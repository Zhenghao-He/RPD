"""RPD decoding loop.

One backbone forward per round. Every still-masked position on the canvas is
scored from the posterior-half layer trajectory (suffix progress) and from its
final-layer confidence; a left-to-right cumulative-entropy gate then decides
which of the admitted positions may actually be committed this round.
"""
import time

import torch

from .coupled_open_block_suffix import CoupledOpenBlockReadout
from .gate import sequential_gate

# Frozen configuration used for the reported results.
DEFAULTS = {
    'gen_len': 256,
    'high_confidence': 0.9,     # commit outright above this confidence
    'c_min': 0.6,               # suffix-progress candidates live in [c_min, high)
    'length_weight': 1.0,       # score = w_len * min(K, cap) - w_drop * drawdown
    'length_cap': 6,
    'kappa': 1.0,               # readout compatibility only
    'tau': 1.0,
    'd': 0.25,
    'entropy_budget': 4.0,      # nats of residual debt tolerated
    'fallback_window': 32,
}
# Per-model score threshold and drawdown weight.
MODELS = {
    'llada': {'drop_weight': 15.0, 'score_threshold': 3.5},
    'dream': {'drop_weight': 20.0, 'score_threshold': 2.5},
}


def config_for(name, **overrides):
    if name not in MODELS:
        raise ValueError(f'unknown model {name!r}; expected one of {sorted(MODELS)}')
    return {**DEFAULTS, **MODELS[name], **overrides}


def candidate_masks(confidence, score, config):
    """Split pending positions into outright commits and suffix-progress commits."""
    high = confidence >= config['high_confidence']
    extra = (score >= config['score_threshold']) & (confidence >= config['c_min'])
    return high, extra & ~high


def select(result, left, length, config):
    """Score the pending positions and apply the entropy gate."""
    metrics = result.metrics
    capped = metrics.consistent_length.float().clamp(max=config['length_cap'])
    metrics.consistency_component = config['length_weight'] * capped
    metrics.score = metrics.consistency_component - config['drop_weight'] * metrics.drawdown
    high, extra = candidate_masks(metrics.final_confidence, metrics.score, config)
    chosen, debt = sequential_gate(high, extra, result.entropy,
                                   config['entropy_budget'], config['entropy_budget'])
    fallback = not bool(chosen.any())
    if fallback:
        # Guarantee progress: commit the most confident position near the frontier.
        positions = result.positions[:, 1] - config['_prompt_length']
        allowed = positions < left + config['fallback_window']
        chosen[torch.where(allowed, metrics.final_confidence, -torch.inf).argmax()] = True
    return chosen, debt, high, extra, fallback


@torch.inference_mode()
def decode(model, name, prompt_ids, config=None):
    """Decode one prompt. Returns the generated ids and per-round diagnostics."""
    config = dict(config or config_for(name))
    length = config['gen_len']
    mask_id = model.config.mask_token_id
    plen = len(prompt_ids)
    config['_prompt_length'] = plen
    layers = model.model.transformer.blocks if name == 'llada' else model.model.layers
    device = next(model.parameters()).device

    forwards = [0]
    hook = model.register_forward_pre_hook(lambda *a: forwards.__setitem__(0, forwards[0] + 1))
    kwargs = {'attention_mask': 'full'} if name == 'dream' else {}
    rounds = []
    try:
        with CoupledOpenBlockReadout(model, name, kappa=config['kappa'], tau=config['tau'],
                                     d=config['d'], start_layer=len(layers) // 2 + 1) as readout:
            torch.cuda.synchronize()
            started = time.perf_counter()
            x = torch.tensor([prompt_ids + [mask_id] * length], device=device)
            while bool((x[0, plen:] == mask_id).any()):
                pending = x == mask_id
                positions = pending[0, plen:].nonzero().flatten()
                left = int(positions[0])
                allowed = torch.zeros_like(pending)
                allowed[:, plen:plen + length] = True

                readout.begin(pending, allowed)
                output = model(x, **kwargs)
                captured = readout.positions[:, 1] - plen
                indices = ((captured >= left) & (captured < length)).nonzero().flatten()
                result = readout.read(output.logits, indices)

                # Predictive entropy of each pending position, from the same forward.
                logits = output.logits[readout.batch[indices],
                                       readout.source_position[indices]].float()
                log_p = logits.log_softmax(-1)
                result.entropy = -(log_p.exp() * log_p).sum(-1)
                del logits, log_p

                chosen, debt, high, extra, fallback = select(result, left, length, config)
                scope = result.positions[:, 1] - plen
                picked = scope[chosen]
                x[0, picked + plen] = result.token_ids[chosen]
                rounds.append({
                    'committed': picked.tolist(),
                    'high_confidence': int((high & chosen).sum()),
                    'suffix_progress': int((extra & chosen).sum()),
                    'fallback': bool(fallback),
                    'frontier': left,
                })
                readout.end()
                del output
            torch.cuda.synchronize()
            seconds = time.perf_counter() - started
    finally:
        hook.remove()

    generated = x[0, plen:].tolist()
    assert mask_id not in generated
    return {'generated_ids': generated, 'nfe': forwards[0], 'seconds': seconds,
            'rounds': rounds, 'model': name}
