"""RPD decoding loop.

One backbone forward per round. Every still-masked position on the canvas is
scored from the posterior-half layer trajectory (suffix progress) and from its
final-layer confidence; a left-to-right cumulative-entropy gate then decides
which of the admitted positions may actually be committed this round.

Release status: the configuration, candidate admission rule and interfaces are
included; `select` and `decode` will be restored after the evaluation pipeline
has been re-verified.
"""
import torch

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
    'lazy_readout': True,
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
    """Score the pending positions and apply the entropy gate.

    Returns:
        (chosen, debt, high, extra, fallback)
    """
    raise NotImplementedError('select is withheld pending re-verification')


@torch.inference_mode()
def decode(model, name, prompt_ids, config=None):
    """Decode one prompt.

    Returns:
        dict with 'generated_ids', 'nfe', 'seconds', 'rounds' and 'model'.
    """
    raise NotImplementedError('decode is withheld pending re-verification')
