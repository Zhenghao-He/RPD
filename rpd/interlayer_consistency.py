"""Final-suffix agreement with a penalty only for net confidence decline.

No learned weights, confidence reward, or decoding fallback is part of the score.
All probabilities refer to the final argmax token before sampling transforms.
"""
from dataclasses import dataclass
import math

import torch
from torch import Tensor
import torch.nn.functional as F


@dataclass
class ConsistencyScore:
    score: Tensor
    consistent_length: Tensor
    delta: Tensor
    final_confidence: Tensor
    slope: Tensor
    penalty: Tensor


def layer_consistency_score(
    layer_argmax: Tensor,
    candidate_probabilities: Tensor,
    *,
    epsilon: float,
    tau: float,
) -> ConsistencyScore:
    """Score trajectories shaped ``[..., L]``, including the final output layer.

    ``candidate_probabilities[..., l]`` must be p_l(y_final), even when the
    intermediate argmax differs. Layers must be consecutive and ordered early
    to late. Earlier agreement interrupted by another prediction is discarded.
    epsilon is an absolute probability tolerance; tau must be positive.
    """
    if not math.isfinite(epsilon) or epsilon < 0:
        raise ValueError('epsilon must be finite and >= 0')
    if not math.isfinite(tau) or tau <= 0:
        raise ValueError('tau must be finite and > 0')
    if layer_argmax.shape != candidate_probabilities.shape or layer_argmax.ndim < 1:
        raise ValueError('argmax and probabilities must have identical [..., L] shapes')
    if not candidate_probabilities.is_floating_point():
        raise TypeError('candidate probabilities must be floating point')
    if layer_argmax.is_floating_point() or layer_argmax.dtype == torch.bool:
        raise TypeError('layer_argmax must contain integer token IDs')
    if layer_argmax.device != candidate_probabilities.device:
        raise ValueError('argmax and probabilities must be on the same device')
    depth = layer_argmax.shape[-1]
    if depth == 0:
        raise ValueError('at least the final output layer is required')
    # Accumulate in FP32 (or preserve FP64 for offline numerical verification).
    q = candidate_probabilities.to(
        torch.float64 if candidate_probabilities.dtype == torch.float64 else torch.float32
    )
    agree = layer_argmax == layer_argmax[..., -1:]
    suffix = agree.flip(-1).to(torch.int64).cumprod(-1).flip(-1).bool()
    length = suffix.sum(-1)
    k = length.to(q.dtype)
    layer = torch.arange(depth, device=q.device, dtype=q.dtype)
    center = (2 * depth - k - 1) / 2
    centered_layer = layer - center[..., None]
    mean_q = torch.where(suffix, q, 0).sum(-1) / k
    numerator = torch.where(suffix, centered_layer * (q - mean_q[..., None]), 0).sum(-1)
    denominator = torch.where(suffix, centered_layer.square(), 0).sum(-1)
    # Replace only a zero denominator: K=2 has denominator 0.5, not 1.
    safe_denominator = torch.where(length > 1, denominator, torch.ones_like(denominator))
    slope = torch.where(length > 1, numerator / safe_denominator, 0)
    delta = (k - 1) * slope
    decline = (-delta - epsilon).clamp_min(0)
    penalty = torch.exp(-decline / tau)
    return ConsistencyScore(k / depth * penalty, length, delta, q[..., -1], slope, penalty)


def consistency_from_logits(layer_logits: Tensor, *, epsilon: float, tau: float) -> ConsistencyScore:
    """Convenience API for unmodified full-vocabulary logits ``[..., L, V]``.

    This materializes probabilities; use LayerConsistencyReadout for live models
    to project eligible positions one layer at a time instead.
    """
    if layer_logits.ndim < 2 or min(layer_logits.shape[-2:]) < 1:
        raise ValueError('expected nonempty [..., L, V] logits')
    probabilities = layer_logits.to(
        torch.float64 if layer_logits.dtype == torch.float64 else torch.float32
    ).softmax(-1)
    winners = probabilities.argmax(-1)
    target = winners[..., -1:, None].expand(*winners.shape, 1)
    q = probabilities.gather(-1, target).squeeze(-1)
    return layer_consistency_score(winners, q, epsilon=epsilon, tau=tau)


@dataclass
class TokenConsistency:
    metrics: ConsistencyScore
    positions: Tensor                 # [N, 2]: original batch index, token position
    token_ids: Tensor                 # [N]: raw final argmax, not sampled IDs
    layer_argmax: Tensor              # [N, L]
    candidate_probabilities: Tensor   # [N, L], final candidate probability
    layer_numbers: tuple
    canvas_shape: tuple

    def select(self, masked: Tensor, allowed: Tensor, *, min_score=None, top_k=None, c_min=None) -> Tensor:
        """Threshold and/or top-k per sequence, restricted to masked & allowed.

        c_min is an optional *separate* final-confidence filter. It never changes
        the score. Empty selections stay empty; fallback belongs to the decoder.
        Ties in score preserve position order. No implicit high-confidence bypass.
        """
        _check_masks(masked, allowed)
        if tuple(masked.shape) != self.canvas_shape or masked.device != self.positions.device:
            raise ValueError('masks must match the scored canvas shape and device')
        if min_score is None and top_k is None:
            raise ValueError('specify min_score and/or top_k explicitly')
        for name, value in [('min_score', min_score), ('c_min', c_min)]:
            if value is not None and (not math.isfinite(value) or not 0 <= value <= 1):
                raise ValueError(f'{name} must be in [0, 1]')
        if top_k is not None and (type(top_k) is not int or top_k < 1):
            raise ValueError('top_k must be a positive integer')
        batch, position = self.positions.unbind(-1)
        keep = masked[batch, position] & allowed[batch, position]
        if c_min is not None:
            keep &= self.metrics.final_confidence >= c_min
        if min_score is not None:
            keep &= self.metrics.score >= min_score
        if top_k is not None:
            ranked_keep = torch.zeros_like(keep)
            for b in range(masked.shape[0]):
                candidates = (keep & (batch == b)).nonzero().flatten()
                order = self.metrics.score[candidates].argsort(descending=True, stable=True)
                ranked_keep[candidates[order[:top_k]]] = True
            keep = ranked_keep
        selected = torch.zeros_like(masked)
        selected[batch[keep], position[keep]] = True
        return selected


def _check_masks(masked, allowed):
    if masked.dtype != torch.bool or allowed.dtype != torch.bool:
        raise TypeError('masked and allowed must be boolean tensors')
    if masked.ndim != 2 or masked.shape != allowed.shape or masked.device != allowed.device:
        raise ValueError('masked and allowed must have identical [batch, sequence] shapes and devices')


class LayerConsistencyReadout:
    """Sparse forward hooks for the repository's official LLaDA and Dream models.

    Call begin(masked, allowed), perform one ordinary full-canvas forward, then
    finish(output.logits). Pass original logits BEFORE any in-place sampling
    transforms. No additional transformer forward or final normalization occurs.
    Default layers are 1..depth; start_layer can select a consecutive suffix
    (e.g. 17 for LLaDA, 15 for Dream). The final output layer is always included.
    Use in eval/inference mode; close hooks via a context manager or close().
    """
    def __init__(self, model, model_name: str, *, epsilon: float, tau: float, start_layer: int = 1):
        if model_name not in ('llada', 'dream'):
            raise ValueError('supported model_name values: llada, dream')
        if not math.isfinite(epsilon) or epsilon < 0 or not math.isfinite(tau) or tau <= 0:
            raise ValueError('require finite epsilon >= 0 and finite tau > 0')
        self.name = model_name
        self.epsilon, self.tau = epsilon, tau
        if model_name == 'llada':
            blocks = model.model.transformer.blocks
            self.norm = model.model.transformer.ln_f
        else:
            blocks = model.model.layers
            self.norm = model.model.norm
        depth = len(blocks)
        if type(start_layer) is not int or not 1 <= start_layer <= depth:
            raise ValueError('start_layer must be a 1-based model layer in [1, depth]')
        self.layers = tuple(range(start_layer, depth + 1))
        self.head = model.get_output_embeddings()
        self.logit_scale = (
            1 / math.sqrt(model.config.d_model)
            if model_name == 'llada' and getattr(model.config, 'scale_logits', False) else 1.0
        )
        self.active = False
        self.closed = False
        self.hidden = {}
        self.hooks = [blocks[layer - 1].register_forward_hook(self._hook(layer)) for layer in self.layers[:-1]]

    def _hook(self, layer):
        def capture(module, args, output):
            if self.active and self.positions.shape[0]:
                if layer in self.hidden:
                    raise RuntimeError('finish the current readout before another forward')
                hidden = output[0] if isinstance(output, tuple) else output
                self.hidden[layer] = hidden[self.batch, self.source_position].detach().clone()
        return capture

    def begin(self, masked: Tensor, allowed: Tensor):
        _check_masks(masked, allowed)
        if self.closed or self.active:
            raise RuntimeError('readout is closed or a previous begin has not been finished')
        self.positions = (masked & allowed).nonzero()
        self.batch, position = self.positions.unbind(-1)
        # Dream official alignment: cat([logits[:, :1], logits[:, :-1]], dim=1).
        self.source_position = (position - 1).clamp_min(0) if self.name == 'dream' else position
        self.canvas_shape = tuple(masked.shape)
        self.hidden = {}
        self.active = True

    @torch.inference_mode()
    def finish(self, raw_final_logits: Tensor) -> TokenConsistency:
        if not self.active:
            raise RuntimeError('call begin before finish')
        self.active = False
        try:
            if raw_final_logits.ndim != 3 or tuple(raw_final_logits.shape[:2]) != self.canvas_shape:
                raise ValueError('expected original full-canvas [batch, sequence, vocabulary] logits')
            if raw_final_logits.device != self.positions.device:
                raise ValueError('logits and masks must be on the same device')
            final_p = raw_final_logits[self.batch, self.source_position].float().softmax(-1)
            target = final_p.argmax(-1)
            final_q = final_p.gather(-1, target[:, None]).squeeze(-1)
            del final_p
            winners, curves = [], []
            if self.positions.shape[0]:
                if set(self.hidden) != set(self.layers[:-1]):
                    raise RuntimeError('missing intermediate layers: perform a full model forward after begin')
                for layer in self.layers[:-1]:
                    normalized = self.norm(self.hidden.pop(layer))
                    if isinstance(self.head, torch.nn.Embedding):
                        logits = F.linear(normalized, self.head.weight)  # tied LLaDA head
                    else:
                        logits = self.head(normalized)
                    if self.logit_scale != 1.0:
                        logits = logits * self.logit_scale
                    probabilities = logits.float().softmax(-1)
                    winners.append(probabilities.argmax(-1))
                    curves.append(probabilities.gather(-1, target[:, None]).squeeze(-1))
                    del logits, probabilities
            else:
                winners = [target for _ in self.layers[:-1]]
                curves = [final_q for _ in self.layers[:-1]]
            winners.append(target)
            curves.append(final_q)
            argmax = torch.stack(winners, dim=-1)
            q = torch.stack(curves, dim=-1)
            metrics = layer_consistency_score(argmax, q, epsilon=self.epsilon, tau=self.tau)
            return TokenConsistency(metrics, self.positions, target, argmax, q, self.layers, self.canvas_shape)
        finally:
            self.hidden.clear()

    def close(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
        self.hidden.clear()
        self.active = False
        self.closed = True

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
