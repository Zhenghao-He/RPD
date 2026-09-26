"""Confidence-weighted, saturating suffix agreement with a hard drawdown veto."""
from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F
from torch import Tensor

from .interlayer_consistency import LayerConsistencyReadout, TokenConsistency


def _parameters(kappa, tau, d):
    for name, value in [('kappa', kappa), ('tau', tau)]:
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f'{name} must be finite and > 0')
    if not math.isfinite(d) or d < 0:
        raise ValueError('d must be finite and >= 0')


@dataclass
class SaturatedScore:
    score: Tensor
    eligible: Tensor
    consistent_length: Tensor
    peak: Tensor
    drawdown: Tensor
    final_confidence: Tensor
    saturation: Tensor
    penalty: Tensor


def saturated_consistency_score(layer_argmax: Tensor, candidate_probabilities: Tensor,
                                *, kappa: float, tau: float, d: float) -> SaturatedScore:
    """Inputs [..., L] follow consecutive layers, ending with the final output.

    Probabilities always track y_final. Only its final uninterrupted argmax
    suffix contributes to K and peak. No epsilon deadzone or division by L.
    Eligibility is returned separately; a tiny/zero score is NOT a hard veto.
    """
    _parameters(kappa, tau, d)
    if layer_argmax.shape != candidate_probabilities.shape or layer_argmax.ndim < 1:
        raise ValueError('argmax and probabilities must have identical [..., L] shapes')
    if not candidate_probabilities.is_floating_point():
        raise TypeError('candidate probabilities must be floating point')
    if layer_argmax.is_floating_point() or layer_argmax.dtype == torch.bool:
        raise TypeError('layer_argmax must contain integer token IDs')
    if layer_argmax.device != candidate_probabilities.device:
        raise ValueError('argmax and probabilities must be on the same device')
    if layer_argmax.shape[-1] == 0:
        raise ValueError('at least the final output layer is required')
    q = candidate_probabilities.to(
        torch.float64 if candidate_probabilities.dtype == torch.float64 else torch.float32)
    agree = layer_argmax == layer_argmax[..., -1:]
    suffix = agree.flip(-1).to(torch.int64).cumprod(-1).flip(-1).bool()
    length = suffix.sum(-1)
    final = q[..., -1]
    peak = q.masked_fill(~suffix, -torch.inf).amax(-1)
    drawdown = torch.where(length == 1, torch.zeros_like(final), (peak-final).clamp_min(0))
    saturation = -torch.expm1(-length.to(q.dtype)/kappa)
    penalty = torch.exp(-drawdown/tau)
    score = final*saturation*penalty
    return SaturatedScore(score, drawdown <= d, length, peak, drawdown, final,
                          saturation, penalty)


@dataclass
class SaturatedTokens(TokenConsistency):
    metrics: SaturatedScore

    def select(self, masked, allowed, *, min_score=None, top_k=None, c_min=None):
        # Apply the veto BEFORE thresholding/top-k, even if min_score == 0 or
        # top_k exceeds the number of eligible candidates.
        eligibility = torch.zeros_like(allowed)
        batch, position = self.positions.unbind(-1)
        eligibility[batch, position] = self.metrics.eligible
        return super().select(masked, allowed & eligibility,
                              min_score=min_score, top_k=top_k, c_min=c_min)


def select_for_commit(result: SaturatedTokens, threshold: float, *, policy='all_score'):
    """Single-sequence block selection, returning (chosen, fallback, blocked).

    Threshold-empty: retain the historical max-confidence fallback, but ONLY
    among E=1 positions. All-veto: return empty with blocked=True; never weaken
    d or silently force a vetoed token. No model forward occurs here.
    """
    if not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError('threshold must be in [0, 1]')
    if result.canvas_shape[0] != 1:
        raise ValueError('commit fallback expects one sequence')
    if policy not in ('all_score', 'hybrid'):
        raise ValueError('policy must be all_score or hybrid')
    m = result.metrics
    if policy == 'all_score':
        chosen = m.eligible & (m.score >= threshold)
    else:
        chosen = m.eligible & ((m.final_confidence >= .9) |
            ((m.final_confidence >= .5) & (m.score >= threshold)))
    if bool(chosen.any()):
        return chosen, False, False
    if not bool(m.eligible.any()):
        return chosen, False, True
    index = m.final_confidence.masked_fill(~m.eligible, -torch.inf).argmax()
    chosen[index] = True
    return chosen, True, False


class SaturatedConsistencyReadout(LayerConsistencyReadout):
    """Batched LM-head projection only for masked, allowed block positions.

    Inherits official model normalization and Dream target/source alignment.
    Intermediate states receive one output norm; final raw logits are reused.
    Distributions precede temperature/noise/top-p. No extra model forward.
    """
    def __init__(self, model, model_name, *, kappa, tau, d, start_layer=1):
        _parameters(kappa, tau, d)
        super().__init__(model, model_name, epsilon=0, tau=tau, start_layer=start_layer)
        self.kappa, self.d = kappa, d

    @torch.inference_mode()
    def finish(self, raw_final_logits: Tensor) -> SaturatedTokens:
        if not self.active:
            raise RuntimeError('call begin before finish')
        self.active = False
        try:
            if raw_final_logits.ndim != 3 or tuple(raw_final_logits.shape[:2]) != self.canvas_shape:
                raise ValueError('expected original full-canvas logits')
            if raw_final_logits.device != self.positions.device:
                raise ValueError('logits and masks must be on the same device')
            p = raw_final_logits[self.batch, self.source_position].float().softmax(-1)
            target = p.argmax(-1)
            confidence = p.gather(-1, target[:, None]).squeeze(-1)
            del p
            depth, n = len(self.layers)-1, self.positions.shape[0]
            if n and set(self.hidden) != set(self.layers[:-1]):
                raise RuntimeError('missing intermediate layers: perform a full model forward')
            if n and depth:
                h = torch.stack([self.hidden[l] for l in self.layers[:-1]], 0)
                h = self.norm(h.flatten(0, 1))
                z = (F.linear(h, self.head.weight) if isinstance(self.head, torch.nn.Embedding)
                     else self.head(h))
                if self.logit_scale != 1:
                    z = z*self.logit_scale
                probability = z.float().softmax(-1)
                arg = probability.argmax(-1).reshape(depth, n).T
                q = probability.gather(-1, target.repeat(depth)[:, None]).reshape(depth, n).T
            else:
                arg = target.new_empty((n, depth))
                q = confidence.new_empty((n, depth))
            arg = torch.cat([arg, target[:, None]], -1)
            q = torch.cat([q, confidence[:, None]], -1)
            metrics = saturated_consistency_score(arg, q, kappa=self.kappa, tau=self.tau, d=self.d)
            return SaturatedTokens(metrics, self.positions, target, arg, q,
                                   self.layers, self.canvas_shape)
        finally:
            self.hidden.clear()
