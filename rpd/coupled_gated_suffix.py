"""Suffix agreement with agreement-conditioned peak-drawdown tolerance."""
from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F
from torch import Tensor

from .interlayer_consistency import LayerConsistencyReadout
from .saturated_consistency import SaturatedTokens


def _parameters(kappa, tau, d):
    for name, value in [('kappa', kappa), ('tau', tau)]:
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f'{name} must be finite and > 0')
    if not math.isfinite(d) or d < 0:
        raise ValueError('d must be finite and >= 0')


@dataclass
class CoupledGatedSuffixScore:
    score: Tensor
    eligible: Tensor
    consistent_length: Tensor
    peak: Tensor
    drawdown: Tensor
    final_confidence: Tensor
    consistency_component: Tensor
    penalty: Tensor


def coupled_gated_suffix_score(layer_argmax: Tensor,
                               candidate_probabilities: Tensor, *,
                               kappa: float = 1.5, tau: float = .2,
                               d: float = .25) -> CoupledGatedSuffixScore:
    """Compute S=a*exp(-r/(tau*a)), where a=1-exp(-K/kappa).

    K and r use only the uninterrupted suffix whose argmax equals the final
    output argmax. K=1 retains the established convention r=0. The Boolean
    drawdown eligibility is returned for diagnostics; admission policy decides
    whether to apply it.
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
    drawdown = torch.where(length == 1, torch.zeros_like(final),
                           (peak-final).clamp_min(0))
    consistency = -torch.expm1(-length.to(q.dtype)/kappa)
    penalty = torch.exp(-drawdown/(tau*consistency))
    return CoupledGatedSuffixScore(
        consistency*penalty, drawdown <= d, length, peak, drawdown, final,
        consistency, penalty)


@dataclass
class CoupledGatedSuffixTokens(SaturatedTokens):
    metrics: CoupledGatedSuffixScore


class CoupledGatedSuffixReadout(LayerConsistencyReadout):
    def __init__(self, model, model_name, *, kappa=1.5, tau=.2, d=.25,
                 start_layer=1):
        _parameters(kappa, tau, d)
        super().__init__(model, model_name, epsilon=0, tau=tau,
                         start_layer=start_layer)
        self.kappa, self.d = kappa, d

    @torch.inference_mode()
    def finish(self, raw_final_logits: Tensor) -> CoupledGatedSuffixTokens:
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
                h = self.norm(torch.stack(
                    [self.hidden[layer] for layer in self.layers[:-1]], 0).flatten(0, 1))
                z = (F.linear(h, self.head.weight)
                     if isinstance(self.head, torch.nn.Embedding) else self.head(h))
                if self.logit_scale != 1:
                    z = z*self.logit_scale
                probability = z.float().softmax(-1)
                arg = probability.argmax(-1).reshape(depth, n).T
                q = probability.gather(
                    -1, target.repeat(depth)[:, None]).reshape(depth, n).T
            else:
                arg = target.new_empty((n, depth))
                q = confidence.new_empty((n, depth))
            arg = torch.cat([arg, target[:, None]], -1)
            q = torch.cat([q, confidence[:, None]], -1)
            metrics = coupled_gated_suffix_score(
                arg, q, kappa=self.kappa, tau=self.tau, d=self.d)
            return CoupledGatedSuffixTokens(
                metrics, self.positions, target, arg, q, self.layers,
                self.canvas_shape)
        finally:
            self.hidden.clear()
