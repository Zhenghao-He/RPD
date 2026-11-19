"""Lazy readout: compute only the layer projections a decision needs.

Positions that the entropy gate can already rule out never have their layer
trajectory materialised. Uncomputed metadata is marked with K=-1 and an explicit
evaluated mask, never fabricated. Commit decisions are identical to the eager
readout; only the amount of computation differs.

Release status: interface only. The implementation will be restored after the
evaluation pipeline has been re-verified.
"""
import torch


def suffix_metrics(argmax, q):
    """Final agreement length K and peak-to-final drop r for each position.

    Args:
        argmax: [n, layers] per-layer argmax token ids.
        q: [n, layers] per-layer probability of the final-layer token.

    Returns:
        (k, drop) tensors of shape [n].
    """
    raise NotImplementedError('suffix_metrics is withheld pending re-verification')


@torch.inference_mode()
def read_selectively(rd, logits, indices, config):
    """Lazily materialise layer trajectories only where the gate still needs them."""
    raise NotImplementedError('read_selectively is withheld pending re-verification')
