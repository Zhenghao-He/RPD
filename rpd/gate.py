"""Cumulative residual-entropy gate.

Scanning the canvas left to right, a position may only be committed while the
entropy accumulated by the still-unresolved positions to its left stays inside
a budget. Rejected positions add their predictive entropy to that shared debt,
so a token far behind an unresolved region can never be committed early.
"""
import torch


def sequential_gate(high, extra, entropy, conf_budget, sp_budget):
    """Left-to-right admission under a shared entropy debt.

    Args:
        high: bool tensor, positions passing the confidence threshold alone.
        extra: bool tensor, positions admitted by the suffix-progress score.
        entropy: float tensor, predictive entropy of every pending position.
        conf_budget: nats of debt tolerated for a `high` position.
        sp_budget: nats of debt tolerated for an `extra` position.

    Returns:
        (chosen, debt) - the committed mask and the debt seen at each position.
    """
    rows = torch.stack([high.float(), extra.float(), entropy], dim=1).tolist()
    chosen, debts = [], []
    debt = 0.0
    for is_high, is_sp, uncertainty in rows:
        debts.append(debt)
        accepted = ((bool(is_high) and debt <= conf_budget) or
                    (bool(is_sp) and debt <= sp_budget))
        chosen.append(accepted)
        if not accepted:
            debt += uncertainty
    return (torch.tensor(chosen, dtype=torch.bool, device=entropy.device),
            torch.tensor(debts, dtype=entropy.dtype, device=entropy.device))
