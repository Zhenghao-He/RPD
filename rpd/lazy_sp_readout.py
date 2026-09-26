"""Lazy readout: compute only the layer projections a decision needs.

Positions that the entropy gate can already rule out never have their layer
trajectory materialised. Uncomputed metadata is marked with K=-1 and an explicit
evaluated mask, never fabricated. Commit decisions are identical to the eager
readout; only the amount of computation differs.
"""
import math
from types import SimpleNamespace
import torch
import torch.nn.functional as F


def possible_prefix(entropy, definitely_rejected, budget):
    """Conservative exclusive prefix debt lower bound; equality remains eligible."""
    debt = torch.where(definitely_rejected, entropy.double(), 0.).cumsum(0)
    before = torch.cat([debt.new_zeros(1), debt[:-1]])
    # FP64 versus the reference Python-double sum: keep a safety band at B.
    return before <= float(budget) + 1e-9


def suffix_metrics(argmax, q):
    suffix = (argmax == argmax[:, -1:]).flip(-1).long().cumprod(-1).flip(-1).bool()
    k = suffix.sum(-1)
    peak = q.masked_fill(~suffix, -torch.inf).amax(-1)
    drop = torch.where(k == 1, torch.zeros_like(peak), (peak-q[:, -1]).clamp_min(0))
    return k, drop


@torch.inference_mode()
def read_selectively(rd, logits, indices, config):
    assert config['candidate_region'] == 'full_canvas'
    assert config['candidate_window_limit'] is None
    assert not config['high_conf_bypass']
    assert config['conf_entropy_budget'] == config['sp_entropy_budget']
    assert config['length_weight'] == 1. and config['drop_weight'] >= 0
    assert config['c_min_comparison'] == '>='
    n = indices.numel(); device = logits.device
    batch, source = rd.batch[indices], rd.source_position[indices]
    z = logits[batch, source].float()
    lp = z.log_softmax(-1); p = lp.exp()
    target = p.argmax(-1); conf = p.gather(-1, target[:, None]).squeeze(-1)
    entropy = -(p*lp).sum(-1)
    assert bool(torch.isfinite(entropy).all())
    # Preserve legacy softmax comparison exactly when close to an admission boundary.
    near = ((conf-config['c_min']).abs() < 2e-6) | ((conf-config['high_confidence']).abs() < 2e-6)
    if bool(near.any()):
        old = z[near].softmax(-1)
        conf[near] = old.gather(-1, target[near, None]).squeeze(-1)
    del z, p, lp
    high = conf >= config['high_confidence']
    mid = (conf >= config['c_min']) & ~high
    rejected = conf < config['c_min']
    possible = possible_prefix(entropy, rejected, config['entropy_budget'])
    candidate = (mid & possible).nonzero().flatten()
    k = torch.full((n,), -1, dtype=torch.long, device=device)
    drop = torch.zeros(n, device=device)
    evaluated = torch.zeros(n, dtype=torch.bool, device=device)
    minimum = math.ceil(config['score_threshold'])
    assert 2 <= minimum <= config['length_cap'] <= len(rd.layers)
    late = rd.layers[-minimum:-1]; early = rd.layers[:-minimum]
    projected_rows = 0

    def project(rows, layers):
        nonlocal projected_rows
        if not layers:
            return (target.new_empty((len(rows), 0)), conf.new_empty((len(rows), 0)))
        hidden = rd.norm(torch.stack([rd.hidden[layer][indices[rows]] for layer in layers], 0).flatten(0, 1))
        out = F.linear(hidden, rd.head.weight) if isinstance(rd.head, torch.nn.Embedding) else rd.head(hidden)
        if rd.logit_scale != 1: out = out*rd.logit_scale
        probability = out.float().softmax(-1)
        arg = probability.argmax(-1).reshape(len(layers), len(rows)).T
        q = probability.gather(-1, target[rows].repeat(len(layers))[:, None]).reshape(len(layers), len(rows)).T
        projected_rows += len(rows)*len(layers)
        return arg, q

    screened = len(candidate)
    surviving = 0
    if screened:
        arg, q = project(candidate, late)
        full_arg = torch.cat([arg, target[candidate, None]], -1)
        full_q = torch.cat([q, conf[candidate, None]], -1)
        sk, sr = suffix_metrics(full_arg, full_q)
        short = sk < minimum
        short_rows = candidate[short]
        k[short_rows] = sk[short]; drop[short_rows] = sr[short]; evaluated[short_rows] = True
        rejected[short_rows] = True
        possible = possible_prefix(entropy, rejected, config['entropy_budget'])
        keep = (~short) & possible[candidate]
        rows = candidate[keep]; surviving = len(rows)
        if surviving:
            ea, eq = project(rows, early)
            fa = torch.cat([ea, full_arg[keep]], -1)
            fq = torch.cat([eq, full_q[keep]], -1)
            kk, rr = suffix_metrics(fa, fq)
            k[rows] = kk; drop[rows] = rr; evaluated[rows] = True
    score = k.clamp(max=config['length_cap']).float()-config['drop_weight']*drop
    # Recheck borderline scores with the unchanged full readout, preserving tie behavior.
    near_score = evaluated & ((score-config['score_threshold']).abs() < 2e-4)
    if bool(near_score.any()):
        old = rd.read(logits, indices[near_score])
        k[near_score] = old.metrics.consistent_length; drop[near_score] = old.metrics.drawdown
        conf[near_score] = old.metrics.final_confidence
        projected_rows += int(near_score.sum())*(len(rd.layers)-1)
        score = k.clamp(max=config['length_cap']).float()-config['drop_weight']*drop
    metrics = SimpleNamespace(final_confidence=conf,consistent_length=k,drawdown=drop,score=score,
                              sp_evaluated=evaluated,projection_rows=projected_rows,
                              screened_positions=screened,fully_read_positions=surviving)
    # Fallback may need max-conf tie breaking even though no SP was evaluated.
    # Cache raw logits for a legacy softmax recheck only if fallback is used.
    return SimpleNamespace(metrics=metrics,positions=rd.positions[indices],token_ids=target,
                           entropy=entropy,final_raw_logits=logits[batch, source])
