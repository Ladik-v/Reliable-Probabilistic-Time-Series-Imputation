import numpy as np
from scipy.stats import norm

from .metrics import PINBALL_LEVELS, level_index

Q21 = np.round(np.concatenate([[0.01], np.arange(1, 20) * 0.05, [0.99]]), 2)
Z90 = float(norm.ppf(0.9))


def hidden_runs(idx):
    breaks = np.nonzero(np.diff(idx) != 1)[0]
    starts = np.concatenate([[0], breaks + 1])
    ends = np.concatenate([breaks + 1, [len(idx)]])
    return [(int(idx[a]), int(idx[b - 1] + 1)) for a, b in zip(starts, ends)]


def run_geometry(idx):
    n, k = idx.shape
    j, h = np.zeros((n, k)), np.zeros((n, k))
    for i in range(n):
        pos = 0
        for s, e in hidden_runs(idx[i]):
            length = e - s
            j[i, pos:pos + length] = np.arange(length)
            h[i, pos:pos + length] = length
            pos += length
    return j, h


def quantiles_at(pred, key, qs):
    qs = np.asarray(qs)
    if f'{key}/mean' in pred:
        return pred[f'{key}/mean'][..., None] + pred[f'{key}/std'][..., None] * norm.ppf(qs)
    if f'{key}/samples' in pred:
        return np.moveaxis(np.quantile(pred[f'{key}/samples'], qs, axis=1), 0, -1)
    return pred[f'{key}/quantiles']


def has_key(pred, key):
    return pred is not None and any(f'{key}/{s}' in pred for s in ('quantiles', 'mean', 'samples'))


def pinball_w(y, Q, qs, levels):
    out = []
    for q in levels:
        e = y - Q[..., level_index(qs, q)]
        out.append(np.maximum(q * e, (q - 1) * e).mean(axis=-1))
    return np.mean(out, axis=0)


def window_scores(y, Q, qs, crps_name='crpsq'):
    qs = np.asarray(qs)
    m = Q[..., level_index(qs, 0.5)]
    lo, hi = Q[..., level_index(qs, 0.1)], Q[..., level_index(qs, 0.9)]
    return {
        'mae': np.abs(y - m).mean(axis=-1),
        'pinball': pinball_w(y, Q, qs, PINBALL_LEVELS),
        'cov80': ((lo <= y) & (y <= hi)).mean(axis=-1),
        'width80': (hi - lo).mean(axis=-1),
        crps_name: 2 * pinball_w(y, Q, qs, list(qs)),
    }


def cluster_draws(clusters, n_boot, rng):
    clusters = np.asarray(clusters)
    ids = [np.nonzero(clusters == c)[0] for c in np.unique(clusters)]
    c_n = len(ids)
    draws = []
    for _ in range(n_boot):
        parts = []
        for c in rng.integers(0, c_n, c_n):
            w = ids[c]
            parts.append(rng.choice(w, size=len(w), replace=True))
        draws.append(np.concatenate(parts))
    return draws


def boot_means(M, draws):
    return np.stack([M[:, d].mean(axis=1) for d in draws])


def interval(x, level=0.95):
    a = (1 - level) / 2
    lo, hi = np.quantile(x, [a, 1 - a])
    return float(lo), float(hi)


def verdict(lo, hi):
    if hi < 0:
        return 'first better'
    if lo > 0:
        return 'second better'
    return 'tie'
