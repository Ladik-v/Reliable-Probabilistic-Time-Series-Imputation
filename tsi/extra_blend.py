import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm

from .common import load_config, load_masks, load_series, mask_configs, parse_args, results_dir
from .evaluate import load_pred, targets
from .extra_stats import Z90, run_geometry
from .metrics import all_metrics, along_gap, level_index

VARIANTS = ('linear', 'uniform', 'poe')
LABELS = {'linear': 'linear weights (current)', 'uniform': 'uniform 0.5', 'poe': 'Gaussian product of experts',
          'left': 'left-only', 'right': 'right-only'}
COLORS = {'linear': '#e08e0b', 'uniform': '#9467bd', 'poe': '#2a6fdb', 'left': '#7a7a7a', 'right': '#3c9d5d'}


def gaussian_from_quantiles(Q, qs):
    qs = np.asarray(qs)
    mu = Q[..., level_index(qs, 0.5)]
    sd = (Q[..., level_index(qs, 0.9)] - Q[..., level_index(qs, 0.1)]) / (2 * Z90)
    return mu, np.maximum(sd, 1e-8)


def combine(QL, QR, qs, idx, kind):
    qs = np.asarray(qs)
    if kind in ('linear', 'uniform'):
        j, h = run_geometry(idx)
        w = ((h - j) / (h + 1) if kind == 'linear' else np.full(h.shape, 0.5))[..., None]
        return w * QL + (1 - w) * QR
    mL, sL = gaussian_from_quantiles(QL, qs)
    mR, sR = gaussian_from_quantiles(QR, qs)
    pL, pR = 1 / sL ** 2, 1 / sR ** 2
    mu = (mL * pL + mR * pR) / (pL + pR)
    sd = 1 / np.sqrt(pL + pR)
    return mu[..., None] + sd[..., None] * norm.ppf(qs)


def main():
    cfg = load_config(parse_args('Blend variants from saved left and right Chronos quantiles').config)
    qs = np.array(cfg['quantiles'])
    tables, figs = results_dir(cfg) / 'tables', results_dir(cfg) / 'figures'
    tables.mkdir(parents=True, exist_ok=True)
    figs.mkdir(parents=True, exist_ok=True)
    for split in ('val', 'test'):
        rows = []
        gaps = {}
        for name in cfg['datasets']:
            series, masks = load_series(cfg, name), load_masks(cfg, split, name)
            left, right = load_pred(cfg, split, name, 'chronos_left'), load_pred(cfg, split, name, 'chronos_right')
            saved = load_pred(cfg, split, name, 'chronos_blend')
            for key in mask_configs(cfg):
                pattern, k = key.split('_')
                y, mu, sd = targets(series, masks, key)
                QL, QR = left[f'{key}/quantiles'], right[f'{key}/quantiles']
                idx = masks[f'{key}/idx']
                variants = {v: combine(QL, QR, qs, idx, v) for v in VARIANTS}
                err = np.abs(variants['linear'] - saved[f'{key}/quantiles']).max()
                if err > 1e-4:
                    print(f'WARNING {split}/{name}/{key}: linear variant differs from saved blend by {err:.2e}')
                variants['left'], variants['right'] = QL, QR
                for v, Q in variants.items():
                    m = Q[..., level_index(qs, 0.5)]
                    met = all_metrics(y, Q, qs, y * sd + mu, m * sd + mu)
                    rows.append({'split': split, 'dataset': name, 'pattern': pattern, 'hidden': int(k), 'variant': v,
                                 **{c: round(met[c], 4) for c in ('MAE', 'pinball', 'CRPS', 'PCE', 'CCE', 'cov80', 'width80', 'SIW')}})
                    if pattern == 'block':
                        gaps[(name, key, v)] = along_gap(y, Q, qs)
        df = pd.DataFrame(rows)
        df.to_csv(tables / f'extra_blend_variants_{split}.csv', index=False)
        keys = [k for k in mask_configs(cfg) if k.startswith('block')]
        for name in cfg['datasets']:
            fig, axes = plt.subplots(2, len(keys), figsize=(5.5 * len(keys), 7), squeeze=False)
            for col, key in enumerate(keys):
                for v in ('linear', 'uniform', 'poe', 'left', 'right'):
                    cov, width = gaps[(name, key, v)]
                    ls = ':' if v in ('left', 'right') else '-'
                    axes[0, col].plot(np.arange(1, len(cov) + 1), cov, ls=ls, c=COLORS[v], label=LABELS[v])
                    axes[1, col].plot(np.arange(1, len(cov) + 1), width, ls=ls, c=COLORS[v])
                axes[0, col].axhline(0.8, c='black', lw=0.8, ls='--')
                axes[0, col].set_title(f'{name}: gap of {key.split("_")[1]}')
                axes[0, col].set_ylabel('80% coverage')
                axes[1, col].set_ylabel('80% width (scaled units)')
                axes[1, col].set_xlabel('position in the gap')
            axes[0, 0].legend(fontsize=8)
            fig.tight_layout()
            fig.savefig(figs / f'extra_blend_variants_{split}_{name}.png', dpi=150)
            plt.close(fig)
        print(f'{split}: {len(df)} rows -> {tables / f"extra_blend_variants_{split}.csv"}')


if __name__ == '__main__':
    main()
