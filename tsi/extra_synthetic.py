
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm

from .chronos_imputer import impute
from .common import load_config, parse_args, results_dir
from .extra_blend import combine
from .extra_chronos import load_pipe, matched_forecasts
from .extra_stats import boot_means, cluster_draws, interval, window_scores
from .kalman import kalman_smoother
from .masks import hidden_index
from .metrics import all_metrics, along_gap, level_index

LENGTH, EDGE = 256, 24
COLORS = {'oracle': 'black', 'left': '#7a7a7a', 'right': '#3c9d5d', 'linear': '#e08e0b', 'uniform': '#9467bd', 'poe': '#2a6fdb'}


def local_level(n, q, r, rng):
    mu = np.cumsum(rng.normal(0, np.sqrt(q), (n, LENGTH)), axis=1)
    return mu + rng.normal(0, np.sqrt(r), (n, LENGTH))


def ar1(n, rng, phi=0.9, burn=200):
    e = rng.normal(size=(n, LENGTH + burn))
    x = np.zeros_like(e)
    for t in range(1, LENGTH + burn):
        x[:, t] = phi * x[:, t - 1] + e[:, t]
    return x[:, burn:]


def shot_noise(n, rng, decay=0.9, p=0.03, scale=3.0, noise=0.05, burn=200):
    jumps = rng.exponential(scale, (n, LENGTH + burn)) * (rng.random((n, LENGTH + burn)) < p)
    eps = rng.normal(0, noise, (n, LENGTH + burn))
    x = np.zeros_like(jumps)
    for t in range(1, LENGTH + burn):
        x[:, t] = decay * x[:, t - 1] + jumps[:, t] + eps[:, t]
    return x[:, burn:]


def block_mask(n, k, rng):
    idx = np.stack([hidden_index('block', k, LENGTH, EDGE, rng) for _ in range(n)])
    return idx


def apply_mask(y, idx):
    x = y.copy()
    np.put_along_axis(x, idx, np.nan, axis=1)
    return x


def run_oracle(cfg, pipe, n, out):
    qs = np.array(cfg['quantiles'])
    batch = cfg['chronos']['batch_size']
    rows, gaps = [], {}
    for ratio in (0.01, 0.1, 1.0):
        for k in cfg['hidden_counts']:
            rng = np.random.default_rng([7, int(ratio * 100), k])
            y_full = local_level(n, 1.0, ratio, rng)
            idx = block_mask(n, k, rng)
            x = apply_mask(y_full, idx)
            y = np.take_along_axis(y_full, idx, 1)
            res = impute(pipe, x, idx, list(qs), batch)
            a_s, p_s = kalman_smoother(x, np.full(n, 1.0), np.full(n, ratio))
            mean = np.take_along_axis(a_s, idx, 1)
            std = np.sqrt(np.take_along_axis(p_s, idx, 1) + ratio)
            models = {'oracle': mean[..., None] + std[..., None] * norm.ppf(qs), 'left': res['left'], 'right': res['right']}
            for kind in ('linear', 'uniform', 'poe'):
                models[kind] = combine(res['left'], res['right'], qs, idx, kind)
            for name, Q in models.items():
                m = Q[..., level_index(qs, 0.5)]
                met = all_metrics(y, Q, qs, y, m)
                gaps[(ratio, k, name)] = along_gap(y, Q, qs)
                rows.append({'noise_ratio': ratio, 'gap': k, 'model': name,
                             **{c: round(met[c], 4) for c in ('MAE', 'pinball', 'CRPS', 'PCE', 'CCE', 'cov80', 'width80', 'SIW')}})
            print(f'oracle test: ratio {ratio}, gap {k} done')
    df = pd.DataFrame(rows)
    ref = df[df['model'] == 'oracle'].set_index(['noise_ratio', 'gap'])['width80']
    df['width_ratio_to_oracle'] = [round(r['width80'] / ref[(r['noise_ratio'], r['gap'])], 3) for _, r in df.iterrows()]
    df.to_csv(out / 'extra_synthetic_oracle.csv', index=False)
    for k in cfg['hidden_counts']:
        fig, axes = plt.subplots(2, 3, figsize=(16, 7), squeeze=False)
        for col, ratio in enumerate((0.01, 0.1, 1.0)):
            cov_o, w_o = gaps[(ratio, k, 'oracle')]
            for name in COLORS:
                cov, w = gaps[(ratio, k, name)]
                pos = np.arange(1, k + 1)
                ls = ':' if name in ('left', 'right') else '-'
                axes[0, col].plot(pos, cov, ls=ls, c=COLORS[name], label=name)
                axes[1, col].plot(pos, w / w_o, ls=ls, c=COLORS[name])
            axes[0, col].axhline(0.8, c='black', lw=0.6, ls='--')
            axes[1, col].axhline(1.0, c='black', lw=0.6, ls='--')
            axes[0, col].set_title(f'local level, r/q = {ratio}, gap {k}')
            axes[0, col].set_ylabel('80% coverage')
            axes[1, col].set_ylabel('80% width / oracle width')
            axes[1, col].set_xlabel('position in the gap')
        axes[0, 0].legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(out.parent / 'figures' / f'extra_synthetic_oracle_gap{k}.png', dpi=150)
        plt.close(fig)
    print(df[df['gap'] == cfg['hidden_counts'][0]].to_string(index=False))


def run_asymmetry(cfg, pipe, n, n_boot, out):
    qs = np.array(cfg['quantiles'])
    batch = cfg['chronos']['batch_size']
    rows = []
    for process, gen in (('ar1_reversible', lambda r: ar1(n, r)), ('shot_noise_irreversible', lambda r: shot_noise(n, r))):
        for k in cfg['hidden_counts']:
            rng = np.random.default_rng([11, k, len(process)])
            y_full = gen(rng)
            y_full = y_full / y_full.std()
            idx = block_mask(n, k, rng)
            x = apply_mask(y_full, idx)
            y = np.take_along_axis(y_full, idx, 1)
            full = impute(pipe, x, idx, list(qs), batch)
            ml, mr, c = matched_forecasts(pipe, x, idx, qs, batch)
            draws = cluster_draws(np.arange(n), n_boot, rng)
            for context, QL, QR, ll, lr in (('full', full['left'], full['right'], idx[:, 0], LENGTH - idx[:, -1] - 1),
                                            ('matched', ml, mr, c, c)):
                SL, SR = window_scores(y, QL, qs), window_scores(y, QR, qs)
                for metric in ('mae', 'pinball', 'cov80', 'width80'):
                    M = np.stack([SL[metric], SR[metric]])
                    B = boot_means(M, draws)
                    lo, hi = interval(B[:, 0] - B[:, 1])
                    rows.append({'process': process, 'gap': k, 'context': context, 'metric': metric,
                                 'mean_left_len': round(float(np.mean(ll)), 1), 'mean_right_len': round(float(np.mean(lr)), 1),
                                 'left': M[0].mean(), 'right': M[1].mean(), 'left_minus_right': M[0].mean() - M[1].mean(),
                                 'lo': lo, 'hi': hi})
            print(f'asymmetry test: {process}, gap {k} done')
    df = pd.DataFrame(rows).round(4)
    df.to_csv(out / 'extra_synthetic_asymmetry.csv', index=False)
    print(df[(df['context'] == 'matched') & df['metric'].isin(['pinball', 'cov80'])].to_string(index=False))


def main():
    def extra(p):
        p.add_argument('--what', nargs='+', default=['oracle', 'asymmetry'], choices=['oracle', 'asymmetry'])
        p.add_argument('--n-windows', type=int, default=200)
        p.add_argument('--n-boot', type=int, default=1000)
    args = parse_args('Synthetic checks for the Chronos blend and for time reversal', extra)
    cfg = load_config(args.config)
    out = results_dir(cfg) / 'tables'
    out.mkdir(parents=True, exist_ok=True)
    (results_dir(cfg) / 'figures').mkdir(parents=True, exist_ok=True)
    pipe = load_pipe(cfg)
    if 'oracle' in args.what:
        run_oracle(cfg, pipe, args.n_windows, out)
    if 'asymmetry' in args.what:
        run_asymmetry(cfg, pipe, args.n_windows, args.n_boot, out)


if __name__ == '__main__':
    main()
