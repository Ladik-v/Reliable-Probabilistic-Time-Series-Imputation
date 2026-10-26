import re

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .common import ROOT, load_config, load_masks, load_series, mask_configs, parse_args, results_dir
from .evaluate import COLORS, MODELS, load_pred, score_split, targets
from .extra_stats import (Q21, boot_means, cluster_draws, has_key, interval, quantiles_at, verdict,
                          window_scores)
from .metrics import crps_quantiles, level_index, reliability

MAIN = ['kalman', 'csdi', 'chronos_blend']
CORE = MAIN + ['chronos_left', 'chronos_right']
EXTRA_COLORS = {'csdi_long': '#8c1c2b'}
FILES = CORE + ['csdi_long', 'chronos_left_q21', 'chronos_right_q21', 'chronos_blend_q21', 'chronos_leftm', 'chronos_rightm']
PAIRS = [('chronos_blend', 'csdi'), ('chronos_blend', 'kalman'), ('csdi', 'kalman'),
         ('chronos_blend', 'chronos_left'), ('chronos_blend', 'chronos_right'), ('csdi_long', 'csdi'),
         ('csdi_long', 'chronos_blend'), ('csdi_long', 'kalman')]
SPLITS = ('val', 'test')


def tables_dir(cfg):
    d = results_dir(cfg) / 'tables'
    d.mkdir(parents=True, exist_ok=True)
    return d


def figs_dir(cfg):
    d = results_dir(cfg) / 'figures'
    d.mkdir(parents=True, exist_ok=True)
    return d


def cells(cfg, split):
    for name in cfg['datasets']:
        series, masks = load_series(cfg, name), load_masks(cfg, split, name)
        preds = {m: load_pred(cfg, split, name, m) for m in FILES}
        for key in mask_configs(cfg):
            y, mu, sd = targets(series, masks, key)
            yield name, key, series, masks, preds, y, mu, sd


def q11(cfg, preds, key, model):
    if not has_key(preds.get(model), key):
        return None
    return quantiles_at(preds[model], key, np.array(cfg['quantiles']))


def q21(preds, key, model):
    if model == 'kalman' or model.startswith('csdi'):
        return quantiles_at(preds[model], key, Q21) if has_key(preds.get(model), key) else None
    p = preds.get(f'{model}_q21')
    return p[f'{key}/quantiles'] if has_key(p, key) else None


def run_check(cfg):
    for split in SPLITS:
        df, _, _, _ = score_split(cfg, split)
        old = pd.read_csv(results_dir(cfg) / 'tables' / f'metrics_{split}.csv')
        on = ['dataset', 'pattern', 'hidden', 'model']
        m = df.merge(old, on=on, suffixes=('', '_saved'))
        cols = ['MAE', 'pinball', 'CRPS', 'PCE', 'CCE', 'cov80', 'width80', 'SIW']
        diffs = {c: float(np.nanmax(np.abs(m[c] - m[f'{c}_saved']))) for c in cols}
        print(f'{split}: {len(m)} rows compared with metrics_{split}.csv; max abs differences:',
              {c: f'{v:.1e}' for c, v in diffs.items()})


def run_crps(cfg):
    for split in SPLITS:
        old = pd.read_csv(results_dir(cfg) / 'tables' / f'metrics_{split}.csv')[['dataset', 'pattern', 'hidden', 'model', 'CRPS']]
        old = old.rename(columns={'CRPS': 'CRPS_orig'})
        qs = np.array(cfg['quantiles'])
        rows = []
        for name, key, series, masks, preds, y, mu, sd in cells(cfg, split):
            pattern, k = key.split('_')
            for model in CORE + ['csdi_long']:
                Q = q11(cfg, preds, key, model)
                if Q is None:
                    continue
                s = window_scores(y, Q, qs, 'crpsq11')
                Q2 = q21(preds, key, model)
                crps21 = np.nan if Q2 is None else crps_quantiles(y, Q2, Q21)
                rows.append({'dataset': name, 'pattern': pattern, 'hidden': int(k), 'model': model,
                             'MAE': s['mae'].mean(), 'pinball': s['pinball'].mean(),
                             'CRPS_q11': s['crpsq11'].mean(), 'CRPS_q21': crps21})
        df = pd.DataFrame(rows).merge(old, on=['dataset', 'pattern', 'hidden', 'model'], how='left')
        df = df[['dataset', 'pattern', 'hidden', 'model', 'MAE', 'pinball', 'CRPS_orig', 'CRPS_q11', 'CRPS_q21']].round(4)
        df.to_csv(tables_dir(cfg) / f'extra_crps_{split}.csv', index=False)
        win = []
        for (name, pattern, hidden), g in df[df['model'].isin(MAIN)].groupby(['dataset', 'pattern', 'hidden'], sort=False):
            row = {'dataset': name, 'pattern': pattern, 'hidden': hidden}
            for c in ('MAE', 'pinball', 'CRPS_orig', 'CRPS_q21'):
                row[f'best_{c}'] = g.loc[g[c].idxmin(), 'model'] if g[c].notna().all() else None
            win.append(row)
        win = pd.DataFrame(win)
        win.to_csv(tables_dir(cfg) / f'extra_winners_{split}.csv', index=False)
        ok = win['best_CRPS_q21'].notna()
        if not ok.any():
            print(f'{split}: CRPS_q21 not available yet, run tsi.extra_chronos --what q21 first')
        else:
            diff = (win.loc[ok, 'best_CRPS_orig'] != win.loc[ok, 'best_CRPS_q21']).sum()
            print(f'{split}: winner changes between original CRPS and CRPS_q21 in {diff} of {int(ok.sum())} cells')


def run_channels(cfg):
    qs = np.array(cfg['quantiles'])
    for split in SPLITS:
        rows = []
        for name, key, series, masks, preds, y, mu, sd in cells(cfg, split):
            pattern, k = key.split('_')
            ch = masks[f'{key}/channel']
            names = series['channels']
            for model in MAIN:
                Q = q11(cfg, preds, key, model)
                if Q is None:
                    continue
                s = window_scores(y, Q, qs)
                for c in np.unique(ch):
                    sel = ch == c
                    rows.append({'dataset': name, 'pattern': pattern, 'hidden': int(k), 'model': model,
                                 'channel': str(names[c]), 'n_windows': int(sel.sum()),
                                 'cov80': s['cov80'][sel].mean(), 'width80': s['width80'][sel].mean(),
                                 'PCE': float(np.abs(qs - reliability(y[sel], Q[sel], qs)).mean()),
                                 'MAE': s['mae'][sel].mean()})
        df = pd.DataFrame(rows)
        df.round(4).to_csv(tables_dir(cfg) / f'extra_per_channel_{split}.csv', index=False)
        summ = []
        for (name, pattern, hidden, model), d in df.groupby(['dataset', 'pattern', 'hidden', 'model'], sort=False):
            summ.append({'dataset': name, 'pattern': pattern, 'hidden': hidden, 'model': model,
                         'n_channels': len(d), 'min_windows': d['n_windows'].min(), 'max_windows': d['n_windows'].max(),
                         'cov80_pooled': np.average(d['cov80'], weights=d['n_windows']), 'cov80_min': d['cov80'].min(),
                         'cov80_median': d['cov80'].median(), 'cov80_max': d['cov80'].max(),
                         'cov80_std': d['cov80'].std(ddof=0),
                         'share_off_by_over_0.10': float((np.abs(d['cov80'] - 0.8) > 0.10).mean()),
                         'PCE_min': d['PCE'].min(), 'PCE_max': d['PCE'].max()})
        summ = pd.DataFrame(summ)
        summ.round(4).to_csv(tables_dir(cfg) / f'extra_per_channel_summary_{split}.csv', index=False)
        fig, axes = plt.subplots(1, len(cfg['datasets']), figsize=(6 * len(cfg['datasets']), 4.5), squeeze=False)
        keys = mask_configs(cfg)
        rng = np.random.default_rng(0)
        for ax, name in zip(axes[0], cfg['datasets']):
            for mi, model in enumerate(MAIN):
                for ki, key in enumerate(keys):
                    pattern, k = key.split('_')
                    d = df[(df['dataset'] == name) & (df['pattern'] == pattern) & (df['hidden'] == int(k)) & (df['model'] == model)]
                    x = ki + (mi - 1) * 0.25 + rng.uniform(-0.05, 0.05, len(d))
                    ax.scatter(x, d['cov80'], s=14, c=COLORS[model], alpha=0.8, label=MODELS[model] if ki == 0 else None)
            ax.axhline(0.8, c='black', lw=0.8, ls='--')
            ax.set_xticks(range(len(keys)))
            ax.set_xticklabels([k.replace('_', ' ') for k in keys], fontsize=8)
            ax.set_title(f'{name}: 80% coverage per channel')
            ax.set_ylabel('coverage')
        axes[0, 0].legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(figs_dir(cfg) / f'extra_per_channel_{split}.png', dpi=150)
        plt.close(fig)
        print(f'{split}: per-channel tables and figure written')


def cell_matrix(cfg, preds, key, y):
    qs = np.array(cfg['quantiles'])
    rows, labels = [], []
    for model in CORE + ['csdi_long']:
        Q = q11(cfg, preds, key, model)
        if Q is None:
            continue
        s = window_scores(y, Q, qs)
        for metric in ('mae', 'pinball', 'cov80'):
            rows.append(s[metric]), labels.append((model, metric))
        Q2 = q21(preds, key, model)
        if Q2 is not None:
            rows.append(window_scores(y, Q2, Q21, 'crpsq21')['crpsq21']), labels.append((model, 'crpsq21'))
    return np.stack(rows), labels


def run_bootstrap(cfg, n_boot):
    rng = np.random.default_rng(0)
    for split in SPLITS:
        diff_rows, cov_rows, win_rows, long_rows = [], [], [], []
        for name, key, series, masks, preds, y, mu, sd in cells(cfg, split):
            pattern, k = key.split('_')
            base = {'dataset': name, 'pattern': pattern, 'hidden': int(k)}
            M, labels = cell_matrix(cfg, preds, key, y)
            pos = {l: i for i, l in enumerate(labels)}
            draws = cluster_draws(masks[f'{key}/channel'], n_boot, rng)
            B = boot_means(M, draws)
            point = M.mean(axis=1)
            for model in CORE + ['csdi_long']:
                if (model, 'cov80') in pos:
                    i = pos[(model, 'cov80')]
                    lo, hi = interval(B[:, i])
                    cov_rows.append({**base, 'model': model, 'cov80': point[i], 'lo': lo, 'hi': hi,
                                     'covers_0.8': bool(lo <= 0.8 <= hi)})
            for a, b in PAIRS:
                for metric in ('mae', 'pinball', 'crpsq21'):
                    if (a, metric) not in pos or (b, metric) not in pos:
                        continue
                    ia, ib = pos[(a, metric)], pos[(b, metric)]
                    lo, hi = interval(B[:, ia] - B[:, ib])
                    diff_rows.append({**base, 'first': a, 'second': b, 'metric': metric,
                                      'diff_first_minus_second': point[ia] - point[ib], 'lo': lo, 'hi': hi,
                                      'verdict': verdict(lo, hi)})
            ip = [pos[(m, 'pinball')] for m in MAIN if (m, 'pinball') in pos]
            if len(ip) == len(MAIN):
                best = int(np.argmin(point[ip]))
                tied = [MAIN[i] for i in range(len(MAIN)) if i != best and
                        verdict(*interval(B[:, ip[i]] - B[:, ip[best]])) == 'tie']
                win_rows.append({**base, 'best_pinball': MAIN[best], 'tied_with_best': ','.join(tied) or '-'})
            ML = MAIN + ['csdi_long']
            if all((m, 'pinball') in pos for m in ML):
                ip = [pos[(m, 'pinball')] for m in ML]
                best = int(np.argmin(point[ip]))
                tied = [ML[i] for i in range(len(ML)) if i != best and
                        verdict(*interval(B[:, ip[i]] - B[:, ip[best]])) == 'tie']
                long_rows.append({**base, 'best_pinball': ML[best], 'tied_with_best': ','.join(tied) or '-'})
        pd.DataFrame(diff_rows).round(4).to_csv(tables_dir(cfg) / f'extra_bootstrap_diffs_{split}.csv', index=False)
        pd.DataFrame(cov_rows).round(4).to_csv(tables_dir(cfg) / f'extra_bootstrap_cov80_{split}.csv', index=False)
        pd.DataFrame(win_rows).to_csv(tables_dir(cfg) / f'extra_winners_with_ties_{split}.csv', index=False)
        pd.DataFrame(long_rows).to_csv(tables_dir(cfg) / f'extra_winners_with_ties_long_{split}.csv', index=False)
        d = pd.DataFrame(diff_rows)
        d = d[d['metric'] == 'pinball']
        print(f'{split}: pinball comparisons: ' + ', '.join(f'{v}={n}' for v, n in d['verdict'].value_counts().items()))


def run_matched(cfg, n_boot):
    rng = np.random.default_rng(0)
    qs = np.array(cfg['quantiles'])
    for split in SPLITS:
        rows = []
        for name, key, series, masks, preds, y, mu, sd in cells(cfg, split):
            if not key.startswith('block') or preds['chronos_leftm'] is None:
                continue
            pattern, k = key.split('_')
            idx = masks[f'{key}/idx']
            L = masks['window'].item()
            full_l, full_r = idx[:, 0], L - (idx[:, -1] + 1)
            ctx = preds['chronos_leftm'][f'{key}/ctx']
            draws = cluster_draws(masks[f'{key}/channel'], n_boot, rng)
            for context, ml, mr, len_l, len_r in (('full', 'chronos_left', 'chronos_right', full_l, full_r),
                                                  ('matched', 'chronos_leftm', 'chronos_rightm', ctx, ctx)):
                SL = window_scores(y, preds[ml][f'{key}/quantiles'], qs)
                SR = window_scores(y, preds[mr][f'{key}/quantiles'], qs)
                for metric in ('mae', 'pinball', 'cov80', 'width80'):
                    M = np.stack([SL[metric], SR[metric]])
                    B = boot_means(M, draws)
                    lo, hi = interval(B[:, 0] - B[:, 1])
                    rows.append({'dataset': name, 'block': int(k), 'context': context, 'metric': metric,
                                 'mean_left_len': round(float(len_l.mean()), 1), 'mean_right_len': round(float(len_r.mean()), 1),
                                 'left': M[0].mean(), 'right': M[1].mean(), 'left_minus_right': M[0].mean() - M[1].mean(),
                                 'lo': lo, 'hi': hi})
        if rows:
            pd.DataFrame(rows).round(4).to_csv(tables_dir(cfg) / f'extra_matched_lr_{split}.csv', index=False)
            print(f'{split}: matched left/right table written')


def parse_curves(path):
    runs, cur = [], []
    pat = re.compile(r'Epoch\s+(\d+)\s+-\s+training loss \(default\):\s+([\d.]+),\s+validation loss:\s+([\d.]+)')
    for line in open(path, errors='ignore'):
        m = pat.search(line)
        if m:
            e = int(m.group(1))
            if cur and e <= cur[-1][0]:
                cur = []
            cur.append((e, float(m.group(2)), float(m.group(3))))
            continue
        s = re.search(r'Saved the model to .*?/(?:csdi|csdi_long)/([^/]+)/', line)
        if s and cur:
            runs.append({'name': s.group(1) + ('_long' if 'csdi_long' in line else ''), 'rows': cur})
            cur = []
    return runs


def run_curves(cfg):
    logs = sorted((ROOT / 'logs').glob('*.log'))
    curves = {}
    for p in logs:
        for r in parse_curves(p):
            curves[r['name']] = r['rows']
    if not curves:
        print('no CSDI training logs found')
        return
    fig, axes = plt.subplots(1, len(curves), figsize=(5 * len(curves), 4), squeeze=False)
    for ax, (name, rows) in zip(axes[0], curves.items()):
        e, tr, va = zip(*rows)
        ax.plot(e, tr, label='train loss')
        ax.plot(e, va, label='validation loss')
        ax.set_title(f'CSDI {name}: best epoch {e[int(np.argmin(va))]} of {e[-1]}')
        ax.set_xlabel('epoch')
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(figs_dir(cfg) / 'extra_csdi_curves.png', dpi=150)
    plt.close(fig)
    out = [{'run': n, 'epochs': r[-1][0], 'best_epoch': r[int(np.argmin([x[2] for x in r]))][0],
            'best_val_loss': min(x[2] for x in r), 'last_val_loss': r[-1][2]} for n, r in curves.items()]
    pd.DataFrame(out).round(4).to_csv(tables_dir(cfg) / 'extra_csdi_curves.csv', index=False)
    print(pd.DataFrame(out).round(4).to_string(index=False))


def main():
    steps = ['check', 'crps', 'channels', 'bootstrap', 'matched', 'curves']

    def extra(p):
        p.add_argument('--what', nargs='+', default=steps, choices=steps)
        p.add_argument('--n-boot', type=int, default=2000)
    args = parse_args('Extra evaluation from saved predictions', extra)
    cfg = load_config(args.config)
    if 'check' in args.what:
        run_check(cfg)
    if 'crps' in args.what:
        run_crps(cfg)
    if 'channels' in args.what:
        run_channels(cfg)
    if 'bootstrap' in args.what:
        run_bootstrap(cfg, args.n_boot)
    if 'matched' in args.what:
        run_matched(cfg, args.n_boot)
    if 'curves' in args.what:
        run_curves(cfg)


if __name__ == '__main__':
    main()
