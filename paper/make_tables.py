import sys
from pathlib import Path
import numpy as np
import pandas as pd

SRC = Path(sys.argv[1])
OUT = Path(sys.argv[2])
OUT.mkdir(exist_ok=True)
T = lambda n: pd.read_csv(SRC / f'{n}.csv')

NAMES = {'kalman': 'Kalman', 'csdi': 'CSDI-40', 'csdi_long': 'CSDI-100', 'chronos_blend': 'Chronos blend',
         'chronos_left': 'Chronos left', 'chronos_right': 'Chronos right'}
DS = {'ETT_h1': 'ETT\\_h1', 'Electricity': 'Electricity', 'BeijingAir': 'BeijingAir'}
MAIN4 = ['kalman', 'csdi', 'chronos_blend', 'csdi_long']
CELLS = [(d, p, h) for d in DS for h in (24, 128) for p in ('point', 'block')]
CELLS = [(d, p, h) for d in DS for p in ('point', 'block') for h in (24, 128)]
CELLS = [(d, p, h) for d in DS for h in (24, 128) for p in ('point', 'block')]
CELLS = [(d, p, h) for d in DS for p, h in (('point', 24), ('block', 24), ('point', 128), ('block', 128))]
nums = {}


def cell_label(d, p, h):
    return f'{DS[d]} {p} {h}'


def f(x, n=4):
    return f'{x:.{n}f}'.replace('-', '$-$')


def sg(x, n=3):
    return f'{x:+.{n}f}'.replace('-', '$-$')


def write(name, s):
    (OUT / name).write_text(s + '\n\\bottomrule\n')


m = T('extra_metrics_test')
q = T('extra_crps_test')
win = T('extra_winners_with_ties_long_test')
M = m.set_index(['dataset', 'pattern', 'hidden', 'model'])

# ---------- main table: pinball and MAE
lines = []
for d, p, h in CELLS:
    row = [cell_label(d, p, h)]
    w = win[(win.dataset == d) & (win.pattern == p) & (win.hidden == h)].iloc[0]
    tied = [] if w.tied_with_best == '-' else w.tied_with_best.split(',')
    for met in ('pinball', 'MAE'):
        if met == 'MAE':
            row.append('')
        vals = {k: M.loc[(d, p, h, k), met] for k in MAIN4}
        best = min(vals, key=vals.get)
        if met == 'pinball':
            assert best == w.best_pinball, (d, p, h, best, w.best_pinball)
        for k in MAIN4:
            s = f(vals[k])
            if k == best:
                s = f'\\textbf{{{s}}}'
            elif met == 'pinball' and k in tied:
                s += '$^\\dagger$'
            row.append(s)
    lines.append(' & '.join(row) + ' \\\\')
    if (p, h) == ('block', 128) and d != 'BeijingAir':
        lines.append('\\midrule')
write('main.tex', '\n'.join(lines))

# ---------- summary over 12 cells
rows = []
q21 = q.groupby('model').CRPS_q21.mean()
g = m.groupby('model')[['MAE', 'pinball', 'PCE', 'CCE', 'cov80', 'width80', 'SIW']].mean()
order = ['kalman', 'csdi', 'csdi_long', 'chronos_left', 'chronos_right', 'chronos_blend']
for k in order:
    r = g.loc[k]
    rows.append(' & '.join([NAMES[k], f(r.MAE, 3), f(r.pinball, 3), f(q21[k], 3), f(r.PCE, 3), f(r.CCE, 3), f(r.cov80, 3), f(r.width80, 3), f(r.SIW, 3)]) + ' \\\\')
write('summary.tex', '\n'.join(rows))
for k in order:
    nums[f'pin_{k}'] = g.loc[k].pinball
    nums[f'pce_{k}'] = g.loc[k].PCE
    nums[f'mae_{k}'] = g.loc[k].MAE

# ---------- per cell calibration (appendix)
rows = []
for d, p, h in CELLS:
    r = [cell_label(d, p, h)]
    for k in MAIN4:
        r.append(f"{M.loc[(d, p, h, k), 'cov80']:.3f}")
    for k in MAIN4:
        r.append(f"{M.loc[(d, p, h, k), 'PCE']:.3f}")
    for k in MAIN4:
        r.append(f"{M.loc[(d, p, h, k), 'width80']:.2f}")
    rows.append(' & '.join(r) + ' \\\\')
    if (p, h) == ('block', 128) and d != 'BeijingAir':
        rows.append('\\midrule')
write('calib_cells.tex', '\n'.join(rows))

# ---------- H1 table: ratio block/point at equal hidden counts
rows = []
h1 = {}
for k in ['kalman', 'csdi', 'csdi_long', 'chronos_blend']:
    r = [NAMES[k]]
    for met in ('MAE', 'CRPS'):
        pv = m[m.model == k].pivot_table(index=['dataset', 'hidden'], columns='pattern', values=met)
        ratio = pv.block / pv.point
        h1[(k, met)] = (int((ratio > 1).sum()), len(ratio))
        r.append(f'{(ratio > 1).sum()}/{len(ratio)}')
        r.append(f'{ratio.min():.2f}--{ratio.max():.2f}')
    rows.append(' & '.join(r) + ' \\\\')
write('h1.tex', '\n'.join(rows))
nums['h1_all'] = all(v[0] == v[1] for v in h1.values())

# ---------- pairwise bootstrap on pinball
d = T('extra_bootstrap_diffs_test')
d = d[d.metric == 'pinball']
pairs = [('csdi_long', 'chronos_blend'), ('csdi_long', 'kalman'), ('csdi_long', 'csdi'), ('csdi', 'kalman'),
         ('chronos_blend', 'kalman'), ('chronos_blend', 'csdi'), ('chronos_blend', 'chronos_left'), ('chronos_blend', 'chronos_right')]
rows = []
pair_res = {}
for a, b in pairs:
    g2 = d[(d['first'] == a) & (d['second'] == b)]
    assert len(g2) == 12, (a, b, len(g2))
    first = (g2.verdict == 'first better').sum()
    second = (g2.verdict == 'second better').sum()
    tie = (g2.verdict == 'tie').sum()
    pt, bl = g2[g2.pattern == 'point'], g2[g2.pattern == 'block']
    pair_res[(a, b)] = (first, tie, second)
    rows.append(f'{NAMES[a]} vs {NAMES[b]} & {first} & {tie} & {second} & '
                f"{(pt.verdict == 'first better').sum()}/{(pt.verdict == 'tie').sum()}/{(pt.verdict == 'second better').sum()} & "
                f"{(bl.verdict == 'first better').sum()}/{(bl.verdict == 'tie').sum()}/{(bl.verdict == 'second better').sum()} \\\\")
write('pairs.tex', '\n'.join(rows))

# ---------- blend variants: real data (12 cells) and synthetic
bv = T('extra_blend_variants_test')
gv = bv.groupby('variant')[['MAE', 'pinball', 'CRPS', 'PCE', 'CCE', 'cov80', 'width80']].mean()
so = T('extra_synthetic_oracle')
rows = []
for k, lab in (('left', 'Left only'), ('right', 'Right only'), ('linear', 'Linear quantile blend'), ('uniform', 'Equal-weight blend'), ('poe', 'Gaussian product')):
    r = gv.loc[k]
    s24 = so[(so.model == k) & (so.gap == 24)]
    s128 = so[(so.model == k) & (so.gap == 128)]
    rows.append(' & '.join([lab, f(r.pinball, 4), f(r.cov80, 3), f(r.width80, 3),
                            f'{s24.width_ratio_to_oracle.min():.2f}--{s24.width_ratio_to_oracle.max():.2f}',
                            f'{s24.cov80.min():.2f}--{s24.cov80.max():.2f}',
                            f'{s128.width_ratio_to_oracle.min():.2f}--{s128.width_ratio_to_oracle.max():.2f}',
                            f'{s128.cov80.min():.2f}--{s128.cov80.max():.2f}']) + ' \\\\')
write('variants.tex', '\n'.join(rows))
rr = so[so.model == 'linear']
nums['lin_ratio_min'] = rr.width_ratio_to_oracle.min()
nums['lin_ratio_max'] = rr.width_ratio_to_oracle.max()
nums['uni_ratio_min'] = so[so.model == 'uniform'].width_ratio_to_oracle.min()
nums['uni_ratio_max'] = so[so.model == 'uniform'].width_ratio_to_oracle.max()
nums['poe_ratio_min'] = so[so.model == 'poe'].width_ratio_to_oracle.min()
nums['poe_ratio_max'] = so[so.model == 'poe'].width_ratio_to_oracle.max()
nums['poe_cov_min'] = so[so.model == 'poe'].cov80.min()
nums['poe_cov_max'] = so[so.model == 'poe'].cov80.max()
nums['lin_cov_min'] = rr.cov80.min()
nums['lin_cov_max'] = rr.cov80.max()
nums['oracle_cov_min'] = so[so.model == 'oracle'].cov80.min()
nums['oracle_cov_max'] = so[so.model == 'oracle'].cov80.max()
nums['lr_ratio_min'] = so[so.model.isin(['left', 'right'])].width_ratio_to_oracle.min()
nums['lr_ratio_max'] = so[so.model.isin(['left', 'right'])].width_ratio_to_oracle.max()

# poe vs linear on real data: cells where poe cov80 < 0.75
pv = bv.pivot_table(index=['dataset', 'pattern', 'hidden'], columns='variant', values='cov80')
nums['poe_cov_real_min'] = pv.poe.min()
nums['poe_cov_real_max'] = pv.poe.max()
nums['lin_cov_real_min'] = pv.linear.min()
nums['lin_cov_real_max'] = pv.linear.max()
pp = bv.pivot_table(index=['dataset', 'pattern', 'hidden'], columns='variant', values='pinball')
nums['lin_beats_uni'] = int((pp.linear < pp.uniform).sum())
nums['lin_beats_poe'] = int((pp.linear < pp.poe).sum())

# ---------- time reversal (synthetic)
sa = T('extra_synthetic_asymmetry')
rows = []
for proc, lab in (('ar1_reversible', 'AR(1), $\\phi=0.9$ (reversible)'), ('shot_noise_irreversible', 'Jump process (irreversible)')):
    for gap in (24, 128):
        for ctx in ('full', 'matched'):
            def get(met):
                r = sa[(sa.process == proc) & (sa.gap == gap) & (sa.context == ctx) & (sa.metric == met)].iloc[0]
                return r
            pb, cv = get('pinball'), get('cov80')
            rows.append(' & '.join([lab if (gap, ctx) == (24, 'full') else '', str(gap), ctx,
                                    f'{pb.mean_left_len:.0f}/{pb.mean_right_len:.0f}',
                                    f'{pb.left:.3f}', f'{pb.right:.3f}',
                                    f'{sg(pb.left_minus_right)} [{sg(pb.lo)}, {sg(pb.hi)}]',
                                    f'{cv.left:.2f}', f'{cv.right:.2f}']) + ' \\\\')
    if proc == 'ar1_reversible':
        rows.append('\\midrule')
write('reversal.tex', '\n'.join(rows))

# real data matched
ml = T('extra_matched_lr_test')
rows = []
ex = {'full': 0, 'matched': 0}
dirs = {'full': [], 'matched': []}
for ds in DS:
    for gap in (24, 128):
        r = [DS[ds] if gap == 24 else '', str(gap)]
        for ctx in ('full', 'matched'):
            x = ml[(ml.dataset == ds) & (ml.block == gap) & (ml.context == ctx) & (ml.metric == 'pinball')].iloc[0]
            sig = not (x.lo <= 0 <= x.hi)
            ex[ctx] += int(sig)
            if sig:
                dirs[ctx].append((ds, gap, 'left' if x.left_minus_right < 0 else 'right'))
            r.append(f'{sg(x.left_minus_right)} [{sg(x.lo)}, {sg(x.hi)}]' + ('$^*$' if sig else ''))
        rows.append(' & '.join(r) + ' \\\\')
write('reversal_real.tex', '\n'.join(rows))
nums['rev_real_sig_full'] = ex['full']
nums['rev_real_sig_matched'] = ex['matched']
print('real reversal significant', dirs)

# ---------- CSDI budget
cv = T('extra_csdi_curves').set_index('run')
rows = []
imp = {}
long40 = pd.concat([m[m.model == 'csdi'], m[m.model == 'csdi_long']])
for ds in DS:
    a, b = cv.loc[ds], cv.loc[ds + '_long']
    pa = m[(m.model == 'csdi') & (m.dataset == ds)].set_index(['pattern', 'hidden']).pinball
    pb = m[(m.model == 'csdi_long') & (m.dataset == ds)].set_index(['pattern', 'hidden']).pinball
    better = int((pb < pa).sum())
    rows.append(' & '.join([DS[ds], f'{int(a.epochs)} / {int(a.best_epoch)}', f(a.best_val_loss),
                            f'{int(b.epochs)} / {int(b.best_epoch)}', f(b.best_val_loss),
                            f'{100 * (1 - b.best_val_loss / a.best_val_loss):.1f}\\%',
                            f(pa.mean()), f(pb.mean()), f'{better}/4']) + ' \\\\')
    imp[ds] = 100 * (1 - b.best_val_loss / a.best_val_loss)
write('budget.tex', '\n'.join(rows))
nums['val_imp_min'] = min(imp.values())
nums['val_imp_max'] = max(imp.values())
nums['csdi_gain'] = 100 * (1 - g.loc['csdi_long'].pinball / g.loc['csdi'].pinball)
nums['csdi_pce_40'] = g.loc['csdi'].PCE
nums['csdi_pce_100'] = g.loc['csdi_long'].PCE

# ---------- calibration facts
pv = m.pivot_table(index=['dataset', 'pattern', 'hidden'], columns='model', values='PCE')
nums['pce_improved'] = int((pv.csdi_long < pv.csdi).sum())
nums['pce_lowest_main'] = int((pv[MAIN4].idxmin(axis=1) == 'csdi_long').sum())
cc = m.pivot_table(index=['dataset', 'pattern', 'hidden'], columns='model', values='CCE')
nums['cce_pos_long'] = int((cc.csdi_long > 0).sum())
nums['cce_closer'] = int((cc.csdi_long.abs() < cc.csdi.abs()).sum())
wd = m.pivot_table(index=['dataset', 'pattern', 'hidden'], columns='model', values='width80')
nums['narrower'] = int((wd.csdi_long < wd.csdi).sum())
cov = m.pivot_table(index=['dataset', 'pattern', 'hidden'], columns='model', values='cov80')
nums['cov_long_min'] = cov.csdi_long.min()
nums['cov_long_max'] = cov.csdi_long.max()
nums['cov_blend_min'] = cov.chronos_blend.min()
nums['cov_blend_max'] = cov.chronos_blend.max()
nums['cov_kal_min'] = cov.kalman.min()
nums['cov_kal_max'] = cov.kalman.max()
cb = T('extra_bootstrap_cov80_test')
nums['cov_ci_cover'] = cb[cb.model == 'csdi_long']['covers_0.8'].sum() if 'csdi_long' in set(cb.model) else -1
print(cb.groupby('model')['covers_0.8'].agg(['sum', 'count']))

# per channel
pc = T('extra_per_channel_summary_test')
gpc = pc.groupby('model')['share_off_by_over_0.10'].mean()
for k in gpc.index:
    nums[f'offshare_{k}'] = gpc[k]

# CRPS rank sensitivity (three original models)
a = q[q.model.isin(['kalman', 'csdi', 'chronos_blend'])]
w1 = a.loc[a.groupby(['dataset', 'pattern', 'hidden']).pinball.idxmin()].set_index(['dataset', 'pattern', 'hidden']).model
w2 = a.loc[a.groupby(['dataset', 'pattern', 'hidden']).CRPS_q21.idxmin()].set_index(['dataset', 'pattern', 'hidden']).model
w3 = a.loc[a.groupby(['dataset', 'pattern', 'hidden']).CRPS_orig.idxmin()].set_index(['dataset', 'pattern', 'hidden']).model
nums['rank_diff_q21'] = int((w1 != w2).sum())
nums['rank_diff_orig'] = int((w1 != w3).sum())
nums['rank_diff_orig_cells'] = '; '.join(f'{i[0]} {i[1]} {i[2]}' for i in w1.index[(w1 != w3).values])

# NLL
nl = m[m.model == 'kalman'][['dataset', 'pattern', 'hidden', 'NLL']]
nums['nll_point_max'] = nl[nl.pattern == 'point'].NLL.max()
nums['nll_block_min'] = nl[nl.pattern == 'block'].NLL.min()

# winners
nums['best_count'] = int((win.best_pinball == 'csdi_long').sum())
nums['tied_best_cells'] = int(((win.best_pinball != 'csdi_long') & win.tied_with_best.str.contains('csdi_long')).sum())

with open(OUT / 'numbers.txt', 'w') as fh:
    for k, v in nums.items():
        fh.write(f'{k} = {v}\n')
print(open(OUT / 'numbers.txt').read())
