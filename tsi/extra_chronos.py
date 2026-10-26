import argparse
import time

import numpy as np
from chronos import Chronos2Pipeline

from .chronos_imputer import forecast, impute
from .common import load_config, load_masks, load_series, mask_configs, masked_windows, parse_args, pred_path
from .extra_stats import Q21
from .metrics import level_index

SIDES = ('left', 'right', 'blend')


def matched_forecasts(pipe, x, idx, qs, batch):
    n, k = idx.shape
    assert (idx[:, -1] - idx[:, 0] + 1 == k).all()
    s, e = idx[:, 0], idx[:, -1] + 1
    c = np.minimum(s, x.shape[1] - e)
    left = forecast(pipe, [x[i, s[i] - c[i]:s[i]] for i in range(n)], k, list(qs), batch)
    right = forecast(pipe, [x[i, e[i]:e[i] + c[i]][::-1].copy() for i in range(n)], k, list(qs), batch)[:, ::-1]
    return left, right, c


def run_q21(cfg, pipe):
    batch = cfg['chronos']['batch_size']
    qs = np.array(cfg['quantiles'])
    worst = 0.0
    for split in ('val', 'test'):
        for name in cfg['datasets']:
            series, masks = load_series(cfg, name), load_masks(cfg, split, name)
            outs = {s: {} for s in SIDES}
            t0 = time.time()
            for key in mask_configs(cfg):
                x, idx = masked_windows(series, masks, key)
                for side, q in impute(pipe, x, idx, list(Q21), batch).items():
                    outs[side][f'{key}/quantiles'] = q
                    old = np.load(pred_path(cfg, split, name, f'chronos_{side}'))[f'{key}/quantiles']
                    shared = [level_index(Q21, q_) for q_ in qs]
                    worst = max(worst, float(np.abs(q[..., shared] - old).max()))
            for side, out in outs.items():
                np.savez(pred_path(cfg, split, name, f'chronos_{side}_q21'), **out)
            print(f'{split}/{name}: q21 done in {time.time() - t0:.1f}s')
    print(f'max abs difference to the saved 11-level quantiles (scaled units): {worst:.2e}')


def run_matched(cfg, pipe):
    batch = cfg['chronos']['batch_size']
    qs = list(cfg['quantiles'])
    for split in ('val', 'test'):
        for name in cfg['datasets']:
            series, masks = load_series(cfg, name), load_masks(cfg, split, name)
            left_out, right_out = {}, {}
            t0 = time.time()
            for key in mask_configs(cfg):
                if not key.startswith('block'):
                    continue
                x, idx = masked_windows(series, masks, key)
                left, right, c = matched_forecasts(pipe, x, idx, qs, batch)
                left_out[f'{key}/quantiles'], right_out[f'{key}/quantiles'] = left, right
                left_out[f'{key}/ctx'] = right_out[f'{key}/ctx'] = c
            np.savez(pred_path(cfg, split, name, 'chronos_leftm'), **left_out)
            np.savez(pred_path(cfg, split, name, 'chronos_rightm'), **right_out)
            print(f'{split}/{name}: matched context done in {time.time() - t0:.1f}s')


def load_pipe(cfg):
    return Chronos2Pipeline.from_pretrained(cfg['chronos']['model'], device_map=cfg['chronos']['device'])


def main():
    def extra(p):
        p.add_argument('--what', nargs='+', default=['q21', 'matched'], choices=['q21', 'matched'])
    args = parse_args('Extra Chronos runs: 21-level quantiles and matched-context left/right', extra)
    cfg = load_config(args.config)
    pipe = load_pipe(cfg)
    if 'q21' in args.what:
        run_q21(cfg, pipe)
    if 'matched' in args.what:
        run_matched(cfg, pipe)


if __name__ == '__main__':
    main()
