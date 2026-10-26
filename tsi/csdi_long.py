import math
import time
import zlib

import numpy as np
import torch
from pypots.imputation import CSDI
from pypots.optim import Adam
from pypots.optim.lr_scheduler import MultiStepLR

from .common import (load_config, load_masks, load_series, mask_configs, masked_windows, parse_args,
                     pred_path, results_dir, save_json)
from .csdi_imputer import draw_samples, training_windows


def build(cfg, name, lcfg):
    c = cfg['csdi']
    steps = math.ceil(c['n_train_windows'] / c['batch_size']) if lcfg.get('scheduler_unit', 'step') == 'step' else 1
    milestones = [int(f * lcfg['epochs'] * steps) for f in lcfg['milestones']]
    print(f'scheduler unit: {lcfg.get("scheduler_unit", "step")}, steps per epoch: {steps}, milestones: {milestones}')
    optimizer = Adam(lr=lcfg['lr'], weight_decay=lcfg['weight_decay'],
                     lr_scheduler=MultiStepLR(milestones=milestones, gamma=lcfg['gamma']))
    ckpt = results_dir(cfg) / 'csdi_long' / name / 'model.pypots'
    torch.manual_seed(lcfg['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(lcfg['seed'])
    model = CSDI(n_steps=cfg['window'], n_features=1, n_layers=c['n_layers'], n_heads=c['n_heads'],
                 n_channels=c['n_channels'], d_time_embedding=c['d_time_embedding'],
                 d_feature_embedding=c['d_feature_embedding'], d_diffusion_embedding=c['d_diffusion_embedding'],
                 n_diffusion_steps=c['n_diffusion_steps'], target_strategy='mix', batch_size=c['batch_size'],
                 epochs=lcfg['epochs'], patience=lcfg['patience'], optimizer=optimizer, device=c['device'],
                 saving_path=str(ckpt.parent / 'runs'), model_saving_strategy='best', verbose=True)
    return model, ckpt, milestones


def final_lr(model):
    try:
        return model.optimizer.torch_optimizer.param_groups[0]['lr']
    except Exception:
        return None


def main():
    def extra(p):
        p.add_argument('--skip-train', action='store_true')
    args = parse_args('CSDI with a longer schedule and step-wise learning rate decay (original CSDI recipe)', extra)
    cfg = load_config(args.config)
    lcfg, c = cfg['csdi_long'], cfg['csdi']
    L, edge, max_hidden = cfg['window'], cfg['edge'], max(cfg['hidden_counts'])
    qs = np.array(cfg['quantiles'])
    log = {}
    for name in lcfg['datasets']:
        series = load_series(cfg, name)
        model, ckpt, milestones = build(cfg, name, lcfg)
        if args.skip_train:
            model.load(str(ckpt))
        else:
            rng = np.random.default_rng([cfg['seeds']['train'], zlib.crc32(name.encode())])
            tr = training_windows(series, 'train', c['n_train_windows'], L, edge, max_hidden, rng)
            va = training_windows(series, 'val', c['n_val_windows'], L, edge, max_hidden, rng)
            va_x = np.where(va['mask'] == 1, va['X'], np.nan)
            t0 = time.time()
            model.fit({'X': tr['X'], 'for_pattern_mask': tr['mask']}, {'X': va_x, 'X_ori': va['X']})
            hours = (time.time() - t0) / 3600
            ckpt.parent.mkdir(parents=True, exist_ok=True)
            model.save(str(ckpt), overwrite=True)
            log[name] = {'train_hours': hours, 'best_val_loss': float(model.best_loss), 'epochs': lcfg['epochs'],
                         'milestones': milestones, 'final_lr': final_lr(model)}
            print(f'{name}: trained in {hours:.2f} h, final learning rate {final_lr(model)} (expected {lcfg["lr"] * lcfg["gamma"] ** 2:.0e})')
        torch.manual_seed(lcfg['seed'] + 1)
        for split in ('val', 'test'):
            masks = load_masks(cfg, split, name)
            out = {}
            t0 = time.time()
            for key in mask_configs(cfg):
                x, idx = masked_windows(series, masks, key)
                samples = draw_samples(model, x, c['n_samples'], c['sample_batch_size'])
                samples = np.take_along_axis(samples, idx[:, None, :], axis=2)
                out[f'{key}/samples'] = samples.astype(np.float32)
                out[f'{key}/quantiles'] = np.moveaxis(np.quantile(samples, qs, axis=1), 0, -1)
            np.savez(pred_path(cfg, split, name, 'csdi_long'), **out)
            log.setdefault(name, {})[f'{split}_sampling_hours'] = (time.time() - t0) / 3600
            print(f'{split}/{name}: sampling done in {(time.time() - t0) / 3600:.2f} h')
    if log:
        save_json(log, results_dir(cfg) / 'csdi_long' / ('log_skip_train.json' if args.skip_train else 'log.json'))


if __name__ == '__main__':
    main()
