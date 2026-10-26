from . import evaluate as ev
from .common import load_config, parse_args, results_dir


def main():
    args = parse_args('Full metric table with CSDI trained for 100 epochs (csdi_long) added')
    cfg = load_config(args.config)
    ev.MODELS = {**ev.MODELS, 'csdi_long': 'CSDI-100'}
    out = results_dir(cfg) / 'tables'
    out.mkdir(parents=True, exist_ok=True)
    for split in ('val', 'test'):
        df = ev.score_split(cfg, split)[0]
        df.round(4).to_csv(out / f'extra_metrics_{split}.csv', index=False)
        print(f'{split}: {len(df)} rows written to extra_metrics_{split}.csv')


if __name__ == '__main__':
    main()
