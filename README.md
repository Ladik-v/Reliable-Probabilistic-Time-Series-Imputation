# Reliable Probabilistic Time Series Imputation

Code for the Models of Sequential Data project (Skoltech, 2026).
Authors: Vladislav Voronin, Kirill Lalayants, Mikhail Bezrukov, Fedor Zhirkov. Supervisor: Vitaliy Pozdnyakov.

**Question.** When are native predictive distributions reliable, and how does a two-sided
forecasting adaptation affect calibration?

**Hypotheses.** (1) Contiguous blocks are harder than scattered points. (2) Median accuracy and
calibration can disagree. (3) Blending left and right forecasts can change calibration in either
direction.

## Results in brief

Test pinball loss (nine levels, lower is better). Bold marks the lowest mean in a cell; a dagger marks
models that are tied with the best, meaning the paired 95% interval of the difference contains zero
(cluster bootstrap over channels and windows, 2000 draws). `CSDI-100` is CSDI trained for 100 epochs
with the original learning-rate schedule and is our main CSDI. `CSDI-40` is the first run
(40 epochs, early stopping), kept for comparison.

| Cell | Kalman | CSDI-40 | Chronos blend | CSDI-100 |
|---|---|---|---|---|
| ETT_h1 point 24 | 0.0882 | 0.0888 | 0.0832 | **0.0755** |
| ETT_h1 block 24 | 0.3520 | 0.1594 | **0.1282** | 0.1302 † |
| ETT_h1 point 128 | 0.1275 | 0.1152 | 0.1077 | **0.0895** |
| ETT_h1 block 128 | 0.3404 | 0.2560 | 0.1711 † | **0.1605** |
| Electricity point 24 | 0.0841 | 0.0705 | 0.0710 | **0.0578** |
| Electricity block 24 | 0.2678 | 0.1142 † | 0.1118 † | **0.1055** |
| Electricity point 128 | 0.1363 | 0.0869 | 0.0968 | **0.0734** |
| Electricity block 128 | 0.3449 | 0.1463 | 0.1580 † | **0.1300** |
| BeijingAir point 24 | 0.0450 | 0.0444 | 0.0495 | **0.0382** |
| BeijingAir block 24 | 0.1599 † | 0.1809 | 0.1612 † | **0.1482** |
| BeijingAir point 128 | 0.0677 | 0.0643 | 0.0831 | **0.0590** |
| BeijingAir block 128 | 0.3513 | 0.3387 | 0.3363 | **0.2968** |

* H1 holds for every model: at equal hidden counts, blocks have higher MAE and CRPS than points.
* CSDI-100 is best in 11 of 12 test cells and tied with the best in the twelfth. Against the
  Chronos blend it is significantly better in all six point cells and tied in five of the six block
  cells. It never loses significantly to Kalman or to the blend.
* CSDI-40 was under-trained: validation loss falls by 8-13% with the longer schedule on all three
  datasets, and test pinball improves in 11 of 12 cells.
* Calibration is not fixed by longer training. CSDI-100 covers 0.71-0.75 of the targets with its 80%
  intervals (nominal 0.8). The Chronos blend is close to nominal or above it in most cells and
  undercovers on long BeijingAir gaps (0.670).
* Quantile averaging (the blend) is wider than the exact conditional distribution: on synthetic
  local-level series its 80% intervals are 1.11-1.24 times the exact width. A Gaussian product of the
  two forecasts is too narrow (0.72-0.87 times, coverage 0.56-0.67).
* Time reversal: on an irreversible synthetic process (jumps with decay) the reversed forecast is
  worse than the forward one; on a reversible AR(1) process there is no difference. On the real data
  we found no consistent direction effect.

The main tables `results/tables/metrics_{val,test}.csv` and the poster were produced before the
longer CSDI run and contain CSDI-40 only. CSDI-100 and all new comparisons are in
`results/tables/extra_*.csv`. A full discussion with every number is in `REVIEW_revised.pdf`
(peer review and author response).

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

Tested on macOS (Apple M4, 24 GB) with Python 3.13. Chronos-2 runs on CPU (faster than MPS at this
size); CSDI trains on MPS. On a CUDA machine set `csdi.device: cuda` in the config (see
`configs/long.yaml`).

The processed data are not in the repository. Run `python -m tsi.data --config configs/main.yaml`
first: `evaluate` and every `extra_*` script load `data/<dataset>.npz` to recover the targets and
the scaling, so saved predictions alone are not enough to rebuild the tables.

## Run

```bash
./run_all.sh configs/pilot.yaml   # small pilot: checks the pipeline end to end
./run_all.sh                      # main experiment (configs/main.yaml)
```

Steps (each can be run on its own; all read the same config):

| Step | Command | Output |
|---|---|---|
| Data | `python -m tsi.data` | `data/<dataset>.npz`: scaled series, split bounds, train-only mean/std |
| Masks | `python -m tsi.masks` | `results/masks/<split>/<dataset>.npz`: windows and hidden indices |
| Kalman | `python -m tsi.kalman` | `results/predictions/<split>/<dataset>/kalman.npz` |
| Chronos-2 | `python -m tsi.chronos_imputer` | `chronos_left.npz`, `chronos_right.npz`, `chronos_blend.npz` |
| CSDI-40 | `python -m tsi.csdi_imputer [--skip-train]` | `csdi.npz` (100 draws), checkpoint in `results/csdi/` |
| Evaluate | `python -m tsi.evaluate` | `results/tables/*.csv|md`, `results/figures/*.png` |

`evaluate` rebuilds every table and figure from saved predictions, without rerunning models.

### Additional analyses (after the peer review)

The new code is separate from the main pipeline and does not change its outputs. The run order,
with every command, is in `colab_extra.ipynb` (written for Google Colab with a GPU); `EXTRA.md` lists
the outputs.

| Analysis | Command | Output |
|---|---|---|
| CSDI-100, Electricity | `python -m tsi.csdi_long --config configs/long.yaml` | `csdi_long.npz`, `results/csdi_long/` |
| CSDI-100, ETT_h1 and BeijingAir | `python -m tsi.csdi_long --config configs/long2.yaml` | same |
| Blend variants (linear, equal weights, Gaussian product) | `python -m tsi.extra_blend` | `extra_blend_variants_*.csv` |
| Chronos with all 21 quantile levels, matched left/right context | `python -m tsi.extra_chronos --what q21 matched` | `chronos_*_q21.npz`, `chronos_leftm.npz`, `chronos_rightm.npz` |
| Synthetic oracle test and time-reversal test | `python -m tsi.extra_synthetic --n-windows 1000` | `extra_synthetic_*.csv` |
| Common-grid CRPS, bootstrap intervals, per-channel calibration, loss curves | `python -m tsi.extra_eval` | `extra_*.csv`, `extra_*.png` |

## Protocol

* **Data.** ETT_h1 (7 channels, hourly); Electricity (2012-2014, hourly means, the first 20 clients
  by id without zero runs longer than 24 h in the train part); BeijingAir (PM2.5 at 12 stations,
  hourly). Loaded with `tsdb`; raw data are not committed.
* **Split.** Chronological 60/20/20 train/validation/test per dataset. Scaling, Kalman variances and
  CSDI use train only. Every window lies fully inside its split.
* **Windows and masks.** Windows of 256 steps without native missing values ("eligible complete
  windows"). In each window we hide 24 or 128 values, either scattered (`point`) or as one
  contiguous gap (`block`). The first and last 24 points stay observed. All four mask configs use
  the same windows, and all models use the same saved masks. Validation and test use different
  seeds. Only hidden targets are scored. 60 windows per dataset and split. The number of eligible
  windows per dataset is not reported yet.
* **Kalman smoother.** Local-level model; process and observation variances fitted per channel by
  maximum likelihood on the last 6000 train points. The predictive distribution of a missing
  measurement includes the observation variance. Parameter uncertainty is excluded.
* **CSDI (PyPOTS).** One model per dataset, pooled over channels (univariate windows), ~0.41M
  parameters, 50 diffusion steps, 6000 training windows, batch size 32. PyPOTS `mix` masking
  strategy: half of the training masks are our point/block masks (1-128 hidden values), half are
  PyPOTS random-ratio masks. 100 draws per window; the median and quantiles are empirical.
  * *CSDI-40:* PyPOTS default optimizer (Adam, constant learning rate 1e-3), at most 40 epochs,
    early stopping on masked validation windows with patience 5.
  * *CSDI-100:* Adam, learning rate 1e-3, divided by 10 at 75% and at 90% of the training steps
    (steps 14,100 and 16,920), 100 epochs, no early stopping, the same training and validation
    windows. The original CSDI recipe uses 200 epochs and batch size 16; ours differs in both. A
    first attempt set the milestones in epochs; PyPOTS steps the scheduler after every batch, so
    the rate fell to 1e-5 inside the first epoch and the run was discarded (validation loss stayed
    near 0.37)
* **Chronos-2-small** (`autogluon/chronos-2-small`, frozen). For every run of hidden values:
  forecast from the left prefix, and from the reversed right suffix; other hidden runs stay NaN in
  the context. The blend averages matching quantiles with linear weights
  `w_left(j) = (k - j) / (k + 1)`. This is a heuristic, not conditional inference.

## Metrics

Point: MAE, MSE, RMSE of the median (train-scaled units), MRE in original units (guarded against a
near-zero denominator). Probabilistic: pinball loss averaged over q = 0.1, 0.2, ..., 0.9 (nine levels);
CRPS in three labelled versions (Kalman: analytic; CSDI: from samples; Chronos: quantile-score
approximation, 2 x mean pinball over the 11 saved levels) and `CRPS_q21`, the same quantile
approximation on one 21-level grid for every model (Kalman from analytic quantiles, CSDI from
samples, Chronos from its native quantiles); NLL (Kalman only, it is the only model with a density).

Calibration: PCE = mean_q |q - c(q)|, the gap between nominal quantile level q and its empirical
frequency c(q). CCE = mean_s [s - coverage(s)] over central intervals s in {0.2, 0.4, 0.6, 0.8, 0.9}:
positive means the intervals undercover, negative means they overcover. 80% coverage and width.
SIW = mean_s mean(Width_s) / IQR_s(y), the width scaled by the spread of the targets; it is a size
measure, not a calibration score. Read coverage together with width: wide intervals cover more
without being better calibrated.

Quantile levels. The main run saved 11 levels: 0.05, 0.1, 0.2, ..., 0.9, 0.95. Chronos-2 outputs 21
levels natively (0.01, 0.05, 0.10, ..., 0.95, 0.99); the 21-level run is in `extra_chronos`.

Uncertainty of differences. `extra_eval` resamples channels, then windows inside each sampled
channel (2000 draws) and keeps the models' outputs on the same window together. A comparison is a tie
when the 95% interval of the paired difference contains zero.

## Checks and ablations

* Left-only vs right-only vs blended Chronos quantiles on identical targets
  (`figures/along_gap_*.png`: 80% coverage and width along the gap position).
* CSDI 50 vs 100 draws (`tables/csdi_draws_*.csv`).
* Scattered vs contiguous at equal hidden counts (24, 128).
* `figures/bridge.png`: analytical Gaussian random walk with noiseless endpoints and a 24-step gap;
  at the midpoint the blended quantiles are about 1.4 times wider than the exact bridge. This is an
  illustration; the synthetic oracle test below measures it.
* Synthetic oracle test (`extra_synthetic_oracle.csv`, `figures/extra_synthetic_oracle_gap*.png`):
  Gaussian local-level series, exact Kalman smoother as the reference, Chronos left, right and blend
  and the blend variants.
* Time-reversal test (`extra_synthetic_asymmetry.csv`): reversible AR(1) and irreversible jump
  process, left and right forecasts with equal context lengths and horizons. On the real data the
  same comparison is in `extra_matched_lr_*.csv`.
* CSDI training curves and the long-run comparison (`extra_csdi_curves.csv`, `extra_csdi_curves.png`).

## Reproducibility and compute

* Windows and masks use fixed NumPy seeds. PyTorch was not seeded in the CSDI-40 run, so its
  outputs are saved in the repository and a fresh fit will not match them bit for bit (MPS is also
  not fully deterministic). The CSDI-100 runs set `torch.manual_seed(0)` and the CUDA seed.
* Tables can be rebuilt exactly from the saved predictions and masks after `python -m tsi.data`.
* Compute. Main run on an Apple M4: CSDI-40 training took 0.35 h (ETT_h1), 0.54 h (Electricity) and
  0.37 h (BeijingAir); the first two are reconstructed from log timestamps because the run was
  interrupted (`results/csdi/log.json`). BeijingAir test sampling alone took about 0.58 h. The
  CSDI-100 runs and the additional analyses ran on Google Colab with an NVIDIA T4: 0.30 h of
  training per dataset and 0.10 h of sampling per split and dataset. The two setups differ, so the
  times are not comparable.

## Limitations

Synthetic masks on eligible complete windows only; channels and windows are dependent, so cases are
not independent repeats, and the bootstrap intervals rest on 7 to 20 channels per dataset; CSDI is
univariate and its long runs use one seed and 100 epochs (the best epochs are 92-97); Kalman is a
local-level model without seasonality; the synthetic tests cover a local-level process and two
control processes, not seasonal data; the numbers of eligible windows per dataset are not reported;
finite diffusion samples.

Chronos-2 pretraining data. We did not check whether ETT_h1, Electricity or BeijingAir (or the
periods we use as test) are part of the pretraining data of Chronos-2. This is unresolved; if there
is overlap, the Chronos results could be optimistic.

## Use of AI assistance

Consulting and assistance in making paper review.

## References

Bergmeir (2023); Du et al. (2024), TSI-Bench; Adler et al. (2026); Tashiro et al. (2021), CSDI;
Gneiting & Raftery (2007); PyPOTS and Chronos documentation.

## Individual contributions

| Author            | Contribution                                            |
| ----------------- | ------------------------------------------------------- |
| Vladislav Voronin | Additional model training                               |
| Kirill Lalayants  | Data collection, initial model training                 |
| Mikhail Bezrukov  | Initial model training                                  |
| Fedor Zhirkov     | Choosing metrics, verifying data and cross-checking res |
