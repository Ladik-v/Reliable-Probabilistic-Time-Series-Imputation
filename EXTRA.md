# Extra experiments (response to peer review)

New files only; the original pipeline and its numbers are untouched. Run order is in `colab_extra.ipynb`.

| Step | Command | Output (under results/) |
|---|---|---|
| Reproduction check | `python -m tsi.extra_eval --what check` | prints max abs difference to `metrics_*.csv` |
| Blend variants | `python -m tsi.extra_blend` | `tables/extra_blend_variants_*.csv`, `figures/extra_blend_variants_*.png` |
| Chronos 21 levels, matched context | `python -m tsi.extra_chronos --what q21 matched` | `predictions/*/*/chronos_*_q21.npz`, `chronos_leftm.npz`, `chronos_rightm.npz` |
| Synthetic oracle and time reversal | `python -m tsi.extra_synthetic` | `tables/extra_synthetic_*.csv`, `figures/extra_synthetic_oracle_gap*.png` |
| Common-scale CRPS, per-channel calibration, paired cluster bootstrap, matched left/right, loss curves | `python -m tsi.extra_eval` | `tables/extra_*.csv`, `figures/extra_*.png` |
| Longer CSDI on Electricity | `python -m tsi.csdi_long --config configs/long.yaml` | `predictions/*/Electricity/csdi_long.npz`, `csdi_long/` |

Notes
* Chronos-2 natively predicts 21 quantile levels (0.01, 0.05, 0.10 ... 0.95, 0.99). The main run used an 11-level subset; `CRPS_q21` scores every model on the 21-level grid (Kalman analytic quantiles, CSDI from samples).
* Bootstrap resamples channels, then windows inside each sampled channel; 2000 draws, 95% intervals. A comparison is a tie when the interval of the difference contains zero.
* `blend_variants`: `linear` reproduces the saved blend, `uniform` uses weight 0.5, `poe` combines two Gaussians (mean and sigma from the 0.1, 0.5, 0.9 quantiles) by precision weights. `poe` is an ablation, not part of the main method.
* Per-channel tables have 2 to 9 windows per channel, so read them as spread, not as per-channel estimates.
* `csdi_long` keeps the original CSDI recipe: Adam, learning rate 1e-3 decayed 10x at 75% and 90% of the epochs; torch seed is fixed.
