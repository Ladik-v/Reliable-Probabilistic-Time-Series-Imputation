# Week 5: peer review and author response (revised)

Project: *Reliable Probabilistic Time Series Imputation*. Repository: nerfff415/mosd_project.

**About this version.** Part 1 is the peer review as written on 20 October for the repository at that time, unchanged. Part 2 replaces the planned revisions with what was done after the review, and says what is still open. Part 3 gives the new results. All new numbers come from the `results/tables/extra_*.csv` files and the logs listed at the end. The original tables `metrics_{val,test}.csv` are not regenerated and still contain the 40-epoch CSDI.

---

## Part 1. Peer review

### 1. Problem, method and contribution

The project asks whether imputation distributions cover missing values at their stated probability levels, how wide the intervals are, and how adapting a forecaster to use both sides affects calibration.

Three model families are compared:

* A local-level Kalman smoother with Gaussian observation and process noise. Variances are fitted separately for each channel by maximum likelihood on the last 6,000 training points. Predictions include observation noise but exclude uncertainty in the fitted parameters.
* CSDI through PyPOTS, with one model per dataset trained on pooled univariate windows. Each model has 413,521 trainable parameters and uses 50 diffusion steps; evaluation retains 100 draws per window. It does not use other channels as simultaneous inputs.
* Frozen autogluon/chronos-2-small. Each contiguous run of hidden values is forecast from the left prefix and from the reversed right suffix. Matching quantiles are then averaged, with more weight assigned to the nearer side. Other hidden runs remain NaN in the context. This blend is a heuristic; it does not calculate the distribution conditional on both sides jointly.

The main configuration and data-loading log give the following dataset sizes:

| Dataset | Channels | Hourly time steps | Native missing values |
|---|---|---|---|
| ETT_h1 | 7 | 17,420 | 0% |
| Electricity | 20 | 26,305 | 0% |
| BeijingAir, PM2.5 | 12 stations | 35,064 | 2.077% |

Electricity is aggregated from 15-minute observations to hourly means, starting in 2012. Client selection uses only the training period and excludes clients with zero runs longer than 24 hours. All datasets use a chronological 60/20/20 split and channel-wise scaling fitted on training data.

Each dataset and evaluation split has 60 windows of length 256. Only windows without native missing values are eligible. The same windows receive all four masks: 24 or 128 hidden values, either scattered (point) or contiguous (block). The first and last 24 positions stay observed. Thus the experiment covers interpolation with context on both sides, rather than gaps at the series boundaries. The hidden fractions are 9.4% and 50% of a window. Metrics are computed only on hidden targets.

The hypotheses are that blocks are harder than scattered points (H1), median accuracy and calibration can disagree (H2), and the blend can change calibration in either direction (H3). The contribution is a comparison under shared masks, with left/right ablations and along-gap diagnostics.

### 2. Strengths

In `tsi/masks.py`, windows are sampled once per dataset and split, then reused across masks and models. Each window lies entirely inside its split; scaling and Kalman fitting use training data. CSDI uses validation windows for early stopping. Validation and test masks use different seeds.

The evaluation reports coverage together with width and SIW. This exposes cases in which high coverage comes with very broad intervals. On Electricity block 128, for example, Kalman has test 80% coverage of 0.948 and SIW of 2.385, compared with CSDI's 0.692 and 0.412. The broader intervals should not be described as better calibrated simply because they contain more targets.

H1 is consistent across the reported settings. At a fixed hidden count, every model has higher MAE and CRPS for blocks than for scattered points, in both validation and test. For Chronos blend with 24 hidden values, test MAE rises from 0.209 to 0.330 on ETT_h1, 0.178 to 0.283 on Electricity, and 0.114 to 0.422 on BeijingAir. These comparisons use the same underlying windows.

The Chronos ablation provides direct evidence for changes caused by the adaptation. On Electricity point 24, blending improves median MAE while moving 80% coverage further above its nominal level:

| Chronos variant | Test MAE | 80% coverage | 80% width |
|---|---|---|---|
| Left-only | 0.221 | 0.815 | 0.734 |
| Right-only | 0.212 | 0.814 | 0.742 |
| Blend | 0.178 | 0.880 | 0.733 |

Here the blend's extra coverage does not come from a larger mean interval width. The change in predicted location also matters. The along-gap figures complement this aggregate comparison by showing coverage and width at each position within a block.

Using the first 50 of CSDI's saved 100 draws changes mean 80% width by about 0.9-2.8% across validation and test settings; the largest absolute CRPS change is about 0.0062. This checks sensitivity to draw count within one run, rather than variation across independent fits or sample batches.

### 3. Major concerns

**M1. The tables do not quantify uncertainty in model differences.** Each row aggregates 60 windows, and nearby windows from the same channel can overlap or be dependent. The 1,440 or 7,680 hidden targets per row are therefore not independent repetitions. Some differences are small: Electricity point 24 test CRPS is 0.127 for the blend and 0.129 for CSDI; BeijingAir point 24 is 0.082 for CSDI and 0.083 for Kalman. These are descriptive rankings, without evidence that the ordering would persist under another window sample. Paired intervals should preserve the common windows across models and account for channel and temporal dependence. Agreement between validation and test is useful supporting evidence, but it does not replace those intervals.

**M2. The CRPS approximations differ across model families.** `tsi/metrics.py` uses analytic Gaussian CRPS for Kalman, a sample-based empirical-distribution score for CSDI, and twice the unweighted mean pinball loss over 11 quantile levels for Chronos. All target the same scoring principle, but the finite-sample and quantile approximations need not introduce the same error. In particular, Chronos uses an unevenly spaced grid from 0.05 to 0.95, without integration weights or tail treatment.

The existing common pinball metric uses nine levels, 0.1 through 0.9. Among Kalman, CSDI and Chronos blend, its test winner differs from the reported CRPS winner in two of twelve settings: Electricity point 24 and BeijingAir block 24. This is a reason to check ranking sensitivity, rather than proof that the estimators alone explain the reversal: nine-level pinball and CRPS also weight the distribution differently. Use common pinball as the main cross-model comparison, and apply the same 11-level approximation to every model as a sensitivity check. Keep analytic and sample CRPS with their method labels.

**M3. H3 needs a clearer distinction between observed behaviour and its explanation.** On BeijingAir point 24, the blend's test 80% coverage is 0.903, compared with 0.821 for left-only and 0.797 for right-only. This is direct evidence that blending can worsen overcoverage. On BeijingAir block 128, blend coverage is 0.670, compared with 0.677 and 0.615 for the one-sided variants: undercoverage is already present before blending. The latter result does not establish that the averaging rule created the undercoverage.

The Gaussian bridge figure shows a separate, analytically tractable case in which quantile blending is wider than exact conditioning. It does not validate a causal explanation for Chronos on real data. Forecast error, asymmetry between the two contexts and the averaging rule can all affect the result. The report should state which effects the ablation measures and which remain hypotheses.

**M4. CSDI conclusions are specific to the univariate setup and training budget.** Each input has `n_features=1`, so the model cannot exploit simultaneous relationships between stations or clients. Training uses 6,000 windows, 200 validation windows, a maximum of 40 epochs and patience 5. The training masks mix the project's point/block masks with PyPOTS random-ratio masks; they are not restricted to the four evaluation masks.

The logs select epoch 26 for ETT_h1, epoch 39 for Electricity, and epoch 30 for BeijingAir. Electricity reaches the epoch limit with a late improvement, which justifies checking a larger budget; it does not establish under-training for all three datasets. CSDI's 80% coverage is below 0.8 in every reported validation and test setting, ranging from 0.523 to 0.765. The supported conclusion is that this univariate implementation undercovers under these settings. More draws alone would not address possible training or model-specification effects.

**M5. Kalman represents a simple local-level baseline.** It models neither daily seasonality nor cross-channel relationships. On ETT_h1 block 24, its test MAE is 0.862 versus 0.330 for the blend; on Electricity block 128, its 80% width is 4.820 versus 1.564 for the blend. These results describe the selected random-walk model, rather than classical probabilistic imputation as a whole. The baseline is still useful because its assumptions and conditional distribution are explicit. A seasonal state-space model would test whether the gap is mainly due to those assumptions.

### 4. Minor concerns and reporting details

* **m1. Randomness.** The code fixes NumPy generators for windows and masks but does not explicitly seed PyTorch for CSDI training and sampling. Add the seed and document any remaining MPS nondeterminism. Repeated training runs would be needed to assess variability across fits.
* **m2. Runtime reporting.** `results/csdi/log.json` reconstructs ETT_h1 and Electricity training times from log timestamps after an interrupted run. Keep this qualification in any runtime table. Training and sampling should be reported separately; BeijingAir test sampling alone took about 0.58 hours, compared with about 0.37 hours for training.
* **m3. Metric definitions.** Explain that positive CCE means undercoverage and negative CCE means overcoverage, averaged over five central interval levels. PCE measures quantile calibration. SIW normalizes widths by the corresponding central spread of the scored targets; it is not a direct calibration score. NLL is available only for Kalman and should be labelled accordingly.
* **m4. Rebuilding results.** Raw and processed data are absent from the repository. The evaluator loads `data/<dataset>.npz` to recover targets and scaling, so saved predictions alone are insufficient. Document `python -m tsi.data --config configs/main.yaml` as a prerequisite for rebuilding tables, and record data versions or checksums.
* **m5. Table presentation.** The CSV already follows the loop order point 24, block 24, point 128, block 128. A compact comparison table would still help readers, but any marked best result should remain a descriptive minimum until paired uncertainty estimates are available.
* **m6. Scope of missingness.** Complete-window filtering excludes native gaps, particularly in BeijingAir. Report how many windows and channels remain eligible. The current results describe artificial masks on complete segments, not the full distribution of real missingness.
* **m7. Pretraining and authorship.** Possible overlap with Chronos pretraining data is acknowledged but not audited. Document known overlap using model provenance, and leave unverified overlap explicitly unresolved. The README's individual-contributions section also remains unfilled.

### 5. Do the experiments support the claims? (as reviewed on 20 October)

| Claim | Assessment | Evidence and qualification |
|---|---|---|
| H1: blocks are harder than scattered points | Supported in this experiment | MAE and CRPS increase at equal hidden counts for every model and dataset in validation and test. This concerns synthetic interior masks. |
| H2: median accuracy and calibration can disagree | Supported | On BeijingAir point 24, Kalman has the lowest test MAE, 0.094, but the largest PCE among the three main models, 0.099. |
| H3: blending can alter calibration, with different outcomes across settings | Supported descriptively | Electricity point 24 shows improved MAE with increased overcoverage. Long BeijingAir gaps remain undercovered, including for the one-sided forecasts; the mechanism is unresolved. |
| One model is reliably best in a particular setting | Not established for small differences | There are no paired intervals, and some rankings change with the scoring summary. Larger numerical gaps are still worth reporting. |

### 6. Suggested additional experiment: compare blending with an exact conditional distribution

Generate Gaussian local-level series with known process and observation variances. For blocks of 24 and 128 steps, compare the exact smoother with the same distance-weighted quantile blend applied to exact one-sided predictive distributions. This control separates the averaging rule from errors made by Chronos. Then run frozen Chronos left, right and blend on the same series.

Use several process-to-observation noise ratios and repeated series. Report coverage, width relative to the exact smoother, median error and reliability curves by gap position. Construct paired intervals by resampling independent generated series, keeping each series' model outputs together. A seasonal variant can follow, but its oracle must include the seasonal component.

The exact one-sided blend tests the rule itself; the Chronos comparison adds model error. Neither result alone would explain calibration on real data. The repository has the smoother, blending logic, masks and metrics, but needs a synthetic generator and evaluation wrapper. No training is required; measure runtime in a pilot.

### 7. Reproducibility assessment

The repository includes a central configuration, `run_all.sh`, pinned direct dependencies, saved masks, predictions and CSDI checkpoints. The documented environment is Python 3.13 on an Apple M4 with 24 GB of memory, Chronos on CPU and CSDI on MPS. Scoring can be changed without retraining.

Rebuilding the tables requires restoring the processed data as well as the saved predictions and masks. Exact agreement depends on recovering the same targets, scaling and environment. A fresh CSDI fit or sampling run should not be expected to match bit for bit with the current seed handling. The logged training times are about 0.35-0.54 hours per dataset, with substantial additional sampling time.

### 8. Recommendation

Accept with major revisions. The matched-mask comparison supports H1 and H2, and the Chronos ablation shows a useful tradeoff between median accuracy and interval calibration. The revisions should focus on uncertainty in small differences, a common scoring comparison, and a narrower interpretation of H3. The synthetic control would help explain the blend, while the existing results already justify reporting its observed behaviour. Claims about CSDI and Kalman should stay tied to the implementations evaluated here.

---

## Part 2. Author response

Status of each concern after the additional experiments. "Done" means the analysis was run and its tables are in `results/tables/`. Details and numbers follow the table.

| Concern | Response | Status | Main result |
|---|---|---|---|
| M1: uncertainty in differences | Agree | Done | Paired cluster bootstrap. 96 pinball comparisons on test: 70 first model better, 21 ties, 5 second better. Both examples named in M1 are ties. |
| M2: different CRPS approximations | Agree | Done | All models scored on one 21-level grid. The winner changes in 0 of 12 validation and 2 of 12 test cells, and both are ties. |
| M3: cause of the blend's calibration | Agree | Done for the local-level case | Averaging widens intervals to 1.11-1.24 times the exact width. Overcoverage on short gaps fits this. Undercoverage on long BeijingAir gaps does not. |
| M4: CSDI budget | Agree | Done for all three datasets | Longer schedule lowers validation loss by 8-13% and gives better pinball in 11 of 12 test cells. Coverage rises slightly and stays below 0.8. |
| M5: Kalman model | Agree | Not done | Role stated; seasonal state-space baseline not built. |
| m1-m7: reporting | Agree | Partly | PyTorch seed set for the new CSDI runs; runtimes reported separately. The other items are not yet in the README. |

### M1. Uncertainty in model differences

We resample channels first and then windows inside each sampled channel (two-stage cluster bootstrap, 2,000 draws), keeping the models' outputs on the same window together. A comparison counts as a tie when the 95% interval of the paired difference contains zero. The bootstrap is applied to MAE, pinball and CRPS on the common 21-level grid, for eight model pairs in each of the 12 cells, and to the 80% coverage of every model.

* On test, 70 of 96 pinball comparisons favour the first model, 21 are ties and 5 favour the second. On validation the counts are 67, 23 and 6.
* The two small differences named in M1 are ties. Electricity point 24, blend minus CSDI: pinball +0.0005, interval (−0.0035, 0.0055). BeijingAir point 24, CSDI minus Kalman: pinball −0.0006, interval (−0.0043, 0.0035).
* Among Kalman, CSDI (40 epochs) and the blend, at least one model is tied with the best in 7 of 12 test cells.

Still open: the cells keep 60 windows and 7 to 20 channels, so the bootstrap has few clusters and the intervals are approximate. More windows would need new model runs.

### M2. CRPS approximations

Chronos-2 outputs 21 quantile levels (0.01, 0.05, 0.10, ..., 0.95, 0.99); the main run kept an 11-level subset. We re-ran Chronos for all 21 levels and scored every model with the same quantile approximation of CRPS on that grid (Kalman from analytic quantiles, CSDI from its samples). The original CRPS columns stay with their method labels. Nine-level pinball is reported as the common comparison in the new tables.

* The winner among the three original models changes between the original CRPS and the common 21-level CRPS in 0 of 12 validation cells and in 2 of 12 test cells (Electricity point 24, BeijingAir block 24).
* Both changed cells are ties in the bootstrap, so the reversal reported in M2 is within noise.

### M3. What the blend does and why

We ran the control suggested in Section 6: Gaussian local-level series with known variances (1,000 generated windows per setting), process-to-observation noise ratios 0.01, 0.1 and 1, gaps of 24 and 128, the exact Kalman smoother with the true parameters as the oracle, and frozen Chronos on the same series.

*Exact one-sided forecasts (analytic).* For a random walk with noiseless endpoints, the exact distribution at position j of a gap of k steps has variance

$$\sigma^2\,\frac{j\,(k+1-j)}{k+1}.$$

Distance-weighted quantile averaging gives the right mean and a standard deviation that is a weighted arithmetic mean of the two one-sided deviations. This is 1.18 to 1.41 times the exact value (largest at the middle of the gap), 1.34 on average, for both k = 24 and k = 128. Equal weights give 1.76 to 1.88 times. A product of the two Gaussians is exact in this case.

*Chronos on the synthetic series (test).*

| Variant | Width / oracle width | 80% coverage |
|---|---|---|
| Left-only | 1.34-1.51 | 0.61-0.70 |
| Right-only | 1.35-1.51 | 0.63-0.71 |
| Blend, linear weights (main method) | 1.11-1.24 | 0.78-0.85 |
| Blend, equal weights | 1.34-1.50 | 0.75-0.84 |
| Gaussian product of experts | 0.72-0.87 | 0.56-0.67 |

The one-sided Chronos intervals undercover, and the averaging rule widens them, so the blend ends up close to nominal but wider than the oracle. The product of experts, exact for exact forecasts, multiplies the one-sided overconfidence and undercovers badly. On the 12 real test cells the linear blend has the lowest pinball in 9, the product of experts in 3, equal weights in none, and the mean 80% coverage is 0.83 for the linear blend and 0.70 for the product.

*What this says about H3.* Overcoverage of the blend on short gaps (Electricity point 24: 0.880, BeijingAir point 24: 0.903) is what the averaging rule produces on synthetic data, so the explanation is consistent with it. We did not test it causally on real data. Undercoverage on BeijingAir block 128 (blend 0.670, one-sided 0.677 and 0.615) is present before blending, and the averaging rule does not explain it.

*Time reversal.* The right forecast uses the reversed series, which is only valid if the process looks the same backwards. We checked this on a reversible AR(1) process and on an irreversible shot-noise process (sudden jumps, geometric decay), with equal context lengths and horizons in both directions (1,000 generated windows per process and gap). For AR(1), all 16 intervals of the left-minus-right difference (4 metrics, 2 gap lengths, 2 context settings) contain zero. For shot noise, all 16 intervals exclude zero: the left forecast has lower pinball by 0.056 to 0.112 and higher coverage, so the reversed forecast is systematically worse and less calibrated when the process is irreversible. On real data with matched contexts, the left-right difference is significant in 1 of 6 cells for pinball (Electricity gap 24, left better), 1 of 6 for coverage (BeijingAir gap 24, right better) and 1 of 6 for width (BeijingAir gap 128), always in different cells. We found no consistent time-direction effect in the real data.

Still open: the seasonal synthetic variant, and the effect of an irreversible process on the final blend (we measured it on the one-sided forecasts only).

### M4. CSDI training budget

We trained CSDI again on all three datasets with a longer schedule and the learning-rate reductions of the original implementation: Adam, learning rate 1e-3, divided by 10 at 75% and at 90% of the training steps, 100 epochs, no early stopping, the same training and validation windows, PyTorch seed 0. The learning rate ended at 1e-5 in all three runs. The differences from the original CSDI recipe are 100 instead of 200 epochs and batch size 32 instead of 16.

A first attempt used the milestones as epochs. PyPOTS steps the scheduler after every batch (188 per epoch here), so the rate fell to 1e-5 within the first half epoch and the validation loss stayed near 0.37. We discarded that run and corrected the milestones to training steps (14,100 and 16,920).

| Dataset | 40-epoch run: epochs run, best epoch, validation loss | Long run: best epoch, validation loss |
|---|---|---|
| ETT_h1 | 31 (patience 5), 26, 0.1966 | 97, 0.1702 |
| Electricity | 40 (limit), 39, 0.2117 | 92, 0.1877 |
| BeijingAir | 35 (patience 5), 30, 0.1454 | 94, 0.1334 |

So the budget was too small on all three datasets: patience 5 ended ETT_h1 and BeijingAir early as well. This corrects M4's remark that under-training was shown for Electricity only.

* Accuracy. The long run has lower pinball than the 40-epoch run in 11 of 12 test cells with intervals excluding zero (the twelfth, Electricity block 24, is a tie).
* Calibration. Coverage on test moves from 0.60-0.77 to 0.71-0.75; it is higher in 10 of 12 cells, most for block 128 (ETT_h1 0.675 to 0.750, BeijingAir 0.598 to 0.717). It stays below 0.8 in all 12 cells, and in 11 of them the 95% interval excludes 0.8. The undercoverage is therefore not caused by the short training.
* Ranking against the other models. Section 3 below.

Still open: the best epochs are 92 to 97 of 100, so more training could help a little more; one seed only; multivariate CSDI not tried; the recipe was fixed before looking at the new test results, and the validation set only selected the epoch.

### M5. Kalman baseline

Not changed. The text states that Kalman is a deliberately simple local-level baseline, and that its results describe that model. A seasonal state-space baseline is not built.

### Minor points

| Point | Status |
|---|---|
| m1 PyTorch seed | Done for the new runs (seed 0, CPU and CUDA). The saved 40-epoch outputs are unchanged. |
| m2 runtimes | Long runs on a Colab T4: training 0.30 h per dataset, sampling 0.10 h per split and dataset. The 40-epoch times (0.35, 0.54, 0.37 h) come from an M4; two of them are reconstructed from log timestamps. Hardware differs, so the times are not comparable. |
| m3 metric definitions | Not yet in the README. |
| m4 data prerequisite | In the Colab notebook (step 3). Not yet in the README. |
| m5 compact table | Table A below. Not yet in the README. |
| m6 eligible windows | Not done. |
| m7 pretraining overlap, contributions | Not done. |

---

## Part 3. Results after the additional experiments

### Table A. Test pinball loss (9 levels, lower is better)

Bold marks the lowest mean in a cell. A dagger marks models whose paired 95% interval against the best one contains zero (a tie).

| Cell | Kalman | CSDI, 40 epochs | Chronos blend | CSDI, 100 epochs |
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

Reading the daggers: in the ETT_h1 block 24 row the Chronos blend has the lowest mean and CSDI (100 epochs) is tied with it. In the other rows with a dagger, the tied models are tied with the bold one. On validation CSDI (100 epochs) is also the best in 11 of 12 cells (BeijingAir block 24 is the exception, where the blend is best and is tied with both CSDI runs), but some ties differ from test.

### Table B. CSDI (100 epochs) against the other models, test pinball, paired 95% intervals

| Compared with | Significantly better | Tie | Significantly worse |
|---|---|---|---|
| CSDI, 40 epochs | 11 | 1 | 0 |
| Chronos blend | 7 | 5 | 0 |
| Kalman | 11 | 1 | 0 |

Against the blend, CSDI (100 epochs) is significantly better in all six point cells. It ties in five of the six block cells and wins in BeijingAir block 128 (difference −0.0395, interval −0.0897 to −0.0043).

### Table C. Assessment of the claims after the additional experiments

| Claim | Assessment | Evidence and qualification |
|---|---|---|
| H1: blocks are harder than scattered points | Supported | Also holds for CSDI with 100 epochs: test MAE for blocks exceeds MAE for points at equal hidden counts in all six dataset and count pairs (for example Electricity 24: 0.147 for points, 0.266 for blocks). Applies to synthetic interior masks. |
| H2: median accuracy and calibration can disagree | Supported | BeijingAir point 24: Kalman has the lowest test MAE (0.094) and ties with CSDI (100 epochs) on MAE (difference 0.0012, interval −0.0072 to 0.0081), but CSDI has lower pinball and CRPS with intervals excluding zero (−0.0068 and −0.0128). The original PCE example is unchanged. |
| H3: blending can alter calibration, with different outcomes across settings | Supported; mechanism partly tested | Overcoverage on short gaps matches the 1.11-1.24 times width of the averaging rule on synthetic data. Undercoverage on long BeijingAir gaps is present before blending and is not explained. No consistent time-direction effect on real data. |
| One model is reliably best in a particular setting | Supported where intervals exclude zero | CSDI (100 epochs) has the lowest pinball in 11 of 12 test cells and is tied with the best in the other. Its margin over the blend is significant in 7 cells; 5 block cells remain ties. A unique best model, without any tie, exists in 7 of 12 cells. |
| CSDI was under-trained with the original budget | Supported on all three datasets | Validation loss falls 8-13%, and test pinball improves in 11 of 12 cells. |
| CSDI is calibrated after longer training | Not supported | Coverage 0.71-0.75 against a nominal 0.8. |

### Limits that remain

* The long CSDI result has one seed and 100 epochs (the original recipe uses 200).
* Intervals come from 7 to 20 channels per dataset, so they are approximate.
* The synthetic control covers the local-level process; the seasonal variant is not run.
* Kalman is the local-level model only; CSDI is univariate.
* The new runs used a different machine (Colab T4) from the main run (Apple M4).
* The original tables `metrics_{val,test}.csv` and the poster still show the 40-epoch CSDI.

### Files

New code: `tsi/extra_stats.py`, `tsi/extra_blend.py`, `tsi/extra_chronos.py`, `tsi/extra_eval.py`, `tsi/extra_synthetic.py`, `tsi/csdi_long.py`, `configs/long.yaml`, `configs/long2.yaml`. Run order: `colab_extra.ipynb`. Tables: `results/tables/extra_*.csv`. Figures: `results/figures/extra_*.png`. Logs: `logs/csdi_long.log` (Electricity), `logs/csdi_long_2.log` (ETT_h1, BeijingAir), `results/csdi_long/log*.json`.
