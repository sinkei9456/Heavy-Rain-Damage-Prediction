# Heavy Rain Damage Prediction: Python port of the R code

[한국어 문서](README_ko.md)

This repository holds a Python port ([python/](python/)) of the R code of a technical manual for predicting heavy rain damage. The R code (`Modeling.R`, `Prediction.R`, `Sub code #1/`) has been removed from the current files; it is kept in the history at commit [4582c73](https://github.com/sinkei9456/Heavy-Rain-Damage-Prediction/blob/4582c73ad2a5da9f4a91c63870243840b019e657), and the R line numbers in this document link to it there. For every district (si-gun-gu) hit by a heavy rain event, the code predicts the damage from rainfall, district statistics and past damage. It fits 18 models per region, picks the best one on the last year of data, and reports the range of the 18 predictions as an interval.

The original data are no longer available. The repository therefore includes a generator of **synthetic data** with the same files, columns and encoding ([python/make_synthetic_data.py](python/make_synthetic_data.py)). Every result in this document comes from that synthetic data and says nothing about real damage.

## Contents

1. [Repository layout](#1-repository-layout)
2. [What the code does](#2-what-the-code-does)
3. [Data the code expects](#3-data-the-code-expects)
4. [Walk-through of the R code](#4-walk-through-of-the-r-code)
5. [Problems and quirks found in the R code](#5-problems-and-quirks-found-in-the-r-code)
6. [Python port](#6-python-port)
7. [Running the Python version](#7-running-the-python-version)
8. [R and Python results compared](#8-r-and-python-results-compared)
9. [References](#9-references)

## 1. Repository layout

```
Heavy-Rain-Damage-Prediction/
├── python/
│   ├── data/                      synthetic data made by make_synthetic_data.py
│   │   ├── Dat_extra/             00_area.csv ... 13_nDRZ.csv (14 tables)
│   │   ├── Dat_input/idat.csv     events to predict (2019, no damage column)
│   │   └── Dat_train/tdat.csv     events with damage (2008-2018)
│   ├── heavyrain/                 shared code
│   │   ├── __init__.py
│   │   ├── clreg.py               two-group regression (clreg.R)
│   │   ├── develop.py             18 models per region, prediction, validation (dvfun.R)
│   │   ├── features.py            reading the data, joining the extra tables (xxfun.R)
│   │   ├── learners.py            lm, glmnet, knn, rf, gbm with caret-style tuning
│   │   ├── linear.py              least squares, logistic and Firth logistic regression, step()
│   │   ├── linesearch.py          line search of the Firth fit (from mbest)
│   │   ├── plots.py               validation figures
│   │   └── preprocess.py          caret preProcess with PCA
│   ├── make_synthetic_data.py     writes data/
│   ├── predict.py                 prediction (Prediction.R)
│   ├── requirements.txt
│   └── train.py                   training and validation (Modeling.R)
├── README.md                      this document (English)
└── README_ko.md                   Korean version
```

## 2. What the code does

```
Dat_train/tdat.csv  (one row per district and event, damage y)
  -> join 14 extra tables (previous year's values), missing values -> 0      makedat()
  -> drop events whose y is below the 36% quantile of y                      Modeling.R
  -> per region 1-4:
       principal components of 4 input groups (90% of variance, at least 2)
       18 models of log(y): 6 methods x 3 input sets                        dev()
  -> validation: fit on all years but the last, predict the last year,
     best model of each region = smallest RMSE                              hval_out()
  -> refit every model on all years and save
Dat_input/idat.csv  (new events)
  -> same inputs -> 18 predictions per event                                 pred_out()
  -> minimum and maximum of the 18 (interval) + value of the best model      Prediction.R
```

**Inputs.** The meaning of most columns is not written in the code; the descriptions marked "probably" are guesses from the names.

| Group | Columns | Meaning | Used as |
|:--|:--|:--|:--|
| 0 | `ndate`, `tot`, `area` (region 1 also `sido_code`) | event length in days (computed), `tot` probably total rainfall, district area | as they are |
| 1 | `d7` ... `d1` | probably daily rainfall 7 to 1 days before the event | principal components |
| 2 | `x1` ... `x24` | probably the largest rainfall over 1 to 24 hours | principal components |
| 3 | `grdp`, `money`, `property_tax`, `building`, `city_rate`, `people`, `vulnerable` | district economy and population, previous year | principal components |
| 4 | `recovery_cost`, `tlength`, `owgr`, `cnt1-4`, `sum1-4`, `avg1-4`, `ccnt`, `ctpc`, `cDRZ`, `nDRZ` | past damage and recovery cost, probably disaster risk zones (`DRZ`) and other facilities, previous year | principal components |

**The 18 models.** Each method is fitted to log(y) three times: type 1 uses group 0 and the components of groups 1-2, type 2 adds group 3, type 3 adds group 4. The models are named `lm1` ... `clreg3`.

| Method | R | Tuning (10-fold cross-validation, smallest RMSE of log y) |
|:--|:--|:--|
| lm | caret `lm` | none |
| glmnet | caret `glmnet`, inputs centred and scaled | 5 alpha values (0.1 to 1) x 5 lambda values (caret default grid) |
| knn | caret `knn` (knnreg) | k = 5, 7, 9, 11, 13 |
| rf | caret `rf`, 1,000 trees | mtry: 5 values from 2 to the number of inputs |
| gbm | caret `gbm` | 500 or 1,000 trees x 10, 15 or 20 splits per tree x learning rate 0.005, 0.01 or 0.02; at least 10 rows per leaf |
| clreg | own code, `clreg.R` | group cut-off by the last year's RMSE, probability cut-off by cross-validated ROC |

**clreg** (two-group regression) splits the events into a small-damage and a large-damage group at a damage cut-off. A Firth logistic regression decides to which group an event belongs, and each group has its own log-linear regression. The cut-off is chosen among the 10%, 15%, ..., 90% quantiles of y.

**Roles of the data.**

| Data | Used for | Role |
|:--|:--|:--|
| all years but the last | fitting the 18 models in the validation step | training |
| last year | comparing the 18 models, best model per region, interval coverage | validation |
| all years | final models saved for prediction | final training |
| 10 folds of the training rows | tuning values of glmnet, knn, rf, gbm; clreg's probability cut-off | cross-validation |
| last year of the rows given to clreg | clreg's damage cut-off | clreg's own validation |

In the validation step clreg receives the years up to the second-to-last, so its own validation year is the second-to-last year; in the final step it is the last year.

## 3. Data the code expects

All files are CSV in EUC-KR. Tables with a name column (`sgg` or `sido`) have it dropped before joining.

| File | Joined by | Columns used |
|:--|:--|:--|
| `Dat_train/tdat.csv` | none | `sgg_code`, `sido_code`, `region` (1-4), `beg_date`, `end_date`, `y`, `tot`, `d1`-`d7`, `x1`-`x24` |
| `Dat_input/idat.csv` | none | the same without `y` |
| `Dat_extra/00_area.csv` | `sgg_code` | `area` |
| `Dat_extra/01_grdp.csv` ... `08_recovery_cost.csv` | `sgg_code`, `year` + 1 | `grdp`, `money`, `property_tax`, `building`, `city_rate`, `people`, `vulnerable`, `recovery_cost` |
| `Dat_extra/09_odat.csv` | `sido_code`, `year` + 1 | not named in the code; synthetic data: `tlength`, `owgr` |
| `Dat_extra/10_ddat.csv` | `sgg_code`, `year` + 1 | not named in the code; synthetic data: `cnt1-4`, `sum1-4`, `avg1-4` |
| `Dat_extra/11_pdat.csv` | `sgg_code`, `year` + 1 | not named in the code; synthetic data: `ccnt`, `ctpc` |
| `Dat_extra/12_cDRZ.csv`, `13_nDRZ.csv` | `sgg_code`, `year` + 1 | `cDRZ`, `nDRZ` |

"`year` + 1" means the table of year t is joined to events of year t+1.

**Synthetic data** ([python/make_synthetic_data.py](python/make_synthetic_data.py), seed 2019):

- 4 regions x 3 provinces x 8 districts = 96 districts; 1,924 events with damage in 2008-2018 and 202 events without damage in 2019.
- Each storm gets an hourly rainfall series; `x1`-`x24` are its largest 1- to 24-hour sums and `tot` its total.
- The damage follows a log-linear formula of rainfall, area, share of vulnerable people, urban share, rain in the week before, region and a hidden district frailty, plus noise. The median damage is 295,674 (thousand KRW in the synthetic data).
- 2% of the rows of three tables are left out (the code fills them with 0), and `nDRZ` is almost always 0 (the PCA filter removes it).

## 4. Walk-through of the R code

The R files are no longer in the current files; the line numbers below link to them at commit [4582c73](https://github.com/sinkei9456/Heavy-Rain-Damage-Prediction/blob/4582c73ad2a5da9f4a91c63870243840b019e657).

### 4.1 Modeling.R

| Lines | What happens |
|:--|:--|
| [16-32](https://github.com/sinkei9456/Heavy-Rain-Damage-Prediction/blob/4582c73ad2a5da9f4a91c63870243840b019e657/Modeling.R#L16-L32) | load packages; `source()` the functions from `Rfun/` |
| [36-49](https://github.com/sinkei9456/Heavy-Rain-Damage-Prediction/blob/4582c73ad2a5da9f4a91c63870243840b019e657/Modeling.R#L36-L49) | read `tdat.csv`; add `ndate` (days) and `year`; `makedat()` joins the extra tables |
| [53-61](https://github.com/sinkei9456/Heavy-Rain-Damage-Prediction/blob/4582c73ad2a5da9f4a91c63870243840b019e657/Modeling.R#L53-L61) | keep events with y between its 36% and 100% quantiles ("Outlier elimination") |
| [67-85](https://github.com/sinkei9456/Heavy-Rain-Damage-Prediction/blob/4582c73ad2a5da9f4a91c63870243840b019e657/Modeling.R#L67-L85) | validation: `dev()` per region on all years but the last, `pred_out()` and `hval_out()` on the last year; `sido = T` only for region 1 |
| [88-117](https://github.com/sinkei9456/Heavy-Rain-Damage-Prediction/blob/4582c73ad2a5da9f4a91c63870243840b019e657/Modeling.R#L88-L117) | best model per region (`optm.rds`), `validation.txt`, `validation.pdf` |
| [120-180](https://github.com/sinkei9456/Heavy-Rain-Damage-Prediction/blob/4582c73ad2a5da9f4a91c63870243840b019e657/Modeling.R#L120-L180) | min-max interval over the 18 models (`pint_mm()`), `coverage.pdf`, `validation.csv` |
| [185-193](https://github.com/sinkei9456/Heavy-Rain-Damage-Prediction/blob/4582c73ad2a5da9f4a91c63870243840b019e657/Modeling.R#L185-L193) | refit `dev()` per region on all years; save `obj1.rds` ... `obj4.rds` |

### 4.2 Prediction.R

| Lines | What happens |
|:--|:--|
| [31-44](https://github.com/sinkei9456/Heavy-Rain-Damage-Prediction/blob/4582c73ad2a5da9f4a91c63870243840b019e657/Prediction.R#L31-L44) | read `idat.csv`, add the same features |
| [49-53](https://github.com/sinkei9456/Heavy-Rain-Damage-Prediction/blob/4582c73ad2a5da9f4a91c63870243840b019e657/Prediction.R#L49-L53) | load the four region models and `optm.rds` |
| [57-72](https://github.com/sinkei9456/Heavy-Rain-Damage-Prediction/blob/4582c73ad2a5da9f4a91c63870243840b019e657/Prediction.R#L57-L72) | `pred_out()` per region inside `tryCatch()` |
| [73-89](https://github.com/sinkei9456/Heavy-Rain-Damage-Prediction/blob/4582c73ad2a5da9f4a91c63870243840b019e657/Prediction.R#L73-L89) | per row: `pred_lwr` = minimum of the 18, `pred_upr` = maximum, `pred_fit` = best model |
| [93-100](https://github.com/sinkei9456/Heavy-Rain-Damage-Prediction/blob/4582c73ad2a5da9f4a91c63870243840b019e657/Prediction.R#L93-L100) | write `prediction.csv` |

### 4.3 Sub code #1/xxfun.R: `makedat()`

[Lines 5-214](https://github.com/sinkei9456/Heavy-Rain-Damage-Prediction/blob/4582c73ad2a5da9f4a91c63870243840b019e657/Sub%20code%20%231/xxfun.R#L5-L214). Joins the 14 tables with `merge(..., all.x = T)`; every table except `00_area` has `year + 1` first ([e.g. line 29](https://github.com/sinkei9456/Heavy-Rain-Damage-Prediction/blob/4582c73ad2a5da9f4a91c63870243840b019e657/Sub%20code%20%231/xxfun.R#L29)). [Line 206](https://github.com/sinkei9456/Heavy-Rain-Damage-Prediction/blob/4582c73ad2a5da9f4a91c63870243840b019e657/Sub%20code%20%231/xxfun.R#L206) replaces every missing value with 0, [lines 209-210](https://github.com/sinkei9456/Heavy-Rain-Damage-Prediction/blob/4582c73ad2a5da9f4a91c63870243840b019e657/Sub%20code%20%231/xxfun.R#L209-L210) sort by district and start date.

### 4.4 Sub code #1/pcfun.R

Defines `calc_pc1()` and `calc_pc2()`, which no other file calls.

### 4.5 Sub code #1/cvfun.R

`cv.folds()` ([lines 5-7](https://github.com/sinkei9456/Heavy-Rain-Damage-Prediction/blob/4582c73ad2a5da9f4a91c63870243840b019e657/Sub%20code%20%231/cvfun.R#L5-L7)) shuffles the rows and deals them into k folds. `cv_cm()` ([lines 11-42](https://github.com/sinkei9456/Heavy-Rain-Damage-Prediction/blob/4582c73ad2a5da9f4a91c63870243840b019e657/Sub%20code%20%231/cvfun.R#L11-L42)) computes a confusion matrix; clreg only uses the threshold it is given back.

### 4.6 Sub code #1/dvfun.R

- **`dev()`** ([lines 4-582](https://github.com/sinkei9456/Heavy-Rain-Damage-Prediction/blob/4582c73ad2a5da9f4a91c63870243840b019e657/Sub%20code%20%231/dvfun.R#L4-L582)): input groups ([11-41](https://github.com/sinkei9456/Heavy-Rain-Damage-Prediction/blob/4582c73ad2a5da9f4a91c63870243840b019e657/Sub%20code%20%231/dvfun.R#L11-L41)); `preProcess(method = c("center", "scale", "zv", "nzv", "pca"), thresh = 0.9)` per group ([43-72](https://github.com/sinkei9456/Heavy-Rain-Damage-Prediction/blob/4582c73ad2a5da9f4a91c63870243840b019e657/Sub%20code%20%231/dvfun.R#L43-L72)); the three input sets and their formulas for log(y) ([74-134](https://github.com/sinkei9456/Heavy-Rain-Damage-Prediction/blob/4582c73ad2a5da9f4a91c63870243840b019e657/Sub%20code%20%231/dvfun.R#L74-L134)); 10-fold cross-validation ([137-142](https://github.com/sinkei9456/Heavy-Rain-Damage-Prediction/blob/4582c73ad2a5da9f4a91c63870243840b019e657/Sub%20code%20%231/dvfun.R#L137-L142)); lm, glmnet, knn, rf, gbm ([150-391](https://github.com/sinkei9456/Heavy-Rain-Damage-Prediction/blob/4582c73ad2a5da9f4a91c63870243840b019e657/Sub%20code%20%231/dvfun.R#L150-L391)); three clreg models ([394-537](https://github.com/sinkei9456/Heavy-Rain-Damage-Prediction/blob/4582c73ad2a5da9f4a91c63870243840b019e657/Sub%20code%20%231/dvfun.R#L394-L537)).
- **`pred_out()`** ([587-797](https://github.com/sinkei9456/Heavy-Rain-Damage-Prediction/blob/4582c73ad2a5da9f4a91c63870243840b019e657/Sub%20code%20%231/dvfun.R#L587-L797)): applies the saved PCA and the 18 models, `exp()` of every log prediction, rows sorted by region, district, year and dates.
- **`hval_out()`** ([802-879](https://github.com/sinkei9456/Heavy-Rain-Damage-Prediction/blob/4582c73ad2a5da9f4a91c63870243840b019e657/Sub%20code%20%231/dvfun.R#L802-L879)): joins the predictions with y; RMSE, NRMSE (RMSE / range of y) and MAPE per model; best = smallest RMSE; bar chart.
- **`pint_mm()`** ([884-1006](https://github.com/sinkei9456/Heavy-Rain-Damage-Prediction/blob/4582c73ad2a5da9f4a91c63870243840b019e657/Sub%20code%20%231/dvfun.R#L884-L1006)): per validation event the minimum and maximum of the 18 predictions, how many observed values fall inside, two figures (raw and log scale).

### 4.7 Sub code #1/clreg.R

- Damage cut-off candidates ([60-76](https://github.com/sinkei9456/Heavy-Rain-Damage-Prediction/blob/4582c73ad2a5da9f4a91c63870243840b019e657/Sub%20code%20%231/clreg.R#L60-L76)): quantiles 0.10 to 0.90 by 0.05 of the training and validation y together; candidates that leave a group empty in the training data are dropped.
- Probability cut-off per candidate ([81-299](https://github.com/sinkei9456/Heavy-Rain-Damage-Prediction/blob/4582c73ad2a5da9f4a91c63870243840b019e657/Sub%20code%20%231/clreg.R#L81-L299)): 10-fold cross-validation on the training rows (`set.seed(i)`); in each fold, PCA on the fitting rows, a Firth logistic regression (`glm(..., method = "firthglm.fit")`) and `step()` between a formula without and with `area:tot`; the threshold is the point of the pooled ROC curve with the largest sensitivity + specificity.
- Choice of the damage cut-off ([304-749](https://github.com/sinkei9456/Heavy-Rain-Damage-Prediction/blob/4582c73ad2a5da9f4a91c63870243840b019e657/Sub%20code%20%231/clreg.R#L304-L749)): for each candidate, one log-linear regression per group (each with its own PCA and `step()`) and the logistic regression are fitted on the training rows, the validation rows are assigned to groups and predicted; the candidate with the smallest RMSE wins.
- Refit ([754-1056](https://github.com/sinkei9456/Heavy-Rain-Damage-Prediction/blob/4582c73ad2a5da9f4a91c63870243840b019e657/Sub%20code%20%231/clreg.R#L754-L1056)) on training and validation rows with the chosen cut-offs.
- `predict.clreg()` ([1090-1238](https://github.com/sinkei9456/Heavy-Rain-Damage-Prediction/blob/4582c73ad2a5da9f4a91c63870243840b019e657/Sub%20code%20%231/clreg.R#L1090-L1238)): logistic regression -> group -> that group's regression -> `exp()`.

## 5. Problems and quirks found in the R code

**Problems.** Only the first two are changed in the Python port, because they stop the code from running; the others are kept so that the port gives the same results, and are listed here for anyone who wants to change them.

| No. | Problem | Where | Python port |
|:--|:--|:--|:--|
| 1 | The scripts load `Rfun/*.R`, but the files are in `Sub code #1/`; the data folders (`Dat_train/`, `Dat_extra/`, `Dat_input/`) and output folders (`Robj/`, `Output/`) are not in the repository, so the code does not run as it is | [Modeling.R 28-32](https://github.com/sinkei9456/Heavy-Rain-Damage-Prediction/blob/4582c73ad2a5da9f4a91c63870243840b019e657/Modeling.R#L28-L32) | folders are command-line options |
| 2 | If the input has no rows for one region, `Prediction.R` stops: `tryCatch()` returns `NULL`, and `apply(ptmp, 1, min)` then fails with "dim(X) must have a positive length"; no output file is written (checked by running it with region 4 left out) | [Prediction.R 57-89](https://github.com/sinkei9456/Heavy-Rain-Damage-Prediction/blob/4582c73ad2a5da9f4a91c63870243840b019e657/Prediction.R#L57-L89) | the region is skipped with a message |
| 3 | "Outlier elimination" removes the 36% smallest damages and nothing at the top, so the models never see small events and are likely to predict them too high | [Modeling.R 53-61](https://github.com/sinkei9456/Heavy-Rain-Damage-Prediction/blob/4582c73ad2a5da9f4a91c63870243840b019e657/Modeling.R#L53-L61) | kept |
| 4 | `sido_code` (province code) enters the region 1 models as a number, not as a category | [dvfun.R 11-15](https://github.com/sinkei9456/Heavy-Rain-Damage-Prediction/blob/4582c73ad2a5da9f4a91c63870243840b019e657/Sub%20code%20%231/dvfun.R#L11-L15) | kept |
| 5 | clreg's damage cut-offs are quantiles of training and validation y together, while the check that both groups are non-empty uses training quantiles only | [clreg.R 63-76](https://github.com/sinkei9456/Heavy-Rain-Damage-Prediction/blob/4582c73ad2a5da9f4a91c63870243840b019e657/Sub%20code%20%231/clreg.R#L63-L76) | kept |
| 6 | Missing values of the extra tables become 0, indistinguishable from a real 0 | [xxfun.R 206](https://github.com/sinkei9456/Heavy-Rain-Damage-Prediction/blob/4582c73ad2a5da9f4a91c63870243840b019e657/Sub%20code%20%231/xxfun.R#L206) | kept |
| 7 | knn measures distance on unscaled inputs, so large-valued columns (area, rainfall) decide the neighbours; in the R run of section 8, knn1, knn2 and knn3 gave identical predictions in every region | [dvfun.R 229-270](https://github.com/sinkei9456/Heavy-Rain-Damage-Prediction/blob/4582c73ad2a5da9f4a91c63870243840b019e657/Sub%20code%20%231/dvfun.R#L229-L270) | kept |
| 8 | Models are fitted to log(y) and turned back with `exp()` without a correction, so the predictions sit below the mean damage (closer to a median) | [dvfun.R 647-652](https://github.com/sinkei9456/Heavy-Rain-Damage-Prediction/blob/4582c73ad2a5da9f4a91c63870243840b019e657/Sub%20code%20%231/dvfun.R#L647-L652) | kept |
| 9 | `pcfun.R` is never used | [pcfun.R](https://github.com/sinkei9456/Heavy-Rain-Damage-Prediction/blob/4582c73ad2a5da9f4a91c63870243840b019e657/Sub%20code%20%231/pcfun.R) | not ported |
| 10 | Minor: when every row falls in group 1, `predict.clreg()` labels it group 0 ([line 1212](https://github.com/sinkei9456/Heavy-Rain-Damage-Prediction/blob/4582c73ad2a5da9f4a91c63870243840b019e657/Sub%20code%20%231/clreg.R#L1212)); `hval_out()` computes R² and throws it away ([line 837](https://github.com/sinkei9456/Heavy-Rain-Damage-Prediction/blob/4582c73ad2a5da9f4a91c63870243840b019e657/Sub%20code%20%231/dvfun.R#L837)); lm2 and lm3 use other cross-validation folds than the other models because `set.seed(1)` is not repeated before them (only the unused CV report of lm is affected) | | kept (no effect on results) |

**Quirks reproduced on purpose.** These are not errors but behaviour of the R packages that changes the numbers; the port copies them.

- **`step()` on a Firth logistic regression.** R's `add1.glm()` and `drop1.glm()` refit the candidate models with ordinary maximum likelihood (`glm.fit`), not with Firth's method, while the current model keeps its Firth deviance. `area:tot` is added when the maximum likelihood deviance + 2 x rank becomes smaller; after the Firth refit, `step()` may drop it again ([linear.py `step_firth()`](python/heavyrain/linear.py)).
- **Firth fit.** mbest stops when (score' I⁻¹ score)² <= 2e-8, which is loose enough that the result depends on the iterations, so the port copies mbest's Newton steps and line search.
- **ROC threshold.** In PRROC's ROC curve the threshold column is one row off against the rates; the threshold taken is the one printed in the chosen row.
- **glmnet in caret.** caret fits glmnet's default path of 100 lambdas and predicts at the chosen lambda by linear interpolation between the two nearest path values; glmnet also standardises y internally, which changes what a given lambda means.
- **knn ties.** caret's C code adds neighbours whose squared distance equals the k-th one exactly.
- **PCA.** caret keeps at least 2 components even when one already reaches 90% of the variance.

## 6. Python port

### 6.1 R to Python

| R | Python |
|:--|:--|
| `Modeling.R` | [train.py](python/train.py) |
| `Prediction.R` | [predict.py](python/predict.py) |
| `makedat()`, quantile cut | `features.makedat()`, `features.drop_small_damage()` |
| caret `preProcess(..., "pca", thresh)` | `preprocess.CaretPCA` |
| caret `train()` for lm, glmnet, knn, rf, gbm | `learners.caret_train()` with `LM`, `Glmnet`, `KNN`, `RF`, `GBM` |
| `glm(..., method = "firthglm.fit")`, `step()` | `linear.FirthLogistic`, `linear.step_firth()`, `linear.step_lm()` |
| `PRROC::roc.curve()` + `which.max()` | `clreg.roc_youden()` |
| `clreg()`, `predict.clreg()` | `clreg.Clreg.fit()`, `.predict()` |
| `dev()`, `pred_out()`, `hval_out()`, `pint_mm()` | `develop.dev()`, `pred_out()`, `hval_out()`, `pint_mm()` |
| ggplot figures | `plots.validation_bars()`, `plots.coverage()` |
| `saveRDS()` / `readRDS()` | joblib (`obj1.joblib` ... `obj4.joblib`) and `optm.json` |

### 6.2 Differences from the R code

| Item | R code | Python port |
|:--|:--|:--|
| Folders | fixed (`Rfun/`, `Dat_train/`, ...) | options `--data-dir`, `--out-dir`, `--model-dir` |
| Region without input rows | stops | skipped |
| Random numbers | R | numpy; cross-validation folds, rf and gbm therefore differ from a given R run (section 8) |
| rf | randomForest | forest of scikit-learn trees grown like randomForest (bootstrap rows as real copies, node size 5, mtry) |
| gbm | gbm package | scikit-learn `GradientBoostingRegressor` with the same settings (splits + 1 leaves grown best first, half the rows per tree, 10 rows per leaf); a different implementation |
| glmnet | Fortran, convergence tolerance 1e-7 | scikit-learn coordinate descent to 1e-12 |
| lm cross-validation report | computed, not used anywhere | skipped |
| Saved models | `.rds` | `.joblib` and `.json`; the CSV outputs have the same columns and encoding |
| Figures | ggplot | matplotlib, same content, English labels |

## 7. Running the Python version

Tested with Python 3.12.3, numpy 2.5.2, pandas 3.0.6, scipy 1.18.1, scikit-learn 1.9.1, joblib 1.5.2 and matplotlib (see [requirements.txt](python/requirements.txt)).

```bash
pip install -r python/requirements.txt
cd python
python make_synthetic_data.py                 # writes data/ (already included)
python train.py --data-dir data --out-dir output --model-dir model --mc 8
python predict.py --data-dir data --model-dir model --out output/prediction.csv
```

| Script | Option | Default | Meaning |
|:--|:--|:--|:--|
| train.py | `--data-dir` | data | folder with `Dat_train/` and `Dat_extra/` |
| | `--out-dir` | output | validation outputs |
| | `--model-dir` | model | saved models |
| | `--mc` | 2 | parallel jobs (R `mc`) |
| | `--seed` | 1 | random seed (R `set.seed(1)`) |
| | `--encoding` | euc-kr | encoding of all CSV files |
| predict.py | `--data-dir` | data | folder with `Dat_extra/` (and `Dat_input/`) |
| | `--input` | data/Dat_input/idat.csv | events to predict |
| | `--model-dir` | model | models saved by train.py |
| | `--out` | output/prediction.csv | output file |
| | `--encoding` | euc-kr | encoding of all CSV files |

**Outputs of train.py**

- `output/validation.txt`: RMSE, NRMSE and MAPE of the 18 models per region on the validation year, and the best model of each region.
- `output/validation.csv`: per validation event the 18 predictions, `pred_lwr` / `pred_upr` (minimum / maximum of the 18), `pred_fit` (best model of the region), `y` and `cover` (1 if y lies inside the interval).
- `output/validation.pdf`: one page per region. Horizontal axis: method (alphabetical, as in the R figure); bars: input set 1, 2, 3 in three colours; vertical axis: RMSE on the validation year; the title names the best model.
- `output/coverage.pdf`: page 1 in damage units, page 2 in log units. Horizontal axis: validation events of all regions in order; vertical black line: from the smallest to the largest of the 18 predictions; filled circle: observed damage (its colour only tells events apart); open circle: mean of the 18 predictions; the title gives how many events lie inside their line.
- `model/obj1.joblib` ... `obj4.joblib`, `model/optm.json`.

**Output of predict.py**: `prediction.csv` with the keys (`region`, `sgg_code`, `year`, `beg_date`, `end_date`), the 18 predictions, `pred_lwr`, `pred_upr` and `pred_fit`.

Notes: model files are pickles; load only files you trust, with the same scikit-learn version. `train.py` with `--mc 16` took 3.6 minutes on the synthetic data on a 20-core server (R `Modeling.R` with its `mc = 2` took 18 minutes; the settings differ, so this is not a fair speed comparison).

## 8. R and Python results compared

Both versions were run on the same synthetic data. R: version 4.5.3 with caret 7.0.1, glmnet 5.0, gbm 2.3.1, randomForest 4.7-1.2, mbest 0.6.1 and PRROC 1.4; the R scripts were not changed, only the folders they expect (`Rfun/`, `Dat_*`, `Robj/`, `Output/`) were set up. The comparison has three levels.

**8.1 Single parts on the same inputs.** These parts have no randomness, so they should agree up to rounding.

Table 1. Largest difference between R and Python (relative difference, no unit)

| Part | Largest difference | Notes |
|:--|--:|:--|
| PCA of the 4 input groups (region 1) | 5e-14 | same columns removed, same number of components (7, 2, 2, 10) |
| lm and its `step()` choice | 8e-13 | 4 cases, with and without `area:tot` chosen; same choice in all |
| Firth logistic regression and its `step()` choice | 2e-14 | 7 cases; same choice in all |
| glmnet lambda grid | 1e-10 | |
| glmnet predictions, R run with tolerance 1e-14 | 2e-6 | with R's default tolerance (1e-7), R itself is off by up to 6e-3 |
| clreg, region 1, with R's folds: table of cut-off candidates | 7e-12 | ROC area, probability cut-off, RMSE of every candidate |
| clreg, region 1, with R's folds: final predictions | 7e-12 | |

**8.2 Whole pipeline with R's random folds.** caret 7 draws one random number and uses it as the seed of the folds; repeating that in R gave exactly the folds stored in R's models (8 of 8 checks). These folds, and those of clreg, were given to the Python pipeline.

Table 2. Largest relative difference of single predictions over 3 versions x 4 regions (no unit)

| Method | Validation (models of 2008-2017 predicting 2018, 115 events) | Prediction (models of all years predicting 2019, 202 events) |
|:--|--:|--:|
| lm | 3e-12 | 1e-13 |
| glmnet | 3e-3 | 3e-3 |
| knn | 8e-15 | 7e-15 |
| clreg | 6e-11 | 1e-10 |
| rf | 0.25 | 0.28 |
| gbm | 0.42 | 0.21 |

With the same folds, Python chose the same tuning values as R everywhere (checked directly against R's saved models for glmnet1, knn1 and clreg1-3 of every region). glmnet differs only by R's own convergence tolerance; rf and gbm differ because each draws its own random numbers inside the model (bootstrap rows, candidate inputs, subsamples).

**8.3 Normal use: each language with its own random numbers.** This is how the two versions behave in practice.

Table 3. Best model per region and its RMSE on the validation year (unit: thousand KRW)

| Region | Best model, R | Best model, Python | RMSE, R | RMSE, Python | Validation events |
|:--|:--|:--|--:|--:|--:|
| 1 | gbm2 | gbm2 | 676,357 | 685,701 | 20 |
| 2 | lm3 | lm3 | 1,546,954 | 1,546,954 | 22 |
| 3 | rf2 | rf2 | 426,548 | 451,201 | 9 |
| 4 | clreg3 | clreg3 | 2,630,022 | 2,674,436 | 64 |

Table 4. Validation RMSE of Python divided by that of R, lowest and highest over 3 versions x 4 regions (no unit)

| Method | Lowest | Highest |
|:--|--:|--:|
| lm | 1.00 | 1.00 |
| glmnet | 0.96 | 1.09 |
| knn | 1.00 | 1.06 |
| rf | 0.98 | 1.06 |
| gbm | 0.74 | 1.14 |
| clreg | 0.76 | 1.04 |

- The min-max interval contained the observed damage for 39 of the 115 validation events in R and 40 in Python.
- In the 2019 prediction (202 events) the final prediction `pred_fit` of Python divided by that of R had a median of 1.000; 99% of the events were within 10% and all within 20% (lowest 0.90, highest 1.10).
- The output files have the same rows and columns as R's.

In short: the parts without randomness give the same numbers as R; the parts with randomness give different single numbers but the same choices and similar errors.

## 9. References

- Breiman, L. (2001). Random forests. *Machine Learning*, 45(1), 5–32.
- Firth, D. (1993). Bias reduction of maximum likelihood estimates. *Biometrika*, 80(1), 27–38.
- Friedman, J. H. (2001). Greedy function approximation: a gradient boosting machine. *Annals of Statistics*, 29(5), 1189–1232.
- Friedman, J., Hastie, T., & Tibshirani, R. (2010). Regularization paths for generalized linear models via coordinate descent. *Journal of Statistical Software*, 33(1), 1–22.
- Kuhn, M. (2008). Building predictive models in R using the caret package. *Journal of Statistical Software*, 28(5), 1–26.
- Moré, J. J., & Thuente, D. J. (1994). Line search algorithms with guaranteed sufficient decrease. *ACM Transactions on Mathematical Software*, 20(3), 286–307.
- R packages used by the R code: caret, glmnet, gbm, randomForest, mbest (Firth regression), PRROC (ROC curves), dplyr, ggplot2, doParallel.
