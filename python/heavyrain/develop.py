"""Model development, prediction and validation per region (R: dvfun.R)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .clreg import Clreg
from .features import KEYS
from .learners import caret_folds, caret_train
from .preprocess import CaretPCA

FEATURE1 = [f"d{k}" for k in range(7, 0, -1)]                       # daily rain before the event
FEATURE2 = [f"x{k}" for k in range(1, 25)]                          # 1..24-hour rainfall
FEATURE3 = ["grdp", "money", "property_tax", "building", "city_rate", "people", "vulnerable"]
FEATURE4 = (["recovery_cost", "tlength", "owgr"]
            + [f"cnt{k}" for k in range(1, 5)] + [f"sum{k}" for k in range(1, 5)]
            + [f"avg{k}" for k in range(1, 5)] + ["ccnt", "ctpc", "cDRZ", "nDRZ"])
FEATURE_GROUPS = [FEATURE1, FEATURE2, FEATURE3, FEATURE4]

CARET_MODELS = ["lm", "glmnet", "knn", "rf", "gbm"]
ALL_MODELS = CARET_MODELS + ["clreg"]
# input sets of the three versions of every model: groups 1-2, 1-3, 1-4
INPUT_SETS = [[0, 1], [0, 1, 2], [0, 1, 2, 3]]
MODEL_COLUMNS = [f"{m}{t}" for m in ALL_MODELS for t in (1, 2, 3)]


def _group_cols(df, group):
    return [c for c in df.columns if c in group]


def dev(dataset: pd.DataFrame, mc=2, sido=False, seed=1, verbose=True):
    """Fit the 18 models of one region (R: dev()).

    dataset: rows of one region after makedat() and the small-damage cut.
    sido: also use sido_code as an input (the R code does this for region 1).
    """
    dataset = dataset.reset_index(drop=True)
    if sido:
        feature0 = ["sido_code", "ndate", "tot", "area"]
        lower_terms = ["sido_code", "area", "tot", "ndate"]
    else:
        feature0 = ["ndate", "tot", "area"]
        lower_terms = ["area", "tot", "ndate"]

    preprocs = [CaretPCA(0.9, prefix="PC", suffix=f"_{g + 1}").fit(dataset[_group_cols(dataset, grp)])
                for g, grp in enumerate(FEATURE_GROUPS)]
    pcs = [p.transform(dataset) for p in preprocs]
    designs = [pd.concat([dataset[feature0]] + [pcs[g] for g in gs], axis=1) for gs in INPUT_SETS]
    ylog = np.log(dataset["y"].to_numpy(dtype=float))
    folds = caret_folds(ylog, 10, np.random.default_rng(seed))

    models = {}
    for name in CARET_MODELS:
        if verbose:
            print(f"  {name}", flush=True)
        models[name] = []
        for X in designs:
            model, best, cv = caret_train(name, X.to_numpy(dtype=float), ylog, folds, seed=seed, n_jobs=mc)
            models[name].append(dict(model=model, columns=list(X.columns), best=best, cv=cv))

    if verbose:
        print("  clreg", flush=True)
    last = dataset["year"].max()
    dat_tr = dataset[dataset["year"] <= last - 1]
    dat_te = dataset[dataset["year"] == last]
    models["clreg"] = []
    for gs in INPUT_SETS:
        pcomps = {f"pc{g + 1}": FEATURE_GROUPS[g] for g in gs}
        models["clreg"].append(Clreg(inputs=feature0, pcomps=pcomps,
                                     thresh={k: 0.9 for k in pcomps},
                                     lower_terms=lower_terms).fit(dat_tr, dat_te))
    return dict(feature0=feature0, preprocs=preprocs, models=models, sido=sido)


def pred_out(obj, pdat: pd.DataFrame) -> pd.DataFrame:
    """Predictions of the 18 models for the rows of one region (R: pred_out()).

    Values are on the damage scale (exp of the log-scale predictions), sorted
    by region, sgg_code, year, beg_date, end_date.
    """
    pdat = pdat.reset_index(drop=True)
    pcs = [p.transform(pdat) for p in obj["preprocs"]]
    base = pd.concat([pdat[obj["feature0"]]] + pcs, axis=1)
    out = pdat[KEYS].copy()
    for name in CARET_MODELS:
        for t, entry in enumerate(obj["models"][name], start=1):
            out[f"{name}{t}"] = np.exp(entry["model"].predict(base[entry["columns"]].to_numpy(dtype=float)))
    for t, cl in enumerate(obj["models"]["clreg"], start=1):
        out[f"clreg{t}"] = cl.predict(pdat)[0]
    return out.sort_values(KEYS, kind="mergesort").reset_index(drop=True)


def hval_out(pout: pd.DataFrame, vdat: pd.DataFrame):
    """Validation of the 18 models on rows with known y (R: hval_out()).

    Returns a dict with the table of RMSE / NRMSE / MAPE per model (vres),
    the predictions joined with y (vout) and the name of the model with the
    smallest RMSE (best).
    """
    vout = pout.merge(vdat[KEYS + ["y"]], on=KEYS).sort_values(KEYS, kind="mergesort").reset_index(drop=True)
    yact = vout["y"].to_numpy(dtype=float)
    rows = []
    for m in ALL_MODELS:
        for t in (1, 2, 3):
            pred = vout[f"{m}{t}"].to_numpy(dtype=float)
            r = float(np.sqrt(np.mean((pred - yact) ** 2)))
            rows.append(dict(ml=m, type=t, rmse=r, nrmse=r / (yact.max() - yact.min()),
                             mape=float(np.mean(np.abs(yact - pred) / yact))))
    vres = pd.DataFrame(rows)
    i = int(vres["rmse"].to_numpy().argmin())
    best = f"{vres.ml[i]}{vres.type[i]}"
    return dict(vres=vres, vout=vout, best=best, region=int(vout["region"].iloc[0]))


def pint_mm(hlist):
    """Min-max prediction interval over the 18 models, on the validation rows (R: pint_mm())."""
    yact = np.concatenate([h["vout"]["y"].to_numpy(dtype=float) for h in hlist])
    pred = np.concatenate([h["vout"][MODEL_COLUMNS].to_numpy(dtype=float) for h in hlist], axis=0).T
    lwr, upr = pred.min(axis=0), pred.max(axis=0)
    ncov = int(np.sum((lwr <= yact) & (yact <= upr)))
    return dict(yact=yact, pred=pred, pred_avg=pred.mean(axis=0), pred_lwr=lwr, pred_upr=upr,
                ncov=ncov, pcov=round(ncov / len(yact), 2))
