"""Port of clreg.R: two-group regression for the damage y.

1. Candidate cut-offs: the 10%, 15%, ..., 90% quantiles of y. For each one the
   rows are split into a small-damage group (y below the cut-off, group 0)
   and a large-damage group (group 1).
2. For each cut-off, a Firth logistic regression that tells the two groups
   apart is checked by 10-fold CV on the training years; the probability
   cut-off (pcut) is the point of the CV ROC curve with the largest
   sensitivity + specificity.
3. For each cut-off, one log-linear regression per group plus the logistic
   regression are fitted on the training years and used to predict the last
   year; the cut-off with the smallest RMSE there is kept.
4. All models are refitted on all years with the chosen cut-offs.

Inputs of every model: the plain inputs (area, tot, ndate and, for region 1,
sido_code) plus principal components of each input group, computed separately
for each subset the model is fitted on. step() decides whether to add the
area:tot interaction.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .linear import step_firth, step_lm
from .preprocess import CaretPCA


def r_cv_folds(n, k, seed) -> list[np.ndarray]:
    """cv.folds() of cvfun.R: split(sample(1:n), rep(1:k, length = n))."""
    perm = np.random.default_rng(seed).permutation(n)
    labels = np.arange(n) % k
    return [perm[labels == j] for j in range(k)]


def roc_youden(p1, p0):
    """PRROC::roc.curve(p1, p0, curve = TRUE), then the threshold at which.max(1 - FPR + TPR).

    The threshold column of PRROC's curve is shifted by one row against the
    rates (row k holds the rates of "score >= k-1-th largest score" but the
    k-th largest score); this is reproduced as it is.
    Returns (area under the ROC curve, threshold).
    """
    scores = np.concatenate([p1, p0])
    pos = np.concatenate([np.ones(len(p1)), np.zeros(len(p0))])
    o = np.argsort(-scores, kind="mergesort")
    s, pos = scores[o], pos[o]
    cp, cn = np.cumsum(pos), np.cumsum(1 - pos)
    use = np.append(s[:-1] != s[1:], True)
    s, cp, cn = s[use], cp[use], cn[use]
    tpr = np.concatenate([[0], cp / len(p1)])
    fpr = np.concatenate([[0], cn / len(p0)])
    auc = float(np.sum(np.diff(fpr) * (tpr[1:] + tpr[:-1]) / 2))
    thr = np.concatenate([s, [s[-1]]])
    return auc, float(thr[np.argmax((1 - fpr) + tpr)])


class PCGroups:
    """One CaretPCA per input group, with R's column names (pc1.PC1, pc1.PC2, ...)."""

    def __init__(self, pcomps: dict, thresh: dict):
        self.pcomps = pcomps
        self.thresh = thresh

    def fit(self, df):
        self.pcas_ = {name: CaretPCA(self.thresh[name], prefix=f"{name}.PC").fit(
            df[[c for c in df.columns if c in cols]]) for name, cols in self.pcomps.items()}
        return self

    def transform(self, df):
        return pd.concat([p.transform(df) for p in self.pcas_.values()], axis=1)


def design(df, pcs, lower_terms, interaction, upper):
    """Model matrix of "~ lower terms + PCs" (+ area:tot when upper), with intercept."""
    cols = [np.ones(len(df))] + [df[t].to_numpy(dtype=float) for t in lower_terms]
    cols += [pcs[c].to_numpy() for c in pcs.columns]
    if upper:
        a, b = interaction
        cols.append(df[a].to_numpy(dtype=float) * df[b].to_numpy(dtype=float))
    return np.column_stack(cols)


class _Fitted:
    """A fitted regression together with its PCA and whether it uses area:tot."""

    def __init__(self, pca, model, upper):
        self.pca, self.model, self.upper = pca, model, upper


class Clreg:
    def __init__(self, inputs, pcomps, thresh, lower_terms, interaction=("area", "tot"),
                 ycut=(0.10, 0.90, 0.05), folds=10):
        self.inputs = inputs
        self.pcomps = pcomps
        self.thresh = thresh
        self.lower_terms = lower_terms
        self.interaction = interaction
        self.ycut = ycut
        self.folds = folds

    # --- helpers
    def _pca(self, df):
        return PCGroups(self.pcomps, self.thresh).fit(df)

    def _X(self, df, pca, upper):
        return design(df, pca.transform(df), self.lower_terms, self.interaction, upper)

    def _fit_reg(self, df):
        pca = self._pca(df)
        y = np.log(df["y"].to_numpy(dtype=float))
        model, upper = step_lm(self._X(df, pca, False), self._X(df, pca, True), y)
        return _Fitted(pca, model, upper)

    def _fit_logit(self, df, ybin):
        pca = self._pca(df)
        model, upper = step_firth(self._X(df, pca, False), self._X(df, pca, True), ybin)
        return _Fitted(pca, model, upper)

    def _prob(self, fitted, df):
        return fitted.model.predict_proba(self._X(df, fitted.pca, fitted.upper))

    def _pred_reg(self, fitted, df):
        return np.exp(fitted.model.predict(self._X(df, fitted.pca, fitted.upper)))

    def _predict_groups(self, logit, reg0, reg1, pcut, df):
        prob = self._prob(logit, df)
        group = (prob >= pcut).astype(int)
        pred = np.empty(len(df))
        for g, reg in ((0, reg0), (1, reg1)):
            rows = group == g
            if rows.any():
                pred[rows] = self._pred_reg(reg, df[rows])
        return pred, group

    # --- training (R: clreg())
    def fit(self, dat_tr: pd.DataFrame, dat_te: pd.DataFrame):
        dat_tr = dat_tr.reset_index(drop=True)
        dat_te = dat_te.reset_index(drop=True)
        y_tr = dat_tr["y"].to_numpy(dtype=float)
        y_te = dat_te["y"].to_numpy(dtype=float)

        lo, hi, by = self.ycut
        yval = lo + by * np.arange(int(np.floor((hi - lo) / by + 1e-10)) + 1)
        ycut = np.quantile(np.concatenate([y_tr, y_te]), yval)
        # keep cut-offs whose training-only quantile leaves both groups non-empty
        both = [len(np.unique(y_tr >= np.quantile(y_tr, v))) == 2 for v in yval]
        yval, ycut = yval[both], ycut[both]

        # pcut by 10-fold CV ROC on the training years
        pauc, pcut = [], []
        for i, yc in enumerate(ycut, start=1):
            ybin = (y_tr >= yc).astype(float)
            probs, acts = [], []
            for test in r_cv_folds(len(dat_tr), self.folds, seed=i):
                train = np.setdiff1d(np.arange(len(dat_tr)), test)
                logit = self._fit_logit(dat_tr.iloc[train], ybin[train])
                probs.append(self._prob(logit, dat_tr.iloc[test]))
                acts.append(ybin[test])
            yhat, yact = np.concatenate(probs), np.concatenate(acts)
            auc, th = roc_youden(yhat[yact == 1], yhat[yact == 0])
            pauc.append(auc)
            pcut.append(th)

        # ycut by RMSE on the last year
        rows = []
        for i, yc in enumerate(ycut):
            ybin = y_tr >= yc
            reg0 = self._fit_reg(dat_tr[~ybin])
            reg1 = self._fit_reg(dat_tr[ybin])
            logit = self._fit_logit(dat_tr, ybin.astype(float))
            with np.errstate(over="ignore", invalid="ignore"):
                # a candidate whose regression extrapolates far can overflow exp();
                # its RMSE is then infinite and it is never chosen (as in R)
                pred, _ = self._predict_groups(logit, reg0, reg1, pcut[i], dat_te)
                rows.append(dict(
                    quantile=yval[i], ycut=yc, pauc=pauc[i], pcut=pcut[i],
                    rmse=float(np.sqrt(np.mean((y_te - pred) ** 2))),
                    rsqr=float(np.corrcoef(y_te, pred)[0, 1] ** 2) if np.all(np.isfinite(pred)) else np.nan,
                    mape=float(np.mean(np.abs((y_te - pred) / y_te)))))
        self.tune_ycut_ = pd.DataFrame(rows)
        rmse = self.tune_ycut_["rmse"].to_numpy()
        best = int(np.argmin(np.where(np.isnan(rmse), np.inf, rmse)))   # which.min skips NaN
        self.ycut_opt_ = float(ycut[best])
        self.pcut_opt_ = float(pcut[best])

        # refit on all years with the chosen cut-offs
        dat = pd.concat([dat_tr, dat_te], ignore_index=True)
        ybin = dat["y"].to_numpy(dtype=float) >= self.ycut_opt_
        self.reg0_ = self._fit_reg(dat[~ybin])
        self.reg1_ = self._fit_reg(dat[ybin])
        self.logit_ = self._fit_logit(dat, ybin.astype(float))
        return self

    # --- prediction (R: predict.clreg())
    def predict(self, newdata: pd.DataFrame):
        """Returns (prediction of y, group 0/1) for each row, in the input order."""
        return self._predict_groups(self.logit_, self.reg0_, self.reg1_, self.pcut_opt_,
                                    newdata.reset_index(drop=True))
