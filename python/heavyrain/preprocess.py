"""Port of caret ``preProcess(method = c("center", "scale", "zv", "nzv", "pca"), thresh)``."""

from __future__ import annotations

import numpy as np
import pandas as pd


def near_zero_var(X: pd.DataFrame, freq_cut=95 / 5, unique_cut=10) -> list[str]:
    """Columns flagged by caret ``nearZeroVar``.

    A column is flagged when it has a single value, or when the most common
    value is more than ``freq_cut`` times as frequent as the second most common
    one AND distinct values make up at most ``unique_cut`` percent of the rows.
    """
    out = []
    for col in X.columns:
        counts = X[col].value_counts(dropna=True).to_numpy()
        n_unique = len(counts)
        if n_unique <= 1:
            out.append(col)
            continue
        top = np.sort(counts)[::-1]
        freq_ratio = top[0] / top[1]
        pct_unique = 100 * n_unique / len(X)
        if freq_ratio > freq_cut and pct_unique <= unique_cut:
            out.append(col)
    return out


class CaretPCA:
    """Drop constant / near-constant columns, standardise, keep principal components.

    Same rules as caret: the number of components is the first one whose
    cumulative share of variance reaches ``thresh``, but at least 2 (capped at
    the number of remaining columns, where caret would stop with an error).
    Columns are always picked by name, so the column order of new data does not
    matter.
    """

    def __init__(self, thresh=0.9, prefix="PC", suffix=""):
        self.thresh = thresh
        self.prefix = prefix
        self.suffix = suffix

    def fit(self, X: pd.DataFrame):
        self.input_columns_ = list(X.columns)
        zero_var = [c for c in X.columns if X[c].nunique(dropna=True) <= 1]
        rest = X.drop(columns=zero_var)
        removed = zero_var + near_zero_var(rest)
        self.removed_ = removed
        self.columns_ = [c for c in X.columns if c not in removed]
        if not self.columns_:
            raise ValueError("no columns left after removing (near) zero-variance columns")

        Z = X[self.columns_].to_numpy(dtype=float)
        self.mean_ = Z.mean(axis=0)
        self.std_ = Z.std(axis=0, ddof=1)
        Z = (Z - self.mean_) / self.std_
        # prcomp(scale = TRUE) on data that already has mean 0 and sd 1
        _, s, vt = np.linalg.svd(Z - Z.mean(axis=0), full_matrices=False)
        var = s ** 2
        cum = np.cumsum(var / var.sum())
        above = np.nonzero(cum >= self.thresh)[0]
        n_comp = len(cum) if len(above) == 0 else max(2, above[0] + 1)
        self.n_components_ = min(n_comp, Z.shape[1])
        self.rotation_ = vt.T[:, :self.n_components_]
        self.names_ = [f"{self.prefix}{i + 1}{self.suffix}" for i in range(self.n_components_)]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        Z = (X[self.columns_].to_numpy(dtype=float) - self.mean_) / self.std_
        return pd.DataFrame(Z @ self.rotation_, columns=self.names_, index=X.index)
