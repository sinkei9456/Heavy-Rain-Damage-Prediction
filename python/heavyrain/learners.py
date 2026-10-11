"""The five caret models of dvfun.R (lm, glmnet, knn, rf, gbm) with caret-style tuning.

caret::train(..., trControl = trainControl(method = "cv", number = 10)) picks the
tuning values with the smallest mean RMSE over 10 folds and refits on all rows.
The R code fits every model to log(y).
"""

from __future__ import annotations

import numpy as np
from joblib import Parallel, delayed
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import enet_path
from sklearn.tree import DecisionTreeRegressor

from .linear import OLS


def caret_folds(y, k, rng) -> list[np.ndarray]:
    """caret createFolds() for a numeric outcome: folds balanced over quartile groups of y.

    Returns the held-out row indices of each fold.
    """
    y = np.asarray(y, dtype=float)
    n = len(y)
    cuts = min(max(n // k, 2), 5)
    breaks = np.unique(np.quantile(y, np.linspace(0, 1, cuts)))
    group = np.clip(np.searchsorted(breaks, y, side="left"), 1, len(breaks) - 1)
    fold = np.zeros(n, dtype=int)
    for g in sorted(np.unique(group), key=str):
        idx = np.nonzero(group == g)[0]
        m = len(idx)
        reps = m // k
        if reps > 0:
            seq = np.tile(np.arange(1, k + 1), reps)
            if m % k:
                seq = np.concatenate([seq, rng.choice(np.arange(1, k + 1), m % k, replace=False)])
            fold[idx] = rng.permutation(seq)
        else:
            fold[idx] = rng.choice(np.arange(1, k + 1), m, replace=False)
    return [np.nonzero(fold == j)[0] for j in range(1, k + 1)]


def rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


# ----------------------------------------------------------------- models
class LM:
    """caret method "lm": ordinary least squares with an intercept."""

    def fit(self, X, y):
        self.ols_ = OLS().fit(np.column_stack([np.ones(len(X)), X]), y)
        return self

    def predict(self, X):
        return self.ols_.predict(np.column_stack([np.ones(len(X)), X]))


class Glmnet:
    """caret method "glmnet" with preProc = c("center", "scale").

    As caret does: centre and scale x, fit the default glmnet path for the given
    alpha (100 lambdas, standardised x and y, early stop as glmnet), then
    predict at ``lam`` by linear interpolation between the two nearest
    lambdas of the path (glmnet predict(..., s = lam)).
    """

    def __init__(self, alpha, lam, nlambda=100):
        self.alpha = alpha
        self.lam = lam
        self.nlambda = nlambda

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        n, p = X.shape
        self.c_mean_, self.c_sd_ = X.mean(0), X.std(0, ddof=1)        # caret preProc
        Xc = (X - self.c_mean_) / self.c_sd_
        self.g_mean_, self.g_sd_ = Xc.mean(0), Xc.std(0)              # glmnet standardize
        Xs = (Xc - self.g_mean_) / self.g_sd_
        self.y_mean_, y_sd = y.mean(), y.std()
        ys = (y - self.y_mean_) / y_sd
        lmax = np.max(np.abs(Xs.T @ (y - self.y_mean_))) / (n * self.alpha)
        ratio = 0.01 if n < p else 1e-4
        lambdas = lmax * ratio ** (np.arange(self.nlambda) / (self.nlambda - 1))
        _, coefs, _ = enet_path(Xs, ys, l1_ratio=self.alpha, alphas=lambdas / y_sd,
                                tol=1e-12, max_iter=100000)
        coefs = coefs * y_sd                                           # back to the y scale
        # glmnet stops the path early (after at least 5 lambdas) when the
        # explained deviance barely changes or exceeds 0.999
        tss = np.sum((y - self.y_mean_) ** 2)
        rsq = 1 - np.array([np.sum((y - self.y_mean_ - Xs @ c) ** 2) for c in coefs.T]) / tss
        last = len(lambdas)
        for m in range(4, len(lambdas)):
            if rsq[m] - rsq[m - 1] < 1e-5 * rsq[m] or rsq[m] > 0.999:
                last = m + 1
                break
        self.lambdas_, self.coefs_ = lambdas[:last], coefs[:, :last]
        self.beta_ = self._interp(self.lam)
        return self

    def _interp(self, s):
        lam = self.lambdas_
        if len(lam) == 1:
            return self.coefs_[:, 0]
        s = min(max(s, lam.min()), lam.max())
        k = len(lam)
        sfrac = (lam[0] - s) / (lam[0] - lam[k - 1])
        lnorm = (lam[0] - lam) / (lam[0] - lam[k - 1])
        coord = np.interp(sfrac, lnorm, np.arange(1, k + 1))
        left, right = int(np.floor(coord)) - 1, int(np.ceil(coord)) - 1
        if left == right or abs(lnorm[left] - lnorm[right]) < np.finfo(float).eps:
            frac = 1.0
        else:
            frac = (sfrac - lnorm[right]) / (lnorm[left] - lnorm[right])
        return self.coefs_[:, left] * frac + self.coefs_[:, right] * (1 - frac)

    def predict(self, X):
        Xc = (np.asarray(X, dtype=float) - self.c_mean_) / self.c_sd_
        return self.y_mean_ + ((Xc - self.g_mean_) / self.g_sd_) @ self.beta_


def glmnet_grid(X, y, length=5):
    """caret's default glmnet grid: 5 alphas x the inner 5 lambdas of a 7-lambda path at alpha 0.5."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    n, p = X.shape
    Xs = (X - X.mean(0)) / X.std(0)
    lmax = np.max(np.abs(Xs.T @ (y - y.mean()))) / (n * 0.5)
    ratio = 0.01 if n < p else 1e-4
    lam = lmax * ratio ** (np.arange(length + 2) / (length + 1))
    lam = lam[1:-1][:length]
    return [dict(alpha=a, lam=l) for l in lam for a in np.linspace(0.1, 1, length)]


class KNN:
    """caret method "knn" (knnreg): mean y of the k nearest rows, Euclidean
    distance on the unscaled inputs. As caret's C code (knn3reg, use.all = TRUE),
    rows whose squared distance is within a factor 1 + 1e-4 of the k-th smallest
    are also included."""

    def __init__(self, k):
        self.k = k

    def fit(self, X, y):
        self.X_ = np.asarray(X, dtype=float)
        self.y_ = np.asarray(y, dtype=float)
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        k = min(self.k, len(self.y_))
        out = np.empty(len(X))
        for i, x in enumerate(X):
            d2 = np.zeros(len(self.X_))
            for c in range(X.shape[1]):          # same summation order as the C code
                d2 += (x[c] - self.X_[:, c]) ** 2
            kth = np.partition(d2, k - 1)[k - 1]
            out[i] = self.y_[d2 <= kth * (1 + 1e-4)].mean()
        return out


class RF:
    """caret method "rf" (randomForest, ntree = 1000), grown like R randomForest:
    bootstrap rows passed as real copies, nodesize 5, ``mtry`` features per split."""

    def __init__(self, mtry, ntree=1000, nodesize=5, random_state=None, n_jobs=None):
        self.mtry = mtry
        self.ntree = ntree
        self.nodesize = nodesize
        self.random_state = random_state
        self.n_jobs = n_jobs

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        n, p = X.shape
        seeds = np.random.default_rng(self.random_state).integers(2**31 - 1, size=self.ntree)

        def grow(seed):
            boot = np.random.default_rng(seed).integers(0, n, size=n)
            tree = DecisionTreeRegressor(max_features=min(self.mtry, p),
                                         min_samples_split=self.nodesize + 1,
                                         min_samples_leaf=1, random_state=int(seed))
            return tree.fit(X[boot], y[boot])

        self.trees_ = Parallel(n_jobs=self.n_jobs, prefer="threads")(delayed(grow)(s) for s in seeds)
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        return np.mean([t.predict(X) for t in self.trees_], axis=0)


def rf_grid(n_features, length=5):
    """caret var_seq(): mtry values for the default rf grid."""
    if n_features <= length:
        seq = np.floor(np.linspace(2, n_features, n_features))
    else:
        seq = np.floor(np.linspace(2, n_features, length))
    return [dict(mtry=int(m)) for m in dict.fromkeys(seq)]


class GBM:
    """caret method "gbm" (gbm::gbm.fit, gaussian): interaction.depth splits per
    tree (= depth + 1 leaves, grown best first), half of the rows per tree
    (bag.fraction 0.5), at least n.minobsinnode rows per leaf."""

    def __init__(self, n_trees, depth, shrinkage, minobs=10, random_state=None):
        self.n_trees = n_trees
        self.depth = depth
        self.shrinkage = shrinkage
        self.minobs = minobs
        self.random_state = random_state

    def fit(self, X, y):
        self.model_ = GradientBoostingRegressor(
            n_estimators=self.n_trees, learning_rate=self.shrinkage, subsample=0.5,
            max_leaf_nodes=self.depth + 1, max_depth=None, min_samples_leaf=self.minobs,
            random_state=self.random_state).fit(np.asarray(X, dtype=float), np.asarray(y, dtype=float))
        return self

    def predict(self, X):
        return self.model_.predict(np.asarray(X, dtype=float))

    def staged(self, X, n_trees_list):
        """Predictions after each number of trees in n_trees_list (caret sub-models)."""
        want = set(n_trees_list)
        out = {}
        for i, pred in enumerate(self.model_.staged_predict(np.asarray(X, dtype=float)), 1):
            if i in want:
                out[i] = pred
        return out


GBM_GRID = [dict(n_trees=t, depth=d, shrinkage=s)
            for t in (500, 1000) for d in (10, 15, 20) for s in (0.005, 0.01, 0.02)]


# ----------------------------------------------------------------- tuning
def _cv_task(name, X, y, train, test, params, seed):
    """Fit one model (or one gbm family of sub-models) on a fold; return RMSEs."""
    Xtr, ytr, Xte, yte = X[train], y[train], X[test], y[test]
    if name == "gbm":
        m = GBM(max(p["n_trees"] for p in params), params[0]["depth"], params[0]["shrinkage"],
                random_state=seed).fit(Xtr, ytr)
        preds = m.staged(Xte, [p["n_trees"] for p in params])
        return [rmse(preds[p["n_trees"]], yte) for p in params]
    return [rmse(make_model(name, p, seed).fit(Xtr, ytr).predict(Xte), yte) for p in params]


def make_model(name, params, seed=None, n_jobs=None):
    if name == "lm":
        return LM()
    if name == "glmnet":
        return Glmnet(params["alpha"], params["lam"])
    if name == "knn":
        return KNN(params["k"])
    if name == "rf":
        return RF(params["mtry"], random_state=seed, n_jobs=n_jobs)
    if name == "gbm":
        return GBM(params["n_trees"], params["depth"], params["shrinkage"], random_state=seed)
    raise ValueError(name)


def default_grid(name, X, y):
    if name == "lm":
        return [dict()]
    if name == "glmnet":
        return glmnet_grid(X, y)
    if name == "knn":
        return [dict(k=k) for k in (5, 7, 9, 11, 13)]
    if name == "rf":
        return rf_grid(X.shape[1])
    if name == "gbm":
        return GBM_GRID
    raise ValueError(name)


def caret_train(name, X, y, folds, seed=1, n_jobs=1):
    """Tune by 10-fold CV (smallest mean RMSE, first on ties) and refit on all rows.

    lm has nothing to tune; caret's CV for lm is only a report, so it is skipped.
    Returns (fitted model, chosen tuning values, table of mean CV RMSE).
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    grid = default_grid(name, X, y)
    if len(grid) == 1:
        return make_model(name, grid[0], seed, n_jobs).fit(X, y), grid[0], None

    if name == "gbm":   # one fit per (depth, shrinkage) serves both numbers of trees
        families = {}
        for p in grid:
            families.setdefault((p["depth"], p["shrinkage"]), []).append(p)
        groups = list(families.values())
    else:
        groups = [[p] for p in grid]
    all_idx = np.arange(len(y))
    tasks = [(gi, fi) for gi in range(len(groups)) for fi in range(len(folds))]
    res = Parallel(n_jobs=n_jobs)(
        delayed(_cv_task)(name, X, y, np.setdiff1d(all_idx, folds[fi]), folds[fi], groups[gi], seed)
        for gi, fi in tasks)
    scores = {}
    for (gi, fi), r in zip(tasks, res):
        for p, s in zip(groups[gi], r):
            scores.setdefault(id(p), []).append(s)
    table = [dict(p, rmse=float(np.mean(scores[id(p)]))) for p in grid]
    best = min(range(len(grid)), key=lambda i: table[i]["rmse"])
    return make_model(name, grid[best], seed, n_jobs).fit(X, y), grid[best], table
