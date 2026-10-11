"""Linear and logistic regression as R's lm(), glm.fit() and mbest::firthglm.fit(),
plus the model choice made by step() in the R code.

The design matrices passed here already contain the intercept column.
"""

from __future__ import annotations

import numpy as np

DBL_EPS = np.finfo(float).eps


def kept_columns(X: np.ndarray, tol=1e-7) -> list[int]:
    """Columns R keeps in its QR with limited pivoting (LINPACK dqrdc2).

    Columns are taken in order; a column is dropped (its coefficient becomes NA
    in R) when what is left of it after removing the earlier kept columns is
    smaller than ``tol`` times its original length.
    """
    keep, basis = [], np.zeros((X.shape[0], 0))
    for j in range(X.shape[1]):
        v = X[:, j]
        norm = np.linalg.norm(v)
        if norm == 0:
            continue
        r = v - basis @ (basis.T @ v)
        r = r - basis @ (basis.T @ r)
        rn = np.linalg.norm(r)
        if rn < tol * norm:
            continue
        keep.append(j)
        basis = np.column_stack([basis, r / rn])
    return keep


class OLS:
    """R lm(): least squares; columns aliased with earlier ones get no coefficient."""

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        self.keep_ = kept_columns(X)
        beta, *_ = np.linalg.lstsq(X[:, self.keep_], y, rcond=None)
        self.coef_ = np.full(X.shape[1], np.nan)
        self.coef_[self.keep_] = beta
        self.rank_ = len(self.keep_)
        resid = y - X[:, self.keep_] @ beta
        self.rss_ = float(resid @ resid)
        n = len(y)
        self.aic_ = n * np.log(self.rss_ / n) + 2 * self.rank_   # extractAIC.lm
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        return X[:, self.keep_] @ self.coef_[self.keep_]


# binomial family with R's logit link (C code clamps eta at +-30)
def _linkinv(eta):
    e = np.exp(np.clip(eta, -30, 30))
    e = np.where(eta < -30, DBL_EPS, np.where(eta > 30, 1 / DBL_EPS, e))
    return e / (1 + e)


def _mu_eta(eta):
    e = np.exp(np.clip(eta, -30, 30))
    return np.where((eta > 30) | (eta < -30), DBL_EPS, e / (1 + e) ** 2)


def _binomial_deviance(y, mu):
    return float(np.sum(np.where(y > 0, -2 * np.log(mu), -2 * np.log(1 - mu))))


def glm_binomial_fit(X, y, maxit=25, epsilon=1e-8):
    """Maximum likelihood logistic regression by IRLS, as R glm.fit().

    Returns (coefficients, rank, deviance). Used for the AIC comparison that
    R's add1.glm() makes inside step().
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    mu = (y + 0.5) / 2
    eta = np.log(mu / (1 - mu))
    dev_old = _binomial_deviance(y, mu)
    keep = None
    beta = None
    for _ in range(maxit):
        mu_eta = _mu_eta(eta)
        var = mu * (1 - mu)
        z = eta + (y - mu) / mu_eta
        w = np.sqrt(mu_eta ** 2 / var)
        Xw = X * w[:, None]
        keep = kept_columns(Xw, tol=min(1e-7, epsilon / 1000))
        b, *_ = np.linalg.lstsq(Xw[:, keep], z * w, rcond=None)
        beta = np.zeros(X.shape[1])
        beta[keep] = b
        eta = X @ beta
        mu = _linkinv(eta)
        dev = _binomial_deviance(y, mu)
        if abs(dev - dev_old) / (abs(dev) + 0.1) < epsilon:
            break
        dev_old = dev
    coef = np.full(X.shape[1], np.nan)
    coef[keep] = beta[keep]
    return coef, len(keep), dev


class FirthLogistic:
    """Logistic regression with Firth's bias correction, following mbest::firthglm.fit.

    Minimises the penalised deviance  deviance - log det(X'WX)  with Newton
    directions on the modified score X'(y - mu + h(0.5 - mu)) and mbest's
    More-Thuente line search, and stops with mbest's rule
    (score' I^-1 score)^2 <= 2 * epsilon. Following the same iterations
    matters because that stopping rule is loose.
    """

    def __init__(self, maxit=25, epsilon=1e-8, linesearch_maxit=20):
        self.maxit = maxit
        self.epsilon = epsilon
        self.linesearch_maxit = linesearch_maxit

    @staticmethod
    def _eval(X, y, beta):
        eta = X @ beta
        mu = _linkinv(eta)
        w = _mu_eta(eta) ** 2 / (mu * (1 - mu))
        q, r = np.linalg.qr(X * np.sqrt(w)[:, None])
        h = np.sum(q ** 2, axis=1)
        dev = _binomial_deviance(y, mu)
        with np.errstate(divide="ignore"):
            logdet = 2 * np.sum(np.log(np.abs(np.diag(r))))
        score = X.T @ (y - mu + h * (0.5 - mu))
        pen = dev - logdet
        ok = bool(np.isfinite(pen) and np.all(np.isfinite(score)))
        return dict(eta=eta, pen=pen, score=score, r=r, dev=dev, ok=ok)

    def fit(self, X, y):
        from .linesearch import LineSearch

        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        self.keep_ = kept_columns(X)
        Xk = X[:, self.keep_]
        # starting values as in firthglm.fit: weighted LS at mu = 0.5
        mustart = (y + 0.5) / 2
        beta, *_ = np.linalg.lstsq(0.5 * Xk, np.log(mustart / (1 - mustart)), rcond=None)
        obj0 = self._eval(Xk, y, beta)
        if not obj0["ok"]:
            raise ValueError("cannot find valid starting values")
        self.converged_ = False
        for _ in range(self.maxit):
            val0 = 0.5 * obj0["pen"]
            grad0 = -obj0["score"]
            r = obj0["r"]
            search = np.linalg.solve(r, np.linalg.solve(r.T, obj0["score"]))
            deriv0 = float(search @ grad0)
            if deriv0 ** 2 <= 2 * self.epsilon:
                self.converged_ = True
                break
            if deriv0 >= 0 or np.linalg.cond(r, 1) >= 1e8:
                search = -grad0
                deriv0 = float(search @ grad0)
            search_eta = Xk @ search
            with np.errstate(divide="ignore"):
                ratio = (np.abs(obj0["eta"]) + 1) / np.abs(search_eta)
            step_max = min(10 * ratio.min(), np.finfo(float).max)
            step_min = max(DBL_EPS * ratio.min(), np.finfo(float).tiny)
            step0 = 1.0 if step_min <= 1.0 <= step_max else step_min + 0.5 * (step_max - step_min)
            while True:
                coef = beta + step0 * search
                obj = self._eval(Xk, y, coef)
                if obj["ok"]:
                    break
                step_max = step0
                if step0 < 0.01:
                    step0 = 2 ** (0.5 * np.log2(step_min) + 0.5 * np.log2(step_max))
                else:
                    step0 = step_min + 0.5 * (step_max - step_min)
            ls = LineSearch(val0, deriv0, step0, step_min=step_min, step_max=step_max)
            for _ in range(self.linesearch_maxit):
                ls.update(0.5 * obj["pen"], float(search @ -obj["score"]))
                if ls.converged:
                    break
                coef = beta + ls.step * search
                obj = self._eval(Xk, y, coef)
            if not ls.converged:
                break       # mbest: "line search failed to converge"
            beta, obj0 = coef, obj
        dev = obj0["dev"]
        self.coef_ = np.full(X.shape[1], np.nan)
        self.coef_[self.keep_] = beta
        self.rank_ = len(self.keep_)
        self.deviance_ = dev
        self.aic_ = dev + 2 * self.rank_
        return self

    def predict_proba(self, X):
        X = np.asarray(X, dtype=float)
        return _linkinv(X[:, self.keep_] @ self.coef_[self.keep_])


def step_lm(X_lower, X_upper, y):
    """R step(lm(lower), scope = list(lower, upper), direction = "both").

    In the R code the two formulas differ only by the area:tot term, so step()
    keeps the larger model exactly when its AIC is smaller.
    Returns (model, used_upper).
    """
    f = OLS().fit(X_lower, y)
    g = OLS().fit(X_upper, y)
    if g.rank_ > f.rank_ and g.aic_ < f.aic_:
        return g, True
    return f, False


def step_firth(X_lower, X_upper, y):
    """R step() on a Firth logistic regression, as it behaves in the R code.

    R's add1.glm() and drop1.glm() refit the candidate models with ordinary
    maximum likelihood (glm.fit), not with the Firth method, while the current
    model keeps its Firth AIC / deviance. step() therefore does:
      1. add area:tot if, by maximum likelihood, deviance + 2 * rank gets smaller;
      2. refit with Firth; stop there if the Firth AIC did not improve;
      3. otherwise ask drop1.glm(): drop the term again if the maximum
         likelihood deviance of the smaller model + 2 * rank is below the
         Firth deviance of the larger model + 2 * rank (step() then ends on
         the smaller model).
    Returns (model, used_upper).
    """
    f = FirthLogistic().fit(X_lower, y)
    _, rank_f, dev_f = glm_binomial_fit(X_lower, y)
    _, rank_g, dev_g = glm_binomial_fit(X_upper, y)
    if not (rank_g > rank_f and dev_g + 2 * rank_g < dev_f + 2 * rank_f):
        return f, False
    g = FirthLogistic().fit(X_upper, y)
    if g.aic_ >= f.aic_ + 1e-7:
        return g, True
    if dev_f + 2 * rank_f < g.deviance_ + 2 * g.rank_:
        return f, False
    return g, True
