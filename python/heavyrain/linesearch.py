"""Line search used by mbest::firthglm.fit (More and Thuente), translated from mbest's linesearch.R.

Kept close to the R code so that Firth fits follow the same iterations as in R.
"""

from __future__ import annotations

import math
import warnings


class LineSearch:
    def __init__(self, value, deriv, step, value_tol=1e-4, deriv_tol=0.9, step_tol=1e-7,
                 step_min=1e-10, step_max=1e10, extrap_lower=1.1, extrap_upper=4.0,
                 extrap_max=0.66, bisection_width=0.66):
        if not (deriv < 0 and math.isfinite(deriv)):
            raise ValueError("initial derivative must be negative and finite")
        self.c = dict(value_tol=value_tol, deriv_tol=deriv_tol, step_tol=step_tol,
                      step_min=step_min, step_max=step_max, extrap_lower=extrap_lower,
                      extrap_upper=extrap_upper, extrap_max=extrap_max,
                      bisection_width=bisection_width)
        self.gtest = deriv * value_tol
        self.ftest = value + step * self.gtest
        self.lower = dict(position=0.0, value=value, deriv=deriv)
        self.upper = dict(self.lower)
        self.interval = [0.0, step + extrap_upper * step]
        self.step = step
        self.value0 = value
        self.deriv0 = deriv
        self.bracket = False
        self.auxfun = True
        self.old_width1 = math.inf
        self.old_width2 = math.inf
        self.converged = False

    def update(self, value, deriv):
        c = self.c
        if value <= self.ftest and abs(deriv) <= -(c["deriv_tol"] * self.deriv0):
            self.converged = True
            return self
        stuck = False
        if (self.bracket and self.step <= self.interval[0]) or self.step >= self.interval[1]:
            stuck = True
        elif self.bracket and (self.interval[1] - self.interval[0]) <= c["step_tol"] * self.interval[0]:
            stuck = True
        elif self.step == c["step_max"] and value <= self.ftest and deriv <= self.gtest:
            stuck = True
        elif self.step == c["step_min"] and (value > self.ftest or deriv >= self.gtest):
            stuck = True
        if stuck:
            return self
        self._update_step(dict(position=self.step, value=value, deriv=deriv))
        return self

    def _update_step(self, test):
        gtest = self.gtest
        aux = self.auxfun and not (test["value"] <= self.ftest and test["deriv"] > 0)
        test0, lower0, upper0 = test, self.lower, self.upper
        if aux:
            def modify(e):
                return dict(position=e["position"], value=e["value"] - e["position"] * gtest,
                            deriv=e["deriv"] - gtest)
            test = modify(test0)
            self.lower = modify(lower0)
            self.upper = modify(upper0)
        else:
            self.auxfun = False
        self._unsafe_step(test)
        if aux:
            def restore(e):
                if e["position"] == test0["position"]:
                    return test0
                if e["position"] == lower0["position"]:
                    return lower0
                return upper0
            self.lower = restore(self.lower)
            self.upper = restore(self.upper)
        self._maybe_bisect()

    def _maybe_bisect(self):
        if not self.bracket:
            return
        l, u = self.lower["position"], self.upper["position"]
        w = self.interval[1] - self.interval[0]
        if w >= self.c["bisection_width"] * self.old_width2:
            self.step = l + 0.5 * (u - l)
            self.ftest = self.value0 + self.step * self.gtest
        self.old_width2 = self.old_width1
        self.old_width1 = w

    def _unsafe_step(self, test):
        c = self.c
        step = _trial_value(self.interval, self.lower, self.upper, test, self.bracket, c)
        self.bracket, self.interval, self.lower, self.upper = _update_interval(
            self.lower, self.upper, test, step, self.bracket, c)
        step = max(c["step_min"], min(c["step_max"], step))
        self.ftest = self.value0 + step * self.gtest
        self.step = step


def _sign(x):
    x = float(x)
    return (x > 0) - (x < 0)


def _update_interval(lower, upper, test, step, bracket, c):
    fl, gl = lower["value"], lower["deriv"]
    t, ft, gt = test["position"], test["value"], test["deriv"]
    if ft > fl:
        return True, sorted([lower["position"], test["position"]]), lower, test
    if _sign(gt) == _sign(gl):
        if bracket:
            return True, sorted([test["position"], upper["position"]]), test, upper
        return False, [step + c["extrap_lower"] * (step - t), step + c["extrap_upper"] * (step - t)], test, upper
    return True, sorted([test["position"], lower["position"]]), test, lower


def _trial_value(interval, lower, upper, test, bracket, c):
    tmin, tmax = interval
    l, fl, gl = lower["position"], lower["value"], lower["deriv"]
    u, fu, gu = upper["position"], upper["value"], upper["deriv"]
    t, ft, gt = test["position"], test["value"], test["deriv"]
    cm = _cubic_min(l, fl, gl, t, ft, gt)          # may be NaN
    q = _quadr_min(l, fl, gl, t, ft)
    s = _secant_min(l, gl, t, gt)
    if ft > fl:
        return cm if abs(cm - l) < abs(q - l) else cm + 0.5 * (q - cm)
    if _sign(gt) != _sign(gl):
        return cm if abs(cm - t) >= abs(s - t) else s
    if abs(gt) <= abs(gl):
        if not math.isnan(cm) and _sign(cm - t) == _sign(cm - l):
            c1 = cm
        else:
            c1 = tmax if t > l else tmin
        if bracket:
            if t > l:
                guard = lambda x: min(x, t + c["extrap_max"] * (u - t))
            else:
                guard = lambda x: max(x, t + c["extrap_max"] * (u - t))
            return guard(c1) if abs(t - c1) < abs(t - s) else guard(s)
        clip = lambda x: max(tmin, min(tmax, x))
        return clip(c1) if abs(t - c1) > abs(t - s) else clip(s)
    if bracket:
        return _cubic_min(t, ft, gt, u, fu, gu)
    return tmax if t > l else tmin


def _quadr_min(u, fu, du, v, fv):
    a = v - u
    return u + (du / ((fu - fv) / a + du) / 2) * a


def _secant_min(u, du, v, dv):
    a = u - v
    return v + dv / (dv - du) * a


def _cubic_min(u, fu, du, v, fv, dv):
    d = v - u
    theta = (fu - fv) * 3 / d + du + dv
    s = max(abs(theta), abs(du), abs(dv))
    a = theta / s
    disc = a * a - (du / s) * (dv / s)
    if disc < 0:
        return math.nan
    gamma0 = s * math.sqrt(disc)
    gamma = -gamma0 if v < u else gamma0
    p = gamma - du + theta
    q = gamma - du + gamma + dv
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = p / q if q != 0 else math.nan
    return u + r * d
