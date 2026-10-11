"""Figures of Modeling.R: validation RMSE bars and min-max coverage (labels in English only)."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402


def validation_bars(houts, path):
    """One page per region: RMSE of the 18 models, grouped by method, bars = input set 1/2/3."""
    with PdfPages(path) as pdf:
        for h in houts:
            vres = h["vres"]
            methods = sorted(vres["ml"].unique())      # ggplot orders the method names alphabetically
            x = np.arange(len(methods))
            fig, ax = plt.subplots(figsize=(9, 6))
            for k, t in enumerate((1, 2, 3)):
                vals = [vres[(vres.ml == m) & (vres.type == t)]["rmse"].iloc[0] for m in methods]
                ax.bar(x + (k - 1) * 0.27, vals, width=0.27, label=f"type {t}")
            ax.set_xticks(x, methods)
            ax.set_xlabel("ml")
            ax.set_ylabel("rmse")
            ax.set_title(f"Region:{h['region']}, The best: {h['best']}")
            ax.legend(title="type", loc="center left", bbox_to_anchor=(1.01, 0.5))
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)


def coverage(pint, path):
    """Two pages (raw and log scale): per validation event, the min-max range of the
    18 predictions (bar), the observed y (filled circle) and the mean prediction (open circle)."""
    n = len(pint["yact"])
    x = np.arange(1, n + 1)
    title = f"min ~ max: {pint['ncov']}/{n}({pint['pcov']})"
    with PdfPages(path) as pdf:
        for log in (False, True):
            f = np.log if log else (lambda v: v)
            fig, ax = plt.subplots(figsize=(9, 6))
            ax.vlines(x, f(pint["pred_lwr"]), f(pint["pred_upr"]), color="black", linewidth=0.8)
            ax.scatter(x, f(pint["yact"]), s=18, c=plt.cm.hsv(np.linspace(0, 1, n, endpoint=False)),
                       edgecolors="black", linewidths=0.4, zorder=3, label="observed y")
            ax.scatter(x, f(pint["pred_avg"]), s=14, facecolors="none", edgecolors="black",
                       zorder=3, label="mean of 18 predictions")
            ax.set_xlabel("Events")
            ax.set_ylabel("log(Predicted values)" if log else "Predicted values")
            ax.set_title(title)
            ax.legend(loc="upper left", fontsize=8)
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)
