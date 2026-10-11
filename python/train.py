"""Training: validation of the 18 models per region, then refit on all years (R: Modeling.R).

Example (synthetic data in data/):
    python train.py --data-dir data --out-dir output --model-dir model --mc 8

Writes
  output/validation.txt   RMSE / NRMSE / MAPE of the 18 models per region and the best ones
  output/validation.pdf   RMSE bars per region
  output/coverage.pdf     min-max range of the 18 predictions per validation event
  output/validation.csv   predictions on the validation year, interval, chosen prediction, y
  model/optm.json         best model per region
  model/obj1..4.joblib    models refitted on all years, one file per region
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import joblib
import pandas as pd

from heavyrain.develop import MODEL_COLUMNS, dev, hval_out, pint_mm, pred_out
from heavyrain.features import ENCODING, KEYS, drop_small_damage, makedat, read_events
from heavyrain.plots import coverage, validation_bars

REGIONS = (1, 2, 3, 4)


def parse_args():
    p = argparse.ArgumentParser(description="Train the heavy rain damage models (Modeling.R).")
    p.add_argument("--data-dir", default="data", help="folder with Dat_train/ and Dat_extra/")
    p.add_argument("--out-dir", default="output", help="folder for validation outputs")
    p.add_argument("--model-dir", default="model", help="folder for the saved models")
    p.add_argument("--mc", type=int, default=2, help="parallel jobs (R mc, default 2)")
    p.add_argument("--seed", type=int, default=1, help="random seed (R set.seed(1))")
    p.add_argument("--encoding", default=ENCODING, help="CSV encoding (default euc-kr)")
    return p.parse_args()


def fit_regions(dat, mc, seed):
    objs = {}
    for r in REGIONS:
        print(f"region {r}", flush=True)
        objs[r] = dev(dat[dat["region"] == r], mc=mc, sido=(r == 1), seed=seed)
    return objs


def main():
    args = parse_args()
    data, out, mdir = Path(args.data_dir), Path(args.out_dir), Path(args.model_dir)
    out.mkdir(parents=True, exist_ok=True)
    mdir.mkdir(parents=True, exist_ok=True)
    start = time.time()

    dat = read_events(data / "Dat_train" / "tdat.csv", args.encoding)
    dat = makedat(dat, data / "Dat_extra", args.encoding)
    dat = drop_small_damage(dat)               # keeps y between its 36% and 100% quantiles

    # validation: fit on all years but the last, check on the last year
    print("validation", flush=True)
    last = dat["year"].max()
    tdat, idat = dat[dat["year"] <= last - 1], dat[dat["year"] == last]
    objs = fit_regions(tdat, args.mc, args.seed)
    houts = [hval_out(pred_out(objs[r], idat[idat["region"] == r]), idat[idat["region"] == r])
             for r in REGIONS]
    optm = {f"region{r}": h["best"] for r, h in zip(REGIONS, houts)}
    (mdir / "optm.json").write_text(json.dumps(optm, indent=2))

    with open(out / "validation.txt", "w") as f:
        for r, h in zip(REGIONS, houts):
            f.write(f"\nregion{r}\n{h['vres'].to_string()}\n")
        f.write("\nThe bests:\n" + pd.Series(optm).to_string() + "\n")
    validation_bars(houts, out / "validation.pdf")
    pint = pint_mm(houts)
    coverage(pint, out / "coverage.pdf")

    vout = pd.concat([h["vout"] for h in houts], ignore_index=True)
    pred = vout[KEYS + MODEL_COLUMNS].copy()
    pred["pred_lwr"] = pint["pred_lwr"]
    pred["pred_upr"] = pint["pred_upr"]
    pred["pred_fit"] = [vout.at[i, optm[f"region{r}"]] for i, r in enumerate(vout["region"])]
    pred["y"] = pint["yact"]
    pred["cover"] = ((pint["pred_lwr"] <= pint["yact"]) & (pint["yact"] <= pint["pred_upr"])).astype(int)
    pred.to_csv(out / "validation.csv", index=False, encoding=args.encoding)
    print("best models:", optm)
    print(f"min-max interval covers {pint['ncov']}/{len(pint['yact'])} validation events")

    # refit every region on all years and save
    print("re-fitting", flush=True)
    objs = fit_regions(dat, args.mc, args.seed)
    for r in REGIONS:
        joblib.dump(objs[r], mdir / f"obj{r}.joblib")
    print(f"done in {(time.time() - start) / 60:.1f} min -> {out}, {mdir}")


if __name__ == "__main__":
    main()
