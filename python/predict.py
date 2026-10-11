"""Prediction with the saved models (R: Prediction.R).

Example (synthetic data in data/):
    python predict.py --data-dir data --model-dir model --out output/prediction.csv

For every input row: the predictions of the 18 models, their minimum and
maximum (pred_lwr, pred_upr) and the prediction of the region's best model
(pred_fit). Regions without input rows are skipped; the R code stops there.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd

from heavyrain.develop import MODEL_COLUMNS, pred_out
from heavyrain.features import ENCODING, makedat, read_events

REGIONS = (1, 2, 3, 4)


def parse_args():
    p = argparse.ArgumentParser(description="Predict heavy rain damage (Prediction.R).")
    p.add_argument("--data-dir", default="data", help="folder with Dat_extra/ (and Dat_input/)")
    p.add_argument("--input", default=None, help="input CSV (default: <data-dir>/Dat_input/idat.csv)")
    p.add_argument("--model-dir", default="model", help="folder with obj1..4.joblib and optm.json")
    p.add_argument("--out", default="output/prediction.csv", help="output CSV")
    p.add_argument("--encoding", default=ENCODING, help="CSV encoding (default euc-kr)")
    return p.parse_args()


def main():
    args = parse_args()
    data, mdir = Path(args.data_dir), Path(args.model_dir)
    src = Path(args.input) if args.input else data / "Dat_input" / "idat.csv"
    pdat = makedat(read_events(src, args.encoding), data / "Dat_extra", args.encoding)
    optm = json.loads((mdir / "optm.json").read_text())

    parts = []
    for r in REGIONS:
        rows = pdat[pdat["region"] == r]
        if rows.empty:
            print(f"region {r}: no input rows, skipped")
            continue
        pobj = pred_out(joblib.load(mdir / f"obj{r}.joblib"), rows)
        pobj["pred_lwr"] = pobj[MODEL_COLUMNS].min(axis=1)
        pobj["pred_upr"] = pobj[MODEL_COLUMNS].max(axis=1)
        pobj["pred_fit"] = pobj[optm[f"region{r}"]]
        parts.append(pobj)
        print(f"region {r}: {len(rows)} rows, best model {optm[f'region{r}']}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.concat(parts, ignore_index=True).to_csv(out, index=False, encoding=args.encoding)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
