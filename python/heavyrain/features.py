"""Reading the event data and adding the extra features (R: xxfun.R makedat())."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ENCODING = "euc-kr"

# (file, join keys, name column dropped before joining); all but 00_area are
# joined with a one-year lag: the table of year t is used for events of year t+1
EXTRA_TABLES = [
    ("00_area", ["sgg_code"], "sgg"),
    ("01_grdp", ["sgg_code", "year"], "sgg"),
    ("02_money", ["sgg_code", "year"], "sgg"),
    ("03_property_tax", ["sgg_code", "year"], "sgg"),
    ("04_building", ["sgg_code", "year"], "sgg"),
    ("05_city_rate", ["sgg_code", "year"], "sgg"),
    ("06_people", ["sgg_code", "year"], "sgg"),
    ("07_vulnerable", ["sgg_code", "year"], "sgg"),
    ("08_recovery_cost", ["sgg_code", "year"], "sgg"),
    ("09_odat", ["sido_code", "year"], "sido"),
    ("10_ddat", ["sgg_code", "year"], "sgg"),
    ("11_pdat", ["sgg_code", "year"], "sgg"),
    ("12_cDRZ", ["sgg_code", "year"], "sgg"),
    ("13_nDRZ", ["sgg_code", "year"], "sgg"),
]

KEYS = ["region", "sgg_code", "year", "beg_date", "end_date"]


def read_events(path, encoding=ENCODING) -> pd.DataFrame:
    """Read an event file and add ndate (days) and year, as Modeling.R / Prediction.R."""
    dat = pd.read_csv(path, encoding=encoding)
    dat["beg_date"] = pd.to_datetime(dat["beg_date"])
    dat["end_date"] = pd.to_datetime(dat["end_date"])
    dat["ndate"] = (dat["end_date"] - dat["beg_date"]).dt.days + 1
    dat["year"] = dat["beg_date"].dt.year
    return dat


def makedat(mydata: pd.DataFrame, extra_dir, encoding=ENCODING) -> pd.DataFrame:
    """Join the 14 extra tables, fill every missing value with 0, sort by district and start date."""
    mdat = mydata.copy()
    for name, keys, name_col in EXTRA_TABLES:
        tab = pd.read_csv(Path(extra_dir) / f"{name}.csv", encoding=encoding)
        tab = tab.drop(columns=[name_col], errors="ignore")
        if "year" in keys:
            tab["year"] = tab["year"] + 1
        mdat = mdat.merge(tab, on=keys, how="left")
    num = mdat.select_dtypes("number").columns
    mdat[num] = mdat[num].fillna(0)
    return mdat.sort_values(["sgg_code", "beg_date"], kind="mergesort").reset_index(drop=True)


def drop_small_damage(dat: pd.DataFrame, lower=0.36, upper=1.00) -> pd.DataFrame:
    """Keep rows with y between its 36% and 100% quantiles (R: "Outlier elimination")."""
    lo, hi = dat["y"].quantile([lower, upper])
    return dat[(dat["y"] >= lo) & (dat["y"] <= hi)].reset_index(drop=True)
