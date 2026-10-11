"""Make a synthetic data set with the layout the heavy rain damage code expects.

The real data of this project are not available, so this script writes fake
data with the same file names, column names and encoding (EUC-KR) as the R
code reads. The numbers are random and only roughly plausible; they are meant
for running and testing the code, not for any conclusion about real damage.

What the columns mean in the real data is not written in the R code. The
meanings below are guesses used only to generate sensible-looking values:
  tot        total rainfall of the event (mm)
  x1..x24    largest rainfall summed over 1..24 consecutive hours (mm)
  d7..d1     daily rainfall 7..1 days before the event (mm)
  y          damage of the event in one district (thousand KRW)

Output (default: data/ next to this script)
  Dat_train/tdat.csv   events of 2008-2018 with damage y
  Dat_input/idat.csv   events of 2019 without y (input for prediction)
  Dat_extra/00..13     district or province tables, one row per year
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

REGIONS = {1: [11, 23, 31], 2: [32, 33, 34], 3: [35, 36, 24], 4: [37, 38, 26]}
N_SGG = 8                       # districts per province
TRAIN_YEARS = range(2008, 2019)
INPUT_YEAR = 2019
TABLE_YEARS = range(2006, 2020)  # tables are joined with a one-year lag
ENCODING = "euc-kr"


def make_districts(rng):
    rows = []
    for region, sidos in REGIONS.items():
        for sido in sidos:
            for j in range(1, N_SGG + 1):
                code = sido * 1000 + 10 * j
                rows.append(dict(region=region, sido_code=sido, sido=f"가상도{sido}",
                                 sgg_code=code, sgg=f"가상군{code}"))
    d = pd.DataFrame(rows)
    n = len(d)
    # fixed district traits used for the tables and for the damage
    d["area"] = np.round(np.exp(rng.normal(np.log(400), 0.7, n)), 2)          # km2
    d["urban"] = rng.beta(2, 3, n)                                               # 0..1
    d["pop0"] = np.exp(rng.normal(np.log(150_000), 0.8, n)) * (0.5 + d["urban"])
    d["frailty"] = rng.normal(0, 1, n)                                           # hidden
    return d


def make_tables(d, rng):
    """One table per extra data source, one row per district (or province) and year."""
    years = np.array(TABLE_YEARS)
    t = years - years[0]
    tabs = {}
    rep = d.loc[d.index.repeat(len(years))].reset_index(drop=True)
    rep["year"] = np.tile(years, len(d))
    tt = np.tile(t, len(d))
    noise = lambda s: rng.normal(0, s, len(rep))
    pop = rep["pop0"] * (1 + 0.005 * tt) * np.exp(noise(0.02))
    key = rep[["sgg_code", "sgg", "year"]]

    tabs["00_area"] = d[["sgg_code", "sgg", "area"]]
    tabs["01_grdp"] = key.assign(grdp=np.round(pop * 0.03 * (1 + rep["urban"]) * (1.03 ** tt) * np.exp(noise(0.1))))
    tabs["02_money"] = key.assign(money=np.round(2e5 + pop * 2.5 * np.exp(noise(0.1))))
    tabs["03_property_tax"] = key.assign(property_tax=np.round(pop * 0.08 * (0.5 + rep["urban"]) * np.exp(noise(0.15))))
    tabs["04_building"] = key.assign(building=np.round(pop / 6 * np.exp(noise(0.1))))
    tabs["05_city_rate"] = key.assign(city_rate=np.round(100 * np.clip(rep["urban"] + 0.004 * tt + noise(0.01), 0, 1), 1))
    tabs["06_people"] = key.assign(people=np.round(pop))
    vul = pop * np.clip(0.12 + 0.04 * rep["frailty"] - 0.05 * rep["urban"] + noise(0.01), 0.02, 0.4)
    tabs["07_vulnerable"] = key.assign(vulnerable=np.round(vul))
    tabs["08_recovery_cost"] = key.assign(recovery_cost=np.round(np.exp(rng.normal(np.log(3e5), 1.2, len(rep)) + 0.5 * rep["frailty"])))

    # province level table (joined by sido_code and year)
    prov = d.groupby("sido_code").agg(sido=("sido", "first"), area=("area", "sum")).reset_index()
    prov = prov.loc[prov.index.repeat(len(years))].reset_index(drop=True)
    prov["year"] = np.tile(years, prov["sido_code"].nunique())
    tabs["09_odat"] = prov.assign(
        tlength=np.round(prov["area"] * 0.3 * np.exp(rng.normal(0, 0.05, len(prov))), 1),
        owgr=np.round(rng.uniform(10, 60, len(prov)), 1),
    )[["sido_code", "sido", "year", "tlength", "owgr"]]

    # past damage counts / sums / averages for four damage types
    lam = np.exp(0.6 * rep["frailty"].to_numpy())[None, :] * np.array([[2.0], [1.0], [0.5], [0.05]])
    cnt = rng.poisson(lam).T
    size = np.exp(rng.normal(np.log(5e4), 1.0, cnt.shape))
    ddat = key.copy()
    for k in range(4):
        ddat[f"cnt{k + 1}"] = cnt[:, k]
        ddat[f"sum{k + 1}"] = np.round(cnt[:, k] * size[:, k])
        ddat[f"avg{k + 1}"] = np.round(np.where(cnt[:, k] > 0, size[:, k], 0))
    tabs["10_ddat"] = ddat
    tabs["11_pdat"] = key.assign(ccnt=rng.poisson(3 + 5 * rep["urban"]),
                                 ctpc=np.round(rng.gamma(2, 50, len(rep)) * (1 + rep["urban"]), 1))
    tabs["12_cDRZ"] = key.assign(cDRZ=rng.poisson(np.exp(0.5 * rep["frailty"])))
    tabs["13_nDRZ"] = key.assign(nDRZ=rng.binomial(1, 0.03, len(rep)))   # mostly zero

    # a few missing rows, which the R code fills with 0
    for name in ["02_money", "07_vulnerable", "10_ddat"]:
        tab = tabs[name]
        tabs[name] = tab.drop(index=rng.choice(tab.index, size=len(tab) // 50, replace=False))
    return tabs


def storm_rain(rng, hours, peak):
    """Hourly rainfall of one storm: a random-width bump plus noise."""
    h = np.arange(hours)
    centre = rng.uniform(0.2, 0.8) * hours
    width = rng.uniform(3, 12)
    shape = np.exp(-0.5 * ((h - centre) / width) ** 2)
    return np.maximum(peak * shape * rng.gamma(2.0, 0.5, hours), 0)


def max_window_sums(rain):
    c = np.concatenate([[0], np.cumsum(rain)])
    return [max(c[k:] - c[:-k]) if k <= len(rain) else c[-1] for k in range(1, 25)]


def make_events(d, years, rng, tabs, with_y):
    rows = []
    for year in years:
        n_events = rng.integers(10, 15)
        for _ in range(n_events):
            beg = pd.Timestamp(year, 6, 1) + pd.Timedelta(days=int(rng.integers(0, 110)))
            ndate = int(rng.choice([1, 2, 3, 4, 5], p=[0.3, 0.3, 0.2, 0.12, 0.08]))
            end = beg + pd.Timedelta(days=ndate - 1)
            regions = rng.choice(list(REGIONS), size=rng.integers(1, 3), replace=False)
            strength = rng.gamma(4, 7)            # storm peak, mm per hour
            for region in regions:
                for sido in REGIONS[region]:
                    if rng.random() < 0.4:
                        continue
                    dd = d[d.sido_code == sido]
                    hit = dd.sample(n=int(rng.integers(2, N_SGG + 1)), random_state=int(rng.integers(1e9)))
                    for _, r in hit.iterrows():
                        rain = storm_rain(rng, 24 * ndate, strength * np.exp(rng.normal(0, 0.3)))
                        x = max_window_sums(rain)
                        row = dict(sgg_code=r.sgg_code, sgg=r.sgg, sido_code=sido, sido=r.sido,
                                   region=region, beg_date=beg.date().isoformat(),
                                   end_date=end.date().isoformat(), tot=round(float(rain.sum()), 1))
                        for k in range(7, 0, -1):
                            row[f"d{k}"] = round(float(rng.gamma(0.6, 15) * (rng.random() < 0.5)), 1)
                        for k in range(24):
                            row[f"x{k + 1}"] = round(float(x[k]), 1)
                        rows.append(row)
    ev = pd.DataFrame(rows)
    # one row per district and event period (two storms may start on the same day)
    ev = ev.drop_duplicates(["sgg_code", "beg_date", "end_date"]).reset_index(drop=True)
    if with_y:
        ev["y"] = damage(ev, d, tabs, rng)
    return ev


def damage(ev, d, tabs, rng):
    """Damage (thousand KRW) from rainfall, district traits and last year's tables."""
    m = ev.merge(d[["sgg_code", "area", "urban", "frailty"]], on="sgg_code", how="left")
    year = pd.to_datetime(m["beg_date"]).dt.year
    vul = tabs["07_vulnerable"].set_index(["sgg_code", "year"])["vulnerable"]
    ppl = tabs["06_people"].set_index(["sgg_code", "year"])["people"]
    idx = pd.MultiIndex.from_arrays([m["sgg_code"], year - 1])
    vul_rate = (vul.reindex(idx).to_numpy() / ppl.reindex(idx).to_numpy())
    vul_rate = np.nan_to_num(vul_rate, nan=np.nanmean(vul_rate))
    antecedent = m[[f"d{k}" for k in range(1, 8)]].sum(axis=1)
    log_y = (0.7
             + 0.9 * np.log1p(m["x3"]) + 0.5 * np.log1p(m["tot"])
             + 0.35 * np.log(m["area"]) + 0.25 * np.log(m["area"]) * np.log1p(m["tot"]) / 5
             + 6.0 * vul_rate + 0.4 * m["frailty"] - 0.8 * m["urban"]
             + 0.004 * antecedent + 0.8 * (m["x1"] > 40)
             + 0.15 * (m["region"] - 2.5)
             + rng.normal(0, 0.8, len(m)))
    return np.round(np.exp(log_y)).astype(int)


def main():
    p = argparse.ArgumentParser(description="Write synthetic data in the layout of the R code.")
    p.add_argument("--out", default=str(Path(__file__).with_name("data")), help="output folder")
    p.add_argument("--seed", type=int, default=2019)
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    out = Path(args.out)
    for sub in ["Dat_train", "Dat_input", "Dat_extra"]:
        (out / sub).mkdir(parents=True, exist_ok=True)

    d = make_districts(rng)
    tabs = make_tables(d, rng)
    for name, tab in tabs.items():
        tab.to_csv(out / "Dat_extra" / f"{name}.csv", index=False, encoding=ENCODING)

    tdat = make_events(d, TRAIN_YEARS, rng, tabs, with_y=True)
    idat = make_events(d, [INPUT_YEAR], rng, tabs, with_y=False)
    tdat.to_csv(out / "Dat_train" / "tdat.csv", index=False, encoding=ENCODING)
    idat.to_csv(out / "Dat_input" / "idat.csv", index=False, encoding=ENCODING)

    print(f"districts {len(d)}, training rows {len(tdat)}, input rows {len(idat)} -> {out}")
    print("training rows per region:", tdat.groupby("region").size().to_dict())
    print("input rows per region   :", idat.groupby("region").size().to_dict())


if __name__ == "__main__":
    main()
