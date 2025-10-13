#!/usr/bin/env python3
"""
Assessment 3 — Group Project Pipeline (Python Script)
Project: Bat vs. Rat — Foraging Behaviour under Rat Presence

USAGE:
    python assessment3_pipeline.py

Assumptions:
    - d1_clean.csv and d2_clean.csv are in the SAME folder where you run this script.
    - Outputs are written to reports/figures and reports/tables.

Dependencies:
    pandas, numpy, matplotlib, statsmodels
"""

import sys
import subprocess
import warnings
from pathlib import Path

def ensure_packages(pkgs):
    for pkg in pkgs:
        try:
            __import__(pkg)
        except ImportError:
            print(f"[setup] Installing missing package: {pkg} ...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])

ensure_packages(["pandas", "numpy", "matplotlib", "statsmodels"])

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf

warnings.filterwarnings("ignore")

BASE = Path(".").resolve()
FIGDIR = BASE / "reports" / "figures"
TABDIR = BASE / "reports" / "tables"
FIGDIR.mkdir(parents=True, exist_ok=True)
TABDIR.mkdir(parents=True, exist_ok=True)

D1 = BASE / "d1_clean.csv"
D2 = BASE / "d2_clean.csv"

def die(msg):
    print(f"[error] {msg}")
    sys.exit(1)

if not D1.exists() or not D2.exists():
    die(f"CSV files not found in {BASE}. Expecting d1_clean.csv and d2_clean.csv")

print(f"[info] Using data at: {D1} and {D2}")

from pandas.api.types import is_datetime64_any_dtype as is_dt

def safe_to_datetime(df, cols):
    for c in cols:
        if c in df.columns and not is_dt(df[c]):
            df[c] = pd.to_datetime(df[c], errors="coerce")

def demote_nullable_ints(df):
    for c in df.columns:
        if str(df[c].dtype) == "Int64":
            df[c] = df[c].astype("float64")

def coerce_numeric(df, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype("float64")

def zscore(s):
    s = pd.to_numeric(s, errors="coerce")
    mu, sd = s.mean(), s.std(ddof=0)
    return (s - mu) / (sd if (sd and sd>0) else 1.0)

print("[step] Loading CSVs ...")
d1 = pd.read_csv(D1, low_memory=False)
d2 = pd.read_csv(D2, low_memory=False)
print(f"[info] d1 shape: {d1.shape} | d2 shape: {d2.shape}")

print("[step] Applying dtype safety patch ...")
likely_time_cols_d1 = [c for c in d1.columns if any(k in c.lower() for k in ["time","start","end","period"])]
likely_time_cols_d2 = [c for c in d2.columns if "time" in c.lower()]

safe_to_datetime(d1, likely_time_cols_d1)
safe_to_datetime(d2, likely_time_cols_d2)
demote_nullable_ints(d1); demote_nullable_ints(d2)
coerce_numeric(d1, [c for c in ["risk","reward","bat_landing_to_food","hours_after_sunset"] if c in d1.columns])
coerce_numeric(d2, [c for c in ["rat_minutes","rat_arrival_number","bat_landing_number","hours_after_sunset"] if c in d2.columns])
if "season" in d1.columns: d1["season"] = d1["season"].astype("category")
if "season" in d2.columns: d2["season"] = d2["season"].astype("category")
print("[info] Dtypes patched.")

print("[step] Feature engineering ...")
if all(c in d1.columns for c in ["start_time","rat_period_start","rat_period_end"]):
    d1["rat_present_during_landing"] = (
        (d1["start_time"] >= d1["rat_period_start"]) &
        (d1["start_time"] <= d1["rat_period_end"])
    ).astype(float)
else:
    d1["rat_present_during_landing"] = np.nan

start_like = None
for candidate in ["start_time","time","landing_time"]:
    if candidate in d1.columns:
        start_like = candidate; break

if start_like is not None:
    d1["time_bin"] = pd.to_datetime(d1[start_like], errors="coerce").dt.floor("30min")
else:
    d1["time_bin"] = pd.NaT

if "time" in d2.columns:
    d2["time_bin"] = pd.to_datetime(d2["time"], errors="coerce").dt.floor("30min")
else:
    d2["time_bin"] = pd.NaT

if all(c in d2.columns for c in ["rat_minutes","rat_arrival_number"]):
    rm = pd.to_numeric(d2["rat_minutes"], errors="coerce")
    ra = pd.to_numeric(d2["rat_arrival_number"], errors="coerce")
    rm_z = (rm - rm.mean()) / (rm.std(ddof=0) if rm.std(ddof=0)!=0 else 1.0)
    ra_z = (ra - ra.mean()) / (ra.std(ddof=0) if ra.std(ddof=0)!=0 else 1.0)
    d2["rat_pressure_idx"] = (rm_z.fillna(0) + ra_z.fillna(0)).astype(float)
else:
    d2["rat_pressure_idx"] = np.nan

d1 = d1.merge(
    d2[["time_bin","rat_pressure_idx"]].drop_duplicates("time_bin"),
    on="time_bin", how="left"
)
print("[info] Feature engineering complete.")

def save_plot(figpath):
    Path(figpath).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(figpath, dpi=300)
    plt.show()
    print(f"[plot] Saved: {figpath}")

print("[step] EDA plots ...")
try:
    if "bat_landing_to_food" in d1.columns and "rat_present_during_landing" in d1.columns:
        plt.figure()
        d1.boxplot(column="bat_landing_to_food", by="rat_present_during_landing")
        plt.title("Vigilance (bat_landing_to_food) by Rat Presence")
        plt.suptitle("")
        plt.xlabel("Rat present during landing (0/1)")
        plt.ylabel("Seconds to approach food")
        save_plot(FIGDIR / "fig_vigilance_by_rat_presence_box.png")
except Exception as e:
    print("[warn] EDA plot 1 failed:", e)

try:
    if "hours_after_sunset" in d2.columns and "rat_arrival_number" in d2.columns:
        plt.figure()
        grp = d2.dropna(subset=["hours_after_sunset"]).copy()
        grp["hrs_bin"] = grp["hours_after_sunset"].round(1)
        ser = grp.groupby("hrs_bin")["rat_arrival_number"].mean().sort_index()
        plt.plot(ser.index.values, ser.values)
        plt.title("Avg Rat Arrivals vs Hours After Sunset")
        plt.xlabel("Hours After Sunset (bin=0.1h)")
        plt.ylabel("Mean Rat Arrivals (per 30-min window)")
        save_plot(FIGDIR / "fig_rat_arrivals_vs_hours.png")
except Exception as e:
    print("[warn] EDA plot 2 failed:", e)

print("[step] Investigation A — Modeling (robust) ...")

def model_investigation_A(d1):
    cols = ["risk"]
    for c in ["rat_present_during_landing","rat_pressure_idx","hours_after_sunset","season"]:
        if c in d1.columns: cols.append(c)
    mf = d1[cols].copy()

    mf["risk"] = pd.to_numeric(mf["risk"], errors="coerce")
    mf = mf[mf["risk"].isin([0,1])]

    for c in ["rat_present_during_landing","rat_pressure_idx","hours_after_sunset"]:
        if c in mf.columns:
            mf[c] = pd.to_numeric(mf[c], errors="coerce")

    if "season" in mf.columns:
        mf = pd.get_dummies(mf, columns=["season"], drop_first=True)

    pred_cols = [c for c in mf.columns if c != "risk"]
    mf = mf.dropna(subset=pred_cols)
    if mf["risk"].nunique() < 2:
        raise RuntimeError("`risk` has one class after cleaning; cannot fit logistic model.")

    for c in ["rat_pressure_idx","hours_after_sunset"]:
        if c in mf.columns:
            mf[c] = zscore(mf[c])

    X = mf[pred_cols].copy()

    zero_var = [c for c in X.columns if X[c].nunique(dropna=True) <= 1]
    if zero_var:
        print("[diag] Dropping zero-variance:", zero_var)
        X = X.drop(columns=zero_var, errors="ignore")

    dup_drop = []
    seen = {}
    for c in X.columns:
        key = tuple(np.nan_to_num(X[c].values, nan=999123).round(12))
        if key in seen:
            dup_drop.append(c)
        else:
            seen[key] = c
    if dup_drop:
        print("[diag] Dropping duplicate columns:", dup_drop)
        X = X.drop(columns=dup_drop, errors="ignore")

    if X.shape[1] >= 2:
        corr = X.corr().abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        to_drop = list(set(upper.columns[(upper > 0.995).any()]))
        if to_drop:
            print("[diag] Dropping highly correlated:", to_drop)
            X = X.drop(columns=to_drop, errors="ignore")

    X = X.astype(float)
    y = mf["risk"].astype(float)
    X = sm.add_constant(X, has_constant="add")

    print("[info] Design matrix:", X.shape, "| predictors:", list(X.columns))

    try:
        fitted = sm.GLM(y, X, family=sm.families.Binomial()).fit()
    except Exception as e:
        print("[warn] Unregularized GLM failed ->", type(e).__name__, str(e))
        fitted = None
        for alpha in [1.0, 5.0, 10.0]:
            try:
                print(f"[info] Trying regularized GLM alpha={alpha} (L2) ...")
                fitted = sm.GLM(y, X, family=sm.families.Binomial()).fit_regularized(alpha=alpha, L1_wt=0.0, maxiter=500)
                break
            except Exception as e2:
                print("  [fail]", type(e2).__name__, str(e2))
        if fitted is None:
            raise RuntimeError("All logistic fits failed (possible separation).")

    params = fitted.params
    try:
        ci = fitted.conf_int()
        OR = np.exp(params)
        OR_lo = np.exp(ci[0]); OR_hi = np.exp(ci[1])
        or_tbl = pd.DataFrame({"OR": OR, "CI_low": OR_lo, "CI_high": OR_hi})
    except Exception:
        or_tbl = pd.DataFrame({"OR": np.exp(params)})
    or_csv = TABDIR / "investigationA_logit_odds_ratios.csv"
    or_tbl.to_csv(or_csv)
    print("[save] ", or_csv)

    lhs = "bat_landing_to_food"
    if lhs in d1.columns:
        use_cols = [c for c in X.columns if c != "const"]
        mfl = d1[[lhs] + [c for c in use_cols if c in d1.columns]].copy()
        mfl[lhs] = pd.to_numeric(mfl[lhs], errors="coerce")
        for c in mfl.columns:
            if c != lhs:
                mfl[c] = pd.to_numeric(mfl[c], errors="coerce")
        mfl = mfl.dropna()
        if len(mfl) > 5 and len(use_cols) > 0:
            X_lin = sm.add_constant(mfl.drop(columns=[lhs]).astype(float), has_constant="add")
            y_lin = mfl[lhs].astype(float)
            lin = sm.OLS(y_lin, X_lin).fit()
            coef_tbl = lin.summary2().tables[1]
            lin_csv = TABDIR / "investigationA_linear_coefficients.csv"
            coef_tbl.to_csv(lin_csv)
            print("[save] ", lin_csv)
        else:
            print("[info] Linear model skipped (insufficient rows/predictors).")

    return True

okA = False
try:
    okA = model_investigation_A(d1)
except Exception as e:
    print("[error] Investigation A failed:", e)

print("[step] Investigation B — Season interaction ...")

def model_investigation_B(d1):
    cols = ["risk","rat_pressure_idx","hours_after_sunset","rat_present_during_landing","season"]
    cols = [c for c in cols if c in d1.columns]
    if "risk" not in cols or "rat_pressure_idx" not in cols or "season" not in cols:
        print("[info] Missing required columns for B; skipping.")
        return False

    mb = d1[cols].copy()
    mb["risk"] = pd.to_numeric(mb["risk"], errors="coerce").astype(float)
    for c in ["rat_pressure_idx","hours_after_sunset","rat_present_during_landing"]:
        if c in mb.columns:
            mb[c] = pd.to_numeric(mb[c], errors="coerce")

    mb = mb[mb["risk"].isin([0.0,1.0])]
    mb = mb.dropna(subset=["risk","rat_pressure_idx"])
    if mb["risk"].nunique() < 2:
        print("[info] `risk` has one class after cleaning; skipping B.")
        return False

    if "season" in mb.columns:
        mb = pd.get_dummies(mb, columns=["season"], drop_first=True)

    for c in ["rat_pressure_idx","hours_after_sunset"]:
        if c in mb.columns:
            mb[c] = zscore(mb[c])

    season_cols = [c for c in mb.columns if c.startswith("season_")]
    for sc in season_cols:
        mb[f"rat_pressure_idx:{sc}"] = mb["rat_pressure_idx"] * mb[sc]

    pred_cols = ["rat_pressure_idx"] + season_cols + [f"rat_pressure_idx:{sc}" for sc in season_cols]
    if "hours_after_sunset" in mb.columns: pred_cols.append("hours_after_sunset")
    if "rat_present_during_landing" in mb.columns: pred_cols.append("rat_present_during_landing")

    mb = mb.dropna(subset=pred_cols)
    X = mb[pred_cols].astype(float)
    y = mb["risk"].astype(float)
    X = sm.add_constant(X, has_constant="add")

    print("[info] B design:", X.shape, "| predictors:", list(X.columns))

    try:
        fitted = sm.GLM(y, X, family=sm.families.Binomial()).fit()
    except Exception as e:
        print("[warn] Unregularized GLM (B) failed ->", type(e).__name__, str(e))
        fitted = None
        for alpha in [1.0, 5.0, 10.0]:
            try:
                print(f"[info] Trying regularized GLM (B) alpha={alpha} ...")
                fitted = sm.GLM(y, X, family=sm.families.Binomial()).fit_regularized(alpha=alpha, L1_wt=0.0, maxiter=500)
                break
            except Exception as e2:
                print("  [fail]", type(e2).__name__, str(e2))
        if fitted is None:
            print("[error] B model could not be fit (separation/collinearity).")
            return False

    try:
        coef_tbl = fitted.summary2().tables[1]
    except Exception:
        coef_tbl = pd.DataFrame({"coef": fitted.params})

    out_csv = TABDIR / "investigationB_logit_with_interaction_coefficients.csv"
    coef_tbl.to_csv(out_csv)
    print("[save] ", out_csv)

    try:
        season_cols = [c for c in mb.columns if c.startswith("season_")]
        if season_cols:
            rp = np.linspace(mb["rat_pressure_idx"].quantile(0.02), mb["rat_pressure_idx"].quantile(0.98), 60)
            plt.figure()
            for sc in season_cols:
                dfp = pd.DataFrame({"rat_pressure_idx": rp})
                for sc2 in season_cols:
                    dfp[sc2] = 1.0 if sc2 == sc else 0.0
                    dfp[f"rat_pressure_idx:{sc2}"] = dfp["rat_pressure_idx"] * dfp[sc2]
                if "hours_after_sunset" in mb.columns:
                    dfp["hours_after_sunset"] = float(mb["hours_after_sunset"].median())
                if "rat_present_during_landing" in mb.columns:
                    dfp["rat_present_during_landing"] = float(mb["rat_present_during_landing"].mode().dropna().iloc[0]) if mb["rat_present_during_landing"].notna().any() else 0.0

                Xg = sm.add_constant(dfp.astype(float), has_constant="add")
                preds = fitted.predict(Xg)
                plt.plot(rp, preds, label=sc.replace("season_",""))
            plt.title("Predicted P(risk=1) vs Rat Pressure by Season")
            plt.xlabel("Rat Pressure Index"); plt.ylabel("Predicted Probability")
            plt.legend(title="Season")
            out = Path("reports/figures/fig_predicted_risk_vs_pressure_by_season.png")
            plt.tight_layout(); plt.savefig(out, dpi=300); plt.show()
            print("[plot] Saved:", out)
    except Exception as e:
        print("[warn] Prediction curves failed:", e)

    return True

okB = False
try:
    okB = model_investigation_B(d1)
except Exception as e:
    print("[error] Investigation B failed:", e)

print("[done] Script finished. See outputs in:", FIGDIR, "and", TABDIR)
