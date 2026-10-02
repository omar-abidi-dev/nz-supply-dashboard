"""Metrics for the NZ supply chain dashboard."""
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing

ROOT = Path(__file__).resolve().parents[1]
CLEAN_FILE = ROOT / "data" / "clean" / "imports.parquet"
APP_FILE = ROOT / "data" / "clean" / "app_data.parquet"

VALUE = "value_cif_nzd"


def load_data(path: Path | None = None) -> pd.DataFrame:
    """Load the small app dataset if it exists, otherwise the full cleaned data."""
    if path is None:
        path = APP_FILE if APP_FILE.exists() else CLEAN_FILE
    df = pd.read_parquet(path)
    if "hs4" not in df.columns:  # full dataset: derive the extra columns
        df["hs4"] = df["hs"].str[:4]
        # Short product label, e.g. "Horses; live, pure-bred..." -> "Horses"
        df["product"] = df["hs_desc"].str.split(";").str[0].str.strip()
    return df


def filter_data(df, sections=None, start=None, end=None):
    """Keep only the chosen sections and date range (None = no filter)."""
    out = df
    if sections:
        out = out[out["section"].isin(sections)]
    if start is not None:
        out = out[out["month"] >= pd.Timestamp(start)]
    if end is not None:
        out = out[out["month"] <= pd.Timestamp(end)]
    return out


# ---------- Basic totals ----------

def monthly_totals(df) -> pd.Series:
    """Total import value per month (NZD)."""
    return df.groupby("month")[VALUE].sum().sort_index()


def top_countries(df, n: int = 10) -> pd.DataFrame:
    out = (
        df.groupby("country", observed=True)[VALUE].sum()
        .sort_values(ascending=False).head(n).reset_index()
    )
    out["share"] = out[VALUE] / df[VALUE].sum()
    return out


def section_totals(df) -> pd.DataFrame:
    return (
        df.groupby("section", observed=True)[VALUE].sum()
        .sort_values(ascending=False).reset_index()
    )


def yoy_growth(monthly: pd.Series) -> float | None:
    """Last 12 months vs the 12 months before (e.g. 0.05 = +5%)."""
    if len(monthly) < 24:
        return None
    last, prev = monthly.iloc[-12:].sum(), monthly.iloc[-24:-12].sum()
    return (last - prev) / prev if prev else None


# ---------- Supplier concentration risk ----------

def concentration(df, level: str = "hs4", min_value: float = 50e6) -> pd.DataFrame:
    """
    Supplier concentration per product group, using the HHI
    (sum of squared supplier shares, 0 = spread out, 1 = one supplier).
    Only groups with total imports >= min_value NZD are kept.
    """

    
    by = df.groupby([level, "country"], observed=True)[VALUE].sum().reset_index()
    by = by[by[VALUE] > 0]

    by[level] = by[level].astype(str)
    by["country"] = by["country"].astype(str)
    
    total = by.groupby(level)[VALUE].transform("sum")
    by["share"] = by[VALUE] / total

    top = by.sort_values("share", ascending=False).drop_duplicates(level)
    out = by.groupby(level).agg(
        total_nzd=(VALUE, "sum"),
        hhi=("share", lambda s: (s ** 2).sum()),
        n_countries=("country", "nunique"),
    )
    out = out.join(top.set_index(level)[["country", "share"]]
                   .rename(columns={"country": "top_country", "share": "top_share"}))

    if level == "hs4":
        info = (df[["hs4", "product", "section"]].astype(str)
        .drop_duplicates("hs4").set_index("hs4"))
        out = out.join(info)

    out["risk"] = np.select(
        [(out["hhi"] > 0.25) | (out["top_share"] > 0.6), out["hhi"] > 0.15],
        ["High", "Medium"], default="Low",
    )
    out = out[out["total_nzd"] >= min_value]
    return out.sort_values(["hhi", "total_nzd"], ascending=False).reset_index()


# ---------- Forecast ----------

def forecast(monthly: pd.Series, horizon: int = 6) -> pd.DataFrame:
    """Holt-Winters forecast with an approximate 95% interval."""
    y = monthly.asfreq("MS").fillna(0)
    seasonal = "add" if len(y) >= 24 else None
    model = ExponentialSmoothing(
        y, trend="add", seasonal=seasonal,
        seasonal_periods=12 if seasonal else None,
    ).fit()
    pred = model.forecast(horizon)
    resid_std = (y - model.fittedvalues).std()
    return pd.DataFrame({
        "forecast": pred,
        "lower": (pred - 1.96 * resid_std).clip(lower=0),
        "upper": pred + 1.96 * resid_std,
    })