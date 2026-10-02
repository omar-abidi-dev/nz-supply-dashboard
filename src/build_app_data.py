"""Build a small, memory-friendly dataset for the online dashboard.

Aggregates the full cleaned data (one row per HS10 product x country x month)
to one row per HS4 product group x country x month, and drops long text columns.
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FULL_FILE = ROOT / "data" / "clean" / "imports.parquet"
APP_FILE = ROOT / "data" / "clean" / "app_data.parquet"


def main() -> None:
    df = pd.read_parquet(FULL_FILE, columns=["month", "hs", "hs_desc", "country",
                                             "section", "value_cif_nzd"])
    df["hs4"] = df["hs"].str[:4]

    # One short label per HS4 group, e.g. "Horses; live, ..." -> "Horses"
    labels = (
        df["hs_desc"].str.split(";").str[0].str.strip()
        .groupby(df["hs4"]).agg(lambda s: s.mode().iat[0])
        .rename("product")
    )

    app = (
        df.groupby(["month", "hs4", "country", "section"], observed=True)["value_cif_nzd"]
        .sum().reset_index()
    )
    app = app[app["value_cif_nzd"] != 0]
    app = app.join(labels, on="hs4")

    for col in ["hs4", "country", "section", "product"]:
        app[col] = app[col].astype("category")
    app["value_cif_nzd"] = app["value_cif_nzd"].astype("float64")

    app.to_parquet(APP_FILE, index=False)
    mb = APP_FILE.stat().st_size / 1e6
    mem = app.memory_usage(deep=True).sum() / 1e6
    print(f"Saved {len(app):,} rows to {APP_FILE.name} ({mb:.1f} MB on disk, {mem:.0f} MB in memory)")
    print(f"Total imports check: NZD {app['value_cif_nzd'].sum() / 1e9:,.1f} billion")


if __name__ == "__main__":
    main()