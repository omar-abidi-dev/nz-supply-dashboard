"""Clean the raw Stats NZ import files into one tidy parquet file."""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
CLEAN_FILE = ROOT / "data" / "clean" / "imports.parquet"

# Some years use Stats NZ's long column names; map them to the short ones.
COLUMN_MAP = {
    "Month": "month",
    "Harmonised System Code": "hs",
    "Harmonised System Description": "hs_desc",
    "Unit Qty": "uom",
    "Country": "country",
    "Imports ($NZD vfd)": "vfd",
    "Imports ($NZD cif)": "cif",
    "Imports Qty": "qty",
    "Status": "status",
}
EXPECTED_COLUMNS = ["month", "hs", "hs_desc", "uom", "country", "vfd", "cif", "qty", "status"]

# HS sections: (first chapter, last chapter, readable name)
HS_SECTIONS = [
    (1, 5, "Animals & animal products"),
    (6, 14, "Vegetable products"),
    (15, 15, "Fats & oils"),
    (16, 24, "Food, beverages & tobacco"),
    (25, 27, "Minerals & fuels"),
    (28, 38, "Chemicals & pharmaceuticals"),
    (39, 40, "Plastics & rubber"),
    (41, 43, "Leather & hides"),
    (44, 46, "Wood products"),
    (47, 49, "Paper & printing"),
    (50, 63, "Textiles & clothing"),
    (64, 67, "Footwear & headgear"),
    (68, 70, "Stone, ceramics & glass"),
    (71, 71, "Precious metals & stones"),
    (72, 83, "Base metals"),
    (84, 85, "Machinery & electronics"),
    (86, 89, "Vehicles, aircraft & ships"),
    (90, 92, "Instruments & medical devices"),
    (93, 93, "Arms & ammunition"),
    (94, 96, "Furniture & other manufactured"),
    (97, 97, "Art & antiques"),
]


def chapter_to_section(chapter: int) -> str:
    for first, last, name in HS_SECTIONS:
        if first <= chapter <= last:
            return name
    return "Other / special"


def read_one(f: Path) -> pd.DataFrame:
    """Read one yearly file and return it with the standard column names."""
    df = pd.read_csv(f, dtype=str)  # read as text first, convert later
    df = df.rename(columns=COLUMN_MAP)
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")]  # drop empty extra columns

    missing = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"{f.name} is missing columns {missing}. Found: {df.columns.tolist()}. "
            "If it was opened/re-saved in Excel or Numbers, download the .csv.xz again."
        )

    # Remove thousands separators, e.g. "6,235" -> "6235"
    for col in ["vfd", "cif", "qty"]:
        df[col] = df[col].str.replace(",", "", regex=False)
    return df[EXPECTED_COLUMNS]


def load_raw() -> pd.DataFrame:
    files = sorted(RAW_DIR.glob("*Imports_HS10_by_Country.csv*"))
    if not files:
        raise FileNotFoundError(f"No import files found in {RAW_DIR}")
    frames = []
    for f in files:
        print(f"Reading {f.name} ...")
        frames.append(read_one(f))
    return pd.concat(frames, ignore_index=True)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns={"vfd": "value_vfd_nzd", "cif": "value_cif_nzd"})

    df["hs"] = df["hs"].str.strip().str.zfill(10)
    df["month"] = pd.to_datetime(df["month"].str.strip(), format="%Y%m", errors="coerce")
    for col in ["value_vfd_nzd", "value_cif_nzd", "qty"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["country"] = df["country"].str.strip()

    df = df.dropna(subset=["month", "hs", "country", "value_cif_nzd"])

    df["chapter"] = df["hs"].str[:2].astype(int)
    df["section"] = df["chapter"].map(chapter_to_section)

    for col in ["country", "section", "uom", "status"]:
        df[col] = df[col].astype("category")
    return df


def main() -> None:
    df = clean(load_raw())
    CLEAN_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(CLEAN_FILE, index=False)

    print(f"\nSaved {len(df):,} rows to {CLEAN_FILE}")
    print(f"Months: {df['month'].min():%Y-%m} to {df['month'].max():%Y-%m}")
    print(f"Countries: {df['country'].nunique()}")
    print("Rows per year:")
    print(df["month"].dt.year.value_counts().sort_index().to_string())
    print(f"Total imports (CIF): NZD {df['value_cif_nzd'].sum() / 1e9:,.1f} billion")


if __name__ == "__main__":
    main()