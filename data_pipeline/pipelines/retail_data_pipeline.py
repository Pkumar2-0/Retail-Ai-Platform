import sqlite3
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ML_DATASETS = PROJECT_ROOT / "ml" / "datasets"
PIPELINE_STORAGE = PROJECT_ROOT / "data_pipeline" / "storage"

RAW_SOURCE = ML_DATASETS / "raw" / "Walmart.csv"
RAW_STORAGE = PIPELINE_STORAGE / "raw"
STAGED_STORAGE = PIPELINE_STORAGE / "staged"
CURATED_STORAGE = PIPELINE_STORAGE / "curated"


def _ensure_storage():
    for path in [RAW_STORAGE, STAGED_STORAGE, CURATED_STORAGE]:
        path.mkdir(parents=True, exist_ok=True)


def _write_dataset(df, output_dir, name):
    csv_path = output_dir / f"{name}.csv"
    parquet_path = output_dir / f"{name}.parquet"

    df.to_csv(csv_path, index=False)
    df.to_parquet(parquet_path, index=False)

    return {
        "csv": str(csv_path),
        "parquet": str(parquet_path),
        "rows": int(df.shape[0]),
        "columns": list(df.columns),
    }


def ingest_raw_data():
    raw_df = pd.read_csv(RAW_SOURCE)
    return raw_df, _write_dataset(raw_df, RAW_STORAGE, "walmart_raw")


def create_staged_data(raw_df):
    staged_df = raw_df.drop_duplicates().copy()
    staged_df["Date"] = pd.to_datetime(staged_df["Date"], format="%d-%m-%Y", errors="coerce")
    staged_df = staged_df.dropna(subset=["Date", "Weekly_Sales"])
    staged_df["Date"] = staged_df["Date"].dt.strftime("%Y-%m-%d")

    ml_staged_path = ML_DATASETS / "staged" / "cleaned_walmart.csv"
    staged_df.to_csv(ml_staged_path, index=False)

    return staged_df, _write_dataset(staged_df, STAGED_STORAGE, "walmart_staged")


def create_curated_data(staged_df):
    curated_df = staged_df.copy()
    curated_df["Date"] = pd.to_datetime(curated_df["Date"], errors="coerce")
    curated_df["Year"] = curated_df["Date"].dt.year
    curated_df["Month"] = curated_df["Date"].dt.month
    curated_df["Day"] = curated_df["Date"].dt.day
    curated_df["Week"] = curated_df["Date"].dt.isocalendar().week.astype(int)
    curated_df["Date"] = curated_df["Date"].dt.strftime("%Y-%m-%d")

    ml_curated_path = ML_DATASETS / "curated" / "featured_walmart.csv"
    curated_df.to_csv(ml_curated_path, index=False)

    return curated_df, _write_dataset(curated_df, CURATED_STORAGE, "walmart_curated")


def create_sql_analytics(curated_df):
    with sqlite3.connect(":memory:") as conn:
        curated_df.to_sql("retail_sales", conn, index=False, if_exists="replace")

        store_sales = pd.read_sql_query(
            """
            SELECT
                Store,
                COUNT(*) AS record_count,
                SUM(Weekly_Sales) AS total_sales,
                AVG(Weekly_Sales) AS avg_weekly_sales
            FROM retail_sales
            GROUP BY Store
            ORDER BY total_sales DESC
            """,
            conn,
        )

        monthly_sales = pd.read_sql_query(
            """
            SELECT
                Year,
                Month,
                SUM(Weekly_Sales) AS total_sales,
                AVG(Weekly_Sales) AS avg_weekly_sales,
                AVG(Fuel_Price) AS avg_fuel_price,
                AVG(CPI) AS avg_cpi,
                AVG(Unemployment) AS avg_unemployment
            FROM retail_sales
            GROUP BY Year, Month
            ORDER BY Year, Month
            """,
            conn,
        )

    return {
        "store_sales": _write_dataset(store_sales, CURATED_STORAGE, "store_sales_summary"),
        "monthly_sales": _write_dataset(monthly_sales, CURATED_STORAGE, "monthly_sales_summary"),
    }


def run_pipeline():
    _ensure_storage()

    raw_df, raw_output = ingest_raw_data()
    staged_df, staged_output = create_staged_data(raw_df)
    curated_df, curated_output = create_curated_data(staged_df)
    analytics_outputs = create_sql_analytics(curated_df)

    manifest = {
        "flow": "raw -> staged -> curated",
        "storage_format": "csv and parquet",
        "raw": raw_output,
        "staged": staged_output,
        "curated": curated_output,
        "sql_analytics": analytics_outputs,
    }

    print("Retail data engineering pipeline completed.")
    print(f"Raw rows: {raw_output['rows']}")
    print(f"Staged rows: {staged_output['rows']}")
    print(f"Curated rows: {curated_output['rows']}")
    print("Parquet outputs written under data_pipeline/storage.")

    return manifest


if __name__ == "__main__":
    run_pipeline()
