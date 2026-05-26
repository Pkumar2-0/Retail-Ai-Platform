import os
from functools import lru_cache

import pandas as pd

DATA_PATHS = [
    "ml/datasets/staged/cleaned_walmart.csv",
    "ml/datasets/curated/featured_walmart.csv",
    "ml/datasets/raw/Walmart.csv",
]


def find_dataset_path():
    for path in DATA_PATHS:
        if os.path.exists(path):
            return path
    return None


@lru_cache()
def load_sales_data():
    path = find_dataset_path()
    if path is None:
        return None

    preview = pd.read_csv(path, nrows=0)
    parse_dates = [col for col in ["Date"] if col in preview.columns]
    df = pd.read_csv(path, parse_dates=parse_dates)

    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    return df


def get_analytics_context():
    df = load_sales_data()
    if df is None:
        return "No analytics dataset is available for this agent."

    total_sales = float(df["Weekly_Sales"].sum())
    avg_sales = float(df["Weekly_Sales"].mean())
    row_count = int(df.shape[0])

    context_lines = [
        "Retail analytics summary from Walmart weekly sales data:",
        f"- Total records: {row_count}",
        f"- Total reported weekly sales: ${total_sales:,.0f}",
        f"- Average weekly sales: ${avg_sales:,.0f}",
    ]

    if "Date" in df.columns:
        start_date = df["Date"].min()
        end_date = df["Date"].max()
        context_lines.append(
            f"- Date range: {start_date.date()} to {end_date.date()}"
        )

    top_store = (
        df.groupby("Store")["Weekly_Sales"].mean().idxmax()
    )
    top_store_sales = float(
        df.groupby("Store")["Weekly_Sales"].mean().max()
    )
    context_lines.append(
        f"- Top store by average weekly sales: Store {top_store} at ${top_store_sales:,.0f} average sales"
    )

    if "Holiday_Flag" in df.columns:
        holiday_sales = float(
            df.loc[df["Holiday_Flag"] == 1, "Weekly_Sales"].sum()
        )
        holiday_ratio = holiday_sales / total_sales if total_sales else 0
        context_lines.append(
            f"- Holiday sales share: {holiday_ratio:.1%} of total sales"
        )

    if "Date" in df.columns:
        monthly_avg = (
            df.assign(Month=df["Date"].dt.month_name())
            .groupby("Month")["Weekly_Sales"]
            .mean()
        )
        best_month = monthly_avg.idxmax()
        best_month_value = float(monthly_avg.max())
        context_lines.append(
            f"- Month with the highest average weekly sales: {best_month} (${best_month_value:,.0f})"
        )

    for feature in ["Temperature", "Fuel_Price", "CPI", "Unemployment"]:
        if feature in df.columns:
            avg_value = float(df[feature].mean())
            context_lines.append(f"- Average {feature}: {avg_value:.2f}")

    return "\n".join(context_lines)


def get_direct_analytics_answer(query):
    df = load_sales_data()
    if df is None:
        return None

    query_lower = query.lower()
    if "top store" in query_lower or "best store" in query_lower or "highest average" in query_lower:
        top_store = df.groupby("Store")["Weekly_Sales"].mean().idxmax()
        top_store_sales = float(df.groupby("Store")["Weekly_Sales"].mean().max())
        return (
            f"Store {top_store} has the highest average weekly sales at ${top_store_sales:,.0f}."
        )

    if "average weekly sales" in query_lower or "average sales" in query_lower:
        avg_sales = float(df["Weekly_Sales"].mean())
        return f"The dataset average weekly sales are approximately ${avg_sales:,.0f}."

    if "holiday" in query_lower:
        holiday_sales = float(df.loc[df["Holiday_Flag"] == 1, "Weekly_Sales"].sum())
        total_sales = float(df["Weekly_Sales"].sum())
        holiday_ratio = holiday_sales / total_sales if total_sales else 0
        return (
            f"Holiday sales represent {holiday_ratio:.1%} of total sales, indicating a meaningful holiday uplift in the dataset."
        )

    if "total sales" in query_lower:
        total_sales = float(df["Weekly_Sales"].sum())
        return f"The dataset contains ${total_sales:,.0f} in total reported weekly sales."

    if "date range" in query_lower or "period" in query_lower or "time frame" in query_lower:
        if "Date" in df.columns:
            start_date = df["Date"].min().date()
            end_date = df["Date"].max().date()
            return f"The dataset covers sales from {start_date} through {end_date}."

    if ("peak" in query_lower or "highest" in query_lower) and "month" in query_lower:
        monthly_avg = (
            df.assign(Month=df["Date"].dt.month_name())
            .groupby("Month")["Weekly_Sales"]
            .mean()
        )
        best_month = monthly_avg.idxmax()
        best_month_value = float(monthly_avg.max())
        return (
            f"The highest average weekly sales occur in {best_month}, at approximately ${best_month_value:,.0f} per week."
        )

    return None


def get_ml_model_context():
    model_context = [
        "Retail ML platform context:",
        "- The demand forecasting model uses features: Store, Holiday_Flag, Temperature, Fuel_Price, CPI, Unemployment, Year, Month, Day, and Week.",
        "- The anomaly detection model uses weekly sales data together with economic indicators to flag abnormal sales patterns.",
        "- Forecasting is designed to support weekly sales planning and stocking decisions.",
        "- Anomaly detection helps identify abnormal events, such as unexpected dips or spikes in demand when economic indicators or holidays deviate from normal patterns.",
    ]

    analytics_context = get_analytics_context()
    return "\n".join(model_context) + "\n" + analytics_context


def get_direct_ml_answer(query):
    query_lower = query.lower()
    if any(keyword in query_lower for keyword in ["anomaly", "outlier", "abnormal"]):
        return (
            "The anomaly detection model is trained to detect unusual weekly sales patterns. "
            "It evaluates Weekly_Sales alongside Temperature, Fuel_Price, CPI, and Unemployment to flag abnormal sales behavior. "
            "If the model finds an outlier, it returns 'Anomaly Detected'; otherwise it returns 'Normal Sales Pattern'."
        )

    if any(keyword in query_lower for keyword in ["forecast", "predict", "prediction", "demand"]):
        return (
            "The demand forecasting model predicts future weekly sales using store data, holiday flags, weather, fuel cost, consumer price index, "
            "unemployment, and calendar features. It is intended to support inventory planning and weekly sales projections."
        )

    if any(keyword in query_lower for keyword in ["feature", "features", "input", "model", "training", "evaluation", "accuracy"]):
        return (
            "The platform uses two ML components: a demand forecast model and an anomaly detector. "
            "The forecast model estimates weekly sales volume from economic and calendar inputs, while the anomaly model identifies unusual sales weeks."
        )

    return None
