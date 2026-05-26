import os
from pathlib import Path

import altair as alt
import pandas as pd
import requests
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_PATH = PROJECT_ROOT / "ml" / "datasets" / "curated" / "featured_walmart.csv"
PARQUET_PATH = PROJECT_ROOT / "data_pipeline" / "storage" / "curated" / "walmart_curated.parquet"


st.set_page_config(page_title="Retail AI Platform", layout="wide")

st.markdown(
    """
    <style>
    [data-testid="stSidebar"] {
        background: #0f172a;
        border-right: 1px solid #1e293b;
    }

    [data-testid="stSidebar"] * {
        color: #f8fafc;
    }

    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        color: #ffffff;
    }

    [data-testid="stSidebar"] .stTextInput label,
    [data-testid="stSidebar"] .stRadio label {
        color: #cbd5e1;
        font-weight: 600;
    }

    [data-testid="stSidebar"] input {
        background: #111827;
        border: 1px solid #334155;
        color: #f8fafc;
        border-radius: 8px;
    }

    [data-testid="stSidebar"] [role="radiogroup"] label {
        background: #111827;
        border: 1px solid #243244;
        border-radius: 8px;
        padding: 8px 10px;
        margin: 5px 0;
    }

    [data-testid="stSidebar"] [role="radiogroup"] label:hover {
        background: #1e293b;
        border-color: #3b82f6;
    }

    .sidebar-brand {
        padding: 12px 4px 16px 4px;
        border-bottom: 1px solid #243244;
        margin-bottom: 14px;
    }

    .sidebar-title {
        font-size: 1.28rem;
        font-weight: 800;
        color: #ffffff;
        line-height: 1.15;
    }

    .sidebar-subtitle {
        color: #94a3b8;
        font-size: 0.86rem;
        margin-top: 6px;
        line-height: 1.35;
    }

    .sidebar-chip {
        display: inline-block;
        background: #0e7490;
        color: #ecfeff;
        border-radius: 999px;
        padding: 4px 9px;
        font-size: 0.75rem;
        font-weight: 700;
        margin-top: 10px;
    }

    .sidebar-help {
        background: #111827;
        border: 1px solid #243244;
        border-radius: 8px;
        padding: 10px 11px;
        color: #cbd5e1;
        font-size: 0.82rem;
        line-height: 1.4;
        margin-top: 18px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def load_sales_data():
    path = PARQUET_PATH if PARQUET_PATH.exists() else DATA_PATH
    df = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    return df.dropna(subset=["Date"])


def backend_url():
    return st.session_state.get("backend_url", "http://127.0.0.1:8000").rstrip("/")


def api_get(path, params=None, timeout=60):
    try:
        response = requests.get(f"{backend_url()}{path}", params=params, timeout=timeout)
        response.raise_for_status()
        return response.json(), None
    except requests.exceptions.RequestException as exc:
        return None, str(exc)


def api_post(path, payload, timeout=60):
    try:
        response = requests.post(f"{backend_url()}{path}", json=payload, timeout=timeout)
        response.raise_for_status()
        return response.json(), None
    except requests.exceptions.RequestException as exc:
        return None, str(exc)


def show_json(title, data):
    st.subheader(title)
    if data is None:
        st.warning("No response received.")
    else:
        st.json(data, expanded=True)


def show_error(error):
    if error:
        st.error(error)


def money(value):
    return f"${float(value):,.0f}"


def render_sidebar():
    st.sidebar.markdown(
        """
        <div class="sidebar-brand">
            <div class="sidebar-title">Retail AI<br>Platform</div>
            <div class="sidebar-subtitle">Sales intelligence, RAG search, Azure agents, and ML tools.</div>
            <div class="sidebar-chip">Azure enabled</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.sidebar.text_input(
        "Backend URL",
        value=os.environ.get("RETAIL_AI_BACKEND_URL", "http://127.0.0.1:8000"),
        key="backend_url",
    )
    st.sidebar.markdown("### Navigation")
    return st.sidebar.radio(
        "Page",
        [
            "Dashboard",
            "Agent Chat",
            "Document Search",
            "Forecast",
            "Anomaly Detection",
        ],
        label_visibility="collapsed",
    )


def render_dashboard(df):
    st.header("Dashboard")
    st.write("Simple sales overview from the curated dataset.")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Sales", money(df["Weekly_Sales"].sum()))
    col2.metric("Average Weekly Sales", money(df["Weekly_Sales"].mean()))
    col3.metric("Stores", int(df["Store"].nunique()))
    col4.metric("Rows", f"{len(df):,}")

    monthly = (
        df.assign(Month=df["Date"].dt.to_period("M").dt.to_timestamp())
        .groupby("Month", as_index=False)["Weekly_Sales"]
        .sum()
    )
    chart = (
        alt.Chart(monthly)
        .mark_line(point=True)
        .encode(
            x=alt.X("Month:T", title="Month"),
            y=alt.Y("Weekly_Sales:Q", title="Total Sales"),
            tooltip=["Month:T", alt.Tooltip("Weekly_Sales:Q", format=",.0f")],
        )
        .properties(height=320)
    )
    st.altair_chart(chart, use_container_width=True)

def render_agent_chat():
    st.header("Agent Chat")
    st.write("Ask the multi-agent system. The final answer and full JSON response are shown below.")

    query = st.text_area(
        "Question",
        value="What inventory policy documents support the forecast model recommendation?",
        height=100,
    )

    if st.button("Ask Agent", type="primary"):
        with st.spinner("Running agents..."):
            result, error = api_get("/agent-chat", params={"query": query}, timeout=120)
        show_error(error)
        if result:
            st.subheader("Final Answer")
            st.info(result.get("response", "No final answer returned."))

            st.subheader("Selected Agents")
            st.write(result.get("selected_agents", []))

            st.subheader("Agent Messages")
            for idx, message in enumerate(result.get("messages", []), start=1):
                with st.container(border=True):
                    st.markdown(f"**Message {idx}: {message.get('sender')} → {message.get('receiver')}**")
                    st.write(f"Intent: {message.get('intent')}")
                    st.write(message.get("content", ""))
                    if message.get("metadata"):
                        st.json(message["metadata"], expanded=True)

            show_json("Full API Response", result)


def render_document_search():
    st.header("Document Search")
    st.write("Search the RAG knowledge base. The backend field should say Azure AI Search when configured.")

    query = st.text_input("Search query", value="inventory policy forecast")

    if st.button("Search Documents", type="primary"):
        with st.spinner("Searching documents..."):
            result, error = api_get("/search-documents", params={"query": query}, timeout=120)
        show_error(error)
        if result:
            results = result.get("results", {})
            st.subheader("Search Backend")
            st.success(results.get("backend", "Unknown backend"))

            if results.get("fallback_reason"):
                st.warning(results["fallback_reason"])

            st.subheader("Answer")
            st.info(results.get("answer", "No answer returned."))

            st.subheader("Retrieved Documents")
            documents = results.get("documents", [])
            sources = results.get("sources", [])
            for idx, doc in enumerate(documents, start=1):
                source = sources[idx - 1] if idx - 1 < len(sources) else "Unknown"
                with st.container(border=True):
                    st.markdown(f"**Document Chunk {idx}**")
                    st.write(f"Source: {source}")
                    st.write(doc)

            show_json("Full API Response", result)


def render_forecast():
    st.header("Forecast")

    with st.form("forecast_form"):
        col1, col2 = st.columns(2)
        with col1:
            store = st.number_input("Store", min_value=1, value=1, step=1)
            holiday_flag = st.selectbox("Holiday Flag", [0, 1])
            temperature = st.number_input("Temperature", value=70.0)
            fuel_price = st.number_input("Fuel Price", value=3.5)
            cpi = st.number_input("CPI", value=211.0)
        with col2:
            unemployment = st.number_input("Unemployment", value=7.5)
            year = st.number_input("Year", min_value=2010, value=2026, step=1)
            month = st.number_input("Month", min_value=1, max_value=12, value=5, step=1)
            day = st.number_input("Day", min_value=1, max_value=31, value=26, step=1)
            week = st.number_input("Week", min_value=1, max_value=53, value=22, step=1)

        submitted = st.form_submit_button("Predict Sales", type="primary")

    if submitted:
        payload = {
            "Store": int(store),
            "Holiday_Flag": int(holiday_flag),
            "Temperature": float(temperature),
            "Fuel_Price": float(fuel_price),
            "CPI": float(cpi),
            "Unemployment": float(unemployment),
            "Year": int(year),
            "Month": int(month),
            "Day": int(day),
            "Week": int(week),
        }
        result, error = api_post("/predict-sales", payload)
        show_error(error)
        if result:
            st.subheader("Prediction")
            st.success(money(result.get("predicted_weekly_sales", 0)))
            show_json("Full API Response", result)


def render_anomaly():
    st.header("Anomaly Detection")

    with st.form("anomaly_form"):
        weekly_sales = st.number_input("Weekly Sales", min_value=0.0, value=1_000_000.0)
        temperature = st.number_input("Temperature", value=65.0)
        fuel_price = st.number_input("Fuel Price", value=3.35)
        cpi = st.number_input("CPI", value=211.0)
        unemployment = st.number_input("Unemployment", value=7.5)
        submitted = st.form_submit_button("Detect Anomaly", type="primary")

    if submitted:
        payload = {
            "Weekly_Sales": float(weekly_sales),
            "Temperature": float(temperature),
            "Fuel_Price": float(fuel_price),
            "CPI": float(cpi),
            "Unemployment": float(unemployment),
        }
        result, error = api_post("/detect-anomaly", payload)
        show_error(error)
        if result:
            st.subheader("Result")
            st.success(result.get("result", "No result returned."))
            show_json("Full API Response", result)


def render_pipeline():
    st.header("Data Pipeline")
    st.write("Raw → Staged → Curated outputs.")

    files = [
        "data_pipeline/storage/raw/walmart_raw.parquet",
        "data_pipeline/storage/staged/walmart_staged.parquet",
        "data_pipeline/storage/curated/walmart_curated.parquet",
        "data_pipeline/storage/curated/store_sales_summary.parquet",
        "data_pipeline/storage/curated/monthly_sales_summary.parquet",
        "deployment/adf_retail_pipeline_template.json",
        "data_pipeline/transformation/retail_pyspark_pipeline.py",
    ]

    rows = []
    for file_name in files:
        path = PROJECT_ROOT / file_name
        rows.append(
            {
                "file": file_name,
                "exists": path.exists(),
                "size_kb": round(path.stat().st_size / 1024, 1) if path.exists() else 0,
            }
        )
    st.dataframe(pd.DataFrame(rows), use_container_width=True)


def main():
    page = render_sidebar()
    st.title("Retail AI Platform")
    st.caption("Simple Streamlit UI for agents, RAG, ML, and data pipeline")

    status, error = api_get("/", timeout=5)
    if status:
        st.success("Backend connected")
    else:
        st.warning("Backend not connected. Start it with: uvicorn backend.main:app --reload")
        show_error(error)

    df = load_sales_data()

    if page == "Dashboard":
        render_dashboard(df)
    elif page == "Agent Chat":
        render_agent_chat()
    elif page == "Document Search":
        render_document_search()
    elif page == "Forecast":
        render_forecast()
    elif page == "Anomaly Detection":
        render_anomaly()

if __name__ == "__main__":
    main()
