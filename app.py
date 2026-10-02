"""NZ Trade & Supply Chain Risk Dashboard (Streamlit)."""
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent / "src"))
import metrics  # noqa: E402

st.set_page_config(page_title="NZ Supply Chain Risk", page_icon="🚢", layout="wide")


@st.cache_resource
def get_data() -> pd.DataFrame:
    return metrics.load_data()


def nzd(x: float) -> str:
    """Format NZD: 1.2B / 350.0M."""
    return f"NZD {x / 1e9:,.1f}B" if abs(x) >= 1e9 else f"NZD {x / 1e6:,.1f}M"


df_all = get_data()

# ---------------- Sidebar filters ----------------
st.sidebar.header("Filters")
all_sections = sorted(df_all["section"].dropna().unique().tolist())
sections = st.sidebar.multiselect("Product categories (empty = all)", all_sections)

months = sorted(df_all["month"].unique())
start, end = st.sidebar.select_slider(
    "Period",
    options=months,
    value=(months[0], months[-1]),
    format_func=lambda m: pd.Timestamp(m).strftime("%b %Y"),
)
min_value_m = st.sidebar.number_input(
    "Risk table: minimum product imports (NZD million)", 1, 5000, 50, step=10
)
horizon = st.sidebar.slider("Forecast horizon (months)", 3, 12, 6)
st.sidebar.caption("Data: Stats NZ Overseas Merchandise Trade (CIF values, NZD).")

df = metrics.filter_data(df_all, sections=sections, start=start, end=end)
if df.empty:
    st.warning("No data for these filters.")
    st.stop()

monthly = metrics.monthly_totals(df)
risk = metrics.concentration(df, min_value=min_value_m * 1e6)

# ---------------- Header + KPIs ----------------
st.title("🇳🇿 NZ Imports & Supply Chain Risk")
st.caption(
    f"{pd.Timestamp(start):%b %Y} – {pd.Timestamp(end):%b %Y} · "
    f"{'All categories' if not sections else ', '.join(sections)}"
)

growth = metrics.yoy_growth(monthly)
k1, k2, k3, k4 = st.columns(4)
k1.metric("Total imports", nzd(df[metrics.VALUE].sum()))
k2.metric("Last 12 months vs previous", f"{growth:+.1%}" if growth is not None else "n/a")
k3.metric("Supplier countries", f"{df['country'].nunique()}")
k4.metric("High-risk product groups", f"{(risk['risk'] == 'High').sum()}")

tab_overview, tab_risk, tab_forecast = st.tabs(["📈 Overview", "⚠️ Supply risk", "🔮 Forecast"])

# ---------------- Overview ----------------
with tab_overview:
    fig = px.line(monthly.reset_index(), x="month", y=metrics.VALUE,
                  labels={"month": "", metrics.VALUE: "Imports (NZD)"},
                  title="Monthly imports")
    st.plotly_chart(fig, width="stretch")

    c1, c2 = st.columns(2)
    top = metrics.top_countries(df, 10)
    fig = px.bar(top.sort_values(metrics.VALUE), x=metrics.VALUE, y="country",
                 orientation="h", title="Top 10 supplier countries",
                 labels={metrics.VALUE: "Imports (NZD)", "country": ""},
                 hover_data={"share": ":.1%"})
    c1.plotly_chart(fig, width="stretch")

    sec = metrics.section_totals(df).head(10)
    fig = px.bar(sec.sort_values(metrics.VALUE), x=metrics.VALUE, y="section",
                 orientation="h", title="Top product categories",
                 labels={metrics.VALUE: "Imports (NZD)", "section": ""})
    c2.plotly_chart(fig, width="stretch")

# ---------------- Supply risk ----------------
with tab_risk:
    st.markdown(
        "**How to read this:** for each product group (4-digit HS code) we measure how "
        "concentrated its suppliers are. **HHI** = sum of squared supplier shares "
        "(1 = one single supplier). **High risk** = HHI > 0.25 or one country > 60%."
    )
    if risk.empty:
        st.info("No product groups above the minimum value. Lower it in the sidebar.")
    else:
        fig = px.scatter(
            risk, x="top_share", y="total_nzd", color="risk", log_y=True,
            hover_name="product", hover_data={"top_country": True, "hhi": ":.2f"},
            color_discrete_map={"High": "#d62728", "Medium": "#ff7f0e", "Low": "#2ca02c"},
            labels={"top_share": "Share of the largest supplier",
                    "total_nzd": "Total imports (NZD, log scale)"},
            title="Dependence on one supplier vs. import value",
        )
        fig.update_xaxes(tickformat=".0%")
        st.plotly_chart(fig, width="stretch")

        level = st.radio("Show", ["High", "Medium", "Low", "All"], horizontal=True)
        table = risk if level == "All" else risk[risk["risk"] == level]
        st.dataframe(
            table[["hs4", "product", "section", "total_nzd", "top_country",
                   "top_share", "hhi", "n_countries", "risk"]],
            hide_index=True, width="stretch",
            column_config={
                "hs4": "HS4", "product": "Product", "section": "Category",
                "total_nzd": st.column_config.NumberColumn("Imports (NZD)", format="%.0f"),
                "top_country": "Main supplier",
                "top_share": st.column_config.ProgressColumn(
                    "Main supplier share", format="percent", min_value=0, max_value=1),
                "hhi": st.column_config.NumberColumn("HHI", format="%.2f"),
                "n_countries": "Countries", "risk": "Risk",
            },
        )

        st.subheader("Drill down into one product group")
        options = risk["hs4"].astype(str) + " – " + risk["product"].astype(str)
        choice = st.selectbox("Product group", options)
        code = choice.split(" – ")[0]
        sub = df[df["hs4"].astype(str) == code]
        by_country = metrics.top_countries(sub, 10)
        fig = px.pie(by_country, names="country", values=metrics.VALUE,
                     title=f"Suppliers of {choice}", hole=0.4)
        st.plotly_chart(fig, width="stretch")

# ---------------- Forecast ----------------
with tab_forecast:
    if len(monthly) < 12:
        st.info("Select at least 12 months to build a forecast.")
    else:
        fc = metrics.forecast(monthly, horizon)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=monthly.index, y=monthly.values, name="Actual"))
        fig.add_trace(go.Scatter(x=fc.index, y=fc["upper"], line=dict(width=0),
                                 showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=fc.index, y=fc["lower"], fill="tonexty",
                                 line=dict(width=0), name="95% range"))
        fig.add_trace(go.Scatter(x=fc.index, y=fc["forecast"], name="Forecast",
                                 line=dict(dash="dash")))
        fig.update_layout(title=f"Imports forecast – next {horizon} months",
                          yaxis_title="Imports (NZD)")
        st.plotly_chart(fig, width="stretch")
        st.caption("Holt-Winters exponential smoothing (trend + yearly seasonality). "
                   "Indicative only.")