"""
ETH Forecasting Executive Dashboard
ECON5380 Group Project: Group 2

Run with:
    streamlit run streamlit_app.py

The app loads precomputed artifacts from ./artifacts (produced by
eth_forecasting.ipynb). If artifacts are missing, the app falls back
to live-pulling data and computing a quick forecast on the fly.
"""

import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="ETH Forecast Dashboard | ECON5380 Group 2",
    page_icon="⟠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Light theme polish
st.markdown(
    """
    <style>
    .main-header {
        background: linear-gradient(90deg, #627EEA 0%, #8A92B2 100%);
        padding: 1.2rem; border-radius: 10px; color: white;
        margin-bottom: 1.2rem;
    }
    .metric-card {
        background: #f7f8fb; padding: 1rem; border-radius: 8px;
        border-left: 4px solid #627EEA;
    }
    div[data-testid="stMetric"] { background: #f7f8fb; padding: 12px;
        border-radius: 8px; border-left: 4px solid #627EEA; }
    </style>
    """,
    unsafe_allow_html=True,
)

ART = Path("artifacts")

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_artifacts():
    if not ART.exists():
        return None
    try:
        panel = pd.read_parquet(ART / "panel.parquet")
        feat = pd.read_parquet(ART / "features.parquet")
        scoreboard = pd.read_csv(ART / "scoreboard.csv")
        sensitivity = pd.read_csv(ART / "sensitivity.csv")
        with open(ART / "model_outputs.pkl", "rb") as f:
            outputs = pickle.load(f)
        return dict(panel=panel, feat=feat, scoreboard=scoreboard,
                    sensitivity=sensitivity, **outputs)
    except Exception as e:
        st.error(f"Could not load artifacts: {e}")
        return None


@st.cache_data(show_spinner=True)
def fallback_live_data():
    """Light fallback if artifacts aren't present yet."""
    import yfinance as yf
    end = pd.Timestamp.today().normalize()
    start = end - pd.DateOffset(years=3)
    eth = yf.download("ETH-USD", start=start, end=end, progress=False, auto_adjust=True)
    if isinstance(eth.columns, pd.MultiIndex):
        eth.columns = eth.columns.get_level_values(0)
    return eth


def simulate_paths(spot, mu_daily, sigma_daily, horizon, n_sim=2000, seed=42):
    rng = np.random.default_rng(seed)
    shocks = rng.normal(mu_daily, sigma_daily, size=(n_sim, horizon))
    return spot * np.exp(np.cumsum(shocks, axis=1))


def _hex_to_rgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))


def rgba(hex_color, alpha):
    r, g, b = _hex_to_rgb(hex_color)
    return f"rgba({r},{g},{b},{alpha})"


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div class="main-header">
        <h1 style="margin:0">⟠ Ethereum 3-Month Forecast Dashboard</h1>
        <div style="opacity:0.9">ECON5380 Group Project · Group 2 · Multi-Model Forecast, Scenarios, Sensitivity</div>
    </div>
    """,
    unsafe_allow_html=True,
)

data = load_artifacts()

if data is None:
    st.warning(
        "Precomputed artifacts not found. Run `eth_forecasting.ipynb` first to "
        "populate `./artifacts/`. Falling back to a lightweight live view."
    )
    eth = fallback_live_data()
    spot = float(eth["Close"].iloc[-1])
    daily_ret = eth["Close"].pct_change().dropna()
    mu, sigma = daily_ret.tail(252).mean(), daily_ret.tail(252).std()
    paths_base = simulate_paths(spot, mu, sigma, 63)
    forecast_dates = pd.bdate_range(eth.index[-1] + pd.Timedelta(days=1), periods=63)

    data = {
        "panel": pd.DataFrame({"ETH_close": eth["Close"]}),
        "feat": None,
        "scoreboard": pd.DataFrame({"Model": ["Live MC"], "RMSE": [np.nan]}),
        "sensitivity": pd.DataFrame(),
        "predictions": {},
        "test_idx": None,
        "actual": None,
        "paths_base": paths_base,
        "scenario_paths": {"Base": paths_base},
        "scenarios": {"Base": {"color": "#1f77b4"}},
        "forecast_dates": forecast_dates,
        "spot": spot,
        "mu": mu,
        "sigma": sigma,
        "horizon": 63,
    }

panel = data["panel"]
spot = float(data["spot"])

# ---------------------------------------------------------------------------
# Top-line KPIs
# ---------------------------------------------------------------------------
col1, col2, col3, col4 = st.columns(4)
ret_1d = panel["ETH_close"].pct_change().iloc[-1]
ret_30d = panel["ETH_close"].pct_change(30).iloc[-1]
ret_90d = panel["ETH_close"].pct_change(90).iloc[-1]

with col1:
    st.metric("Spot Price (ETH/USD)", f"${spot:,.2f}", f"{ret_1d:+.2%} (1d)")
with col2:
    st.metric("30-Day Change", f"{ret_30d:+.2%}")
with col3:
    st.metric("90-Day Change", f"{ret_90d:+.2%}")
with col4:
    median_3m = np.percentile(data["paths_base"][:, -1], 50)
    st.metric("3-Month Median Forecast", f"${median_3m:,.0f}",
              f"{(median_3m / spot - 1):+.2%}")

# ---------------------------------------------------------------------------
# Sidebar controls
# ---------------------------------------------------------------------------
st.sidebar.header("Controls")

scenario_choice = st.sidebar.selectbox(
    "Scenario",
    options=list(data["scenario_paths"].keys()) if data["scenario_paths"] else ["Base"],
    index=min(1, len(data["scenario_paths"]) - 1) if data["scenario_paths"] else 0,
)

horizon_user = st.sidebar.slider("Horizon (trading days)", 5, 126,
                                 value=int(data["horizon"]))
ci_level = st.sidebar.slider("Confidence band", 50, 99, 80, step=5)
lo_pct, hi_pct = (100 - ci_level) / 2, 100 - (100 - ci_level) / 2

st.sidebar.markdown("---")
st.sidebar.markdown("**Custom Sensitivity (Live MC)**")
drift_adj = st.sidebar.slider("Drift adjustment (annualized %)", -100, 100, 0, step=5)
vol_mult = st.sidebar.slider("Volatility multiplier", 0.5, 3.0, 1.0, step=0.1)
n_sim = st.sidebar.slider("Number of simulations", 500, 5000, 2000, step=500)

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["📈 Forecast", "🤖 Model Comparison", "🎭 Scenarios", "📊 Sensitivity", "⚠️ Risk"]
)

# ----------------------------- TAB 1: FORECAST -----------------------------
with tab1:
    st.subheader("3-Month Probabilistic Forecast")

    mu_use = data["mu"] + (drift_adj / 100) / 252
    sig_use = data["sigma"] * vol_mult
    paths_live = simulate_paths(spot, mu_use, sig_use, horizon_user, n_sim=n_sim)
    fdates = pd.bdate_range(panel.index[-1] + pd.Timedelta(days=1), periods=horizon_user)

    p_lo = np.percentile(paths_live, lo_pct, axis=0)
    p_md = np.percentile(paths_live, 50, axis=0)
    p_hi = np.percentile(paths_live, hi_pct, axis=0)

    fig = go.Figure()
    # Historical
    hist_window = panel["ETH_close"].iloc[-180:]
    fig.add_trace(go.Scatter(x=hist_window.index, y=hist_window.values,
                             name="Historical", line=dict(color="black", width=2)))
    # CI band
    fig.add_trace(go.Scatter(x=fdates, y=p_hi, name=f"{int(hi_pct)}th pct",
                             line=dict(width=0), showlegend=False))
    fig.add_trace(go.Scatter(x=fdates, y=p_lo, name=f"{int(lo_pct)}th pct",
                             fill="tonexty", fillcolor="rgba(98,126,234,0.25)",
                             line=dict(width=0), showlegend=True))
    # Median
    fig.add_trace(go.Scatter(x=fdates, y=p_md, name="Median Forecast",
                             line=dict(color="#627EEA", width=3)))
    # A few sample paths
    for i in range(20):
        fig.add_trace(go.Scatter(x=fdates, y=paths_live[i],
                                 line=dict(color="rgba(150,150,150,0.25)", width=0.8),
                                 showlegend=False, hoverinfo="skip"))
    fig.add_vline(x=panel.index[-1], line_dash="dash", line_color="red")

    fig.update_layout(height=520, hovermode="x unified",
                      xaxis_title="Date", yaxis_title="ETH / USD",
                      margin=dict(l=20, r=20, t=10, b=20))
    st.plotly_chart(fig, use_container_width=True)

    c1, c2, c3, c4 = st.columns(4)
    end = paths_live[:, -1]
    c1.metric(f"{int(lo_pct)}th percentile", f"${np.percentile(end, lo_pct):,.0f}")
    c2.metric("Median",                       f"${np.percentile(end, 50):,.0f}")
    c3.metric(f"{int(hi_pct)}th percentile", f"${np.percentile(end, hi_pct):,.0f}")
    c4.metric("Mean return",                  f"{(end / spot - 1).mean():+.1%}")


# ------------------------ TAB 2: MODEL COMPARISON --------------------------
with tab2:
    st.subheader("Out-of-Sample Performance: 120-Day Walk-Forward Test")

    sb = data["scoreboard"].copy()
    if len(sb) > 1:
        sb = sb.sort_values("RMSE").reset_index(drop=True)

        c1, c2 = st.columns(2)
        with c1:
            fig = go.Figure(go.Bar(x=sb["Model"], y=sb["RMSE"],
                                   marker_color="#627EEA"))
            fig.update_layout(title="RMSE (lower is better)", height=360,
                              margin=dict(l=10, r=10, t=40, b=10))
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            fig = go.Figure(go.Bar(x=sb["Model"], y=sb["Dir Acc (%)"],
                                   marker_color="#2ca02c"))
            fig.add_hline(y=50, line_dash="dash", line_color="red",
                          annotation_text="Coin flip")
            fig.update_layout(title="Directional Accuracy (%)", height=360,
                              margin=dict(l=10, r=10, t=40, b=10))
            st.plotly_chart(fig, use_container_width=True)

        st.dataframe(sb.style.format({"RMSE": "{:,.2f}", "MAE": "{:,.2f}",
                                      "MAPE (%)": "{:.2f}", "Dir Acc (%)": "{:.2f}"}),
                     use_container_width=True)

    # Plot predictions vs actual
    if data["predictions"] and data["actual"] is not None:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=data["test_idx"], y=data["actual"],
                                 name="Actual", line=dict(color="black", width=2.5)))
        for name, pred in data["predictions"].items():
            fig.add_trace(go.Scatter(x=data["test_idx"], y=pred,
                                     name=name, line=dict(width=1.2),
                                     opacity=0.75))
        fig.update_layout(height=480, hovermode="x unified",
                          title="Predictions vs Actual",
                          xaxis_title="Date", yaxis_title="ETH / USD",
                          margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig, use_container_width=True)


# --------------------------- TAB 3: SCENARIOS ------------------------------
with tab3:
    st.subheader("Scenario Forecasts: 3-Month Horizon")

    fig = go.Figure()
    hist_window = panel["ETH_close"].iloc[-90:]
    fig.add_trace(go.Scatter(x=hist_window.index, y=hist_window.values,
                             name="Historical", line=dict(color="black", width=2)))

    rows = []
    fdates = data["forecast_dates"]
    for name, paths in data["scenario_paths"].items():
        color = data["scenarios"][name].get("color", "#888888")
        p10, p50, p90 = np.percentile(paths, [10, 50, 90], axis=0)
        fig.add_trace(go.Scatter(x=fdates, y=p90, line=dict(width=0),
                                 showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=fdates, y=p10, fill="tonexty",
                                 fillcolor=rgba(color, 0.12),
                                 line=dict(width=0), showlegend=False,
                                 hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=fdates, y=p50, name=f"{name} (median)",
                                 line=dict(color=color, width=2.5)))
        end = paths[:, -1]
        rows.append({
            "Scenario": name,
            "Median ($)": np.percentile(end, 50),
            "P10 ($)":    np.percentile(end, 10),
            "P90 ($)":    np.percentile(end, 90),
            "Mean Return (%)": (end / spot - 1).mean() * 100,
            "P(loss > 30%)":   (end < spot * 0.70).mean() * 100,
            "P(gain > 50%)":   (end > spot * 1.50).mean() * 100,
        })

    fig.add_vline(x=panel.index[-1], line_dash="dash", line_color="red")
    fig.update_layout(height=520, hovermode="x unified",
                      xaxis_title="Date", yaxis_title="ETH / USD",
                      margin=dict(l=20, r=20, t=10, b=20))
    st.plotly_chart(fig, use_container_width=True)

    df_sc = pd.DataFrame(rows)
    st.dataframe(
        df_sc.style.format({
            "Median ($)": "${:,.0f}", "P10 ($)": "${:,.0f}", "P90 ($)": "${:,.0f}",
            "Mean Return (%)": "{:+.1f}%",
            "P(loss > 30%)": "{:.1f}%", "P(gain > 50%)": "{:.1f}%"
        }), use_container_width=True
    )

    with st.expander("Scenario narratives"):
        st.markdown("""
- **Bull**: Spot ETH ETF launches with strong inflows, dovish Fed pivot, L2 adoption accelerates, staking yields hold 3-4%.
- **Base**: Continued institutional flows, neutral macro, status quo regulation.
- **Bear**: SEC reclassifies ETH as a security, stablecoin contagion, sticky inflation forces higher rates.
- **Black Swan**: Major smart-contract exploit, exchange insolvency, or outright ban in a G7 jurisdiction.
""")


# -------------------------- TAB 4: SENSITIVITY -----------------------------
with tab4:
    st.subheader("Sensitivity: One-at-a-Time Driver Shocks")

    if not data["sensitivity"].empty:
        sens = data["sensitivity"].copy()
        sens = sens.sort_values("Range ($)", ascending=True)
        base_price = (sens["Price Low"] + sens["Price High"]).mean() / 2

        fig = go.Figure()
        fig.add_trace(go.Bar(y=sens["Driver"],
                             x=sens["Price High"] - base_price,
                             base=base_price, orientation="h",
                             marker_color="#2ca02c", name="Upside shock"))
        fig.add_trace(go.Bar(y=sens["Driver"],
                             x=sens["Price Low"] - base_price,
                             base=base_price, orientation="h",
                             marker_color="#d62728", name="Downside shock"))
        fig.add_vline(x=base_price, line_color="black", line_width=2)
        fig.update_layout(barmode="overlay", height=480,
                          xaxis_title="Next-day ETH price ($)",
                          margin=dict(l=20, r=20, t=10, b=20))
        st.plotly_chart(fig, use_container_width=True)

        st.dataframe(sens.style.format({
            "Price Low": "${:,.2f}", "Price High": "${:,.2f}", "Range ($)": "${:,.2f}"
        }), use_container_width=True)
    else:
        st.info("Run the notebook to populate sensitivity artifacts.")

    st.markdown("---")
    st.subheader("Interactive: Joint Sensitivity (Live MC)")
    st.caption("Adjust drift and vol on the sidebar to see how a stress scenario "
               "propagates through the 3-month forecast band.")


# ------------------------------ TAB 5: RISK --------------------------------
with tab5:
    st.subheader("Risk Metrics: Base Scenario")

    paths = data["paths_base"]
    end_ret = paths[:, -1] / spot - 1
    var95, var99 = np.percentile(end_ret, 5), np.percentile(end_ret, 1)
    cvar95 = end_ret[end_ret <= var95].mean()
    cvar99 = end_ret[end_ret <= var99].mean()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("VaR 95% (3M)", f"{var95:.1%}")
    c2.metric("VaR 99% (3M)", f"{var99:.1%}")
    c3.metric("CVaR 95% (3M)", f"{cvar95:.1%}")
    c4.metric("CVaR 99% (3M)", f"{cvar99:.1%}")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("P(loss > 20%)", f"{(end_ret < -0.20).mean():.1%}")
    c2.metric("P(loss > 50%)", f"{(end_ret < -0.50).mean():.1%}")
    c3.metric("P(gain > 20%)", f"{(end_ret >  0.20).mean():.1%}")
    c4.metric("Expected return", f"{end_ret.mean():+.1%}")

    fig = go.Figure(go.Histogram(x=end_ret * 100, nbinsx=60,
                                 marker_color="#627EEA"))
    fig.add_vline(x=var95 * 100, line_dash="dash", line_color="red",
                  annotation_text="VaR 95%")
    fig.add_vline(x=var99 * 100, line_dash="dash", line_color="darkred",
                  annotation_text="VaR 99%")
    fig.update_layout(title="Distribution of 3-Month Returns",
                      xaxis_title="Return (%)", yaxis_title="Frequency",
                      height=420, margin=dict(l=20, r=20, t=40, b=20))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.markdown(
        """
**Reading the numbers.** A 95% VaR of, say, -28% means there is a 5% probability
that ETH falls more than 28% over the 3-month horizon under the base scenario.
CVaR is the average loss conditional on being in that worst-5% tail. It captures
how bad the bad outcomes get, not just how often they happen.

Both numbers are sensitive to the volatility input. See the **Sensitivity** tab
for how the price band moves under stressed inputs, and the **Scenarios** tab
for narrative-driven stress tests.
        """
    )

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.markdown("---")
st.caption("ECON5380 · Group 2 · ETH Price Forecasting · "
           f"Data through {panel.index[-1].date()}. "
           "Forecasts are model-driven and not investment advice.")
