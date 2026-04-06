"""
Stock Trend Analyzer
=====================
A Streamlit app for exploring historical stock prices,
running Monte Carlo simulations, and forecasting with Prophet.

Run locally:
    pip install -r requirements.txt
    streamlit run app.py
"""

import streamlit as st
import yfinance as yf
from prophet import Prophet
from datetime import date, timedelta
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Stock Trend Analyzer",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Constants ─────────────────────────────────────────────────────────────────
START_DATE = "2015-01-01"
END_DATE   = date.today().strftime("%Y-%m-%d")

STOCK_OPTIONS = {
    "Apple (AAPL)":        "AAPL",
    "Google (GOOGL)":      "GOOGL",
    "Microsoft (MSFT)":    "MSFT",
    "Amazon (AMZN)":       "AMZN",
    "Meta (META)":         "META",
    "Tesla (TSLA)":        "TSLA",
    "NVIDIA (NVDA)":       "NVDA",
    "Netflix (NFLX)":      "NFLX",
    "Disney (DIS)":        "DIS",
    "GameStop (GME)":      "GME",
    "Berkshire Hathaway (BRK-B)": "BRK-B",
    "JPMorgan Chase (JPM)":"JPM",
    "S&P 500 ETF (SPY)":   "SPY",
}

# ── Data loading ──────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_stock_data(ticker: str, start: str, end: str) -> pd.DataFrame:
    """
    Download OHLCV + Adj Close data for a single ticker.
    Returns a flat DataFrame with columns:
        Date, Open, High, Low, Close, Adj Close, Volume
    Raises ValueError if no data was returned.
    """
    raw = yf.download(ticker, start=start, end=end, auto_adjust=False, progress=False)
    if raw.empty:
        raise ValueError(f"No data found for ticker '{ticker}'. "
                         "Check that the symbol is correct and has data in the selected range.")

    raw.reset_index(inplace=True)

    # yfinance ≥0.2 returns MultiIndex columns like ('Close', 'AAPL').
    # Flatten to single-level by keeping only the first level.
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = [col[0] for col in raw.columns]

    # Ensure 'Date' column is datetime
    raw['Date'] = pd.to_datetime(raw['Date'])

    # Drop any fully-empty rows
    raw.dropna(how='all', subset=['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume'],
               inplace=True)
    raw.reset_index(drop=True, inplace=True)
    return raw


# ── Technical indicators ──────────────────────────────────────────────────────
def add_moving_averages(df: pd.DataFrame, windows=(20, 50, 200)) -> pd.DataFrame:
    df = df.copy()
    for w in windows:
        if len(df) >= w:
            df[f'MA{w}'] = df['Close'].rolling(w).mean()
    return df


def add_rsi(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    df = df.copy()
    delta = df['Close'].diff()
    gain  = delta.clip(lower=0)
    loss  = -delta.clip(upper=0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs  = avg_gain / avg_loss.replace(0, np.nan)
    df['RSI'] = 100 - (100 / (1 + rs))
    return df


def add_bollinger_bands(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    df = df.copy()
    if len(df) >= window:
        rolling = df['Close'].rolling(window)
        df['BB_Mid']   = rolling.mean()
        df['BB_Upper'] = df['BB_Mid'] + 2 * rolling.std()
        df['BB_Lower'] = df['BB_Mid'] - 2 * rolling.std()
    return df


# ── Monte Carlo simulation ────────────────────────────────────────────────────
def monte_carlo_simulation(
    data: pd.DataFrame,
    days: int = 365,
    num_simulations: int = 200,
    price_col: str = 'Adj Close',
) -> np.ndarray:
    """
    Geometric Brownian Motion simulation.
    Returns array of shape (num_simulations, days).
    """
    prices  = data[price_col].dropna()
    returns = prices.pct_change().dropna()
    mu      = returns.mean()
    sigma   = returns.std()
    last_price = float(prices.iloc[-1])

    rng         = np.random.default_rng(seed=42)  # reproducible per stock
    rand_matrix = rng.standard_normal((num_simulations, days))
    daily_ret   = np.exp((mu - 0.5 * sigma ** 2) + sigma * rand_matrix)

    simulations = np.zeros((num_simulations, days))
    simulations[:, 0] = last_price * daily_ret[:, 0]
    for d in range(1, days):
        simulations[:, d] = simulations[:, d - 1] * daily_ret[:, d]

    return simulations


# ── Prophet forecast ──────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def run_prophet_forecast(data: pd.DataFrame, periods: int) -> tuple[pd.DataFrame, object]:
    """Fit Prophet on Close prices and return (forecast_df, model)."""
    df_prophet = data[['Date', 'Close']].rename(columns={'Date': 'ds', 'Close': 'y'})
    df_prophet['ds'] = pd.to_datetime(df_prophet['ds'])
    model = Prophet(daily_seasonality=False, yearly_seasonality=True, weekly_seasonality=True)
    model.fit(df_prophet)
    future   = model.make_future_dataframe(periods=periods)
    forecast = model.predict(future)
    return forecast, model


# ── Helpers ───────────────────────────────────────────────────────────────────
def fmt_currency(val: float) -> str:
    sign = "+" if val >= 0 else ""
    return f"{sign}${val:,.2f}"


def safe_idx(arr: np.ndarray, idx: int) -> float:
    return float(arr[min(idx, len(arr) - 1)])


# ══════════════════════════════════════════════════════════════════════════════
# UI
# ══════════════════════════════════════════════════════════════════════════════

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("📈 Stock Trend Analyzer")
    st.markdown("---")

    # Stock selection
    st.subheader("Stock Selection")
    selection_mode = st.radio("Choose stock by", ["Preset list", "Custom ticker"], horizontal=True)

    if selection_mode == "Preset list":
        display_name  = st.selectbox("Select a stock", list(STOCK_OPTIONS.keys()))
        selected_ticker = STOCK_OPTIONS[display_name]
    else:
        raw_ticker = st.text_input("Enter ticker symbol (e.g. BABA, TSM, COIN)", value="AAPL")
        selected_ticker = raw_ticker.strip().upper()

    st.markdown("---")

    # Date range
    st.subheader("Date Range")
    col_s, col_e = st.columns(2)
    with col_s:
        start_input = st.date_input("Start", value=date(2015, 1, 1), max_value=date.today())
    with col_e:
        end_input = st.date_input("End", value=date.today(), max_value=date.today())

    if start_input >= end_input:
        st.error("Start date must be before end date.")
        st.stop()

    start_str = start_input.strftime("%Y-%m-%d")
    end_str   = end_input.strftime("%Y-%m-%d")

    st.markdown("---")

    # Tabs to show
    st.subheader("Features")
    show_indicators  = st.checkbox("Technical Indicators", value=True)
    show_monte_carlo = st.checkbox("Monte Carlo Simulation", value=True)
    show_prophet     = st.checkbox("Prophet Forecast", value=True)

    st.markdown("---")
    st.caption("Data provided by Yahoo Finance. Not financial advice.")


# ── Load data ─────────────────────────────────────────────────────────────────
with st.spinner(f"Loading data for **{selected_ticker}**…"):
    try:
        stock_data = load_stock_data(selected_ticker, start_str, end_str)
    except Exception as e:
        st.error(f"❌ Could not load data: {e}")
        st.stop()

if stock_data.empty:
    st.warning("No data returned. Try a different ticker or date range.")
    st.stop()

# ── Header KPIs ───────────────────────────────────────────────────────────────
latest   = stock_data.iloc[-1]
previous = stock_data.iloc[-2] if len(stock_data) > 1 else latest
day_chg  = float(latest['Close']) - float(previous['Close'])
day_pct  = (day_chg / float(previous['Close'])) * 100 if float(previous['Close']) != 0 else 0

period_start_price = float(stock_data.iloc[0]['Close'])
period_end_price   = float(stock_data.iloc[-1]['Close'])
period_return      = ((period_end_price - period_start_price) / period_start_price) * 100

st.title(f"📈 {selected_ticker} — Stock Trend Analyzer")
st.caption(f"Data from {start_str} to {end_str}  ·  {len(stock_data):,} trading days")

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Latest Close",  f"${latest['Close']:.2f}", f"{day_chg:+.2f} ({day_pct:+.2f}%)")
k2.metric("52-Week High",  f"${stock_data['High'].tail(252).max():.2f}")
k3.metric("52-Week Low",   f"${stock_data['Low'].tail(252).min():.2f}")
k4.metric("Avg Volume",    f"{int(stock_data['Volume'].mean()):,}")
k5.metric(f"Return ({start_input.year}–now)", f"{period_return:+.1f}%")

st.markdown("---")

# ── Raw data table ────────────────────────────────────────────────────────────
with st.expander("📋 Raw Data (last 10 rows)", expanded=False):
    display_df = stock_data.tail(10).copy()
    display_df['Date'] = display_df['Date'].dt.strftime('%Y-%m-%d')
    st.dataframe(display_df.set_index('Date').style.format({
        'Open': '${:.2f}', 'High': '${:.2f}', 'Low': '${:.2f}',
        'Close': '${:.2f}', 'Adj Close': '${:.2f}',
        'Volume': '{:,.0f}',
    }), use_container_width=True)

# ── Price History Chart ───────────────────────────────────────────────────────
st.subheader("📊 Price History")

price_tab, candle_tab = st.tabs(["Line Chart", "Candlestick"])

with price_tab:
    fig_line = go.Figure()
    fig_line.add_trace(go.Scatter(
        x=stock_data['Date'], y=stock_data['Close'],
        mode='lines', name='Close', line=dict(color='#00b4d8', width=1.5)
    ))
    fig_line.add_trace(go.Bar(
        x=stock_data['Date'], y=stock_data['Volume'],
        name='Volume', yaxis='y2',
        marker_color='rgba(100,100,200,0.25)', showlegend=True
    ))
    fig_line.update_layout(
        height=420,
        margin=dict(l=0, r=0, t=30, b=0),
        yaxis=dict(title='Price (USD)', side='left', showgrid=True),
        yaxis2=dict(title='Volume', side='right', overlaying='y', showgrid=False),
        xaxis=dict(title='Date', rangeslider=dict(visible=True)),
        legend=dict(orientation='h', yanchor='bottom', y=1.02),
        hovermode='x unified',
    )
    st.plotly_chart(fig_line, use_container_width=True)

with candle_tab:
    fig_candle = go.Figure(data=[go.Candlestick(
        x=stock_data['Date'],
        open=stock_data['Open'],
        high=stock_data['High'],
        low=stock_data['Low'],
        close=stock_data['Close'],
        name=selected_ticker,
    )])
    fig_candle.update_layout(
        height=420,
        margin=dict(l=0, r=0, t=30, b=0),
        xaxis_rangeslider_visible=True,
        hovermode='x unified',
    )
    st.plotly_chart(fig_candle, use_container_width=True)

# ── Technical Indicators ──────────────────────────────────────────────────────
if show_indicators:
    st.subheader("📐 Technical Indicators")

    ind_col1, ind_col2 = st.columns(2)

    with ind_col1:
        # Moving averages
        ma_windows = st.multiselect(
            "Moving Averages",
            options=[10, 20, 50, 100, 200],
            default=[20, 50, 200],
        )

    with ind_col2:
        show_bb  = st.checkbox("Bollinger Bands (20-day)", value=True)
        show_rsi = st.checkbox("RSI (14-day)", value=True)

    df_ind = stock_data.copy()
    if ma_windows:
        df_ind = add_moving_averages(df_ind, windows=ma_windows)
    if show_bb:
        df_ind = add_bollinger_bands(df_ind)
    if show_rsi:
        df_ind = add_rsi(df_ind)

    # Price + MA + BB chart
    MA_COLORS = ['#f4a261', '#e9c46a', '#2a9d8f', '#e76f51', '#264653']
    fig_ma = go.Figure()
    fig_ma.add_trace(go.Scatter(
        x=df_ind['Date'], y=df_ind['Close'],
        mode='lines', name='Close',
        line=dict(color='#00b4d8', width=1.5)
    ))
    for i, w in enumerate(sorted(ma_windows)):
        col_name = f'MA{w}'
        if col_name in df_ind.columns:
            fig_ma.add_trace(go.Scatter(
                x=df_ind['Date'], y=df_ind[col_name],
                mode='lines', name=f'{w}-day MA',
                line=dict(color=MA_COLORS[i % len(MA_COLORS)], width=1.4, dash='dot')
            ))
    if show_bb and 'BB_Upper' in df_ind.columns:
        fig_ma.add_trace(go.Scatter(
            x=df_ind['Date'], y=df_ind['BB_Upper'],
            mode='lines', name='BB Upper',
            line=dict(color='rgba(200,200,200,0.4)', width=1),
        ))
        fig_ma.add_trace(go.Scatter(
            x=df_ind['Date'], y=df_ind['BB_Lower'],
            mode='lines', name='BB Lower',
            line=dict(color='rgba(200,200,200,0.4)', width=1),
            fill='tonexty',
            fillcolor='rgba(200,200,200,0.07)',
        ))

    fig_ma.update_layout(
        height=400, margin=dict(l=0, r=0, t=30, b=0),
        yaxis_title='Price (USD)', xaxis_title='Date',
        hovermode='x unified',
        legend=dict(orientation='h', yanchor='bottom', y=1.02),
    )
    st.plotly_chart(fig_ma, use_container_width=True)

    # RSI chart
    if show_rsi and 'RSI' in df_ind.columns:
        fig_rsi = go.Figure()
        fig_rsi.add_trace(go.Scatter(
            x=df_ind['Date'], y=df_ind['RSI'],
            mode='lines', name='RSI (14)',
            line=dict(color='#e76f51', width=1.5)
        ))
        fig_rsi.add_hline(y=70, line_dash='dash', line_color='red',   annotation_text='Overbought (70)')
        fig_rsi.add_hline(y=30, line_dash='dash', line_color='green', annotation_text='Oversold (30)')
        fig_rsi.update_layout(
            height=220, margin=dict(l=0, r=0, t=30, b=0),
            yaxis=dict(title='RSI', range=[0, 100]),
            xaxis_title='Date',
            hovermode='x unified',
        )
        st.plotly_chart(fig_rsi, use_container_width=True)

        # RSI signal today
        current_rsi = df_ind['RSI'].dropna().iloc[-1]
        if current_rsi > 70:
            st.warning(f"⚠️ RSI is **{current_rsi:.1f}** — stock may be **overbought**.")
        elif current_rsi < 30:
            st.success(f"💡 RSI is **{current_rsi:.1f}** — stock may be **oversold**.")
        else:
            st.info(f"RSI is **{current_rsi:.1f}** — within neutral range.")

# ── Monte Carlo Simulation ────────────────────────────────────────────────────
if show_monte_carlo:
    st.subheader("🎲 Monte Carlo Simulation")
    st.caption(
        "Uses Geometric Brownian Motion (GBM) to simulate possible future price paths. "
        "Each path is one equally-likely scenario — not a prediction."
    )

    mc_col1, mc_col2 = st.columns(2)
    with mc_col1:
        num_years = st.slider("Forecast horizon (years)", 1, 5, value=1)
    with mc_col2:
        num_sims  = st.slider("Number of simulations", 50, 500, value=200, step=50)

    prediction_days = num_years * 252   # trading days

    with st.spinner("Running simulations…"):
        simulations = monte_carlo_simulation(
            stock_data, days=prediction_days,
            num_simulations=num_sims
        )

    last_price = float(stock_data['Adj Close'].iloc[-1])

    # Plot
    fig_mc = go.Figure()
    # Individual paths (thin, transparent)
    for i, sim in enumerate(simulations):
        fig_mc.add_trace(go.Scatter(
            x=list(range(1, prediction_days + 1)), y=sim,
            mode='lines',
            line=dict(width=0.5, color='rgba(100,180,255,0.15)'),
            showlegend=False,
        ))
    # Percentile bands
    p10 = np.percentile(simulations, 10, axis=0)
    p50 = np.percentile(simulations, 50, axis=0)
    p90 = np.percentile(simulations, 90, axis=0)
    x_axis = list(range(1, prediction_days + 1))
    fig_mc.add_trace(go.Scatter(x=x_axis, y=p90, mode='lines', name='90th pct',
                                line=dict(color='#2ecc71', width=2)))
    fig_mc.add_trace(go.Scatter(x=x_axis, y=p50, mode='lines', name='Median',
                                line=dict(color='#f4a261', width=2.5)))
    fig_mc.add_trace(go.Scatter(x=x_axis, y=p10, mode='lines', name='10th pct',
                                line=dict(color='#e74c3c', width=2)))
    fig_mc.update_layout(
        height=420, margin=dict(l=0, r=0, t=30, b=0),
        xaxis_title='Trading Days',
        yaxis_title='Simulated Price (USD)',
        hovermode='x unified',
        legend=dict(orientation='h', yanchor='bottom', y=1.02),
    )
    st.plotly_chart(fig_mc, use_container_width=True)

    # Return distribution at end of period
    final_prices  = simulations[:, -1]
    final_returns = (final_prices - last_price) / last_price * 100
    fig_hist = px.histogram(
        final_returns, nbins=60,
        labels={'value': f'Simulated Return after {num_years}y (%)'},
        title=f"Distribution of Simulated Returns ({num_years}-year horizon)",
        color_discrete_sequence=['#00b4d8'],
    )
    fig_hist.add_vline(x=0, line_dash='dash', line_color='white', annotation_text='Break-even')
    fig_hist.update_layout(height=300, margin=dict(l=0, r=0, t=50, b=0), showlegend=False)
    st.plotly_chart(fig_hist, use_container_width=True)

    # Simulation summary stats
    sim_mean = np.mean(simulations, axis=0)
    trading_days_map = {
        '1 Week':   min(5,   prediction_days - 1),
        '1 Month':  min(21,  prediction_days - 1),
        '6 Months': min(126, prediction_days - 1),
        '1 Year':   min(252, prediction_days - 1),
    }

    # Investment calculator
    st.subheader("💰 Investment Return Estimator")
    st.caption("Based on the median Monte Carlo path. Not a guarantee of future returns.")
    investment_amount = st.slider("Investment amount ($)", 100, 50_000, 1_000, step=100)

    cols = st.columns(len(trading_days_map))
    for col, (label, day_idx) in zip(cols, trading_days_map.items()):
        projected_price  = safe_idx(sim_mean, day_idx)
        price_change_pct = (projected_price - last_price) / last_price
        profit           = investment_amount * price_change_pct
        col.metric(label, fmt_currency(profit), f"{price_change_pct * 100:+.1f}%")

    st.info(
        "⚠️ These projections are simulated estimates based on historical volatility. "
        "Actual returns will differ. This is not financial advice."
    )

    # VaR / CVaR
    var_95  = np.percentile(final_returns, 5)
    cvar_95 = final_returns[final_returns <= var_95].mean()
    v1, v2 = st.columns(2)
    v1.metric("Value at Risk (95%, full period)",
              f"{var_95:.1f}%",
              help="95% of simulations resulted in a return better than this.")
    v2.metric("Conditional VaR (Expected Shortfall)",
              f"{cvar_95:.1f}%",
              help="Average return in the worst 5% of simulated outcomes.")


# ── Prophet Forecast ──────────────────────────────────────────────────────────
if show_prophet:
    st.subheader("🔮 Prophet Forecast")
    st.caption(
        "Facebook Prophet decomposes the time series into trend, weekly, and yearly seasonality components."
    )

    prophet_years = st.slider("Forecast horizon (years)", 1, 4, value=1, key="prophet_years")
    prophet_days  = prophet_years * 365

    with st.spinner("Fitting Prophet model…"):
        try:
            forecast, model = run_prophet_forecast(stock_data, periods=prophet_days)
        except Exception as e:
            st.error(f"Prophet forecast failed: {e}")
            st.stop()

    # Forecast chart
    hist_end = stock_data['Date'].max()
    fig_prophet = go.Figure()
    fig_prophet.add_trace(go.Scatter(
        x=stock_data['Date'], y=stock_data['Close'],
        mode='lines', name='Historical Close',
        line=dict(color='#00b4d8', width=1.5)
    ))
    fig_prophet.add_trace(go.Scatter(
        x=forecast['ds'], y=forecast['yhat'],
        mode='lines', name='Forecast',
        line=dict(color='#f4a261', width=2)
    ))
    fig_prophet.add_trace(go.Scatter(
        x=pd.concat([forecast['ds'], forecast['ds'][::-1]]),
        y=pd.concat([forecast['yhat_upper'], forecast['yhat_lower'][::-1]]),
        fill='toself',
        fillcolor='rgba(244,162,97,0.15)',
        line=dict(color='rgba(244,162,97,0.3)', width=0),
        name='Confidence interval',
    ))
    fig_prophet.add_vline(
        x=hist_end.timestamp() * 1000,
        line_dash='dash', line_color='gray',
        annotation_text='Today',
    )
    fig_prophet.update_layout(
        height=440, margin=dict(l=0, r=0, t=30, b=0),
        yaxis_title='Price (USD)', xaxis_title='Date',
        hovermode='x unified',
        legend=dict(orientation='h', yanchor='bottom', y=1.02),
    )
    st.plotly_chart(fig_prophet, use_container_width=True)

    # Trend + seasonality components
    with st.expander("📉 Trend & Seasonality Components", expanded=False):
        comp_fig = go.Figure()
        comp_fig.add_trace(go.Scatter(
            x=forecast['ds'], y=forecast['trend'],
            mode='lines', name='Trend', line=dict(color='#e76f51')
        ))
        comp_fig.update_layout(height=260, yaxis_title='Trend', xaxis_title='Date',
                                margin=dict(l=0, r=0, t=20, b=0))
        st.plotly_chart(comp_fig, use_container_width=True)

    # Forecast summary
    fut = forecast[forecast['ds'] > hist_end][['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
    if not fut.empty:
        p1w  = fut[fut['ds'] <= hist_end + timedelta(days=7)].iloc[-1]   if len(fut) >= 1  else None
        p1m  = fut[fut['ds'] <= hist_end + timedelta(days=30)].iloc[-1]  if len(fut) >= 1  else None
        p6m  = fut[fut['ds'] <= hist_end + timedelta(days=180)].iloc[-1] if len(fut) >= 1  else None
        p1y  = fut[fut['ds'] <= hist_end + timedelta(days=365)].iloc[-1] if len(fut) >= 1  else None

        rows = []
        for label, row in [("1 Week", p1w), ("1 Month", p1m), ("6 Months", p6m), ("1 Year", p1y)]:
            if row is not None:
                chg = ((row['yhat'] - float(latest['Close'])) / float(latest['Close'])) * 100
                rows.append({
                    "Horizon":    label,
                    "Forecast":   f"${row['yhat']:.2f}",
                    "Low (80%)":  f"${row['yhat_lower']:.2f}",
                    "High (80%)": f"${row['yhat_upper']:.2f}",
                    "Change %":   f"{chg:+.1f}%",
                })
        if rows:
            st.table(pd.DataFrame(rows).set_index("Horizon"))

    st.info(
        "⚠️ Prophet extrapolates historical patterns. "
        "Forecasts become less reliable further into the future. Not financial advice."
    )

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption(
    "Stock Trend Analyzer · Data via [Yahoo Finance](https://finance.yahoo.com) · "
    "Built with Streamlit, Plotly, Prophet & yfinance"
)
