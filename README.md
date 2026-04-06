# 📈 Stock Trend Analyzer

An interactive stock analysis dashboard built with Streamlit. Enter any ticker, explore historical prices with technical indicators, run Monte Carlo simulations, and get a Prophet-based forecast — all in the browser.

---

## Features

- **Price History** — Line chart and candlestick chart with a range slider and volume overlay
- **Key Metrics** — Latest close, day change, 52-week high/low, average volume, period return
- **Technical Indicators** — Configurable moving averages (10/20/50/100/200-day), Bollinger Bands, RSI with overbought/oversold signals
- **Monte Carlo Simulation** — Geometric Brownian Motion with configurable horizon (1–5 years) and number of simulations (50–500). Shows percentile bands, return distribution histogram, VaR/CVaR, and an investment return estimator
- **Prophet Forecast** — Facebook Prophet time-series forecast with confidence intervals, trend decomposition, and a horizon summary table (1W / 1M / 6M / 1Y)
- **Custom ticker input** — Analyze any stock beyond the preset list
- **Custom date range** — Filter historical data by any start/end date

---

## Run Locally

**Prerequisites:** Python 3.10+

```bash
# 1. Clone the repo
git clone https://github.com/shahriar3174/Stock_Trend_Analyzer.git
cd Stock_Trend_Analyzer

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the app
streamlit run app.py
```

The app will open at `http://localhost:8501`.

---

## Deploy to Streamlit Community Cloud (free)

1. Fork or push this repo to your GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in
3. Click **"New app"** → select your repo → set **Main file path** to `app.py`
4. Click **Deploy** — it's free for public repos

---

## File Structure

```
├── app.py             # Main Streamlit application
├── requirements.txt   # Python dependencies
└── README.md          # This file
```

---

## Tech Stack

| Library | Purpose |
|---|---|
| [Streamlit](https://streamlit.io) | Web UI framework |
| [yfinance](https://github.com/ranaroussi/yfinance) | Yahoo Finance data |
| [Plotly](https://plotly.com/python/) | Interactive charts |
| [Prophet](https://facebook.github.io/prophet/) | Time-series forecasting |
| [pandas](https://pandas.pydata.org/) / [numpy](https://numpy.org/) | Data processing |

---

## Disclaimer

This tool is for educational and informational purposes only. Nothing here constitutes financial advice. Stock market investments carry risk and past performance does not guarantee future results.
