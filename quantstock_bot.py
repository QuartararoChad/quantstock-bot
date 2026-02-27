import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

# ── Try optional auto-refresh (install: pip install streamlit-autorefresh) ──
try:
    from streamlit_autorefresh import st_autorefresh
    HAS_AUTOREFRESH = True
except ImportError:
    HAS_AUTOREFRESH = False

st.set_page_config(page_title="QuantStock Bot v3.0", layout="wide", page_icon="🚀")
st.title("🚀 QuantStock Bot v3.0")
st.caption("Fixed & Improved • Pure numpy/pandas indicators • yfinance MultiIndex safe • Auto-refresh")

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Controls")
    tickers_input = st.text_input("Stocks (comma-separated)", "AAPL,TSLA,NVDA,AMZN")
    tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]
    tf = st.selectbox("Timeframe", ["5m", "15m", "30m", "1h", "1d"], index=3)
    refresh_mins = st.slider("Auto-refresh every (minutes)", 1, 15, 5)
    if st.button("🔄 Manual Refresh Now"):
        st.cache_data.clear()
        st.rerun()

if not tickers:
    st.error("Enter at least one ticker.")
    st.stop()

# Auto-refresh if library is available
if HAS_AUTOREFRESH:
    st_autorefresh(interval=refresh_mins * 60 * 1000, limit=9999, key="quantrefresh")
else:
    st.info("💡 Install `streamlit-autorefresh` for automatic updates. Using manual refresh for now.")

# ── Data fetching ─────────────────────────────────────────────────────────────
@st.cache_data(ttl=60)
def get_data(ticker: str, interval: str) -> pd.DataFrame | None:
    period = "60d" if interval == "1d" else "10d"
    try:
        df = yf.download(
            ticker,
            period=period,
            interval=interval,
            progress=False,
            auto_adjust=True,   # keeps columns simple: Open High Low Close Volume
            prepost=False,
        )
    except Exception as e:
        st.warning(f"⚠️ Download error for {ticker}: {e}")
        return None

    # ── FIX: Flatten MultiIndex columns returned by yfinance >= 0.2 ──
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # Drop duplicate columns if any
    df = df.loc[:, ~df.columns.duplicated()]

    if df.empty or len(df) < 20:
        st.warning(f"⚠️ Not enough data for {ticker} on {interval} timeframe.")
        return None

    return df


# ── Technical Indicators ──────────────────────────────────────────────────────
def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    close = df["Close"].squeeze()   # ensure 1-D Series
    high  = df["High"].squeeze()
    low   = df["Low"].squeeze()

    # SMA
    df["SMA_20"] = close.rolling(20).mean()
    df["SMA_50"] = close.rolling(50).mean()

    # RSI (Wilder smoothing)
    delta     = close.diff()
    gain      = delta.clip(lower=0)
    loss      = -delta.clip(upper=0)
    avg_gain  = gain.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    avg_loss  = loss.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    rs        = avg_gain / avg_loss.replace(0, np.nan)
    df["RSI"] = 100 - (100 / (1 + rs))

    # MACD
    ema12        = close.ewm(span=12, adjust=False).mean()
    ema26        = close.ewm(span=26, adjust=False).mean()
    df["MACD"]   = ema12 - ema26
    df["MACD_S"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_H"] = df["MACD"] - df["MACD_S"]

    # ATR (True Range average)
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low  - prev_close).abs(),
    ], axis=1).max(axis=1)
    df["ATR"] = tr.ewm(span=10, adjust=False).mean()

    # SuperTrend (proper implementation)
    factor    = 3.0
    period_st = 10
    atr_st    = tr.rolling(period_st).mean()
    hl2       = (high + low) / 2
    upper_raw = hl2 + factor * atr_st
    lower_raw = hl2 - factor * atr_st

    upper = upper_raw.copy()
    lower = lower_raw.copy()
    direction = pd.Series(index=df.index, dtype=int)

    for i in range(1, len(df)):
        # Lower band
        lower.iloc[i] = (
            lower_raw.iloc[i]
            if lower_raw.iloc[i] > lower.iloc[i-1] or close.iloc[i-1] < lower.iloc[i-1]
            else lower.iloc[i-1]
        )
        # Upper band
        upper.iloc[i] = (
            upper_raw.iloc[i]
            if upper_raw.iloc[i] < upper.iloc[i-1] or close.iloc[i-1] > upper.iloc[i-1]
            else upper.iloc[i-1]
        )
        # Direction
        if close.iloc[i] > upper.iloc[i-1]:
            direction.iloc[i] = 1
        elif close.iloc[i] < lower.iloc[i-1]:
            direction.iloc[i] = -1
        else:
            direction.iloc[i] = direction.iloc[i-1] if i > 0 else 1

    df["ST_upper"]     = upper
    df["ST_lower"]     = lower
    df["ST_dir"]       = direction
    df["SuperTrend"]   = np.where(direction == 1, lower, upper)

    # Bollinger Bands
    sma20        = close.rolling(20).mean()
    std20        = close.rolling(20).std()
    df["BB_mid"] = sma20
    df["BB_up"]  = sma20 + 2 * std20
    df["BB_low"] = sma20 - 2 * std20

    # Volume SMA
    df["Vol_SMA"] = df["Volume"].rolling(20).mean()

    return df


# ── Signal Engine ─────────────────────────────────────────────────────────────
def get_signal(df: pd.DataFrame):
    if len(df) < 10:
        close = float(df["Close"].iloc[-1])
        return "NEUTRAL", 50, [], "⏳ HOLD - Not enough data", round(close, 2)

    r  = df.iloc[-1]   # latest row
    p  = df.iloc[-2]   # previous row
    close = float(r["Close"])

    score   = 0.0
    signals = []

    def val(col, default=0):
        v = r.get(col, default)
        return float(v) if pd.notna(v) else default

    rsi  = val("RSI", 50)
    macd = val("MACD", 0)
    macs = val("MACD_S", 0)
    st_d = val("ST_dir", 0)
    sma20 = val("SMA_20", close)
    bb_up = val("BB_up", close * 1.05)
    bb_lo = val("BB_low", close * 0.95)
    vol   = float(r.get("Volume", 0) or 0)
    vol_a = float(r.get("Vol_SMA", vol) or vol)

    # ── Bullish signals ──
    if close > sma20:
        score += 1.5
        signals.append("Above SMA20 ✅")

    if rsi < 35:
        score += 2.5
        signals.append(f"RSI Oversold ({rsi:.0f}) 🟢")
    elif rsi > 65:
        score -= 2.5
        signals.append(f"RSI Overbought ({rsi:.0f}) 🔴")

    prev_macd = float(p.get("MACD", 0) or 0)
    prev_macs = float(p.get("MACD_S", 0) or 0)
    if macd > macs and prev_macd <= prev_macs:
        score += 2.5
        signals.append("MACD Bull Cross 🟢")
    elif macd < macs and prev_macd >= prev_macs:
        score -= 2.5
        signals.append("MACD Bear Cross 🔴")
    elif macd > macs:
        score += 1.0
    else:
        score -= 1.0

    if st_d == 1:
        score += 3.0
        signals.append("SuperTrend BULL 🟢")
    elif st_d == -1:
        score -= 3.0
        signals.append("SuperTrend BEAR 🔴")

    if close < bb_lo:
        score += 1.5
        signals.append("Below BB Lower 🟢")
    elif close > bb_up:
        score -= 1.5
        signals.append("Above BB Upper 🔴")

    if vol_a > 0 and vol > vol_a * 1.5:
        score += 1.0 if score > 0 else -1.0
        signals.append("High Volume Spike ⚡")

    # ── Determine direction ──
    if score >= 4:
        direction = "BULLISH"
    elif score <= -4:
        direction = "BEARISH"
    else:
        direction = "NEUTRAL"

    confidence = max(20, min(98, int(50 + score * 6)))

    atr = float(r.get("ATR", (float(r["High"]) - float(r["Low"]))))
    if direction == "BULLISH" and confidence >= 65:
        tp  = round(close + atr * 3, 2)
        sl  = round(close - atr * 1.5, 2)
        rec = f"🚀 BUY  •  TP ${tp}  •  SL ${sl}"
    elif direction == "BEARISH" and confidence >= 65:
        tp  = round(close - atr * 3, 2)
        sl  = round(close + atr * 1.5, 2)
        rec = f"🔻 SELL  •  TP ${tp}  •  SL ${sl}"
    else:
        rec = "⏳ HOLD — Waiting for stronger signal"

    return direction, confidence, signals[:5], rec, round(close, 2)


# ── Dashboard ─────────────────────────────────────────────────────────────────
for ticker in tickers:
    st.markdown(f"---")
    df_raw = get_data(ticker, tf)
    if df_raw is None:
        st.error(f"❌ Could not load data for **{ticker}**")
        continue

    df = add_indicators(df_raw)
    direction, conf, sigs, rec, price = get_signal(df)

    # Header row
    col1, col2, col3 = st.columns([1, 2, 2])
    with col1:
        color = "🟢" if direction == "BULLISH" else "🔴" if direction == "BEARISH" else "🟡"
        st.metric(label=f"{color} **{ticker}**", value=f"${price:,.2f}")
    with col2:
        st.markdown(f"**Signal:** {direction} — Confidence: `{conf}%`")
        st.markdown(f"**Rec:** {rec}")
    with col3:
        if sigs:
            st.markdown("**Active Signals:**")
            for s in sigs:
                st.markdown(f"- {s}")

    # Chart: 3 sub-plots — Candlestick+indicators | Volume | RSI
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        row_heights=[0.55, 0.2, 0.25],
        vertical_spacing=0.03,
    )

    # Candlesticks
    fig.add_trace(go.Candlestick(
        x=df.index, open=df["Open"], high=df["High"],
        low=df["Low"], close=df["Close"],
        name="Price", increasing_line_color="#00c853", decreasing_line_color="#ff1744"
    ), row=1, col=1)

    # SMA20
    fig.add_trace(go.Scatter(x=df.index, y=df["SMA_20"], name="SMA20",
                             line=dict(color="#ff9800", width=1.5)), row=1, col=1)

    # Bollinger Bands (shaded)
    fig.add_trace(go.Scatter(x=df.index, y=df["BB_up"], name="BB Upper",
                             line=dict(color="rgba(100,180,255,0.5)", width=1), showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["BB_low"], name="BB Lower",
                             fill="tonexty", fillcolor="rgba(100,180,255,0.07)",
                             line=dict(color="rgba(100,180,255,0.5)", width=1), showlegend=False), row=1, col=1)

    # SuperTrend
    st_color = "#00e676" if df["ST_dir"].iloc[-1] == 1 else "#ff1744"
    fig.add_trace(go.Scatter(x=df.index, y=df["SuperTrend"], name="SuperTrend",
                             line=dict(color=st_color, width=2, dash="dot")), row=1, col=1)

    # Volume
    vol_colors = ["#00c853" if c >= o else "#ff1744"
                  for c, o in zip(df["Close"].squeeze(), df["Open"].squeeze())]
    fig.add_trace(go.Bar(x=df.index, y=df["Volume"], name="Volume",
                         marker_color=vol_colors, opacity=0.7), row=2, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["Vol_SMA"], name="Vol SMA",
                             line=dict(color="#ff9800", width=1.5)), row=2, col=1)

    # RSI
    fig.add_trace(go.Scatter(x=df.index, y=df["RSI"], name="RSI",
                             line=dict(color="#7c4dff", width=2)), row=3, col=1)
    fig.add_hline(y=70, row=3, col=1, line=dict(color="red",   dash="dot", width=1))
    fig.add_hline(y=30, row=3, col=1, line=dict(color="green", dash="dot", width=1))
    fig.add_hline(y=50, row=3, col=1, line=dict(color="gray",  dash="dot", width=1))

    fig.update_layout(
        height=750,
        template="plotly_dark",
        title_text=f"{ticker} • {tf} • {direction} {conf}%",
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1),
        margin=dict(l=40, r=40, t=60, b=20),
    )
    fig.update_xaxes(rangeslider_visible=False)
    fig.update_yaxes(title_text="Price",  row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)
    fig.update_yaxes(title_text="RSI",    row=3, col=1, range=[0, 100])

    st.plotly_chart(fig, use_container_width=True)

st.success("✅ QuantStock Bot v3.0 — Running")

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
---
**⚠️ Disclaimer:** This tool is for educational/research purposes only.  
Not financial advice. Always do your own research before trading.
""")
