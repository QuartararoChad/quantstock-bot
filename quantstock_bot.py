import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from streamlit_autorefresh import st_autorefresh

st.set_page_config(page_title="QuantStock Bot v2.1", layout="wide", page_icon="🚀")
st.title("🚀 QuantStock Bot v2.1")
st.caption("Stable Version • No pandas_ta • Works on Streamlit Cloud • Auto 5-min updates")

# Sidebar
with st.sidebar:
    st.header("⚙️ Controls")
    tickers_input = st.text_input("Stocks (comma-separated)", "AAPL,TSLA,NVDA,AMZN")
    tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]
    tf = st.selectbox("Timeframe", ["5m", "15m", "30m", "1h", "1d"], index=3)
    refresh_mins = st.slider("Auto-refresh every (minutes)", 1, 15, 5)
    poly_key = st.text_input("Polygon.io Key (optional)", type="password")
    if st.button("🔄 Manual Refresh Now"):
        st.rerun()

if not tickers:
    st.error("Enter at least one ticker")
    st.stop()

st_autorefresh(interval=refresh_mins * 60 * 1000, limit=9999, key="quantrefresh")

@st.cache_data(ttl=60)
def get_data(ticker, interval):
    period = "60d" if interval == "1d" else "10d"
    df = yf.download(ticker, period=period, interval=interval, progress=False, prepost=True)
    if df.empty or len(df) < 30:   # ← Lowered from 50 so it works on short timeframes
        st.warning(f"⚠️ Limited data for {ticker} — showing what we have")
        if df.empty:
            return None
    return df

def add_advanced_indicators(df):
    df = df.copy()
    # SMA
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    
    # RSI
    delta = df['Close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    df['RSI_14'] = 100 - (100 / (1 + rs))
    
    # MACD
    ema12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD_12_26_9'] = ema12 - ema26
    df['MACDs_12_26_9'] = df['MACD_12_26_9'].ewm(span=9, adjust=False).mean()
    
    # SuperTrend (NaN-safe)
    atr = df['High'].rolling(10).max() - df['Low'].rolling(10).min()
    hl2 = (df['High'] + df['Low']) / 2
    df['SUPERT_10_3.0'] = hl2 + 3 * atr
    df['SUPERTd_10_3.0'] = np.where(df['Close'] > df['SUPERT_10_3.0'].fillna(df['Close']), 1, -1)
    
    return df

def get_quant_signal(df):
    if len(df) < 5:
        return "NEUTRAL", 50, [], "HOLD - Not enough data", round(df['Close'].iloc[-1], 2)
    
    latest = df.iloc[-1]
    prev = df.iloc[-2]
    
    score = 0.0
    signals = []
    
    if latest['Close'] > latest.get('SMA_20', latest['Close']): 
        score += 2
    if latest.get('RSI_14', 50) < 35:
        score += 2.5
        signals.append("RSI Oversold")
    if latest.get('MACD_12_26_9', 0) > latest.get('MACDs_12_26_9', 0) and latest.get('MACD_12_26_9', 0) > prev.get('MACD_12_26_9', 0):
        score += 2
        signals.append("MACD Bullish Cross")
    if latest.get('SUPERTd_10_3.0', 0) == 1:
        score += 3
        signals.append("SuperTrend BULLISH")
    
    confidence = max(20, min(98, int(50 + score * 7)))
    direction = "BULLISH" if score > 3 else "BEARISH" if score < -2 else "NEUTRAL"
    
    if direction == "BULLISH" and confidence > 70:
        tp = round(latest['Close'] + (latest['High'] - latest['Low']) * 3, 2)
        sl = round(latest['Close'] - (latest['High'] - latest['Low']) * 1.5, 2)
        rec = f"🚀 STRONG BUY • TP ${tp} • SL ${sl}"
    elif direction == "BEARISH" and confidence > 70:
        rec = "🔻 CONSIDER SHORT / SELL"
    else:
        rec = "⏳ HOLD - Waiting for stronger signal"
    
    return direction, confidence, signals[:4], rec, round(latest['Close'], 2)

# Main dashboard
cols = st.columns(len(tickers))
for idx, ticker in enumerate(tickers):
    with cols[idx]:
        df = get_data(ticker, tf)
        if df is None:
            continue
        df = add_advanced_indicators(df)
        direction, conf, sigs, rec, price = get_quant_signal(df)
        
        st.metric(f"**{ticker}**", f"${price}", f"{direction} {conf}%")
        st.write(f"**Recommendation:** {rec}")
        if sigs:
            st.caption("Signals: " + " • ".join(sigs))
        
        # Chart
        fig = make_subplots(rows=3, cols=1, shared_xaxes=True, row_heights=[0.6, 0.2, 0.2])
        fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close']), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['SMA_20'], name="SMA20", line=dict(color="#ff9900")), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['SUPERT_10_3.0'], name="SuperTrend", line=dict(color="#00ff88" if df['SUPERTd_10_3.0'].iloc[-1] == 1 else "#ff4444", width=3)), row=1, col=1)
        fig.add_trace(go.Bar(x=df.index, y=df['Volume'], name="Volume"), row=2, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['RSI_14'], name="RSI"), row=3, col=1)
        fig.add_hline(y=70, row=3, col=1, line_dash="dot")
        fig.add_hline(y=30, row=3, col=1, line_dash="dot")
        
        fig.update_layout(height=800, template="plotly_dark", title_text=f"{ticker} • {tf}")
        fig.update_xaxes(rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)

st.success("✅ QuantStock Bot v2.1 is LIVE and stable!")
