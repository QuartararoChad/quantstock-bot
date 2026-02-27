import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime
from streamlit_autorefresh import st_autorefresh
import pandas_ta as ta

st.set_page_config(page_title="QuantStock Bot v2.0", layout="wide", page_icon="🚀")
st.title("🚀 QuantStock Bot v2.0")
st.caption("Advanced Quant Edition • 12+ pro indicators • pandas_ta • Auto 5-min updates • Multi-API cross-check")

# Sidebar
with st.sidebar:
    st.header("⚙️ Controls")
    tickers_input = st.text_input("Stocks (comma-separated)", "AAPL,TSLA,NVDA,AMZN")
    tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]
    tf = st.selectbox("Timeframe", ["5m", "15m", "30m", "1h", "1d"], index=3)
    refresh_mins = st.slider("Auto-refresh every (minutes)", 1, 15, 5)
    poly_key = st.text_input("Polygon.io Key (optional – extra accuracy)", type="password")
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
    if df.empty or len(df) < 50:
        st.error(f"Insufficient data for {ticker}")
        return None
    return df

def add_advanced_indicators(df):
    df = df.copy()
    df.ta.rsi(append=True)
    df.ta.macd(append=True)
    df.ta.bbands(append=True)
    df.ta.atr(append=True)
    df.ta.supertrend(length=10, multiplier=3, append=True)
    df.ta.adx(append=True)
    df.ta.stoch(append=True)
    df.ta.cci(append=True)
    df.ta.obv(append=True)
    df.ta.mfi(append=True)
    df.ta.vwap(append=True)
    df.ta.keltner(append=True)
    df.ta.sma(length=20, append=True)
    df.ta.sma(length=50, append=True)
    return df

def get_quant_signal(df):
    latest = df.iloc[-1]
    prev = df.iloc[-2] if len(df) > 1 else latest
    
    score = 0.0
    signals_list = []
    
    if latest.get('SUPERTd_10_3.0', 0) == 1:
        score += 3.0
        signals_list.append("SuperTrend BULLISH")
    elif latest.get('SUPERTd_10_3.0', 0) == -1:
        score -= 3.0
        signals_list.append("SuperTrend BEARISH")
    
    if latest.get('ADX_14', 0) > 25:
        if latest.get('DMP_14', 0) > latest.get('DMN_14', 0):
            score += 2.0
            signals_list.append("Strong Uptrend (ADX+DMI)")
        else:
            score -= 2.0
            signals_list.append("Strong Downtrend (ADX+DMI)")
    
    if latest.get('STOCHk_14_3_3', 50) < 20:
        score += 1.8
        signals_list.append("Stochastic Oversold")
    elif latest.get('STOCHk_14_3_3', 50) > 80:
        score -= 1.8
        signals_list.append("Stochastic Overbought")
    
    if latest.get('CCI_14_0.015', 0) < -100:
        score += 1.5
        signals_list.append("CCI Oversold")
    elif latest.get('CCI_14_0.015', 0) > 100:
        score -= 1.5
    
    if latest.get('MFI_14', 50) < 25:
        score += 1.5
        signals_list.append("MFI Oversold")
    
    if len(df) > 5 and df['OBV'].iloc[-1] > df['OBV'].iloc[-5]:
        score += 0.8 if latest['Close'] > latest['Open'] else -0.8
    
    if 'VWAP' in df.columns and latest['Close'] > latest.get('VWAP', latest['Close']):
        score += 1.0
    
    if latest.get('RSI_14', 50) < 35: score += 1.2
    if latest.get('MACD_12_26_9', 0) > latest.get('MACDs_12_26_9', 0) and latest.get('MACD_12_26_9', 0) > prev.get('MACD_12_26_9', 0):
        score += 1.8
        signals_list.append("MACD Bullish Cross")
    
    confidence = max(15, min(99, int(48 + score * 7)))
    direction = "BULLISH" if score > 2 else "BEARISH" if score < -2 else "NEUTRAL"
    
    if direction == "BULLISH" and confidence > 70:
        tp = round(latest['Close'] + latest.get('ATR_14', 1) * 4, 2)
        sl = round(latest['Close'] - latest.get('ATR_14', 1) * 2, 2)
        rec = f"🚀 STRONG BUY • TP ${tp} • SL ${sl} (SuperTrend trailing)"
    elif direction == "BEARISH" and confidence > 70:
        rec = "🔻 CONSIDER SHORT / SELL"
    else:
        rec = "⏳ HOLD – Waiting for higher confluence"
    
    return direction, confidence, signals_list[:6], rec, round(latest['Close'], 2)

# Dashboard
cols = st.columns(len(tickers))
for idx, ticker in enumerate(tickers):
    with cols[idx]:
        df = get_data(ticker, tf)
        if df is None: continue
            
        df = add_advanced_indicators(df)
        direction, conf, sigs, rec, price = get_quant_signal(df)
        
        poly_price = None
        if poly_key:
            try:
                import requests
                r = requests.get(f"https://api.polygon.io/v2/last/trade/{ticker}?apiKey={poly_key}", timeout=5)
                if r.status_code == 200:
                    poly_price = round(r.json()['results']['p'], 2)
            except:
                pass
        
        st.metric(f"**{ticker}**", f"${price:.2f}", f"{direction} {conf}%")
        st.write(f"**Recommendation:** {rec}")
        if poly_price:
            st.caption(f"Polygon cross-check: ${poly_price}")
        if sigs:
            st.caption("Signals: " + " • ".join(sigs))
        
        fig = make_subplots(rows=4, cols=1, shared_xaxes=True,
                            row_heights=[0.55, 0.12, 0.15, 0.18],
                            vertical_spacing=0.03,
                            subplot_titles=("Price + SuperTrend + VWAP", "Volume + OBV", "ADX Trend Strength", "Stochastic + RSI"))
        
        fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name="OHLC"), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['SMA_20'], name="SMA20", line=dict(color="#ff9900")), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['SMA_50'], name="SMA50", line=dict(color="#1f77b4")), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['SUPERT_10_3.0'], name="SuperTrend", line=dict(color="#00ff88" if df['SUPERTd_10_3.0'].iloc[-1] == 1 else "#ff4444", width=3)), row=1, col=1)
        if 'VWAP' in df.columns:
            fig.add_trace(go.Scatter(x=df.index, y=df['VWAP'], name="VWAP", line=dict(color="#ff00ff", dash="dash")), row=1, col=1)
        
        fig.add_trace(go.Bar(x=df.index, y=df['Volume'], name="Volume", marker_color="rgba(100,180,255,0.6)"), row=2, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['OBV'], name="OBV", line=dict(color="#ffcc00")), row=2, col=1)
        
        fig.add_trace(go.Scatter(x=df.index, y=df['ADX_14'], name="ADX", line=dict(color="#ff00ff")), row=3, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['DMP_14'], name="+DI", line=dict(color="#00ff88")), row=3, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['DMN_14'], name="-DI", line=dict(color="#ff4444")), row=3, col=1)
        fig.add_hline(y=25, line_dash="dot", row=3, col=1)
        
        fig.add_trace(go.Scatter(x=df.index, y=df['STOCHk_14_3_3'], name="Stoch %K", line=dict(color="#00ffff")), row=4, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['STOCHd_14_3_3'], name="Stoch %D", line=dict(color="#ff9900")), row=4, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['RSI_14'], name="RSI", line=dict(color="#cc00ff")), row=4, col=1)
        fig.add_hline(y=80, line_dash="dot", row=4, col=1)
        fig.add_hline(y=20, line_dash="dot", row=4, col=1)
        
        fig.update_layout(height=950, template="plotly_dark", showlegend=True, title_text=f"{ticker} • {tf} • Advanced Quant Analysis")
        fig.update_xaxes(rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)

st.success("✅ QuantStock Bot v2.0 LIVE – 12 advanced quant indicators running with maximum confluence")