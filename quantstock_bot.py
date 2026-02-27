import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import time
from datetime import datetime

# ─────────────────────────────────────────────
#  PAGE CONFIG  (must be first Streamlit call)
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="QuantStock",
    layout="wide",
    page_icon="📈",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
#  CUSTOM CSS  –  terminal / trading-terminal aesthetic
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;600;700&family=Rajdhani:wght@400;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'JetBrains Mono', monospace;
    background-color: #080c10;
    color: #c8d6e5;
}
.main { background: #080c10; }
.block-container { padding: 1rem 2rem 2rem 2rem; max-width: 100%; }
#MainMenu, footer, header { visibility: hidden; }
.stDeployButton { display: none; }

section[data-testid="stSidebar"] {
    background: #0d1117 !important;
    border-right: 1px solid #1e2d3d;
}
section[data-testid="stSidebar"] * { font-family: 'JetBrains Mono', monospace !important; }

.app-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0.6rem 0 1.2rem 0;
    border-bottom: 1px solid #1e2d3d;
    margin-bottom: 1.4rem;
}
.app-title {
    font-family: 'Rajdhani', sans-serif;
    font-size: 2rem;
    font-weight: 700;
    color: #00d4ff;
    letter-spacing: 3px;
    text-transform: uppercase;
}
.app-subtitle {
    font-size: 0.68rem;
    color: #4a6a8a;
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-top: 2px;
}
.live-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(0,212,100,0.08);
    border: 1px solid rgba(0,212,100,0.3);
    color: #00d464;
    font-size: 0.72rem;
    padding: 4px 12px;
    border-radius: 3px;
    letter-spacing: 2px;
    font-weight: 600;
}
.live-dot {
    width: 7px; height: 7px;
    background: #00d464;
    border-radius: 50%;
    animation: pulse 1.4s ease-in-out infinite;
    display: inline-block;
}
@keyframes pulse {
    0%,100% { opacity:1; transform:scale(1); }
    50%      { opacity:0.3; transform:scale(0.7); }
}

.ticker-card {
    background: #0d1117;
    border: 1px solid #1e2d3d;
    border-radius: 6px;
    padding: 1.2rem 1.4rem;
    margin-bottom: 0.4rem;
}
.ticker-card.bullish { border-left: 3px solid #00d464; }
.ticker-card.bearish { border-left: 3px solid #ff3b5c; }
.ticker-card.neutral { border-left: 3px solid #ffa500; }
.ticker-name {
    font-family: 'Rajdhani', sans-serif;
    font-size: 1.5rem; font-weight: 700;
    color: #e8f4fd; letter-spacing: 2px;
}
.ticker-price { font-size: 1.6rem; font-weight: 700; color: #00d4ff; }
.chg-pos { color: #00d464; font-size: 0.85rem; }
.chg-neg { color: #ff3b5c; font-size: 0.85rem; }
.signal-badge {
    display: inline-block;
    padding: 3px 10px; border-radius: 3px;
    font-size: 0.72rem; font-weight: 700;
    letter-spacing: 1.5px; text-transform: uppercase;
}
.sb-bull { background:rgba(0,212,100,0.12); color:#00d464; border:1px solid rgba(0,212,100,0.3); }
.sb-bear { background:rgba(255,59,92,0.12);  color:#ff3b5c; border:1px solid rgba(255,59,92,0.3); }
.sb-neut { background:rgba(255,165,0,0.12);  color:#ffa500; border:1px solid rgba(255,165,0,0.3); }
.rec-box {
    background: #111822; border: 1px solid #1e2d3d;
    border-radius: 4px; padding: 0.5rem 0.9rem;
    font-size: 0.82rem; margin-top: 0.6rem; color: #8aadcc;
}
.conf-wrap { background:#111822; border-radius:3px; height:5px; width:100%; margin-top:6px; }
.conf-fill  { height:5px; border-radius:3px; }
.sig-tag {
    display: inline-block;
    background: #111822; border: 1px solid #1e2d3d;
    color: #6a8aaa; font-size: 0.68rem;
    padding: 2px 8px; border-radius: 2px;
    margin: 2px 2px 0 0; letter-spacing: 0.5px;
}
.stat-label { font-size:0.65rem; color:#4a6a8a; letter-spacing:1px; text-transform:uppercase; margin-bottom:2px; }
.stat-value { font-size:0.9rem; color:#c8d6e5; font-weight:600; }
.divider    { border:none; border-top:1px solid #1e2d3d; margin:0.8rem 0; }
.sec-hdr    {
    font-family:'Rajdhani',sans-serif; font-size:0.75rem;
    letter-spacing:3px; text-transform:uppercase; color:#4a6a8a;
    border-bottom:1px solid #1e2d3d; padding-bottom:6px;
    margin-bottom:12px; margin-top:20px;
}

div[data-testid="stTextInput"] input {
    background:#111822 !important; border:1px solid #1e2d3d !important;
    color:#c8d6e5 !important; border-radius:4px !important;
    font-family:'JetBrains Mono',monospace !important;
}
div[data-testid="stTextInput"] input:focus {
    border-color:#00d4ff !important;
    box-shadow:0 0 0 2px rgba(0,212,255,0.12) !important;
}
.stButton > button {
    background:#111822 !important; border:1px solid #1e2d3d !important;
    color:#8aadcc !important; border-radius:4px !important;
    font-family:'JetBrains Mono',monospace !important; font-size:0.78rem !important;
    transition:all 0.2s !important;
}
.stButton > button:hover {
    border-color:#00d4ff !important; color:#00d4ff !important;
    background:rgba(0,212,255,0.06) !important;
}
div[data-testid="stSelectbox"] > div > div {
    background:#111822 !important; border:1px solid #1e2d3d !important;
    color:#c8d6e5 !important; border-radius:4px !important;
}
div[data-testid="metric-container"] {
    background:#111822; border:1px solid #1e2d3d; border-radius:4px; padding:0.6rem 0.9rem;
}
div[data-testid="metric-container"] label { color:#4a6a8a !important; font-size:0.7rem !important; }
div[data-testid="metric-container"] div[data-testid="stMetricValue"] { color:#00d4ff !important; font-size:1.1rem !important; }
::-webkit-scrollbar { width:5px; height:5px; }
::-webkit-scrollbar-track { background:#080c10; }
::-webkit-scrollbar-thumb { background:#1e2d3d; border-radius:3px; }
button[data-baseweb="tab"] { font-family:'JetBrains Mono',monospace !important; font-size:0.75rem !important; color:#4a6a8a !important; }
button[data-baseweb="tab"][aria-selected="true"] { color:#00d4ff !important; border-bottom:2px solid #00d4ff !important; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  SESSION STATE
# ─────────────────────────────────────────────
if "tickers"       not in st.session_state: st.session_state.tickers       = ["AAPL","TSLA","NVDA","AMZN"]
if "last_refresh"  not in st.session_state: st.session_state.last_refresh  = time.time()
if "refresh_count" not in st.session_state: st.session_state.refresh_count = 0

# ─────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sec-hdr">Add Ticker</div>', unsafe_allow_html=True)
    new_ticker = st.text_input("", placeholder="Search symbol  e.g. MSFT", label_visibility="collapsed", key="new_t")
    if st.button("＋  Add to Watchlist", use_container_width=True):
        symbol = new_ticker.strip().upper()
        if symbol and symbol not in st.session_state.tickers:
            with st.spinner(f"Verifying {symbol}..."):
                try:
                    info = yf.Ticker(symbol).fast_info
                    if info.last_price and info.last_price > 0:
                        st.session_state.tickers.append(symbol)
                        st.success(f"✓ {symbol} added")
                        st.rerun()
                    else:
                        st.error(f"Symbol not found: {symbol}")
                except Exception:
                    st.error(f"Could not verify {symbol}")
        elif symbol in st.session_state.tickers:
            st.warning(f"{symbol} already in watchlist")

    st.markdown('<div class="sec-hdr">Watchlist</div>', unsafe_allow_html=True)
    to_remove = None
    for t in st.session_state.tickers:
        c1, c2 = st.columns([3, 1])
        with c1: st.markdown(f"<span style='color:#c8d6e5;font-size:0.85rem;line-height:2.2;display:block'>{t}</span>", unsafe_allow_html=True)
        with c2:
            if st.button("✕", key=f"rm_{t}"): to_remove = t
    if to_remove:
        st.session_state.tickers.remove(to_remove)
        st.rerun()

    st.markdown('<div class="sec-hdr">Chart Settings</div>', unsafe_allow_html=True)
    tf          = st.selectbox("Timeframe", ["5m","15m","30m","1h","1d"], index=3)
    show_bb     = st.toggle("Bollinger Bands", value=True)
    show_sma50  = st.toggle("SMA 50",          value=False)
    show_volume = st.toggle("Volume Panel",    value=True)

    st.markdown('<div class="sec-hdr">Auto-Refresh</div>', unsafe_allow_html=True)
    auto_refresh = st.toggle("Enable", value=True)
    refresh_secs = st.select_slider("Interval", options=[30,60,120,300,600], value=60,
                                    format_func=lambda x: f"{x}s" if x<60 else f"{x//60}m")

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    rc1, rc2 = st.columns(2)
    with rc1:
        if st.button("⟳ Refresh", use_container_width=True):
            st.cache_data.clear()
            st.session_state.last_refresh  = time.time()
            st.session_state.refresh_count += 1
            st.rerun()
    with rc2:
        if st.button("⊘ Cache", use_container_width=True):
            st.cache_data.clear(); st.rerun()

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    last_dt = datetime.fromtimestamp(st.session_state.last_refresh).strftime("%H:%M:%S")
    st.markdown(f"""
    <div style="font-size:0.65rem;color:#4a6a8a;line-height:2;">
        LAST UPDATE &nbsp; {last_dt}<br>
        REFRESH #&nbsp;&nbsp;&nbsp;&nbsp; {st.session_state.refresh_count}<br>
        TICKERS &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; {len(st.session_state.tickers)}
    </div>""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  AUTO-REFRESH
# ─────────────────────────────────────────────
if auto_refresh:
    try:
        from streamlit_autorefresh import st_autorefresh
        st_autorefresh(interval=refresh_secs * 1000, limit=99999, key="ar")
    except ImportError:
        st.markdown(f"<script>setTimeout(()=>window.location.reload(),{refresh_secs*1000})</script>",
                    unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  APP HEADER
# ─────────────────────────────────────────────
now_str = datetime.now().strftime("%a %d %b %Y  •  %H:%M:%S")
st.markdown(f"""
<div class="app-header">
    <div>
        <div class="app-title">QuantStock</div>
        <div class="app-subtitle">Real-Time Signal Dashboard  •  {now_str}</div>
    </div>
    <div class="live-badge"><span class="live-dot"></span> LIVE</div>
</div>""", unsafe_allow_html=True)

if not st.session_state.tickers:
    st.markdown("""<div style="text-align:center;padding:4rem;color:#4a6a8a;">
        <div style="font-size:2rem;margin-bottom:1rem;">📭</div>
        <div style="font-family:'Rajdhani',sans-serif;font-size:1.2rem;letter-spacing:2px;">WATCHLIST IS EMPTY</div>
        <div style="font-size:0.78rem;margin-top:0.5rem;">Use the sidebar to add tickers</div>
    </div>""", unsafe_allow_html=True)
    st.stop()

# ─────────────────────────────────────────────
#  DATA + INDICATORS
# ─────────────────────────────────────────────
@st.cache_data(ttl=30)
def get_data(ticker: str, interval: str):
    period = "60d" if interval == "1d" else "10d"
    try:
        df = yf.download(ticker, period=period, interval=interval,
                         progress=False, auto_adjust=True, prepost=False)
    except Exception:
        return None
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.loc[:, ~df.columns.duplicated()]
    return None if (df.empty or len(df) < 20) else df


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    c = df["Close"].squeeze(); h = df["High"].squeeze(); l = df["Low"].squeeze()

    df["SMA_20"] = c.rolling(20).mean()
    df["SMA_50"] = c.rolling(50).mean()

    delta = c.diff(); gain = delta.clip(lower=0); loss = -delta.clip(upper=0)
    ag = gain.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    al = loss.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    df["RSI"] = 100 - (100 / (1 + ag / al.replace(0, np.nan)))

    ema12 = c.ewm(span=12, adjust=False).mean(); ema26 = c.ewm(span=26, adjust=False).mean()
    df["MACD"]   = ema12 - ema26
    df["MACD_S"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_H"] = df["MACD"] - df["MACD_S"]

    prev_c = c.shift(1)
    tr = pd.concat([(h-l),(h-prev_c).abs(),(l-prev_c).abs()],axis=1).max(axis=1)
    df["ATR"] = tr.ewm(span=14, adjust=False).mean()

    sma20 = c.rolling(20).mean(); std20 = c.rolling(20).std()
    df["BB_mid"] = sma20; df["BB_up"] = sma20 + 2*std20; df["BB_lo"] = sma20 - 2*std20
    df["Vol_SMA"] = df["Volume"].rolling(20).mean()

    # SuperTrend
    atr_st = tr.rolling(10).mean(); hl2 = (h+l)/2
    ur_raw = hl2 + 3*atr_st; lr_raw = hl2 - 3*atr_st
    upper = ur_raw.copy(); lower = lr_raw.copy(); st_dir = pd.Series(0, index=df.index)
    for i in range(1, len(df)):
        lower.iloc[i] = lr_raw.iloc[i] if lr_raw.iloc[i]>lower.iloc[i-1] or c.iloc[i-1]<lower.iloc[i-1] else lower.iloc[i-1]
        upper.iloc[i] = ur_raw.iloc[i] if ur_raw.iloc[i]<upper.iloc[i-1] or c.iloc[i-1]>upper.iloc[i-1] else upper.iloc[i-1]
        if   c.iloc[i] > upper.iloc[i-1]: st_dir.iloc[i] =  1
        elif c.iloc[i] < lower.iloc[i-1]: st_dir.iloc[i] = -1
        else: st_dir.iloc[i] = st_dir.iloc[i-1]
    df["ST_dir"]  = st_dir
    df["ST_line"] = np.where(st_dir==1, lower, upper)
    return df


def get_signal(df: pd.DataFrame):
    r = df.iloc[-1]; p = df.iloc[-2]
    c = float(r["Close"])
    def v(col, default=0):
        val = r.get(col, default)
        return float(val) if pd.notna(val) else default

    score = 0.0; sigs = []
    rsi   = v("RSI",50); macd = v("MACD",0); macs = v("MACD_S",0)
    pmacd = float(p.get("MACD",0) or 0); pmacs = float(p.get("MACD_S",0) or 0)
    st_d  = v("ST_dir",0); sma20 = v("SMA_20",c); sma50 = v("SMA_50",c)
    bb_up = v("BB_up",c*1.05); bb_lo = v("BB_lo",c*0.95)
    vol   = float(r.get("Volume",0) or 0); vola = float(r.get("Vol_SMA",vol) or vol)

    if c > sma20: score += 1.5
    else:         score -= 1.0
    if c > sma50: score += 1.0
    else:         score -= 0.5

    if   rsi < 30: score += 3.0; sigs.append("RSI Oversold")
    elif rsi < 40: score += 1.5; sigs.append("RSI Bullish Zone")
    elif rsi > 70: score -= 3.0; sigs.append("RSI Overbought")
    elif rsi > 60: score -= 1.5; sigs.append("RSI Bearish Zone")

    if   macd > macs and pmacd <= pmacs: score += 3.0; sigs.append("MACD Bull Cross")
    elif macd < macs and pmacd >= pmacs: score -= 3.0; sigs.append("MACD Bear Cross")
    elif macd > macs: score += 1.0
    else:             score -= 1.0

    if   st_d ==  1: score += 3.0; sigs.append("SuperTrend Bull")
    elif st_d == -1: score -= 3.0; sigs.append("SuperTrend Bear")

    if   c < bb_lo: score += 2.0; sigs.append("BB Oversold")
    elif c > bb_up: score -= 2.0; sigs.append("BB Overbought")

    if vola > 0 and vol > vola * 1.5:
        score += 1.0 if score > 0 else -1.0; sigs.append("Volume Spike")

    conf      = max(20, min(98, int(50 + score * 5.5)))
    direction = "BULLISH" if score >= 3.5 else "BEARISH" if score <= -3.5 else "NEUTRAL"
    atr       = v("ATR", float(r["High"]) - float(r["Low"]))

    if direction == "BULLISH" and conf >= 65:
        tp = round(c + atr*3, 2); sl = round(c - atr*1.5, 2)
        rec = f"🚀 BUY  •  TP ${tp:,.2f}  •  SL ${sl:,.2f}"
    elif direction == "BEARISH" and conf >= 65:
        tp = round(c - atr*3, 2); sl = round(c + atr*1.5, 2)
        rec = f"🔻 SELL  •  TP ${tp:,.2f}  •  SL ${sl:,.2f}"
    else:
        rec = "⏳ HOLD — Waiting for stronger signal"

    open_p  = float(r.get("Open", c) or c)
    chg     = c - open_p
    chg_pct = (chg / open_p * 100) if open_p else 0
    return direction, conf, sigs[:5], rec, round(c,2), chg, chg_pct, round(score,2)


# ─────────────────────────────────────────────
#  SUMMARY METRIC ROW
# ─────────────────────────────────────────────
summary_data = {}
summary_cols = st.columns(len(st.session_state.tickers))
for idx, ticker in enumerate(st.session_state.tickers):
    df_raw = get_data(ticker, tf)
    if df_raw is None:
        with summary_cols[idx]: st.metric(ticker, "N/A")
        continue
    df_ind = add_indicators(df_raw)
    result = get_signal(df_ind)
    summary_data[ticker] = (df_ind,) + result
    direction, conf, sigs, rec, price, chg, chg_pct, score = result
    sign = "+" if chg >= 0 else ""
    with summary_cols[idx]:
        st.metric(ticker, f"${price:,.2f}", f"{sign}{chg_pct:.2f}%")

st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  TICKER CARDS + CHARTS
# ─────────────────────────────────────────────
for ticker in st.session_state.tickers:
    if ticker not in summary_data:
        st.markdown(f'<div class="ticker-card neutral"><span class="ticker-name">{ticker}</span>'
                    f'<span style="color:#ff3b5c;font-size:0.8rem;margin-left:1rem;">⚠ No data</span></div>',
                    unsafe_allow_html=True)
        continue

    df, direction, conf, sigs, rec, price, chg, chg_pct, score = summary_data[ticker]

    card_cls   = "bullish" if direction=="BULLISH" else "bearish" if direction=="BEARISH" else "neutral"
    sig_cls    = "sb-bull" if direction=="BULLISH" else "sb-bear" if direction=="BEARISH" else "sb-neut"
    sig_icon   = "▲" if direction=="BULLISH" else "▼" if direction=="BEARISH" else "◆"
    chg_cls    = "chg-pos" if chg >= 0 else "chg-neg"
    chg_sign   = "+" if chg >= 0 else ""
    conf_color = "#00d464" if direction=="BULLISH" else "#ff3b5c" if direction=="BEARISH" else "#ffa500"
    tags_html  = "".join(f'<span class="sig-tag">{s}</span>' for s in sigs)

    rsi_val   = float(df["RSI"].iloc[-1])  if pd.notna(df["RSI"].iloc[-1])  else 0
    macd_val  = float(df["MACD"].iloc[-1]) if pd.notna(df["MACD"].iloc[-1]) else 0
    atr_val   = float(df["ATR"].iloc[-1])  if pd.notna(df["ATR"].iloc[-1])  else 0
    vol_val   = float(df["Volume"].iloc[-1])
    vol_avg   = float(df["Vol_SMA"].iloc[-1]) if pd.notna(df["Vol_SMA"].iloc[-1]) else vol_val
    vol_ratio = vol_val / vol_avg if vol_avg > 0 else 1.0
    rsi_color = "#00d464" if rsi_val<40 else "#ff3b5c" if rsi_val>60 else "#c8d6e5"
    macd_color= "#00d464" if macd_val>0 else "#ff3b5c"
    vol_color = "#ffa500" if vol_ratio>1.5 else "#c8d6e5"

    st.markdown(f"""
    <div class="ticker-card {card_cls}">
        <div style="display:flex;justify-content:space-between;align-items:flex-start;flex-wrap:wrap;gap:1rem;">
            <div>
                <div style="display:flex;align-items:center;gap:12px;margin-bottom:4px;">
                    <span class="ticker-name">{ticker}</span>
                    <span class="signal-badge {sig_cls}">{sig_icon} {direction}</span>
                </div>
                <div style="display:flex;align-items:baseline;gap:10px;">
                    <span class="ticker-price">${price:,.2f}</span>
                    <span class="{chg_cls}">{chg_sign}{chg:.2f} ({chg_sign}{chg_pct:.2f}%)</span>
                </div>
                <div style="margin-top:8px;">
                    <div style="font-size:0.65rem;color:#4a6a8a;margin-bottom:3px;">CONFIDENCE {conf}%</div>
                    <div class="conf-wrap"><div class="conf-fill" style="width:{conf}%;background:{conf_color};"></div></div>
                </div>
            </div>
            <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:0.8rem;min-width:300px;">
                <div><div class="stat-label">RSI</div><div class="stat-value" style="color:{rsi_color}">{rsi_val:
