import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import time
from datetime import datetime

st.set_page_config(page_title="QuantStock", layout="wide", initial_sidebar_state="expanded")

st.markdown("<style>html,body,[class*='css']{font-family:'Courier New',monospace;background:#080c10;color:#c8d6e5;}.main{background:#080c10;}.block-container{padding:1rem 2rem;max-width:100%;}#MainMenu,footer,header{visibility:hidden;}section[data-testid='stSidebar']{background:#0d1117 !important;border-right:1px solid #1e2d3d;}.hdr{display:flex;justify-content:space-between;align-items:center;padding:0.6rem 0 1.2rem;border-bottom:1px solid #1e2d3d;margin-bottom:1.4rem;}.appt{font-size:1.8rem;font-weight:700;color:#00d4ff;letter-spacing:3px;}.apps{font-size:0.68rem;color:#4a6a8a;letter-spacing:2px;margin-top:2px;}.lbadge{display:inline-flex;align-items:center;gap:6px;background:rgba(0,212,100,0.08);border:1px solid rgba(0,212,100,0.3);color:#00d464;font-size:0.72rem;padding:4px 12px;border-radius:3px;letter-spacing:2px;font-weight:600;}.ldot{width:7px;height:7px;background:#00d464;border-radius:50%;animation:pulse 1.4s infinite;display:inline-block;}@keyframes pulse{0%,100%{opacity:1;}50%{opacity:0.3;}}.card{background:#0d1117;border:1px solid #1e2d3d;border-radius:6px;padding:1.2rem 1.4rem;margin-bottom:0.4rem;}.cbull{border-left:3px solid #00d464;}.cbear{border-left:3px solid #ff3b5c;}.cneut{border-left:3px solid #ffa500;}.tkr{font-size:1.4rem;font-weight:700;color:#e8f4fd;letter-spacing:2px;}.prc{font-size:1.5rem;font-weight:700;color:#00d4ff;}.cup{color:#00d464;font-size:0.85rem;}.cdn{color:#ff3b5c;font-size:0.85rem;}.sbadge{display:inline-block;padding:3px 10px;border-radius:3px;font-size:0.72rem;font-weight:700;letter-spacing:1.5px;text-transform:uppercase;}.sbull{background:rgba(0,212,100,0.12);color:#00d464;border:1px solid rgba(0,212,100,0.3);}.sbear{background:rgba(255,59,92,0.12);color:#ff3b5c;border:1px solid rgba(255,59,92,0.3);}.sneut{background:rgba(255,165,0,0.12);color:#ffa500;border:1px solid rgba(255,165,0,0.3);}.rec{background:#111822;border:1px solid #1e2d3d;border-radius:4px;padding:0.5rem 0.9rem;font-size:0.82rem;margin-top:0.6rem;color:#8aadcc;}.cbw{background:#111822;border-radius:3px;height:5px;width:100%;margin-top:6px;}.cbf{height:5px;border-radius:3px;}.stg{display:inline-block;background:#111822;border:1px solid #1e2d3d;color:#6a8aaa;font-size:0.68rem;padding:2px 8px;border-radius:2px;margin:2px 2px 0 0;}.slbl{font-size:0.65rem;color:#4a6a8a;letter-spacing:1px;text-transform:uppercase;margin-bottom:2px;}.sval{font-size:0.9rem;color:#c8d6e5;font-weight:600;}.div{border:none;border-top:1px solid #1e2d3d;margin:0.8rem 0;}.sec{font-size:0.75rem;letter-spacing:3px;text-transform:uppercase;color:#4a6a8a;border-bottom:1px solid #1e2d3d;padding-bottom:6px;margin-bottom:12px;margin-top:20px;}div[data-testid='metric-container']{background:#111822;border:1px solid #1e2d3d;border-radius:4px;padding:0.6rem 0.9rem;}div[data-testid='metric-container'] label{color:#4a6a8a !important;font-size:0.7rem !important;}div[data-testid='metric-container'] div[data-testid='stMetricValue']{color:#00d4ff !important;font-size:1.1rem !important;}.stButton > button{background:#111822 !important;border:1px solid #1e2d3d !important;color:#8aadcc !important;border-radius:4px !important;}</style>", unsafe_allow_html=True)

if "tickers" not in st.session_state:
    st.session_state.tickers = ["AAPL", "TSLA", "NVDA", "AMZN"]
if "last_refresh" not in st.session_state:
    st.session_state.last_refresh = time.time()
if "refresh_count" not in st.session_state:
    st.session_state.refresh_count = 0

with st.sidebar:
    st.markdown('<div class="sec">Add Ticker</div>', unsafe_allow_html=True)
    new_ticker = st.text_input("", placeholder="Symbol e.g. MSFT", label_visibility="collapsed", key="new_t")
    if st.button("+ Add to Watchlist", use_container_width=True):
        sym = new_ticker.strip().upper()
        if sym and sym not in st.session_state.tickers:
            with st.spinner("Verifying..."):
                try:
                    px = yf.Ticker(sym).fast_info.last_price
                    if px and px > 0:
                        st.session_state.tickers.append(sym)
                        st.success(sym + " added")
                        st.rerun()
                    else:
                        st.error("Not found: " + sym)
                except Exception:
                    st.error("Could not verify: " + sym)
        elif sym in st.session_state.tickers:
            st.warning(sym + " already in watchlist")
    st.markdown('<div class="sec">Watchlist</div>', unsafe_allow_html=True)
    to_remove = None
    for _t in st.session_state.tickers:
        _c1, _c2 = st.columns([3, 1])
        with _c1:
            st.markdown("<span style='color:#c8d6e5;font-size:0.85rem'>" + _t + "</span>", unsafe_allow_html=True)
        with _c2:
            if st.button("x", key="rm_" + _t):
                to_remove = _t
    if to_remove:
        st.session_state.tickers.remove(to_remove)
        st.rerun()
    st.markdown('<div class="sec">Settings</div>', unsafe_allow_html=True)
    tf = st.selectbox("Timeframe", ["5m", "15m", "30m", "1h", "1d"], index=3)
    show_bb = st.toggle("Bollinger Bands", value=True)
    show_sma50 = st.toggle("SMA 50", value=False)
    show_vol = st.toggle("Volume Panel", value=True)
    st.markdown('<div class="sec">Auto-Refresh</div>', unsafe_allow_html=True)
    auto_ref = st.toggle("Enable", value=True)
    ref_secs = st.select_slider("Interval", options=[30, 60, 120, 300, 600], value=60, format_func=lambda x: str(x) + "s" if x < 60 else str(x // 60) + "m")
    st.markdown('<div class="div"></div>', unsafe_allow_html=True)
    _b1, _b2 = st.columns(2)
    with _b1:
        if st.button("Refresh", use_container_width=True):
            st.cache_data.clear()
            st.session_state.last_refresh = time.time()
            st.session_state.refresh_count += 1
            st.rerun()
    with _b2:
        if st.button("Clear Cache", use_container_width=True):
            st.cache_data.clear()
            st.rerun()
    st.markdown('<div class="div"></div>', unsafe_allow_html=True)
    _ldt = datetime.fromtimestamp(st.session_state.last_refresh).strftime("%H:%M:%S")
    st.markdown("<div style='font-size:0.65rem;color:#4a6a8a;line-height:2;'>UPDATED: " + _ldt + "<br>REFRESHES: " + str(st.session_state.refresh_count) + "<br>TICKERS: " + str(len(st.session_state.tickers)) + "</div>", unsafe_allow_html=True)

if auto_ref:
    try:
        from streamlit_autorefresh import st_autorefresh
        st_autorefresh(interval=ref_secs * 1000, limit=99999, key="ar")
    except ImportError:
        pass

now_str = datetime.now().strftime("%a %d %b %Y  |  %H:%M:%S")
st.markdown("<div class='hdr'><div><div class='appt'>QuantStock</div><div class='apps'>Real-Time Signal Dashboard  |  " + now_str + "</div></div><div class='lbadge'><span class='ldot'></span> LIVE</div></div>", unsafe_allow_html=True)

if not st.session_state.tickers:
    st.info("Watchlist is empty. Add tickers using the sidebar.")
    st.stop()


@st.cache_data(ttl=30)
def get_data(ticker, interval):
    period = "60d" if interval == "1d" else "10d"
    try:
        df = yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=True, prepost=False)
    except Exception:
        return None
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.loc[:, ~df.columns.duplicated()]
    if df.empty or len(df) < 20:
        return None
    return df


def add_indicators(df):
    df = df.copy()
    c = df["Close"].squeeze()
    h = df["High"].squeeze()
    lo = df["Low"].squeeze()
    df["SMA20"] = c.rolling(20).mean()
    df["SMA50"] = c.rolling(50).mean()
    delta = c.diff()
    ag = delta.clip(lower=0).ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    al = (-delta.clip(upper=0)).ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    df["RSI"] = 100 - (100 / (1 + ag / al.replace(0, np.nan)))
    df["MACD"] = c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean()
    df["MACDS"] = df["MACD"].ewm(span=9, adjust=False).mean()
    prev_c = c.shift(1)
    tr = pd.concat([(h - lo), (h - prev_c).abs(), (lo - prev_c).abs()], axis=1).max(axis=1)
    df["ATR"] = tr.ewm(span=14, adjust=False).mean()
    sma20 = c.rolling(20).mean()
    std20 = c.rolling(20).std()
    df["BBU"] = sma20 + 2 * std20
    df["BBL"] = sma20 - 2 * std20
    df["VOLSMA"] = df["Volume"].rolling(20).mean()
    atr10 = tr.rolling(10).mean()
    hl2 = (h + lo) / 2
    ur = hl2 + 3 * atr10
    lr = hl2 - 3 * atr10
    upper = ur.copy()
    lower = lr.copy()
    stdir = pd.Series(0, index=df.index)
    for i in range(1, len(df)):
        pl = lower.iloc[i - 1]
        pu = upper.iloc[i - 1]
        pc = c.iloc[i - 1]
        lower.iloc[i] = lr.iloc[i] if (lr.iloc[i] > pl or pc < pl) else pl
        upper.iloc[i] = ur.iloc[i] if (ur.iloc[i] < pu or pc > pu) else pu
        if c.iloc[i] > pu:
            stdir.iloc[i] = 1
        elif c.iloc[i] < pl:
            stdir.iloc[i] = -1
        else:
            stdir.iloc[i] = stdir.iloc[i - 1]
    df["STDIR"] = stdir
    df["STLINE"] = np.where(stdir == 1, lower, upper)
    return df


def get_signal(df):
    r = df.iloc[-1]
    p = df.iloc[-2]
    close = float(r["Close"])
    def v(col, d=0.0):
        x = r.get(col, d)
        return float(x) if pd.notna(x) else d
    score = 0.0
    sigs = []
    rsi = v("RSI", 50)
    macd = v("MACD", 0)
    macds = v("MACDS", 0)
    pmacd = float(p.get("MACD", 0) or 0)
    pmacs = float(p.get("MACDS", 0) or 0)
    stdir = v("STDIR", 0)
    bbu = v("BBU", close * 1.05)
    bbl = v("BBL", close * 0.95)
    vol = float(r.get("Volume", 0) or 0)
    vola = float(r.get("VOLSMA", vol) or vol)
    score += 1.5 if close > v("SMA20", close) else -1.0
    score += 1.0 if close > v("SMA50", close) else -0.5
    if rsi < 30:
        score += 3.0
        sigs.append("RSI Oversold")
    elif rsi < 40:
        score += 1.5
        sigs.append("RSI Bullish")
    elif rsi > 70:
        score -= 3.0
        sigs.append("RSI Overbought")
    elif rsi > 60:
        score -= 1.5
        sigs.append("RSI Bearish")
    if macd > macds and pmacd <= pmacs:
        score += 3.0
        sigs.append("MACD Bull Cross")
    elif macd < macds and pmacd >= pmacs:
        score -= 3.0
        sigs.append("MACD Bear Cross")
    elif macd > macds:
        score += 1.0
    else:
        score -= 1.0
    if stdir == 1:
        score += 3.0
        sigs.append("SuperTrend Bull")
    elif stdir == -1:
        score -= 3.0
        sigs.append("SuperTrend Bear")
    if close < bbl:
        score += 2.0
        sigs.append("Below BB Lower")
    elif close > bbu:
        score -= 2.0
        sigs.append("Above BB Upper")
    if vola > 0 and vol > vola * 1.5:
        score += 1.0 if score > 0 else -1.0
        sigs.append("Volume Spike")
    conf = max(20, min(98, int(50 + score * 5.5)))
    direction = "BULLISH" if score >= 3.5 else "BEARISH" if score <= -3.5 else "NEUTRAL"
    atr = v("ATR", float(r["High"]) - float(r["Low"]))
    if direction == "BULLISH" and conf >= 65:
        rec = "BUY  |  TP $" + str(round(close + atr * 3, 2)) + "  |  SL $" + str(round(close - atr * 1.5, 2))
    elif direction == "BEARISH" and conf >= 65:
        rec = "SELL  |  TP $" + str(round(close - atr * 3, 2)) + "  |  SL $" + str(round(close + atr * 1.5, 2))
    else:
        rec = "HOLD  |  Waiting for stronger signal"
    open_p = float(r.get("Open", close) or close)
    chg = close - open_p
    chg_pct = (chg / open_p * 100) if open_p else 0.0
    return direction, conf, sigs[:5], rec, round(close, 2), chg, chg_pct


def safe_f(val, d=0.0):
    try:
        x = float(val)
        return x if not np.isnan(x) else d
    except Exception:
        return d


def build_card(ticker, direction, conf, sigs, rec, price, chg, chg_pct, rsi_v, macd_v, atr_v, vol_r):
    cc = "cbull" if direction == "BULLISH" else "cbear" if direction == "BEARISH" else "cneut"
    sc = "sbull" if direction == "BULLISH" else "sbear" if direction == "BEARISH" else "sneut"
    sl = "BULL" if direction == "BULLISH" else "BEAR" if direction == "BEARISH" else "NEUT"
    fc = "#00d464" if direction == "BULLISH" else "#ff3b5c" if direction == "BEARISH" else "#ffa500"
    dc = "cup" if chg >= 0 else "cdn"
    ds = "+" if chg >= 0 else ""
    rc = "#00d464" if rsi_v < 40 else "#ff3b5c" if rsi_v > 60 else "#c8d6e5"
    mc = "#00d464" if macd_v > 0 else "#ff3b5c"
    vc = "#ffa500" if vol_r > 1.5 else "#c8d6e5"
    ps = "${:,.2f}".format(price)
    cs = "{}{:.2f} ({}{:.2f}%)".format(ds, chg, ds, chg_pct)
    ns = str(conf) + "%"
    tags = "".join(["<span class='stg'>" + s + "</span>" for s in sigs])
    h = "<div class='card " + cc + "'>"
    h += "<div style='display:flex;justify-content:space-between;align-items:flex-start;flex-wrap:wrap;gap:1rem;'>"
    h += "<div>"
    h += "<div style='display:flex;align-items:center;gap:12px;margin-bottom:4px;'>"
    h += "<span class='tkr'>" + ticker + "</span>"
    h += "<span class='sbadge " + sc + "'>" + sl + " " + direction + "</span>"
    h += "</div>"
    h += "<div style='display:flex;align-items:baseline;gap:10px;'>"
    h += "<span class='prc'>" + ps + "</span>"
    h += "<span class='" + dc + "'>" + cs + "</span>"
    h += "</div>"
    h += "<div style='margin-top:8px;'>"
    h += "<div style='font-size:0.65rem;color:#4a6a8a;margin-bottom:3px;'>CONFIDENCE " + ns + "</div>"
    h += "<div class='cbw'><div class='cbf' style='width:" + ns + ";background:" + fc + ";'></div></div>"
    h += "</div></div>"
    h += "<div style='display:grid;grid-template-columns:1fr 1fr 1fr 1fr;gap:0.8rem;min-width:280px;'>"
    h += "<div><div class='slbl'>RSI</div><div class='sval' style='color:" + rc + "'>" + "{:.1f}".format(rsi_v) + "</div></div>"
    h += "<div><div class='slbl'>MACD</div><div class='sval' style='color:" + mc + "'>" + "{:+.3f}".format(macd_v) + "</div></div>"
    h += "<div><div class='slbl'>ATR</div><div class='sval'>" + "{:.2f}".format(atr_v) + "</div></div>"
    h += "<div><div class='slbl'>VOL</div><div class='sval' style='color:" + vc + "'>" + "{:.1f}x".format(vol_r) + "</div></div>"
    h += "</div></div>"
    h += "<div class='rec'>" + rec + "</div>"
    h += "<div style='margin-top:8px;'>" + tags + "</div>"
    h += "</div>"
    return h


summary = {}
mcols = st.columns(len(st.session_state.tickers))
for idx, ticker in enumerate(st.session_state.tickers):
    df_raw = get_data(ticker, tf)
    if df_raw is None:
        with mcols[idx]:
            st.metric(ticker, "N/A")
        continue
    df_ind = add_indicators(df_raw)
    result = get_signal(df_ind)
    summary[ticker] = (df_ind,) + result
    direction, conf, sigs, rec, price, chg, chg_pct = result
    sgn = "+" if chg >= 0 else ""
    with mcols[idx]:
        st.metric(ticker, "${:,.2f}".format(price), "{}{:.2f}%".format(sgn, chg_pct))

st.markdown("<div class='div'></div>", unsafe_allow_html=True)

for ticker in st.session_state.tickers:
    if ticker not in summary:
        st.warning(ticker + ": no data available for this timeframe")
        continue
    df, direction, conf, sigs, rec, price, chg, chg_pct = summary[ticker]
    rsi_v = safe_f(df["RSI"].iloc[-1])
    macd_v = safe_f(df["MACD"].iloc[-1])
    atr_v = safe_f(df["ATR"].iloc[-1])
    vol_v = safe_f(df["Volume"].iloc[-1])
    vol_a = safe_f(df["VOLSMA"].iloc[-1], vol_v)
    vol_r = (vol_v / vol_a) if vol_a > 0 else 1.0
    st.markdown(build_card(ticker, direction, conf, sigs, rec, price, chg, chg_pct, rsi_v, macd_v, atr_v, vol_r), unsafe_allow_html=True)
    rows = 3 if show_vol else 2
    heights = [0.55, 0.22, 0.23] if show_vol else [0.65, 0.35]
    fig = make_subplots(rows=rows, cols=1, shared_xaxes=True, row_heights=heights, vertical_spacing=0.02)
    fig.add_trace(go.Candlestick(x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"], name="Price", increasing=dict(line=dict(color="#00d464", width=1), fillcolor="#00d46422"), decreasing=dict(line=dict(color="#ff3b5c", width=1), fillcolor="#ff3b5c22")), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["SMA20"], name="SMA20", line=dict(color="#ff9800", width=1.5, dash="dot")), row=1, col=1)
    if show_sma50:
        fig.add_trace(go.Scatter(x=df.index, y=df["SMA50"], name="SMA50", line=dict(color="#ab47bc", width=1.5, dash="dot")), row=1, col=1)
    if show_bb:
        fig.add_trace(go.Scatter(x=df.index, y=df["BBU"], name="BB Upper", line=dict(color="rgba(100,180,255,0.4)", width=1), showlegend=False), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df["BBL"], name="BB Lower", fill="tonexty", fillcolor="rgba(100,180,255,0.05)", line=dict(color="rgba(100,180,255,0.4)", width=1), showlegend=False), row=1, col=1)
    stc = "#00d464" if df["STDIR"].iloc[-1] == 1 else "#ff3b5c"
    fig.add_trace(go.Scatter(x=df.index, y=df["STLINE"], name="SuperTrend", line=dict(color=stc, width=2)), row=1, col=1)
    rsi_row = 2
    if show_vol:
        cl = df["Close"].squeeze().tolist()
        op = df["Open"].squeeze().tolist()
        vc = ["#00d46455" if cl[i] >= op[i] else "#ff3b5c55" for i in range(len(cl))]
        fig.add_trace(go.Bar(x=df.index, y=df["Volume"], name="Volume", marker_color=vc, showlegend=False), row=2, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df["VOLSMA"], name="Vol SMA", line=dict(color="#ff9800", width=1), showlegend=False), row=2, col=1)
        rsi_row = 3
    fig.add_trace(go.Scatter(x=df.index, y=df["RSI"], name="RSI", line=dict(color="#7c4dff", width=1.5), fill="tozeroy", fillcolor="rgba(124,77,255,0.06)"), row=rsi_row, col=1)
    fig.add_hline(y=70, row=rsi_row, col=1, line=dict(color="#ff3b5c", dash="dot", width=1))
    fig.add_hline(y=30, row=rsi_row, col=1, line=dict(color="#00d464", dash="dot", width=1))
    fig.add_hrect(y0=30, y1=70, row=rsi_row, col=1, fillcolor="rgba(255,255,255,0.015)", line_width=0)
    fig.update_layout(height=680, template="plotly_dark", paper_bgcolor="#0d1117", plot_bgcolor="#080c10", font=dict(family="Courier New, monospace", size=11, color="#6a8aaa"), legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1, bgcolor="rgba(0,0,0,0)", font=dict(size=10)), margin=dict(l=10, r=10, t=10, b=10), xaxis_rangeslider_visible=False)
    fig.update_xaxes(showgrid=True, gridcolor="#1e2d3d", showline=True, linecolor="#1e2d3d", tickfont=dict(size=10))
    fig.update_yaxes(showgrid=True, gridcolor="#1e2d3d", showline=True, linecolor="#1e2d3d", tickfont=dict(size=10))
    fig.update_yaxes(title_text="Price", row=1, col=1, title_font=dict(size=10))
    if show_vol:
        fig.update_yaxes(title_text="Vol", row=2, col=1, title_font=dict(size=10))
    fig.update_yaxes(title_text="RSI", row=rsi_row, col=1, title_font=dict(size=10), range=[0, 100])
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)

st.markdown("<div style='text-align:center;padding:2rem 0 1rem;font-size:0.65rem;color:#2e4a6a;letter-spacing:1.5px;text-transform:uppercase;'>QuantStock  |  Educational Use Only  |  Not Financial Advice<br>Data via Yahoo Finance  |  " + now_str + "</div>", unsafe_allow_html=True)
    
