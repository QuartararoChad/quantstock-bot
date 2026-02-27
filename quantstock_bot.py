import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import time
from datetime import datetime

st.set_page_config(page_title="QuantStock", layout="wide", page_icon="chart_with_upwards_trend", initial_sidebar_state="expanded")

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;600;700&family=Rajdhani:wght@400;600;700&display=swap');
html,body,[class*="css"]{font-family:'JetBrains Mono',monospace;background:#080c10;color:#c8d6e5;}
.main{background:#080c10;}
.block-container{padding:1rem 2rem 2rem 2rem;max-width:100%;}
#MainMenu,footer,header{visibility:hidden;}
section[data-testid="stSidebar"]{background:#0d1117 !important;border-right:1px solid #1e2d3d;}
.app-title{font-family:'Rajdhani',sans-serif;font-size:2rem;font-weight:700;color:#00d4ff;letter-spacing:3px;text-transform:uppercase;}
.app-sub{font-size:0.68rem;color:#4a6a8a;letter-spacing:2px;text-transform:uppercase;margin-top:2px;}
.live-badge{display:inline-flex;align-items:center;gap:6px;background:rgba(0,212,100,0.08);border:1px solid rgba(0,212,100,0.3);color:#00d464;font-size:0.72rem;padding:4px 12px;border-radius:3px;letter-spacing:2px;font-weight:600;}
.live-dot{width:7px;height:7px;background:#00d464;border-radius:50%;animation:pulse 1.4s ease-in-out infinite;display:inline-block;}
@keyframes pulse{0%,100%{opacity:1;transform:scale(1);}50%{opacity:0.3;transform:scale(0.7);}}
.hdr{display:flex;justify-content:space-between;align-items:center;padding:0.6rem 0 1.2rem;border-bottom:1px solid #1e2d3d;margin-bottom:1.4rem;}
.card{background:#0d1117;border:1px solid #1e2d3d;border-radius:6px;padding:1.2rem 1.4rem;margin-bottom:0.4rem;}
.card-bull{border-left:3px solid #00d464;}
.card-bear{border-left:3px solid #ff3b5c;}
.card-neut{border-left:3px solid #ffa500;}
.tkr{font-family:'Rajdhani',sans-serif;font-size:1.5rem;font-weight:700;color:#e8f4fd;letter-spacing:2px;}
.prc{font-size:1.6rem;font-weight:700;color:#00d4ff;}
.up{color:#00d464;font-size:0.85rem;}
.dn{color:#ff3b5c;font-size:0.85rem;}
.sbadge{display:inline-block;padding:3px 10px;border-radius:3px;font-size:0.72rem;font-weight:700;letter-spacing:1.5px;text-transform:uppercase;}
.sb-bull{background:rgba(0,212,100,0.12);color:#00d464;border:1px solid rgba(0,212,100,0.3);}
.sb-bear{background:rgba(255,59,92,0.12);color:#ff3b5c;border:1px solid rgba(255,59,92,0.3);}
.sb-neut{background:rgba(255,165,0,0.12);color:#ffa500;border:1px solid rgba(255,165,0,0.3);}
.rec{background:#111822;border:1px solid #1e2d3d;border-radius:4px;padding:0.5rem 0.9rem;font-size:0.82rem;margin-top:0.6rem;color:#8aadcc;}
.cbar-wrap{background:#111822;border-radius:3px;height:5px;width:100%;margin-top:6px;}
.cbar-fill{height:5px;border-radius:3px;}
.stag{display:inline-block;background:#111822;border:1px solid #1e2d3d;color:#6a8aaa;font-size:0.68rem;padding:2px 8px;border-radius:2px;margin:2px 2px 0 0;}
.slbl{font-size:0.65rem;color:#4a6a8a;letter-spacing:1px;text-transform:uppercase;margin-bottom:2px;}
.sval{font-size:0.9rem;color:#c8d6e5;font-weight:600;}
.divider{border:none;border-top:1px solid #1e2d3d;margin:0.8rem 0;}
.sec{font-family:'Rajdhani',sans-serif;font-size:0.75rem;letter-spacing:3px;text-transform:uppercase;color:#4a6a8a;border-bottom:1px solid #1e2d3d;padding-bottom:6px;margin-bottom:12px;margin-top:20px;}
div[data-testid="stTextInput"] input{background:#111822 !important;border:1px solid #1e2d3d !important;color:#c8d6e5 !important;border-radius:4px !important;}
.stButton > button{background:#111822 !important;border:1px solid #1e2d3d !important;color:#8aadcc !important;border-radius:4px !important;font-size:0.78rem !important;}
.stButton > button:hover{border-color:#00d4ff !important;color:#00d4ff !important;background:rgba(0,212,255,0.06) !important;}
div[data-testid="metric-container"]{background:#111822;border:1px solid #1e2d3d;border-radius:4px;padding:0.6rem 0.9rem;}
div[data-testid="metric-container"] label{color:#4a6a8a !important;font-size:0.7rem !important;}
div[data-testid="metric-container"] div[data-testid="stMetricValue"]{color:#00d4ff !important;font-size:1.1rem !important;}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

if "tickers"       not in st.session_state: st.session_state.tickers       = ["AAPL","TSLA","NVDA","AMZN"]
if "last_refresh"  not in st.session_state: st.session_state.last_refresh  = time.time()
if "refresh_count" not in st.session_state: st.session_state.refresh_count = 0

with st.sidebar:
    st.markdown('<div class="sec">Add Ticker</div>', unsafe_allow_html=True)
    new_ticker = st.text_input("", placeholder="Symbol e.g. MSFT", label_visibility="collapsed", key="new_t")
    if st.button("+ Add to Watchlist", use_container_width=True):
        symbol = new_ticker.strip().upper()
        if symbol and symbol not in st.session_state.tickers:
            with st.spinner("Verifying..."):
                try:
                    p = yf.Ticker(symbol).fast_info.last_price
                    if p and p > 0:
                        st.session_state.tickers.append(symbol)
                        st.success(symbol + " added")
                        st.rerun()
                    else:
                        st.error("Symbol not found: " + symbol)
                except Exception:
                    st.error("Could not verify " + symbol)
        elif symbol in st.session_state.tickers:
            st.warning(symbol + " already in watchlist")

    st.markdown('<div class="sec">Watchlist</div>', unsafe_allow_html=True)
    to_remove = None
    for t in st.session_state.tickers:
        c1, c2 = st.columns([3, 1])
        with c1:
            st.markdown("<span style='color:#c8d6e5;font-size:0.85rem;line-height:2.2;display:block'>" + t + "</span>", unsafe_allow_html=True)
        with c2:
            if st.button("x", key="rm_" + t):
                to_remove = t
    if to_remove:
        st.session_state.tickers.remove(to_remove)
        st.rerun()

    st.markdown('<div class="sec">Settings</div>', unsafe_allow_html=True)
    tf         = st.selectbox("Timeframe", ["5m","15m","30m","1h","1d"], index=3)
    show_bb    = st.toggle("Bollinger Bands", value=True)
    show_sma50 = st.toggle("SMA 50", value=False)
    show_vol   = st.toggle("Volume Panel", value=True)

    st.markdown('<div class="sec">Auto-Refresh</div>', unsafe_allow_html=True)
    auto_ref = st.toggle("Enable", value=True)
    ref_secs = st.select_slider(
        "Interval",
        options=[30, 60, 120, 300, 600],
        value=60,
        format_func=lambda x: (str(x) + "s") if x < 60 else (str(x // 60) + "m")
    )

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    b1, b2 = st.columns(2)
    with b1:
        if st.button("Refresh", use_container_width=True):
            st.cache_data.clear()
            st.session_state.last_refresh  = time.time()
            st.session_state.refresh_count += 1
            st.rerun()
    with b2:
        if st.button("Clear Cache", use_container_width=True):
            st.cache_data.clear()
            st.rerun()

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    last_dt = datetime.fromtimestamp(st.session_state.last_refresh).strftime("%H:%M:%S")
    sidebar_info = (
        "<div style='font-size:0.65rem;color:#4a6a8a;line-height:2;'>"
        "LAST UPDATE: " + last_dt + "<br>"
        "REFRESHES: " + str(st.session_state.refresh_count) + "<br>"
        "TICKERS: " + str(len(st.session_state.tickers)) + "</div>"
    )
    st.markdown(sidebar_info, unsafe_allow_html=True)

if auto_ref:
    try:
        from streamlit_autorefresh import st_autorefresh
        st_autorefresh(interval=ref_secs * 1000, limit=99999, key="ar")
    except ImportError:
        pass

now_str = datetime.now().strftime("%a %d %b %Y  |  %H:%M:%S")
header_html = (
    "<div class='hdr'>"
    "<div><div class='app-title'>QuantStock</div>"
    "<div class='app-sub'>Real-Time Signal Dashboard  |  " + now_str + "</div></div>"
    "<div class='live-badge'><span class='live-dot'></span> LIVE</div>"
    "</div>"
)
st.markdown(header_html, unsafe_allow_html=True)

if not st.session_state.tickers:
    st.info("Watchlist is empty. Add tickers in the sidebar.")
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
    gain  = delta.clip(lower=0)
    loss  = -delta.clip(upper=0)
    ag    = gain.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    al    = loss.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    df["RSI"] = 100 - (100 / (1 + ag / al.replace(0, np.nan)))

    ema12      = c.ewm(span=12, adjust=False).mean()
    ema26      = c.ewm(span=26, adjust=False).mean()
    df["MACD"] = ema12 - ema26
    df["MACDS"] = df["MACD"].ewm(span=9, adjust=False).mean()

    prev_c = c.shift(1)
    tr = pd.concat([(h - lo), (h - prev_c).abs(), (lo - prev_c).abs()], axis=1).max(axis=1)
    df["ATR"] = tr.ewm(span=14, adjust=False).mean()

    sma20       = c.rolling(20).mean()
    std20       = c.rolling(20).std()
    df["BBU"]   = sma20 + 2 * std20
    df["BBL"]   = sma20 - 2 * std20
    df["VOLSMA"] = df["Volume"].rolling(20).mean()

    atr10  = tr.rolling(10).mean()
    hl2    = (h + lo) / 2
    ur_raw = hl2 + 3 * atr10
    lr_raw = hl2 - 3 * atr10
    upper  = ur_raw.copy()
    lower  = lr_raw.copy()
    stdir  = pd.Series(0, index=df.index)
    for i in range(1, len(df)):
        prev_lower = lower.iloc[i - 1]
        prev_upper = upper.iloc[i - 1]
        prev_close = c.iloc[i - 1]
        lower.iloc[i] = lr_raw.iloc[i] if (lr_raw.iloc[i] > prev_lower or prev_close < prev_lower) else prev_lower
        upper.iloc[i] = ur_raw.iloc[i] if (ur_raw.iloc[i] < prev_upper or prev_close > prev_upper) else prev_upper
        if c.iloc[i] > prev_upper:
            stdir.iloc[i] = 1
        elif c.iloc[i] < prev_lower:
            stdir.iloc[i] = -1
        else:
            stdir.iloc[i] = stdir.iloc[i - 1]
    df["STDIR"]  = stdir
    df["STLINE"] = np.where(stdir == 1, lower, upper)
    return df


def get_signal(df):
    r     = df.iloc[-1]
    p     = df.iloc[-2]
    close = float(r["Close"])

    def v(col, default=0.0):
        val = r.get(col, default)
        return float(val) if pd.notna(val) else default

    score = 0.0
    sigs  = []
    rsi   = v("RSI",   50)
    macd  = v("MACD",   0)
    macds = v("MACDS",  0)
    pmacd = float(p.get("MACD",  0) or 0)
    pmacs = float(p.get("MACDS", 0) or 0)
    stdir = v("STDIR",  0)
    sma20 = v("SMA20", close)
    sma50 = v("SMA50", close)
    bbu   = v("BBU",   close * 1.05)
    bbl   = v("BBL",   close * 0.95)
    vol   = float(r.get("Volume", 0) or 0)
    vola  = float(r.get("VOLSMA", vol) or vol)

    if close > sma20: score += 1.5
    else:             score -= 1.0
    if close > sma50: score += 1.0
    else:             score -= 0.5

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

    conf      = max(20, min(98, int(50 + score * 5.5)))
    direction = "BULLISH" if score >= 3.5 else "BEARISH" if score <= -3.5 else "NEUTRAL"
    atr       = v("ATR", float(r["High"]) - float(r["Low"]))

    if direction == "BULLISH" and conf >= 65:
        tp  = round(close + atr * 3,   2)
        sl  = round(close - atr * 1.5, 2)
        rec = "BUY  |  TP $" + str(tp) + "  |  SL $" + str(sl)
    elif direction == "BEARISH" and conf >= 65:
        tp  = round(close - atr * 3,   2)
        sl  = round(close + atr * 1.5, 2)
        rec = "SELL  |  TP $" + str(tp) + "  |  SL $" + str(sl)
    else:
        rec = "HOLD  |  Waiting for stronger signal"

    open_p  = float(r.get("Open", close) or close)
    chg     = close - open_p
    chg_pct = (chg / open_p * 100) if open_p else 0.0
    return direction, conf, sigs[:5], rec, round(close, 2), chg, chg_pct


def build_card(ticker, direction, conf, sigs, rec, price, chg, chg_pct, rsi_val, macd_val, atr_val, vol_ratio):
    card_cls = "card-bull" if direction == "BULLISH" else "card-bear" if direction == "BEARISH" else "card-neut"
    sig_cls  = "sb-bull"   if direction == "BULLISH" else "sb-bear"   if direction == "BEARISH" else "sb-neut"
    sig_lbl  = "BULL"      if direction == "BULLISH" else "BEAR"      if direction == "BEARISH" else "NEUT"
    chg_cls  = "up" if chg >= 0 else "dn"
    chg_sign = "+" if chg >= 0 else ""
    conf_col = "#00d464" if direction == "BULLISH" else "#ff3b5c" if direction == "BEARISH" else "#ffa500"
    rsi_col  = "#00d464" if rsi_val < 40 else "#ff3b5c" if rsi_val > 60 else "#c8d6e5"
    macd_col = "#00d464" if macd_val > 0 else "#ff3b5c"
    vol_col  = "#ffa500" if vol_ratio > 1.5 else "#c8d6e5"

    price_s  = "${:,.2f}".format(price)
    chg_s    = "{}{:.2f} ({}{:.2f}%)".format(chg_sign, chg, chg_sign, chg_pct)
    conf_s   = str(conf) + "%"
    rsi_s    = "{:.1f}".format(rsi_val)
    macd_s   = "{:+.3f}".format(macd_val)
    atr_s    = "{:.2f}".format(atr_val)
    vol_s    = "{:.1f}x".format(vol_ratio)
    tags     = "".join(['<span class="stag">' + s + '</span>' for s in sigs])

    html  = '<div class="card ' + card_cls + '">'
    html += '<div style="display:flex;justify-content:space-between;align-items:flex-start;flex-wrap:wrap;gap:1rem;">'
    html += '<div>'
    html += '<div style="display:flex;align-items:center;gap:12px;margin-bottom:4px;">'
    html += '<span class="tkr">' + ticker + '</span>'
    html += '<span class="sbadge ' + sig_cls + '">' + sig_lbl + ' ' + direction + '</span>'
    html += '</div>'
    html += '<div style="display:flex;align-items:baseline;gap:10px;">'
    html += '<span class="prc">' + price_s + '</span>'
    html += '<span class="' + chg_cls + '">' + chg_s + '</span>'
    html += '</div>'
    html += '<div style="margin-top:8px;">'
    html += '<div style="font-size:0.65rem;color:#4a6a8a;margin-bottom:3px;">CONFIDENCE ' + conf_s + '</div>'
    html += '<div class="cbar-wrap"><div class="cbar-fill" style="width:' + conf_s + ';background:' + conf_col + ';"></div></div>'
    html += '</div></div>'
    html += '<div style="display:grid;grid-template-columns:1fr 1fr 1fr 1fr;gap:0.8rem;min-width:280px;">'
    html += '<div><div class="slbl">RSI</div><div class="sval" style="color:' + rsi_col + '">' + rsi_s + '</div></div>'
    html += '<div><div class="slbl">MACD</div><div class="sval" style="color:' + macd_col + '">' + macd_s + '</div></div>'
    html += '<div><div class="slbl">ATR</div><div class="sval">' + atr_s + '</div></div>'
    html += '<div><div class="slbl">VOL</div><div class="sval" style="color:' + vol_col + '">' + vol_s + '</div></div>'
    html += '</div></div>'
    html += '<div class="rec">' + rec + '</div>'
    html += '<div style="margin-top:8px;">' + tags + '</div>'
    html += '</div>'
    return html


summary = {}
metric_cols = st.columns(len(st.session_state.tickers))
for idx, ticker in enumerate(st.session_state.tickers):
    df_raw = get_data(ticker, tf)
    if df_raw is None:
        with metric_cols[idx]:
            st.metric(ticker, "N/A")
        continue
    df_ind = add_indicators(df_raw)
    result = get_signal(df_ind)
    summary[ticker] = (df_ind,) + result
    direction, conf, sigs, rec, price, chg, chg_pct = result
    sign = "+" if chg >= 0 else ""
    with metric_cols[idx]:
        st.metric(ticker, "${:,.2f}".format(price), "{}{:.2f}%".format(sign, chg_pct))

st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

for ticker in st.session_state.tickers:
    if ticker not in summary:
        st.warning(ticker + ": no data available for this timeframe")
        continue

    df, direction, conf, sigs, rec, price, chg, chg_pct = summary[ticker]

    def safe_float(series_val, default=0.0):
        try:
            v = float(series_val)
            return v if not np.isnan(v) else default
        except Exception:
            return default

    rsi_val  = safe_float(df["RSI"].iloc[-1])
    macd_val = safe_float(df["MACD"].iloc[-1])
    atr_val  = safe_float(df["ATR"].iloc[-1])
    vol_val  = safe_float(df["Volume"].iloc[-1])
    vol_avg  = safe_float(df["VOLSMA"].iloc[-1], vol_val)
    vol_ratio = (vol_val / vol_avg) if vol_avg > 0 else 1.0

    st.markdown(
        build_card(ticker, direction, conf, sigs, rec, price, chg, chg_pct, rsi_val, macd_val, atr_val, vol_ratio),
        unsafe_allow_html=True
    )

    rows    = 3 if show_vol else 2
    heights = [0.55, 0.22, 0.23] if show_vol else [0.65, 0.35]

    fig = make_subplots(
        rows=rows, cols=1,
        shared_xaxes=True,
        row_heights=heights,
        vertical_spacing=0.02
    )

    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df["Open"],
            high=df["High"],
            low=df["Low"],
            close=df["Close"],
            name="Price",
            increasing=dict(line=dict(color="#00d464", width=1), fillcolor="#00d46422"),
            decreasing=dict(line=dict(color="#ff3b5c", width=1), fillcolor="#ff3b5c22")
        ),
        row=1, col=1
    )

    fig.add_trace(
        go.Scatter(
            x=df.index, y=df["SMA20"],
            name="SMA20",
            line=dict(color="#ff9800", width=1.5, dash="dot")
        ),
        row=1, col=1
    )

    if show_sma50:
        fig.add_trace(
            go.Scatter(
                x=df.index, y=df["SMA50"],
                name="SMA50",
                line=dict(color="#ab47bc", width=1.5, dash="dot")
            ),
            row=1, col=1
        )

    if show_bb:
        fig.add_trace(
            go.Scatter(
                x=df.index, y=df["BBU"],
                name="BB Upper",
                line=dict(color="rgba(100,180,255,0.4)", width=1),
                showlegend=False
            ),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=df.index, y=df["BBL"],
                name="BB Lower",
                fill="tonexty",
                fillcolor="rgba(100,180,255,0.05)",
                line=dict(color="rgba(100,180,255,0.4)", width=1),
                showlegend=False
            ),
            row=1, col=1
        )

    st_color = "#00d464" if df["STDIR"].iloc[-1] == 1 else "#ff3b5c"
    fig.add_trace(
        go.Scatter(
            x=df.index, y=df["STLINE"],
            name="SuperTrend",
            line=dict(color=st_color, width=2)
        ),
        row=1, co
