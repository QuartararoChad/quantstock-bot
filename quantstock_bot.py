import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import time
from datetime import datetime

st.set_page_config(page_title="QuantStock", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
html, body, [class*="css"] { font-family: 'Segoe UI', sans-serif; background: #0a0e1a; color: #e2e8f0; }
.main { background: #0a0e1a; }
.block-container { padding: 1.5rem 2rem; max-width: 100%; }
#MainMenu, footer, header { visibility: hidden; }
section[data-testid="stSidebar"] { background: #0d1117 !important; }
.top-bar { display: flex; justify-content: space-between; align-items: center; margin-bottom: 1.5rem; border-bottom: 1px solid #1e3a5f; padding-bottom: 1rem; }
.app-logo { font-size: 1.6rem; font-weight: 800; color: #38bdf8; letter-spacing: 2px; }
.app-time { font-size: 0.75rem; color: #4a6a8a; }
.live-pill { background: rgba(34,197,94,0.1); border: 1px solid rgba(34,197,94,0.4); color: #22c55e; font-size: 0.7rem; padding: 3px 10px; border-radius: 20px; font-weight: 600; letter-spacing: 1px; }
.search-wrap { background: #0d1625; border: 1px solid #1e3a5f; border-radius: 10px; padding: 1rem 1.2rem; margin-bottom: 1.5rem; }
.search-label { font-size: 0.7rem; color: #4a6a8a; letter-spacing: 2px; text-transform: uppercase; margin-bottom: 0.5rem; }
.watchlist-bar { display: flex; flex-wrap: wrap; gap: 0.5rem; margin-bottom: 1.5rem; align-items: center; }
.wtag { background: #0d1625; border: 1px solid #1e3a5f; border-radius: 6px; padding: 6px 14px; font-size: 0.8rem; color: #94a3b8; display: inline-flex; align-items: center; gap: 8px; }
.wtag-active { border-color: #38bdf8; color: #38bdf8; background: rgba(56,189,248,0.08); }
.card { background: #0d1625; border: 1px solid #1e3a5f; border-radius: 10px; padding: 1.2rem 1.5rem; margin-bottom: 1rem; }
.card-bull { border-left: 3px solid #22c55e; }
.card-bear { border-left: 3px solid #ef4444; }
.card-neut { border-left: 3px solid #f59e0b; }
.ticker-sym { font-size: 1.6rem; font-weight: 800; color: #f1f5f9; letter-spacing: 1px; }
.ticker-name { font-size: 0.75rem; color: #4a6a8a; margin-top: 2px; }
.price-big { font-size: 1.8rem; font-weight: 700; color: #38bdf8; }
.chg-up { color: #22c55e; font-size: 0.9rem; font-weight: 600; }
.chg-dn { color: #ef4444; font-size: 0.9rem; font-weight: 600; }
.sig-bull { background: rgba(34,197,94,0.1); color: #22c55e; border: 1px solid rgba(34,197,94,0.3); padding: 2px 10px; border-radius: 4px; font-size: 0.72rem; font-weight: 700; letter-spacing: 1px; }
.sig-bear { background: rgba(239,68,68,0.1); color: #ef4444; border: 1px solid rgba(239,68,68,0.3); padding: 2px 10px; border-radius: 4px; font-size: 0.72rem; font-weight: 700; letter-spacing: 1px; }
.sig-neut { background: rgba(245,158,11,0.1); color: #f59e0b; border: 1px solid rgba(245,158,11,0.3); padding: 2px 10px; border-radius: 4px; font-size: 0.72rem; font-weight: 700; letter-spacing: 1px; }
.stat-box { background: #0a0e1a; border: 1px solid #1e3a5f; border-radius: 6px; padding: 0.6rem 0.8rem; text-align: center; }
.stat-lbl { font-size: 0.6rem; color: #4a6a8a; letter-spacing: 1px; text-transform: uppercase; }
.stat-val { font-size: 1rem; font-weight: 700; margin-top: 2px; }
.rec-bar { background: #0a0e1a; border: 1px solid #1e3a5f; border-radius: 6px; padding: 0.6rem 1rem; margin-top: 0.8rem; font-size: 0.82rem; color: #94a3b8; }
.stag { display: inline-block; background: #0a0e1a; border: 1px solid #1e3a5f; color: #64748b; font-size: 0.68rem; padding: 2px 8px; border-radius: 4px; margin: 3px 2px 0 0; }
.conf-wrap { background: #0a0e1a; border-radius: 4px; height: 4px; width: 100%; margin-top: 6px; }
.section-label { font-size: 0.65rem; letter-spacing: 2px; text-transform: uppercase; color: #4a6a8a; margin: 1.5rem 0 0.5rem 0; }
div[data-testid="metric-container"] { background: #0d1625; border: 1px solid #1e3a5f; border-radius: 8px; padding: 0.8rem 1rem; }
div[data-testid="metric-container"] label { color: #4a6a8a !important; font-size: 0.7rem !important; }
div[data-testid="metric-container"] div[data-testid="stMetricValue"] { color: #38bdf8 !important; font-size: 1.2rem !important; font-weight: 700 !important; }
.stButton > button { background: #0d1625 !important; border: 1px solid #1e3a5f !important; color: #94a3b8 !important; border-radius: 6px !important; transition: all 0.15s !important; }
.stButton > button:hover { border-color: #38bdf8 !important; color: #38bdf8 !important; }
div[data-testid="stTextInput"] input { background: #0a0e1a !important; border: 1px solid #1e3a5f !important; color: #e2e8f0 !important; border-radius: 6px !important; }
div[data-testid="stTextInput"] input:focus { border-color: #38bdf8 !important; box-shadow: 0 0 0 2px rgba(56,189,248,0.15) !important; }
</style>
""", unsafe_allow_html=True)

if "tickers" not in st.session_state:
    st.session_state.tickers = ["AAPL", "TSLA", "NVDA", "AMZN"]
if "selected" not in st.session_state:
    st.session_state.selected = "AAPL"
if "last_refresh" not in st.session_state:
    st.session_state.last_refresh = time.time()
if "refresh_count" not in st.session_state:
    st.session_state.refresh_count = 0

now_str = datetime.now().strftime("%a %d %b %Y  |  %H:%M:%S")

st.markdown("<div class='top-bar'><div><div class='app-logo'>QUANTSTOCK</div><div class='app-time'>" + now_str + "</div></div><div class='live-pill'>● LIVE</div></div>", unsafe_allow_html=True)

st.markdown("<div class='search-label'>Search &amp; Add Ticker</div>", unsafe_allow_html=True)
col_in, col_btn, col_ref = st.columns([4, 1, 1])
with col_in:
    new_ticker = st.text_input("", placeholder="Enter symbol e.g. MSFT, GOOGL, BTC-USD ...", label_visibility="collapsed", key="new_t")
with col_btn:
    add_clicked = st.button("+ Add", use_container_width=True)
with col_ref:
    if st.button("⟳ Refresh", use_container_width=True):
        st.cache_data.clear()
        st.session_state.last_refresh = time.time()
        st.session_state.refresh_count += 1
        st.rerun()

if add_clicked:
    sym = new_ticker.strip().upper()
    if sym and sym not in st.session_state.tickers:
        with st.spinner("Verifying " + sym + "..."):
            try:
                px = yf.Ticker(sym).fast_info.last_price
                if px and px > 0:
                    st.session_state.tickers.append(sym)
                    st.session_state.selected = sym
                    st.success(sym + " added to watchlist")
                    st.rerun()
                else:
                    st.error("Symbol not found: " + sym)
            except Exception:
                st.error("Could not verify: " + sym)
    elif sym in st.session_state.tickers:
        st.warning(sym + " is already in your watchlist")

st.markdown("<div class='section-label'>Watchlist</div>", unsafe_allow_html=True)

ticker_cols = st.columns(len(st.session_state.tickers) + 1)
to_remove = None
for i, t in enumerate(st.session_state.tickers):
    with ticker_cols[i]:
        is_sel = (t == st.session_state.selected)
        label = ("▶ " if is_sel else "") + t
        if st.button(label, key="sel_" + t, use_container_width=True):
            st.session_state.selected = t
            st.rerun()
with ticker_cols[len(st.session_state.tickers)]:
    if len(st.session_state.tickers) > 1:
        if st.button("✕ Remove " + st.session_state.selected, use_container_width=True):
            st.session_state.tickers.remove(st.session_state.selected)
            st.session_state.selected = st.session_state.tickers[0]
            st.rerun()

st.markdown("---")

col_tf, col_bb, col_sma, col_vol, col_ar = st.columns([2, 1, 1, 1, 2])
with col_tf:
    tf = st.selectbox("Timeframe", ["5m", "15m", "30m", "1h", "1d"], index=3, label_visibility="visible")
with col_bb:
    show_bb = st.toggle("BB", value=True)
with col_sma:
    show_sma50 = st.toggle("SMA50", value=False)
with col_vol:
    show_vol = st.toggle("Volume", value=True)
with col_ar:
    auto_ref = st.toggle("Auto-refresh", value=True)

if auto_ref:
    try:
        from streamlit_autorefresh import st_autorefresh
        st_autorefresh(interval=60000, limit=99999, key="ar")
    except ImportError:
        pass

ticker = st.session_state.selected


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
    df["MACDH"] = df["MACD"] - df["MACDS"]
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


df_raw = get_data(ticker, tf)
if df_raw is None:
    st.error("No data available for " + ticker + " on " + tf + " timeframe. Try a different timeframe.")
    st.stop()

df = add_indicators(df_raw)
direction, conf, sigs, rec, price, chg, chg_pct = get_signal(df)

rsi_v = safe_f(df["RSI"].iloc[-1])
macd_v = safe_f(df["MACD"].iloc[-1])
atr_v = safe_f(df["ATR"].iloc[-1])
vol_v = safe_f(df["Volume"].iloc[-1])
vol_a = safe_f(df["VOLSMA"].iloc[-1], vol_v)
vol_r = (vol_v / vol_a) if vol_a > 0 else 1.0

if direction == "BULLISH":
    sig_cls = "sig-bull"
    card_cls = "card-bull"
    conf_col = "#22c55e"
elif direction == "BEARISH":
    sig_cls = "sig-bear"
    card_cls = "card-bear"
    conf_col = "#ef4444"
else:
    sig_cls = "sig-neut"
    card_cls = "card-neut"
    conf_col = "#f59e0b"

chg_cls = "chg-up" if chg >= 0 else "chg-dn"
chg_sign = "+" if chg >= 0 else ""
rsi_col = "#22c55e" if rsi_v < 40 else "#ef4444" if rsi_v > 60 else "#94a3b8"
macd_col = "#22c55e" if macd_v > 0 else "#ef4444"
vol_col = "#f59e0b" if vol_r > 1.5 else "#94a3b8"
price_s = "${:,.2f}".format(price)
chg_s = "{}{:.2f} ({}{:.2f}%)".format(chg_sign, chg, chg_sign, chg_pct)
conf_s = str(conf) + "%"
tags = "".join(["<span class='stag'>" + s + "</span>" for s in sigs])

info_col, chart_col = st.columns([1, 3])

with info_col:
    h = "<div class='card " + card_cls + "'>"
    h += "<div style='margin-bottom:0.8rem;'>"
    h += "<div style='display:flex;align-items:center;gap:10px;'>"
    h += "<span class='ticker-sym'>" + ticker + "</span>"
    h += "<span class='" + sig_cls + "'>" + direction + "</span>"
    h += "</div></div>"
    h += "<div class='price-big'>" + price_s + "</div>"
    h += "<div class='" + chg_cls + "' style='margin-top:4px;'>" + chg_s + "</div>"
    h += "<div style='margin-top:1rem;font-size:0.65rem;color:#4a6a8a;margin-bottom:4px;'>CONFIDENCE</div>"
    h += "<div class='conf-wrap'>"
    h += "<div style='height:4px;border-radius:4px;background:" + conf_col + ";width:" + conf_s + ";'></div>"
    h += "</div>"
    h += "<div style='font-size:0.75rem;color:" + conf_col + ";margin-top:4px;font-weight:600;'>" + conf_s + "</div>"
    h += "<div class='rec-bar'>" + rec + "</div>"
    h += "<div style='margin-top:0.8rem;'>" + tags + "</div>"
    h += "</div>"
    h += "<div style='display:grid;grid-template-columns:1fr 1fr;gap:0.5rem;margin-top:0.5rem;'>"
    h += "<div class='stat-box'><div class='stat-lbl'>RSI</div><div class='stat-val' style='color:" + rsi_col + "'>" + "{:.1f}".format(rsi_v) + "</div></div>"
    h += "<div class='stat-box'><div class='stat-lbl'>MACD</div><div class='stat-val' style='color:" + macd_col + "'>" + "{:+.3f}".format(macd_v) + "</div></div>"
    h += "<div class='stat-box'><div class='stat-lbl'>ATR</div><div class='stat-val'>" + "{:.2f}".format(atr_v) + "</div></div>"
    h += "<div class='stat-box'><div class='stat-lbl'>VOL RATIO</div><div class='stat-val' style='color:" + vol_col + "'>" + "{:.1f}x".format(vol_r) + "</div></div>"
    h += "</div>"
    st.markdown(h, unsafe_allow_html=True)

with chart_col:
    rows = 3 if show_vol else 2
    heights = [0.58, 0.20, 0.22] if show_vol else [0.72, 0.28]
    row_titles = ["Price", "Volume", "RSI"] if show_vol else ["Price", "RSI"]
    fig = make_subplots(rows=rows, cols=1, shared_xaxes=True, row_heights=heights, vertical_spacing=0.03, row_titles=row_titles)
    fig.add_trace(go.Candlestick(x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"], name="Price", increasing_line_color="#22c55e", decreasing_line_color="#ef4444"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["SMA20"], name="SMA20", line=dict(color="#f59e0b", width=1.5, dash="dot")), row=1, col=1)
    if show_sma50:
        fig.add_trace(go.Scatter(x=df.index, y=df["SMA50"], name="SMA50", line=dict(color="#a78bfa", width=1.5, dash="dot")), row=1, col=1)
    if show_bb:
        fig.add_trace(go.Scatter(x=df.index, y=df["BBU"], name="BB Upper", line=dict(color="rgba(56,189,248,0.5)", width=1), showlegend=False), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df["BBL"], name="BB Lower", fill="tonexty", fillcolor="rgba(56,189,248,0.04)", line=dict(color="rgba(56,189,248,0.5)", width=1), showlegend=False), row=1, col=1)
    stc = "#22c55e" if df["STDIR"].iloc[-1] == 1 else "#ef4444"
    fig.add_trace(go.Scatter(x=df.index, y=df["STLINE"], name="SuperTrend", line=dict(color=stc, width=2)), row=1, col=1)
    rsi_row = 2
    if show_vol:
        cl = df["Close"].squeeze().tolist()
        op = df["Open"].squeeze().tolist()
        vc = ["rgba(34,197,94,0.4)" if cl[i] >= op[i] else "rgba(239,68,68,0.4)" for i in range(len(cl))]
        fig.add_trace(go.Bar(x=df.index, y=df["Volume"], name="Volume", marker_color=vc, showlegend=False), row=2, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df["VOLSMA"], name="Vol MA", line=dict(color="#f59e0b", width=1.2), showlegend=False), row=2, col=1)
        rsi_row = 3
    fig.add_trace(go.Scatter(x=df.index, y=df["RSI"], name="RSI", line=dict(color="#818cf8", width=1.8)), row=rsi_row, col=1)
    fig.add_hline(y=70, row=rsi_row, col=1, line=dict(color="#ef4444", dash="dash", width=1))
    fig.add_hline(y=30, row=rsi_row, col=1, line=dict(color="#22c55e", dash="dash", width=1))
    fig.add_hline(y=50, row=rsi_row, col=1, line=dict(color="#334155", dash="dot", width=1))
    fig.add_hrect(y0=30, y1=70, row=rsi_row, col=1, fillcolor="rgba(129,140,248,0.03)", line_width=0)
    fig.update_layout(height=700, template="plotly_dark", paper_bgcolor="#0d1625", plot_bgcolor="#0a0e1a", font=dict(family="Segoe UI, sans-serif", size=11, color="#64748b"), legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0, bgcolor="rgba(0,0,0,0)", font=dict(size=11, color="#94a3b8")), margin=dict(l=0, r=20, t=30, b=0), xaxis_rangeslider_visible=False, hovermode="x unified")
    fig.update_xaxes(showgrid=True, gridcolor="#1e3a5f", gridwidth=1, showline=False, zeroline=False, tickfont=dict(size=10, color="#4a6a8a"))
    fig.update_yaxes(showgrid=True, gridcolor="#1e3a5f", gridwidth=1, showline=False, zeroline=False, tickfont=dict(size=10, color="#4a6a8a"), tickprefix="$", row=1, col=1)
    fig.update_yaxes(showgrid=True, gridcolor="#1e3a5f", gridwidth=1, showline=False, zeroline=False, tickfont=dict(size=10, color="#4a6a8a"), row=2, col=1)
    fig.update_yaxes(showgrid=True, gridcolor="#1e3a5f", gridwidth=1, showline=False, zeroline=False, tickfont=dict(size=10, color="#4a6a8a"), range=[0, 100], row=rsi_row, col=1)
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": True, "modeBarButtonsToRemove": ["autoScale2d", "lasso2d", "select2d"], "displaylogo": False})

st.markdown("<div style='text-align:center;padding:1.5rem 0 0.5rem;font-size:0.65rem;color:#1e3a5f;letter-spacing:1px;'>QUANTSTOCK  |  FOR EDUCATIONAL USE ONLY  |  NOT FINANCIAL ADVICE  |  DATA VIA Y
