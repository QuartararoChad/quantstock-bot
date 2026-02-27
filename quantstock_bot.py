# QuantStock Bot v5.0
import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import time
from datetime import datetime

st.set_page_config(
    page_title="QuantStock",
    layout="wide",
    page_icon="chart_with_upwards_trend",
    initial_sidebar_state="expanded"
)

_C1 = "@import url('https://fonts.googleapis.com/css2?family="
_C1 += "JetBrains+Mono:wght@300;400;600;700"
_C1 += "&family=Rajdhani:wght@400;600;700&display=swap');"

_C2 = "html,body,[class*='css']"
_C2 += "{font-family:'JetBrains Mono',monospace;"
_C2 += "background:#080c10;color:#c8d6e5;}"

_C3 = ".main{background:#080c10;}"
_C3 += ".block-container{padding:1rem 2rem 2rem 2rem;"
_C3 += "max-width:100%;}"
_C3 += "#MainMenu,footer,header{visibility:hidden;}"

_C4 = "section[data-testid='stSidebar']"
_C4 += "{background:#0d1117 !important;"
_C4 += "border-right:1px solid #1e2d3d;}"

_C5 = ".app-title{font-family:'Rajdhani',sans-serif;"
_C5 += "font-size:2rem;font-weight:700;color:#00d4ff;"
_C5 += "letter-spacing:3px;text-transform:uppercase;}"
_C5 += ".app-sub{font-size:0.68rem;color:#4a6a8a;"
_C5 += "letter-spacing:2px;text-transform:uppercase;"
_C5 += "margin-top:2px;}"

_C6 = ".live-badge{display:inline-flex;align-items:center;"
_C6 += "gap:6px;background:rgba(0,212,100,0.08);"
_C6 += "border:1px solid rgba(0,212,100,0.3);"
_C6 += "color:#00d464;font-size:0.72rem;padding:4px 12px;"
_C6 += "border-radius:3px;letter-spacing:2px;font-weight:600;}"
_C6 += ".live-dot{width:7px;height:7px;background:#00d464;"
_C6 += "border-radius:50%;"
_C6 += "animation:pulse 1.4s ease-in-out infinite;"
_C6 += "display:inline-block;}"
_C6 += "@keyframes pulse{"
_C6 += "0%,100%{opacity:1;transform:scale(1);}"
_C6 += "50%{opacity:0.3;transform:scale(0.7);}}"

_C7 = ".hdr{display:flex;justify-content:space-between;"
_C7 += "align-items:center;"
_C7 += "padding:0.6rem 0 1.2rem;"
_C7 += "border-bottom:1px solid #1e2d3d;"
_C7 += "margin-bottom:1.4rem;}"

_C8 = ".card{background:#0d1117;"
_C8 += "border:1px solid #1e2d3d;"
_C8 += "border-radius:6px;"
_C8 += "padding:1.2rem 1.4rem;"
_C8 += "margin-bottom:0.4rem;}"
_C8 += ".card-bull{border-left:3px solid #00d464;}"
_C8 += ".card-bear{border-left:3px solid #ff3b5c;}"
_C8 += ".card-neut{border-left:3px solid #ffa500;}"

_C9 = ".tkr{font-family:'Rajdhani',sans-serif;"
_C9 += "font-size:1.5rem;font-weight:700;"
_C9 += "color:#e8f4fd;letter-spacing:2px;}"
_C9 += ".prc{font-size:1.6rem;font-weight:700;color:#00d4ff;}"
_C9 += ".up{color:#00d464;font-size:0.85rem;}"
_C9 += ".dn{color:#ff3b5c;font-size:0.85rem;}"

_CA = ".sbadge{display:inline-block;padding:3px 10px;"
_CA += "border-radius:3px;font-size:0.72rem;font-weight:700;"
_CA += "letter-spacing:1.5px;text-transform:uppercase;}"
_CA += ".sb-bull{background:rgba(0,212,100,0.12);"
_CA += "color:#00d464;border:1px solid rgba(0,212,100,0.3);}"
_CA += ".sb-bear{background:rgba(255,59,92,0.12);"
_CA += "color:#ff3b5c;border:1px solid rgba(255,59,92,0.3);}"
_CA += ".sb-neut{background:rgba(255,165,0,0.12);"
_CA += "color:#ffa500;border:1px solid rgba(255,165,0,0.3);}"

_CB = ".rec{background:#111822;"
_CB += "border:1px solid #1e2d3d;border-radius:4px;"
_CB += "padding:0.5rem 0.9rem;font-size:0.82rem;"
_CB += "margin-top:0.6rem;color:#8aadcc;}"
_CB += ".cbar-wrap{background:#111822;border-radius:3px;"
_CB += "height:5px;width:100%;margin-top:6px;}"
_CB += ".cbar-fill{height:5px;border-radius:3px;}"

_CC = ".stag{display:inline-block;background:#111822;"
_CC += "border:1px solid #1e2d3d;color:#6a8aaa;"
_CC += "font-size:0.68rem;padding:2px 8px;"
_CC += "border-radius:2px;margin:2px 2px 0 0;}"
_CC += ".slbl{font-size:0.65rem;color:#4a6a8a;"
_CC += "letter-spacing:1px;text-transform:uppercase;"
_CC += "margin-bottom:2px;}"
_CC += ".sval{font-size:0.9rem;color:#c8d6e5;font-weight:600;}"
_CC += ".divider{border:none;"
_CC += "border-top:1px solid #1e2d3d;margin:0.8rem 0;}"

_CD = ".sec{font-family:'Rajdhani',sans-serif;"
_CD += "font-size:0.75rem;letter-spacing:3px;"
_CD += "text-transform:uppercase;color:#4a6a8a;"
_CD += "border-bottom:1px solid #1e2d3d;"
_CD += "padding-bottom:6px;margin-bottom:12px;"
_CD += "margin-top:20px;}"

_CE = "div[data-testid='stTextInput'] input"
_CE += "{background:#111822 !important;"
_CE += "border:1px solid #1e2d3d !important;"
_CE += "color:#c8d6e5 !important;"
_CE += "border-radius:4px !important;}"

_CF = ".stButton > button"
_CF += "{background:#111822 !important;"
_CF += "border:1px solid #1e2d3d !important;"
_CF += "color:#8aadcc !important;"
_CF += "border-radius:4px !important;"
_CF += "font-size:0.78rem !important;}"
_CF += ".stButton > button:hover"
_CF += "{border-color:#00d4ff !important;"
_CF += "color:#00d4ff !important;"
_CF += "background:rgba(0,212,255,0.06) !important;}"

_CG = "div[data-testid='metric-container']"
_CG += "{background:#111822;"
_CG += "border:1px solid #1e2d3d;"
_CG += "border-radius:4px;"
_CG += "padding:0.6rem 0.9rem;}"
_CG += "div[data-testid='metric-container'] label"
_CG += "{color:#4a6a8a !important;font-size:0.7rem !important;}"
_CG += "div[data-testid='metric-container'] "
_CG += "div[data-testid='stMetricValue']"
_CG += "{color:#00d4ff !important;"
_CG += "font-size:1.1rem !important;}"

CSS = '<style>' + _C1+_C2+_C3+_C4+_C5+_C6+_C7+_C8+_C9
CSS += _CA+_CB+_CC+_CD+_CE+_CF+_CG + '</style>'
st.markdown(CSS, unsafe_allow_html=True)

if 'tickers' not in st.session_state:
    st.session_state.tickers = ['AAPL', 'TSLA', 'NVDA', 'AMZN']
if 'last_refresh' not in st.session_state:
    st.session_state.last_refresh = time.time()
if 'refresh_count' not in st.session_state:
    st.session_state.refresh_count = 0

with st.sidebar:
    st.markdown('<div class="sec">Add Ticker</div>',
                unsafe_allow_html=True)
    new_ticker = st.text_input(
        '',
        placeholder='Symbol e.g. MSFT',
        label_visibility='collapsed',
        key='new_t'
    )
    if st.button('+ Add to Watchlist', use_container_width=True):
        symbol = new_ticker.strip().upper()
        already = symbol in st.session_state.tickers
        if symbol and not already:
            with st.spinner('Verifying...'):
                try:
                    p = yf.Ticker(symbol).fast_info.last_price
                    if p and p > 0:
                        st.session_state.tickers.append(symbol)
                        st.success(symbol + ' added')
                        st.rerun()
                    else:
                        st.error('Not found: ' + symbol)
                except Exception:
                    st.error('Could not verify: ' + symbol)
        elif already:
            st.warning(symbol + ' already in watchlist')

    st.markdown('<div class="sec">Watchlist</div>',
                unsafe_allow_html=True)
    to_remove = None
    for t in st.session_state.tickers:
        c1, c2 = st.columns([3, 1])
        with c1:
            _s = "<span style='color:#c8d6e5;"
            _s += "font-size:0.85rem;"
            _s += "line-height:2.2;"
            _s += "display:block'>"
            _s += t + "</span>"
            st.markdown(_s, unsafe_allow_html=True)
        with c2:
            if st.button('x', key='rm_' + t):
                to_remove = t
    if to_remove:
        st.session_state.tickers.remove(to_remove)
        st.rerun()

    st.markdown('<div class="sec">Settings</div>',
                unsafe_allow_html=True)
    tf = st.selectbox(
        'Timeframe',
        ['5m', '15m', '30m', '1h', '1d'],
        index=3
    )
    show_bb = st.toggle('Bollinger Bands', value=True)
    show_sma50 = st.toggle('SMA 50', value=False)
    show_vol = st.toggle('Volume Panel', value=True)

    st.markdown('<div class="sec">Auto-Refresh</div>',
                unsafe_allow_html=True)
    auto_ref = st.toggle('Enable', value=True)

    def _fmt(x):
        return (str(x) + 's') if x < 60 else (str(x // 60) + 'm')

    ref_secs = st.select_slider(
        'Interval',
        options=[30, 60, 120, 300, 600],
        value=60,
        format_func=_fmt
    )

    st.markdown('<div class="divider"></div>',
                unsafe_allow_html=True)
    b1, b2 = st.columns(2)
    with b1:
        if st.button('Refresh', use_container_width=True):
            st.cache_data.clear()
            st.session_state.last_refresh = time.time()
            st.session_state.refresh_count += 1
            st.rerun()
    with b2:
        if st.button('Clear Cache', use_container_width=True):
            st.cache_data.clear()
            st.rerun()

    st.markdown('<div class="divider"></div>',
                unsafe_allow_html=True)
    _ldt = datetime.fromtimestamp(
        st.session_state.last_refresh
    ).strftime('%H:%M:%S')
    _rc = str(st.session_state.refresh_count)
    _tc = str(len(st.session_state.tickers))
    _si = "<div style='font-size:0.65rem;"
    _si += "color:#4a6a8a;line-height:2;'>"
    _si += 'LAST UPDATE: ' + _ldt + '<br>'
    _si += 'REFRESHES: ' + _rc + '<br>'
    _si += 'TICKERS: ' + _tc + '</div>'
    st.markdown(_si, unsafe_allow_html=True)

if auto_ref:
    try:
        from streamlit_autorefresh import st_autorefresh
        st_autorefresh(interval=ref_secs * 1000,
                       limit=99999, key='ar')
    except ImportError:
        pass

now_str = datetime.now().strftime('%a %d %b %Y  |  %H:%M:%S')
_h = "<div class='hdr'>"
_h += "<div><div class='app-title'>QuantStock</div>"
_h += "<div class='app-sub'>"
_h += 'Real-Time Signal Dashboard  |  ' + now_str
_h += "</div></div>"
_h += "<div class='live-badge'>"
_h += "<span class='live-dot'></span> LIVE</div></div>"
st.markdown(_h, unsafe_allow_html=True)

if not st.session_state.tickers:
    st.info('Watchlist is empty. Add tickers in the sidebar.')
    st.stop()


@st.cache_data(ttl=30)
def get_data(ticker, interval):
    period = '60d' if interval == '1d' else '10d'
    try:
        df = yf.download(
            ticker,
            period=period,
            interval=interval,
            progress=False,
            auto_adjust=True,
            prepost=False
        )
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
    c = df['Close'].squeeze()
    h = df['High'].squeeze()
    lo = df['Low'].squeeze()
    df['SMA20'] = c.rolling(20).mean()
    df['SMA50'] = c.rolling(50).mean()
    delta = c.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    ag = gain.ewm(alpha=1/14, min_periods=14,
                  adjust=False).mean()
    al = loss.ewm(alpha=1/14, min_periods=14,
                  adjust=False).mean()
    df['RSI'] = 100 - (100 / (1 + ag / al.replace(0, np.nan)))
    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    df['MACD'] = ema12 - ema26
    df['MACDS'] = df['MACD'].ewm(span=9, adjust=False).mean()
    prev_c = c.shift(1)
    tr = pd.concat(
        [(h - lo), (h - prev_c).abs(), (lo - prev_c).abs()],
        axis=1
    ).max(axis=1)
    df['ATR'] = tr.ewm(span=14, adjust=False).mean()
    sma20 = c.rolling(20).mean()
    std20 = c.rolling(20).std()
    df['BBU'] = sma20 + 2 * std20
    df['BBL'] = sma20 - 2 * std20
    df['VOLSMA'] = df['Volume'].rolling(20).mean()
    atr10 = tr.rolling(10).mean()
    hl2 = (h + lo) / 2
    ur_raw = hl2 + 3 * atr10
    lr_raw = hl2 - 3 * atr10
    upper = ur_raw.copy()
    lower = lr_raw.copy()
    stdir = pd.Series(0, index=df.index)
    for i in range(1, len(df)):
        pl = lower.iloc[i - 1]
        pu = upper.iloc[i - 1]
        pc = c.iloc[i - 1]
        if lr_raw.iloc[i] > pl or pc < pl:
            lower.iloc[i] = lr_raw.iloc[i]
        else:
            lower.iloc[i] = pl
        if ur_raw.iloc[i] < pu or pc > pu:
            upper.iloc[i] = ur_raw.iloc[i]
        else:
            upper.iloc[i] = pu
        if c.iloc[i] > pu:
            stdir.iloc[i] = 1
        elif c.iloc[i] < pl:
            stdir.iloc[i] = -1
        else:
            stdir.iloc[i] = stdir.iloc[i - 1]
    df['STDIR'] = stdir
    df['STLINE'] = np.where(stdir == 1, lower, upper)
    return df


def get_signal(df):
    r = df.iloc[-1]
    p = df.iloc[-2]
    close = float(r['Close'])

    def v(col, default=0.0):
        val = r.get(col, default)
        return float(val) if pd.notna(val) else default

    score = 0.0
    sigs = []
    rsi = v('RSI', 50)
    macd = v('MACD', 0)
    macds = v('MACDS', 0)
    pmacd = float(p.get('MACD', 0) or 0)
    pmacs = float(p.get('MACDS', 0) or 0)
    stdir = v('STDIR', 0)
    sma20 = v('SMA20', close)
    sma50 = v('SMA50', close)
    bbu = v('BBU', close * 1.05)
    bbl = v('BBL', close * 0.95)
    vol = float(r.get('Volume', 0) or 0)
    vola = float(r.get('VOLSMA', vol) or vol)
    if close > sma20:
        score += 1.5
    else:
        score -= 1.0
    if close > sma50:
        score += 1.0
    else:
        score -= 0.5
    if rsi < 30:
        score += 3.0
        sigs.append('RSI Oversold')
    elif rsi < 40:
        score += 1.5
        sigs.append('RSI Bullish')
    elif rsi > 70:
        score -= 3.0
        sigs.append('RSI Overbought')
    elif rsi > 60:
        score -= 1.5
        sigs.append('RSI Bearish')
    if macd > macds and pmacd <= pmacs:
        score += 3.0
        sigs.append('MACD Bull Cross')
    elif macd < macds and pmacd >= pmacs:
        score -= 3.0
        sigs.append('MACD Bear Cross')
    elif macd > macds:
        score += 1.0
    else:
        score -= 1.0
    if stdir == 1:
        score += 3.0
        sigs.append('SuperTrend Bull')
    elif stdir == -1:
        score -= 3.0
        sigs.append('SuperTrend Bear')
    if close < bbl:
        score += 2.0
        sigs.append('Below BB Lower')
    elif close > bbu:
        score -= 2.0
        sigs.append('Above BB Upper')
    if vola > 0 and vol > vola * 1.5:
        score += 1.0 if score > 0 else -1.0
        sigs.append('Volume Spike')
    conf = max(20, min(98, int(50 + score * 5.5)))
    if score >= 3.5:
        direction = 'BULLISH'
    elif score <= -3.5:
        direction = 'BEARISH'
    else:
        direction = 'NEUTRAL'
    atr = v('ATR', float(r['High']) - float(r['Low']))
    if direction == 'BULLISH' and conf >= 65:
        tp = round(close + atr * 3, 2)
        sl = round(close - atr * 1.5, 2)
        rec = 'BUY  |  TP $' + str(tp) + '  |  SL $' + str(sl)
    elif direction == 'BEARISH' and conf >= 65:
        tp = round(close - atr * 3, 2)
        sl = round(close + atr * 1.5, 2)
        rec = 'SELL  |  TP $' + str(tp) + '  |  SL $' + str(sl)
    else:
        rec = 'HOLD  |  Waiting for stronger signal'
    open_p = float(r.get('Open', close) or close)
    chg = close - open_p
    chg_pct = (chg / open_p * 100) if open_p else 0.0
    return direction, conf, sigs[:5], rec, round(close, 2), chg, chg_pct


def build_card(ticker, direction, conf, sigs, rec,
               price, chg, chg_pct,
               rsi_val, macd_val, atr_val, vol_ratio):
    if direction == 'BULLISH':
        card_cls = 'card-bull'
        sig_cls = 'sb-bull'
        sig_lbl = 'BULL'
        conf_col = '#00d464'
    elif direction == 'BEARISH':
        card_cls = 'card-bear'
        sig_cls = 'sb-bear'
        sig_lbl = 'BEAR'
        conf_col = '#ff3b5c'
    else:
        card_cls = 'card-neut'
        sig_cls = 'sb-neut'
        sig_lbl = 'NEUT'
        conf_col = '#ffa500'
    chg_cls = 'up' if chg >= 0 else 'dn'
    chg_sign = '+' if chg >= 0 else ''
    if rsi_val < 40:
        rsi_col = '#00d464'
    elif rsi_val > 60:
        rsi_col = '#ff3b5c'
    else:
        rsi_col = '#c8d6e5'
    macd_col = '#00d464' if macd_val > 0 else '#ff3b5c'
    vol_col = '#ffa500' if vol_ratio > 1.5 else '#c8d6e5'
    price_s = '${:,.2f}'.format(price)
    chg_s = '{}{:.2f} ({}{:.2f}%)'.format(
        chg_sign, chg, chg_sign, chg_pct)
    conf_s = str(conf) + '%'
    rsi_s = '{:.1f}'.format(rsi_val)
    macd_s = '{:+.3f}'.format(macd_val)
    atr_s = '{:.2f}'.format(atr_val)
    vol_s = '{:.1f}x'.format(vol_ratio)
    tags = ''.join([
        "<span class='stag'>" + s + "</span>"
        for s in sigs
    ])
    html = "<div class='card " + card_cls + "'>"
    html += "<div style='display:flex;"
    html += "justify-content:space-between;"
    html += "align-items:flex-start;"
    html += "flex-wrap:wrap;gap:1rem;'>"
    html += '<div>'
    html += "<div style='display:flex;"
    html += "align-items:center;"
    html += "gap:12px;margin-bottom:4px;'>"
    html += "<span class='tkr'>" + ticker + "</span>"
    html += "<span class='sbadge " + sig_cls + "'>"
    html += sig_lbl + ' ' + direction + "</span>"
    html += '</div>'
    html += "<div style='display:flex;"
    html += "align-items:baseline;gap:10px;'>"
    html += "<span class='prc'>" + price_s + "</span>"
    html += "<span class='" + chg_cls + "'>" + chg_s + "</span>"
    html += '</div>'
    html += "<div style='margin-top:8px;'>"
    html += "<div style='font-size:0.65rem;"
    html += "color:#4a6a8a;margin-bottom:3px;'>"
    html += 'CONFIDENCE ' + conf_s + '</div>'
    html += "<div class='cbar-wrap'>"
    html += "<div class='cbar-fill' style='width:"
    html += conf_s + ";background:" + conf_col + ";'>"
    html += '</div></div>'
    html += '</div></div>'
    html += "<div style='display:grid;"
    html += "grid-template-columns:1fr 1fr 1fr 1fr;"
    html += "gap:0.8rem;min-width:280px;'>"
    html += '<div>'
    html += "<div class='slbl'>RSI</div>"
    html += "<div class='sval' style='color:" + rsi_col + "'>"
    html += rsi_s + '</div></div>'
    html += '<div>'
    html += "<div class='slbl'>MACD</div>"
    html += "<div class='sval' style='color:" + macd_col + "'>"
    html += macd_s + '</div></div>'
    html += '<div>'
    html += "<div class='slbl'>ATR</div>"
    html += "<div class='sval'>" + atr_s + '</div></div>'
    html += '<div>'
    html += "<div class='slbl'>VOL</div>"
    html += "<div class='sval' style='color:" + vol_col + "'>"
    html += vol_s + '</div></div>'
    html += '</div></div>'
    html += "<div class='rec'>" + rec + '</div>'
    html += "<div style='margin-top:8px;'>" + tags + '</div>'
    html += '</div>'
    return html


summary = {}
metric_cols = st.columns(len(st.session_state.tickers))
for idx, ticker in enumerate(st.session_state.tickers):
    df_raw = get_data(ticker, tf)
    if df_raw is None:
        with metric_cols[idx]:
            st.metric(ticker, 'N/A')
        continue
    df_ind = add_indicators(df_raw)
    result = get_signal(df_ind)
    summary[ticker] = (df_ind,) + result
    direction, conf, sigs, rec, price, chg, chg_pct = result
    sign = '+' if chg >= 0 else ''
    with metric_cols[idx]:
        st.metric(
            ticker,
            '${:,.2f}'.format(price),
            '{}{:.2f}%'.format(sign, chg_pct)
        )

st.markdown('<div class="divider"></div>',
            unsafe_allow_html=True)

for ticker in st.session_state.tickers:
    if ticker not in summary:
        st.warning(ticker + ': no data available')
        continue

    df, direction, conf, sigs, rec, price, chg, chg_pct = (
        summary[ticker]
    )

    def safe_float(val, default=0.0):
        try:
            result = float(val)
            return result if not np.isnan(result) else default
        except Exception:
            return default

    rsi_val = safe_float(df['RSI'].iloc[-1])
    macd_val = safe_float(df['MACD'].iloc[-1])
    atr_val = safe_float(df['ATR'].iloc[-1])
    vol_val = safe_float(df['Volume'].iloc[-1])
    vol_avg = safe_float(df['VOLSMA'].iloc[-1], vol_val)
    vol_ratio = (vol_val / vol_avg) if vol_avg > 0 else 1.0

    st.markdown(
        build_card(
            ticker, direction, conf, sigs, rec,
            price, chg, chg_pct,
            rsi_val, macd_val, atr_val, vol_ratio
        ),
                   
