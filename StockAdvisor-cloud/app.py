"""
Stock Advisor - a simple NSE stock helper (English, light Hindi).
Run:  streamlit run app.py   (or double-click "Start Stock Advisor.bat")
"""
from __future__ import annotations

import os
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

import data
import engine
import history
import report
import tracker
import notify

APP_DIR = os.path.dirname(os.path.abspath(__file__))

st.set_page_config(page_title="Stock Advisor", page_icon="📈", layout="wide",
                   initial_sidebar_state="expanded")

# =============================================================================
# THEME / DESIGN SYSTEM
# =============================================================================
st.markdown("""<style>
:root{
  --bg:#0e1117; --card:#171b23; --card2:#1d2230; --line:rgba(255,255,255,.09);
  --txt:#e8eaf0; --mut:#9aa3b2;
  --green:#22c55e; --greenbg:rgba(34,197,94,.14);
  --red:#f87171;   --redbg:rgba(248,113,113,.14);
  --amber:#fbbf24; --amberbg:rgba(251,191,36,.14);
  --blue:#60a5fa;  --bluebg:rgba(96,165,250,.14);
  --purple:#c084fc;--purplebg:rgba(192,132,252,.14);
  --accent:#6366f1;
}
.block-container{padding-top:1.1rem;max-width:1280px}
h1,h2,h3,h4{letter-spacing:-.01em}
/* top gradient banner */
.hero{background:linear-gradient(120deg,rgba(99,102,241,.22),rgba(34,197,94,.14) 70%);
  border:1px solid var(--line);border-radius:18px;padding:16px 20px;margin-bottom:12px}
.hero h1{margin:0;font-size:1.65rem;display:flex;align-items:center;gap:10px}
.hero .tag{opacity:.85;font-size:.92rem;margin-top:3px}
/* cards */
.card{border:1px solid var(--line);border-radius:16px;padding:15px 16px;margin-bottom:10px;
  background:linear-gradient(180deg,var(--card),var(--card2));transition:transform .12s ease,border-color .12s ease}
.card:hover{transform:translateY(-2px);border-color:rgba(99,102,241,.45)}
.card h4{margin:0 0 2px 0;font-size:1.08rem}
.card .sub{color:var(--mut);font-size:.82rem;margin-bottom:8px}
.badge{display:inline-block;padding:3px 10px;border-radius:999px;font-size:.76rem;font-weight:700;margin:0 4px 4px 0}
.b-hi{background:var(--greenbg);color:var(--green)}
.b-mid{background:var(--amberbg);color:var(--amber)}
.b-lo{background:var(--redbg);color:var(--red)}
.b-info{background:var(--bluebg);color:var(--blue)}
.b-sec{background:var(--purplebg);color:var(--purple)}
.row3{display:flex;gap:8px;margin-top:10px}
.row3 div{flex:1;border-radius:11px;padding:7px 9px;font-size:.8rem;color:var(--mut)}
.r-buy{background:var(--bluebg)}.r-sl{background:var(--redbg)}.r-tg{background:var(--greenbg)}
.row3 b{display:block;font-size:1.02rem;color:var(--txt);margin-top:1px}
/* verdict */
.verdict{border-radius:18px;padding:18px 22px;margin:8px 0 12px 0}
.v-buy{background:linear-gradient(135deg,var(--greenbg),transparent);border:2px solid rgba(34,197,94,.5)}
.v-sell{background:linear-gradient(135deg,var(--redbg),transparent);border:2px solid rgba(248,113,113,.5)}
.v-wait{background:linear-gradient(135deg,rgba(148,163,184,.14),transparent);border:2px solid rgba(148,163,184,.4)}
.verdict .big{font-size:2rem;font-weight:800;line-height:1.1}
.verdict .msg{font-size:1.02rem;margin-top:6px;color:var(--txt)}
.verdict .sub{color:var(--mut);font-size:.82rem;margin-top:6px}
/* price */
.price{font-size:2.15rem;font-weight:800}
.up{color:var(--green)}.dn{color:var(--red)}
/* stat box */
.box{border:1px solid var(--line);border-radius:14px;padding:12px 14px;height:100%;
  background:linear-gradient(180deg,var(--card),var(--card2))}
.box .lbl{font-size:.82rem;color:var(--mut)}.box .val{font-size:1.5rem;font-weight:800}
.box .hint{font-size:.78rem;color:var(--mut);margin-top:2px}
.chart-tag{display:inline-block;background:var(--bluebg);color:var(--blue);padding:4px 12px;border-radius:999px;font-weight:700}
.pnl-up{color:var(--green);font-weight:800}.pnl-dn{color:var(--red);font-weight:800}
/* section label */
.seclabel{display:flex;align-items:center;gap:8px;margin:14px 0 4px;font-weight:700;font-size:1.05rem}
.meter{height:9px;border-radius:999px;background:rgba(255,255,255,.08);overflow:hidden;margin-top:6px}
.meter > span{display:block;height:100%;border-radius:999px}
@media (max-width:640px){
  .block-container{padding-left:.6rem;padding-right:.6rem}
  .hero h1{font-size:1.3rem}.verdict .big{font-size:1.5rem}
  .price{font-size:1.6rem}.box .val{font-size:1.2rem}.card h4{font-size:1rem}
  .row3{flex-wrap:wrap}
}
</style>""", unsafe_allow_html=True)

# =============================================================================
# LABELS
# =============================================================================
HZ_LABEL = {"Intraday": "⚡ Intraday (buy & sell same day)",
            "Weekly": "📅 Weekly (hold up to 1 week)",
            "Monthly": "🗓️ Monthly (hold up to 1 month)"}
HZ_SHORT = {"Intraday": "⚡ Intraday", "Weekly": "📅 Weekly", "Monthly": "🗓️ Monthly"}
HZ_EXPLAIN = {
    "Intraday": "Buy in the morning, sell before 3:10 PM. Small profit, fast decisions, highest risk.",
    "Weekly": "Hold for up to 1 week (5 trading days). Exit on target or stop-loss, whichever comes first.",
    "Monthly": "Hold for about 1 month. Bigger target, less stress — best for beginners.",
}
P_HOME = "🏠 Best Stocks"
P_SECTOR = "📊 Sectors"
P_BUCKET = "📌 My Bucket"
P_TRACK = "🔁 What Changed"
P_CHECK = "🔍 Check a Stock"
P_ALERTS = "📲 Telegram Alerts"
P_LEARN = "📚 Learn"
P_ACC = "🎯 How Accurate?"
PAGES = [P_HOME, P_SECTOR, P_BUCKET, P_TRACK, P_CHECK, P_ALERTS, P_LEARN, P_ACC]

for k, v in (("page", P_HOME), ("sel_sym", "RELIANCE"), ("hz", "Weekly"), ("nav_stack", [])):
    st.session_state.setdefault(k, v)


# ---- navigation helpers (power the Back button) -----------------------------
def navigate(to, remember=True):
    if remember and st.session_state.page != to:
        st.session_state.nav_stack = (st.session_state.nav_stack + [st.session_state.page])[-12:]
    st.session_state.page = to


def go_to(sym, hz):
    st.session_state.sel_sym = sym
    st.session_state.hz = hz
    navigate(P_CHECK)


def go_back():
    if st.session_state.nav_stack:
        st.session_state.page = st.session_state.nav_stack.pop()


def back_button():
    if st.session_state.nav_stack:
        prev = st.session_state.nav_stack[-1]
        st.button(f"← Back to {prev}", key=f"back_{st.session_state.page}", on_click=go_back)


# =============================================================================
# DATA (cached)
# =============================================================================
@st.cache_data(ttl=900, show_spinner=False)
def load_daily(universe: str, extra: tuple, demo: bool):
    syms = list(dict.fromkeys(data.UNIVERSES[universe] + list(extra)))
    if demo:
        return data.demo_daily(syms), data.demo_index(), True, datetime.now()
    try:
        d = data.download_daily(syms)
        idx = data.download_index()
    except Exception:
        d, idx = {}, pd.DataFrame()
    return d, idx, False, datetime.now()


@st.cache_data(ttl=900, show_spinner=False)
def run_scan(universe: str, extra: tuple, demo: bool, _stamp):
    daily, idx, is_demo, _ = load_daily(universe, extra, demo)
    status, _ = engine.market_status()
    today = engine.ist_now().date()
    nifty_ret60 = float(idx.Close.pct_change(60).iloc[-1] * 100) if len(idx) > 61 else 0.0
    results = {h: [] for h in engine.HORIZONS}
    for sym, raw in daily.items():
        try:
            d = engine.add_indicators(raw)
            for h in engine.HORIZONS:
                if h == "Intraday" and status == "open" and raw.index[-1].date() == today and not is_demo:
                    r = engine.analyse(sym, raw.iloc[:-1], h)
                    if r:
                        r["price"] = float(raw.Close.iloc[-1])
                else:
                    r = engine.analyse(sym, raw, h, d=d)
                if r:
                    r["name"] = data.NAMES.get(sym, sym)
                    r["rs"] = float(d.ret60.iloc[-1]) - nifty_ret60
                    results[h].append(r)
        except Exception:
            continue
    for h, rows in results.items():
        for side in ("BUY", "SELL"):
            rr = [r for r in rows if r["side"] == side and r["bt_trades"] > 0]
            tot = sum(r["bt_trades"] for r in rr)
            wins = sum(r["bt_trades"] * r["bt_profit"] / 100 for r in rr)
            prior = wins / tot * 100 if tot else 45.0
            for r in rows:
                if r["side"] == side:
                    w = r["bt_trades"] * (r["bt_profit"] if r["bt_trades"] else 0) / 100
                    r["chance"] = engine.adjusted_chance(w, r["bt_trades"], prior)
                    r["wins"] = int(round(w))
    return results, engine.market_regime(idx)


@st.cache_data(ttl=60, show_spinner=False)
def load_intraday(sym: str, demo: bool, base: float = 1000.0):
    if demo:
        return data.demo_intraday(sym, base), True
    try:
        df = data.download_intraday(sym)
        if df.empty:
            raise RuntimeError
        return df, False
    except Exception:
        return data.demo_intraday(sym, base), True


@st.cache_data(ttl=25, show_spinner=False)
def load_live(sym: str, demo: bool):
    if demo:
        return None
    return data.live_price(sym)


@st.cache_data(ttl=3600, show_spinner=False)
def load_fundamentals(sym: str):
    return data.fundamentals(sym)


@st.cache_data(ttl=86400, show_spinner=False)
def get_universe_map():
    allmap = data.nse_all_equities()
    n500 = data.nse_500()
    return allmap, n500


@st.cache_data(ttl=900, show_spinner=False)
def analyse_one(sym: str, demo: bool, _stamp):
    if demo:
        raw = data.demo_daily([sym]).get(sym)
    else:
        raw = data.download_daily([sym]).get(sym)
    if raw is None or len(raw) < 260:
        return None
    d = engine.add_indicators(raw)
    out = {}
    for h in engine.HORIZONS:
        r = engine.analyse(sym, raw, h, d=d)
        if not r:
            continue
        r["name"] = data.NAMES.get(sym, sym)
        r["rs"] = 0.0
        w = r["bt_trades"] * (r["bt_profit"] if r["bt_trades"] else 0) / 100
        r["chance"] = engine.adjusted_chance(w, r["bt_trades"], 45.0)
        r["wins"] = int(round(w))
        out[h] = r
    if out:
        out["_raw"] = raw
    return out or None


# =============================================================================
# SIDEBAR
# =============================================================================
allmap, n500 = get_universe_map()
data.NAMES.update(allmap)
ALL_NSE = sorted(allmap.keys())
UNIV = {
    "⚡ Nifty 50 (fastest)": data.NIFTY50,
    f"Top {len(data.ALL_STOCKS)} big stocks (fast)": data.ALL_STOCKS,
    f"Nifty 500 (~{len(n500)}) — medium": list(dict.fromkeys(n500)),
    f"🌐 All NSE stocks (~{len(ALL_NSE)}) — slow": ALL_NSE,
}
data.UNIVERSES.clear()
data.UNIVERSES.update(UNIV)

PRICE_BANDS = {
    "Any price": (0.0, 1e12),
    "₹10 – 50": (10.0, 50.0),
    "₹50 – 100": (50.0, 100.0),
    "₹100 – 200": (100.0, 200.0),
    "₹200 – 500": (200.0, 500.0),
    "₹500 – 700": (500.0, 700.0),
    "₹700 – 1000": (700.0, 1000.0),
    "Above ₹1000": (1000.0, 1e12),
}

st.sidebar.markdown("### 📈 Stock Advisor")
st.sidebar.radio("Go to", PAGES, key="page", label_visibility="collapsed")
st.sidebar.divider()

with st.sidebar.expander("⚙️ Settings", expanded=False):
    universe = st.selectbox("Which stocks to scan", list(data.UNIVERSES.keys()), index=1,
                            help="More stocks = more time. 'All NSE' is the fullest but slow (first run 3-5 min).")
    if universe.startswith("🌐"):
        st.caption("⏳ 'All NSE' has ~2000 stocks — first run can take 3-5 min. To check just one instantly, "
                   "use **Check a Stock** (any NSE stock works there).")
    extra_txt = st.text_input("Add more NSE stocks (comma separated)", placeholder="e.g. IDEA, SUZLON")
    extra = tuple(s.strip().upper() for s in extra_txt.split(",") if s.strip())
    demo = st.toggle("Demo mode (fake data)", value=False)
    capital = st.number_input("Your trading capital (₹)", 5000, 100_000_000, 100000, step=5000)
    risk_pct = st.slider("Max loss per trade (% of capital)", 0.5, 3.0, 1.0, 0.25,
                         help="1% means in ₹1,00,000 you risk at most ₹1,000 per trade.")

if st.sidebar.button("🔄 Get fresh data (refresh)", width="stretch"):
    st.cache_data.clear()
    st.rerun()

with st.spinner("Fetching and checking all stocks… first run takes 1-2 minutes ⏳"):
    daily, nifty, is_demo, fetched = load_daily(universe, extra, demo)
    if not demo and len(daily) == 0:
        st.markdown("<div class='hero'><h1>📈 Stock Advisor</h1></div>", unsafe_allow_html=True)
        st.error("### ❌ Live market data didn't load\n\n"
                 "Your internet or office network seems to be blocking Yahoo Finance. "
                 "**Fake (demo) data is deliberately not shown** — so you never trade on wrong numbers.")
        st.markdown("**What to do:**")
        st.markdown("- Press **🔄 Get fresh data** in the sidebar (try again).\n"
                    "- Try **home wifi / mobile hotspot** (office networks often block Yahoo).\n"
                    "- For learning/testing only: Sidebar → Settings → **Demo mode** "
                    "(this is FAKE data, don't trade on it).")
        if st.button("🔄 Try again"):
            st.cache_data.clear()
            st.rerun()
        st.stop()
    results, regime = run_scan(universe, extra, demo, fetched)

_last_dates = [engine.last_bar_date(df) for df in daily.values() if len(df)]
data_date = max([d for d in _last_dates if d], default=None)
age = (engine.ist_now().date() - data_date).days if data_date else None

st.sidebar.caption(f"Data pulled: {fetched:%d %b, %I:%M %p} · {len(daily)} stocks"
                   + (f" · candle {data_date:%d %b}" if data_date else ""))
snapshot_html = report.build_snapshot(results, regime, universe)
st.sidebar.download_button("⬇️ Download report (saved to Downloads)", snapshot_html,
                           file_name=f"stock_report_{datetime.now():%Y%m%d_%H%M}.html", mime="text/html",
                           width="stretch")
st.sidebar.caption("⚠️ Not SEBI-registered advice. Decide yourself, and always use a stop-loss.")

status, status_txt = engine.market_status()

if is_demo:
    st.error("⚠️ Showing DEMO / FAKE data — demo mode is ON. Don't trade on these numbers. "
             "Turn demo off in Settings and press 'Get fresh data'.")
elif age is not None and age >= 4:
    st.warning(f"🕒 Data is a bit old (last candle {data_date:%d %b}). "
               "Press **🔄 Get fresh data** in the sidebar for today's picks.")


# =============================================================================
# SHARED HELPERS
# =============================================================================
def money_plan(r):
    risk_per_share = abs(r["entry"] - r["stop"])
    qty = int((capital * risk_pct / 100) // risk_per_share) if risk_per_share > 0 else 0
    qty = min(qty, int(capital // r["entry"])) if r["entry"] > 0 else qty
    return qty, qty * r["entry"], qty * risk_per_share, qty * abs(r["target"] - r["entry"])


def chance_badge(ch):
    if ch >= 60:
        return "<span class='badge b-hi'>🔥 High chance</span>"
    if ch >= 52:
        return "<span class='badge b-mid'>👍 Decent chance</span>"
    return "<span class='badge b-lo'>⚠️ Low chance</span>"


def chance_word(ch):
    return "Good" if ch >= 60 else "Okay" if ch >= 52 else "Low"


def regime_line():
    ok = regime.get("ok")
    icon = "🟢" if ok else "🟡" if ok is None else "🔴"
    extra_ = f" · Nifty {regime['close']:,.0f} ({regime['chg']:+.2f}%)" if "close" in regime else ""
    st.markdown(f"{icon} **Market mood:** {regime.get('text')}{extra_}  \n🕘 {status_txt}")


def hero(title, tag=""):
    t = f"<div class='tag'>{tag}</div>" if tag else ""
    st.markdown(f"<div class='hero'><h1>{title}</h1>{t}</div>", unsafe_allow_html=True)


# =============================================================================
# TELEGRAM ALERT SCAN (runs whenever the app is open)
# =============================================================================
def scan_alerts():
    """Check tracked stocks against live price. Returns in-app alert rows and
    fires Telegram messages (de-duplicated) when something important happens."""
    items = tracker.load(APP_DIR)
    out = []
    if not items or demo or is_demo:
        return out
    ready = notify.is_ready(APP_DIR)
    today = engine.ist_now().date()
    for it in items[:25]:
        lv = load_live(it["symbol"], demo or is_demo)
        price = lv["price"] if lv else None
        chg = lv["chg"] if lv and lv.get("chg") is not None else None
        stt = tracker.status(it, price, chg, today)
        kind = None
        if stt["hit"] == "target":
            kind = "target"
        elif stt["hit"] == "stop":
            kind = "stop"
        elif stt["window_over"]:
            kind = "window"
        elif stt["sharp_fall"]:
            kind = "fall"
        if kind:
            out.append((it, stt, kind))
            if ready and not notify.already_sent(APP_DIR, tracker.alert_key(it, kind)):
                notify.broadcast(APP_DIR, _alert_text(it, stt, kind, price))
    return out


def _alert_text(it, stt, kind, price):
    nm, sym, hz = it["name"], it["symbol"], it["horizon"]
    pstr = f"₹{price:,.2f}" if price else "—"
    head = {
        "target": f"🎯 <b>{nm}</b> hit TARGET",
        "stop": f"🛑 <b>{nm}</b> hit STOP-LOSS",
        "window": f"⏰ <b>{nm}</b> — {hz} window over",
        "fall": f"⚠️ <b>{nm}</b> sharp fall today",
    }[kind]
    return (f"{head}\n{sym} · {hz} · now {pstr}\n{stt['msg']}\n"
            f"<i>Buy pt ₹{stt['buy_point']:,.1f} · SL ₹{stt['stop']:,.1f} · Target ₹{stt['target']:,.1f}</i>")


ALERTS = scan_alerts()
if ALERTS:
    with st.container():
        st.markdown("#### 🔔 Alerts for your bucket")
        for it, stt, kind in ALERTS:
            getattr(st, stt["level"])(f"**{it['name']} ({it['symbol']}, {it['horizon']})** — {stt['msg']}")
        if not notify.is_ready(APP_DIR):
            st.caption("💡 Connect Telegram (📲 Telegram Alerts page) to also get these on your phone when the app is closed.")


# =============================================================================
# DAY-OVER-DAY TRACKING (history)
# =============================================================================
today_str = str(engine.ist_now().date())

# default price window (the Home page filter bar overrides these globals)
price_lo, price_hi = 0.0, 1e12


def top_buys(h, n=9, sector="All", min_chance=0, sort_by="Best chance"):
    rows = [r for r in results[h] if r["signal"] == "BUY" and r["bt_trades"] >= 6
            and r["bt_avg"] > 0 and price_lo <= r["price"] < price_hi]
    ok = regime.get("ok")
    if ok is False:
        rows = [r for r in rows if r["rs"] > 0 and r["chance"] >= 52]
    elif ok is None:
        rows = [r for r in rows if r["rs"] > -3]
    if sector != "All":
        rows = [r for r in rows if data.sector_of(r["symbol"]) == sector]
    if min_chance:
        rows = [r for r in rows if r["chance"] >= min_chance]
    if sort_by == "Highest price move":
        rows.sort(key=lambda r: -((r["target"] / r["entry"] - 1) * 100))
    elif sort_by == "Most tested":
        rows.sort(key=lambda r: -r["bt_trades"])
    elif sort_by == "Strongest trend":
        rows.sort(key=lambda r: -r["score"])
    else:  # Best chance
        rows.sort(key=lambda r: (-(r["chance"] + 0.05 * r["score"] + (2 if r["rs"] > 0 else 0))))
    return rows[:n] if n else rows


# default picks (unfiltered) used by history snapshot
picks_by_h = {h: [r["symbol"] for r in top_buys(h, n=9)] for h in engine.HORIZONS}
curr_snap = history.build_snapshot(today_str, universe, results, picks_by_h, is_demo or demo)
prev_snap = history.previous_snapshot(APP_DIR, today_str)
if not (is_demo or demo):
    try:
        history.save_snapshot(APP_DIR, curr_snap)
    except Exception:
        pass


def _mini_levels(p):
    if not p:
        return ""
    try:
        return (f"<span class='sub'>Entry ₹{p['entry']:,.1f} · SL ₹{p['stop']:,.1f} · "
                f"Target ₹{p['target']:,.1f}</span>")
    except Exception:
        return ""


def render_changes(h):
    diff = history.compare(prev_snap, curr_snap, h)
    if not diff["have_prev"]:
        st.info("📅 This is the first record. From tomorrow the app will show what **changed** since last time — "
                "new picks, still-good ones, and which to exit.")
        return
    pd_ = diff["prev_date"]
    st.caption(f"Changes from last time ({pd_}) to now ({today_str}):")
    exited = diff["exited"]
    if exited:
        st.markdown("#### 🚪 Exit these now (no longer good)")
        for r in exited:
            res, note = r["outcome"]
            tag = {"win": "b-hi", "loss": "b-lo", "running": "b-mid"}.get(res, "b-mid")
            sig = r["today_signal"]
            why = "no longer a BUY" if not r["still_buy"] else "dropped out of the top list"
            st.markdown(
                f"<div class='card'><h4>{r['name']} <span class='badge {tag}'>{note}</span></h4>"
                f"<div class='sub'>{r['symbol']} · was a BUY, now {why} (today: {sig}). "
                f"Don't buy fresh. If you already hold it, exit at your stop-loss/target.</div>"
                f"{_mini_levels(r['prev'])}</div>", unsafe_allow_html=True)
    newp = diff["new"]
    if newp:
        st.markdown("#### 🆕 New picks (weren't there last time)")
        st.write(" · ".join(f"**{r['name']}**" for r in newp))
    cont = diff["continued"]
    if cont:
        st.markdown("#### ✅ Still good (were there last time too)")
        for r in cont:
            res, note = (r["outcome"] or ("", ""))
            tag = {"win": "b-hi", "loss": "b-lo", "running": "b-mid"}.get(res, "b-mid")
            badge = f"<span class='badge {tag}'>{note}</span>" if note else ""
            st.markdown(f"<div class='sub'>• <b>{r['name']}</b> ({r['symbol']}) {badge}</div>",
                        unsafe_allow_html=True)
    if not (exited or newp or cont):
        st.success("No change in this type since last time — same picks are running.")


def track_button(r, key_prefix):
    """Small add/remove-to-bucket control for an analysed dict r."""
    tracked = tracker.is_tracked(APP_DIR, r["symbol"], r["horizon"])
    if tracked:
        if st.button("📌 Remove from bucket", key=f"untrk_{key_prefix}", width="stretch"):
            tracker.remove(APP_DIR, r["symbol"], r["horizon"])
            st.rerun()
    else:
        if st.button("➕ Add to bucket", key=f"trk_{key_prefix}", width="stretch"):
            tracker.add(APP_DIR, r)
            st.toast(f"Added {r['name']} to your bucket")
            st.rerun()


# =============================================================================
# PAGE: HOME
# =============================================================================
page = st.session_state.page

if page == P_HOME:
    hero("🏠 Best Stocks Today", "Pick a style, see stocks worth buying — with buy price, stop-loss and target.")
    back_button()
    regime_line()
    h = st.segmented_control("How do you want to trade?", list(engine.HORIZONS),
                             format_func=lambda x: HZ_LABEL[x], default="Weekly", key="home_hz") or "Weekly"
    st.info(f"**{HZ_SHORT[h]}:** {HZ_EXPLAIN[h]}")

    # ---- change summary
    _diff = history.compare(prev_snap, curr_snap, h)
    if _diff["have_prev"] and (_diff["exited"] or _diff["new"]):
        n_exit, n_new = len(_diff["exited"]), len(_diff["new"])
        with st.expander(f"🔁 Since last time: {n_new} new · {n_exit} to exit", expanded=bool(n_exit)):
            render_changes(h)

    # ---- FILTER BAR
    with st.container():
        st.markdown("<div class='seclabel'>🧰 Filters</div>", unsafe_allow_html=True)
        f1, f2, f3, f4 = st.columns([1.1, 1, 1, 1])
        price_band = f1.selectbox("💰 Price range", list(PRICE_BANDS), index=0)
        sectors_present = sorted({data.sector_of(r["symbol"]) for r in results[h] if r["signal"] == "BUY"})
        sector_pick = f2.selectbox("🏭 Sector", ["All"] + sectors_present, index=0)
        min_chance = f3.select_slider("🎯 Min chance %", options=[0, 50, 52, 55, 58, 60, 65], value=0)
        sort_by = f4.selectbox("↕️ Sort by", ["Best chance", "Highest price move", "Most tested", "Strongest trend"])
    price_lo, price_hi = PRICE_BANDS[price_band]

    active = []
    if price_band != "Any price":
        active.append(f"price {price_band}")
    if sector_pick != "All":
        active.append(f"sector {sector_pick}")
    if min_chance:
        active.append(f"chance ≥ {min_chance}%")
    if active:
        st.caption("Filters on: " + " · ".join(active))

    picks = top_buys(h, n=9, sector=sector_pick, min_chance=min_chance, sort_by=sort_by)
    if not picks:
        st.warning("No good BUY stock for this type with these filters. **Saving money is also profit** — "
                   "loosen the filters, try another style, or check back tomorrow.")
    else:
        st.markdown(f"<div class='seclabel'>✅ Worth buying ({len(picks)})</div>", unsafe_allow_html=True)
        st.caption("Top one has the best chance. 'Chance' = how often this same signal made a profit over the last 2 years.")
        cols = st.columns(3)
        for i, r in enumerate(picks):
            with cols[i % 3]:
                up = (r["target"] / r["entry"] - 1) * 100
                dn = (r["stop"] / r["entry"] - 1) * 100
                trig = "Buy above" if h == "Intraday" else "Buy near"
                sec = data.sector_of(r["symbol"])
                st.markdown(f"""<div class='card'>
<h4>{r['name']}</h4><div class='sub'>{r['symbol']} · now ₹{r['price']:,.2f} · <span class='badge b-sec'>{sec}</span></div>
{chance_badge(r['chance'])}<span class='badge b-mid'>{r['chance']:.0f}% chance</span>
<div class='row3'><div class='r-buy'>{trig}<b>₹{r['entry']:,.1f}</b></div>
<div class='r-sl'>Stop-loss<b>₹{r['stop']:,.1f}</b>{dn:.1f}%</div>
<div class='r-tg'>Target<b>₹{r['target']:,.1f}</b>+{up:.1f}%</div></div>
<div class='sub' style='margin-top:8px'>Profit in {r['wins']} of last {r['bt_trades']} signals</div>
</div>""", unsafe_allow_html=True)
                c1, c2 = st.columns(2)
                c1.button("Full plan ➜", key=f"go_{h}_{r['symbol']}", on_click=go_to, args=(r["symbol"], h), width="stretch")
                with c2:
                    if tracker.is_tracked(APP_DIR, r["symbol"], h):
                        if st.button("📌 In bucket", key=f"ut_{h}_{r['symbol']}", width="stretch"):
                            tracker.remove(APP_DIR, r["symbol"], h)
                            st.rerun()
                    else:
                        if st.button("➕ Bucket", key=f"t_{h}_{r['symbol']}", width="stretch"):
                            tracker.add(APP_DIR, r)
                            st.toast(f"Added {r['name']}")
                            st.rerun()

    sells = sorted([r for r in results[h] if r["signal"].startswith("SELL") and price_lo <= r["price"] < price_hi],
                   key=lambda r: -r["score"])[:12]
    if sells:
        st.markdown("<div class='seclabel'>🚫 Stay away (trend down)</div>", unsafe_allow_html=True)
        st.write(" · ".join(f"**{r['name']}** ({r['change']:+.1f}%)" for r in sells))

    with st.expander("📋 Full list of all stocks (table)"):
        df = pd.DataFrame([{"Stock": r["name"], "Symbol": r["symbol"], "Sector": data.sector_of(r["symbol"]),
                            "Call": r["signal"], "Score": int(r["score"]), "Price ₹": round(r["price"], 2),
                            "Chance %": round(r["chance"]), "Entry ₹": round(r["entry"], 2),
                            "Stop ₹": round(r["stop"], 2), "Target ₹": round(r["target"], 2)}
                           for r in sorted(results[h], key=lambda r: -r["bull"])
                           if price_lo <= r["price"] < price_hi])
        st.dataframe(df, hide_index=True, width="stretch")

    with st.expander("📄 See the full report here (no download needed)"):
        try:
            components.html(snapshot_html, height=650, scrolling=True)
        except Exception:
            st.download_button("⬇️ Download report", snapshot_html,
                               file_name=f"stock_report_{datetime.now():%Y%m%d_%H%M}.html", mime="text/html")

# =============================================================================
# PAGE: SECTORS
# =============================================================================
elif page == P_SECTOR:
    hero("📊 Sector Strength", "Where is the market strong right now? Don't put all your money in one sector.")
    back_button()
    regime_line()
    h = st.segmented_control("Trade style", list(engine.HORIZONS),
                             format_func=lambda x: HZ_SHORT[x], default="Weekly", key="sec_hz") or "Weekly"

    # aggregate by sector
    agg = {}
    for r in results[h]:
        sec = data.sector_of(r["symbol"])
        a = agg.setdefault(sec, {"buy": 0, "sell": 0, "chances": [], "buys": [], "n": 0})
        a["n"] += 1
        if r["signal"] == "BUY" and r["bt_trades"] >= 6 and r["bt_avg"] > 0:
            a["buy"] += 1
            a["chances"].append(r["chance"])
            a["buys"].append(r)
        elif r["signal"].startswith("SELL"):
            a["sell"] += 1

    def strength(a):
        if a["n"] == 0:
            return -999
        return (a["buy"] - a["sell"]) / a["n"] * 100 + (np.mean(a["chances"]) if a["chances"] else 0) * 0.3

    ordered = sorted([(s, a) for s, a in agg.items() if s != "Other"], key=lambda x: -strength(x[1]))
    st.caption("Sectors with the most BUY signals and best average chance are on top.")
    for sec, a in ordered:
        if a["n"] == 0:
            continue
        avg_ch = np.mean(a["chances"]) if a["chances"] else 0
        tone = "b-hi" if a["buy"] > a["sell"] and avg_ch >= 55 else "b-lo" if a["sell"] > a["buy"] else "b-mid"
        bar = max(2, min(100, (a["buy"] / a["n"] * 100)))
        colr = "var(--green)" if tone == "b-hi" else "var(--red)" if tone == "b-lo" else "var(--amber)"
        names = " · ".join(f"{x['name']} ({x['chance']:.0f}%)" for x in sorted(a["buys"], key=lambda r: -r["chance"])[:6])
        st.markdown(f"""<div class='card'>
<h4>{sec} <span class='badge {tone}'>{a['buy']} buy · {a['sell']} avoid</span>
<span class='badge b-info'>avg chance {avg_ch:.0f}%</span></h4>
<div class='meter'><span style='width:{bar:.0f}%;background:{colr}'></span></div>
<div class='sub' style='margin-top:8px'>{names or 'No strong buy right now.'}</div>
</div>""", unsafe_allow_html=True)
        if a["buys"]:
            cols = st.columns(min(4, len(a["buys"][:4])))
            for i, x in enumerate(sorted(a["buys"], key=lambda r: -r["chance"])[:4]):
                cols[i].button(f"{x['symbol']} ➜", key=f"sec_{h}_{x['symbol']}", on_click=go_to,
                               args=(x["symbol"], h), width="stretch")

# =============================================================================
# PAGE: MY BUCKET (tracking)
# =============================================================================
elif page == P_BUCKET:
    hero("📌 My Bucket", "Stocks you're tracking. Buy point, sell/exit point, live profit/loss and what to do now.")
    back_button()
    items = tracker.load(APP_DIR)
    if not items:
        st.info("Your bucket is empty. Add stocks from **Best Stocks** or **Check a Stock** "
                "(look for the ➕ Bucket button). The app will then track each one and alert you.")
    else:
        st.caption(f"{len(items)} stock(s) tracked. Live price refreshes below.")
        order = {"Intraday": 0, "Weekly": 1, "Monthly": 2}
        for it in sorted(items, key=lambda x: (order.get(x["horizon"], 9), x["symbol"])):
            lv = load_live(it["symbol"], demo or is_demo)
            price = lv["price"] if lv else it.get("price_at_add")
            chg = lv["chg"] if lv and lv.get("chg") is not None else None
            stt = tracker.status(it, price, chg, engine.ist_now().date())
            tone = {"success": "b-hi", "error": "b-lo", "warning": "b-mid", "info": "b-info"}.get(stt["level"], "b-info")
            pnl_html = ""
            if stt["pnl_pct"] is not None:
                cls = "pnl-up" if stt["pnl_pct"] >= 0 else "pnl-dn"
                amt = f" (₹{stt['pnl_amt']:,.0f})" if stt["pnl_amt"] is not None else ""
                pnl_html = f"<span class='{cls}'>{stt['pnl_pct']:+.1f}%{amt}</span>"
            pstr = f"₹{price:,.2f}" if price else "—"
            wl = (f"{stt['window_left']} trading day(s) left" if not stt["window_over"]
                  else "window over — time to exit")
            st.markdown(f"""<div class='card'>
<h4>{it['name']} <span class='badge {tone}'>{stt['verdict']}</span>
<span class='badge b-sec'>{HZ_SHORT[it['horizon']]}</span></h4>
<div class='sub'>{it['symbol']} · now {pstr} {pnl_html} · added {it['added_date']} · {wl}</div>
<div class='row3'><div class='r-buy'>Buy point<b>₹{stt['buy_point']:,.1f}</b></div>
<div class='r-sl'>Stop-loss (sell)<b>₹{stt['stop']:,.1f}</b></div>
<div class='r-tg'>Target (sell)<b>₹{stt['target']:,.1f}</b></div></div>
<div class='sub' style='margin-top:8px'>👉 {stt['msg']}</div>
</div>""", unsafe_allow_html=True)
            c1, c2 = st.columns(2)
            c1.button("Full plan ➜", key=f"bk_go_{it['symbol']}_{it['horizon']}", on_click=go_to,
                      args=(it["symbol"], it["horizon"]), width="stretch")
            if c2.button("🗑️ Remove", key=f"bk_rm_{it['symbol']}_{it['horizon']}", width="stretch"):
                tracker.remove(APP_DIR, it["symbol"], it["horizon"])
                st.rerun()

        # live intraday mini-panel
        intr = [it for it in items if it["horizon"] == "Intraday"]
        if intr and not (demo or is_demo):
            st.divider()
            st.markdown("<div class='seclabel'>⚡ Live intraday watch (auto-refresh)</div>", unsafe_allow_html=True)

            @st.fragment(run_every=30)
            def live_intraday_panel():
                st.caption(f"Updated {engine.ist_now():%I:%M:%S %p} · {status_txt}")
                for it in intr:
                    lv = load_live(it["symbol"], False)
                    price = lv["price"] if lv else it.get("price_at_add")
                    chg = lv["chg"] if lv and lv.get("chg") is not None else 0.0
                    df5, _ = load_intraday(it["symbol"], False, it.get("price_at_add", 1000))
                    il = engine.intraday_live(df5)
                    bias = il["bias"] if il else "No intraday data yet."
                    stt = tracker.status(it, price, chg, engine.ist_now().date())
                    cls = "up" if chg >= 0 else "dn"
                    st.markdown(f"**{it['name']}** — ₹{price:,.2f} "
                                f"<span class='{cls}'>{chg:+.2f}%</span> · {bias}", unsafe_allow_html=True)
                    getattr(st, stt["level"])(stt["msg"])
            live_intraday_panel()

# =============================================================================
# PAGE: WHAT CHANGED (history)
# =============================================================================
elif page == P_TRACK:
    hero("🔁 What Changed Since Last Time", "The app remembers your picks each trading day and shows the difference.")
    back_button()
    regime_line()
    if not prev_snap:
        st.info("📅 Only today's record exists so far. **Next time** you open the app, you'll see the full "
                "difference from last time. (A record is saved only when real data loads.)")
        st.caption(f"Today's record saved: {today_str}")
    else:
        for h in engine.HORIZONS:
            st.subheader(HZ_SHORT[h])
            render_changes(h)
            st.divider()

# =============================================================================
# PAGE: CHECK A STOCK
# =============================================================================
elif page == P_CHECK:
    hero("🔍 Check a Stock", "Any NSE stock — get a clear BUY / DON'T BUY / WAIT with levels and a chart.")
    back_button()
    syms = ALL_NSE if ALL_NSE else sorted(daily.keys())
    if st.session_state.sel_sym not in syms:
        st.session_state.sel_sym = "RELIANCE" if "RELIANCE" in syms else syms[0]
    sym = st.selectbox("Type or pick a stock 👇", syms, key="sel_sym",
                       format_func=lambda s: f"{data.NAMES.get(s, s)}  ({s})")
    h = st.segmented_control("For how long?", list(engine.HORIZONS), key="hz",
                             format_func=lambda x: HZ_LABEL[x]) or "Weekly"
    r = next((x for x in results[h] if x["symbol"] == sym), None)
    if r is None:
        with st.spinner(f"Fetching {data.NAMES.get(sym, sym)}…"):
            od = analyse_one(sym, demo or is_demo, fetched)
        if not od or h not in od:
            st.error("Couldn't get 2 years of data for this stock (maybe newly listed or wrong symbol). Try another one.")
            st.stop()
        r = od[h]
        if sym not in daily and od.get("_raw") is not None:
            daily[sym] = od["_raw"]

    # ---- live price + verdict (auto refresh 30s)
    @st.fragment(run_every=30)
    def live_box():
        lv = load_live(sym, demo or is_demo)
        price = lv["price"] if lv else r["price"]
        chg = lv["chg"] if lv and lv.get("chg") is not None else r["change"]
        vwap = None
        if h == "Intraday":
            df5, _ = load_intraday(sym, demo or is_demo, r['price'])
            il = engine.intraday_live(df5)
            vwap = il["vwap"] if il else None
        level, msg = engine.live_verdict(r, price, vwap, status)
        c1, c2 = st.columns([1, 2])
        with c1:
            cls = "up" if chg >= 0 else "dn"
            st.markdown(f"<div class='sub'>{data.NAMES.get(sym, sym)} · LIVE</div>"
                        f"<div class='price'>₹{price:,.2f} <span class='{cls}' style='font-size:1.1rem'>{chg:+.2f}%</span></div>"
                        f"<div class='sub'>Updated {engine.ist_now():%I:%M:%S %p} · every 30s · Yahoo may lag 1-15 min</div>",
                        unsafe_allow_html=True)
        with c2:
            getattr(st, level)(f"**What to do right now?** {msg}")
            if h == "Intraday":
                st.caption(f"🎲 Current chance: **{r['chance']:.0f}% ({chance_word(r['chance'])})** · "
                           f"Square off by **3:10 PM** (don't carry intraday overnight).")
    live_box()

    # ---- big verdict
    kind = "buy" if r["signal"] == "BUY" else "sell" if r["signal"].startswith("SELL") else "wait"
    if kind == "buy":
        big, msg = "✅ BUY", (f"Trend and momentum are together. Buy {'above' if h == 'Intraday' else 'near'} "
                             f"₹{r['entry']:,.2f}, put a stop-loss at ₹{r['stop']:,.2f}, book profit at ₹{r['target']:,.2f}.")
    elif kind == "sell":
        if h == "Intraday":
            big, msg = "🔻 SHORT SELL (intraday only)", (f"Stock is weak. Short if it breaks below ₹{r['entry']:,.2f}, "
                                                        f"target ₹{r['target']:,.2f}, stop-loss ₹{r['stop']:,.2f}. Beginners avoid.")
        else:
            big, msg = "❌ DON'T BUY / SELL", "Trend is down. Don't buy fresh. If you already hold it, consider selling."
    else:
        big, msg = "⏸️ WAIT", "No clear signal yet. Don't force a trade — check again tomorrow."
    st.markdown(f"<div class='verdict v-{kind}'><div class='big'>{big}</div>"
                f"<div class='msg'>{msg}</div><div class='sub'>Signal strength: {r['score']:.0f}/100 "
                f"(65+ = signal) · {HZ_SHORT[h]} · data {r['asof']}</div></div>", unsafe_allow_html=True)

    # add-to-bucket row
    bc1, bc2 = st.columns([1, 2])
    with bc1:
        track_button(r, f"check_{sym}_{h}")
    with bc2:
        if tracker.is_tracked(APP_DIR, sym, h):
            st.caption("✅ In your bucket — tracked on the 📌 My Bucket page with live updates & alerts.")
        else:
            st.caption("Add to bucket to track live P&L and get exit alerts for this stock.")

    # ---- plan boxes
    qty, invest, max_loss, max_profit = money_plan(r)
    b1, b2, b3, b4 = st.columns(4)
    b1.markdown(f"<div class='box r-buy'><div class='lbl'>🛒 {'Buy trigger' if h=='Intraday' else 'Buy price'}</div>"
                f"<div class='val'>₹{r['entry']:,.2f}</div><div class='hint'>{'Buy when price crosses above' if h=='Intraday' else 'Buy around here'}</div></div>", unsafe_allow_html=True)
    b2.markdown(f"<div class='box r-sl'><div class='lbl'>🛑 Stop-loss (exit here)</div>"
                f"<div class='val'>₹{r['stop']:,.2f}</div><div class='hint'>If price hits this, sell immediately — keep the loss small</div></div>", unsafe_allow_html=True)
    b3.markdown(f"<div class='box r-tg'><div class='lbl'>🎯 Target (book profit)</div>"
                f"<div class='val'>₹{r['target']:,.2f}</div><div class='hint'>Sell here, or sell half</div></div>", unsafe_allow_html=True)
    if qty > 0:
        hint4 = f"Needs ₹{invest:,.0f} · max loss ₹{max_loss:,.0f} · possible profit ₹{max_profit:,.0f}"
    else:
        hint4 = "This share is pricey for your capital/risk. Raise capital or risk % in Settings, or pick a cheaper stock."
    b4.markdown(f"<div class='box' style='background:linear-gradient(180deg,var(--card),var(--card2))'><div class='lbl'>💰 How many shares</div>"
                f"<div class='val'>{qty:,}</div><div class='hint'>{hint4}</div></div>", unsafe_allow_html=True)

    # ---- chance
    st.markdown("<div class='seclabel'>🎲 What's the chance of profit?</div>", unsafe_allow_html=True)
    if r["bt_trades"] == 0:
        st.info("This signal never appeared on this stock in the last 2 years, so there's no record.")
    else:
        c1, c2 = st.columns([1, 2])
        c1.markdown(f"<div class='price'>{r['chance']:.0f}%</div>{chance_badge(r['chance'])}", unsafe_allow_html=True)
        c2.markdown(f"Over the last 2 years this **same signal came {r['bt_trades']} times**. "
                    f"**{r['wins']} were profitable**, **{r['bt_trades'] - r['wins']} were losses**. "
                    f"Average per trade: **{r['bt_avg']:+.2f}%**.")
        c2.progress(min(1.0, max(0.0, r["chance"] / 100)))
        c2.caption("Chance is balanced against the overall market record, so a lucky 3-of-3 doesn't show as 100%. "
                   "55-65% chance + a stop-loss = a very good system. Nobody can honestly promise 90%.")

    # ---- why
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("<div class='seclabel'>👍 In favour</div>", unsafe_allow_html=True)
        if r["pros"]:
            for _, t in r["pros"]:
                st.markdown(f"- ✅ {t}")
        else:
            st.caption("No strong point.")
    with c2:
        st.markdown("<div class='seclabel'>👎 What's missing</div>", unsafe_allow_html=True)
        if r["cons"]:
            for _, t in r["cons"]:
                st.markdown(f"- ➖ {t}")
        else:
            st.caption("All points are in favour.")
    st.markdown("<div class='seclabel'>⏰ When to buy, when to sell</div>", unsafe_allow_html=True)
    for t in r["timing"]:
        st.markdown(f"- {t}")

    # ---- chart (unchanged logic)
    st.divider()
    d_all = engine.add_indicators(daily[sym])

    def level_lines(fig):
        fig.add_hrect(y0=min(r["entry"], r["target"]), y1=max(r["entry"], r["target"]), fillcolor="rgba(34,197,94,.10)", line_width=0)
        fig.add_hrect(y0=min(r["entry"], r["stop"]), y1=max(r["entry"], r["stop"]), fillcolor="rgba(248,113,113,.10)", line_width=0)
        for y, nm, col in ((r["entry"], "Buy", "#60a5fa"), (r["stop"], "Stop-loss", "#f87171"), (r["target"], "Target", "#22c55e")):
            fig.add_hline(y=y, line_dash="dash", line_color=col, annotation_text=f"{nm} ₹{y:,.1f}", annotation_position="right")

    def markers(fig, d_view, hz):
        buys, sells = engine.signal_markers(d_all, hz, last_n=len(d_view) if hz != "Monthly" else 520)
        buys = [x for x in buys if x >= d_view.index[0]]
        sells = [x for x in sells if x >= d_view.index[0]]
        if buys:
            fig.add_trace(go.Scatter(x=buys, y=d_all.loc[buys, "Low"] * 0.97, mode="markers+text", text=["BUY"] * len(buys),
                                     textposition="bottom center", marker=dict(symbol="triangle-up", size=13, color="#22c55e"),
                                     name="Past BUY signal"))
        if sells:
            fig.add_trace(go.Scatter(x=sells, y=d_all.loc[sells, "High"] * 1.03, mode="markers+text", text=["SELL"] * len(sells),
                                     textposition="top center", marker=dict(symbol="triangle-down", size=13, color="#f87171"),
                                     name="Past SELL signal"))

    layout = dict(height=560, xaxis_rangeslider_visible=False, margin=dict(l=10, r=90, t=10, b=10),
                  legend=dict(orientation="h", y=1.06), hovermode="x unified",
                  paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")

    if h == "Intraday":
        st.markdown("<span class='chart-tag'>📊 INTRADAY chart · each candle = 5 minutes · today only</span>", unsafe_allow_html=True)
        df5, _ = load_intraday(sym, demo or is_demo, r['price'])
        il = engine.intraday_live(df5)
        if il is None:
            st.info("No intraday data right now (market closed / holiday).")
        else:
            t = il["bars"]
            fig = go.Figure(go.Candlestick(x=t.index, open=t.Open, high=t.High, low=t.Low, close=t.Close, name="5-min candle",
                                           increasing_line_color="#22c55e", decreasing_line_color="#f87171"))
            fig.add_trace(go.Scatter(x=t.index, y=t.vwap, name="VWAP (today's avg)", line=dict(color="#fbbf24", width=2)))
            fig.add_hline(y=il["orb_hi"], line_dash="dot", line_color="#86efac", annotation_text="First 15-min HIGH", annotation_position="left")
            fig.add_hline(y=il["orb_lo"], line_dash="dot", line_color="#fca5a5", annotation_text="First 15-min LOW", annotation_position="left")
            level_lines(fig)
            fig.update_layout(**layout)
            st.plotly_chart(fig, width="stretch")
            st.markdown(f"**What the chart says now:** {il['bias']}")
        with st.expander("🧑‍🏫 How to read this intraday chart", expanded=True):
            st.markdown("""
- **Each candle = 5 minutes.** 🟢 green = price went up in those 5 min, 🔴 red = went down.
- **Yellow line (VWAP)** = today's average price. Price **above** = buyers strong 💪, **below** = sellers strong.
- **Green dotted line** = first 15-minute high (9:15-9:30). A break above it = momentum starting.
- **Blue dashed line (Buy)** = trigger. Price above it **and** above VWAP → **BUY**.
- **Red line (Stop-loss)** = sell immediately if hit. **Green line (Target)** = book profit.
- **When to sell:** target, stop-loss, or **3:10 PM** — whichever comes first.
- No new trade after 2:30 PM. Two losses in a day → stop for the day.
""")
    elif h == "Weekly":
        st.markdown("<span class='chart-tag'>📊 WEEKLY chart · each candle = 1 day · last 6 months</span>", unsafe_allow_html=True)
        dv = d_all.iloc[-130:]
        fig = go.Figure(go.Candlestick(x=dv.index, open=dv.Open, high=dv.High, low=dv.Low, close=dv.Close, name="Daily candle",
                                       increasing_line_color="#22c55e", decreasing_line_color="#f87171"))
        fig.add_trace(go.Scatter(x=dv.index, y=dv.ema21, name="21-day line", line=dict(color="#fbbf24", width=1.6)))
        fig.add_trace(go.Scatter(x=dv.index, y=dv.ema50, name="50-day line", line=dict(color="#60a5fa", width=1.6)))
        markers(fig, dv, "Weekly")
        level_lines(fig)
        fig.update_layout(**layout)
        st.plotly_chart(fig, width="stretch")
        with st.expander("🧑‍🏫 How to read this weekly chart", expanded=True):
            st.markdown("""
- **Each candle = 1 day.** 🟢 green = closed up that day, 🔴 red = closed down.
- **Orange line = 21-day average**, **blue line = 50-day average**.
- **Uptrend:** price **above** both lines and orange above blue → good to buy ✅.
- **Downtrend:** price **below** both lines → stay away ❌.
- **▲ green (BUY)** and **▼ red (SELL)** arrows = past signals. See what happened after — that's where 'chance %' comes from.
- **Best entry:** in an uptrend, when price **dips to the orange line** and a green candle forms there.
- **When to sell:** target (green line), stop-loss (red line), or after 5 days — whichever first.
""")
    else:
        st.markdown("<span class='chart-tag'>📊 MONTHLY chart · each candle = 1 week · last 2 years</span>", unsafe_allow_html=True)
        wk = engine.to_weekly(daily[sym])
        w = wk.iloc[-104:].copy()
        w["ma10"] = wk.Close.rolling(10).mean().iloc[-104:]
        w["ma40"] = wk.Close.rolling(40).mean().iloc[-104:]
        fig = go.Figure(go.Candlestick(x=w.index, open=w.Open, high=w.High, low=w.Low, close=w.Close, name="Weekly candle",
                                       increasing_line_color="#22c55e", decreasing_line_color="#f87171"))
        fig.add_trace(go.Scatter(x=w.index, y=w.ma10, name="10-week line (~50 days)", line=dict(color="#fbbf24", width=1.8)))
        fig.add_trace(go.Scatter(x=w.index, y=w.ma40, name="40-week line (~200 days)", line=dict(color="#c084fc", width=1.8)))
        markers(fig, d_all[d_all.index >= w.index[0]], "Monthly")
        level_lines(fig)
        fig.update_layout(**layout)
        st.plotly_chart(fig, width="stretch")
        with st.expander("🧑‍🏫 How to read this monthly chart", expanded=True):
            st.markdown("""
- **Each candle = 1 week.** Gives the big picture without daily noise.
- **Orange line ≈ 50 days**, **purple line ≈ 200 days** average.
- **Long uptrend:** price above both lines and orange above purple → best for monthly BUY ✅.
- **Price below the purple line** = long trend weak → don't buy for the long term ❌.
- **▲/▼ arrows** = past monthly BUY/SELL signals. See the next 4 weeks after each.
- **How to buy:** in 2-3 parts. First now, the rest on dips toward the orange line.
- **When to sell:** target, stop-loss, or about 1 month (20 trading days).
""")

    if not (demo or is_demo):
        with st.expander("🏢 Company details (fundamentals)"):
            f = load_fundamentals(sym)
            if f:
                st.table(pd.DataFrame(f.items(), columns=["Item", "Value"]))
            else:
                st.caption("Not available right now.")

# =============================================================================
# PAGE: TELEGRAM ALERTS
# =============================================================================
elif page == P_ALERTS:
    hero("📲 Telegram Alerts", "Get a message on Telegram when a tracked stock hits target, stop-loss, a sharp fall, or its exit time.")
    back_button()
    cfg = notify.load_config(APP_DIR)
    token = cfg.get("bot_token", "")

    st.markdown("<div class='seclabel'>Step 1 — Bot token (one-time, by the owner)</div>", unsafe_allow_html=True)
    st.caption("Create a free bot in Telegram: open **@BotFather** → send `/newbot` → follow the steps → copy the token it gives. "
               "Paste it here once. (You can also set it via the TELEGRAM_BOT_TOKEN environment variable.)")
    new_token = st.text_input("Bot token", value=token, type="password", placeholder="123456789:ABCdef...")
    if st.button("💾 Save token"):
        cfg["bot_token"] = new_token.strip()
        notify.save_config(APP_DIR, cfg)
        st.success("Token saved.")
        st.rerun()

    if not token:
        st.info("Add the bot token above to enable Telegram alerts.")
    else:
        uname = notify.bot_username(token)
        st.markdown("<div class='seclabel'>Step 2 — Add your number (chat ID)</div>", unsafe_allow_html=True)
        if uname:
            st.markdown(f"**A.** Open your bot and press **Start**: 👉 [t.me/{uname}](https://t.me/{uname}) "
                        "(do this in the Telegram app, from the phone/account you want alerts on).")
        else:
            st.warning("Couldn't reach Telegram with this token. Check the token is correct.")
        st.markdown("**B.** Then press the button below — it reads who just pressed Start and adds them automatically.")
        if st.button("🔗 Find & connect my Telegram"):
            found = notify.recent_chat_ids(token)
            if not found:
                st.warning("No one found. Open the bot link above, press **Start**, send any message, then click this again.")
            else:
                for f in found:
                    notify.add_subscriber(APP_DIR, f["chat_id"], f["name"])
                st.success(f"Connected: {', '.join(f['name'] for f in found)}")
                st.rerun()

        st.markdown("<div class='seclabel'>Or add a chat ID number by hand</div>", unsafe_allow_html=True)
        colA, colB = st.columns([2, 1])
        manual = colA.text_input("Chat ID (a number)", placeholder="e.g. 123456789", label_visibility="collapsed")
        if colB.button("➕ Add number", width="stretch"):
            try:
                notify.add_subscriber(APP_DIR, int(manual.strip()))
                st.success("Added.")
                st.rerun()
            except Exception:
                st.error("Please enter a valid numeric chat ID.")

        # subscribers list
        cfg = notify.load_config(APP_DIR)
        subs = cfg.get("subscribers", [])
        st.markdown("<div class='seclabel'>Connected people</div>", unsafe_allow_html=True)
        if not subs:
            st.caption("Nobody connected yet.")
        for s in subs:
            c1, c2, c3 = st.columns([2, 1, 1])
            c1.markdown(f"**{s.get('name', s['chat_id'])}** · `{s['chat_id']}`")
            if c2.button("✉️ Test", key=f"test_{s['chat_id']}", width="stretch"):
                ok = notify.send_test(APP_DIR, s["chat_id"])
                st.toast("Sent! Check Telegram." if ok else "Failed — check token/chat.")
            if c3.button("🗑️ Remove", key=f"rm_{s['chat_id']}", width="stretch"):
                notify.remove_subscriber(APP_DIR, s["chat_id"])
                st.rerun()

        enabled = st.toggle("Alerts enabled", value=cfg.get("enabled", True))
        if enabled != cfg.get("enabled", True):
            cfg["enabled"] = enabled
            notify.save_config(APP_DIR, cfg)
            st.rerun()

        st.divider()
        st.markdown("<div class='seclabel'>Alerts while the app is CLOSED</div>", unsafe_allow_html=True)
        st.markdown(
            "This app only runs while a browser tab is open. To get alerts even when it's closed, run the "
            "**background watchdog** on this PC: double-click **`Start Watchdog.bat`** in your StockAdvisor folder "
            "and keep that small black window open. It checks your bucket every few minutes and pings Telegram on "
            "sudden falls, stop-loss, target, and when a holding window is over.")

# =============================================================================
# PAGE: LEARN
# =============================================================================
elif page == P_LEARN:
    hero("📚 Learn — simple words", "Candles, trend, stop-loss, risk — the basics you actually need.")
    back_button()
    st.markdown("""
### 1️⃣ How to use this app (3 steps)
1. Open **🏠 Best Stocks** → pick a style: Intraday / Weekly / Monthly.
2. On any card press **Full plan** → the app tells you **BUY / DON'T BUY / WAIT**.
3. In your broker app (Zerodha, Groww, Upstox…) place the same **Buy price, Stop-loss, Target**. **Always put the stop-loss order.**

### 2️⃣ What is a candle? 🕯️
- Each candle covers one time block (5 min / 1 day / 1 week).
- 🟢 **Green** = closed higher than it opened (price up). 🔴 **Red** = closed lower (price down).
- The thin lines above/below (wicks) = the highest and lowest price in that time.

### 3️⃣ Trend — the most important thing 📈
- **"Trend is your friend."** Buy stocks going up, not ones falling.
- Look at the average lines (21, 50, 200 day). Price **above** = uptrend, **below** = downtrend.

### 4️⃣ Support and Resistance 🧱
- **Support** = a price the stock keeps bouncing up from (the floor).
- **Resistance** = a price it keeps falling from (the ceiling).
- Closing above resistance = **breakout** (a good buy chance).

### 5️⃣ Stop-loss and Target 🛑🎯
- **Stop-loss** = "if I'm wrong, I exit here." It keeps losses small.
- **Target** = "I'll book profit here."
- In this app the target is about **2× further** than the stop-loss. So even 4 wins out of 10 can still make money.

### 6️⃣ How much money to use? 💰
- Never risk more than **1-2%** of your total money on one trade.
- Example: ₹1,00,000, 1% risk = ₹1,000. If stop-loss is ₹10 away → buy **100 shares**. The app does this for you.
- Never put all your money in one stock.

### 7️⃣ Intraday vs Weekly vs Monthly
| | ⚡ Intraday | 📅 Weekly | 🗓️ Monthly |
|---|---|---|---|
| Hold time | Same day | 1 week | ~1 month |
| Chart | 5-min candle | Daily candle | Weekly candle |
| Risk | Highest | Medium | Lowest |
| Good for | Full-day screen watchers | Working people | Beginners ✅ |

### 8️⃣ Why no 90% guarantee? 🤝
- News, results and global events move markets — no chart knows these in advance.
- Even the world's best traders are right only 50-65% of the time. They win because **profits are big and losses are small**.
- Anyone promising 90% sure profit — **it may be a scam**, stay away.
- This app shows **honest numbers** so you can pick the best-chance stocks.

### 9️⃣ Golden rules ⭐
- Always use a **stop-loss**. ✋
- When the **market mood** is 🔴, trade less.
- Avoid new trades just before a company's **results**.
- After a loss, don't take a big "revenge" trade.
- Write your trades in a diary/Excel — review after a month to see what works.
""")

# =============================================================================
# PAGE: ACCURACY
# =============================================================================
elif page == P_ACC:
    hero("🎯 How Accurate Is This App?", "Every BUY/SELL rule tested on each stock's last 2 years — the honest record.")
    back_button()
    st.write("Each rule was replayed on every stock's **last 2 years** (after a 0.1% cost). "
             "If both stop-loss and target were hit on the same day, it was counted as a **loss** (safe side).")

    @st.cache_data(ttl=900, show_spinner="Running the test on all stocks…")
    def all_returns(universe, extra, demo, _stamp):
        dd, *_ = load_daily(universe, extra, demo)
        out = {}
        for _, raw in dd.items():
            d = engine.add_indicators(raw)
            for hh in engine.HORIZONS:
                for side in ("BUY", "SELL"):
                    out.setdefault((hh, side), []).extend(engine.backtest(d, hh, side).returns)
        return out

    allr = all_returns(universe, extra, demo, fetched)
    rows = []
    for hh in engine.HORIZONS:
        for side in ("BUY", "SELL"):
            arr = np.array(allr.get((hh, side), []))
            if len(arr):
                rows.append({"Type": HZ_SHORT[hh], "Signal": side, "Trades": len(arr),
                             "Profitable %": round((arr > 0).mean() * 100, 1),
                             "Avg per trade %": round(arr.mean(), 2),
                             "Avg profit %": round(arr[arr > 0].mean(), 2) if (arr > 0).any() else 0,
                             "Avg loss %": round(arr[arr <= 0].mean(), 2) if (arr <= 0).any() else 0})
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    st.info("**How to read it:** if 'Avg per trade %' is **positive**, the system made money over time — even if fewer "
            "than 50% of trades won, because profits are kept bigger than losses. The Best Stocks page only shows stocks "
            "whose own record is also positive.")

st.markdown("<div class='sub' style='margin-top:30px;color:var(--mut)'>Education/research tool, not SEBI-registered advice · "
            "Past record is not a guarantee of the future.</div>", unsafe_allow_html=True)
