"""
Stock Advisor - simple Hinglish NSE stock helper.
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

APP_DIR = os.path.dirname(os.path.abspath(__file__))

st.set_page_config(page_title="Stock Advisor", page_icon="📈", layout="wide")

st.markdown("""<style>
.block-container{padding-top:1.2rem;max-width:1250px}
.card{border:1px solid rgba(128,128,128,.28);border-radius:14px;padding:14px 16px;margin-bottom:6px;background:rgba(128,128,128,.06)}
.card h4{margin:0 0 2px 0;font-size:1.05rem}
.card .sub{opacity:.7;font-size:.82rem;margin-bottom:8px}
.badge{display:inline-block;padding:2px 9px;border-radius:999px;font-size:.78rem;font-weight:600;margin-right:4px}
.b-hi{background:rgba(22,163,74,.18);color:#22c55e}.b-mid{background:rgba(234,179,8,.18);color:#eab308}
.b-lo{background:rgba(239,68,68,.16);color:#f87171}
.row3{display:flex;gap:8px;margin-top:8px}.row3 div{flex:1;border-radius:10px;padding:6px 8px;font-size:.82rem}
.r-buy{background:rgba(59,130,246,.14)}.r-sl{background:rgba(239,68,68,.14)}.r-tg{background:rgba(22,163,74,.14)}
.row3 b{display:block;font-size:1rem}
.verdict{border-radius:16px;padding:18px 22px;margin:6px 0 10px 0}
.v-buy{background:rgba(22,163,74,.16);border:2px solid rgba(22,163,74,.55)}
.v-sell{background:rgba(239,68,68,.14);border:2px solid rgba(239,68,68,.5)}
.v-wait{background:rgba(128,128,128,.12);border:2px solid rgba(128,128,128,.4)}
.verdict .big{font-size:2rem;font-weight:800;line-height:1.1}
.verdict .msg{font-size:1.02rem;margin-top:6px}
.price{font-size:2.1rem;font-weight:800}
.up{color:#22c55e}.dn{color:#f87171}
.box{border-radius:14px;padding:12px 14px;height:100%}
.box .lbl{font-size:.85rem;opacity:.8}.box .val{font-size:1.5rem;font-weight:800}.box .hint{font-size:.8rem;opacity:.75}
.chart-tag{display:inline-block;background:rgba(59,130,246,.18);color:#60a5fa;padding:4px 12px;border-radius:999px;font-weight:700}
/* phone-friendly: smaller fonts & padding on narrow screens */
@media (max-width:640px){
  .block-container{padding-left:.6rem;padding-right:.6rem}
  .verdict .big{font-size:1.5rem}
  .price{font-size:1.6rem}
  .box .val{font-size:1.2rem}
  .card h4{font-size:1rem}
  .row3{flex-wrap:wrap}
}
</style>""", unsafe_allow_html=True)

HZ_LABEL = {"Intraday": "⚡ Intraday (aaj hi khareedo-becho)",
            "Weekly": "📅 Weekly (1 hafte tak rakho)",
            "Monthly": "🗓️ Monthly (1 mahine tak rakho)"}
HZ_SHORT = {"Intraday": "⚡ Intraday", "Weekly": "📅 Weekly", "Monthly": "🗓️ Monthly"}
HZ_EXPLAIN = {
    "Intraday": "Subah khareedo, shaam 3:10 se pehle becho. Chhota profit, tez faisla. Sabse zyada risk.",
    "Weekly": "Kharid ke 1 hafte (5 trading din) tak rakho. Target ya stop-loss jo pehle aaye.",
    "Monthly": "Kharid ke lagbhag 1 mahina rakho. Bada target, kam tension, beginners ke liye sabse achha.",
}
P_HOME = "🏠 Aaj ke Best Stocks"
P_TRACK = "🔁 Pichle picks (tracking)"
P_CHECK = "🔍 Stock Check Karo"
P_LEARN = "📚 Seekho (Hinglish)"
P_ACC = "🎯 App kitna sahi hai?"
PAGES = [P_HOME, P_TRACK, P_CHECK, P_LEARN, P_ACC]

for k, v in (("page", P_HOME), ("sel_sym", "RELIANCE"), ("hz", "Weekly")):
    st.session_state.setdefault(k, v)


def go_to(sym, hz):
    st.session_state.sel_sym = sym
    st.session_state.hz = hz
    st.session_state.page = P_CHECK


# --------------------------------------------------------------------------- data
@st.cache_data(ttl=900, show_spinner=False)
def load_daily(universe: str, extra: tuple, demo: bool):
    """Returns (daily_dict, nifty_df, is_demo, fetched_at).

    Only ever returns demo (fake) data when the user explicitly turns the demo
    toggle on. If real data can't be fetched we return an EMPTY result so the
    app shows a clear error instead of silently trading on fake numbers.
    """
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
                    # market is running: plan from YESTERDAY's completed candle
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
    # overall average win-rate per horizon/side -> used to make each stock's chance realistic
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
    """Fetch the full NSE equity list + Nifty 500 once a day (falls back to curated)."""
    allmap = data.nse_all_equities()
    n500 = data.nse_500()
    return allmap, n500


@st.cache_data(ttl=900, show_spinner=False)
def analyse_one(sym: str, demo: bool, _stamp):
    """Analyse a SINGLE stock on demand (for stocks not in the scanned universe)."""
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
        # simple standalone chance: shrink own win-rate toward 45% (no scan prior here)
        w = r["bt_trades"] * (r["bt_profit"] if r["bt_trades"] else 0) / 100
        r["chance"] = engine.adjusted_chance(w, r["bt_trades"], 45.0)
        r["wins"] = int(round(w))
        out[h] = r
    if out:
        out["_raw"] = raw
    return out or None


# --------------------------------------------------------------------------- sidebar
# build the stock universe (full NSE list, fetched & cached daily)
allmap, n500 = get_universe_map()
data.NAMES.update(allmap)                 # name lookups work for EVERY NSE stock
ALL_NSE = sorted(allmap.keys())
UNIV = {
    "⚡ Nifty 50 (sabse fast)": data.NIFTY50,
    f"Top {len(data.ALL_STOCKS)} bade stocks (fast)": data.ALL_STOCKS,
    f"Nifty 500 (~{len(n500)}) — medium": list(dict.fromkeys(n500)),
    f"🌐 Sab NSE stocks (~{len(ALL_NSE)}) — dheere": ALL_NSE,
}
data.UNIVERSES.clear()
data.UNIVERSES.update(UNIV)

PRICE_BANDS = {
    "Sab price (koi filter nahi)": (0.0, 1e12),
    "₹10 – 50": (10.0, 50.0),
    "₹50 – 100": (50.0, 100.0),
    "₹100 – 200": (100.0, 200.0),
    "₹200 – 500": (200.0, 500.0),
    "₹500 – 700": (500.0, 700.0),
    "₹700 – 1000": (700.0, 1000.0),
    "₹1000 se upar": (1000.0, 1e12),
}

st.sidebar.title("📈 Stock Advisor")
st.sidebar.radio("Kahan jaana hai?", PAGES, key="page")

# ---- price filter: always visible (not hidden in Settings) ----
price_band = st.sidebar.selectbox("💰 Price range filter (share ka daam)", list(PRICE_BANDS), index=0,
                                  help="Sirf is daam ke beech wale stocks dikhao. Jaise ₹100-200.")
price_lo, price_hi = PRICE_BANDS[price_band]

with st.sidebar.expander("⚙️ Settings", expanded=False):
    universe = st.selectbox("Kaunse stocks scan karein", list(data.UNIVERSES.keys()), index=1,
                            help="Zyada stocks = zyada time. 'Sab NSE' sabse poora par dheere (pehli baar 3-5 min).")
    if universe.startswith("🌐"):
        st.caption("⏳ 'Sab NSE' me ~2000 stocks – pehli baar 3-5 min lag sakte hain. Kisi ek ko turant "
                   "dekhna ho to **Stock Check Karo** page use karo (wahan koi bhi NSE stock chalta hai).")
    extra_txt = st.text_input("Aur NSE stock jodo (comma se)", placeholder="jaise: IDEA, SUZLON")
    extra = tuple(s.strip().upper() for s in extra_txt.split(",") if s.strip())
    demo = st.toggle("Demo mode (nakli data)", value=False)
    capital = st.number_input("Aapka trading paisa (₹)", 5000, 100_000_000, 100000, step=5000)
    risk_pct = st.slider("Ek trade me max kitna nuksaan sahenge (% of paisa)", 0.5, 3.0, 1.0, 0.25,
                         help="1% matlab ₹1,00,000 me se ek trade me max ₹1,000 ka nuksaan.")
if st.sidebar.button("🔄 Naya data lao (refresh)", width="stretch"):
    st.cache_data.clear()

with st.spinner("Saare stocks ka data aa raha hai aur check ho raha hai… pehli baar 1-2 minute lagenge ⏳"):
    daily, nifty, is_demo, fetched = load_daily(universe, extra, demo)
    # No real data and demo is OFF -> STOP. Never show fake numbers as real.
    if not demo and len(daily) == 0:
        st.title("📈 Stock Advisor")
        st.error("### ❌ Abhi asli market data nahi aaya\n\n"
                 "Internet ya office network ne Yahoo Finance ka data block kar diya lagta hai. "
                 "**Nakli (demo) data jaan-bujh ke nahi dikhaya ja raha** – taaki aap galat numbers pe trade na karo.")
        st.markdown("**Kya karein:**")
        st.markdown("- Sidebar me **🔄 Naya data lao** dabao (dobara koshish).\n"
                    "- **Ghar ka wifi / mobile hotspot** try karo (office network aksar Yahoo block karta hai).\n"
                    "- Sirf seekhne/testing ke liye: Sidebar → Settings → **Demo mode** on karo "
                    "(ye NAKLI data hai, ispe trade mat karna).")
        if st.button("🔄 Dobara koshish karo"):
            st.cache_data.clear()
            st.rerun()
        st.stop()
    results, regime = run_scan(universe, extra, demo, fetched)

# ---- data freshness: newest daily bar date across the universe ----
_last_dates = [engine.last_bar_date(df) for df in daily.values() if len(df)]
data_date = max([d for d in _last_dates if d], default=None)
age = (engine.ist_now().date() - data_date).days if data_date else None

st.sidebar.caption(f"Data pulled: {fetched:%d %b, %I:%M %p} · {len(daily)} stocks"
                   + (f" · candle {data_date:%d %b}" if data_date else ""))
snapshot_html = report.build_snapshot(results, regime, universe)
st.sidebar.download_button("⬇️ Report download (Downloads me save hogi)", snapshot_html,
                           file_name=f"stock_report_{datetime.now():%Y%m%d_%H%M}.html", mime="text/html",
                           width="stretch")
st.sidebar.caption("Download file aapke **Downloads** folder me jaati hai. Kholne ke liye us file pe "
                   "right-click → **Open with → Chrome/Edge**. Ya niche Home page pe **'Report yahin dekho'** use karo.")
st.sidebar.caption("⚠️ Ye SEBI registered advice nahi hai. Apna faisla khud lo, hamesha stop-loss lagao.")

status, status_txt = engine.market_status()

if is_demo:
    st.error("⚠️ DEMO / NAKLI DATA dikh raha hai – demo mode ON hai. "
             "In numbers pe trade mat karna. Settings me demo band karke 'Naya data lao' dabao.")
elif age is not None and age >= 4:
    st.warning(f"🕒 Data thoda purana hai (aakhri candle {data_date:%d %b}). "
               "Sidebar me **🔄 Naya data lao** dabao taaki aaj ke hisaab se suggestion mile.")


def money_plan(r):
    risk_per_share = abs(r["entry"] - r["stop"])
    qty = int((capital * risk_pct / 100) // risk_per_share) if risk_per_share > 0 else 0
    qty = min(qty, int(capital // r["entry"])) if r["entry"] > 0 else qty
    return qty, qty * r["entry"], qty * risk_per_share, qty * abs(r["target"] - r["entry"])


def chance_badge(ch):
    if ch >= 60:
        return "<span class='badge b-hi'>🔥 High chance</span>"
    if ch >= 52:
        return "<span class='badge b-mid'>👍 Theek chance</span>"
    return "<span class='badge b-lo'>⚠️ Kam chance</span>"


def in_band(r):
    return price_lo <= r["price"] < price_hi


def top_buys(h, n=9):
    rows = [r for r in results[h] if r["signal"] == "BUY" and r["bt_trades"] >= 6
            and r["bt_avg"] > 0 and in_band(r)]
    ok = regime.get("ok")
    if ok is False:
        # Weak market (Nifty below its long trend): BUYs fail more, so only keep
        # stocks that are STRONGER than the market and have a real edge.
        rows = [r for r in rows if r["rs"] > 0 and r["chance"] >= 52]
    elif ok is None:
        rows = [r for r in rows if r["rs"] > -3]
    rows.sort(key=lambda r: (-(r["chance"] + 0.05 * r["score"] + (2 if r["rs"] > 0 else 0))))
    return rows[:n]


def regime_line():
    ok = regime.get("ok")
    icon = "🟢" if ok else "🟡" if ok is None else "🔴"
    extra_ = f" · Nifty {regime['close']:,.0f} ({regime['chg']:+.2f}%)" if "close" in regime else ""
    st.markdown(f"{icon} **Market ka mood:** {regime.get('text')}{extra_}  \n🕘 {status_txt}")


# --------------------------------------------------------------------------- day-over-day tracking
# Build today's snapshot (picks shown + every stock's signal), save it once per
# trading day, and load the previous trading day's snapshot for comparison.
today_str = str(engine.ist_now().date())
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


def render_changes(h, compact=False):
    """Show what changed for horizon h vs the previous trading day."""
    diff = history.compare(prev_snap, curr_snap, h)
    if not diff["have_prev"]:
        st.info("📅 Ye pehla record hai. Kal se app aapko batayega ki pichle baar se kya **badla** – "
                "naye picks, abhi bhi achhe, aur kaun se nikal jaane chahiye.")
        return

    pd_ = diff["prev_date"]
    st.caption(f"Pichli baar ({pd_}) se abhi ({today_str}) tak ka farak:")

    exited = diff["exited"]
    if exited:
        st.markdown("#### 🚪 Ab in picks se nikal jao (data ab achha nahi)")
        for r in exited:
            res, note = r["outcome"]
            tag = {"win": "b-hi", "loss": "b-lo", "running": "b-mid"}.get(res, "b-mid")
            sig = r["today_signal"]
            why = "ab BUY nahi raha" if not r["still_buy"] else "ab top list me nahi"
            st.markdown(
                f"<div class='card'><h4>{r['name']} <span class='badge {tag}'>{note}</span></h4>"
                f"<div class='sub'>{r['symbol']} · pichli baar BUY tha, {why} (aaj: {sig}). "
                f"Naya mat kharido. Pehle se hai to apne stop-loss/target pe nikal jao.</div>"
                f"{_mini_levels(r['prev'])}</div>", unsafe_allow_html=True)

    newp = diff["new"]
    if newp:
        st.markdown("#### 🆕 Naye picks (pichli baar nahi the)")
        st.write(" · ".join(f"**{r['name']}**" for r in newp))

    cont = diff["continued"]
    if cont:
        st.markdown("#### ✅ Abhi bhi achhe (pichli baar bhi the)")
        for r in cont:
            res, note = (r["outcome"] or ("", ""))
            tag = {"win": "b-hi", "loss": "b-lo", "running": "b-mid"}.get(res, "b-mid")
            badge = f"<span class='badge {tag}'>{note}</span>" if note else ""
            st.markdown(f"<div class='sub'>• <b>{r['name']}</b> ({r['symbol']}) {badge}</div>",
                        unsafe_allow_html=True)

    if not (exited or newp or cont):
        st.success("Is type me pichli baar se koi badlav nahi – wahi picks chal rahe hain.")


# ============================================================================ HOME
if st.session_state.page == P_HOME:
    st.title("🏠 Aaj ke Best Stocks")
    regime_line()
    h = st.segmented_control("Aap kaisa trade karna chahte ho?", list(engine.HORIZONS),
                             format_func=lambda x: HZ_LABEL[x], default="Weekly", key="home_hz") or "Weekly"
    st.info(f"**{HZ_SHORT[h]}:** {HZ_EXPLAIN[h]}")
    _diff = history.compare(prev_snap, curr_snap, h)
    if _diff["have_prev"] and (_diff["exited"] or _diff["new"]):
        n_exit, n_new = len(_diff["exited"]), len(_diff["new"])
        with st.expander(f"🔁 Pichli baar se badlav: {n_new} naye · {n_exit} nikalne wale", expanded=bool(n_exit)):
            render_changes(h)
    if price_band != "Sab price (koi filter nahi)":
        st.caption(f"💰 Price filter ON: sirf **{price_band}** wale stocks dikh rahe hain. "
                   "(Sidebar → Settings me badlo ya hatao.)")
    picks = top_buys(h)
    if not picks:
        extra_note = (f" Is price range (**{price_band}**) me aaj kuch nahi mila – "
                      "range badlo ya 'Sab price' karo.") if price_band != "Sab price (koi filter nahi)" else ""
        st.warning("Aaj is type ke liye koi achha BUY stock nahi mila. **Paisa bachana bhi profit hai** – "
                   "dusra type try karo ya kal dekhna." + extra_note)
    else:
        st.subheader(f"✅ Kharidne layak stocks ({len(picks)})")
        st.caption("Sabse upar wale ka chance sabse zyada hai. 'Chance' = pichle 2 saal me isi signal ne kitni baar profit diya.")
        cols = st.columns(3)
        for i, r in enumerate(picks):
            with cols[i % 3]:
                up = (r["target"] / r["entry"] - 1) * 100
                dn = (r["stop"] / r["entry"] - 1) * 100
                trig = "Kharido (iske upar)" if h == "Intraday" else "Kharido (iske paas)"
                st.markdown(f"""<div class='card'>
<h4>{r['name']}</h4><div class='sub'>{r['symbol']} · abhi ₹{r['price']:,.2f}</div>
{chance_badge(r['chance'])}<span class='badge b-mid'>{r['chance']:.0f}% chance</span>
<div class='row3'><div class='r-buy'>{trig}<b>₹{r['entry']:,.1f}</b></div>
<div class='r-sl'>Stop-loss<b>₹{r['stop']:,.1f}</b>{dn:.1f}%</div>
<div class='r-tg'>Target<b>₹{r['target']:,.1f}</b>+{up:.1f}%</div></div>
<div class='sub' style='margin-top:8px'>Pichle {r['bt_trades']} signals me {r['wins']} baar profit</div>
</div>""", unsafe_allow_html=True)
                st.button("Poora plan dekho ➜", key=f"go_{h}_{r['symbol']}", on_click=go_to,
                          args=(r["symbol"], h), width="stretch")
    sells = sorted([r for r in results[h] if r["signal"].startswith("SELL") and in_band(r)],
                   key=lambda r: -r["score"])[:12]
    if sells:
        st.subheader("🚫 In stocks se abhi door raho (trend neeche)" if h != "Intraday" else "🔻 Intraday me kamzor stocks")
        st.write(" · ".join(f"**{r['name']}** ({r['change']:+.1f}%)" for r in sells))
    with st.expander("📋 Saare stocks ki list (table)"):
        df = pd.DataFrame([{"Stock": r["name"], "Symbol": r["symbol"], "Faisla": r["signal"],
                            "Score": int(r["score"]), "Price ₹": round(r["price"], 2),
                            "Chance %": round(r["chance"]), "Entry ₹": round(r["entry"], 2),
                            "Stop-loss ₹": round(r["stop"], 2), "Target ₹": round(r["target"], 2)}
                           for r in sorted(results[h], key=lambda r: -r["bull"]) if in_band(r)])
        st.dataframe(df, hide_index=True, width="stretch")

    with st.expander("📄 Poori report yahin dekho (download kholne ki zaroorat nahi)"):
        st.caption("Ye wahi report hai jo download hoti hai – yahin padh lo, ya download karke "
                   "kisi ko WhatsApp/email pe bhejo.")
        try:
            components.html(snapshot_html, height=650, scrolling=True)
        except Exception:
            st.download_button("⬇️ Report download", snapshot_html,
                               file_name=f"stock_report_{datetime.now():%Y%m%d_%H%M}.html",
                               mime="text/html")

# ============================================================================ TRACKING
elif st.session_state.page == P_TRACK:
    st.title("🔁 Pichle picks – kya badla?")
    regime_line()
    st.caption("App har trading din aapke picks yaad rakhta hai. Yahan dekho: pichli baar se kaun naye aaye, "
               "kaun abhi bhi achhe hain, aur kaun se **ab nikal jaane** chahiye (data kharab ho gaya).")
    if not prev_snap:
        st.info("📅 Abhi tak sirf aaj ka record bana hai. **Kal ya agli baar** jab app khologe, "
                "tab yahan pichli baar se poora farak dikhega. (Record tabhi banta hai jab asli data aata hai.)")
        st.caption(f"Aaj ka record save ho gaya: {today_str}")
    else:
        for h in engine.HORIZONS:
            st.subheader(HZ_SHORT[h])
            render_changes(h)
            st.divider()

# ============================================================================ STOCK CHECK
elif st.session_state.page == P_CHECK:
    st.title("🔍 Stock Check Karo")
    st.caption("Yahan **koi bhi NSE stock** check kar sakte ho – chahe wo upar wale scan me ho ya nahi.")
    # every NSE stock is selectable here, not just the scanned ones
    syms = ALL_NSE if ALL_NSE else sorted(daily.keys())
    if st.session_state.sel_sym not in syms:
        st.session_state.sel_sym = "RELIANCE" if "RELIANCE" in syms else syms[0]
    sym = st.selectbox("Stock ka naam likho ya chuno 👇", syms, key="sel_sym",
                       format_func=lambda s: f"{data.NAMES.get(s, s)}  ({s})")
    h = st.segmented_control("Kitne time ke liye?", list(engine.HORIZONS), key="hz",
                             format_func=lambda x: HZ_LABEL[x]) or "Weekly"
    r = next((x for x in results[h] if x["symbol"] == sym), None)
    if r is None:
        # stock not in today's scan -> fetch & analyse it on demand
        with st.spinner(f"{data.NAMES.get(sym, sym)} ka data la rahe hain…"):
            od = analyse_one(sym, demo or is_demo, fetched)
        if not od or h not in od:
            st.error("Is stock ka 2 saal ka data nahi mila (bahut naya listed ho sakta hai, ya symbol galat hai). "
                     "Kisi aur stock ko try karo.")
            st.stop()
        r = od[h]
        if sym not in daily and od.get("_raw") is not None:
            daily[sym] = od["_raw"]   # so the charts below can draw this stock

    # ---------------- live price + live verdict (auto refresh every 30 s)
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
            st.markdown(f"<div class='sub'>{data.NAMES.get(sym, sym)} · LIVE price</div>"
                        f"<div class='price'>₹{price:,.2f} <span class='{cls}' style='font-size:1.1rem'>{chg:+.2f}%</span></div>"
                        f"<div class='sub'>Update: {engine.ist_now():%I:%M:%S %p} · har 30 sec · Yahoo data 1-15 min late ho sakta hai</div>",
                        unsafe_allow_html=True)
        with c2:
            getattr(st, level)(f"**Abhi kya karein?** {msg}")
    live_box()

    # ---------------- big verdict
    kind = "buy" if r["signal"] == "BUY" else "sell" if r["signal"].startswith("SELL") else "wait"
    if kind == "buy":
        big, msg = "✅ KHARIDO (BUY)", (f"Trend aur momentum saath hain. ₹{r['entry']:,.2f} "
                                        f"{'ke upar jaaye tab' if h == 'Intraday' else 'ke paas'} kharido, "
                                        f"₹{r['stop']:,.2f} pe stop-loss lagao, ₹{r['target']:,.2f} pe profit book karo.")
    elif kind == "sell":
        if h == "Intraday":
            big, msg = "🔻 SHORT SELL (sirf intraday)", (f"Stock kamzor hai. ₹{r['entry']:,.2f} ke neeche toote to short karo, "
                                                         f"target ₹{r['target']:,.2f}, stop-loss ₹{r['stop']:,.2f}. Naye log isse avoid karein.")
        else:
            big, msg = "❌ MAT KHARIDO / BECHO", "Trend neeche hai. Naya mat kharido. Agar pehle se aapke paas hai to bechne ka socho."
    else:
        big, msg = "⏸️ RUKO (WAIT)", "Abhi clear signal nahi hai. Zabardasti trade mat karo – kal dobara check karo."
    st.markdown(f"<div class='verdict v-{kind}'><div class='big'>{big}</div>"
                f"<div class='msg'>{msg}</div><div class='sub' style='margin-top:6px'>Signal strength: {r['score']:.0f}/100 "
                f"(65+ = signal) · {HZ_SHORT[h]} · data {r['asof']}</div></div>", unsafe_allow_html=True)

    # ---------------- plan boxes
    qty, invest, max_loss, max_profit = money_plan(r)
    b1, b2, b3, b4 = st.columns(4)
    b1.markdown(f"<div class='box r-buy'><div class='lbl'>🛒 {'Kharidne ka trigger' if h=='Intraday' else 'Kharidne ka price'}</div>"
                f"<div class='val'>₹{r['entry']:,.2f}</div><div class='hint'>{'Price iske upar jaaye tab' if h=='Intraday' else 'Iske aas-paas kharido'}</div></div>", unsafe_allow_html=True)
    b2.markdown(f"<div class='box r-sl'><div class='lbl'>🛑 Stop-loss (yahan nikal jao)</div>"
                f"<div class='val'>₹{r['stop']:,.2f}</div><div class='hint'>Price yahan aaye to turant becho – nuksaan chhota rakho</div></div>", unsafe_allow_html=True)
    b3.markdown(f"<div class='box r-tg'><div class='lbl'>🎯 Target (yahan profit lo)</div>"
                f"<div class='val'>₹{r['target']:,.2f}</div><div class='hint'>Yahan pahunche to becho ya aadha becho</div></div>", unsafe_allow_html=True)
    if qty > 0:
        hint4 = f"Lagega ₹{invest:,.0f} · max nuksaan ₹{max_loss:,.0f} · possible profit ₹{max_profit:,.0f}"
    else:
        hint4 = "Aapke paise/risk ke hisaab se ye share mehenga hai. Settings me paisa ya risk % badhao, ya sasta stock chuno."
    b4.markdown(f"<div class='box' style='background:rgba(168,85,247,.13)'><div class='lbl'>💰 Kitne share lein</div>"
                f"<div class='val'>{qty:,} shares</div><div class='hint'>{hint4}</div></div>", unsafe_allow_html=True)

    # ---------------- chance
    st.subheader("🎲 Profit ka chance kitna hai?")
    if r["bt_trades"] == 0:
        st.info("Pichle 2 saal me is stock pe ye signal kabhi nahi aaya, isliye record nahi hai.")
    else:
        c1, c2 = st.columns([1, 2])
        c1.markdown(f"<div class='price'>{r['chance']:.0f}%</div>{chance_badge(r['chance'])}", unsafe_allow_html=True)
        c2.markdown(f"Pichle 2 saal me ye **same signal {r['bt_trades']} baar** aaya. **{r['wins']} baar profit** hua, "
                    f"**{r['bt_trades'] - r['wins']} baar loss**. Average har trade: **{r['bt_avg']:+.2f}%**.")
        c2.progress(min(1.0, max(0.0, r["chance"] / 100)))
        c2.caption("Chance % ko overall market ke record se balance kiya gaya hai, taaki 3 me se 3 jaisa luck 100% na dikhe. "
                   "55-65% chance + stop-loss = bahut achha system. 90% ka wada koi bhi sach me nahi kar sakta.")

    # ---------------- why
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("👍 Kyun? (paksh me)")
        if r["pros"]:
            for _, t in r["pros"]:
                st.markdown(f"- ✅ {t}")
        else:
            st.caption("Koi strong point nahi.")
    with c2:
        st.subheader("👎 Kya kami hai?")
        if r["cons"]:
            for _, t in r["cons"]:
                st.markdown(f"- ➖ {t}")
        else:
            st.caption("Sab points paksh me hain.")
    st.subheader("⏰ Kab kharidein, kab bechein?")
    for t in r["timing"]:
        st.markdown(f"- {t}")

    # ---------------- horizon specific chart
    st.divider()
    d_all = engine.add_indicators(daily[sym])

    def level_lines(fig):
        fig.add_hrect(y0=min(r["entry"], r["target"]), y1=max(r["entry"], r["target"]), fillcolor="rgba(22,163,74,.10)", line_width=0)
        fig.add_hrect(y0=min(r["entry"], r["stop"]), y1=max(r["entry"], r["stop"]), fillcolor="rgba(239,68,68,.10)", line_width=0)
        for y, nm, col in ((r["entry"], "Kharido", "#3b82f6"), (r["stop"], "Stop-loss", "#ef4444"), (r["target"], "Target", "#22c55e")):
            fig.add_hline(y=y, line_dash="dash", line_color=col, annotation_text=f"{nm} ₹{y:,.1f}",
                          annotation_position="right")

    def markers(fig, d_view, hz):
        buys, sells = engine.signal_markers(d_all, hz, last_n=len(d_view) if hz != "Monthly" else 520)
        buys = [x for x in buys if x >= d_view.index[0]]
        sells = [x for x in sells if x >= d_view.index[0]]
        if buys:
            fig.add_trace(go.Scatter(x=buys, y=d_all.loc[buys, "Low"] * 0.97, mode="markers+text", text=["BUY"] * len(buys),
                                     textposition="bottom center", marker=dict(symbol="triangle-up", size=13, color="#22c55e"),
                                     name="Pichle BUY signal"))
        if sells:
            fig.add_trace(go.Scatter(x=sells, y=d_all.loc[sells, "High"] * 1.03, mode="markers+text", text=["SELL"] * len(sells),
                                     textposition="top center", marker=dict(symbol="triangle-down", size=13, color="#ef4444"),
                                     name="Pichle SELL signal"))

    layout = dict(height=560, xaxis_rangeslider_visible=False, margin=dict(l=10, r=90, t=10, b=10),
                  legend=dict(orientation="h", y=1.06), hovermode="x unified")

    if h == "Intraday":
        st.markdown("<span class='chart-tag'>📊 Ye chart INTRADAY ke liye hai · har candle = 5 minute · sirf aaj ka din</span>", unsafe_allow_html=True)
        df5, _ = load_intraday(sym, demo or is_demo, r['price'])
        il = engine.intraday_live(df5)
        if il is None:
            st.info("Aaj ka intraday data abhi nahi hai (market band / chhutti).")
        else:
            t = il["bars"]
            fig = go.Figure(go.Candlestick(x=t.index, open=t.Open, high=t.High, low=t.Low, close=t.Close, name="5-min candle",
                                           increasing_line_color="#22c55e", decreasing_line_color="#ef4444"))
            fig.add_trace(go.Scatter(x=t.index, y=t.vwap, name="VWAP (aaj ka average)", line=dict(color="#f59e0b", width=2)))
            fig.add_hline(y=il["orb_hi"], line_dash="dot", line_color="#86efac", annotation_text="Pehle 15 min ka HIGH", annotation_position="left")
            fig.add_hline(y=il["orb_lo"], line_dash="dot", line_color="#fca5a5", annotation_text="Pehle 15 min ka LOW", annotation_position="left")
            level_lines(fig)
            fig.update_layout(**layout)
            st.plotly_chart(fig, width="stretch")
            st.markdown(f"**Abhi chart kya bol raha hai:** {il['bias']}")
        with st.expander("🧑‍🏫 Ye intraday chart kaise padhein? (zaroor padho)", expanded=True):
            st.markdown("""
- **Har candle = 5 minute.** 🟢 Hari candle = us 5 min me price upar gaya, 🔴 laal = neeche gaya.
- **Peeli line (VWAP)** = aaj ka average price. Price **upar** = buyers strong 💪, **neeche** = sellers strong.
- **Hari dotted line** = pehle 15 minute (9:15-9:30) ka high. Price iske upar toote = tezi shuru.
- **Neeli dashed line (Kharido)** = trigger. Price iske upar jaaye **aur** VWAP ke upar ho → **BUY**.
- **Laal line (Stop-loss)** = price yahan aaye to turant becho. **Hari line (Target)** = yahan profit lo.
- **Kab becho:** target aaye, stop-loss aaye, ya **3:10 PM** – jo pehle ho.
- 2:30 PM ke baad naya trade mat lo. Ek din me 2 loss ho gaye to us din band.
""")
    elif h == "Weekly":
        st.markdown("<span class='chart-tag'>📊 Ye chart WEEKLY trade ke liye hai · har candle = 1 din · pichle 6 mahine</span>", unsafe_allow_html=True)
        dv = d_all.iloc[-130:]
        fig = go.Figure(go.Candlestick(x=dv.index, open=dv.Open, high=dv.High, low=dv.Low, close=dv.Close, name="Daily candle",
                                       increasing_line_color="#22c55e", decreasing_line_color="#ef4444"))
        fig.add_trace(go.Scatter(x=dv.index, y=dv.ema21, name="21-din line", line=dict(color="#f59e0b", width=1.6)))
        fig.add_trace(go.Scatter(x=dv.index, y=dv.ema50, name="50-din line", line=dict(color="#3b82f6", width=1.6)))
        markers(fig, dv, "Weekly")
        level_lines(fig)
        fig.update_layout(**layout)
        st.plotly_chart(fig, width="stretch")
        with st.expander("🧑‍🏫 Ye weekly chart kaise padhein? (zaroor padho)", expanded=True):
            st.markdown("""
- **Har candle = 1 din.** 🟢 hara = us din upar band hua, 🔴 laal = neeche band hua.
- **Orange line = 21 din ka average**, **neeli line = 50 din ka average**.
- **Uptrend kab:** price dono lines ke **upar** aur orange line neeli ke **upar** → kharidne ka mahaul ✅.
- **Downtrend kab:** price dono lines ke **neeche** → door raho ❌.
- **▲ Hare teer (BUY)** aur **▼ laal teer (SELL)** = pichli baar jab app ne signal diya tha. Dekho uske baad price kya hua – isi se "chance %" banta hai.
- **Sabse achhi entry:** uptrend me jab price **orange line tak gire** aur wahan se hari candle bane.
- **Kab becho:** target (hari line), stop-loss (laal line), ya 5 din poore – jo pehle ho.
""")
    else:
        st.markdown("<span class='chart-tag'>📊 Ye chart MONTHLY trade ke liye hai · har candle = 1 hafta · pichle 2 saal</span>", unsafe_allow_html=True)
        w = engine.to_weekly(daily[sym]).iloc[-104:]
        w["ma10"] = engine.to_weekly(daily[sym]).Close.rolling(10).mean().iloc[-104:]
        w["ma40"] = engine.to_weekly(daily[sym]).Close.rolling(40).mean().iloc[-104:]
        fig = go.Figure(go.Candlestick(x=w.index, open=w.Open, high=w.High, low=w.Low, close=w.Close, name="Weekly candle",
                                       increasing_line_color="#22c55e", decreasing_line_color="#ef4444"))
        fig.add_trace(go.Scatter(x=w.index, y=w.ma10, name="10-hafte line (~50 din)", line=dict(color="#f59e0b", width=1.8)))
        fig.add_trace(go.Scatter(x=w.index, y=w.ma40, name="40-hafte line (~200 din)", line=dict(color="#a855f7", width=1.8)))
        markers(fig, d_all[d_all.index >= w.index[0]], "Monthly")
        level_lines(fig)
        fig.update_layout(**layout)
        st.plotly_chart(fig, width="stretch")
        with st.expander("🧑‍🏫 Ye monthly chart kaise padhein? (zaroor padho)", expanded=True):
            st.markdown("""
- **Har candle = 1 hafta.** Isse bada picture dikhta hai, roz ka shor nahi.
- **Orange line ≈ 50 din**, **baingani line ≈ 200 din** ka average.
- **Lamba uptrend:** price dono lines ke upar aur orange line baingani ke upar → monthly BUY ke liye best ✅.
- **Price baingani line ke neeche** = lamba trend kamzor → long-term ke liye mat kharido ❌.
- **▲/▼ teer** = pichle monthly BUY/SELL signal. Dekho unke baad agle 4 hafte me kya hua.
- **Kaise kharidein:** 2-3 hisson me. Pehla abhi, baaki orange line ki taraf girne pe.
- **Kab becho:** target aaye, stop-loss aaye, ya lagbhag 1 mahina (20 trading din) ho jaye.
""")

    if not (demo or is_demo):
        with st.expander("🏢 Company ki details (fundamentals)"):
            f = load_fundamentals(sym)
            if f:
                st.table(pd.DataFrame(f.items(), columns=["Cheez", "Value"]))
            else:
                st.caption("Abhi available nahi.")

# ============================================================================ LEARN
elif st.session_state.page == P_LEARN:
    st.title("📚 Seekho – Hinglish me, aasaan bhasha me")
    st.markdown("""
### 1️⃣ Ye app kaise use karein (3 step)
1. **🏠 Aaj ke Best Stocks** kholo → upar se chuno: Intraday / Weekly / Monthly.
2. Kisi card pe **"Poora plan dekho"** dabao → app batayega **KHARIDO / MAT KHARIDO / RUKO**.
3. App jo **Kharido price, Stop-loss, Target** bataye, broker app (Zerodha, Groww, Upstox…) me wahi order lagao. **Stop-loss order zaroor lagao.**

### 2️⃣ Candle kya hoti hai? 🕯️
- Har candle ek time ka hisaab hai (5 min / 1 din / 1 hafta).
- 🟢 **Hari** = shuru se zyada pe band hua (price badha). 🔴 **Laal** = kam pe band hua (price gira).
- Upar-neeche ki patli line (wick) = us time ka sabse high aur sabse low price.

### 3️⃣ Trend – sabse zaroori cheez 📈
- **"Trend is your friend."** Upar jaate stock ko kharido, girte stock ko nahi.
- Chart pe average lines (21, 50, 200 din) dekho. Price lines ke **upar** = uptrend, **neeche** = downtrend.

### 4️⃣ Support aur Resistance 🧱
- **Support** = wo price jahan se stock baar-baar upar uchhalta hai (farsh).
- **Resistance** = wo price jahan se stock baar-baar neeche aata hai (chhat).
- Resistance todke upar band ho = **breakout** (achha BUY mauka).

### 5️⃣ Stop-loss aur Target 🛑🎯
- **Stop-loss** = "agar main galat hua to yahan nikal jaunga". Isse bada nuksaan nahi hota.
- **Target** = "yahan profit le lunga".
- Is app me target hamesha stop-loss se **lagbhag 2 guna door** hota hai. Matlab 10 me se 4 trade bhi sahi hue to bhi kamaai ho sakti hai.

### 6️⃣ Kitna paisa lagayein? 💰
- Ek trade me apne total paise ka **1-2% se zyada nuksaan** mat hone do.
- Example: ₹1,00,000 hai, 1% risk = ₹1,000. Stop-loss ₹10 door hai → **100 shares** lo. App ye calculation khud karta hai.
- Saara paisa ek stock me kabhi mat lagao.

### 7️⃣ Intraday vs Weekly vs Monthly
| | ⚡ Intraday | 📅 Weekly | 🗓️ Monthly |
|---|---|---|---|
| Kitne time | Same din | 1 hafta | ~1 mahina |
| Chart | 5-min candle | Daily candle | Weekly candle |
| Risk | Sabse zyada | Medium | Sabse kam |
| Kiske liye | Jo poora din screen dekh sake | Job wale | Beginners ✅ |

### 8️⃣ Sach baat – 90% guarantee kyun nahi? 🤝
- Market news, results, global events se hilta hai – koi chart ye pehle se nahi jaanta.
- Duniya ke best traders bhi 50-65% trade hi sahi karte hain. Woh kamaate hain kyunki **unka profit bada aur loss chhota** hota hai.
- Jo bhi 90% pakka profit ka wada kare – **dhoka ho sakta hai**, door raho.
- Ye app **sach ke numbers** dikhata hai, taaki aap sabse achhe chance wale stock chun sako.

### 9️⃣ Golden rules ⭐
- Hamesha **stop-loss** lagao. ✋
- **Market mood** 🔴 ho to kam trade karo.
- Company ke **results** se pehle naya trade avoid karo.
- Loss ke baad "wapas kamana hai" soch ke bada trade mat lo.
- Apne trades ek diary/Excel me likho – 1 mahine baad dekho kya kaam kar raha hai.
""")

# ============================================================================ ACCURACY
elif st.session_state.page == P_ACC:
    st.title("🎯 App kitna sahi hai?")
    st.write("App ke har BUY/SELL rule ko har stock ke **pichle 2 saal** pe chala ke dekha gaya (0.1% kharcha kaat ke). "
             "Agar ek hi din stop-loss aur target dono aa gaye, to use **loss** gina gaya (safe side).")

    @st.cache_data(ttl=900, show_spinner="Saare stocks pe test chal raha hai…")
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
                rows.append({"Type": HZ_SHORT[hh], "Signal": side, "Kitne trade": len(arr),
                             "Profit wale %": round((arr > 0).mean() * 100, 1),
                             "Average har trade %": round(arr.mean(), 2),
                             "Average profit %": round(arr[arr > 0].mean(), 2) if (arr > 0).any() else 0,
                             "Average loss %": round(arr[arr <= 0].mean(), 2) if (arr <= 0).any() else 0})
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    st.info("**Kaise samjhein:** 'Average har trade %' **positive** hai to system ne pichle time me paisa banaya – chahe "
            "profit wale trade 50% se kam hon, kyunki profit loss se bada rakha jaata hai. "
            "Home page pe sirf wahi stocks dikhte hain jinka apna record bhi positive raha hai.")

st.markdown("<div class='sub' style='margin-top:30px'>Education/research tool hai, SEBI registered advice nahi · "
            "Pichla record future ki guarantee nahi.</div>", unsafe_allow_html=True)
