"""
StockAdvisor analysis engine.

Everything here is plain pandas/numpy so it can be tested without a network.
Signals are rule-based technical scores. Every "chance of profit" number is
MEASURED by replaying the exact same rule on the stock's own price history
(a backtest) - it is never a made-up number.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# Horizon settings
# --------------------------------------------------------------------------
HORIZONS = {
    "Intraday": dict(bars=1, stop_atr=0.5, target_atr=0.8, label="Same day (next session)"),
    "Weekly": dict(bars=5, stop_atr=1.5, target_atr=3.0, label="Up to 1 week (5 sessions)"),
    "Monthly": dict(bars=20, stop_atr=2.5, target_atr=5.0, label="Up to 1 month (20 sessions)"),
}
SIGNAL_THRESHOLD = 65  # score (0-100) needed for a BUY / SELL call


# --------------------------------------------------------------------------
# Indicators
# --------------------------------------------------------------------------
def _ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()


def _wilder(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(alpha=1 / n, adjust=False).mean()


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """df needs Open, High, Low, Close, Volume columns (daily bars)."""
    d = df.copy()
    c, h, l, v = d["Close"], d["High"], d["Low"], d["Volume"]

    for n in (9, 21, 50, 200):
        d[f"ema{n}"] = _ema(c, n)

    # RSI (Wilder)
    delta = c.diff()
    gain = _wilder(delta.clip(lower=0), 14)
    loss = _wilder(-delta.clip(upper=0), 14)
    rs = gain / loss.replace(0, np.nan)
    d["rsi"] = (100 - 100 / (1 + rs)).fillna(50)

    # MACD
    macd = _ema(c, 12) - _ema(c, 26)
    d["macd"] = macd
    d["macd_sig"] = _ema(macd, 9)
    d["macd_hist"] = macd - d["macd_sig"]

    # Bollinger
    mid = c.rolling(20).mean()
    sd = c.rolling(20).std()
    d["bb_up"], d["bb_mid"], d["bb_lo"] = mid + 2 * sd, mid, mid - 2 * sd

    # ATR
    prev_c = c.shift(1)
    tr = pd.concat([h - l, (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)
    d["atr"] = _wilder(tr, 14)
    d["atr_pct"] = d["atr"] / c * 100

    # ADX / DI
    up = h.diff()
    dn = -l.diff()
    plus_dm = np.where((up > dn) & (up > 0), up, 0.0)
    minus_dm = np.where((dn > up) & (dn > 0), dn, 0.0)
    atr_w = _wilder(tr, 14)
    d["pdi"] = 100 * _wilder(pd.Series(plus_dm, index=d.index), 14) / atr_w
    d["mdi"] = 100 * _wilder(pd.Series(minus_dm, index=d.index), 14) / atr_w
    dx = 100 * (d["pdi"] - d["mdi"]).abs() / (d["pdi"] + d["mdi"]).replace(0, np.nan)
    d["adx"] = _wilder(dx.fillna(0), 14)

    # Volume
    d["vol_ratio"] = v / v.rolling(20).mean()
    d["vol_ratio5"] = v.rolling(5).mean() / v.rolling(50).mean()

    # Momentum / range position
    d["ret5"] = c.pct_change(5) * 100
    d["ret20"] = c.pct_change(20) * 100
    d["ret60"] = c.pct_change(60) * 100
    d["hi252"] = h.rolling(252, min_periods=60).max()
    d["lo252"] = l.rolling(252, min_periods=60).min()
    d["from_hi"] = (c / d["hi252"] - 1) * 100
    rng = (h - l).replace(0, np.nan)
    d["close_pos"] = ((c - l) / rng).fillna(0.5)  # 1 = closed at high

    # Supertrend (10, 3)
    d["supertrend_up"] = _supertrend(d, 10, 3.0)
    return d


def _supertrend(d: pd.DataFrame, n: int, mult: float) -> pd.Series:
    hl2 = (d["High"] + d["Low"]) / 2
    atr = d["atr"].values
    upper = (hl2 + mult * d["atr"]).values
    lower = (hl2 - mult * d["atr"]).values
    close = d["Close"].values
    fu, fl = upper.copy(), lower.copy()
    trend = np.ones(len(d), dtype=bool)
    for i in range(1, len(d)):
        if np.isnan(atr[i]):
            continue
        fu[i] = upper[i] if (upper[i] < fu[i - 1] or close[i - 1] > fu[i - 1]) else fu[i - 1]
        fl[i] = lower[i] if (lower[i] > fl[i - 1] or close[i - 1] < fl[i - 1]) else fl[i - 1]
        if close[i] > fu[i - 1]:
            trend[i] = True
        elif close[i] < fl[i - 1]:
            trend[i] = False
        else:
            trend[i] = trend[i - 1]
    return pd.Series(trend, index=d.index)


# --------------------------------------------------------------------------
# Scoring rules. Each rule = (points, bull_condition, bear_condition, reason_bull, reason_bear)
# Vectorised so the same rule is used for "today" and for the backtest.
# --------------------------------------------------------------------------
def _rules(d: pd.DataFrame, horizon: str):
    c = d["Close"]
    if horizon == "Intraday":
        return [
            (15, (c > d.ema9) & (d.ema9 > d.ema21), (c < d.ema9) & (d.ema9 < d.ema21),
             "Short-term trend is UP (price above the 9 & 21-day average lines)", "Short-term trend is DOWN (price below the 9 & 21-day average lines)"),
            (15, d.rsi.between(52, 72), d.rsi.between(28, 48),
             "RSI is in a strong zone (buyers active)", "RSI is in a weak zone (sellers active)"),
            (15, (d.macd_hist > 0) & (d.macd_hist > d.macd_hist.shift(1)),
             (d.macd_hist < 0) & (d.macd_hist < d.macd_hist.shift(1)),
             "Momentum is rising (MACD moving up)", "Momentum is falling (MACD moving down)"),
            (15, (d.vol_ratio > 1.3) & (c > c.shift(1)), (d.vol_ratio > 1.3) & (c < c.shift(1)),
             "Price rose on above-normal volume — big players buying", "Price fell on high volume — big players selling"),
            (10, d.close_pos > 0.7, d.close_pos < 0.3,
             "Closed near the day's HIGH — buyers in control", "Closed near the day's LOW — sellers in control"),
            (10, c > d.High.shift(1), c < d.Low.shift(1),
             "Closed above the previous day's high (breakout)", "Closed below the previous day's low (breakdown)"),
            (10, (d.adx > 20) & (d.pdi > d.mdi), (d.adx > 20) & (d.mdi > d.pdi),
             "Trend has strength (ADX above 20, buyers ahead)", "Downtrend has strength (ADX above 20, sellers ahead)"),
            (10, c < d.bb_up, c > d.bb_lo,
             "Price isn't overstretched (still room to rise)", "Price isn't overstretched to the downside"),
        ]
    if horizon == "Weekly":
        return [
            (20, (c > d.ema21) & (d.ema21 > d.ema50), (c < d.ema21) & (d.ema21 < d.ema50),
             "Medium-term trend is UP (price above the 21 & 50-day lines)", "Medium-term trend is DOWN (price below the 21 & 50-day lines)"),
            (15, d.rsi.between(50, 68), d.rsi.between(32, 50),
             "RSI is healthy (50-68) — not weak, not overheated", "RSI is weak (32-50)"),
            (15, d.macd_hist > 0, d.macd_hist < 0, "MACD above its signal line — momentum positive", "MACD below its signal line — momentum negative"),
            (15, (d.adx > 20) & (d.pdi > d.mdi), (d.adx > 20) & (d.mdi > d.pdi),
             "Strong trend with buyers ahead (ADX)", "Strong trend with sellers ahead (ADX)"),
            (10, ((c / d.ema21 - 1).abs() < 0.03) & (c > d.ema21),
             ((c / d.ema21 - 1).abs() < 0.03) & (c < d.ema21),
             "Price near the 21-day support line — lower-risk entry", "Stalled at the 21-day line and fell back (resistance)"),
            (10, d.vol_ratio5 > 1.0, d.vol_ratio5 > 1.0,
             "Volume above normal over the last 5 days — rising interest", "High volume on the decline — selling pressure"),
            (15, d.supertrend_up, ~d.supertrend_up, "Supertrend is GREEN (buy mode)", "Supertrend is RED (sell mode)"),
        ]
    # Monthly
    return [
        (25, (c > d.ema50) & (d.ema50 > d.ema200), (c < d.ema50) & (d.ema50 < d.ema200),
         "Long-term trend is UP (price above the 50 & 200-day lines)", "Long-term trend is DOWN (price below the 50 & 200-day lines)"),
        (15, d.ret60 > 5, d.ret60 < -5, "Up more than 5% over the last 3 months", "Down more than 5% over the last 3 months"),
        (15, d.from_hi > -12, d.from_hi < -30, "Within 12% of its 1-year high — a market leader", "More than 30% below its 1-year high — a weak stock"),
        (10, d.rsi.between(50, 70), d.rsi < 45, "RSI is strong", "RSI is weak"),
        (15, d.supertrend_up, ~d.supertrend_up, "Supertrend is GREEN (buy mode)", "Supertrend is RED (sell mode)"),
        (10, d.adx > 20, d.adx > 20, "Trend has strength (ADX above 20)", "Trend has strength (ADX above 20)"),
        (10, d.macd > 0, d.macd < 0, "MACD above zero — long-term momentum positive", "MACD below zero — long-term momentum negative"),
    ]


def score_series(d: pd.DataFrame, horizon: str):
    """Return bull and bear score series (0-100) for every bar."""
    bull = pd.Series(0.0, index=d.index)
    bear = pd.Series(0.0, index=d.index)
    for pts, b, s, *_ in _rules(d, horizon):
        bull += np.where(b.fillna(False), pts, 0)
        bear += np.where(s.fillna(False), pts, 0)
    return bull, bear


def reasons_today(d: pd.DataFrame, horizon: str, side: str):
    out_for, out_against = [], []
    for pts, b, s, rb, rs in _rules(d, horizon):
        bull_ok = bool(b.iloc[-1]) if not pd.isna(b.iloc[-1]) else False
        bear_ok = bool(s.iloc[-1]) if not pd.isna(s.iloc[-1]) else False
        if side == "BUY":
            (out_for if bull_ok else out_against).append((pts, rb if bull_ok else f"Missing: {rb}"))
        else:
            (out_for if bear_ok else out_against).append((pts, rs if bear_ok else f"Missing: {rs}"))
    return out_for, out_against


# --------------------------------------------------------------------------
# Backtest: replay the signal on history, measure real outcomes
# --------------------------------------------------------------------------
@dataclass
class BacktestResult:
    trades: int = 0
    target_hit: float = float("nan")    # % of trades where target was hit before stop
    profit_rate: float = float("nan")   # % of trades closed in profit (target, or positive at time exit)
    profit_low: float = float("nan")    # Wilson 90% lower bound for profit_rate (honest worst case)
    avg_ret: float = float("nan")       # average % return per trade
    stop_hit: float = float("nan")
    returns: list = field(default_factory=list)


def wilson_lower(p: float, n: int, z: float = 1.645) -> float:
    if n == 0:
        return float("nan")
    denom = 1 + z * z / n
    centre = p + z * z / (2 * n)
    adj = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return max(0.0, (centre - adj) / denom)


def backtest(d: pd.DataFrame, horizon: str, side: str = "BUY", lookback: int = 500,
             threshold: int = SIGNAL_THRESHOLD) -> BacktestResult:
    cfg = HORIZONS[horizon]
    hbars, k_stop, k_tgt = cfg["bars"], cfg["stop_atr"], cfg["target_atr"]
    bull, bear = score_series(d, horizon)
    sig = (bull >= threshold) & (bull > bear) if side == "BUY" else (bear >= threshold) & (bear > bull)

    o, h, l, c, atr = (d[x].values for x in ("Open", "High", "Low", "Close", "atr"))
    n = len(d)
    start = max(200, n - lookback)  # skip warm-up period
    rets, tgt, stp = [], 0, 0
    i = start
    sig_v = sig.values
    while i < n - hbars - (1 if horizon == "Intraday" else 0):
        if not sig_v[i] or np.isnan(atr[i]):
            i += 1
            continue
        if horizon == "Intraday":
            # signal seen at close of day i -> next day, enter ONLY if price breaks
            # yesterday's high (BUY) / low (SELL), same as the live plan; exit by close.
            trigger = h[i] if side == "BUY" else l[i]
            triggered = h[i + 1] >= trigger if side == "BUY" else l[i + 1] <= trigger
            if not triggered:
                i += 1
                continue
            entry = max(o[i + 1], trigger) if side == "BUY" else min(o[i + 1], trigger)
            path = range(i + 1, i + 2)
        else:
            entry = c[i]
            path = range(i + 1, i + 1 + hbars)
        risk = k_stop * atr[i]
        reward = k_tgt * atr[i]
        if side == "BUY":
            stop, target = entry - risk, entry + reward
        else:
            stop, target = entry + risk, entry - reward
        result = None
        for j in path:
            hit_stop = l[j] <= stop if side == "BUY" else h[j] >= stop
            hit_tgt = h[j] >= target if side == "BUY" else l[j] <= target
            if hit_stop:  # conservative: if both hit in the same bar, count the loss
                result = -risk / entry * 100
                stp += 1
                break
            if hit_tgt:
                result = reward / entry * 100
                tgt += 1
                break
        if result is None:
            exit_px = c[path[-1]]
            result = (exit_px - entry) / entry * 100 * (1 if side == "BUY" else -1)
        result -= 0.1  # brokerage + taxes + slippage allowance (round trip)
        rets.append(result)
        # skip ahead so trades don't overlap (each trade counted once)
        i = path[-1] + 1 if horizon != "Intraday" else i + 1

    r = BacktestResult(trades=len(rets), returns=rets)
    if rets:
        arr = np.array(rets)
        r.target_hit = tgt / len(rets) * 100
        r.stop_hit = stp / len(rets) * 100
        pr = float((arr > 0).mean())
        r.profit_rate = pr * 100
        r.profit_low = wilson_lower(pr, len(rets)) * 100
        r.avg_ret = float(arr.mean())
    return r


# --------------------------------------------------------------------------
# Full analysis for one stock & horizon
# --------------------------------------------------------------------------
def analyse(symbol: str, raw: pd.DataFrame, horizon: str, d: pd.DataFrame | None = None) -> dict | None:
    if raw is None or len(raw) < 260:
        return None
    d = add_indicators(raw) if d is None else d
    bull, bear = score_series(d, horizon)
    b, s = float(bull.iloc[-1]), float(bear.iloc[-1])
    if b >= SIGNAL_THRESHOLD and b > s:
        side, signal = "BUY", "BUY"
    elif s >= SIGNAL_THRESHOLD and s > b:
        side = "SELL"
        signal = "SELL (short)" if horizon == "Intraday" else "SELL / AVOID"
    else:
        side, signal = ("BUY" if b >= s else "SELL"), "WAIT"

    cfg = HORIZONS[horizon]
    last = d.iloc[-1]
    price, atr = float(last.Close), float(last.atr)
    if horizon == "Intraday":
        # trigger levels for next session
        entry = float(last.High) if side == "BUY" else float(last.Low)
    else:
        entry = price
    if side == "BUY":
        stop, target = entry - cfg["stop_atr"] * atr, entry + cfg["target_atr"] * atr
    else:
        stop, target = entry + cfg["stop_atr"] * atr, entry - cfg["target_atr"] * atr

    bt = backtest(d, horizon, side)
    pros, cons = reasons_today(d, horizon, side)

    return dict(
        symbol=symbol, horizon=horizon, signal=signal, side=side,
        score=b if side == "BUY" else s, bull=b, bear=s,
        price=price, change=float((last.Close / d.Close.iloc[-2] - 1) * 100),
        entry=entry, stop=stop, target=target,
        rr=cfg["target_atr"] / cfg["stop_atr"],
        rsi=float(last.rsi), adx=float(last.adx), atr_pct=float(last.atr_pct),
        from_hi=float(last.from_hi), ret20=float(last.ret20),
        bt_trades=bt.trades, bt_profit=bt.profit_rate, bt_profit_low=bt.profit_low,
        bt_target=bt.target_hit, bt_stop=bt.stop_hit, bt_avg=bt.avg_ret,
        pros=pros, cons=cons, timing=timing_advice(horizon, side, entry, last),
        asof=str(d.index[-1].date()) if hasattr(d.index[-1], "date") else str(d.index[-1]),
    )


def timing_advice(horizon: str, side: str, entry: float, last) -> list[str]:
    e = f"₹{entry:,.2f}"
    if horizon == "Intraday":
        if side == "BUY":
            return [
                "Do nothing from 9:15–9:30 — the first 15 minutes are very choppy.",
                f"Between 9:30–11:00, buy when price goes ABOVE {e} (yesterday's high) and holds.",
                "Even better if price is above the yellow VWAP line on the live chart.",
                "If there's no breakout by 11:30, skip the trade for the day.",
                "Sell by 3:10 PM no matter what — never hold intraday shares overnight.",
            ]
        return [
            "Do nothing from 9:15–9:30.",
            f"Short only when price goes BELOW {e} (yesterday's low) and holds (9:30–11:00).",
            "The signal is stronger if price is below the VWAP line.",
            "Close the position by 3:10 PM (brokers auto-close around 3:15–3:20 with a penalty).",
            "Beginners should avoid short selling — practice the buy side first.",
        ]
    if horizon == "Weekly":
        if side == "BUY":
            return [
                f"Good entry: around {e}, or when price dips toward the 21-day line (₹{float(last.ema21):,.2f}).",
                "Buy in the last hour (2:30–3:20 PM) once the day's direction is clear.",
                "If the stock gaps up more than 2%, don't chase — wait for a small dip.",
                "Sell on target, stop-loss, or after 5 trading days — whichever comes first.",
            ]
        return [
            "Trend is weak — don't buy fresh. If you already hold it, exit on an up-move.",
            f"Look again when price closes above the 21-day line (₹{float(last.ema21):,.2f}).",
        ]
    if side == "BUY":
        return [
            f"Buy in 2–3 parts: first near {e}, the rest on dips toward the 50-day line (₹{float(last.ema50):,.2f}).",
            "Don't buy right before the company's quarterly results — check the results date.",
            "Hold about 1 month. When half the target is reached, move the stop-loss to your buy price (a no-loss trade).",
        ]
    return [
        "Long-term trend is down — stay away. If you hold it, sell in parts on up-moves.",
        f"Look again when price rises above the 50-day (₹{float(last.ema50):,.2f}) and 200-day (₹{float(last.ema200):,.2f}) lines.",
    ]


# --------------------------------------------------------------------------
# Intraday live helpers (5-minute bars)
# --------------------------------------------------------------------------
def intraday_live(df5: pd.DataFrame) -> dict | None:
    """df5: 5-minute bars (may contain several days). Uses the latest session."""
    if df5 is None or df5.empty:
        return None
    df5 = df5.copy()
    day = df5.index[-1].date()
    t = df5[df5.index.date == day].copy()
    if t.empty:
        return None
    tp = (t.High + t.Low + t.Close) / 3
    t["vwap"] = (tp * t.Volume).cumsum() / t.Volume.replace(0, np.nan).cumsum()
    orb = t.iloc[:3]  # first 15 minutes (3 x 5 min)
    orb_hi, orb_lo = float(orb.High.max()), float(orb.Low.min())
    last = float(t.Close.iloc[-1])
    vwap = float(t.vwap.iloc[-1]) if not np.isnan(t.vwap.iloc[-1]) else last
    if last > orb_hi and last > vwap:
        bias = "BULLISH — price above the opening range and above VWAP"
    elif last < orb_lo and last < vwap:
        bias = "BEARISH — price below the opening range and below VWAP"
    elif last > vwap:
        bias = "MILDLY BULLISH — above VWAP, but the range hasn't broken yet"
    else:
        bias = "MILDLY BEARISH — below VWAP, but the range hasn't broken yet"
    return dict(bars=t, orb_hi=orb_hi, orb_lo=orb_lo, vwap=vwap, last=last, bias=bias, day=str(day))


def market_regime(nifty: pd.DataFrame) -> dict:
    if nifty is None or len(nifty) < 210:
        return dict(ok=None, text="Nifty data unavailable")
    d = add_indicators(nifty)
    last = d.iloc[-1]
    above50 = last.Close > last.ema50
    above200 = last.Close > last.ema200
    if above50 and above200:
        txt, ok = "GOOD — Nifty is in an uptrend. BUY signals work more often.", True
    elif above200:
        txt, ok = "MIXED — Nifty is a bit weak. Trade smaller sizes.", None
    else:
        txt, ok = "WEAK — Nifty is falling. BUY signals fail more often; trade very carefully.", False
    return dict(ok=ok, text=txt, close=float(last.Close), chg=float((last.Close / d.Close.iloc[-2] - 1) * 100),
                rsi=float(last.rsi))


def make_sample_data(seed: int = 0, n: int = 750, start_price: float = 1000.0) -> pd.DataFrame:
    """Synthetic OHLCV for offline testing / demo mode."""
    rng = np.random.default_rng(seed)
    drift = rng.normal(0.0004, 0.0004)
    rets = rng.normal(drift, 0.017, n)
    close = start_price * np.exp(np.cumsum(rets))
    open_ = close * np.exp(rng.normal(0, 0.006, n))
    high = np.maximum(open_, close) * np.exp(np.abs(rng.normal(0, 0.008, n)))
    low = np.minimum(open_, close) * np.exp(-np.abs(rng.normal(0, 0.008, n)))
    vol = rng.lognormal(13, 0.4, n)
    idx = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=n)
    return pd.DataFrame(dict(Open=open_, High=high, Low=low, Close=close, Volume=vol), index=idx)


# --------------------------------------------------------------------------
# Extras used by the simple UI
# --------------------------------------------------------------------------
def signal_markers(d: pd.DataFrame, horizon: str, last_n: int = 250):
    """Dates where a NEW buy / sell signal started (for arrows on the chart)."""
    bull, bear = score_series(d, horizon)
    buy = (bull >= SIGNAL_THRESHOLD) & (bull > bear)
    sell = (bear >= SIGNAL_THRESHOLD) & (bear > bull)
    buy_start = buy & ~buy.shift(1, fill_value=False)
    sell_start = sell & ~sell.shift(1, fill_value=False)
    d2 = d.iloc[-last_n:]
    return d2.index[buy_start.iloc[-last_n:].values], d2.index[sell_start.iloc[-last_n:].values]


def to_weekly(df: pd.DataFrame) -> pd.DataFrame:
    w = df.resample("W-FRI").agg({"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"})
    return w.dropna(subset=["Close"])


def ist_now():
    return pd.Timestamp.now(tz="Asia/Kolkata")


def last_bar_date(raw: pd.DataFrame):
    try:
        return raw.index[-1].date()
    except Exception:
        return None


def data_age_days(raw: pd.DataFrame):
    """Trading-ish age of the newest daily bar vs today (calendar days)."""
    d = last_bar_date(raw)
    if d is None:
        return None
    return (ist_now().date() - d).days


def market_status():
    now = ist_now()
    if now.weekday() >= 5:
        return "closed", "Market is closed today (weekend). The plan is for the next trading day."
    t = now.hour * 60 + now.minute
    if t < 9 * 60 + 15:
        return "pre", "Market isn't open yet (opens at 9:15 AM). The plan is ready for today."
    if t <= 15 * 60 + 30:
        return "open", "Market is OPEN — live prices are updating."
    return "closed", "Today's market has closed (3:30 PM). The plan is for the next trading day."


def adjusted_chance(wins: float, trades: int, prior: float, k: int = 10) -> float:
    """Shrink a stock's own win rate towards the overall average so that
    a lucky 3-out-of-3 doesn't look like 100%. Returns %."""
    if trades <= 0:
        return prior
    return (wins + prior / 100 * k) / (trades + k) * 100


# --------------------------------------------------------------------------
# Fundamental quality + factor scores + overall conviction
# (how a research desk judges a stock: company strength + valuation, not just
#  the chart). All graceful: missing data just lowers that part's weight.
# --------------------------------------------------------------------------
def _band(v, thresholds, scores):
    """thresholds high->low with matching scores; picks first threshold v meets."""
    for t, s in zip(thresholds, scores):
        if v >= t:
            return s
    return scores[-1]


def fundamental_score(stats: dict | None) -> dict:
    """Company-quality score 0-100 from growth, profitability, debt, valuation.
    Returns {score, have, pros, cons, rows} ; score=None if no data at all."""
    if not stats:
        return dict(score=None, have=False, pros=[], cons=[], rows=[])
    parts, pros, cons, rows = [], [], [], []

    def add(label, val, weight, sub, good_txt, bad_txt, fmt):
        if val is None:
            return
        parts.append((weight, sub))
        rows.append((label, fmt(val), sub))
        if sub >= 70:
            pros.append(good_txt)
        elif sub <= 40:
            cons.append(bad_txt)

    roe = stats.get("roe")
    add("Return on equity (ROE)", roe, 0.22,
        _band(roe, [0.18, 0.12, 0.08, 0.0], [100, 75, 55, 30, 10]) if roe is not None else None,
        "Uses shareholders' money very efficiently (high ROE)", "Weak returns on capital (low ROE)",
        lambda v: f"{v*100:.0f}%")
    eg = stats.get("earnings_growth")
    add("Profit growth", eg, 0.18,
        _band(eg, [0.15, 0.05, 0.0], [100, 75, 55, 20]) if eg is not None else None,
        "Profits are growing strongly", "Profits are shrinking", lambda v: f"{v*100:.0f}%")
    rg = stats.get("rev_growth")
    add("Sales growth", rg, 0.15,
        _band(rg, [0.15, 0.08, 0.0], [100, 75, 55, 25]) if rg is not None else None,
        "Sales are growing fast", "Sales are falling", lambda v: f"{v*100:.0f}%")
    pm = stats.get("profit_margin")
    add("Profit margin", pm, 0.12,
        _band(pm, [0.15, 0.08, 0.03, 0.0], [100, 78, 55, 30, 10]) if pm is not None else None,
        "Keeps a healthy profit on each sale", "Very thin profit margins", lambda v: f"{v*100:.0f}%")
    de = stats.get("de")
    if de is not None:
        sub = 100 if de <= 0.3 else 80 if de <= 0.7 else 60 if de <= 1.0 else 40 if de <= 1.5 else 20
        add("Debt vs equity", de, 0.15, sub, "Low debt — financially safe", "High debt — risky", lambda v: f"{v:.2f}x")
    peg = stats.get("peg")
    pe = stats.get("pe")
    if peg is not None:
        sub = 100 if peg <= 1 else 75 if peg <= 2 else 50 if peg <= 3 else 25
        add("Valuation (PEG)", peg, 0.10, sub, "Fairly priced for its growth", "Expensive vs its growth", lambda v: f"{v:.2f}")
    elif pe is not None and pe > 0:
        sub = 90 if pe <= 15 else 70 if pe <= 25 else 45 if pe <= 40 else 20
        add("Valuation (P/E)", pe, 0.10, sub, "Reasonably valued", "Looks expensive (high P/E)", lambda v: f"{v:.1f}")
    mc = stats.get("market_cap")
    if mc is not None:
        sub = 100 if mc >= 2e12 else 80 if mc >= 5e11 else 60 if mc >= 1e11 else 45
        add("Company size", mc, 0.08, sub, "Large, established company", "Smaller company — more volatile",
            lambda v: f"₹{v/1e7:,.0f} Cr")

    if not parts:
        return dict(score=None, have=False, pros=[], cons=[], rows=[])
    wsum = sum(w for w, _ in parts)
    score = sum(w * s for w, s in parts) / wsum
    return dict(score=round(score), have=True, pros=pros[:4], cons=cons[:4], rows=rows)


def factor_scores(last, fund_score, rs: float = 0.0) -> dict:
    """Five 0-100 factor bars, like a quick quant snapshot. `last` = latest
    indicator row (d.iloc[-1]). Skips Quality if no fundamentals."""
    c = float(last.Close)
    trend = (int(c > last.ema21) + int(last.ema21 > last.ema50) + int(c > last.ema50)
             + int(c > last.ema200) + int(last.ema50 > last.ema200)) / 5 * 100
    mom = 50 + float(last.ret20) * 1.2 + (float(last.rsi) - 50) * 0.8 + (10 if last.macd_hist > 0 else -10)
    mom = max(0, min(100, mom)) * 0.6 + max(0, min(100, 50 + rs * 2)) * 0.4
    atrp = float(last.atr_pct)
    stability = max(0, min(100, 100 - (atrp - 1.0) * 22))
    out = {"Trend": round(trend), "Momentum": round(mom), "Stability": round(stability)}
    if fund_score is not None:
        out["Quality"] = int(fund_score)
    return out


def conviction(chance: float, fund_score, rs: float, regime_ok) -> dict:
    """Blend the measured technical 'chance' with company quality, relative
    strength and market mood into one High/Medium/Low rating."""
    base = chance if chance else 50.0
    if fund_score is not None:
        combined = 0.6 * base + 0.4 * fund_score
    else:
        combined = base
    combined += 3 if rs > 0 else -3
    combined *= 1.0 if regime_ok else 0.95 if regime_ok is None else 0.90
    combined = max(0, min(100, combined))
    if combined >= 62:
        label, tone = "High conviction", "b-hi"
    elif combined >= 52:
        label, tone = "Medium conviction", "b-mid"
    else:
        label, tone = "Low conviction", "b-lo"
    return dict(score=round(combined), label=label, tone=tone)


def live_verdict(r: dict, live_price: float, vwap: float | None, status: str) -> tuple[str, str]:
    """Simple 'what should I do right now' message. Returns (level, text)."""
    side, h = r["side"], r["horizon"]
    if r["signal"] == "WAIT":
        return "info", "No clear signal right now — WAIT. Doing nothing is also a good decision."
    entry, stop, target = r["entry"], r["stop"], r["target"]
    up = side == "BUY"
    if (up and live_price <= stop) or (not up and live_price >= stop):
        return "error", "Price has reached the stop-loss — this signal has FAILED. Stay away from this trade."
    if (up and live_price >= target) or (not up and live_price <= target):
        return "warning", "The target is already hit — entering now is too late. Don't chase."
    if h == "Intraday":
        if status != "open":
            return "info", f"Check at market open: trade only if price goes {'above' if up else 'below'} ₹{entry:,.2f}."
        crossed = live_price > entry if up else live_price < entry
        vw_ok = vwap is None or (live_price > vwap if up else live_price < vwap)
        now = ist_now()
        late = now.hour * 60 + now.minute > 14 * 60 + 30
        if late:
            return "warning", "No new intraday trade after 2:30 PM — too little time left."
        if crossed and vw_ok:
            return "success", f"You can {'BUY' if up else 'SHORT'} NOW — price is {'above' if up else 'below'} the trigger ₹{entry:,.2f} and VWAP agrees. Set the stop-loss ₹{stop:,.2f} immediately."
        if crossed:
            return "warning", "Trigger broke but price is on the wrong side of VWAP — wait a bit for confirmation."
        return "info", f"WAIT — price is ₹{live_price:,.2f} now. Trade when it goes {'above' if up else 'below'} ₹{entry:,.2f}."
    # weekly / monthly
    if not up:
        return "error", "Trend is weak — DON'T BUY. If you already hold it, consider selling."
    risk = abs(entry - stop)
    if live_price > entry + 0.5 * risk:
        return "warning", f"Price has run well above entry (₹{entry:,.2f}) — don't chase; wait for a dip back near ₹{entry:,.2f}."
    return "success", f"You can BUY — price is near the entry zone (₹{entry:,.2f}). Stop-loss ₹{stop:,.2f}, target ₹{target:,.2f}."
