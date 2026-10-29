"""
Bucket / tracking list for Stock Advisor.

The user adds stocks they are watching or have bought. For each one we remember
the horizon (Intraday / Weekly / Monthly), the day it was added, the plan levels
(buy point, stop-loss, target) and optionally the real buy price + quantity.

Given a live price we work out, in plain language:
  * live profit / loss,
  * whether the stop-loss or target has been hit,
  * whether the holding window is over (weekly ~5 trading days, monthly ~20,
    intraday the same day) and it's time to exit and rotate into a fresh stock,
  * a simple HOLD / ADD / BOOK PROFIT / EXIT verdict.

Pure standard-library + the analysed dicts from engine, so both the Streamlit app
and the background watchdog can use it with no extra dependencies.
"""
from __future__ import annotations

import json
import os
from datetime import date, datetime

STORE_NAME = "tracked.json"

# how long each horizon is meant to be held, in *trading* days
WINDOW_DAYS = {"Intraday": 1, "Weekly": 5, "Monthly": 20}
# a one-session drop bigger than this (vs previous close) counts as a "sharp fall"
SHARP_FALL_PCT = {"Intraday": 1.5, "Weekly": 4.0, "Monthly": 6.0}


# --------------------------------------------------------------------------- store
def _path(app_folder: str) -> str:
    return os.path.join(app_folder, STORE_NAME)


def load(app_folder: str) -> list[dict]:
    path = _path(app_folder)
    if not os.path.exists(path):
        return []
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except Exception:
        return []


def save(app_folder: str, items: list[dict]) -> None:
    path = _path(app_folder)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(items, f, indent=2)
    os.replace(tmp, path)


def _key(symbol: str, horizon: str) -> str:
    return f"{symbol}|{horizon}"


def add(app_folder: str, r: dict, buy_price: float | None = None,
        qty: int | None = None, today: str | None = None) -> list[dict]:
    """Add an analysed stock dict `r` (from engine.analyse) to the bucket."""
    items = load(app_folder)
    today = today or str(date.today())
    entry = {
        "symbol": r["symbol"],
        "name": r.get("name", r["symbol"]),
        "horizon": r["horizon"],
        "side": r.get("side", "BUY"),
        "added_date": today,
        "added_at": datetime.now().isoformat(timespec="seconds"),
        "buy_point": float(r["entry"]),
        "stop": float(r["stop"]),
        "target": float(r["target"]),
        "price_at_add": float(r.get("price", r["entry"])),
        "chance": float(r.get("chance", 0) or 0),
        "buy_price": float(buy_price) if buy_price else None,
        "qty": int(qty) if qty else None,
    }
    k = _key(entry["symbol"], entry["horizon"])
    items = [it for it in items if _key(it["symbol"], it["horizon"]) != k]
    items.append(entry)
    save(app_folder, items)
    return items


def remove(app_folder: str, symbol: str, horizon: str) -> list[dict]:
    items = [it for it in load(app_folder) if _key(it["symbol"], it["horizon"]) != _key(symbol, horizon)]
    save(app_folder, items)
    return items


def is_tracked(app_folder: str, symbol: str, horizon: str) -> bool:
    return any(_key(it["symbol"], it["horizon"]) == _key(symbol, horizon) for it in load(app_folder))


# --------------------------------------------------------------------------- status
def _trading_days_between(d1: str, d2: date) -> int:
    """Rough count of trading days (Mon-Fri) from d1 up to d2."""
    try:
        start = datetime.strptime(d1, "%Y-%m-%d").date()
    except Exception:
        return 0
    if d2 <= start:
        return 0
    days = 0
    cur = start
    while cur < d2:
        cur = date.fromordinal(cur.toordinal() + 1)
        if cur.weekday() < 5:
            days += 1
    return days


def status(entry: dict, live_price: float | None, live_chg: float | None = None,
           today: date | None = None) -> dict:
    """Work out the current situation for one tracked stock."""
    today = today or date.today()
    side = entry.get("side", "BUY")
    buy_point = entry.get("buy_point")
    stop = entry.get("stop")
    target = entry.get("target")
    ref = entry.get("buy_price") or buy_point          # what we measure P&L from
    qty = entry.get("qty")
    horizon = entry.get("horizon", "Weekly")

    used = 0
    held = _trading_days_between(entry.get("added_date", str(today)), today)
    window = WINDOW_DAYS.get(horizon, 5)
    window_left = max(0, window - held)
    window_over = held >= window

    pnl_pct = pnl_amt = None
    if live_price is not None and ref:
        direction = 1 if side == "BUY" else -1
        pnl_pct = (live_price / ref - 1) * 100 * direction
        if qty:
            pnl_amt = (live_price - ref) * qty * direction

    hit = None
    if live_price is not None:
        if side == "BUY":
            if target is not None and live_price >= target:
                hit = "target"
            elif stop is not None and live_price <= stop:
                hit = "stop"
        else:
            if target is not None and live_price <= target:
                hit = "target"
            elif stop is not None and live_price >= stop:
                hit = "stop"

    sharp_fall = (live_chg is not None and live_chg <= -SHARP_FALL_PCT.get(horizon, 4.0))

    # ---- verdict
    if hit == "target":
        verdict, level, msg = "BOOK PROFIT", "success", "Target hit — book your profit (or sell half and trail the rest)."
    elif hit == "stop":
        verdict, level, msg = "EXIT NOW", "error", "Stop-loss hit — exit now, keep the loss small. Don't average down."
    elif window_over:
        verdict, level, msg = "EXIT & ROTATE", "warning", (
            f"Your {horizon.lower()} holding window is over ({window} trading days). "
            "Book it and move to a fresh pick — don't hold out of hope.")
    elif sharp_fall:
        verdict, level, msg = "WATCH CLOSELY", "warning", (
            f"Sharp fall today ({live_chg:+.1f}%). If it closes below the stop-loss, exit. "
            "Check for any news before adding more.")
    elif side != "BUY":
        verdict, level, msg = "HOLD SHORT", "info", "Short trade still valid. Keep the stop-loss in place."
    else:
        verdict, level, msg = "HOLD", "info", "Trade is on track. Keep your stop-loss; let it run toward target."

    return dict(
        verdict=verdict, level=level, msg=msg, hit=hit,
        pnl_pct=pnl_pct, pnl_amt=pnl_amt,
        held_days=held, window=window, window_left=window_left, window_over=window_over,
        sharp_fall=sharp_fall, buy_point=buy_point, stop=stop, target=target,
        live_price=live_price, ref=ref,
    )


def alert_key(entry: dict, kind: str) -> str:
    """Stable key so an alert isn't repeated (used with notify.already_sent)."""
    return f"{entry['symbol']}|{entry['horizon']}|{entry.get('added_date','')}|{kind}"
