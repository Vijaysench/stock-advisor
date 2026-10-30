"""
Day-over-day tracking for Stock Advisor.

Saves one snapshot per trading day (the picks the app showed + every stock's
signal that day) into <app>/history/snapshot_YYYY-MM-DD.json, and compares
today's scan with the most recent earlier day so the app can say, in plain
Hinglish:

  - Naye picks        (new BUYs today that were not there last time)
  - Abhi bhi achhe     (still recommended)
  - Ab nikal jao       (was recommended before, data ab kharab -> EXIT/AVOID)
  - plus whether each earlier pick hit its target / stop-loss by now.

Pure standard-library + the analysed dicts from engine.analyse, so it can be
unit-tested offline with no network.
"""
from __future__ import annotations

import glob
import json
import os
from datetime import datetime

# fields kept for every analysed stock in a snapshot
_FIELDS = ("name", "signal", "side", "score", "chance", "price",
           "entry", "stop", "target", "bt_trades", "wins", "asof")


def _dir(app_folder: str) -> str:
    d = os.path.join(app_folder, "history")
    os.makedirs(d, exist_ok=True)
    return d


def _compact(r: dict) -> dict:
    return {k: r.get(k) for k in _FIELDS}


def build_snapshot(date_str: str, universe: str, results: dict,
                   picks_by_h: dict, is_demo: bool) -> dict:
    """results: {horizon: [analysed dicts]}; picks_by_h: {horizon: [symbol,...]} shown to user."""
    horizons = {}
    for h, rows in results.items():
        horizons[h] = {
            "picks": list(picks_by_h.get(h, [])),
            "signals": {r["symbol"]: _compact(r) for r in rows},
        }
    return dict(date=date_str, universe=universe, is_demo=bool(is_demo),
                saved_at=datetime.now().isoformat(timespec="seconds"), horizons=horizons)


def save_snapshot(app_folder: str, snapshot: dict) -> str | None:
    """Write today's snapshot. Never records demo/fake data. Returns the path (or None)."""
    if snapshot.get("is_demo"):
        return None
    path = os.path.join(_dir(app_folder), f"snapshot_{snapshot['date']}.json")
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(snapshot, f)
    os.replace(tmp, path)
    _prune(app_folder, keep=120)
    return path


def _prune(app_folder: str, keep: int = 120) -> None:
    files = sorted(glob.glob(os.path.join(_dir(app_folder), "snapshot_*.json")))
    for p in files[:-keep]:
        try:
            os.remove(p)
        except OSError:
            pass


def load_snapshot(app_folder: str, date_str: str) -> dict | None:
    path = os.path.join(_dir(app_folder), f"snapshot_{date_str}.json")
    return _read(path)


def previous_snapshot(app_folder: str, today_str: str) -> dict | None:
    """Most recent snapshot strictly before today_str."""
    files = sorted(glob.glob(os.path.join(_dir(app_folder), "snapshot_*.json")))
    for p in reversed(files):
        base = os.path.basename(p)
        date = base[len("snapshot_"):-len(".json")]
        if date < today_str:
            snap = _read(p)
            if snap:
                return snap
    return None


def _read(path: str) -> dict | None:
    if not os.path.exists(path):
        return None
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _outcome(prev_pick: dict, today_price) -> tuple[str, str]:
    """How a previous pick played out, judged by today's price vs its stop/target."""
    side = prev_pick.get("side")
    entry, stop, target = prev_pick.get("entry"), prev_pick.get("stop"), prev_pick.get("target")
    if today_price is None or entry in (None, 0):
        return "unknown", "No price"
    if side == "BUY":
        if target is not None and today_price >= target:
            return "win", "🎯 Target was hit — profit"
        if stop is not None and today_price <= stop:
            return "loss", "🛑 Stop-loss hit — loss"
        return "running", f"Running ({(today_price / entry - 1) * 100:+.1f}%)"
    # SELL / short
    if target is not None and today_price <= target:
        return "win", "🎯 Target was hit — profit"
    if stop is not None and today_price >= stop:
        return "loss", "🛑 Stop-loss hit — loss"
    return "running", f"Running ({(entry / today_price - 1) * 100:+.1f}%)"


def compare(prev_snap: dict | None, curr_snap: dict, horizon: str) -> dict:
    """Return day-over-day changes for one horizon.

    Keys: have_prev, prev_date, new, continued, exited, where each list item is a
    dict {symbol, name, ...} ready for the UI.
    """
    curr_h = curr_snap.get("horizons", {}).get(horizon, {})
    today_picks = list(curr_h.get("picks", []))
    today_sig = curr_h.get("signals", {})

    if not prev_snap:
        return dict(have_prev=False, prev_date=None,
                    new=[_row(s, today_sig.get(s, {}), None, None) for s in today_picks],
                    continued=[], exited=[])

    prev_h = prev_snap.get("horizons", {}).get(horizon, {})
    prev_picks = set(prev_h.get("picks", []))
    prev_sig = prev_h.get("signals", {})

    new, continued, exited = [], [], []
    for s in today_picks:
        prev_pick = prev_sig.get(s) if s in prev_picks else None
        row = _row(s, today_sig.get(s, {}), prev_pick,
                   _outcome(prev_pick, (today_sig.get(s) or {}).get("price")) if prev_pick else None)
        (continued if s in prev_picks else new).append(row)

    for s in prev_picks - set(today_picks):
        prev_pick = prev_sig.get(s, {})
        today = today_sig.get(s)  # may be None if dropped from universe/data
        today_price = today.get("price") if today else None
        today_signal = today.get("signal") if today else None
        still_buy = bool(today_signal and today_signal == "BUY")
        exited.append(dict(
            symbol=s, name=prev_pick.get("name", s),
            prev=prev_pick, today=today,
            still_buy=still_buy,
            today_signal=today_signal or "no data",
            outcome=_outcome(prev_pick, today_price),
        ))
    return dict(have_prev=True, prev_date=prev_snap.get("date"),
                new=new, continued=continued, exited=exited)


def _row(symbol: str, today_pick: dict, prev_pick: dict | None, outcome) -> dict:
    return dict(symbol=symbol, name=today_pick.get("name", symbol),
                today=today_pick, prev=prev_pick, outcome=outcome)
