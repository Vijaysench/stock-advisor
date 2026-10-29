"""
Background watchdog for Stock Advisor.

Run this in its own window (double-click "Start Watchdog.bat") and keep it open.
It checks the stocks in your bucket every few minutes and sends a Telegram
message when something important happens — even when the main app / browser is
closed:

  * 🎯 target hit            * 🛑 stop-loss hit
  * ⚠️ sharp fall today      * ⏰ holding window over (time to exit & rotate)

Needs the Telegram bot token + at least one connected person, set on the
"📲 Telegram Alerts" page of the app. Uses the same tracked.json and
notify_config.json, so nothing extra to configure here.
"""
from __future__ import annotations

import os
import time
from datetime import datetime

import data
import engine
import notify
import tracker

APP_DIR = os.path.dirname(os.path.abspath(__file__))
POLL_SECONDS_OPEN = 180      # how often to check while the market is open
POLL_SECONDS_CLOSED = 900    # slower when the market is closed


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


def check_once() -> int:
    items = tracker.load(APP_DIR)
    if not items or not notify.is_ready(APP_DIR):
        return 0
    status, _ = engine.market_status()
    today = engine.ist_now().date()
    sent = 0
    for it in items:
        try:
            lv = data.live_price(it["symbol"])
        except Exception:
            lv = None
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
        elif stt["sharp_fall"] and status == "open":
            kind = "fall"
        if kind and not notify.already_sent(APP_DIR, tracker.alert_key(it, kind)):
            if notify.broadcast(APP_DIR, _alert_text(it, stt, kind, price)):
                sent += 1
    return sent


def main():
    print("=" * 54)
    print(" Stock Advisor — Telegram watchdog")
    print(" Keep this window open. Alerts go to Telegram.")
    print(" Press Ctrl+C (or close this window) to stop.")
    print("=" * 54)
    while True:
        try:
            status, _ = engine.market_status()
            n = check_once()
            stamp = datetime.now().strftime("%d %b %I:%M:%S %p")
            ready = notify.is_ready(APP_DIR)
            tracked = len(tracker.load(APP_DIR))
            note = "" if ready else "  (Telegram not set up yet — do it on the Alerts page)"
            print(f"[{stamp}] checked {tracked} tracked · market {status} · {n} alert(s) sent{note}")
            time.sleep(POLL_SECONDS_OPEN if status == "open" else POLL_SECONDS_CLOSED)
        except KeyboardInterrupt:
            print("\nWatchdog stopped. Bye!")
            break
        except Exception as e:
            print(f"  (temporary error: {e}; retrying in 60s)")
            time.sleep(60)


if __name__ == "__main__":
    main()
