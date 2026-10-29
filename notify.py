"""
Telegram notifications for Stock Advisor.

How it works (important):
  * A Telegram *bot* can only message a person AFTER that person has opened the
    bot and pressed Start once. At that moment Telegram gives the bot a numeric
    "chat id" for that person. That chat id (a NUMBER) is what the app stores –
    so in the UI the user effectively "adds their number" to get alerts.
  * Sending is a simple HTTPS call to the Telegram Bot API, so it works from the
    Streamlit app AND from the standalone background watchdog.

Config is kept in <app>/notify_config.json:
    {
      "bot_token": "123456:ABC...",      # set once by the owner
      "subscribers": [                    # people who will get alerts
          {"chat_id": 12345678, "name": "Vijay"}
      ],
      "enabled": true
    }

Pure standard-library + requests, so it can be imported by both the app and the
watchdog with no Streamlit dependency.
"""
from __future__ import annotations

import json
import os
from datetime import datetime

CONFIG_NAME = "notify_config.json"


# --------------------------------------------------------------------------- config
def config_path(app_folder: str) -> str:
    return os.path.join(app_folder, CONFIG_NAME)


def load_config(app_folder: str) -> dict:
    path = config_path(app_folder)
    data = {"bot_token": "", "subscribers": [], "enabled": True}
    if os.path.exists(path):
        try:
            with open(path, encoding="utf-8") as f:
                saved = json.load(f)
            if isinstance(saved, dict):
                data.update(saved)
        except Exception:
            pass
    # allow the token to come from an environment variable too (handy on the cloud)
    data["bot_token"] = (data.get("bot_token") or os.environ.get("TELEGRAM_BOT_TOKEN", "")).strip()
    if not isinstance(data.get("subscribers"), list):
        data["subscribers"] = []
    return data


def save_config(app_folder: str, cfg: dict) -> None:
    path = config_path(app_folder)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)
    os.replace(tmp, path)


def add_subscriber(app_folder: str, chat_id, name: str = "") -> dict:
    cfg = load_config(app_folder)
    chat_id = int(chat_id)
    subs = cfg.get("subscribers", [])
    if not any(int(s.get("chat_id")) == chat_id for s in subs):
        subs.append({"chat_id": chat_id, "name": name or str(chat_id)})
        cfg["subscribers"] = subs
        save_config(app_folder, cfg)
    return cfg


def remove_subscriber(app_folder: str, chat_id) -> dict:
    cfg = load_config(app_folder)
    chat_id = int(chat_id)
    cfg["subscribers"] = [s for s in cfg.get("subscribers", []) if int(s.get("chat_id")) != chat_id]
    save_config(app_folder, cfg)
    return cfg


def is_ready(app_folder: str) -> bool:
    cfg = load_config(app_folder)
    return bool(cfg.get("bot_token")) and bool(cfg.get("subscribers")) and cfg.get("enabled", True)


# --------------------------------------------------------------------------- telegram api
def _api(token: str, method: str, params: dict, timeout: int = 10):
    import requests
    url = f"https://api.telegram.org/bot{token}/{method}"
    r = requests.post(url, data=params, timeout=timeout)
    r.raise_for_status()
    return r.json()


def bot_username(token: str) -> str | None:
    """Return the bot's @username so the UI can build a one-tap start link."""
    if not token:
        return None
    try:
        js = _api(token, "getMe", {}, timeout=8)
        return js.get("result", {}).get("username")
    except Exception:
        return None


def recent_chat_ids(token: str) -> list[dict]:
    """People who recently messaged the bot: [{chat_id, name}] – used to let a
    user tap 'Start', then pick themselves without typing their id by hand."""
    if not token:
        return []
    try:
        js = _api(token, "getUpdates", {"timeout": 0, "allowed_updates": json.dumps(["message"])}, timeout=8)
    except Exception:
        return []
    found, seen = [], set()
    for upd in js.get("result", []):
        msg = upd.get("message") or upd.get("edited_message") or {}
        chat = msg.get("chat") or {}
        cid = chat.get("id")
        if cid is None or cid in seen:
            continue
        seen.add(cid)
        name = (chat.get("first_name", "") + " " + chat.get("last_name", "")).strip() \
            or chat.get("username", "") or str(cid)
        found.append({"chat_id": int(cid), "name": name})
    return found


def send_to(token: str, chat_id, text: str) -> bool:
    try:
        js = _api(token, "sendMessage",
                  {"chat_id": chat_id, "text": text, "parse_mode": "HTML",
                   "disable_web_page_preview": "true"})
        return bool(js.get("ok"))
    except Exception:
        return False


def broadcast(app_folder: str, text: str) -> int:
    """Send `text` to every subscriber. Returns how many got it."""
    cfg = load_config(app_folder)
    token = cfg.get("bot_token")
    if not token or not cfg.get("enabled", True):
        return 0
    sent = 0
    for s in cfg.get("subscribers", []):
        if send_to(token, s.get("chat_id"), text):
            sent += 1
    return sent


def send_test(app_folder: str, chat_id) -> bool:
    cfg = load_config(app_folder)
    token = cfg.get("bot_token")
    if not token:
        return False
    stamp = datetime.now().strftime("%d %b %Y, %I:%M %p")
    msg = ("✅ <b>Stock Advisor connected!</b>\n"
           f"You will now get alerts here.\n<i>{stamp}</i>")
    return send_to(token, chat_id, msg)


# --------------------------------------------------------------------------- de-dupe
# The app and the watchdog both send alerts. We keep a tiny log so the same alert
# (e.g. "TARGET hit for RELIANCE weekly") is not sent again and again.
def _sent_log_path(app_folder: str) -> str:
    return os.path.join(app_folder, "notify_sent.json")


def already_sent(app_folder: str, key: str, cooldown_hours: float = 6.0) -> bool:
    path = _sent_log_path(app_folder)
    log = {}
    if os.path.exists(path):
        try:
            with open(path, encoding="utf-8") as f:
                log = json.load(f)
        except Exception:
            log = {}
    last = log.get(key)
    now = datetime.now().timestamp()
    if last and (now - last) < cooldown_hours * 3600:
        return True
    log[key] = now
    # keep it from growing forever
    if len(log) > 2000:
        log = dict(sorted(log.items(), key=lambda kv: kv[1])[-1000:])
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(log, f)
    except Exception:
        pass
    return False
