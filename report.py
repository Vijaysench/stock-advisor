"""Export a self-contained, shareable HTML snapshot of the current scan."""
from __future__ import annotations

import html
from datetime import datetime


def _pct(v, d=0):
    return "–" if v is None or v != v else f"{v:.{d}f}%"


def build_snapshot(results: dict[str, list[dict]], regime: dict, universe: str) -> str:
    now = datetime.now().strftime("%d %b %Y, %I:%M %p")
    sections = []
    for horizon, rows in results.items():
        buys = [r for r in rows if r["signal"] == "BUY"]
        sells = [r for r in rows if r["signal"].startswith("SELL")]
        buys.sort(key=lambda r: (-(r["bt_profit_low"] if r["bt_profit_low"] == r["bt_profit_low"] else 0), -r["score"]))
        sells.sort(key=lambda r: -r["score"])

        def table(rs, kind):
            if not rs:
                return f"<p class='muted'>No {kind} signals right now.</p>"
            trs = []
            for r in rs[:15]:
                why = "; ".join(t for _, t in r["pros"][:3])
                trs.append(
                    f"<tr><td><b>{html.escape(r['symbol'])}</b></td><td class='{ 'g' if kind=='BUY' else 'r'}'>{html.escape(r['signal'])}</td>"
                    f"<td>₹{r['price']:,.2f}</td><td>₹{r['entry']:,.2f}</td><td>₹{r['stop']:,.2f}</td><td>₹{r['target']:,.2f}</td>"
                    f"<td>{r['score']:.0f}</td><td>{_pct(r['bt_profit'])} <span class='muted'>({r['bt_trades']} trades)</span></td>"
                    f"<td class='why'>{html.escape(why)}</td></tr>")
            return ("<div class='tw'><table><thead><tr><th>Stock</th><th>Signal</th><th>Price</th><th>Entry</th><th>Stop-loss</th>"
                    "<th>Target</th><th>Score</th><th>Past win rate</th><th>Why</th></tr></thead><tbody>"
                    + "".join(trs) + "</tbody></table></div>")

        sections.append(f"<h2>{horizon}</h2><h3>Buy candidates</h3>{table(buys,'BUY')}<h3>Sell / avoid</h3>{table(sells,'SELL')}")

    return f"""<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Stock Advisor Snapshot</title><style>
body{{font-family:Segoe UI,system-ui,sans-serif;margin:0;background:#f6f7f9;color:#1b1f24}}
main{{max-width:1150px;margin:auto;padding:16px}}h1{{margin:.2em 0}}h2{{margin-top:1.6em;border-bottom:2px solid #dde1e6;padding-bottom:4px}}
.tw{{overflow-x:auto}}table{{border-collapse:collapse;width:100%;background:#fff;font-size:14px}}
th,td{{padding:7px 9px;border-bottom:1px solid #eceef1;text-align:left;white-space:nowrap}}td.why{{white-space:normal;min-width:260px;color:#4a525c}}
th{{background:#eef1f4}}.g{{color:#0a7d33;font-weight:600}}.r{{color:#c0262d;font-weight:600}}.muted{{color:#6b7480}}
.warn{{background:#fff4e5;border:1px solid #f3c27a;padding:10px 14px;border-radius:8px}}
@media (prefers-color-scheme:dark){{body{{background:#14171b;color:#e6e8eb}}table{{background:#1c2025}}th{{background:#242a31}}td{{border-color:#2b3138}}td.why,.muted{{color:#9aa3ad}}.warn{{background:#3a2c14;border-color:#7a5a20}}}}
</style></head><body><main>
<h1>Stock Advisor – NSE snapshot</h1>
<p class="muted">Generated {now} · Universe: {html.escape(universe)} · Market mood: {html.escape(regime.get('text',''))}</p>
<p class="warn"><b>Not investment advice.</b> Signals are technical rules. "Past win rate" is how often the same signal made a profit on this stock's own history (after costs) – it is not a guarantee. No method gives 80–90% certainty. Always use the stop-loss and risk at most 1–2% of your capital per trade.</p>
{''.join(sections)}
</main></body></html>"""
