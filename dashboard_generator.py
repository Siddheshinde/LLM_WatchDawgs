"""
dashboard_generator.py
Generate the light-themed dashboard (dashboard.html + reports_data.js).
"""

import json
import shutil
from pathlib import Path

import numpy as np
from datetime import datetime
from utils import load_logs, format_timestamp
from generate_reports_data import KEEP_FIELDS, load_reports, write_js
from risk_engine import analyze_risk_distribution
from temporal_preview import generate_temporal_report

try:
    from temporal_engine import run_temporal_analysis
    PHASE2_AVAILABLE = True
except ImportError:
    PHASE2_AVAILABLE = False

# =========================
# DATA AGGREGATION
# =========================

def aggregate_dashboard_data(records):
    """Aggregate all data needed for dashboard, including Phase 2 temporal analysis."""
    if not records:
        return None

    total = len(records)
    uncertainties = [r["uncertainty_score"] for r in records if r.get("uncertainty_score") is not None]
    consistencies  = [r["consistency_score"] for r in records if r.get("consistency_score") is not None]
    calibrations   = [r.get("calibration_score") for r in records if r.get("calibration_score") is not None]
    risk_scores    = [r.get("risk_score") for r in records if r.get("risk_score") is not None]

    risk_dist = analyze_risk_distribution(records)
    temporal  = generate_temporal_report(records)

    # Category breakdown
    categories = {}
    for record in records:
        cat = record.get("category", "unknown")
        categories.setdefault(cat, []).append(record)

    def safe_mean(values):
        return float(np.mean(values)) if values else 0.0

    category_stats = {}
    for cat, recs in categories.items():
        cat_uncertainties = [r["uncertainty_score"] for r in recs if r.get("uncertainty_score") is not None]
        cat_consistencies = [r["consistency_score"] for r in recs if r.get("consistency_score") is not None]
        cat_calibrations = [r.get("calibration_score") for r in recs if r.get("calibration_score") is not None]
        cat_risks = [r.get("risk_score") for r in recs if r.get("risk_score") is not None]

        category_stats[cat] = {
            "count":             len(recs),
            "mean_uncertainty":  safe_mean(cat_uncertainties),
            "mean_consistency":  safe_mean(cat_consistencies),
            "mean_calibration":  safe_mean(cat_calibrations),
            "mean_risk":         safe_mean(cat_risks),
            "reliable_count":     sum(1 for r in recs if r.get("risk_zone") == "RELIABLE"),
            "overconfident_count":sum(1 for r in recs if r.get("risk_zone") == "OVERCONFIDENT"),
            "unstable_count":     sum(1 for r in recs if r.get("risk_zone") == "UNSTABLE"),
            "ambiguous_count":    sum(1 for r in recs if r.get("risk_zone") == "AMBIGUOUS"),
        }

    # Phase 2 temporal windows
    temporal_windows = []
    phase2_stats = {"total_windows": 0, "drift_alert_count": 0, "change_point_count": 0}
    if PHASE2_AVAILABLE:
        try:
            temporal_windows = run_temporal_analysis("qa_monitoring_logs.jsonl", "24h")
            phase2_stats = {
                "total_windows":     len(temporal_windows),
                "drift_alert_count": sum(1 for w in temporal_windows if w.get("drift_alert")),
                "change_point_count":sum(1 for w in temporal_windows if w.get("change_point")),
            }
        except Exception as e:
            print(f"  [WARN] Phase 2 analysis failed: {e}")

    # Chart series
    reliable_pct = round(
        risk_dist["distribution"].get("RELIABLE", 0) / total * 100
        if total else 0, 1
    )

    def safe_mean(values):
        return float(np.mean(values)) if values else 0.0

    return {
        "total_interactions": total,
        "model":  records[0].get("model", "llama3"),
        "time_range": {
            "first": records[0].get("timestamp"),
            "last":  records[-1].get("timestamp"),
        },
        "overall_stats": {
            "mean_uncertainty":  safe_mean(uncertainties),
            "std_uncertainty":   float(np.std(uncertainties)) if uncertainties else 0.0,
            "mean_consistency":  safe_mean(consistencies),
            "std_consistency":   float(np.std(consistencies)) if consistencies else 0.0,
            "mean_calibration":  safe_mean(calibrations),
            "mean_risk":         safe_mean(risk_scores),
        },
        "reliable_pct": reliable_pct,
        "risk_distribution":  risk_dist,
        "temporal":           temporal,
        "temporal_windows":   temporal_windows,
        "phase2_stats":       phase2_stats,
        "category_stats":     category_stats,
        "time_series": {
            "timestamps":  list(range(len(records))),
            "uncertainty": [r.get("uncertainty_score") for r in records],
            "consistency": [r.get("consistency_score") for r in records],
            "calibration": [r.get("calibration_score") for r in records],
        },
        "recent_interactions": records[-15:],
    }


# =========================
# HTML HELPERS
# =========================

def _score_cls(v, invert=False):
    """Return CSS class based on score value."""
    if v is None:
        return "warn"
    hi = v > 0.6; lo = v < 0.35
    if invert:
        return "good" if hi else ("warn" if not lo else "danger")
    return "danger" if hi else ("warn" if not lo else "good")

def _pill(v, invert=False):
    if v is None:
        return "mid"
    hi = v > 0.6; lo = v < 0.35
    if invert:
        return "lo" if hi else ("mid" if not lo else "hi")
    return "hi" if hi else ("mid" if not lo else "lo")

def _fmt_score(v, ndigits=3):
    if v is None:
        return "—"
    return f"{v:.{ndigits}f}"


def generate_alerts_html(data):
    alerts = []
    dist = data["risk_distribution"]["distribution"]

    oc = dist.get("OVERCONFIDENT", 0)
    if oc:
        alerts.append(
            f'<div class="alert-item critical">'
            f'<div class="alert-dot"></div>'
            f'<div class="alert-body">'
            f'<strong>{oc} OVERCONFIDENT</strong> interactions — '
            f'model appears confident but changes answers when rephrased.'
            f'</div><span class="alert-count">{oc}</span></div>'
        )

    us = dist.get("UNSTABLE", 0)
    if us:
        alerts.append(
            f'<div class="alert-item warning">'
            f'<div class="alert-dot"></div>'
            f'<div class="alert-body">'
            f'<strong>{us} UNSTABLE</strong> interactions — '
            f'high response variability detected, possible hallucination.'
            f'</div><span class="alert-count">{us}</span></div>'
        )

    cal = data["overall_stats"]["mean_calibration"]
    if cal < 0.5:
        alerts.append(
            f'<div class="alert-item warning">'
            f'<div class="alert-dot"></div>'
            f'<div class="alert-body">'
            f'Poor calibration (mean: <strong>{cal:.3f}</strong>) — '
            f'confidence does not match reliability.'
            f'</div></div>'
        )

    p2 = data["phase2_stats"]
    if p2.get("drift_alert_count", 0):
        alerts.append(
            f'<div class="alert-item info">'
            f'<div class="alert-dot"></div>'
            f'<div class="alert-body">'
            f'<strong>{p2["drift_alert_count"]} semantic drift alert(s)</strong> detected '
            f'across {p2["total_windows"]} time windows.'
            f'</div></div>'
        )

    if p2.get("change_point_count", 0):
        alerts.append(
            f'<div class="alert-item critical">'
            f'<div class="alert-dot"></div>'
            f'<div class="alert-body">'
            f'<strong>{p2["change_point_count"]} behavioral change point(s)</strong> detected — '
            f'abrupt shift in uncertainty (Δ &gt; 0.25).'
            f'</div></div>'
        )

    if data["temporal"]["trends"]["uncertainty"]["trend"] in ("DEGRADING", "SLIGHTLY_DEGRADING"):
        alerts.append(
            '<div class="alert-item info">'
            '<div class="alert-dot"></div>'
            '<div class="alert-body">Uncertainty trend is <strong>degrading</strong> — '
            'model may be destabilising over time.</div></div>'
        )

    if not alerts:
        alerts.append(
            '<div class="alert-item ok">'
            '<div class="alert-dot"></div>'
            '<div class="alert-body">All systems normal — no critical alerts.</div></div>'
        )

    return "\n".join(alerts)


def generate_category_table_html(category_stats):
    rows = []
    for cat, s in sorted(category_stats.items()):
        total_z = max(s["reliable_count"] + s["overconfident_count"] +
                      s["unstable_count"] + s["ambiguous_count"], 1)
        def pct(n): return max(int(n / total_z * 100), 0 if n == 0 else 3)

        zone_bar = (
            f'<div class="zone-bar">'
            f'<div class="zone-bar-seg" style="flex:{pct(s["reliable_count"])};background:var(--green)"></div>'
            f'<div class="zone-bar-seg" style="flex:{pct(s["overconfident_count"])};background:var(--red)"></div>'
            f'<div class="zone-bar-seg" style="flex:{pct(s["unstable_count"])};background:var(--yellow)"></div>'
            f'<div class="zone-bar-seg" style="flex:{pct(s["ambiguous_count"])};background:var(--accent2)"></div>'
            f'</div>'
        )
        rows.append(f"""<tr>
          <td><span class="cat-name">{cat.replace('_',' ')}</span></td>
          <td class="num-cell">{s['count']}</td>
          <td><span class="score-pill {_pill(s['mean_uncertainty'])}">{s['mean_uncertainty']:.3f}</span></td>
          <td><span class="score-pill {_pill(s['mean_consistency'],True)}">{s['mean_consistency']:.3f}</span></td>
          <td><span class="score-pill">{s['mean_calibration']:.3f}</span></td>
          <td><span class="score-pill {_pill(s['mean_risk'])}">{s['mean_risk']:.3f}</span></td>
          <td>{zone_bar}</td>
          <td class="z-reliable">{s['reliable_count']}</td>
          <td class="z-overconf">{s['overconfident_count']}</td>
          <td class="z-unstable">{s['unstable_count']}</td>
          <td class="z-ambig">{s['ambiguous_count']}</td>
        </tr>""")

    return f"""<table class="cat-table">
      <thead><tr>
        <th>Category</th><th>n</th><th>Uncertainty</th><th>Consistency</th>
        <th>Calibration</th><th>Risk</th><th>Zone Split</th>
        <th>✅</th><th>⛔</th><th>⚠</th><th>ℹ</th>
      </tr></thead>
      <tbody>{"".join(rows)}</tbody>
    </table>"""


def generate_phase2_html(temporal_windows, phase2_stats):
    """Generate Phase 2 Temporal Intelligence HTML section."""
    if not temporal_windows:
        return '<div class="empty-state">No temporal window data available. Run with more log data to enable Phase 2 analysis.</div>'

    # Mini KPI row
    w = phase2_stats["total_windows"]
    da = phase2_stats["drift_alert_count"]
    cp = phase2_stats["change_point_count"]
    da_cls = "danger" if da > 0 else "good"
    cp_cls = "danger" if cp > 0 else "good"

    mini_kpis = f"""<div class="phase2-kpis">
      <div class="phase2-kpi"><div class="phase2-kpi-label">Windows Analyzed</div>
        <div class="phase2-kpi-value">{w}</div></div>
      <div class="phase2-kpi"><div class="phase2-kpi-label">Drift Alerts</div>
        <div class="phase2-kpi-value {da_cls}">{da}</div></div>
      <div class="phase2-kpi"><div class="phase2-kpi-label">Change Points</div>
        <div class="phase2-kpi-value {cp_cls}">{cp}</div></div>
    </div>"""

    # Window table rows
    rows = []
    for w in temporal_windows:
        key = w["window_key"]
        n   = w["num_samples"]
        mu  = w["mean_uncertainty"]
        mc  = w["mean_consistency"]
        du  = w.get("delta_uncertainty") or 0.0
        ds  = w.get("drift_score")
        da  = w.get("drift_alert", False)
        cp  = w.get("change_point", False)

        du_str  = f'<span class="delta {("neg" if du < 0 else "pos")}">{du:+.3f}</span>'
        ds_str  = f'<span class="drift-val {"alert" if da else ""}">{ds:.3f}</span>' if ds is not None else '<span class="muted">—</span>'

        drift_bar = ""
        if ds is not None:
            pct = min(int(ds * 400), 100)
            col = "#ff4560" if da else ("#ffcb47" if ds > 0.08 else "#00e5ff")
            drift_bar = f'<div class="drift-bar-wrap"><div class="drift-bar-fill" style="width:{pct}%;background:{col}"></div></div>'

        alert_badge = '<span class="w-badge alert">DRIFT</span>' if da else ''
        cp_badge    = '<span class="w-badge change-pt">CHANGE⚡</span>' if cp else ''

        rows.append(f"""<tr class="{'w-row-alert' if (da or cp) else ''}">
          <td class="w-key">{key}</td>
          <td class="num-cell">{n}</td>
          <td><span class="score-pill {_pill(mu)}">{mu:.3f}</span></td>
          <td><span class="score-pill {_pill(mc,True)}">{mc:.3f}</span></td>
          <td>{du_str}</td>
          <td>{ds_str}{drift_bar}</td>
          <td>{alert_badge}{cp_badge}</td>
        </tr>""")

    table = f"""<div class="window-table-wrap">
      <table class="window-table">
        <thead><tr>
          <th>Window</th><th>Samples</th><th>Uncertainty</th><th>Consistency</th>
          <th>Δ Uncertainty</th><th>Drift Score</th><th>Flags</th>
        </tr></thead>
        <tbody>{"".join(rows)}</tbody>
      </table>
    </div>"""

    return mini_kpis + '\n' + table


def generate_temporal_html(temporal):
    """Generate rolling-window trend cards."""
    u_trend = temporal["trends"]["uncertainty"]
    c_trend = temporal["trends"]["consistency"]

    def tc(t, invert=False):
        if "DEGRADING" in t: return "bad" if not invert else "good"
        if "IMPROVING" in t: return "good" if not invert else "bad"
        return "neutral"

    rw = temporal["rolling_window_5"] or {}
    return f"""<div class="temporal-grid">
      <div class="trend-card">
        <div class="trend-label">Uncertainty Trend</div>
        <div class="trend-value {tc(u_trend['trend'])}">{u_trend['trend']}</div>
        <div class="trend-desc">{u_trend['description']}</div>
      </div>
      <div class="trend-card">
        <div class="trend-label">Consistency Trend</div>
        <div class="trend-value {tc(c_trend['trend'], True)}">{c_trend['trend']}</div>
        <div class="trend-desc">{c_trend['description']}</div>
      </div>
      <div class="trend-card">
        <div class="trend-label">Rolling Window · Last 5</div>
        <div class="stat-group">
          <div class="stat-row"><span class="stat-key">Uncertainty</span>
            <span class="stat-val">{rw.get('mean_uncertainty',0):.3f}</span></div>
          <div class="stat-row"><span class="stat-key">Consistency</span>
            <span class="stat-val">{rw.get('mean_consistency',0):.3f}</span></div>
          <div class="stat-row"><span class="stat-key">Calibration</span>
            <span class="stat-val">{rw.get('mean_calibration',0):.3f}</span></div>
        </div>
      </div>
    </div>"""


def generate_recent_table_html(recent):
    rows = []
    for record in reversed(recent):
        zone  = record.get("risk_zone", "UNKNOWN")
        badge = f"risk-{zone.lower()}"
        u   = record.get("uncertainty_score")
        c   = record.get("consistency_score")
        cal = record.get("calibration_score")
        q     = record["question"]
        q_d   = (q[:68] + "…") if len(q) > 68 else q

        rows.append(f"""<tr>
          <td class="ts-cell">{format_timestamp(record['timestamp'])}</td>
          <td><span class="q-text" title="{q}">{q_d}</span></td>
          <td><span class="risk-badge {badge}">{zone}</span></td>
          <td><span class="score-pill {_pill(u)}" style="font-family:var(--mono);font-size:11px">{_fmt_score(u)}</span></td>
          <td><span class="score-pill {_pill(c,True)}" style="font-family:var(--mono);font-size:11px">{_fmt_score(c)}</span></td>
          <td style="font-family:var(--mono);font-size:11px;color:var(--accent)">{_fmt_score(cal)}</td>
        </tr>""")

    return f"""<table class="recent-table">
      <thead><tr>
        <th>Timestamp</th><th>Question</th><th>Zone</th>
        <th>Uncertainty</th><th>Consistency</th><th>Calibration</th>
      </tr></thead>
      <tbody>{"".join(rows)}</tbody>
    </table>"""


# =========================
# MAIN HTML GENERATOR
# =========================

def generate_html_dashboard(data):
    ts_data       = json.dumps(data["time_series"])
    risk_data_js  = json.dumps(data["risk_distribution"]["distribution"])
    phase2_json   = json.dumps([
        {k: v for k, v in w.items() if k != "risk_distribution"}
        for w in data["temporal_windows"]
    ])
    cat_json      = json.dumps({
        cat: {
            "uncertainty": round(s["mean_uncertainty"], 3),
            "consistency": round(s["mean_consistency"], 3),
            "count":       s["count"],
        }
        for cat, s in data["category_stats"].items()
    })

    health     = data["risk_distribution"]["health_score"]
    health_cls = "good" if health >= 0.7 else "warn" if health >= 0.4 else "danger"
    cal        = data["overall_stats"]["mean_calibration"]
    cal_cls    = "good" if cal >= 0.6 else "warn" if cal >= 0.4 else "danger"
    crit       = data["risk_distribution"]["critical_count"]
    crit_cls   = "danger" if crit > 0 else "good"
    da         = data["phase2_stats"]["drift_alert_count"]
    da_cls     = "danger" if da > 0 else "good"

    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>LLM Watchdog — Behavioral Reliability Monitor</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Inter:wght@300;400;500;600&display=swap" rel="stylesheet">
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>
<style>
*, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}

:root {{
  --bg:       #080a0f;
  --surface:  #0e1117;
  --surface2: #141720;
  --surface3: #1a1e2a;
  --border:   #1e2535;
  --border2:  #2a3048;
  --text:     #c4cde0;
  --muted:    #4e566b;
  --muted2:   #6b7591;
  --accent:   #00e5ff;
  --accent2:  #7b61ff;
  --green:    #00d47e;
  --yellow:   #f5c842;
  --red:      #ff3d5a;
  --orange:   #ff8c42;
  --mono:     'Space Mono', monospace;
  --sans:     'Inter', sans-serif;
  --r:        8px;
  --r-sm:     5px;
}}

html {{ scroll-behavior: smooth; }}

body {{
  background: var(--bg);
  color: var(--text);
  font-family: var(--sans);
  font-size: 13.5px;
  line-height: 1.55;
  min-height: 100vh;
  overflow-x: hidden;
}}

/* subtle dot grid background */
body::before {{
  content: '';
  position: fixed; inset: 0;
  background-image: radial-gradient(circle, #ffffff06 1px, transparent 1px);
  background-size: 28px 28px;
  pointer-events: none;
  z-index: 0;
}}

/* ── SCROLLBAR ── */
::-webkit-scrollbar {{ width: 5px; height: 5px; }}
::-webkit-scrollbar-track {{ background: var(--bg); }}
::-webkit-scrollbar-thumb {{ background: var(--border2); border-radius: 4px; }}

/* ── HEADER ── */
header {{
  position: sticky; top: 0; z-index: 200;
  background: rgba(8,10,15,0.92);
  backdrop-filter: blur(16px);
  border-bottom: 1px solid var(--border);
  display: flex; align-items: center; justify-content: space-between;
  padding: 0 24px;
  height: 52px;
}}

.logo {{ display: flex; align-items: center; gap: 12px; }}
.logo-icon {{
  width: 26px; height: 26px;
  background: linear-gradient(135deg, var(--accent), var(--accent2));
  clip-path: polygon(50% 0%, 100% 25%, 100% 75%, 50% 100%, 0% 75%, 0% 25%);
  animation: hex-pulse 3s ease-in-out infinite;
}}
@keyframes hex-pulse {{
  0%,100% {{ filter: brightness(1) drop-shadow(0 0 4px rgba(0,229,255,0.4)); }}
  50%      {{ filter: brightness(1.3) drop-shadow(0 0 10px rgba(0,229,255,0.7)); }}
}}
.logo-text {{
  font-family: var(--mono);
  font-size: 12px; font-weight: 700;
  letter-spacing: 0.14em; text-transform: uppercase;
  color: #fff;
}}
.logo-sub {{
  font-family: var(--mono); font-size: 9.5px;
  color: var(--muted); letter-spacing: 0.1em; text-transform: uppercase;
}}

.header-right {{ display: flex; align-items: center; gap: 18px; }}
.live-badge {{
  display: flex; align-items: center; gap: 6px;
  font-family: var(--mono); font-size: 9.5px;
  color: var(--green); letter-spacing: 0.1em; text-transform: uppercase;
}}
.live-dot {{
  width: 6px; height: 6px; border-radius: 50%;
  background: var(--green);
  box-shadow: 0 0 6px var(--green);
  animation: blink 1.4s step-end infinite;
}}
@keyframes blink {{ 0%,100% {{ opacity:1; }} 50% {{ opacity:0.2; }} }}
.header-clock {{
  font-family: var(--mono); font-size: 10.5px;
  color: var(--muted2);
}}
.model-badge {{
  background: var(--surface2);
  border: 1px solid var(--border2);
  border-radius: 20px;
  padding: 2px 10px;
  font-family: var(--mono); font-size: 9.5px;
  color: var(--accent); letter-spacing: 0.06em;
}}

/* ── TAB NAV ── */
.tab-nav {{
  position: sticky; top: 52px; z-index: 190;
  background: rgba(8,10,15,0.9);
  backdrop-filter: blur(12px);
  border-bottom: 1px solid var(--border);
  display: flex; align-items: center;
  padding: 0 24px; gap: 0;
}}
.tab-btn {{
  padding: 12px 18px;
  font-family: var(--mono); font-size: 10px;
  letter-spacing: 0.12em; text-transform: uppercase;
  color: var(--muted2);
  background: none; border: none;
  border-bottom: 2px solid transparent;
  cursor: pointer;
  transition: color 0.2s, border-color 0.2s;
}}
.tab-btn:hover {{ color: var(--text); }}
.tab-btn.active {{
  color: var(--accent);
  border-bottom-color: var(--accent);
}}

/* ── MAIN ── */
main {{
  position: relative; z-index: 1;
  max-width: 1480px; margin: 0 auto;
  padding: 24px 24px 48px;
  display: flex; flex-direction: column; gap: 32px;
}}

section {{ display: flex; flex-direction: column; gap: 16px; }}

/* ── SECTION LABEL ── */
.section-label {{
  display: flex; align-items: center; gap: 10px;
  font-family: var(--mono); font-size: 9.5px;
  letter-spacing: 0.22em; text-transform: uppercase;
  color: var(--muted);
}}
.section-label .s-tag {{
  background: var(--surface2);
  border: 1px solid var(--border);
  border-radius: 3px;
  padding: 2px 8px;
  white-space: nowrap;
}}
.section-label::after {{
  content: ''; flex: 1;
  height: 1px; background: var(--border);
}}

/* ── KPI CARDS ── */
.kpi-grid {{
  display: grid;
  grid-template-columns: repeat(6, 1fr);
  gap: 12px;
}}
.kpi-card {{
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: var(--r);
  padding: 18px 20px;
  position: relative; overflow: hidden;
  transition: border-color 0.2s, transform 0.18s;
}}
.kpi-card:hover {{ border-color: var(--border2); transform: translateY(-2px); }}
.kpi-card::before {{
  content: '';
  position: absolute; top: 0; left: 0; right: 0; height: 2px;
  background: var(--kpi-accent, var(--border2));
}}
.kpi-card.k-cyan   {{ --kpi-accent: var(--accent); }}
.kpi-card.k-purple {{ --kpi-accent: var(--accent2); }}
.kpi-card.k-green  {{ --kpi-accent: var(--green); }}
.kpi-card.k-red    {{ --kpi-accent: var(--red); }}
.kpi-card.k-yellow {{ --kpi-accent: var(--yellow); }}
.kpi-card.k-orange {{ --kpi-accent: var(--orange); }}
.kpi-label {{
  font-family: var(--mono); font-size: 9px;
  letter-spacing: 0.16em; text-transform: uppercase;
  color: var(--muted); margin-bottom: 10px;
}}
.kpi-value {{
  font-family: var(--mono); font-size: 32px;
  font-weight: 700; line-height: 1;
  letter-spacing: -0.02em; color: #fff;
}}
.kpi-value.good   {{ color: var(--green); }}
.kpi-value.warn   {{ color: var(--yellow); }}
.kpi-value.danger {{ color: var(--red); }}
.kpi-sub {{
  font-size: 10.5px; color: var(--muted);
  margin-top: 7px; line-height: 1.3;
}}
.kpi-glow {{
  position: absolute; bottom: 0; right: 0;
  width: 60px; height: 60px;
  border-radius: 50%;
  opacity: 0.05;
  background: var(--kpi-accent, transparent);
  transform: translate(20px, 20px);
}}

/* ── ALERTS ── */
.alerts-box {{
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: var(--r);
  overflow: hidden;
}}
.alerts-header {{
  padding: 11px 16px;
  border-bottom: 1px solid var(--border);
  display: flex; align-items: center; gap: 8px;
  font-family: var(--mono); font-size: 9px;
  letter-spacing: 0.18em; text-transform: uppercase;
  color: var(--muted);
}}
.alert-item {{
  display: flex; align-items: center; gap: 12px;
  padding: 11px 16px;
  border-bottom: 1px solid var(--border);
  font-size: 12.5px;
  animation: slide-in 0.35s ease;
}}
.alert-item:last-child {{ border-bottom: none; }}
@keyframes slide-in {{ from {{ opacity:0; transform:translateX(-6px); }} to {{ opacity:1; transform:translateX(0); }} }}
.alert-dot {{
  width: 8px; height: 8px; border-radius: 50%; flex-shrink: 0;
}}
.alert-item.critical {{ border-left: 2px solid var(--red);    background: rgba(255,61,90,0.04); }}
.alert-item.critical .alert-dot {{ background: var(--red);    box-shadow: 0 0 6px var(--red); }}
.alert-item.warning  {{ border-left: 2px solid var(--yellow); background: rgba(245,200,66,0.04); }}
.alert-item.warning  .alert-dot {{ background: var(--yellow); box-shadow: 0 0 6px var(--yellow); }}
.alert-item.info     {{ border-left: 2px solid var(--accent);  background: rgba(0,229,255,0.04); }}
.alert-item.info     .alert-dot {{ background: var(--accent);  box-shadow: 0 0 6px var(--accent); }}
.alert-item.ok       {{ border-left: 2px solid var(--green);   background: rgba(0,212,126,0.04); }}
.alert-item.ok       .alert-dot {{ background: var(--green);   box-shadow: 0 0 6px var(--green); }}
.alert-body {{ flex: 1; line-height: 1.4; }}
.alert-count {{
  font-family: var(--mono); font-size: 11px; font-weight: 700;
  color: var(--muted2);
}}

/* ── PANELS ── */
.panel {{
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: var(--r);
  overflow: hidden;
}}
.panel-header {{
  padding: 12px 18px;
  border-bottom: 1px solid var(--border);
  display: flex; align-items: center; justify-content: space-between;
}}
.panel-title {{
  font-family: var(--mono); font-size: 10px;
  letter-spacing: 0.14em; text-transform: uppercase;
  color: var(--text);
}}
.panel-meta {{
  font-family: var(--mono); font-size: 9.5px;
  color: var(--muted);
}}
.panel-body {{ padding: 18px; }}

/* ── CHART LAYOUT ── */
.chart-row-main {{
  display: grid;
  grid-template-columns: 2fr 1fr;
  gap: 12px;
}}
.chart-row-full {{
  display: grid;
  grid-template-columns: 1fr;
}}

/* donut center */
.donut-wrap {{
  position: relative;
  display: flex; align-items: center; justify-content: center;
}}
.donut-center {{
  position: absolute;
  text-align: center; pointer-events: none;
}}
.donut-center-val {{
  font-family: var(--mono); font-size: 26px; font-weight: 700;
  line-height: 1; color: #fff;
}}
.donut-center-lbl {{
  font-family: var(--mono); font-size: 9px;
  letter-spacing: 0.1em; text-transform: uppercase;
  color: var(--muted); margin-top: 4px;
}}

/* ── TABLES ── */
.cat-table, .window-table, .recent-table {{
  width: 100%; border-collapse: collapse; font-size: 12px;
}}
.cat-table th, .window-table th, .recent-table th {{
  font-family: var(--mono); font-size: 9px;
  letter-spacing: 0.15em; text-transform: uppercase;
  color: var(--muted);
  padding: 9px 12px;
  text-align: left;
  border-bottom: 1px solid var(--border);
  white-space: nowrap;
  background: var(--surface2);
}}
.cat-table td, .window-table td, .recent-table td {{
  padding: 9px 12px;
  border-bottom: 1px solid var(--border);
  vertical-align: middle;
  font-family: var(--mono); font-size: 11px;
}}
.cat-table tr:last-child td, .window-table tr:last-child td,
.recent-table tr:last-child td {{ border-bottom: none; }}
.cat-table tr:hover td, .window-table tr:hover td,
.recent-table tr:hover td {{ background: rgba(255,255,255,0.018); }}

.cat-name {{
  color: #fff; font-weight: 700; font-size: 11.5px;
  letter-spacing: 0.04em; text-transform: capitalize;
}}
.num-cell {{ color: #fff; font-weight: 700; }}

/* zone bar */
.zone-bar {{
  display: flex; height: 5px; border-radius: 4px;
  overflow: hidden; gap: 2px; min-width: 90px;
}}
.zone-bar-seg {{ border-radius: 2px; min-width: 3px; }}

/* score pill */
.score-pill {{ font-family: var(--mono); font-size: 11px; }}
.score-pill.hi  {{ color: var(--red); }}
.score-pill.mid {{ color: var(--yellow); }}
.score-pill.lo  {{ color: var(--green); }}

/* risk badge */
.risk-badge {{
  display: inline-block;
  padding: 2px 8px; border-radius: 4px;
  font-family: var(--mono); font-size: 9px;
  letter-spacing: 0.12em; text-transform: uppercase; font-weight: 700;
}}
.risk-reliable      {{ background: rgba(0,212,126,0.12);  color: var(--green);  border: 1px solid rgba(0,212,126,0.25); }}
.risk-overconfident {{ background: rgba(255,61,90,0.12);   color: var(--red);    border: 1px solid rgba(255,61,90,0.25); }}
.risk-unstable      {{ background: rgba(245,200,66,0.12);  color: var(--yellow); border: 1px solid rgba(245,200,66,0.25); }}
.risk-ambiguous     {{ background: rgba(123,97,255,0.12);  color: var(--accent2);border: 1px solid rgba(123,97,255,0.25); }}
.risk-unknown       {{ background: rgba(78,86,107,0.12);   color: var(--muted2); border: 1px solid rgba(78,86,107,0.25); }}

/* zone count cells */
.z-reliable {{ color: var(--green); }}
.z-overconf {{ color: var(--red); }}
.z-unstable {{ color: var(--yellow); }}
.z-ambig    {{ color: var(--accent2); }}

/* ── TEMPORAL SECTION ── */
.phase2-kpis {{
  display: grid; grid-template-columns: repeat(3,1fr); gap: 12px;
}}
.phase2-kpi {{
  background: var(--surface2);
  border: 1px solid var(--border);
  border-radius: var(--r-sm);
  padding: 14px 16px;
}}
.phase2-kpi-label {{
  font-family: var(--mono); font-size: 9px;
  letter-spacing: 0.14em; text-transform: uppercase;
  color: var(--muted); margin-bottom: 8px;
}}
.phase2-kpi-value {{
  font-family: var(--mono); font-size: 26px; font-weight: 700;
  color: var(--accent);
}}
.phase2-kpi-value.good   {{ color: var(--green); }}
.phase2-kpi-value.warn   {{ color: var(--yellow); }}
.phase2-kpi-value.danger {{ color: var(--red); }}

.window-table-wrap {{
  overflow-x: auto;
  border: 1px solid var(--border);
  border-radius: var(--r);
  background: var(--surface);
}}
.w-key {{ color: #fff; font-weight: 700; letter-spacing: 0.04em; }}
.w-row-alert td {{ background: rgba(255,61,90,0.04); }}

.drift-bar-wrap {{
  height: 4px; background: var(--border);
  border-radius: 2px; margin-top: 5px;
  overflow: hidden; min-width: 80px;
}}
.drift-bar-fill {{ height: 100%; border-radius: 2px; transition: width 0.6s ease; }}

.drift-val {{ font-family: var(--mono); font-size: 11px; }}
.drift-val.alert {{ color: var(--red); }}

.delta {{ font-family: var(--mono); font-size: 11px; }}
.delta.pos {{ color: var(--red); }}
.delta.neg {{ color: var(--green); }}

.w-badge {{
  display: inline-block; padding: 1px 6px; border-radius: 3px;
  font-family: var(--mono); font-size: 8px;
  letter-spacing: 0.1em; font-weight: 700;
  margin-right: 3px;
}}
.w-badge.alert     {{ background: rgba(255,61,90,0.18); color: var(--red); border: 1px solid rgba(255,61,90,0.3); }}
.w-badge.change-pt {{ background: rgba(245,200,66,0.18); color: var(--yellow); border: 1px solid rgba(245,200,66,0.3); }}

.empty-state {{
  padding: 40px; text-align: center;
  color: var(--muted); font-size: 13px;
  background: var(--surface); border: 1px solid var(--border);
  border-radius: var(--r);
}}

/* ── TREND CARDS ── */
.temporal-grid {{
  display: grid; grid-template-columns: repeat(3,1fr); gap: 12px;
}}
.trend-card {{
  background: var(--surface2); border: 1px solid var(--border);
  border-radius: var(--r); padding: 16px;
}}
.trend-label {{
  font-family: var(--mono); font-size: 9px;
  letter-spacing: 0.15em; text-transform: uppercase;
  color: var(--muted); margin-bottom: 9px;
}}
.trend-value {{
  font-family: var(--mono); font-size: 13px; font-weight: 700;
  letter-spacing: 0.04em; margin-bottom: 5px;
}}
.trend-value.good    {{ color: var(--green); }}
.trend-value.bad     {{ color: var(--red); }}
.trend-value.neutral {{ color: var(--yellow); }}
.trend-desc {{ font-size: 11.5px; color: var(--muted2); line-height: 1.4; }}

.stat-group {{ display: flex; flex-direction: column; gap: 7px; }}
.stat-row {{
  display: flex; justify-content: space-between; align-items: center;
  padding: 7px 10px;
  background: var(--surface); border-radius: 4px;
  border: 1px solid var(--border);
}}
.stat-key {{
  font-family: var(--mono); font-size: 9.5px;
  letter-spacing: 0.1em; text-transform: uppercase;
  color: var(--muted);
}}
.stat-val {{
  font-family: var(--mono); font-size: 13px; font-weight: 700;
  color: var(--accent);
}}

/* ── RECENT TABLE ── */
.q-text {{
  color: var(--text); max-width: 360px;
  overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
  font-family: var(--sans); font-size: 12px;
}}
.ts-cell {{
  font-family: var(--mono); font-size: 10px;
  color: var(--muted2); white-space: nowrap;
}}

/* ── FOOTER ── */
footer {{
  border-top: 1px solid var(--border);
  padding: 14px 24px;
  display: flex; align-items: center; justify-content: space-between;
  font-family: var(--mono); font-size: 9.5px;
  color: var(--muted); letter-spacing: 0.1em;
}}
footer .f-brand {{ color: var(--accent); font-weight: 700; }}
.muted {{ color: var(--muted2); }}

/* ── RESPONSIVE ── */
@media (max-width: 1100px) {{ .kpi-grid {{ grid-template-columns: repeat(3,1fr); }} }}
@media (max-width: 800px)  {{
  .kpi-grid {{ grid-template-columns: repeat(2,1fr); }}
  .chart-row-main {{ grid-template-columns: 1fr; }}
  .temporal-grid {{ grid-template-columns: 1fr; }}
  .phase2-kpis {{ grid-template-columns: 1fr; }}
}}
@media (max-width: 500px)  {{ .kpi-grid {{ grid-template-columns: 1fr; }} }}
</style>
</head>
<body>

<header>
  <div class="logo">
    <div class="logo-icon"></div>
    <div>
      <div class="logo-text">LLM Watchdog</div>
      <div class="logo-sub">Behavioral Reliability Monitor</div>
    </div>
  </div>
  <div class="header-right">
    <div class="live-badge"><div class="live-dot"></div>Live</div>
    <div class="model-badge">{data['model']}</div>
    <div class="header-clock" id="live-clock">{now_str}</div>
  </div>
</header>

<nav class="tab-nav">
  <button class="tab-btn active" data-target="overview">Overview</button>
  <button class="tab-btn" data-target="temporal">Temporal</button>
  <button class="tab-btn" data-target="categories">Categories</button>
  <button class="tab-btn" data-target="recent">Recent</button>
</nav>

<main>

<!-- ══════════════════════════════════════════════
     SECTION 1 — OVERVIEW
══════════════════════════════════════════════ -->
<section id="overview">
  <div class="section-label"><span class="s-tag">01</span> System Overview</div>

  <!-- KPI CARDS -->
  <div class="kpi-grid">
    <div class="kpi-card k-cyan">
      <div class="kpi-label">Total Interactions</div>
      <div class="kpi-value" data-count="{data['total_interactions']}">{data['total_interactions']}</div>
      <div class="kpi-sub">Monitoring sessions logged</div>
      <div class="kpi-glow"></div>
    </div>
    <div class="kpi-card k-purple">
      <div class="kpi-label">System Health</div>
      <div class="kpi-value {health_cls}" data-count="{health:.2f}">{health:.2f}</div>
      <div class="kpi-sub">Out of 1.0 — weighted reliability</div>
      <div class="kpi-glow"></div>
    </div>
    <div class="kpi-card k-green">
      <div class="kpi-label">Reliable Interactions</div>
      <div class="kpi-value good" data-count="{data['reliable_pct']}">{data['reliable_pct']}%</div>
      <div class="kpi-sub">Classified as RELIABLE zone</div>
      <div class="kpi-glow"></div>
    </div>
    <div class="kpi-card k-red">
      <div class="kpi-label">Critical Issues</div>
      <div class="kpi-value {crit_cls}" data-count="{crit}">{crit}</div>
      <div class="kpi-sub">OVERCONFIDENT + UNSTABLE</div>
      <div class="kpi-glow"></div>
    </div>
    <div class="kpi-card k-yellow">
      <div class="kpi-label">Mean Calibration</div>
      <div class="kpi-value {cal_cls}" data-count="{cal:.3f}">{cal:.3f}</div>
      <div class="kpi-sub">Consistency / (1 + Uncertainty)</div>
      <div class="kpi-glow"></div>
    </div>
    <div class="kpi-card k-orange">
      <div class="kpi-label">Drift Alerts</div>
      <div class="kpi-value {da_cls}" data-count="{da}">{da}</div>
      <div class="kpi-sub">Phase 2 · semantic drift events</div>
      <div class="kpi-glow"></div>
    </div>
  </div>

  <!-- ALERTS -->
  <div class="alerts-box">
    <div class="alerts-header">◈ &nbsp;Active Alerts</div>
    {generate_alerts_html(data)}
  </div>

  <!-- MAIN CHARTS -->
  <div class="chart-row-main">
    <div class="panel">
      <div class="panel-header">
        <span class="panel-title">Metrics Over Time</span>
        <span class="panel-meta">{len(data['time_series']['timestamps'])} samples &nbsp;·&nbsp; raw + MA-10 rolling average</span>
      </div>
      <div class="panel-body"><canvas id="tsChart" height="210"></canvas></div>
    </div>
    <div class="panel">
      <div class="panel-header">
        <span class="panel-title">Risk Zone Distribution</span>
        <span class="panel-meta">{data['total_interactions']} total</span>
      </div>
      <div class="panel-body">
        <div class="donut-wrap">
          <canvas id="riskChart" height="220"></canvas>
          <div class="donut-center">
            <div class="donut-center-val">{health:.0%}</div>
            <div class="donut-center-lbl">health</div>
          </div>
        </div>
      </div>
    </div>
  </div>
</section>

<!-- ══════════════════════════════════════════════
     SECTION 2 — TEMPORAL INTELLIGENCE
══════════════════════════════════════════════ -->
<section id="temporal">
  <div class="section-label"><span class="s-tag">02</span> Phase 2 · Temporal Intelligence</div>

  {generate_phase2_html(data['temporal_windows'], data['phase2_stats'])}

  <!-- Drift Timeline Chart -->
  <div class="panel">
    <div class="panel-header">
      <span class="panel-title">Semantic Drift Timeline</span>
      <span class="panel-meta">drift score per 24h window &nbsp;·&nbsp; red bar = alert threshold crossed</span>
    </div>
    <div class="panel-body"><canvas id="driftChart" height="180"></canvas></div>
  </div>

  <!-- Rolling trends -->
  <div class="section-label"><span class="s-tag">↳</span> Rolling Window Trends</div>
  {generate_temporal_html(data['temporal'])}
</section>

<!-- ══════════════════════════════════════════════
     SECTION 3 — CATEGORIES
══════════════════════════════════════════════ -->
<section id="categories">
  <div class="section-label"><span class="s-tag">03</span> Category Performance</div>

  <!-- Category chart -->
  <div class="panel">
    <div class="panel-header">
      <span class="panel-title">Uncertainty vs Consistency by Category</span>
      <span class="panel-meta">lower uncertainty + higher consistency = more reliable</span>
    </div>
    <div class="panel-body"><canvas id="catChart" height="200"></canvas></div>
  </div>

  <!-- Category table -->
  <div class="panel">
    <div style="overflow-x:auto;">
      {generate_category_table_html(data['category_stats'])}
    </div>
  </div>
</section>

<!-- ══════════════════════════════════════════════
     SECTION 4 — RECENT INTERACTIONS
══════════════════════════════════════════════ -->
<section id="recent">
  <div class="section-label"><span class="s-tag">04</span> Recent Interactions</div>
  <div class="panel">
    <div style="overflow-x:auto;">
      {generate_recent_table_html(data['recent_interactions'])}
    </div>
  </div>
</section>

</main>

<footer>
  <span class="f-brand">LLM WATCHDOG</span>
  <div>Generated: {now_str} UTC</div>
  <div>Data range: {data['time_range']['first']} → {data['time_range']['last']}</div>
</footer>

<script>
// ══════════════════════════════════════════════
// LIVE CLOCK
// ══════════════════════════════════════════════
(function tick() {{
  const el = document.getElementById('live-clock');
  if (el) el.textContent = new Date().toISOString().replace('T',' ').slice(0,19);
  setTimeout(tick, 1000);
}})();

// ══════════════════════════════════════════════
// TAB NAVIGATION (smooth scroll + active state)
// ══════════════════════════════════════════════
document.querySelectorAll('.tab-btn').forEach(btn => {{
  btn.addEventListener('click', () => {{
    document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    const target = document.getElementById(btn.dataset.target);
    if (target) target.scrollIntoView({{ behavior: 'smooth', block: 'start' }});
  }});
}});

// Highlight tab on scroll
const sections = ['overview','temporal','categories','recent'];
const observer = new IntersectionObserver(entries => {{
  entries.forEach(e => {{
    if (e.isIntersecting) {{
      document.querySelectorAll('.tab-btn').forEach(b => {{
        b.classList.toggle('active', b.dataset.target === e.target.id);
      }});
    }}
  }});
}}, {{ threshold: 0.3 }});
sections.forEach(id => {{ const el = document.getElementById(id); if (el) observer.observe(el); }});

// ══════════════════════════════════════════════
// CHART.JS GLOBAL DEFAULTS
// ══════════════════════════════════════════════
Chart.defaults.color = '#4e566b';
Chart.defaults.borderColor = '#1e2535';
Chart.defaults.font.family = "'Space Mono', monospace";
Chart.defaults.font.size = 10;

const TOOLTIP_STYLE = {{
  backgroundColor: '#0e1117',
  borderColor: '#2a3048',
  borderWidth: 1,
  titleColor: '#c4cde0',
  bodyColor: '#6b7591',
  padding: 10,
  cornerRadius: 6,
}};

// ══════════════════════════════════════════════
// TIME SERIES CHART
// ══════════════════════════════════════════════
const tsData = {ts_data};

// Rolling average helper (window = 10 samples)
function rollingAvg(arr, win) {{
  return arr.map((_, i) => {{
    const sl = arr.slice(Math.max(0, i - win + 1), i + 1);
    return sl.reduce((a, b) => a + b, 0) / sl.length;
  }});
}}
const RA_WIN = 10;
const raUnc = rollingAvg(tsData.uncertainty, RA_WIN);
const raCon = rollingAvg(tsData.consistency, RA_WIN);

new Chart(document.getElementById('tsChart').getContext('2d'), {{
  type: 'line',
  data: {{
    labels: tsData.timestamps,
    datasets: [
      {{
        label: 'Uncertainty',
        data: tsData.uncertainty,
        borderColor: 'rgba(255,61,90,0.35)',
        backgroundColor: (ctx) => {{
          const g = ctx.chart.ctx.createLinearGradient(0,0,0,ctx.chart.height);
          g.addColorStop(0,'rgba(255,61,90,0.10)'); g.addColorStop(1,'rgba(255,61,90,0)');
          return g;
        }},
        borderWidth: 1, pointRadius: 0, tension: 0.4, fill: true, order: 2,
      }},
      {{
        label: 'Consistency',
        data: tsData.consistency,
        borderColor: 'rgba(0,229,255,0.3)',
        backgroundColor: (ctx) => {{
          const g = ctx.chart.ctx.createLinearGradient(0,0,0,ctx.chart.height);
          g.addColorStop(0,'rgba(0,229,255,0.07)'); g.addColorStop(1,'rgba(0,229,255,0)');
          return g;
        }},
        borderWidth: 1, pointRadius: 0, tension: 0.4, fill: true, order: 2,
      }},
      {{
        label: 'Calibration',
        data: tsData.calibration,
        borderColor: 'rgba(0,212,126,0.4)',
        backgroundColor: 'transparent',
        borderWidth: 1, pointRadius: 0, tension: 0.4, fill: false,
        borderDash: [4,3], order: 2,
      }},
      {{
        label: 'Unc MA-10',
        data: raUnc,
        borderColor: '#ff3d5a',
        backgroundColor: 'transparent',
        borderWidth: 2.2, pointRadius: 0, tension: 0.4, fill: false, order: 1,
      }},
      {{
        label: 'Con MA-10',
        data: raCon,
        borderColor: '#00e5ff',
        backgroundColor: 'transparent',
        borderWidth: 2.2, pointRadius: 0, tension: 0.4, fill: false, order: 1,
      }},
    ],
  }},
  options: {{
    responsive: true,
    interaction: {{ mode: 'index', intersect: false }},
    plugins: {{
      legend: {{ labels: {{ boxWidth: 10, boxHeight: 2, padding: 16, color: '#6b7591' }} }},
      tooltip: TOOLTIP_STYLE,
    }},
    scales: {{
      x: {{ ticks: {{ maxTicksLimit: 10, color: '#4e566b' }}, grid: {{ color: '#14181f' }} }},
      y: {{ min: -0.05, max: 1.05, ticks: {{ color: '#4e566b', stepSize: 0.25 }}, grid: {{ color: '#14181f' }} }},
    }},
  }},
}});

// ══════════════════════════════════════════════
// RISK DONUT CHART
// ══════════════════════════════════════════════
const riskData = {risk_data_js};
const ZONE_COLORS = {{
  RELIABLE:'#00d47e', OVERCONFIDENT:'#ff3d5a',
  UNSTABLE:'#f5c842',  AMBIGUOUS:'#7b61ff',
}};
const rLabels = Object.keys(riskData);

new Chart(document.getElementById('riskChart').getContext('2d'), {{
  type: 'doughnut',
  data: {{
    labels: rLabels,
    datasets: [{{
      data: Object.values(riskData),
      backgroundColor: rLabels.map(l => (ZONE_COLORS[l]||'#4e566b') + 'bb'),
      borderColor:     rLabels.map(l => ZONE_COLORS[l]||'#4e566b'),
      borderWidth: 1.5,
      hoverOffset: 8,
    }}],
  }},
  options: {{
    responsive: true, cutout: '70%',
    plugins: {{
      legend: {{ position: 'bottom', labels: {{ boxWidth: 10, boxHeight: 10, padding: 14, color: '#6b7591' }} }},
      tooltip: TOOLTIP_STYLE,
    }},
  }},
}});

// ══════════════════════════════════════════════
// DRIFT TIMELINE CHART (Phase 2)
// ══════════════════════════════════════════════
const temporalWindows = {phase2_json};

if (temporalWindows.length > 0 && document.getElementById('driftChart')) {{
  const wLabels   = temporalWindows.map(w => w.window_key);
  const driftVals = temporalWindows.map(w => w.drift_score ?? 0);
  const cpVals    = temporalWindows.map(w => w.change_point ? (w.drift_score ?? 0) : null);
  const DRIFT_THR = 0.15;

  new Chart(document.getElementById('driftChart').getContext('2d'), {{
    type: 'bar',
    data: {{
      labels: wLabels,
      datasets: [
        {{
          label: 'Drift Score',
          data: driftVals,
          backgroundColor: driftVals.map(d =>
            d > DRIFT_THR ? 'rgba(255,61,90,0.75)'
            : d > 0.08 ? 'rgba(245,200,66,0.65)'
            : 'rgba(0,229,255,0.5)'
          ),
          borderColor: driftVals.map(d =>
            d > DRIFT_THR ? '#ff3d5a'
            : d > 0.08 ? '#f5c842'
            : '#00e5ff'
          ),
          borderWidth: 1,
          borderRadius: 3,
          order: 2,
        }},
        {{
          label: 'Change Point',
          data: cpVals,
          type: 'scatter',
          pointStyle: 'triangle',
          pointRadius: 9,
          backgroundColor: '#f5c842',
          borderColor: '#fff',
          borderWidth: 1.5,
          order: 1,
        }},
        {{
          label: 'Alert Threshold',
          data: wLabels.map(() => DRIFT_THR),
          type: 'line',
          borderColor: 'rgba(255,61,90,0.4)',
          borderWidth: 1,
          borderDash: [5,4],
          pointRadius: 0,
          fill: false,
          order: 3,
        }},
      ],
    }},
    options: {{
      responsive: true,
      interaction: {{ mode: 'index', intersect: false }},
      plugins: {{
        legend: {{ labels: {{ boxWidth: 10, boxHeight: 10, padding: 14, color: '#6b7591' }} }},
        tooltip: TOOLTIP_STYLE,
      }},
      scales: {{
        x: {{ ticks: {{ color: '#4e566b' }}, grid: {{ color: '#14181f' }} }},
        y: {{
          min: 0, suggestedMax: 0.4,
          ticks: {{ color: '#4e566b', stepSize: 0.1 }},
          grid: {{ color: '#14181f' }},
          title: {{ display: true, text: 'Drift Score', color: '#4e566b', font: {{ size: 9 }} }},
        }},
      }},
    }},
  }});
}}

// ══════════════════════════════════════════════
// CATEGORY CHART (Horizontal bars)
// ══════════════════════════════════════════════
const catData = {cat_json};

if (document.getElementById('catChart')) {{
  const catLabels = Object.keys(catData);
  new Chart(document.getElementById('catChart').getContext('2d'), {{
    type: 'bar',
    data: {{
      labels: catLabels,
      datasets: [
        {{
          label: 'Uncertainty',
          data: catLabels.map(c => catData[c].uncertainty),
          backgroundColor: 'rgba(255,61,90,0.65)',
          borderColor: '#ff3d5a',
          borderWidth: 1, borderRadius: 3,
        }},
        {{
          label: 'Consistency',
          data: catLabels.map(c => catData[c].consistency),
          backgroundColor: 'rgba(0,229,255,0.55)',
          borderColor: '#00e5ff',
          borderWidth: 1, borderRadius: 3,
        }},
      ],
    }},
    options: {{
      responsive: true,
      plugins: {{
        legend: {{ labels: {{ boxWidth: 10, boxHeight: 10, padding: 14, color: '#6b7591' }} }},
        tooltip: TOOLTIP_STYLE,
      }},
      scales: {{
        x: {{ ticks: {{ color: '#4e566b' }}, grid: {{ color: '#14181f' }} }},
        y: {{
          min: 0, max: 1,
          ticks: {{ color: '#4e566b', stepSize: 0.25 }},
          grid: {{ color: '#14181f' }},
        }},
      }},
    }},
  }});
}}
</script>
</body>
</html>"""
    return html


# =========================
# LIGHT DASHBOARD (template + reports_data.js)
# =========================

TEMPLATE_FILE = Path(__file__).with_name("dashboard_template.html")
REPORTS_DATA_FILE = "reports_data.js"


def _is_pipeline_report(record):
    return "run_timestamp" in record or "pipeline_version" in record


def _qa_log_to_report(record):
    ts = record.get("timestamp", "")
    return {
        "run_timestamp": ts,
        "question": record.get("question"),
        "category": record.get("category"),
        "model": record.get("model"),
        "num_samples": record.get("num_samples"),
        "num_paraphrases": record.get("num_paraphrases"),
        "uncertainty_score": record.get("uncertainty_score"),
        "consistency_score": record.get("consistency_score"),
        "calibration_score": record.get("calibration_score"),
        "risk_score": record.get("risk_score"),
        "risk_zone": record.get("risk_zone"),
        "severity": record.get("severity"),
        "window_key": ts[:10] if ts else "unknown",
        "drift_score": 0.0,
        "drift_alert": False,
        "change_point": False,
        "delta_uncertainty": 0.0,
        "delta_consistency": 0.0,
        "alerts": [],
        "alert_count": 0,
        "execution_time_seconds": record.get("execution_time_seconds"),
    }


def _records_for_dashboard(log_file):
    path = Path(log_file)
    if not path.exists():
        return []

    if log_file.endswith("final_risk_reports.jsonl") or "risk_reports" in path.name:
        return [{k: v for k, v in r.items() if k in KEEP_FIELDS} for r in load_reports(log_file)]

    records = load_logs(log_file)
    if not records:
        return []
    if _is_pipeline_report(records[0]):
        return [{k: v for k, v in r.items() if k in KEEP_FIELDS} for r in records]
    return [_qa_log_to_report(r) for r in records]


def _resolve_template(output_file):
    template = TEMPLATE_FILE
    if not template.exists():
        candidate = Path(output_file)
        if candidate.exists() and "--bg:#F6F7F9" in candidate.read_text(encoding="utf-8"):
            return candidate
        raise FileNotFoundError(
            f"Dashboard template not found: {TEMPLATE_FILE}. "
            "Restore dashboard_template.html from imp_files/."
        )
    return template


# =========================
# MAIN ENTRY POINT
# =========================

def generate_dashboard(log_file="qa_monitoring_logs.jsonl", output_file="dashboard.html"):
    reports_path = Path("final_risk_reports.jsonl")
    source = log_file
    if log_file == "qa_monitoring_logs.jsonl" and reports_path.exists():
        source = str(reports_path)
        print(f"Using pipeline reports: {source}")

    print("Loading records...")
    records = _records_for_dashboard(source)
    if not records and source != log_file:
        print(f"  No records in {source}, falling back to {log_file}")
        records = _records_for_dashboard(log_file)

    if not records:
        print(f"No records found in {source}" + (f" or {log_file}" if source != log_file else ""))
        return False

    print(f"Loaded {len(records)} records")
    print(f"Writing {REPORTS_DATA_FILE}...")
    write_js(records, REPORTS_DATA_FILE)

    template = _resolve_template(output_file)
    print(f"Copying light dashboard template -> {output_file}...")
    shutil.copy2(template, output_file)

    print(f"Dashboard generated: {output_file}")
    print(f"Data file written:   {REPORTS_DATA_FILE}")
    return True


if __name__ == "__main__":
    generate_dashboard()