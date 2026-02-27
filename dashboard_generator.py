"""
dashboard_generator.py
Generate interactive HTML dashboard from monitoring logs
"""

import json
import numpy as np
from datetime import datetime
from utils import load_logs, format_timestamp
from risk_engine import analyze_risk_distribution
from temporal_preview import generate_temporal_report

# =========================
# DATA AGGREGATION
# =========================

def aggregate_dashboard_data(records):
    """Aggregate all data needed for dashboard"""
    if not records:
        return None

    total = len(records)
    uncertainties = [r["uncertainty_score"] for r in records]
    consistencies = [r["consistency_score"] for r in records]
    calibrations = [r.get("calibration_score", 0) for r in records]
    risk_scores = [r.get("risk_score", 0) for r in records]

    risk_dist = analyze_risk_distribution(records)
    temporal = generate_temporal_report(records)

    categories = {}
    for record in records:
        cat = record.get("category", "unknown")
        if cat not in categories:
            categories[cat] = []
        categories[cat].append(record)

    category_stats = {}
    for cat, cat_records in categories.items():
        category_stats[cat] = {
            "count": len(cat_records),
            "mean_uncertainty": np.mean([r["uncertainty_score"] for r in cat_records]),
            "mean_consistency": np.mean([r["consistency_score"] for r in cat_records]),
            "mean_calibration": np.mean([r.get("calibration_score", 0) for r in cat_records]),
            "mean_risk": np.mean([r.get("risk_score", 0) for r in cat_records]),
            "reliable_count": sum(1 for r in cat_records if r.get("risk_zone") == "RELIABLE"),
            "overconfident_count": sum(1 for r in cat_records if r.get("risk_zone") == "OVERCONFIDENT"),
            "unstable_count": sum(1 for r in cat_records if r.get("risk_zone") == "UNSTABLE"),
            "ambiguous_count": sum(1 for r in cat_records if r.get("risk_zone") == "AMBIGUOUS")
        }

    timestamps = list(range(len(records)))
    uncertainty_series = [r["uncertainty_score"] for r in records]
    consistency_series = [r["consistency_score"] for r in records]
    calibration_series = [r.get("calibration_score", 0) for r in records]

    recent = records[-10:]

    return {
        "total_interactions": total,
        "model": records[0].get("model", "unknown"),
        "time_range": {
            "first": records[0]["timestamp"],
            "last": records[-1]["timestamp"]
        },
        "overall_stats": {
            "mean_uncertainty": np.mean(uncertainties),
            "std_uncertainty": np.std(uncertainties),
            "mean_consistency": np.mean(consistencies),
            "std_consistency": np.std(consistencies),
            "mean_calibration": np.mean(calibrations),
            "mean_risk": np.mean(risk_scores)
        },
        "risk_distribution": risk_dist,
        "temporal": temporal,
        "category_stats": category_stats,
        "time_series": {
            "timestamps": timestamps,
            "uncertainty": uncertainty_series,
            "consistency": consistency_series,
            "calibration": calibration_series
        },
        "recent_interactions": recent
    }


# =========================
# HTML GENERATION
# =========================

def generate_html_dashboard(data):
    """Generate complete HTML dashboard"""

    ts_data = json.dumps(data['time_series'])
    risk_dist_data = json.dumps(data['risk_distribution']['distribution'])

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>LLM Watchdog — Behavioral Reliability Monitor</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Space+Mono:ital,wght@0,400;0,700;1,400&family=DM+Sans:wght@300;400;500&display=swap" rel="stylesheet">
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>
<style>
  *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}

  :root {{
    --bg:        #0a0b0f;
    --surface:   #111318;
    --surface2:  #181b22;
    --border:    #1e2230;
    --border2:   #2a2f3e;
    --text:      #c8d0e0;
    --muted:     #5a6275;
    --accent:    #00e5ff;
    --accent2:   #7b61ff;
    --green:     #00d97e;
    --yellow:    #ffcb47;
    --red:       #ff4560;
    --orange:    #ff7f50;
    --mono:      'Space Mono', monospace;
    --sans:      'DM Sans', sans-serif;
    --gutter:    24px;
    --radius:    6px;
  }}

  html {{ scroll-behavior: smooth; }}

  body {{
    background: var(--bg);
    color: var(--text);
    font-family: var(--sans);
    font-size: 14px;
    line-height: 1.6;
    min-height: 100vh;
    overflow-x: hidden;
  }}

  /* ── noise overlay ── */
  body::before {{
    content: '';
    position: fixed; inset: 0;
    background-image: url("data:image/svg+xml,%3Csvg viewBox='0 0 256 256' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)' opacity='0.04'/%3E%3C/svg%3E");
    pointer-events: none; z-index: 999; opacity: 0.35;
  }}

  /* ── scanline ── */
  body::after {{
    content: '';
    position: fixed; inset: 0;
    background: repeating-linear-gradient(
      to bottom,
      transparent 0px, transparent 3px,
      rgba(0,0,0,0.08) 3px, rgba(0,0,0,0.08) 4px
    );
    pointer-events: none; z-index: 998;
  }}

  /* ── HEADER ── */
  header {{
    border-bottom: 1px solid var(--border);
    padding: 0 var(--gutter);
    display: flex;
    align-items: center;
    justify-content: space-between;
    height: 56px;
    position: sticky; top: 0; z-index: 100;
    background: rgba(10,11,15,0.92);
    backdrop-filter: blur(12px);
  }}

  .logo {{
    display: flex; align-items: center; gap: 10px;
  }}
  .logo-icon {{
    width: 28px; height: 28px;
    background: var(--accent);
    clip-path: polygon(50% 0%, 100% 25%, 100% 75%, 50% 100%, 0% 75%, 0% 25%);
    animation: pulse-hex 3s ease-in-out infinite;
  }}
  @keyframes pulse-hex {{
    0%,100% {{ box-shadow: 0 0 0 0 rgba(0,229,255,0.4); }}
    50%      {{ box-shadow: 0 0 0 8px rgba(0,229,255,0); }}
  }}
  .logo-text {{
    font-family: var(--mono);
    font-size: 13px;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #fff;
  }}
  .logo-sub {{
    font-family: var(--mono);
    font-size: 10px;
    color: var(--muted);
    letter-spacing: 0.1em;
    text-transform: uppercase;
  }}

  .header-meta {{
    display: flex; align-items: center; gap: 20px;
  }}
  .live-badge {{
    display: flex; align-items: center; gap: 6px;
    font-family: var(--mono);
    font-size: 10px;
    letter-spacing: 0.1em;
    color: var(--green);
    text-transform: uppercase;
  }}
  .live-dot {{
    width: 6px; height: 6px; border-radius: 50%;
    background: var(--green);
    animation: blink 1.4s step-end infinite;
  }}
  @keyframes blink {{
    0%,100% {{ opacity: 1; }}
    50%      {{ opacity: 0.2; }}
  }}
  .ts {{
    font-family: var(--mono);
    font-size: 11px;
    color: var(--muted);
  }}

  /* ── MAIN LAYOUT ── */
  main {{
    padding: var(--gutter);
    max-width: 1440px;
    margin: 0 auto;
    display: flex;
    flex-direction: column;
    gap: var(--gutter);
  }}

  /* ── section labels ── */
  .section-label {{
    font-family: var(--mono);
    font-size: 10px;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 12px;
    display: flex;
    align-items: center;
    gap: 8px;
  }}
  .section-label::after {{
    content: '';
    flex: 1;
    height: 1px;
    background: var(--border);
  }}

  /* ── KPI ROW ── */
  .kpi-row {{
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 12px;
  }}

  .kpi-card {{
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 20px;
    position: relative;
    overflow: hidden;
    transition: border-color 0.2s, transform 0.2s;
  }}
  .kpi-card:hover {{
    border-color: var(--border2);
    transform: translateY(-2px);
  }}
  .kpi-card::before {{
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
  }}
  .kpi-card.accent-cyan::before  {{ background: var(--accent); }}
  .kpi-card.accent-purple::before {{ background: var(--accent2); }}
  .kpi-card.accent-green::before  {{ background: var(--green); }}
  .kpi-card.accent-red::before    {{ background: var(--red); }}

  .kpi-label {{
    font-family: var(--mono);
    font-size: 10px;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 10px;
  }}
  .kpi-value {{
    font-family: var(--mono);
    font-size: 36px;
    font-weight: 700;
    line-height: 1;
    color: #fff;
    letter-spacing: -0.02em;
  }}
  .kpi-value.danger {{ color: var(--red); }}
  .kpi-value.warn   {{ color: var(--yellow); }}
  .kpi-value.good   {{ color: var(--green); }}

  .kpi-sub {{
    font-size: 11px;
    color: var(--muted);
    margin-top: 6px;
  }}

  /* ── ALERTS ── */
  .alerts-box {{
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    overflow: hidden;
  }}
  .alerts-header {{
    padding: 12px 16px;
    border-bottom: 1px solid var(--border);
    font-family: var(--mono);
    font-size: 10px;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: var(--muted);
    display: flex;
    align-items: center;
    gap: 8px;
  }}
  .alert-item {{
    padding: 12px 16px;
    border-bottom: 1px solid var(--border);
    display: flex;
    align-items: flex-start;
    gap: 10px;
    font-size: 13px;
    animation: slide-in 0.4s ease;
  }}
  .alert-item:last-child {{ border-bottom: none; }}
  @keyframes slide-in {{
    from {{ opacity: 0; transform: translateX(-8px); }}
    to   {{ opacity: 1; transform: translateX(0); }}
  }}
  .alert-item.critical {{ border-left: 3px solid var(--red); background: rgba(255,69,96,0.04); }}
  .alert-item.warning  {{ border-left: 3px solid var(--yellow); background: rgba(255,203,71,0.04); }}
  .alert-item.info     {{ border-left: 3px solid var(--accent); background: rgba(0,229,255,0.04); }}
  .alert-item.ok       {{ border-left: 3px solid var(--green); background: rgba(0,217,126,0.04); }}
  .alert-icon          {{ font-size: 15px; flex-shrink: 0; margin-top: 1px; }}

  /* ── CHARTS ROW ── */
  .charts-row {{
    display: grid;
    grid-template-columns: 2fr 1fr;
    gap: 12px;
  }}

  .panel {{
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    overflow: hidden;
  }}
  .panel-header {{
    padding: 14px 18px;
    border-bottom: 1px solid var(--border);
    display: flex;
    align-items: center;
    justify-content: space-between;
  }}
  .panel-title {{
    font-family: var(--mono);
    font-size: 11px;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--text);
  }}
  .panel-body {{
    padding: 18px;
  }}

  /* ── CATEGORY TABLE ── */
  .cat-table {{
    width: 100%;
    border-collapse: collapse;
    font-size: 12px;
  }}
  .cat-table th {{
    font-family: var(--mono);
    font-size: 9px;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: var(--muted);
    padding: 8px 12px;
    text-align: left;
    border-bottom: 1px solid var(--border);
    white-space: nowrap;
  }}
  .cat-table td {{
    padding: 10px 12px;
    border-bottom: 1px solid var(--border);
    color: var(--text);
    font-family: var(--mono);
    font-size: 11px;
    vertical-align: middle;
  }}
  .cat-table tr:last-child td {{ border-bottom: none; }}
  .cat-table tr:hover td {{ background: rgba(255,255,255,0.02); }}

  .cat-name {{
    color: #fff;
    font-weight: 700;
    font-size: 12px;
    letter-spacing: 0.05em;
    text-transform: capitalize;
  }}

  /* inline zone bars */
  .zone-bar {{
    display: flex;
    height: 6px;
    border-radius: 3px;
    overflow: hidden;
    gap: 2px;
    min-width: 80px;
  }}
  .zone-bar-seg {{ border-radius: 2px; }}

  /* risk badge */
  .risk-badge {{
    display: inline-block;
    padding: 2px 8px;
    border-radius: 3px;
    font-family: var(--mono);
    font-size: 9px;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    font-weight: 700;
  }}
  .risk-reliable     {{ background: rgba(0,217,126,0.15); color: var(--green); border: 1px solid rgba(0,217,126,0.3); }}
  .risk-overconfident{{ background: rgba(255,69,96,0.15);  color: var(--red);   border: 1px solid rgba(255,69,96,0.3); }}
  .risk-unstable     {{ background: rgba(255,203,71,0.15); color: var(--yellow);border: 1px solid rgba(255,203,71,0.3); }}
  .risk-ambiguous    {{ background: rgba(123,97,255,0.15); color: var(--accent2);border: 1px solid rgba(123,97,255,0.3); }}
  .risk-unknown      {{ background: rgba(90,98,117,0.15);  color: var(--muted); border: 1px solid rgba(90,98,117,0.3); }}

  /* score pill */
  .score-pill {{
    font-family: var(--mono);
    font-size: 11px;
  }}
  .score-pill.hi  {{ color: var(--red); }}
  .score-pill.mid {{ color: var(--yellow); }}
  .score-pill.lo  {{ color: var(--green); }}

  /* ── TEMPORAL PANEL ── */
  .temporal-grid {{
    display: grid;
    grid-template-columns: 1fr 1fr 1fr;
    gap: 12px;
  }}
  .trend-card {{
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 16px;
  }}
  .trend-label {{
    font-family: var(--mono);
    font-size: 9px;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 8px;
  }}
  .trend-value {{
    font-family: var(--mono);
    font-size: 13px;
    font-weight: 700;
    letter-spacing: 0.05em;
    margin-bottom: 4px;
  }}
  .trend-value.good    {{ color: var(--green); }}
  .trend-value.bad     {{ color: var(--red); }}
  .trend-value.neutral {{ color: var(--yellow); }}
  .trend-desc {{ font-size: 11px; color: var(--muted); line-height: 1.4; }}

  .stat-group {{ display: flex; flex-direction: column; gap: 8px; }}
  .stat-row {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 8px 10px;
    background: var(--surface2);
    border-radius: 4px;
    border: 1px solid var(--border);
  }}
  .stat-key {{
    font-family: var(--mono);
    font-size: 10px;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: var(--muted);
  }}
  .stat-val {{
    font-family: var(--mono);
    font-size: 13px;
    font-weight: 700;
    color: var(--accent);
  }}

  /* ── RECENT TABLE ── */
  .recent-table {{
    width: 100%;
    border-collapse: collapse;
    font-size: 12px;
  }}
  .recent-table th {{
    font-family: var(--mono);
    font-size: 9px;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: var(--muted);
    padding: 8px 12px;
    text-align: left;
    border-bottom: 1px solid var(--border);
    white-space: nowrap;
    background: var(--surface2);
    position: sticky; top: 56px;
  }}
  .recent-table td {{
    padding: 10px 12px;
    border-bottom: 1px solid var(--border);
    vertical-align: middle;
  }}
  .recent-table tr:last-child td {{ border-bottom: none; }}
  .recent-table tr:hover td {{ background: rgba(255,255,255,0.02); }}
  .q-text {{
    color: var(--text);
    max-width: 380px;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }}
  .ts-cell {{
    font-family: var(--mono);
    font-size: 10px;
    color: var(--muted);
    white-space: nowrap;
  }}

  /* ── FOOTER ── */
  footer {{
    border-top: 1px solid var(--border);
    padding: 16px var(--gutter);
    display: flex;
    align-items: center;
    justify-content: space-between;
    font-family: var(--mono);
    font-size: 10px;
    color: var(--muted);
    letter-spacing: 0.1em;
  }}
  footer span {{ color: var(--accent); }}

  /* ── RESPONSIVE ── */
  @media (max-width: 900px) {{
    .kpi-row       {{ grid-template-columns: 1fr 1fr; }}
    .charts-row    {{ grid-template-columns: 1fr; }}
    .temporal-grid {{ grid-template-columns: 1fr; }}
  }}
  @media (max-width: 500px) {{
    .kpi-row {{ grid-template-columns: 1fr; }}
  }}
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
  <div class="header-meta">
    <div class="live-badge">
      <div class="live-dot"></div>
      Live
    </div>
    <div class="ts" id="live-clock">{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</div>
  </div>
</header>

<main>

  <!-- KPI ROW -->
  <div>
    <div class="section-label">System Overview</div>
    <div class="kpi-row">
      <div class="kpi-card accent-cyan">
        <div class="kpi-label">Total Interactions</div>
        <div class="kpi-value">{data['total_interactions']}</div>
        <div class="kpi-sub">Model: {data['model']}</div>
      </div>
      <div class="kpi-card accent-purple">
        <div class="kpi-label">Health Score</div>
        <div class="kpi-value {'good' if data['risk_distribution']['health_score'] >= 0.7 else 'warn' if data['risk_distribution']['health_score'] >= 0.4 else 'danger'}">{data['risk_distribution']['health_score']:.2f}</div>
        <div class="kpi-sub">Out of 1.0</div>
      </div>
      <div class="kpi-card accent-green">
        <div class="kpi-label">Mean Calibration</div>
        <div class="kpi-value {'good' if data['overall_stats']['mean_calibration'] >= 0.6 else 'warn' if data['overall_stats']['mean_calibration'] >= 0.4 else 'danger'}">{data['overall_stats']['mean_calibration']:.3f}</div>
        <div class="kpi-sub">Calibration Score</div>
      </div>
      <div class="kpi-card accent-red">
        <div class="kpi-label">Critical Issues</div>
        <div class="kpi-value {'danger' if data['risk_distribution']['critical_count'] > 0 else 'good'}">{data['risk_distribution']['critical_count']}</div>
        <div class="kpi-sub">High Risk Interactions</div>
      </div>
    </div>
  </div>

  <!-- ALERTS -->
  <div>
    <div class="section-label">Alerts &amp; Warnings</div>
    <div class="alerts-box">
      <div class="alerts-header">
        ◈ Active Alerts
      </div>
      {generate_alerts_html(data)}
    </div>
  </div>

  <!-- CHARTS ROW -->
  <div>
    <div class="section-label">Metrics &amp; Distribution</div>
    <div class="charts-row">
      <div class="panel">
        <div class="panel-header">
          <span class="panel-title">Metrics Over Time</span>
          <span style="font-family:var(--mono);font-size:10px;color:var(--muted);">
            {len(data['time_series']['timestamps'])} samples
          </span>
        </div>
        <div class="panel-body">
          <canvas id="tsChart" height="200"></canvas>
        </div>
      </div>
      <div class="panel">
        <div class="panel-header">
          <span class="panel-title">Risk Zone Distribution</span>
        </div>
        <div class="panel-body">
          <canvas id="riskChart" height="200"></canvas>
        </div>
      </div>
    </div>
  </div>

  <!-- CATEGORY TABLE -->
  <div>
    <div class="section-label">Category Performance</div>
    <div class="panel">
      <div class="panel-body" style="padding:0; overflow-x:auto;">
        {generate_category_table_html(data['category_stats'])}
      </div>
    </div>
  </div>

  <!-- TEMPORAL ANALYSIS -->
  <div>
    <div class="section-label">Temporal Analysis</div>
    {generate_temporal_html(data['temporal'])}
  </div>

  <!-- RECENT INTERACTIONS -->
  <div>
    <div class="section-label">Recent Interactions</div>
    <div class="panel">
      <div style="overflow-x:auto;">
        {generate_recent_table_html(data['recent_interactions'])}
      </div>
    </div>
  </div>

</main>

<footer>
  <span>LLM WATCHDOG</span>
  <div>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}</div>
  <div>Range: {data['time_range']['first']} → {data['time_range']['last']}</div>
</footer>

<script>
// Live clock
function tick() {{
  const el = document.getElementById('live-clock');
  if (el) el.textContent = new Date().toISOString().replace('T',' ').slice(0,19);
}}
setInterval(tick, 1000);

// Chart.js defaults
Chart.defaults.color = '#5a6275';
Chart.defaults.borderColor = '#1e2230';
Chart.defaults.font.family = "'Space Mono', monospace";
Chart.defaults.font.size = 10;

const tsData = {ts_data};
const riskData = {risk_dist_data};

// ── Time series chart ──
const tsCtx = document.getElementById('tsChart').getContext('2d');
new Chart(tsCtx, {{
  type: 'line',
  data: {{
    labels: tsData.timestamps,
    datasets: [
      {{
        label: 'Uncertainty',
        data: tsData.uncertainty,
        borderColor: '#ff4560',
        backgroundColor: 'rgba(255,69,96,0.06)',
        borderWidth: 1.5,
        pointRadius: 0,
        tension: 0.4,
        fill: true
      }},
      {{
        label: 'Consistency',
        data: tsData.consistency,
        borderColor: '#00e5ff',
        backgroundColor: 'rgba(0,229,255,0.04)',
        borderWidth: 1.5,
        pointRadius: 0,
        tension: 0.4,
        fill: true
      }},
      {{
        label: 'Calibration',
        data: tsData.calibration,
        borderColor: '#00d97e',
        backgroundColor: 'rgba(0,217,126,0.04)',
        borderWidth: 1.5,
        pointRadius: 0,
        tension: 0.4,
        fill: true
      }}
    ]
  }},
  options: {{
    responsive: true,
    interaction: {{ mode: 'index', intersect: false }},
    plugins: {{
      legend: {{
        labels: {{
          boxWidth: 10, boxHeight: 2,
          usePointStyle: false,
          color: '#5a6275',
          padding: 16
        }}
      }},
      tooltip: {{
        backgroundColor: '#111318',
        borderColor: '#1e2230',
        borderWidth: 1,
        titleColor: '#c8d0e0',
        bodyColor: '#5a6275',
        padding: 10
      }}
    }},
    scales: {{
      x: {{
        ticks: {{ maxTicksLimit: 8, color: '#5a6275' }},
        grid: {{ color: '#1a1d26' }}
      }},
      y: {{
        min: 0, max: 1,
        ticks: {{ color: '#5a6275', stepSize: 0.25 }},
        grid: {{ color: '#1a1d26' }}
      }}
    }}
  }}
}});

// ── Risk donut chart ──
const rCtx = document.getElementById('riskChart').getContext('2d');
const rLabels = Object.keys(riskData);
const rValues = Object.values(riskData);
const rColors = {{
  RELIABLE:     '#00d97e',
  OVERCONFIDENT:'#ff4560',
  UNSTABLE:     '#ffcb47',
  AMBIGUOUS:    '#7b61ff'
}};
new Chart(rCtx, {{
  type: 'doughnut',
  data: {{
    labels: rLabels,
    datasets: [{{
      data: rValues,
      backgroundColor: rLabels.map(l => (rColors[l] || '#5a6275') + '99'),
      borderColor: rLabels.map(l => rColors[l] || '#5a6275'),
      borderWidth: 1.5,
      hoverOffset: 6
    }}]
  }},
  options: {{
    responsive: true,
    cutout: '68%',
    plugins: {{
      legend: {{
        position: 'bottom',
        labels: {{
          boxWidth: 10, boxHeight: 10,
          padding: 12,
          color: '#5a6275'
        }}
      }},
      tooltip: {{
        backgroundColor: '#111318',
        borderColor: '#1e2230',
        borderWidth: 1,
        titleColor: '#c8d0e0',
        bodyColor: '#5a6275',
        padding: 10
      }}
    }}
  }}
}});
</script>
</body>
</html>"""
    return html


def generate_alerts_html(data):
    """Generate alerts section HTML"""
    alerts = []

    oc = data['risk_distribution']['distribution'].get('OVERCONFIDENT', 0)
    if oc > 0:
        alerts.append(
            f'<div class="alert-item critical">'
            f'<span class="alert-icon">⛔</span>'
            f'<div><strong>{oc} OVERCONFIDENT</strong> interactions detected — '
            f'model may be hallucinating with false confidence.</div></div>'
        )

    us = data['risk_distribution']['distribution'].get('UNSTABLE', 0)
    if us > 0:
        alerts.append(
            f'<div class="alert-item warning">'
            f'<span class="alert-icon">⚠</span>'
            f'<div><strong>{us} UNSTABLE</strong> interactions detected — '
            f'high variability in responses.</div></div>'
        )

    if data['overall_stats']['mean_calibration'] < 0.5:
        alerts.append(
            f'<div class="alert-item warning">'
            f'<span class="alert-icon">⚠</span>'
            f'<div>Low calibration detected '
            f'(mean: {data["overall_stats"]["mean_calibration"]:.3f}) — '
            f'model may be poorly calibrated.</div></div>'
        )

    if data['temporal']['trends']['uncertainty']['trend'] == 'DEGRADING':
        alerts.append(
            '<div class="alert-item info">'
            '<span class="alert-icon">📉</span>'
            '<div>Degrading trend in uncertainty — '
            'model behavior may be destabilizing.</div></div>'
        )

    if not alerts:
        alerts.append(
            '<div class="alert-item ok">'
            '<span class="alert-icon">✓</span>'
            '<div>No critical alerts. System is operating normally.</div></div>'
        )

    return '\n'.join(alerts)


def generate_category_table_html(category_stats):
    """Generate category performance table HTML"""
    rows = []
    for cat, stats in sorted(category_stats.items()):
        total_zoned = (stats['reliable_count'] + stats['overconfident_count'] +
                       stats['unstable_count'] + stats['ambiguous_count']) or 1
        def pct(n): return max(int(n / total_zoned * 100), 0 if n == 0 else 4)

        u_class = 'hi' if stats['mean_uncertainty'] > 0.6 else 'mid' if stats['mean_uncertainty'] > 0.35 else 'lo'
        c_class = 'lo' if stats['mean_consistency'] > 0.6 else 'mid' if stats['mean_consistency'] > 0.35 else 'hi'
        r_class = 'hi' if stats['mean_risk'] > 0.6 else 'mid' if stats['mean_risk'] > 0.3 else 'lo'

        zone_bar = (
            f'<div class="zone-bar">'
            f'<div class="zone-bar-seg" style="width:{pct(stats["reliable_count"])}%;background:#00d97e88"></div>'
            f'<div class="zone-bar-seg" style="width:{pct(stats["overconfident_count"])}%;background:#ff456088"></div>'
            f'<div class="zone-bar-seg" style="width:{pct(stats["unstable_count"])}%;background:#ffcb4788"></div>'
            f'<div class="zone-bar-seg" style="width:{pct(stats["ambiguous_count"])}%;background:#7b61ff88"></div>'
            f'</div>'
        )

        rows.append(f"""<tr>
          <td><span class="cat-name">{cat.replace('_',' ')}</span></td>
          <td style="color:#fff;font-weight:700">{stats['count']}</td>
          <td><span class="score-pill {u_class}">{stats['mean_uncertainty']:.3f}</span></td>
          <td><span class="score-pill {c_class}">{stats['mean_consistency']:.3f}</span></td>
          <td><span class="score-pill">{stats['mean_calibration']:.3f}</span></td>
          <td><span class="score-pill {r_class}">{stats['mean_risk']:.3f}</span></td>
          <td>{zone_bar}</td>
          <td style="color:var(--green)">{stats['reliable_count']}</td>
          <td style="color:var(--red)">{stats['overconfident_count']}</td>
          <td style="color:var(--yellow)">{stats['unstable_count']}</td>
          <td style="color:var(--accent2)">{stats['ambiguous_count']}</td>
        </tr>""")

    return f"""<table class="cat-table">
      <thead><tr>
        <th>Category</th>
        <th>Count</th>
        <th>Uncertainty</th>
        <th>Consistency</th>
        <th>Calibration</th>
        <th>Risk</th>
        <th>Zone Split</th>
        <th>Reliable</th>
        <th>Overconf.</th>
        <th>Unstable</th>
        <th>Ambiguous</th>
      </tr></thead>
      <tbody>{''.join(rows)}</tbody>
    </table>"""


def generate_temporal_html(temporal):
    """Generate temporal analysis HTML"""
    u_trend = temporal['trends']['uncertainty']
    c_trend = temporal['trends']['consistency']

    def trend_class(trend_str, invert=False):
        if 'DEGRADING' in trend_str:
            return 'bad' if not invert else 'good'
        if 'IMPROVING' in trend_str:
            return 'good' if not invert else 'bad'
        return 'neutral'

    u_cls = trend_class(u_trend['trend'])
    c_cls = trend_class(c_trend['trend'], invert=True)
    rw = temporal['rolling_window_5']

    return f"""<div class="temporal-grid">
      <div class="trend-card">
        <div class="trend-label">Uncertainty Trend</div>
        <div class="trend-value {u_cls}">{u_trend['trend']}</div>
        <div class="trend-desc">{u_trend['description']}</div>
      </div>
      <div class="trend-card">
        <div class="trend-label">Consistency Trend</div>
        <div class="trend-value {c_cls}">{c_trend['trend']}</div>
        <div class="trend-desc">{c_trend['description']}</div>
      </div>
      <div class="trend-card">
        <div class="trend-label">Rolling Window (last 5)</div>
        <div class="stat-group">
          <div class="stat-row">
            <span class="stat-key">Uncertainty</span>
            <span class="stat-val">{rw['mean_uncertainty']:.3f}</span>
          </div>
          <div class="stat-row">
            <span class="stat-key">Consistency</span>
            <span class="stat-val">{rw['mean_consistency']:.3f}</span>
          </div>
          <div class="stat-row">
            <span class="stat-key">Calibration</span>
            <span class="stat-val">{rw['mean_calibration']:.3f}</span>
          </div>
        </div>
      </div>
    </div>"""


def generate_recent_table_html(recent):
    """Generate recent interactions table HTML"""
    rows = []
    for record in reversed(recent):
        risk_zone = record.get('risk_zone', 'UNKNOWN')
        badge_class = f"risk-{risk_zone.lower()}"
        u = record['uncertainty_score']
        c = record['consistency_score']
        cal = record.get('calibration_score', 0)

        u_cls = 'hi' if u > 0.6 else 'mid' if u > 0.35 else 'lo'
        c_cls = 'lo' if c > 0.6 else 'mid' if c > 0.35 else 'hi'

        q = record['question']
        q_display = (q[:72] + '…') if len(q) > 72 else q

        rows.append(f"""<tr>
          <td class="ts-cell">{format_timestamp(record['timestamp'])}</td>
          <td><span class="q-text" title="{q}">{q_display}</span></td>
          <td><span class="risk-badge {badge_class}">{risk_zone}</span></td>
          <td><span class="score-pill {u_cls}" style="font-family:var(--mono);font-size:11px">{u:.3f}</span></td>
          <td><span class="score-pill {c_cls}" style="font-family:var(--mono);font-size:11px">{c:.3f}</span></td>
          <td style="font-family:var(--mono);font-size:11px;color:var(--accent)">{cal:.3f}</td>
        </tr>""")

    return f"""<table class="recent-table">
      <thead><tr>
        <th>Timestamp</th>
        <th>Question</th>
        <th>Risk Zone</th>
        <th>Uncertainty</th>
        <th>Consistency</th>
        <th>Calibration</th>
      </tr></thead>
      <tbody>{''.join(rows)}</tbody>
    </table>"""


# =========================
# MAIN GENERATION FUNCTION
# =========================

def generate_dashboard(log_file="qa_monitoring_logs.jsonl", output_file="dashboard.html"):
    """Generate dashboard from logs"""
    print("Loading logs...")
    records = load_logs(log_file)

    if not records:
        print(f"❌ No records found in {log_file}")
        return False

    print(f"Found {len(records)} records")
    print("Aggregating data...")
    data = aggregate_dashboard_data(records)

    print("Generating HTML...")
    html = generate_html_dashboard(data)

    print(f"Writing to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html)

    print(f"✅ Dashboard generated: {output_file}")
    print(f"Open in browser to view")
    return True


if __name__ == "__main__":
    generate_dashboard()