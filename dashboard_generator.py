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
    
    # Basic stats
    total = len(records)
    uncertainties = [r["uncertainty_score"] for r in records]
    consistencies = [r["consistency_score"] for r in records]
    calibrations = [r.get("calibration_score", 0) for r in records]
    risk_scores = [r.get("risk_score", 0) for r in records]
    
    # Risk distribution
    risk_dist = analyze_risk_distribution(records)
    
    # Temporal report
    temporal = generate_temporal_report(records)
    
    # Per-category stats
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
    
    # Time series data
    timestamps = [i for i in range(len(records))]
    uncertainty_series = [r["uncertainty_score"] for r in records]
    consistency_series = [r["consistency_score"] for r in records]
    calibration_series = [r.get("calibration_score", 0) for r in records]
    
    # Recent interactions
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
    
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>LLM Watchdog Dashboard</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }}
        
        .container {{
            max-width: 1400px;
            margin: 0 auto;
        }}
        
        .header {{
            background: white;
            border-radius: 12px;
            padding: 30px;
            margin-bottom: 20px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }}
        
        .header h1 {{
            font-size: 36px;
            color: #1a202c;
            margin-bottom: 10px;
        }}
        
        .header p {{
            color: #718096;
            font-size: 16px;
        }}
        
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 20px;
        }}
        
        .stat-card {{
            background: white;
            border-radius: 12px;
            padding: 25px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            text-align: center;
        }}
        
        .stat-card h3 {{
            font-size: 14px;
            color: #718096;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-bottom: 10px;
        }}
        
        .stat-value {{
            font-size: 36px;
            font-weight: bold;
            color: #1a202c;
            margin-bottom: 5px;
        }}
        
        .stat-label {{
            font-size: 12px;
            color: #a0aec0;
        }}
        
        .card {{
            background: white;
            border-radius: 12px;
            padding: 25px;
            margin-bottom: 20px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }}
        
        .card h2 {{
            font-size: 20px;
            color: #1a202c;
            margin-bottom: 20px;
            padding-bottom: 10px;
            border-bottom: 2px solid #e2e8f0;
        }}
        
        .chart-container {{
            position: relative;
            height: 300px;
            margin-top: 20px;
        }}
        
        table {{
            width: 100%;
            border-collapse: collapse;
        }}
        
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #e2e8f0;
        }}
        
        th {{
            background: #f7fafc;
            color: #4a5568;
            font-weight: 600;
            font-size: 13px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}
        
        td {{
            color: #2d3748;
        }}
        
        .risk-badge {{
            display: inline-block;
            padding: 4px 12px;
            border-radius: 12px;
            font-size: 12px;
            font-weight: 600;
        }}
        
        .risk-reliable {{ background: #c6f6d5; color: #22543d; }}
        .risk-overconfident {{ background: #fed7d7; color: #742a2a; }}
        .risk-unstable {{ background: #feebc8; color: #7c2d12; }}
        .risk-ambiguous {{ background: #e9d8fd; color: #44337a; }}
        
        .alert-box {{
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 15px;
            border-left: 4px solid;
        }}
        
        .alert-warning {{
            background: #fffaf0;
            border-color: #ed8936;
            color: #7c2d12;
        }}
        
        .alert-danger {{
            background: #fff5f5;
            border-color: #f56565;
            color: #742a2a;
        }}
        
        .alert-info {{
            background: #ebf8ff;
            border-color: #4299e1;
            color: #2c5282;
        }}
        
        .metric {{
            display: inline-block;
            margin-right: 20px;
        }}
        
        .metric-label {{
            font-size: 12px;
            color: #718096;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}
        
        .metric-value {{
            font-size: 24px;
            font-weight: bold;
            color: #1a202c;
        }}
        
        .timestamp {{
            font-size: 12px;
            color: #a0aec0;
        }}
        
        .trend-up {{
            color: #f56565;
        }}
        
        .trend-down {{
            color: #48bb78;
        }}
        
        .trend-stable {{
            color: #4299e1;
        }}
    </style>
</head>
<body>
    <div class="container">
        <!-- Header -->
        <div class="header">
            <h1>🐕 LLM Watchdog Dashboard</h1>
            <p>Behavioral Reliability Monitoring System</p>
            <p class="timestamp">Last Updated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
        </div>
        
        <!-- Stats Grid -->
        <div class="stats-grid">
            <div class="stat-card">
                <h3>Total Interactions</h3>
                <div class="stat-value">{data['total_interactions']}</div>
                <div class="stat-label">Questions Monitored</div>
            </div>
            
            <div class="stat-card">
                <h3>System Health</h3>
                <div class="stat-value">{data['risk_distribution']['health_score']:.2f}</div>
                <div class="stat-label">Out of 1.0</div>
            </div>
            
            <div class="stat-card">
                <h3>Mean Calibration</h3>
                <div class="stat-value">{data['overall_stats']['mean_calibration']:.3f}</div>
                <div class="stat-label">Calibration Score</div>
            </div>
            
            <div class="stat-card">
                <h3>Critical Issues</h3>
                <div class="stat-value">{data['risk_distribution']['critical_count']}</div>
                <div class="stat-label">High Risk Interactions</div>
            </div>
        </div>
        
        <!-- Alerts -->
        <div class="card">
            <h2>🚨 Alerts & Warnings</h2>
            {generate_alerts_html(data)}
        </div>
        
        <!-- Risk Distribution -->
        <div class="card">
            <h2>📊 Risk Zone Distribution</h2>
            <div class="chart-container">
                <canvas id="riskPieChart"></canvas>
            </div>
        </div>
        
        <!-- Time Series -->
        <div class="card">
            <h2>📈 Metrics Over Time</h2>
            <div class="chart-container">
                <canvas id="timeSeriesChart"></canvas>
            </div>
        </div>
        
        <!-- Category Performance -->
        <div class="card">
            <h2>🏷️ Category Performance</h2>
            {generate_category_table_html(data['category_stats'])}
        </div>
        
        <!-- Temporal Analysis -->
        <div class="card">
            <h2>⏱️ Temporal Analysis</h2>
            {generate_temporal_html(data['temporal'])}
        </div>
        
        <!-- Recent Interactions -->
        <div class="card">
            <h2>📝 Recent Interactions</h2>
            {generate_recent_table_html(data['recent_interactions'])}
        </div>
    </div>
    
    <script>
        // Risk Pie Chart
        const riskCtx = document.getElementById('riskPieChart').getContext('2d');
        new Chart(riskCtx, {{
            type: 'doughnut',
            data: {{
                labels: ['RELIABLE', 'OVERCONFIDENT', 'UNSTABLE', 'AMBIGUOUS'],
                datasets: [{{
                    data: [
                        {data['risk_distribution']['distribution']['RELIABLE']},
                        {data['risk_distribution']['distribution']['OVERCONFIDENT']},
                        {data['risk_distribution']['distribution']['UNSTABLE']},
                        {data['risk_distribution']['distribution']['AMBIGUOUS']}
                    ],
                    backgroundColor: ['#10b981', '#ef4444', '#f59e0b', '#6366f1']
                }}]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                plugins: {{
                    legend: {{
                        position: 'bottom'
                    }}
                }}
            }}
        }});
        
        // Time Series Chart
        const timeCtx = document.getElementById('timeSeriesChart').getContext('2d');
        new Chart(timeCtx, {{
            type: 'line',
            data: {{
                labels: {json.dumps(data['time_series']['timestamps'])},
                datasets: [
                    {{
                        label: 'Uncertainty',
                        data: {json.dumps(data['time_series']['uncertainty'])},
                        borderColor: '#ef4444',
                        backgroundColor: 'rgba(239, 68, 68, 0.1)',
                        tension: 0.4
                    }},
                    {{
                        label: 'Consistency',
                        data: {json.dumps(data['time_series']['consistency'])},
                        borderColor: '#10b981',
                        backgroundColor: 'rgba(16, 185, 129, 0.1)',
                        tension: 0.4
                    }},
                    {{
                        label: 'Calibration',
                        data: {json.dumps(data['time_series']['calibration'])},
                        borderColor: '#6366f1',
                        backgroundColor: 'rgba(99, 102, 241, 0.1)',
                        tension: 0.4
                    }}
                ]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                scales: {{
                    y: {{
                        beginAtZero: true,
                        max: 1
                    }}
                }},
                plugins: {{
                    legend: {{
                        position: 'bottom'
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
    
    # Check for critical issues
    if data['risk_distribution']['distribution']['OVERCONFIDENT'] > 0:
        count = data['risk_distribution']['distribution']['OVERCONFIDENT']
        alerts.append(f'<div class="alert-box alert-danger">⛔ <strong>{count} OVERCONFIDENT</strong> interactions detected. Model may be hallucinating with false confidence.</div>')
    
    if data['risk_distribution']['distribution']['UNSTABLE'] > 0:
        count = data['risk_distribution']['distribution']['UNSTABLE']
        alerts.append(f'<div class="alert-box alert-warning">⚠️ <strong>{count} UNSTABLE</strong> interactions detected. High variability in responses.</div>')
    
    if data['overall_stats']['mean_calibration'] < 0.5:
        alerts.append(f'<div class="alert-box alert-warning">⚠️ <strong>Low calibration</strong> detected (mean: {data["overall_stats"]["mean_calibration"]:.3f}). Model may be poorly calibrated.</div>')
    
    # Check temporal trends
    if data['temporal']['trends']['uncertainty']['trend'] == 'DEGRADING':
        alerts.append('<div class="alert-box alert-warning">📉 <strong>Degrading trend</strong> in uncertainty. Model behavior may be destabilizing.</div>')
    
    if not alerts:
        alerts.append('<div class="alert-box alert-info">✅ No critical alerts. System is operating normally.</div>')
    
    return '\n'.join(alerts)

def generate_category_table_html(category_stats):
    """Generate category performance table HTML"""
    rows = []
    
    for cat, stats in sorted(category_stats.items()):
        risk_color = "🟢" if stats['mean_risk'] < 0.3 else "🟡" if stats['mean_risk'] < 0.6 else "🔴"
        
        rows.append(f"""
            <tr>
                <td><strong>{cat.replace('_', ' ').title()}</strong></td>
                <td>{stats['count']}</td>
                <td>{stats['mean_uncertainty']:.3f}</td>
                <td>{stats['mean_consistency']:.3f}</td>
                <td>{stats['mean_calibration']:.3f}</td>
                <td>{stats['mean_risk']:.3f} {risk_color}</td>
                <td>
                    <span class="risk-badge risk-reliable">{stats['reliable_count']}</span>
                    <span class="risk-badge risk-overconfident">{stats['overconfident_count']}</span>
                    <span class="risk-badge risk-unstable">{stats['unstable_count']}</span>
                    <span class="risk-badge risk-ambiguous">{stats['ambiguous_count']}</span>
                </td>
            </tr>
        """)
    
    return f"""
        <table>
            <thead>
                <tr>
                    <th>Category</th>
                    <th>Count</th>
                    <th>Uncertainty</th>
                    <th>Consistency</th>
                    <th>Calibration</th>
                    <th>Risk</th>
                    <th>Distribution</th>
                </tr>
            </thead>
            <tbody>
                {''.join(rows)}
            </tbody>
        </table>
    """

def generate_temporal_html(temporal):
    """Generate temporal analysis HTML"""
    u_trend = temporal['trends']['uncertainty']
    c_trend = temporal['trends']['consistency']
    
    u_class = "trend-up" if "DEGRADING" in u_trend['trend'] else "trend-down" if "IMPROVING" in u_trend['trend'] else "trend-stable"
    c_class = "trend-down" if "DEGRADING" in c_trend['trend'] else "trend-up" if "IMPROVING" in c_trend['trend'] else "trend-stable"
    
    return f"""
        <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 20px;">
            <div>
                <h3 style="font-size: 16px; color: #4a5568; margin-bottom: 15px;">Uncertainty Trend</h3>
                <div class="metric">
                    <div class="metric-label">Trend</div>
                    <div class="metric-value {u_class}">{u_trend['trend']}</div>
                </div>
                <p style="margin-top: 10px; color: #718096;">{u_trend['description']}</p>
            </div>
            
            <div>
                <h3 style="font-size: 16px; color: #4a5568; margin-bottom: 15px;">Consistency Trend</h3>
                <div class="metric">
                    <div class="metric-label">Trend</div>
                    <div class="metric-value {c_class}">{c_trend['trend']}</div>
                </div>
                <p style="margin-top: 10px; color: #718096;">{c_trend['description']}</p>
            </div>
        </div>
        
        <div style="margin-top: 30px;">
            <h3 style="font-size: 16px; color: #4a5568; margin-bottom: 15px;">Rolling Window (Last 5)</h3>
            <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 15px;">
                <div class="metric">
                    <div class="metric-label">Mean Uncertainty</div>
                    <div class="metric-value">{temporal['rolling_window_5']['mean_uncertainty']:.3f}</div>
                </div>
                <div class="metric">
                    <div class="metric-label">Mean Consistency</div>
                    <div class="metric-value">{temporal['rolling_window_5']['mean_consistency']:.3f}</div>
                </div>
                <div class="metric">
                    <div class="metric-label">Mean Calibration</div>
                    <div class="metric-value">{temporal['rolling_window_5']['mean_calibration']:.3f}</div>
                </div>
            </div>
        </div>
    """

def generate_recent_table_html(recent):
    """Generate recent interactions table HTML"""
    rows = []
    
    for record in reversed(recent):
        risk_zone = record.get('risk_zone', 'UNKNOWN')
        risk_class = f"risk-{risk_zone.lower()}"
        
        rows.append(f"""
            <tr>
                <td class="timestamp">{format_timestamp(record['timestamp'])}</td>
                <td>{record['question'][:60]}...</td>
                <td><span class="risk-badge {risk_class}">{risk_zone}</span></td>
                <td>{record['uncertainty_score']:.3f}</td>
                <td>{record['consistency_score']:.3f}</td>
                <td>{record.get('calibration_score', 0):.3f}</td>
            </tr>
        """)
    
    return f"""
        <table>
            <thead>
                <tr>
                    <th>Timestamp</th>
                    <th>Question</th>
                    <th>Risk Zone</th>
                    <th>Uncertainty</th>
                    <th>Consistency</th>
                    <th>Calibration</th>
                </tr>
            </thead>
            <tbody>
                {''.join(rows)}
            </tbody>
        </table>
    """

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