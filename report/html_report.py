"""Interactive HTML report generator (Plotly.js via CDN — zero extra Python deps) + CSV exporter."""
import json
import os
from typing import Dict, List

import pandas as pd

from engine.portfolio import TradeRecord


def export_html_report(
    output_dir: str,
    summary_tables: Dict[str, pd.DataFrame],
    portfolio_runs: Dict[str, dict],
) -> str:
    """Write trades CSVs and a clean interactive HTML dashboard."""
    os.makedirs(output_dir, exist_ok=True)

    # Export CSVs
    for name, df in summary_tables.items():
        df.to_csv(os.path.join(output_dir, f"{name}.csv"), index=False)

    equity_traces = []
    run_cards_html = []

    for run_name, info in portfolio_runs.items():
        eq_df: pd.DataFrame = info["equity_curve"]
        trades: List[TradeRecord] = info["trades"]
        if trades:
            pd.DataFrame([t.__dict__ for t in trades]).to_csv(
                os.path.join(output_dir, f"trades_{run_name}.csv"), index=False
            )

        if not eq_df.empty:
            equity_traces.append(
                {
                    "name": info["label"],
                    "x": [str(t) for t in eq_df["timestamp"].tolist()],
                    "y": [round(float(v), 2) for v in eq_df["equity"].tolist()],
                    "mode": "lines",
                }
            )

        m = info["metrics"]
        run_cards_html.append(
            f"""
            <div class="card">
                <h3>{info['label']}</h3>
                <div class="grid">
                    <div><span>Vốn cuối</span><strong>${m['final_equity']:,.2f} ({m['return_pct']:+.2f}%)</strong></div>
                    <div><span>Bậc lãi kép cao nhất</span><strong>Bậc {m['max_step']} / 20</strong></div>
                    <div><span>Tổng lệnh</span><strong>{m['total_trades']}</strong></div>
                    <div><span>Winrate (sau phí)</span><strong>{m['winrate_pct']:.2f}%</strong></div>
                    <div><span>Profit Factor</span><strong>{m['profit_factor']:.2f}</strong></div>
                    <div><span>Max Drawdown</span><strong>{m['max_dd_pct']:.2f}%</strong></div>
                    <div><span>Tổng phí sàn</span><strong>${m['total_fees']:,.2f}</strong></div>
                    <div><span>Lệnh Full / Half</span><strong>{m['full_count']} / {m['half_count']}</strong></div>
                </div>
            </div>
            """
        )

    tables_html = []
    for title, df in summary_tables.items():
        tables_html.append(f"<h2>{title}</h2>" + df.to_html(index=False, classes="tbl", float_format=lambda x: f"{x:,.2f}"))

    # 20-step horizontal reference lines
    step_shapes = [
        {
            "type": "line",
            "xref": "paper",
            "x0": 0,
            "x1": 1,
            "y0": round(10_000 * (1.08 ** s), 2),
            "y1": round(10_000 * (1.08 ** s), 2),
            "line": {"color": "rgba(148,163,184,0.22)", "width": 1, "dash": "dot"},
        }
        for s in range(1, 21)
    ]

    html = f"""<!DOCTYPE html>
<html lang="vi">
<head>
<meta charset="UTF-8">
<title>CHICKTRADE — Báo Cáo Backtest 1 Năm M5 (EMA 34-89-200)</title>
<script src="https://cdn.plot.ly/plotly-2.32.0.min.js"></script>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; background: #0f172a; color: #e2e8f0; margin: 0; padding: 24px; }}
  h1 {{ margin-top: 0; color: #38bdf8; }}
  h2 {{ margin-top: 32px; color: #93c5fd; border-bottom: 1px solid #334155; padding-bottom: 6px; }}
  .cards {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(340px, 1fr)); gap: 16px; margin: 20px 0; }}
  .card {{ background: #1e293b; border: 1px solid #334155; border-radius: 10px; padding: 16px; }}
  .card h3 {{ margin: 0 0 12px 0; color: #f8fafc; font-size: 16px; }}
  .grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }}
  .grid div {{ background: #0f172a; padding: 8px 10px; border-radius: 6px; }}
  .grid span {{ display: block; font-size: 11px; color: #94a3b8; }}
  .grid strong {{ font-size: 15px; color: #f8fafc; }}
  #equity-chart {{ width: 100%; height: 460px; background: #1e293b; border-radius: 10px; padding: 8px; box-sizing: border-box; }}
  table.tbl {{ width: 100%; border-collapse: collapse; margin-top: 10px; background: #1e293b; font-size: 13px; }}
  table.tbl th, table.tbl td {{ border: 1px solid #334155; padding: 7px 10px; text-align: right; }}
  table.tbl th {{ background: #0f172a; color: #93c5fd; text-align: center; }}
  table.tbl td:first-child, table.tbl td:nth-child(2) {{ text-align: left; }}
</style>
</head>
<body>
  <h1>CHICKTRADE — Báo Cáo Backtest 1 Năm Khung M5 (EMA 34–89–200)</h1>
  <p>Dữ liệu Binance miễn phí 12 tháng gần nhất | Vốn khởi điểm $10,000 | Quy tắc bao 200 Unit (ATR-normalized) | 20 bậc lãi kép (+8%/bậc).</p>
  <div class="cards">
    {''.join(run_cards_html)}
  </div>
  <h2>Biểu đồ Tăng trưởng Vốn (Equity Curve & 20 Bậc Lãi Kép)</h2>
  <div id="equity-chart"></div>
  {''.join(tables_html)}
  <script>
    const traces = {json.dumps(equity_traces)};
    const layout = {{
      paper_bgcolor: '#1e293b',
      plot_bgcolor: '#0f172a',
      font: {{ color: '#e2e8f0' }},
      xaxis: {{ title: 'Thời gian (UTC)', gridcolor: '#1e293b' }},
      yaxis: {{ title: 'Vốn ($)', gridcolor: '#334155' }},
      shapes: {json.dumps(step_shapes)},
      legend: {{ orientation: 'h', y: 1.08 }}
    }};
    Plotly.newPlot('equity-chart', traces, layout, {{ responsive: true }});
  </script>
</body>
</html>
"""
    report_path = os.path.join(output_dir, "backtest_report.html")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(html)
    return report_path
