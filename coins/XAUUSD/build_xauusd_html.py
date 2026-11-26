"""Build standalone Light-Mode HTML Report for CHICKTRADE v3.3 — XAUUSD 100x Quant Study (2020-2024 + 2025 Blind OOS)."""
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
IN_JSON = os.path.join(ROOT, "coins", "XAUUSD", "xauusd_study_results.json")
OUT_HTML = os.path.join(ROOT, "core", "reports", "chicktrade_xauusd_100x_report.html")


def build_html():
    with open(IN_JSON, encoding="utf-8") as f:
        data = json.load(f)

    data_js = json.dumps(data, ensure_ascii=False)

    html = f"""<!DOCTYPE html>
<html lang="vi">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>CHICKTRADE v3.3 — Báo Cáo Quant XAUUSD (Vàng 100× · 2020–2024 &amp; Blind OOS 2025)</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@500;600;700;800&display=swap" rel="stylesheet">
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
  :root {{
    --bg: #F8FAFC;
    --card: #FFFFFF;
    --surface: #F1F5F9;
    --border: #E2E8F0;
    --text: #0F172A;
    --sec: #475569;
    --muted: #64748B;
    --gold: #D97706;
    --gold-deep: #B45309;
    --gold-bg: #FFFBEB;
    --emerald: #059669;
    --emerald-bg: #ECFDF5;
    --rose: #DC2626;
    --rose-bg: #FFF1F2;
    --sky: #0284C7;
    --sky-bg: #F0F9FF;
    --purple: #6D28D9;
    --sans: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    --mono: 'JetBrains Mono', ui-monospace, Consolas, monospace;
  }}
  * {{ box-sizing: border-box; }}
  body {{
    margin: 0;
    background: var(--bg);
    color: var(--text);
    font-family: var(--sans);
    font-size: 13.5px;
    line-height: 1.5;
    -webkit-font-smoothing: antialiased;
  }}
  .num {{ font-family: var(--mono); font-variant-numeric: tabular-nums; }}
  .wrap {{ max-width: 1260px; margin: 0 auto; padding: 20px 16px 48px 16px; }}

  /* Header */
  .hero {{
    background: linear-gradient(135deg, #FFFFFF 0%, #FFFBEB 100%);
    border: 1px solid #FDE68A;
    border-top: 5px solid var(--gold);
    border-radius: 16px;
    padding: 22px 24px;
    box-shadow: 0 4px 14px rgba(180, 83, 9, 0.06);
    margin-bottom: 18px;
  }}
  .hero-top {{ display: flex; flex-wrap: wrap; justify-content: space-between; align-items: flex-start; gap: 12px; }}
  .hero h1 {{ margin: 0; font-size: 22px; font-weight: 800; letter-spacing: -0.02em; color: #0F172A; }}
  .hero-sub {{ margin: 5px 0 0 0; font-size: 12.8px; color: var(--sec); }}
  .pill-row {{ display: flex; flex-wrap: wrap; gap: 7px; margin-top: 12px; }}
  .pill {{
    display: inline-flex; align-items: center; gap: 5px;
    padding: 4px 10px; border-radius: 999px; font-size: 11.5px; font-weight: 700;
    border: 1px solid var(--border); background: #FFFFFF; color: var(--sec);
  }}
  .pill-gold {{ background: var(--gold-bg); color: var(--gold-deep); border-color: #FCD34D; }}
  .pill-green {{ background: var(--emerald-bg); color: var(--emerald); border-color: #A7F3D0; }}
  .pill-sky {{ background: var(--sky-bg); color: #0369A1; border-color: #BAE6FD; }}

  /* KPI Strip */
  .kpi-grid {{
    display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 12px; margin-bottom: 18px;
  }}
  @media (max-width: 900px) {{ .kpi-grid {{ grid-template-columns: repeat(2, 1fr); }} }}
  @media (max-width: 520px) {{ .kpi-grid {{ grid-template-columns: 1fr; }} }}
  .kpi-card {{
    background: var(--card); border: 1px solid var(--border); border-radius: 14px;
    padding: 15px 16px; box-shadow: 0 1px 3px rgba(15,23,42,0.03);
  }}
  .kpi-lbl {{ font-size: 11px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.04em; color: var(--muted); }}
  .kpi-val {{ font-size: 23px; font-weight: 800; margin: 4px 0 2px 0; }}
  .kpi-note {{ font-size: 11.5px; color: var(--sec); }}

  /* Section Card */
  .card {{
    background: var(--card); border: 1px solid var(--border); border-radius: 14px;
    padding: 18px 20px; margin-bottom: 18px; box-shadow: 0 1px 3px rgba(15,23,42,0.03);
  }}
  .card-head {{
    display: flex; flex-wrap: wrap; justify-content: space-between; align-items: center;
    gap: 10px; margin-bottom: 14px;
  }}
  .card-head h2 {{ margin: 0; font-size: 16px; font-weight: 800; }}
  .card-head p {{ margin: 2px 0 0 0; font-size: 12px; color: var(--sec); }}

  /* Sweet Spot 4 TF Cards */
  .tf-grid {{ display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 12px; }}
  @media (max-width: 980px) {{ .tf-grid {{ grid-template-columns: repeat(2, 1fr); }} }}
  @media (max-width: 540px) {{ .tf-grid {{ grid-template-columns: 1fr; }} }}
  .tf-card {{
    border: 1px solid var(--border); border-top: 3.5px solid var(--sky);
    border-radius: 12px; padding: 14px; background: #FFFFFF;
  }}
  .tf-card.best-tf {{ border-top-color: var(--gold); background: linear-gradient(180deg, #FFFBEB 0%, #FFFFFF 36%); }}
  .tf-card-top {{ display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px; }}
  .tf-name {{ font-size: 16px; font-weight: 800; }}
  .kv-row {{ display: flex; justify-content: space-between; font-size: 12px; padding: 3px 0; border-bottom: 1px dashed #F1F5F9; }}
  .kv-row:last-child {{ border-bottom: none; }}

  /* 3 Buffer Cards (100x) */
  .buf-grid {{ display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 12px; margin-bottom: 14px; }}
  @media (max-width: 860px) {{ .buf-grid {{ grid-template-columns: 1fr; }} }}
  .buf-card {{
    border: 1px solid var(--border); border-radius: 12px; padding: 14px; background: #FFFFFF;
    cursor: pointer; transition: all 0.15s ease;
  }}
  .buf-card:hover {{ border-color: #93C5FD; }}
  .buf-card.active {{ border: 2px solid var(--gold); background: #FFFDF7; box-shadow: 0 3px 10px rgba(217,119,6,0.12); }}

  /* Calculator */
  .calc-grid {{ display: grid; grid-template-columns: 1fr 1.6fr; gap: 16px; align-items: start; }}
  @media (max-width: 820px) {{ .calc-grid {{ grid-template-columns: 1fr; }} }}
  .calc-box {{ background: #F8FAFC; border: 1px solid var(--border); border-radius: 12px; padding: 14px; }}
  .calc-row {{ margin-bottom: 10px; }}
  .calc-row label {{ display: block; font-size: 11.5px; font-weight: 700; color: var(--sec); margin-bottom: 4px; }}
  .calc-row input, .calc-row select {{
    width: 100%; padding: 8px 10px; border: 1px solid #CBD5E1; border-radius: 8px;
    font-family: var(--mono); font-size: 13px; background: #FFFFFF;
  }}
  .calc-out-grid {{ display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 10px; }}
  @media (max-width: 560px) {{ .calc-out-grid {{ grid-template-columns: repeat(2, 1fr); }} }}
  .calc-out-item {{
    background: #FFFFFF; border: 1px solid var(--border); border-radius: 10px; padding: 10px 12px;
  }}
  .calc-out-item span {{ display: block; font-size: 10.8px; color: var(--muted); font-weight: 700; text-transform: uppercase; }}
  .calc-out-item strong {{ display: block; font-size: 15.5px; margin-top: 2px; }}

  /* Tables & Badges */
  .filter-bar {{ display: flex; flex-wrap: wrap; gap: 6px; margin-bottom: 12px; }}
  .f-btn {{
    border: 1px solid var(--border); background: #FFFFFF; color: var(--sec);
    border-radius: 999px; padding: 4px 11px; font-size: 11.5px; font-weight: 700; cursor: pointer;
  }}
  .f-btn.active {{ background: #0F172A; color: #FFFFFF; border-color: #0F172A; }}
  .tbl-wrap {{ overflow-x: auto; border: 1px solid var(--border); border-radius: 10px; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 12.2px; }}
  th {{
    background: #F8FAFC; color: var(--sec); text-align: left; padding: 9px 10px;
    font-size: 11px; text-transform: uppercase; letter-spacing: 0.03em; border-bottom: 1px solid var(--border);
    white-space: nowrap;
  }}
  td {{ padding: 8px 10px; border-bottom: 1px solid #F1F5F9; white-space: nowrap; }}
  tr:hover td {{ background: #F8FAFC; }}
  .badge {{ display: inline-block; padding: 2px 7px; border-radius: 5px; font-size: 10.5px; font-weight: 800; }}
  .b-core {{ background: var(--emerald-bg); color: var(--emerald); border: 1px solid #A7F3D0; }}
  .b-scout {{ background: var(--sky-bg); color: #0369A1; border: 1px solid #BAE6FD; }}
  .b-drop {{ background: var(--surface); color: var(--muted); border: 1px solid var(--border); }}
  .b-oos {{ background: var(--gold-bg); color: var(--gold-deep); border: 1px solid #FDE68A; }}

  @keyframes shimmer {{
    0% {{ transform: translateX(-140%); }}
    45%, 100% {{ transform: translateX(180%); }}
  }}
  .must-badge {{
    position: relative; display: inline-block; overflow: hidden;
    padding: 2px 7px; border-radius: 5px; font-size: 10px; font-weight: 800;
    background: linear-gradient(135deg, #F59E0B 0%, #EA580C 50%, #D97706 100%);
    color: #FFFFFF; border: 1px solid #FDE68A; box-shadow: 0 2px 5px rgba(245,158,11,0.3);
    vertical-align: middle; margin-left: 4px;
  }}
  .must-badge::after {{
    content: ""; position: absolute; top: 0; left: 0; width: 60%; height: 100%;
    background: linear-gradient(110deg, transparent 20%, rgba(255,255,255,0.65) 50%, transparent 80%);
    animation: shimmer 3.2s ease-in-out infinite;
  }}
  .pos {{ color: var(--emerald); font-weight: 700; }}
  .neg {{ color: var(--rose); font-weight: 700; }}
  .callout {{
    background: #FFFBEB; border: 1px solid #FDE68A; border-left: 4px solid var(--gold);
    border-radius: 10px; padding: 12px 14px; font-size: 12.5px; color: #78350F;
  }}
</style>
</head>
<body>
<div class="wrap">

  <!-- HERO -->
  <section class="hero">
    <div class="hero-top">
      <div>
        <h1>🥇 CHICKTRADE v3.3 — Báo Cáo Kiểm Định Quant XAUUSD (Vàng · Đòn Bẩy 100×)</h1>
        <p class="hero-sub">Quét trọn 4 khung <b>M5 · M15 · M30 · 1H</b> (48 Setup EMA 34/89/200) trên <b>560,741 nến 5m</b> thực tế (2020–2024 Train + 2025 Blind Out-of-Sample)</p>
      </div>
      <div style="text-align:right;font-size:11.5px;color:#64748B;" class="num">
        <div>Xuất báo cáo: <b id="gen-utc">--</b></div>
        <div>Phí mô phỏng: <b>IC Markets Raw (0.012% RT)</b></div>
      </div>
    </div>
    <div class="pill-row">
      <span class="pill pill-gold">⚡ Đòn bẩy: 100× Cross (IC Markets)</span>
      <span class="pill pill-green">🎯 Điểm Ngọt Vàng: ONLY_TP1 (Chốt 100% tại TP1)</span>
      <span class="pill pill-sky">📐 Thước đo: U3_HTF_MACRO_ATR 4H (Sàn 0.04%)</span>
      <span class="pill">🕒 Lọc phiên Forex: Thứ 2 → Thứ 6 (bỏ cuối tuần)</span>
      <span class="pill">🛡️ Phanh 2-SL Shadow-Win + Lãi kép S2 (+8% / -10%)</span>
    </div>
  </section>

  <!-- TOP KPI STRIP -->
  <section class="kpi-grid" id="kpi-strip"></section>

  <!-- KEY DISCOVERY CALLOUT -->
  <div class="callout" style="margin-bottom:18px;">
    <b>💡 Phát hiện định lượng quan trọng nhất trên Vàng (XAUUSD) so với Crypto:</b>
    Khác với BTC/Altcoin (hay chạy xu hướng dài nên gồng TP2 `8U–12U` rất ngon), <b>Vàng có đặc tính quét râu 2 đầu mạnh ở phiên Âu/Mỹ</b>.
    Kết quả quét lưới toàn bộ 4 khung <code>M5 / M15 / M30 / 1H</code> suốt 4.25 năm (<code>2020–2024</code>) cho thấy cơ chế <b><code>ONLY_TP1</code> (Chốt gọn 100% tại TP1)</b> áp đảo hoàn toàn cơ chế gồng TP2 trên cả 4 khung:
    <b>M5 / M15 / M30 đều đạt điểm ngọt tại <code>SL 6.0U · TP1 3.5U</code></b> (WR ~67–69%), còn <b>1H đạt điểm ngọt tại <code>SL 5.0U · TP1 3.0U</code></b>.
    Đặc biệt, khi mang đúng <b>27 Setup tuyển chọn từ 2020–2024</b> sang kiểm định mù năm <b>2025</b> (218 lệnh chưa từng thấy): <b>WR đạt 71.6%, Thắng tiền (Profit WR) đạt 57.5%, lãi +37.78% (Bao 400G) đến +65.99% (Bao 250G)</b>!
  </div>

  <!-- SWEET SPOT 4 TIMEFRAMES -->
  <section class="card">
    <div class="card-head">
      <div>
        <h2>🎯 Điểm Ngọt Theo Từng Khung Thời Gian (2020–2024 · Chưa Lọc Setup)</h2>
        <p>Quét toàn bộ tổ hợp SL (4U/5U/6U) × TP1 (3U/3.5U/4U) × TP2 (8U/10U/12U) × Kiến trúc thoát lệnh trên dữ liệu 4.25 năm</p>
      </div>
    </div>
    <div class="tf-grid" id="tf-sweet-grid"></div>
  </section>

  <!-- 100X PORTFOLIO SIMULATIONS & EQUITY CURVE -->
  <section class="card">
    <div class="card-head">
      <div>
        <h2>🚀 Mô Phỏng Quỹ $10,000 Tại Đòn Bẩy 100× (IC Markets Raw · 27 Setup Đạt Chuẩn)</h2>
        <p>Bấm chọn chế độ Bao (400G / 250G / 150G) để so sánh đường cong tài sản liên tục 2020–2024 (Train) &amp; 2025 (Blind OOS)</p>
      </div>
    </div>
    <div class="buf-grid" id="buf-cards"></div>
    <div id="eq-chart" style="width:100%;height:370px;"></div>
  </section>

  <!-- YEARLY BREAKDOWN -->
  <section class="card">
    <div class="card-head">
      <div>
        <h2>📅 Hiệu Suất Từng Năm (2020 → 2025 · Kiểm Định Mù 2025)</h2>
        <p>Đánh giá độ ổn định qua mọi mùa sóng Vàng: Covid 2020, Sideway 2021–2022, Tăng tốc 2023–2024 và Siêu sóng 2025 (OOS)</p>
      </div>
    </div>
    <div class="tbl-wrap">
      <table>
        <thead>
          <tr>
            <th>Năm</th>
            <th>Giai đoạn</th>
            <th>Số lệnh</th>
            <th>WR Lệnh</th>
            <th>Thắng tiền (PWR)</th>
            <th>Profit Factor</th>
            <th>Net (U)</th>
            <th>ROI Bao 400G (100×)</th>
            <th>MaxDD 400G</th>
            <th>ROI Bao 250G (100×)</th>
            <th>MaxDD 250G</th>
          </tr>
        </thead>
        <tbody id="yearly-tbody"></tbody>
      </table>
    </div>
  </section>

  <!-- 100X LOT & MARGIN CALCULATOR -->
  <section class="card">
    <div class="card-head">
      <div>
        <h2>🧮 Máy Tính Lot Size &amp; Ký Quỹ Đòn Bẩy 100× (IC Markets XAUUSD)</h2>
        <p>Tính chính xác số Lot (1 Lot = 100 oz), tiền ký quỹ chiếm dụng và khoảng cách an toàn so với cháy tài khoản ở đòn bẩy 100×</p>
      </div>
    </div>
    <div class="calc-grid">
      <div class="calc-box">
        <div class="calc-row">
          <label>Vốn tài khoản (USD)</label>
          <input type="number" id="c-eq" value="10000" min="100" step="500" oninput="recalc100x()">
        </div>
        <div class="calc-row">
          <label>Giá Vàng XAUUSD hiện tại ($/oz)</label>
          <input type="number" id="c-px" value="3550" min="1000" step="10" oninput="recalc100x()">
        </div>
        <div class="calc-row">
          <label>Chế độ Bao (Số U chia vốn)</label>
          <select id="c-buf" onchange="recalc100x()">
            <option value="400" selected>Bao 400G (An toàn tối ưu · MaxDD -9.7%)</option>
            <option value="250">Bao 250G (Tăng tốc chuẩn · MaxDD -15.1%)</option>
            <option value="150">Bao 150G (Rủi ro cao · MaxDD -24.5%)</option>
          </select>
        </div>
        <div class="calc-row">
          <label>Khung thời gian vào lệnh</label>
          <select id="c-tf" onchange="recalc100x()">
            <option value="5m">M5 (1U TB = 0.0531% · SL 6U / TP1 3.5U)</option>
            <option value="15m">M15 (1U TB = 0.0906% · SL 6U / TP1 3.5U)</option>
            <option value="30m" selected>M30 (1U TB = 0.1281% · SL 6U / TP1 3.5U)</option>
            <option value="1h">1H (1U TB = 0.1844% · SL 5U / TP1 3.0U)</option>
          </select>
        </div>
      </div>
      <div class="calc-out-grid num" id="calc-outputs"></div>
    </div>
  </section>

  <!-- FULL 48-SETUP AUDIT TABLE -->
  <section class="card">
    <div class="card-head">
      <div>
        <h2>📋 Bảng Kiểm Định Đầy Đủ 48 Setup XAUUSD (M5 · M15 · M30 · 1H)</h2>
        <p>Tiêu chí tuyển chọn chỉ dùng 2020–2024 (CORE: N≥18, PWR≥56%, PF≥1.28, Net≥+20U | SCOUT: N≥10, PWR≥53%, PF≥1.13, Net≥+8U) — Cột 2025 là kiểm định mù hoàn toàn</p>
      </div>
    </div>
    <div class="filter-bar">
      <button class="f-btn tf-f active" onclick="setTfFilter('ALL', this)">Tất cả 4 khung</button>
      <button class="f-btn tf-f" onclick="setTfFilter('5m', this)">M5</button>
      <button class="f-btn tf-f" onclick="setTfFilter('15m', this)">M15</button>
      <button class="f-btn tf-f" onclick="setTfFilter('30m', this)">M30</button>
      <button class="f-btn tf-f" onclick="setTfFilter('1h', this)">1H</button>
      <span style="width:10px;"></span>
      <button class="f-btn st-f active" onclick="setStFilter('ALL', this)">Tất cả (48)</button>
      <button class="f-btn st-f" onclick="setStFilter('ACTIVE', this)">Đạt chuẩn (CORE + SCOUT)</button>
      <button class="f-btn st-f" onclick="setStFilter('CORE', this)">Chỉ CORE</button>
      <button class="f-btn st-f" onclick="setStFilter('DROP', this)">Đã loại (DROP)</button>
    </div>
    <div class="tbl-wrap">
      <table>
        <thead>
          <tr>
            <th>Khung</th>
            <th>Setup</th>
            <th>Hạng</th>
            <th>SL / TP1</th>
            <th>2020–24 Lệnh</th>
            <th>2020–24 WR</th>
            <th>2020–24 PWR</th>
            <th>2020–24 PF</th>
            <th>2020–24 Net (U)</th>
            <th>2025 OOS Lệnh</th>
            <th>2025 OOS WR</th>
            <th>2025 OOS Net (U)</th>
          </tr>
        </thead>
        <tbody id="audit-tbody"></tbody>
      </table>
    </div>
  </section>

</div>

<script>
const D = {data_js};
let activeBuf = '400';
let tfFilter = 'ALL';
let stFilter = 'ALL';

document.getElementById('gen-utc').textContent = D.generated_utc;

function fmtUsd(v) {{
  return '$' + Number(v || 0).toLocaleString('en-US', {{ minimumFractionDigits: 2, maximumFractionDigits: 2 }});
}}

function renderKpis() {{
  const s400 = D.sims_train['400'];
  const s250 = D.sims_train['250'];
  const oos400 = D.sims_oos['400'];
  const oos250 = D.sims_oos['250'];
  const coreCnt = D.audit_48.filter(r => r.status === 'CORE').length;
  const scoutCnt = D.audit_48.filter(r => r.status === 'SCOUT').length;

  document.getElementById('kpi-strip').innerHTML = `
    <div class="kpi-card" style="border-top:3.5px solid #059669;">
      <div class="kpi-lbl">Bao 400G @ 100× (2020–2024)</div>
      <div class="kpi-val num" style="color:#059669;">+${{s400.roi_pct.toFixed(1)}}%</div>
      <div class="kpi-note num">$10k → <b>${{fmtUsd(s400.final_eq)}}</b> · DD <b>${{s400.max_dd_pct}}%</b> · Margin ≤ <b>${{s400.max_margin_pct}}%</b></div>
    </div>
    <div class="kpi-card" style="border-top:3.5px solid #D97706;">
      <div class="kpi-lbl">Bao 250G @ 100× (2020–2024)</div>
      <div class="kpi-val num" style="color:#B45309;">+${{s250.roi_pct.toFixed(1)}}%</div>
      <div class="kpi-note num">$10k → <b>${{fmtUsd(s250.final_eq)}}</b> · DD <b>${{s250.max_dd_pct}}%</b> · Margin ≤ <b>${{s250.max_margin_pct}}%</b></div>
    </div>
    <div class="kpi-card" style="border-top:3.5px solid #0284C7;">
      <div class="kpi-lbl">Kiểm Định Mù 2025 (Blind OOS)</div>
      <div class="kpi-val num" style="color:#0284C7;">+${{oos400.roi_pct.toFixed(1)}}% / +${{oos250.roi_pct.toFixed(1)}}%</div>
      <div class="kpi-note num">${{oos400.trades}} lệnh · WR <b>${{oos400.wr}}%</b> · Thắng tiền <b>${{oos400.pwr}}%</b></div>
    </div>
    <div class="kpi-card" style="border-top:3.5px solid #6D28D9;">
      <div class="kpi-lbl">Setup Đạt Chuẩn / 48 Quét</div>
      <div class="kpi-val num" style="color:#6D28D9;">${{coreCnt + scoutCnt}} / 48</div>
      <div class="kpi-note"><b>${{coreCnt}} CORE</b> (1.0×) + <b>${{scoutCnt}} SCOUT</b> (0.5×) · Top 8 THẤY LÀ QUẤT</div>
    </div>
  `;
}}

function renderSweetSpots() {{
  const tfs = ['5m', '15m', '30m', '1h'];
  const lbls = {{ '5m': 'M5 (5 Phút)', '15m': 'M15 (15 Phút)', '30m': 'M30 (30 Phút)', '1h': '1H (1 Giờ)' }};
  document.getElementById('tf-sweet-grid').innerHTML = tfs.map(tf => {{
    const b = D.best_cfg_by_tf[tf];
    const u = D.u_stats[tf];
    const isBest = tf === '30m';
    const px = 3550;
    const uUsd = px * (u.avg_pct / 100);
    return `
      <div class="tf-card ${{isBest ? 'best-tf' : ''}}">
        <div class="tf-card-top">
          <span class="tf-name">${{lbls[tf]}}</span>
          ${{isBest ? '<span class="badge b-oos">👑 Khung Đẹp Nhất</span>' : '<span class="badge b-scout">ONLY_TP1</span>'}}
        </div>
        <div class="kv-row"><span>Điểm ngọt SL / TP1</span><b class="num" style="color:#0F172A;">-${{b.sl_u}}U / +${{b.tp1_u}}U</b></div>
        <div class="kv-row"><span>1U trung bình (4H)</span><b class="num">${{u.avg_pct.toFixed(3)}}% (~$${{uUsd.toFixed(2)}})</b></div>
        <div class="kv-row"><span>Số lệnh (2020–24)</span><b class="num">${{b.n}} lệnh</b></div>
        <div class="kv-row"><span>WR / Thắng tiền (PWR)</span><b class="num">${{b.wr}}% / <span class="pos">${{b.pwr}}%</span></b></div>
        <div class="kv-row"><span>PF / Tổng Net (chưa lọc)</span><b class="num">${{b.pf}} / <span class="pos">+${{b.net_u}}U</span></b></div>
      </div>
    `;
  }}).join('');
}}

function selectBuf(buf) {{
  activeBuf = String(buf);
  renderBufCards();
  renderEqChart();
}}

function renderBufCards() {{
  const items = [
    {{ buf: '400', title: '🛡️ Bao 400G @ 100× (Khuyên Dùng)', desc: 'Rủi ro ~1.4%/lệnh · DD siêu thấp < 10% · Margin tối đa 6.7%' }},
    {{ buf: '250', title: '⚡ Bao 250G @ 100× (Tăng Tốc Chuẩn)', desc: 'Rủi ro ~2.3%/lệnh · Cân bằng lợi nhuận/DD · Margin tối đa 10.9%' }},
    {{ buf: '150', title: '🔥 Bao 150G @ 100× (Hiệu Suất Cao)', desc: 'Rủi ro ~3.8%/lệnh · Lãi kép cực nhanh nhưng DD ~24%' }},
  ];
  document.getElementById('buf-cards').innerHTML = items.map(it => {{
    const tr = D.sims_train[it.buf];
    const oo = D.sims_oos[it.buf];
    return `
      <div class="buf-card ${{activeBuf === it.buf ? 'active' : ''}}" onclick="selectBuf('${{it.buf}}')">
        <div style="font-weight:800;font-size:13.8px;margin-bottom:4px;">${{it.title}}</div>
        <div style="font-size:11.5px;color:#64748B;margin-bottom:8px;">${{it.desc}}</div>
        <div class="num" style="font-size:12.2px;display:grid;grid-template-columns:1fr 1fr;gap:4px;">
          <span>2020–24: <b class="pos">+${{tr.roi_pct.toFixed(1)}}%</b></span>
          <span>MaxDD: <b class="neg">${{tr.max_dd_pct}}%</b></span>
          <span>2025 OOS: <b style="color:#0284C7;">+${{oo.roi_pct.toFixed(1)}}%</b></span>
          <span>Lot TB: <b>${{tr.avg_lots}} lot</b></span>
        </div>
      </div>
    `;
  }}).join('');
}}

function renderEqChart() {{
  const sim = D.sims_full[activeBuf];
  const x = sim.curve.map(p => p.dt);
  const y = sim.curve.map(p => p.eq);
  Plotly.newPlot('eq-chart', [{{
    x, y, type: 'scatter', mode: 'lines',
    fill: 'tozeroy',
    fillcolor: 'rgba(217, 119, 6, 0.10)',
    line: {{ color: '#D97706', width: 2.5 }},
    hovertemplate: '<b>%{{x}}</b><br>Equity: <b>$%{{y:,.2f}}</b><extra></extra>'
  }}], {{
    paper_bgcolor: '#FFFFFF',
    plot_bgcolor: '#F8FAFC',
    margin: {{ t: 24, r: 20, l: 68, b: 36 }},
    font: {{ family: 'Inter, sans-serif', size: 11.5, color: '#475569' }},
    xaxis: {{ gridcolor: '#E2E8F0' }},
    yaxis: {{ gridcolor: '#E2E8F0', tickprefix: '$', type: activeBuf === '150' ? 'log' : 'linear' }},
    shapes: [{{
      type: 'rect', xref: 'x', yref: 'paper',
      x0: '2025-01-01', x1: '2025-12-31', y0: 0, y1: 1,
      fillcolor: 'rgba(2, 132, 199, 0.08)', line: {{ width: 0 }}
    }}],
    annotations: [{{
      x: '2025-07-01', y: 0.96, xref: 'x', yref: 'paper',
      text: '<b>VÙNG KIỂM ĐỊNH MÙ 2025 (BLIND OOS)</b>',
      showarrow: false, font: {{ size: 11, color: '#0369A1' }}
    }}]
  }}, {{ responsive: true, displayModeBar: false }});
}}

function renderYearly() {{
  const yrs = ['2020', '2021', '2022', '2023', '2024', '2025'];
  document.getElementById('yearly-tbody').innerHTML = yrs.map(yr => {{
    const r = D.yearly[yr];
    const isOos = yr === '2025';
    return `
      <tr style="${{isOos ? 'background:#F0F9FF;' : ''}}">
        <td class="num"><b>${{yr}}</b></td>
        <td>${{isOos ? '<span class="badge b-oos">BLIND OOS (Mù)</span>' : '<span class="badge b-drop">Train (2020–24)</span>'}}</td>
        <td class="num">${{r.trades}}</td>
        <td class="num"><b>${{r.wr}}%</b></td>
        <td class="num"><b class="${{r.pwr >= 50 ? 'pos' : 'neg'}}">${{r.pwr}}%</b></td>
        <td class="num"><b>${{r.pf}}</b></td>
        <td class="num"><span class="${{r.net_u >= 0 ? 'pos' : 'neg'}}">${{r.net_u >= 0 ? '+' : ''}}${{r.net_u}}U</span></td>
        <td class="num"><span class="${{r.roi_400g >= 0 ? 'pos' : 'neg'}}">${{r.roi_400g >= 0 ? '+' : ''}}${{r.roi_400g}}%</span></td>
        <td class="num neg">${{r.dd_400g}}%</td>
        <td class="num"><span class="${{r.roi_250g >= 0 ? 'pos' : 'neg'}}">${{r.roi_250g >= 0 ? '+' : ''}}${{r.roi_250g}}%</span></td>
        <td class="num neg">${{r.dd_250g}}%</td>
      </tr>
    `;
  }}).join('');
}}

function recalc100x() {{
  const eq = Math.max(100, Number(document.getElementById('c-eq').value || 10000));
  const px = Math.max(500, Number(document.getElementById('c-px').value || 3550));
  const buf = Number(document.getElementById('c-buf').value || 400);
  const tf = document.getElementById('c-tf').value || '30m';
  const uPct = D.u_stats[tf].avg_pct;
  const uFrac = uPct / 100;
  const best = D.best_cfg_by_tf[tf];

  const uUsd = eq / buf;
  const notional = uUsd / uFrac;
  const lots = notional / (px * 100);
  const margin100 = notional / 100;
  const marginPct = (margin100 / eq) * 100;
  const slUsd = uUsd * best.sl_u;
  const slPctEq = (slUsd / eq) * 100;
  const tp1Usd = uUsd * best.tp1_u;
  const tp1PctEq = (tp1Usd / eq) * 100;
  const slDistPx = px * uFrac * best.sl_u;
  const tp1DistPx = px * uFrac * best.tp1_u;

  document.getElementById('calc-outputs').innerHTML = `
    <div class="calc-out-item"><span>Khối lượng vào lệnh</span><strong style="color:#0284C7;">${{lots.toFixed(2)}} Lot</strong><small style="color:#64748B;">Notional $${{Math.round(notional).toLocaleString()}}</small></div>
    <div class="calc-out-item"><span>Ký quỹ @ 100×</span><strong>$${{margin100.toFixed(2)}} (${{marginPct.toFixed(1)}}%)</strong><small style="color:#059669;">Dư địa ký quỹ rất rộng</small></div>
    <div class="calc-out-item"><span>Giá trị 1U</span><strong>$${{uUsd.toFixed(2)}}</strong><small style="color:#64748B;">1U = $${{(px * uFrac).toFixed(2)}}/oz</small></div>
    <div class="calc-out-item"><span>Chốt lời TP1 (+${{best.tp1_u}}U)</span><strong class="pos">+$${{tp1Usd.toFixed(2)}} (+${{tp1PctEq.toFixed(2)}}%)</strong><small style="color:#64748B;">Biên độ ~$${{tp1DistPx.toFixed(1)}}/oz</small></div>
    <div class="calc-out-item"><span>Cắt lỗ SL (-${{best.sl_u}}U)</span><strong class="neg">-$${{slUsd.toFixed(2)}} (-${{slPctEq.toFixed(2)}}%)</strong><small style="color:#64748B;">Biên độ ~$${{slDistPx.toFixed(1)}}/oz</small></div>
    <div class="calc-out-item"><span>An toàn thanh lý</span><strong class="pos">Gấp ${{(100 / Math.max(slPctEq, 0.1)).toFixed(0)}}× SL</strong><small style="color:#64748B;">Không thể cháy bởi 1 lệnh</small></div>
  `;
}}

function setTfFilter(tf, btn) {{
  tfFilter = tf;
  document.querySelectorAll('.tf-f').forEach(b => b.classList.remove('active'));
  btn.classList.add('active');
  renderAuditTable();
}}

function setStFilter(st, btn) {{
  stFilter = st;
  document.querySelectorAll('.st-f').forEach(b => b.classList.remove('active'));
  btn.classList.add('active');
  renderAuditTable();
}}

function renderAuditTable() {{
  const rows = D.audit_48.filter(r => {{
    if (tfFilter !== 'ALL' && r.tf !== tfFilter) return false;
    if (stFilter === 'ACTIVE' && r.status === 'DROP') return false;
    if (stFilter === 'CORE' && r.status !== 'CORE') return false;
    if (stFilter === 'DROP' && r.status !== 'DROP') return false;
    return true;
  }});
  document.getElementById('audit-tbody').innerHTML = rows.map(r => {{
    const stCls = r.status === 'CORE' ? 'b-core' : (r.status === 'SCOUT' ? 'b-scout' : 'b-drop');
    const mustHtml = r.is_must ? '<span class="must-badge">THẤY LÀ QUẤT</span>' : '';
    return `
      <tr>
        <td class="num"><b>${{r.tf.toUpperCase()}}</b></td>
        <td class="num"><b>${{r.setup}}</b>${{mustHtml}}</td>
        <td><span class="badge ${{stCls}}">${{r.status}}</span></td>
        <td class="num">-${{r.sl_u}}U / +${{r.tp1_u}}U</td>
        <td class="num">${{r.train_n}}</td>
        <td class="num">${{r.train_wr}}%</td>
        <td class="num"><b class="${{r.train_pwr >= 55 ? 'pos' : (r.train_pwr < 50 ? 'neg' : '')}}">${{r.train_pwr}}%</b></td>
        <td class="num">${{r.train_pf}}</td>
        <td class="num"><span class="${{r.train_net_u >= 0 ? 'pos' : 'neg'}}">${{r.train_net_u >= 0 ? '+' : ''}}${{r.train_net_u}}U</span></td>
        <td class="num">${{r.oos_n}}</td>
        <td class="num">${{r.oos_wr}}%</td>
        <td class="num"><span class="${{r.oos_net_u >= 0 ? 'pos' : 'neg'}}">${{r.oos_net_u >= 0 ? '+' : ''}}${{r.oos_net_u}}U</span></td>
      </tr>
    `;
  }}).join('');
}}

renderKpis();
renderSweetSpots();
renderBufCards();
renderEqChart();
renderYearly();
recalc100x();
renderAuditTable();
</script>
</body>
</html>
"""
    with open(OUT_HTML, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"[OK] Built {OUT_HTML} ({len(html):,} bytes)")


if __name__ == "__main__":
    build_html()
