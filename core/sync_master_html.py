"""Inject v1.4 Adaptive Volume (Bayesian-Kelly + EMA Risk) & Per-Setup Optimal Leg2 SL Tab into Master HTML."""
import json
import os
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HTML_PATH = os.path.join(ROOT, "core", "reports", "chicktrade_6coin_master_report.html")
RULES_PATH = os.path.join(ROOT, "core", "core_strategy_rules.json")
VOL_CSV = os.path.join(ROOT, "core", "reports", "adaptive_volume_matrix.csv")
SL_CSV = os.path.join(ROOT, "core", "reports", "leg2_sl_whipsaw_matrix.csv")


def main():
    with open(HTML_PATH, encoding="utf-8") as f:
        html = f.read()

    # Prevent duplicate injection if re-run
    if 'id="tab-vol-matrix"' in html:
        print("Tab tab-vol-matrix already present; stripping old injection first.")
        # Cleanly remove previous block between markers
        s_btn = "<!-- VOL_TAB_BTN_START -->"
        e_btn = "<!-- VOL_TAB_BTN_END -->"
        s_div = "<!-- VOL_TAB_DIV_START -->"
        e_div = "<!-- VOL_TAB_DIV_END -->"
        if s_btn in html and e_btn in html:
            html = html[: html.index(s_btn)] + html[html.index(e_btn) + len(e_btn) :]
        if s_div in html and e_div in html:
            html = html[: html.index(s_div)] + html[html.index(e_div) + len(e_div) :]

    with open(RULES_PATH, encoding="utf-8") as f:
        core_cfg = json.load(f)

    vol_df = pd.read_csv(VOL_CSV)
    sl_df = pd.read_csv(SL_CSV)

    # Merge the 33 setups by (Coin, TF, Setup)
    merged = pd.merge(
        sl_df,
        vol_df[["Coin", "TF", "Setup", "SL_Full_%", "Kelly_Full_%", "Vol_Init", "Vol_Hot(+5W)", "Vol_Cold(-3SL)", "Tier"]],
        on=["Coin", "TF", "Setup"],
        how="left",
    )
    ema_mult_map = core_cfg.get("ema_risk_multipliers", {"EMA34": 1.0, "EMA89": 0.95, "EMA200": 0.90})

    def _ema_from_setup(s: str) -> str:
        return s.split("_")[0]

    rows_html = []
    for _, r in merged.iterrows():
        ema = _ema_from_setup(str(r["Setup"]))
        emult = ema_mult_map.get(ema, 1.0)
        tier = str(r["Tier"])
        is_scout = "Scout" in tier
        badge_cls = "badge-warn" if is_scout else ("badge-good" if "S+" in tier or "A+" in tier else "badge-info")
        rows_html.append(
            f"""
            <tr>
              <td><strong>{r['Coin']}</strong></td>
              <td><span class="tf-pill tf-{r['TF']}">{r['TF']}</span></td>
              <td><code>{r['Setup']}</code></td>
              <td>{int(r['Lenh'])}</td>
              <td>{r['TP1']} / {r['TP2']}</td>
              <td><strong style="color:#38bdf8;">{r['Doi_SL_Toi_Uu']}</strong></td>
              <td style="color:#f87171;">{r['Quet_Neu_BEPhi_%']:.1f}%</td>
              <td style="color:#4ade80;"><strong>{r['Quet_Sau_Toi_Uu_%']:.1f}%</strong></td>
              <td><strong>{r['Cham_TP2_%']:.1f}%</strong></td>
              <td style="color:#4ade80;"><strong>{r['WR_Bao_Toan_%']:.2f}%</strong></td>
              <td>{r['SL_Full_%']:.1f}%</td>
              <td><strong>{r['PF']:.2f}</strong></td>
              <td>{r['Kelly_Full_%']:.1f}%</td>
              <td>{emult:.2f}x</td>
              <td><strong style="color:#fbbf24;font-size:14px;">{r['Vol_Init']}</strong></td>
              <td style="color:#4ade80;">{r['Vol_Hot(+5W)']}</td>
              <td style="color:#f87171;">{r['Vol_Cold(-3SL)']}</td>
              <td><span class="{badge_cls}">{tier}</span></td>
            </tr>
            """
        )

    tab_btn = (
        '<!-- VOL_TAB_BTN_START -->'
        '<button class="tab-btn" onclick="openTab(\'tab-vol-matrix\', this)">📊 Volume &amp; Dời SL (v1.4)</button>'
        '<!-- VOL_TAB_BTN_END -->'
    )

    tab_div = f"""<!-- VOL_TAB_DIV_START -->
    <div id="tab-vol-matrix" class="tab-content">
      <div class="coin-banner">
        <div>
          <h2>📊 Ma Trận Volume Toán Học (Bayesian-Kelly + Rủi Ro EMA) &amp; Dời SL Chân 2 Tối Ưu Riêng Từng Setup (v1.4)</h2>
          <p class="rationale-text">
            <strong>Cấu hình Vô Địch Pareto Chốt Cuối Cùng:</strong> 🏆 <strong>Bao 400 giá (Max 5 lệnh đồng thời)</strong> · Bỏ hẳn 16 setup rủi ro cao · Giữ đúng <strong>33 Setup (25 Core + 8 Scout hiếm 0.45×)</strong> · Calmar = <strong>40.91</strong> · Lãi 12T: <strong>+359.97% (Bậc 20/20)</strong> · MaxDD: <strong>-8.80%</strong> (OOS 6T: <strong>-6.45%</strong>).<br>
            <strong>Hệ số rủi ro đường EMA:</strong> <code>EMA34 = 1.00×</code> · <code>EMA89 = 0.95×</code> · <code>EMA200 = 0.90×</code> |
            <strong>Luật số lệnh theo Bao giá:</strong> <code>250→2</code> · <code>300→3</code> · <code>400→5</code> · <code>500→7</code>.
          </p>
        </div>
      </div>

      <div class="kpi-grid">
        <div class="kpi-card">
          <span>Tỷ Lệ Chạm TP2 (Dời SL Tối Ưu vs Entry+Phí)</span>
          <strong style="color:#4ade80;">40.41% vs 28.28%</strong>
          <small>Cứu thêm +100 lệnh ăn trọn TP2 (+239.90U = +27.1%)</small>
        </div>
        <div class="kpi-card">
          <span>Winrate Bảo Toàn Tối Đa (33 Setups)</span>
          <strong style="color:#4ade80;">75.73%</strong>
          <small>0% giảm Winrate do rò rỉ phí sàn ở chân 2</small>
        </div>
        <div class="kpi-card">
          <span>Cơ Chế Tự Học Volume (Bayesian-Kelly)</span>
          <strong style="color:#fbbf24;">0.40× &rarr; 1.30×</strong>
          <small>Tăng max 1.30× khi WR&ge;76% &amp; SL&le;22%; tự PAUSE (0.00×) nếu WR&lt;60%</small>
        </div>
        <div class="kpi-card">
          <span>Tổng Lãi Ròng 33 Setup (1 Năm)</span>
          <strong style="color:#38bdf8;">+1,124.17 U</strong>
          <small>So với +884.27 U nếu dời cứng Entry+Phí</small>
        </div>
      </div>

      <h2>Bảng Chi Tiết Tỷ Lệ Volume &amp; Mức Dời SL Chân 2 Tối Ưu Cho Toàn Bộ 33 Setups</h2>
      <div style="overflow-x:auto;">
        <table class="tbl">
          <thead>
            <tr>
              <th>Coin</th>
              <th>Khung</th>
              <th>Setup</th>
              <th>Lệnh 1Y</th>
              <th>TP1 / TP2</th>
              <th>Dời SL Chân 2 Tối Ưu</th>
              <th>Quét nếu BE+Phí</th>
              <th>Quét Sau Tối Ưu</th>
              <th>Chạm TP2</th>
              <th>WR Bảo Toàn</th>
              <th>SL Full %</th>
              <th>PF</th>
              <th>Kelly Full</th>
              <th>HS EMA</th>
              <th>Vol Khởi Điểm</th>
              <th>Vol Nóng (+5W)</th>
              <th>Vol Lạnh (-3SL)</th>
              <th>Phân Hạng</th>
            </tr>
          </thead>
          <tbody>
            {''.join(rows_html)}
          </tbody>
        </table>
      </div>
    </div>
    <!-- VOL_TAB_DIV_END -->"""

    # Inject button right after the Overview tab button
    overview_btn_anchor = "openTab('tab-overview', this)"
    pos_b = html.find(overview_btn_anchor)
    if pos_b == -1:
        raise RuntimeError("Could not find Overview tab button anchor in HTML")
    close_btn_pos = html.find("</button>", pos_b) + len("</button>")
    html = html[:close_btn_pos] + "\n      " + tab_btn + html[close_btn_pos:]

    # Inject tab content div right before <div id="tab-BTCUSDT"
    btc_tab_anchor = '<div id="tab-BTCUSDT"'
    pos_d = html.find(btc_tab_anchor)
    if pos_d == -1:
        raise RuntimeError("Could not find tab-BTCUSDT anchor in HTML")
    html = html[:pos_d] + tab_div + "\n\n    " + html[pos_d:]

    # Update stale text in Overview method box to reflect v1.4 accurately
    html = html.replace("Chỉ giữ <strong>25 setup tinh nhuệ</strong>", "Chốt <strong>33 setup (25 Core + 8 Scout 0.45×)</strong>")
    html = html.replace("Tối đa <code>3 lệnh mở cùng lúc</code> (tối đa 1 lệnh/coin, tổng hệ số mở &le; 2.40x)", "Số lệnh mở theo Bao giá: <code>250&rarr;2 | 300&rarr;3 | 400&rarr;5 (Chốt Pareto) | 500&rarr;7</code>")

    with open(HTML_PATH, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"[DONE] Synced v1.4 Volume & Leg2 SL Matrix into {HTML_PATH} ({len(merged)} setups).")


if __name__ == "__main__":
    main()
