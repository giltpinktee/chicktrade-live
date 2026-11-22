"""Build CHICKTRADE v2.3 Master HTML Report — Deep Slate-Navy Fintech Edition (75 Active Setups, Profit WR + 4-Branch %, Interactive 20-Step Calculator, Zero Horizontal Scrollbar).

Strictly writes ONLY to:
  - e:\CHICKTRADE\core\reports\chicktrade_6coin_master_report.html
Reads from:
  - e:\CHICKTRADE\core\core_strategy_rules.json (v2.3-final-75setups-rog25)
  - e:\CHICKTRADE\coins\<SYMBOL>\audit_48.csv
  - e:\CHICKTRADE\coins\<SYMBOL>\data\<SYMBOL>_5m_1y.csv.gz
"""
import json
import os
from typing import Dict, List, Tuple

import pandas as pd

CORE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(CORE_DIR)
REPORTS_DIR = os.path.join(CORE_DIR, "reports")
os.makedirs(REPORTS_DIR, exist_ok=True)

SIX_COINS = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "LINKUSDT", "XRPUSDT", "SOLUSDT"]
COIN_META = {
    "BTCUSDT": {
        "icon": "🪙",
        "short": "BTC",
        "sl_u": 5.0,
        "title": "BTCUSDT — Anh Cả Dẫn Sóng (Sóng Thuận Dài, Râu Ngắn)",
        "sl_tp": "SL 5.0U · TP1 4.0U (15m/30m) & 3.5U (1h) · TP2 12.0U",
        "summary": "BTC có biên độ râu nến quét ngược rất gọn (trung vị chỉ 2.6U–3.1U) nên SL 5.0U đã nằm ngoài vùng nhiễu. Khi vào sóng thuận EMA 34/89, BTC đi rất xa nên TP2 đặt ở 12.0U giúp ăn trọn sóng lớn.",
    },
    "ETHUSDT": {
        "icon": "💎",
        "short": "ETH",
        "sl_u": 6.0,
        "title": "ETHUSDT — Chuyên Gia Quét Râu Rồi Chạy & Đảo Chiều 1h",
        "sl_tp": "SL 6.0U · TP1 3.5U · TP2 10.0U",
        "summary": "ETH thường quét râu thanh khoản sâu trước khi chạy thật nên dùng SL 6.0U để tránh bị cắt lỗ oan. Kết hợp chốt 100% ở 1R (6U) trên khung 30m/1h và giữ SL gốc trên Setup Đảo chiều 1h EMA34C4 (PF 2.90, Thắng Tiền 74.4%).",
    },
    "BNBUSDT": {
        "icon": "🔶",
        "short": "BNB",
        "sl_u": 6.0,
        "title": "BNBUSDT — Đảo Chiều Cạn Sóng 4 & Đỡ Giá EMA200 (1h Chỉ Long)",
        "sl_tp": "SL 6.0U · TP1 4.0U (15m) & 3.5U (30m/1h) · TP2 10.0U",
        "summary": "BNB vận động biên rộng rất tôn trọng EMA 89/200 và đảo chiều chuẩn ở nhịp Chạm 4 (C4). Setup 1h EMA200C1 khóa CHỈ ĐÁNH LONG (Luật 1B) vì lực cầu nắm giữ BNB đỡ giá mạnh tại EMA200.",
    },
    "LINKUSDT": {
        "icon": "🔗",
        "short": "LINK",
        "sl_u": 6.0,
        "title": "LINKUSDT — Tỷ Lệ Thắng Đỉnh Cao Ngay Nhịp Chạm Đầu Tiên (C1)",
        "sl_tp": "SL 6.0U · TP1 3.5U · TP2 8.0U",
        "summary": "LINK bật rất nhạy ngay lần chạm đầu tiên (C1) vào EMA 34/89 khung 15m và EMA 200 khung 1h, cùng Setup 5m EMA89C2. Chốt TP2 ở 8.0U giúp tỷ lệ ăn trọn 2 nhịp đạt tối đa.",
    },
    "XRPUSDT": {
        "icon": "⚡",
        "short": "XRP",
        "sl_u": 6.0,
        "title": "XRPUSDT — Bùng Nổ Nhịp Sóng 3 & Bất Đối Xứng 1h (1h EMA34C2 Chỉ Short)",
        "sl_tp": "SL 6.0U · TP1 3.5U · TP2 8.0U",
        "summary": "XRP nén chặt rồi bùng nổ ở nhịp Chạm 3 (15m EMA34C3) và tôn trọng EMA 200 khung 1h. Setup 1h EMA34C2 khóa CHỈ ĐÁNH SHORT (Luật 1B) với PF 6T đầu 3.75, 6T cuối 1.57.",
    },
    "SOLUSDT": {
        "icon": "☀️",
        "short": "SOL",
        "sl_u": 6.0,
        "title": "SOLUSDT — Biến Động Mạnh Chọn Lọc Tinh Nhuệ (Đánh Nhanh Thắng Gọn)",
        "sl_tp": "SL 6.0U · TP1 3.5U · TP2 8.0U",
        "summary": "SOL có biên độ dao động nhanh nên hệ thống tập trung vào các nhịp hồi rõ xu hướng (15m EMA89C2, 30m EMA34C1) và các Setup Sniper/Mở rộng đảo chiều có Thắng Tiền (Profit WR) vượt trội.",
    },
}


# ============================================================================
# LIGHT MODE FINTECH (UNSLOP) 2-LAYER COLOR ARCHITECTURE HELPERS (v2.3)
# Layer A: Game RPG Tier Colors (Rank #, Col 1 S+ tag, Col 6 Vol, Col 7 Pill, Legend)
# Layer B: Financial PnL Colors (Col 4 Thắng tiền % & Net U, Col 5 PF, Tab 1 KPIs)
# ============================================================================
def color_for_pwr(pwr: float) -> str:
    """Layer B — Financial PnL color scale for Profit-Weighted Winrate (Thắng Tiền %):
    - >= 65.0% (High profit): Emerald Dark #059669
    - 50.0% - 64.9% (Solid / Moderate): Amber Dark #B45309
    - < 50.0% (Loss / Weak): Red Dark #DC2626
    """
    if pwr >= 65.0:
        return "#059669"
    if pwr >= 50.0:
        return "#B45309"
    return "#DC2626"


def color_for_pf(pf: float) -> str:
    """Layer B — Financial PnL color scale for Profit Factor (PF):
    - >= 2.00: Emerald Dark #059669
    - 1.20 - 1.99: Amber Dark #B45309
    - 1.00 - 1.19: Slate Dark #475569
    - < 1.00: Red Dark #DC2626
    """
    if pf >= 2.0:
        return "#059669"
    if pf >= 1.2:
        return "#B45309"
    if pf >= 1.0:
        return "#475569"
    return "#DC2626"


def get_unified_row_tier(r: dict, is_top_must: bool = False) -> Tuple[str, str, str, str]:
    """Layer A — Volume Tier Scale (tier_key, tier_text_col, pill_cls, tier_label) for Col 6 & Col 7:
    - S+ (Vol >= 1.10x): Burnt Orange #C2410C / bg #FFEDD5
    - S  (Vol = 1.00x - 1.09x): Royal Purple #6D28D9 / bg #F3E8FF
    - A  (Vol 0.50x - 0.99x): Deep Blue #0369A1 / bg #E0F2FE
    - B  (Vol 0.45x - 0.49x): Teal #0F766E / bg #F0FDFA
    - C  (Vol <= 0.40x): Slate #475569 / bg #F1F5F9
    - D  (SKIP / Vol 0): Rose #BE123C / bg #FFF1F2
    """
    status = str(r.get("status", ""))
    vol = float(r.get("adaptive_volume", {}).get("init_vol_mult", 0.0))

    if status == "SKIP" or vol <= 0.0:
        return ("D", "#BE123C", "st-d", "Không vào")
    if vol >= 1.10:
        return ("S+", "#C2410C", "st-sp", "Bậc S+")
    if vol >= 1.00:
        return ("S", "#6D28D9", "st-s", "Bậc S")
    if vol >= 0.50:
        return ("A", "#0369A1", "st-a", "Bậc A")
    if vol >= 0.45:
        return ("B", "#0F766E", "st-b", "Bậc B")
    return ("C", "#475569", "st-c", "Bậc C")


def get_status_pill_meta(status: str) -> Tuple[str, str]:
    """Return (status_label, tooltip_title) for Column 7 single clean status pill."""
    if status == "CORE":
        return (
            "ĐÁNH CHÍNH",
            "Setup chủ lực (Core) đủ mẫu >= 15 lệnh/năm và thắng bền vững ở cả 2 nửa năm",
        )
    if status == "CORE_EXP":
        return (
            "MỞ RỘNG",
            "Setup mở rộng bền cả 2 nửa năm (Thắng Tiền > 58%), đánh hệ số 0.75x",
        )
    if status == "SCOUT_EXP":
        return (
            "MỞ RỘNG",
            "Setup mở rộng thăm dò bền cả 2 nửa năm, đánh hệ số 0.45x",
        )
    if status == "SNIPER":
        return (
            "SNIPER",
            "Setup Sniper hiếm lệnh (3–6 lệnh/năm) nhưng Thắng Tiền cao, đánh 0.45x",
        )
    if status == "SCOUT":
        return (
            "SCOUT",
            "Setup hiếm (7–14 lệnh/năm) nhưng Thắng Tiền cao ở cả 2 nửa năm, đánh thăm dò 0.35x–0.45x",
        )
    if status == "HOT_ADAPTIVE":
        return (
            "THÍCH ỨNG",
            "Setup thích ứng bùng nổ ở 6 tháng cuối (PF OOS >= 1.80), chỉ đánh nhỏ 0.40x và không kích hoạt đóng băng danh mục",
        )
    return ("BỎ QUA", "Setup không đạt chuẩn vào tiền")


def format_short_setup_code(setup_code: str) -> Tuple[str, str]:
    """Convert 'EMA34_W1' -> ('EMA34C1', 'Chạm 1 · Khởi sóng thuận'), 'EMA200_W4_REV' -> ('EMA200C4', 'Chạm 4 · Đảo chiều cạn sóng')."""
    sc = str(setup_code).strip()
    parts = sc.split("_")
    ema_part = parts[0] if parts else "EMA34"
    w_num = "1"
    for p in parts:
        if p.startswith("W") and len(p) >= 2 and p[1].isdigit():
            w_num = p[1]
            break
    short_code = f"{ema_part}C{w_num}"
    desc_map = {
        "1": "Chạm 1 · Khởi sóng thuận",
        "2": "Chạm 2 · Tiếp diễn thuận",
        "3": "Chạm 3 · Chín muồi thuận",
        "4": "Chạm 4 · Đảo chiều cạn sóng",
    }
    return short_code, desc_map.get(w_num, f"Chạm {w_num}")


def compute_1u_usd_reference_table() -> pd.DataFrame:
    """Compute 30-day median 1U = 0.5 * ATR(14) across 6 coins and active timeframes (15m/30m/1h + 5m for LINK)
    for a $10,000 account at Bao 400G (1.00x base volume -> 1U = $25.00 USD risk unit).
    """
    rows = []
    base_cap = 10_000.0
    buf_g = 400.0
    usd_per_1u = base_cap / buf_g  # $25.00 per 1U at 1.00x Volume

    tf_resample_map = {"5m": None, "15m": "15min", "30m": "30min", "1h": "1h"}

    for sym in SIX_COINS:
        gz_path = os.path.join(ROOT, "coins", sym, "data", f"{sym}_5m_1y.csv.gz")
        if not os.path.exists(gz_path):
            continue
        df5 = pd.read_csv(gz_path)
        df5["timestamp"] = pd.to_datetime(df5["timestamp"], utc=True)
        df5 = df5.sort_values("timestamp").reset_index(drop=True)

        cutoff_45d = df5["timestamp"].max() - pd.Timedelta(days=45)
        cutoff_30d = df5["timestamp"].max() - pd.Timedelta(days=30)
        sub5 = df5[df5["timestamp"] >= cutoff_45d].copy()

        tfs_to_calc = ["5m", "15m", "30m", "1h"] if sym == "LINKUSDT" else ["15m", "30m", "1h"]
        tf_stats = {}

        for tfl in tfs_to_calc:
            rule_str = tf_resample_map[tfl]
            if rule_str is None:
                d = sub5.copy()
            else:
                d = (
                    sub5.set_index("timestamp")
                    .resample(rule_str)
                    .agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
                    .dropna()
                    .reset_index()
                )
            high = d["high"]
            low = d["low"]
            close = d["close"]
            tr = pd.concat([high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()], axis=1).max(axis=1)
            atr14 = tr.ewm(alpha=1.0 / 14, adjust=False).mean()
            d["u_price"] = 0.5 * atr14
            d30 = d[d["timestamp"] >= cutoff_30d]
            med_close = float(d30["close"].median())
            med_u = float(d30["u_price"].median())
            u_pct = (med_u / med_close) * 100.0
            qty_coin = usd_per_1u / med_u
            notional_usd = qty_coin * med_close
            tf_stats[tfl] = {
                "price": med_close,
                "u_price": med_u,
                "u_pct": u_pct,
                "qty": qty_coin,
                "notional": notional_usd,
            }

        sl_u = COIN_META[sym]["sl_u"]
        sl_usd = sl_u * usd_per_1u
        rows.append({
            "sym": sym,
            "price": tf_stats["15m"]["price"],
            "sl_u": sl_u,
            "usd_per_1u": usd_per_1u,
            "sl_usd": sl_usd,
            "tf_stats": tf_stats,
        })
    return pd.DataFrame(rows)


def format_price_smart(val: float) -> str:
    if val >= 1000:
        return f"${val:,.1f}"
    if val >= 10:
        return f"${val:,.2f}"
    if val >= 1:
        return f"${val:,.3f}"
    return f"${val:,.4f}"


def format_qty_smart(qty: float, sym: str) -> str:
    coin_unit = sym.replace("USDT", "")
    if qty >= 100:
        return f"{qty:,.0f} {coin_unit}"
    if qty >= 10:
        return f"{qty:,.1f} {coin_unit}"
    if qty >= 1:
        return f"{qty:,.2f} {coin_unit}"
    return f"{qty:,.3f} {coin_unit}"


def render_1u_conversion_table_html(conv_df: pd.DataFrame) -> str:
    """Render compact 6-column reference table converting 1U -> USD & Coin Qty at $10,000 capital (Bao 400G)."""
    rows_html = []
    for _, r in conv_df.iterrows():
        sym = r["sym"]
        meta = COIN_META[sym]
        p_str = format_price_smart(r["price"])
        sl_u = int(r["sl_u"])
        sl_usd = r["sl_usd"]
        tfs = r["tf_stats"]

        def _fmt_tf_cell(tfl: str) -> str:
            st = tfs[tfl]
            u_str = format_price_smart(st["u_price"])
            q_str = format_qty_smart(st["qty"], sym)
            notional_str = f"${st['notional']:,.0f}"
            return (
                f'<div class="cell-main num">1U = <span style="color:#0369A1;">{u_str}</span> <span style="color:#64748B;font-size:11px;">({st["u_pct"]:.2f}%)</span></div>'
                f'<div class="cell-sub num">Vào: <strong>{q_str}</strong> (~{notional_str})</div>'
            )

        extra_5m = ""
        if "5m" in tfs:
            st5 = tfs["5m"]
            extra_5m = f'<div class="cell-sub num" style="margin-top:3px;color:#B45309;">[Setup 5m LINK]: 1U = {format_price_smart(st5["u_price"])} ({st5["u_pct"]:.2f}%) · Vào {format_qty_smart(st5["qty"], sym)} (~${st5["notional"]:,.0f})</div>'

        rows_html.append(
            f"""<tr class="row-active">
                <td>
                    <div class="cell-main">{meta['icon']} {sym}</div>
                    <div class="cell-sub num">Giá TB 30 ngày: <strong>{p_str}</strong></div>
                </td>
                <td>
                    <div class="cell-main num" style="color:#059669;">1U = $25.00 USD</div>
                    <div class="cell-sub num">Vốn $10,000 &divide; Bao 400 giá (Vol 1.0&times;)</div>
                </td>
                <td>
                    <div class="cell-main num" style="color:#DC2626;">SL {sl_u}U = -${sl_usd:,.0f} USD</div>
                    <div class="cell-sub num">Rủi ro <strong>-{sl_usd/100:.2f}%</strong> vốn gốc / lệnh 1.0&times;</div>
                </td>
                <td>{_fmt_tf_cell('15m')}{extra_5m}</td>
                <td>{_fmt_tf_cell('30m')}</td>
                <td>{_fmt_tf_cell('1h')}</td>
            </tr>"""
        )

    return f"""
    <table class="compact-tbl">
        <colgroup>
            <col style="width:14%">
            <col style="width:15%">
            <col style="width:16%">
            <col style="width:19%">
            <col style="width:18%">
            <col style="width:18%">
        </colgroup>
        <thead>
            <tr>
                <th title="Đồng Coin và mức giá trung vị 30 ngày gần nhất trong dữ liệu">1. Đồng Coin &amp; Giá</th>
                <th title="Tại vốn $10,000 và Bao 400 giá (Vol 1.0x), mỗi 1U giá chạy tương đương đúng $25.00 USD tiền thật">2. Giá Trị 1U Bằng Tiền USD</th>
                <th title="Số tiền USD mất tối đa nếu lệnh dính Full SL (5U cho BTC = $125, 6U cho các coin khác = $150 tại Vol 1.0x)">3. Rủi Ro Cắt Lỗ (SL &rarr; USD)</th>
                <th title="Biên độ 1U (0.5 x ATR14) và Khối lượng vào lệnh thực tế trên khung 15m (kèm 5m cho LINK)">4. Khung 15m (1U &amp; Vol Lệnh)</th>
                <th title="Biên độ 1U (0.5 x ATR14) và Khối lượng vào lệnh thực tế trên khung 30m">5. Khung 30m (1U &amp; Vol Lệnh)</th>
                <th title="Biên độ 1U (0.5 x ATR14) và Khối lượng vào lệnh thực tế trên khung 1h">6. Khung 1h (1U &amp; Vol Lệnh)</th>
            </tr>
        </thead>
        <tbody>
            {''.join(rows_html)}
        </tbody>
    </table>
    """


def format_exit_vn(exit_raw: str, sl_u: float, tp1_u: float, tp2_u: float, leg2_label: str = "") -> Tuple[str, str]:
    """Return (Line 1 plain Vietnamese exit style, Line 2 exact SL/TP parameters with .nw guards)."""
    clean = str(exit_raw).replace("SKIP (", "").replace(")", "").strip()
    sl_s = f'<span class="nw">SL <b>{sl_u:.0f}U</b></span>'
    tp1_s = f'<span class="nw">TP1 <b>{tp1_u:.1f}U</b></span>'
    tp2_s = f'<span class="nw">TP2 <b>{tp2_u:.0f}U</b></span>'
    if clean == "ONLY_TP1":
        return "Chốt 100% tại TP1 (Ăn nhanh)", f'{sl_s} · <span class="nw">Chốt hết ở TP1 <b>{tp1_u:.1f}U</b></span>'
    if clean == "ONLY_1R":
        return "Chốt 100% tại 1R (Tỷ lệ 1:1)", f'{sl_s} · <span class="nw">Chốt hết ở <b>+{sl_u:.0f}U (1R)</b></span>'
    if clean == "PARTIAL_NO_MOVE":
        return "Chia 2 nhịp (50/50) · Giữ SL gốc", f'{sl_s} · {tp1_s} · {tp2_s} · <span class="nw">(Không dời SL)</span>'
    if "Entry-2" in clean or "-2" in leg2_label:
        return "Chia 2 nhịp (50/50) · Dời SL -2U", f'{sl_s} · {tp1_s} · {tp2_s} · <span class="nw">Chân 2: <b>Entry-2U</b></span>'
    if "Entry-1" in clean or "-1" in leg2_label:
        return "Chia 2 nhịp (50/50) · Dời SL -1U", f'{sl_s} · {tp1_s} · {tp2_s} · <span class="nw">Chân 2: <b>Entry-1U</b></span>'
    if "Entry0U" in clean or "0U" in leg2_label:
        return "Chia 2 nhịp (50/50) · Dời về Entry", f'{sl_s} · {tp1_s} · {tp2_s} · <span class="nw">Chân 2: <b>Hòa vốn 0U</b></span>'
    if "BE+Phi" in clean:
        return "Chia 2 nhịp (50/50) · Dời Entry+Phí", f'{sl_s} · {tp1_s} · {tp2_s} · <span class="nw">Chân 2: <b>Entry+Phí</b></span>'
    return f"Chia 2 nhịp ({clean})", f'{sl_s} · {tp1_s} · {tp2_s} · <span class="nw">{leg2_label}</span>'


def explain_skip_reason_vn(row: pd.Series) -> str:
    """Return concise plain-Vietnamese reason why a setup in audit_48.csv is not traded."""
    tf = str(row["TF"])
    n = int(row["Lenh_1Y"])
    pf_is = float(row["PF_IS"])
    pf_oos = float(row["PF_OOS"])
    pf_1y = float(row["PF_1Y"])
    net_u = float(row["Net_U_1Y"])
    if tf == "5m":
        if pf_1y < 1.0 or net_u <= 0:
            return f"Phí sàn 5m bào mòn ({net_u:+.1f}U)"
        if pf_is < 1.15 or pf_oos < 1.15:
            return "Nhiễu 5m, không bền 2 nửa năm"
        return "Loại khung 5m chống trượt giá"
    if n < 3:
        return f"Quá ít lệnh ({n} lệnh/năm)"
    if pf_is < 1.0 and pf_oos < 1.0:
        return f"Thắng Tiền < 50% (PF {pf_1y:.2f})"
    if pf_is < 1.05:
        return f"Yếu 6T đầu (PF đầu {pf_is:.2f})"
    if pf_oos < 1.05:
        return f"Gãy 6T cuối (PF cuối {pf_oos:.2f})"
    return f"Biên lãi mỏng (PF {pf_1y:.2f})"


def load_all_v23_data() -> Tuple[dict, List[dict], Dict[str, pd.DataFrame]]:
    with open(os.path.join(CORE_DIR, "core_strategy_rules.json"), encoding="utf-8") as f:
        core_cfg = json.load(f)

    all_75_rules: List[dict] = list(core_cfg["rules"])

    coin_audits: Dict[str, pd.DataFrame] = {}
    for sym in SIX_COINS:
        acsv = os.path.join(ROOT, "coins", sym, "audit_48.csv")
        adf = pd.read_csv(acsv)
        coin_audits[sym] = adf

    return core_cfg, all_75_rules, coin_audits


def render_direction_badge(dir_str: str) -> str:
    if dir_str == "LONG_ONLY":
        return '<span class="dir-badge dir-long" title="Luật 1B: Setup này chỉ vào lệnh chiều LONG (Chiều Short không hiệu quả)">Chỉ LONG</span>'
    if dir_str == "SHORT_ONLY":
        return '<span class="dir-badge dir-short" title="Luật 1B: Setup này chỉ vào lệnh chiều SHORT (Chiều Long không hiệu quả)">Chỉ SHORT</span>'
    return '<span class="dir-text nw">Thuận 2 chiều</span>'


def render_tier_badge(r: dict, is_top_must: bool = False) -> Tuple[str, str, str, str]:
    """Return (tier_key, tier_col, Vol cell HTML, Status pill HTML) aligned with unified row tier."""
    status = str(r.get("status", ""))
    vol = float(r.get("adaptive_volume", {}).get("init_vol_mult", 0.0))
    tier_key, tier_col, pill_cls, tier_label = get_unified_row_tier(r, is_top_must=is_top_must)
    status_label, status_title = get_status_pill_meta(status)
    vol_html = (
        f'<div class="cell-main num" style="color:{tier_col};">{vol:.2f}&times;</div>'
        f'<div class="cell-sub"><span class="nw">{tier_label}</span></div>'
    )
    st_html = f'<span class="status-pill {pill_cls}" title="{status_title}">{status_label}</span>'
    return tier_key, tier_col, vol_html, st_html


def get_profit_wr(p: dict) -> float:
    if "profit_wr_pct" in p and p["profit_wr_pct"] is not None:
        return float(p["profit_wr_pct"])
    pf = float(p.get("pf_1y", 0.0))
    return (pf / (pf + 1.0)) * 100.0 if pf > 0 else 0.0


def compute_must_order_score(r: dict) -> float:
    """Compute composite 'MUST ORDER — ĐỤNG LÀ QUẤT' score combining Profit WR (Thắng Tiền %) and Net U, prioritizing CORE setups."""
    p = r["prior_1y"]
    net_u = float(p["net_u"])
    pwr = get_profit_wr(p)
    pf = float(p["pf_1y"])
    st = r["status"]
    status_bonus = 24.0 if st == "CORE" else (12.0 if st == "CORE_EXP" else (5.0 if st in ("SCOUT", "HOT_ADAPTIVE", "SCOUT_EXP") else 0.0))
    return status_bonus + (net_u / 90.0) * 55.0 + (pwr / 100.0) * 45.0 + min(pf, 3.5) * 2.0


def compute_top_must_keys(all_rules: List[dict]) -> set:
    """Identify global S+ top-must setup keys consistently across Tab 2 and per-coin tabs."""
    sorted_all = sorted(all_rules, key=lambda r: -compute_must_order_score(r))
    keys = set()
    for idx, r in enumerate(sorted_all, start=1):
        p = r["prior_1y"]
        pwr_val = get_profit_wr(p)
        net_u_val = float(p["net_u"])
        if r["status"] in ("CORE", "CORE_EXP") and (idx <= 14 or (pwr_val >= 67.0 and net_u_val >= 34.0)):
            keys.add((r["coin"], r["tf"], r["setup_code"]))
    return keys


def build_75_rules_table_html(
    rules_list: List[dict],
    table_id: str = "tbl-75-all",
    top_must_keys: set = None,
) -> str:
    """Render 7-column zero-horizontal-scroll table for active setups, sorted by MUST ORDER (Profit WR + Net U)."""
    rows_html = []
    coin_ord_map = {c: i for i, c in enumerate(SIX_COINS)}
    sorted_rules = sorted(
        rules_list,
        key=lambda r: -compute_must_order_score(r),
    )
    if top_must_keys is None:
        top_must_keys = compute_top_must_keys(sorted_rules)

    for rank_idx, r in enumerate(sorted_rules, start=1):
        sym = r["coin"]
        tf = r["tf"]
        sc = str(r["setup_code"])
        short_code, touch_desc = format_short_setup_code(sc)
        p = r["prior_1y"]
        dir_mode = r.get("allowed_direction", "BOTH")
        must_score = compute_must_order_score(r)

        raw_ex = r.get("best_exit_raw") or (
            f"PARTIAL_Entry{int(r.get('leg2_sl_u', -2))}U"
            if r.get("exit_arch") == "PARTIAL_MOVE_SL"
            else r.get("exit_arch", "PARTIAL_Entry-2U")
        )
        ex_line1, ex_line2 = format_exit_vn(
            raw_ex, float(r["sl_u"]), float(r["tp1_u"]), float(r["tp2_u"]), r.get("leg2_sl_label", "")
        )

        wr_val = float(p["wr_pct"])
        pwr_val = get_profit_wr(p)
        net_u_val = float(p["net_u"])
        pf_val = float(p["pf_1y"])
        pf_is_val = float(p["pf_is"])
        pf_oos_val = float(p["pf_oos"])

        sl_pct = float(p.get("sl_full_pct", 0.0))
        l2_pct = float(p.get("leg2_sl_pct", 0.0))
        tp1_pct = float(p.get("tp1_hit_pct", 0.0))
        tp2_pct = float(p.get("tp2_hit_pct", 0.0))

        is_top_must = (sym, tf, r["setup_code"]) in top_must_keys
        tier_key, tier_col, vol_html, st_html = render_tier_badge(r, is_top_must=is_top_must)
        pwr_col = color_for_pwr(pwr_val)
        pf_col = color_for_pf(pf_val)
        net_col = "#15803D" if (is_top_must or net_u_val >= 35.0) else ("#059669" if net_u_val > 0 else "#DC2626")

        rank_cls = "rank-top" if is_top_must else "rank-normal"
        must_badge = (
            ' <span class="must-badge nw" title="Huyền thoại MMORPG — Top Siêu Setup THẤY LÀ QUẤT: Thắng Tiền (Profit WR) cao kết hợp kiếm nhiều tiền nhất trong năm">THẤY LÀ QUẤT</span>'
            if is_top_must
            else ""
        )

        is_col_style = ' style="color:#DC2626;"' if pf_is_val < 1.0 else ""
        oos_col_style = ' style="color:#DC2626;"' if pf_oos_val < 1.0 else ""
        st_group = "EXP" if r["status"] in ("CORE_EXP", "SCOUT_EXP") else r["status"]

        rows_html.append(
            f"""<tr class="{'row-must' if is_top_must else 'row-active'}" data-coin="{sym}" data-coin-ord="{coin_ord_map.get(sym, 99)}" data-tf="{tf}" data-status="{r['status']}" data-group="{st_group}" data-must="{must_score:.4f}" data-netu="{net_u_val:.4f}" data-pwr="{pwr_val:.4f}" data-wr="{wr_val:.4f}">
                <td>
                    <div class="cell-main"><span class="rank-badge num {rank_cls}">#{rank_idx}</span> <span class="nw">{COIN_META[sym]['icon']} {sym}</span> <span class="tf-badge tf-{tf}">{tf}</span></div>
                    <div class="cell-sub">{render_direction_badge(dir_mode)}{must_badge}</div>
                </td>
                <td>
                    <div class="cell-main"><span class="code-pill">{short_code}</span> <span class="nw muted-meta">&middot; {int(p['trades'])} lệnh</span></div>
                    <div class="cell-sub">{touch_desc}</div>
                </td>
                <td>
                    <div class="cell-main">{ex_line1}</div>
                    <div class="cell-sub num">{ex_line2}</div>
                </td>
                <td>
                    <div class="cell-main num"><span class="nw" style="color:{pwr_col};">Thắng tiền {pwr_val:.1f}%</span> <span class="nw" style="color:{net_col};">(+{net_u_val:.1f}U)</span> <span class="nw muted-meta">&middot; WR {wr_val:.0f}%</span></div>
                    <div class="cell-sub num"><span class="nw">SL <b>{sl_pct:.0f}%</b></span> &middot; <span class="nw">DờiSL <b>{l2_pct:.0f}%</b></span> &middot; <span class="nw">TP1 <b>{tp1_pct:.0f}%</b></span> &middot; <span class="nw">TP2 <b>{tp2_pct:.0f}%</b></span></div>
                </td>
                <td>
                    <div class="cell-main num" style="color:{pf_col};">PF {pf_val:.2f}</div>
                    <div class="cell-sub num"><span class="nw">6T đầu <b{is_col_style}>{pf_is_val:.2f}</b></span> &middot; <span class="nw">6T cuối <b{oos_col_style}>{pf_oos_val:.2f}</b></span></div>
                </td>
                <td style="text-align:center;">{vol_html}</td>
                <td style="text-align:center;">{st_html}</td>
            </tr>"""
        )

    return f"""
    <table class="compact-tbl" id="{table_id}">
        <colgroup>
            <col style="width:14%">
            <col style="width:13%">
            <col style="width:19%">
            <col style="width:24%">
            <col style="width:12%">
            <col style="width:9%">
            <col style="width:9%">
        </colgroup>
        <thead>
            <tr>
                <th title="Thứ hạng ưu tiên Đụng Là Quất, Tên đồng Coin, Khung giờ và Chiều giao dịch">1. Hạng &amp; Coin</th>
                <th title="Mã Setup rút gọn (VD: EMA34C1 = EMA 34 Chạm 1, EMA200C4 = EMA 200 Chạm 4 Đảo chiều) và Tổng số lệnh trong 1 năm">2. Mã Setup</th>
                <th title="Cách chốt lời tối ưu cho riêng Setup này (Chốt 100% hay Chia 2 nhịp 50/50 và vị trí dời SL chân 2)">3. Chốt Lời &amp; Dời SL</th>
                <th title="Winrate Lợi Nhuận (Thắng Tiền % = Tổng Lãi / (Tổng Lãi + Tổng Lỗ), >50% là chắc chắn có lãi), Tổng lãi +U, WR số lệnh và đủ 4 nhánh %: Dính SL gốc / DờiSL chân 2 / Chạm TP1 / Ăn trọn TP2">4. Thắng Tiền &amp; 4 Nhánh %</th>
                <th title="Hệ số Lãi/Lỗ cả năm (PF) và kiểm chứng riêng trên 6 tháng đầu (IS) & 6 tháng cuối (OOS)">5. Hệ Số Lãi/Lỗ (PF)</th>
                <th title="Hệ số nhân khối lượng khuyến nghị theo công thức Kelly và bậc chất lượng">6. Mức Volume</th>
                <th title="Phân loại Setup theo 5 Bậc Màu thống nhất: ĐÁNH CHÍNH, MỞ RỘNG, SNIPER, SCOUT, THÍCH ỨNG">7. Trạng Thái</th>
            </tr>
        </thead>
        <tbody>
            {''.join(rows_html)}
        </tbody>
    </table>
    """


def build_coin_48_matrix_html(
    sym: str,
    adf: pd.DataFrame,
    rules_by_key: Dict[Tuple[str, str, str], dict],
    top_must_keys: set = None,
) -> str:
    """Render full 48-row reference matrix (5m, 15m, 30m, 1h x EMA 34/89/200 x Touch 1..4) in 7 compact columns, 100% synced."""
    rows_html = []
    sl_default = 5.0 if sym == "BTCUSDT" else 6.0
    tp2_default = 12.0 if sym == "BTCUSDT" else (10.0 if sym in ("ETHUSDT", "BNBUSDT") else 8.0)
    if top_must_keys is None:
        top_must_keys = set()

    for _, row in adf.iterrows():
        tf = str(row["TF"])
        scode = str(row["Setup"])
        short_code, touch_desc = format_short_setup_code(scode)
        k = (sym, tf, scode)
        in_port = k in rules_by_key

        n_1y = int(row["Lenh_1Y"])
        n_l = int(row["Long_Lenh"])
        wr_l = float(row["Long_WR_%"])
        n_s = int(row["Short_Lenh"])
        wr_s = float(row["Short_WR_%"])
        wr_1y = float(row["WR_1Y_%"])
        pwr_1y = float(row["Profit_WR_%"]) if "Profit_WR_%" in row else 0.0
        pf_is = float(row["PF_IS"])
        pf_oos = float(row["PF_OOS"])
        pf_1y = float(row["PF_1Y"])
        net_u = float(row["Net_U_1Y"])
        sl_pct = float(row["SL_Full_%"])
        l2_pct = float(row["Dinh_DoiSL_%"])
        tp1_pct = float(row["TP1_Hit_%"])
        tp2_pct = float(row["TP2_Hit_%"])

        tp1_default = 4.0 if (sym == "BTCUSDT" and tf != "1h") or (sym == "BNBUSDT" and tf in ("5m", "15m")) else 3.5

        if in_port:
            r = rules_by_key[k]
            p = r["prior_1y"]
            dir_mode = r.get("allowed_direction", "BOTH")
            raw_ex = r.get("best_exit_raw") or (
                f"PARTIAL_Entry{int(r.get('leg2_sl_u', -2))}U"
                if r.get("exit_arch") == "PARTIAL_MOVE_SL"
                else r.get("exit_arch", "PARTIAL_Entry-2U")
            )
            ex_l1, ex_l2 = format_exit_vn(raw_ex, float(r["sl_u"]), float(r["tp1_u"]), float(r["tp2_u"]), r.get("leg2_sl_label", ""))
            pwr_1y = get_profit_wr(p)
            wr_1y = float(p["wr_pct"])
            net_u = float(p["net_u"])
            sl_pct = float(p.get("sl_full_pct", sl_pct))
            l2_pct = float(p.get("leg2_sl_pct", l2_pct))
            tp1_pct = float(p.get("tp1_hit_pct", tp1_pct))
            tp2_pct = float(p.get("tp2_hit_pct", tp2_pct))
            pf_is = float(p.get("pf_is", pf_is))
            pf_oos = float(p.get("pf_oos", pf_oos))
            pf_1y = float(p.get("pf_1y", pf_1y))

            is_top_must = k in top_must_keys
            tier_key, tier_col, vol_html, st_html = render_tier_badge(r, is_top_must=is_top_must)
            must_badge = ' <span class="must-badge nw">THẤY LÀ QUẤT</span>' if is_top_must else ""
            tr_cls = "row-must" if is_top_must else "row-active"
        else:
            dir_mode = "BOTH"
            is_top_must = False
            must_badge = ""
            ex_l1, ex_l2 = format_exit_vn(str(row["Best_Exit"]), sl_default, tp1_default, tp2_default)
            ex_l1 = f"(Thử tốt nhất: {ex_l1})"
            if pwr_1y <= 0 and pf_1y > 0:
                pwr_1y = (pf_1y / (pf_1y + 1.0)) * 100.0
            skip_r = {
                "status": "SKIP",
                "prior_1y": {"profit_wr_pct": pwr_1y, "net_u": net_u},
                "adaptive_volume": {"init_vol_mult": 0.0},
            }
            tier_key, tier_col, pill_cls, _ = get_unified_row_tier(skip_r, is_top_must=False)
            vol_html = '<div class="cell-main num" style="color:#64748B;">0.00&times;</div><div class="cell-sub"><span class="nw">Không vào</span></div>'
            reason_vn = explain_skip_reason_vn(row)
            st_html = f'<span class="status-pill {pill_cls}">BỎ QUA</span><div class="cell-sub" style="margin-top:3px;">{reason_vn}</div>'
            tr_cls = "row-m5-skip" if tf == "5m" else "row-skip"

        net_col = "#15803D" if (is_top_must or net_u >= 35.0) else ("#059669" if net_u > 0 else "#DC2626")
        pwr_col = color_for_pwr(pwr_1y)
        pf_col = color_for_pf(pf_1y)
        is_col_style = ' style="color:#DC2626;"' if pf_is < 1.0 else ""
        oos_col_style = ' style="color:#DC2626;"' if pf_oos < 1.0 else ""

        rows_html.append(
            f"""<tr class="{tr_cls}" data-tf="{tf}" data-picked="{'YES' if in_port else 'NO'}">
                <td>
                    <div class="cell-main"><span class="nw">{COIN_META[sym]['icon']} {sym}</span> <span class="tf-badge tf-{tf}">{tf}</span></div>
                    <div class="cell-sub">{render_direction_badge(dir_mode)}{must_badge}</div>
                </td>
                <td>
                    <div class="cell-main"><span class="code-pill">{short_code}</span> <span class="nw muted-meta">&middot; {n_1y} lệnh</span></div>
                    <div class="cell-sub num">{touch_desc} &middot; <span class="nw">L <b>{n_l}</b> ({wr_l:.0f}%)</span> &middot; <span class="nw">S <b>{n_s}</b> ({wr_s:.0f}%)</span></div>
                </td>
                <td>
                    <div class="cell-main">{ex_l1}</div>
                    <div class="cell-sub num">{ex_l2}</div>
                </td>
                <td>
                    <div class="cell-main num"><span class="nw" style="color:{pwr_col};">Thắng tiền {pwr_1y:.1f}%</span> <span class="nw" style="color:{net_col};">({net_u:+.1f}U)</span> <span class="nw muted-meta">&middot; WR {wr_1y:.0f}%</span></div>
                    <div class="cell-sub num"><span class="nw">SL <b>{sl_pct:.0f}%</b></span> &middot; <span class="nw">DờiSL <b>{l2_pct:.0f}%</b></span> &middot; <span class="nw">TP1 <b>{tp1_pct:.0f}%</b></span> &middot; <span class="nw">TP2 <b>{tp2_pct:.0f}%</b></span></div>
                </td>
                <td>
                    <div class="cell-main num" style="color:{pf_col};">PF {pf_1y:.2f}</div>
                    <div class="cell-sub num"><span class="nw">6T đầu <b{is_col_style}>{pf_is:.2f}</b></span> &middot; <span class="nw">6T cuối <b{oos_col_style}>{pf_oos:.2f}</b></span></div>
                </td>
                <td style="text-align:center;">{vol_html}</td>
                <td style="text-align:center;">{st_html}</td>
            </tr>"""
        )

    return f"""
    <table class="compact-tbl" id="matrix-48-{sym}">
        <colgroup>
            <col style="width:14%">
            <col style="width:13%">
            <col style="width:19%">
            <col style="width:24%">
            <col style="width:12%">
            <col style="width:9%">
            <col style="width:9%">
        </colgroup>
        <thead>
            <tr>
                <th>1. Coin &amp; Khung</th>
                <th>2. Mã Setup</th>
                <th>3. Chốt Lời &amp; Dời SL</th>
                <th>4. Thắng Tiền &amp; 4 Nhánh %</th>
                <th>5. Hệ Số Lãi/Lỗ (PF)</th>
                <th>6. Mức Volume</th>
                <th>7. Trạng Thái</th>
            </tr>
        </thead>
        <tbody>
            {''.join(rows_html)}
        </tbody>
    </table>
    """


def render_color_tier_legend_html() -> str:
    """Render slim single-line Volume Tier + Legendary Shimmer + Financial PnL legend bar."""
    return """
    <div class="tier-legend-bar">
      <span style="color:#0F172A;font-weight:700;">Thang Mức Volume:</span>
      <span class="nw"><span class="ldot" style="background:#EA580C;"></span><b style="color:#C2410C;">S+</b> (&ge;1.10&times;)</span>
      <span class="nw"><span class="ldot" style="background:#7C3AED;"></span><b style="color:#6D28D9;">S</b> (1.00&times;)</span>
      <span class="nw"><span class="ldot" style="background:#0284C7;"></span><b style="color:#0369A1;">A</b> (0.50&ndash;0.99&times;)</span>
      <span class="nw"><span class="ldot" style="background:#0D9488;"></span><b style="color:#0F766E;">B</b> (0.45&times;)</span>
      <span class="nw"><span class="ldot" style="background:#64748B;"></span><b style="color:#475569;">C</b> (&le;0.40&times;)</span>
      <span class="nw"><span class="must-badge" style="margin-left:0;">THẤY LÀ QUẤT</span> Siêu Setup Huyền Thoại</span>
      <span class="nw" style="margin-left:auto;color:#64748B;font-size:11.5px;">Lãi/PF: <b style="color:#059669;">Xanh lá</b> Cao &middot; <b style="color:#B45309;">Cam</b> Ổn &middot; <b style="color:#DC2626;">Đỏ</b> Lỗ</span>
    </div>
    """


def render_rog_calculator_and_ladder_html() -> str:
    """Render Utility 1 (Interactive USD & 1U Calculator) and Utility 2 (20-Step Compounding 1U Ladder Table) in Tab 1 (Deep Slate-Navy Fintech style)."""
    step_options = []
    initial_rows = []
    base_default = 10000.0
    for s in range(21):
        mult = 1.08 ** s
        pct_gain = (mult - 1.0) * 100.0
        cap_s = base_default * mult
        u400_10 = cap_s / 400.0
        u400_045 = u400_10 * 0.45
        u250_10 = cap_s / 250.0
        u250_045 = u250_10 * 0.45
        sl5_400 = -5.0 * u400_10
        sl5_250 = -5.0 * u250_10
        sel_attr = ' selected="selected"' if s == 0 else ""
        turbo_tag = " [Đủ mốc >= 1.25x bật Turbo 250G]" if mult >= 1.25 else ""
        step_options.append(
            f'<option value="{s}"{sel_attr}>Bậc {s} (&times;{mult:.3f} · +{pct_gain:.1f}%){turbo_tag}</option>'
        )

        active_cls = " step-active" if s == 0 else ""
        active_badge = '<span class="step-active-pill" id="step-pill-0">Đang chọn</span>' if s == 0 else f'<span class="step-active-pill" id="step-pill-{s}" style="display:none;">Đang chọn</span>'
        turbo_badge = '<span class="turbo-pill nw" title="Từ Bậc 3 (vốn >= 1.25x vốn gốc), chế độ Auto-Turbo tự động sang số Bao 250 giá">Turbo 250G</span>' if mult >= 1.25 else ""
        initial_rows.append(
            f"""<tr class="step-row{active_cls}" id="ladder-row-{s}" data-step="{s}" onclick="selectStepRow({s})" title="Bấm vào dòng này để chọn nhanh Bậc {s} trên Máy Tính">
                <td>
                    <div class="cell-main num" style="color:#0F172A;"><span class="nw">Bậc {s}</span> <span class="nw muted-meta">(+{pct_gain:.1f}%)</span>{active_badge}</div>
                    <div class="cell-sub num"><span class="nw">Hệ số lãi kép: <b>&times;{mult:.3f}</b></span>{turbo_badge}</div>
                </td>
                <td>
                    <div class="cell-main num" id="lad-cap-{s}" style="color:#0F172A;">${cap_s:,.2f}</div>
                    <div class="cell-sub num" id="lad-cap-sub-{s}"><span class="nw">Lãi tích lũy: <b>+${cap_s - base_default:,.2f}</b></span></div>
                </td>
                <td>
                    <div class="cell-main num"><span class="nw" style="color:#059669;">1.0&times;: <strong id="lad-400-10-{s}">${u400_10:,.2f}</strong></span> &middot; <span class="nw" style="color:#0369A1;">0.45&times;: <strong id="lad-400-045-{s}">${u400_045:,.2f}</strong></span></div>
                    <div class="cell-sub num" id="lad-400-vol-{s}"><span class="nw">Theo Vol đang chọn (1.00&times;): <b>${u400_10:,.2f}/1U</b></span></div>
                </td>
                <td>
                    <div class="cell-main num"><span class="nw" style="color:#B45309;">1.0&times;: <strong id="lad-250-10-{s}">${u250_10:,.2f}</strong></span> &middot; <span class="nw" style="color:#0369A1;">0.45&times;: <strong id="lad-250-045-{s}">${u250_045:,.2f}</strong></span></div>
                    <div class="cell-sub num" id="lad-250-vol-{s}"><span class="nw">Theo Vol đang chọn (1.00&times;): <b>${u250_10:,.2f}/1U</b></span></div>
                </td>
                <td>
                    <div class="cell-main num"><span class="nw" style="color:#DC2626;">400G: <strong id="lad-sl400-{s}">-${abs(sl5_400):,.2f}</strong></span> &middot; <span class="nw" style="color:#D97706;">250G: <strong id="lad-sl250-{s}">-${abs(sl5_250):,.2f}</strong></span></div>
                    <div class="cell-sub num" id="lad-sl-sub-{s}"><span class="nw">SL 5U tại Vol 1.00&times; (<b>-1.25%</b> / <b>-2.00%</b> vốn bậc)</span></div>
                </td>
            </tr>"""
        )

    return f"""
    <!-- UTILITY 1: DEEP SLATE-NAVY INTERACTIVE CALCULATOR -->
    <div class="fintech-calc-card">
      <div class="section-head">
        <h3>🧮 Máy Tính Nhanh &ldquo;Lãi USD &amp; Giá Trị 1U Theo Bậc Vốn&rdquo; (Interactive Calculator — Tính Tức Thì Khi Gõ/Chọn)</h3>
        <span class="ver-badge num" style="font-size:11.5px;">Công thức: 1U = (Vốn bậc &divide; Bao giá) &times; Volume</span>
      </div>
      <div class="calc-inputs-grid">
        <!-- Ô 1: Vốn gốc khởi điểm ($) -->
        <div class="calc-field">
          <div>
            <label for="calc-base-cap">1. Vốn Gốc Khởi Điểm ($ USD)</label>
            <input type="number" id="calc-base-cap" class="calc-input num" value="10000" min="100" step="500" oninput="updateRogCalculator()">
          </div>
          <div class="calc-quick-btns" id="q-cap-group">
            <button type="button" class="q-btn num" onclick="setCalcBaseCap(1000, this)">$1k</button>
            <button type="button" class="q-btn num" onclick="setCalcBaseCap(5000, this)">$5k</button>
            <button type="button" class="q-btn num active" onclick="setCalcBaseCap(10000, this)">$10k</button>
            <button type="button" class="q-btn num" onclick="setCalcBaseCap(20000, this)">$20k</button>
            <button type="button" class="q-btn num" onclick="setCalcBaseCap(50000, this)">$50k</button>
          </div>
        </div>

        <!-- Ô 2: Chọn Bậc lãi kép (Bậc 0 -> Bậc 20) -->
        <div class="calc-field">
          <div>
            <label for="calc-step">2. Bậc Lãi Kép (Bậc 0 &rarr; 20 · +8%/bậc)</label>
            <select id="calc-step" class="calc-select num" onchange="updateRogCalculator()">
              {''.join(step_options)}
            </select>
          </div>
          <div class="calc-quick-btns" id="q-step-group">
            <button type="button" class="q-btn num active" onclick="selectStepRow(0)">Bậc 0</button>
            <button type="button" class="q-btn num" onclick="selectStepRow(3)">Bậc 3 (Turbo)</button>
            <button type="button" class="q-btn num" onclick="selectStepRow(10)">Bậc 10</button>
            <button type="button" class="q-btn num" onclick="selectStepRow(20)">Bậc 20</button>
          </div>
        </div>

        <!-- Ô 3: Chế độ Bao giá -->
        <div class="calc-field">
          <div>
            <label for="calc-bao-mode">3. Chế Độ Bao Giá (400G / 250G)</label>
            <select id="calc-bao-mode" class="calc-select num" onchange="updateRogCalculator()">
              <option value="400" selected="selected">Bao 400 giá (Tiêu chuẩn)</option>
              <option value="250">Auto-Turbo 250 giá (Tăng tốc)</option>
            </select>
          </div>
          <div class="calc-quick-btns" id="q-bao-group">
            <button type="button" class="q-btn num active" id="q-bao-400" onclick="setCalcBao(400)">Bao 400G</button>
            <button type="button" class="q-btn num" id="q-bao-250" onclick="setCalcBao(250)">Turbo 250G</button>
          </div>
        </div>

        <!-- Ô 4: Số U lợi nhuận muốn tính -->
        <div class="calc-field">
          <div>
            <label for="calc-u-profit">4. Số U Lợi Nhuận Muốn Tính (+U)</label>
            <input type="number" id="calc-u-profit" class="calc-input num" value="27.2" step="0.5" oninput="updateRogCalculator()">
          </div>
          <div class="calc-quick-btns" id="q-u-group">
            <button type="button" class="q-btn num" onclick="setCalcU(10.0, this)">+10U</button>
            <button type="button" class="q-btn num active" onclick="setCalcU(27.2, this)">+27.2U</button>
            <button type="button" class="q-btn num" onclick="setCalcU(50.0, this)">+50U</button>
            <button type="button" class="q-btn num" onclick="setCalcU(100.0, this)">+100U</button>
          </div>
        </div>

        <!-- Ô 5: Mức Volume (x) -->
        <div class="calc-field">
          <div>
            <label for="calc-vol-mult">5. Mức Volume Hệ Số (&times;)</label>
            <input type="number" id="calc-vol-mult" class="calc-input num" value="1.0" min="0.1" max="3.0" step="0.05" oninput="updateRogCalculator()">
          </div>
          <div class="calc-quick-btns" id="q-vol-group">
            <button type="button" class="q-btn num" onclick="setCalcVol(0.45, this)">0.45&times;</button>
            <button type="button" class="q-btn num" onclick="setCalcVol(0.75, this)">0.75&times;</button>
            <button type="button" class="q-btn num active" onclick="setCalcVol(1.0, this)">1.0&times;</button>
            <button type="button" class="q-btn num" onclick="setCalcVol(1.15, this)">1.15&times;</button>
          </div>
        </div>
      </div>

      <!-- 4 KPI Result Cards -->
      <div class="calc-kpi-grid">
        <div class="calc-kpi-box" style="border-top: 3px solid #D97706;">
          <span class="kpi-lbl">1. Vốn Tại Bậc Đang Chọn ($)</span>
          <strong class="kpi-val num" id="calc-out-step-cap" style="color:#B45309;">$10,000.00</strong>
          <span class="kpi-sub num" id="calc-out-step-sub">Bậc 0/20 (&times;1.000 vốn gốc · +0.0%)</span>
        </div>
        <div class="calc-kpi-box" style="border-top: 3px solid #0284C7;">
          <span class="kpi-lbl">2. Giá Trị 1U Thực Tế ($)</span>
          <strong class="kpi-val num" id="calc-out-1u-actual" style="color:#0369A1;">$25.00 / 1U</strong>
          <span class="kpi-sub num" id="calc-out-1u-sub">1U chuẩn 1.0&times; = $25.00 · Bao 400G · Vol 1.00&times;</span>
        </div>
        <div class="calc-kpi-box" style="border-top: 3px solid #059669;">
          <span class="kpi-lbl">3. Lãi Ròng Thực Nhận (+$)</span>
          <strong class="kpi-val num" id="calc-out-net-usd" style="color:#059669;">+$680.00</strong>
          <span class="kpi-sub num" id="calc-out-net-sub">+6.80% trên vốn Bậc 0 (+6.80% trên vốn gốc)</span>
        </div>
        <div class="calc-kpi-box" style="border-top: 3px solid #DC2626;">
          <span class="kpi-lbl">4. Rủi Ro Cắt Lỗ SL (5U / 6U)</span>
          <strong class="kpi-val num" id="calc-out-sl-usd" style="color:#DC2626;">-$125.00 / -$150.00</strong>
          <span class="kpi-sub num" id="calc-out-sl-sub">SL 5U (BTC): -1.25% · SL 6U (Alt): -1.50% vốn bậc</span>
        </div>
      </div>
    </div>

    <!-- UTILITY 2: 20-STEP COMPOUNDING 1U LADDER TABLE -->
    <div class="section-card">
      <div class="section-head">
        <h3>📈 Bảng Lộ Trình Giá Trị 1U Qua 20 Bậc Lãi Kép (Bậc 0 &rarr; Bậc 20 · +8%/Bậc — Tự Động Cập Nhật Theo Vốn Gốc)</h3>
        <span style="font-size:12px;color:#0369A1;font-weight:600;">Bấm vào dòng Bậc bất kỳ bên dưới để đưa số liệu lên Máy Tính</span>
      </div>
      <table class="compact-tbl" id="tbl-20-step-ladder">
        <colgroup>
          <col style="width:18%">
          <col style="width:18%">
          <col style="width:21%">
          <col style="width:21%">
          <col style="width:22%">
        </colgroup>
        <thead>
          <tr>
            <th title="Bậc lãi kép từ Bậc 0 (Vốn gốc) đến Bậc 20 (+8% mỗi bậc = x4.661 vốn gốc). Từ Bậc 3 (>= 1.25x vốn gốc) đủ điều kiện bật Auto-Turbo 250G">1. Bậc Lãi Kép (0 &rarr; 20)</th>
            <th title="Mốc vốn tại từng bậc = Vốn gốc x (1.08 ^ Bậc)">2. Mốc Vốn Tại Bậc ($)</th>
            <th title="Giá trị 1U ở chế độ Bao 400 giá chuẩn tại mức Volume 1.0x (Core), 0.45x (Sniper/Scout) và mức Volume đang chọn">3. 1U — Bao 400G (1.0&times; / 0.45&times;)</th>
            <th title="Giá trị 1U ở chế độ Auto-Turbo Bao 250 giá tại mức Volume 1.0x, 0.45x và mức Volume đang chọn">4. 1U — Turbo 250G (1.0&times; / 0.45&times;)</th>
            <th title="Rủi ro cắt lỗ 1 lệnh SL 5U ở chế độ Bao 400G và Turbo 250G theo mức Volume đang chọn trên Máy Tính">5. Rủi Ro SL 5U (400G / 250G)</th>
          </tr>
        </thead>
        <tbody>
          {''.join(initial_rows)}
        </tbody>
      </table>
    </div>
    """


def main():
    core_cfg, all_75_rules, coin_audits = load_all_v23_data()
    conv_df = compute_1u_usd_reference_table()
    top_must_keys = compute_top_must_keys(all_75_rules)
    summary = core_cfg.get("summary", {})
    n_total = len(all_75_rules)
    n_core = sum(1 for r in all_75_rules if r["status"] == "CORE")
    n_exp = sum(1 for r in all_75_rules if r["status"] in ("CORE_EXP", "SCOUT_EXP"))
    n_sniper = sum(1 for r in all_75_rules if r["status"] == "SNIPER")
    n_scout = sum(1 for r in all_75_rules if r["status"] == "SCOUT")
    n_hot = sum(1 for r in all_75_rules if r["status"] == "HOT_ADAPTIVE")

    b12 = summary.get("base_400g_12m", {"ret_pct": 841.7, "final_eq_usd": 94170, "net_profit_usd": 84170, "max_dd_pct": -9.72, "calmar": 86.6})
    boos = summary.get("base_400g_oos_6m", {"ret_pct": 485.2, "max_dd_pct": -6.88, "calmar": 70.5})
    t12 = summary.get("auto_turbo_12m", {"ret_pct": 1481.0, "final_eq_usd": 158100, "net_profit_usd": 148100, "max_dd_pct": -12.13, "calmar": 122.1})
    toos = summary.get("auto_turbo_oos_6m", {"ret_pct": 828.1, "max_dd_pct": -8.62, "calmar": 96.1})

    # Build 6 Individual Coin Tabs HTML
    coin_tabs_html = []
    coin_nav_btns = []
    for sym in SIX_COINS:
        meta = COIN_META[sym]
        adf = coin_audits[sym]
        coin_rules = [r for r in all_75_rules if r["coin"] == sym]
        rules_by_key = {(r["coin"], r["tf"], r["setup_code"]): r for r in coin_rules}

        coin_nav_btns.append(
            f'<button class="tab-btn" onclick="switchTab(\'tab-{sym}\', this)">{meta["icon"]} {sym} ({len(coin_rules)})</button>'
        )

        tot_trades = sum(int(r["prior_1y"]["trades"]) for r in coin_rules)
        avg_pwr = sum(get_profit_wr(r["prior_1y"]) * int(r["prior_1y"]["trades"]) for r in coin_rules) / max(tot_trades, 1)
        avg_wr = sum(float(r["prior_1y"]["wr_pct"]) * int(r["prior_1y"]["trades"]) for r in coin_rules) / max(tot_trades, 1)
        avg_pf = sum(float(r["prior_1y"]["pf_1y"]) for r in coin_rules) / max(len(coin_rules), 1)
        tot_net_u = sum(float(r["prior_1y"]["net_u"]) for r in coin_rules)

        coin_tabs_html.append(
            f"""
            <section id="tab-{sym}" class="tab-panel">
                <div class="coin-hero">
                    <div>
                        <h2>{meta['icon']} {meta['title']}</h2>
                        <p class="coin-desc">{meta['summary']}</p>
                        <div class="coin-tags">
                            <span class="param-tag num">Cấu hình chuẩn: {meta['sl_tp']}</span>
                            <span class="param-tag num">Số Setup đang chạy: <strong>{len(coin_rules)} Setup</strong> ({tot_trades} lệnh/năm)</span>
                            <span class="param-tag num">Thắng Tiền (Profit WR) TB: <strong style="color:{color_for_pwr(avg_pwr)};">{avg_pwr:.1f}%</strong> (WR lệnh: {avg_wr:.1f}%) · PF TB: <strong style="color:{color_for_pf(avg_pf)};">{avg_pf:.2f}</strong> · Tổng lãi: <strong style="color:#059669;">+{tot_net_u:.1f}U</strong></span>
                        </div>
                    </div>
                </div>

                <div class="section-card">
                    <div class="section-head">
                        <h3>Danh Sách {len(coin_rules)} Setup Đang Chạy Của {sym} (Xếp Theo Thứ Tự &ldquo;Đụng Là Quất&rdquo;)</h3>
                    </div>
                    {render_color_tier_legend_html()}
                    {build_75_rules_table_html(coin_rules, table_id=f"tbl-picked-{sym}", top_must_keys=top_must_keys)}
                </div>

                <details class="ref-details">
                    <summary>Xem thêm toàn bộ 48 tổ hợp tham khảo của {sym} (5m · 15m · 30m · 1h &times; EMA 34 · 89 · 200 &times; Chạm 1–4 — Đồng bộ 100%)</summary>
                    <div class="details-body">
                        <div class="section-head" style="margin-top:8px;">
                            <h3>Bảng Tham Khảo Đầy Đủ 48 Tổ Hợp Của {sym}</h3>
                            <div class="filter-group">
                                <button class="f-btn active" onclick="filterCoinMatrix('{sym}', 'ALL', this)">Tất cả (48)</button>
                                <button class="f-btn" onclick="filterCoinMatrix('{sym}', '5m', this)">5m (12)</button>
                                <button class="f-btn" onclick="filterCoinMatrix('{sym}', '15m', this)">15m (12)</button>
                                <button class="f-btn" onclick="filterCoinMatrix('{sym}', '30m', this)">30m (12)</button>
                                <button class="f-btn" onclick="filterCoinMatrix('{sym}', '1h', this)">1h (12)</button>
                            </div>
                        </div>
                        {build_coin_48_matrix_html(sym, adf, rules_by_key, top_must_keys=top_must_keys)}
                    </div>
                </details>
            </section>
            """
        )

    # Single Chart Data: Per-coin Net U & Profit WR of the active 75 setups
    c_labels = [f"{COIN_META[c]['short']} ({sum(1 for r in all_75_rules if r['coin']==c)} Setup)" for c in SIX_COINS]
    c_net_u = [round(sum(float(r["prior_1y"]["net_u"]) for r in all_75_rules if r["coin"] == c), 1) for c in SIX_COINS]
    c_pwr_avg = [
        round(
            sum(get_profit_wr(r["prior_1y"]) * int(r["prior_1y"]["trades"]) for r in all_75_rules if r["coin"] == c)
            / max(sum(int(r["prior_1y"]["trades"]) for r in all_75_rules if r["coin"] == c), 1),
            1,
        )
        for c in SIX_COINS
    ]

    html_doc = f"""<!DOCTYPE html>
<html lang="vi">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>CHICKTRADE v2.3 — Hệ Thống Giao Dịch 6 Coin ({n_total} Setup Đang Chạy)</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@500;600;700&display=swap" rel="stylesheet">
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
  :root {{
    --bg: #F8FAFC;
    --card: #FFFFFF;
    --card-2: #F1F5F9;
    --surface-input: #FFFFFF;
    --border: #E2E8F0;
    --border-strong: #CBD5E1;
    --text: #0F172A;
    --text-sec: #475569;
    --muted: #64748B;
    --font-sans: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    --font-mono: 'JetBrains Mono', ui-monospace, SFMono-Regular, Consolas, monospace;
  }}
  * {{
    box-sizing: border-box;
  }}
  html, body {{
    margin: 0;
    padding: 0;
    width: 100%;
    max-width: 100vw;
    overflow-x: hidden;
    font-family: var(--font-sans);
    background: #F8FAFC;
    color: var(--text);
    line-height: 1.48;
    font-size: 13.5px;
    -webkit-font-smoothing: antialiased;
  }}
  .num {{
    font-variant-numeric: tabular-nums;
  }}
  .page-shell {{
    max-width: 1350px;
    margin: 0 auto;
    padding: 0 20px 60px 20px;
  }}

  /* Top Header — Clean Light Fintech */
  .hero-bar {{
    background: #FFFFFF;
    border-bottom: 1px solid #E2E8F0;
    box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
    padding: 18px 0 15px 0;
  }}
  .hero-inner {{
    max-width: 1350px;
    margin: 0 auto;
    padding: 0 20px;
    display: flex;
    flex-wrap: wrap;
    justify-content: space-between;
    align-items: center;
    gap: 14px;
  }}
  .hero-title h1 {{
    margin: 0;
    font-size: 20px;
    font-weight: 700;
    color: #0F172A;
    letter-spacing: -0.2px;
  }}
  .hero-title p {{
    margin: 4px 0 0 0;
    font-size: 13px;
    color: #475569;
  }}
  .hero-actions {{
    display: flex;
    align-items: center;
    gap: 10px;
    flex-wrap: wrap;
  }}
  .dict-toggle-btn {{
    background: #F8FAFC;
    border: 1px solid #CBD5E1;
    color: #0F172A;
    padding: 7px 14px;
    border-radius: 8px;
    cursor: pointer;
    font-weight: 600;
    font-size: 12.8px;
    transition: all 0.15s ease;
  }}
  .dict-toggle-btn:hover {{
    background: #F1F5F9;
    border-color: #94A3B8;
  }}
  .ver-badge {{
    background: #ECFDF5;
    border: 1px solid #A7F3D0;
    color: #059669;
    padding: 6px 12px;
    border-radius: 8px;
    font-family: var(--font-mono);
    font-weight: 600;
    font-size: 12px;
  }}

  /* Sticky Navigation Bar */
  .nav-tabs {{
    position: sticky;
    top: 0;
    z-index: 100;
    background: rgba(255, 255, 255, 0.92);
    backdrop-filter: blur(8px);
    border-bottom: 1px solid #E2E8F0;
    padding: 10px 0;
    margin-bottom: 18px;
  }}
  .nav-tabs-inner {{
    max-width: 1350px;
    margin: 0 auto;
    padding: 0 20px;
    display: flex;
    flex-wrap: wrap;
    gap: 7px;
  }}
  .tab-btn {{
    background: #FFFFFF;
    border: 1px solid #CBD5E1;
    color: #475569;
    padding: 8px 14px;
    border-radius: 8px;
    cursor: pointer;
    font-weight: 600;
    font-size: 12.8px;
    transition: all 0.15s ease;
  }}
  .tab-btn:hover {{
    background: #F1F5F9;
    border-color: #94A3B8;
    color: #0F172A;
  }}
  .tab-btn.active {{
    background: #0F172A;
    border-color: #0F172A;
    color: #FFFFFF;
    box-shadow: none;
  }}
  .tab-panel {{
    display: none;
  }}
  .tab-panel.active {{
    display: block;
  }}

  /* Quick Glossary Drawer */
  .glossary-drawer {{
    display: none;
    background: #FFFFFF;
    border: 1px solid #CBD5E1;
    border-radius: 12px;
    padding: 18px 20px;
    margin: 0 0 18px 0;
    box-shadow: 0 4px 16px rgba(15, 23, 42, 0.08);
  }}
  .glossary-drawer.open {{
    display: block;
  }}
  .glossary-drawer h3 {{
    margin: 0 0 12px 0;
    color: #0F172A;
    font-size: 15px;
    display: flex;
    justify-content: space-between;
    align-items: center;
  }}
  .glossary-grid {{
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 12px;
  }}
  @media (max-width: 960px) {{
    .glossary-grid {{ grid-template-columns: 1fr; }}
  }}
  .g-item {{
    background: #F8FAFC;
    border: 1px solid #E2E8F0;
    border-left: 3px solid #0284C7;
    border-radius: 8px;
    padding: 12px 14px;
    font-size: 12.5px;
    color: #475569;
  }}
  .g-item strong {{
    color: #0F172A;
    display: block;
    margin-bottom: 5px;
    font-size: 13px;
  }}
  code {{
    font-family: var(--font-mono);
    font-size: 12px;
    color: #0F172A;
    background: #F1F5F9;
    border: 1px solid #E2E8F0;
    padding: 1px 5px;
    border-radius: 4px;
  }}

  /* 2-Layer Legend Bar (Slim 1-Row) */
  .tier-legend-bar {{
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 16px;
    background: #F8FAFC;
    border: 1px solid #E2E8F0;
    border-radius: 8px;
    padding: 7px 14px;
    margin-bottom: 10px;
    font-size: 12px;
    color: #475569;
  }}
  .ldot {{
    display: inline-block;
    width: 8px;
    height: 8px;
    border-radius: 50%;
    margin-right: 6px;
    vertical-align: middle;
  }}
  .nw {{
    white-space: nowrap;
    display: inline-block;
  }}
  .muted-meta {{
    color: #64748B;
    font-size: 11.5px;
    font-weight: 500;
  }}
  .must-tag {{
    color: #C2410C;
    font-weight: 700;
  }}
  .dir-text {{
    color: #64748B;
    font-weight: 500;
  }}

  /* Two Mode Banner Cards */
  .dual-mode-grid {{
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 16px;
    margin-bottom: 18px;
  }}
  @media (max-width: 920px) {{
    .dual-mode-grid {{ grid-template-columns: 1fr; }}
  }}
  .mode-card {{
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 12px;
    padding: 18px 20px;
    box-shadow: 0 1px 3px rgba(15, 23, 42, 0.05);
  }}
  .mode-card h3 {{
    margin: 0 0 8px 0;
    font-size: 15.5px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 8px;
  }}
  .mode-metrics {{
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 10px;
    margin-top: 12px;
  }}
  .m-item {{
    background: #F8FAFC;
    padding: 11px 12px;
    border-radius: 8px;
    border: 1px solid #E2E8F0;
  }}
  .m-item span {{
    display: block;
    font-size: 11.4px;
    color: #64748B;
    font-weight: 600;
  }}
  .m-item strong {{
    display: block;
    font-family: var(--font-mono);
    font-size: 16px;
    font-weight: 700;
    margin-top: 4px;
  }}
  .m-item small {{
    display: block;
    font-size: 11px;
    color: #64748B;
    margin-top: 3px;
    line-height: 1.32;
  }}

  /* Section Card */
  .section-card {{
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 12px;
    padding: 18px 20px;
    margin-bottom: 18px;
    width: 100%;
    overflow: hidden;
    box-shadow: 0 1px 3px rgba(15, 23, 42, 0.05);
  }}
  .section-head {{
    display: flex;
    flex-wrap: wrap;
    justify-content: space-between;
    align-items: center;
    gap: 10px;
    margin-bottom: 12px;
    border-bottom: 1px solid #E2E8F0;
    padding-bottom: 10px;
  }}
  .section-head h3 {{
    margin: 0;
    font-size: 15.5px;
    color: #0F172A;
  }}

  /* Collapsible Reference Details */
  details.ref-details {{
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 12px;
    padding: 14px 20px;
    margin-bottom: 18px;
    box-shadow: 0 1px 3px rgba(15, 23, 42, 0.05);
  }}
  details.ref-details summary {{
    cursor: pointer;
    font-weight: 600;
    color: #0369A1;
    font-size: 14px;
    padding: 4px 0;
    user-select: none;
  }}
  details.ref-details summary:hover {{
    color: #0F172A;
  }}
  .details-body {{
    margin-top: 12px;
    border-top: 1px solid #E2E8F0;
    padding-top: 10px;
  }}

  /* Filter & Sort Buttons */
  .filter-group {{
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
  }}
  .f-btn {{
    background: #FFFFFF;
    border: 1px solid #CBD5E1;
    color: #475569;
    padding: 6px 12px;
    border-radius: 7px;
    cursor: pointer;
    font-size: 12px;
    font-weight: 600;
    transition: all 0.14s ease;
  }}
  .f-btn:hover {{
    background: #F1F5F9;
    border-color: #94A3B8;
    color: #0F172A;
  }}
  .f-btn.active {{
    background: #0F172A;
    border-color: #0F172A;
    color: #FFFFFF;
  }}
  .sort-bar {{
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 7px;
    margin-bottom: 10px;
    padding: 8px 12px;
    background: #F8FAFC;
    border: 1px solid #E2E8F0;
    border-radius: 9px;
  }}
  .sort-label {{
    font-size: 12px;
    font-weight: 700;
    color: #0F172A;
    margin-right: 4px;
  }}
  .s-btn {{
    background: #FFFFFF;
    border: 1px solid #CBD5E1;
    color: #475569;
    padding: 5px 12px;
    border-radius: 7px;
    cursor: pointer;
    font-size: 12px;
    font-weight: 600;
    transition: all 0.14s ease;
  }}
  .s-btn:hover {{
    background: #F1F5F9;
    border-color: #94A3B8;
    color: #0F172A;
  }}
  .s-btn.active {{
    background: #0F172A;
    border-color: #0F172A;
    color: #FFFFFF;
  }}

  /* ZERO HORIZONTAL SCROLL COMPACT TABLE */
  table.compact-tbl {{
    width: 100%;
    table-layout: fixed;
    border-collapse: collapse;
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 10px;
    overflow: hidden;
  }}
  table.compact-tbl th,
  table.compact-tbl td {{
    border: 1px solid #E2E8F0;
    padding: 9px 10px;
    vertical-align: middle;
    word-wrap: break-word;
    overflow-wrap: break-word;
    white-space: normal;
  }}
  table.compact-tbl th {{
    background: #F1F5F9;
    color: #334155;
    font-size: 12px;
    font-weight: 700;
    text-align: left;
    border-bottom: 2px solid #CBD5E1;
    cursor: help;
  }}
  .cell-main {{
    font-weight: 600;
    font-size: 13px;
    color: #0F172A;
    line-height: 1.36;
  }}
  .cell-sub {{
    font-size: 11.5px;
    color: #64748B;
    margin-top: 3px;
    line-height: 1.32;
  }}
  .cell-sub b {{
    color: #334155;
    font-weight: 600;
    font-family: var(--font-mono);
  }}
  .cell-sub strong {{
    color: #334155;
    font-weight: 600;
  }}
  .code-pill {{
    display: inline-block;
    font-family: var(--font-mono);
    font-weight: 700;
    font-size: 12.6px;
    color: #0F172A;
    background: #F1F5F9;
    border: 1px solid #CBD5E1;
    padding: 1px 7px;
    border-radius: 5px;
    font-variant-numeric: tabular-nums;
  }}
  tr.row-must {{
    background: linear-gradient(90deg, #FFFBEB 0%, #FEF08A44 35%, #FFEDD5 75%, #FFF7ED 100%);
  }}
  tr.row-must td:first-child {{
    border-left: 4px solid #F59E0B;
  }}
  tr.row-must:hover {{
    background: #FEF3C7;
  }}
  tr.row-active {{
    background: #FFFFFF;
  }}
  tr.row-active:hover {{
    background: #F8FAFC;
  }}
  tr.row-skip {{
    background: #F8FAFC;
    color: #94A3B8;
  }}
  tr.row-m5-skip {{
    background: #FFF5F5;
    color: #94A3B8;
  }}
  tr.row-skip:hover, tr.row-m5-skip:hover {{
    background: #F1F5F9;
  }}

  /* MMORPG Legendary Shimmer & Rank Badges */
  @keyframes legendary-shimmer {{
    0% {{ transform: translateX(-140%); }}
    45%, 100% {{ transform: translateX(180%); }}
  }}
  .rank-badge {{
    display: inline-block;
    min-width: 27px;
    text-align: center;
    padding: 1px 5px;
    border-radius: 5px;
    font-family: var(--font-mono);
    font-size: 11px;
    font-weight: 700;
    margin-right: 4px;
    vertical-align: middle;
  }}
  .rank-top,
  .must-badge {{
    position: relative;
    overflow: hidden;
    background: linear-gradient(135deg, #F59E0B 0%, #EA580C 50%, #D97706 100%);
    color: #FFFFFF;
    border: 1px solid #FDE68A;
    box-shadow: 0 2px 8px rgba(245, 158, 11, 0.35);
  }}
  .rank-top::after,
  .must-badge::after {{
    content: "";
    position: absolute;
    top: 0;
    left: 0;
    width: 60%;
    height: 100%;
    background: linear-gradient(110deg, transparent 20%, rgba(255, 255, 255, 0.65) 50%, transparent 80%);
    animation: legendary-shimmer 3.2s ease-in-out infinite;
    pointer-events: none;
  }}
  .must-badge {{
    display: inline-block;
    padding: 1px 6px;
    border-radius: 5px;
    font-size: 9.8px;
    font-weight: 700;
    letter-spacing: 0.2px;
    margin-left: 4px;
    vertical-align: middle;
  }}
  .rank-normal {{
    background: #F1F5F9;
    color: #475569;
    border: 1px solid #E2E8F0;
  }}

  /* Badges & Volume Tier Pills */
  .tf-badge {{
    display: inline-block;
    padding: 1px 7px;
    border-radius: 5px;
    font-family: var(--font-mono);
    font-size: 11px;
    font-weight: 700;
    margin-left: 3px;
  }}
  .tf-5m {{ background: #FFF1F2; color: #BE123C; border: 1px solid #FECDD3; }}
  .tf-15m {{ background: #E0F2FE; color: #0369A1; border: 1px solid #BAE6FD; }}
  .tf-30m {{ background: #F3E8FF; color: #6D28D9; border: 1px solid #DDD6FE; }}
  .tf-1h {{ background: #ECFDF5; color: #047857; border: 1px solid #A7F3D0; }}
  .dir-badge {{
    display: inline-block;
    padding: 1px 6px;
    border-radius: 5px;
    font-size: 10.8px;
    font-weight: 600;
  }}
  .dir-long {{ background: #ECFDF5; color: #047857; border: 1px solid #A7F3D0; }}
  .dir-short {{ background: #FFF1F2; color: #BE123C; border: 1px solid #FECDD3; }}
  .status-pill {{
    display: inline-block;
    padding: 3px 8px;
    border-radius: 6px;
    font-size: 11px;
    font-weight: 700;
    white-space: nowrap;
  }}
  .st-sp {{ background: #FFEDD5; color: #C2410C; border: 1px solid #FDBA74; }}
  .st-s {{ background: #F3E8FF; color: #6D28D9; border: 1px solid #DDD6FE; }}
  .st-a {{ background: #E0F2FE; color: #0369A1; border: 1px solid #BAE6FD; }}
  .st-b {{ background: #F0FDFA; color: #0F766E; border: 1px solid #99F6E4; }}
  .st-c {{ background: #F1F5F9; color: #475569; border: 1px solid #CBD5E1; }}
  .st-d {{ background: #FFF1F2; color: #BE123C; border: 1px solid #FECDD3; }}

  /* Single Full-Width Chart Box */
  .chart-single-box {{
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 12px;
    padding: 14px;
    height: 360px;
    margin-bottom: 18px;
    box-shadow: 0 1px 3px rgba(15, 23, 42, 0.05);
  }}

  /* Light Fintech Interactive Calculator & 20-Step Ladder */
  .fintech-calc-card {{
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    box-shadow: 0 1px 3px rgba(15, 23, 42, 0.05);
    border-radius: 12px;
    padding: 18px 20px;
    margin-bottom: 18px;
  }}
  .calc-inputs-grid {{
    display: grid;
    grid-template-columns: repeat(5, 1fr);
    gap: 11px;
    margin-bottom: 14px;
  }}
  @media (max-width: 1120px) {{
    .calc-inputs-grid {{ grid-template-columns: repeat(2, 1fr); }}
  }}
  .calc-field {{
    background: #F8FAFC;
    border: 1px solid #E2E8F0;
    border-radius: 9px;
    padding: 10px 12px;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
  }}
  .calc-field label {{
    display: block;
    font-size: 11.5px;
    font-weight: 700;
    color: #334155;
    margin-bottom: 6px;
    line-height: 1.25;
  }}
  .calc-input, .calc-select {{
    width: 100%;
    background: #FFFFFF;
    border: 1px solid #CBD5E1;
    color: #0F172A;
    font-family: var(--font-mono);
    font-size: 13.6px;
    font-weight: 700;
    padding: 7px 9px;
    border-radius: 7px;
    outline: none;
    transition: border-color 0.15s ease, box-shadow 0.15s ease;
  }}
  .calc-input:focus, .calc-select:focus {{
    border-color: #0284C7;
    box-shadow: 0 0 0 2px rgba(2, 132, 199, 0.15);
  }}
  .calc-quick-btns {{
    display: flex;
    flex-wrap: wrap;
    gap: 4px;
    margin-top: 7px;
  }}
  .q-btn {{
    background: #FFFFFF;
    border: 1px solid #CBD5E1;
    color: #475569;
    font-family: var(--font-mono);
    font-size: 10.8px;
    font-weight: 600;
    padding: 3px 6px;
    border-radius: 5px;
    cursor: pointer;
    flex: 1;
    text-align: center;
    transition: all 0.14s ease;
  }}
  .q-btn:hover {{
    background: #F1F5F9;
    border-color: #94A3B8;
    color: #0F172A;
  }}
  .q-btn.active {{
    background: #0F172A;
    border-color: #0F172A;
    color: #FFFFFF;
    font-weight: 700;
  }}
  .calc-kpi-grid {{
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 11px;
  }}
  @media (max-width: 1024px) {{
    .calc-kpi-grid {{ grid-template-columns: repeat(2, 1fr); }}
  }}
  .calc-kpi-box {{
    background: #F8FAFC;
    border-radius: 9px;
    padding: 12px 14px;
    border: 1px solid #E2E8F0;
  }}
  .calc-kpi-box .kpi-lbl {{
    display: block;
    font-size: 11.4px;
    font-weight: 600;
    color: #64748B;
  }}
  .calc-kpi-box .kpi-val {{
    display: block;
    font-family: var(--font-mono);
    font-size: 18px;
    font-weight: 700;
    margin: 4px 0 3px 0;
  }}
  .calc-kpi-box .kpi-sub {{
    display: block;
    font-size: 11px;
    color: #64748B;
    line-height: 1.32;
  }}
  tr.step-row {{
    background: #FFFFFF;
    cursor: pointer;
    transition: background 0.12s ease;
  }}
  tr.step-row:hover {{
    background: #F8FAFC;
  }}
  tr.step-row.step-active {{
    background: #EFF6FF !important;
  }}
  tr.step-row.step-active td:first-child {{
    border-left: 3px solid #0284C7;
  }}
  .step-active-pill {{
    display: inline-block;
    background: #E0F2FE;
    color: #0369A1;
    border: 1px solid #BAE6FD;
    font-size: 10.4px;
    font-weight: 700;
    padding: 1px 6px;
    border-radius: 5px;
    margin-left: 6px;
  }}
  .turbo-pill {{
    display: inline-block;
    background: #FFEDD5;
    color: #C2410C;
    border: 1px solid #FDBA74;
    font-size: 10.2px;
    font-weight: 600;
    padding: 1px 6px;
    border-radius: 5px;
    margin-left: 5px;
  }}

  /* Coin Hero */
  .coin-hero {{
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 12px;
    padding: 18px 20px;
    margin-bottom: 16px;
    box-shadow: 0 1px 3px rgba(15, 23, 42, 0.05);
  }}
  .coin-hero h2 {{
    margin: 0 0 6px 0;
    font-size: 18px;
    color: #0F172A;
  }}
  .coin-desc {{
    margin: 0 0 10px 0;
    color: #475569;
    font-size: 13px;
  }}
  .coin-tags {{
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
  }}
  .param-tag {{
    background: #F8FAFC;
    border: 1px solid #E2E8F0;
    padding: 5px 11px;
    border-radius: 7px;
    font-size: 12px;
    font-weight: 600;
    color: #334155;
  }}

  /* Step Flow Grid */
  .flow-grid {{
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 14px;
  }}
  @media (max-width: 960px) {{
    .flow-grid {{ grid-template-columns: 1fr; }}
  }}
  .flow-box {{
    background: #F8FAFC;
    border: 1px solid #E2E8F0;
    border-top: 3px solid #0284C7;
    border-radius: 10px;
    padding: 15px 16px;
    font-size: 13px;
    color: #475569;
  }}
  .flow-box h4 {{
    margin: 0 0 8px 0;
    color: #0F172A;
    font-size: 14.2px;
  }}
  .hero-meta-wrap {{
    max-width: 1350px;
    margin: 11px auto 0 auto;
    padding: 0 20px;
  }}
  .hero-meta-bar {{
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 8px 18px;
    background: #F8FAFC;
    border: 1px solid #E2E8F0;
    border-radius: 8px;
    padding: 8px 14px;
    font-size: 12px;
    color: #334155;
    line-height: 1.45;
  }}
  .hero-meta-bar b {{
    color: #0F172A;
    font-weight: 700;
  }}
</style>
</head>
<body>

<header class="hero-bar">
  <div class="hero-inner">
    <div class="hero-title">
      <h1>CHICKTRADE v2.3 — Hệ Thống Giao Dịch Định Lượng 6 Coin ({n_total} Setup Đang Chạy)</h1>
      <p>Giao diện Light Fintech (Unslop) &middot; 2 Lớp Màu Độc Lập (Thang Volume &amp; Chữ Số Tài Chính) &middot; Bao 400G / Auto-Turbo 250G &middot; 0 Cuộn Ngang</p>
    </div>
    <div class="hero-actions">
      <button class="dict-toggle-btn" onclick="toggleGlossary()">📖 Từ Điển Giải Thích Nhanh (30 Giây Hiểu Liền)</button>
      <span class="ver-badge num">FINTECH v2.3 · {n_total} SETUPS</span>
    </div>
  </div>
  <div class="hero-meta-wrap">
    <div class="hero-meta-bar">
      <span><b>Dữ liệu thống kê:</b> <span class="nw"><code>01/11/2025 &rarr; 31/10/2026</code></span> <em class="nw">(12 tháng trọn vẹn &middot; 105,120 nến 5m/coin &middot; Binance USD-M Futures)</em></span>
      <span><b>Chia kiểm chứng:</b> <span class="nw"><code>6T đầu (IS): 01/11/2025 &ndash; 23/04/2026</code></span> &middot; <span class="nw"><code>6T cuối (OOS): 23/04/2026 &ndash; 31/10/2026</code></span></span>
      <span><b>Phương pháp tính:</b> <span class="nw"><code>1U = 0.5 &times; ATR(14)</code></span> &middot; <span class="nw">Đã trừ phí sàn <code>0.02%</code> Maker / <code>0.05%</code> Taker + trượt giá <code>0.01%</code></span> &middot; <span class="nw">Vốn mẫu <code>$10,000</code> lãi kép 20 bậc (<code>+8%/bậc</code>)</span> &middot; <span class="nw">Phanh <code>PORTFOLIO 2-SL Core-Only &rarr; Shadow-Win</code></span></span>
    </div>
  </div>
</header>

<nav class="nav-tabs">
  <div class="nav-tabs-inner">
    <button class="tab-btn active" onclick="switchTab('tab-overview', this)">🌐 Tổng Quan v2.3 ({n_total} Setup)</button>
    <button class="tab-btn" onclick="switchTab('tab-75-rules', this)">🎯 {n_total} Setup Đang Chạy</button>
    {''.join(coin_nav_btns)}
    <button class="tab-btn" onclick="switchTab('tab-guide', this)">📖 Luật Vận Hành Hiện Tại</button>
  </div>
</nav>

<div class="page-shell">

  <!-- COLLAPSIBLE QUICK GLOSSARY DRAWER (6 CORE EXPLANATIONS) -->
  <div id="glossary-drawer" class="glossary-drawer">
    <h3>
      <span>📖 Từ Điển Giải Thích Nhanh 6 Điểm Cốt Lõi (Rê chuột vào tiêu đề cột bất kỳ để xem gợi ý)</span>
      <button class="f-btn" onclick="toggleGlossary()">Đóng &times;</button>
    </h3>
    <div class="glossary-grid">
      <div class="g-item" style="border-left-color:#0284C7;">
        <strong>1. Cách Đọc Mã Setup Rút Gọn (EMA89C1 / EMA200C4)</strong>
        &bull; <code>EMA34 / EMA89 / EMA200</code> là đường trung bình động đóng vai trò hỗ trợ/kháng cự động.<br>
        &bull; Chữ <code>C1, C2, C3, C4</code> là viết tắt của <strong>Chạm lần 1, Chạm lần 2, Chạm lần 3, Chạm lần 4</strong> tính từ khi bắt đầu xu hướng: <code>C1–C3</code> đánh <strong>Thuận xu hướng</strong>, còn <code>C4</code> đánh <strong>Đảo chiều cạn sóng</strong>.
      </div>
      <div class="g-item" style="border-left-color:#059669;">
        <strong style="color:#059669;">2. Phân Biệt Thắng Tiền % (Profit WR) vs WR Số Lệnh %</strong>
        &bull; <code>WR Số Lệnh %</code> chỉ đếm số lần thắng (nếu SL 6U mà TP1 chỉ 3.5U thì WR lệnh 60% vẫn có thể lỗ tiền).<br>
        &bull; <code>Thắng Tiền % (Profit WR) = Tổng Tiền Thắng &divide; (Tổng Thắng + Tổng Thua) &times; 100%</code>. <strong>Cứ Thắng Tiền &gt; 50% là chắc chắn 100% có lãi tiền thật sau phí sàn!</strong>
      </div>
      <div class="g-item" style="border-left-color:#DC2626;">
        <strong style="color:#DC2626;">3. Giải Nghĩa 4 Nhánh Kết Quả &amp; Cơ Chế Phanh 2-SL</strong>
        &bull; <code>SL %</code>: Dính cắt lỗ gốc ban đầu &rarr; <strong>CÓ tính vào bộ đếm phanh 2-SL Shadow-Win</strong> (với Setup Core <code>Vol &gt; 0.45&times;</code>).<br>
        &bull; <code>DờiSL %</code>: Đã chạm TP1 (chốt lãi 50%), sau đó giá vòng lại chạm SL chân 2 &rarr; <strong>KHÔNG tính là SL kích hoạt phanh 2-SL</strong>.<br>
        &bull; <code>TP1 %</code> &amp; <code>TP2 %</code>: Tỷ lệ chạm đích chốt lời nhịp 1 và ăn trọn nhịp 2.
      </div>
      <div class="g-item" style="border-left-color:#6D28D9;">
        <strong style="color:#6D28D9;">4. Bốn Nhóm Trong {n_total} Setup Đang Chạy</strong>
        &bull; <strong>Đánh Chính Core ({n_core}) &amp; Mở Rộng ({n_exp}):</strong> Nhóm chủ lực bền cả 2 nửa năm (Volume <code>0.57&times; – 1.23&times;</code>).<br>
        &bull; <strong>Sniper Hiếm Lệnh ({n_sniper}) &amp; Scout ({n_scout}):</strong> Xuất hiện ít (3–14 lệnh/năm) nhưng Thắng Tiền rất cao (Volume <code>0.35&times; – 0.45&times;</code>).<br>
        &bull; <strong>Thích Ứng 6T Cuối ({n_hot}):</strong> Bắt sóng mới nổi ở nửa cuối năm (Volume thăm dò <code>0.40&times;</code>).
      </div>
      <div class="g-item" style="border-left-color:#0284C7;">
        <strong>5. Lý Do Tách 2 Nửa Năm: 6T Đầu (IS) &amp; 6T Cuối (OOS)</strong>
        Chia dữ liệu 1 năm thành 2 giai đoạn 6 tháng độc lập (6 tháng đầu năm và 6 tháng cuối năm) để kiểm tra chéo. Việc một Setup duy trì hệ số Lãi/Lỗ <code>PF &gt; 1.10</code> ở cả 2 nửa năm chứng minh hệ thống có lợi thế thật, không bị học vẹt quá khứ (Overfitting).
      </div>
      <div class="g-item" style="border-left-color:#D97706;">
        <strong style="color:#B45309;">6. Công Thức Nhẩm Nhanh Tiền Thật Tại Bậc 0 ($10,000 · Bao 400G)</strong>
        Tại vốn khởi điểm <strong>$10,000</strong> và mức <strong>Bao 400 giá</strong>, mỗi <code>1U = $25.00 USD</code>:<br>
        <code>Lãi USD = Số U &times; $25 &times; Mức Volume</code>.<br>
        Ví dụ: Setup thắng <code>+40U</code> với <code>Volume 1.0&times;</code> &rarr; Lãi đơn <code>40 &times; $25 &times; 1.0 = +$1,000 USD</code> (chưa cộng lãi kép 20 bậc).
      </div>
    </div>
  </div>

  <!-- TAB 1: OVERVIEW -->
  <section id="tab-overview" class="tab-panel active">

    <div class="dual-mode-grid">
      <div class="mode-card" style="border-top: 4px solid #059669;">
        <h3>
          <span style="color:#059669;">🛡️ Chế Độ 1: Tiêu Chuẩn An Toàn (Bao 400 Giá · Tối Đa 5 Lệnh)</span>
          <span class="status-pill" style="background:#ECFDF5;color:#059669;border:1px solid #A7F3D0;">MẶC ĐỊNH ĐANG BẬT</span>
        </h3>
        <p style="margin:0;color:#475569;font-size:12.8px;">
          Vận hành đồng thời <strong>{n_total} Setup đang chạy</strong> ({n_core} Core · {n_exp} Mở rộng · {n_sniper} Sniper · {n_scout} Scout · {n_hot} Thích ứng 6T cuối) tại mức <strong>Bao 400 giá</strong> (mỗi 1U = $25 USD ở vốn gốc $10,000). Tự động kích hoạt <strong>Đánh nháp ngầm (Shadow-Win)</strong> sau 2 lệnh SL Core liên tiếp.
        </p>
        <div class="mode-metrics">
          <div class="m-item">
            <span>Lãi Ròng 1 Năm (USD &amp; %)</span>
            <strong class="num" style="color:#059669;">+${int(b12['net_profit_usd']):,d} (+{float(b12['ret_pct']):.1f}%)</strong>
            <small class="num">Tiền USD lãi ròng trên vốn gốc $10,000 (Lãi kép 20 bậc &rarr; Tổng tài sản ${int(b12['final_eq_usd']):,d})</small>
          </div>
          <div class="m-item">
            <span>Sụt Giảm Tối Đa (MaxDD 1 Năm)</span>
            <strong class="num" style="color:#DC2626;">{float(b12['max_dd_pct']):.2f}% <span style="color:#0369A1;font-size:13px;">(Calmar {float(b12['calmar']):.1f})</span></strong>
            <small>Đường cong vốn mượt, kiểm soát rủi ro chặt chẽ trên cả 6 Coin</small>
          </div>
          <div class="m-item">
            <span>Hiệu Suất 6 Tháng Cuối (OOS)</span>
            <strong class="num" style="color:#059669;">+{float(boos['ret_pct']):.1f}% <span style="color:#DC2626;font-size:13px;">(DD {float(boos['max_dd_pct']):.2f}%)</span></strong>
            <small class="num">Sụt giảm 6T cuối cực thấp chỉ <strong>{float(boos['max_dd_pct']):.2f}%</strong> · Điểm Calmar OOS <strong>{float(boos['calmar']):.1f}</strong></small>
          </div>
        </div>
      </div>

      <div class="mode-card" style="border-top: 4px solid #D97706;">
        <h3>
          <span style="color:#B45309;">🚀 Chế Độ 2: Tăng Tốc Auto-Turbo (400G &rarr; 250G · Khóa Sàn &ge; $11,000)</span>
          <span class="status-pill st-sp">CÔNG TẮC TÙY CHỌN (ON/OFF)</span>
        </h3>
        <p style="margin:0;color:#475569;font-size:12.8px;">
          Khởi đầu ở Bao 400 giá để bảo vệ vốn gốc $10,000. Khi vốn tăng lên <strong>&ge; $12,500</strong>, hệ thống tự động sang số <strong>Bao 250 giá</strong> để bứt tốc lãi kép. Nếu tổng rủi ro SL của các lệnh đang mở có nguy cơ chạm sàn bảo vệ <strong>$11,000 (+10% lãi khóa cứng)</strong>, bot tự lùi về Bao 400 giá ngay lập tức.
        </p>
        <div class="mode-metrics">
          <div class="m-item">
            <span>Lãi Ròng 1 Năm (USD &amp; %)</span>
            <strong class="num" style="color:#15803D;">+${int(t12['net_profit_usd']):,d} (+{float(t12['ret_pct']):,.1f}%)</strong>
            <small class="num">Tiền USD lãi ròng trên vốn gốc $10,000 (Lãi kép 20 bậc &rarr; Tổng tài sản ${int(t12['final_eq_usd']):,d})</small>
          </div>
          <div class="m-item">
            <span>Sụt Giảm Tối Đa (MaxDD 1 Năm)</span>
            <strong class="num" style="color:#DC2626;">{float(t12['max_dd_pct']):.2f}% <span style="color:#B45309;font-size:13px;">(Calmar {float(t12['calmar']):.1f})</span></strong>
            <small class="num">Điểm Calmar đạt Bậc S+ (<strong>{float(t12['calmar']):.1f}</strong>) · Giữ cứng sàn tài sản &ge; <strong>$11,000</strong></small>
          </div>
          <div class="m-item">
            <span>Hiệu Suất 6 Tháng Cuối (OOS)</span>
            <strong class="num" style="color:#15803D;">+{float(toos['ret_pct']):.1f}% <span style="color:#DC2626;font-size:13px;">(DD {float(toos['max_dd_pct']):.2f}%)</span></strong>
            <small class="num">Sụt giảm 6T cuối chỉ <strong>{float(toos['max_dd_pct']):.2f}%</strong> · Điểm Calmar OOS đạt <strong>{float(toos['calmar']):.1f}</strong></small>
          </div>
        </div>
      </div>
    </div>

    <div class="chart-single-box" id="chart-overview-coins"></div>

    {render_rog_calculator_and_ladder_html()}

    <div class="section-card">
      <div class="section-head">
        <h3>💵 Bảng Quy Đổi Tham Khảo 1U &rarr; Tiền USD &amp; Khối Lượng Vào Lệnh Thực Tế (Mẫu Vốn Gốc $10,000 · Bao 400 Giá · Vol 1.0&times;)</h3>
      </div>
      <p style="margin:0 0 10px 0;font-size:12.5px;color:#475569;">
        Tính từ trung vị 30 ngày gần nhất của từng đồng Coin (<code>1U = 0.5 &times; ATR(14)</code>). Tại vốn gốc <strong>$10,000</strong> và mức <strong>Bao 400 giá</strong>, mỗi <strong>1U biến động giá tương đương đúng $25.00 USD</strong> (Công thức nhẩm nhanh: <code>Lãi USD = Số U &times; $25 &times; Mức Volume</code>):
      </p>
      {render_1u_conversion_table_html(conv_df)}
    </div>
  </section>

  <!-- TAB 2: ALL 75 ACTIVE SETUPS -->
  <section id="tab-75-rules" class="tab-panel">
    <div class="section-card">
      <div class="section-head">
        <h3>🎯 Bảng Xếp Hạng {n_total} Setup Đang Chạy (&ldquo;ĐỤNG LÀ QUẤT&rdquo;: Ưu Tiên Thắng Tiền Cao &amp; Kiếm Nhiều Tiền Nhất Lên Đầu)</h3>
        <div class="filter-group">
          <button class="f-btn active" onclick="filter75Table('ALL', this)">Tất cả ({n_total})</button>
          <button class="f-btn" onclick="filter75Table('CORE', this)">Core ({n_core})</button>
          <button class="f-btn" onclick="filter75Table('EXP', this)">Mở rộng ({n_exp})</button>
          <button class="f-btn" onclick="filter75Table('SNIPER', this)">Sniper ({n_sniper})</button>
          <button class="f-btn" onclick="filter75Table('SCOUT', this)">Scout ({n_scout})</button>
          <button class="f-btn" onclick="filter75Table('HOT_ADAPTIVE', this)">Thích ứng 6T cuối ({n_hot})</button>
          <button class="f-btn" onclick="filter75Table('BTCUSDT', this)">🪙 BTC</button>
          <button class="f-btn" onclick="filter75Table('ETHUSDT', this)">💎 ETH</button>
          <button class="f-btn" onclick="filter75Table('BNBUSDT', this)">🔶 BNB</button>
          <button class="f-btn" onclick="filter75Table('LINKUSDT', this)">🔗 LINK</button>
          <button class="f-btn" onclick="filter75Table('XRPUSDT', this)">⚡ XRP</button>
          <button class="f-btn" onclick="filter75Table('SOLUSDT', this)">☀️ SOL</button>
        </div>
      </div>
      {render_color_tier_legend_html()}
      <div class="sort-bar">
        <span class="sort-label">Sắp xếp 1-Click:</span>
        <button class="s-btn active" onclick="sort75Table('MUST', this)">Ưu tiên hàng đầu (Đụng là quất)</button>
        <button class="s-btn" onclick="sort75Table('NETU', this)">Lãi ròng (Net U)</button>
        <button class="s-btn" onclick="sort75Table('PWR', this)">Thắng tiền (%)</button>
        <button class="s-btn" onclick="sort75Table('WR', this)">Winrate lệnh (%)</button>
        <button class="s-btn" onclick="sort75Table('COIN', this)">Theo mã coin</button>
      </div>
      {build_75_rules_table_html(all_75_rules, table_id="tbl-75-master", top_must_keys=top_must_keys)}
    </div>
  </section>

  <!-- TABS 3..8: 6 PER-COIN TABS -->
  {''.join(coin_tabs_html)}

  <!-- TAB 9: CURRENT LIVE OPERATING RULES -->
  <section id="tab-guide" class="tab-panel">
    <div class="section-card">
      <div class="section-head">
        <h3>📖 Tổng Hợp Bộ Luật Vận Hành Thực Tế Hiện Tại Của Hệ Thống CHICKTRADE v2.3 ({n_total} Setup Chạy Tự Động 24/7)</h3>
      </div>
      <div class="flow-grid">
        <div class="flow-box">
          <h4>1. Đa Khung Giờ &amp; Đồng Thuận Xu Hướng</h4>
          &bull; Giao dịch 24/7 trên các khung <code>15m</code>, <code>30m</code>, <code>1h</code> (và Setup chọn lọc trên <code>5m</code> là <code>LINK 5m EMA89C2</code>).<br>
          &bull; Khung <code>5m &amp; 15m</code> bắt buộc thuận theo xu hướng đóng nến của khung <code>1h</code>; khung <code>30m &amp; 1h</code> bắt buộc thuận theo khung <code>4h</code> (<code>Close &gt; EMA200 &amp; EMA34 &gt; EMA200</code> cho Long; ngược lại cho Short) kèm <code>ADX(14) &ge; 20</code>.
        </div>
        <div class="flow-box">
          <h4>2. Mã Setup (C1–C4) &amp; Khóa Chiều (Luật 1B)</h4>
          &bull; Sau khi giá bứt phá khỏi EMA (<code>Impulse &ge; 0.8–1.2 ATR</code>), chờ giá hồi về chạm vùng đệm EMA (<code>&pm;0.20 ATR</code>) và rút chân giữ cấu trúc:<br>
          &bull; <strong>Chạm 1, 2, 3 (C1, C2, C3):</strong> Đánh thuận xu hướng chính.<br>
          &bull; <strong>Chạm 4 (C4):</strong> Đánh đảo chiều cạn sóng.<br>
          &bull; <strong>Khóa 1 chiều:</strong> <code>BNBUSDT 1h EMA200C1</code> chỉ đánh <strong>LONG</strong>; <code>XRPUSDT 1h EMA34C2</code> chỉ đánh <strong>SHORT</strong>.
        </div>
        <div class="flow-box">
          <h4>3. Thoát Lệnh Tối Ưu &amp; 4 Nhánh Xác Suất</h4>
          Mỗi Setup trong {n_total} Setup sử dụng đúng kiến trúc chốt lời sinh lời cao nhất của nó:<br>
          &bull; <strong>Chốt 100% tại TP1 (3.5U–4U) hoặc 1R (5U–6U):</strong> Khóa lãi gọn cho các Setup bật nhanh.<br>
          &bull; <strong>Chia 2 nhịp (50% TP1 + 50% TP2):</strong> Chốt 50% ở TP1, phần còn lại gồng lên TP2 (8U–12U) với SL chân 2 đặt tại <code>Entry - 1U</code>, <code>Entry - 2U</code>, <code>Hòa vốn</code> hoặc <code>Giữ SL gốc</code>.
        </div>
        <div class="flow-box">
          <h4>4. Phân Bổ Khối Lượng ({n_total} Setup Đang Chạy)</h4>
          &bull; <strong>{n_core} Setup Chính (Core):</strong> Volume từ <code>0.57&times; đến 1.23&times;</code> theo điểm Kelly.<br>
          &bull; <strong>{n_exp} Setup Mở Rộng (Core/Scout Exp):</strong> Bền cả 2 nửa năm, Volume <code>0.75&times;</code> (8 Setup) và <code>0.45&times;</code> (3 Setup).<br>
          &bull; <strong>{n_sniper} Setup Sniper &amp; {n_scout} Setup Scout:</strong> Hiếm lệnh nhưng Thắng Tiền cao, Volume <code>0.35&times; – 0.45&times;</code>.<br>
          &bull; <strong>{n_hot} Setup Thích Ứng 6T Cuối:</strong> Volume thăm dò <code>0.40&times;</code>.
        </div>
        <div class="flow-box">
          <h4>5. Khiên Chống Bão 3 Lớp &amp; Đánh Nháp Ngầm (Luật 4E)</h4>
          &bull; <strong>Storm Guard 4h:</strong> Tự khóa vào lệnh mới 4 giờ khi nến giật bão 2 đầu (<code>ATR &ge; 1.8&times;MA50</code>, nến quét <code>&ge; 2.8 ATR</code>, hoặc <code>Kaufman ER &lt; 0.15</code>) trên coin đó hoặc trên BTC.<br>
          &bull; <strong>Đánh Nháp Ngầm (Shadow-Win):</strong> Khi cả danh mục dính <strong>2 lệnh SL gốc liên tiếp</strong> ở nhóm Core (<code>Vol &gt; 0.45&times;</code> theo Luật 4E; nhánh <code>Dời/QuaySL</code> và các Setup &le; 0.45&times; không làm đóng băng danh mục), bot tạm dừng tiền thật và đợi <strong>1 lệnh thắng nháp</strong> mới mở lại.
        </div>
        <div class="flow-box">
          <h4>6. Quản Trị Danh Mục, Lãi Kép 20 Bậc &amp; Auto-Turbo</h4>
          &bull; <strong>Trần Danh Mục:</strong> Tối đa <strong>5 lệnh mở cùng lúc</strong> (mỗi coin tối đa 1 lệnh, tổng hệ số Volume mở &le; <code>4.25&times;</code>), tối đa <strong>10 lệnh/ngày</strong>, nghỉ trong ngày nếu lỗ ngày chạm <code>-5%</code>.<br>
          &bull; <strong>Lãi Kép 20 Bậc (+8%/bậc) &amp; Auto-Turbo:</strong> Mặc định chạy <strong>Bao 400 giá (+$84,170)</strong>. Khi bật <strong>Auto-Turbo</strong>, hễ vốn &ge; <code>$12,500</code> bot tự chuyển sang <strong>Bao 250 giá (+$148,100)</strong> và luôn giữ cứng sàn tài sản <strong>&ge; $11,000</strong>.
        </div>
      </div>
    </div>
  </section>

</div>

<script>
  function toggleGlossary() {{
    const el = document.getElementById('glossary-drawer');
    el.classList.toggle('open');
  }}

  function switchTab(tabId, btnEl) {{
    document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
    document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
    document.getElementById(tabId).classList.add('active');
    btnEl.classList.add('active');
    window.scrollTo({{ top: 0, behavior: 'smooth' }});
    if (tabId === 'tab-overview') {{
      setTimeout(() => {{
        Plotly.Plots.resize('chart-overview-coins');
      }}, 60);
    }}
  }}

  function filter75Table(filterVal, btnEl) {{
    const grp = btnEl.parentElement;
    grp.querySelectorAll('.f-btn').forEach(b => b.classList.remove('active'));
    btnEl.classList.add('active');
    const rows = document.querySelectorAll('#tbl-75-master tbody tr');
    rows.forEach(r => {{
      const c = r.getAttribute('data-coin');
      const st = r.getAttribute('data-status');
      const g = r.getAttribute('data-group');
      if (filterVal === 'ALL' || c === filterVal || st === filterVal || g === filterVal) {{
        r.style.display = '';
      }} else {{
        r.style.display = 'none';
      }}
    }});
  }}

  function sort75Table(mode, btnEl) {{
    const bar = btnEl.parentElement;
    bar.querySelectorAll('.s-btn').forEach(b => b.classList.remove('active'));
    btnEl.classList.add('active');
    const tbody = document.querySelector('#tbl-75-master tbody');
    const rows = Array.from(tbody.querySelectorAll('tr'));
    rows.sort((a, b) => {{
      if (mode === 'MUST') {{
        return parseFloat(b.getAttribute('data-must')) - parseFloat(a.getAttribute('data-must'));
      }}
      if (mode === 'NETU') {{
        return parseFloat(b.getAttribute('data-netu')) - parseFloat(a.getAttribute('data-netu'))
          || parseFloat(b.getAttribute('data-pwr')) - parseFloat(a.getAttribute('data-pwr'));
      }}
      if (mode === 'PWR') {{
        return parseFloat(b.getAttribute('data-pwr')) - parseFloat(a.getAttribute('data-pwr'))
          || parseFloat(b.getAttribute('data-netu')) - parseFloat(a.getAttribute('data-netu'));
      }}
      if (mode === 'WR') {{
        return parseFloat(b.getAttribute('data-wr')) - parseFloat(a.getAttribute('data-wr'))
          || parseFloat(b.getAttribute('data-netu')) - parseFloat(a.getAttribute('data-netu'));
      }}
      if (mode === 'COIN') {{
        return parseInt(a.getAttribute('data-coin-ord')) - parseInt(b.getAttribute('data-coin-ord'))
          || parseFloat(b.getAttribute('data-must')) - parseFloat(a.getAttribute('data-must'));
      }}
      return 0;
    }});
    rows.forEach((r, idx) => {{
      const badge = r.querySelector('.rank-badge');
      if (badge) badge.textContent = '#' + (idx + 1);
      tbody.appendChild(r);
    }});
  }}

  function filterCoinMatrix(sym, filterVal, btnEl) {{
    const grp = btnEl.parentElement;
    grp.querySelectorAll('.f-btn').forEach(b => b.classList.remove('active'));
    btnEl.classList.add('active');
    const rows = document.querySelectorAll('#matrix-48-' + sym + ' tbody tr');
    rows.forEach(r => {{
      const tf = r.getAttribute('data-tf');
      if (filterVal === 'ALL' || tf === filterVal) {{
        r.style.display = '';
      }} else {{
        r.style.display = 'none';
      }}
    }});
  }}

  // Interactive Calculator & 20-Step Compounding Ladder
  function fmtUsd(v) {{
    return '$' + Number(v).toLocaleString('en-US', {{ minimumFractionDigits: 2, maximumFractionDigits: 2 }});
  }}

  function setCalcBaseCap(val, btnEl) {{
    document.getElementById('calc-base-cap').value = val;
    if (btnEl) {{
      document.querySelectorAll('#q-cap-group .q-btn').forEach(b => b.classList.remove('active'));
      btnEl.classList.add('active');
    }}
    updateRogCalculator();
  }}

  function setCalcBao(val) {{
    document.getElementById('calc-bao-mode').value = String(val);
    updateRogCalculator();
  }}

  function setCalcU(val, btnEl) {{
    document.getElementById('calc-u-profit').value = val;
    if (btnEl) {{
      document.querySelectorAll('#q-u-group .q-btn').forEach(b => b.classList.remove('active'));
      btnEl.classList.add('active');
    }}
    updateRogCalculator();
  }}

  function setCalcVol(val, btnEl) {{
    document.getElementById('calc-vol-mult').value = val;
    if (btnEl) {{
      document.querySelectorAll('#q-vol-group .q-btn').forEach(b => b.classList.remove('active'));
      btnEl.classList.add('active');
    }}
    updateRogCalculator();
  }}

  function selectStepRow(stepIdx) {{
    document.getElementById('calc-step').value = String(stepIdx);
    updateRogCalculator();
  }}

  function updateRogCalculator() {{
    const rawCap = parseFloat(document.getElementById('calc-base-cap').value);
    const baseCap = (!isNaN(rawCap) && rawCap > 0) ? rawCap : 10000;
    const step = parseInt(document.getElementById('calc-step').value) || 0;
    const baoMode = parseFloat(document.getElementById('calc-bao-mode').value) || 400;
    const rawU = parseFloat(document.getElementById('calc-u-profit').value);
    const uProfit = !isNaN(rawU) ? rawU : 0;
    const rawVol = parseFloat(document.getElementById('calc-vol-mult').value);
    const volMult = (!isNaN(rawVol) && rawVol > 0) ? rawVol : 1.0;

    // Sync quick button active states
    document.querySelectorAll('#q-cap-group .q-btn').forEach(b => {{
      const txt = b.textContent.replace('$', '').replace('k', '000');
      b.classList.toggle('active', parseFloat(txt) === baseCap);
    }});
    document.querySelectorAll('#q-step-group .q-btn').forEach(b => {{
      const m = b.textContent.match(/Bậc\s+(\d+)/);
      b.classList.toggle('active', m && parseInt(m[1]) === step);
    }});
    document.getElementById('q-bao-400').classList.toggle('active', baoMode === 400);
    document.getElementById('q-bao-250').classList.toggle('active', baoMode === 250);
    document.querySelectorAll('#q-u-group .q-btn').forEach(b => {{
      const v = parseFloat(b.textContent.replace('+', '').replace('U', ''));
      b.classList.toggle('active', Math.abs(v - uProfit) < 0.001);
    }});
    document.querySelectorAll('#q-vol-group .q-btn').forEach(b => {{
      const v = parseFloat(b.textContent);
      b.classList.toggle('active', Math.abs(v - volMult) < 0.001);
    }});

    const mult = Math.pow(1.08, step);
    const stepCap = baseCap * mult;
    const pctStepGain = (mult - 1.0) * 100.0;
    const u1Std = stepCap / baoMode;
    const u1Actual = u1Std * volMult;
    const netUsd = uProfit * u1Actual;
    const netPctOnStep = stepCap > 0 ? (netUsd / stepCap) * 100.0 : 0.0;
    const netPctOnBase = baseCap > 0 ? (netUsd / baseCap) * 100.0 : 0.0;
    const sl5Actual = 5.0 * u1Actual;
    const sl6Actual = 6.0 * u1Actual;
    const sl5Pct = stepCap > 0 ? (sl5Actual / stepCap) * 100.0 : 0.0;
    const sl6Pct = stepCap > 0 ? (sl6Actual / stepCap) * 100.0 : 0.0;

    document.getElementById('calc-out-step-cap').textContent = fmtUsd(stepCap);
    document.getElementById('calc-out-step-sub').textContent =
      'Bậc ' + step + '/20 (\u00d7' + mult.toFixed(3) + ' vốn gốc \u00b7 +' + pctStepGain.toFixed(1) + '%)';

    document.getElementById('calc-out-1u-actual').textContent = fmtUsd(u1Actual) + ' / 1U';
    document.getElementById('calc-out-1u-sub').textContent =
      '1U chuẩn 1.0\u00d7 = ' + fmtUsd(u1Std) + ' \u00b7 Bao ' + baoMode + 'G \u00b7 Vol ' + volMult.toFixed(2) + '\u00d7';

    const signStr = netUsd >= 0 ? '+' : '-';
    document.getElementById('calc-out-net-usd').textContent = signStr + fmtUsd(Math.abs(netUsd));
    document.getElementById('calc-out-net-usd').style.color = netUsd >= 0 ? '#059669' : '#DC2626';
    document.getElementById('calc-out-net-sub').textContent =
      (netPctOnStep >= 0 ? '+' : '') + netPctOnStep.toFixed(2) + '% trên vốn Bậc ' + step +
      ' (' + (netPctOnBase >= 0 ? '+' : '') + netPctOnBase.toFixed(2) + '% trên vốn gốc)';

    document.getElementById('calc-out-sl-usd').textContent = '-' + fmtUsd(sl5Actual) + ' / -' + fmtUsd(sl6Actual);
    document.getElementById('calc-out-sl-sub').textContent =
      'SL 5U (BTC): -' + sl5Pct.toFixed(2) + '% \u00b7 SL 6U (Alt): -' + sl6Pct.toFixed(2) + '% vốn bậc';

    // Update all 21 rows (Bậc 0 -> Bậc 20) in the 20-Step Compounding Ladder table
    for (let s = 0; s <= 20; s++) {{
      const m_s = Math.pow(1.08, s);
      const cap_s = baseCap * m_s;
      const u400_10 = cap_s / 400.0;
      const u400_045 = u400_10 * 0.45;
      const u400_vol = u400_10 * volMult;
      const u250_10 = cap_s / 250.0;
      const u250_045 = u250_10 * 0.45;
      const u250_vol = u250_10 * volMult;
      const sl400_vol = 5.0 * u400_vol;
      const sl250_vol = 5.0 * u250_vol;
      const sl400_pct = (sl400_vol / cap_s) * 100.0;
      const sl250_pct = (sl250_vol / cap_s) * 100.0;

      const rowEl = document.getElementById('ladder-row-' + s);
      const pillEl = document.getElementById('step-pill-' + s);
      if (rowEl) {{
        rowEl.classList.toggle('step-active', s === step);
      }}
      if (pillEl) {{
        pillEl.style.display = (s === step) ? 'inline-block' : 'none';
      }}

      document.getElementById('lad-cap-' + s).textContent = fmtUsd(cap_s);
      document.getElementById('lad-cap-sub-' + s).innerHTML =
        '<span class="nw">Lãi tích lũy: <b>+' + fmtUsd(cap_s - baseCap) + '</b></span>';

      document.getElementById('lad-400-10-' + s).textContent = fmtUsd(u400_10);
      document.getElementById('lad-400-045-' + s).textContent = fmtUsd(u400_045);
      document.getElementById('lad-400-vol-' + s).innerHTML =
        '<span class="nw">Theo Vol đang chọn (' + volMult.toFixed(2) + '\u00d7): <b>' + fmtUsd(u400_vol) + '/1U</b></span>';

      document.getElementById('lad-250-10-' + s).textContent = fmtUsd(u250_10);
      document.getElementById('lad-250-045-' + s).textContent = fmtUsd(u250_045);
      document.getElementById('lad-250-vol-' + s).innerHTML =
        '<span class="nw">Theo Vol đang chọn (' + volMult.toFixed(2) + '\u00d7): <b>' + fmtUsd(u250_vol) + '/1U</b></span>';

      document.getElementById('lad-sl400-' + s).textContent = '-' + fmtUsd(sl400_vol);
      document.getElementById('lad-sl250-' + s).textContent = '-' + fmtUsd(sl250_vol);
      document.getElementById('lad-sl-sub-' + s).innerHTML =
        '<span class="nw">SL 5U tại Vol ' + volMult.toFixed(2) + '\u00d7 (<b>-' + sl400_pct.toFixed(2) + '%</b> / <b>-' + sl250_pct.toFixed(2) + '%</b> vốn bậc)</span>';
    }}
  }}

  // Single Chart: Per-Coin Net U & Profit-Weighted Winrate of the active 75 setups (Light Mode Unslop Palette)
  const coinTraces = [
    {{
      name: 'Tổng Lãi Đơn Vị Biến Động (+U / năm)',
      x: {json.dumps(c_labels)},
      y: {json.dumps(c_net_u)},
      type: 'bar',
      marker: {{
        color: 'rgba(2, 132, 199, 0.18)',
        line: {{ color: '#0284C7', width: 1.5 }}
      }},
      yaxis: 'y1',
      text: {json.dumps([f"+{v:.1f}U" for v in c_net_u])},
      textposition: 'auto',
      textfont: {{ color: '#0F172A', family: 'JetBrains Mono, monospace', weight: 'bold' }}
    }},
    {{
      name: 'Winrate Lợi Nhuận — Thắng Tiền TB (%)',
      x: {json.dumps(c_labels)},
      y: {json.dumps(c_pwr_avg)},
      type: 'scatter',
      mode: 'lines+markers+text',
      marker: {{ color: '#059669', size: 10, line: {{ color: '#FFFFFF', width: 1.5 }} }},
      line: {{ color: '#059669', width: 3 }},
      yaxis: 'y2',
      text: {json.dumps([f"{v:.1f}%" for v in c_pwr_avg])},
      textposition: 'top center',
      textfont: {{ color: '#059669', family: 'JetBrains Mono, monospace', weight: 'bold' }}
    }}
  ];
  const coinLayout = {{
    title: {{ text: 'Đóng Góp Tổng Lãi Đơn Vị (+U) & Winrate Lợi Nhuận (Thắng Tiền %) Của 6 Coin Hiện Tại ({n_total} Setup Đang Chạy)', font: {{ color: '#0F172A', size: 14.5, family: 'Inter, sans-serif' }} }},
    paper_bgcolor: '#FFFFFF',
    plot_bgcolor: '#F8FAFC',
    font: {{ color: '#334155', size: 12, family: 'Inter, sans-serif' }},
    xaxis: {{ gridcolor: '#E2E8F0' }},
    yaxis: {{ title: 'Tổng Lãi Đơn Vị (+U)', gridcolor: '#E2E8F0' }},
    yaxis2: {{ title: 'Thắng Tiền — Profit WR (%)', overlaying: 'y', side: 'right', range: [50, 85], showgrid: false }},
    legend: {{ orientation: 'h', y: -0.20 }},
    margin: {{ t: 42, r: 55, l: 60, b: 60 }}
  }};
  Plotly.newPlot('chart-overview-coins', coinTraces, coinLayout, {{ responsive: true, displayModeBar: false }});
</script>
</body>
</html>
"""
    out_html = os.path.join(REPORTS_DIR, "chicktrade_6coin_master_report.html")
    with open(out_html, "w", encoding="utf-8") as f:
        f.write(html_doc)
    print(f"[DONE] Generated v2.3 Light Mode Fintech (Unslop) Master HTML Report ({n_total} setups, 2-layer color system, 0 horizontal scroll) at: {out_html}")


if __name__ == "__main__":
    main()
