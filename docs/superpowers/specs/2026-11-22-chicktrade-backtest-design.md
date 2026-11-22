# CHICKTRADE — Thiết kế Hệ thống Backtest 1 Năm (EMA 34–89–200 M5)

**Ngày tạo:** 2026-11-22  
**Thư mục dự án:** `e:\CHICKTRADE`

## 1. Mục tiêu
Xây dựng hệ thống backtest 1 năm trên dữ liệu lịch sử miễn phí của Binance (khung **M5**) cho danh mục **BTCUSDT, ETHUSDT, BNBUSDT** (kèm **PAXGUSDT** làm mốc tham chiếu vàng) nhằm:
1. **Thống kê thực nghiệm (Giai đoạn 1):** Đo chính xác tỷ lệ thắng (Winrate %), Profit Factor, Expectancy của từng loại tín hiệu (Sóng 1, Sóng 2, Sóng 3, Sóng 4+ đảo chiều) và từng bộ lọc (Price Action, Volume, EMA 200) trên 1 năm dữ liệu M5.
2. **Mô phỏng Portfolio & Lãi kép 20 bậc (Giai đoạn 2):** Áp dụng bộ quản lý vốn **Bao 200 giá + Phân tầng Full/Half theo Winrate (≥51% / >70%) + 20 bậc lãi kép (+8%/bậc)** kèm phí sàn Binance Futures để đánh giá hiệu quả thực chiến (có kiểm chứng Out-of-Sample: 6 tháng đầu đo Winrate → 6 tháng sau chạy thực chiến).

---

## 2. Kiến trúc Module

```text
e:\CHICKTRADE\
├── config.py                    # Cấu hình toàn cục (symbols, EMA, unit_size, risk, fees)
├── data/
│   ├── binance_loader.py        # Tải dữ liệu M5 miễn phí từ data.binance.vision + fallback REST API
│   └── cache/                   # Lưu file .parquet / .csv 1 năm dữ liệu cục bộ
├── strategy/
│   ├── indicators.py            # Tính EMA 34/89/200, ATR(14), Volume MA(20), nến rút râu
│   └── ema_wave.py              # Logic xác định trend, đếm sóng chạm EMA 34 (1, 2, 3, 4+), sinh tín hiệu
├── risk/
│   └── compounding.py           # Quy đổi Unit (bao 200 giá), phân tầng Full/Half (>70% vs 51-70%), 20 bậc +8%
├── engine/
│   └── portfolio.py             # Lõi Event-Driven Portfolio Backtest đa coin (SL, BE, TP, phí sàn, max 2 lệnh, 10 lệnh/ngày)
├── report/
│   └── html_report.py           # Sinh báo cáo HTML tương tác (Plotly) + CSV lịch sử lệnh + bảng thống kê
├── tests/
│   └── test_core.py             # Unit tests cho indicators, wave counter, compounding, SL/BE/TP
└── run_backtest.py              # CLI chạy toàn bộ pipeline (Giai đoạn 1 + Giai đoạn 2 + Report)
```

---

## 3. Quy tắc Chiến lược Số hóa (Phương án 1 – M5)

- **Chỉ báo:** `EMA 34`, `EMA 89`, `EMA 200`, `ATR(14)`, `Volume MA(20)`.
- **Xác định Trend & Reset chu kỳ sóng:**
  - **Uptrend:** `Close > EMA 200` và `EMA 34 > EMA 89`.
  - **Downtrend:** `Close < EMA 200` và `EMA 34 < EMA 89`.
  - Khi giá đóng cửa cắt phá `EMA 200` sang phía ngược lại → **Reset `wave_count = 0`** (đảo trend).
  - Yêu cầu có **nhịp mở đầu (impulse)** tách khỏi `EMA 34` ít nhất `min_impulse_atr × ATR` (mặc định `0.8 × ATR`) trước khi tính lần chạm tiếp theo.
- **Đếm lần chạm EMA 34 (`wave_count`):**
  - **Chạm Buy (Uptrend):** Giá đã tách lên trên `EMA 34` đủ ngưỡng, sau đó hồi về `Low <= EMA 34 + touch_buffer` và đóng nến giữ trên cấu trúc (`Close >= EMA 34 - breach_buffer`) → tăng `wave_count += 1`.
  - **Chạm Sell (Downtrend):** Đối xứng (`High >= EMA 34 - touch_buffer`, `Close <= EMA 34 + breach_buffer`) → tăng `wave_count += 1`.
  - Chống đếm trùng liên tiếp: sau mỗi lần chạm, giá phải rời khỏi vùng `EMA 34` một khoảng tối thiểu trước khi được tính là sóng mới.
- **Bộ lọc loại bỏ & Xác nhận:**
  - **Mất thanh khoản (bỏ lệnh):** Bỏ qua nếu nến tín hiệu có biên độ `High - Low > 3.0 × ATR` (quét tin/bão bất thường).
  - **Lọc Price Action + Volume (cho Sóng 2 và thống kê so sánh):** Nến có râu rút hướng thuận trend (`wick_ratio >= 0.30`) hoặc thân nến đóng thuận chiều, và `Volume >= 0.8 × Volume MA(20)`.
  - **Sóng 1:** Vào lệnh thuận trend.
  - **Sóng 2:** Vào lệnh thuận trend (có gắn cờ `pa_vol_ok` để thống kê cả bản gốc và bản có lọc PA+Vol).
  - **Sóng 3:** Ghi nhận vào thống kê Giai đoạn 1 để kiểm chứng thực tế, mặc định bỏ ở Giai đoạn 2 nếu Winrate < 51%.
  - **Sóng 4+:** Tín hiệu **ngược xu hướng** (chờ nến xác nhận đảo chiều / rút râu quét đỉnh-đáy sau lần chạm thứ 4).

---

## 4. Quy tắc Quản lý vốn & Lãi kép 20 Bậc

- **Vốn khởi điểm:** `\$10,000`.
- **Đơn vị quy đổi `1 Unit`:**
  - `PAXGUSDT` (Vàng): `1 Unit = \$1.00` (hoặc tỷ lệ tương đương).
  - `BTCUSDT`, `ETHUSDT`, `BNBUSDT`: `1 Unit = unit_pct × EntryPrice` (mặc định `0.10%` giá vào lệnh; có thể quét tham số `0.08%`, `0.10%`, `0.15%` để tìm biên độ tối ưu cho M5).
- **Công thức Volume "Bao 200 giá":**
  - **Full Size (`1.0×` — khi setup có Winrate thực nghiệm > 70%):**
    $$\text{Qty}_{\text{full}} = \frac{\text{Vốn bậc hiện tại}}{200 \times \text{unit\_size}}$$
    *(Cho phép mở tối đa 2 lệnh Full Size cùng lúc nếu cả 2 setup đều > 70%).*
  - **Half Size (`0.5×` — khi setup có Winrate thực nghiệm từ 51% đến 70%):**
    $$\text{Qty}_{\text{half}} = \frac{\text{Vốn bậc hiện tại}}{400 \times \text{unit\_size}}$$
    *(Tối đa 2 lệnh cùng lúc → tổng 2 lệnh gánh đúng 200 giá).*
  - **Winrate < 51%:** Bỏ qua, không vào lệnh.
- **Giới hạn vận hành:**
  - Tối đa **10 lệnh/ngày** (UTC day).
  - Tối đa **2 lệnh mở đồng thời** trên toàn danh mục (`BTCUSDT`, `ETHUSDT`, `BNBUSDT`).
- **Quản lý lệnh trong nến M5:**
  - **SL cố định:** `sl_units` (mặc định test cả `3 Unit` và `5 Unit`).
  - **Dời SL hòa vốn (BE):** Khi giá đi thuận chiều đạt `be_trigger_units` (mặc định `3 Unit`), dời SL về `Entry ± fee_buffer` (hòa vốn thực kể cả phí).
  - **TP:** `tp_units` (mặc định test `3–5 Unit`).
  - **Thứ tự ưu tiên thận trọng trong cùng 1 nến:** Nếu cả SL và TP cùng nằm trong `[Low, High]` của 1 nến M5 → tính chạm **SL trước** để không bao giờ phóng đại lợi nhuận.
- **Lộ trình 20 bậc lãi kép (+8%/bậc):**
  - Bậc $k$ ($k = 1 \dots 20$): vốn chuẩn bậc $E_k = 10{,}000 \times 1.08^{k-1}$, mục tiêu lên bậc tiếp theo là $E_k \times 1.08$.
  - Khi Equity đạt mốc $E_{k+1}$, tự động tăng bậc và tính lại `Qty` theo $E_{k+1}$.
- **Phí sàn Binance Futures:**
  - Hỗ trợ 2 chế độ: **Maker (`0.02%/chiều`)** và **Taker (`0.05%/chiều`)** + trượt giá cấu hình (`slippage_bps`).

---

## 5. Đầu ra & Kiểm chứng
1. Bảng thống kê thực nghiệm 1 năm theo **từng loại Sóng (1, 2, 3, 4+)**, **từng Coin**, **mức SL/TP** (hiển thị rõ Winrate %, Số lệnh, Profit Factor, Expectancy theo Unit, PnL ròng sau phí).
2. Kiểm chứng **In-Sample (6 tháng đầu)** để đo Winrate phân tầng (>70% / 51–70% / <51%) → chạy **Out-of-Sample (6 tháng sau)** theo đúng luật 20 bậc lãi kép để tránh tự lừa mình bằng overfitting.
3. Báo cáo HTML tương tác (`outputs/backtest_report.html`) + CSV lịch sử lệnh (`outputs/trades.csv`).
