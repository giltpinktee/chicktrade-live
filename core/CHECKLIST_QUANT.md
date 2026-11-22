# CHICKTRADE — CHECKLIST QUANT CHUẨN HÓA (12 TIÊU CHUẨN BẮT BUỘC)

Mọi đồng coin, khung thời gian và chiến lược mới trước khi đưa vào vận hành **CHICKTRADE** đều bắt buộc vượt qua đầy đủ **12 tiêu chuẩn định lượng** dưới đây. Không bỏ bước, không nới lỏng tham số bằng cảm tính.

---

## 1. Giao Dịch 2 Chiều (Long / Short) & Luật 1 Hướng / 1 Coin
- Hệ thống phải quét và kiểm định đồng đều cả **Long (+1)** và **Short (-1)** theo xu hướng HTF (Higher Timeframe) và cấu trúc sóng 3-EMA (`EMA34 / EMA89 / EMA200`).
- **Bất biến 1 lệnh / 1 coin**: Tại bất kỳ thời điểm nào, mỗi `symbol` chỉ được phép tồn tại **tối đa 1 vị thế đang mở** (`DUPLICATE_SYMBOL_OPEN` guard trong `bot/signal_engine.py`). Tuyệt đối cấm mở chồng lệnh cùng chiều hoặc ngược chiều trên cùng một đồng coin.

## 2. Quét Toàn Diện 4 Khung Thời Gian (`5m`, `15m`, `30m`, `1h`)
- Mỗi coin phải được quét đầy đủ toàn bộ lưới `5m`, `15m`, `30m`, `1h` kết hợp lọc xu hướng khung lớn (`5m/15m -> 1h`, `30m/1h -> 4h`) và bộ lọc sức mạnh xu hướng `ADX(14) >= 20`.
- Chỉ sử dụng **nến đã đóng hoàn toàn (strictly closed candles)** — tuyệt đối không dùng dữ liệu tương lai (zero-lookahead).

## 3. Giải Phẫu 4 Xác Suất Thoát Lệnh (`SL` / `TP1` / `Leg2-SL` / `TP2`)
- Mọi setup phải được bóc tách đầy đủ 4 kịch bản kết thúc thực tế:
  1. **Full SL (`P_SL`)**: Chạm cắt lỗ ban đầu ngay trước TP1.
  2. **TP1 Only (`P_TP1`)**: Chốt lời 100% hoặc chốt một phần tại TP1.
  3. **Leg2-SL / BE (`P_Leg2_SL`)**: Đã ăn TP1, phần còn lại quay đầu chạm SL dời của Leg 2.
  4. **Full TP2 (`P_TP2`)**: Đã ăn TP1 và phần còn lại chạy trọn tới TP2.
- Quy tắc xử lý nến xung đột (**Conservative SL-First**): Nếu cùng một cây nến chạm cả SL và TP thì **bắt buộc ghi nhận là SL trước**.

## 4. Bất Biến Zero Fee-Leak (Không Rò Rỉ Phí & Trượt Giá)
- Toàn bộ backtest, paper và live phải tính đầy đủ phí sàn Binance USD-M Futures (**Maker 0.02%**, **Taker 0.05%**) và trượt giá (**Slippage >= 1 bps**).
- Khi dời SL Leg 2 về hòa vốn (`leg2_sl_u > 0` / Breakeven), giá `leg2_sl_price` bắt buộc cộng thêm `fee_pad = maker_fee + taker_fee + slippage` để đảm bảo đóng lệnh hòa vốn **không bao giờ bị âm ròng vì phí**.

## 5. Tự Động Chọn Lọc Trong 7 Kiến Trúc Thoát Lệnh (7 Exit Architectures)
- Khi onboard mỗi setup `(coin, tf, setup_code)`, bộ tối ưu phải benchmark đầy đủ **7 kiến trúc thoát lệnh** (nhóm single-leg: `ONLY_TP1`, `ONLY_1R`; nhóm partial 2-leg: `PARTIAL_NO_MOVE`, `PARTIAL_MOVE_SL` với các mức dời SL Leg 2: BE+fee, `-1U`, `+1U`, và cấu hình tỷ trọng TP1/TP2 tối ưu) rồi tự động chọn kiến trúc đạt Expectancy & Profit Factor sau phí tốt nhất, không ép cứng một kiểu thoát lệnh cho mọi coin.

## 6. Kiểm Định Ổn Định 2 Nửa Thời Gian (Dual-Half IS / OOS Stability)
- Dữ liệu lịch sử phải được chia đôi thành **Half 1 (In-Sample)** và **Half 2 (Out-of-Sample)** độc lập.
- Một setup chỉ được phê duyệt khi có lãi dương, kỳ vọng (`net_u > 0`), và tỷ lệ thắng/thua ổn định ở **cả 2 nửa thời gian** — loại bỏ ngay các setup chỉ thắng nhờ một đợt sóng đơn lẻ (anti-overfitting).

## 7. Điều Tiết Khối Lượng Động: EMA Risk Multiplier + Bayesian-Kelly Live
- Khối lượng lệnh thực tế kết hợp 2 tầng:
  - **EMA Risk Multiplier**: Phân bổ rủi ro theo độ sâu sóng (`EMA34 = 1.00x`, `EMA89 = 0.95x`, `EMA200 = 0.90x`).
  - **Bayesian-Kelly Adaptive Sizer** (`core/adaptive_sizing.py`): Cập nhật xác suất hậu nghiệm (`posterior_wr`) sau từng lệnh thực tế từ `prior_1y` + chuỗi `live_outcomes` lưu trong SQLite. Setup đang "lạnh" tự động hạ hệ số volume và tự ngắt (`BAYESIAN_KELLY_PAUSE`) khi Kelly âm.
- Giới hạn tổng hệ số mở cùng lúc: `max_open_mult_cap <= 2.4`.

## 8. Bộ Điều Khiển Buffer Động Theo Drawdown (`AdaptiveBufferController`)
- Triển khai tại `core/adaptive_buffer.py` và tích hợp trực tiếp vào `bot/signal_engine.py`:
  - **Mục tiêu Drawdown**: Giữ `Max DD <= 12%`, trần cứng `15%`, ngưỡng khẩn cấp `18%`.
  - **Khi `|DD| > 12%`**: Tự động nhân tỷ lệ nới buffer `scaled = current_buffer * (|DD| / 12%)`, làm tròn bước `10G`, chặn trần `500G`.
  - **Khi `|DD| >= 15%`**: Nhảy lập tức tối thiểu về `max(base_buffer, 400G)` để chặn chảy máu tài khoản; nếu `|DD| >= 18%` đẩy thẳng lên **`500G`**.
  - **Ánh xạ số lệnh đồng thời (`max_concurrent`)**: `<= 250G -> 2`, `<= 300G -> 3`, `<= 400G -> 5`, `> 400G -> 7`.

## 9. Auto-Turbo ON/OFF Kèm Sàn Cứng Bảo Vệ Vốn `$11,000` (`1.10x`)
- **Hysteresis kích hoạt**: Chỉ bật Auto-Turbo (hạ buffer về `250G` hoặc `230G` khi lập đỉnh vốn mới `equity >= peak_equity`) khi `equity >= 1.25x` vốn gốc (`$12,500`), và tự động tắt hoàn toàn khi `equity < 1.15x` (`$11,500`).
- **Sàn cứng quét SL đồng thời (`1.10x` / `$11,000` trên vốn `$10,000`)**: Trước khi vào bất kỳ lệnh Turbo nào, hệ thống tính:
  `worst_case_equity = equity - (existing_open_sl_risk_usd + candidate_turbo_sl_risk_usd)`.
  Nếu `worst_case_equity < 1.10 * initial_capital`, **lập tức hủy Turbo và lùi về `400G`** (`FLOOR_GUARD`).

## 10. Lá Chắn Bão 3 Lớp (3-Layer Storm Guard) & Phanh Rủi Ro Ngày
- **Lớp 1 (Local Volatility)**: Chặn vào lệnh khi `ATR_ratio >= 1.8x` hoặc xuất hiện nến quét râu (`Sweep Candle`).
- **Lớp 2 (BTC Macro Whipsaw)**: Chặn toàn bộ altcoin khi BTC rơi vào trạng thái giật 2 đầu (`Kaufman ER(10) < 0.15` & `ATR_ratio >= 1.2x`).
- **Lớp 3 (Cooldown & Daily Stops)**: Kích hoạt thời gian nghỉ cứng **4 giờ** lưu bền vững trong SQLite; giảm 50% size khi `elevated_vol` / `DD <= -6%` / `2 SL liên tiếp`; ngắt giao dịch trong ngày khi chạm `3 Full SL/ngày` hoặc lỗ ngày `<= -5%`.

## 11. Cô Lập Thư Mục Riêng Cho Từng Coin (`coins/<SYMBOL>/`)
- Mỗi đồng coin bắt buộc có không gian lưu trữ độc lập tại `coins/<SYMBOL>/` chứa toàn bộ dữ liệu nến, kết quả quét, cấu hình rule riêng và báo cáo kiểm định.
- Không trộn lẫn file tạm giữa các coin; chỉ những rule đã vượt đủ tiêu chuẩn mới được hợp nhất vào `core/core_strategy_rules.json`.

## 12. Cổng Forward-Test Paper / Testnet Trước Khi Chạy Live
- Mọi thay đổi về chiến lược, buffer, sizing hoặc broker bắt buộc:
  1. Vượt 100% bộ unit test tự động (`python -m unittest discover -s tests -v`).
  2. Chạy kiểm chứng thực tế trên `EXECUTION_MODE=PAPER` và `EXECUTION_MODE=TESTNET` (xác minh khớp lệnh, dời SL Leg 2, ghi nhận SQLite khi restart, cảnh báo Telegram).
  3. Chỉ kích hoạt `EXECUTION_MODE=LIVE` khi đã bật khóa an toàn `LIVE_CONFIRMED=true` và đối chiếu đúng số dư, đòn bẩy, bước giá sàn Binance.

---

### 13. Cơ Chế Pause & Đánh Lại Tự Động (`Shadow-Win Resume` — Điểm Ngọt Thực Nghiệm)
- **Kết quả thực nghiệm 12 kịch bản (`research/run_pause_resume_sweetspot.py`):**
  - **Pause cấp Setup đơn lẻ** quá chậm (1 setup trên 15m/30m/1h ít khi ra 2–3 SL cùng ngày → chỉ lọc được 4–13 lệnh/năm).
  - **🏆 Điểm ngọt tuyệt đối — `PORTFOLIO Pause: 2 SL liên tiếp → Chế độ Shadow-Win (Đánh nháp ngầm)`:**
    - Khi toàn danh mục dính **2 lệnh SL toàn phần liên tiếp** → Bot tự động chuyển sang **Chế độ Đánh Nháp Ngầm (không vào tiền thật)**.
    - Bot vẫn quét tín hiệu bình thường trong bóng tối; **ngay khi có 1 lệnh thắng ngầm (`1 Shadow Win`)** (hoặc hết tối đa `72 giờ`) → **tự động mở khóa đánh tiền thật trở lại**!
    - Hiệu quả thực nghiệm (Bao `400G`, 46 setups): **Calmar nhảy hơn 2× (`36.6 → 80.38`)**, **Lãi 12T vọt từ `+410%` → `+740.8%`**, **OOS 6T `+312.5%`**, **MaxDD giảm từ `-11.19%` → `-9.22%`**, **PF tăng `1.63 → 1.86`**!
  - **Lớp phụ cấp Coin:** Riêng từng coin nếu dính **2 SL liên tiếp** có thể nghỉ riêng coin đó `12h–48h` (Calmar `52.9`, OOS MaxDD `-7.60%`).

---

### Mục 14: Bộ 6 Luật Khép Kín Chọn Điểm Ngọt Chuẩn (v2.3 — 75 Setup)
1. **Lọc khung `5m`:** Loại bỏ chống trượt giá & phí sàn (chỉ giữ ngoại lệ đã chứng minh thắng cả 2 nửa năm & dư biên phí như `LINKUSDT 5m EMA89C2`).
2. **Đánh Chính (`CORE` / `CORE_EXP`, `0.75×–1.15×`, tính vào phanh `2-SL`):** `15m–1h`, `N ≥ 8`, `PF_IS ≥ 1.15`, `PF_OOS ≥ 1.15`, `Thắng Tiền (Profit WR) ≥ 58%`, `WR lệnh ≥ 60%`.
3. **Thăm Dò (`SCOUT` / `SCOUT_EXP`, `0.45×`, KHÔNG kích hoạt `2-SL`):** `15m–1h`, `N ≥ 8`, `PF_IS ≥ 1.05`, `PF_OOS ≥ 1.05`, `Thắng Tiền ≥ 55.5%`.
4. **Sniper Hiếm Lệnh (`SNIPER`, `0.45×`, KHÔNG kích hoạt `2-SL`):** `15m–1h`, `N = 2..7` lệnh/năm, `WR lệnh ≥ 60%`, `Thắng Tiền ≥ 57.5%`, không lỗ nặng ở nửa nào.
5. **Thích Ứng 6T Cuối (`HOT_ADAPTIVE`, `0.40×–0.45×`, KHÔNG kích hoạt `2-SL`):** Cho phép 6T đầu yếu, bắt buộc `PF_OOS ≥ 1.50`, `Thắng Tiền ≥ 60%`, `WR lệnh ≥ 60%` (tự ngắt nếu live tụt `< 60%`).
6. **LOẠI TUYỆT ĐỐI (`SKIP`):** Mọi setup gãy 6T gần nhất (`PF_OOS < 1.0`) hoặc lãi quá mỏng (`Thắng Tiền < 55%`).
7. **Chỉ số Winrate chuẩn:** Dùng **Winrate Lợi Nhuận (`Profit WR = GP / (GP + GL)`)** làm thước đo sức kiếm tiền (mốc `> 50%` chắc chắn có lãi, giải quyết bất đối xứng `SL ≠ TP`), hiển thị song song `WR Số Lệnh %` và 4 nhánh `% (SL gốc · Dời/QuaySL · TP1 · TP2)`.
