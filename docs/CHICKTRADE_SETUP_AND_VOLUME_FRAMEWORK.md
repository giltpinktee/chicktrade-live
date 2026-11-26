# CHICKTRADE v5.1 — Khung Tiêu Chuẩn Tuyển Chọn Setup & Phân Bổ Volume

> Tài liệu chuẩn hóa tiêu chí đánh giá setup và cơ chế phân bổ Volume **V3 Barbell Alpha (`0.25× → 1.75×`)**, xây dựng từ phân tích tương quan định lượng trên **68 setup / 7 Coin** (`BTC, ETH, BNB, LINK, XRP, SOL, NEAR`) giai đoạn `2023-11 → 2026-10` dưới động cơ **Binance-Exact** (Max 5 lệnh mở, 1 lệnh/coin, ≤ 3 lệnh cùng chiều, LIMIT IOC thật, Gap-SL, Funding 8h thật).

---

## 1. Bản đồ 7 Chỉ số cốt lõi & Tương quan thực nghiệm

| Chỉ số | Tương quan với `MaxDD_U` riêng | Tương quan với `Recovery Factor (Net_U / MaxDD_U)` | Vai trò trong quyết định |
|---|---:|---:|---|
| **`EPR = TB_U / SL_U`** *(Edge-per-Risk)* | `-0.408` *(giảm DD)* | **`+0.735` (#1 tuyệt đối)** | Chỉ số quan trọng nhất: đo biên lãi trên mỗi `1U` rủi ro SL |
| **`PWR`** *(Profit Win Rate)* | `-0.468` *(giảm DD)* | **`+0.649` (#2)** | Tỉ lệ thắng **tiền** (`GP / (GP + GL)`), phản ánh đúng chiến lược 2 chân `TP1 + TP2` |
| **`Net_U`** *(Tổng lãi 36M)* | `+0.366` | `+0.613` (#3) | Quy mô lợi nhuận, nhưng phải đi kèm `EPR` & `N` mới có ý nghĩa |
| **`S5 Resilience`** *(`Net_S5 / Net_S4`)* | — | Bắt buộc `≥ 68%` | Đo khả năng sống sót khi trượt giá xấu thêm `+0.10U` cả vào lẫn SL |
| **`Max Consec SL`** *(Chuỗi SL dài nhất)* | **`+0.672` (#1 gây DD)** | `-0.280` | Thủ phạm lớn nhất gây sụt sâu → kèo chuỗi SL dài bị khóa volume nhỏ |
| **`N`** *(Số lệnh trong 36 tháng)* | **`+0.554` (#2 gây DD)** | `+0.252` | `< 15`: thiếu mẫu; `≥ 60` + biên mỏng: bẫy cày phí chiếm chỗ trần 5 lệnh |
| **`SL_U`** *(Độ rộng SL `5/6/7U`)* | **`+0.421` (#3 gây DD)** | `-0.325` | SL 7U dễ kéo DD sâu hơn SL 5U nếu không giảm hệ số volume |

---

## 2. Phễu 5 Cửa Ải Tuyển Chọn Setup (Đậu hay Loại?)

Mọi setup ứng viên phải vượt qua đủ **5 cửa ải**:

1. **Cửa 1 — Khung thời gian & Chuẩn khớp lệnh thật:**
   - Chỉ nhận `15m / 30m / 1h` (loại bỏ hoàn toàn `5m` do nhiễu phí/vi cấu trúc).
   - Bắt buộc chạy trên động cơ **Binance-Exact** (warmup từ `2021-01`, LIMIT IOC `±0.25U` thật, SL trước TP, Gap-Through SL, không ăn kép TP1+TP2 cùng nến, Funding 8h lịch sử thật).
2. **Cửa 2 — Ngưỡng lợi nhuận tối thiểu (`S4 & S5`):**
   - `N ≥ 15`, `PWR ≥ 53.5%`, `PF ≥ 1.15`, `Net_U ≥ +12.0U`, `TB_U ≥ +0.25U`, và **`Net_S5 > 0`**.
3. **Cửa 3 — Kháng trượt giá & Chống bẫy cày phí (Argon Gate):**
   - `S5 Resilience ≥ 68%` và `PF_S5 ≥ 1.15`.
   - `EPR ≥ 0.095`, `PF ≥ 1.22`, `TB_U ≥ 0.45U`.
   - Anti-Fee-Grind: Nếu `N ≥ 60` mà `EPR < 0.11` hoặc `TB_U < 0.65U` hoặc `PF < 1.28` → **LOẠI**.
4. **Cửa 4 — Sống sót ở chế độ gần nhất (`Y3 Survival` → Ghế Dự Bị `BENCH_WATCH`):**
   - Nếu 12 tháng gần nhất (`Y3`) bị `Net_Y3 ≤ -15.0U` và `PF_Y3 < 0.70` → **KHÔNG xóa vĩnh viễn**, mà đưa vào trạng thái **Tạm Dừng Theo Dõi (`BENCH_WATCH`)** để ghi nhận lệnh nháp tương lai chờ hồi sinh.
5. **Cửa 5 — Hợp rổ (Portfolio Fit dưới trần Max 5 lệnh):**
   - Ghép vào rổ hiện tại phải giúp **tăng Calmar hoặc giảm MaxDD** (không gây trùng nhịp xấu hoặc chiếm slot của siêu phẩm).

---

## 3. Phân Loại 6 Hạng Setup & Mức Volume Chuẩn (`V3 Barbell Alpha`)

| Hạng | Tên gọi | Volume | Tiêu chí định lượng bắt buộc | Vai trò trong danh mục |
|---|---|---:|---|---|
| **`S++`** | Siêu Phẩm | **`1.75×`** | `PWR ≥ 69.0%` · `EPR ≥ 0.28` · `N ≥ 25` | Đầu tàu lợi nhuận chính (mẫu lớn, biên dày, DD riêng nhỏ) |
| **`S+`** | Thấy Là Quất | **`1.50×`** | `PWR ≥ 65.5%` · `EPR ≥ 0.22` · `N ≥ 25` | Nhóm chủ lực chất lượng cao |
| **`A`** | Cốt Lõi Chuẩn | **`1.00×`** | `PWR ≥ 61.5%` · `EPR ≥ 0.17` · `N ≥ 20` | Xương sống giữ nhịp tăng trưởng ổn định |
| **`B`** | Vệ Tinh Khá | **`0.75×`** | `PWR ≥ 58.5%` · `EPR ≥ 0.13` | Bổ sung cơ hội đều đặn |
| **`C`** | Đánh Nhẹ / SL Rộng | **`0.50×`** | `EPR ≥ 0.10` | Ăn thêm lãi dài hạn nhưng khóa rủi ro nhỏ (ghìm kèo SL 7U / chuỗi SL dài) |
| **`D`** | Thăm Dò / Hạ Lạnh | **`0.25×`** | `EPR < 0.10` hoặc dính `4 SL` liên tiếp | Cầm máu khi setup vào chuỗi lạnh |

### Các lớp bảo vệ động đi kèm:
- **Cold-Only Cooling:** Dính `3 SL` liên tiếp → hạ trần xuống `0.50×`; `4 SL` liên tiếp → hạ xuống `0.25×` tới khi thắng lại (**tuyệt đối không bơm nóng vượt trần**).
- **Lọc bão (`ATR_ratio ≥ 1.35`):** Tự động chia đôi `×0.5` volume.
- **Khóa cứng theo Bậc S2 (`+8%` lên bậc / `-10%` lùi bậc):** Volume cố định trong từng bậc để giữ trọn sức gỡ lỗ.
- **Tự Học Chậm (`core/live_learning.py`):** Lưu lịch sử lệnh thật vào `core/live_learning_state.json`, kết hợp ký ức nền 3 năm (`prior = 60`), chỉ đổi bậc khi tích lũy đủ `≥ 15` lệnh thật.
- **Vòng Đời Ghế Dự Bị & Hồi Sinh (`ACTIVE ↔ BENCH_WATCH ↔ PROBATION`):**
  - **Đưa lên Ghế Dự Bị (`ACTIVE → BENCH_WATCH`):** Khi rớt Y3 hoặc `10` lệnh thật gần nhất bị `Net ≤ -12U` & `PF < 0.75`. Trong trạng thái này, **tuyệt đối không vào tiền thật (`vol = 0.0×`)**, nhưng vẫn ghi nhận 100% lệnh nháp tương lai (`record_shadow_trade`).
  - **Khởi động lại Thử Việc (`BENCH_WATCH → PROBATION`):** Khi `6` lệnh nháp gần nhất chứng minh hồi sinh (`Net ≥ +6U`, `PWR ≥ 58%`, `PF ≥ 1.35`, lệnh cuối thắng) → mở lại đánh thật ở mức **`0.50×`** trong `5` lệnh đầu.
  - **Tốt nghiệp về đánh chính (`PROBATION → ACTIVE`):** Nếu `5` lệnh thử việc kết thúc có lãi (`Net > 0`) → trả lại đúng bậc Volume V3 Barbell; nếu tái lỗ `≤ -8U` → đẩy lại về `BENCH_WATCH` ngay.

---

## 4. Năm Bài Học Cốt Lõi Khi Tuyển Setup & Đi Volume

1. **Ưu tiên `EPR + PWR`, không nhìn mỗi Winrate (`WR`) hay Tổng lãi (`Net_U`):** Kèo `WR 49%` nhưng `EPR` dày vẫn là siêu phẩm `1.75×`; kèo tổng lãi lớn nhờ nhồi lệnh mỏng (`N > 100`) dễ phá hỏng toàn rổ.
2. **Bảo vệ thống kê cho mẫu nhỏ (`N < 25`):** Không bao giờ cấp `1.50×–1.75×` cho setup dưới `25` lệnh trong 3 năm.
3. **Ghìm kèo tần suất cao & SL 7U về `0.50×`:** Giảm thiểu nguồn gây Drawdown lớn nhất (`MaxSL` & `SL_U`).
4. **Chiến lược Barbell (Tạ đòn) vượt trội dàn đều:** Giữ Volume tĩnh trung bình thấp (`0.82×`), nhưng tập trung hỏa lực vào top setup `S++/S+` giúp tăng gấp đôi lợi nhuận mà DD lại giảm.
5. **Tách biệt tối ưu Hiệu suất (`Volume`) và chọn Khẩu vị Rủi ro (`Bao Giá`):**
   - `Volume V3 Barbell` đã đẩy `Calmar` và `PF` (`1.66`) lên vùng tối ưu.
   - `Bao Giá` quyết định trực tiếp biên độ Drawdown theo tỷ lệ tuyến tính (`DD ≈ 11.56% × 400 / Bao_Giá`):
     - **Bao `400G`:** DD `~11.5%` (`$352k`, PF `1.66`)
     - **Turbo `320G–350G`:** DD `~13.3%–14.3%` (`$442k–$491k`)
     - **Turbo `250G`:** DD `~19%` (`$618k`)
