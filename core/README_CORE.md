# CHICKTRADE — Cấu Trúc Hệ Thống Chuẩn & Phương Pháp Cốt Lõi

## 1. Cấu trúc Thư mục Thông minh (`e:\CHICKTRADE\`)

```text
e:\CHICKTRADE\
├── core\                              # [CHỐT CHÍNH THỨC — PRODUCTION CORE]
│   ├── README_CORE.md                 # Tài liệu này
│   ├── core_strategy_rules.json       # 25 setup chốt + Bayesian Prior 1Y + Kelly Volume
│   ├── adaptive_sizing.py             # Động cơ toán học Bayesian + Fractional Kelly tự cập nhật Volume
│   ├── build_core_package.py          # Script đồng bộ & tái tạo bộ chốt core
│   └── reports\
│       ├── chicktrade_6coin_master_report.html  # Master HTML Dashboard (7 Tabs: Tổng quan + 6 Coin 5m-1h)
│       └── adaptive_volume_matrix.csv           # Bảng tổng hợp Volume Toán học cho 25 setup
├── output_test\                       # [KẾT QUẢ TEST & BACKTEST LỊCH SỬ — TÁCH RIÊNG]
│   ├── per_coin_study\                # Thống kê Chạm 1-6 riêng từng coin
│   └── *.csv / *.html                 # Toàn bộ file kết quả chạy thử nghiệm
├── research\                          # [MÃ NGUỒN NGHIÊN CỨU & BACKTEST]
│   └── run_*.py                       # Các script nghiên cứu MTF, Storm Guard, 6-coin Swarm
├── data\                              # Bộ tải & cache dữ liệu Binance miễn phí
├── strategy\                          # Chỉ báo EMA 34/89/200, ATR, Volume, đếm sóng
├── risk\                              # Quản lý vốn 20 bậc lãi kép (+8%/bậc)
├── engine\                            # Lõi mô phỏng Portfolio
├── report\                            # Trình xuất báo cáo HTML
├── tests\                             # Unit tests kiểm chứng lõi
├── docs\                              # Tài liệu thiết kế (specs) & kế hoạch (plans)
└── config.py                          # Cấu hình mặc định toàn cục
```

## 2. Cơ Chế Tự Học & Điều Chỉnh Volume Toán Học (`core/adaptive_sizing.py`)
- Kết hợp **Prior 1 năm quá khứ** (giới hạn trọng số tối đa `35 lệnh` để dữ liệu cũ không át dữ liệu mới) với **Dữ liệu đánh thực tế mới** (có trọng số suy giảm `0.96` ưu tiên lệnh gần nhất).
- Cập nhật liên tục **Xác suất thắng hậu nghiệm ($\hat{p}$)** và **Tỷ lệ thưởng/rủi ro ($\hat{b} = \text{AvgWin}/\text{AvgLoss}$)** → tính **Kelly Criterion**:
  $$f^* = \hat{p} - \frac{1 - \hat{p}}{\hat{b}}$$
- Quy đổi ra **Hệ số Volume thực tế** (có trần an toàn):
  - **`1.30×` (Kim cương S+):** Chỉ mở khóa khi `WR ≥ 76%` **VÀ** `SL_Full ≤ 22%`.
  - **`1.00× – 1.15×` (Core A+):** `WR ≥ 73%` & `Kelly ≥ 25%`.
  - **`0.40× – 0.95×` (A / B / C):** Tỷ lệ thuận theo Kelly thực nghiệm.
  - **`0.00×` (PAUSE — Tự ngắt):** Tự động tạm dừng setup khi `WR hậu nghiệm < 60%` hoặc `Kelly ≤ 2%`.

## 3. Quy Tắc Số Lệnh Mở Đồng Thời Theo Mức Bao Giá (`buffer_max_concurrent`)
| Mức Bao Giá | Tối đa lệnh mở cùng lúc | Rủi ro cực đoan nếu quét sạch SL 6U cùng lúc |
| :---: | :---: | :---: |
| **250 giá** | **2 lệnh** | `-4.8%` vốn |
| **300 giá** *(mặc định)* | **3 lệnh** | `-6.0%` vốn |
| **400 giá** | **5 lệnh** | `-7.5%` vốn |
| **500 giá** | **7 lệnh** | `-8.4%` vốn |
