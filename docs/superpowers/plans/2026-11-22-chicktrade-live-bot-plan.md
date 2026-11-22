# CHICKTRADE — Kế Hoạch Triển Khai Bot Giao Dịch 6 Coin Chuẩn Thực Chiến (Final Plan)

**Ngày chốt:** 2026-11-22  
**Thư mục dự án:** `e:\CHICKTRADE`  
**Trạng thái:** Sẵn sàng triển khai từ bộ thông số đồng thuận Swarm (`Coordinator + Gemini 4 Argon High`)

---

## 1. Phạm Vi & Thông Số Chốt Của Hệ Thống

- **Danh mục 6 Coin:** `BTCUSDT`, `ETHUSDT`, `BNBUSDT`, `LINKUSDT`, `XRPUSDT`, `SOLUSDT` (Binance USD-M Futures).
- **25 Setup Bền Vững Cả 2 Nửa Năm:** Lưu cứng trong `bot/strategy_rules.json` (khớp 100% bảng [final_6coin_selected_setups.csv](file:///e:/CHICKTRADE/outputs/final_6coin_selected_setups.csv)):
  - **BTCUSDT (5 setups):** `SL 5U`, `TP1 3.5–4.0U`, `SL chân 2 = Entry - 2U`, `TP2 12U` (`M15/M30/H1`).
  - **ETHUSDT (5 setups) & BNBUSDT (6 setups):** `SL 6U`, `TP1 3.5–4.0U`, `SL chân 2 = Entry - 2U`, `TP2 10U`.
  - **LINKUSDT (3 setups), XRPUSDT (4 setups), SOLUSDT (2 setups):** `SL 6U`, `TP1 3.5U`, `SL chân 2 = Entry - 2U`, `TP2 8U`.
- **Quản Lý Vốn & Lãi Kép:**
  - Vốn khởi điểm cấu hình được (mặc định `\$10,000`), **20 bậc lãi kép (+8%/bậc)**.
  - Mức bao giá mặc định **`300 giá`** *(điểm ngọt: MaxDD ≤ 10.3%; cấu hình 1 dòng đổi sang `250` hoặc `400` giá bất cứ lúc nào)*.
  - Phân hạng Volume theo chất lượng setup: **A+ (`1.00×`)**, **A (`0.80×`)**, **B (`0.65×`)**, **C (`0.50×`)**.
  - Giới hạn vận hành: **Tối đa 3 lệnh mở cùng lúc** (1 lệnh/coin, tổng exposure `≤ 1.50×`), **tối đa 10 lệnh/ngày**.
- **Hệ Thống Bảo Vệ Storm Guard 3 Lớp:**
  1. **Lớp 1 (Volatility Spike & Wick Sweep):** Cooldown **4 giờ** nếu `ATR(14) ≥ 1.8× ATR_MA(50)` hoặc nến quét râu 2 đầu (`Range ≥ 2.8× ATR` hoặc `Range ≥ 2.0× ATR` với `Wick ≥ 70%`).
  2. **Lớp 2 (BTC Macro Choppiness):** Nếu `BTCUSDT` có `Kaufman ER(10) < 0.15` kèm `ATR ≥ 1.2×` → **Khóa vào lệnh mới toàn bộ 6 coin trong 4 giờ**.
  3. **Lớp 3 (Drawdown & Daily Circuit Breaker):** Giảm 50% size khi `ATR 1.35×–1.8×`; hạ về **`0.25×`** khi `DD > 6%` hoặc dính **2 SL liên tiếp**; khóa bot nghỉ hết ngày UTC khi dính **3 SL toàn phần** hoặc lỗ ngày **`-5%`**.
- **3 Chế Độ Chạy Trên Cùng 1 Kiến Trúc (`EXECUTION_MODE`):**
  - `PAPER` *(mặc định — 100% miễn phí, không cần API key)*: Lấy nến thật real-time từ Binance Public API, tự động khớp lệnh giả lập 2 chân (TP1 50% + dời SL chân 2 = `Entry - 2U` + TP2 50%), lưu sổ lệnh vào SQLite để không mất trạng thái khi tắt mở máy.
  - `TESTNET`: Kết nối `testnet.binancefuture.com` đặt lệnh Limit/Stop-Market/Take-Profit thật trên tài khoản tiền ảo Binance.
  - `LIVE`: Chỉ mở khóa khi cấu hình tường minh `LIVE_CONFIRMED=true` + API key có giới hạn IP & tắt quyền rút tiền.

---

## 2. Kiến Trúc Thư Mục Bot (`e:\CHICKTRADE\bot\`)

```text
e:\CHICKTRADE\
├── bot/
│   ├── __init__.py
│   ├── strategy_rules.json      # 25 setup chốt từng coin (SL, TP1, Leg2_SL, TP2, Vol_Mult)
│   ├── bot_config.py            # Đọc cấu hình từ .env (MODE, BUFFER_UNITS, TELEGRAM, API_KEY)
│   ├── market_feed.py           # Kéo nến đóng cửa real-time M15/M30/H1/H4 từ Binance Futures Public API
│   ├── signal_engine.py         # Quét tín hiệu 3-EMA + HTF/ADX + Storm Guard 3 lớp (tái sử dụng lõi đã backtest)
│   ├── state_store.py           # Lưu SQLite: vị thế đang mở (trạng thái TP1/Leg2), lịch sử lệnh, bậc lãi kép, cooldown
│   ├── execution/
│   │   ├── base.py              # Interface BrokerAdapter (place_order, close_partial, move_sl, get_price)
│   │   ├── paper_broker.py      # Bộ khớp lệnh Paper Trading real-time (có phí Maker/Taker + slippage)
│   │   └── binance_broker.py    # Bộ đặt lệnh Binance Futures Testnet / Live (ký HMAC-SHA256, quản lý TP1/SL/TP2)
│   ├── notifier.py              # Bắn thông báo kèo, khớp TP1/dời SL, đóng lệnh, cảnh báo Storm Guard qua Telegram + Terminal
│   └── main_bot.py              # Vòng lặp chính: đồng bộ đóng nến M15/M30/H1 + monitor SL/TP mỗi 15 giây
├── tests/
│   └── test_bot.py              # Unit & integration tests cho state_store, paper_broker, signal_engine, Storm Guard
└── .env.example                 # Mẫu file cấu hình an toàn (không commit key thật)
```

---

## 3. Lộ Trình Triển Khai Từng Bước (4 Giai Đoạn)

### Giai đoạn 1: Khóa Quy Tắc & Trạng Thái Bền Vững (`strategy_rules.json`, `bot_config.py`, `state_store.py`)
1. Xuất chính xác **25 setup** từ `outputs/final_6coin_selected_setups.csv` sang `bot/strategy_rules.json` (kèm thông số từng coin).
2. Xây dựng `bot/bot_config.py` và `.env.example` (mặc định `MODE=PAPER`, `BUFFER_UNITS=300`, `INITIAL_CAPITAL=10000`, `MAX_CONCURRENT=3`, `MAX_DAILY_TRADES=10`).
3. Xây dựng `bot/state_store.py` dùng **SQLite (`data/chicktrade_state.db`)** lưu:
   - Vị thế đang mở (`symbol, tf, setup, side, entry, qty_total, qty_remaining, sl, tp1, tp2, tp1_filled, leg2_sl_moved, unit_size`).
   - Tiến độ 20 bậc lãi kép, số lệnh trong ngày, chuỗi SL, mốc hết hạn Storm Guard.
   - Đảm bảo **tắt máy / crash mở lại vẫn nhớ nguyên lệnh đang mở và không vào trùng nến**.

### Giai đoạn 2: Luồng Dữ Liệu Real-Time & Động Cơ Tín Hiệu (`market_feed.py`, `signal_engine.py`)
1. `market_feed.py`: Gọi Binance Futures Public API (`/fapi/v1/klines`) lấy **300 nến đã đóng cửa gần nhất** cho 6 coin trên `15m`, `30m`, `1h`, `4h` (có retry + xử lý rate limit, chỉ dùng nến đã đóng để **zero-repaint**).
2. `signal_engine.py`: Tái sử dụng đúng logic `EMA 34/89/200 + ADX(14) >= 20 + HTF Trend + Storm Guard 3 lớp` đã kiểm chứng trong `run_final_6coin_swarm.py`, đối chiếu trực tiếp với `strategy_rules.json` và trả về danh sách `ActionableSignal` kèm đầy đủ `Qty`, `SL`, `TP1`, `Leg2_SL (Entry - 2U)`, `TP2`.

### Giai đoạn 3: Bộ Khớp Lệnh 2 Chân & Thông Báo (`paper_broker.py`, `binance_broker.py`, `notifier.py`, `main_bot.py`)
1. `paper_broker.py`: Theo dõi giá Mark/Last real-time mỗi **15 giây**, tự động thực thi:
   - Chạm `SL` trước `TP1` → Đóng 100%, ghi nhận `SL_FULL`, cập nhật chuỗi SL & phí.
   - Chạm `TP1` → Đóng **50%**, tự động dời `SL` phần 50% còn lại về **`Entry - 2U`**, bắn thông báo *"Đã chốt TP1 + Dời SL về Entry-2U"*.
   - Chạm `TP2` hoặc `SL chân 2` → Đóng 50% còn lại, cập nhật Equity & kiểm tra nâng **Bậc lãi kép (+8%)**.
2. `binance_broker.py`: Triển khai đầy đủ API ký HMAC-SHA256 cho **Binance Futures Testnet** (`testnet.binancefuture.com`) & Live (tự động set đòn bẩy, làm tròn `stepSize`/`tickSize` theo `exchangeInfo`, đặt `LIMIT/MARKET` + `STOP_MARKET` + `TAKE_PROFIT_MARKET`).
3. `notifier.py`: In bảng trạng thái gọn đẹp ra Terminal + hỗ trợ gửi tin nhắn **Telegram Bot** (tự động bỏ qua êm đẹp nếu chưa cấu hình `TELEGRAM_BOT_TOKEN`).
4. `main_bot.py`: Hỗ trợ 2 cờ chạy CLI:
   - `python bot/main_bot.py --once` *(quét 1 vòng kiểm tra ngay lập tức — dùng để test & xem trạng thái 6 coin hiện tại)*.
   - `python bot/main_bot.py` *(chạy liên tục 24/7 theo nhịp đóng nến M15/M30/H1)*.

### Giai đoạn 4: Kiểm Thử Toàn Diện & Chạy Thử Nghiệm (`tests/test_bot.py`)
1. Viết unit test kiểm tra:
   - Khớp lệnh 2 chân (`TP1 → Dời SL = Entry - 2U → TP2` và `TP1 → Quét SL chân 2 vẫn lãi ròng`).
   - Khôi phục trạng thái SQLite sau restart.
   - Khóa Storm Guard, khóa Max 3 lệnh, khóa 3 SL/ngày, tự hạ Volume `0.25×` khi dính 2 SL liên tiếp.
2. Chạy thật `python bot/main_bot.py --once` kết nối trực tiếp Binance Public API để xác nhận kéo nến sống 6 coin thành công, tính đúng trạng thái Storm Guard & các thông số hiện tại.

---

## 4. Quy Trình Go-Live An Toàn (Khuyến Nghị)
1. **Tuần 1–2:** Chạy `MODE=PAPER` 24/7 (không cần API key), đối chiếu sổ lệnh SQLite + thông báo Telegram với biểu đồ TradingView.
2. **Tuần 3–4:** Chuyển sang `MODE=TESTNET` với tài khoản Binance Futures Testnet (tiền ảo) để kiểm tra độ trễ, làm tròn lot và trượt giá thực tế.
3. **Chạy thật:** Nạp vốn nhỏ (hoặc bật `BUFFER_UNITS=400` trước), sau 1 tháng khớp kỳ vọng mới nâng lên `300` hoặc `250` giá.
