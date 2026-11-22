# Hướng Dẫn Triển Khai CHICKTRADE v3.0 Paper-Test $10,000 ($0/Tháng · 5 Phút Xong)

Kiến trúc miễn phí 100%:
- **GitHub Actions (Cron 15 phút):** Chạy `cloud_bot/paper_engine_v3.py`, kéo nến Binance Futures realtime, cập nhật `docs/data/live_dashboard.json` và bắn thông báo Telegram khi có lệnh.
- **GitHub Pages (`docs/index.html`):** Giao diện Telegram Mini App Popup + Web Dashboard (tự gọi giá Binance Futures mỗi **5 giây** để nhảy PnL nổi & hiển thị bảng `1U / SL / TP1 / TP2 / Dời SL (%)` kèm mũi tên `▲ / ▼`).
- **Google Apps Script (`cloud_bot/telegram_gas_webhook.gs`):** Bot Webhook 24/7 để bạn bè tự gõ `/start`, `/menu`, `/status`, `/orders`, `/1u` và mở Mini App ngay trong Telegram (+ tùy chọn ghi sang Google Sheets).

---

## Bước 1: Tạo Bot Telegram với `@BotFather` (1 phút)
1. Mở Telegram, chat với `@BotFather` → gõ `/newbot`, đặt tên & username (ví dụ `chicktrade_v3_paper_bot`).
2. Lưu lại **Bot Token** (dạng `123456789:AAH...`).
3. Thêm Bot vào **nhóm Telegram của bạn & bạn bè**, nhắn 1 tin bất kỳ rồi mở `https://api.telegram.org/bot<BOT_TOKEN>/getUpdates` để lấy `chat.id` của nhóm (thường là số âm `-100...`).

## Bước 2: Đẩy lên GitHub & Bật GitHub Pages (2 phút)
1. Push repository lên GitHub (nếu dùng Public repo thì GitHub Actions & Pages **miễn phí không giới hạn**).
2. Vào **Settings → Pages**:
   - **Source:** `Deploy from a branch`
   - **Branch:** `main` · Folder: `/docs` → nhấn **Save**.
3. Sau ~1 phút, link Web/Mini App của bạn sẽ có dạng:
   `https://<TEN_GITHUB>.github.io/<TEN_REPO>/`

## Bước 3: Thêm Secrets & Variables cho GitHub Actions (1 phút)
Vào **Settings → Secrets and variables → Actions**:
- Tab **Secrets** → *New repository secret*:
  - `TELEGRAM_BOT_TOKEN`: Token từ BotFather
  - `TELEGRAM_CHAT_ID`: ID nhóm Telegram nhận thông báo
- Tab **Variables** → *New repository variable*:
  - `WEBAPP_URL`: `https://<TEN_GITHUB>.github.io/<TEN_REPO>/`
  - `TURBO_ENABLED`: `0` (mặc định Bao 400G an toàn) hoặc `1` (bật Auto-Turbo 250G)

Vào tab **Actions → CHICKTRADE v3.0 Free Paper Bot → Run workflow** để kiểm tra chạy tay lần đầu.

## Bước 4: Cài Google Apps Script làm Bot Tương Tác 24/7 + Google Sheets (1 phút)
1. Mở Google Sheets mới (hoặc vào thẳng `https://script.google.com/` → **New project**).
2. Dán toàn bộ nội dung file `cloud_bot/telegram_gas_webhook.gs` vào `Code.gs`.
3. Điền 3 dòng trong `CONFIG` (hoặc vào *Project Settings → Script Properties*):
   - `BOT_TOKEN`
   - `DASHBOARD_JSON_URL`: `https://<TEN_GITHUB>.github.io/<TEN_REPO>/data/live_dashboard.json`
   - `WEBAPP_URL`: `https://<TEN_GITHUB>.github.io/<TEN_REPO>/`
4. Bấm **Deploy → New deployment**:
   - Type: **Web app**
   - Execute as: **Me**
   - Who has access: **Anyone** → bấm **Deploy** & cấp quyền.
5. Chọn hàm **`setWebhook`** trên thanh công cụ và bấm **Run** 1 lần duy nhất (`{"ok":true,"result":true}`).
6. *(Tùy chọn Google Sheets)*: Vào **Triggers → Add Trigger** cho hàm `syncHistoryToSheet` chạy mỗi 15 phút để tự lưu lịch sử lệnh sang Google Sheets.

Xong! Gửi `/start` hoặc `/menu` trong Telegram để bạn bè cùng xem Dashboard Live 5 giây & bảng tỷ lệ `1U / SL / TP1 / TP2 (%)`!
