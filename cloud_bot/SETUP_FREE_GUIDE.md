# Hướng Dẫn Clone & Triển Khai CHICKTRADE v3.3 ($0/Tháng · 5 Phút Xong)

Kiến trúc miễn phí 100% — **không chứa bất kỳ khóa bí mật (token/secret) nào trong mã nguồn**:
- **GitHub Actions (Cron 15 phút):** Chạy `cloud_bot/paper_engine_v3.py`, kéo nến Binance Futures realtime, xuất `docs/data/live_dashboard.json` + `docs/data/chart_candles.json` và bắn thông báo mở/đóng lệnh (kèm ảnh Chart PNG khi mở lệnh, tự Reply khi chạm TP1/đóng lệnh).
- **GitHub Pages (`docs/index.html`):** Giao diện Telegram Mini App Popup + Web Dashboard (cache RAM `chart_candles.json` vẽ tức thì `0ms`, ticker Binance Futures 5 giây nhảy PnL nổi & bảng `1U / SL / TP1 / TP2 (%)`).
- **Cloudflare Workers (`cloud_bot/cloudflare_worker.js`):** Bot Telegram Webhook `HTTP 200 OK` phản hồi `< 0.2s` (miễn phí 100,000 req/ngày) cho `/start`, `/status`, `/orders`, `/1u`, `/chart`, `/stats`, `/mute`, `/subscribe` và quản trị `/admin`, `/ban`, `/unban`.

---

## Bước 1: Tạo Bot Telegram với `@BotFather` (1 phút)
1. Mở Telegram, chat với `@BotFather` → gõ `/newbot`, đặt tên & username cho bot của bạn.
2. Lưu lại **Bot Token** (dạng `123456789:AAH...`).
3. Lấy **Chat ID** của bạn (hoặc nhóm nhận lệnh) bằng cách nhắn `/start` cho `@userinfobot` hoặc gọi `https://api.telegram.org/bot<BOT_TOKEN>/getUpdates`.

## Bước 2: Fork / Clone Repo & Bật GitHub Pages (1 phút)
1. Fork hoặc clone repo này lên tài khoản GitHub của bạn (Public repo giúp GitHub Actions & Pages miễn phí không giới hạn).
2. Vào **Settings → Pages**:
   - **Source:** `Deploy from a branch`
   - **Branch:** `master` (hoặc `main`) · Folder: `/docs` → bấm **Save**.
3. Link Web / Mini App của bạn sẽ có dạng:
   `https://<TEN_GITHUB>.github.io/<TEN_REPO>/`

## Bước 3: Triển Khai Telegram Bot Trên Cloudflare Workers (2 phút)
1. Đăng nhập [Cloudflare Dashboard](https://dash.cloudflare.com/) → **Workers & Pages → Create Worker** → đặt tên (ví dụ `chicktrade-bot`).
2. Bấm **Edit code**, dán toàn bộ nội dung file `cloud_bot/cloudflare_worker.js` → bấm **Deploy**.
3. Vào **Settings → Variables and Secrets** của Worker, thêm:
   - `BOT_TOKEN` *(Secret)*: Bot Token từ `@BotFather`
   - `BROADCAST_SECRET` *(Secret)*: Chuỗi mật khẩu bất kỳ tự đặt (dùng để xác thực broadcast từ GitHub Actions)
   - `OWNER_CHAT_ID` *(Variable)*: Chat ID Telegram của bạn (Admin dùng `/admin`, `/ban`, `/unban`)
   - `DASHBOARD_JSON_URL` *(Variable)*: `https://<TEN_GITHUB>.github.io/<TEN_REPO>/data/live_dashboard.json`
   - `CHART_CANDLES_JSON_URL` *(Variable)*: `https://<TEN_GITHUB>.github.io/<TEN_REPO>/data/chart_candles.json`
   - `WEBAPP_URL` *(Variable)*: `https://<TEN_GITHUB>.github.io/<TEN_REPO>/`
4. *(Khuyên dùng)* Vào **Settings → Bindings → Add KV Namespace** đặt tên biến `CHICKTRADE_KV` để lưu bền vững danh sách người đăng ký & liên kết Reply-to-Open.
5. Kích hoạt Webhook 1 lần duy nhất bằng cách mở link trên trình duyệt:
   `https://<TEN_WORKER>.workers.dev/set-webhook?secret=<BROADCAST_SECRET>`
   → Thấy `{"ok":true,...}` là thành công!

## Bước 4: Thêm Secrets & Variables cho GitHub Actions (1 phút)
Vào repo GitHub của bạn → **Settings → Secrets and variables → Actions**:
- Tab **Secrets** (*New repository secret*):
  - `TELEGRAM_BOT_TOKEN`: Bot Token từ `@BotFather`
  - `TELEGRAM_CHAT_ID`: Chat ID nhận thông báo chính
  - `GAS_BROADCAST_SECRET`: Giống hệt `BROADCAST_SECRET` đã đặt ở Cloudflare Worker
- Tab **Variables** (*New repository variable*):
  - `GAS_BROADCAST_URL`: `https://<TEN_WORKER>.workers.dev/`
  - `WEBAPP_URL`: `https://<TEN_GITHUB>.github.io/<TEN_REPO>/`
  - `TURBO_ENABLED`: `0` (mặc định Bao 400G an toàn) hoặc `1` (bật Auto-Turbo 250G)

Vào tab **Actions → CHICKTRADE v3.0 Free Paper Bot → Run workflow** để chạy thử lần đầu. Xong! Mở Telegram gõ `/start` để dùng ngay.
