/**
 * CHICKTRADE v3.0 — Free 24/7 Telegram Webhook + Google Sheets Sync (Google Apps Script)
 *
 * Huong dan nhanh:
 * 1. Tao Script moi tai https://script.google.com/
 * 2. Dien BOT_TOKEN, DASHBOARD_JSON_URL, WEBAPP_URL ben duoi (hoac luu vao Script Properties).
 * 3. Deploy -> New deployment -> Web app (Execute as: Me, Who has access: Anyone).
 * 4. Chay ham setWebhook() 1 lan duy nhat (hoac goi URL Telegram API setWebhook).
 */

const CONFIG = {
  BOT_TOKEN: PropertiesService.getScriptProperties().getProperty('BOT_TOKEN') || 'PASTE_TELEGRAM_BOT_TOKEN_HERE',
  DASHBOARD_JSON_URL: PropertiesService.getScriptProperties().getProperty('DASHBOARD_JSON_URL') || 'https://YOUR_GITHUB_USER.github.io/CHICKTRADE/data/live_dashboard.json',
  WEBAPP_URL: PropertiesService.getScriptProperties().getProperty('WEBAPP_URL') || 'https://YOUR_GITHUB_USER.github.io/CHICKTRADE/',
  SHEET_NAME: 'CHICKTRADE_Orders',
};

function doPost(e) {
  try {
    const update = JSON.parse(e.postData.contents);
    if (update.message && update.message.text) {
      handleMessage_(update.message);
    } else if (update.callback_query) {
      handleCallback_(update.callback_query);
    }
  } catch (err) {
    Logger.log('doPost error: ' + err);
  }
  return ContentService.createTextOutput('OK');
}

function doGet() {
  return ContentService.createTextOutput('CHICKTRADE v3.0 Telegram GAS Webhook is active.');
}

function handleMessage_(msg) {
  const chatId = msg.chat.id;
  const text = (msg.text || '').trim().split(/\s+/)[0].toLowerCase();

  if (text === '/start' || text === '/menu') {
    const welcome =
      '🤖 <b>CHICKTRADE v3.0 — Paper-Test $10,000 Live</b>\n' +
      'Chuẩn <code>U3_HTF_MACRO_ATR</code> · 58 Setups · S2 50 Bậc.\n\n' +
      'Chọn lệnh nhanh bên dưới hoặc bấm nút <b>📊 Mở Mini App Popup</b> để xem giá &amp; PnL nhảy realtime mỗi 5 giây!';
    sendTelegram_(chatId, welcome, buildMainMenuKeyboard_());
    return;
  }

  if (text === '/status') {
    sendTelegram_(chatId, formatStatusText_(fetchDashboard_()), buildMainMenuKeyboard_());
    return;
  }

  if (text === '/orders') {
    sendTelegram_(chatId, formatOrdersText_(fetchDashboard_()), buildMainMenuKeyboard_());
    return;
  }

  if (text === '/1u') {
    sendTelegram_(chatId, format1uRatesText_(fetchDashboard_()), buildMainMenuKeyboard_());
    return;
  }
}

function handleCallback_(cb) {
  const chatId = cb.message.chat.id;
  const data = cb.data || '';
  answerCallback_(cb.id);

  if (data === 'cmd_status') {
    sendTelegram_(chatId, formatStatusText_(fetchDashboard_()), buildMainMenuKeyboard_());
  } else if (data === 'cmd_orders') {
    sendTelegram_(chatId, formatOrdersText_(fetchDashboard_()), buildMainMenuKeyboard_());
  } else if (data === 'cmd_1u') {
    sendTelegram_(chatId, format1uRatesText_(fetchDashboard_()), buildMainMenuKeyboard_());
  }
}

function fetchDashboard_() {
  const url = CONFIG.DASHBOARD_JSON_URL + '?t=' + Date.now();
  const res = UrlFetchApp.fetch(url, { muteHttpExceptions: true });
  if (res.getResponseCode() !== 200) {
    throw new Error('HTTP ' + res.getResponseCode() + ' fetching live_dashboard.json');
  }
  return JSON.parse(res.getContentText());
}

function formatStatusText_(d) {
  const acc = d.account || {};
  const st = d.stats || {};
  const mode = acc.turbo_active ? 'Auto-Turbo 250G ⚡' : 'Bao ' + (acc.buffer_units || 400) + 'G';
  const brake = acc.shadow_paused ? '🛑 ĐANG PHANH SHADOW-WIN' : '🟢 Bình thường (' + (acc.portfolio_consec_sl || 0) + '/2 SL)';
  return (
    '📊 <b>TỔNG QUAN TÀI KHOẢN PAPER ($10,000)</b>\n' +
    '• Cập nhật: <code>' + (d.updated_at_utc || '--') + '</code>\n' +
    '• Tài sản (Equity): <b>$' + Number(acc.equity || 0).toLocaleString('en-US', { minimumFractionDigits: 2 }) + '</b> (' + (Number(acc.roi_pct || 0) >= 0 ? '+' : '') + Number(acc.roi_pct || 0).toFixed(2) + '%)\n' +
    '• Bậc lãi kép S2: <b>Bậc ' + (acc.step || 0) + '/50</b> (Base $' + Number(acc.tier_capital || 10000).toLocaleString('en-US') + ')\n' +
    '• Chế độ: <b>' + mode + ' · ' + (acc.leverage || 20) + '×</b>\n' +
    '• Phanh 2-SL Core: ' + brake + '\n' +
    '• Lệnh đang mở: <b>' + (d.open_positions || []).length + '/5</b> | Đã đóng: <b>' + (st.total_trades || 0) + '</b>\n' +
    '• Winrate: <b>' + Number(st.wr_pct || 0).toFixed(1) + '%</b> | Thắng tiền: <b>' + Number(st.profit_wr_pct || 0).toFixed(1) + '%</b> | PF: <b>' + Number(st.pf || 0).toFixed(2) + '</b>\n' +
    '• MaxDD: <b>' + Number(acc.max_dd_pct || 0).toFixed(2) + '%</b>'
  );
}

function formatOrdersText_(d) {
  const open = d.open_positions || [];
  const closed = (d.closed_trades || []).slice(0, 5);
  let out = '⚡ <b>LỆNH ĐANG MỞ (' + open.length + '/5)</b>\n';
  if (open.length === 0) {
    out += '<i>Không có lệnh đang mở.</i>\n';
  } else {
    open.forEach(p => {
      const totPnl = Number(p.realized_pnl_usd || 0) + Number(p.unrealized_pnl_usd || 0);
      out += '• <b>' + p.symbol + '</b> (' + p.tf + ') <code>' + p.setup_code + '</code> [' + p.direction + ']\n' +
        '  Entry <code>' + p.entry_price + '</code> · PnL: <b>' + (totPnl >= 0 ? '+' : '') + '$' + totPnl.toFixed(2) + '</b> · SL <code>' + p.active_sl_price + '</code> · TP1 <code>' + p.tp1_price + '</code>\n';
    });
  }
  out += '\n📜 <b>5 LỆNH ĐÓNG GẦN NHẤT</b>\n';
  if (closed.length === 0) {
    out += '<i>Chưa có lệnh đã đóng.</i>';
  } else {
    closed.forEach(t => {
      const icon = Number(t.pnl_usd) > 0 ? '✅' : '❌';
      out += icon + ' <b>' + t.symbol + '</b> (' + t.tf + ') ' + t.outcome + ': <b>' + (Number(t.pnl_usd) >= 0 ? '+' : '') + '$' + Number(t.pnl_usd).toFixed(2) + '</b>\n';
    });
  }
  return out;
}

function format1uRatesText_(d) {
  const rates = d.live_u3_rates || [];
  let out = '📏 <b>BẢNG TỶ LỆ 1U CHUẨN U3_HTF_MACRO_ATR (KHUNG 15m / 1h)</b>\n';
  rates.forEach(r => {
    const d15 = (r.tf_details && r.tf_details['15m']) || {};
    const d1h = (r.tf_details && r.tf_details['1h']) || {};
    const arrow15 = Number(r.delta_pct_15m) > 0 ? '▲ +' : (Number(r.delta_pct_15m) < 0 ? '▼ ' : '— ');
    out += '• <b>' + r.coin.replace('USDT', '') + '</b> ($' + r.price + '): 15m <code>' + Number(r.u_pct_15m).toFixed(3) + '%</code> (' + arrow15 + Number(r.delta_pct_15m).toFixed(4) + '%) · SL <code>' + Number(d15.sl_pct || 0).toFixed(2) + '%</code> · TP2 <code>' + Number(d15.tp2_pct || 0).toFixed(2) + '%</code> | 1h <code>' + Number(r.u_pct_1h).toFixed(3) + '%</code>\n';
  });
  return out;
}

function buildMainMenuKeyboard_() {
  const rows = [];
  if (CONFIG.WEBAPP_URL && CONFIG.WEBAPP_URL.startsWith('https://')) {
    rows.push([{ text: '📊 Mở Mini App Popup (Live 5s)', web_app: { url: CONFIG.WEBAPP_URL } }]);
  }
  rows.push([
    { text: '💼 Trạng Thái ($10k)', callback_data: 'cmd_status' },
    { text: '⚡ Lệnh Mở / Lịch Sử', callback_data: 'cmd_orders' },
  ]);
  rows.push([{ text: '📏 Bảng 1U / SL / TP % Live', callback_data: 'cmd_1u' }]);
  return { inline_keyboard: rows };
}

function sendTelegram_(chatId, text, replyMarkup) {
  const payload = {
    chat_id: chatId,
    text: text,
    parse_mode: 'HTML',
    disable_web_page_preview: true,
  };
  if (replyMarkup) payload.reply_markup = replyMarkup;
  UrlFetchApp.fetch('https://api.telegram.org/bot' + CONFIG.BOT_TOKEN + '/sendMessage', {
    method: 'post',
    contentType: 'application/json',
    payload: JSON.stringify(payload),
    muteHttpExceptions: true,
  });
}

function answerCallback_(callbackId) {
  UrlFetchApp.fetch('https://api.telegram.org/bot' + CONFIG.BOT_TOKEN + '/answerCallbackQuery', {
    method: 'post',
    contentType: 'application/json',
    payload: JSON.stringify({ callback_query_id: callbackId }),
    muteHttpExceptions: true,
  });
}

/**
 * Ham tuy chon: Cai Trigger 15 phut hoac chay tay de ghi lich su lenh sang Google Sheets.
 */
function syncHistoryToSheet() {
  const d = fetchDashboard_();
  const ss = SpreadsheetApp.getActiveSpreadsheet();
  if (!ss) return;
  let sheet = ss.getSheetByName(CONFIG.SHEET_NAME);
  if (!sheet) {
    sheet = ss.insertSheet(CONFIG.SHEET_NAME);
    sheet.appendRow(['ID', 'Symbol', 'TF', 'Setup', 'Direction', 'Entry', 'Exit', 'Qty', 'VolMult', 'Outcome', 'PnL_USD', 'Opened_UTC', 'Closed_UTC']);
  }
  const existingIds = new Set(
    sheet.getLastRow() > 1 ? sheet.getRange(2, 1, sheet.getLastRow() - 1, 1).getValues().flat() : []
  );
  const closed = (d.closed_trades || []).slice().reverse();
  closed.forEach(t => {
    if (!existingIds.has(t.id)) {
      sheet.appendRow([t.id, t.symbol, t.tf, t.setup_code, t.direction, t.entry_price, t.exit_price, t.qty, t.vol_mult, t.outcome, t.pnl_usd, t.opened_utc, t.closed_utc]);
    }
  });
}

/**
 * Chay ham nay 1 lan duy nhat sau khi Deploy Web App de gan Webhook cho Bot Telegram.
 * Dien WEBHOOK_URL la URL Web App /exec nhan duoc tu buoc Deploy.
 */
function setWebhook() {
  const webAppExecUrl = ScriptApp.getService().getUrl();
  const res = UrlFetchApp.fetch('https://api.telegram.org/bot' + CONFIG.BOT_TOKEN + '/setWebhook?url=' + encodeURIComponent(webAppExecUrl));
  Logger.log(res.getContentText());
}
