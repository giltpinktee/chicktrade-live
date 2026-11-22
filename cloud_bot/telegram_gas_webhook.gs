/**
 * CHICKTRADE QUANT v3.3 — Free $0/mo Multi-User Telegram Webhook + Broadcast + Sheets Sync (GAS)
 *
 * Huong dan nhanh (1 phut):
 * 1. Tao du an moi tai https://script.google.com/ -> dan toan bo code nay vao Code.gs.
 * 2. Vao Project Settings -> Script Properties, them:
 *      - BOT_TOKEN: Token cua @chicktrade_v3_bot (lay tu @BotFather)
 *      - BROADCAST_SECRET: Chuoi bi mat bat ky (khop voi GitHub Secret GAS_BROADCAST_SECRET)
 *      - OWNER_CHAT_ID: (Tuy chon) Chat ID cua ban de khong gui trung khi GitHub Actions da gui rieng
 * 3. Deploy -> New deployment -> Web app (Execute as: Me, Who has access: Anyone).
 * 4. Copy URL Web App (/exec) dan vao GitHub Variable GAS_BROADCAST_URL, va chay ham setWebhook() 1 lan!
 */

const CONFIG = {
  BOT_TOKEN: PropertiesService.getScriptProperties().getProperty('BOT_TOKEN') || 'PASTE_TELEGRAM_BOT_TOKEN_HERE',
  DASHBOARD_JSON_URL:
    PropertiesService.getScriptProperties().getProperty('DASHBOARD_JSON_URL') ||
    'https://giltpinktee.github.io/chicktrade-live/data/live_dashboard.json',
  WEBAPP_URL:
    PropertiesService.getScriptProperties().getProperty('WEBAPP_URL') ||
    'https://giltpinktee.github.io/chicktrade-live/',
  BROADCAST_SECRET: PropertiesService.getScriptProperties().getProperty('BROADCAST_SECRET') || '',
  OWNER_CHAT_ID: PropertiesService.getScriptProperties().getProperty('OWNER_CHAT_ID') || '',
  SUBSCRIBERS_PROP_KEY: 'SUBSCRIBERS_JSON',
  SHEET_NAME: 'CHICKTRADE_Orders',
};

function doPost(e) {
  try {
    if (!e || !e.postData || !e.postData.contents) {
      return ContentService.createTextOutput('OK');
    }
    const payload = JSON.parse(e.postData.contents);

    // 1. Multi-user broadcast request from GitHub Actions paper_engine_v3.py
    if (payload.action === 'broadcast') {
      if (CONFIG.BROADCAST_SECRET && String(payload.secret || '') !== CONFIG.BROADCAST_SECRET) {
        Logger.log('Broadcast unauthorized: secret mismatch');
        return ContentService.createTextOutput(JSON.stringify({ ok: false, error: 'UNAUTHORIZED' })).setMimeType(
          ContentService.MimeType.JSON
        );
      }
      const res = broadcastToSubscribers_(
        String(payload.text || ''),
        String(payload.webapp_url || CONFIG.WEBAPP_URL),
        String(payload.exclude_chat_id || CONFIG.OWNER_CHAT_ID || '')
      );
      return ContentService.createTextOutput(JSON.stringify(res)).setMimeType(ContentService.MimeType.JSON);
    }

    // 2. Standard Telegram Bot Webhook update
    if (payload.message && payload.message.text) {
      handleMessage_(payload.message);
    } else if (payload.callback_query) {
      handleCallback_(payload.callback_query);
    }
  } catch (err) {
    Logger.log('doPost error: ' + err);
  }
  return ContentService.createTextOutput('OK');
}

function doGet() {
  const subs = getSubscribers_();
  const activeCount = Object.keys(subs).filter(k => !subs[k].muted).length;
  return ContentService.createTextOutput(
    'CHICKTRADE v3.3 Multi-User Telegram GAS Webhook active. Active subscribers: ' + activeCount
  );
}

// ---------- Subscriber Store (PropertiesService) ----------

function getSubscribers_() {
  const raw = PropertiesService.getScriptProperties().getProperty(CONFIG.SUBSCRIBERS_PROP_KEY);
  if (!raw) return {};
  try {
    return JSON.parse(raw) || {};
  } catch (e) {
    return {};
  }
}

function saveSubscribers_(subs) {
  PropertiesService.getScriptProperties().setProperty(CONFIG.SUBSCRIBERS_PROP_KEY, JSON.stringify(subs));
}

function upsertSubscriber_(chatObj, fromObj, mutedOverride) {
  const chatId = String(chatObj.id);
  const subs = getSubscribers_();
  const prev = subs[chatId] || {};
  const displayName =
    (chatObj.title && String(chatObj.title)) ||
    (fromObj && (fromObj.username ? '@' + fromObj.username : fromObj.first_name)) ||
    prev.name ||
    'Trader';
  subs[chatId] = {
    name: displayName,
    muted: typeof mutedOverride === 'boolean' ? mutedOverride : Boolean(prev.muted),
    joined_utc: prev.joined_utc || new Date().toISOString().slice(0, 16).replace('T', ' '),
  };
  saveSubscribers_(subs);
  return subs[chatId];
}

function broadcastToSubscribers_(text, webappUrl, excludeChatId) {
  if (!text) return { ok: true, sent: 0, skipped: 0 };
  const subs = getSubscribers_();
  const ids = Object.keys(subs);
  let sent = 0;
  let skipped = 0;
  let changed = false;

  ids.forEach(cid => {
    const rec = subs[cid];
    if (!rec || rec.muted === true) {
      skipped += 1;
      return;
    }
    if (excludeChatId && String(cid) === String(excludeChatId).trim()) {
      skipped += 1;
      return;
    }
    const markup = buildOpenAppInlineMarkup_(cid, webappUrl);
    const resp = sendTelegram_(cid, text, markup);
    if (resp && resp.ok) {
      sent += 1;
    } else if (resp && (resp.error_code === 403 || resp.error_code === 400)) {
      // User blocked bot or chat unavailable -> auto-mute to avoid retrying dead chats
      rec.muted = true;
      changed = true;
      skipped += 1;
    } else {
      skipped += 1;
    }
  });

  if (changed) {
    saveSubscribers_(subs);
  }
  return { ok: true, sent: sent, skipped: skipped };
}

// ---------- Command & Callback Routing ----------

function handleMessage_(msg) {
  const chatId = msg.chat.id;
  const rawText = (msg.text || '').trim();
  const firstToken = rawText.split(/\s+/)[0].toLowerCase().replace(/@[a-z0-9_]+$/i, '');

  // Map quick-reply keyboard labels to commands
  let cmd = firstToken;
  if (rawText === '📊 Tài sản') cmd = '/status';
  else if (rawText === '📂 Lệnh đang mở') cmd = '/orders';
  else if (rawText === '📐 Tỷ lệ 1U') cmd = '/1u';
  else if (rawText === '📈 Thống kê') cmd = '/stats';

  if (cmd === '/start' || cmd === '/menu') {
    upsertSubscriber_(msg.chat, msg.from, false);
    const welcome =
      '🤖 <b>CHICKTRADE QUANT v3.3 — Paper-Test $10,000 Live</b>\n' +
      'Chuẩn <code>U3_HTF_MACRO_ATR 4H</code> · 58 Setups · S2 Không Giới Hạn Bậc.\n' +
      '✅ <b>Đã bật nhận thông báo tự động</b> (Mở lệnh, TP1, Đóng lệnh, Phanh 2-SL, Tổng kết ngày/tuần).\n\n' +
      '<b>Lệnh tra cứu nhanh:</b>\n' +
      '• /status — Tài sản, Bậc S2 &amp; Phanh 2-SL\n' +
      '• /orders — Lệnh đang mở &amp; Dự kiến Win/Loss $\n' +
      '• /1u — Bảng tỷ lệ 1U % 6 Coin (15m/30m/1h)\n' +
      '• /stats — Hiệu suất Profit WR, PF &amp; 5 lệnh gần nhất\n' +
      '• /mute | /subscribe — Tạm tắt / Bật lại thông báo tự động';
    sendTelegram_(chatId, welcome, buildReplyKeyboard_(chatId));
    return;
  }

  if (cmd === '/subscribe') {
    upsertSubscriber_(msg.chat, msg.from, false);
    sendTelegram_(
      chatId,
      '🔔 <b>Đã BẬT thông báo lệnh tự động!</b>\nBạn sẽ nhận tin ngay khi Bot mở lệnh, chốt TP1, đóng lệnh, bật/mở phanh 2-SL và tổng kết sáng 07:00 VN.',
      buildInlineDashboardKeyboard_(chatId)
    );
    return;
  }

  if (cmd === '/mute') {
    upsertSubscriber_(msg.chat, msg.from, true);
    sendTelegram_(
      chatId,
      '🔕 <b>Đã TẠM TẮT thông báo tự động.</b>\nBạn vẫn có thể bấm nút hoặc gõ <code>/status</code>, <code>/orders</code>, <code>/1u</code>, <code>/stats</code> bình thường. Gõ /subscribe để bật lại.',
      buildInlineDashboardKeyboard_(chatId)
    );
    return;
  }

  if (cmd === '/status') {
    sendTelegram_(chatId, formatStatusText_(fetchDashboard_()), buildInlineDashboardKeyboard_(chatId));
    return;
  }

  if (cmd === '/orders') {
    sendTelegram_(chatId, formatOrdersText_(fetchDashboard_()), buildInlineDashboardKeyboard_(chatId));
    return;
  }

  if (cmd === '/1u') {
    sendTelegram_(chatId, format1uRatesText_(fetchDashboard_()), buildInlineDashboardKeyboard_(chatId));
    return;
  }

  if (cmd === '/stats') {
    sendTelegram_(chatId, formatStatsText_(fetchDashboard_()), buildInlineDashboardKeyboard_(chatId));
    return;
  }
}

function handleCallback_(cb) {
  const chatId = cb.message.chat.id;
  const data = cb.data || '';
  answerCallback_(cb.id);

  if (data === 'cmd_status') {
    sendTelegram_(chatId, formatStatusText_(fetchDashboard_()), buildInlineDashboardKeyboard_(chatId));
  } else if (data === 'cmd_orders') {
    sendTelegram_(chatId, formatOrdersText_(fetchDashboard_()), buildInlineDashboardKeyboard_(chatId));
  } else if (data === 'cmd_1u') {
    sendTelegram_(chatId, format1uRatesText_(fetchDashboard_()), buildInlineDashboardKeyboard_(chatId));
  } else if (data === 'cmd_stats') {
    sendTelegram_(chatId, formatStatsText_(fetchDashboard_()), buildInlineDashboardKeyboard_(chatId));
  }
}

// ---------- Data Fetch & Formatters (< 1 Mobile Screen) ----------

function fetchDashboard_() {
  const url = CONFIG.DASHBOARD_JSON_URL + '?t=' + Date.now();
  const res = UrlFetchApp.fetch(url, { muteHttpExceptions: true });
  if (res.getResponseCode() !== 200) {
    throw new Error('HTTP ' + res.getResponseCode() + ' fetching live_dashboard.json');
  }
  return JSON.parse(res.getContentText());
}

function fmtUsd_(val, signed) {
  const n = Number(val || 0);
  const absStr = Math.abs(n).toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 });
  if (signed) return (n >= 0 ? '+$' : '-$') + absStr;
  return '$' + absStr;
}

function fmtPx_(px) {
  const a = Math.abs(Number(px || 0));
  if (a >= 1000) return '$' + a.toLocaleString('en-US', { minimumFractionDigits: 1, maximumFractionDigits: 1 });
  if (a >= 10) return '$' + a.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 });
  if (a >= 1) return '$' + a.toFixed(3);
  return '$' + a.toFixed(4);
}

function formatStatusText_(d) {
  const acc = d.account || {};
  const st = d.stats || {};
  const eq = Number(acc.equity || 10000);
  const initCap = Number(acc.initial_capital || 10000);
  const netDiff = eq - initCap;
  const roiPct = Number(acc.roi_pct || 0);
  const step = Number(acc.step || 0);
  const tierCap = Number(acc.tier_capital || 10000);
  const buf = Number(acc.buffer_units || 400);
  const unitUsd = tierCap / Math.max(buf, 1);
  const curBase = 10000 * Math.pow(1.08, step);
  const nextEq = 10000 * Math.pow(1.08, step + 1);
  const progPct = Math.max(0, Math.min(100, ((eq - curBase) / Math.max(nextEq - curBase, 1)) * 100));
  const remUsd = Math.max(0, nextEq - eq);

  const mode = acc.turbo_active ? 'Auto-Turbo 250G ⚡' : 'Bao ' + buf + 'G';
  const brake = acc.shadow_paused
    ? '🛡️ ĐANG PHANH SHADOW-WIN'
    : '🟢 An toàn (' + (acc.portfolio_consec_sl || 0) + '/2 SL)';

  return (
    '💼 <b>TRẠNG THÁI QUỸ CHICKTRADE v3.3</b>\n' +
    '🕒 <code>' + (d.updated_at_utc || '--') + '</code>\n\n' +
    '• Tổng tài sản (Equity): <b>' + fmtUsd_(eq, false) + '</b> (' + (roiPct >= 0 ? '+' : '') + roiPct.toFixed(2) + '% · ' + fmtUsd_(netDiff, true) + ')\n' +
    '• Bậc lãi kép S2: <b>Bậc ' + step + ' (∞)</b> · 1U = <b>' + fmtUsd_(unitUsd, false) + '</b>\n' +
    '• Lên Bậc ' + (step + 1) + ' (' + fmtUsd_(nextEq, false) + '): đạt <b>' + progPct.toFixed(1) + '%</b> (còn +' + fmtUsd_(remUsd, false) + ')\n' +
    '• Chế độ: <b>' + mode + ' · ' + (acc.leverage || 20) + '× Cross</b>\n' +
    '• Phanh 2-SL Core: ' + brake + '\n' +
    '• Lệnh đang mở: <b>' + (d.open_positions || []).length + '/5</b> | Đã đóng: <b>' + (st.total_trades || 0) + '</b>'
  );
}

function formatOrdersText_(d) {
  const open = d.open_positions || [];
  let out = '📂 <b>LỆNH ĐANG MỞ (' + open.length + '/5)</b>\n';
  if (open.length === 0) {
    out += '<i>Hiện không có lệnh đang mở — hệ thống đang canh tín hiệu 24/7.</i>';
    return out;
  }
  open.forEach((p, idx) => {
    const totPnl = Number(p.realized_pnl_usd || 0) + Number(p.unrealized_pnl_usd || 0);
    const dirIc = p.direction === 'LONG' ? '🟢' : '🔴';
    const mustTag = p.is_must_order ? ' 🔥<b>THẤY LÀ QUẤT</b>' : '';
    const wrLive = Number(p.setup_live_wr_pct != null ? p.setup_live_wr_pct : p.wr_pct || 0).toFixed(1);
    const pwrLive = Number(p.setup_live_profit_wr_pct != null ? p.setup_live_profit_wr_pct : p.profit_wr_pct || 0).toFixed(1);
    const samples = p.setup_total_samples != null ? ' (' + p.setup_total_samples + ' lệnh)' : '';
    const expTp1 = fmtUsd_(p.expected_tp1_usd || 0, true);
    const expTp2 = fmtUsd_(p.expected_tp2_usd || 0, true);
    const expSl = fmtUsd_(p.expected_sl_usd || 0, true);
    const expSlPct = Number(p.expected_sl_eq_pct || 0).toFixed(2);

    out +=
      (idx > 0 ? '\n' : '') +
      dirIc + ' <b>' + p.symbol + ' · ' + String(p.tf || '').toUpperCase() + ' [' + p.setup_code + ']</b> ' + p.direction + mustTag + '\n' +
      '• Entry <code>' + fmtPx_(p.entry_price) + '</code> → Live <code>' + fmtPx_(p.current_price) + '</code> | PnL: <b>' + fmtUsd_(totPnl, true) + '</b>\n' +
      '• SL <code>' + fmtPx_(p.active_sl_price) + '</code> · TP1 <code>' + fmtPx_(p.tp1_price) + '</code> · TP2 <code>' + fmtPx_(p.tp2_price) + '</code>\n' +
      '• Dự kiến Win: <b>' + expTp1 + ' / ' + expTp2 + '</b> | Thua SL: <b>' + expSl + '</b> (' + expSlPct + '%)\n' +
      '• WR (3Y+Live): <b>' + wrLive + '%</b> · Thắng tiền: <b>' + pwrLive + '%</b>' + samples + '\n';
  });
  return out;
}

function format1uRatesText_(d) {
  const rates = d.live_u3_rates || [];
  let out = '📐 <b>BẢNG TỶ LỆ 1U CHUẨN 4H (U3_HTF_MACRO_ATR)</b>\n';
  if (rates.length === 0) {
    return out + '<i>Chưa có dữ liệu 1U.</i>';
  }
  rates.forEach(r => {
    const coin = String(r.coin || '').replace('USDT', '');
    const d15 = Number(r.delta_pct_15m || 0);
    const arrow = d15 > 0 ? '▲+' : d15 < 0 ? '▼' : '—';
    out +=
      '• <b>' + coin + '</b> (' + fmtPx_(r.price) + '): ' +
      '15m <code>' + Number(r.u_pct_15m || 0).toFixed(3) + '%</code> (' + arrow + d15.toFixed(4) + '%) · ' +
      '30m <code>' + Number(r.u_pct_30m || 0).toFixed(3) + '%</code> · ' +
      '1h <code>' + Number(r.u_pct_1h || 0).toFixed(3) + '%</code>\n';
  });
  return out;
}

function formatStatsText_(d) {
  const acc = d.account || {};
  const st = d.stats || {};
  const closed = (d.closed_trades || []).slice(0, 5);
  let out =
    '📈 <b>THỐNG KÊ HIỆU SUẤT TÍCH LŨY</b>\n' +
    '• Tổng lệnh đã đóng: <b>' + (st.total_trades || 0) + '</b> (' + (st.wins || 0) + ' Thắng)\n' +
    '• Thắng tiền (Profit WR): <b>' + Number(st.profit_wr_pct || 0).toFixed(1) + '%</b> | WR lệnh: <b>' + Number(st.wr_pct || 0).toFixed(1) + '%</b>\n' +
    '• Profit Factor: <b>' + Number(st.pf || 0).toFixed(2) + '</b> | MaxDD: <b>' + Number(acc.max_dd_pct || 0).toFixed(2) + '%</b>\n' +
    '• Lãi/Lỗ ròng đã chốt: <b>' + fmtUsd_(st.net_pnl_usd || 0, true) + '</b>\n\n' +
    '📜 <b>5 LỆNH ĐÓNG GẦN NHẤT:</b>\n';
  if (closed.length === 0) {
    out += '<i>Chưa có lệnh nào đóng.</i>';
  } else {
    closed.forEach(t => {
      const pnl = Number(t.pnl_usd || 0);
      const ic = pnl > 0 ? '✅' : '❌';
      out += ic + ' <b>' + t.symbol + '</b> (' + t.tf + ' ' + t.setup_code + ') ' + t.outcome + ': <b>' + fmtUsd_(pnl, true) + '</b>\n';
    });
  }
  return out;
}

// ---------- Keyboards & Telegram HTTP Helpers ----------

function isPrivateChat_(chatId) {
  return Number(chatId) > 0;
}

function buildOpenAppButton_(chatId, url) {
  if (!url || !url.startsWith('https://')) return null;
  if (isPrivateChat_(chatId)) {
    return { text: '📊 Mở Popup Theo Dõi Lệnh Live', web_app: { url: url } };
  }
  return { text: '📊 Mở Dashboard Live', url: url };
}

function buildOpenAppInlineMarkup_(chatId, url) {
  const btn = buildOpenAppButton_(chatId, url);
  if (!btn) return null;
  return { inline_keyboard: [[btn]] };
}

function buildReplyKeyboard_(chatId) {
  const rows = [
    [{ text: '📊 Tài sản' }, { text: '📂 Lệnh đang mở' }],
    [{ text: '📐 Tỷ lệ 1U' }, { text: '📈 Thống kê' }],
  ];
  if (isPrivateChat_(chatId) && CONFIG.WEBAPP_URL && CONFIG.WEBAPP_URL.startsWith('https://')) {
    rows.push([{ text: '🚀 Mở Popup Mini App Live', web_app: { url: CONFIG.WEBAPP_URL } }]);
  }
  return {
    keyboard: rows,
    resize_keyboard: true,
    is_persistent: true,
  };
}

function buildInlineDashboardKeyboard_(chatId) {
  const rows = [];
  const appBtn = buildOpenAppButton_(chatId, CONFIG.WEBAPP_URL);
  if (appBtn) rows.push([appBtn]);
  rows.push([
    { text: '💼 Tài sản', callback_data: 'cmd_status' },
    { text: '📂 Lệnh mở', callback_data: 'cmd_orders' },
  ]);
  rows.push([
    { text: '📐 Tỷ lệ 1U', callback_data: 'cmd_1u' },
    { text: '📈 Thống kê', callback_data: 'cmd_stats' },
  ]);
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
  try {
    const res = UrlFetchApp.fetch('https://api.telegram.org/bot' + CONFIG.BOT_TOKEN + '/sendMessage', {
      method: 'post',
      contentType: 'application/json',
      payload: JSON.stringify(payload),
      muteHttpExceptions: true,
    });
    return JSON.parse(res.getContentText());
  } catch (err) {
    Logger.log('sendTelegram_ error for ' + chatId + ': ' + err);
    return { ok: false };
  }
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
    sheet.appendRow([
      'ID', 'Symbol', 'TF', 'Setup', 'Direction', 'Entry', 'Exit', 'Qty',
      'VolMult', 'Outcome', 'PnL_USD', 'PnL_U', 'Opened_UTC', 'Closed_UTC',
    ]);
  }
  const existingIds = new Set(
    sheet.getLastRow() > 1 ? sheet.getRange(2, 1, sheet.getLastRow() - 1, 1).getValues().flat() : []
  );
  const closed = (d.closed_trades || []).slice().reverse();
  closed.forEach(t => {
    if (!existingIds.has(t.id)) {
      sheet.appendRow([
        t.id, t.symbol, t.tf, t.setup_code, t.direction, t.entry_price, t.exit_price,
        t.qty, t.vol_mult, t.outcome, t.pnl_usd, t.pnl_u || 0, t.opened_utc, t.closed_utc,
      ]);
    }
  });
}

/**
 * Chay ham nay 1 lan duy nhat sau khi Deploy Web App de gan Webhook cho Bot Telegram.
 */
function setWebhook() {
  const webAppExecUrl = ScriptApp.getService().getUrl();
  const res = UrlFetchApp.fetch(
    'https://api.telegram.org/bot' + CONFIG.BOT_TOKEN + '/setWebhook?url=' + encodeURIComponent(webAppExecUrl)
  );
  Logger.log(res.getContentText());
}
