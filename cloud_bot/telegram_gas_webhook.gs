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
  PENDING_DELETES_PROP_KEY: 'PENDING_DELETES_JSON',
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

    // 2. Standard Telegram Bot Webhook update (deduplicate update_id to block GAS 302 retry spam!)
    if (payload.update_id !== undefined && payload.update_id !== null) {
      const cache = CacheService.getScriptCache();
      const dedupKey = 'tg_upd_' + String(payload.update_id);
      if (cache.get(dedupKey)) {
        return ContentService.createTextOutput('OK');
      }
      cache.put(dedupKey, '1', 21600); // 6 hours TTL
    }
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

function isAdmin_(chatId) {
  return Boolean(CONFIG.OWNER_CHAT_ID) && String(chatId) === String(CONFIG.OWNER_CHAT_ID).trim();
}

function isBlocked_(chatId) {
  if (isAdmin_(chatId)) return false;
  const subs = getSubscribers_();
  const rec = subs[String(chatId)];
  return Boolean(rec && rec.blocked === true);
}

function escapeHtml_(str) {
  return String(str || '')
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;');
}

function nowUtcStr_() {
  return new Date().toISOString().slice(0, 16).replace('T', ' ');
}

function upsertSubscriber_(chatObj, fromObj, mutedOverride) {
  const chatId = String(chatObj.id);
  const subs = getSubscribers_();
  const prev = subs[chatId] || {};
  const firstName = (fromObj && fromObj.first_name ? String(fromObj.first_name) : prev.first_name) || '';
  const username = (fromObj && fromObj.username ? String(fromObj.username) : prev.username) || '';
  const displayName =
    (chatObj.title && String(chatObj.title)) ||
    firstName ||
    (username ? '@' + username : '') ||
    prev.name ||
    'Trader';
  const nowStr = nowUtcStr_();
  subs[chatId] = {
    name: displayName,
    username: username,
    first_name: firstName,
    muted: typeof mutedOverride === 'boolean' ? mutedOverride : Boolean(prev.muted),
    blocked: Boolean(prev.blocked),
    joined_utc: prev.joined_utc || nowStr,
    last_seen_utc: nowStr,
  };
  saveSubscribers_(subs);
  return subs[chatId];
}

function setUserBlocked_(targetChatId, blocked) {
  const tid = String(targetChatId || '').trim();
  const subs = getSubscribers_();
  const prev = subs[tid] || {};
  const nowStr = nowUtcStr_();
  subs[tid] = {
    name: prev.name || ('User ' + tid),
    username: prev.username || '',
    first_name: prev.first_name || '',
    muted: Boolean(prev.muted),
    blocked: Boolean(blocked),
    joined_utc: prev.joined_utc || nowStr,
    last_seen_utc: prev.last_seen_utc || nowStr,
  };
  saveSubscribers_(subs);
  return subs[tid];
}

function buildAdminUsersView_() {
  const subs = getSubscribers_();
  const ids = Object.keys(subs);
  if (ids.length === 0) {
    return {
      text: '👑 <b>QUẢN TRỊ NGƯỜI DÙNG BOT</b>\n<i>Chưa có người dùng nào tương tác.</i>',
      markup: null,
    };
  }
  let text = '👑 <b>QUẢN TRỊ NGƯỜI DÙNG BOT (' + ids.length + ' tài khoản)</b>\n\n';
  const rows = [];
  ids.forEach((cid, idx) => {
    const r = subs[cid] || {};
    const uname = r.username ? ' (@' + escapeHtml_(r.username) + ')' : '';
    const statusTxt = r.blocked ? '⛔ Đã chặn' : (r.muted ? '🔕 Tắt tin' : '✅ Hoạt động');
    const roleTag = isAdmin_(cid) ? ' 👑' : '';
    text +=
      (idx + 1) + '. <b>' + escapeHtml_(r.name || 'Trader') + '</b>' + uname + roleTag +
      ' · ID: <code>' + escapeHtml_(cid) + '</code>\n' +
      '   ' + statusTxt + ' · Tham gia: <code>' + escapeHtml_(r.joined_utc || '--') + '</code>\n';
    if (!isAdmin_(cid)) {
      const shortName = String(r.name || 'User').slice(0, 14);
      if (r.blocked) {
        rows.push([{ text: '✅ Mở chặn ' + shortName + ' (' + cid + ')', callback_data: 'admin_unban_' + cid }]);
      } else {
        rows.push([{ text: '⛔ Chặn ' + shortName + ' (' + cid + ')', callback_data: 'admin_ban_' + cid }]);
      }
    }
  });
  text += '\n💡 <i>Lệnh nhanh:</i> <code>/ban &lt;id&gt;</code> · <code>/unban &lt;id&gt;</code>';
  return {
    text: text,
    markup: rows.length > 0 ? { inline_keyboard: rows } : null,
  };
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
    if (!rec || rec.blocked === true || rec.muted === true) {
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
  if (isBlocked_(chatId)) {
    sendTelegram_(chatId, '⛔ Bạn không có quyền truy cập bot này.');
    return;
  }

  const rawText = (msg.text || '').trim();
  const tokens = rawText.split(/\s+/);
  const firstToken = tokens[0].toLowerCase().replace(/@[a-z0-9_]+$/i, '');

  // Map quick-reply keyboard labels to commands (supports both 4-col compact & legacy labels)
  let cmd = firstToken;
  if (rawText === '📊 Tài sản') cmd = '/status';
  else if (rawText === '📂 Lệnh đang mở' || rawText === '📂 Lệnh mở') cmd = '/orders';
  else if (rawText === '📐 Tỷ lệ 1U' || rawText === '📐 1U (4H)') cmd = '/1u';
  else if (rawText === '📈 Thống kê') cmd = '/stats';
  else if (rawText === '🔔 Bật tin') cmd = '/subscribe';
  else if (rawText === '🔕 Tắt tin') cmd = '/mute';
  else if (rawText === '🕯️ Chart') cmd = '/chart';
  else if (rawText === '🔄 Menu') cmd = '/menu';

  // Auto-record / refresh active user profile on any interaction (unless /start, /subscribe, /mute set muted explicitly)
  if (cmd !== '/start' && cmd !== '/menu' && cmd !== '/subscribe' && cmd !== '/mute') {
    upsertSubscriber_(msg.chat, msg.from);
  }

  if (cmd === '/admin' || cmd === '/users') {
    if (!isAdmin_(chatId)) {
      sendTelegram_(chatId, '⛔ Lệnh này chỉ dành riêng cho Admin.');
      return;
    }
    const view = buildAdminUsersView_();
    sendTelegram_(chatId, view.text, view.markup);
    return;
  }

  if (cmd === '/ban' || cmd === '/unban') {
    if (!isAdmin_(chatId)) {
      sendTelegram_(chatId, '⛔ Lệnh này chỉ dành riêng cho Admin.');
      return;
    }
    const targetId = (tokens[1] || '').trim();
    if (!targetId) {
      sendTelegram_(chatId, '⚠️ Cú pháp: <code>' + cmd + ' &lt;chat_id&gt;</code>');
      return;
    }
    if (cmd === '/ban' && isAdmin_(targetId)) {
      sendTelegram_(chatId, '⚠️ Không thể tự chặn Admin (OWNER_CHAT_ID).');
      return;
    }
    const isBan = cmd === '/ban';
    const rec = setUserBlocked_(targetId, isBan);
    sendTelegram_(
      chatId,
      isBan
        ? '⛔ Đã chặn người dùng <b>' + escapeHtml_(rec.name) + '</b> (<code>' + escapeHtml_(targetId) + '</code>).'
        : '✅ Đã mở chặn người dùng <b>' + escapeHtml_(rec.name) + '</b> (<code>' + escapeHtml_(targetId) + '</code>).'
    );
    return;
  }

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
      '• /chart — Gửi ảnh biểu đồ nến + EMA 34/89/200 (tự xóa sau 2 phút)\n' +
      '• /stats — Hiệu suất Profit WR, PF &amp; 5 lệnh gần nhất\n' +
      '• /mute | /subscribe — Tạm tắt / Bật lại thông báo tự động';
    sendTelegram_(chatId, welcome, buildReplyKeyboard_(chatId));
    return;
  }

  if (cmd === '/chart') {
    sendChartCoinPicker_(chatId);
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
  const msgId = cb.message.message_id;
  const data = cb.data || '';
  answerCallback_(cb.id);

  if (isBlocked_(chatId)) {
    sendTelegram_(chatId, '⛔ Bạn không có quyền truy cập bot này.');
    return;
  }

  upsertSubscriber_(cb.message.chat, cb.from);

  if (data.indexOf('admin_ban_') === 0 || data.indexOf('admin_unban_') === 0) {
    if (!isAdmin_(chatId)) {
      sendTelegram_(chatId, '⛔ Chỉ Admin mới được thao tác.');
      return;
    }
    const isBan = data.indexOf('admin_ban_') === 0;
    const targetId = data.replace(isBan ? 'admin_ban_' : 'admin_unban_', '').trim();
    if (isBan && isAdmin_(targetId)) {
      return;
    }
    setUserBlocked_(targetId, isBan);
    const view = buildAdminUsersView_();
    editMessageText_(chatId, msgId, view.text, view.markup);
    return;
  }

  if (data === 'cmd_status') {
    sendTelegram_(chatId, formatStatusText_(fetchDashboard_()), buildInlineDashboardKeyboard_(chatId));
  } else if (data === 'cmd_orders') {
    sendTelegram_(chatId, formatOrdersText_(fetchDashboard_()), buildInlineDashboardKeyboard_(chatId));
  } else if (data === 'cmd_1u') {
    sendTelegram_(chatId, format1uRatesText_(fetchDashboard_()), buildInlineDashboardKeyboard_(chatId));
  } else if (data === 'cmd_stats') {
    sendTelegram_(chatId, formatStatsText_(fetchDashboard_()), buildInlineDashboardKeyboard_(chatId));
  } else if (data === 'chart_back') {
    editMessageText_(chatId, msgId, '🕯️ <b>XEM BIỂU ĐỒ NẾN + EMA 34/89/200 LIVE</b>\nChọn đồng Coin muốn xem:', buildChartCoinKeyboard_());
  } else if (data.indexOf('chart_coin_') === 0) {
    const sym = data.replace('chart_coin_', '');
    editMessageText_(chatId, msgId, '🕯️ <b>' + sym + '</b> — Chọn khung thời gian muốn vẽ biểu đồ:', buildChartTfKeyboard_(sym));
  } else if (data.indexOf('chart_go_') === 0) {
    const rest = data.replace('chart_go_', '');
    const parts = rest.split('_');
    if (parts.length === 2) {
      sendChartPhotoWithAutoDelete_(chatId, msgId, parts[0], parts[1]);
    }
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
  const row1 = [
    { text: '📊 Tài sản' },
    { text: '📂 Lệnh mở' },
    { text: '📐 1U (4H)' },
    { text: '📈 Thống kê' },
  ];
  const popupBtn =
    isPrivateChat_(chatId) && CONFIG.WEBAPP_URL && CONFIG.WEBAPP_URL.startsWith('https://')
      ? { text: '🚀 Popup', web_app: { url: CONFIG.WEBAPP_URL } }
      : { text: '🔄 Menu' };
  const row2 = [
    popupBtn,
    { text: '🕯️ Chart' },
    { text: '🔔 Bật tin' },
    { text: '🔕 Tắt tin' },
  ];
  return {
    keyboard: [row1, row2],
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

function editMessageText_(chatId, messageId, text, replyMarkup) {
  const payload = {
    chat_id: chatId,
    message_id: messageId,
    text: text,
    parse_mode: 'HTML',
    disable_web_page_preview: true,
  };
  if (replyMarkup) payload.reply_markup = replyMarkup;
  try {
    const res = UrlFetchApp.fetch('https://api.telegram.org/bot' + CONFIG.BOT_TOKEN + '/editMessageText', {
      method: 'post',
      contentType: 'application/json',
      payload: JSON.stringify(payload),
      muteHttpExceptions: true,
    });
    return JSON.parse(res.getContentText());
  } catch (err) {
    return { ok: false };
  }
}

function deleteMessage_(chatId, messageId) {
  try {
    const res = UrlFetchApp.fetch('https://api.telegram.org/bot' + CONFIG.BOT_TOKEN + '/deleteMessage', {
      method: 'post',
      contentType: 'application/json',
      payload: JSON.stringify({ chat_id: chatId, message_id: messageId }),
      muteHttpExceptions: true,
    });
    return JSON.parse(res.getContentText());
  } catch (err) {
    return { ok: false };
  }
}

// ---------- QuickChart Live Candlestick + EMA 34/89/200 + 2-Min Auto-Delete ----------

function buildChartCoinKeyboard_() {
  return {
    inline_keyboard: [
      [
        { text: 'BTC', callback_data: 'chart_coin_BTCUSDT' },
        { text: 'ETH', callback_data: 'chart_coin_ETHUSDT' },
        { text: 'BNB', callback_data: 'chart_coin_BNBUSDT' },
      ],
      [
        { text: 'LINK', callback_data: 'chart_coin_LINKUSDT' },
        { text: 'XRP', callback_data: 'chart_coin_XRPUSDT' },
        { text: 'SOL', callback_data: 'chart_coin_SOLUSDT' },
      ],
    ],
  };
}

function buildChartTfKeyboard_(sym) {
  return {
    inline_keyboard: [
      [
        { text: '15m', callback_data: 'chart_go_' + sym + '_15m' },
        { text: '30m', callback_data: 'chart_go_' + sym + '_30m' },
        { text: '1h', callback_data: 'chart_go_' + sym + '_1h' },
        { text: '4h', callback_data: 'chart_go_' + sym + '_4h' },
      ],
      [{ text: '« Đổi Coin', callback_data: 'chart_back' }],
    ],
  };
}

function sendChartCoinPicker_(chatId) {
  sendTelegram_(
    chatId,
    '🕯️ <b>XEM BIỂU ĐỒ NẾN + EMA 34/89/200 LIVE</b>\nChọn đồng Coin muốn xem:',
    buildChartCoinKeyboard_()
  );
}

function fetchKlines220_(sym, tf) {
  const urls = [
    'https://fapi.binance.com/fapi/v1/klines?symbol=' + encodeURIComponent(sym) + '&interval=' + encodeURIComponent(tf) + '&limit=220',
    'https://data-api.binance.vision/api/v3/klines?symbol=' + encodeURIComponent(sym) + '&interval=' + encodeURIComponent(tf) + '&limit=220',
  ];
  for (let i = 0; i < urls.length; i++) {
    try {
      const res = UrlFetchApp.fetch(urls[i], { muteHttpExceptions: true });
      if (res.getResponseCode() === 200) {
        const arr = JSON.parse(res.getContentText());
        if (Array.isArray(arr) && arr.length > 0) return arr;
      }
    } catch (e) {}
  }
  throw new Error('Khong tai duoc du lieu nen Binance cho ' + sym + ' ' + tf);
}

function computeEmaArr_(closes, period) {
  if (!closes.length) return [];
  const k = 2 / (period + 1);
  const out = [closes[0]];
  for (let i = 1; i < closes.length; i++) {
    out.push(closes[i] * k + out[i - 1] * (1 - k));
  }
  return out;
}

function pad2_(n) {
  return (n < 10 ? '0' : '') + n;
}

function sendChartPhotoWithAutoDelete_(chatId, pickerMsgId, sym, tf) {
  try {
    editMessageText_(chatId, pickerMsgId, '⏳ Đang tải 220 nến &amp; vẽ biểu đồ <b>' + sym + ' (' + tf + ')</b>...');
    const raw = fetchKlines220_(sym, tf);
    const closes = raw.map(k => parseFloat(k[4]));
    const e34All = computeEmaArr_(closes, 34);
    const e89All = computeEmaArr_(closes, 89);
    const e200All = computeEmaArr_(closes, 200);

    const nShow = Math.min(60, raw.length);
    const startIdx = raw.length - nShow;
    const labels = [];
    const wicks = [];
    const bodies = [];
    const colors = [];
    const e34Show = [];
    const e89Show = [];
    const e200Show = [];

    let yMin = Infinity;
    let yMax = -Infinity;
    for (let i = startIdx; i < raw.length; i++) {
      const k = raw[i];
      const o = parseFloat(k[1]);
      const h = parseFloat(k[2]);
      const l = parseFloat(k[3]);
      const c = parseFloat(k[4]);
      yMin = Math.min(yMin, l, e34All[i], e89All[i], e200All[i]);
      yMax = Math.max(yMax, h, e34All[i], e89All[i], e200All[i]);
    }
    const padRange = Math.max((yMax - yMin) * 0.05, yMax * 0.001);
    const minBody = Math.max((yMax - yMin) * 0.0025, yMax * 0.00015);

    for (let i = startIdx; i < raw.length; i++) {
      const k = raw[i];
      const dtVn = new Date(Number(k[0]) + 7 * 3600 * 1000);
      const lbl =
        tf === '4h'
          ? pad2_(dtVn.getUTCDate()) + '/' + pad2_(dtVn.getUTCMonth() + 1) + ' ' + pad2_(dtVn.getUTCHours()) + 'h'
          : pad2_(dtVn.getUTCHours()) + ':' + pad2_(dtVn.getUTCMinutes());
      const o = parseFloat(k[1]);
      const h = parseFloat(k[2]);
      const l = parseFloat(k[3]);
      const c = parseFloat(k[4]);
      const isUp = c >= o;
      labels.push(lbl);
      wicks.push([l, h]);
      if (Math.abs(c - o) < minBody) {
        bodies.push([o - minBody * 0.5, o + minBody * 0.5]);
      } else {
        bodies.push([Math.min(o, c), Math.max(o, c)]);
      }
      colors.push(isUp ? '#059669' : '#DC2626');
      e34Show.push(Number(e34All[i].toFixed(6)));
      e89Show.push(Number(e89All[i].toFixed(6)));
      e200Show.push(Number(e200All[i].toFixed(6)));
    }

    const lastPx = closes[closes.length - 1];
    const lastE34 = e34All[e34All.length - 1];
    const lastE89 = e89All[e89All.length - 1];
    const lastE200 = e200All[e200All.length - 1];

    const chartCfg = {
      type: 'bar',
      data: {
        labels: labels,
        datasets: [
          {
            type: 'line',
            label: 'EMA 34',
            data: e34Show,
            borderColor: '#0284C7',
            borderWidth: 2,
            pointRadius: 0,
            fill: false,
            tension: 0.15,
            order: 1,
          },
          {
            type: 'line',
            label: 'EMA 89',
            data: e89Show,
            borderColor: '#D97706',
            borderWidth: 2,
            pointRadius: 0,
            fill: false,
            tension: 0.15,
            order: 1,
          },
          {
            type: 'line',
            label: 'EMA 200',
            data: e200Show,
            borderColor: '#7C3AED',
            borderWidth: 2.2,
            pointRadius: 0,
            fill: false,
            tension: 0.15,
            order: 1,
          },
          {
            type: 'bar',
            label: 'Wick',
            data: wicks,
            backgroundColor: colors,
            barThickness: 2,
            grouped: false,
            order: 2,
          },
          {
            type: 'bar',
            label: 'Candle',
            data: bodies,
            backgroundColor: colors,
            barPercentage: 0.72,
            categoryPercentage: 0.88,
            grouped: false,
            order: 3,
          },
        ],
      },
      options: {
        plugins: {
          title: {
            display: true,
            text: sym + ' · ' + tf.toUpperCase() + ' (60 nến gần nhất · EMA 34/89/200)',
            color: '#0F172A',
            font: { size: 14, weight: 'bold' },
          },
          legend: {
            display: true,
            labels: {
              color: '#334155',
              filter: function (item) {
                return item.text.indexOf('EMA') === 0;
              },
            },
          },
        },
        scales: {
          x: {
            ticks: { color: '#475569', maxTicksLimit: 10, maxRotation: 0 },
            grid: { color: '#F1F5F9' },
          },
          y: {
            position: 'right',
            min: yMin - padRange,
            max: yMax + padRange,
            ticks: { color: '#334155' },
            grid: { color: '#E2E8F0' },
          },
        },
      },
    };

    const qcRes = UrlFetchApp.fetch('https://quickchart.io/chart', {
      method: 'post',
      contentType: 'application/json',
      payload: JSON.stringify({
        version: '3',
        width: 800,
        height: 460,
        backgroundColor: '#FFFFFF',
        format: 'png',
        chart: chartCfg,
      }),
      muteHttpExceptions: true,
    });
    if (qcRes.getResponseCode() !== 200) {
      throw new Error('QuickChart HTTP ' + qcRes.getResponseCode());
    }
    const pngBlob = qcRes.getBlob().setName(sym + '_' + tf + '.png');

    const popupChartUrl = CONFIG.WEBAPP_URL + '?coin=' + encodeURIComponent(sym) + '&tf=' + encodeURIComponent(tf);
    const openBtn = buildOpenAppButton_(chatId, popupChartUrl);
    const markup = openBtn ? { inline_keyboard: [[{ text: '📈 Mở Popup TradingView', web_app: openBtn.web_app, url: openBtn.url }]] } : null;

    const caption =
      '🕯️ <b>' + sym + ' · ' + tf.toUpperCase() + '</b> · Giá Live: <b>' + fmtPx_(lastPx) + '</b>\n' +
      '• EMA34: <code>' + fmtPx_(lastE34) + '</code> · EMA89: <code>' + fmtPx_(lastE89) + '</code> · EMA200: <code>' + fmtPx_(lastE200) + '</code>\n' +
      '⏱️ <i>Ảnh tự xóa sau 2 phút để giữ nhóm sạch sẽ</i>';

    const sent = sendPhoto_(chatId, pngBlob, caption, markup);
    deleteMessage_(chatId, pickerMsgId);
    if (sent && sent.ok && sent.result && sent.result.message_id) {
      scheduleChartDelete_(chatId, sent.result.message_id, 120000);
    }
  } catch (err) {
    Logger.log('Chart error: ' + err);
    editMessageText_(chatId, pickerMsgId, '⚠️ Không thể vẽ biểu đồ lúc này: ' + String(err));
  }
}

function sendPhoto_(chatId, photoBlob, caption, replyMarkup) {
  const form = {
    chat_id: String(chatId),
    photo: photoBlob,
    caption: caption,
    parse_mode: 'HTML',
  };
  if (replyMarkup) {
    form.reply_markup = JSON.stringify(replyMarkup);
  }
  const res = UrlFetchApp.fetch('https://api.telegram.org/bot' + CONFIG.BOT_TOKEN + '/sendPhoto', {
    method: 'post',
    payload: form,
    muteHttpExceptions: true,
  });
  return JSON.parse(res.getContentText());
}

function scheduleChartDelete_(chatId, messageId, delayMs) {
  const props = PropertiesService.getScriptProperties();
  let list = [];
  try {
    list = JSON.parse(props.getProperty(CONFIG.PENDING_DELETES_PROP_KEY) || '[]') || [];
  } catch (e) {
    list = [];
  }
  list.push({
    chat_id: String(chatId),
    message_id: Number(messageId),
    delete_at: Date.now() + Number(delayMs || 120000),
  });
  props.setProperty(CONFIG.PENDING_DELETES_PROP_KEY, JSON.stringify(list));
  ScriptApp.newTrigger('cleanupExpiredCharts_').timeBased().after(Number(delayMs || 120000)).create();
}

function cleanupExpiredCharts_(e) {
  const props = PropertiesService.getScriptProperties();
  let list = [];
  try {
    list = JSON.parse(props.getProperty(CONFIG.PENDING_DELETES_PROP_KEY) || '[]') || [];
  } catch (err) {
    list = [];
  }
  const now = Date.now();
  const remaining = [];
  list.forEach(item => {
    if (Number(item.delete_at) <= now + 5000) {
      deleteMessage_(item.chat_id, item.message_id);
    } else {
      remaining.push(item);
    }
  });
  if (remaining.length === 0) {
    props.deleteProperty(CONFIG.PENDING_DELETES_PROP_KEY);
  } else {
    props.setProperty(CONFIG.PENDING_DELETES_PROP_KEY, JSON.stringify(remaining));
  }

  const firedUid = e && e.triggerUid ? String(e.triggerUid) : '';
  const triggers = ScriptApp.getProjectTriggers();
  triggers.forEach(t => {
    if (t.getHandlerFunction() === 'cleanupExpiredCharts_') {
      if ((firedUid && String(t.getUniqueId()) === firedUid) || remaining.length === 0) {
        ScriptApp.deleteTrigger(t);
      }
    }
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
