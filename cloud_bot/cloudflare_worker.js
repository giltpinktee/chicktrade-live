/**
 * CHICKTRADE QUANT v3.3 — Free $0/mo Cloudflare Worker Telegram Bot (ES Module)
 *
 * Cach cai dat nhanh tren Cloudflare Dashboard (100,000 req/ngay mien phi, HTTP 200 OK < 0.2s):
 * 1. Workers & Pages -> Create Worker -> Dan toan bo file nay vao -> Deploy.
 * 2. Settings -> Variables and Secrets -> Them:
 *      - BOT_TOKEN (Secret): Token Telegram Bot (lay tu @BotFather)
 *      - BROADCAST_SECRET (Secret): Chuoi bi mat tu chon (khop voi GitHub Secret GAS_BROADCAST_SECRET)
 *      - OWNER_CHAT_ID (Variable): Chat ID Telegram cua Admin (de dung /admin, /ban, /unban)
 *      - DASHBOARD_JSON_URL (Variable): https://<GITHUB_USER>.github.io/<REPO>/data/live_dashboard.json
 *      - CHART_CANDLES_JSON_URL (Variable): https://<GITHUB_USER>.github.io/<REPO>/data/chart_candles.json
 *      - WEBAPP_URL (Variable): https://<GITHUB_USER>.github.io/<REPO>/
 * 3. Settings -> Bindings -> Add KV Namespace (dat ten bien: CHICKTRADE_KV) de luu ben vung subscribers & reply-to-open.
 * 4. Goi URL setWebhook cua Telegram (hoac mo https://<worker>.workers.dev/set-webhook?secret=<BROADCAST_SECRET>) 1 lan duy nhat!
 */

const MEM_STORE = {
  subscribers: {},
  openMsgs: {},
  dedup: new Map(),
};

function getConfig(env) {
  const e = env || {};
  return {
    BOT_TOKEN: String(e.BOT_TOKEN || '').trim(),
    BROADCAST_SECRET: String(e.BROADCAST_SECRET || '').trim(),
    OWNER_CHAT_ID: String(e.OWNER_CHAT_ID || '').trim(),
    DASHBOARD_JSON_URL: String(
      e.DASHBOARD_JSON_URL || 'https://<YOUR_GITHUB_USER>.github.io/<YOUR_REPO>/data/live_dashboard.json'
    ).trim(),
    CHART_CANDLES_JSON_URL: String(
      e.CHART_CANDLES_JSON_URL || 'https://<YOUR_GITHUB_USER>.github.io/<YOUR_REPO>/data/chart_candles.json'
    ).trim(),
    WEBAPP_URL: String(e.WEBAPP_URL || 'https://<YOUR_GITHUB_USER>.github.io/<YOUR_REPO>/').trim(),
    KV: e.CHICKTRADE_KV || null,
  };
}

export default {
  async fetch(request, env, ctx) {
    const cfg = getConfig(env);
    const url = new URL(request.url);

    if (request.method === 'GET') {
      if (url.pathname === '/set-webhook') {
        const sec = url.searchParams.get('secret') || '';
        if (cfg.BROADCAST_SECRET && sec !== cfg.BROADCAST_SECRET) {
          return jsonResponse({ ok: false, error: 'UNAUTHORIZED' }, 401);
        }
        if (!cfg.BOT_TOKEN) {
          return jsonResponse({ ok: false, error: 'Missing BOT_TOKEN env secret' }, 400);
        }
        const hookUrl = `${url.origin}/`;
        const res = await fetch(
          `https://api.telegram.org/bot${cfg.BOT_TOKEN}/setWebhook?url=${encodeURIComponent(hookUrl)}`
        );
        const data = await res.json();
        return jsonResponse({ ok: Boolean(data.ok), webhook: hookUrl, telegram: data }, 200);
      }

      const subs = await getSubscribers(cfg);
      return jsonResponse(
        {
          ok: true,
          service: 'CHICKTRADE v3.3 Cloudflare Worker',
          subscribers: Object.keys(subs).length,
          kv_bound: Boolean(cfg.KV),
        },
        200
      );
    }

    if (request.method !== 'POST') {
      return new Response('Method Not Allowed', { status: 405 });
    }

    let payload;
    try {
      payload = await request.json();
    } catch {
      return new Response('OK', { status: 200 });
    }

    // 1. Multi-user broadcast request from GitHub Actions paper_engine_v3.py
    if (payload && payload.action === 'broadcast') {
      if (cfg.BROADCAST_SECRET && String(payload.secret || '') !== cfg.BROADCAST_SECRET) {
        return jsonResponse({ ok: false, error: 'UNAUTHORIZED' }, 401);
      }
      const res = await broadcastToSubscribers(
        cfg,
        String(payload.text || ''),
        String(payload.exclude_chat_id || cfg.OWNER_CHAT_ID || ''),
        String(payload.event_type || ''),
        String(payload.pos_id || ''),
        String(payload.photo_base64 || ''),
        String(payload.photo_filename || 'chart.png')
      );
      return jsonResponse(res, 200);
    }

    // 2. Standard Telegram Bot Webhook update (return 200 OK immediately via ctx.waitUntil)
    if (payload && payload.update_id !== undefined && payload.update_id !== null) {
      if (isDuplicateUpdate(payload.update_id)) {
        return new Response('OK', { status: 200 });
      }
    }

    const work = handleTelegramUpdate(cfg, env, ctx, payload).catch(err => {
      console.error('Worker update error:', err);
    });
    if (ctx && typeof ctx.waitUntil === 'function') {
      ctx.waitUntil(work);
    } else {
      await work;
    }

    return new Response('OK', { status: 200 });
  },
};

function jsonResponse(obj, status = 200) {
  return new Response(JSON.stringify(obj), {
    status,
    headers: { 'content-type': 'application/json; charset=utf-8' },
  });
}

function isDuplicateUpdate(updateId) {
  const now = Date.now();
  const key = String(updateId);
  if (MEM_STORE.dedup.has(key)) return true;
  MEM_STORE.dedup.set(key, now);
  if (MEM_STORE.dedup.size > 500) {
    const cutoff = now - 6 * 3600 * 1000;
    for (const [k, ts] of MEM_STORE.dedup.entries()) {
      if (ts < cutoff) MEM_STORE.dedup.delete(k);
    }
  }
  return false;
}

// ---------- Subscriber & Open-Message Store (KV with Memory Fallback) ----------

async function getSubscribers(cfg) {
  if (cfg.KV) {
    try {
      const val = await cfg.KV.get('subscribers', { type: 'json' });
      if (val && typeof val === 'object') {
        MEM_STORE.subscribers = val;
        return val;
      }
    } catch (e) {
      console.error('KV getSubscribers error:', e);
    }
  }
  return MEM_STORE.subscribers;
}

async function saveSubscribers(cfg, subs) {
  MEM_STORE.subscribers = subs;
  if (cfg.KV) {
    try {
      await cfg.KV.put('subscribers', JSON.stringify(subs));
    } catch (e) {
      console.error('KV saveSubscribers error:', e);
    }
  }
}

async function getOpenMsgMap(cfg, posId) {
  const key = `open_msgs_${posId}`;
  if (cfg.KV) {
    try {
      const val = await cfg.KV.get(key, { type: 'json' });
      if (val && typeof val === 'object') return val;
    } catch (e) {
      console.error('KV getOpenMsgMap error:', e);
    }
  }
  return MEM_STORE.openMsgs[key] || {};
}

async function saveOpenMsgMap(cfg, posId, map) {
  const key = `open_msgs_${posId}`;
  MEM_STORE.openMsgs[key] = map;
  if (cfg.KV) {
    try {
      await cfg.KV.put(key, JSON.stringify(map), { expirationTtl: 14 * 86400 });
    } catch (e) {
      console.error('KV saveOpenMsgMap error:', e);
    }
  }
}

async function deleteOpenMsgMap(cfg, posId) {
  const key = `open_msgs_${posId}`;
  delete MEM_STORE.openMsgs[key];
  if (cfg.KV) {
    try {
      await cfg.KV.delete(key);
    } catch (e) {
      console.error('KV deleteOpenMsgMap error:', e);
    }
  }
}

function isAdmin(cfg, chatId) {
  return Boolean(cfg.OWNER_CHAT_ID) && String(chatId) === String(cfg.OWNER_CHAT_ID);
}

async function isBlocked(cfg, chatId) {
  if (isAdmin(cfg, chatId)) return false;
  const subs = await getSubscribers(cfg);
  const rec = subs[String(chatId)];
  return Boolean(rec && rec.blocked === true);
}

function escapeHtml(str) {
  return String(str || '')
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;');
}

function nowUtcStr() {
  return new Date().toISOString().slice(0, 16).replace('T', ' ');
}

async function upsertSubscriber(cfg, chatObj, fromObj, mutedOverride) {
  const chatId = String(chatObj.id);
  const subs = await getSubscribers(cfg);
  const prev = subs[chatId] || {};
  const firstName = (fromObj && fromObj.first_name ? String(fromObj.first_name) : prev.first_name) || '';
  const username = (fromObj && fromObj.username ? String(fromObj.username) : prev.username) || '';
  const displayName =
    (chatObj.title && String(chatObj.title)) ||
    firstName ||
    (username ? `@${username}` : '') ||
    prev.name ||
    'Trader';
  const nowStr = nowUtcStr();
  subs[chatId] = {
    name: displayName,
    username,
    first_name: firstName,
    muted: typeof mutedOverride === 'boolean' ? mutedOverride : Boolean(prev.muted),
    blocked: Boolean(prev.blocked),
    joined_utc: prev.joined_utc || nowStr,
    last_seen_utc: nowStr,
  };
  await saveSubscribers(cfg, subs);
  return subs[chatId];
}

async function setUserBlocked(cfg, targetChatId, blocked) {
  const tid = String(targetChatId || '').trim();
  const subs = await getSubscribers(cfg);
  const prev = subs[tid] || {};
  const nowStr = nowUtcStr();
  subs[tid] = {
    name: prev.name || `User ${tid}`,
    username: prev.username || '',
    first_name: prev.first_name || '',
    muted: Boolean(prev.muted),
    blocked: Boolean(blocked),
    joined_utc: prev.joined_utc || nowStr,
    last_seen_utc: prev.last_seen_utc || nowStr,
  };
  await saveSubscribers(cfg, subs);
  return subs[tid];
}

async function buildAdminUsersView(cfg) {
  const subs = await getSubscribers(cfg);
  const ids = Object.keys(subs);
  if (ids.length === 0) {
    return {
      text: '👑 <b>QUẢN TRỊ NGƯỜI DÙNG BOT</b>\n<i>Chưa có người dùng nào tương tác.</i>',
      markup: null,
    };
  }
  let text = `👑 <b>QUẢN TRỊ NGƯỜI DÙNG BOT (${ids.length} tài khoản)</b>\n\n`;
  const rows = [];
  ids.forEach((cid, idx) => {
    const r = subs[cid] || {};
    const uname = r.username ? ` (@${escapeHtml(r.username)})` : '';
    const statusTxt = r.blocked ? '⛔ Đã chặn' : r.muted ? '🔕 Tắt tin' : '✅ Hoạt động';
    const roleTag = isAdmin(cfg, cid) ? ' 👑' : '';
    text +=
      `${idx + 1}. <b>${escapeHtml(r.name || 'Trader')}</b>${uname}${roleTag} · ID: <code>${escapeHtml(cid)}</code>\n` +
      `   ${statusTxt} · Tham gia: <code>${escapeHtml(r.joined_utc || '--')}</code>\n`;
    if (!isAdmin(cfg, cid)) {
      const shortName = String(r.name || 'User').slice(0, 14);
      if (r.blocked) {
        rows.push([{ text: `✅ Mở chặn ${shortName} (${cid})`, callback_data: `admin_unban_${cid}` }]);
      } else {
        rows.push([{ text: `⛔ Chặn ${shortName} (${cid})`, callback_data: `admin_ban_${cid}` }]);
      }
    }
  });
  text += '\n💡 <i>Lệnh nhanh:</i> <code>/ban &lt;id&gt;</code> · <code>/unban &lt;id&gt;</code>';
  return {
    text,
    markup: rows.length > 0 ? { inline_keyboard: rows } : null,
  };
}

function base64ToUint8Array(b64) {
  const bin = atob(b64);
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) {
    bytes[i] = bin.charCodeAt(i);
  }
  return bytes;
}

async function broadcastToSubscribers(
  cfg,
  text,
  excludeChatId,
  eventType = '',
  posId = '',
  photoBase64 = '',
  photoFilename = 'chart.png'
) {
  if (!text) return { ok: true, sent: 0, skipped: 0 };
  const subs = await getSubscribers(cfg);
  const ids = Object.keys(subs);
  let sent = 0;
  let skipped = 0;
  let changed = false;

  const isFollowUp = Boolean(posId) && (eventType === 'TP1_HIT' || eventType === 'CLOSED');
  const openMap = isFollowUp ? await getOpenMsgMap(cfg, posId) : {};
  const newOpenMap = eventType === 'OPEN' && posId ? {} : null;
  let pngBytes = null;
  if (photoBase64) {
    try {
      pngBytes = base64ToUint8Array(photoBase64);
    } catch {
      pngBytes = null;
    }
  }

  for (const cid of ids) {
    const rec = subs[cid];
    if (!rec || rec.blocked === true || rec.muted === true) {
      skipped += 1;
      continue;
    }
    if (excludeChatId && String(cid) === String(excludeChatId).trim()) {
      skipped += 1;
      continue;
    }
    const replyToMsgId = isFollowUp ? openMap[cid] || null : null;
    let resp;
    if (pngBytes && text.length <= 1020) {
      resp = await sendPhoto(cfg, cid, pngBytes, photoFilename || 'chart.png', text, null, replyToMsgId);
      if (!resp || !resp.ok) {
        resp = await sendTelegram(cfg, cid, text, null, replyToMsgId);
      }
    } else {
      resp = await sendTelegram(cfg, cid, text, null, replyToMsgId);
    }
    if (resp && resp.ok) {
      sent += 1;
      if (newOpenMap && resp.result && resp.result.message_id) {
        newOpenMap[cid] = resp.result.message_id;
      }
    } else if (resp && (resp.error_code === 403 || resp.error_code === 400)) {
      rec.muted = true;
      changed = true;
      skipped += 1;
    } else {
      skipped += 1;
    }
  }

  if (changed) {
    await saveSubscribers(cfg, subs);
  }
  if (newOpenMap && Object.keys(newOpenMap).length > 0) {
    await saveOpenMsgMap(cfg, posId, newOpenMap);
  }
  if (eventType === 'CLOSED' && posId) {
    await deleteOpenMsgMap(cfg, posId);
  }
  return { ok: true, sent, skipped };
}

// ---------- Update, Command & Callback Routing ----------

async function handleTelegramUpdate(cfg, env, ctx, payload) {
  if (payload && payload.message && payload.message.text) {
    await handleMessage(cfg, ctx, payload.message);
  } else if (payload && payload.callback_query) {
    await handleCallback(cfg, ctx, payload.callback_query);
  }
}

async function handleMessage(cfg, ctx, msg) {
  const chatId = msg.chat.id;
  if (await isBlocked(cfg, chatId)) {
    await sendTelegram(cfg, chatId, '⛔ Bạn không có quyền truy cập bot này.');
    return;
  }

  const rawText = String(msg.text || '').trim();
  const tokens = rawText.split(/\s+/);
  const firstToken = tokens[0].toLowerCase().replace(/@[a-z0-9_]+$/i, '');

  let cmd = firstToken;
  if (rawText === '📊 Tài sản') cmd = '/status';
  else if (rawText === '📂 Lệnh đang mở' || rawText === '📂 Lệnh mở') cmd = '/orders';
  else if (rawText === '📐 Tỷ lệ 1U' || rawText === '📐 1U (4H)') cmd = '/1u';
  else if (rawText === '📈 Thống kê') cmd = '/stats';
  else if (rawText === '🔔 Bật tin') cmd = '/subscribe';
  else if (rawText === '🔕 Tắt tin') cmd = '/mute';
  else if (rawText === '🕯️ Chart') cmd = '/chart';
  else if (rawText === '🔄 Menu') cmd = '/menu';

  if (cmd !== '/start' && cmd !== '/menu' && cmd !== '/subscribe' && cmd !== '/mute') {
    await upsertSubscriber(cfg, msg.chat, msg.from);
  }

  if (cmd === '/admin' || cmd === '/users') {
    if (!isAdmin(cfg, chatId)) {
      await sendTelegram(cfg, chatId, '⛔ Lệnh này chỉ dành riêng cho Admin.');
      return;
    }
    const view = await buildAdminUsersView(cfg);
    await sendTelegram(cfg, chatId, view.text, view.markup);
    return;
  }

  if (cmd === '/ban' || cmd === '/unban') {
    if (!isAdmin(cfg, chatId)) {
      await sendTelegram(cfg, chatId, '⛔ Lệnh này chỉ dành riêng cho Admin.');
      return;
    }
    const targetId = (tokens[1] || '').trim();
    if (!targetId) {
      await sendTelegram(cfg, chatId, `⚠️ Cú pháp: <code>${cmd} &lt;chat_id&gt;</code>`);
      return;
    }
    if (cmd === '/ban' && isAdmin(cfg, targetId)) {
      await sendTelegram(cfg, chatId, '⚠️ Không thể tự chặn Admin (OWNER_CHAT_ID).');
      return;
    }
    const isBan = cmd === '/ban';
    const rec = await setUserBlocked(cfg, targetId, isBan);
    await sendTelegram(
      cfg,
      chatId,
      isBan
        ? `⛔ Đã chặn người dùng <b>${escapeHtml(rec.name)}</b> (<code>${escapeHtml(targetId)}</code>).`
        : `✅ Đã mở chặn người dùng <b>${escapeHtml(rec.name)}</b> (<code>${escapeHtml(targetId)}</code>).`
    );
    return;
  }

  if (cmd === '/start' || cmd === '/menu') {
    await upsertSubscriber(cfg, msg.chat, msg.from, false);
    const welcome =
      '🤖 <b>CHICKTRADE QUANT v3.3 — Paper-Test $10,000 Live</b>\n' +
      'Chuẩn <code>U3_HTF_MACRO_ATR 4H</code> · 58 Setups · S2 Không Giới Hạn Bậc.\n' +
      '✅ <b>Đã bật nhận thông báo tự động</b> (Mở lệnh, TP1, Đóng lệnh, Phanh 2-SL, Tổng kết ngày/tuần).\n\n' +
      '<b>Lệnh tra cứu nhanh:</b>\n' +
      '• /status — Tài sản, Bậc S2 &amp; Phanh 2-SL\n' +
      '• /orders — Lệnh đang mở &amp; Dự kiến Win/Loss $\n' +
      '• /1u — Bảng tỷ lệ 1U % 6 Coin (15m/30m/1h)\n' +
      '• /chart — Gửi ảnh biểu đồ nến + EMA 34/89/200 (kèm Entry/TP/SL nếu có lệnh)\n' +
      '• /stats — Hiệu suất Profit WR, PF &amp; 5 lệnh gần nhất\n' +
      '• /mute | /subscribe — Tạm tắt / Bật lại thông báo tự động';
    await sendTelegram(cfg, chatId, welcome, buildReplyKeyboard(cfg, chatId));
    return;
  }

  if (cmd === '/chart') {
    await sendChartCoinPicker(cfg, chatId);
    return;
  }

  if (cmd === '/subscribe') {
    await upsertSubscriber(cfg, msg.chat, msg.from, false);
    await sendTelegram(
      cfg,
      chatId,
      '🔔 <b>Đã BẬT thông báo lệnh tự động!</b>\nBạn sẽ nhận tin ngay khi Bot mở lệnh, chốt TP1, đóng lệnh, bật/mở phanh 2-SL và tổng kết sáng 07:00 VN.'
    );
    return;
  }

  if (cmd === '/mute') {
    await upsertSubscriber(cfg, msg.chat, msg.from, true);
    await sendTelegram(
      cfg,
      chatId,
      '🔕 <b>Đã TẠM TẮT thông báo tự động.</b>\nBạn vẫn có thể bấm nút hoặc gõ <code>/status</code>, <code>/orders</code>, <code>/1u</code>, <code>/stats</code> bình thường. Gõ /subscribe để bật lại.'
    );
    return;
  }

  if (cmd === '/status') {
    const dash = await fetchDashboard(cfg);
    await sendTelegram(cfg, chatId, formatStatusText(dash));
    return;
  }

  if (cmd === '/orders') {
    const dash = await fetchDashboard(cfg);
    await sendTelegram(cfg, chatId, formatOrdersText(dash));
    return;
  }

  if (cmd === '/1u') {
    const dash = await fetchDashboard(cfg);
    await sendTelegram(cfg, chatId, format1uRatesText(dash));
    return;
  }

  if (cmd === '/stats') {
    const dash = await fetchDashboard(cfg);
    await sendTelegram(cfg, chatId, formatStatsText(dash));
  }
}

async function handleCallback(cfg, ctx, cb) {
  const chatId = cb.message.chat.id;
  const msgId = cb.message.message_id;
  const data = cb.data || '';
  await answerCallback(cfg, cb.id);

  if (await isBlocked(cfg, chatId)) {
    await sendTelegram(cfg, chatId, '⛔ Bạn không có quyền truy cập bot này.');
    return;
  }

  await upsertSubscriber(cfg, cb.message.chat, cb.from);

  if (data.startsWith('admin_ban_') || data.startsWith('admin_unban_')) {
    if (!isAdmin(cfg, chatId)) {
      await sendTelegram(cfg, chatId, '⛔ Chỉ Admin mới được thao tác.');
      return;
    }
    const isBan = data.startsWith('admin_ban_');
    const targetId = data.replace(isBan ? 'admin_ban_' : 'admin_unban_', '').trim();
    if (isBan && isAdmin(cfg, targetId)) return;
    await setUserBlocked(cfg, targetId, isBan);
    const view = await buildAdminUsersView(cfg);
    await editMessageText(cfg, chatId, msgId, view.text, view.markup);
    return;
  }

  if (data === 'cmd_status') {
    const dash = await fetchDashboard(cfg);
    await sendTelegram(cfg, chatId, formatStatusText(dash));
  } else if (data === 'cmd_orders') {
    const dash = await fetchDashboard(cfg);
    await sendTelegram(cfg, chatId, formatOrdersText(dash));
  } else if (data === 'cmd_1u') {
    const dash = await fetchDashboard(cfg);
    await sendTelegram(cfg, chatId, format1uRatesText(dash));
  } else if (data === 'cmd_stats') {
    const dash = await fetchDashboard(cfg);
    await sendTelegram(cfg, chatId, formatStatsText(dash));
  } else if (data === 'chart_back') {
    const dash = await fetchDashboard(cfg).catch(() => null);
    const openPos = dash && Array.isArray(dash.open_positions) ? dash.open_positions : [];
    await editMessageText(
      cfg,
      chatId,
      msgId,
      '🕯️ <b>XEM BIỂU ĐỒ NẾN + EMA 34/89/200 LIVE</b>\nChọn đồng Coin muốn xem:',
      buildChartCoinKeyboard(openPos)
    );
  } else if (data.startsWith('chart_coin_')) {
    const sym = data.replace('chart_coin_', '');
    await editMessageText(
      cfg,
      chatId,
      msgId,
      `🕯️ <b>${sym}</b> — Chọn khung thời gian muốn vẽ biểu đồ:`,
      buildChartTfKeyboard(sym)
    );
  } else if (data.startsWith('chart_go_')) {
    const parts = data.replace('chart_go_', '').split('_');
    if (parts.length === 2) {
      await sendChartPhoto(cfg, chatId, msgId, parts[0], parts[1]);
    }
  }
}

// ---------- Dashboard Fetch & Formatters (< 1 Mobile Screen) ----------

async function fetchDashboard(cfg) {
  const res = await fetch(`${cfg.DASHBOARD_JSON_URL}?t=${Date.now()}`);
  if (!res.ok) {
    throw new Error(`HTTP ${res.status} fetching live_dashboard.json`);
  }
  return await res.json();
}

function fmtUsd(val, signed) {
  const n = Number(val || 0);
  const absStr = Math.abs(n).toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 });
  if (signed) return (n >= 0 ? '+$' : '-$') + absStr;
  return '$' + absStr;
}

function fmtPx(px) {
  const a = Math.abs(Number(px || 0));
  if (a >= 1000) return '$' + a.toLocaleString('en-US', { minimumFractionDigits: 1, maximumFractionDigits: 1 });
  if (a >= 10) return '$' + a.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 });
  if (a >= 1) return '$' + a.toFixed(3);
  return '$' + a.toFixed(4);
}

function formatStatusText(d) {
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

  const mode = acc.turbo_active ? 'Auto-Turbo 250G ⚡' : `Bao ${buf}G`;
  const brake = acc.shadow_paused
    ? '🛡️ ĐANG PHANH SHADOW-WIN'
    : `🟢 An toàn (${acc.portfolio_consec_sl || 0}/2 SL)`;

  return (
    '💼 <b>TRẠNG THÁI QUỸ CHICKTRADE v3.3</b>\n' +
    `🕒 <code>${d.updated_at_utc || '--'}</code>\n\n` +
    `• Số dư hiện tại (Balance): <b>${fmtUsd(eq, false)}</b> (${roiPct >= 0 ? '+' : ''}${roiPct.toFixed(2)}% · ${fmtUsd(netDiff, true)})\n` +
    `• Bậc lãi kép S2: <b>Bậc ${step} (∞)</b> · 1U = <b>${fmtUsd(unitUsd, false)}</b>\n` +
    `• Lên Bậc ${step + 1} (${fmtUsd(nextEq, false)}): đạt <b>${progPct.toFixed(1)}%</b> (còn +${fmtUsd(remUsd, false)})\n` +
    `• Chế độ: <b>${mode} · ${acc.leverage || 20}× Cross</b>\n` +
    `• Phanh 2-SL Core: ${brake}\n` +
    `• Lệnh đang mở: <b>${(d.open_positions || []).length}/5</b> | Đã đóng: <b>${st.total_trades || 0}</b>`
  );
}

function formatOrdersText(d) {
  const open = d.open_positions || [];
  let out = `📂 <b>LỆNH ĐANG MỞ (${open.length}/5)</b>\n`;
  if (open.length === 0) {
    return out + '<i>Hiện không có lệnh mở — hệ thống đang canh 24/7.</i>';
  }
  open.forEach((p, idx) => {
    const totPnl = Number(p.realized_pnl_usd || 0) + Number(p.unrealized_pnl_usd || 0);
    const dirIc = p.direction === 'LONG' ? '🟢' : '🔴';
    const mustTag = p.is_must_order ? ' 🔥' : '';
    const wrLive = Number(p.setup_live_wr_pct != null ? p.setup_live_wr_pct : p.wr_pct || 0).toFixed(1);
    const pwrLive = Number(
      p.setup_live_profit_wr_pct != null ? p.setup_live_profit_wr_pct : p.profit_wr_pct || 0
    ).toFixed(1);

    out +=
      (idx > 0 ? '\n' : '') +
      `${dirIc} <b>${p.symbol} · ${String(p.tf || '').toUpperCase()} [${p.setup_code}]</b> ${p.direction}${mustTag} | PnL: <b>${fmtUsd(totPnl, true)}</b>\n` +
      `• Entry <code>${fmtPx(p.entry_price)}</code> → Live <code>${fmtPx(p.current_price)}</code> · SL <code>${fmtPx(p.active_sl_price)}</code>\n` +
      `• TP1 <code>${fmtPx(p.tp1_price)}</code> · TP2 <code>${fmtPx(p.tp2_price)}</code> | Win <b>${fmtUsd(p.expected_tp1_usd || 0, true)}/${fmtUsd(p.expected_tp2_usd || 0, true)}</b> · SL <b>${fmtUsd(p.expected_sl_usd || 0, true)}</b>\n` +
      `• WR (3Y+Live): <b>${wrLive}%</b> · Thắng tiền: <b>${pwrLive}%</b>\n`;
  });
  return out;
}

function format1uRatesText(d) {
  const rates = d.live_u3_rates || [];
  let out = '📐 <b>BẢNG TỶ LỆ 1U CHUẨN 4H (U3_HTF_MACRO_ATR)</b>\n';
  if (rates.length === 0) return out + '<i>Chưa có dữ liệu 1U.</i>';
  rates.forEach(r => {
    const coin = String(r.coin || '').replace('USDT', '');
    const d15 = Number(r.delta_pct_15m || 0);
    const arrow = d15 > 0 ? '▲+' : d15 < 0 ? '▼' : '—';
    out +=
      `• <b>${coin}</b> (${fmtPx(r.price)}): ` +
      `15m <code>${Number(r.u_pct_15m || 0).toFixed(3)}%</code> (${arrow}${d15.toFixed(4)}%) · ` +
      `30m <code>${Number(r.u_pct_30m || 0).toFixed(3)}%</code> · ` +
      `1h <code>${Number(r.u_pct_1h || 0).toFixed(3)}%</code>\n`;
  });
  return out;
}

function formatStatsText(d) {
  const acc = d.account || {};
  const st = d.stats || {};
  const closed = (d.closed_trades || []).slice(0, 5);
  let out =
    '📈 <b>THỐNG KÊ HIỆU SUẤT TÍCH LŨY</b>\n' +
    `• Tổng lệnh đã đóng: <b>${st.total_trades || 0}</b> (${st.wins || 0} Thắng)\n` +
    `• Thắng tiền (Profit WR): <b>${Number(st.profit_wr_pct || 0).toFixed(1)}%</b> | WR lệnh: <b>${Number(st.wr_pct || 0).toFixed(1)}%</b>\n` +
    `• Profit Factor: <b>${Number(st.pf || 0).toFixed(2)}</b> | MaxDD: <b>${Number(acc.max_dd_pct || 0).toFixed(2)}%</b>\n` +
    `• Lãi/Lỗ ròng đã chốt: <b>${fmtUsd(st.net_pnl_usd || 0, true)}</b>\n\n` +
    '📜 <b>5 LỆNH ĐÓNG GẦN NHẤT:</b>\n';
  if (closed.length === 0) {
    out += '<i>Chưa có lệnh nào đóng.</i>';
  } else {
    closed.forEach(t => {
      const pnl = Number(t.pnl_usd || 0);
      const ic = pnl > 0 ? '✅' : '❌';
      out += `${ic} <b>${t.symbol}</b> (${t.tf} ${t.setup_code}) ${t.outcome}: <b>${fmtUsd(pnl, true)}</b>\n`;
    });
  }
  return out;
}

// ---------- Keyboards & Telegram HTTP Helpers ----------

function isPrivateChat(chatId) {
  return Number(chatId) > 0;
}

function buildOpenAppButton(chatId, url) {
  if (!url || !url.startsWith('https://')) return null;
  if (isPrivateChat(chatId)) {
    return { text: '📊 Mở Popup Theo Dõi Lệnh Live', web_app: { url } };
  }
  return { text: '📊 Mở Dashboard Live', url };
}

function buildOpenAppInlineMarkup(chatId, url) {
  const btn = buildOpenAppButton(chatId, url);
  if (!btn) return null;
  return { inline_keyboard: [[btn]] };
}

function buildReplyKeyboard(cfg, chatId) {
  const row1 = [
    { text: '📊 Tài sản' },
    { text: '📂 Lệnh mở' },
    { text: '📐 1U (4H)' },
    { text: '📈 Thống kê' },
  ];
  const popupBtn =
    isPrivateChat(chatId) && cfg.WEBAPP_URL && cfg.WEBAPP_URL.startsWith('https://')
      ? { text: '🚀 Popup', web_app: { url: cfg.WEBAPP_URL } }
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

function buildInlineDashboardKeyboard(cfg, chatId) {
  const rows = [];
  const appBtn = buildOpenAppButton(chatId, cfg.WEBAPP_URL);
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

async function tgApi(cfg, method, body) {
  try {
    const res = await fetch(`https://api.telegram.org/bot${cfg.BOT_TOKEN}/${method}`, {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify(body),
    });
    return await res.json();
  } catch (err) {
    console.error(`tgApi ${method} error:`, err);
    return { ok: false };
  }
}

async function sendTelegram(cfg, chatId, text, replyMarkup = null, replyToMessageId = null) {
  const payload = {
    chat_id: chatId,
    text,
    parse_mode: 'HTML',
    disable_web_page_preview: true,
  };
  if (replyMarkup) payload.reply_markup = replyMarkup;
  if (replyToMessageId) {
    payload.reply_to_message_id = Number(replyToMessageId);
    payload.allow_sending_without_reply = true;
  }
  return await tgApi(cfg, 'sendMessage', payload);
}

async function editMessageText(cfg, chatId, messageId, text, replyMarkup) {
  const payload = {
    chat_id: chatId,
    message_id: messageId,
    text,
    parse_mode: 'HTML',
    disable_web_page_preview: true,
  };
  if (replyMarkup) payload.reply_markup = replyMarkup;
  return await tgApi(cfg, 'editMessageText', payload);
}

async function deleteMessage(cfg, chatId, messageId) {
  return await tgApi(cfg, 'deleteMessage', { chat_id: chatId, message_id: messageId });
}

async function answerCallback(cfg, callbackId) {
  return await tgApi(cfg, 'answerCallbackQuery', { callback_query_id: callbackId });
}

// ---------- QuickChart Live Candlestick + EMA 34/89/200 (Permanent Archive) ----------

function buildChartCoinKeyboard(openPositions = []) {
  const openBySym = {};
  for (const p of openPositions || []) {
    if (p && p.symbol) openBySym[p.symbol] = p;
  }
  const mkBtn = (sym, shortName) => {
    const pos = openBySym[sym];
    const label = pos
      ? `${pos.direction === 'LONG' ? '🟢' : '🔴'} ${shortName} (${String(pos.tf || '').toUpperCase()})`
      : shortName;
    return { text: label, callback_data: `chart_coin_${sym}` };
  };
  return {
    inline_keyboard: [
      [mkBtn('BTCUSDT', 'BTC'), mkBtn('ETHUSDT', 'ETH'), mkBtn('BNBUSDT', 'BNB')],
      [mkBtn('LINKUSDT', 'LINK'), mkBtn('XRPUSDT', 'XRP'), mkBtn('SOLUSDT', 'SOL')],
    ],
  };
}

function buildChartTfKeyboard(sym) {
  return {
    inline_keyboard: [
      [
        { text: '15m', callback_data: `chart_go_${sym}_15m` },
        { text: '30m', callback_data: `chart_go_${sym}_30m` },
        { text: '1h', callback_data: `chart_go_${sym}_1h` },
        { text: '4h', callback_data: `chart_go_${sym}_4h` },
      ],
      [{ text: '« Đổi Coin', callback_data: 'chart_back' }],
    ],
  };
}

async function sendChartCoinPicker(cfg, chatId) {
  const dash = await fetchDashboard(cfg).catch(() => null);
  const openPos = dash && Array.isArray(dash.open_positions) ? dash.open_positions : [];
  await sendTelegram(
    cfg,
    chatId,
    '🕯️ <b>XEM BIỂU ĐỒ NẾN + EMA 34/89/200 LIVE</b>\nChọn đồng Coin muốn xem:',
    buildChartCoinKeyboard(openPos)
  );
}

async function fetchSelfHostedCandles(cfg, sym, tf) {
  try {
    const res = await fetch(`${cfg.CHART_CANDLES_JSON_URL}?t=${Date.now()}`);
    if (!res.ok) return null;
    const j = await res.json();
    const rows = j && j.candles && j.candles[sym] ? j.candles[sym][tf] : null;
    if (Array.isArray(rows) && rows.length >= 20) return rows;
  } catch {}
  return null;
}

async function fetchKlines220(sym, tf) {
  const hdrs = { 'User-Agent': 'curl/8.4.0', Accept: 'application/json' };
  const base = sym.replace(/USDT$/i, '').toUpperCase();
  const errs = [];

  // 1. Binance Vision (global CDN) & Binance Futures
  const binanceUrls = [
    ['BV', `https://data-api.binance.vision/api/v3/klines?symbol=${encodeURIComponent(sym)}&interval=${encodeURIComponent(tf)}&limit=220`],
    ['BF', `https://fapi.binance.com/fapi/v1/klines?symbol=${encodeURIComponent(sym)}&interval=${encodeURIComponent(tf)}&limit=220`],
  ];
  for (const [tag, u] of binanceUrls) {
    try {
      const res = await fetch(u, { headers: hdrs });
      if (res.ok) {
        const arr = await res.json();
        if (Array.isArray(arr) && arr.length >= 20) return arr;
      }
      errs.push(`${tag}:${res.status}`);
    } catch { errs.push(`${tag}:err`); }
  }

  // 2. KuCoin Public API (works cleanly from Cloudflare Workers globally)
  try {
    const kcTf = { '15m': '15min', '30m': '30min', '1h': '1hour', '4h': '4hour' }[tf] || '1hour';
    const kcUrl = `https://api.kucoin.com/api/v1/market/candles?type=${encodeURIComponent(kcTf)}&symbol=${encodeURIComponent(base)}-USDT`;
    const res = await fetch(kcUrl, { headers: hdrs });
    if (res.ok) {
      const j = await res.json();
      if (j && Array.isArray(j.data) && j.data.length >= 20) {
        // KuCoin format: [time_sec, open, close, high, low, vol, turnover] newest-first
        return j.data.slice(0, 220).reverse().map(c => [Number(c[0]) * 1000, c[1], c[3], c[4], c[2]]);
      }
    }
    errs.push(`KC:${res.status}`);
  } catch { errs.push('KC:err'); }

  // 3. Kraken Public OHLC API (US-friendly, never 451 on Cloudflare US POPs)
  try {
    const krTf = { '15m': 15, '30m': 30, '1h': 60, '4h': 240 }[tf] || 60;
    const krPair = (base === 'BTC' ? 'XBT' : base) + 'USD';
    const krUrl = `https://api.kraken.com/0/public/OHLC?pair=${encodeURIComponent(krPair)}&interval=${krTf}`;
    const res = await fetch(krUrl, { headers: hdrs });
    if (res.ok) {
      const j = await res.json();
      const firstKey = j && j.result ? Object.keys(j.result).find(k => k !== 'last') : null;
      const arr = firstKey ? j.result[firstKey] : null;
      if (Array.isArray(arr) && arr.length >= 20) {
        return arr.slice(-220).map(c => [Number(c[0]) * 1000, String(c[1]), String(c[2]), String(c[3]), String(c[4])]);
      }
    }
    errs.push(`KR:${res.status}`);
  } catch { errs.push('KR:err'); }

  // 4. OKX Public Perpetual Swap API
  try {
    const okxBar = { '15m': '15m', '30m': '30m', '1h': '1H', '4h': '4H' }[tf] || '1H';
    const okxUrl = `https://www.okx.com/api/v5/market/candles?instId=${encodeURIComponent(base)}-USDT-SWAP&bar=${encodeURIComponent(okxBar)}&limit=100`;
    const res = await fetch(okxUrl, { headers: hdrs });
    if (res.ok) {
      const j = await res.json();
      if (j && Array.isArray(j.data) && j.data.length >= 20) {
        return j.data.slice().reverse().map(c => [Number(c[0]), c[1], c[2], c[3], c[4]]);
      }
    }
    errs.push(`OKX:${res.status}`);
  } catch { errs.push('OKX:err'); }

  // 5. Gate.io Public Futures API
  try {
    const gateUrl = `https://api.gateio.ws/api/v4/futures/usdt/candlesticks?contract=${encodeURIComponent(base)}_USDT&interval=${encodeURIComponent(tf)}&limit=220`;
    const res = await fetch(gateUrl, { headers: hdrs });
    if (res.ok) {
      const arr = await res.json();
      if (Array.isArray(arr) && arr.length >= 20) {
        return arr.map(c => [Number(c.t) * 1000, String(c.o), String(c.h), String(c.l), String(c.c)]);
      }
    }
    errs.push(`GT:${res.status}`);
  } catch { errs.push('GT:err'); }

  throw new Error(`Không tải được nến ${sym} ${tf} (${errs.join(', ')})`);
}

function computeEmaArr(closes, period) {
  if (!closes.length) return [];
  const k = 2 / (period + 1);
  const out = [closes[0]];
  for (let i = 1; i < closes.length; i++) {
    out.push(closes[i] * k + out[i - 1] * (1 - k));
  }
  return out;
}

function pad2(n) {
  return (n < 10 ? '0' : '') + n;
}

async function sendChartPhoto(cfg, chatId, pickerMsgId, sym, tf) {
  try {
    await editMessageText(cfg, chatId, pickerMsgId, `⏳ Đang tải nến &amp; vẽ biểu đồ <b>${sym} (${tf})</b>...`);

    // Priority #1: self-hosted chart_candles.json (0.1s, 0 rate limit / 0 451, engine-exact EMAs)
    const [selfRows, dashData] = await Promise.all([
      fetchSelfHostedCandles(cfg, sym, tf),
      fetchDashboard(cfg).catch(() => null),
    ]);

    let bars = [];
    if (selfRows) {
      const tail = selfRows.slice(-60);
      bars = tail.map(r => ({
        tSec: Number(r[0]),
        o: Number(r[1]),
        h: Number(r[2]),
        l: Number(r[3]),
        c: Number(r[4]),
        e34: Number(r[5]),
        e89: Number(r[6]),
        e200: Number(r[7]),
      }));
    } else {
      const raw = await fetchKlines220(sym, tf);
      const closes = raw.map(k => parseFloat(k[4]));
      const e34All = computeEmaArr(closes, 34);
      const e89All = computeEmaArr(closes, 89);
      const e200All = computeEmaArr(closes, 200);
      const startIdx = Math.max(0, raw.length - 60);
      for (let i = startIdx; i < raw.length; i++) {
        const k = raw[i];
        bars.push({
          tSec: Math.floor(Number(k[0]) / 1000),
          o: parseFloat(k[1]),
          h: parseFloat(k[2]),
          l: parseFloat(k[3]),
          c: parseFloat(k[4]),
          e34: Number(e34All[i].toFixed(6)),
          e89: Number(e89All[i].toFixed(6)),
          e200: Number(e200All[i].toFixed(6)),
        });
      }
    }

    const nShow = bars.length;
    const labels = [];
    const wicks = [];
    const bodies = [];
    const colors = [];
    const e34Show = [];
    const e89Show = [];
    const e200Show = [];

    let yMin = Infinity;
    let yMax = -Infinity;
    for (const b of bars) {
      yMin = Math.min(yMin, b.l, b.e34, b.e89, b.e200);
      yMax = Math.max(yMax, b.h, b.e34, b.e89, b.e200);
    }

    const openList = dashData && Array.isArray(dashData.open_positions) ? dashData.open_positions : [];
    const pos = openList.find(p => p.symbol === sym && p.tf === tf) || openList.find(p => p.symbol === sym) || null;
    let entryIdx = -1;
    if (pos && nShow > 0) {
      const tfSec = { '15m': 900, '30m': 1800, '1h': 3600, '4h': 14400 }[tf] || 3600;
      let rawEntrySec = Number(pos.entry_bar_ts || 0);
      if (!isFinite(rawEntrySec) || rawEntrySec <= 0) {
        const ms = Date.parse(String(pos.opened_utc || '').replace(' ', 'T') + ':00Z');
        if (isFinite(ms)) rawEntrySec = Math.floor(ms / 1000);
      }
      const alignedEntrySec = rawEntrySec > 0 ? Math.floor(rawEntrySec / tfSec) * tfSec : 0;
      if (alignedEntrySec >= bars[0].tSec && alignedEntrySec <= bars[nShow - 1].tSec) {
        for (let i = 0; i < nShow; i++) {
          if (bars[i].tSec === alignedEntrySec || (bars[i].tSec <= alignedEntrySec && (i === nShow - 1 || bars[i + 1].tSec > alignedEntrySec))) {
            entryIdx = i;
            break;
          }
        }
      }
      const lvls = [
        Number(pos.entry_price),
        Number(pos.active_sl_price),
        Number(pos.tp1_price),
        Number(pos.tp2_price),
      ].filter(v => isFinite(v) && v > 0);
      for (const lv of lvls) {
        yMin = Math.min(yMin, lv);
        yMax = Math.max(yMax, lv);
      }
    }

    const padRange = Math.max((yMax - yMin) * 0.05, yMax * 0.001);
    const minBody = Math.max((yMax - yMin) * 0.0025, yMax * 0.00015);

    for (let i = 0; i < nShow; i++) {
      const b = bars[i];
      const dtVn = new Date((b.tSec + 7 * 3600) * 1000);
      const lbl =
        tf === '4h'
          ? `${pad2(dtVn.getUTCDate())}/${pad2(dtVn.getUTCMonth() + 1)} ${pad2(dtVn.getUTCHours())}h`
          : `${pad2(dtVn.getUTCHours())}:${pad2(dtVn.getUTCMinutes())}`;
      const isUp = b.c >= b.o;
      labels.push(lbl);
      wicks.push([b.l, b.h]);
      if (Math.abs(b.c - b.o) < minBody) {
        bodies.push([b.o - minBody * 0.5, b.o + minBody * 0.5]);
      } else {
        bodies.push([Math.min(b.o, b.c), Math.max(b.o, b.c)]);
      }
      // Keep original green/red candle colors (never recolor the entry candle)
      colors.push(isUp ? '#059669' : '#DC2626');
      e34Show.push(b.e34);
      e89Show.push(b.e89);
      e200Show.push(b.e200);
    }

    const lastBar = bars[nShow - 1];
    const lastPx = lastBar.c;
    const lastE34 = lastBar.e34;
    const lastE89 = lastBar.e89;
    const lastE200 = lastBar.e200;

    const datasets = [
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
    ];

    if (pos) {
      const addHLine = (label, price, color, dashed) => {
        const p = Number(price);
        if (!isFinite(p) || p <= 0) return;
        const ds = {
          type: 'line',
          label,
          data: Array(nShow).fill(p),
          borderColor: color,
          borderWidth: dashed ? 1.6 : 2,
          pointRadius: 0,
          fill: false,
          order: 0,
        };
        if (dashed) ds.borderDash = [6, 4];
        datasets.push(ds);
      };
      addHLine(`Entry ${fmtPx(pos.entry_price)}`, pos.entry_price, '#2563EB', true);
      addHLine(`SL ${fmtPx(pos.active_sl_price)}`, pos.active_sl_price, '#DC2626', false);
      addHLine(`TP1 ${fmtPx(pos.tp1_price)}`, pos.tp1_price, '#059669', true);
      addHLine(`TP2 ${fmtPx(pos.tp2_price)}`, pos.tp2_price, '#15803D', false);
    }

    if (entryIdx >= 0 && pos) {
      const entryDotData = Array(nShow).fill(null);
      entryDotData[entryIdx] = Number(pos.entry_price);
      datasets.push({
        type: 'line',
        label: 'Entry Point',
        data: entryDotData,
        showLine: false,
        pointRadius: 7,
        pointBackgroundColor: '#2563EB',
        pointBorderColor: '#FFFFFF',
        pointBorderWidth: 2,
        order: 0,
      });
    }

    const chartCfg = {
      type: 'bar',
      data: { labels, datasets },
      options: {
        plugins: {
          title: {
            display: true,
            text: `${sym} · ${tf.toUpperCase()} (${nShow} nến · EMA 34/89/200${pos ? ` · ${pos.direction} ${pos.setup_code}` : ''})`,
            color: '#0F172A',
            font: { size: 14, weight: 'bold' },
          },
          legend: {
            display: true,
            labels: {
              color: '#334155',
              font: { size: 11 },
              filter: item => item.text !== 'Wick' && item.text !== 'Candle',
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

    const qcRes = await fetch('https://quickchart.io/chart', {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({
        version: '3',
        width: 800,
        height: 460,
        backgroundColor: '#FFFFFF',
        format: 'png',
        chart: chartCfg,
      }),
    });
    if (!qcRes.ok) {
      throw new Error(`QuickChart HTTP ${qcRes.status}`);
    }
    const pngBuf = await qcRes.arrayBuffer();

    let caption =
      `🕯️ <b>${sym} · ${tf.toUpperCase()}</b> · Giá Live: <b>${fmtPx(lastPx)}</b>\n` +
      `• EMA34: <code>${fmtPx(lastE34)}</code> · EMA89: <code>${fmtPx(lastE89)}</code> · EMA200: <code>${fmtPx(lastE200)}</code>`;

    if (pos) {
      const totPnl = Number(pos.realized_pnl_usd || 0) + Number(pos.unrealized_pnl_usd || 0);
      const dirIc = pos.direction === 'LONG' ? '🟢' : '🔴';
      const wrLive = Number(pos.setup_live_wr_pct != null ? pos.setup_live_wr_pct : pos.wr_pct || 0).toFixed(1);
      const pwrLive = Number(
        pos.setup_live_profit_wr_pct != null ? pos.setup_live_profit_wr_pct : pos.profit_wr_pct || 0
      ).toFixed(1);
      const samples = pos.setup_total_samples != null ? ` (${pos.setup_total_samples} lệnh)` : '';
      caption +=
        `\n\n${dirIc} <b>LỆNH ĐANG MỞ: ${pos.symbol} · ${String(pos.tf || '').toUpperCase()} [${pos.setup_code}] ${pos.direction}</b>\n` +
        `• Entry: <code>${fmtPx(pos.entry_price)}</code> | PnL Live: <b>${fmtUsd(totPnl, true)}</b>\n` +
        `• SL: <code>${fmtPx(pos.active_sl_price)}</code> · TP1: <code>${fmtPx(pos.tp1_price)}</code> · TP2: <code>${fmtPx(pos.tp2_price)}</code>\n` +
        `• Dự kiến Win: <b>${fmtUsd(pos.expected_tp1_usd || 0, true)} / ${fmtUsd(pos.expected_tp2_usd || 0, true)}</b> | Thua SL: <b>${fmtUsd(pos.expected_sl_usd || 0, true)}</b>\n` +
        `• WR (3Y+Live): <b>${wrLive}%</b> · Thắng tiền: <b>${pwrLive}%</b>${samples}`;
    }

    await sendPhoto(cfg, chatId, pngBuf, `${sym}_${tf}.png`, caption, null);
    await deleteMessage(cfg, chatId, pickerMsgId);
  } catch (err) {
    await editMessageText(cfg, chatId, pickerMsgId, `⚠️ Không thể vẽ biểu đồ lúc này: ${escapeHtml(String(err))}`);
  }
}

async function sendPhoto(cfg, chatId, pngBuffer, filename, caption, replyMarkup = null, replyToMessageId = null) {
  const form = new FormData();
  form.append('chat_id', String(chatId));
  form.append('photo', new Blob([pngBuffer], { type: 'image/png' }), filename);
  form.append('caption', caption);
  form.append('parse_mode', 'HTML');
  if (replyMarkup) {
    form.append('reply_markup', JSON.stringify(replyMarkup));
  }
  if (replyToMessageId) {
    form.append('reply_to_message_id', String(replyToMessageId));
    form.append('allow_sending_without_reply', 'true');
  }
  const res = await fetch(`https://api.telegram.org/bot${cfg.BOT_TOKEN}/sendPhoto`, {
    method: 'POST',
    body: form,
  });
  return await res.json();
}
