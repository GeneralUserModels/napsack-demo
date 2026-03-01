/* ============================================================
   NAPsack Demo Dashboard – app.js
   ============================================================ */

// ── State ──────────────────────────────────────────────────────────────────
let serverState = {};
let gtData = null;
let gtChunkIdx = 0;
let frameIdx = 0;
let humanChunks = null;
let humanChunkIdx = 0;
let activeStep = null;

// ── Lightbox ───────────────────────────────────────────────────────────────
function openLightbox(src) {
  const overlay = document.getElementById('lightbox-overlay');
  const img = document.getElementById('lightbox-img');
  img.src = src;
  overlay.classList.remove('hidden');
}

function closeLightbox() {
  document.getElementById('lightbox-overlay').classList.add('hidden');
}

document.addEventListener('keydown', e => {
  if (e.key === 'Escape') closeLightbox();
});

// ── SSE Log ────────────────────────────────────────────────────────────────
const logEl = document.getElementById('log-console');

function appendLog(msg, isErr = false) {
  const line = document.createElement('div');
  line.className = 'log-line' + (isErr ? ' log-error' : '');
  line.textContent = msg;
  logEl.appendChild(line);
  logEl.scrollTop = logEl.scrollHeight;
}

function clearLog() { logEl.innerHTML = ''; }

(function initSSE() {
  const es = new EventSource('/events');
  es.addEventListener('log', e => {
    const msg = e.data;
    // Intercept __progress__ messages (sent by single_judge.py, double-wrapped by server)
    // Format: [judge] [judge] __progress__ N M label
    const pm = msg.match(/^\[([a-z_]+)\] (?:\[[a-z_]+\] )?__progress__ (\d+) (\d+) (.*)$/);
    if (pm) { _updateProgress(pm[1], parseInt(pm[2]), parseInt(pm[3]), pm[4]); return; }
    // Intercept CHUNK_PROGRESS messages (sent by processor.py)
    const cp = msg.match(/^\[([a-z_]+)\] CHUNK_PROGRESS: (\d+)\/(\d+)$/);
    if (cp) { _updateProgress(cp[1], parseInt(cp[2]), parseInt(cp[3]), `chunk ${cp[2]}/${cp[3]}`); return; }
    appendLog(msg);
  });
  es.addEventListener('error', () => appendLog('[SSE] connection lost', true));
})();

// ── Progress bars ──────────────────────────────────────────────────────────
function _updateProgress(key, done, total, label) {
  const pct = total > 0 ? Math.round(done / total * 100) : 0;
  if (key === 'judge') {
    const wrap = document.getElementById('judge-progress-wrap');
    const fill = document.getElementById('judge-progress-fill');
    const lbl  = document.getElementById('judge-progress-label');
    if (wrap) wrap.classList.remove('hidden');
    if (fill) { fill.classList.remove('indeterminate'); fill.style.width = pct + '%'; }
    if (lbl)  lbl.textContent = `${pct}% – ${label}`;
  } else {
    const wrap = document.getElementById(`progress-wrap-${key}`);
    const fill = document.getElementById(`progress-${key}`);
    if (wrap) wrap.classList.remove('hidden');
    if (fill) { fill.classList.remove('indeterminate'); fill.style.width = pct + '%'; }
  }
}

// ── Step card selection ────────────────────────────────────────────────────
function selectStep(step) {
  activeStep = step;
  // Update card highlights
  document.querySelectorAll('.step-card').forEach(c => {
    c.classList.toggle('active', parseInt(c.dataset.step) === step);
  });
  // Show/hide canvases
  document.querySelectorAll('.canvas').forEach(c => c.classList.add('hidden'));
  const canvas = document.getElementById(`canvas-${step}`);
  if (canvas) canvas.classList.remove('hidden');
}

// ── Session directory dropdown ─────────────────────────────────────────────
async function loadSessionList() {
  try {
    const res = await fetch('/api/sessions/list');
    const data = await res.json();
    const sel = document.getElementById('session-dropdown');
    sel.innerHTML = '<option value="">— Select session —</option>';
    (data.sessions || []).forEach(s => {
      const opt = document.createElement('option');
      opt.value = s;
      // Show just the directory name for readability
      opt.textContent = s.split('/').pop() || s;
      sel.appendChild(opt);
    });
  } catch (_) {}
}

document.addEventListener('DOMContentLoaded', () => {
  loadSessionList();

  document.getElementById('session-dropdown').addEventListener('change', (e) => {
    const val = e.target.value;
    if (val) {
      document.getElementById('session-dir-input').value = val;
      loadSession();
    }
  });
});

// ── Polling ────────────────────────────────────────────────────────────────
let _initialSessionLoaded = false;
let _appRevealed = false;

async function pollStatus() {
  try {
    const res = await fetch('/api/status');
    serverState = await res.json();
    applyState(serverState);

    if (!_appRevealed) {
      _appRevealed = true;
      document.getElementById('loading-overlay').classList.add('hidden');
      document.getElementById('app-wrapper').classList.remove('hidden');
    }

    if (!_initialSessionLoaded) {
      _initialSessionLoaded = true;
      const inp = document.getElementById('session-dir-input');
      const sel = document.getElementById('session-dropdown');
      if (serverState.session_dir) {
        // Server already has an active session (restored from state.json) – sync UI
        inp.value = serverState.session_dir;
        sel.value = serverState.session_dir;
        document.getElementById('session-status').textContent =
          `✓ ${serverState.session_dir.split('/').pop()}`;
      } else if (serverState.default_session_dir) {
        // Just pre-fill the input, don't auto-load
        inp.value = serverState.default_session_dir;
      }
    }
  } catch (_) {}
}
setInterval(pollStatus, 3000);
setTimeout(pollStatus, 50);

function applyState(s) {
  // Gemini key warning
  const warn = document.getElementById('gemini-warning');
  if (!s.gemini_key_ok) warn.classList.remove('hidden');
  else warn.classList.add('hidden');

  // Recording button
  const recBtn = document.getElementById('record-btn');
  const recStatus = document.getElementById('record-status');
  if (s.recording && s.recording.running) {
    recBtn.textContent = '⏹ Stop Recording';
    recBtn.classList.add('btn-red');
    recStatus.textContent = '🔴 Recording in progress…';
    _setBadge(1, 'running', '●');
  } else {
    recBtn.textContent = '▶ Start Recording';
    if (s.recording?.end_time) {
      recStatus.textContent = `✓ Stopped at ${new Date(s.recording.end_time * 1000).toLocaleTimeString()}`;
      _setBadge(1, 'done', '✓');
    }
  }

  // Processing statuses
  const methods = ['naive', 'split', 'split_compress', 'split_compress_io'];
  let allDone = true;
  let anyRunning = false;
  methods.forEach(m => {
    const st = (s.processing?.status || {})[m] || 'pending';
    const sz = (s.processing?.mp4_sizes_mb || {})[m];
    const el = document.getElementById(`status-${m}`);
    const szEl = document.getElementById(`size-${m}`);
    if (el) {
      el.textContent = st;
      el.className = 'method-status ' + st;
    }
    if (szEl) szEl.textContent = sz != null ? `${sz} MB` : '–';
    if (st !== 'done') allDone = false;
    if (st === 'running') anyRunning = true;
    // Progress bar
    const wrap = document.getElementById(`progress-wrap-${m}`);
    const fill = document.getElementById(`progress-${m}`);
    if (wrap && fill) {
      if (st === 'pending') {
        wrap.classList.add('hidden');
        fill.style.width = '0%';
        fill.className = 'method-progress-fill';
      } else if (st === 'running') {
        wrap.classList.remove('hidden');
        if (!fill.style.width || fill.style.width === '0%') {
          fill.classList.add('indeterminate');
        }
      } else if (st === 'done') {
        wrap.classList.remove('hidden');
        fill.classList.remove('indeterminate');
        fill.className = 'method-progress-fill';
        fill.style.width = '100%';
      } else if (st === 'error') {
        wrap.classList.remove('hidden');
        fill.classList.remove('indeterminate');
        fill.className = 'method-progress-fill error-fill';
        fill.style.width = '100%';
      }
    }
  });
  if (allDone && methods.some(m => (s.processing?.status || {})[m])) _setBadge(2, 'done', '✓');
  else if (anyRunning) _setBadge(2, 'running', '●');

  // GT badge
  if (s.gt?.done) _setBadge(3, 'done', '✓');

  // Judge status
  const judgeStatus = document.getElementById('judge-status');
  if (judgeStatus && s.judge) {
    const js = s.judge.status || 'pending';
    judgeStatus.textContent = js === 'done' ? '✓ Judge complete' : js === 'running' ? '⏳ Running…' : js;
    if (js === 'done') _setBadge(4, 'done', '✓');
    else if (js === 'running') _setBadge(4, 'running', '●');
    // Judge progress bar
    const jWrap = document.getElementById('judge-progress-wrap');
    const jFill = document.getElementById('judge-progress-fill');
    const jLbl  = document.getElementById('judge-progress-label');
    if (jWrap && jFill) {
      if (js === 'pending') {
        jWrap.classList.add('hidden');
        jFill.style.width = '0%';
        jFill.className = 'judge-progress-fill';
      } else if (js === 'running') {
        jWrap.classList.remove('hidden');
        if (!jFill.style.width || jFill.style.width === '0%') {
          jFill.classList.add('indeterminate');
        }
      } else if (js === 'done') {
        jWrap.classList.remove('hidden');
        jFill.classList.remove('indeterminate');
        jFill.style.width = '100%';
        if (jLbl && jLbl.textContent === '') jLbl.textContent = 'Complete';
      }
    }
  }

  // Human eval badge
  if (s.human_eval?.status === 'done') _setBadge(5, 'done', '✓');
}

function _setBadge(step, cls, text) {
  const badge = document.getElementById(`badge-${step}`);
  if (!badge) return;
  badge.className = 'step-badge ' + cls;
  badge.textContent = text;
}

// ── Session ────────────────────────────────────────────────────────────────
async function loadSession() {
  const dir = document.getElementById('session-dir-input').value.trim();
  if (!dir) return alert('Enter a session directory path');
  const res = await fetch('/api/session', {
    method: 'POST', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({ session_dir: dir })
  });
  const data = await res.json();
  serverState = data;
  applyState(data);
  document.getElementById('session-status').textContent = `✓ ${dir.split('/').pop()}`;
  appendLog(`[session] Loaded: ${dir}`);

  // Sync dropdown selection
  const sel = document.getElementById('session-dropdown');
  sel.value = dir;

  // Refresh session list in case this was a new session
  loadSessionList();

  // Auto-select step 1 if none selected
  if (!activeStep) selectStep(1);
}

// ── Step 1: Recording ──────────────────────────────────────────────────────
async function toggleRecording() {
  if (!serverState?.session_dir) {
    await loadSession();
    if (!serverState?.session_dir) {
      appendLog('[record] Cannot start — no session directory set', true);
      return;
    }
  }
  if (serverState?.recording?.running) {
    const res = await fetch('/api/record/stop', { method: 'POST' });
    if (!res.ok) appendLog('[record] Stop failed: ' + await res.text(), true);
    else appendLog('[record] Stop requested');
  } else {
    const w = parseInt(document.getElementById('max-res-w').value) || 1920;
    const h = parseInt(document.getElementById('max-res-h').value) || 1080;
    const res = await fetch('/api/record/start', {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({ max_res: [w, h] })
    });
    if (!res.ok) appendLog('[record] Start failed: ' + await res.text(), true);
    else appendLog('[record] Start requested');
  }
  await pollStatus();
}

// ── Step 2: Processing ─────────────────────────────────────────────────────
async function runMethod(method) {
  if (method === 'split_compress_io') {
    const scStatus = (serverState.processing?.status || {})['split_compress'];
    if (scStatus !== 'done') {
      appendLog('[process] Cannot run split_compress_io: "split + compress" must be completed first.', true);
      alert('"+ split + compress" must be completed before running "+ split + compress + IO".');
      return;
    }
  }
  const workers = parseInt(document.getElementById('caption-workers').value) || 4;
  const res = await fetch(`/api/process/${method}`, {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({ num_workers: workers })
  });
  const data = await res.json();
  appendLog(`[process] ${method}: ${data.status}`);
  await pollStatus();
}

// ── Step 3: GT Annotation ──────────────────────────────────────────────────
async function loadGT() {
  const res = await fetch('/api/gt');
  if (!res.ok) { appendLog('[gt] Failed to load: ' + await res.text(), true); return; }
  gtData = await res.json();
  gtChunkIdx = 0;
  frameIdx = 0;

  if (gtData.existing_gt && gtData.existing_gt.length > 0) {
    const chunkSize = 8;
    for (let i = 0; i < gtData.chunks.length; i++) {
      const slice = gtData.existing_gt.slice(i * chunkSize, (i + 1) * chunkSize);
      if (slice.length > 0) gtData.chunks[i].captions = slice;
    }
  }

  document.getElementById('gt-nav').classList.remove('hidden');
  renderGTChunk();
}

function renderGTChunk() {
  if (!gtData) return;
  const chunk = gtData.chunks[gtChunkIdx];
  const total = gtData.chunks.length;
  document.getElementById('gt-chunk-indicator').textContent = `Chunk ${gtChunkIdx + 1} / ${total}`;

  const lines = chunk.captions.map(c => c.caption ?? '');
  document.getElementById('gt-textarea').value = lines.join('\n');

  frameIdx = 0;
  renderGTFrame();
}

function renderGTFrame() {
  if (!gtData) return;
  const chunk = gtData.chunks[gtChunkIdx];
  const frames = chunk.frames || [];
  const imgEl = document.getElementById('gt-frame-img');
  const labelEl = document.getElementById('gt-frame-label');
  const indicator = document.getElementById('gt-frame-indicator');

  if (frames.length === 0) {
    imgEl.src = '';
    imgEl.style.display = 'none';
    labelEl.textContent = '';
    indicator.textContent = 'No frames';
    return;
  }

  const idx = Math.min(frameIdx, frames.length - 1);
  const frame = frames[idx];
  if (frame) {
    imgEl.src = `/api/media/${encodeURIComponent(frame)}`;
    imgEl.style.display = 'block';
    imgEl.onclick = () => openLightbox(imgEl.src);
  } else {
    imgEl.src = '';
    imgEl.style.display = 'none';
  }

  const cap = chunk.captions?.[idx];
  if (cap) {
    const start = cap.start ?? cap.start_time ?? '?';
    const end = cap.end ?? cap.end_time ?? '?';
    labelEl.textContent = `Caption ${idx + 1}: [${start} → ${end}]`;
  } else {
    labelEl.textContent = '';
  }
  indicator.textContent = `${idx + 1} / ${frames.length}`;
}

function prevGTFrame() {
  if (!gtData) return;
  frameIdx = Math.max(0, frameIdx - 1);
  renderGTFrame();
  _highlightTextareaLine(frameIdx);
}

function nextGTFrame() {
  if (!gtData) return;
  const maxF = (gtData.chunks[gtChunkIdx].frames?.length ?? 1) - 1;
  frameIdx = Math.min(maxF, frameIdx + 1);
  renderGTFrame();
  _highlightTextareaLine(frameIdx);
}

function _highlightTextareaLine(lineIdx) {
  const ta = document.getElementById('gt-textarea');
  const lines = ta.value.split('\n');
  if (lineIdx < 0 || lineIdx >= lines.length) return;
  let start = 0;
  for (let i = 0; i < lineIdx; i++) start += lines[i].length + 1;
  const end = start + lines[lineIdx].length;
  ta.focus();
  ta.setSelectionRange(start, end);
}

document.addEventListener('DOMContentLoaded', () => {
  const ta = document.getElementById('gt-textarea');
  if (ta) {
    ta.addEventListener('click', () => {
      const pos = ta.selectionStart;
      const before = ta.value.substring(0, pos);
      const lineIdx = before.split('\n').length - 1;
      if (lineIdx !== frameIdx) {
        frameIdx = lineIdx;
        renderGTFrame();
      }
    });
  }
});

function prevGTChunk() {
  if (!gtData || gtChunkIdx === 0) return;
  _syncTextareaToChunk();
  gtChunkIdx--; frameIdx = 0; renderGTChunk();
}
function nextGTChunk() {
  if (!gtData || gtChunkIdx >= gtData.chunks.length - 1) return;
  _syncTextareaToChunk();
  gtChunkIdx++; frameIdx = 0; renderGTChunk();
}

function _syncTextareaToChunk() {
  if (!gtData) return;
  const ta = document.getElementById('gt-textarea');
  const lines = ta.value.split('\n').filter(l => l.trim() !== '');
  const chunk = gtData.chunks[gtChunkIdx];
  const updatedCaptions = [];
  for (let i = 0; i < lines.length; i++) {
    if (i < chunk.captions.length) {
      const orig = chunk.captions[i];
      updatedCaptions.push({ ...orig, caption: lines[i] });
    } else {
      updatedCaptions.push({ start: '', end: '', caption: lines[i] });
    }
  }
  chunk.captions = updatedCaptions;
}

document.addEventListener('keydown', e => {
  if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;
  if (e.key === 'ArrowLeft') prevGTFrame();
  if (e.key === 'ArrowRight') nextGTFrame();
});

async function saveGT() {
  if (!gtData) return;
  _syncTextareaToChunk();
  const allCaptions = gtData.chunks.flatMap(c => c.captions);
  const res = await fetch('/api/gt/save', {
    method: 'POST', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({ captions: allCaptions })
  });
  const data = await res.json();
  appendLog(`[gt] Saved ${data.num_captions} captions ✓`);
  await pollStatus();
}

// ── Step 4: LLM Judge ──────────────────────────────────────────────────────
async function runJudge() {
  const runs = parseInt(document.getElementById('judge-runs').value) || 3;
  const workers = parseInt(document.getElementById('judge-workers').value) || 4;
  const res = await fetch('/api/judge/run', {
    method: 'POST', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({ runs, num_workers: workers })
  });
  const data = await res.json();
  appendLog(`[judge] ${data.status}`);
  await pollStatus();
}

// ── Step 5: Human Evaluation ───────────────────────────────────────────────
async function loadHumanChunks() {
  const res = await fetch('/api/human/chunks');
  if (!res.ok) { appendLog('[human] Failed: ' + await res.text(), true); return; }
  const data = await res.json();
  humanChunks = data.chunks;
  humanChunkIdx = 0;
  document.getElementById('human-eval-ui').classList.remove('hidden');
  renderHumanChunk();
}

function renderHumanChunk() {
  if (!humanChunks) return;
  const chunk = humanChunks[humanChunkIdx];
  document.getElementById('human-chunk-indicator').textContent =
    `Chunk ${humanChunkIdx + 1} / ${humanChunks.length}`;

  // GT video
  const vidContainer = document.getElementById('human-gt-video-container');
  const vidEl = document.getElementById('human-gt-video');
  if (chunk.gt_video) {
    vidContainer.classList.remove('hidden');
    vidEl.src = `/api/media/${encodeURIComponent(chunk.gt_video)}`;
  } else {
    vidContainer.classList.add('hidden');
  }

  // GT captions
  const gtEl = document.getElementById('human-gt-captions');
  gtEl.innerHTML = chunk.gt_captions.map(c =>
    `<p>${escHtml(c.caption ?? '')}</p>`
  ).join('');

  // Candidates
  const panel = document.getElementById('human-candidates-panel');
  panel.innerHTML = '';
  chunk.candidates.forEach(cand => {
    const card = document.createElement('div');
    card.className = 'candidate-card';
    const capText = cand.captions.map(c =>
      `<p>${escHtml(c.caption ?? '')}</p>`
    ).join('');
    card.innerHTML = `
      <h4>Candidate ${cand.slot}</h4>
      <div class="caption-list" style="max-height:260px">${capText || '<em>No captions</em>'}</div>
      <div class="rank-input">
        <label>Rank (1=best):</label>
        <input type="number" min="1" max="4" id="rank-slot-${cand.slot}"
               data-slot="${cand.slot}" data-method="${cand.method_hidden}"
               oninput="onRankInput()" />
        <span class="rank-saved" id="rank-saved-${cand.slot}"></span>
      </div>
    `;
    panel.appendChild(card);
  });
}

/** Auto-save ranking when all 4 rank inputs are filled */
async function onRankInput() {
  if (!humanChunks) return;
  const chunk = humanChunks[humanChunkIdx];
  const slots = chunk.candidates.map(cand => {
    const inp = document.getElementById(`rank-slot-${cand.slot}`);
    return {
      slot: cand.slot,
      method: cand.method_hidden,
      rank: inp ? parseInt(inp.value) || null : null,
    };
  });

  const filled = slots.filter(s => s.rank !== null);
  if (filled.length < chunk.candidates.length) return; // not all filled yet

  // All filled → auto-save
  const res = await fetch('/api/human/rank', {
    method: 'POST', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({ chunk_idx: humanChunkIdx, slots })
  });
  const data = await res.json();
  document.getElementById('human-status').textContent =
    `✓ Auto-saved chunk ${humanChunkIdx + 1} (total: ${data.total})`;
  appendLog(`[human] Chunk ${humanChunkIdx + 1} ranked (auto-saved)`);

  // Show saved indicator on each input
  chunk.candidates.forEach(cand => {
    const indicator = document.getElementById(`rank-saved-${cand.slot}`);
    if (indicator) indicator.textContent = '✓';
  });
}

function prevHumanChunk() {
  if (!humanChunks || humanChunkIdx === 0) return;
  humanChunkIdx--; renderHumanChunk();
}
function nextHumanChunk() {
  if (!humanChunks || humanChunkIdx >= humanChunks.length - 1) return;
  humanChunkIdx++; renderHumanChunk();
}

async function finalizeHuman() {
  const res = await fetch('/api/human/finalize', { method: 'POST' });
  if (!res.ok) { appendLog('[human] Finalize failed: ' + await res.text(), true); return; }
  const data = await res.json();
  appendLog(`[human] Pearson r=${data.pearson_r.toFixed(3)}  Spearman ρ=${data.spearman_rho.toFixed(3)}`);
  await loadResults();
}

// ── Results ────────────────────────────────────────────────────────────────
async function loadResults() {
  const res = await fetch('/api/results');
  const data = await res.json();

  const methods = data.methods || ['naive', 'split', 'split_compress', 'split_compress_io'];
  const sizes = data.mp4_sizes_mb || {};
  const judgeData = data.judge?.methods || {};
  const humanRanks = data.human_eval?.mean_human_rank || {};

  // Table
  const tbody = document.getElementById('results-tbody');
  tbody.innerHTML = '';
  methods.forEach(m => {
    const jd = judgeData[m] || {};
    const mean = jd.mean != null ? jd.mean.toFixed(2) : '–';
    const se = jd.bootstrap_se != null ? jd.bootstrap_se.toFixed(2) : '–';
    const rank = humanRanks[m] != null ? humanRanks[m].toFixed(2) : '–';
    const sz = sizes[m] != null ? `${sizes[m]} MB` : '–';
    const tr = document.createElement('tr');
    tr.innerHTML = `<td>${escHtml(m)}</td><td>${mean} ± ${se}</td><td>${rank}</td><td>${sz}</td>`;
    tbody.appendChild(tr);
  });
  document.getElementById('results-table-container').classList.remove('hidden');

  // Correlation table
  if (data.human_eval) {
    const he = data.human_eval;
    const cTbody = document.getElementById('corr-tbody');
    cTbody.innerHTML = `
      <tr><td>Pearson <em>r</em></td><td>${he.pearson_r?.toFixed(3) ?? '–'}</td><td>${he.pearson_p?.toFixed(4) ?? '–'}</td></tr>
      <tr><td>Spearman <em>ρ</em></td><td>${he.spearman_rho?.toFixed(3) ?? '–'}</td><td>${he.spearman_p?.toFixed(4) ?? '–'}</td></tr>
    `;
    document.getElementById('correlation-container').classList.remove('hidden');
  }
}

// ── Utilities ──────────────────────────────────────────────────────────────
function escHtml(s) {
  return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}

// Initial load
pollStatus();
