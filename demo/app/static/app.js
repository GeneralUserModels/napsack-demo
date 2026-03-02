/* ============================================================
   NAPsack Demo Dashboard – app.js
   ============================================================ */

// ── State ──────────────────────────────────────────────────────────────────
let serverState = {};
let gtData = null;
let gtChunkIdx = 0;
let frameIdx = 0;
let pairwiseTrials = null;  // array of trial objects from server
let trialIdx = 0;           // current position in the trials list
let pairwiseJudgments = {}; // keyed by "pair_id::chunk_idx"
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

// ── Step 5: Pairwise Human Evaluation ──────────────────────────────────────
async function loadPairwiseTrials() {
  const nSamples = parseInt(document.getElementById('human-n-samples').value) || 10;
  const res = await fetch(`/api/human/chunks?n=${nSamples}`);
  if (!res.ok) { appendLog('[human] Failed: ' + await res.text(), true); return; }
  const data = await res.json();
  pairwiseTrials = data.trials;
  trialIdx = 0;

  // Load existing judgments
  pairwiseJudgments = {};
  try {
    const jr = await fetch('/api/human/rankings');
    if (jr.ok) {
      const jd = await jr.json();
      (jd.judgments || []).forEach(j => {
        const key = `${j.pair_id}::${j.chunk_idx}`;
        pairwiseJudgments[key] = j;
      });
    }
  } catch (_) {}

  // Skip to first unjudged trial
  let firstOpen = pairwiseTrials.length;
  for (let i = 0; i < pairwiseTrials.length; i++) {
    const t = pairwiseTrials[i];
    const key = `${t.pair_id}::${t.chunk_idx}`;
    if (!pairwiseJudgments[key]) { firstOpen = i; break; }
  }
  trialIdx = firstOpen < pairwiseTrials.length ? firstOpen : 0;

  const doneCount = Object.keys(pairwiseJudgments).length;
  if (doneCount > 0) {
    appendLog(`[human] Loaded ${doneCount}/${pairwiseTrials.length} existing judgments, ` +
              (firstOpen < pairwiseTrials.length
                ? `starting at trial ${firstOpen + 1}`
                : 'all done — showing trial 1'));
  }

  document.getElementById('human-eval-ui').classList.remove('hidden');
  renderTrial();
}

function renderTrial() {
  if (!pairwiseTrials) return;
  const trial = pairwiseTrials[trialIdx];
  const total = pairwiseTrials.length;
  const doneCount = Object.keys(pairwiseJudgments).length;
  const key = `${trial.pair_id}::${trial.chunk_idx}`;
  const judged = !!pairwiseJudgments[key];

  document.getElementById('human-chunk-indicator').textContent =
    `Trial ${trialIdx + 1} / ${total} (${doneCount} done)` + (judged ? ' ✓' : '');

  // GT video
  const vidContainer = document.getElementById('human-gt-video-container');
  const vidEl = document.getElementById('human-gt-video');
  if (trial.gt_video) {
    vidContainer.classList.remove('hidden');
    vidEl.src = `/api/media/${encodeURIComponent(trial.gt_video)}`;
  } else {
    vidContainer.classList.add('hidden');
  }

  // GT captions
  document.getElementById('human-gt-captions').innerHTML =
    trial.gt_captions.map(c => `<p>${escHtml(c.caption ?? '')}</p>`).join('');

  // Left candidate
  document.getElementById('pw-left-captions').innerHTML =
    trial.left.captions.map(c => `<p>${escHtml(c.caption ?? '')}</p>`).join('') || '<em>No captions</em>';

  // Right candidate
  document.getElementById('pw-right-captions').innerHTML =
    trial.right.captions.map(c => `<p>${escHtml(c.caption ?? '')}</p>`).join('') || '<em>No captions</em>';

  // Highlight previous choice if any
  const leftCard = document.getElementById('pairwise-left');
  const rightCard = document.getElementById('pairwise-right');
  leftCard.classList.remove('pw-selected', 'pw-draw');
  rightCard.classList.remove('pw-selected', 'pw-draw');
  if (judged) {
    const prev = pairwiseJudgments[key];
    if (prev.draw) {
      leftCard.classList.add('pw-draw');
      rightCard.classList.add('pw-draw');
    } else if (prev.winner_method === trial.left.method_hidden) {
      leftCard.classList.add('pw-selected');
    } else {
      rightCard.classList.add('pw-selected');
    }
  }
}

async function pickWinner(side) {
  if (!pairwiseTrials) return;
  const trial = pairwiseTrials[trialIdx];
  const isDraw = side === 'draw';

  const body = isDraw
    ? {
        trial_idx: trial.trial_idx,
        pair_id: trial.pair_id,
        chunk_idx: trial.chunk_idx,
        draw: true,
        method_a: trial.left.method_hidden,
        method_b: trial.right.method_hidden,
      }
    : {
        trial_idx: trial.trial_idx,
        pair_id: trial.pair_id,
        chunk_idx: trial.chunk_idx,
        draw: false,
        winner_method: (side === 'left' ? trial.left : trial.right).method_hidden,
        loser_method:  (side === 'left' ? trial.right : trial.left).method_hidden,
      };

  const res = await fetch('/api/human/rank', {
    method: 'POST', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify(body),
  });
  await res.json();

  const key = `${trial.pair_id}::${trial.chunk_idx}`;
  pairwiseJudgments[key] = body;

  const doneCount = Object.keys(pairwiseJudgments).length;
  document.getElementById('human-status').textContent =
    `✓ Saved trial ${trialIdx + 1} (${doneCount}/${pairwiseTrials.length} done)`;
  if (isDraw) {
    appendLog(`[human] Trial ${trialIdx + 1}: draw (${trial.left.method_hidden} = ${trial.right.method_hidden})`);
  } else {
    const winner = side === 'left' ? trial.left : trial.right;
    const loser  = side === 'left' ? trial.right : trial.left;
    appendLog(`[human] Trial ${trialIdx + 1}: ${winner.label} wins (${winner.method_hidden} > ${loser.method_hidden})`);
  }

  // Highlight selection
  const leftCard = document.getElementById('pairwise-left');
  const rightCard = document.getElementById('pairwise-right');
  leftCard.classList.remove('pw-selected', 'pw-draw');
  rightCard.classList.remove('pw-selected', 'pw-draw');
  if (isDraw) {
    leftCard.classList.add('pw-draw');
    rightCard.classList.add('pw-draw');
  } else if (side === 'left') {
    leftCard.classList.add('pw-selected');
  } else {
    rightCard.classList.add('pw-selected');
  }

  // Update indicator
  document.getElementById('human-chunk-indicator').textContent =
    `Trial ${trialIdx + 1} / ${pairwiseTrials.length} (${doneCount} done) ✓`;

  // Auto-advance after short delay
  setTimeout(() => {
    if (trialIdx < pairwiseTrials.length - 1) {
      trialIdx++;
      renderTrial();
    }
  }, 400);
}

function prevTrial() {
  if (!pairwiseTrials || trialIdx === 0) return;
  trialIdx--; renderTrial();
}
function nextTrial() {
  if (!pairwiseTrials || trialIdx >= pairwiseTrials.length - 1) return;
  trialIdx++; renderTrial();
}

async function finalizeHuman() {
  const res = await fetch('/api/human/finalize', { method: 'POST' });
  if (!res.ok) { appendLog('[human] Finalize failed: ' + await res.text(), true); return; }
  const data = await res.json();
  const methods = data.methods || {};
  for (const [m, r] of Object.entries(methods)) {
    appendLog(`[human] ${m}: win_rate=${r.win_rate.toFixed(3)} [${r.ci_lo.toFixed(3)}, ${r.ci_hi.toFixed(3)}]`);
  }
  await loadResults();
}

// ── Results ────────────────────────────────────────────────────────────────
async function loadResults() {
  const res = await fetch('/api/results');
  const data = await res.json();

  const methods = data.methods || ['naive', 'split', 'split_compress', 'split_compress_io'];
  const sizes = data.mp4_sizes_mb || {};
  const judgeData = data.judge?.methods || {};
  const humanWinRates = data.human_eval?.methods || {};

  // Table
  const tbody = document.getElementById('results-tbody');
  tbody.innerHTML = '';
  methods.forEach(m => {
    const jd = judgeData[m] || {};
    const mean = jd.mean != null ? jd.mean.toFixed(2) : '–';
    const se = jd.bootstrap_se != null ? jd.bootstrap_se.toFixed(2) : '–';
    const wr = humanWinRates[m];
    const winRateStr = wr != null
      ? `${(wr.win_rate * 100).toFixed(1)}% [${(wr.ci_lo * 100).toFixed(1)}, ${(wr.ci_hi * 100).toFixed(1)}]`
      : '–';
    const sz = sizes[m] != null ? `${sizes[m]} MB` : '–';
    const tr = document.createElement('tr');
    tr.innerHTML = `<td>${escHtml(m)}</td><td>${mean} ± ${se}</td><td>${winRateStr}</td><td>${sz}</td>`;
    tbody.appendChild(tr);
  });
  document.getElementById('results-table-container').classList.remove('hidden');
}

// ── Utilities ──────────────────────────────────────────────────────────────
function escHtml(s) {
  return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}

// Initial load
pollStatus();
