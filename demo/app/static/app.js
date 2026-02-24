/* ============================================================
   NAPsack Demo Dashboard – app.js
   ============================================================ */

// ── State ──────────────────────────────────────────────────────────────────
let serverState = {};
let gtData = null;          // {chunks, existing_gt, gt_videos, num_chunks}
let gtChunkIdx = 0;
let frameIdx = 0;
let humanChunks = null;     // array from /api/human/chunks
let humanChunkIdx = 0;

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
  es.addEventListener('log', e => appendLog(e.data));
  es.addEventListener('error', () => appendLog('[SSE] connection lost', true));
})();

// ── Polling ────────────────────────────────────────────────────────────────
let _initialSessionLoaded = false;
let _appRevealed = false;
async function pollStatus() {
  try {
    const res = await fetch('/api/status');
    serverState = await res.json();
    applyState(serverState);

    // Hide loading overlay and reveal app on first successful response
    if (!_appRevealed) {
      _appRevealed = true;
      document.getElementById('loading-overlay').classList.add('hidden');
      document.getElementById('app-wrapper').classList.remove('hidden');
    }

    // On very first poll, pre-populate input and auto-load default session
    const inp = document.getElementById('session-dir-input');
    if (!_initialSessionLoaded && serverState.default_session_dir) {
      _initialSessionLoaded = true;
      if (!inp.value) inp.value = serverState.default_session_dir;
      // Auto-load if no session is active yet
      if (!serverState.session_dir) {
        await loadSession();
      }
    }
  } catch (_) {}
}
setInterval(pollStatus, 3000);
// Defer first poll so the loading overlay has time to paint
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
  } else {
    recBtn.textContent = '▶ Start Recording';
    recStatus.textContent = s.recording?.end_time
      ? `✓ Stopped at ${new Date(s.recording.end_time * 1000).toLocaleTimeString()}`
      : '';
  }

  // Processing statuses
  const methods = ['naive', 'split', 'split_compress', 'split_compress_io'];
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
  });

  // Judge status
  const judgeStatus = document.getElementById('judge-status');
  if (judgeStatus && s.judge) {
    const js = s.judge.status || 'pending';
    judgeStatus.textContent = js === 'done' ? '✓ Judge complete' : js === 'running' ? '⏳ Running…' : js;
  }
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
  document.getElementById('session-status').textContent = `✓ Session: ${dir}`;
  appendLog(`[session] Loaded: ${dir}`);
}

// ── Step 1: Recording ──────────────────────────────────────────────────────
async function toggleRecording() {
  // Ensure session is loaded before starting
  if (!serverState?.session_dir) {
    await loadSession();
    if (!serverState?.session_dir) {
      appendLog('[record] Cannot start — no session directory set', true);
      return;
    }
  }
  if (serverState?.recording?.running) {
    const res = await fetch('/api/record/stop', { method: 'POST' });
    if (!res.ok) { appendLog('[record] Stop failed: ' + await res.text(), true); }
    else { appendLog('[record] Stop requested'); }
  } else {
    const w = parseInt(document.getElementById('max-res-w').value) || 1920;
    const h = parseInt(document.getElementById('max-res-h').value) || 1080;
    const res = await fetch('/api/record/start', {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({ max_res: [w, h] })
    });
    if (!res.ok) { appendLog('[record] Start failed: ' + await res.text(), true); }
    else { appendLog('[record] Start requested'); }
  }
  await pollStatus();
}

// ── Step 2: Processing ─────────────────────────────────────────────────────
async function runMethod(method) {
  // Enforce dependency: split_compress_io requires split_compress to be done
  if (method === 'split_compress_io') {
    const scStatus = (serverState.processing?.status || {})['split_compress'];
    if (scStatus !== 'done') {
      appendLog('[process] Cannot run split_compress_io: "split + compress" must be completed first.', true);
      alert('"+ split + compress" must be completed before running "+ split + compress + IO".');
      return;
    }
  }
  const res = await fetch(`/api/process/${method}`, { method: 'POST' });
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

  // Pre-fill from existing GT if available
  if (gtData.existing_gt && gtData.existing_gt.length > 0) {
    // Re-distribute existing GT back into chunk structure
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

  // Frame viewer
  renderFrame(chunk, frameIdx);

  // GT video
  const gtVideo = gtData.gt_videos?.[gtChunkIdx];
  const vidContainer = document.getElementById('gt-video-container');
  const vidEl = document.getElementById('gt-video');
  if (gtVideo) {
    vidContainer.classList.remove('hidden');
    vidEl.src = `/api/media/${encodeURIComponent(gtVideo)}`;
  } else {
    vidContainer.classList.add('hidden');
  }

  // Caption editor
  renderCaptionEditor(chunk.captions);
}

function renderFrame(chunk, idx) {
  const imgEl = document.getElementById('frame-img');
  const labelEl = document.getElementById('frame-caption-label');
  const frame = chunk.frames?.[idx];
  if (frame) {
    imgEl.src = `/api/media/${encodeURIComponent(frame)}`;
    imgEl.style.display = 'block';
  } else {
    imgEl.src = ''; imgEl.style.display = 'none';
  }
  const cap = chunk.captions?.[idx];
  labelEl.textContent = cap
    ? `Caption ${idx + 1}: [${cap.start ?? cap.start_time ?? '?'} → ${cap.end ?? cap.end_time ?? '?'}]`
    : '';
}

function renderCaptionEditor(captions) {
  const el = document.getElementById('caption-editor');
  el.innerHTML = '';
  captions.forEach((cap, i) => {
    const row = document.createElement('div');
    row.className = 'caption-item';
    row.innerHTML = `
      <input type="text" placeholder="start" value="${escHtml(String(cap.start ?? cap.start_time ?? ''))}"
             onchange="updateCaption(${i},'start',this.value)" />
      <input type="text" placeholder="end" value="${escHtml(String(cap.end ?? cap.end_time ?? ''))}"
             onchange="updateCaption(${i},'end',this.value)" />
      <textarea rows="2" onchange="updateCaption(${i},'caption',this.value)">${escHtml(cap.caption ?? '')}</textarea>
      <button class="caption-delete" onclick="deleteCaption(${i})">✕</button>
    `;
    el.appendChild(row);
    // Sync textarea
    const ta = row.querySelector('textarea');
    ta.value = cap.caption ?? '';
  });
}

function updateCaption(idx, field, val) {
  if (!gtData) return;
  const cap = gtData.chunks[gtChunkIdx].captions[idx];
  if (!cap) return;
  cap[field] = val;
  // Also keep start_time / end_time in sync for judge alignment
  if (field === 'start') cap.start_time = val;
  if (field === 'end') cap.end_time = val;
}

function deleteCaption(idx) {
  if (!gtData) return;
  gtData.chunks[gtChunkIdx].captions.splice(idx, 1);
  gtData.chunks[gtChunkIdx].frames?.splice(idx, 1);
  renderGTChunk();
}

function addCaption() {
  if (!gtData) return;
  gtData.chunks[gtChunkIdx].captions.push({ start: '', end: '', caption: '' });
  renderCaptionEditor(gtData.chunks[gtChunkIdx].captions);
}

function prevGTChunk() {
  if (!gtData || gtChunkIdx === 0) return;
  gtChunkIdx--; frameIdx = 0; renderGTChunk();
}
function nextGTChunk() {
  if (!gtData || gtChunkIdx >= gtData.chunks.length - 1) return;
  gtChunkIdx++; frameIdx = 0; renderGTChunk();
}
function prevFrame() {
  if (!gtData) return;
  const maxF = (gtData.chunks[gtChunkIdx].frames?.length ?? 1) - 1;
  frameIdx = Math.max(0, frameIdx - 1);
  renderFrame(gtData.chunks[gtChunkIdx], frameIdx);
}
function nextFrame() {
  if (!gtData) return;
  const maxF = (gtData.chunks[gtChunkIdx].frames?.length ?? 1) - 1;
  frameIdx = Math.min(maxF, frameIdx + 1);
  renderFrame(gtData.chunks[gtChunkIdx], frameIdx);
}

// Keyboard navigation for frames
document.addEventListener('keydown', e => {
  if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;
  if (e.key === 'ArrowLeft') prevFrame();
  if (e.key === 'ArrowRight') nextFrame();
});

async function saveGT() {
  if (!gtData) return;
  // Flatten all chunks
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
  const res = await fetch('/api/judge/run', {
    method: 'POST', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({ runs })
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
    `<p><strong>[${escHtml(String(c.start ?? c.start_time ?? ''))} → ${escHtml(String(c.end ?? c.end_time ?? ''))}]</strong> ${escHtml(c.caption ?? '')}</p>`
  ).join('');

  // Candidates
  const panel = document.getElementById('human-candidates-panel');
  panel.innerHTML = '';
  chunk.candidates.forEach(cand => {
    const card = document.createElement('div');
    card.className = 'candidate-card';
    const capText = cand.captions.map(c =>
      `<p>[${escHtml(String(c.start ?? c.start_time ?? ''))} → ${escHtml(String(c.end ?? c.end_time ?? ''))}] ${escHtml(c.caption ?? '')}</p>`
    ).join('');
    card.innerHTML = `
      <h4>Candidate ${cand.slot}</h4>
      <div class="caption-list" style="max-height:260px">${capText || '<em>No captions</em>'}</div>
      <div class="rank-input">
        <label>Rank (1=best):</label>
        <input type="number" min="1" max="4" id="rank-slot-${cand.slot}"
               data-slot="${cand.slot}" data-method="${cand.method_hidden}" />
      </div>
    `;
    panel.appendChild(card);
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

async function submitRanking() {
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
  const missing = slots.filter(s => s.rank === null);
  if (missing.length > 0) {
    document.getElementById('human-status').textContent = '⚠️ Please enter a rank for every candidate.';
    return;
  }
  const res = await fetch('/api/human/rank', {
    method: 'POST', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({ chunk_idx: humanChunkIdx, slots })
  });
  const data = await res.json();
  document.getElementById('human-status').textContent = `✓ Saved chunk ${humanChunkIdx + 1} (total: ${data.total})`;
  appendLog(`[human] Chunk ${humanChunkIdx + 1} ranked`);
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

  // Table
  const tbody = document.getElementById('results-tbody');
  tbody.innerHTML = '';
  methods.forEach(m => {
    const jd = judgeData[m] || {};
    const mean = jd.mean != null ? jd.mean.toFixed(2) : '–';
    const se = jd.bootstrap_se != null ? jd.bootstrap_se.toFixed(2) : '–';
    const sz = sizes[m] != null ? `${sizes[m]} MB` : '–';
    const tr = document.createElement('tr');
    tr.innerHTML = `<td>${escHtml(m)}</td><td>${mean} ± ${se}</td><td>${sz}</td>`;
    tbody.appendChild(tr);
  });
  document.getElementById('results-table-container').classList.remove('hidden');

  // Correlation table
  if (data.human_eval) {
    const he = data.human_eval;
    const cTbody = document.getElementById('corr-tbody');
    cTbody.innerHTML = `
      <tr><td>Pearson r</td><td>${he.pearson_r?.toFixed(3) ?? '–'} (p=${he.pearson_p?.toFixed(3) ?? '–'})</td></tr>
      <tr><td>Spearman ρ</td><td>${he.spearman_rho?.toFixed(3) ?? '–'} (p=${he.spearman_p?.toFixed(3) ?? '–'})</td></tr>
    `;
    document.getElementById('correlation-container').classList.remove('hidden');
  }

  // LaTeX
  buildLatex(methods, sizes, judgeData, data.judge?.num_gt_chunks);
}

function buildLatex(methods, sizes, judgeData, numGT) {
  const methodLabels = {
    naive: 'naive',
    split: '+ split',
    split_compress: '+ split + compress',
    split_compress_io: '+ split + compress + IO',
  };
  let rows = methods.map(m => {
    const jd = judgeData[m] || {};
    const mean = jd.mean != null ? jd.mean.toFixed(2) : '?';
    const se = jd.bootstrap_se != null ? jd.bootstrap_se.toFixed(2) : '?';
    const sz = sizes[m] != null ? sizes[m].toFixed(0) : '?';
    const label = methodLabels[m] || m;
    const hrule = m === 'naive' ? '\\midrule\n' : '';
    return `${hrule}        ${label} & $${mean} \\pm ${se}$ & \\SI{${sz}}{\\mega\\byte} \\\\`;
  }).join('\n');

  const nStr = numGT != null ? `$n={${numGT} \\cdot 8}$` : '$n={?}$';
  const latex = `\\begin{table}[]
    \\centering
    \\begin{tabular}{lrr}
        \\toprule
        Method & Judge Score ($\\uparrow$) & Size ($\\downarrow$) \\\\
        \\midrule
${rows}
        \\bottomrule
    \\end{tabular}
    \\caption{\\textbf{NAPsack reduces the amount of data we have to save for effective captioning by 70\\% without compromising quality}. When comparing to ${nStr} ground truth trajectories, our LLM-as-a-judge scores (in $[0,1]$; $\\pm$ bootstrap standard error) show that accuracy increases most when splitting the data into smaller chunks for the VLM to label. Event-driven compression (where frames are saved only when a user interacts with their computer) yields the best data efficiency.}
    \\label{tab:pack-eval}
\\end{table}`;

  document.getElementById('latex-output').textContent = latex;
  document.getElementById('latex-container').classList.remove('hidden');
}

function copyLatex() {
  const text = document.getElementById('latex-output').textContent;
  navigator.clipboard.writeText(text).then(() => appendLog('[latex] Copied to clipboard'));
}

// ── Utilities ──────────────────────────────────────────────────────────────
function escHtml(s) {
  return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}

// Initial load
pollStatus();
