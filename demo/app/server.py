"""
FastAPI server for the NAPsack demo dashboard.
…
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any, AsyncIterator, Dict, List, Optional

# Ensure repo root (containing demo/) is on sys.path so that
# `from demo.app.state import …` works regardless of how uvicorn was started.
_ROOT = Path(__file__).parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from sse_starlette.sse import EventSourceResponse

from demo.app.state import DemoState, METHODS
from demo.ffmpeg_recorder import FFmpegRecorder

# ---------------------------------------------------------------------------
# Env / startup
# ---------------------------------------------------------------------------

load_dotenv()

REPO_ROOT = Path(__file__).parent.parent.parent
DEFAULT_SESSION_DIR = str(REPO_ROOT / "demo-session")
STATIC_DIR = Path(__file__).parent / "static"
STATIC_DIR.mkdir(exist_ok=True)

app = FastAPI(title="NAPsack Demo Dashboard")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# Global mutable state
_state: DemoState = DemoState()
_ffmpeg_recorder: Optional[FFmpegRecorder] = None
_pack_proc: Optional[asyncio.subprocess.Process] = None

# SSE event queue
_log_queue: asyncio.Queue = asyncio.Queue()


def _log(msg: str):
    """Push a message to the SSE log queue (thread-safe)."""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            asyncio.run_coroutine_threadsafe(_log_queue.put(msg), loop)
        else:
            _log_queue.put_nowait(msg)
    except Exception:
        pass
    print(msg, flush=True)


# ---------------------------------------------------------------------------
# Root + static
# ---------------------------------------------------------------------------

@app.get("/")
async def serve_index():
    index = STATIC_DIR / "index.html"
    if index.exists():
        return FileResponse(str(index))
    return JSONResponse({"error": "index.html not found"}, status_code=404)


@app.get("/api/media/{path:path}")
async def serve_media(path: str):
    """Serve arbitrary files from the filesystem (frames, videos)."""
    p = Path(path)
    if not p.exists():
        raise HTTPException(status_code=404, detail=f"File not found: {path}")
    return FileResponse(str(p))


# ---------------------------------------------------------------------------
# SSE
# ---------------------------------------------------------------------------

@app.get("/events")
async def sse_events(request: Request):
    async def generator():
        yield {"event": "connected", "data": "stream open"}
        while True:
            if await request.is_disconnected():
                break
            try:
                msg = await asyncio.wait_for(_log_queue.get(), timeout=1.0)
                yield {"event": "log", "data": msg}
            except asyncio.TimeoutError:
                yield {"event": "ping", "data": ""}
    return EventSourceResponse(generator())


# ---------------------------------------------------------------------------
# Session
# ---------------------------------------------------------------------------

@app.get("/api/status")
async def get_status():
    gemini_ok = bool(os.environ.get("GEMINI_API_KEY", ""))
    d = _state.to_json()
    d["gemini_key_ok"] = gemini_ok
    d["default_session_dir"] = DEFAULT_SESSION_DIR
    return JSONResponse(d)


@app.post("/api/session")
async def set_session(body: dict):
    global _state
    session_dir = body.get("session_dir", "") or DEFAULT_SESSION_DIR
    p = Path(session_dir)
    p.mkdir(parents=True, exist_ok=True)
    _state = DemoState.load(str(p))
    _state.session_dir = str(p)
    _state.save()
    _log(f"[session] Loaded session: {p}")
    return JSONResponse(_state.to_json())


# ---------------------------------------------------------------------------
# Step 1: Recording
# ---------------------------------------------------------------------------

@app.post("/api/record/start")
async def start_recording(body: dict):
    global _state, _ffmpeg_recorder, _pack_proc

    if _state.session_dir is None:
        raise HTTPException(status_code=400, detail="Set session_dir first")
    if _state.recording.running:
        return JSONResponse({"status": "already_running"})

    session_dir = Path(_state.session_dir)
    pack_session = session_dir / "pack_session"
    ffmpeg_dir = session_dir / "ffmpeg"
    pack_session.mkdir(parents=True, exist_ok=True)
    ffmpeg_dir.mkdir(parents=True, exist_ok=True)

    max_res = body.get("max_res", [1920, 1080])

    # Start FFmpeg recorder
    _log("[record] Starting FFmpeg recorder…")
    _ffmpeg_recorder = FFmpegRecorder(ffmpeg_dir)
    _ffmpeg_recorder.start()

    # Start pack recorder as subprocess
    _log("[record] Starting pack recorder…")
    cmd = [
        sys.executable, "-m", "record",
        "--max-res", str(max_res[0]), str(max_res[1]),
        "--session-dir", str(pack_session),
    ]
    _pack_proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        cwd=str(REPO_ROOT),
    )

    # Stream pack stdout to SSE in background
    async def _stream_pack():
        assert _pack_proc and _pack_proc.stdout
        async for line in _pack_proc.stdout:
            _log(f"[pack] {line.decode(errors='replace').rstrip()}")
    asyncio.create_task(_stream_pack())

    _state.recording.running = True
    _state.recording.pack_session_dir = str(pack_session)
    _state.recording.ffmpeg_dir = str(ffmpeg_dir)
    _state.recording.start_time = time.time()
    _state.save()

    _log("[record] Both recorders started ✓")
    return JSONResponse({"status": "started", "pack_session": str(pack_session), "ffmpeg_dir": str(ffmpeg_dir)})


@app.post("/api/record/stop")
async def stop_recording():
    global _state, _ffmpeg_recorder, _pack_proc

    if not _state.recording.running:
        return JSONResponse({"status": "not_running"})

    _log("[record] Stopping recorders…")

    # Stop FFmpeg
    if _ffmpeg_recorder:
        meta = _ffmpeg_recorder.stop()
        _log(f"[record] FFmpeg stopped. {len(meta)} monitor(s) saved.")
        _ffmpeg_recorder = None

    # Stop pack (send SIGINT)
    if _pack_proc and _pack_proc.returncode is None:
        try:
            _pack_proc.terminate()
            await asyncio.wait_for(_pack_proc.wait(), timeout=30)
        except asyncio.TimeoutError:
            _pack_proc.kill()
        _pack_proc = None

    _state.recording.running = False
    _state.recording.end_time = time.time()
    _state.save()

    _log("[record] Both recorders stopped ✓")
    return JSONResponse({"status": "stopped"})


# ---------------------------------------------------------------------------
# Step 2: Processing
# ---------------------------------------------------------------------------

@app.post("/api/process/{method}")
async def process_method(method: str):
    if method not in METHODS:
        raise HTTPException(status_code=400, detail=f"Unknown method: {method}. Valid: {METHODS}")
    if _state.session_dir is None:
        raise HTTPException(status_code=400, detail="Set session_dir first")
    if _state.processing.status.get(method) == "running":
        return JSONResponse({"status": "already_running"})

    # Dependency: split_compress_io requires split_compress to be done
    if method == "split_compress_io":
        sc_status = _state.processing.status.get("split_compress")
        if sc_status != "done":
            raise HTTPException(
                status_code=400,
                detail="split_compress must be completed before running split_compress_io",
            )

    session_dir = Path(_state.session_dir)

    async def _run():
        from demo.app.processing import run_method
        try:
            async for line in run_method(method, _state, session_dir):
                _log(line)
        except Exception as e:
            _log(f"[{method}] FAILED: {e}")

    asyncio.create_task(_run())
    return JSONResponse({"status": "started", "method": method})


# ---------------------------------------------------------------------------
# Step 3: GT annotation
# ---------------------------------------------------------------------------

@app.get("/api/gt")
async def get_gt_data():
    """
    Return the split_compress_io captions in 8-caption chunks,
    along with per-caption frame paths for the annotation UI.
    """
    if _state.session_dir is None:
        raise HTTPException(status_code=400, detail="Set session_dir first")

    session_dir = Path(_state.session_dir)

    # Find fused captions
    fused_dir = session_dir / "split_compress_io" / "fused"
    captions: List[Dict] = []
    for chunk_dir in sorted(fused_dir.glob("chunk_??*")):
        cap_file = chunk_dir / "captions.jsonl"
        if cap_file.exists():
            with open(cap_file) as f:
                for line in f:
                    if line.strip():
                        captions.append(json.loads(line))

    # Find pack screenshots dir for frame serving
    pack_dir = Path(_state.recording.pack_session_dir or "")
    screenshots_dir = pack_dir / "screenshots"

    # Build 8-caption chunks
    chunk_size = 8
    chunks = []
    for i in range(0, len(captions), chunk_size):
        chunk_caps = captions[i:i + chunk_size]
        # Attach representative frame paths
        frames = []
        for cap in chunk_caps:
            frame = _find_nearest_frame(screenshots_dir, cap)
            frames.append(str(frame) if frame else None)
        chunks.append({"captions": chunk_caps, "frames": frames})

    # Also load existing GT if already saved
    gt_path = session_dir / "gt" / "gt_captions.jsonl"
    existing_gt: List[Dict] = []
    if gt_path.exists():
        with open(gt_path) as f:
            for line in f:
                if line.strip():
                    existing_gt.append(json.loads(line))

    # Build GT video clips: one per 8-caption chunk
    gt_videos = _get_gt_videos(session_dir, captions, chunk_size)

    return JSONResponse({
        "chunks": chunks,
        "existing_gt": existing_gt,
        "gt_videos": gt_videos,
        "num_chunks": len(chunks),
    })


def _find_nearest_frame(screenshots_dir: Path, cap: Dict) -> Optional[Path]:
    """Find the screenshot closest to the caption's start_time."""
    if not screenshots_dir.exists():
        return None
    start = cap.get("start_time") or cap.get("start_seconds") or 0
    try:
        start = float(start)
    except (TypeError, ValueError):
        # Try parsing MM:SS
        try:
            parts = str(start).split(":")
            start = int(parts[0]) * 60 + int(parts[1])
        except Exception:
            start = 0.0

    best: Optional[Path] = None
    best_diff = float("inf")
    for img in screenshots_dir.glob("*.[jp][pn]g"):
        try:
            ts = float(img.stem.split("_")[0]) if "_" in img.stem else float(img.stem)
            diff = abs(ts - start)
            if diff < best_diff:
                best_diff = diff
                best = img
        except ValueError:
            pass
    return best


def _get_gt_videos(session_dir: Path, captions: List[Dict], chunk_size: int) -> List[Optional[str]]:
    """
    Return per-chunk video clip paths (or None).
    For each 8-caption chunk we look for a pre-existing ffmpeg clip;
    if not found we return None (the UI will hide the video player).
    """
    videos = []
    for i in range(0, len(captions), chunk_size):
        chunk_caps = captions[i:i + chunk_size]
        # Check if a clip already exists for this chunk
        clip_dir = session_dir / "gt_clips"
        clip_idx = i // chunk_size
        clip_path = clip_dir / f"gt_chunk_{clip_idx:03d}.mp4"
        if clip_path.exists():
            videos.append(str(clip_path))
        else:
            # Try to find the ffmpeg monitor recording and slice it
            clip = _extract_gt_clip(session_dir, chunk_caps, clip_path)
            videos.append(str(clip) if clip else None)
    return videos


def _extract_gt_clip(
    session_dir: Path,
    chunk_caps: List[Dict],
    clip_path: Path,
) -> Optional[Path]:
    """Extract a video clip from the ffmpeg recording for a GT chunk."""
    import subprocess
    ffmpeg_dir = Path(_state.recording.ffmpeg_dir or "")
    if not ffmpeg_dir.exists():
        return None

    mp4s = sorted(ffmpeg_dir.glob("screen_*.mp4"))
    if not mp4s:
        return None
    src = mp4s[0]  # use first monitor

    # Get time range from captions
    def _to_sec(v):
        if isinstance(v, str) and ":" in v:
            p = v.split(":")
            return int(p[0]) * 60 + int(p[1])
        try:
            return float(v)
        except (TypeError, ValueError):
            return 0.0

    starts = [_to_sec(c.get("start_time") or c.get("start") or 0) for c in chunk_caps]
    ends = [_to_sec(c.get("end_time") or c.get("end") or 0) for c in chunk_caps]
    t_start = max(0, min(starts) - 1)
    t_dur = max(ends) - t_start + 1

    clip_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.check_call([
            "ffmpeg", "-y",
            "-ss", str(t_start),
            "-i", str(src),
            "-t", str(t_dur),
            "-c", "copy",
            str(clip_path),
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return clip_path
    except Exception:
        return None


@app.post("/api/gt/save")
async def save_gt(body: dict):
    """Save annotated GT captions."""
    if _state.session_dir is None:
        raise HTTPException(status_code=400, detail="Set session_dir first")

    session_dir = Path(_state.session_dir)
    gt_dir = session_dir / "gt"
    gt_dir.mkdir(exist_ok=True)
    gt_path = gt_dir / "gt_captions.jsonl"

    captions = body.get("captions", [])
    with open(gt_path, "w") as f:
        for cap in captions:
            f.write(json.dumps(cap) + "\n")

    _state.gt.done = True
    _state.gt.gt_captions_path = str(gt_path)
    _state.gt.num_chunks = (len(captions) + 7) // 8
    _state.save()

    _log(f"[gt] Saved {len(captions)} GT captions to {gt_path}")
    return JSONResponse({"status": "saved", "num_captions": len(captions), "path": str(gt_path)})


# ---------------------------------------------------------------------------
# Step 4: LLM Judge
# ---------------------------------------------------------------------------

@app.post("/api/judge/run")
async def run_judge(body: dict):
    if _state.session_dir is None:
        raise HTTPException(status_code=400, detail="Set session_dir first")
    if not _state.gt.done:
        raise HTTPException(status_code=400, detail="GT captions not yet annotated (step 3)")
    if _state.judge.status == "running":
        return JSONResponse({"status": "already_running"})

    session_dir = Path(_state.session_dir)
    gt_path = Path(_state.gt.gt_captions_path)
    output_dir = session_dir / "judge"
    num_runs = body.get("runs", 3)

    _state.judge.status = "running"
    _state.save()

    async def _run():
        try:
            sys_path_entry = str(Path(__file__).parent.parent.parent / "demo")
            cmd = [
                sys.executable, "-m", "demo.single_judge",
                "--session-dir", str(session_dir),
                "--output-dir", str(output_dir),
                "--runs", str(num_runs),
            ]
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                cwd=str(REPO_ROOT),
            )
            assert proc.stdout
            async for line in proc.stdout:
                _log(f"[judge] {line.decode(errors='replace').rstrip()}")
            await proc.wait()

            summary_path = output_dir / "summary.json"
            if summary_path.exists():
                with open(summary_path) as f:
                    summary = json.load(f)
                _state.judge.results_path = str(summary_path)
                _state.judge.summary = summary
                _state.judge.status = "done"
            else:
                _state.judge.status = "error"
            _state.save()
            _log("[judge] ✓ done")
        except Exception as e:
            _state.judge.status = "error"
            _state.save()
            _log(f"[judge] ERROR: {e}")

    asyncio.create_task(_run())
    return JSONResponse({"status": "started"})


# ---------------------------------------------------------------------------
# Step 5: Human evaluation
# ---------------------------------------------------------------------------

@app.get("/api/human/chunks")
async def get_human_chunks():
    """
    Return data for each GT chunk: GT captions, all 4 method captions
    (shuffled / blinded), and the GT video clip path.
    """
    if _state.session_dir is None:
        raise HTTPException(status_code=400, detail="Set session_dir first")
    if not _state.gt.done:
        raise HTTPException(status_code=400, detail="GT not done")

    import random as _random

    session_dir = Path(_state.session_dir)
    gt_path = Path(_state.gt.gt_captions_path)

    # Load GT chunks (groups of 8)
    gt_entries: List[Dict] = []
    with open(gt_path) as f:
        for line in f:
            if line.strip():
                gt_entries.append(json.loads(line))
    gt_chunks = [gt_entries[i:i+8] for i in range(0, len(gt_entries), 8)]

    # Build aligned method chunks
    from demo.single_judge import _load_method_chunks, _chunk_captions_by_gt
    result_chunks = []
    for ci, gt_chunk in enumerate(gt_chunks):
        methods_data: Dict[str, List[Dict]] = {}
        for m in METHODS:
            raw = _load_method_chunks(session_dir, m)
            flat = [c for chunk in raw for c in chunk]
            aligned = _chunk_captions_by_gt(flat, [gt_chunk])
            methods_data[m] = aligned[0] if aligned else []

        # Shuffle method order for blind presentation
        shuffled_methods = METHODS[:]
        _random.shuffle(shuffled_methods)
        candidates = [
            {"slot": i + 1, "method_hidden": m, "captions": methods_data[m]}
            for i, m in enumerate(shuffled_methods)
        ]

        # GT video
        gt_videos = _get_gt_videos(session_dir, gt_entries, 8)
        gt_video = gt_videos[ci] if ci < len(gt_videos) else None

        result_chunks.append({
            "chunk_idx": ci,
            "gt_captions": gt_chunk,
            "candidates": candidates,
            "gt_video": gt_video,
        })

    return JSONResponse({"chunks": result_chunks})


@app.post("/api/human/rank")
async def save_human_rank(body: dict):
    """Save one chunk's rankings from the human evaluator."""
    if _state.session_dir is None:
        raise HTTPException(status_code=400, detail="Set session_dir first")

    session_dir = Path(_state.session_dir)
    hr_dir = session_dir / "human_eval"
    hr_dir.mkdir(exist_ok=True)
    rankings_path = hr_dir / "rankings.json"

    rankings: List[Dict] = []
    if rankings_path.exists():
        with open(rankings_path) as f:
            rankings = json.load(f)

    rankings.append(body)
    with open(rankings_path, "w") as f:
        json.dump(rankings, f, indent=2)

    _state.human_eval.rankings_path = str(rankings_path)
    _state.save()
    return JSONResponse({"status": "saved", "total": len(rankings)})


@app.post("/api/human/finalize")
async def finalize_human_eval():
    """Compute Pearson r and Spearman ρ between human ranks and LLM scores."""
    if _state.session_dir is None:
        raise HTTPException(status_code=400, detail="Set session_dir first")
    if _state.judge.status != "done":
        raise HTTPException(status_code=400, detail="LLM judge not done yet (step 4)")

    session_dir = Path(_state.session_dir)
    hr_dir = session_dir / "human_eval"
    rankings_path = hr_dir / "rankings.json"
    if not rankings_path.exists():
        raise HTTPException(status_code=400, detail="No human rankings found")

    with open(rankings_path) as f:
        rankings: List[Dict] = json.load(f)

    # Build per-method human rank and LLM score vectors
    judge_summary = _state.judge.summary or {}
    judge_methods = judge_summary.get("methods", {})

    # Aggregate mean human rank per method across all chunks
    human_ranks: Dict[str, List[float]] = {m: [] for m in METHODS}
    for entry in rankings:
        for slot in entry.get("slots", []):
            m = slot.get("method")
            r = slot.get("rank")
            if m and r is not None:
                human_ranks[m].append(float(r))

    mean_human: Dict[str, float] = {
        m: float(sum(v) / len(v)) if v else 0.0
        for m, v in human_ranks.items()
    }
    mean_llm: Dict[str, float] = {
        m: judge_methods.get(m, {}).get("mean", 0.0) for m in METHODS
    }

    # Vectors for correlation (same order)
    h_vec = [mean_human[m] for m in METHODS]
    l_vec = [mean_llm[m] for m in METHODS]

    from scipy.stats import pearsonr, spearmanr  # type: ignore
    pear_r, pear_p = pearsonr(h_vec, l_vec)
    spear_r, spear_p = spearmanr(h_vec, l_vec)

    correlations = {
        "pearson_r": float(pear_r),
        "pearson_p": float(pear_p),
        "spearman_rho": float(spear_r),
        "spearman_p": float(spear_p),
        "mean_human_rank": mean_human,
        "mean_llm_score": mean_llm,
    }

    corr_path = hr_dir / "correlations.json"
    with open(corr_path, "w") as f:
        json.dump(correlations, f, indent=2)

    _state.human_eval.correlations_path = str(corr_path)
    _state.human_eval.correlations = correlations
    _state.human_eval.status = "done"
    _state.save()

    _log(f"[human_eval] Pearson r={pear_r:.3f}  Spearman ρ={spear_r:.3f}")
    return JSONResponse(correlations)


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------

@app.get("/api/results")
async def get_results():
    """Return full results for the dashboard table."""
    d: Dict[str, Any] = {
        "methods": METHODS,
        "mp4_sizes_mb": _state.processing.mp4_sizes_mb,
        "judge": _state.judge.summary,
        "human_eval": _state.human_eval.correlations,
        "gemini_key_ok": bool(os.environ.get("GEMINI_API_KEY", "")),
    }
    return JSONResponse(d)
