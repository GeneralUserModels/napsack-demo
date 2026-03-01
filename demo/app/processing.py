"""
Processing pipeline – four captioning modes run as subprocesses.

Each mode writes its output to a sub-directory of the demo session dir
and updates DemoState with status, output paths and MP4 sizes.
"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import AsyncIterator, List, Optional

from demo.app.state import DemoState, METHODS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _uv_run(*args: str) -> List[str]:
    """Build a `uv run` command list."""
    return ["uv", "run", *args]


def _total_mp4_size_mb(directory: Path, exclude_master: bool = True) -> float:
    """Sum up all .mp4 files (recursive) in a directory, in decimal megabytes.
    Optionally excludes master.mp4 to avoid double-counting with chunks."""
    total = sum(
        p.stat().st_size
        for p in directory.rglob("*.mp4")
        if p.is_file() and not (exclude_master and p.name == "master.mp4")
    )
    return round(total / 1_000_000, 2)


async def _stream_subprocess(cmd: List[str], cwd: Optional[str] = None) -> AsyncIterator[str]:
    """Run a subprocess and yield stdout/stderr lines as they arrive."""
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        cwd=cwd,
        env=env,
    )
    assert proc.stdout is not None
    async for line in proc.stdout:
        yield line.decode(errors="replace").rstrip("\n")
    await proc.wait()
    if proc.returncode != 0:
        raise subprocess.CalledProcessError(proc.returncode, cmd)


# ---------------------------------------------------------------------------
# Ffmpeg video → screenshots extraction (for naive / split modes)
# ---------------------------------------------------------------------------

def extract_frames_from_video(video_path: Path, out_dir: Path, fps: int = 1) -> Path:
    """
    Extract frames from an MP4 at `fps` frames/second into out_dir.
    Files are named frame_000001.png, frame_000002.png, etc.
    Default 1fps matches Gemini's native sampling rate.
    Returns out_dir.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y",
        "-i", str(video_path),
        "-vf", f"fps={fps}",
        str(out_dir / "frame_%06d.png"),
    ]
    subprocess.check_call(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    return out_dir


# ---------------------------------------------------------------------------
# Individual mode runners
# ---------------------------------------------------------------------------

async def run_naive(
    state: DemoState,
    ffmpeg_dir: Path,
    session_dir: Path,
    label_workers: int = 4,
) -> AsyncIterator[str]:
    """
    Naive: whole-video Gemini captioning (no splitting, no compression).
    Frames are extracted at 1fps from the ffmpeg screen recording, then
    labelled with 10-min chunk duration (max 12 min tolerance).
    Gemini samples at 1fps natively, so all extracted frames are processed.
    """
    out_dir = session_dir / "naive"
    out_dir.mkdir(parents=True, exist_ok=True)
    screenshots_dir = out_dir / "screenshots"
    screenshots_dir.mkdir(exist_ok=True)

    # Extract frames from the single focus-following recording
    output_mp4 = ffmpeg_dir / "output.mp4"
    if not output_mp4.exists():
        yield "[naive] ERROR: output.mp4 not found in ffmpeg dir"
        state.processing.status["naive"] = "error"
        state.processing.errors["naive"] = "output.mp4 missing"
        state.save()
        return
    yield f"[naive] Extracting frames from {output_mp4.name}…"
    extract_frames_from_video(output_mp4, screenshots_dir)

    # Determine chunk duration: aim for 10-min chunks, tolerate up to 12 min.
    # If total video ≤ 12 min, use a single chunk (no splitting needed).
    num_frames = len(list(screenshots_dir.glob("frame_*.png")))
    total_seconds = num_frames  # 1 fps → 1 frame per second
    if total_seconds <= 720:  # ≤ 12 minutes → single chunk
        chunk_dur = str(total_seconds + 1)
        yield f"[naive] Video is {total_seconds}s (≤12 min) → single chunk"
    else:
        chunk_dur = "600"  # 10-minute chunks
        n_chunks = -(-total_seconds // 600)  # ceil division
        yield f"[naive] Video is {total_seconds}s → splitting into ~{n_chunks} chunks of ≤10 min"

    yield "[naive] Running label pipeline…"
    cmd = _uv_run(
        "-m", "label",
        "--session", str(out_dir),
        "--screenshots-only",
        "--chunk-duration", chunk_dur,
        "--client", "gemini",
        "--model", "gemini-3-flash-preview",
        "--label-workers", str(label_workers),
    )
    async for line in _stream_subprocess(cmd):
        yield f"[naive] {line}"

    state.processing.output_dirs["naive"] = str(out_dir)
    state.processing.mp4_sizes_mb["naive"] = _total_mp4_size_mb(ffmpeg_dir)
    state.processing.status["naive"] = "done"
    state.save()
    yield "[naive] ✓ done"


async def run_split(
    state: DemoState,
    ffmpeg_dir: Path,
    session_dir: Path,
    label_workers: int = 4,
) -> AsyncIterator[str]:
    """
    Split: same as naive but split into 1-min chunks before labelling.
    """
    out_dir = session_dir / "split"
    out_dir.mkdir(parents=True, exist_ok=True)
    screenshots_dir = out_dir / "screenshots"
    screenshots_dir.mkdir(exist_ok=True)

    output_mp4 = ffmpeg_dir / "output.mp4"
    if not output_mp4.exists():
        yield "[split] ERROR: output.mp4 not found in ffmpeg dir"
        state.processing.status["split"] = "error"
        state.processing.errors["split"] = "output.mp4 missing"
        state.save()
        return
    yield f"[split] Extracting frames from {output_mp4.name}…"
    extract_frames_from_video(output_mp4, screenshots_dir)

    yield "[split] Running label pipeline (60 s chunks)…"
    cmd = _uv_run(
        "-m", "label",
        "--session", str(out_dir),
        "--screenshots-only",
        "--chunk-duration", "60",
        "--client", "gemini",
        "--model", "gemini-3-flash-preview",
        "--label-workers", str(label_workers),
    )
    async for line in _stream_subprocess(cmd):
        yield f"[split] {line}"

    state.processing.output_dirs["split"] = str(out_dir)
    state.processing.mp4_sizes_mb["split"] = _total_mp4_size_mb(ffmpeg_dir)
    state.processing.status["split"] = "done"
    state.save()
    yield "[split] ✓ done"


async def run_split_compress(
    state: DemoState,
    napsack_session_dir: Path,
    session_dir: Path,
    label_workers: int = 4,
) -> AsyncIterator[str]:
    """
    Split + compress: use napsack screenshots (event-driven, already compressed)
    and build 1-min chunk videos, then label with Gemini.
    Data files are copied into the method's own directory so label outputs
    stay isolated from the original napsack_session.
    """
    import shutil

    out_dir = session_dir / "split_compress"
    out_dir.mkdir(parents=True, exist_ok=True)

    napsack_screenshots = napsack_session_dir / "screenshots"
    if not napsack_screenshots.exists():
        yield "[split_compress] ERROR: napsack screenshots dir not found"
        state.processing.status["split_compress"] = "error"
        state.processing.errors["split_compress"] = "screenshots dir missing"
        state.save()
        return

    # Copy napsack screenshots into our own directory
    local_screenshots = out_dir / "screenshots"
    if local_screenshots.exists():
        shutil.rmtree(local_screenshots)
    yield "[split_compress] Copying napsack screenshots → split_compress/screenshots …"
    shutil.copytree(str(napsack_screenshots), str(local_screenshots))

    # Copy aggregations.jsonl if present
    agg_src = napsack_session_dir / "aggregations.jsonl"
    if agg_src.exists():
        shutil.copy2(str(agg_src), str(out_dir / "aggregations.jsonl"))

    # Point label at our isolated directory
    yield "[split_compress] Running label pipeline on napsack screenshots (60 s chunks)…"
    cmd = _uv_run(
        "-m", "label",
        "--session", str(out_dir),
        "--screenshots-only",
        "--chunk-duration", "60",
        "--client", "gemini",
        "--model", "gemini-3-flash-preview",
        "--label-workers", str(label_workers),
    )
    async for line in _stream_subprocess(cmd):
        yield f"[split_compress] {line}"

    # Clean up redundant master.mp4 if present
    master_mp4 = out_dir / "chunks" / "master.mp4"
    if master_mp4.exists():
        master_mp4.unlink()
        yield "[split_compress] Removed redundant master.mp4"

    state.processing.output_dirs["split_compress"] = str(out_dir)
    # MP4 size = chunk videos built from napsack screenshots
    state.processing.mp4_sizes_mb["split_compress"] = _total_mp4_size_mb(out_dir)
    state.processing.status["split_compress"] = "done"
    state.save()
    yield "[split_compress] ✓ done"


async def run_split_compress_io(
    state: DemoState,
    napsack_session_dir: Path,
    session_dir: Path,
    label_workers: int = 4,
) -> AsyncIterator[str]:
    """
    Split + compress + IO: full napsack labelling (screenshots + events), then
    fuse with split_compress video-only captions via napsack_fuse.py.
    Data files are copied into the method's own directory so label outputs
    stay isolated from the original napsack_session.
    """
    import shutil

    out_dir = session_dir / "split_compress_io"
    out_dir.mkdir(parents=True, exist_ok=True)
    fused_dir = out_dir / "fused"
    fused_dir.mkdir(parents=True, exist_ok=True)

    napsack_screenshots = napsack_session_dir / "screenshots"
    if not napsack_screenshots.exists():
        yield "[split_compress_io] ERROR: napsack screenshots dir not found"
        state.processing.status["split_compress_io"] = "error"
        state.processing.errors["split_compress_io"] = "screenshots dir missing"
        state.save()
        return

    # Copy napsack data into our own directory
    local_screenshots = out_dir / "screenshots"
    if local_screenshots.exists():
        shutil.rmtree(local_screenshots)
    yield "[split_compress_io] Copying napsack screenshots → split_compress_io/screenshots …"
    shutil.copytree(str(napsack_screenshots), str(local_screenshots))

    # Copy aggregations.jsonl if present
    agg_src = napsack_session_dir / "aggregations.jsonl"
    if agg_src.exists():
        shutil.copy2(str(agg_src), str(out_dir / "aggregations.jsonl"))

    # Step 1: label our isolated copy with full IO context
    yield "[split_compress_io] Running full napsack label (screenshots + events)…"
    cmd = _uv_run(
        "-m", "label",
        "--session", str(out_dir),
        "--chunk-duration", "60",
        "--client", "gemini",
        "--model", "gemini-3-flash-preview",
        "--label-workers", str(label_workers),
    )
    async for line in _stream_subprocess(cmd):
        yield f"[split_compress_io] {line}"

    # Clean up redundant master.mp4 (chunks already split from it)
    master_mp4 = out_dir / "chunks" / "master.mp4"
    if master_mp4.exists():
        master_mp4.unlink()
        yield "[split_compress_io] Removed redundant master.mp4"

    # split_compress output dir (video-only captions already produced)
    sc_dir = state.processing.output_dirs.get("split_compress") or str(session_dir / "split_compress")

    # Step 2: fuse video-only + napsack captions
    yield "[split_compress_io] Fusing captions with napsack_fuse.py…"
    demo_dir = Path(__file__).parent.parent  # demo/
    fuse_script = demo_dir / "napsack_fuse.py"
    # video chunks were built by the label pipeline inside out_dir (isolated)
    cmd2 = [
        sys.executable, str(fuse_script),
        "--video-only-name", sc_dir,
        "--napsack-name", str(out_dir),
        "--video-dir", str(out_dir),
        "--output-dir", str(fused_dir),
    ]
    async for line in _stream_subprocess(cmd2):
        yield f"[split_compress_io] {line}"

    state.processing.output_dirs["split_compress_io"] = str(fused_dir)
    state.processing.mp4_sizes_mb["split_compress_io"] = _total_mp4_size_mb(out_dir)
    state.processing.status["split_compress_io"] = "done"
    state.save()
    yield "[split_compress_io] ✓ done"


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

async def run_method(
    method: str,
    state: DemoState,
    session_dir: Path,
    label_workers: int = 4,
) -> AsyncIterator[str]:
    napsack_dir = Path(state.recording.napsack_session_dir)
    ffmpeg_dir = Path(state.recording.ffmpeg_dir)

    state.processing.status[method] = "running"
    state.processing.errors[method] = None
    state.save()
    try:
        if method == "naive":
            async for line in run_naive(state, ffmpeg_dir, session_dir, label_workers):
                yield line
        elif method == "split":
            async for line in run_split(state, ffmpeg_dir, session_dir, label_workers):
                yield line
        elif method == "split_compress":
            async for line in run_split_compress(state, napsack_dir, session_dir, label_workers):
                yield line
        elif method == "split_compress_io":
            async for line in run_split_compress_io(state, napsack_dir, session_dir, label_workers):
                yield line
        else:
            raise ValueError(f"Unknown method: {method}")
    except Exception as exc:
        state.processing.status[method] = "error"
        state.processing.errors[method] = str(exc)
        state.save()
        yield f"[{method}] ERROR: {exc}"
        raise
