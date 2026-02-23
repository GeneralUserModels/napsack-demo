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


def _total_mp4_size_mb(directory: Path) -> float:
    """Sum up all .mp4 files (recursive) in a directory, in megabytes."""
    total = sum(p.stat().st_size for p in directory.rglob("*.mp4") if p.is_file())
    return round(total / (1024 ** 2), 2)


async def _stream_subprocess(cmd: List[str], cwd: Optional[str] = None) -> AsyncIterator[str]:
    """Run a subprocess and yield stdout/stderr lines as they arrive."""
    env = {**os.environ}
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
    Returns out_dir.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y",
        "-i", str(video_path),
        "-vf", f"fps={fps}",
        "-q:v", "5",  # ~JPEG quality 70
        str(out_dir / "%017.3f.jpg"),  # 17-char float timestamp prefix (seconds)
    ]
    subprocess.check_call(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return out_dir


# ---------------------------------------------------------------------------
# Individual mode runners
# ---------------------------------------------------------------------------

async def run_naive(
    state: DemoState,
    ffmpeg_dir: Path,
    session_dir: Path,
) -> AsyncIterator[str]:
    """
    Naive: whole-video Gemini captioning (no splitting, no compression).
    Frames are extracted at 1fps from the ffmpeg screen recording, then
    labelled with a very large chunk duration to avoid splitting.
    """
    out_dir = session_dir / "naive"
    out_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = out_dir / "screenshots"

    # Extract frames from every monitor recording
    yield "[naive] Extracting frames from ffmpeg recordings…"
    for mp4 in sorted(ffmpeg_dir.glob("screen_*.mp4")):
        target = frames_dir / mp4.stem
        yield f"[naive]   {mp4.name} → {target}"
        extract_frames_from_video(mp4, target)
    # Merge all frame dirs into one by symlinking / copying
    merged = out_dir / "screenshots_merged"
    merged.mkdir(exist_ok=True)
    import shutil
    for sub in sorted(frames_dir.iterdir()):
        for img in sorted(sub.iterdir()):
            dest = merged / img.name
            if not dest.exists():
                shutil.copy2(img, dest)

    yield "[naive] Running label pipeline (no split)…"
    cmd = _uv_run(
        "-m", "label",
        "--session", str(out_dir),
        "--video-only",
        "--chunk-duration", "99999",
        "--client", "gemini",
        "--model", "gemini-2.0-flash-preview",
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
) -> AsyncIterator[str]:
    """
    Split: same as naive but split into 1-min chunks before labelling.
    """
    out_dir = session_dir / "split"
    out_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = out_dir / "screenshots_merged"
    frames_dir.mkdir(exist_ok=True)

    yield "[split] Extracting frames from ffmpeg recordings…"
    import shutil
    for mp4 in sorted(ffmpeg_dir.glob("screen_*.mp4")):
        target = out_dir / "screenshots" / mp4.stem
        yield f"[split]   {mp4.name} → {target}"
        extract_frames_from_video(mp4, target)
        for img in sorted(target.iterdir()):
            dest = frames_dir / img.name
            if not dest.exists():
                shutil.copy2(img, dest)

    yield "[split] Running label pipeline (60 s chunks)…"
    cmd = _uv_run(
        "-m", "label",
        "--session", str(out_dir),
        "--video-only",
        "--chunk-duration", "60",
        "--client", "gemini",
        "--model", "gemini-2.0-flash-preview",
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
    pack_session_dir: Path,
    session_dir: Path,
) -> AsyncIterator[str]:
    """
    Split + compress: use pack screenshots (event-driven, already compressed)
    and build 1-min chunk videos, then label with Gemini.
    """
    out_dir = session_dir / "split_compress"
    out_dir.mkdir(parents=True, exist_ok=True)

    screenshots_dir = pack_session_dir / "screenshots"
    if not screenshots_dir.exists():
        yield "[split_compress] ERROR: pack screenshots dir not found"
        state.processing.status["split_compress"] = "error"
        state.processing.errors["split_compress"] = "screenshots dir missing"
        state.save()
        return

    # Point label at the pack session which already has screenshots/
    yield "[split_compress] Running label pipeline on pack screenshots (60 s chunks)…"
    cmd = _uv_run(
        "-m", "label",
        "--session", str(pack_session_dir),
        "--video-only",
        "--chunk-duration", "60",
        "--client", "gemini",
        "--model", "gemini-2.0-flash-preview",
    )
    async for line in _stream_subprocess(cmd):
        yield f"[split_compress] {line}"

    state.processing.output_dirs["split_compress"] = str(pack_session_dir)
    # MP4 size = chunk videos built from pack screenshots
    state.processing.mp4_sizes_mb["split_compress"] = _total_mp4_size_mb(pack_session_dir)
    state.processing.status["split_compress"] = "done"
    state.save()
    yield "[split_compress] ✓ done"


async def run_split_compress_io(
    state: DemoState,
    pack_session_dir: Path,
    session_dir: Path,
) -> AsyncIterator[str]:
    """
    Split + compress + IO: full pack labelling (screenshots + events), then
    fuse with split_compress video-only captions via pack_fuse.py.
    """
    out_dir = session_dir / "split_compress_io"
    out_dir.mkdir(parents=True, exist_ok=True)
    fused_dir = out_dir / "fused"
    fused_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: label pack session with full IO context
    yield "[split_compress_io] Running full pack label (screenshots + events)…"
    cmd = _uv_run(
        "-m", "label",
        "--session", str(pack_session_dir),
        "--chunk-duration", "60",
        "--client", "gemini",
        "--model", "gemini-2.0-flash-preview",
    )
    async for line in _stream_subprocess(cmd):
        yield f"[split_compress_io] {line}"

    # split_compress output dir (video-only captions already produced)
    sc_dir = state.processing.output_dirs.get("split_compress") or str(pack_session_dir)

    # Step 2: fuse video-only + pack captions
    yield "[split_compress_io] Fusing captions with pack_fuse.py…"
    demo_dir = Path(__file__).parent.parent  # demo/
    fuse_script = demo_dir / "pack_fuse.py"
    # video chunks were built by the label pipeline inside pack_session_dir
    chunk_video_dir = pack_session_dir  # label writes chunk_NNN.mp4 here
    cmd2 = [
        sys.executable, str(fuse_script),
        "--video-only-name", sc_dir,
        "--pack-name", str(pack_session_dir),
        "--video-dir", str(chunk_video_dir),
        "--output-dir", str(fused_dir),
    ]
    async for line in _stream_subprocess(cmd2):
        yield f"[split_compress_io] {line}"

    state.processing.output_dirs["split_compress_io"] = str(fused_dir)
    state.processing.mp4_sizes_mb["split_compress_io"] = _total_mp4_size_mb(pack_session_dir)
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
) -> AsyncIterator[str]:
    pack_dir = Path(state.recording.pack_session_dir)
    ffmpeg_dir = Path(state.recording.ffmpeg_dir)

    state.processing.status[method] = "running"
    state.processing.errors[method] = None
    state.save()
    try:
        if method == "naive":
            async for line in run_naive(state, ffmpeg_dir, session_dir):
                yield line
        elif method == "split":
            async for line in run_split(state, ffmpeg_dir, session_dir):
                yield line
        elif method == "split_compress":
            async for line in run_split_compress(state, pack_dir, session_dir):
                yield line
        elif method == "split_compress_io":
            async for line in run_split_compress_io(state, pack_dir, session_dir):
                yield line
        else:
            raise ValueError(f"Unknown method: {method}")
    except Exception as exc:
        state.processing.status[method] = "error"
        state.processing.errors[method] = str(exc)
        state.save()
        yield f"[{method}] ERROR: {exc}"
        raise
