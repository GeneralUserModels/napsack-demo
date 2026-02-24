"""
Cross-platform FFmpeg screen recorder that follows the focused monitor.

Instead of recording every monitor as a separate file, this recorder
detects which monitor currently has the focused workspace/window and
records only that one.  When focus moves to a different monitor the
current ffmpeg process is stopped and a new one is started for the new
monitor.  On stop() the segments are concatenated into a single output
file (output.mp4).

Writes recordings_meta.json with per-segment start/end Unix timestamps
in the same format used by the pack recording pipeline.

Supports:
  Linux   – i3-msg + xrandr  (X11)
  macOS   – avfoundation (single-screen fallback)
  Windows – gdigrab      (single-screen fallback)
"""

import json
import os
import platform
import subprocess
import threading
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional


# ---------------------------------------------------------------------------
# Platform-specific monitor detection
# ---------------------------------------------------------------------------

@dataclass
class MonitorInfo:
    index: int
    width: int
    height: int
    x: int           # offset from left edge of virtual desktop
    y: int           # offset from top edge of virtual desktop
    name: str        # display / output name


def _get_monitors_linux() -> List[MonitorInfo]:
    """Use xrandr to enumerate connected monitors on Linux/X11."""
    try:
        out = subprocess.check_output(["xrandr", "--query"], text=True)
    except FileNotFoundError:
        raise RuntimeError("xrandr not found. Ensure X11/xrandr is installed.")

    monitors: List[MonitorInfo] = []
    idx = 0
    for line in out.splitlines():
        if " connected" in line and "disconnected" not in line:
            parts = line.split()
            name = parts[0]
            for token in parts:
                if "x" in token and "+" in token and not token.startswith("("):
                    try:
                        w_h, ox, oy = token.split("+")[0], token.split("+")[1], token.split("+")[2]
                        w, h = w_h.split("x")
                        monitors.append(MonitorInfo(idx, int(w), int(h), int(ox), int(oy), name))
                        idx += 1
                        break
                    except Exception:
                        pass

    if monitors:
        return monitors

    # Fallback
    try:
        out2 = subprocess.check_output(["xdpyinfo"], text=True)
        for line2 in out2.splitlines():
            if "dimensions:" in line2:
                dims = line2.strip().split()[1]
                w, h = dims.split("x")
                return [MonitorInfo(0, int(w), int(h), 0, 0, ":0")]
    except Exception:
        pass
    return [MonitorInfo(0, 1920, 1080, 0, 0, ":0")]


def _get_monitors_macos() -> List[MonitorInfo]:
    """Use system_profiler to enumerate displays on macOS."""
    try:
        out = subprocess.check_output(
            ["system_profiler", "SPDisplaysDataType", "-json"], text=True
        )
        data = json.loads(out)
        displays = []
        for gpu in data.get("SPDisplaysDataType", []):
            for disp in gpu.get("spdisplays_ndrvs", []):
                res_str = disp.get("_spdisplays_resolution", "")
                parts = res_str.split("@")[0].strip().replace(" x ", "x")
                if "x" in parts:
                    w, h = parts.split("x")
                    displays.append((int(w.strip()), int(h.strip())))
        monitors = []
        for idx, (w, h) in enumerate(displays):
            monitors.append(MonitorInfo(idx, w, h, 0, 0, str(idx + 1)))
        return monitors if monitors else [MonitorInfo(0, 1920, 1080, 0, 0, "1")]
    except Exception:
        return [MonitorInfo(0, 1920, 1080, 0, 0, "1")]


def _get_monitors_windows() -> List[MonitorInfo]:
    """Use ctypes to enumerate monitors on Windows."""
    try:
        import ctypes
        user32 = ctypes.windll.user32
        width = user32.GetSystemMetrics(0)
        height = user32.GetSystemMetrics(1)
        return [MonitorInfo(0, width, height, 0, 0, "desktop")]
    except Exception:
        return [MonitorInfo(0, 1920, 1080, 0, 0, "desktop")]


def get_monitors() -> List[MonitorInfo]:
    system = platform.system()
    if system == "Linux":
        return _get_monitors_linux()
    elif system == "Darwin":
        return _get_monitors_macos()
    elif system == "Windows":
        return _get_monitors_windows()
    else:
        raise RuntimeError(f"Unsupported platform: {system}")


# ---------------------------------------------------------------------------
# Focused-monitor detection
# ---------------------------------------------------------------------------

def _get_focused_output_linux() -> Optional[str]:
    """Use i3-msg to get the output name of the focused workspace."""
    try:
        out = subprocess.check_output(
            ["i3-msg", "-t", "get_workspaces"], text=True, stderr=subprocess.DEVNULL
        )
        workspaces = json.loads(out)
        for ws in workspaces:
            if ws.get("focused"):
                return ws.get("output")
    except Exception:
        pass

    # Fallback: use xdotool to get cursor position
    try:
        out = subprocess.check_output(
            ["xdotool", "getmouselocation", "--shell"],
            text=True, stderr=subprocess.DEVNULL,
        )
        mx, my = 0, 0
        for line in out.splitlines():
            if line.startswith("X="):
                mx = int(line.split("=")[1])
            elif line.startswith("Y="):
                my = int(line.split("=")[1])
        monitors = _get_monitors_linux()
        for m in monitors:
            if m.x <= mx < m.x + m.width and m.y <= my < m.y + m.height:
                return m.name
    except Exception:
        pass
    return None


def get_focused_monitor(monitors: List[MonitorInfo]) -> MonitorInfo:
    """Return the MonitorInfo for the currently focused output."""
    system = platform.system()

    focused_name: Optional[str] = None
    if system == "Linux":
        focused_name = _get_focused_output_linux()

    if focused_name:
        for m in monitors:
            if m.name == focused_name:
                return m

    # Fallback: return first monitor
    return monitors[0] if monitors else MonitorInfo(0, 1920, 1080, 0, 0, ":0")


# ---------------------------------------------------------------------------
# FFmpeg command builder
# ---------------------------------------------------------------------------

def _build_ffmpeg_cmd(monitor: MonitorInfo, output_path: str,
                      target_w: int = 1920, target_h: int = 1080) -> List[str]:
    """Build an ffmpeg command that records *monitor* and scales output to
    target_w x target_h with aspect-preserving padding."""
    system = platform.system()
    display = os.environ.get("DISPLAY", ":0.0")

    scale_vf = (
        f"scale={target_w}:{target_h}"
        f":force_original_aspect_ratio=decrease,"
        f"pad={target_w}:{target_h}:(ow-iw)/2:(oh-ih)/2:black,"
        f"format=yuv420p"
    )

    if system == "Linux":
        return [
            "ffmpeg", "-y",
            "-f", "x11grab",
            "-framerate", "30",
            "-video_size", f"{monitor.width}x{monitor.height}",
            "-i", f"{display}+{monitor.x},{monitor.y}",
            "-vf", scale_vf,
            "-c:v", "libx264",
            "-crf", "23",
            "-preset", "ultrafast",
            "-tune", "zerolatency",
            "-threads", "1",
            "-flush_packets", "1",
            "-movflags", "+faststart",
            output_path,
        ]
    elif system == "Darwin":
        return [
            "ffmpeg", "-y",
            "-f", "avfoundation",
            "-capture_cursor", "1",
            "-framerate", "30",
            "-i", f"{monitor.name}:none",
            "-vf", scale_vf,
            "-c:v", "libx264",
            "-crf", "23",
            "-preset", "ultrafast",
            "-tune", "zerolatency",
            "-threads", "1",
            "-flush_packets", "1",
            "-movflags", "+faststart",
            output_path,
        ]
    elif system == "Windows":
        return [
            "ffmpeg", "-y",
            "-f", "gdigrab",
            "-framerate", "30",
            "-i", "desktop",
            "-vf", scale_vf,
            "-c:v", "libx264",
            "-crf", "23",
            "-preset", "ultrafast",
            "-tune", "zerolatency",
            "-threads", "1",
            "-flush_packets", "1",
            "-movflags", "+faststart",
            output_path,
        ]
    else:
        raise RuntimeError(f"Unsupported platform: {system}")


# ---------------------------------------------------------------------------
# Recording segment metadata
# ---------------------------------------------------------------------------

@dataclass
class RecordingMeta:
    segment_index: int
    monitor_index: int
    monitor_name: str
    width: int
    height: int
    video_path: str        # absolute path to the segment / final MP4 file
    start_time: float      # Unix timestamp
    end_time: Optional[float] = None


# ---------------------------------------------------------------------------
# Recorder
# ---------------------------------------------------------------------------

class FFmpegRecorder:
    """
    Records the focused monitor as a continuous MP4.  When the focused
    monitor changes, the current ffmpeg segment is finalized and a new
    one is started.  On stop() all segments are concatenated into a
    single ``output.mp4``.

    Call start() to begin, stop() to finish.
    Writes ``recordings_meta.json`` with per-segment timestamps.
    """

    def __init__(self, output_dir: Path, target_w: int = 1920, target_h: int = 1080):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.target_w = target_w
        self.target_h = target_h

        self._monitors: List[MonitorInfo] = []
        self._current_proc: Optional[subprocess.Popen] = None
        self._current_monitor: Optional[MonitorInfo] = None
        self._segment_idx: int = 0
        self._meta: List[RecordingMeta] = []
        self._stop_event = threading.Event()
        self._watcher_thread: Optional[threading.Thread] = None

    # ── public API ────────────────────────────────────────────────────────

    def start(self) -> List[RecordingMeta]:
        self._monitors = get_monitors()
        print(f"[FFmpegRecorder] Detected {len(self._monitors)} monitor(s)")

        focused = get_focused_monitor(self._monitors)
        self._start_segment(focused)

        # Background thread polls for monitor changes
        self._stop_event.clear()
        self._watcher_thread = threading.Thread(target=self._watch_focus, daemon=True)
        self._watcher_thread.start()

        return self._meta

    def stop(self) -> List[RecordingMeta]:
        self._stop_event.set()
        if self._watcher_thread:
            self._watcher_thread.join(timeout=5)
            self._watcher_thread = None

        self._stop_current_segment()
        self._save_meta()
        self._concatenate_segments()
        return self._meta

    # ── internals ─────────────────────────────────────────────────────────

    def _start_segment(self, monitor: MonitorInfo):
        seg_path = self.output_dir / f"segment_{self._segment_idx:03d}.mp4"
        cmd = _build_ffmpeg_cmd(monitor, str(seg_path), self.target_w, self.target_h)
        print(f"[FFmpegRecorder] Segment {self._segment_idx}: "
              f"{monitor.name} ({monitor.width}x{monitor.height}) -> {seg_path.name}")

        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        meta = RecordingMeta(
            segment_index=self._segment_idx,
            monitor_index=monitor.index,
            monitor_name=monitor.name,
            width=monitor.width,
            height=monitor.height,
            video_path=str(seg_path.resolve()),
            start_time=time.time(),
        )
        self._current_proc = proc
        self._current_monitor = monitor
        self._meta.append(meta)
        self._segment_idx += 1

    def _stop_current_segment(self):
        if self._current_proc is None:
            return
        end_time = time.time()
        try:
            self._current_proc.stdin.write(b"q")
            self._current_proc.stdin.flush()
        except Exception:
            pass
        try:
            self._current_proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            self._current_proc.kill()
            self._current_proc.wait()

        # Update end_time on the last segment meta
        if self._meta:
            self._meta[-1].end_time = end_time

        self._current_proc = None

    def _watch_focus(self):
        """Poll every 0.5 s for focus changes and switch segments."""
        while not self._stop_event.is_set():
            self._stop_event.wait(0.5)
            if self._stop_event.is_set():
                break
            try:
                focused = get_focused_monitor(self._monitors)
                if self._current_monitor and focused.name != self._current_monitor.name:
                    print(f"[FFmpegRecorder] Focus changed: "
                          f"{self._current_monitor.name} -> {focused.name}")
                    self._stop_current_segment()
                    self._start_segment(focused)
            except Exception as exc:
                print(f"[FFmpegRecorder] watcher error: {exc}")

    def _concatenate_segments(self):
        """Concatenate all segment MP4s into a single output.mp4."""
        segments = [Path(m.video_path) for m in self._meta if Path(m.video_path).exists()]
        if not segments:
            print("[FFmpegRecorder] No segments to concatenate")
            return

        out = self.output_dir / "output.mp4"

        if len(segments) == 1:
            # Just rename the single segment
            segments[0].rename(out)
            self._meta[0].video_path = str(out.resolve())
            print(f"[FFmpegRecorder] Single segment -> {out}")
            return

        # Write ffmpeg concat list
        concat_list = self.output_dir / "_concat.txt"
        with open(concat_list, "w") as f:
            for seg in segments:
                f.write(f"file '{seg}'\n")

        print(f"[FFmpegRecorder] Concatenating {len(segments)} segments -> {out.name}")
        try:
            cmd = [
                "ffmpeg", "-y",
                "-f", "concat", "-safe", "0",
                "-i", str(concat_list),
                "-c", "copy",
                str(out),
            ]
            subprocess.check_call(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except subprocess.CalledProcessError:
            # If stream-copy concat fails, re-encode
            cmd_re = [
                "ffmpeg", "-y",
                "-f", "concat", "-safe", "0",
                "-i", str(concat_list),
                "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
                "-pix_fmt", "yuv420p",
                str(out),
            ]
            subprocess.check_call(cmd_re, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        finally:
            concat_list.unlink(missing_ok=True)

        # Clean up individual segments
        for seg in segments:
            seg.unlink(missing_ok=True)

        print(f"[FFmpegRecorder] Concatenated output: {out}")

    def _save_meta(self):
        meta_path = self.output_dir / "recordings_meta.json"
        data = []
        for m in self._meta:
            entry = asdict(m)
            entry["start_datetime"] = datetime.fromtimestamp(m.start_time).strftime("%Y%m%d_%H%M%S")
            if m.end_time is not None:
                entry["end_datetime"] = datetime.fromtimestamp(m.end_time).strftime("%Y%m%d_%H%M%S")
            data.append(entry)

        with open(meta_path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"[FFmpegRecorder] Metadata saved to {meta_path}")


# ---------------------------------------------------------------------------
# CLI entry point (for manual testing)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Focus-following FFmpeg screen recorder")
    parser.add_argument("--output-dir", required=True, help="Directory to save recordings")
    parser.add_argument("--duration", type=int, default=10, help="Recording duration (0 = until Ctrl+C)")
    parser.add_argument("--width", type=int, default=1920, help="Target output width")
    parser.add_argument("--height", type=int, default=1080, help="Target output height")
    args = parser.parse_args()

    recorder = FFmpegRecorder(Path(args.output_dir), args.width, args.height)
    recorder.start()

    if args.duration > 0:
        print(f"Recording for {args.duration} seconds...")
        time.sleep(args.duration)
        recorder.stop()
        print("Done.")
    else:
        print("Recording... Press Ctrl+C to stop.")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            recorder.stop()
            print("Done.")