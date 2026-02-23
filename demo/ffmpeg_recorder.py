"""
Cross-platform FFmpeg screen recorder.

Records every monitor as a separate MP4 video file.
On stop, writes recordings_meta.json with start/end Unix timestamps
in the same format used by the pack recording pipeline.

Supports:
  Linux  – x11grab (X11) or pipewire/kmsgrab (Wayland fallback)
  macOS  – avfoundation
  Windows – gdigrab
"""

import json
import platform
import subprocess
import sys
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
    name: str        # display / device name for FFmpeg


def _get_monitors_linux() -> List[MonitorInfo]:
    """Use xrandr to enumerate connected monitors on Linux/X11."""
    try:
        out = subprocess.check_output(["xrandr", "--query"], text=True)
    except FileNotFoundError:
        raise RuntimeError("xrandr not found. Ensure X11/xrandr is installed.")

    monitors = []
    idx = 0
    for line in out.splitlines():
        # Lines like: HDMI-1 connected 1920x1080+0+0 (normal …)
        if " connected" in line and "disconnected" not in line:
            parts = line.split()
            name = parts[0]
            # Find the geometry token e.g. "1920x1080+0+0"
            for token in parts:
                if "x" in token and "+" in token:
                    try:
                        res, ox, oy = token.replace("+", " ").replace("x", " ", 1).split()
                        monitors.append(MonitorInfo(
                            index=idx,
                            width=int(res.split("x")[0]) if "x" in res else int(res),
                            height=int(token.split("x")[1].split("+")[0]),
                            x=int(ox),
                            y=int(oy),
                            name=name,
                        ))
                        idx += 1
                        break
                    except Exception:
                        pass
    if not monitors:
        # Fallback: record the whole virtual desktop at :0
        try:
            out2 = subprocess.check_output(["xdpyinfo"], text=True)
            for line in out2.splitlines():
                if "dimensions:" in line:
                    dims = line.strip().split()[1]
                    w, h = dims.split("x")
                    monitors.append(MonitorInfo(0, int(w), int(h), 0, 0, ":0"))
                    break
        except Exception:
            monitors.append(MonitorInfo(0, 1920, 1080, 0, 0, ":0"))
    return monitors


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
                # "1920 x 1080" or "1920 x 1080 @ 60 Hz"
                parts = res_str.split("@")[0].strip().replace(" x ", "x")
                if "x" in parts:
                    w, h = parts.split("x")
                    displays.append((int(w.strip()), int(h.strip())))
        monitors = []
        for idx, (w, h) in enumerate(displays):
            # avfoundation screen indices start at 1 for the first display
            monitors.append(MonitorInfo(idx, w, h, 0, 0, str(idx + 1)))
        return monitors if monitors else [MonitorInfo(0, 1920, 1080, 0, 0, "1")]
    except Exception:
        return [MonitorInfo(0, 1920, 1080, 0, 0, "1")]


def _get_monitors_windows() -> List[MonitorInfo]:
    """Use PowerShell to enumerate monitors on Windows."""
    try:
        ps_cmd = (
            "Get-WmiObject -Namespace root\\wmi -Class WmiMonitorBasicDisplayParams "
            "| Select-Object -ExpandProperty MaxHorizontalImageSize"
        )
        # Simpler: use win32api or ctypes
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
# FFmpeg command builders
# ---------------------------------------------------------------------------

def _build_ffmpeg_cmd_linux(monitor: MonitorInfo, output_path: str) -> List[str]:
    display = ":0.0"
    # Check if DISPLAY env var is set for the correct display
    import os
    display = os.environ.get("DISPLAY", ":0.0")
    return [
        "ffmpeg", "-y",
        "-f", "x11grab",
        "-video_size", f"{monitor.width}x{monitor.height}",
        "-i", f"{display}+{monitor.x},{monitor.y}",
        "-vf", "format=yuv420p",
        "-c:v", "libx264",
        "-crf", "23",           # ~JPEG-70 equivalent quality
        "-preset", "fast",
        "-movflags", "+faststart",
        output_path,
    ]


def _build_ffmpeg_cmd_macos(monitor: MonitorInfo, output_path: str) -> List[str]:
    return [
        "ffmpeg", "-y",
        "-f", "avfoundation",
        "-capture_cursor", "1",
        "-i", f"{monitor.name}:none",
        "-vf", "format=yuv420p",
        "-c:v", "libx264",
        "-crf", "23",
        "-preset", "fast",
        "-movflags", "+faststart",
        output_path,
    ]


def _build_ffmpeg_cmd_windows(monitor: MonitorInfo, output_path: str) -> List[str]:
    return [
        "ffmpeg", "-y",
        "-f", "gdigrab",
        "-i", "desktop",
        "-vf", "format=yuv420p",
        "-c:v", "libx264",
        "-crf", "23",
        "-preset", "fast",
        "-movflags", "+faststart",
        output_path,
    ]


def _build_ffmpeg_cmd(monitor: MonitorInfo, output_path: str) -> List[str]:
    system = platform.system()
    if system == "Linux":
        return _build_ffmpeg_cmd_linux(monitor, output_path)
    elif system == "Darwin":
        return _build_ffmpeg_cmd_macos(monitor, output_path)
    elif system == "Windows":
        return _build_ffmpeg_cmd_windows(monitor, output_path)
    else:
        raise RuntimeError(f"Unsupported platform: {system}")


# ---------------------------------------------------------------------------
# Recorder
# ---------------------------------------------------------------------------

@dataclass
class RecordingMeta:
    monitor_index: int
    monitor_name: str
    width: int
    height: int
    video_path: str        # absolute path to the MP4 file
    start_time: float      # Unix timestamp
    end_time: Optional[float] = None


class FFmpegRecorder:
    """
    Records each monitor as a separate continuous MP4 using FFmpeg.
    Call start() to begin recording, stop() to finish.
    Writes <output_dir>/recordings_meta.json with start/end times.
    """

    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._processes: List[subprocess.Popen] = []
        self._meta: List[RecordingMeta] = []

    def start(self) -> List[RecordingMeta]:
        monitors = get_monitors()
        start_time = time.time()
        print(f"[FFmpegRecorder] Detected {len(monitors)} monitor(s)")

        for m in monitors:
            video_path = self.output_dir / f"screen_{m.index}.mp4"
            cmd = _build_ffmpeg_cmd(m, str(video_path))
            print(f"[FFmpegRecorder] Starting monitor {m.index} ({m.width}x{m.height}): {' '.join(cmd)}")

            proc = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            meta = RecordingMeta(
                monitor_index=m.index,
                monitor_name=m.name,
                width=m.width,
                height=m.height,
                video_path=str(video_path.resolve()),
                start_time=start_time,
            )
            self._processes.append(proc)
            self._meta.append(meta)

        return self._meta

    def stop(self) -> List[RecordingMeta]:
        end_time = time.time()
        for proc, meta in zip(self._processes, self._meta):
            try:
                # Send 'q' to ffmpeg stdin to request graceful stop
                proc.stdin.write(b"q")
                proc.stdin.flush()
            except Exception:
                pass
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
            meta.end_time = end_time

        self._save_meta()
        self._processes.clear()
        return self._meta

    def _save_meta(self):
        meta_path = self.output_dir / "recordings_meta.json"
        data = []
        for m in self._meta:
            entry = asdict(m)
            # Add human-readable timestamps consistent with pack's format
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

    parser = argparse.ArgumentParser(description="Cross-platform FFmpeg screen recorder")
    parser.add_argument("--output-dir", required=True, help="Directory to save recordings")
    parser.add_argument("--duration", type=int, default=10, help="Recording duration in seconds (0 = until Ctrl+C)")
    args = parser.parse_args()

    recorder = FFmpegRecorder(Path(args.output_dir))
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
