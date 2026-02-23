"""
DemoState – single source of truth for the demo pipeline.

Everything is persisted to <session_dir>/state.json so the dashboard
can resume from wherever it left off.
"""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional


METHODS = ["naive", "split", "split_compress", "split_compress_io"]


@dataclass
class RecordingState:
    running: bool = False
    pack_session_dir: Optional[str] = None       # path to pack session dir
    ffmpeg_dir: Optional[str] = None             # path to ffmpeg output dir
    start_time: Optional[float] = None
    end_time: Optional[float] = None


@dataclass
class ProcessingState:
    # per-method status: "pending" | "running" | "done" | "error"
    status: Dict[str, str] = field(default_factory=lambda: {m: "pending" for m in METHODS})
    output_dirs: Dict[str, Optional[str]] = field(default_factory=lambda: {m: None for m in METHODS})
    mp4_sizes_mb: Dict[str, Optional[float]] = field(default_factory=lambda: {m: None for m in METHODS})
    errors: Dict[str, Optional[str]] = field(default_factory=lambda: {m: None for m in METHODS})


@dataclass
class GTState:
    done: bool = False
    gt_captions_path: Optional[str] = None   # <session>/gt/gt_captions.jsonl
    num_chunks: int = 0                       # number of 8-caption groups


@dataclass
class JudgeState:
    status: str = "pending"   # pending | running | done | error
    results_path: Optional[str] = None        # <session>/judge/summary.json
    summary: Optional[Dict[str, Any]] = None  # mean ± se per method


@dataclass
class HumanEvalState:
    status: str = "pending"   # pending | running | done
    rankings_path: Optional[str] = None       # <session>/human_eval/rankings.json
    correlations_path: Optional[str] = None   # <session>/human_eval/correlations.json
    correlations: Optional[Dict[str, Any]] = None


@dataclass
class DemoState:
    session_dir: Optional[str] = None
    recording: RecordingState = field(default_factory=RecordingState)
    processing: ProcessingState = field(default_factory=ProcessingState)
    gt: GTState = field(default_factory=GTState)
    judge: JudgeState = field(default_factory=JudgeState)
    human_eval: HumanEvalState = field(default_factory=HumanEvalState)

    # ------------------------------------------------------------------ #
    _lock: threading.Lock = field(default_factory=threading.Lock, compare=False, repr=False)

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #

    @property
    def state_path(self) -> Optional[Path]:
        if self.session_dir:
            return Path(self.session_dir) / "state.json"
        return None

    def save(self):
        path = self.state_path
        if path is None:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            data = _to_dict(self)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, session_dir: str | Path) -> "DemoState":
        session_dir = str(session_dir)
        state_path = Path(session_dir) / "state.json"
        if state_path.exists():
            with open(state_path) as f:
                data = json.load(f)
            state = _from_dict(data)
            state.session_dir = session_dir  # ensure consistency
        else:
            state = cls(session_dir=session_dir)
        return state

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    def to_json(self) -> Dict[str, Any]:
        with self._lock:
            return _to_dict(self)


def _to_dict(state: DemoState) -> Dict[str, Any]:
    """Recursively convert dataclass to plain dict (skips _lock)."""
    def _conv(obj: Any) -> Any:
        if isinstance(obj, threading.Lock):
            return None  # skip
        if hasattr(obj, "__dataclass_fields__"):
            return {k: _conv(v) for k, v in asdict(obj).items() if k != "_lock"}
        if isinstance(obj, dict):
            return {k: _conv(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_conv(v) for v in obj]
        return obj

    d = {k: _conv(v) for k, v in asdict(state).items() if k != "_lock"}
    return d


def _from_dict(data: Dict[str, Any]) -> DemoState:
    """Reconstruct DemoState from a plain dict (e.g. loaded from JSON)."""
    def _pick(cls, d: dict):
        if d is None:
            return cls()
        fields = {f for f in cls.__dataclass_fields__}
        kwargs = {k: v for k, v in d.items() if k in fields}
        return cls(**kwargs)

    state = DemoState(
        session_dir=data.get("session_dir"),
        recording=_pick(RecordingState, data.get("recording")),
        processing=_pick(ProcessingState, data.get("processing")),
        gt=_pick(GTState, data.get("gt")),
        judge=_pick(JudgeState, data.get("judge")),
        human_eval=_pick(HumanEvalState, data.get("human_eval")),
    )
    return state
