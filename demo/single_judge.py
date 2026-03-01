"""
single_judge.py – pairwise LLM evaluation for the NAPsack demo pipeline.

Demo-mode CLI (--session-dir):
  Evaluates all four methods (naive, split, split_compress, split_compress_io)
  against GT captions that are grouped in chunks of 8 captions.
  Runs --runs times with independent Gemini calls, computes per-method
  mean ± bootstrap SE, and saves to <session_dir>/judge/summary.json.

Legacy standalone mode: same as before (no --session-dir).
"""

import bisect
import json
import os
import random
import argparse
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import google.generativeai as genai


# ---------------------------------------------------------------------------
# Default judge prompt (used when no external prompt file is given)
# ---------------------------------------------------------------------------
DEFAULT_JUDGE_PROMPT = """\
You are an expert evaluator comparing computer-use action captions.

GROUND TRUTH:
{GT}

CANDIDATE:
{CANDIDATE}

Score the candidate from 0.0 to 1.0 based on:
- Action Accuracy: are the described actions correct?
- Order & Temporal Consistency: are events in the right order?
- Semantic Fidelity: is the meaning preserved?
- Specificity / Entity Fidelity: are apps, files, commands named correctly?
- Verbosity: is the level of detail appropriate (not too terse, not too verbose)?

Return ONLY valid JSON: {"score": <float>, "rationale": "<one sentence>"}
"""


# ---------------------------------------------------------------------------
# Caption loader
# ---------------------------------------------------------------------------

class CaptionLoader:
    """Load and chunk caption data from different methods."""

    def __init__(self, base_path: str = ".", num_gt_chunks: int = 3):
        self.base_path = Path(base_path)
        self.num_gt_chunks = num_gt_chunks

    # ------------------------------------------------------------------ #
    # GT loading (two formats)
    # ------------------------------------------------------------------ #

    def load_ground_truth_txt(self) -> List[str]:
        """Load ground truth from gt/gt_N.txt files (legacy format)."""
        gt_chunks = []
        for i in range(self.num_gt_chunks):
            gt_file = self.base_path / "gt" / f"gt_{i}.txt"
            if gt_file.exists():
                with open(gt_file, 'r') as f:
                    gt_chunks.append(f.read())
            else:
                print(f"Warning: GT file not found: {gt_file}")
                gt_chunks.append("")
        return gt_chunks

    def load_ground_truth_jsonl(self, gt_captions_path: Path) -> List[List[Dict]]:
        """
        Load GT captions from gt_captions.jsonl and group into chunks of 8.
        Each chunk is a list of caption dicts with keys: caption, start_time, end_time, etc.
        """
        entries = self._load_jsonl(gt_captions_path)
        chunk_size = 8
        chunks = [entries[i:i + chunk_size] for i in range(0, len(entries), chunk_size)]
        return chunks

    # ------------------------------------------------------------------ #
    # Method caption loading
    # ------------------------------------------------------------------ #

    def load_from_chunk_dirs(self, name: str) -> List[List[Dict]]:
        """Load captions from chunk_NNN/captions.jsonl (label pipeline output)."""
        base = self.base_path / name if not Path(name).is_absolute() else Path(name)
        chunks = []
        for chunk_dir in sorted(base.glob("chunk_??*")):
            cap_file = chunk_dir / "captions.jsonl"
            chunks.append(self._load_jsonl(cap_file))
        return chunks

    def load_video_only(self, name: str) -> List[List[Dict]]:
        """Load compression without key (chunk_NNN/captions.jsonl)."""
        return self.load_from_chunk_dirs(name)

    def load_napsack(self, name: str = "napsack") -> List[List[Dict]]:
        """Load compression with key and chunk into 10-minute segments."""
        data_file = self.base_path / name / "data.jsonl"
        all_entries = self._load_jsonl(data_file)

        if not all_entries:
            return []

        # Chunk by 10-minute intervals
        chunks = []
        current_chunk = []
        start_time = all_entries[0]["start_time"]
        chunk_duration = 600  # 10 minutes in seconds

        for entry in all_entries:
            if entry["start_time"] - start_time < chunk_duration:
                current_chunk.append(entry)
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                current_chunk = [entry]
                start_time = entry["start_time"]

        if current_chunk:
            chunks.append(current_chunk)

        return chunks

    def _load_jsonl(self, filepath: Path) -> List[Dict]:
        """Load JSONL file."""
        if not filepath.exists():
            return []

        entries = []
        with open(filepath, 'r') as f:
            for line in f:
                if line.strip():
                    entries.append(json.loads(line))
        return entries


class GeminiEvaluator:
    """Manage automated evaluation using Gemini API."""

    def __init__(
        self,
        api_key: str,
        methods: List[str],
        prompt_file: Optional[str] = None,
        results_file: str = "gemini_evaluation_results.json",
    ):
        import threading as _th
        self.results_file = results_file
        self.methods = methods
        self._lock = _th.Lock()

        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel("gemini-3-flash-preview")

        self.results = self._load_results()
        if prompt_file and Path(prompt_file).exists():
            with open(prompt_file) as f:
                self.evaluation_prompt_template = f.read()
        else:
            self.evaluation_prompt_template = DEFAULT_JUDGE_PROMPT

    def _load_results(self) -> Dict:
        if Path(self.results_file).exists():
            with open(self.results_file) as f:
                data = json.load(f)
            data["completed_pairs"] = set(data.get("completed_pairs", []))
            return data
        return {"evaluations": [], "completed_pairs": set()}

    def _save_results(self):
        save_data = {
            "evaluations": self.results["evaluations"],
            "completed_pairs": list(self.results.get("completed_pairs", set())),
        }
        with open(self.results_file, "w") as f:
            json.dump(save_data, f, indent=2)

    def is_pair_evaluated(self, chunk_idx: int, method: str, run: int = 0) -> bool:
        with self._lock:
            pair_id = f"{chunk_idx}_{method}_run{run}"
            return pair_id in set(self.results.get("completed_pairs", []))

    def format_captions_for_eval(self, captions: List[Dict]) -> str:
        lines = []
        for entry in captions:
            if "caption" in entry:
                text = entry["caption"].replace("<action>", "").replace("</action>", "")
                lines.append(text)
        return "\n".join(lines)

    def format_gt_chunk(self, gt_chunk: List[Dict]) -> str:
        lines = []
        for entry in gt_chunk:
            text = entry.get("caption", "")
            text = text.replace("<action>", "").replace("</action>", "")
            if text:
                lines.append(text)
        return "\n".join(lines)

    def evaluate_pair(
        self,
        chunk_idx: int,
        method: str,
        ground_truth: str,
        candidate: List[Dict],
        run: int = 0,
    ) -> Dict:
        candidate_text = self.format_captions_for_eval(candidate)
        prompt = self.evaluation_prompt_template.replace("{GT}", ground_truth)
        prompt = prompt.replace("{CANDIDATE}", candidate_text)
        try:
            response = self.model.generate_content(prompt)
            response_text = response.text
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text:
                response_text = response_text.split("```")[1].split("```")[0].strip()
            evaluation = json.loads(response_text)
            return {"chunk_idx": chunk_idx, "method": method, "run": run,
                    "evaluation": evaluation, "raw_response": response.text}
        except Exception as e:
            print(f"Error evaluating chunk {chunk_idx}, method {method}, run {run}: {e}")
            return {"chunk_idx": chunk_idx, "method": method, "run": run,
                    "error": str(e), "evaluation": None}

    def add_evaluation(self, chunk_idx: int, method: str, result: Dict, run: int = 0):
        with self._lock:
            self.results["evaluations"].append(result)
            pair_id = f"{chunk_idx}_{method}_run{run}"
            if "completed_pairs" not in self.results:
                self.results["completed_pairs"] = set()
            self.results["completed_pairs"] = set(self.results["completed_pairs"])
            self.results["completed_pairs"].add(pair_id)
            self._save_results()

    def get_statistics(self) -> Dict:
        scores: Dict[str, List[float]] = defaultdict(list)
        for ev in self.results["evaluations"]:
            if ev.get("evaluation") and "score" in ev["evaluation"]:
                scores[ev["method"]].append(ev["evaluation"]["score"])
        statistics = {}
        for method in self.methods:
            s = scores.get(method, [])
            statistics[method] = {
                "mean_score": float(np.mean(s)) if s else 0.0,
                "min_score": float(np.min(s)) if s else 0.0,
                "max_score": float(np.max(s)) if s else 0.0,
                "num_evaluations": len(s),
                "all_scores": s,
            }
        return statistics


# ---------------------------------------------------------------------------
# Bootstrap SE
# ---------------------------------------------------------------------------

def bootstrap_se(scores: List[float], n_resamples: int = 1000) -> float:
    if len(scores) < 2:
        return 0.0
    arr = np.array(scores)
    means = [np.mean(arr[np.random.randint(0, len(arr), len(arr))]) for _ in range(n_resamples)]
    return float(np.std(means))


# ---------------------------------------------------------------------------
# Demo-mode evaluation (called by the dashboard)
# ---------------------------------------------------------------------------

DEMO_METHODS = ["naive", "split", "split_compress", "split_compress_io"]


def _load_method_chunks(session_dir: Path, method: str) -> List[List[Dict]]:
    """Load flat captions from {method}/captions.jsonl and return as a
    single-element list so callers can flatten→re-align via _chunk_captions_by_gt.
    """
    cap_file = session_dir / method / "captions.jsonl"
    if not cap_file.exists():
        print(f"Warning: captions file not found: {cap_file}")
        return []
    entries: List[Dict] = []
    with open(cap_file) as f:
        for line in f:
            if line.strip():
                entries.append(json.loads(line))
    # Return as [[all_captions]] – callers flatten then re-align by GT windows
    return [entries] if entries else []


def _to_sec(v) -> float:
    """Convert a timestamp value to seconds. Handles MM:SS strings, unix epochs, and floats."""
    if isinstance(v, str):
        v = v.strip()
        if ":" in v:
            p = v.split(":")
            try:
                return int(p[0]) * 60 + int(p[1])
            except ValueError:
                return 0.0
        if not v:
            return 0.0
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0.0


def _build_screenshot_time_map(screenshots_dir: Path) -> List[float]:
    """
    Build a sorted list of unix timestamps from screenshot filenames.
    Each screenshot filename starts with a unix timestamp, e.g.
    '1772145150.608936_reason_key_start_stale.png'.
    The index in this list corresponds to the compressed-video second.
    """
    timestamps: List[float] = []
    for p in sorted(screenshots_dir.iterdir()):
        if p.suffix == ".png":
            try:
                ts_str = p.name.split("_")[0]
                timestamps.append(float(ts_str))
            except (ValueError, IndexError):
                pass
    return timestamps


def _load_ffmpeg_start(session_dir: Path) -> float:
    """Load the ffmpeg recording start time from recordings_meta.json."""
    meta_path = session_dir / "ffmpeg" / "recordings_meta.json"
    if not meta_path.exists():
        return 0.0
    with open(meta_path) as f:
        segments = json.load(f)
    if segments:
        return segments[0]["start_time"]
    return 0.0


def _load_scio_data(session_dir: Path) -> List[Dict]:
    """Load split_compress_io/data.jsonl entries (which carry unix timestamps)."""
    data_file = session_dir / "split_compress_io" / "data.jsonl"
    if not data_file.exists():
        return []
    entries: List[Dict] = []
    with open(data_file) as f:
        for line in f:
            if line.strip():
                entries.append(json.loads(line))
    return entries


def _unix_to_screenshot_index(
    unix_time: float, screenshot_timestamps: List[float],
) -> int:
    """Find the screenshot index whose timestamp is closest to *unix_time*."""
    if not screenshot_timestamps:
        return 0
    idx = bisect.bisect_left(screenshot_timestamps, unix_time)
    if idx == 0:
        return 0
    if idx >= len(screenshot_timestamps):
        return len(screenshot_timestamps) - 1
    # Return whichever neighbour is closer
    if unix_time - screenshot_timestamps[idx - 1] <= screenshot_timestamps[idx] - unix_time:
        return idx - 1
    return idx


def _scio_sec_to_unix(
    target_sec: float,
    scio_entries: List[Dict],
    use_end: bool = False,
) -> float:
    """
    Convert a formatted-time second (shared by GT & SCIO) to unix time
    using split_compress_io data.jsonl entries as a lookup table.

    use_end=False → match start_formatted, return start_time
    use_end=True  → match end_formatted,   return end_time
    """
    if not scio_entries:
        return 0.0
    fmt_key = "end_formatted" if use_end else "start_formatted"
    unix_key = "end_time" if use_end else "start_time"
    best_entry = None
    best_diff = float('inf')
    for e in scio_entries:
        e_sec = _to_sec(e.get(fmt_key, ""))
        diff = abs(e_sec - target_sec)
        if diff < best_diff:
            best_diff = diff
            best_entry = e
            if diff == 0:
                break  # Exact match
    return best_entry.get(unix_key, 0.0) if best_entry else 0.0


def _ranges_overlap(
    a_start: float, a_end: float, b_start: float, b_end: float,
) -> bool:
    """Check whether time ranges [a_start, a_end] and [b_start, b_end] overlap."""
    return not (a_end < b_start or a_start > b_end)


def _chunk_captions_by_gt(
    method_captions: List[Dict],
    gt_chunks: List[List[Dict]],
    method: str,
    scio_data: List[Dict],
    screenshot_timestamps: List[float],
    ffmpeg_start: float,
) -> List[List[Dict]]:
    """
    Re-group flat method captions to align with GT chunk time windows.

    Uses split_compress_io/data.jsonl as a *time bridge*:
    - GT and SCIO share the same formatted-time system (mm:ss → index seconds).
    - SCIO data.jsonl provides unix timestamps for cross-method mapping.

    Time conversion per method:
    - naive / split:      video_sec = unix_time − ffmpeg_start
    - split_compress:     screenshot-index via binary search in timestamp list
    - split_compress_io:  direct formatted-time matching (same system as GT)
    """
    result: List[List[Dict]] = []

    for gt_chunk in gt_chunks:
        # 1. Determine GT chunk time window in formatted seconds
        valid_starts = [
            _to_sec(e.get("start_seconds", e.get("start", 0)))
            for e in gt_chunk if e.get("start")
        ]
        valid_ends = [
            _to_sec(e.get("end_seconds", e.get("end", 0)))
            for e in gt_chunk if e.get("end")
        ]
        if not valid_starts or not valid_ends:
            result.append([])
            continue

        gt_start_sec = min(valid_starts)
        gt_end_sec = max(valid_ends)

        # 2. split_compress_io: direct formatted-time matching
        if method == "split_compress_io":
            window = [
                c for c in method_captions
                if gt_start_sec
                <= _to_sec(c.get("start_seconds", c.get("start", 0)))
                <= gt_end_sec
            ]
            result.append(window)
            continue

        # 3. Convert GT formatted seconds → unix time via SCIO bridge
        unix_start = _scio_sec_to_unix(gt_start_sec, scio_data, use_end=False)
        unix_end = _scio_sec_to_unix(gt_end_sec, scio_data, use_end=True)

        if unix_start == 0.0 or unix_end == 0.0:
            # Fallback when no SCIO data is available
            window = [
                c for c in method_captions
                if gt_start_sec
                <= _to_sec(c.get("start_seconds", c.get("start", 0)))
                <= gt_end_sec
            ]
            result.append(window)
            continue

        # 4a. naive / split → ffmpeg video seconds
        if method in ("naive", "split"):
            vid_start = unix_start - ffmpeg_start
            vid_end = unix_end - ffmpeg_start
            window = [
                c for c in method_captions
                if _ranges_overlap(
                    _to_sec(c.get("start_seconds", c.get("start", 0))),
                    _to_sec(c.get("end_seconds", c.get("end", 0))),
                    vid_start, vid_end,
                )
            ]

        # 4b. split_compress → screenshot indices
        elif method == "split_compress":
            idx_start = _unix_to_screenshot_index(unix_start, screenshot_timestamps)
            idx_end = _unix_to_screenshot_index(unix_end, screenshot_timestamps)
            window = [
                c for c in method_captions
                if _ranges_overlap(
                    _to_sec(c.get("start_seconds", c.get("start", 0))),
                    _to_sec(c.get("end_seconds", c.get("end", 0))),
                    float(idx_start), float(idx_end),
                )
            ]

        else:
            window = []

        result.append(window)

    return result


def run_demo_judge(
    session_dir: Path,
    gt_captions_path: Path,
    output_dir: Path,
    num_runs: int = 3,
    num_workers: int = 4,
    prompt_file: Optional[str] = None,
    n_bootstrap: int = 1000,
) -> Dict:
    """
    Run the LLM judge in demo mode.

    Args:
        session_dir: root demo session directory
        gt_captions_path: path to gt_captions.jsonl
        output_dir: where to save per-run results and summary
        num_runs: number of independent LLM evaluation runs
        num_workers: number of parallel Gemini evaluation workers
        prompt_file: optional path to external judge prompt
        n_bootstrap: number of bootstrap resamples for SE

    Returns:
        summary dict with per-method mean ± bootstrap SE
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY not set")

    loader = CaptionLoader()
    gt_chunks = loader.load_ground_truth_jsonl(gt_captions_path)
    num_gt_chunks = len(gt_chunks)
    print(f"GT: {num_gt_chunks} chunks of 8 captions each", flush=True)

    # ── Build screenshot timestamp map ──
    screenshots_dir: Optional[Path] = None
    for candidate_dir in [
        session_dir / "napsack_session" / "screenshots",
        session_dir / "split_compress" / "screenshots",
    ]:
        if candidate_dir.exists():
            screenshots_dir = candidate_dir
            break

    screenshot_timestamps: List[float] = []
    if screenshots_dir:
        screenshot_timestamps = _build_screenshot_time_map(screenshots_dir)
        if screenshot_timestamps:
            napsack_start = screenshot_timestamps[0]
            print(f"  Screenshots: {len(screenshot_timestamps)} frames, "
                  f"span = {screenshot_timestamps[-1] - napsack_start:.1f}s", flush=True)

    # ── Load ffmpeg start time ──
    ffmpeg_start = _load_ffmpeg_start(session_dir)
    if ffmpeg_start:
        print(f"  ffmpeg start: {ffmpeg_start}", flush=True)

    # ── Load SCIO data.jsonl (unix timestamps) as time bridge ──
    scio_data = _load_scio_data(session_dir)
    if scio_data:
        print(f"  SCIO data.jsonl: {len(scio_data)} entries (time bridge)", flush=True)
    else:
        print(f"  WARNING: split_compress_io/data.jsonl not found — time alignment may be inaccurate", flush=True)

    # ── Load method captions and align to GT chunks ──
    method_data: Dict[str, List[List[Dict]]] = {}
    for m in DEMO_METHODS:
        raw_chunks = _load_method_chunks(session_dir, m)
        flat = [c for chunk in raw_chunks for c in chunk]
        aligned = _chunk_captions_by_gt(
            flat, gt_chunks, m, scio_data, screenshot_timestamps, ffmpeg_start,
        )
        method_data[m] = aligned
        print(f"  {m}: {sum(len(c) for c in aligned)} captions aligned to {len(aligned)} GT chunks", flush=True)

    all_run_scores: Dict[str, List[float]] = defaultdict(list)

    total_pairs = num_runs * num_gt_chunks * len(DEMO_METHODS)
    done_pairs = 0
    print(f"[judge] {num_runs} run(s) × {num_gt_chunks} chunk(s) × {len(DEMO_METHODS)} methods = {total_pairs} total pairs (workers={num_workers})", flush=True)

    # Thread-safe counter for progress
    import threading
    _progress_lock = threading.Lock()

    for run in range(num_runs):
        print(f"\n{'='*60}", flush=True)
        print(f"=== Run {run + 1}/{num_runs} ===", flush=True)
        print(f"{'='*60}", flush=True)
        results_file = str(output_dir / f"run_{run}.json")
        evaluator = GeminiEvaluator(
            api_key=api_key,
            methods=DEMO_METHODS,
            prompt_file=prompt_file,
            results_file=results_file,
        )
        run_scores: Dict[str, List[float]] = defaultdict(list)

        # Build list of tasks for this run
        tasks = []
        for chunk_idx, gt_chunk in enumerate(gt_chunks):
            gt_text = evaluator.format_gt_chunk(gt_chunk)
            for method in DEMO_METHODS:
                tasks.append((chunk_idx, method, gt_text))

        def _eval_task(task_info):
            """Evaluate a single (chunk, method) pair. Returns (chunk_idx, method, result)."""
            nonlocal done_pairs
            chunk_idx, method, gt_text = task_info

            if evaluator.is_pair_evaluated(chunk_idx, method, run):
                # Re-collect score from existing results (resume)
                score_found = None
                for ev in evaluator.results["evaluations"]:
                    if (ev.get("chunk_idx") == chunk_idx and ev.get("method") == method
                            and ev.get("run") == run and ev.get("evaluation")):
                        score_found = ev["evaluation"]["score"]
                with _progress_lock:
                    done_pairs += 1
                    dp = done_pairs
                print(f"[judge] __progress__ {dp} {total_pairs} Run {run+1}/{num_runs} | chunk {chunk_idx+1}/{num_gt_chunks} | {method} (cached)", flush=True)
                return (chunk_idx, method, score_found, None)

            if chunk_idx >= len(method_data.get(method, [])):
                with _progress_lock:
                    done_pairs += 1
                    dp = done_pairs
                print(f"[judge] __progress__ {dp} {total_pairs} Run {run+1}/{num_runs} | chunk {chunk_idx+1}/{num_gt_chunks} | {method} (skipped)", flush=True)
                return (chunk_idx, method, None, "no captions")

            candidate = method_data[method][chunk_idx]
            result = evaluator.evaluate_pair(chunk_idx, method, gt_text, candidate, run)
            evaluator.add_evaluation(chunk_idx, method, result, run)

            with _progress_lock:
                done_pairs += 1
                dp = done_pairs

            if result.get("evaluation"):
                score = result["evaluation"].get("score", 0.0)
                rationale = result["evaluation"].get("rationale", "")[:100]
                print(f"  → {method} chunk {chunk_idx+1}: score={score:.3f} | {rationale}", flush=True)
                print(f"[judge] __progress__ {dp} {total_pairs} Run {run+1}/{num_runs} | chunk {chunk_idx+1}/{num_gt_chunks} | {method}: {score:.3f}", flush=True)
                return (chunk_idx, method, score, None)
            else:
                err = result.get("error", "unknown error")
                print(f"  → {method} chunk {chunk_idx+1}: ERROR – {err}", flush=True)
                print(f"[judge] __progress__ {dp} {total_pairs} Run {run+1}/{num_runs} | chunk {chunk_idx+1}/{num_gt_chunks} | {method}: error", flush=True)
                return (chunk_idx, method, None, err)

        # Execute tasks with ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = {executor.submit(_eval_task, t): t for t in tasks}
            for future in as_completed(futures):
                chunk_idx, method, score, err = future.result()
                if score is not None:
                    all_run_scores[method].append(score)
                    run_scores[method].append(score)

        # Per-run summary
        print(f"\n--- Run {run+1}/{num_runs} summary ---", flush=True)
        for m in DEMO_METHODS:
            s = run_scores[m]
            if s:
                print(f"  {m:30s}: mean={np.mean(s):.3f}  (n={len(s)})", flush=True)
            else:
                print(f"  {m:30s}: no scores this run", flush=True)

    # Compute summary
    summary: Dict[str, Dict] = {}
    for method in DEMO_METHODS:
        s = all_run_scores[method]
        summary[method] = {
            "mean": float(np.mean(s)) if s else 0.0,
            "bootstrap_se": bootstrap_se(s, n_bootstrap),
            "n": len(s),
            "all_scores": s,
        }

    summary_path = output_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump({"methods": summary, "num_gt_chunks": num_gt_chunks, "num_runs": num_runs}, f, indent=2)
    print(f"\nSummary saved to {summary_path}")
    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="NAPsack LLM judge for demo pipeline")
    parser.add_argument("--session-dir", help="Demo session directory (enables demo mode)")
    parser.add_argument("--output-dir", help="Output directory for judge results (default: <session-dir>/judge)")
    parser.add_argument("--runs", type=int, default=3, help="Number of independent evaluation runs (default: 3)")
    parser.add_argument("--num-workers", type=int, default=4, help="Number of parallel Gemini judge workers (default: 4)")
    parser.add_argument("--prompt-file", default=None, help="Path to judge prompt template file")
    parser.add_argument("--n-bootstrap", type=int, default=1000, help="Bootstrap resamples for SE (default: 1000)")
    # Legacy args
    parser.add_argument("--num-gt-chunks", type=int, default=3, help="(legacy) Number of GT chunks")
    args = parser.parse_args()

    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        print("Error: Please set GEMINI_API_KEY environment variable")
        return

    if args.session_dir:
        # ---- Demo mode ----
        session_dir = Path(args.session_dir)
        gt_path = session_dir / "gt" / "gt_captions.jsonl"
        if not gt_path.exists():
            print(f"Error: GT captions not found at {gt_path}")
            print("Please run GT annotation (Step 3) first.")
            return
        out_dir = Path(args.output_dir) if args.output_dir else session_dir / "judge"
        summary = run_demo_judge(
            session_dir=session_dir,
            gt_captions_path=gt_path,
            output_dir=out_dir,
            num_runs=args.runs,
            num_workers=args.num_workers,
            prompt_file=args.prompt_file,
            n_bootstrap=args.n_bootstrap,
        )
        print("\n=== RESULTS ===")
        for m, s in summary.items():
            print(f"  {m:30s}: {s['mean']:.3f} ± {s['bootstrap_se']:.3f}  (n={s['n']})")
    else:
        # ---- Legacy standalone mode ----
        print("Running in legacy standalone mode (no --session-dir given).")
        loader = CaptionLoader(num_gt_chunks=args.num_gt_chunks)
        gt_chunks = loader.load_ground_truth_txt()
        data = {
            "pack_io_fused_2": loader.load_video_only("fused_2"),
        }
        evaluator = GeminiEvaluator(
            api_key=api_key,
            methods=list(data.keys()),
            prompt_file=args.prompt_file or "verifier.txt",
        )
        for chunk_idx, gt in enumerate(gt_chunks[:args.num_gt_chunks]):
            for method in evaluator.methods:
                if chunk_idx >= len(data.get(method, [])):
                    continue
                result = evaluator.evaluate_pair(chunk_idx, method, gt, data[method][chunk_idx])
                evaluator.add_evaluation(chunk_idx, method, result)
        stats = evaluator.get_statistics()
        for method, s in sorted(stats.items(), key=lambda x: -x[1]["mean_score"]):
            print(f"  {method}: {s['mean_score']:.3f} ({s['num_evaluations']} evals)")


if __name__ == "__main__":
    main()
