"""
single_judge.py – pairwise LLM evaluation for the NAPsack demo pipeline.

Demo-mode CLI (--session-dir):
  Evaluates all four methods (naive, split, split_compress, split_compress_io)
  against GT captions that are grouped in chunks of 8 captions.
  Runs --runs times with independent Gemini calls, computes per-method
  mean ± bootstrap SE, and saves to <session_dir>/judge/summary.json.

Legacy standalone mode: same as before (no --session-dir).
"""

import json
import os
import random
import argparse
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from collections import defaultdict
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

    def load_pack(self, name: str = "pack") -> List[List[Dict]]:
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
        self.results_file = results_file
        self.methods = methods

        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel("gemini-2.0-flash")

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
    """Load captions from the appropriate sub-directory for each method."""
    loader = CaptionLoader(base_path=str(session_dir))
    if method == "naive":
        return loader.load_from_chunk_dirs(str(session_dir / "naive"))
    elif method == "split":
        return loader.load_from_chunk_dirs(str(session_dir / "split"))
    elif method == "split_compress":
        # label writes chunks into pack_session; find it via state.json
        state_file = session_dir / "state.json"
        if state_file.exists():
            with open(state_file) as f:
                state = json.load(f)
            sc_dir = (state.get("processing", {}).get("output_dirs", {}) or {}).get("split_compress")
            if sc_dir:
                return loader.load_from_chunk_dirs(sc_dir)
        return []
    elif method == "split_compress_io":
        fused_dir = session_dir / "split_compress_io" / "fused"
        return loader.load_from_chunk_dirs(str(fused_dir))
    return []


def _chunk_captions_by_gt(method_chunks_flat: List[Dict], gt_chunks: List[List[Dict]]) -> List[List[Dict]]:
    """
    Re-group flat method captions to align with GT 8-caption time windows.
    Each GT chunk covers [gt[0].start_time, gt[-1].end_time].
    Method captions whose start_time falls in that window are assigned to it.
    """
    result: List[List[Dict]] = []
    for gt_chunk in gt_chunks:
        # Get time window from GT
        starts = [c.get("start_time", c.get("start", 0)) for c in gt_chunk]
        ends = [c.get("end_time", c.get("end", 0)) for c in gt_chunk]
        # Convert MM:SS strings if needed
        def _to_sec(v):
            if isinstance(v, str) and ":" in v:
                p = v.split(":")
                return int(p[0]) * 60 + int(p[1])
            return float(v) if v else 0.0
        t_start = min(_to_sec(s) for s in starts)
        t_end = max(_to_sec(e) for e in ends)
        window = [
            c for c in method_chunks_flat
            if _to_sec(c.get("start_time", c.get("start", 0))) >= t_start
            and _to_sec(c.get("start_time", c.get("start", 0))) <= t_end
        ]
        result.append(window)
    return result


def run_demo_judge(
    session_dir: Path,
    gt_captions_path: Path,
    output_dir: Path,
    num_runs: int = 3,
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
    print(f"GT: {num_gt_chunks} chunks of 8 captions each")

    # Load method captions and align to GT windows
    method_data: Dict[str, List[List[Dict]]] = {}
    for m in DEMO_METHODS:
        raw_chunks = _load_method_chunks(session_dir, m)
        # Flatten all method chunks to re-align by GT timestamps
        flat = [c for chunk in raw_chunks for c in chunk]
        aligned = _chunk_captions_by_gt(flat, gt_chunks)
        method_data[m] = aligned
        print(f"  {m}: {sum(len(c) for c in aligned)} captions aligned to {len(aligned)} GT chunks")

    all_run_scores: Dict[str, List[float]] = defaultdict(list)

    for run in range(num_runs):
        print(f"\n=== Run {run + 1}/{num_runs} ===")
        results_file = str(output_dir / f"run_{run}.json")
        evaluator = GeminiEvaluator(
            api_key=api_key,
            methods=DEMO_METHODS,
            prompt_file=prompt_file,
            results_file=results_file,
        )

        for chunk_idx, gt_chunk in enumerate(gt_chunks):
            gt_text = evaluator.format_gt_chunk(gt_chunk)
            for method in DEMO_METHODS:
                if evaluator.is_pair_evaluated(chunk_idx, method, run):
                    # Re-collect score from existing results
                    for ev in evaluator.results["evaluations"]:
                        if (ev.get("chunk_idx") == chunk_idx and ev.get("method") == method
                                and ev.get("run") == run and ev.get("evaluation")):
                            all_run_scores[method].append(ev["evaluation"]["score"])
                    continue
                if chunk_idx >= len(method_data.get(method, [])):
                    print(f"  Skip {method} chunk {chunk_idx} (not available)")
                    continue
                candidate = method_data[method][chunk_idx]
                result = evaluator.evaluate_pair(chunk_idx, method, gt_text, candidate, run)
                evaluator.add_evaluation(chunk_idx, method, result, run)
                if result.get("evaluation"):
                    score = result["evaluation"].get("score", 0.0)
                    all_run_scores[method].append(score)
                    print(f"  [run{run}] {method} chunk {chunk_idx}: {score:.3f}")

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
