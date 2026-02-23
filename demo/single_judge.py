import json
import os
from pathlib import Path
from typing import List, Dict
from collections import defaultdict
import google.generativeai as genai


class CaptionLoader:
    """Load and chunk caption data from different methods."""

    def __init__(self, base_path: str = ".", num_gt_chunks: int = 3):
        self.base_path = Path(base_path)
        self.num_gt_chunks = num_gt_chunks

    def load_ground_truth(self) -> List[str]:
        """Load ground truth completions from gt/gt_N/.txt files."""
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

    def load_video_only(self, name) -> List[List[Dict]]:
        """Load compression without key."""
        chunks = []
        for i in range(6):
            chunk_file = self.base_path / name / f"chunk_{str(i).zfill(3)}" / "captions.jsonl"
            chunks.append(self._load_jsonl(chunk_file))
        return chunks

    def load_pack(self, name="pack") -> List[List[Dict]]:
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

    def __init__(self, api_key: str, prompt_file: str = "verifier.txt",
                 results_file: str = "gemini_evaluation_results.json"):
        self.results_file = results_file
        self.methods = ["flash_10m", "split_1m", "pack_no_key", "pack_key"]

        # Configure Gemini API
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-3-flash-preview')

        self.results = self._load_results()
        self.evaluation_prompt_template = self._load_prompt_from_file(prompt_file)

    def _load_prompt_from_file(self, prompt_file: str) -> str:
        """Load the evaluation prompt template from a text file."""
        prompt_path = Path(prompt_file)
        if not prompt_path.exists():
            raise FileNotFoundError(f"Prompt file not found: {prompt_file}")

        with open(prompt_path, 'r') as f:
            return f.read()

    def _load_results(self) -> Dict:
        """Load existing results if available."""
        return {"evaluations": [], "completed_pairs": set()}

    def _save_results(self):
        """Save results to file."""
        save_data = {
            "evaluations": self.results["evaluations"],
            "completed_pairs": list(self.results.get("completed_pairs", set()))
        }
        with open(self.results_file, 'w') as f:
            json.dump(save_data, f, indent=2)

    def is_pair_evaluated(self, chunk_idx: int, method: str) -> bool:
        """Check if a chunk-method pair has already been evaluated."""
        completed = set(self.results.get("completed_pairs", []))
        pair_id = f"{chunk_idx}_{method}"
        return pair_id in completed

    def format_captions_for_eval(self, captions: List[Dict]) -> str:
        """Format captions for evaluation."""
        lines = []
        for entry in captions:
            if "caption" in entry:
                # Remove any existing <action> tags and add plain text
                caption_text = entry['caption'].replace('<action>', '').replace('</action>', '')
                lines.append(caption_text)
        return "\n".join(lines)

    def evaluate_pair(self, chunk_idx: int, method: str, ground_truth: str,
                      candidate: List[Dict]) -> Dict:
        """Evaluate a single candidate against ground truth using Gemini."""

        # Format candidate
        candidate_text = self.format_captions_for_eval(candidate)

        # Create prompt by replacing placeholders
        prompt = self.evaluation_prompt_template.replace("{GT}", ground_truth)
        prompt = prompt.replace("{CANDIDATE}", candidate_text)

        # Call Gemini API
        try:
            print(prompt)
            response = self.model.generate_content(prompt)
            response_text = response.text

            # Parse JSON response
            # Remove markdown code blocks if present
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text:
                response_text = response_text.split("```")[1].split("```")[0].strip()

            evaluation = json.loads(response_text)

            return {
                "chunk_idx": chunk_idx,
                "method": method,
                "evaluation": evaluation,
                "raw_response": response.text
            }

        except Exception as e:
            print(f"Error evaluating chunk {chunk_idx}, method {method}: {e}")
            return {
                "chunk_idx": chunk_idx,
                "method": method,
                "error": str(e),
                "evaluation": None
            }

    def add_evaluation(self, chunk_idx: int, method: str, evaluation_result: Dict):
        """Add an evaluation result."""
        self.results["evaluations"].append(evaluation_result)

        # Mark as completed
        if "completed_pairs" not in self.results:
            self.results["completed_pairs"] = set()
        else:
            self.results["completed_pairs"] = set(self.results["completed_pairs"])

        pair_id = f"{chunk_idx}_{method}"
        self.results["completed_pairs"].add(pair_id)
        self._save_results()

    def get_statistics(self) -> Dict:
        """Calculate aggregate statistics across all evaluations."""
        scores = defaultdict(list)

        for eval_result in self.results["evaluations"]:
            if eval_result.get("evaluation") and "score" in eval_result["evaluation"]:
                method = eval_result.get("method")
                score = eval_result["evaluation"].get("score", 0.0)
                scores[method].append(score)

        statistics = {}
        for method in self.methods:
            if method in scores and len(scores[method]) > 0:
                statistics[method] = {
                    "mean_score": sum(scores[method]) / len(scores[method]),
                    "min_score": min(scores[method]),
                    "max_score": max(scores[method]),
                    "num_evaluations": len(scores[method]),
                    "all_scores": scores[method]
                }
            else:
                statistics[method] = {
                    "mean_score": 0.0,
                    "min_score": 0.0,
                    "max_score": 0.0,
                    "num_evaluations": 0,
                    "all_scores": []
                }

        return statistics


def main():
    # Configuration
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    if not GEMINI_API_KEY:
        print("Error: Please set GEMINI_API_KEY environment variable")
        return

    NUM_GT_CHUNKS = 3  # Configurable number of GT chunks
    PROMPT_FILE = "verifier.txt"  # Path to prompt template file
    # PROMPT_FILE = "coverage.txt"  # Path to prompt template file
    # PROMPT_FILE = "correctness.txt"  # Path to prompt template file

    print("Loading caption data...")
    loader = CaptionLoader(num_gt_chunks=NUM_GT_CHUNKS)

    # Load ground truth
    ground_truth_chunks = loader.load_ground_truth()
    print(f"Loaded {len(ground_truth_chunks)} ground truth chunks")

    # Load all candidate methods
    data = {
        # "flash_10m": loader.load_video_only("chunks_10m"),
        # "split_1m": loader.load_video_only("chunks_1m"),
        # "pack_no_key": loader.load_video_only("pack_no_key"),
        # "pack_no_key": loader.load_video_only("pack_no_key_dedup"),
        # "pack_io": loader.load_pack("pack_io"),
        "pack_io_fused_2": loader.load_video_only("fused_2"),
        # "pack_io_dedup": loader.load_pack("pack_io_dedup"),
        # "fused": loader.load_video_only("fused"),
        # "fused_dedup": loader.load_video_only("fused_dedup"),
        # "pack_key_new": loader.load_pack("pack"),
        # "pack_key_only_new": loader.load_pack("real_pack_new_no_anno"),
        # "pack_only_annotations": loader.load_pack("pack_only_annotations"),
        # "pack_key_no_annotations": loader.load_pack("pack_only_annotations"),
        # "pack_key_no_annotations_fixed": loader.load_pack("pack_key_no_annotations_fixed"),
        # "real_pack_new_key_only": loader.load_pack("real_pack_new_key_only"),
    }

    for method_name, chunks in data.items():
        total_captions = sum([len(chunk) for chunk in chunks])
        print(f"Loaded {len(chunks)} chunks with {total_captions} total captions for method '{method_name}'")

    # Initialize evaluator
    evaluator = GeminiEvaluator(api_key=GEMINI_API_KEY, prompt_file=PROMPT_FILE)
    evaluator.methods = list(data.keys())

    # Evaluate each GT chunk against each method
    num_chunks_to_evaluate = min(NUM_GT_CHUNKS, len(ground_truth_chunks))

    print(f"\nStarting pairwise evaluation of {num_chunks_to_evaluate} chunks...")

    for chunk_idx in range(num_chunks_to_evaluate):
        print(f"\n--- Evaluating chunk {chunk_idx} ---")

        # Get ground truth
        gt = ground_truth_chunks[chunk_idx]

        # Evaluate against each method
        for method in evaluator.methods:
            pair_id = f"{chunk_idx}_{method}"

            # if evaluator.is_pair_evaluated(chunk_idx, method):
            #     print(f"  Pair {pair_id} already evaluated, skipping...")
            #     continue

            # Check if method has this chunk
            if chunk_idx >= len(data[method]):
                print(f"  Warning: Method {method} does not have chunk {chunk_idx}, skipping...")
                continue

            print(f"  Evaluating against {method}...")

            # Get candidate
            candidate = data[method][chunk_idx]

            # Evaluate
            result = evaluator.evaluate_pair(chunk_idx, method, gt, candidate)

            if result.get("evaluation"):
                score = result['evaluation'].get('score', 'N/A')
                print(f"    ✓ Score: {score}")
            else:
                print(f"    ✗ Failed to evaluate")

            evaluator.add_evaluation(chunk_idx, method, result)

    # Final statistics
    print("\n" + "=" * 80)
    print("EVALUATION COMPLETED!")
    print("=" * 80 + "\n")

    stats = evaluator.get_statistics()

    print("=== AGGREGATE SCORES ===")
    sorted_methods = sorted(stats.items(), key=lambda x: x[1]["mean_score"], reverse=True)

    for method, method_stats in sorted_methods:
        print(f"\n{method}:")
        print(f"  Mean Score: {method_stats['mean_score']:.3f}")
        print(f"  Min Score:  {method_stats['min_score']:.3f}")
        print(f"  Max Score:  {method_stats['max_score']:.3f}")
        print(f"  Evaluations: {method_stats['num_evaluations']}")

    print(f"\nDetailed results saved to {evaluator.results_file}")


if __name__ == "__main__":
    main()
