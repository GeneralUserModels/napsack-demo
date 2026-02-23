import json
import os
import random
from pathlib import Path
from typing import List, Dict, Tuple
from collections import defaultdict
import sys
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

    def load_flash_10m(self) -> List[List[Dict]]:
        """Load 10-minute chunks from chunks_10m."""
        chunks = []
        for i in range(6):
            chunk_file = self.base_path / "chunks_10m" / f"chunk_{str(i).zfill(3)}" / "captions.jsonl"
            chunks.append(self._load_jsonl(chunk_file))
        return chunks

    def load_split_1m(self) -> List[List[Dict]]:
        """Load 1-minute chunks and group into 10-minute chunks."""
        all_captions = []
        for i in range(6):
            chunk_file = self.base_path / "chunks_1m" / f"chunk_{str(i).zfill(3)}" / "captions.jsonl"
            all_captions.append(self._load_jsonl(chunk_file))
        return all_captions

    def load_pack_no_key(self) -> List[List[Dict]]:
        """Load compression without key."""
        chunks = []
        for i in range(6):
            chunk_file = self.base_path / "pack_no_key" / f"chunk_{str(i).zfill(3)}" / "captions.jsonl"
            chunks.append(self._load_jsonl(chunk_file))
        return chunks

    def load_pack_key(self) -> List[List[Dict]]:
        """Load compression with key and chunk into 10-minute segments."""
        all_captions = []
        for i in range(3):
            chunk_file = self.base_path / "fused_3" / f"chunk_{str(i).zfill(3)}" / "captions.jsonl"
            all_captions.append(self._load_jsonl(chunk_file))
        return all_captions
        # data_file = self.base_path / "fised_old_new" / "data.jsonl"
        # all_entries = self._load_jsonl(data_file)
        #
        # if not all_entries:
        #     return []
        #
        # # Chunk by 10-minute intervals
        # chunks = []
        # current_chunk = []
        # start_time = all_entries[0]["start_time"]
        # chunk_duration = 600  # 10 minutes in seconds
        #
        # for entry in all_entries:
        #     if entry["start_time"] - start_time < chunk_duration:
        #         current_chunk.append(entry)
        #     else:
        #         if current_chunk:
        #             chunks.append(current_chunk)
        #         current_chunk = [entry]
        #         start_time = entry["start_time"]
        #
        # if current_chunk:
        #     chunks.append(current_chunk)
        #
        # return chunks

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

    def __init__(self, api_key: str, results_file: str = "gemini_evaluation_results.json"):
        self.results_file = results_file
        self.methods = ["flash_10m", "split_1m", "pack_no_key", "pack_key"]

        # Configure Gemini API
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-3-flash-preview')

        self.results = self._load_results()
        self.evaluation_prompt_template = self._load_prompt_template()

    def _load_prompt_template(self) -> str:
        """Load the evaluation prompt template."""
        return """You are an expert evaluator.

Your task is to compare a **ground truth sequence of actions** with several **candidate predicted sequences of actions**, and assign a **quality score** to each candidate.

Each action is a timestamped event describing what the user did on a computer.

Your goal is to evaluate **how closely each candidate reproduces the same user trajectory** as the ground truth.

---

# Evaluation Dimensions

Evaluate candidates holistically using the following dimensions.

### 1. Action Accuracy
- Do predicted actions correspond to real actions in the ground truth?
- Are there hallucinated actions not present in the ground truth?
- Are important ground truth actions missing?

Minor extra or missing actions may be acceptable if the overall trajectory is preserved.

---

### 2. Order & Temporal Consistency
- Are actions in the correct order relative to the ground truth?
- Small local reordering is acceptable if it does not change task meaning.
- Major reordering that changes the task flow should be penalized.

---

### 3. Semantic Fidelity
- Even if wording differs, does the candidate describe the same intent as the ground truth?
- Clear paraphrases count as matches.
- Actions describing different user intent should be penalized.

---

### 4. Specificity / Entity Fidelity
- When the ground truth includes specific identifying information (e.g., exact post title, filename, email subject, UI label, query text, numeric value), the candidate should preserve that specificity via exact match or close paraphrase.
- Loss of specific identifiers should be treated as:
  - **Minor penalty** if the intended action is still unambiguous.
  - **Major penalty** only if the missing specificity makes it unclear which ground truth action is being referenced.
- If the ground truth itself is generic, do not penalize candidates for being generic.

Examples:
- Ground truth: "Clicked on article titled `Fed Signals Interest Rate Cuts Could Begin in 2026`."
  - Good: "Clicked the news story about the Fed signaling rate cuts starting in 2026."
  - Poor: "Clicked on a post."

---

### 5. Verbosity & Redundancy
- Actions should be concise and factual.
- Verbosity, explanatory phrasing, or unnecessary repetition should result in **penalty**.
- Hedging and listing of multiple possibilities (e.g. clicked or scrolled or ...) should also result in **penalty**.
- Do not heavily penalize verbosity if the actions are otherwise correct.

---

# Scoring Guidelines

Assign a **single score between 0.0 and 1.0** to each candidate.

Use **continuous values** (up to two decimal places), and distribute scores meaningfully across candidates.

### Score Calibration

- **0.90 – 1.00**  
  Near-perfect or perfect match.  
  The candidate closely reproduces the ground truth trajectory. Minor wording differences or small omissions are acceptable.

- **0.75 – 0.89**  
  Largely correct trajectory.  
  Most actions match in intent and order, with minor missing details, mild genericness, or small ordering issues.

- **0.50 – 0.74**  
  Partial match.  
  Key actions are correct, but multiple missing steps, incorrect ordering, or loss of important specificity.

- **0.25 – 0.49**  
  Weak match.  
  Minimal overlap with the ground truth; many hallucinated or incorrect actions.

- **0.00 – 0.24**  
  No meaningful overlap.  
  The candidate does not resemble the ground truth trajectory.

---

# Important Rules

- Always **rank candidates relative to each other**.
- Do NOT assign identical scores unless candidates are truly equivalent.
- If one candidate is even slightly better than another, their scores must reflect that difference.
- Avoid over-penalizing minor flaws: assess the **overall severity** of errors rather than subtracting for every issue.
- Hallucinated actions and missing core steps should lower the score proportionally to how much they distort the trajectory.

---

# Input Format

## Ground Truth
{ground_truth}

## Candidates
{candidates}

---

# Output Format

Return a JSON object with scores for each candidate:

```json
{{
  "reasoning": "Brief justification summarizing why each candidate received its score.",
  "candidates": [
    {{ "id": "Candidate 1", "score": }},
    {{ "id": "Candidate 2", "score": }},
    {{ "id": "Candidate 3", "score": }},
    {{ "id": "Candidate 4", "score": }}
  ]
}}
```
"""

    def _load_results(self) -> Dict:
        """Load existing results if available."""
        if os.path.exists(self.results_file):
            with open(self.results_file, 'r') as f:
                return json.load(f)
        return {"evaluations": [], "completed_chunks": set()}

    def _save_results(self):
        """Save results to file."""
        save_data = {
            "evaluations": self.results["evaluations"],
            "completed_chunks": list(self.results.get("completed_chunks", set()))
        }
        with open(self.results_file, 'w') as f:
            json.dump(save_data, f, indent=2)

    def is_chunk_evaluated(self, chunk_idx: int) -> bool:
        """Check if a chunk has already been evaluated."""
        completed = set(self.results.get("completed_chunks", []))
        return chunk_idx in completed

    def format_captions_for_eval(self, captions: List[Dict]) -> str:
        """Format captions for evaluation."""
        lines = []
        for entry in captions:
            if "caption" in entry:
                # Remove any existing <action> tags and add plain text
                caption_text = entry['caption'].replace('<action>', '').replace('</action>', '')
                lines.append(caption_text)
        return "\n".join(lines)

    def evaluate_chunk(self, chunk_idx: int, ground_truth: str, candidates: Dict[str, List[Dict]]) -> Dict:
        """Evaluate all candidates for a chunk against ground truth using Gemini."""

        # Create shuffled list of methods to blind the evaluation
        method_order = self.methods.copy()
        random.shuffle(method_order)

        # Create mapping from blind ID to actual method
        blind_to_method = {f"Candidate {i + 1}": method for i, method in enumerate(method_order)}
        method_to_blind = {method: f"Candidate {i + 1}" for i, method in enumerate(method_order)}

        # Format candidates with blind IDs
        candidates_text = ""
        for i, method in enumerate(method_order):
            candidate_text = self.format_captions_for_eval(candidates[method])
            candidates_text += f"\n## Candidate {i + 1}\n{candidate_text}\n"

        # Create prompt
        prompt = self.evaluation_prompt_template.format(
            ground_truth=ground_truth,
            candidates=candidates_text
        )

        # Call Gemini API
        try:
            response = self.model.generate_content(prompt)
            response_text = response.text

            # Parse JSON response
            # Remove markdown code blocks if present
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text:
                response_text = response_text.split("```")[1].split("```")[0].strip()

            evaluation = json.loads(response_text)

            # Map blind IDs back to actual method names
            unblinded_candidates = []
            for candidate in evaluation.get("candidates", []):
                blind_id = candidate.get("id")
                if blind_id in blind_to_method:
                    unblinded_candidates.append({
                        "id": blind_to_method[blind_id],
                        "score": candidate.get("score", 0.0),
                        "blind_id": blind_id
                    })
                else:
                    print(f"Warning: Unknown candidate ID {blind_id}")

            evaluation["candidates"] = unblinded_candidates

            return {
                "chunk_idx": chunk_idx,
                "evaluation": evaluation,
                "raw_response": response.text,
                "blind_mapping": blind_to_method
            }

        except Exception as e:
            print(f"Error evaluating chunk {chunk_idx}: {e}")
            return {
                "chunk_idx": chunk_idx,
                "error": str(e),
                "evaluation": None
            }

    def add_evaluation(self, chunk_idx: int, evaluation_result: Dict):
        """Add an evaluation result."""
        self.results["evaluations"].append(evaluation_result)

        # Mark as completed
        if "completed_chunks" not in self.results:
            self.results["completed_chunks"] = set()
        else:
            self.results["completed_chunks"] = set(self.results["completed_chunks"])

        self.results["completed_chunks"].add(chunk_idx)
        self._save_results()

    def get_statistics(self) -> Dict:
        """Calculate aggregate statistics across all evaluations."""
        scores = defaultdict(list)

        for eval_result in self.results["evaluations"]:
            if eval_result.get("evaluation") and "candidates" in eval_result["evaluation"]:
                for candidate in eval_result["evaluation"]["candidates"]:
                    method_id = candidate.get("id")
                    score = candidate.get("score", 0.0)
                    scores[method_id].append(score)

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

    print("Loading caption data...")
    loader = CaptionLoader(num_gt_chunks=NUM_GT_CHUNKS)

    # Load ground truth
    ground_truth_chunks = loader.load_ground_truth()
    print(f"Loaded {len(ground_truth_chunks)} ground truth chunks")

    # Load all candidate methods
    data = {
        "flash_10m": loader.load_flash_10m(),
        "split_1m": loader.load_split_1m(),
        "pack_no_key": loader.load_pack_no_key(),
        "pack_key": loader.load_pack_key()
    }

    for method_name, chunks in data.items():
        total_captions = sum([len(chunk) for chunk in chunks])
        print(f"Loaded {len(chunks)} chunks with {total_captions} total captions for method '{method_name}'")

    # Initialize evaluator
    evaluator = GeminiEvaluator(api_key=GEMINI_API_KEY)

    # Evaluate each GT chunk
    num_chunks_to_evaluate = min(NUM_GT_CHUNKS, len(ground_truth_chunks))

    print(f"\nStarting evaluation of {num_chunks_to_evaluate} chunks...")

    for chunk_idx in range(num_chunks_to_evaluate):
        if evaluator.is_chunk_evaluated(chunk_idx):
            print(f"Chunk {chunk_idx} already evaluated, skipping...")
            continue

        print(f"\nEvaluating chunk {chunk_idx}...")

        # Get ground truth
        gt = ground_truth_chunks[chunk_idx]

        # Get candidates (ensure all methods have this chunk)
        candidates = {}
        all_have_chunk = True
        for method in evaluator.methods:
            if chunk_idx < len(data[method]):
                candidates[method] = data[method][chunk_idx]
            else:
                print(f"Warning: Method {method} does not have chunk {chunk_idx}")
                all_have_chunk = False
                break

        if not all_have_chunk:
            print(f"Skipping chunk {chunk_idx} - not all methods have this chunk")
            continue

        # Evaluate
        result = evaluator.evaluate_chunk(chunk_idx, gt, candidates)

        if result.get("evaluation"):
            print(f"Successfully evaluated chunk {chunk_idx}")
            print(f"Blind mapping: {result.get('blind_mapping', {})}")
            print(f"Scores: {result['evaluation'].get('candidates', [])}")
        else:
            print(f"Failed to evaluate chunk {chunk_idx}")

        evaluator.add_evaluation(chunk_idx, result)

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
