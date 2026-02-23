from google import genai
from google.genai import types
import json
from pathlib import Path
from typing import List, Dict, Optional
import time
from dataclasses import dataclass


from typing import Optional, Any, Dict
import json
import os
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, List, Union

CAPTION_SCHEMA = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "start": {"type": "string"},
            "end": {"type": "string"},
            "caption": {"type": "string"}
        },
        "required": ["start", "end", "caption"]
    }
}


class VLMClient(ABC):
    @abstractmethod
    def upload_file(self, path: str) -> Any:
        pass

    @abstractmethod
    def generate(self, prompt: Union[str, List[str]],
                 file_descriptor: Optional[Union[Any, List[Any]]] = None,
                 schema: Optional[Dict] = None) -> Union[Any, List[Any]]:
        pass


class GeminiResponse:
    def __init__(self, response):
        self.response = response
        self._json = None

    @property
    def text(self) -> str:
        return self.response.text

    @property
    def json(self):
        if self._json is None:
            self._json = json.loads(self.text)
        return self._json


class GeminiClient(VLMClient):
    def __init__(self, api_key: Optional[str] = None, model_name: str = "gemini-2.5-flash"):
        if genai is None:
            raise RuntimeError("google-genai not installed")

        api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY not set")

        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name

    def upload_file(self, path: str) -> Any:
        video_file = self.client.files.upload(file=path)

        while True:
            video_file = self.client.files.get(name=video_file.name)
            state = getattr(getattr(video_file, "state", None), "name", None)

            if state == "PROCESSING":
                time.sleep(2)
            elif state == "FAILED":
                raise RuntimeError("Gemini failed processing file")
            elif state == "ACTIVE":
                break
            else:
                break

        return video_file

    def generate(self, prompt: str, file_descriptor: Optional[Any] = None,
                 schema: Optional[Dict] = None) -> GeminiResponse:
        inputs = []
        if file_descriptor:
            inputs.append(file_descriptor)
        inputs.append(prompt)

        if "gemini-3" in self.model_name:
            config = types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0.0,
                response_schema=schema or CAPTION_SCHEMA,
                thinking_config=types.ThinkingConfig(),
                media_resolution=types.MediaResolution.MEDIA_RESOLUTION_HIGH
            )
        else:
            config = types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0.0,
                response_schema=schema or CAPTION_SCHEMA,
            )

        res = self.client.models.generate_content(
            model=self.model_name,
            contents=inputs,
            config=config
        )

        response = GeminiResponse(res)
        return response

    def get_token_stats(self) -> Dict[str, int]:
        """Get current token usage statistics."""
        return {
            "input_tokens": self.total_input_tokens,
            "output_tokens": self.total_output_tokens,
            "total_tokens": self.total_input_tokens + self.total_output_tokens
        }


@dataclass
class CaptionLoader:
    """Handles loading captions from both candidate formats."""

    def load_video_only(self, name) -> List[List[Dict]]:
        """Load compression without key."""
        chunks = []
        for i in range(6):
            chunk_file = Path(name) / f"chunk_{str(i).zfill(3)}" / "captions.jsonl"
            chunks.append(self._load_jsonl(chunk_file))
        return chunks

    def load_from_chunks(self, name: str) -> List[List[Dict]]:
        """Load pack captions from chunk_NNN/captions.jsonl (label pipeline output format)."""
        base = Path(name)
        chunks = []
        for chunk_dir in sorted(base.glob("chunk_????")):
            cap_file = chunk_dir / "captions.jsonl"
            chunks.append(self._load_jsonl(cap_file))
        # Also try chunk_NNN (3-digit) naming
        if not chunks:
            for chunk_dir in sorted(base.glob("chunk_???")):
                cap_file = chunk_dir / "captions.jsonl"
                chunks.append(self._load_jsonl(cap_file))
        return chunks

    def load_pack(self, name="pack") -> List[List[Dict]]:
        """Load compression with key and chunk into 10-minute segments."""
        data_file = Path(name) / "data.jsonl"
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


def format_video_only_captions(captions: List[Dict]) -> str:
    """Format video-only captions for the prompt."""
    lines = []
    for cap in captions:
        lines.append(
            f"[{cap['start']} → {cap['end']}] {cap['caption']}"
        )
    return "\n".join(lines)


def format_pack_captions(captions: List[Dict], reference_timestamp: Optional[float] = None) -> str:
    """Format pack captions for the prompt with relative timestamps."""
    if not captions:
        return "No pack captions available."

    def sec_to_mmss(seconds: float) -> str:
        mm = int(seconds) // 60
        ss = int(seconds) % 60
        return f"{mm:02d}:{ss:02d}"

    lines = []
    fitst_cap_start = captions[0]['start_formatted']  # mm:ss
    first_cap_start_seconds = int(fitst_cap_start.split(":")[0]) * 60 + int(fitst_cap_start.split(":")[1])
    for cap in captions:
        cap_start_seconds = int(cap['start_formatted'].split(":")[0]) * 60 + int(cap['start_formatted'].split(":")[1])
        cap_end_seconds = int(cap['end_formatted'].split(":")[0]) * 60 + int(cap['end_formatted'].split(":")[1])
        lines.append(
            f"[{sec_to_mmss(cap_start_seconds - first_cap_start_seconds)} → {sec_to_mmss(cap_end_seconds - first_cap_start_seconds)}] {cap['caption']}"
        )

    return "\n".join(lines)


def create_fusion_prompt(video_only_captions: List[Dict], pack_captions: List[Dict]) -> str:
    video_only_text = format_video_only_captions(video_only_captions)
    pack_text = format_pack_captions(pack_captions)

    prompt = f"""You are a Technical Video Analyst. Your task is to generate a high-granularity timeline of computer interactions.

## Input Sources:
1. Visual Stream (Video-Only): Captions that only describe what is visible on the screen.
2. Input Stream (Pack/Telemetry): Captions that rather focus on integrating recorded keystroke and user input events.

---

## Your Core Logic: Causal Synthesis
Do not simply merge these lists. You must **map causes to effects**.

### 1. Verify Keystrokes via Visual Feedback
Only include a shortcut or command from the "Input Stream" if it logically explains the next visual change in the "Visual Stream."
* *Example:* If Input Stream says "cmd + up" and the Visual Stream says "Switched to the browser," synthesize this as: "Switched to the browser by pressing 'cmd + up'."
* *Sanity Check:* If a shortcut appears in the Input Stream but the screen shows an unrelated action (e.g., Input says `:q` but the file stays open), prioritize the Visual Stream and ignore the conflicting input.

### 2. Maintain Atomic Granularity
Every significant input or visual transition deserves its own caption.
* **Trigger:** The specific keypress or click (e.g., "Typed ':w'").
* **Context:** Where it happened (e.g., "in the Neovim command line").
* **Result:** What changed on screen (e.g., "to save the changes to 'grpo_trainer.py'").

### 3. Style Guidelines (Mirroring Ground Truth)
* **Formula:** "[Action] in [Application/Location] using [Input], which [Resulting State]."
* **Vim/CLI:** Be precise with Vim commands (e.g., "shift + v + g", ":w", "gg").
* **Navigation:** Describe workspace or tab switches explicitly with the keys used.
* **Tense:** Use Past Tense ("Clicked", "Navigated", "Pressed").

---

## Input Data:
### Visual Stream:
{video_only_text}

### Input Stream:
{pack_text}

---

## Output Format
Return a JSON array. Ensure timestamps align with the visual change.
```json
[
  {{
    "start": "MM:SS",
    "end": "MM:SS",
    "caption": "Detailed action description linking input to visual result"
  }}
]
Generate the logically synthesized timeline now:"""
    return prompt


def parse_time_to_seconds(time_str: str) -> int:
    """Convert MM:SS to seconds."""
    parts = time_str.split(":")
    return int(parts[0]) * 60 + int(parts[1])


def seconds_to_time_str(seconds: int) -> str:
    """Convert seconds to MM:SS format."""
    mm = seconds // 60
    ss = seconds % 60
    return f"{mm:02d}:{ss:02d}"


def add_derived_fields(caption: Dict, chunk_index: int) -> Dict:
    """Add start_seconds, end_seconds, and chunk_index fields."""
    caption["start_seconds"] = parse_time_to_seconds(caption["start"])
    caption["end_seconds"] = parse_time_to_seconds(caption["end"])
    caption["chunk_index"] = chunk_index
    return caption


def fuse_captions_for_chunk(
    video_only_chunk: List[Dict],
    pack_chunk: List[Dict],
    video_file_path: str,
    chunk_index: int,
    client: GeminiClient
) -> List[Dict]:
    """Fuse captions from both candidates for a single chunk using Gemini."""

    print(f"Processing chunk {chunk_index}...")
    print(f"  Video-only captions: {len(video_only_chunk)}")
    print(f"  Pack captions: {len(pack_chunk)}")

    # Upload video file
    print(f"  Uploading video file: {video_file_path}")
    video_descriptor = client.upload_file(video_file_path)

    # Create fusion prompt
    prompt = create_fusion_prompt(video_only_chunk, pack_chunk)

    # Generate fused captions
    print(f"  Generating fused captions...")
    response = client.generate(
        prompt=prompt,
        file_descriptor=video_descriptor,
        schema=CAPTION_SCHEMA
    )

    # Parse response
    fused_captions = response.json

    # Add derived fields
    fused_captions = [add_derived_fields(cap, chunk_index) for cap in fused_captions]

    print(f"  Generated {len(fused_captions)} fused captions")

    return fused_captions


def main(
    video_only_name: str,
    pack_name: str,
    video_dir: str,
    output_dir: str,
    api_key: Optional[str] = None
):
    """
    Main function to fuse captions from both candidates.

    Args:
        video_only_name: Name of the video-only candidate folder
        pack_name: Name of the pack candidate folder
        video_dir: Directory containing video chunk files (e.g., "chunk_000.mp4")
        output_dir: Directory to save fused captions
        api_key: Gemini API key (optional, can use GEMINI_API_KEY env var)
    """

    video_dir = Path(video_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize loader and client
    loader = CaptionLoader()
    client = GeminiClient(api_key=api_key, model_name="gemini-2.5-flash")

    # Load captions from both candidates
    print("Loading captions from both candidates...")
    video_only_chunks = loader.load_video_only(video_only_name)
    # Use chunk-directory format (label pipeline output) if available, else data.jsonl
    pack_path = Path(pack_name)
    has_chunks = any(True for _ in list(pack_path.glob("chunk_??*")))
    if has_chunks:
        print(f"  (pack-name has chunk dirs – using load_from_chunks)")
        pack_chunks = loader.load_from_chunks(pack_name)
    else:
        pack_chunks = loader.load_pack(pack_name)

    print(f"Loaded {len(video_only_chunks)} video-only chunks")
    print(f"Loaded {len(pack_chunks)} pack chunks")

    # Process each chunk
    num_chunks = len(video_only_chunks)
    all_fused_captions = []

    for i in range(num_chunks):
        video_only_chunk = video_only_chunks[i] if i < len(video_only_chunks) else []
        pack_chunk = pack_chunks[i] if i < len(pack_chunks) else []

        # Construct video file path
        video_file = video_dir / f"chunk_{str(i).zfill(3)}.mp4"

        if not video_file.exists():
            print(f"Warning: Video file not found: {video_file}")
            continue

        # Fuse captions for this chunk
        fused = fuse_captions_for_chunk(
            video_only_chunk=video_only_chunk,
            pack_chunk=pack_chunk,
            video_file_path=str(video_file),
            chunk_index=i,
            client=client
        )

        all_fused_captions.extend(fused)

        # Save chunk output
        chunk_output_file = output_dir / f"chunk_{str(i).zfill(3)}_fused.jsonl"
        with open(chunk_output_file, 'w') as f:
            for caption in fused:
                f.write(json.dumps(caption) + "\n")
        # if exists, save as output_dir/chunk_{str(i).zfill(3)}/captions.jsonl
        if (output_dir / f"chunk_{str(i).zfill(3)}").exists():
            chunk_output_file_alt = output_dir / f"chunk_{str(i).zfill(3)}" / "captions.jsonl"
            with open(chunk_output_file_alt, 'w') as f:
                for caption in fused:
                    f.write(json.dumps(caption) + "\n")
            print(f"  Also saved to: {chunk_output_file_alt}")

        print(f"  Saved to: {chunk_output_file}")

    # Save all fused captions
    all_output_file = output_dir / "all_fused_captions.jsonl"
    with open(all_output_file, 'w') as f:
        for caption in all_fused_captions:
            f.write(json.dumps(caption) + "\n")

    print(f"\n✓ Fusion complete!")
    print(f"  Total fused captions: {len(all_fused_captions)}")
    print(f"  Output directory: {output_dir}")
    print(f"  All captions saved to: {all_output_file}")

    # Print token usage if available
    try:
        stats = client.get_token_stats()
        print(f"\nToken usage:")
        print(f"  Input tokens: {stats['input_tokens']:,}")
        print(f"  Output tokens: {stats['output_tokens']:,}")
        print(f"  Total tokens: {stats['total_tokens']:,}")
    except:
        pass


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Fuse captions from video-only and pack candidates")
    parser.add_argument("--video-only-name", required=True, help="Name of video-only candidate folder")
    parser.add_argument("--pack-name", required=True, help="Name of pack candidate folder")
    parser.add_argument("--video-dir", required=True, help="Directory containing video chunk files")
    parser.add_argument("--output-dir", required=True, help="Directory to save fused captions")
    parser.add_argument("--api-key", help="Gemini API key (optional, uses GEMINI_API_KEY env var)")

    args = parser.parse_args()

    main(
        video_only_name=args.video_only_name,
        pack_name=args.pack_name,
        video_dir=args.video_dir,
        output_dir=args.output_dir,
        api_key=args.api_key
    )
