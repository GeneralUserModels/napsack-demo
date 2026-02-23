"""Demo dashboard entry point – run with: uv run python -m demo"""
import sys
from pathlib import Path

# Ensure repo root (containing demo/) is on sys.path
_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import uvicorn


def main():
    uvicorn.run(
        "demo.app.server:app",
        host="0.0.0.0",
        port=8765,
        reload=False,
        log_level="info",
    )


if __name__ == "__main__":
    main()
