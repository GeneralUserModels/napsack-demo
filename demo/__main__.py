"""Demo dashboard entry point – run with: uv run python -m demo"""
import sys
import threading
import time
import webbrowser
from pathlib import Path

# Ensure repo root (containing demo/) is on sys.path
_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import uvicorn

HOST = "127.0.0.1"
PORT = 8765
URL = f"http://{HOST}:{PORT}"


def _open_browser():
    """Wait briefly for uvicorn to start, then open the dashboard."""
    time.sleep(1.5)
    print(f"\n  → Opening {URL} in your browser…\n")
    webbrowser.open(URL)


def main():
    threading.Thread(target=_open_browser, daemon=True).start()
    uvicorn.run(
        "demo.app.server:app",
        host=HOST,
        port=PORT,
        reload=False,
        log_level="info",
    )


if __name__ == "__main__":
    main()
