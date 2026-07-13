"""Entry point for the packaged desktop app (built by packaging/build.py).

Starts the backend (which also serves the bundled frontend) and opens the
browser. Everything else — corpus, index, settings, API keys — lives in the
OS per-user locations (see config.DATA_ROOT and backend/settings.py).
"""
import threading
import time
import webbrowser

import uvicorn

from backend.main import app

URL = "http://127.0.0.1:8642"


def _open_browser():
    time.sleep(1.5)  # give uvicorn a moment to bind
    webbrowser.open(URL)


def main():
    print(f"production-rag — opening {URL} (close this window to quit)")
    threading.Thread(target=_open_browser, daemon=True).start()
    uvicorn.run(app, host="127.0.0.1", port=8642, log_level="info")


if __name__ == "__main__":
    main()
