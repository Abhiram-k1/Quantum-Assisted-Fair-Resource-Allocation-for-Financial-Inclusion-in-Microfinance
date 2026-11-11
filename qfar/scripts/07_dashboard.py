"""Launch the interactive Q-FAR Dash + Plotly dashboard.

Usage:
    python scripts/07_dashboard.py --port 8050
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from qfar.dashboard import create_app


def main():
    ap = argparse.ArgumentParser(description="Q-FAR interactive web dashboard")
    ap.add_argument("--config", default=None)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8050)
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    app = create_app(cfg_path=args.config)
    print(f"Starting Q-FAR dashboard on http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()

