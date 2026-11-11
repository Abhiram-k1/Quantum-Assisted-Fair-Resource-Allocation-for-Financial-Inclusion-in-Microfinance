"""Generate publication-grade Q-FAR figures into results/figures/."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from qfar.visuals import generate_all_visuals


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default=None)
    ap.add_argument("--fig-dir", default=None)
    ap.add_argument("--config", default=None)
    args = ap.parse_args()

    paths = generate_all_visuals(
        results_dir=Path(args.results_dir) if args.results_dir else None,
        fig_dir=Path(args.fig_dir) if args.fig_dir else None,
        cfg_path=args.config,
    )
    print(f"Generated {len(paths)} figure(s):")
    for p in paths:
        print(f"  - {p}")


if __name__ == "__main__":
    main()

