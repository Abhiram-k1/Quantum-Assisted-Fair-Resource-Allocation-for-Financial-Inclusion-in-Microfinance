"""Configuration loading and run manifests (reproducibility, fix #20)."""
from __future__ import annotations

import copy
import json
import pathlib
import platform
import subprocess
from datetime import datetime, timezone

import yaml

ROOT = pathlib.Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs" / "default.yaml"


def load_config(path: str | pathlib.Path | None = None, overrides: dict | None = None) -> dict:
    """Load a YAML config and apply nested ``overrides`` (dict of dicts)."""
    with open(path or DEFAULT_CONFIG) as fh:
        cfg = yaml.safe_load(fh)
    if overrides:
        cfg = _merge(cfg, overrides)
    w = cfg["scores"]["weights"]
    if abs(w["F"] + w["N"] + w["S"] - 1) > 1e-9 or min(w.values()) < 0:
        raise ValueError("score weights must be non-negative and sum to 1")
    return cfg


def _merge(base: dict, upd: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in upd.items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"],
                                       stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return None


def write_manifest(out_dir: pathlib.Path, cfg: dict, **extra) -> pathlib.Path:
    """Save everything needed to reproduce a run next to its results."""
    import numpy, scipy, sklearn, pandas
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = dict(
        created_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        git_commit=_git_commit(), python=platform.python_version(),
        packages=dict(numpy=numpy.__version__, scipy=scipy.__version__,
                      sklearn=sklearn.__version__, pandas=pandas.__version__),
        config=cfg, **extra)
    p = out_dir / "manifest.json"
    p.write_text(json.dumps(manifest, indent=2, default=str))
    return p
