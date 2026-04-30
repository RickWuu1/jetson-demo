"""
I/O utilities: JSON read/write, path helpers, result saving.
"""

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict


def save_json(data: Dict, path: str, indent: int = 2):
    """Save dict to JSON, creating parent dirs automatically."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=indent)


def load_json(path: str) -> Dict:
    with open(path) as f:
        return json.load(f)


def timestamp() -> str:
    """Return current timestamp string for output naming."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def ensure_dir(path: str) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_results(results: Dict, output_dir: str, filename: str = "results.json"):
    """Save experiment results with auto timestamp. Does NOT mutate the input dict."""
    to_save = dict(results)       # shallow copy to avoid mutating caller's dict
    to_save["_saved_at"] = timestamp()
    path = Path(output_dir) / filename
    save_json(to_save, str(path))
    return str(path)
