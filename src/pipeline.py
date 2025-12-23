from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import subprocess
import sys

from config import MODELS_DIR, MODEL_NAME_PREFIX


def run(cmd: list[str]) -> None:
    """Run a command and fail fast if it errors."""
    print("\n$", " ".join(cmd))
    result = subprocess.run(cmd, text=True)
    if result.returncode != 0:
        raise SystemExit(result.returncode)


def main():
    parser = argparse.ArgumentParser(description="End-to-end pipeline: train -> evaluate")
    parser.add_argument("--epochs", type=int, default=30, help="Max epochs (EarlyStopping may stop earlier)")
    parser.add_argument(
        "--tag",
        type=str,
        default="",
        help="Optional tag to include in model filename (e.g., v1, aug, lr1e-3)",
    )
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag_part = f"_{args.tag}" if args.tag else ""
    model_path = MODELS_DIR / f"{MODEL_NAME_PREFIX}{tag_part}_{timestamp}.keras"

    # 1) Train
    run([sys.executable, "src/train.py", "--epochs", str(args.epochs)])

    # train.py saves a timestamped model itself; to keep pipeline deterministic,
    # we re-save the latest model to our chosen model_path if needed.
    # Simplest: find the most recently modified .keras in MODELS_DIR.
    latest = max(MODELS_DIR.glob(f"{MODEL_NAME_PREFIX}*.keras"), key=lambda p: p.stat().st_mtime)
    if latest != model_path:
        print(f"\nLatest model found:\n  {latest}")
        print(f"Copying to pipeline target:\n  {model_path}")
        model_path.write_bytes(latest.read_bytes())

    # 2) Evaluate
    run([sys.executable, "src/evaluate.py", "--model", str(model_path)])

    print("\n✅ Pipeline complete")
    print("Model:", model_path)
    print("Reports: reports/confusion_matrix.png , reports/metrics.json")


if __name__ == "__main__":
    main()
