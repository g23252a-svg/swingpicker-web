"""Build audited swing rankings without running collectors or sending notifications."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from services.swing_model import run

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--asof", required=True)
    args = parser.parse_args()
    picks, report = run(args.data_dir, args.asof)
    print(json.dumps({k: v for k, v in report.items()
                      if k not in {"calibration_bins", "features", "policy"}}, ensure_ascii=False, indent=2))
    print(picks.head(10).to_string(index=False))
