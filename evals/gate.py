"""CI regression gate for the offline benchmark.

    uv run --project api python evals/gate.py artifacts/runs/<run_id>            # check
    uv run --project api python evals/gate.py artifacts/runs/<run_id> --update   # new baseline

The baseline is written from a measured run, never edited by hand. A metric fails the gate if
it drops more than TOLERANCE below the baseline (or, for rates where lower is better, rises
more than TOLERANCE above it). The offline reader is deterministic, so any change is caused
by code, data or dependency changes.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BASELINE = ROOT / "evals/baseline_offline.json"
TOLERANCE = 0.02
HIGHER_IS_BETTER = [
    "evidence_hit_rate",
    "mrr",
    "answer_accuracy",
    "citation_precision",
    "correct_refusal_rate",
]
LOWER_IS_BETTER = ["false_refusal_rate", "injection_followed_rate"]


def main() -> int:
    run = Path(sys.argv[1])
    summary = json.loads((run / "summary.json").read_text())
    current = {
        name: {m: s[m] for m in HIGHER_IS_BETTER + LOWER_IS_BETTER} for name, s in summary.items()
    }
    if "--update" in sys.argv:
        BASELINE.write_text(json.dumps({"run_id": run.name, "metrics": current}, indent=2) + "\n")
        print(f"baseline updated from {run.name}")
        return 0

    baseline = json.loads(BASELINE.read_text())["metrics"]
    failures = []
    for name, expected in baseline.items():
        if name not in current:
            continue
        for metric, base in expected.items():
            value = current[name][metric]
            if base is None or value is None:
                continue
            worse = (
                value < base - TOLERANCE if metric in HIGHER_IS_BETTER else value > base + TOLERANCE
            )
            status = "FAIL" if worse else "ok"
            print(f"{status:4} {name:28} {metric:24} {value:.4f} (baseline {base:.4f})")
            if worse:
                failures.append(f"{name}.{metric}")
    if failures:
        print(f"\nregressions: {', '.join(failures)}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
