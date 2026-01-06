"""
Thin CLI wrapper for the incident-aware-agents harness.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from src.harness.models import ExperimentConfig
from src.harness.report import build_results_payload, generate_report, write_preventions
from src.harness.run_experiment import run_experiment


def main() -> int:
    parser = argparse.ArgumentParser(description="Run incident-aware-agents experiment")
    parser.add_argument("--corpus", type=Path, default=Path("tasks/corpus.json"))
    parser.add_argument("--runs-dir", type=Path, default=Path("./runs"))
    parser.add_argument("--results-dir", type=Path, default=Path("./results"))
    parser.add_argument("--conditions", type=str, default="A,B,C")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    config = ExperimentConfig(
        corpus_path=args.corpus,
        runs_dir=args.runs_dir,
        results_dir=args.results_dir,
        incidents_path=args.results_dir / "incidents.jsonl",
        preventions_path=args.results_dir / "preventions.jsonl",
        seed=args.seed,
        run_conditions=tuple(x.strip() for x in args.conditions.split(",") if x.strip()),
        run_id=datetime.now(timezone.utc).strftime("run_%Y%m%dT%H%M%SZ"),
    )

    config.runs_dir.mkdir(parents=True, exist_ok=True)
    config.results_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("INCIDENT-AWARE AGENTS EXPERIMENT")
    print("=" * 60)
    print(f"Corpus: {config.corpus_path}")
    print(f"Conditions: {', '.join(config.run_conditions)}")

    results = run_experiment(config)

    report = generate_report(results, config)
    report_path = config.results_dir / "report.md"
    report_path.write_text(report)
    print(f"\nReport saved to: {report_path}")

    preventions_path = config.results_dir / "preventions.jsonl"
    write_preventions(results, preventions_path)
    print(f"Preventions saved to: {preventions_path}")

    results_payload = build_results_payload(results)
    (config.results_dir / "results.json").write_text(json.dumps(results_payload, indent=2))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
