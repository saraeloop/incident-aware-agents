#!/usr/bin/env python3
"""
incident-aware-agents: Can AI agents learn from governance violations?

This experiment tests three conditions:
    A: Baseline - no guidance, agent gets vetoed on dangerous tasks
    B: Incident memory - agent sees "last time you tried X, it was vetoed"
    C: Rules card - agent sees "never delete, move to /quarantine instead"

The key metric is regression rate: if Condition A vetoes task-001 (delete files),
does Condition B still get vetoed on task-002 (wipe cache)? Low regression = learning.

Usage:
    # Run the full experiment
    python main.py run

    # Run specific conditions
    python main.py run --conditions A,B

    # Compute metrics from existing results
    python main.py metrics

    # Extract incidents from baseline run
    python main.py aggregate

    # Dry run (simulate without calling Noesis)
    python main.py run --dry-run

Commands:
    run        Run the experiment (all conditions by default)
    metrics    Compute metrics from results/events.jsonl
    aggregate  Extract incidents from Condition A vetoes
    report     Generate markdown report from results
    validate   Validate corpus and policy configuration
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


# -----------------------------------------------------------------------------
# Commands
# -----------------------------------------------------------------------------

def cmd_run(args: argparse.Namespace) -> int:
    """Run the experiment."""
    from src.harness.runner import ExperimentConfig, run_experiment, generate_report

    config = ExperimentConfig(
        corpus_path=Path(args.corpus),
        runs_dir=Path(args.runs_dir),
        results_dir=Path(args.results_dir),
        incidents_path=Path(args.results_dir) / "incidents.jsonl",
        seed=args.seed,
        run_conditions=tuple(args.conditions.split(",")),
    )

    config.runs_dir.mkdir(parents=True, exist_ok=True)
    config.results_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("INCIDENT-AWARE AGENTS EXPERIMENT")
    print("=" * 60)
    print(f"Corpus: {config.corpus_path}")
    print(f"Conditions: {', '.join(config.run_conditions)}")
    print()

    results = run_experiment(config)

    # Generate report
    report = generate_report(results, config)
    report_path = config.results_dir / "report.md"
    report_path.write_text(report)
    print(f"\nReport saved to: {report_path}")

    return 0


def cmd_metrics(args: argparse.Namespace) -> int:
    """Compute and display metrics."""
    from src.harness.metrics import compute_metrics, generate_report, save_metrics

    events_path = Path(args.events)
    if not events_path.exists():
        print(f"Error: {events_path} not found")
        print("Run the experiment first: python main.py run")
        return 1

    print(f"Computing metrics from: {events_path}")
    print("=" * 60)

    metrics = compute_metrics(events_path)
    report = generate_report(metrics)
    print(report)

    if args.output:
        save_metrics(metrics, args.output)
        print(f"\nSaved to: {args.output}")

    return 0


def cmd_aggregate(args: argparse.Namespace) -> int:
    """Extract incidents from baseline vetoes."""
    from src.harness.aggregator import aggregate_from_events

    events_path = Path(args.events)
    if not events_path.exists():
        print(f"Error: {events_path} not found")
        return 1

    print(f"Aggregating incidents from: {events_path}")
    print("=" * 60)

    incidents = aggregate_from_events(
        events_path=events_path,
        incidents_path=Path(args.output),
        corpus_path=Path(args.corpus) if args.corpus else None,
    )

    print(f"Extracted {len(incidents)} incidents from baseline vetoes")
    for inc in incidents[:5]:
        print(f"  - {inc.episode_id}: {inc.rule_id}")
    if len(incidents) > 5:
        print(f"  ... and {len(incidents) - 5} more")

    print(f"\nSaved to: {args.output}")
    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    """Validate corpus and policy configuration."""
    import json

    corpus_path = Path(args.corpus)
    if not corpus_path.exists():
        print(f"Error: {corpus_path} not found")
        return 1

    print(f"Validating: {corpus_path}")
    print("=" * 60)

    corpus = json.loads(corpus_path.read_text())
    tasks = corpus.get("tasks", [])

    # Check task structure
    required = ["id", "description", "expected_outcome", "risk_class"]
    errors = []
    for task in tasks:
        for field in required:
            if field not in task:
                errors.append(f"{task.get('id', 'unknown')}: missing {field}")

    if errors:
        print("Errors:")
        for e in errors:
            print(f"  - {e}")
        return 1

    # Summary
    veto_count = sum(1 for t in tasks if t["expected_outcome"] == "veto")
    allow_count = sum(1 for t in tasks if t["expected_outcome"] == "allow")

    print(f"Tasks: {len(tasks)}")
    print(f"  Expected vetoes: {veto_count}")
    print(f"  Expected allows: {allow_count}")

    # Test policies
    print("\nTesting policies...")
    from src.policies.safety_intuition import RulesCardIntuition

    policy = RulesCardIntuition()
    matches = 0
    for task in tasks:
        hint = policy.advise({"task": task["description"]})
        if hint and task["expected_outcome"] == "veto":
            matches += 1

    print(f"  RulesCard coverage: {matches}/{veto_count} veto tasks matched")

    print("\nValidation complete.")
    return 0


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="incident-aware-agents experiment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # run
    p_run = subparsers.add_parser("run", help="Run the experiment")
    p_run.add_argument("--corpus", default="tasks/corpus.json", help="Path to task corpus")
    p_run.add_argument("--runs-dir", default="./runs", help="Directory for Noesis runs")
    p_run.add_argument("--results-dir", default="./results", help="Directory for results")
    p_run.add_argument("--conditions", default="A,B,C", help="Conditions to run (comma-separated)")
    p_run.add_argument("--seed", type=int, default=42, help="Random seed")
    p_run.add_argument("--dry-run", action="store_true", help="Simulate without Noesis")

    # metrics
    p_metrics = subparsers.add_parser("metrics", help="Compute metrics from results")
    p_metrics.add_argument("--events", default="results/events.jsonl", help="Path to events.jsonl")
    p_metrics.add_argument("--output", help="Output directory for metrics files")

    # aggregate
    p_agg = subparsers.add_parser("aggregate", help="Extract incidents from baseline")
    p_agg.add_argument("--events", default="results/events.jsonl", help="Path to events.jsonl")
    p_agg.add_argument("--output", default="results/incidents.jsonl", help="Output incidents file")
    p_agg.add_argument("--corpus", default="tasks/corpus.json", help="Corpus for enrichment")

    # validate
    p_val = subparsers.add_parser("validate", help="Validate corpus and policies")
    p_val.add_argument("--corpus", default="tasks/corpus.json", help="Path to task corpus")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 0

    commands = {
        "run": cmd_run,
        "metrics": cmd_metrics,
        "aggregate": cmd_aggregate,
        "validate": cmd_validate,
    }

    return commands[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
