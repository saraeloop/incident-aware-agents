"""
Experiment Runner for incident-aware-agents.

Runs the task corpus under three conditions:
    A: Baseline (no intuition)
    B: Incident memory (learns from past vetoes)
    C: Rules card (static guidance)

Usage:
    python -m src.harness.runner --corpus tasks/corpus.json
    
    # Or run specific conditions
    python -m src.harness.runner --conditions A,B
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import noesis as ns

from src.agent import create_adapter
from src.policies.safety_intuition import RulesCardIntuition
from src.policies.incident_intuition import IncidentAwareIntuition
from src.memory.incident_store import IncidentStore
from src.harness.aggregator import aggregate_incidents


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

@dataclass
class ExperimentConfig:
    """Configuration for the experiment run."""
    
    corpus_path: Path = Path("tasks/corpus.json")
    runs_dir: Path = Path("./runs")
    results_dir: Path = Path("./results")
    incidents_path: Path = Path("./results/incidents.jsonl")
    
    governance_mode: str = "enforce"
    planner_mode: str = "meta"
    seed: int = 42
    run_conditions: tuple[str, ...] = ("A", "B", "C")


# -----------------------------------------------------------------------------
# Results
# -----------------------------------------------------------------------------

@dataclass
class TaskResult:
    """Result of running a single task."""
    
    task_id: str
    episode_id: str
    condition: str
    expected_outcome: str
    actual_outcome: str
    vetoed: bool
    success: bool
    veto_rule_id: str | None
    duration_sec: float
    
    @property
    def correct(self) -> bool:
        if self.expected_outcome == "veto":
            return self.vetoed
        return not self.vetoed and self.success


@dataclass
class ConditionResults:
    """Aggregated results for one condition."""
    
    condition: str
    task_results: list[TaskResult] = field(default_factory=list)
    
    @property
    def total_tasks(self) -> int:
        return len(self.task_results)
    
    @property
    def veto_count(self) -> int:
        return sum(1 for r in self.task_results if r.vetoed)
    
    @property
    def veto_rate(self) -> float:
        if not self.task_results:
            return 0.0
        return self.veto_count / len(self.task_results)
    
    @property
    def success_count(self) -> int:
        return sum(1 for r in self.task_results if r.success and not r.vetoed)
    
    @property
    def success_rate(self) -> float:
        non_vetoed = [r for r in self.task_results if not r.vetoed]
        if not non_vetoed:
            return 0.0
        return sum(1 for r in non_vetoed if r.success) / len(non_vetoed)
    
    @property
    def correct_count(self) -> int:
        return sum(1 for r in self.task_results if r.correct)
    
    @property
    def accuracy(self) -> float:
        if not self.task_results:
            return 0.0
        return self.correct_count / len(self.task_results)
    
    def veto_breakdown(self) -> dict[str, int]:
        breakdown: dict[str, int] = {}
        for r in self.task_results:
            if r.vetoed and r.veto_rule_id:
                breakdown[r.veto_rule_id] = breakdown.get(r.veto_rule_id, 0) + 1
        return breakdown
    
    def regression_rate(self, baseline_vetoes: set[str]) -> float:
        """Vetoes for task_ids that were also vetoed in baseline."""
        if not baseline_vetoes:
            return 0.0
        repeat_vetoes = sum(
            1 for r in self.task_results 
            if r.vetoed and r.task_id in baseline_vetoes
        )
        return repeat_vetoes / len(baseline_vetoes)


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def load_corpus(path: Path) -> list[dict[str, Any]]:
    """Load task corpus from JSON file."""
    data = json.loads(path.read_text())
    return data["tasks"]


def is_enforced_veto(event: dict[str, Any]) -> bool:
    """Check if event is an enforced governance veto."""
    if event.get("phase") != "governance":
        return False
    payload = event.get("payload") or {}
    return payload.get("decision") == "veto" and payload.get("enforced") is True


def get_veto_rule_id(events: list[dict[str, Any]]) -> str | None:
    """Extract rule_id from veto events."""
    for event in events:
        if is_enforced_veto(event):
            payload = event.get("payload") or {}
            return payload.get("rule_id")
    return None


def get_episode_outcome(episode_id: str) -> tuple[bool, bool, str | None]:
    """Get (vetoed, success, veto_rule_id) for an episode."""
    try:
        summary = ns.summary.read(episode_id)
        events = list(ns.events.read(episode_id))
    except Exception as e:
        print(f"    Warning: Could not read episode {episode_id}: {e}")
        return False, False, None
    
    vetoed = any(is_enforced_veto(e) for e in events)
    veto_rule_id = get_veto_rule_id(events) if vetoed else None
    
    status = summary.get("status", "")
    metrics = summary.get("metrics") or {}
    insight = summary.get("insight") or {}
    insight_metrics = insight.get("metrics") or {}
    
    success = (
        status == "success" 
        or metrics.get("success") is True
        or insight_metrics.get("success") is True
    )
    
    return vetoed, success, veto_rule_id


# -----------------------------------------------------------------------------
# Task Runner
# -----------------------------------------------------------------------------

def run_task(
    task: dict[str, Any],
    condition: str,
    intuition: Any,
    config: ExperimentConfig,
    adapter: Any = None,
) -> TaskResult:
    """Run a single task and return the result."""
    task_id = task["id"]
    description = task["description"]
    expected = task["expected_outcome"]

    start = time.time()

    try:
        episode_id = ns.solve(
            description,
            using=adapter,
            seed=config.seed,
            intuition=intuition,
            tags={
                "experiment": "incident-aware-agents",
                "condition": condition,
                "task_id": task_id,
                "expected_outcome": expected,
            },
        )
    except Exception as e:
        print(f"    Error running task {task_id}: {e}")
        return TaskResult(
            task_id=task_id,
            episode_id="error",
            condition=condition,
            expected_outcome=expected,
            actual_outcome="error",
            vetoed=False,
            success=False,
            veto_rule_id=None,
            duration_sec=time.time() - start,
        )
    
    duration = time.time() - start
    vetoed, success, veto_rule_id = get_episode_outcome(episode_id)
    actual = "veto" if vetoed else ("success" if success else "failure")
    
    return TaskResult(
        task_id=task_id,
        episode_id=episode_id,
        condition=condition,
        expected_outcome=expected,
        actual_outcome=actual,
        vetoed=vetoed,
        success=success,
        veto_rule_id=veto_rule_id,
        duration_sec=duration,
    )


# -----------------------------------------------------------------------------
# Condition Runners
# -----------------------------------------------------------------------------

def run_condition_a(
    tasks: list[dict[str, Any]],
    config: ExperimentConfig,
    adapter: Any,
) -> ConditionResults:
    """Condition A: Baseline (no intuition)."""
    print("\n" + "=" * 60)
    print("CONDITION A: Baseline (no intuition)")
    print("=" * 60)

    results = ConditionResults(condition="A")

    for i, task in enumerate(tasks, 1):
        print(f"  [{i}/{len(tasks)}] {task['id']}: {task['description'][:50]}...")
        result = run_task(task, "A", intuition=False, config=config, adapter=adapter)
        results.task_results.append(result)
        status = "VETO" if result.vetoed else ("✓" if result.success else "✗")
        print(f"           → {status}")

    return results


def run_condition_b(
    tasks: list[dict[str, Any]],
    config: ExperimentConfig,
    incident_store: IncidentStore,
    adapter: Any,
) -> ConditionResults:
    """Condition B: Incident memory."""
    print("\n" + "=" * 60)
    print(f"CONDITION B: Incident memory ({len(incident_store)} incidents)")
    print("=" * 60)

    intuition = IncidentAwareIntuition(incident_store)
    results = ConditionResults(condition="B")

    for i, task in enumerate(tasks, 1):
        print(f"  [{i}/{len(tasks)}] {task['id']}: {task['description'][:50]}...")
        result = run_task(task, "B", intuition=intuition, config=config, adapter=adapter)
        results.task_results.append(result)
        status = "VETO" if result.vetoed else ("✓" if result.success else "✗")
        print(f"           → {status}")

    return results


def run_condition_c(
    tasks: list[dict[str, Any]],
    config: ExperimentConfig,
    adapter: Any,
) -> ConditionResults:
    """Condition C: Rules card."""
    print("\n" + "=" * 60)
    print("CONDITION C: Rules card (static guidance)")
    print("=" * 60)

    intuition = RulesCardIntuition()
    results = ConditionResults(condition="C")

    for i, task in enumerate(tasks, 1):
        print(f"  [{i}/{len(tasks)}] {task['id']}: {task['description'][:50]}...")
        result = run_task(task, "C", intuition=intuition, config=config, adapter=adapter)
        results.task_results.append(result)
        status = "VETO" if result.vetoed else ("✓" if result.success else "✗")
        print(f"           → {status}")

    return results


# -----------------------------------------------------------------------------
# Report
# -----------------------------------------------------------------------------

def generate_report(results: dict[str, ConditionResults], config: ExperimentConfig) -> str:
    """Generate markdown comparison report."""
    baseline = results.get("A")
    baseline_vetoes = set(
        r.task_id for r in (baseline.task_results if baseline else []) if r.vetoed
    )
    
    lines = [
        "# Incident-Aware Agents: Experiment Results",
        "",
        f"**Generated:** {datetime.now(timezone.utc).isoformat()}",
        f"**Corpus:** {config.corpus_path}",
        "",
        "## Summary",
        "",
        "| Metric | A (Baseline) | B (Memory) | C (Rules) |",
        "|--------|--------------|------------|-----------|",
    ]
    
    metrics = [
        ("Veto count", lambda r: str(r.veto_count)),
        ("Veto rate", lambda r: f"{r.veto_rate:.1%}"),
        ("Success rate", lambda r: f"{r.success_rate:.1%}"),
        ("Accuracy", lambda r: f"{r.accuracy:.1%}"),
        ("Regression", lambda r: f"{r.regression_rate(baseline_vetoes):.1%}"),
    ]
    
    for name, fn in metrics:
        row = f"| {name} |"
        for cond in ["A", "B", "C"]:
            row += f" {fn(results[cond]) if cond in results else '-'} |"
        lines.append(row)
    
    return "\n".join(lines)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def run_experiment(config: ExperimentConfig) -> dict[str, ConditionResults]:
    """Run the full experiment."""

    ns.set(
        runs_dir=str(config.runs_dir),
        governance_mode=config.governance_mode,
        planner_mode=config.planner_mode,
    )

    tasks = load_corpus(config.corpus_path)
    print(f"Loaded {len(tasks)} tasks from {config.corpus_path}")

    # Create the LangGraph agent adapter
    print("Creating LangGraph agent adapter...")
    adapter = create_adapter()

    results: dict[str, ConditionResults] = {}

    # Condition A
    if "A" in config.run_conditions:
        results["A"] = run_condition_a(tasks, config, adapter)

        # Generate incidents for B
        print("\nAggregating incidents from Condition A...")
        incidents = aggregate_incidents(config.runs_dir, filter_tags={"condition": "A"})
        store = IncidentStore(incidents)
        store.to_file(config.incidents_path)
        print(f"Saved {len(incidents)} incidents to {config.incidents_path}")

    # Condition B
    if "B" in config.run_conditions:
        store = IncidentStore.from_file(config.incidents_path) if config.incidents_path.exists() else IncidentStore()
        results["B"] = run_condition_b(tasks, config, store, adapter)

    # Condition C
    if "C" in config.run_conditions:
        results["C"] = run_condition_c(tasks, config, adapter)

    return results


def main() -> int:
    import argparse
    
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
        seed=args.seed,
        run_conditions=tuple(args.conditions.split(",")),
    )
    
    config.runs_dir.mkdir(parents=True, exist_ok=True)
    config.results_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("INCIDENT-AWARE AGENTS EXPERIMENT")
    print("=" * 60)
    
    results = run_experiment(config)
    
    # Save report
    report = generate_report(results, config)
    report_path = config.results_dir / "report.md"
    report_path.write_text(report)
    print(f"\nReport: {report_path}")
    
    # Save raw results
    raw = {
        cond: {
            "veto_rate": r.veto_rate,
            "accuracy": r.accuracy,
            "veto_count": r.veto_count,
        }
        for cond, r in results.items()
    }
    raw_path = config.results_dir / "results.json"
    raw_path.write_text(json.dumps(raw, indent=2))
    
    # Print summary
    print("\n" + "=" * 60)
    print(f"{'Condition':<12} {'Veto Rate':<12} {'Accuracy':<12}")
    print("-" * 36)
    for cond in ["A", "B", "C"]:
        if cond in results:
            r = results[cond]
            print(f"{cond:<12} {r.veto_rate:<12.1%} {r.accuracy:<12.1%}")
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
