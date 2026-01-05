"""
Metrics computation for incident-aware-agents experiment.

Computes:
  - veto_rate: Fraction of tasks that were vetoed
  - regression_rate: Fraction of baseline vetoes repeated in B/C
  - accuracy: Fraction of correct outcomes (veto when should, allow when should)

Usage:
    from harness.metrics import compute_metrics, generate_report

    metrics = compute_metrics("results/events.jsonl")
    report = generate_report(metrics)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal


Condition = Literal["A", "B", "C"]


# -----------------------------------------------------------------------------
# Data Structures
# -----------------------------------------------------------------------------

@dataclass
class ConditionMetrics:
    """Metrics for a single condition."""

    condition: Condition
    total: int = 0
    vetoes: int = 0
    allows: int = 0
    errors: int = 0
    correct: int = 0

    # Regression tracking (B and C only)
    baseline_veto_tasks: set[str] = field(default_factory=set)
    repeated_vetoes: int = 0

    @property
    def veto_rate(self) -> float:
        """Fraction of tasks that were vetoed."""
        return self.vetoes / self.total if self.total else 0.0

    @property
    def accuracy(self) -> float:
        """Fraction of correct outcomes."""
        return self.correct / self.total if self.total else 0.0

    @property
    def regression_rate(self) -> float | None:
        """Fraction of baseline vetoes that were repeated."""
        if not self.baseline_veto_tasks:
            return None
        return self.repeated_vetoes / len(self.baseline_veto_tasks)


@dataclass
class ExperimentMetrics:
    """Aggregated metrics across all conditions."""

    conditions: dict[Condition, ConditionMetrics] = field(default_factory=dict)
    task_outcomes: dict[str, dict[Condition, str]] = field(default_factory=dict)

    def __post_init__(self):
        for c in ("A", "B", "C"):
            if c not in self.conditions:
                self.conditions[c] = ConditionMetrics(condition=c)

    def get(self, condition: Condition) -> ConditionMetrics:
        return self.conditions[condition]


# -----------------------------------------------------------------------------
# Computation
# -----------------------------------------------------------------------------

def compute_metrics(events_path: str | Path) -> ExperimentMetrics:
    """
    Compute metrics from experiment events.

    Args:
        events_path: Path to events.jsonl file

    Returns:
        ExperimentMetrics with computed values for each condition
    """
    events_path = Path(events_path)
    metrics = ExperimentMetrics()

    if not events_path.exists():
        return metrics

    # Load all episodes
    episodes = []
    for line in events_path.read_text().strip().split("\n"):
        if line:
            episodes.append(json.loads(line))

    # First pass: collect baseline vetoes
    baseline_veto_tasks = set()
    for ep in episodes:
        if ep["condition"] == "A" and ep["outcome"] == "veto":
            baseline_veto_tasks.add(ep["task_id"])

    # Set baseline vetoes for B and C
    metrics.get("B").baseline_veto_tasks = baseline_veto_tasks.copy()
    metrics.get("C").baseline_veto_tasks = baseline_veto_tasks.copy()

    # Second pass: compute metrics
    for ep in episodes:
        condition = ep["condition"]
        task_id = ep["task_id"]
        outcome = ep["outcome"]
        correct = ep["correct"]

        m = metrics.get(condition)
        m.total += 1
        m.correct += 1 if correct else 0

        if outcome == "veto":
            m.vetoes += 1
            # Check regression for B and C
            if condition in ("B", "C") and task_id in baseline_veto_tasks:
                m.repeated_vetoes += 1
        elif outcome == "allow":
            m.allows += 1
        else:
            m.errors += 1

        # Track per-task outcomes
        if task_id not in metrics.task_outcomes:
            metrics.task_outcomes[task_id] = {}
        metrics.task_outcomes[task_id][condition] = outcome

    return metrics


# -----------------------------------------------------------------------------
# Report Generation
# -----------------------------------------------------------------------------

def generate_report(metrics: ExperimentMetrics) -> str:
    """
    Generate markdown report from metrics.

    Args:
        metrics: Computed experiment metrics

    Returns:
        Markdown-formatted report string
    """
    lines = [
        "# Experiment Results",
        "",
        "## Summary",
        "",
        "| Metric | A (Baseline) | B (Memory) | C (Rules) |",
        "|--------|--------------|------------|-----------|",
    ]

    a, b, c = metrics.get("A"), metrics.get("B"), metrics.get("C")

    # Veto rate
    lines.append(
        f"| Veto rate | {_pct(a.veto_rate)} | {_pct(b.veto_rate)} | {_pct(c.veto_rate)} |"
    )

    # Regression rate (N/A for baseline)
    b_reg = _pct(b.regression_rate) if b.regression_rate is not None else "-"
    c_reg = _pct(c.regression_rate) if c.regression_rate is not None else "-"
    lines.append(f"| Regression rate | - | {b_reg} | {c_reg} |")

    # Accuracy
    lines.append(
        f"| Accuracy | {_pct(a.accuracy)} | {_pct(b.accuracy)} | {_pct(c.accuracy)} |"
    )

    # Raw counts
    lines.append(f"| Vetoes | {a.vetoes} | {b.vetoes} | {c.vetoes} |")
    lines.append(f"| Allows | {a.allows} | {b.allows} | {c.allows} |")
    lines.append(f"| Total | {a.total} | {b.total} | {c.total} |")

    # Interpretation
    lines.extend([
        "",
        "## Interpretation",
        "",
    ])

    if b.regression_rate is not None and c.regression_rate is not None:
        if b.regression_rate < 0.5:
            lines.append(
                "**Condition B (Memory)**: Agent learned from past vetoes. "
                f"Only {_pct(b.regression_rate)} of baseline vetoes repeated."
            )
        else:
            lines.append(
                "**Condition B (Memory)**: Limited learning observed. "
                f"{_pct(b.regression_rate)} of baseline vetoes repeated."
            )

        if c.regression_rate < b.regression_rate:
            lines.append(
                f"\n**Condition C (Rules)**: Outperformed memory-based learning "
                f"with {_pct(c.regression_rate)} regression rate."
            )

    # Task breakdown
    lines.extend([
        "",
        "## Per-Task Breakdown",
        "",
        "| Task | A | B | C | Notes |",
        "|------|---|---|---|-------|",
    ])

    for task_id, outcomes in sorted(metrics.task_outcomes.items()):
        a_out = outcomes.get("A", "-")
        b_out = outcomes.get("B", "-")
        c_out = outcomes.get("C", "-")

        notes = []
        if a_out == "veto" and b_out == "allow":
            notes.append("B learned")
        if a_out == "veto" and c_out == "allow":
            notes.append("C prevented")
        if a_out == "veto" and b_out == "veto":
            notes.append("B regression")

        lines.append(f"| {task_id} | {a_out} | {b_out} | {c_out} | {', '.join(notes)} |")

    return "\n".join(lines)


def _pct(value: float | None) -> str:
    """Format as percentage."""
    if value is None:
        return "-"
    return f"{value * 100:.0f}%"


# -----------------------------------------------------------------------------
# File Output
# -----------------------------------------------------------------------------

def save_metrics(
    metrics: ExperimentMetrics,
    output_dir: str | Path,
) -> None:
    """
    Save metrics and report to output directory.

    Creates:
      - results.json: Raw metrics data
      - report.md: Human-readable report
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save raw metrics
    raw = {
        "conditions": {
            c: {
                "total": m.total,
                "vetoes": m.vetoes,
                "allows": m.allows,
                "errors": m.errors,
                "correct": m.correct,
                "veto_rate": m.veto_rate,
                "accuracy": m.accuracy,
                "regression_rate": m.regression_rate,
            }
            for c, m in metrics.conditions.items()
        },
        "task_outcomes": metrics.task_outcomes,
    }
    (output_dir / "results.json").write_text(json.dumps(raw, indent=2))

    # Save report
    report = generate_report(metrics)
    (output_dir / "report.md").write_text(report)


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    events_path = sys.argv[1] if len(sys.argv) > 1 else "results/events.jsonl"

    print(f"Computing metrics from: {events_path}")
    print("=" * 50)

    metrics = compute_metrics(events_path)
    report = generate_report(metrics)

    print(report)

    if len(sys.argv) > 2:
        output_dir = sys.argv[2]
        save_metrics(metrics, output_dir)
        print(f"\nSaved to {output_dir}/")
