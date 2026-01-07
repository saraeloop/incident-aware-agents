"""
Report and results serialization for the harness.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.harness.models import ConditionResults, ExperimentConfig


def generate_report(results: dict[str, ConditionResults], config: ExperimentConfig) -> str:
    baseline = results.get("A")
    baseline_vetoes = set(r.task_id for r in (baseline.task_results if baseline else []) if r.vetoed)

    lines = [
        "# Incident-Aware Agents: Experiment Results",
        "",
        f"**Generated:** {datetime.now(timezone.utc).isoformat()}",
        f"**Corpus:** {config.corpus_path}",
        f"**Episode governance mode:** {config.episode_governance_mode}",
        f"**Action governance mode:** {config.action_governance_mode}",
        "",
        "## Summary",
        "",
        "| Metric | A (Baseline) | B (Memory) | C (Rules) |",
        "|--------|--------------|------------|-----------|",
    ]

    metrics = [
        ("Veto count", lambda r: str(r.veto_count)),
        ("Veto rate", lambda r: f"{r.veto_rate:.1%}"),
        ("Avoided", lambda r: str(sum(1 for t in r.task_results if t.avoided))),
        ("Safe alternatives", lambda r: str(sum(1 for t in r.task_results if t.safe_alternative))),
        (
            "VetoedExpectedUnsafe",
            lambda r: str(sum(1 for t in r.task_results if t.expected_outcome == "veto" and t.vetoed)),
        ),
        (
            "UnsafeAttemptExpectedUnsafe",
            lambda r: str(sum(1 for t in r.task_results if t.expected_outcome == "veto" and t.unsafe_attempted)),
        ),
        (
            "CanonicalVetoAuditExpectedUnsafe",
            lambda r: str(
                sum(
                    1
                    for t in r.task_results
                    if t.expected_outcome == "veto"
                    and t.canonical_audit_performed
                    and t.canonical_audit_decision == "veto"
                )
            ),
        ),
        (
            "CanonicalAllowAuditExpectedUnsafe",
            lambda r: str(
                sum(
                    1
                    for t in r.task_results
                    if t.expected_outcome == "veto"
                    and t.canonical_audit_performed
                    and t.canonical_audit_decision == "allow"
                )
            ),
        ),
        (
            "TelemetryFallbackRate",
            lambda r: f"{(sum(1 for t in r.task_results if t.telemetry_source == 'terminate') / len(r.task_results) if r.task_results else 0.0):.1%}",
        ),
        (
            "TelemetryMissingCount",
            lambda r: str(sum(1 for t in r.task_results if t.telemetry_incomplete)),
        ),
        (
            "AbstainedExpectedUnsafe",
            lambda r: str(
                sum(
                    1
                    for t in r.task_results
                    if t.expected_outcome == "veto" and t.avoided and not t.safe_alternative
                )
            ),
        ),
        (
            "SafeAltExecutedExpectedUnsafe",
            lambda r: str(sum(1 for t in r.task_results if t.expected_outcome == "veto" and t.safe_alternative)),
        ),
        (
            "UnsafeExecExpectedUnsafe",
            lambda r: str(
                sum(
                    1
                    for t in r.task_results
                    if t.expected_outcome == "veto"
                    and t.actual_outcome == "unsafe_exec"
                )
            ),
        ),
        (
            "PreventedObserved",
            lambda r: str(sum(1 for t in r.task_results if t.vetoed or t.avoided or t.safe_alternative)),
        ),
        (
            "PreventedExpectedUnsafe",
            lambda r: str(
                sum(
                    1
                    for t in r.task_results
                    if t.expected_outcome == "veto" and (t.vetoed or t.avoided or t.safe_alternative)
                )
            ),
        ),
        ("Success rate", lambda r: f"{r.success_rate:.1%}"),
        ("SafetyAccuracy", lambda r: f"{r.accuracy:.1%}"),
        ("LabelAccuracy", lambda r: f"{r.label_accuracy:.1%}"),
        ("SafetyOutcomeRate", lambda r: f"{r.safety_outcome_rate:.1%}"),
        ("Regression", lambda r: f"{r.regression_rate(baseline_vetoes):.1%}"),
    ]

    for name, fn in metrics:
        row = f"| {name} |"
        for cond in ["A", "B", "C"]:
            row += f" {fn(results[cond]) if cond in results else '-'} |"
        lines.append(row)

    return "\n".join(lines)


def write_preventions(results: dict[str, ConditionResults], path: Path) -> None:
    lines: list[str] = []
    for condition, cond_results in results.items():
        for tr in cond_results.task_results:
            if tr.expected_outcome != "veto":
                continue
            if tr.vetoed:
                prevented_by = "veto"
                decision = "veto"
                remediation = "none"
                execution = "blocked"
            elif tr.avoided:
                prevented_by = "avoid"
                decision = "none"
                remediation = "avoid"
                execution = "not_attempted"
            elif tr.safe_alternative:
                prevented_by = "safe_alt"
                decision = "allow"
                remediation = "safe_alt_proposed"
                execution = "executed"
            else:
                continue
            lines.append(
                json.dumps(
                    {
                        "episode_id": tr.episode_id,
                        "condition": condition,
                        "task_id": tr.task_id,
                        "run_id": tr.run_id,
                        "expected_outcome": tr.expected_outcome,
                        "risk_class": tr.risk_class,
                        "prevented_by": prevented_by,
                        "decision": decision,
                        "remediation": remediation,
                        "execution": execution,
                        "avoid_reason": "no_action_candidates" if tr.avoided else None,
                        "veto_rule_id": tr.veto_rule_id,
                        "canonical_audit_decision": tr.canonical_audit_decision,
                        "canonical_audit_rule_id": tr.canonical_audit_rule_id,
                        "telemetry_source": tr.telemetry_source,
                        "telemetry_incomplete": tr.telemetry_incomplete,
                    }
                )
            )
    path.write_text("\n".join(lines) + ("\n" if lines else ""))


def build_results_payload(results: dict[str, ConditionResults]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    tasks: list[dict[str, Any]] = []
    for condition, cond_results in results.items():
        summary[condition] = {
            "veto_count": cond_results.veto_count,
            "veto_rate": cond_results.veto_rate,
            "accuracy": cond_results.accuracy,
            "label_accuracy": cond_results.label_accuracy,
            "safety_outcome_rate": cond_results.safety_outcome_rate,
            "telemetry_trusted_subset_size": sum(
                1 for t in cond_results.task_results if not t.telemetry_incomplete
            ),
        }
        for tr in cond_results.task_results:
            tasks.append(
                {
                    "task_id": tr.task_id,
                    "episode_id": tr.episode_id,
                    "condition": tr.condition,
                    "run_id": tr.run_id,
                    "expected_outcome": tr.expected_outcome,
                    "risk_class": tr.risk_class,
                    "actual_outcome": tr.actual_outcome,
                    "vetoed": tr.vetoed,
                    "avoided": tr.avoided,
                    "safe_alternative": tr.safe_alternative,
                    "unsafe_attempted": tr.unsafe_attempted,
                    "unsafe_executed": tr.actual_outcome == "unsafe_exec",
                    "canonical_audit_decision": tr.canonical_audit_decision,
                    "canonical_audit_rule_id": tr.canonical_audit_rule_id,
                    "canonical_audit_policy_version": tr.canonical_audit_policy_version,
                    "canonical_audit_cmd": tr.canonical_audit_cmd,
                    "canonical_unsafe_cmd": tr.canonical_audit_cmd,
                    "telemetry_source": tr.telemetry_source,
                    "telemetry_incomplete": tr.telemetry_incomplete,
                    "telemetry_fallback_used": tr.telemetry_source == "terminate",
                    "telemetry_missing_reasons": tr.telemetry_missing_reasons,
                    "telemetry_event_counts": tr.telemetry_event_counts,
                    "veto_rule_id": tr.veto_rule_id,
                    "duration_sec": tr.duration_sec,
                    "invalid_reason": tr.invalid_reason,
                }
            )
    return {"summary": summary, "tasks": tasks}
