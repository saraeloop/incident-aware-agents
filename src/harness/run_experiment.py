"""
Experiment orchestration for incident-aware-agents.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any

import noesis as ns

from src.agent import build_agent, configure_noesis_for_agent
from src.harness.aggregator import aggregate_incidents
from src.harness.execute_task import (
    build_adapter_with_hints,
    hint_from_incidents,
    hint_from_rules,
    run_condition,
    unsafe_commands_for_task,
)
from src.harness.models import ConditionResults, ExperimentConfig
from src.memory.incident_store import IncidentStore
from src.policies.governance_policy import ExperimentGovernor
from src.policies.incident_intuition import IncidentAwareIntuition
from src.policies.safety_intuition import RulesCardIntuition


def load_corpus(path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text())
    return list(data["tasks"])


def run_experiment(config: ExperimentConfig) -> dict[str, ConditionResults]:
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set. Planning will fail; the experiment would be invalid.")

    if not config.run_id:
        config.run_id = datetime.now(timezone.utc).strftime("run_%Y%m%dT%H%M%SZ")

    _ensure_noesis_intuition_mode()
    ns.set(
        runs_dir=str(config.runs_dir),
        governance_mode=config.episode_governance_mode,
        planner_mode=config.planner_mode,
        intuition_mode="advisory",
        governance_policy=ExperimentGovernor(),
    )
    cfg = ns.get()
    print("episode_governance_mode =", cfg.get("governance_mode"))
    print("action_governance_mode  =", config.action_governance_mode)
    print("planner_mode            =", cfg.get("planner_mode"))
    print("run_id                  =", config.run_id)
    for key in ["governance_policy", "governor", "pre_act_governor", "governance"]:
        if key in cfg:
            print(f"{key} =", cfg[key])

    configure_noesis_for_agent(image="incident-sandbox")

    tasks = load_corpus(config.corpus_path)
    print(f"Loaded {len(tasks)} tasks from {config.corpus_path}")

    print("Creating LangGraph agent...")
    agent = build_agent()
    safe_goal = "Execute the task using the adapter."

    results: dict[str, ConditionResults] = {}

    if "A" in config.run_conditions:
        def adapter_builder_a(task: dict[str, Any]) -> Any:
            return build_adapter_with_hints(
                agent,
                task_override=str(task.get("description") or ""),
                forced_commands=unsafe_commands_for_task(task),
                action_governance_mode=config.action_governance_mode,
                tags={
                    "experiment": "incident-aware-agents",
                    "condition": "A",
                    "task_id": task.get("id"),
                    "expected_outcome": task.get("expected_outcome"),
                    "run_id": config.run_id,
                },
            )

        results["A"] = run_condition(
            tasks,
            label="A",
            intuition=False,
            config=config,
            adapter_builder=adapter_builder_a,
            goal_override=safe_goal,
        )

        print("\nAggregating incidents from Condition A...")
        incidents = aggregate_incidents(
            config.runs_dir,
            filter_tags={"condition": "A", "run_id": config.run_id},
            corpus_path=config.corpus_path,
        )
        store = IncidentStore(incidents)
        store.to_file(config.incidents_path)
        print(f"Saved {len(incidents)} incidents to {config.incidents_path}")

    if "B" in config.run_conditions:
        store = IncidentStore.from_file(config.incidents_path) if config.incidents_path.exists() else IncidentStore()
        intuition = IncidentAwareIntuition(store)

        def adapter_builder_b(task: dict[str, Any]) -> Any:
            return build_adapter_with_hints(
                agent,
                hints_fn=lambda t: hint_from_incidents(store, t),
                retry_on_veto=True,
                task_override=str(task.get("description") or ""),
                action_governance_mode=config.action_governance_mode,
                tags={
                    "experiment": "incident-aware-agents",
                    "condition": "B",
                    "task_id": task.get("id"),
                    "expected_outcome": task.get("expected_outcome"),
                    "run_id": config.run_id,
                },
            )

        results["B"] = run_condition(
            tasks,
            label="B",
            intuition=intuition,
            config=config,
            adapter_builder=adapter_builder_b,
            goal_override=safe_goal,
        )

    if "C" in config.run_conditions:
        intuition = RulesCardIntuition()

        def adapter_builder_c(task: dict[str, Any]) -> Any:
            return build_adapter_with_hints(
                agent,
                hints_fn=hint_from_rules,
                retry_on_veto=True,
                task_override=str(task.get("description") or ""),
                action_governance_mode=config.action_governance_mode,
                tags={
                    "experiment": "incident-aware-agents",
                    "condition": "C",
                    "task_id": task.get("id"),
                    "expected_outcome": task.get("expected_outcome"),
                    "run_id": config.run_id,
                },
            )

        results["C"] = run_condition(
            tasks,
            label="C",
            intuition=intuition,
            config=config,
            adapter_builder=adapter_builder_c,
            goal_override=safe_goal,
        )

    return results


def _ensure_noesis_intuition_mode() -> None:
    """
    Ensure NoesisState exposes intuition_mode for older Noesis layouts.

    This is a lab-side runtime shim (no file changes in Noesis).
    """
    try:
        from noesis.domain.state import NoesisState
    except Exception:
        return
    if hasattr(NoesisState, "intuition_mode"):
        return

    def _get_intuition_mode(self: object) -> str:
        return "advisory"

    setattr(NoesisState, "intuition_mode", property(_get_intuition_mode))
