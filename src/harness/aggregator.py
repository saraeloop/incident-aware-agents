"""
Aggregator for incident-aware-agents experiment.

Extracts veto incidents from Noesis runs or events.jsonl files.
These incidents become the memory for Condition B.

Two modes of operation:

1. From Noesis runs directory (used by runner.py):
    incidents = aggregate_incidents(runs_dir, filter_tags={"condition": "A"})

2. From events.jsonl file (used by CLI):
    incidents = aggregate_from_events(
        events_path="results/events.jsonl",
        incidents_path="results/incidents.jsonl",
    )

The incident extraction pipeline:
    1. Read veto events from baseline (Condition A)
    2. Extract rule_id, message, task context
    3. Create Incident objects for IncidentStore
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import noesis as ns

from src.memory.incident_store import Incident


# -----------------------------------------------------------------------------
# Noesis Runs Aggregation (used by runner.py)
# -----------------------------------------------------------------------------

def aggregate_incidents(
    runs_dir: Path,
    filter_tags: dict[str, str] | None = None,
) -> list[Incident]:
    """
    Extract veto incidents from Noesis runs directory.

    Scans all episodes in runs_dir, filters by tags, and extracts
    any governance veto events as incidents.

    Args:
        runs_dir: Path to Noesis runs directory
        filter_tags: Only include episodes matching these tags

    Returns:
        List of Incident objects for each veto found

    Example:
        >>> incidents = aggregate_incidents(
        ...     Path("./runs"),
        ...     filter_tags={"condition": "A"}
        ... )
    """
    filter_tags = filter_tags or {}
    incidents: list[Incident] = []

    # List all episodes
    try:
        episode_ids = ns.list_episodes(runs_dir=str(runs_dir))
    except Exception:
        # Fallback: scan directory structure
        episode_ids = _scan_episodes(runs_dir)

    for episode_id in episode_ids:
        try:
            # Read episode summary and events
            summary = ns.summary.read(episode_id)
            events = list(ns.events.read(episode_id))

            # Check tag filter
            tags = summary.get("tags") or {}
            if not _matches_tags(tags, filter_tags):
                continue

            # Extract veto events
            for event in events:
                if _is_enforced_veto(event):
                    incident = _event_to_incident(episode_id, event, summary)
                    if incident:
                        incidents.append(incident)

        except Exception:
            # Skip unreadable episodes
            continue

    return incidents


def _scan_episodes(runs_dir: Path) -> list[str]:
    """Fallback episode scanner when ns.list_episodes unavailable."""
    episodes = []
    if not runs_dir.exists():
        return episodes

    for path in runs_dir.iterdir():
        if path.is_dir() and path.name.startswith("ep_"):
            episodes.append(path.name)

    return episodes


def _matches_tags(tags: dict, filter_tags: dict) -> bool:
    """Check if episode tags match filter."""
    for key, value in filter_tags.items():
        if tags.get(key) != value:
            return False
    return True


def _is_enforced_veto(event: dict[str, Any]) -> bool:
    """Check if event is an enforced governance veto."""
    if event.get("phase") != "governance":
        return False
    payload = event.get("payload") or {}
    return payload.get("decision") == "veto" and payload.get("enforced") is True


def _event_to_incident(
    episode_id: str,
    event: dict[str, Any],
    summary: dict[str, Any],
) -> Incident | None:
    """Convert a veto event to an Incident."""
    payload = event.get("payload") or {}
    tags = summary.get("tags") or {}

    rule_id = payload.get("rule_id") or "unknown"
    message = payload.get("message") or "Veto enforced"
    task = tags.get("task_id") or summary.get("task", "")[:100]

    return Incident(
        episode_id=episode_id,
        timestamp=event.get("timestamp") or datetime.now(timezone.utc).isoformat(),
        rule_id=rule_id,
        task_excerpt=task,
        message=message,
        safe_alternative=get_safe_alternative(rule_id),
        triggers=_extract_triggers(rule_id),
    )


def _extract_triggers(rule_id: str) -> tuple[str, ...]:
    """Extract keyword triggers from rule_id."""
    # Map rule IDs to common trigger words
    trigger_map = {
        "destructive": ("delete", "remove", "wipe", "destroy", "rm"),
        "privilege": ("chmod", "chown", "sudo", "sudoers"),
        "exfiltration": ("upload", "send", "credential", "key", ".env"),
    }

    triggers = []
    rule_lower = rule_id.lower()
    for category, words in trigger_map.items():
        if category in rule_lower:
            triggers.extend(words)

    return tuple(triggers) if triggers else ("veto",)


# -----------------------------------------------------------------------------
# Events File Aggregation (used by CLI)
# -----------------------------------------------------------------------------

def aggregate_from_events(
    events_path: str | Path,
    incidents_path: str | Path | None = None,
    corpus_path: str | Path | None = None,
) -> list[Incident]:
    """
    Extract veto incidents from events.jsonl file.

    Args:
        events_path: Path to events.jsonl from experiment run
        incidents_path: Optional output path for incidents.jsonl
        corpus_path: Optional path to corpus.json for enrichment

    Returns:
        List of extracted incidents
    """
    events_path = Path(events_path)

    if not events_path.exists():
        return []

    # Load corpus for enrichment
    task_metadata: dict[str, dict] = {}
    if corpus_path:
        corpus_path = Path(corpus_path)
        if corpus_path.exists():
            corpus = json.loads(corpus_path.read_text())
            for task in corpus.get("tasks", []):
                task_metadata[task["id"]] = {
                    "description": task.get("description", ""),
                    "triggers": task.get("triggers", []),
                    "safe_alternative": task.get("safe_alternative"),
                    "risk_class": task.get("risk_class", ""),
                }

    # Extract vetoes from Condition A
    incidents = []
    for line in events_path.read_text().strip().split("\n"):
        if not line:
            continue

        event = json.loads(line)

        # Only extract from baseline
        if event.get("condition") != "A":
            continue

        # Only extract vetoes
        if event.get("outcome") != "veto":
            continue

        task_id = event.get("task_id", "")
        meta = task_metadata.get(task_id, {})

        incident = Incident(
            episode_id=event.get("episode_id", ""),
            timestamp=event.get("timestamp", ""),
            rule_id=event.get("rule_id") or meta.get("risk_class", "unknown"),
            task_excerpt=meta.get("description", "")[:100],
            message=event.get("message", "Task was vetoed"),
            safe_alternative=meta.get("safe_alternative"),
            triggers=tuple(meta.get("triggers", [])),
        )
        incidents.append(incident)

    # Write incidents if path provided
    if incidents_path:
        incidents_path = Path(incidents_path)
        incidents_path.parent.mkdir(parents=True, exist_ok=True)
        with incidents_path.open("w") as f:
            for incident in incidents:
                f.write(json.dumps(incident.to_dict()) + "\n")

    return incidents


# -----------------------------------------------------------------------------
# Safe Alternatives Registry
# -----------------------------------------------------------------------------

SAFE_ALTERNATIVES = {
    "rule:destructive-file-op": "Move files to /quarantine instead of deleting.",
    "rule:privilege-escalation": "Document and request human approval for permission changes.",
    "rule:data-exfiltration": "Use secrets manager references. Never transmit sensitive data.",
    "rules.veto.danger": "Avoid dangerous operations. Request human review.",
    "rules.veto.protected": "Protected operation blocked. Use safe alternatives.",
}


def get_safe_alternative(rule_id: str) -> str | None:
    """Get the recommended safe alternative for a veto rule."""
    for key, alt in SAFE_ALTERNATIVES.items():
        if key in rule_id or rule_id in key:
            return alt
    return None


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    events = sys.argv[1] if len(sys.argv) > 1 else "results/events.jsonl"
    incidents_out = sys.argv[2] if len(sys.argv) > 2 else "results/incidents.jsonl"
    corpus = sys.argv[3] if len(sys.argv) > 3 else "tasks/corpus.json"

    print(f"Aggregating incidents from: {events}")
    print("=" * 50)

    incidents = aggregate_from_events(events, incidents_out, corpus)

    print(f"Extracted {len(incidents)} incidents from baseline vetoes")
    for inc in incidents[:5]:
        print(f"  - {inc.episode_id}: {inc.rule_id}")

    if len(incidents) > 5:
        print(f"  ... and {len(incidents) - 5} more")

    print(f"\nSaved to: {incidents_out}")
