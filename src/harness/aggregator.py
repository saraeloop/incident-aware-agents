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
    *,
    corpus_path: Path | None = None,
) -> list[Incident]:
    """
    Extract veto incidents from Noesis runs directory.

    Scans all episodes in runs_dir, filters by tags, and extracts
    any governance veto events as incidents.

    Args:
        runs_dir: Path to Noesis runs directory
        filter_tags: Only include episodes matching these tags
        corpus_path: Optional path to corpus.json for task text enrichment

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
    task_metadata = _load_corpus_metadata(corpus_path)

    # List all episodes
    try:
        rows = ns.list_runs(limit=100000)
        episode_ids = [row.get("episode_id") for row in rows if row.get("episode_id")]
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

            # Extract veto events (prefer adapter result; fallback to governance events)
            veto_events = _extract_veto_events(events)
            if veto_events:
                for veto in veto_events:
                    incident = _veto_to_incident(
                        episode_id,
                        veto,
                        summary,
                        task_metadata=task_metadata,
                    )
                    if incident:
                        incidents.append(incident)
            else:
                for event in events:
                    if _is_enforced_veto(event):
                        incident = _event_to_incident(
                            episode_id,
                            event,
                            summary,
                            task_metadata=task_metadata,
                        )
                        if incident:
                            incidents.append(incident)

        except Exception:
            # Skip unreadable episodes
            continue

    return _dedupe_incidents(incidents)


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
    *,
    task_metadata: dict[str, dict[str, Any]] | None = None,
) -> Incident | None:
    """Convert a veto event to an Incident."""
    payload = event.get("payload") or {}
    tags = summary.get("tags") or {}

    rule_id = str(payload.get("rule_id") or "unknown")
    message = str(payload.get("message") or "Veto enforced")
    task_id = str(tags.get("task_id") or "")
    run_id = str(tags.get("run_id") or "") or None
    expected_outcome = str(tags.get("expected_outcome") or "") or None
    risk_class = None
    if task_id and task_id in (task_metadata or {}):
        risk_class = task_metadata[task_id].get("risk_class")
    task_text = _get_task_text(task_id, summary, task_metadata or {})
    task_excerpt = (task_text or task_id)[:100]
    safe_alt, safe_alt_source = _resolve_safe_alternative(rule_id, task_id, task_metadata or {})
    triggers = _resolve_triggers(rule_id, task_id, task_metadata or {})

    return Incident(
        episode_id=episode_id,
        task_id=task_id or None,
        run_id=run_id,
        expected_outcome=expected_outcome,
        risk_class=str(risk_class) if risk_class else None,
        timestamp=event.get("timestamp") or datetime.now(timezone.utc).isoformat(),
        rule_id=rule_id,
        task_excerpt=task_excerpt,
        message=message,
        safe_alternative=safe_alt,
        safe_alternative_source=safe_alt_source,
        triggers=triggers,
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


def _load_corpus_metadata(corpus_path: Path | None) -> dict[str, dict[str, Any]]:
    if not corpus_path:
        return {}
    if not corpus_path.exists():
        return {}
    corpus = json.loads(corpus_path.read_text())
    metadata: dict[str, dict[str, Any]] = {}
    for task in corpus.get("tasks", []):
        metadata[str(task["id"])] = {
            "description": task.get("description", ""),
            "triggers": task.get("triggers", []),
            "safe_alternative": task.get("safe_alternative"),
            "risk_class": task.get("risk_class", ""),
        }
    return metadata


def _get_task_text(task_id: str, summary: dict[str, Any], metadata: dict[str, dict[str, Any]]) -> str:
    if task_id and task_id in metadata:
        return str(metadata[task_id].get("description") or "")
    task_text = summary.get("task")
    return str(task_text or "")


def _resolve_triggers(
    rule_id: str,
    task_id: str,
    metadata: dict[str, dict[str, Any]],
) -> tuple[str, ...]:
    if task_id and task_id in metadata:
        triggers = metadata[task_id].get("triggers", [])
        if triggers:
            return tuple(str(t) for t in triggers)
    return _extract_triggers(rule_id)


def _resolve_safe_alternative(
    rule_id: str,
    task_id: str,
    metadata: dict[str, dict[str, Any]],
) -> tuple[str | None, str | None]:
    if task_id and task_id in metadata:
        alt = metadata[task_id].get("safe_alternative")
        if alt:
            return str(alt), "corpus"
    registry_alt = get_safe_alternative(rule_id)
    if registry_alt:
        return registry_alt, "registry"
    return None, None


def _extract_veto_events(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Extract veto records from adapter terminate payloads.
    Returns a list of dicts with keys: timestamp, rule_id, message.
    """
    terminate = next((e for e in reversed(events) if e.get("phase") == "terminate"), None)
    if not terminate:
        return []
    payload = terminate.get("payload") or {}
    message = payload.get("message")
    if not isinstance(message, str):
        return []
    try:
        import ast

        data = ast.literal_eval(message)
    except Exception:
        return []
    if not isinstance(data, dict):
        return []
    result = data.get("result")
    if not isinstance(result, dict):
        return []
    veto_records: list[dict[str, Any]] = []
    results = result.get("results")
    if not isinstance(results, list):
        return []
    for entry in results:
        if not isinstance(entry, dict):
            continue
        if entry.get("status") != "vetoed":
            continue
        veto = entry.get("veto") or {}
        rule_id = veto.get("rule_id") or "unknown"
        message = veto.get("advice") or "Task blocked by governance policy"
        veto_records.append(
            {
                "timestamp": terminate.get("timestamp") or datetime.now(timezone.utc).isoformat(),
                "rule_id": rule_id,
                "message": message,
            }
        )
    return veto_records


def _extract_tags_from_events(events: list[dict[str, Any]]) -> dict[str, Any]:
    for event in events:
        tags = event.get("tags")
        if isinstance(tags, dict) and tags:
            return tags
        payload = event.get("payload") or {}
        if isinstance(payload.get("tags"), dict):
            return payload.get("tags") or {}
        snapshot = payload.get("experimental", {}).get("snapshot", {})
        if isinstance(snapshot.get("tags"), dict):
            return snapshot.get("tags") or {}
        state = snapshot.get("state", {})
        if isinstance(state.get("episode", {}).get("tags"), dict):
            return state.get("episode", {}).get("tags") or {}
    return {}


def _veto_to_incident(
    episode_id: str,
    veto: dict[str, Any],
    summary: dict[str, Any],
    *,
    task_metadata: dict[str, dict[str, Any]] | None = None,
) -> Incident | None:
    tags = summary.get("tags") or {}
    task_id = str(tags.get("task_id") or "")
    run_id = str(tags.get("run_id") or "") or None
    expected_outcome = str(tags.get("expected_outcome") or "") or None
    risk_class = None
    if task_id and task_id in (task_metadata or {}):
        risk_class = task_metadata[task_id].get("risk_class")
    task_text = _get_task_text(task_id, summary, task_metadata or {})
    task_excerpt = (task_text or task_id)[:100]
    rule_id = str(veto.get("rule_id") or "unknown")
    safe_alt, safe_alt_source = _resolve_safe_alternative(rule_id, task_id, task_metadata or {})
    triggers = _resolve_triggers(rule_id, task_id, task_metadata or {})

    return Incident(
        episode_id=episode_id,
        task_id=task_id or None,
        run_id=run_id,
        expected_outcome=expected_outcome,
        risk_class=str(risk_class) if risk_class else None,
        timestamp=str(veto.get("timestamp") or datetime.now(timezone.utc).isoformat()),
        rule_id=rule_id,
        task_excerpt=task_excerpt,
        message=str(veto.get("message") or "Task blocked by governance policy"),
        safe_alternative=safe_alt,
        safe_alternative_source=safe_alt_source,
        triggers=triggers,
    )


def _dedupe_incidents(incidents: list[Incident]) -> list[Incident]:
    by_task: dict[str, Incident] = {}
    for incident in incidents:
        task_key = incident.task_id or incident.task_excerpt
        if not task_key:
            task_key = incident.episode_id
        current = by_task.get(task_key)
        if not current:
            by_task[task_key] = incident
            continue
        if incident.timestamp and current.timestamp and incident.timestamp > current.timestamp:
            by_task[task_key] = incident
    return list(by_task.values())


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
    task_metadata = _load_corpus_metadata(Path(corpus_path) if corpus_path else None)

    # Extract vetoes from Condition A
    incidents: list[Incident] = []
    raw_lines = [line for line in events_path.read_text().splitlines() if line.strip()]
    if not raw_lines:
        return incidents

    parsed = [json.loads(line) for line in raw_lines]
    if parsed and "phase" in parsed[0]:
        by_episode: dict[str, list[dict[str, Any]]] = {}
        for event in parsed:
            episode_id = event.get("episode_id")
            if not episode_id:
                continue
            by_episode.setdefault(str(episode_id), []).append(event)

        for episode_id, events in by_episode.items():
            tags = _extract_tags_from_events(events)
            if tags.get("condition") != "A":
                continue
            summary = {"tags": tags, "task": tags.get("task")}
            veto_events = _extract_veto_events(events)
            if veto_events:
                for veto in veto_events:
                    incident = _veto_to_incident(
                        episode_id,
                        veto,
                        summary,
                        task_metadata=task_metadata,
                    )
                    if incident:
                        incidents.append(incident)
            else:
                for event in events:
                    if _is_enforced_veto(event):
                        incident = _event_to_incident(
                            episode_id,
                            event,
                            summary,
                            task_metadata=task_metadata,
                        )
                        if incident:
                            incidents.append(incident)
    else:
        for event in parsed:
            if event.get("condition") != "A":
                continue
            if event.get("outcome") != "veto":
                continue
            task_id = str(event.get("task_id", ""))
            meta = task_metadata.get(task_id, {})
            safe_alt, safe_alt_source = _resolve_safe_alternative(
                str(event.get("rule_id") or meta.get("risk_class", "unknown")),
                task_id,
                task_metadata,
            )
            triggers = _resolve_triggers(
                str(event.get("rule_id") or meta.get("risk_class", "unknown")),
                task_id,
                task_metadata,
            )
            incident = Incident(
                episode_id=event.get("episode_id", ""),
                task_id=task_id or None,
                run_id=str(event.get("run_id") or "") or None,
                expected_outcome=str(event.get("expected_outcome") or "") or None,
                risk_class=str(meta.get("risk_class") or "") or None,
                timestamp=event.get("timestamp", ""),
                rule_id=event.get("rule_id") or meta.get("risk_class", "unknown"),
                task_excerpt=str(meta.get("description", "") or "")[:100],
                message=event.get("message", "Task was vetoed"),
                safe_alternative=safe_alt,
                safe_alternative_source=safe_alt_source,
                triggers=triggers,
            )
            incidents.append(incident)

    # Write incidents if path provided
    if incidents_path:
        incidents_path = Path(incidents_path)
        incidents_path.parent.mkdir(parents=True, exist_ok=True)
        with incidents_path.open("w") as f:
            for incident in _dedupe_incidents(incidents):
                f.write(json.dumps(incident.to_dict()) + "\n")

    return _dedupe_incidents(incidents)


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
