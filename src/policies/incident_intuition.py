"""
Condition B: Incident-Aware Intuition.

Queries past vetoes from incident memory to guide the agent.
This represents "learning from bruises" - the agent sees what was
vetoed before and receives guidance based on similar patterns.

Usage:
    import noesis as ns
    from policies.incident_intuition import IncidentAwareIntuition, IncidentStore
    
    store = IncidentStore.from_file(Path("incidents.jsonl"))
    ns.set(governance_mode="enforce")
    ns.run(task, intuition=IncidentAwareIntuition(store))
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import noesis as ns


# -----------------------------------------------------------------------------
# Incident Data Model
# -----------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class Incident:
    """A recorded veto incident from a previous episode."""
    
    episode_id: str
    timestamp: str
    rule_id: str
    task_excerpt: str
    message: str
    safe_alternative: str | None = None
    triggers: tuple[str, ...] = ()


# -----------------------------------------------------------------------------
# Incident Store
# -----------------------------------------------------------------------------

class IncidentStore:
    """
    In-memory store of past veto incidents.
    
    Provides keyword-based retrieval for the experiment.
    Production would use embeddings/FAISS, but keyword matching
    is sufficient to demonstrate the feedback loop.
    """
    
    def __init__(self, incidents: Sequence[Incident] = ()) -> None:
        self._incidents = list(incidents)
        self._keyword_index: dict[str, list[Incident]] = {}
        self._build_index()
    
    def _build_index(self) -> None:
        """Build keyword -> incident mapping for fast lookup."""
        self._keyword_index.clear()
        
        for incident in self._incidents:
            # Index by triggers
            for trigger in incident.triggers:
                key = trigger.lower()
                self._keyword_index.setdefault(key, []).append(incident)
            
            # Index by rule_id keywords
            rule_words = incident.rule_id.replace("rule:", "").replace("-", " ").split()
            for word in rule_words:
                if len(word) > 3:
                    self._keyword_index.setdefault(word.lower(), []).append(incident)
    
    def query(self, task: str, k: int = 3) -> list[Incident]:
        """Find incidents relevant to a task."""
        task_lower = task.lower()
        matches: dict[str, Incident] = {}
        
        for keyword, incidents in self._keyword_index.items():
            if keyword in task_lower:
                for incident in incidents:
                    if incident.episode_id not in matches:
                        matches[incident.episode_id] = incident
        
        return list(matches.values())[:k]
    
    def add(self, incident: Incident) -> None:
        """Add a new incident to the store."""
        self._incidents.append(incident)
        
        for trigger in incident.triggers:
            key = trigger.lower()
            self._keyword_index.setdefault(key, []).append(incident)
    
    @classmethod
    def from_file(cls, path: Path) -> "IncidentStore":
        """Load incidents from a JSONL file."""
        incidents: list[Incident] = []
        
        if not path.exists():
            return cls(incidents)
        
        for line in path.read_text().strip().split("\n"):
            if not line:
                continue
            data = json.loads(line)
            incidents.append(Incident(
                episode_id=data["episode_id"],
                timestamp=data["timestamp"],
                rule_id=data["rule_id"],
                task_excerpt=data.get("task_excerpt", ""),
                message=data.get("message", ""),
                safe_alternative=data.get("safe_alternative"),
                triggers=tuple(data.get("triggers", [])),
            ))
        
        return cls(incidents)
    
    def to_file(self, path: Path) -> None:
        """Save incidents to a JSONL file."""
        lines = []
        for i in self._incidents:
            lines.append(json.dumps({
                "episode_id": i.episode_id,
                "timestamp": i.timestamp,
                "rule_id": i.rule_id,
                "task_excerpt": i.task_excerpt,
                "message": i.message,
                "safe_alternative": i.safe_alternative,
                "triggers": list(i.triggers),
            }))
        path.write_text("\n".join(lines) + "\n" if lines else "")
    
    def __len__(self) -> int:
        return len(self._incidents)


# -----------------------------------------------------------------------------
# Intuition Policy
# -----------------------------------------------------------------------------

class IncidentAwareIntuition(ns.DirectedIntuition):
    """
    Queries past vetoes from incident memory to guide the agent.
    
    Unlike RulesCardIntuition (static rules), this policy:
    - Retrieves incidents dynamically based on task similarity
    - Includes evidence_ids linking to the original veto episodes
    - Can surface multiple relevant incidents
    
    Example:
        >>> store = IncidentStore.from_file(Path("incidents.jsonl"))
        >>> policy = IncidentAwareIntuition(store)
        >>> event = policy.advise({"task": "wipe the cache directory"})
        >>> "Previously vetoed" in event.advice
        True
    """
    
    __version__ = "1.0"
    
    def __init__(self, incident_store: IncidentStore) -> None:
        self._store = incident_store
    
    def advise(self, state: dict[str, Any]) -> ns.IntuitionEvent | None:
        task = state.get("task") or ""
        if not task:
            return None
        
        incidents = self._store.query(task, k=3)
        if not incidents:
            return None
        
        top = incidents[0]
        advice = (
            f"Previously vetoed: {top.task_excerpt}\n"
            f"Rule: {top.rule_id}\n"
            f"Guidance: {top.safe_alternative or top.message}"
        )
        
        evidence_ids = [i.episode_id for i in incidents]
        
        return self.hint(
            advice=advice,
            confidence=0.85,
            rationale=f"incident_memory:matched:{top.rule_id}",
            evidence_ids=evidence_ids,
            target="plan",
        )
    
    @property
    def incident_count(self) -> int:
        """Number of incidents in the store."""
        return len(self._store)


# -----------------------------------------------------------------------------
# Safe Alternatives (used by aggregator)
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