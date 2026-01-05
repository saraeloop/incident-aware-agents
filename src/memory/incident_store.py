"""
Incident Store: Memory backend for veto incidents.

This is the "memory" that enables Condition B. It stores past vetoes
and retrieves relevant ones based on task similarity.

Architecture:
    
    Condition A (baseline)
           ↓
    Governance vetoes → events.jsonl
           ↓
    Aggregator extracts vetoes
           ↓
    IncidentStore (this file)
           ↓
    Condition B queries store → IncidentAwareIntuition

The store is intentionally simple: keyword matching, no embeddings.
This keeps the experiment focused on the feedback loop, not retrieval quality.

Usage:
    from memory.incident_store import IncidentStore, Incident
    
    # Load from file
    store = IncidentStore.from_file(Path("incidents.jsonl"))
    
    # Query
    incidents = store.query("delete all temp files", k=3)
    
    # Add new
    store.add(Incident(...))
    
    # Save
    store.to_file(Path("incidents.jsonl"))
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


@dataclass(frozen=True, slots=True)
class Incident:
    """
    A recorded veto incident from a previous episode.
    
    This is what the agent "remembers" about past failures.
    
    Attributes:
        episode_id: The Noēsis episode where this veto occurred
        timestamp: When the veto happened
        rule_id: Which governance rule triggered (e.g., "rule:destructive-file-op")
        task_excerpt: What the agent was trying to do
        message: The governance message explaining why
        safe_alternative: What the agent should do instead
        triggers: Keywords that match this incident to future tasks
    
    Example:
        Incident(
            episode_id="ep_01JG...",
            timestamp="2026-01-04T12:00:00Z",
            rule_id="rule:destructive-file-op",
            task_excerpt="Delete all temp files in /workspace/tmp",
            message="File deletion is forbidden",
            safe_alternative="Move to /quarantine instead",
            triggers=("delete", "temp", "tmp"),
        )
    """
    
    episode_id: str
    timestamp: str
    rule_id: str
    task_excerpt: str
    message: str
    safe_alternative: str | None = None
    triggers: tuple[str, ...] = ()
    
    def to_dict(self) -> dict:
        """Serialize for JSON storage."""
        return {
            "episode_id": self.episode_id,
            "timestamp": self.timestamp,
            "rule_id": self.rule_id,
            "task_excerpt": self.task_excerpt,
            "message": self.message,
            "safe_alternative": self.safe_alternative,
            "triggers": list(self.triggers),
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "Incident":
        """Deserialize from JSON."""
        return cls(
            episode_id=data["episode_id"],
            timestamp=data["timestamp"],
            rule_id=data["rule_id"],
            task_excerpt=data.get("task_excerpt", ""),
            message=data.get("message", ""),
            safe_alternative=data.get("safe_alternative"),
            triggers=tuple(data.get("triggers", [])),
        )


class IncidentStore:
    """
    In-memory store of past veto incidents.
    
    How it works:
    
    1. INDEXING: When incidents are loaded, we build a keyword index.
       Each trigger word maps to the incidents it appears in.
       
       Index example:
           "delete" → [Incident A, Incident B]
           "chmod"  → [Incident C]
           "tmp"    → [Incident A]
    
    2. QUERYING: When a new task comes in, we scan for keyword matches.
       "wipe the tmp directory" matches "tmp" → returns Incident A
       
    3. RANKING: First match wins (by insertion order). 
       Production would use embeddings + cosine similarity.
    
    Why keyword matching?
        - Simple to debug
        - Deterministic (same query → same results)
        - Sufficient to prove the feedback loop works
        - Embeddings are an optimization, not the hypothesis
    """
    
    def __init__(self, incidents: Sequence[Incident] = ()) -> None:
        self._incidents: list[Incident] = list(incidents)
        self._keyword_index: dict[str, list[Incident]] = {}
        self._build_index()
    
    def _build_index(self) -> None:
        """
        Build keyword → incident mapping.
        
        Sources of keywords:
        1. Explicit triggers from the incident
        2. Words from the rule_id (e.g., "destructive", "file", "op")
        """
        self._keyword_index.clear()
        
        for incident in self._incidents:
            # Index by explicit triggers
            for trigger in incident.triggers:
                key = trigger.lower()
                self._keyword_index.setdefault(key, []).append(incident)
            
            # Index by rule_id words (skip short ones)
            rule_words = incident.rule_id.replace("rule:", "").replace("-", " ").split()
            for word in rule_words:
                if len(word) > 3:
                    self._keyword_index.setdefault(word.lower(), []).append(incident)
    
    def query(self, task: str, k: int = 3) -> list[Incident]:
        """
        Find incidents relevant to a task.
        
        Args:
            task: The task description to match against
            k: Maximum number of incidents to return
            
        Returns:
            List of matching incidents, ordered by first match
            
        Example:
            >>> store.query("wipe the cache directory")
            [Incident(rule_id="rule:destructive-file-op", ...)]
        """
        task_lower = task.lower()
        matches: dict[str, Incident] = {}  # episode_id → incident (deduped)
        
        for keyword, incidents in self._keyword_index.items():
            if keyword in task_lower:
                for incident in incidents:
                    # Keep first occurrence only
                    if incident.episode_id not in matches:
                        matches[incident.episode_id] = incident
        
        return list(matches.values())[:k]
    
    def add(self, incident: Incident) -> None:
        """
        Add a new incident to the store.
        
        Also updates the keyword index.
        """
        self._incidents.append(incident)
        
        # Update index
        for trigger in incident.triggers:
            key = trigger.lower()
            self._keyword_index.setdefault(key, []).append(incident)
        
        rule_words = incident.rule_id.replace("rule:", "").replace("-", " ").split()
        for word in rule_words:
            if len(word) > 3:
                self._keyword_index.setdefault(word.lower(), []).append(incident)
    
    def clear(self) -> None:
        """Remove all incidents."""
        self._incidents.clear()
        self._keyword_index.clear()
    
    @classmethod
    def from_file(cls, path: Path) -> "IncidentStore":
        """
        Load incidents from a JSONL file.
        
        File format (one JSON object per line):
            {"episode_id": "...", "rule_id": "...", ...}
            {"episode_id": "...", "rule_id": "...", ...}
        """
        incidents: list[Incident] = []
        
        if not path.exists():
            return cls(incidents)
        
        content = path.read_text().strip()
        if not content:
            return cls(incidents)
        
        for line in content.split("\n"):
            if line.strip():
                data = json.loads(line)
                incidents.append(Incident.from_dict(data))
        
        return cls(incidents)
    
    def to_file(self, path: Path) -> None:
        """
        Save incidents to a JSONL file.
        
        Creates parent directories if needed.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        
        lines = [json.dumps(i.to_dict()) for i in self._incidents]
        path.write_text("\n".join(lines) + "\n" if lines else "")
    
    def __len__(self) -> int:
        """Number of incidents in the store."""
        return len(self._incidents)
    
    def __iter__(self):
        """Iterate over all incidents."""
        return iter(self._incidents)
    
    def stats(self) -> dict:
        """
        Return statistics about the store.
        
        Useful for debugging and reporting.
        """
        rule_counts: dict[str, int] = {}
        for incident in self._incidents:
            rule_counts[incident.rule_id] = rule_counts.get(incident.rule_id, 0) + 1
        
        all_triggers: set[str] = set()
        for incident in self._incidents:
            all_triggers.update(incident.triggers)
        
        return {
            "total_incidents": len(self._incidents),
            "unique_rules": len(rule_counts),
            "rule_breakdown": rule_counts,
            "unique_triggers": len(all_triggers),
            "index_size": len(self._keyword_index),
        }