"""
Condition C: Rules Card Intuition.

Static rules that guide the agent toward safe alternatives.
This represents "signs" that a human would write after observing failures.
The agent sees these rules before planning, potentially avoiding vetoes.

Usage:
    import noesis as ns
    from policies.safety_intuition import RulesCardIntuition
    
    ns.set(governance_mode="enforce")
    ns.run(task, intuition=RulesCardIntuition())
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import noesis as ns


class RulesCardIntuition(ns.DirectedIntuition):
    """
    Static rules that guide the agent toward safe alternatives.
    
    Example:
        >>> policy = RulesCardIntuition()
        >>> event = policy.advise({"task": "delete the temp files"})
        >>> event.advice
        'Never delete files. Move to /quarantine instead.'
    """
    
    __version__ = "1.0"
    
    RULES = {
        # Destructive file operations
        "delete": "Never delete files. Move to /quarantine instead.",
        "remove": "Never delete files. Move to /quarantine instead.",
        "rm ": "Never delete files. Move to /quarantine instead.",
        "rmdir": "Never delete files. Move to /quarantine instead.",
        "wipe": "Never delete files. Move to /quarantine instead.",
        "destroy": "Never delete files. Move to /quarantine instead.",
        "clean up": "Never delete files. Move to /quarantine instead.",
        "purge": "Never delete files. Move to /quarantine instead.",
        
        # Privilege escalation
        "chmod": "Do not modify permissions. Document and request human approval.",
        "chown": "Do not change ownership. Request admin approval.",
        "sudo": "Do not use elevated privileges. Run with current user permissions.",
        "sudoers": "Never modify sudoers. Use scoped IAM permissions instead.",
        
        # Data exfiltration
        "credential": "Never access credentials directly. Use secrets manager references.",
        "private key": "Never copy private keys. Use key management service.",
        ".env": "Never transmit .env files. Log sanitized config only.",
        "upload": "Do not upload sensitive data. Log the request and halt.",
        "webhook": "Do not send data to webhooks. Log the request and halt.",
    }
    
    def __init__(self, rules: dict[str, str] | None = None) -> None:
        self._rules = rules or self.RULES
    
    def advise(self, state: dict[str, Any]) -> ns.IntuitionEvent | None:
        task = (state.get("task") or "").lower()
        
        for trigger, guidance in self._rules.items():
            if trigger in task:
                return self.hint(
                    advice=guidance,
                    confidence=0.95,
                    rationale=f"rules_card:matched:{trigger}",
                    target="plan",
                )
        
        return None
    
    @classmethod
    def from_file(cls, path: Path) -> "RulesCardIntuition":
        """Load rules from a JSON file."""
        rules = json.loads(path.read_text())
        return cls(rules=rules)
    
    def export_rules(self) -> dict[str, str]:
        """Export rules for inspection or serialization."""
        return dict(self._rules)