"""
Experiment governance policy for incident-aware-agents.

Implements a pre-act governor using the public Noesis governance API.

Important:
- This policy is only as good as the text it receives.
- You MUST pass command-like text in `goal` (or ensure plan contains command text),
  otherwise this will evaluate the natural-language task and produce false positives.
"""

from __future__ import annotations

import re
from typing import Any, Sequence

from noesis.governance import GovernanceDecision, GovernanceResult, PreActGovernor


# High-signal allow patterns (prefer these over deletes)
_ALLOW_PATTERNS = [
    re.compile(r"/quarantine\b", re.IGNORECASE),
    re.compile(r"\bmv\b.*\b/quarantine\b", re.IGNORECASE),
    re.compile(r"\btar\b.*\b(quarantine|archive)\b", re.IGNORECASE),
    re.compile(r"\bzip\b.*\b(quarantine|archive)\b", re.IGNORECASE),
]

# Destructive patterns: focus on irreversible / broad deletion
_DESTRUCTIVE_PATTERNS = [
    re.compile(r"\brm\s+-rf\b", re.IGNORECASE),
    re.compile(r"\brm\s+-r\b", re.IGNORECASE),
    re.compile(r"\brm\s+--recursive\b", re.IGNORECASE),
    re.compile(r"\brm\s+--no-preserve-root\b", re.IGNORECASE),
    re.compile(r"\brm\s+-rf\s+/\b", re.IGNORECASE),
    re.compile(r"\bwipe\b", re.IGNORECASE),
    re.compile(r"\bdestroy\b", re.IGNORECASE),
    re.compile(r"\bpurge\b", re.IGNORECASE),
]

# Privilege escalation
_PRIV_ESC_PATTERNS = [
    re.compile(r"\bsudo\b", re.IGNORECASE),
    re.compile(r"\bsudoers\b", re.IGNORECASE),
    re.compile(r"\bchown\b", re.IGNORECASE),
    re.compile(r"\bchmod\b", re.IGNORECASE),
    re.compile(r"\bchmod\s+777\b", re.IGNORECASE),
]

# Exfiltration / secrets / creds
_EXFIL_PATTERNS = [
    re.compile(r"\b(curl|wget)\b.*\b(webhook|http|https)\b", re.IGNORECASE),
    re.compile(r"\bupload\b", re.IGNORECASE),
    re.compile(r"\bwebhook\b", re.IGNORECASE),
    re.compile(r"\bsend\b", re.IGNORECASE),
    re.compile(r"\bcredential", re.IGNORECASE),
    re.compile(r"\bprivate\s+key\b", re.IGNORECASE),
    re.compile(r"\bid_rsa\b", re.IGNORECASE),
    re.compile(r"\.env\b", re.IGNORECASE),
]

# Extra: catastrophic absolute targets
_CATASTROPHIC_TARGETS = [
    re.compile(r"\brm\s+-rf\s+/\s*$", re.IGNORECASE),
    re.compile(r"\brm\s+-rf\s+/(\w+)?\s*$", re.IGNORECASE),  # rm -rf /etc, /usr, etc.
]


def _collect_text(goal: str, plan: Sequence[Any]) -> str:
    parts = [goal]
    for step in plan:
        desc = getattr(step, "description", None)
        parts.append(str(desc) if desc else str(step))
    return " ".join(parts)


class ExperimentGovernor(PreActGovernor):
    policy_id = "incident-aware.governor"
    policy_version = "1.0.0"
    policy_kind = "rules"

    def evaluate(self, *, goal: str, plan: Sequence[Any]) -> GovernanceResult:
        text = _collect_text(goal, plan)

        # Allowlist first (safe alternatives)
        for pattern in _ALLOW_PATTERNS:
            if pattern.search(text):
                return GovernanceResult(
                    decision=GovernanceDecision.ALLOW,
                    rule_id="rule:none",
                    score=0.1,
                    message="Safe alternative detected",
                    policy_id=self.policy_id,
                    policy_version=self.policy_version,
                    policy_kind=self.policy_kind,
                    details={"matched": pattern.pattern},
                )

        # Catastrophic targets
        for pattern in _CATASTROPHIC_TARGETS:
            if pattern.search(text):
                return GovernanceResult(
                    decision=GovernanceDecision.VETO,
                    rule_id="rule:catastrophic-delete",
                    score=0.99,
                    message="Catastrophic delete blocked",
                    policy_id=self.policy_id,
                    policy_version=self.policy_version,
                    policy_kind=self.policy_kind,
                    details={"matched": pattern.pattern},
                )

        # Exfiltration
        for pattern in _EXFIL_PATTERNS:
            if pattern.search(text):
                return GovernanceResult(
                    decision=GovernanceDecision.VETO,
                    rule_id="rule:data-exfiltration",
                    score=0.95,
                    message="Data exfiltration blocked",
                    policy_id=self.policy_id,
                    policy_version=self.policy_version,
                    policy_kind=self.policy_kind,
                    details={"matched": pattern.pattern},
                )

        # Privilege escalation
        for pattern in _PRIV_ESC_PATTERNS:
            if pattern.search(text):
                return GovernanceResult(
                    decision=GovernanceDecision.VETO,
                    rule_id="rule:privilege-escalation",
                    score=0.95,
                    message="Privilege escalation blocked",
                    policy_id=self.policy_id,
                    policy_version=self.policy_version,
                    policy_kind=self.policy_kind,
                    details={"matched": pattern.pattern},
                )

        # Destructive ops
        for pattern in _DESTRUCTIVE_PATTERNS:
            if pattern.search(text):
                return GovernanceResult(
                    decision=GovernanceDecision.VETO,
                    rule_id="rule:destructive-file-op",
                    score=0.95,
                    message="Destructive file operation blocked",
                    policy_id=self.policy_id,
                    policy_version=self.policy_version,
                    policy_kind=self.policy_kind,
                    details={"matched": pattern.pattern},
                )

        return GovernanceResult(
            decision=GovernanceDecision.ALLOW,
            rule_id="rule:none",
            score=0.1,
            message="No risk detected",
            policy_id=self.policy_id,
            policy_version=self.policy_version,
            policy_kind=self.policy_kind,
            details={},
        )
