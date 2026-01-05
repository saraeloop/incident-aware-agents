"""
Noesis adapter for LangGraph agents.

Wraps the plan→act LangGraph agent for use with Noesis governance.
This is the bridge between the agent and ns.solve().

Usage:
    from src.agent.adapter import create_adapter
    from src.policies.incident_intuition import IncidentAwareIntuition

    adapter = create_adapter()
    ns.solve(task, using=adapter, intuition=intuition)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable

from noesis.adapters import LangGraphAdapter

if TYPE_CHECKING:
    from src.agent.graph import PlanActAgent


# -----------------------------------------------------------------------------
# Adapter Factory
# -----------------------------------------------------------------------------

def build_adapter(
    agent: "PlanActAgent",
    input_mapper: Callable[[str], dict[str, Any]] | None = None,
) -> LangGraphAdapter:
    """
    Wrap a PlanActAgent's LangGraph for Noesis.

    This is what you pass to ns.solve(using=adapter).

    Args:
        agent: The PlanActAgent to wrap
        input_mapper: Optional custom input mapper (default: {"task": task})

    Returns:
        LangGraphAdapter ready for ns.solve()

    Example:
        >>> agent = build_agent()
        >>> adapter = build_adapter(agent)
        >>> ns.solve("Delete temp files", using=adapter)
    """
    from src.agent.graph import build_graph

    app = build_graph(agent)

    if input_mapper is None:
        def input_mapper(task: str) -> dict[str, Any]:
            return {"task": task}

    return LangGraphAdapter(app, input_mapper=input_mapper)


def create_adapter(model: str | None = None) -> LangGraphAdapter:
    """
    One-liner to create a ready-to-use adapter.

    Args:
        model: Optional LLM model override (default: gpt-4o-mini)

    Returns:
        LangGraphAdapter configured with PlanActAgent

    Example:
        >>> adapter = create_adapter()
        >>> ns.solve("List Python files", using=adapter)
    """
    from src.agent.graph import build_agent

    return build_adapter(build_agent(model))


# -----------------------------------------------------------------------------
# Adapter Configuration
# -----------------------------------------------------------------------------

class AdapterConfig:
    """
    Configuration for adapter behavior.

    Allows customizing how tasks are mapped to agent input
    and how results are extracted.
    """

    def __init__(
        self,
        model: str | None = None,
        timeout: int = 60,
        sandbox_mode: str = "docker",
    ) -> None:
        self.model = model
        self.timeout = timeout
        self.sandbox_mode = sandbox_mode

    def create_adapter(self) -> LangGraphAdapter:
        """Create adapter with this configuration."""
        return create_adapter(model=self.model)
