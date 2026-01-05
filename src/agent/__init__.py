"""
Agent module for incident-aware-agents.

Provides LangGraph-based plan→act agent with Noesis adapter.

Usage:
    from src.agent import build_agent, create_adapter

    agent = build_agent()
    adapter = create_adapter()
    ns.solve(task, using=adapter, intuition=...)
"""

from src.agent.graph import PlanActAgent, build_agent, build_graph
from src.agent.adapter import build_adapter, create_adapter

__all__ = [
    "PlanActAgent",
    "build_agent",
    "build_graph",
    "build_adapter",
    "create_adapter",
]
