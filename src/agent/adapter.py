"""
Noesis adapter for LangGraph agents.

This is the bridge between your LangGraph graph and ns.solve(...).

Key rule:
- Do NOT manually emit governance events from the adapter.
- Governance must happen at the side-effect boundary: ns.governed_act(kind="shell", ...).
"""

from __future__ import annotations

from typing import Any, Callable, TYPE_CHECKING

if TYPE_CHECKING:
    from src.agent.graph import PlanActAgent


class NoesisGraphWrapper:
    """
    Wrapper that exposes __noesis_input_mapper__ for Noesis AdapterActuator.
    """

    __slots__ = ("graph", "__noesis_input_mapper__")

    def __init__(
        self,
        graph: Any,
        input_mapper: Callable[[str], dict[str, Any]] | None,
    ) -> None:
        self.graph = graph
        self.__noesis_input_mapper__ = input_mapper

    def invoke(self, payload: Any) -> Any:
        if hasattr(self.graph, "invoke"):
            return self.graph.invoke(payload)
        if hasattr(self.graph, "run"):
            return self.graph.run(payload)
        if callable(self.graph):
            return self.graph(payload)
        raise TypeError("object is neither runnable nor callable")

    def run(self, payload: Any) -> Any:
        return self.invoke(payload)

    def __call__(self, payload: Any) -> Any:
        return self.invoke(payload)

    def execute(
        self,
        *,
        task: str,
        episode_id: str,
        run_dir: Any,
        intuition: Any | None = None,
        seed: int = 0,
        tags: dict[str, Any] | None = None,
    ) -> Any:
        """
        Noesis adapter entrypoint to preserve episode/run context in the graph state.
        """
        _ = (intuition, seed)
        payload = self.__noesis_input_mapper__(task) if callable(self.__noesis_input_mapper__) else {"task": task}
        if not isinstance(payload, dict):
            payload = {"task": task}
        payload.setdefault("task", task)
        payload["episode_id"] = episode_id
        payload["run_dir"] = str(run_dir)
        if tags:
            payload["tags"] = dict(tags)
        return self.invoke(payload)


def build_adapter(
    agent: "PlanActAgent",
    input_mapper: Callable[[str], dict[str, Any]] | None = None,
) -> NoesisGraphWrapper:
    """
    Wrap a PlanActAgent's LangGraph for Noesis.

    Pass the returned wrapper to ns.solve(using=...).

    Default input shape: {"task": task}
    """
    from src.agent.graph import build_graph

    app = build_graph(agent)

    if input_mapper is None:

        def input_mapper(task: str) -> dict[str, Any]:
            return {"task": task}

    return NoesisGraphWrapper(app, input_mapper)


def create_adapter(model: str | None = None) -> NoesisGraphWrapper:
    """
    One-liner to create a ready-to-use Noesis wrapper for ns.solve(using=...).
    """
    from src.agent.graph import build_agent

    agent = build_agent(model=model)
    return build_adapter(agent)
