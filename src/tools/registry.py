"""Tool registry and execution wrapper for the Agentic RAG system."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class ToolResult:
    """Standardized output returned by any agent tool."""

    tool_name: str
    success: bool
    summary: str
    data: Any = None
    sources: list[dict] = field(default_factory=list)


@dataclass
class ToolSpec:
    """Metadata and callable for a registered agent tool."""

    name: str
    description: str
    parameters: dict[str, str]
    fn: Callable[..., ToolResult]


class ToolRegistry:
    """Registry that stores, describes, and invokes agent tools."""

    def __init__(self) -> None:
        self._tools: dict[str, ToolSpec] = {}

    def register(self, spec: ToolSpec) -> None:
        self._tools[spec.name] = spec

    def get(self, name: str) -> ToolSpec | None:
        return self._tools.get(name)

    def list_tools(self) -> list[ToolSpec]:
        return list(self._tools.values())

    def describe_for_prompt(self) -> str:
        lines = []
        for spec in self._tools.values():
            params = ", ".join(f"{k}: {v}" for k, v in spec.parameters.items())
            lines.append(f"- {spec.name}({params}): {spec.description}")
        return "\n".join(lines)

    def invoke(self, name: str, **kwargs: Any) -> ToolResult:
        spec = self._tools.get(name)
        if spec is None:
            return ToolResult(
                tool_name=name,
                success=False,
                summary=f"Unknown tool '{name}'. Available: {list(self._tools.keys())}",
            )
        try:
            return spec.fn(**kwargs)
        except Exception as exc:
            return ToolResult(
                tool_name=name,
                success=False,
                summary=f"Tool '{name}' failed: {exc}",
            )

