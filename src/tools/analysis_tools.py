"""Safe mathematical calculation and data analysis tools."""

from __future__ import annotations

import ast
import math
import operator
from typing import Any

import numpy as np
import pandas as pd

from src.pipeline.vector_store import VectorStore
from src.tools.registry import ToolResult, ToolSpec

_SAFE_FUNCS: dict[str, Any] = {
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
    "sum": sum,
    "pow": pow,
    "sqrt": math.sqrt,
    "log": math.log,
    "log2": math.log2,
    "log10": math.log10,
    "ceil": math.ceil,
    "floor": math.floor,
    "factorial": math.factorial,
    "comb": math.comb,
    "perm": math.perm,
    "pi": math.pi,
    "e": math.e,
}

_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}

_UNARY_OPS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


def _eval_ast(node: ast.AST) -> Any:
    if isinstance(node, ast.Expression):
        return _eval_ast(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.Name):
        if node.id in _SAFE_FUNCS and not callable(_SAFE_FUNCS[node.id]):
            return _SAFE_FUNCS[node.id]
        raise ValueError(f"Unsupported symbol: {node.id}")
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
        return _BIN_OPS[type(node.op)](_eval_ast(node.left), _eval_ast(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        return _UNARY_OPS[type(node.op)](_eval_ast(node.operand))
    if isinstance(node, ast.List):
        return [_eval_ast(elt) for elt in node.elts]
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        fname = node.func.id
        if fname in _SAFE_FUNCS and callable(_SAFE_FUNCS[fname]):
            args = [_eval_ast(a) for a in node.args]
            return _SAFE_FUNCS[fname](*args)
        raise ValueError(f"Unsupported function: {fname}")
    raise ValueError(f"Unsupported expression node: {type(node).__name__}")


def calculate(expression: str) -> ToolResult:
    """Safely evaluate arithmetic, logarithmic, combinatorial, and Big-O math expressions."""
    expr = expression.strip().replace("^", "**")
    try:
        tree = ast.parse(expr, mode="eval")
        value = _eval_ast(tree)
        if isinstance(value, float):
            formatted = f"{value:.6g}"
        else:
            formatted = str(value)
        return ToolResult(
            tool_name="calculate",
            success=True,
            summary=f"{expression} = {formatted}",
            data={"expression": expression, "result": value},
        )
    except Exception as exc:
        return ToolResult(
            tool_name="calculate",
            success=False,
            summary=f"Calculation error for '{expression}': {exc}",
        )


def analyze_data(
    operation: str,
    store: VectorStore | None = None,
    target: str | None = None,
    values: list[float] | None = None,
) -> ToolResult:
    """Perform structured data analysis over the knowledge base or numeric inputs.

    Supported operations:
    - 'corpus_stats': Per-document pages, chunks, and word-count statistics.
    - 'topic_frequency': Count how often `target` appears in each document.
    - 'descriptive_stats': Compute count, mean, median, std, min, max for `values`.
    - 'complexity_table': Compare Big-O operation counts for input sizes `values` (e.g. [10, 100, 1000]).
    """
    op = operation.lower().strip()

    if op == "corpus_stats":
        if store is None or not store.is_ready:
            return ToolResult("analyze_data", False, "Knowledge base is not loaded.")
        df = pd.DataFrame(
            [
                {
                    "filename": c.filename,
                    "page": c.page,
                    "words": c.word_count,
                    "chars": c.char_count,
                }
                for c in store.chunks
            ]
        )
        grouped = (
            df.groupby("filename")
            .agg(pages=("page", "nunique"), chunks=("words", "count"), total_words=("words", "sum"), avg_words=("words", "mean"))
            .reset_index()
        )
        grouped["avg_words"] = grouped["avg_words"].round(1)
        records = grouped.to_dict(orient="records")
        lines = [
            f"- {r['filename']}: {r['pages']} pages, {r['chunks']} chunks, {r['total_words']} words (avg {r['avg_words']} words/chunk)"
            for r in records
        ]
        return ToolResult(
            tool_name="analyze_data",
            success=True,
            summary="Corpus statistics:\n" + "\n".join(lines),
            data=records,
        )

    if op == "topic_frequency":
        if store is None or not store.is_ready:
            return ToolResult("analyze_data", False, "Knowledge base is not loaded.")
        term = (target or "").strip().lower()
        if not term:
            return ToolResult("analyze_data", False, "Provide 'target' term for topic_frequency.")
        counts: dict[str, int] = {}
        for c in store.chunks:
            occ = c.text.lower().count(term)
            if occ > 0:
                counts[c.filename] = counts.get(c.filename, 0) + occ
        sorted_counts = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
        total = sum(counts.values())
        summary = f"Occurrences of '{term}' across KB (total={total}): " + (
            ", ".join(f"{fn}: {cnt}" for fn, cnt in sorted_counts) if sorted_counts else "none"
        )
        return ToolResult("analyze_data", True, summary, data=dict(sorted_counts))

    if op == "descriptive_stats":
        if not values:
            return ToolResult("analyze_data", False, "Provide numeric 'values' list.")
        arr = np.array(values, dtype=float)
        stats = {
            "count": int(arr.size),
            "sum": float(np.sum(arr)),
            "mean": round(float(np.mean(arr)), 4),
            "median": round(float(np.median(arr)), 4),
            "std": round(float(np.std(arr)), 4),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
        }
        return ToolResult(
            tool_name="analyze_data",
            success=True,
            summary=f"Stats for {len(values)} values: mean={stats['mean']}, median={stats['median']}, std={stats['std']}, min={stats['min']}, max={stats['max']}",
            data=stats,
        )

    if op == "complexity_table":
        ns = values or [10, 100, 1000, 10000]
        rows = []
        for n in ns:
            n_val = max(1, int(n))
            rows.append(
                {
                    "n": n_val,
                    "O(1)": 1,
                    "O(log n)": round(math.log2(n_val), 2),
                    "O(n)": n_val,
                    "O(n log n)": round(n_val * math.log2(n_val), 1),
                    "O(n^2)": n_val * n_val,
                }
            )
        df = pd.DataFrame(rows)
        return ToolResult(
            tool_name="analyze_data",
            success=True,
            summary="Complexity growth comparison:\n" + df.to_string(index=False),
            data=rows,
        )

    return ToolResult("analyze_data", False, f"Unknown operation '{operation}'.")


def build_analysis_tools(store: VectorStore) -> list[ToolSpec]:
    """Create calculation and data-analysis tools."""
    return [
        ToolSpec(
            name="calculate",
            description="Safely evaluate math & algorithmic expressions (arithmetic, log2, sqrt, factorial, comb, ceil, floor).",
            parameters={"expression": "str"},
            fn=calculate,
        ),
        ToolSpec(
            name="analyze_data",
            description="Analyze KB corpus stats, topic frequencies, descriptive statistics, or Big-O complexity tables.",
            parameters={
                "operation": "'corpus_stats' | 'topic_frequency' | 'descriptive_stats' | 'complexity_table'",
                "target": "str | None",
                "values": "list[float] | None",
            },
            fn=lambda operation, target=None, values=None: analyze_data(
                operation=operation, store=store, target=target, values=values
            ),
        ),
    ]

