"""Agentic RAG Engine: Query Planning, Adaptive Retrieval, Tool Use, Refinement, and Cited Synthesis."""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Any

from src.agent.llm_client import LLMClient, extractive_synthesize
from src.config import AgentConfig
from src.pipeline.vector_store import VectorStore
from src.tools import ToolRegistry, build_analysis_tools, build_kb_tools


@dataclass
class AgentStep:
    """One reasoning + tool execution step in the agent's trace."""

    step_number: int
    thought: str
    action: str
    action_input: dict[str, Any]
    observation: str
    sources_found: int = 0


@dataclass
class AgentResponse:
    """Complete structured output from the Agentic RAG system."""

    query: str
    query_type: str
    retrieval_strategy: str
    plan: list[str]
    steps: list[AgentStep]
    answer: str
    sources: list[dict] = field(default_factory=list)
    tools_used: list[str] = field(default_factory=list)
    refined_queries: list[str] = field(default_factory=list)
    latency_ms: float = 0.0
    provider: str = "local"


_MATH_TERM = r"(?:(?:log2|log10|sqrt|factorial|comb|perm|ceil|floor)\s*\([^)]+\)|\d+(?:\.\d+)?)"
_MATH_EXPR_RE = re.compile(
    rf"(?:(?:log2|log10|sqrt|factorial|comb|perm|ceil|floor)\s*\([^)]+\)(?:\s*[\+\-\*/\^]\s*{_MATH_TERM})*)"
    rf"|(?:\b\d+(?:\.\d+)?(?:\s*[\+\-\*/\^]\s*{_MATH_TERM})+)"
)


class AgenticRAG:
    """Intelligent RAG Agent that plans, selects tools, refines queries, and verifies sources."""

    def __init__(
        self,
        store: VectorStore,
        config: AgentConfig | None = None,
        llm_client: LLMClient | None = None,
    ) -> None:
        self.store = store
        self.config = config or AgentConfig()
        self.llm = llm_client or LLMClient(self.config.llm)
        self.registry = ToolRegistry()
        for spec in build_kb_tools(store):
            self.registry.register(spec)
        for spec in build_analysis_tools(store):
            self.registry.register(spec)

    # ------------------------------------------------------------------
    # Query Understanding & Planning
    # ------------------------------------------------------------------
    @staticmethod
    def classify_query(query: str) -> tuple[str, str]:
        """Classify query intent and select best retrieval strategy."""
        q = query.lower()
        if any(w in q for w in ["compare", "difference", "contrast", "versus", " vs ", "between"]):
            return "comparison", "hybrid"
        if any(w in q for w in ["summarize", "summary", "overview", "main topics", "what is covered"]):
            return "summarization", "hybrid"
        if any(w in q for w in ["how many", "statistics", "stats", "frequency", "distribution", "complexity table"]):
            return "data_analysis", "bm25"
        if _MATH_EXPR_RE.search(q) or any(w in q for w in ["calculate", "compute", "height of", "how many nodes", "operations for n="]):
            return "calculation_and_retrieval", "hybrid"
        if any(w in q for w in [" and ", "also ", "how does", "why does", "explain how", "step by step"]):
            return "multi_hop", "hybrid"
        return "factual", "hybrid"

    @staticmethod
    def decompose_query(query: str) -> list[str]:
        """Generate refined/decomposed sub-queries for multi-hop or comparison questions."""
        q_clean = re.sub(r"[?!.]+$", "", query.strip())
        # Comparison patterns: "compare X and Y" / "difference between X and Y" / "X vs Y"
        m = re.search(r"(?:compare|difference between|contrast)\s+(.+?)\s+(?:and|vs\.?|versus)\s+(.+)", q_clean, re.I)
        if m:
            return [m.group(1).strip(), m.group(2).strip(), q_clean]

        m_vs = re.search(r"(.+?)\s+(?:vs\.?|versus)\s+(.+)", q_clean, re.I)
        if m_vs:
            return [m_vs.group(1).strip(), m_vs.group(2).strip()]

        # Conjunction split for multi-part questions
        parts = [p.strip() for p in re.split(r"\b(?:and also|and how|and what|;)\b", q_clean, flags=re.I) if len(p.strip()) > 6]
        if len(parts) >= 2:
            return parts[:3]

        # Keyword-focused reformulation
        keywords = re.findall(r"[A-Za-z0-9_+-]{3,}", q_clean)
        stop = {"what", "when", "where", "which", "does", "explain", "describe", "between", "compare", "about", "with"}
        focused = " ".join(w for w in keywords if w.lower() not in stop)
        if focused and focused.lower() != q_clean.lower():
            return [q_clean, focused]
        return [q_clean]

    def build_plan(self, query: str, query_type: str, strategy: str) -> list[str]:
        """Create an explicit human-readable execution plan before acting."""
        if query_type == "comparison":
            return [
                "Inspect available documents to identify relevant lectures.",
                "Decompose comparison into entity-specific sub-queries via refine_and_search.",
                "Cross-check retrieved passages across documents and synthesize comparison with citations.",
            ]
        if query_type == "summarization":
            return [
                "List knowledge-base documents to locate target document(s).",
                "Run summarize_document and targeted search_knowledge_base for key concepts.",
                "Synthesize structured overview with page citations.",
            ]
        if query_type == "data_analysis":
            return [
                "Run analyze_data to compute structured statistics/frequencies.",
                "Supplement with search_knowledge_base for conceptual context if needed.",
                "Synthesize quantitative findings and source-backed explanation.",
            ]
        if query_type == "calculation_and_retrieval":
            return [
                "Retrieve relevant formulas/definitions using search_knowledge_base.",
                "Evaluate numerical/complexity expressions using calculate or analyze_data.",
                "Combine computed results with cited lecture definitions.",
            ]
        if query_type == "multi_hop":
            return [
                f"Perform initial {strategy} search on the knowledge base.",
                "Refine and decompose into sub-queries if coverage is incomplete.",
                "Synthesize multi-step answer with inline source citations.",
            ]
        return [
            f"Search knowledge base using {strategy} retrieval.",
            "Verify relevance; refine query if top score is below threshold.",
            "Generate concise, source-cited answer.",
        ]

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------
    def run(self, query: str) -> AgentResponse:
        t0 = time.perf_counter()
        query_type, strategy = self.classify_query(query)
        if self.config.retrieval_strategy:
            strategy = self.config.retrieval_strategy
        plan = self.build_plan(query, query_type, strategy)

        provider = self.llm.active_provider
        if provider in {"gemini", "openai"}:
            steps, sources, extra_notes, refined_queries, final_answer = self._run_llm_loop(
                query, query_type, strategy, plan
            )
        else:
            steps, sources, extra_notes, refined_queries, final_answer = self._run_local_loop(
                query, query_type, strategy
            )

        tools_used = []
        for s in steps:
            if s.action not in tools_used:
                tools_used.append(s.action)

        latency_ms = round((time.perf_counter() - t0) * 1000, 1)
        return AgentResponse(
            query=query,
            query_type=query_type,
            retrieval_strategy=strategy,
            plan=plan,
            steps=steps,
            answer=final_answer,
            sources=sources,
            tools_used=tools_used,
            refined_queries=refined_queries,
            latency_ms=latency_ms,
            provider=provider,
        )

    def _run_local_loop(
        self, query: str, query_type: str, strategy: str
    ) -> tuple[list[AgentStep], list[dict], list[str], list[str], str]:
        """Deterministic multi-step agent loop for local/offline mode."""
        steps: list[AgentStep] = []
        collected_sources: dict[str, dict] = {}
        extra_notes: list[str] = []
        refined_queries: list[str] = []
        step_num = 1

        def _merge_sources(new_sources: list[dict]) -> None:
            for src in new_sources:
                cid = src["chunk_id"]
                if cid not in collected_sources or src["score"] > collected_sources[cid]["score"]:
                    collected_sources[cid] = src

        # Step 1: Route based on query_type
        if query_type == "summarization":
            res_list = self.registry.invoke("list_documents")
            steps.append(
                AgentStep(
                    step_number=step_num,
                    thought="Identify which documents exist in the knowledge base before summarizing.",
                    action="list_documents",
                    action_input={},
                    observation=res_list.summary,
                )
            )
            step_num += 1

            # Match mentioned lecture/filename or summarize all
            docs = res_list.data or []
            matched = [d["filename"] for d in docs if any(tok in d["filename"].lower() for tok in query.lower().split() if len(tok) > 2)]
            targets = matched[:2] if matched else [d["filename"] for d in docs[:2]]
            for fname in targets:
                res_sum = self.registry.invoke("summarize_document", filename=fname)
                _merge_sources(res_sum.sources)
                extra_notes.append(f"Document overview ({fname}): {res_sum.summary.splitlines()[0]}")
                steps.append(
                    AgentStep(
                        step_number=step_num,
                        thought=f"Extract key topics and representative passages from '{fname}'.",
                        action="summarize_document",
                        action_input={"filename": fname},
                        observation=res_sum.summary[:500],
                        sources_found=len(res_sum.sources),
                    )
                )
                step_num += 1

        elif query_type == "data_analysis":
            q_lower = query.lower()
            if "complexity" in q_lower:
                res_ad = self.registry.invoke("analyze_data", operation="complexity_table", values=[10, 100, 1000, 10000])
            elif any(w in q_lower for w in ["frequency", "often", "mention", "appear"]):
                # Extract candidate topic word
                words = [w for w in re.findall(r"[A-Za-z]+", q_lower) if w not in {"how", "many", "times", "does", "appear", "frequency", "mention", "document", "documents", "across"}]
                term = words[-1] if words else "heap"
                res_ad = self.registry.invoke("analyze_data", operation="topic_frequency", target=term)
            else:
                res_ad = self.registry.invoke("analyze_data", operation="corpus_stats")

            if res_ad.success:
                extra_notes.append(res_ad.summary)
            steps.append(
                AgentStep(
                    step_number=step_num,
                    thought="Use data analysis tool to compute structured quantitative metrics.",
                    action="analyze_data",
                    action_input={"operation": "auto"},
                    observation=res_ad.summary,
                )
            )
            step_num += 1

            res_kb = self.registry.invoke(
                "search_knowledge_base", query=query, top_k=self.config.default_top_k, strategy=strategy
            )
            _merge_sources(res_kb.sources)
            steps.append(
                AgentStep(
                    step_number=step_num,
                    thought="Retrieve supporting conceptual passages from the knowledge base.",
                    action="search_knowledge_base",
                    action_input={"query": query, "top_k": self.config.default_top_k, "strategy": strategy},
                    observation=res_kb.summary,
                    sources_found=len(res_kb.sources),
                )
            )
            step_num += 1

        elif query_type in {"comparison", "multi_hop"}:
            sub_qs = self.decompose_query(query)
            refined_queries.extend(sub_qs)
            res_ref = self.registry.invoke(
                "refine_and_search", original_query=query, sub_queries=sub_qs, top_k_per_query=4
            )
            _merge_sources(res_ref.sources)
            steps.append(
                AgentStep(
                    step_number=step_num,
                    thought=f"Query requires multi-part/comparative evidence; decompose into sub-queries {sub_qs}.",
                    action="refine_and_search",
                    action_input={"original_query": query, "sub_queries": sub_qs, "top_k_per_query": 4},
                    observation=res_ref.summary,
                    sources_found=len(res_ref.sources),
                )
            )
            step_num += 1

        else:
            # Factual or calculation_and_retrieval
            res_kb = self.registry.invoke(
                "search_knowledge_base", query=query, top_k=self.config.default_top_k, strategy=strategy
            )
            _merge_sources(res_kb.sources)
            top_score = (res_kb.data or {}).get("top_score", 0.0) if res_kb.success else 0.0
            steps.append(
                AgentStep(
                    step_number=step_num,
                    thought=f"Run initial {strategy} retrieval to locate relevant lecture passages.",
                    action="search_knowledge_base",
                    action_input={"query": query, "top_k": self.config.default_top_k, "strategy": strategy},
                    observation=res_kb.summary,
                    sources_found=len(res_kb.sources),
                )
            )
            step_num += 1

            # Adaptive query refinement if relevance is low
            if top_score < self.config.min_relevance_threshold and step_num <= self.config.max_steps:
                sub_qs = self.decompose_query(query)
                refined_queries.extend(sub_qs)
                res_ref = self.registry.invoke(
                    "refine_and_search", original_query=query, sub_queries=sub_qs, top_k_per_query=4
                )
                _merge_sources(res_ref.sources)
                steps.append(
                    AgentStep(
                        step_number=step_num,
                        thought=f"Initial top score ({top_score:.3f}) is below threshold; refine query into {sub_qs}.",
                        action="refine_and_search",
                        action_input={"original_query": query, "sub_queries": sub_qs, "top_k_per_query": 4},
                        observation=res_ref.summary,
                        sources_found=len(res_ref.sources),
                    )
                )
                step_num += 1

        # Optional calculation step if math expression is present
        math_match = _MATH_EXPR_RE.search(query)
        if math_match and step_num <= self.config.max_steps:
            expr = math_match.group(0)
            res_calc = self.registry.invoke("calculate", expression=expr)
            if res_calc.success:
                extra_notes.append(f"Calculated: {res_calc.summary}")
            steps.append(
                AgentStep(
                    step_number=step_num,
                    thought=f"Evaluate mathematical expression '{expr}' using the calculator tool.",
                    action="calculate",
                    action_input={"expression": expr},
                    observation=res_calc.summary,
                )
            )

        sorted_sources = sorted(collected_sources.values(), key=lambda s: s["score"], reverse=True)
        answer = extractive_synthesize(query, sorted_sources, extra_notes=extra_notes)
        return steps, sorted_sources, extra_notes, refined_queries, answer

    def _run_llm_loop(
        self, query: str, query_type: str, strategy: str, plan: list[str]
    ) -> tuple[list[AgentStep], list[dict], list[str], list[str], str]:
        """LLM-driven ReAct loop using Gemini or OpenAI with JSON tool-calling."""
        steps: list[AgentStep] = []
        collected_sources: dict[str, dict] = {}
        extra_notes: list[str] = []
        refined_queries: list[str] = []

        system_prompt = (
            "You are an Agentic RAG assistant. You plan before answering and use tools to retrieve, "
            "refine queries, analyze data, summarize documents, and calculate.\n"
            "Available tools:\n"
            f"{self.registry.describe_for_prompt()}\n\n"
            "At each turn, respond ONLY with valid JSON:\n"
            '{"thought": "...", "action": "<tool_name_or_FINISH>", "action_input": {...}, "final_answer": "..."}'
            "\nWhen action is 'FINISH', include 'final_answer' with inline citations [Source: filename, p. X]."
        )

        scratchpad = [
            f"User Question: {query}",
            f"Detected Query Type: {query_type} | Recommended Strategy: {strategy}",
            f"Initial Plan: {plan}",
        ]

        final_answer = ""
        for step_idx in range(1, self.config.max_steps + 1):
            raw = self.llm.generate(system_prompt, "\n\n".join(scratchpad))
            parsed = self._parse_json_action(raw)
            if parsed is None:
                final_answer = raw
                break

            thought = parsed.get("thought", "")
            action = parsed.get("action", "FINISH")
            action_input = parsed.get("action_input") or {}

            if action.upper() == "FINISH":
                final_answer = parsed.get("final_answer", "") or raw
                break

            result = self.registry.invoke(action, **action_input)
            for src in result.sources:
                cid = src["chunk_id"]
                if cid not in collected_sources or src["score"] > collected_sources[cid]["score"]:
                    collected_sources[cid] = src
            if action == "refine_and_search" and isinstance(action_input.get("sub_queries"), list):
                refined_queries.extend(action_input["sub_queries"])
            if action in {"calculate", "analyze_data"} and result.success:
                extra_notes.append(result.summary)

            steps.append(
                AgentStep(
                    step_number=step_idx,
                    thought=thought,
                    action=action,
                    action_input=action_input,
                    observation=result.summary[:800],
                    sources_found=len(result.sources),
                )
            )
            scratchpad.append(
                f"Step {step_idx} Thought: {thought}\nAction: {action}({action_input})\nObservation: {result.summary[:1000]}"
            )

        sorted_sources = sorted(collected_sources.values(), key=lambda s: s["score"], reverse=True)
        if not final_answer:
            # Synthesize final answer from gathered evidence
            context = "\n\n".join(
                f"[{s['filename']}, p. {s['page']}] {s['text']}" for s in sorted_sources[:8]
            )
            synth_prompt = (
                f"Question: {query}\n\nTool Notes:\n{chr(10).join(extra_notes)}\n\n"
                f"Retrieved Evidence:\n{context}\n\n"
                "Write a comprehensive, accurate answer citing sources inline as [Source: filename, p. X]."
            )
            final_answer = self.llm.generate(
                "You are a precise academic assistant. Cite every claim with [Source: filename, p. X].",
                synth_prompt,
            ) or extractive_synthesize(query, sorted_sources, extra_notes=extra_notes)

        return steps, sorted_sources, extra_notes, refined_queries, final_answer

    @staticmethod
    def _parse_json_action(text: str) -> dict | None:
        cleaned = text.strip()
        fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", cleaned, re.S)
        if fence:
            cleaned = fence.group(1)
        else:
            brace = re.search(r"(\{.*\})", cleaned, re.S)
            if brace:
                cleaned = brace.group(1)
        try:
            return json.loads(cleaned)
        except Exception:
            return None

