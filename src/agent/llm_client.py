"""Unified LLM client supporting Google Gemini, OpenAI, and Local Extractive mode."""

from __future__ import annotations

import os
import re
from src.config import LLMConfig


class LLMClient:
    """Wrapper around Gemini, OpenAI, and a zero-dependency local extractive fallback."""

    def __init__(self, config: LLMConfig | None = None) -> None:
        self.config = config or LLMConfig()

    @property
    def active_provider(self) -> str:
        return self.config.resolved_provider()

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        provider = self.active_provider
        if provider == "gemini":
            return self._call_gemini(system_prompt, user_prompt)
        if provider == "openai":
            return self._call_openai(system_prompt, user_prompt)
        return ""

    def _call_gemini(self, system_prompt: str, user_prompt: str) -> str:
        from google import genai
        from google.genai import types

        api_key = self.config.api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        model = self.config.model or "gemini-2.5-flash"
        client = genai.Client(api_key=api_key)
        resp = client.models.generate_content(
            model=model,
            contents=user_prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=self.config.temperature,
            ),
        )
        return (resp.text or "").strip()

    def _call_openai(self, system_prompt: str, user_prompt: str) -> str:
        from openai import OpenAI

        api_key = self.config.api_key or os.getenv("OPENAI_API_KEY")
        model = self.config.model or "gpt-4o-mini"
        kwargs: dict = {"api_key": api_key}
        if self.config.base_url:
            kwargs["base_url"] = self.config.base_url
        client = OpenAI(**kwargs)
        resp = client.chat.completions.create(
            model=model,
            temperature=self.config.temperature,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        return (resp.choices[0].message.content or "").strip()


def extractive_synthesize(question: str, sources: list[dict], extra_notes: list[str] | None = None) -> str:
    """Deterministic local synthesis from retrieved source chunks and tool outputs.

    Used when running in 'local' mode without an external LLM API key.
    """
    q_tokens = set(re.findall(r"[A-Za-z0-9_+-]{3,}", question.lower()))
    lines: list[str] = []

    if extra_notes:
        lines.append("### Tool & Analysis Results")
        for note in extra_notes:
            lines.append(f"- {note}")
        lines.append("")

    if not sources:
        if lines:
            return "\n".join(lines)
        return "No relevant passages were found in the knowledge base to answer this question."

    lines.append("### Source-Grounded Answer")
    seen_snippets: set[str] = set()
    scored_sentences: list[tuple[int, str, str, int]] = []

    for src in sources[:8]:
        text = src.get("text", "")
        fname = src.get("filename", "unknown")
        page = src.get("page", 1)
        sentences = re.split(r"(?<=[.!?])\s+|\n+", text)
        for sent in sentences:
            s = sent.strip()
            if len(s) < 25:
                continue
            s_tokens = set(re.findall(r"[A-Za-z0-9_+-]{3,}", s.lower()))
            overlap = len(q_tokens & s_tokens)
            scored_sentences.append((overlap, s, fname, page))

    scored_sentences.sort(key=lambda item: item[0], reverse=True)
    picked = 0
    for overlap, sent, fname, page in scored_sentences:
        norm = sent.lower()[:80]
        if norm in seen_snippets:
            continue
        seen_snippets.add(norm)
        lines.append(f"- {sent} `[Source: {fname}, p. {page}]`")
        picked += 1
        if picked >= 5:
            break

    if picked == 0:
        for src in sources[:3]:
            lines.append(
                f"- {src['text'][:240].strip()}... `[Source: {src['filename']}, p. {src['page']}]`"
            )

    return "\n".join(lines)

