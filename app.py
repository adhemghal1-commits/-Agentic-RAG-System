"""Streamlit deployment for the Agentic RAG System."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src.agent import AgenticRAG, LLMClient, StandardRAG
from src.config import AgentConfig, FILES_DIR, LLMConfig, PipelineConfig, VECTOR_STORE_DIR
from src.pipeline import IngestionPipeline, VectorStore

st.set_page_config(
    page_title="Agentic RAG System",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

DEMO_QUESTIONS = [
    "What is a heap and what are the main properties of a min-heap vs max-heap?",
    "Compare the topics covered in Lecture 1 and Lecture 5 Heaps.",
    "Calculate log2(1024) + 1 and explain what it represents for a binary heap of 1024 nodes.",
    "Show corpus statistics and explain how many chunks are in each lecture document.",
    "Summarize Heaps+Lec+5+CP.pdf",
    "Compare time complexity growth across input sizes and explain Big-O notation.",
]


@st.cache_resource(show_spinner=False)
def load_vector_store() -> VectorStore:
    store = VectorStore(VECTOR_STORE_DIR)
    store.load()
    return store


def render_sources(sources: list[dict], key_prefix: str) -> None:
    if not sources:
        st.info("No source chunks retrieved.")
        return
    for i, src in enumerate(sources[:8], start=1):
        label = (
            f"[{i}] {src['filename']} — Page {src['page']} "
            f"(Score: {src['score']:.3f} | Sem: {src.get('semantic_score', 0):.3f} | BM25: {src.get('bm25_score', 0):.3f})"
        )
        with st.expander(label, expanded=(i == 1)):
            st.write(src["text"])
            if src.get("keywords"):
                st.caption("Keywords: " + ", ".join(src["keywords"]))


def main() -> None:
    st.title("🧠 Agentic RAG System")
    st.caption(
        "Intelligent Retrieval-Augmented Generation with Query Planning, Hybrid Search, "
        "Adaptive Refinement, Tool Use, and Source Citations."
    )

    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        provider = st.selectbox(
            "LLM Provider",
            options=["auto", "local", "gemini", "openai"],
            index=0,
            help="'auto' uses Gemini/OpenAI if an API key is set, otherwise runs 100% locally.",
        )
        api_key = ""
        model_name = ""
        if provider in {"gemini", "openai"}:
            api_key = st.text_input(f"{provider.upper()} API Key", type="password")
            default_model = "gemini-2.5-flash" if provider == "gemini" else "gpt-4o-mini"
            model_name = st.text_input("Model", value=default_model)

        strategy = st.selectbox(
            "Retrieval Strategy",
            options=["hybrid", "semantic", "bm25"],
            index=0,
        )
        top_k = st.slider("Top-K Chunks", min_value=2, max_value=12, value=5)
        max_steps = st.slider("Max Agent Steps", min_value=2, max_value=8, value=5)

        st.divider()
        store = load_vector_store()
        if store.is_ready:
            st.success(f"KB Ready: {len(store.chunks)} chunks across {len(store.documents_meta)} docs")
        else:
            st.warning("Knowledge base not built yet. Go to 'Knowledge Base & Pipeline' tab.")

    llm_cfg = LLMConfig(provider=provider, model=model_name, api_key=api_key)
    agent_cfg = AgentConfig(max_steps=max_steps, default_top_k=top_k, retrieval_strategy=strategy, llm=llm_cfg)
    llm_client = LLMClient(llm_cfg)

    tab_qa, tab_compare, tab_kb, tab_arch = st.tabs(
        [
            "🤖 Agentic Q&A",
            "⚖️ Standard vs. Agentic RAG",
            "📚 Knowledge Base & Pipeline",
            "🏗️ Architecture & Demo Suite",
        ]
    )

    # ------------------------------------------------------------------
    # TAB 1: Agentic Q&A
    # ------------------------------------------------------------------
    with tab_qa:
        st.subheader("Ask a Complex Question")
        selected_demo = st.selectbox("Quick Demo Questions (optional)", ["(Custom question)"] + DEMO_QUESTIONS)
        default_q = "" if selected_demo == "(Custom question)" else selected_demo
        query = st.text_area("Your Question", value=default_q, height=80, placeholder="e.g., Compare heaps and priority queues, or calculate log2(1024)...")

        if st.button("🚀 Run Agentic RAG", type="primary", use_container_width=True):
            if not store.is_ready:
                st.error("Please build the Knowledge Base first in the 'Knowledge Base & Pipeline' tab.")
            elif not query.strip():
                st.warning("Please enter a question.")
            else:
                with st.spinner("Agent is planning, retrieving, and reasoning..."):
                    agent = AgenticRAG(store=store, config=agent_cfg, llm_client=llm_client)
                    resp = agent.run(query.strip())

                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Query Type", resp.query_type)
                c2.metric("Strategy", resp.retrieval_strategy)
                c3.metric("Agent Steps", len(resp.steps))
                c4.metric("Latency", f"{resp.latency_ms} ms")

                st.markdown("### 📝 Answer")
                st.markdown(resp.answer)

                col_plan, col_tools = st.columns(2)
                with col_plan:
                    st.markdown("### 🗺️ Execution Plan")
                    for idx, step_txt in enumerate(resp.plan, start=1):
                        st.markdown(f"{idx}. {step_txt}")
                with col_tools:
                    st.markdown("### 🛠️ Tools & Query Refinement")
                    st.write("**Tools Used:**", ", ".join(resp.tools_used) if resp.tools_used else "None")
                    if resp.refined_queries:
                        st.write("**Refined Sub-Queries:**", resp.refined_queries)
                    st.write("**LLM Backend:**", resp.provider)

                st.markdown("### 🔍 Step-by-Step Reasoning Trace")
                for step in resp.steps:
                    with st.expander(f"Step {step.step_number}: `{step.action}` (sources={step.sources_found})", expanded=True):
                        st.markdown(f"**Thought:** {step.thought}")
                        st.code(f"{step.action}({step.action_input})", language="python")
                        st.markdown(f"**Observation:**\n{step.observation}")

                st.markdown("### 📄 Retrieved Source Evidence")
                render_sources(resp.sources, key_prefix="qa")

    # ------------------------------------------------------------------
    # TAB 2: Standard vs Agentic RAG
    # ------------------------------------------------------------------
    with tab_compare:
        st.subheader("Side-by-Side Comparison: Standard RAG vs. Agentic RAG")
        comp_q = st.text_input(
            "Comparison Question",
            value=DEMO_QUESTIONS[2],
            key="comp_q",
        )
        if st.button("⚡ Compare Both Pipelines", type="primary"):
            if not store.is_ready:
                st.error("Build the Knowledge Base first.")
            else:
                with st.spinner("Running Standard RAG and Agentic RAG..."):
                    std = StandardRAG(store=store, llm_client=llm_client).run(comp_q, top_k=top_k, strategy="semantic")
                    ag = AgenticRAG(store=store, config=agent_cfg, llm_client=llm_client).run(comp_q)

                col_l, col_r = st.columns(2)
                with col_l:
                    st.markdown("### 📦 Standard RAG (1-Shot)")
                    m1, m2 = st.columns(2)
                    m1.metric("Latency", f"{std.latency_ms} ms")
                    m2.metric("Retrieved Chunks", len(std.sources))
                    st.info("No planning • No tools • Single semantic query")
                    st.markdown(std.answer)
                    with st.expander("Standard RAG Sources"):
                        render_sources(std.sources, key_prefix="std")

                with col_r:
                    st.markdown("### 🧠 Agentic RAG (Plan + Tools)")
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Latency", f"{ag.latency_ms} ms")
                    m2.metric("Steps", len(ag.steps))
                    m3.metric("Unique Chunks", len(ag.sources))
                    st.success(f"Plan + Tools: {', '.join(ag.tools_used)}")
                    st.markdown(ag.answer)
                    with st.expander("Agentic RAG Reasoning & Sources"):
                        for step in ag.steps:
                            st.markdown(f"- **Step {step.step_number}** `{step.action}`: {step.thought}")
                        render_sources(ag.sources, key_prefix="ag")

    # ------------------------------------------------------------------
    # TAB 3: Knowledge Base & Pipeline
    # ------------------------------------------------------------------
    with tab_kb:
        st.subheader("5-Stage Document Processing Pipeline")
        st.markdown(
            "`1. Text Extraction (PDF + Auto OCR)` → `2. Cleaning & Normalization` → "
            "`3. Recursive Chunking` → `4. Dense + BM25 Embedding` → `5. Persistent Vector Store`"
        )

        uploaded_files = st.file_uploader(
            "Upload additional documents (PDF, TXT, MD, CSV, JSON)",
            type=["pdf", "txt", "md", "csv", "json"],
            accept_multiple_files=True,
        )
        if uploaded_files:
            FILES_DIR.mkdir(parents=True, exist_ok=True)
            for uf in uploaded_files:
                dest = FILES_DIR / uf.name
                dest.write_bytes(uf.getvalue())
            st.success(f"Saved {len(uploaded_files)} file(s) to `{FILES_DIR.name}/`. Click 'Rebuild Knowledge Base' below.")

        c1, c2 = st.columns(2)
        chunk_size = c1.number_input("Chunk Size (chars)", min_value=200, max_value=2000, value=800, step=100)
        chunk_overlap = c2.number_input("Chunk Overlap (chars)", min_value=0, max_value=500, value=150, step=50)

        if st.button("🔄 Build / Rebuild Knowledge Base", type="primary"):
            status_box = st.empty()
            cfg = PipelineConfig(chunk_size=int(chunk_size), chunk_overlap=int(chunk_overlap))
            pipeline = IngestionPipeline(cfg)
            with st.spinner("Running 5-stage ingestion pipeline..."):
                new_store, stats = pipeline.run(progress_cb=lambda msg: status_box.info(msg))
            status_box.success("Ingestion complete!")
            load_vector_store.clear()
            st.json(stats.to_dict())
            st.rerun()

        if store.is_ready and store.documents_meta:
            st.markdown("### 📊 Indexed Documents")
            st.dataframe(pd.DataFrame(store.documents_meta), use_container_width=True)

            st.markdown("### 🔎 Inspect Chunks")
            doc_names = ["All"] + [d["filename"] for d in store.documents_meta]
            sel_doc = st.selectbox("Filter by Document", doc_names)
            filtered = [
                c for c in store.chunks if sel_doc == "All" or c.filename == sel_doc
            ][:15]
            for ch in filtered:
                with st.expander(f"{ch.chunk_id} ({ch.word_count} words)"):
                    st.write(ch.text)
                    st.caption(f"Method: {ch.extraction_method} | Keywords: {', '.join(ch.keywords)}")

    # ------------------------------------------------------------------
    # TAB 4: Architecture & Demo Suite
    # ------------------------------------------------------------------
    with tab_arch:
        st.subheader("Standard RAG vs. Agentic RAG")
        comp_df = pd.DataFrame(
            [
                {"Dimension": "Workflow", "Standard RAG": "Fixed 1-shot Retrieve -> Generate", "Agentic RAG": "Dynamic Plan -> Act -> Observe -> Refine -> Synthesize"},
                {"Dimension": "Query Handling", "Standard RAG": "Raw query as-is", "Agentic RAG": "Intent classification, decomposition, & query refinement"},
                {"Dimension": "Retrieval Strategy", "Standard RAG": "Single top-k semantic lookup", "Agentic RAG": "Hybrid (Dense + BM25 RRF), doc filtering, multi-query"},
                {"Dimension": "Tool Usage", "Standard RAG": "None", "Agentic RAG": "Search, Refine, Summarize, Data Analysis, Calculator"},
                {"Dimension": "Complex Multi-Step Qs", "Standard RAG": "Often misses sub-parts or math", "Agentic RAG": "Breaks down & verifies each sub-part iteratively"},
            ]
        )
        st.table(comp_df)


if __name__ == "__main__":
    main()

