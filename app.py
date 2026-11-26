"""
Campus Knowledge Assistant — Streamlit Web Application
LU10 AI Capstone Project
"""

import streamlit as st
from typing import List, Dict, Any
import os

from config import (
    DEFAULT_CHUNK_SIZE,
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_TOP_K,
    DEFAULT_GEMINI_MODEL,
    SUPPORTED_MODELS,
    SUGGESTED_QUESTIONS,
    ROUTE_DOCS,
    ROUTE_TOOLS,
    ROUTE_DECLINE,
)
from rag_core import build_vector_store
from graph_core import create_assistant_graph, AssistantState

# Page Configuration
st.set_page_config(
    page_title="Campus Knowledge Assistant",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling for Modern AI SaaS Look
st.markdown("""
<style>
    /* Metric & Badge styling */
    .route-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 12px;
        font-weight: 600;
        font-size: 0.85rem;
        margin-bottom: 8px;
    }
    .badge-docs {
        background-color: #e3f2fd;
        color: #0d47a1;
        border: 1px solid #90caf9;
    }
    .badge-tools {
        background-color: #fff3e0;
        color: #e65100;
        border: 1px solid #ffcc80;
    }
    .badge-decline {
        background-color: #fbe9e7;
        color: #c62828;
        border: 1px solid #ffab91;
    }
    .exec-path {
        font-family: monospace;
        font-size: 0.82rem;
        color: #555;
        background-color: #f8f9fa;
        padding: 6px 12px;
        border-radius: 6px;
        border: 1px solid #e9ecef;
        margin-bottom: 10px;
    }
    .stButton>button {
        border-radius: 8px;
        transition: all 0.2s;
    }
</style>
""", unsafe_allow_html=True)


def init_session_state():
    """Initialize persistent Streamlit session state."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None
    if "graph_app" not in st.session_state:
        st.session_state.graph_app = None
    if "active_api_key" not in st.session_state:
        st.session_state.active_api_key = ""
    if "submitted_prompt" not in st.session_state:
        st.session_state.submitted_prompt = None


init_session_state()

# ----------------- SIDEBAR -----------------
with st.sidebar:
    st.title("🎓 Assistant Settings")
    st.markdown("Configure your AI Campus Assistant credentials and RAG parameters.")

    # 1. API Key Input
    env_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or ""
    api_key_input = st.text_input(
        "Google Gemini API Key",
        value=env_key if not st.session_state.active_api_key else st.session_state.active_api_key,
        type="password",
        placeholder="AIzaSy...",
        help="Enter your Gemini API key from Google AI Studio. Your key is kept in memory and never stored or logged.",
    )

    if api_key_input:
        st.session_state.active_api_key = api_key_input.strip()
        st.success("API Key detected and active.", icon="🔑")
    else:
        st.info("Operating in offline/heuristic mode. Provide an API key for full Gemini generative capabilities.", icon="ℹ️")

    st.markdown("---")

    # 2. Model Selection
    selected_model = st.selectbox(
        "Gemini Model",
        options=SUPPORTED_MODELS,
        index=0,
        help="Select the generative model for intent routing, query rewriting, and answer generation.",
    )

    # 3. RAG Parameters
    st.subheader("📚 RAG Parameters")
    chunk_size = st.slider("Chunk Size", min_value=200, max_value=1200, value=DEFAULT_CHUNK_SIZE, step=50)
    chunk_overlap = st.slider("Chunk Overlap", min_value=0, max_value=300, value=DEFAULT_CHUNK_OVERLAP, step=25)
    top_k = st.slider("Retrieval Top-K", min_value=1, max_value=6, value=DEFAULT_TOP_K, step=1)

    # Rebuild Index Button
    if st.button("🔄 Rebuild Vector Index", use_container_width=True):
        with st.spinner("Rebuilding vector index..."):
            st.session_state.vector_store = build_vector_store(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                google_api_key=st.session_state.active_api_key,
            )
            st.session_state.graph_app = create_assistant_graph(
                vector_store=st.session_state.vector_store,
                google_api_key=st.session_state.active_api_key,
                model_name=selected_model,
                top_k=top_k,
            )
            st.success("Index rebuilt successfully!")

    st.markdown("---")

    # 4. Clear Chat
    if st.button("🗑️ Clear Conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.chat_history = []
        st.rerun()

    st.markdown("---")
    st.markdown("""
    **🔒 Safety & Grounding Notice:**
    - Document answers are strictly grounded in retrieved campus documents.
    - Hallucination and out-of-scope queries are automatically declined.
    - Prompt injections are rejected by security guardrails.
    - Academic tools perform deterministic computations.
    """)


# Ensure Vector Store and Graph App are initialized
if st.session_state.vector_store is None or st.session_state.graph_app is None:
    st.session_state.vector_store = build_vector_store(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        google_api_key=st.session_state.active_api_key,
    )
    st.session_state.graph_app = create_assistant_graph(
        vector_store=st.session_state.vector_store,
        google_api_key=st.session_state.active_api_key,
        model_name=selected_model,
        top_k=top_k,
    )


# ----------------- MAIN UI -----------------
st.title("🎓 Campus Knowledge Assistant")
st.caption(
    "A production-grade, stateful AI assistant powered by LangChain, LangGraph, and Google Gemini. "
    "Designed to answer academic policy questions, assist with course syllabi, calculate deadlines, and structure study plans."
)

# Suggested Questions Buttons
st.markdown("##### 💡 Suggested Questions")
cols = st.columns(4)
for i, question_text in enumerate(SUGGESTED_QUESTIONS):
    with cols[i]:
        if st.button(question_text, key=f"sugg_{i}", use_container_width=True):
            st.session_state.submitted_prompt = question_text

st.markdown("---")

# Render Existing Chat Messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "assistant":
            # Show route badge
            route = msg.get("route", "")
            if route == ROUTE_DOCS:
                st.markdown('<span class="route-badge badge-docs">Route: DOCS (Knowledge Base)</span>', unsafe_allow_html=True)
            elif route == ROUTE_TOOLS:
                st.markdown('<span class="route-badge badge-tools">Route: TOOLS (Academic Tool)</span>', unsafe_allow_html=True)
            elif route == ROUTE_DECLINE:
                st.markdown('<span class="route-badge badge-decline">Route: DECLINE (Out of Scope / Guardrail)</span>', unsafe_allow_html=True)

            # Show execution path
            exec_path = msg.get("execution_path", [])
            if exec_path:
                st.markdown(f'<div class="exec-path"><strong>Execution Path:</strong> {" ➔ ".join(exec_path)}</div>', unsafe_allow_html=True)

        st.markdown(msg["content"])

        # Show Sources Expander if available
        sources = msg.get("sources", [])
        if sources:
            with st.expander(f"📚 Retrieved Sources ({len(sources)} passages)"):
                for s in sources:
                    st.markdown(f"**Document:** `{s.get('source')}` (Passage #{s.get('index')})")
                    st.caption(s.get("snippet"))
                    st.markdown("---")

        # Show Tool Steps Expander if available
        tool_calls = msg.get("tool_calls", [])
        tool_results = msg.get("tool_results", [])
        if tool_calls or tool_results:
            with st.expander("⚙️ Agent Tool Execution Details"):
                st.markdown("**Tool Calls:**")
                st.json(tool_calls)
                st.markdown("**Tool Outputs:**")
                st.json(tool_results)


# Handle Input (From Chat Input or Suggested Question Button)
user_prompt = st.chat_input("Ask about courses, academic policies, deadlines, or study schedules...")
prompt_to_process = None

if st.session_state.submitted_prompt:
    prompt_to_process = st.session_state.submitted_prompt
    st.session_state.submitted_prompt = None
elif user_prompt:
    prompt_to_process = user_prompt

if prompt_to_process:
    # 1. Render User Message
    st.session_state.messages.append({"role": "user", "content": prompt_to_process})
    with st.chat_message("user"):
        st.markdown(prompt_to_process)

    # 2. Invoke LangGraph Assistant
    with st.chat_message("assistant"):
        with st.spinner("Processing request through LangGraph workflow..."):
            # Prepare state
            initial_state: AssistantState = {
                "question": prompt_to_process,
                "rewritten_question": "",
                "chat_history": list(st.session_state.chat_history),
                "route": "",
                "route_reason": "",
                "documents": [],
                "context": "",
                "sources": [],
                "tool_calls": [],
                "tool_results": [],
                "response": "",
                "execution_path": [],
                "step_count": 0,
            }

            result_state = st.session_state.graph_app.invoke(initial_state)

            answer = result_state.get("response", "I could not generate an answer.")
            route = result_state.get("route", "")
            exec_path = result_state.get("execution_path", [])
            sources = result_state.get("sources", [])
            tool_calls = result_state.get("tool_calls", [])
            tool_results = result_state.get("tool_results", [])

            # Display Route Badge
            if route == ROUTE_DOCS:
                st.markdown('<span class="route-badge badge-docs">Route: DOCS (Knowledge Base)</span>', unsafe_allow_html=True)
            elif route == ROUTE_TOOLS:
                st.markdown('<span class="route-badge badge-tools">Route: TOOLS (Academic Tool)</span>', unsafe_allow_html=True)
            elif route == ROUTE_DECLINE:
                st.markdown('<span class="route-badge badge-decline">Route: DECLINE (Out of Scope / Guardrail)</span>', unsafe_allow_html=True)

            # Display Execution Path
            if exec_path:
                st.markdown(f'<div class="exec-path"><strong>Execution Path:</strong> {" ➔ ".join(exec_path)}</div>', unsafe_allow_html=True)

            # Display Main Response
            st.markdown(answer)

            # Display Sources Expander
            if sources:
                with st.expander(f"📚 Retrieved Sources ({len(sources)} passages)"):
                    for s in sources:
                        st.markdown(f"**Document:** `{s.get('source')}` (Passage #{s.get('index')})")
                        st.caption(s.get("snippet"))
                        st.markdown("---")

            # Display Tool Execution Expander
            if tool_calls or tool_results:
                with st.expander("⚙️ Agent Tool Execution Details"):
                    st.markdown("**Tool Calls:**")
                    st.json(tool_calls)
                    st.markdown("**Tool Outputs:**")
                    st.json(tool_results)

    # 3. Save into History
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "route": route,
        "execution_path": exec_path,
        "sources": sources,
        "tool_calls": tool_calls,
        "tool_results": tool_results,
    })

    # Update conversation memory
    st.session_state.chat_history.append({"role": "user", "content": prompt_to_process})
    st.session_state.chat_history.append({"role": "assistant", "content": answer})
