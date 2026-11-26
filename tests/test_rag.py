"""
Unit tests for RAG core engine.
Tests document loading, chunking, indexing, retrieval, and source attribution.
"""

import pytest
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import KNOWLEDGE_BASE_DIR, RAG_FALLBACK_RESPONSE
from rag_core import (
    load_knowledge_base_documents,
    build_vector_store,
    retrieve_documents,
    format_context_and_sources,
)


class TestRAGCore:
    """Test suite for RAG loading, chunking, and retrieval."""

    def test_load_all_knowledge_base_documents(self):
        docs = load_knowledge_base_documents(KNOWLEDGE_BASE_DIR)
        assert len(docs) >= 3, "Expected at least 3 knowledge base documents"
        source_names = {d.metadata.get("source") for d in docs}
        assert "course_overview.md" in source_names
        assert "academic_policies.md" in source_names
        assert "student_guide.md" in source_names

    def test_document_metadata_preservation(self):
        docs = load_knowledge_base_documents(KNOWLEDGE_BASE_DIR)
        for doc in docs:
            assert "source" in doc.metadata
            assert "file_path" in doc.metadata
            assert len(doc.page_content.strip()) > 500

    def test_build_vector_store_and_chunking(self):
        vs = build_vector_store(chunk_size=400, chunk_overlap=50)
        assert vs is not None

    def test_retrieval_accuracy_for_policies(self):
        vs = build_vector_store(chunk_size=500, chunk_overlap=50)
        results = retrieve_documents(vs, "attendance policy minimum percentage", top_k=2)
        assert len(results) > 0
        found_policy = any("academic_policies.md" in d.metadata.get("source", "") for d in results)
        assert found_policy, "Expected academic_policies.md to be retrieved for attendance policy query"

    def test_retrieval_accuracy_for_course_overview(self):
        vs = build_vector_store(chunk_size=500, chunk_overlap=50)
        results = retrieve_documents(vs, "CS 402 grading breakdown Capstone project", top_k=2)
        assert len(results) > 0
        found_course = any("course_overview.md" in d.metadata.get("source", "") for d in results)
        assert found_course, "Expected course_overview.md to be retrieved for grading breakdown query"

    def test_retrieval_accuracy_for_facilities(self):
        vs = build_vector_store(chunk_size=500, chunk_overlap=50)
        results = retrieve_documents(vs, "Turing computing center lab operating hours", top_k=3)
        assert len(results) > 0
        found_guide = any("student_guide.md" in d.metadata.get("source", "") for d in results)
        assert found_guide, "Expected student_guide.md to be retrieved for lab hours query"

    def test_format_context_and_sources(self):
        vs = build_vector_store(chunk_size=500, chunk_overlap=50)
        results = retrieve_documents(vs, "late submission penalty", top_k=2)
        formatted = format_context_and_sources(results)
        assert "context_str" in formatted
        assert "sources" in formatted
        assert len(formatted["sources"]) == len(results)
        for s in formatted["sources"]:
            assert "source" in s
            assert "snippet" in s
            assert "full_text" in s
