"""
RAG (Retrieval-Augmented Generation) Core module.
Handles document loading, text chunking, embedding generation,
vector storage, similarity retrieval, and source attribution.
"""

from pathlib import Path
from typing import List, Dict, Any, Optional
import os

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.embeddings import Embeddings

from config import (
    KNOWLEDGE_BASE_DIR,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_TOP_K,
    RAG_FALLBACK_RESPONSE,
)


class LocalTfidfEmbeddings(Embeddings):
    """
    Offline-capable TF-IDF embeddings fallback for testing
    and environments without an active Google API key.
    """

    def __init__(self):
        from sklearn.feature_extraction.text import TfidfVectorizer
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
        self.fitted = False

    def fit_corpus(self, texts: List[str]) -> None:
        """Fit vectorizer on document corpus."""
        if texts:
            self.vectorizer.fit(texts)
            self.fitted = True

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not self.fitted:
            self.fit_corpus(texts)
        matrix = self.vectorizer.transform(texts).toarray()
        return matrix.tolist()

    def embed_query(self, text: str) -> List[float]:
        if not self.fitted:
            return [0.0] * 32
        vec = self.vectorizer.transform([text]).toarray()[0]
        return vec.tolist()


def load_knowledge_base_documents(kb_dir: Optional[Path] = None) -> List[Document]:
    """
    Load all markdown files from the knowledge base directory.
    Attaches source filename to document metadata.
    """
    target_dir = kb_dir or KNOWLEDGE_BASE_DIR
    if not target_dir.exists():
        raise FileNotFoundError(f"Knowledge base directory not found at: {target_dir}")

    documents: List[Document] = []
    md_files = list(target_dir.glob("*.md"))

    if not md_files:
        raise ValueError(f"No markdown documents found in {target_dir}")

    for file_path in sorted(md_files):
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        doc = Document(
            page_content=content,
            metadata={
                "source": file_path.name,
                "file_path": str(file_path),
            },
        )
        documents.append(doc)

    return documents


def get_embeddings(google_api_key: Optional[str] = None) -> Embeddings:
    """
    Return GoogleGenerativeAIEmbeddings when an API key is available,
    otherwise fallback to LocalTfidfEmbeddings for testing/offline use.
    """
    api_key = google_api_key or os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if api_key and api_key.strip():
        try:
            from langchain_google_genai import GoogleGenerativeAIEmbeddings
            return GoogleGenerativeAIEmbeddings(
                model="models/text-embedding-004",
                google_api_key=api_key.strip(),
            )
        except Exception:
            # Fallback if connection or package fails
            pass

    return LocalTfidfEmbeddings()


def build_vector_store(
    documents: Optional[List[Document]] = None,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    google_api_key: Optional[str] = None,
    kb_dir: Optional[Path] = None,
) -> InMemoryVectorStore:
    """
    Split documents into chunks and index them in an in-memory vector store.
    """
    raw_docs = documents if documents is not None else load_knowledge_base_documents(kb_dir)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n## ", "\n### ", "\n- ", "\n\n", "\n", " ", ""],
    )
    chunked_docs = text_splitter.split_documents(raw_docs)

    embeddings = get_embeddings(google_api_key)

    # If using local TF-IDF, pre-fit on all chunk contents
    if isinstance(embeddings, LocalTfidfEmbeddings):
        all_texts = [d.page_content for d in chunked_docs]
        embeddings.fit_corpus(all_texts)

    vector_store = InMemoryVectorStore.from_documents(
        documents=chunked_docs,
        embedding=embeddings,
    )
    return vector_store


def retrieve_documents(
    vector_store: InMemoryVectorStore,
    query: str,
    top_k: int = DEFAULT_TOP_K,
) -> List[Document]:
    """
    Retrieve top_k most similar document chunks for a given query.
    """
    if not query or not query.strip():
        return []
    return vector_store.similarity_search(query, k=top_k)


def format_context_and_sources(documents: List[Document]) -> Dict[str, Any]:
    """
    Format retrieved documents into structured context string and citation list.
    """
    if not documents:
        return {
            "context_str": "",
            "sources": [],
        }

    formatted_parts: List[str] = []
    sources: List[Dict[str, Any]] = []

    for idx, doc in enumerate(documents, start=1):
        source_name = doc.metadata.get("source", "Unknown Document")
        clean_text = doc.page_content.strip()

        formatted_parts.append(
            f"--- [Document Passage {idx}] (Source: {source_name}) ---\n{clean_text}\n"
        )

        # Snippet for UI display (first 250 characters)
        snippet = clean_text[:250].replace("\n", " ") + ("..." if len(clean_text) > 250 else "")
        sources.append({
            "index": idx,
            "source": source_name,
            "snippet": snippet,
            "full_text": clean_text,
        })

    return {
        "context_str": "\n".join(formatted_parts),
        "sources": sources,
    }
