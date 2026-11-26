"""
Configuration settings and constants for the Campus Knowledge Assistant.
"""

from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent
KNOWLEDGE_BASE_DIR = BASE_DIR / "knowledge_base"
ASSETS_DIR = BASE_DIR / "assets"

# RAG Defaults (Configurable)
DEFAULT_CHUNK_SIZE = 600
DEFAULT_CHUNK_OVERLAP = 100
DEFAULT_TOP_K = 3

# LangGraph Limits & Fallback Constants
STEP_LIMIT = 20

# Mandatory Fallback Strings
RAG_FALLBACK_RESPONSE = "I couldn't find that information in the available documents."
DECLINE_RESPONSE = (
    "I'm designed to assist with campus and academic information. "
    "I can't help with that request."
)

# Supported Gemini Models
DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"
SUPPORTED_MODELS = [
    "gemini-2.5-flash",
    "gemini-1.5-flash",
    "gemini-1.5-pro",
]

# Routing Constants
ROUTE_DOCS = "DOCS"
ROUTE_TOOLS = "TOOLS"
ROUTE_DECLINE = "DECLINE"
VALID_ROUTES = {ROUTE_DOCS, ROUTE_TOOLS, ROUTE_DECLINE}

# Suggested Questions for UI
SUGGESTED_QUESTIONS = [
    "What topics are covered in the course?",
    "What are the academic policies?",
    "How many days remain until my deadline?",
    "Help me create a study schedule.",
]
