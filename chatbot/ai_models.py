"""
AI Models Integration for Vitas AI
FILE: chatbot/ai_models.py

Routing:
  - Medicinal questions  → Local GGUF model (medical_model.py)
                           Shubh769/medical_phi3_q4km.gguf via llama-cpp-python
  - Ayurvedic questions  → Local GGUF model (phi3_model.py)
                           Shubh769/Vitas-Ayurveda-Phi3 via llama-cpp-python

Both models run fully locally on Apple M2 (Metal GPU offload).
No external API keys are required.

RAG Pipeline (Phase 3):
  - On every send_message call that has an uploaded file, `rag_engine.query_documents()`
    fetches the top-4 most relevant text chunks via ChromaDB + sentence-transformers.
  - Falls back to raw extracted text (capped at 3000 chars) if no RAG index exists.
"""

import os
import logging
from typing import Optional

import PyPDF2
from PIL import Image
import pytesseract

logger = logging.getLogger(__name__)


# ============================================
# PDF TEXT EXTRACTION
# ============================================

def extract_text_from_pdf(file_path: str) -> str:
    """Extract text from a PDF file using PyPDF2."""
    try:
        text = ""
        with open(file_path, "rb") as f:
            pdf_reader = PyPDF2.PdfReader(f)
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
        return text.strip()
    except Exception as e:
        return f"Error reading PDF: {str(e)}"


# ============================================
# IMAGE TEXT EXTRACTION (OCR)
# ============================================

def extract_text_from_image(file_path: str) -> str:
    """Extract text from an image file using pytesseract OCR."""
    try:
        image = Image.open(file_path)
        return pytesseract.image_to_string(image).strip()
    except Exception as e:
        return f"Error reading image: {str(e)}"


def get_groq_rag_response(message: str, file_context: str) -> str:
    """
    RAG queries are exclusively routed to the Groq API as requested.
    Bypasses local models when a document is uploaded.
    """
    import os
    try:
        from groq import Groq
    except ImportError:
        return (
            "⚠️ **Missing Dependency:** The `groq` package is not installed.\n\n"
            "Please run this command in your VS Code terminal:\n"
            "```bash\npip install groq\n```\n"
            "Then restart the server."
        )

    try:
        # Client automatically picks up GROQ_API_KEY from environment
        client = Groq()
        
        # Strict RAG prompt optimized for LONG, DETAILED, and STRUCTURED answers
        system_prompt = (
            "You are a professional medical document analyst. Your goal is to provide a LONG, COMPREHENSIVE, and HIGHLY DETAILED response.\n\n"
            "STRICT RULES:\n"
            "- Answer the user's query using ONLY the document context provided below.\n"
            "- Use a structured format with headings and bullet points.\n"
            "- DO NOT be brief. Provide a deep-dive into the facts, data, and findings mentioned in the text.\n"
            "- Quote directly from the document where it adds value.\n"
            "- If the information is not in the context, say 'Not mentioned in the document'.\n\n"
            f"DOCUMENT CONTEXT:\n{file_context}"
        )
        
        # Exact snippet provided by the user
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": message}
            ],
            temperature=1,
            max_completion_tokens=1024,
            top_p=1,
            stream=True,
            stop=None
        )
        
        response_text = ""
        for chunk in completion:
            response_text += chunk.choices[0].delta.content or ""
            
        return response_text
        
    except Exception as e:
        logger.error(f"[GROQ RAG] API Error: {e}", exc_info=True)
        return f"⚠️ **Groq API Error:** {str(e)}"


# ============================================
# FILE PROCESSING
# ============================================

def process_uploaded_file(file_path: str, file_type: str) -> Optional[str]:
    """
    Process an uploaded file and return its extracted text content.

    Args:
        file_path: Absolute path to the file on disk.
        file_type: MIME type string (e.g. 'application/pdf', 'image/png').

    Returns:
        Extracted text string, or None if the type is unsupported.
    """
    try:
        if "pdf" in file_type.lower():
            return extract_text_from_pdf(file_path)
        elif any(t in file_type.lower() for t in ["image", "jpeg", "jpg", "png"]):
            return extract_text_from_image(file_path)
        elif "text" in file_type.lower():
            with open(file_path, "r", encoding="utf-8") as f:
                return f.read()
        return None
    except Exception as e:
        logger.error(f"Error processing uploaded file: {e}")
        return None


# ============================================
# MEDICINAL AI MODEL (LOCAL GGUF)
# ============================================

def get_medicinal_response(
    message: str,
    file_context: Optional[str] = None,
) -> str:
    """
    Get a response from the local Medical GGUF model.

    Uses medical_model.py which loads Shubh769/medical_phi3_q4km.gguf
    via llama-cpp-python with full M2 Metal GPU offload.

    Args:
        message:      The user's medical question.
        file_context: Optional text extracted from an uploaded file.

    Returns:
        str: Structured medical response prefixed with 💊
    """
    from .medical_model import get_medicinal_response as medical_respond
    return medical_respond(message, file_context=file_context)


# ============================================
# AYURVEDIC AI MODEL (LOCAL GGUF)
# ============================================

def get_ayurvedic_response(
    message: str,
    file_context: Optional[str] = None,
    conversation_history: list = None,
) -> str:
    """
    Get a response from the local GGUF Ayurveda model (llama-cpp-python).

    Uses phi3_model.py which loads Shubh769/Vitas-Ayurveda-Phi3 with
    chained generation: 4 focused sub-calls assembled into one structured
    response. Falls back to a friendly error message if the model fails to load.

    The file_context (RAG-retrieved chunks) is passed directly to phi3_model
    so each of the 4 sub-calls is individually grounded in the document.

    Args:
        message:              The user's latest question.
        file_context:         Optional RAG context block from rag_engine, or
                              fallback raw text excerpt.
        conversation_history: Prior [{'role': ..., 'content': ...}] turns
                              (accepted for API compatibility; chained generation
                              is stateless so history is not injected).

    Returns:
        str: Structured Ayurvedic markdown response prefixed with 🌿
    """
    from .phi3_model import get_ayurvedic_response as phi3_respond
    return phi3_respond(message, file_context=file_context, conversation_history=conversation_history)


# ============================================
# MAIN RESPONSE GENERATOR
# ============================================

def generate_ai_response(
    message: str,
    model_type: str,
    file_path: Optional[str] = None,
    file_type: Optional[str] = None,
    conversation_history: list = None,
    conversation_id: Optional[int] = None,
) -> str:
    """
    Main dispatch function — routes a user message to the correct local model.

    RAG-augmented: If a conversation has indexed documents, retrieves the top-4
    most relevant chunks via ChromaDB instead of stuffing the raw file text.
    Falls back to raw extracted text (capped at 3000 chars) if RAG unavailable.

    Args:
        message:              The user's latest question.
        model_type:           'medicinal' (Medical GGUF) or 'ayurvedic' (Ayurveda GGUF).
        file_path:            Optional path to an uploaded file (used for fallback extraction).
        file_type:            MIME type of the uploaded file.
        conversation_history: Prior [{'role': ..., 'content': ...}] turns.
        conversation_id:      Django ChatConversation PK — used to query the RAG index.

    Returns:
        str: The AI response string.
    """
    # ── Phase 3: RAG Retrieval ────────────────────────────────────────────────
    file_context = None

    if conversation_id is not None:
        try:
            from .rag_engine import query_documents, has_documents
            if has_documents(conversation_id):
                file_context = query_documents(message, conversation_id)
                if file_context:
                    logger.info(
                        f"[RAG] Retrieved context ({len(file_context)} chars) "
                        f"for conversation {conversation_id}"
                    )
        except Exception as rag_err:
            logger.warning(f"[RAG] Retrieval failed, falling back to raw text: {rag_err}")
            file_context = None

    # Fallback: extract raw text from the file if RAG returned nothing
    if file_context is None and file_path and file_type:
        raw_text = process_uploaded_file(file_path, file_type)
        if raw_text and raw_text.strip():
            logger.info(f"[RAG] Fallback — using raw extracted text ({len(raw_text)} chars)")
            # Cap at 3000 chars to stay within the model's context window
            trimmed = raw_text[:3000]
            file_context = (
                "--- DOCUMENT EXCERPT (fallback, no RAG index) ---\n"
                f"{trimmed}\n"
                "--------------------------------------------------"
            )
        else:
            return (
                "⚠️ **Document Extraction Failed:** I could not read any text from the uploaded document. "
                "If this is a scanned PDF, an image, or an unsupported format like .docx, please ensure it contains "
                "selectable text, or try copying and pasting the text directly into the chat."
            )
    # ── End RAG Retrieval ─────────────────────────────────────────────────────

    # ── API Routing Logic ──
    # If a document is uploaded, exclusively use the Groq API
    if file_context:
        logger.info("[ROUTING] Document detected. Routing to Groq API.")
        return get_groq_rag_response(message, file_context)

    # If NO document is uploaded, route to the local GGUF models
    logger.info(f"[ROUTING] No document. Routing to local {model_type} model.")
    if model_type == "medicinal":
        return get_medicinal_response(message, None)
    elif model_type == "ayurvedic":
        return get_ayurvedic_response(message, None, conversation_history)
    else:
        return "Invalid model type. Please select either 'medicinal' or 'ayurvedic'."


# ============================================
# UTILITY — MODEL STATUS
# ============================================

def get_model_status() -> dict:
    """
    Return the load status of both local GGUF models.
    Useful for health-check endpoints or admin views.

    Returns:
        dict: {'medicinal_loaded': bool, 'ayurvedic_loaded': bool}
    """
    try:
        from .medical_model import is_model_loaded as medical_loaded
    except ImportError:
        medical_loaded = lambda: False  # noqa: E731

    try:
        from .phi3_model import is_model_loaded as ayurvedic_loaded
    except ImportError:
        ayurvedic_loaded = lambda: False  # noqa: E731

    return {
        "medicinal_loaded":  medical_loaded(),
        "ayurvedic_loaded":  ayurvedic_loaded(),
    }