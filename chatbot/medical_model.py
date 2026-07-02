"""
medical_model.py — Vitas AI | GGUF Medical Model for Django
============================================================
Replaces the Google Gemini API backend for medicinal mode with a
local llama-cpp-python GGUF model (medical_phi3_q4km.gguf) trained on
NIH MedQuad, USMLE MedQA, WikiDoc, and PubMed data.

Architecture mirrors phi3_model.py exactly:
  • Full Metal/MPS GPU offload via n_gpu_layers=-1 (Apple M2 optimised)
  • Lazy load — model loads on the first medicinal request, NOT at Django
    startup, so Ayurvedic-only usage never blocks on the medical model
  • Thread-safe singleton with double-checked locking
  • HuggingFace Hub auto-cache → ~/.cache/huggingface/hub/

Public API (drop-in replacement for Gemini-based get_medicinal_response):
  get_medicinal_response(message, file_context=None) -> str
  is_model_loaded() -> bool
"""

import logging
import threading
import re
from typing import Optional

logger = logging.getLogger(__name__)

# ─── llama-cpp-python import (graceful if not installed) ──────────────────────
try:
    from llama_cpp import Llama
    LLAMA_AVAILABLE = True
except ImportError:
    LLAMA_AVAILABLE = False
    logger.warning(
        "[MEDICAL-GGUF] llama-cpp-python is not installed. "
        "Run:  pip install llama-cpp-python"
    )

# ─── MODEL CONFIG ─────────────────────────────────────────────────────────────
MEDICAL_REPO_ID  = "Shubh769/medical_phi3_q4km.gguf"
MEDICAL_FILENAME = "medical_phi3_q4km.gguf"
N_CTX            = 4096   # Reverted to native 4K context to prevent RoPE scaling degradation
N_GPU_LAYERS     = -1     # offload ALL layers to M2 GPU (Metal)
VERBOSE          = False  # suppress llama.cpp low-level load logs

# Generation params — calibrated for strict RAG extraction and factual medical answers
_GEN_KWARGS = dict(
    temperature    = 0.5,    # Increased to encourage longer, more descriptive output
    top_p          = 0.95,
    top_k          = 50,
    max_tokens     = 1500,   # Safely fits within 4K context alongside RAG chunks
    repeat_penalty = 1.05,   # Gentle penalty to break loops without heavily fragmenting headings
)

# ─── SYSTEM PROMPT ────────────────────────────────────────────────────────────
MEDICAL_SYSTEM_PROMPT = """You are Vitas, a board-certified medical AI assistant \
with deep expertise in pathophysiology, pharmacology, clinical medicine, and \
evidence-based treatment. You always produce EXHAUSTIVELY LONG, comprehensive, well-structured answers.

CRITICAL OUTPUT RULES:
1. ONLY answer medical and health-related questions.
2. ALWAYS write a COMPLETE, VERY DETAILED response covering ALL six sections below. Never truncate.
3. Be highly thorough, exhaustive, and professional. Write AT LEAST 2-3 full paragraphs per section.
4. NEVER output literal instructions or bracketed text like "[Insert symptoms here]". You must actually write the medical content.

MANDATORY ANSWER FORMAT (use these exact headings and provide detailed content under each):
## Overview
(Write a detailed introduction to the topic)

## Pathophysiology
(Explain the underlying biological mechanisms)

## Clinical Features
(List specific signs and symptoms)

## Diagnosis
(Describe diagnostic criteria, lab tests, and imaging)

## Management
(Describe first-line treatment and pharmacological options)

## Key Teaching Points
(List high-yield clinical pearls)

DISCLAIMER: This is for informational purposes only. Please consult a licensed physician for personalized medical advice."""

# ─── SINGLETON STATE ──────────────────────────────────────────────────────────
_llm        = None          # the Llama instance
_lock       = threading.Lock()
_load_error = None          # last exception from _load_model, surfaced to caller


# ─── MODEL LOADING ────────────────────────────────────────────────────────────

def is_model_loaded() -> bool:
    """Return True if the Medical GGUF model is currently in memory."""
    return _llm is not None


def _load_model():
    """
    Internal: download (first run) and load the Medical GGUF model.
    Called exactly once, guarded by _lock + double-checked locking.
    The model is pulled from HuggingFace Hub and cached locally by
    llama-cpp-python's standard cache mechanism (~/.cache/huggingface/).
    """
    global _llm, _load_error

    if not LLAMA_AVAILABLE:
        raise RuntimeError(
            "llama-cpp-python is not installed.\n"
            "Run:  pip install llama-cpp-python"
        )

    logger.info(f"[MEDICAL-GGUF] Loading model: {MEDICAL_REPO_ID}/{MEDICAL_FILENAME}")
    logger.info("[MEDICAL-GGUF] Cache: ~/.cache/huggingface/hub/ — first run downloads ~2.4 GB")

    try:
        _llm = Llama.from_pretrained(
            repo_id      = MEDICAL_REPO_ID,
            filename     = MEDICAL_FILENAME,
            n_ctx        = N_CTX,
            n_gpu_layers = N_GPU_LAYERS,
            verbose      = VERBOSE,
        )
        _load_error = None
        logger.info("[MEDICAL-GGUF] ✅ Medical model loaded and ready.")
    except Exception as exc:
        _load_error = exc
        _llm = None
        logger.error(f"[MEDICAL-GGUF] ❌ Failed to load model: {exc}")
        raise


def _ensure_loaded():
    """
    Guarantee the model is loaded exactly once (thread-safe lazy init).
    Safe to call from any Django view thread.
    """
    global _llm
    if _llm is not None:
        return
    with _lock:
        if _llm is None:      # double-checked locking
            _load_model()


# ─── POST-PROCESSOR ───────────────────────────────────────────────────────────

_JUNK_RE = re.compile(
    r"https?://\S+"                                    # URLs
    r"|www\.\S+"                                       # www links
    r"|@[A-Za-z0-9_]+"                                # @mentions
    r"|#[A-Za-z0-9_]+"                                # hashtags
    r"|[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}"  # emails
)

_SPAM_KEYWORDS = [
    "instagram", "twitter", "linkedin", "facebook",
    "follow us", "join us", "like/share", "subscribe",
]


def _postprocess(text: str) -> str:
    """Remove hallucinated junk: URLs, social-media references, emails."""
    clean_lines = []
    for line in text.splitlines():
        cleaned = _JUNK_RE.sub("", line).strip()
        if any(kw in cleaned.lower() for kw in _SPAM_KEYWORDS):
            continue
        if len(cleaned) > 2:
            clean_lines.append(cleaned)
    return "\n".join(clean_lines).strip()


# ─── CORE INFERENCE ───────────────────────────────────────────────────────────

def _run_medical_inference(prompt_text: str, system_prompt: str = MEDICAL_SYSTEM_PROMPT) -> str:
    """
    Run inference locally using llama-cpp-python.
    Uses the ChatML format expected by Phi-3 fine-tunes.
    """
    global _llm
    if not _llm:
        raise RuntimeError("Model is not loaded. Call _ensure_loaded() first.")

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt_text}
    ]
    resp = _llm.create_chat_completion(messages=messages, **_GEN_KWARGS)
    raw  = resp["choices"][0]["message"]["content"]
    return _postprocess(raw)


# ─── PUBLIC API ───────────────────────────────────────────────────────────────

def get_medicinal_response(
    message: str,
    file_context: Optional[str] = None,
) -> str:
    """
    Generate a medical response using the local GGUF model.

    Drop-in replacement for the Gemini-based get_medicinal_response()
    in ai_models.py. Signature is identical so ai_models.py routing
    needs only a one-line import change.

    When a RAG context block is provided (from rag_engine.query_documents),
    the model is explicitly instructed to ground its answer in the retrieved
    document chunks and flag when the answer is not in the context.

    Args:
        message:      The user's medical question.
        file_context: Optional formatted context block from RAG retrieval
                      (or raw extracted text for fallback). Detected by the
                      '--- RETRIEVED DOCUMENT CONTEXT ---' header.

    Returns:
        str: Structured markdown medical response prefixed with 💊
    """
    # Lazy-load the model on first medicinal request
    try:
        _ensure_loaded()
    except Exception as exc:
        return (
            f"💊 ⚠️ Medical model failed to load.\n\n"
            f"**Error:** {exc}\n\n"
            "**Fix:** Make sure `llama-cpp-python` is installed:\n"
            "```\npip install llama-cpp-python\n```\n"
            "Then restart the Django server."
        )

    # ── Phase 4: Build RAG-grounded prompt ──────────────────────────────────
    user_prompt = message
    system_prompt_to_use = MEDICAL_SYSTEM_PROMPT

    if file_context:
        # STRICT RAG EXTRACTION SYSTEM PROMPT (Bypasses generic textbook template entirely)
        system_prompt_to_use = (
            "You are given a medical report.\n\n"
            "STRICT RULES:\n"
            "- Use ONLY the provided context.\n"
            "- DO NOT add external medical knowledge.\n"
            "- DO NOT generalize.\n"
            "- If your answer is not directly supported by the context, do not include it.\n\n"
            "Extract and return exactly these sections based ONLY on the text:\n"
            "1. Diagnosis:\n"
            "2. Key Findings:\n"
            "3. Important Negative Findings:\n"
            "4. Final Impression:\n\n"
            "If something is not present, write 'Not mentioned'."
        )

        user_prompt = (
            f"DOCUMENT CONTEXT:\n{file_context}\n\n"
            f"User Question: {message}"
        )
    # ── End Phase 4 ──────────────────────────────────────────────────

    # Run inference
    try:
        response_text = _run_medical_inference(user_prompt, system_prompt=system_prompt_to_use)
        return f"💊 {response_text}"
    except Exception as exc:
        logger.error(f"[MEDICAL-GGUF] Generation error: {exc}", exc_info=True)
        return (
            f"💊 ⚠️ Generation failed: {exc}\n\n"
            "Try rephrasing your question or restarting the server."
        )
