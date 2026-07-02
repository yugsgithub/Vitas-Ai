"""
phi3_model.py — Vitas AI | GGUF Ayurveda Model for Django
==========================================================
Replaces the previous Transformers + PEFT LoRA backend with a
llama-cpp-python GGUF backend for 3-5× faster inference on Apple Silicon.

Key improvements:
  • Full Metal/MPS GPU offload via n_gpu_layers=-1 (no Transformers needed)
  • Chained generation — 4 focused sub-calls eliminate prompt-leak in small
    fine-tuned models, producing structured ~600-word responses
  • RAG-grounded: each sub-call receives the retrieved document context so
    all 4 sections are anchored in the uploaded document
  • Lazy load — model loads on the first Ayurvedic request, NOT at Django
    startup, so medicinal-only usage never blocks
  • Thread-safe singleton with double-checked locking

Public API (unchanged — no edits needed in ai_models.py or views.py):
  get_ayurvedic_response(message, file_context=None, conversation_history=None) -> str
  is_model_loaded() -> bool
"""

import os
import re
import time
import logging
import threading
from typing import Optional

logger = logging.getLogger(__name__)

# ─── llama-cpp-python import (graceful if not installed) ──────────────────────
try:
    from llama_cpp import Llama
    LLAMA_AVAILABLE = True
except ImportError:
    LLAMA_AVAILABLE = False
    logger.warning(
        "[GGUF] llama-cpp-python is not installed. "
        "Run:  pip install llama-cpp-python"
    )

# ─── MODEL CONFIG ─────────────────────────────────────────────────────────────
REPO_ID      = "Shubh769/Vitas-Ayurveda-Phi3"
FILENAME     = "ayurveda_phi3_q4km.gguf"
N_CTX        = 4096   # Reverted to native 4K context to prevent RoPE scaling degradation
N_GPU_LAYERS     = -1     # offload ALL layers to M2 GPU (Metal)
VERBOSE      = False  # suppress llama.cpp low-level load logs

# Generation params — applied identically to every sub-call in chained mode
_GEN_KWARGS = dict(
    temperature    = 0.65,   # Increased to encourage longer, more diverse, and deeper generation
    top_p          = 0.95,
    top_k          = 50,
    max_tokens     = 1024,   # Increased to allow full paragraph development per section
    repeat_penalty = 1.15,   # Crucial: Forces the model to move on to new concepts instead of looping
)

# ─── SYSTEM PROMPT ────────────────────────────────────────────────────────────
# Intentionally short — this model was fine-tuned with a compact system prompt.
# Long prompts cause the model to echo the prompt back inside its own answer
# ("prompt-leak"). Structure is enforced externally via chained generation below.
SYSTEM_PROMPT = (
    "You are Vitas, an expert Ayurvedic health assistant trained on classical "
    "Ayurvedic texts. Answer questions using accurate Ayurvedic knowledge. "
    "Include relevant Sanskrit terms with English explanations where helpful. "
    "Be detailed, thorough, and professional."
)

# ─── CHAINED GENERATION SUB-PROMPTS ──────────────────────────────────────────
# Instead of one long prompt → one output (which leaks), we make 4 short,
# focused calls and stitch the results into a single structured response.
# Each sub-prompt demands DETAILED, LONG answers — multiple paragraphs per section.
_SUB_PROMPTS = {
    "overview": (
        "Write a DETAILED, COMPREHENSIVE Ayurvedic overview for the following topic. "
        "Your response MUST include: (1) A thorough explanation of the Ayurvedic root "
        "cause and disease origin (Nidana), (2) Which doshas (Vata, Pitta, Kapha) are "
        "imbalanced and WHY, with specific doshic qualities involved, (3) How the "
        "condition progresses according to Samprapti (pathogenesis), and (4) The "
        "classical Ayurvedic perspective including references to relevant texts. "
        "Write at least 4-5 full paragraphs. Topic: "
    ),
    "herbs": (
        "Provide a DETAILED and COMPREHENSIVE list of Ayurvedic herbs and classical "
        "formulations for the following topic. For EACH of at least 6 herbs/formulations, "
        "provide: (1) Full Sanskrit name with English translation and botanical name, "
        "(2) Primary therapeutic actions (Karma) and active properties (Guna, Virya, Vipaka), "
        "(3) Specific form of use (churna/kwath/tablet/ghee/arishta), "
        "(4) Standard dosage with anupana (vehicle), and "
        "(5) Why it is specifically beneficial for this condition. "
        "Write in paragraph or structured list format — be thorough and detailed. Topic: "
    ),
    "diet": (
        "Write a DETAILED and COMPREHENSIVE Ayurvedic diet (Ahara) and lifestyle (Vihara) "
        "guide for the following topic. Include: "
        "(1) At least 6 foods/drinks to FAVOUR with Ayurvedic reasoning and their doshic effects, "
        "(2) At least 5 foods/drinks to AVOID and why they aggravate the condition, "
        "(3) Recommended cooking methods and spices (Aushadha Ahara), "
        "(4) At least 5 daily routine (Dinacharya) and seasonal (Ritucharya) practices, "
        "(5) Yoga asanas, pranayama, and meditation recommendations where applicable. "
        "Write in depth — each point should have at least 2 sentences of explanation. Topic: "
    ),
    "takeaway": (
        "Provide a DETAILED Ayurvedic summary with key takeaways and clinical cautions for "
        "the following topic. Include: "
        "(1) The 4-5 MOST IMPORTANT Ayurvedic principles and insights for this condition, "
        "each explained in 2-3 sentences, "
        "(2) Important contraindications and precautions — which herbs or practices to avoid "
        "and under what circumstances, "
        "(3) When to seek urgent conventional medical care alongside Ayurvedic treatment, "
        "(4) Expected timeline for Ayurvedic improvement with realistic expectations. "
        "End with: 'Please consult a qualified Vaidya (Ayurvedic physician) for a "
        "personalised treatment plan before beginning any herbal regimen.' Topic: "
    ),
}

# Section headings used when assembling the final response
_SECTION_HEADINGS = [
    "1. Ayurvedic Overview",
    "2. Recommended Herbs & Formulations",
    "3. Diet (Ahara) & Lifestyle (Vihara)",
    "4. Key Takeaway & Caution",
]

# ─── SINGLETON STATE ──────────────────────────────────────────────────────────
_llm        = None          # the Llama instance
_lock       = threading.Lock()
_load_error = None          # last exception from _load_model, surfaced to caller


# ─── MODEL LOADING ────────────────────────────────────────────────────────────

def is_model_loaded() -> bool:
    """Return True if the GGUF model is currently in memory."""
    return _llm is not None


def _load_model():
    """
    Internal: download (first run) and load the GGUF model.
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

    logger.info(f"[GGUF] Loading model from HuggingFace: {REPO_ID}/{FILENAME}")
    logger.info("[GGUF] First run will download the model (~2.7 GB). Please wait…")

    try:
        _llm = Llama.from_pretrained(
            repo_id      = REPO_ID,
            filename     = FILENAME,
            n_ctx        = N_CTX,
            n_gpu_layers = N_GPU_LAYERS,
            verbose      = VERBOSE,
        )
        _load_error = None
        logger.info("[GGUF] ✅ Model loaded and ready.")
    except Exception as exc:
        _load_error = exc
        _llm = None
        logger.error(f"[GGUF] ❌ Failed to load model: {exc}")
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


# ─── PROMPT-LEAK CLEANER ──────────────────────────────────────────────────────
# The fine-tuned model occasionally echoes fragments of the system/user prompt
# back into its answer. These heuristics strip the most common patterns.

_LEAK_MARKERS = [
    "You are Vitas",
    "Answer the following question",
    "Include relevant Sanskrit terms",
    "accurate Ayurvedic knowledge",
]


def _clean_leak(text: str) -> str:
    """Strip prompt-leak artifacts that the model sometimes echoes."""
    for marker in _LEAK_MARKERS:
        idx = text.find(marker)
        if idx != -1:
            text = text[:idx].rstrip()
    return text.strip()


# ─── SOCIAL-MEDIA / URL POST-PROCESSOR ───────────────────────────────────────

_JUNK_RE = re.compile(
    r"https?://\S+"                                   # URLs
    r"|www\.\S+"                                      # www links
    r"|@[A-Za-z0-9_]+"                               # @mentions
    r"|#[A-Za-z0-9_]+"                               # hashtags
    r"|[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}"  # emails
)

_SPAM_KEYWORDS = [
    "instagram", "twitter", "linkedin", "facebook",
    "follow us", "join us", "like/share", "subscribe",
]


def _postprocess(text: str) -> str:
    """
    Remove hallucinated junk: URLs, social media references, emails.
    Mirrors the cleanup logic in the original phi3_model.py.
    """
    clean_lines = []
    for line in text.splitlines():
        cleaned = _JUNK_RE.sub("", line).strip()
        if any(kw in cleaned.lower() for kw in _SPAM_KEYWORDS):
            continue
        if len(cleaned) > 3:
            clean_lines.append(cleaned)
    return "\n".join(clean_lines).strip()


# ─── SINGLE FOCUSED MODEL CALL ────────────────────────────────────────────────

def _single_call(user_prompt: str) -> str:
    """
    One focused model call.  Returns the cleaned response text.
    _llm must already be loaded before calling this.
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": user_prompt},
    ]
    resp = _llm.create_chat_completion(messages=messages, **_GEN_KWARGS)
    raw  = resp["choices"][0]["message"]["content"]
    return _clean_leak(raw)


# ─── CHAINED GENERATION ───────────────────────────────────────────────────────

def _chat_with_vitas(user_message: str, file_context: Optional[str] = None) -> str:
    """
    Build a full structured Ayurvedic response with 4 sequential sub-calls.

    Why chained?  A small fine-tuned model given one long, complex system
    prompt tends to loop and reproduce the prompt inside its own answer.
    Four short, directive sub-prompts each get a focused, clean answer which
    are then assembled into a professional structured document.

    RAG-grounded: When file_context is provided (from rag_engine), it is
    prepended to the topic string for ALL 4 sub-calls so every section
    (Overview, Herbs, Diet, Takeaway) is anchored in the document content.

    Sections:
        1. Ayurvedic Overview    (~400 tokens)
        2. Herbs & Formulations  (~400 tokens)
        3. Diet & Lifestyle      (~400 tokens)
        4. Key Takeaway          (~400 tokens)
    Total: ≈ 600–800 words of high-quality, structured content.
    """
    q  = user_message.strip()
    t0 = time.time()

    # ── Phase 4: Build RAG-grounded topic string ────────────────────────────
    if file_context:
        is_rag = "--- RETRIEVED DOCUMENT CONTEXT ---" in file_context
        if is_rag:
            topic = (
                "You are a strict document extraction assistant. "
                "INSTRUCTION: Answer the prompt ONLY using the retrieved document excerpts below. "
                "DO NOT generate generic Ayurvedic knowledge. DO NOT hallucinate. "
                "If the document does not contain the specific herbs, diet, or information requested, "
                "you MUST explicitly say 'Not mentioned in the document'.\n\n"
                f"DOCUMENT CONTEXT:\n{file_context}\n\n"
                f"Topic: {q}"
            )
        else:
            topic = (
                "You are a strict document extraction assistant. "
                "INSTRUCTION: Answer the prompt ONLY using the document excerpt below. "
                "DO NOT generate generic Ayurvedic knowledge. If not mentioned, say 'Not mentioned'.\n\n"
                f"DOCUMENT EXCERPT:\n{file_context}\n\n"
                f"Topic: {q}"
            )
    else:
        topic = q
    # ── End Phase 4 ─────────────────────────────────────────────────────

    raw_sections = [
        _single_call(_SUB_PROMPTS["overview"]  + topic),
        _single_call(_SUB_PROMPTS["herbs"]     + topic),
        _single_call(_SUB_PROMPTS["diet"]      + topic),
        _single_call(_SUB_PROMPTS["takeaway"]  + topic),
    ]

    elapsed = time.time() - t0
    logger.info(f"[GGUF] Chained generation complete in {elapsed:.1f}s")

    # Assemble into a clearly labelled markdown document
    parts = []
    for heading, body in zip(_SECTION_HEADINGS, raw_sections):
        cleaned = _postprocess(body)
        if cleaned:
            parts.append(f"**{heading}**\n\n{cleaned}")

    return "\n\n---\n\n".join(parts)


# ─── PUBLIC API ───────────────────────────────────────────────────────────────

def get_ayurvedic_response(
    message: str,
    file_context: Optional[str] = None,
    conversation_history: list = None,
) -> str:
    """
    Generate an Ayurvedic response using the local GGUF model.

    Drop-in replacement for the old Transformers+LoRA version.
    Signature is intentionally compatible so ai_models.py needs no changes.

    When file_context is provided (from rag_engine.query_documents), it is
    injected into all 4 chained sub-calls so every section of the response
    is grounded in the uploaded document.

    Args:
        message:              The user's latest question.
        file_context:         Optional RAG context block from rag_engine, or
                              fallback raw text excerpt.
        conversation_history: Accepted for API compatibility; chained generation
                              is stateless per sub-call so history is not injected
                              into section prompts (avoids context overflow in a
                              4-call chain of a 4k-context model).

    Returns:
        str: Structured markdown response prefixed with 🌿
    """
    # Lazy-load the model on first Ayurvedic request
    try:
        _ensure_loaded()
    except Exception as exc:
        return (
            f"🌿 ⚠️ Ayurveda model failed to load.\n\n"
            f"**Error:** {exc}\n\n"
            "**Fix:** Make sure `llama-cpp-python` is installed:\n"
            "```\npip install llama-cpp-python\n```\n"
            "Then restart the Django server."
        )

    # Run chained generation (with optional RAG context)
    try:
        response_text = _chat_with_vitas(message, file_context=file_context)
        return f"🌿 {response_text}"
    except Exception as exc:
        logger.error(f"[GGUF] Generation error: {exc}", exc_info=True)
        return (
            f"🌿 ⚠️ Generation failed: {exc}\n\n"
            "Try rephrasing your question or restarting the server."
        )
