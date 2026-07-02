"""
=============================================================================
  Vitas AI — Apple M2 Air Inference & Chatbot Integration Script
  Target Hardware : Apple MacBook M2 Air (16 GB Unified RAM)
  Runtime         : llama-cpp-python (loads GGUF directly via Metal / MPS)

  Prerequisites (run once):
    pip install llama-cpp-python

  Model is loaded directly from HuggingFace Hub on first run and cached
  locally — no Ollama server required.

  Sections:
    A. Model Initialisation (lazy singleton)
    B. System Prompt (strict structured-output contract)
    C. Python API Wrapper  (chat_with_vitas / stream_vitas)
    D. Response Post-Processor (formatting & display)
    E. CLI Benchmark
    F. Interactive Chat Loop
    G. Django Integration Notes
=============================================================================
"""

import os
import sys
import time
import textwrap
import re

# ─── A. MODEL INITIALISATION ──────────────────────────────────────────────────
# llama-cpp-python loads GGUF weights directly and offloads all layers to the
# M2 Neural Engine / Metal via n_gpu_layers=-1.
# n_ctx=4096 fills the full Phi-3-mini context window on 16 GB Unified RAM.

try:
    from llama_cpp import Llama
    LLAMA_AVAILABLE = True
except ImportError:
    LLAMA_AVAILABLE = False

REPO_ID  = "Shubh769/Vitas-Ayurveda-Phi3"
FILENAME = "ayurveda_phi3_q4km.gguf"

_llm: "Llama | None" = None   # lazy singleton — loaded on first call


def _get_model() -> "Llama":
    """Return the (cached) Llama instance, loading it on first call."""
    global _llm
    if _llm is not None:
        return _llm

    if not LLAMA_AVAILABLE:
        raise RuntimeError(
            "llama-cpp-python is not installed.\n"
            "Run:  pip install llama-cpp-python"
        )

    print("🔄  Loading Vitas model (first run may take a moment)…")
    _llm = Llama.from_pretrained(
        repo_id      = REPO_ID,
        filename     = FILENAME,
        n_ctx        = 4096,    # full Phi-3-mini context
        n_gpu_layers = -1,      # offload ALL layers to M2 GPU (Metal)
        verbose      = False,
    )
    print("✅  Model loaded and ready.\n")
    return _llm


# ─── B. SYSTEM PROMPT ─────────────────────────────────────────────────────────
# IMPORTANT: This model (Phi-3 Mini Q4_K_M) was fine-tuned with a very short
# system prompt baked into its weights. Injecting a long complex prompt causes
# the model to loop back and reproduce the prompt inside its own answer
# ("prompt leak"). The fix is to use the exact short prompt it was trained with
# and handle structure externally via chained generation (see section C).

# This matches the prompt embedded during LoRA fine-tuning:
SYSTEM_PROMPT = (
    "You are Vitas, an expert Ayurvedic health assistant trained on classical "
    "Ayurvedic texts. Answer questions using accurate Ayurvedic knowledge. "
    "Include relevant Sanskrit terms with English explanations where helpful. "
    "Be detailed, thorough, and professional."
)

# Sub-prompts used by the chained generation pipeline (Section C).
# Each is short enough for the model to follow without looping.
_SUB_PROMPTS = {
    "overview": (
        "Provide a detailed Ayurvedic overview for the following. "
        "Explain the Ayurvedic root cause, which doshas (Vata, Pitta, or Kapha) "
        "are involved and why. Write at least 3 full paragraphs. Topic: "
    ),
    "herbs": (
        "List at least 5 specific Ayurvedic herbs or classical formulations "
        "for the following. For each herb give: Sanskrit name with English "
        "translation, therapeutic action, form of use (churna/kwath/tablet/ghee), "
        "and typical dosage. Topic: "
    ),
    "diet": (
        "Give detailed Ayurvedic diet (Ahara) and lifestyle (Vihara) guidance "
        "for the following. List at least 4 foods to favour and 3 to avoid with "
        "Ayurvedic reasoning. Also include 3 daily routine (Dinacharya) practices. Topic: "
    ),
    "takeaway": (
        "Summarise the 3 most important Ayurvedic takeaways and cautions for the "
        "following. End with: 'Please consult a qualified Vaidya (Ayurvedic "
        "physician) for a personalised treatment plan before beginning any herbal "
        "regimen.' Topic: "
    ),
}


# ─── C. PYTHON API WRAPPER ────────────────────────────────────────────────────
# Strategy: Chained Generation
# ─────────────────────────────
# Instead of one call with a long system prompt (which causes prompt-leak in
# small fine-tuned models), we make 4 focused sequential API calls — one for
# each section of the answer — then stitch them together.
# Each sub-call uses a short, directive user prompt the model can handle.

# Shared generation kwargs — same params for every sub-call
_GEN_KWARGS = dict(
    temperature    = 0.70,
    top_p          = 0.90,
    top_k          = 40,
    max_tokens     = 400,   # per section; 4 sections ≈ 1 600 tokens total
    repeat_penalty = 1.15,
)


def _clean(text: str) -> str:
    """
    Strip any prompt-leak artefacts from a model response.
    The model sometimes echoes fragments of the system or user prompt;
    this heuristic removes the most common patterns.
    """
    # Known training-prompt fragments the model tends to reproduce
    leak_markers = [
        "You are Vitas",
        "Answer the following question",
        "Include relevant Sanskrit terms",
        "accurate Ayurvedic knowledge",
    ]
    for marker in leak_markers:
        idx = text.find(marker)
        if idx != -1:
            text = text[:idx].rstrip()
    return text.strip()


def _single_call(user_prompt: str) -> str:
    """One focused model call. Returns cleaned response text."""
    llm = _get_model()
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": user_prompt},
    ]
    resp = llm.create_chat_completion(messages=messages, **_GEN_KWARGS)
    raw  = resp["choices"][0]["message"]["content"]
    return _clean(raw)


def chat_with_vitas(user_message: str, history: list | None = None) -> str:
    """
    Generate a long, structured Ayurvedic response using chained generation.

    Makes 4 focused API calls (Overview → Herbs → Diet+Lifestyle → Takeaway)
    and stitches the results into one professional structured answer.

    Args:
        user_message : The user's question or message.
        history      : Ignored in chained mode (stateless per section call).
                       Pass history only when using _single_call directly.

    Returns:
        A fully structured multi-section string (~400–600 words).

    Example:
        response = chat_with_vitas("What are the benefits of Ashwagandha?")
        print(response)
    """
    q = user_message.strip()

    overview  = _single_call(_SUB_PROMPTS["overview"]  + q)
    herbs     = _single_call(_SUB_PROMPTS["herbs"]     + q)
    diet      = _single_call(_SUB_PROMPTS["diet"]      + q)
    takeaway  = _single_call(_SUB_PROMPTS["takeaway"]  + q)

    # Assemble into a clearly labelled structured document
    sections = [
        ("1. Ayurvedic Overview",                    overview),
        ("2. Recommended Herbs & Formulations",      herbs),
        ("3. Diet (Ahara) & Lifestyle (Vihara)",     diet),
        ("4. Key Takeaway & Caution",                takeaway),
    ]
    parts = []
    for heading, body in sections:
        if body:
            parts.append(f"**{heading}**\n\n{body}")

    return "\n\n".join(parts)


def stream_vitas(user_message: str, history: list | None = None):
    """
    Streaming version of chained generation.
    Yields tokens section-by-section with section headers interleaved.

    Useful for real-time web UI (e.g., Server-Sent Events / WebSocket).

    Example:
        for chunk in stream_vitas("What is Vata dosha?"):
            print(chunk, end="", flush=True)
    """
    llm = _get_model()
    q   = user_message.strip()

    plan = [
        ("1. Ayurvedic Overview",               _SUB_PROMPTS["overview"]  + q),
        ("2. Recommended Herbs & Formulations", _SUB_PROMPTS["herbs"]     + q),
        ("3. Diet (Ahara) & Lifestyle (Vihara)",_SUB_PROMPTS["diet"]      + q),
        ("4. Key Takeaway & Caution",           _SUB_PROMPTS["takeaway"]  + q),
    ]

    for i, (heading, user_prompt) in enumerate(plan):
        # Yield the section header
        prefix = "\n\n" if i > 0 else ""
        yield f"{prefix}**{heading}**\n\n"

        # Stream tokens for this section
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_prompt},
        ]
        stream = llm.create_chat_completion(
            messages = messages,
            stream   = True,
            **_GEN_KWARGS,
        )
        buffer = ""
        for chunk in stream:
            delta = chunk["choices"][0]["delta"]
            token = delta.get("content", "")
            if not token:
                continue
            buffer += token
            # Detect and suppress prompt-leak on the fly
            leak = any(
                marker in buffer
                for marker in ["You are Vitas", "Answer the following",
                               "Include relevant Sanskrit", "accurate Ayurvedic knowledge"]
            )
            if leak:
                break   # stop streaming this section; move to next
            yield token


# ─── D. RESPONSE POST-PROCESSOR & DISPLAY ─────────────────────────────────────

# Terminal width for wrapping
_TERM_WIDTH = 80

# ANSI colour codes (gracefully degraded if terminal does not support them)
_C = {
    "reset"  : "\033[0m",
    "bold"   : "\033[1m",
    "green"  : "\033[32m",
    "cyan"   : "\033[36m",
    "yellow" : "\033[33m",
    "white"  : "\033[97m",
    "dim"    : "\033[2m",
}

# Detect whether the terminal supports colour
_COLOUR = sys.stdout.isatty()


def _col(code: str, text: str) -> str:
    """Wrap text in an ANSI colour code if the terminal supports it."""
    if not _COLOUR:
        return text
    return f"{_C[code]}{text}{_C['reset']}"


def _fmt_bold(text: str) -> str:
    if not _COLOUR:
        return text
    return f"{_C['bold']}{text}{_C['reset']}"


def _word_count(text: str) -> int:
    return len(text.split())


def _wrap(text: str, indent: int = 3) -> str:
    """Wrap a paragraph to terminal width with a leading indent."""
    prefix = " " * indent
    return textwrap.fill(
        text,
        width           = _TERM_WIDTH,
        initial_indent  = prefix,
        subsequent_indent = prefix,
    )


def format_and_print_response(raw: str, question: str, elapsed: float) -> None:
    """
    Pretty-print a Vitas response to stdout with section highlighting,
    word-wrap, and a performance footer.

    Args:
        raw      : Raw model output string.
        question : The original user question (for the header).
        elapsed  : Wall-clock time in seconds that inference took.
    """
    bar   = _col("cyan", "─" * _TERM_WIDTH)
    dbar  = _col("cyan", "═" * _TERM_WIDTH)

    print(f"\n{dbar}")
    print(_col("green", _fmt_bold(f"  🌿  Vitas Response")))
    print(_col("dim", f"  Q: {question[:_TERM_WIDTH - 6]}"))
    print(dbar)

    # Split into lines and render with light section-header colouring
    lines = raw.strip().splitlines()
    in_section = False

    for line in lines:
        stripped = line.strip()

        # Detect markdown-style bold section headers  **Heading**
        if re.match(r"^\*\*(.+?)\*\*\s*$", stripped) or re.match(r"^\d+\.\s+\*\*(.+?)\*\*", stripped):
            print()
            # Remove markdown asterisks and print as coloured header
            clean_header = re.sub(r"\*\*", "", stripped)
            print(_col("yellow", _fmt_bold(f"  {clean_header}")))
            print(_col("dim", "  " + "·" * (_TERM_WIDTH - 2)))
            in_section = True
            continue

        # Bullet points
        if stripped.startswith(("-", "•", "*")) and not stripped.startswith("**"):
            bullet_text = re.sub(r"^[-•*]\s*", "", stripped)
            print(_wrap(f"• {bullet_text}", indent=4))
            continue

        # Numbered list items  1. / 2. etc
        m = re.match(r"^(\d+)\.\s+(.+)", stripped)
        if m and not re.search(r"\*\*", stripped):
            print(_wrap(f"{m.group(1)}. {m.group(2)}", indent=4))
            continue

        # Separator lines the model sometimes produces
        if set(stripped) <= set("═─=- ") and len(stripped) > 4:
            print(_col("dim", "  " + "─" * (_TERM_WIDTH - 2)))
            continue

        # Empty line → blank spacer
        if not stripped:
            print()
            continue

        # Regular paragraph text
        print(_wrap(stripped, indent=3))

    # Footer with stats
    wc = _word_count(raw)
    print(f"\n{bar}")
    print(
        _col("dim",
             f"  ⏱  {elapsed:.1f}s  |  ~{wc} words  |  "
             f"Model: {FILENAME}")
    )
    print(bar + "\n")


# ─── E. BENCHMARK ─────────────────────────────────────────────────────────────

BENCHMARK_QUESTIONS = [
    "What is Ashwagandha and what are its primary Ayurvedic benefits?",
    "Explain the three doshas: Vata, Pitta, and Kapha.",
    "What Ayurvedic herbs are recommended for improving digestion (Agni)?",
    "How does Panchakarma detoxification work?",
    "What dietary recommendations does Ayurveda suggest for a Pitta-dominant person?",
]


def run_benchmark() -> None:
    """Run the 5 benchmark questions and display structured responses with timing."""
    _get_model()  # pre-load before the benchmark starts
    dbar = "═" * _TERM_WIDTH
    print(f"\n{dbar}")
    print("  Vitas Ayurveda — M2 Air Benchmark")
    print(f"{dbar}\n")

    total_start = time.time()
    for i, q in enumerate(BENCHMARK_QUESTIONS, 1):
        print(f"\n[{i}/{len(BENCHMARK_QUESTIONS)}] Running: {q[:70]}…")
        t0  = time.time()
        ans = chat_with_vitas(q)
        elapsed = time.time() - t0
        format_and_print_response(ans, q, elapsed)

    total = time.time() - total_start
    print(f"\n✅  Benchmark complete!  Total time: {total:.1f}s\n")


# ─── F. INTERACTIVE CHAT LOOP ─────────────────────────────────────────────────

def run_interactive() -> None:
    """
    Interactive multi-turn chat loop with streaming output and
    structured post-render.

    The response is collected in full while streaming so that the
    post-processor can apply formatting and word-count stats after
    the model finishes.
    """
    _get_model()  # pre-load before any prompts

    dbar = _col("green", "═" * _TERM_WIDTH)
    print(f"\n{dbar}")
    print(_col("green", _fmt_bold("  🌿  Vitas Ayurveda AI")))
    print(_col("dim",   "  Powered by Phi-3 Mini · Apple M2 Neural Engine"))
    print(_col("dim",   "  Type 'exit' or 'quit' to end the session"))
    print(f"{dbar}\n")

    history: list = []

    while True:
        try:
            user_input = input(_col("cyan", "You › ")).strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\n👋  Goodbye! Take care and stay healthy.")
            break

        if not user_input:
            continue
        if user_input.lower() in ("exit", "quit"):
            print("\n👋  Goodbye! Take care and stay healthy.")
            break

        # ── Stream tokens live while collecting the full reply ──────────────
        print(_col("yellow", "\n  Vitas is thinking…\n"))
        t0         = time.time()
        full_reply = ""

        # Live stream indicator without re-printing the whole response
        # (dots give feedback while the model generates)
        sys.stdout.write(_col("dim", "  "))
        for token in stream_vitas(user_input, history=history):
            full_reply += token
            # Print every 5th token as a dot to show progress
            if len(full_reply) % 40 < len(token):
                sys.stdout.write(_col("dim", "·"))
                sys.stdout.flush()
        print()  # end the dot line

        elapsed = time.time() - t0

        # ── Pretty-print the finished response ──────────────────────────────
        format_and_print_response(full_reply, user_input, elapsed)

        # Keep rolling history for multi-turn conversations
        history.append({"role": "user",      "content": user_input})
        history.append({"role": "assistant", "content": full_reply})


# ─── G. DJANGO INTEGRATION NOTES ─────────────────────────────────────────────
# Drop-in replacement for views that currently call Gemini / Ollama.
#
# In your views.py:
#   from inference_mac_m2 import chat_with_vitas, stream_vitas
#
# Blocking (simple) usage:
#   answer = chat_with_vitas(request.POST["question"])
#
# Streaming (StreamingHttpResponse):
#   from django.http import StreamingHttpResponse
#   from inference_mac_m2 import stream_vitas
#
#   def vitas_view(request):
#       q = request.GET.get("q", "")
#       return StreamingHttpResponse(
#           stream_vitas(q),
#           content_type="text/plain; charset=utf-8"
#       )
#
# Note: _get_model() is called lazily on the first request.
# For production, call _get_model() once in your Django AppConfig.ready()
# to warm the model before the first HTTP request arrives.

# ─── MAIN ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    if len(sys.argv) > 1 and sys.argv[1] == "setup":
        print("ℹ️  No manual setup required — the model downloads automatically.")
        print(f"   Repo  : {REPO_ID}")
        print(f"   File  : {FILENAME}")
        print("   Run   : python inference_mac_m2.py")

    elif len(sys.argv) > 1 and sys.argv[1] == "benchmark":
        run_benchmark()

    elif len(sys.argv) > 1:
        # Quick single-question mode:  python inference_mac_m2.py "your question"
        question = " ".join(sys.argv[1:])
        _get_model()
        t0  = time.time()
        ans = chat_with_vitas(question)
        elapsed = time.time() - t0
        format_and_print_response(ans, question, elapsed)

    else:
        # Default: interactive chat loop
        run_interactive()

