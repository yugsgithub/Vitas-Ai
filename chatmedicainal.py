#!/usr/bin/env python3
"""
Vitas AI — Medical Q&A Terminal Interface (M2 Optimised)
=========================================================
Uses llama-cpp-python directly for Metal acceleration on Apple Silicon.
Model: Shubh769/medical_phi3_q4km.gguf (loaded from ~/.cache/huggingface/hub/)

Training data:  NIH MedQuad · USMLE MedQA · WikiDoc · PubMed
"""

import sys
import time
import os
import re
import textwrap
from pathlib import Path

# ─── llama-cpp-python ─────────────────────────────────────────────────────────
try:
    from llama_cpp import Llama
except ImportError:
    print("\n  ❌  llama-cpp-python is not installed.")
    print("      Run:  pip install llama-cpp-python\n")
    sys.exit(1)

# ─── Model Configuration ──────────────────────────────────────────────────────
# Model loaded from HuggingFace Hub cache — no hardcoded path needed.
# llama-cpp-python resolves it automatically from ~/.cache/huggingface/hub/
MEDICAL_REPO_ID  = "Shubh769/medical_phi3_q4km.gguf"
MEDICAL_FILENAME = "medical_phi3_q4km.gguf"
THREADS = os.cpu_count() or 4

# ─── PostgreSQL DB Config (for status banner) ─────────────────────────────────
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent / ".env")
    DB_NAME = os.environ.get("DB_NAME", "vitas_db")
    DB_HOST = os.environ.get("DB_HOST", "localhost")
    DB_PORT = os.environ.get("DB_PORT", "5432")
    _ENV_LOADED = True
except ImportError:
    DB_NAME, DB_HOST, DB_PORT = "vitas_db", "localhost", "5432"
    _ENV_LOADED = False


# ─── ANSI Palette ─────────────────────────────────────────────────────────────
class A:
    RST    = "\033[0m"
    BOLD   = "\033[1m"
    DIM    = "\033[2m"
    ITAL   = "\033[3m"
    UL     = "\033[4m"
    WHITE  = "\033[97m"
    CYAN   = "\033[96m"
    TEAL   = "\033[36m"
    GREEN  = "\033[92m"
    LIME   = "\033[32m"
    YELLOW = "\033[93m"
    ORANGE = "\033[38;5;214m"
    RED    = "\033[91m"
    BLUE   = "\033[94m"
    PURPLE = "\033[95m"
    GREY   = "\033[90m"


W = 70   # terminal width


def hr(char="─", color=A.GREY):
    print(f"{color}{char * W}{A.RST}")


def box_top(color=A.CYAN):
    print(f"{color}╔{'═' * (W-2)}╗{A.RST}")


def box_mid(color=A.CYAN):
    print(f"{color}╠{'═' * (W-2)}╣{A.RST}")


def box_bot(color=A.CYAN):
    print(f"{color}╚{'═' * (W-2)}╝{A.RST}")


def box_row(text, color=A.CYAN, text_color=A.WHITE):
    pad = W - 2 - len(text)
    print(f"{color}║{A.RST}{text_color} {text}{' ' * (max(0, pad-1))}{A.RST}{color}║{A.RST}")


def section_header(title: str, icon: str, color: str):
    inner = f"  {icon}  {title}  "
    pad   = W - 2 - len(inner)
    print(f"\n{color}┌{'─' * (W-2)}┐")
    print(f"│{A.BOLD}{A.WHITE}{inner}{' ' * max(0, pad)}{A.RST}{color}│")
    print(f"└{'─' * (W-2)}┘{A.RST}")


def wrap_print(text: str, indent=4, width=None, color=A.WHITE):
    if width is None:
        width = W - indent
    for para in text.split("\n"):
        if not para.strip():
            print()
            continue
        wrapped = textwrap.fill(
            para.strip(), width=width,
            initial_indent=" " * indent,
            subsequent_indent=" " * indent
        )
        print(f"{color}{wrapped}{A.RST}")


def bullet_print(text: str, color=A.GREEN):
    line = text.strip().lstrip("-•").strip()
    print(f"{color}    ◆  {A.WHITE}{line}{A.RST}")


# ─── DB Status Check ──────────────────────────────────────────────────────────
def _check_db_status() -> str:
    """Return a one-line status string for the PostgreSQL connection."""
    try:
        import psycopg2
        conn = psycopg2.connect(
            dbname=DB_NAME,
            user=os.environ.get("DB_USER", "vitas_user"),
            password=os.environ.get("DB_PASSWORD", ""),
            host=DB_HOST,
            port=int(DB_PORT),
            connect_timeout=2,
        )
        conn.close()
        return f"PostgreSQL {DB_NAME}@{DB_HOST}:{DB_PORT}  ✓"
    except Exception as e:
        return f"PostgreSQL  ✗  ({str(e)[:40]})"


# ─── Banner ───────────────────────────────────────────────────────────────────
def banner(model_gb: float = 2.4):
    db_status = _check_db_status()
    print()
    box_top(A.CYAN)
    box_row("", A.CYAN, A.WHITE)
    box_row("  🏥  VITAS AI  —  Medical Intelligence Assistant", A.CYAN, A.BOLD + A.CYAN)
    box_row(f"      Phi-3-mini 3.8B  ·  Q4_K_M  ·  {model_gb:.2f} GB  ·  M2 GPU", A.CYAN, A.DIM + A.WHITE)
    box_row("", A.CYAN, A.WHITE)
    box_mid(A.CYAN)
    box_row("  Trained on:  NIH MedQuad  ·  USMLE MedQA  ·  WikiDoc  ·  PubMed", A.CYAN, A.TEAL)
    box_row(f"  DB:  {db_status}", A.CYAN, A.GREEN if "✓" in db_status else A.YELLOW)
    box_row("  Type your medical question and press Enter", A.CYAN, A.DIM + A.WHITE)
    box_row("  Type  'quit'  to exit", A.CYAN, A.DIM + A.WHITE)
    box_bot(A.CYAN)
    print()


# ─── System Prompt ────────────────────────────────────────────────────────────
SYSTEM = """You are Vitas, a board-certified medical AI assistant with expertise \
in pathophysiology, pharmacology, clinical medicine, and evidence-based treatment.

For every question, write a complete, detailed, well-structured clinical answer \
using these exact sections:

## Overview
Briefly introduce the topic (2-3 sentences).

## Pathophysiology
Explain the underlying biological mechanism in depth.

## Clinical Features
Describe signs, symptoms, and presentation.

## Diagnosis
Describe investigations, lab tests, imaging, and criteria.

## Management
Cover lifestyle, pharmacological, and procedural treatment.

## Key Teaching Points
3-5 bullet points with clinical pearls.

Always end with: "DISCLAIMER: Please consult a licensed physician."
"""

# ─── Response Renderer ────────────────────────────────────────────────────────
SECTION_MAP = {
    "overview":           ("📋", "OVERVIEW",           A.BLUE),
    "pathophysiology":    ("🔬", "PATHOPHYSIOLOGY",    A.PURPLE),
    "clinical features":  ("🩺", "CLINICAL FEATURES",  A.ORANGE),
    "diagnosis":          ("🧪", "DIAGNOSIS",           A.YELLOW),
    "management":         ("💊", "MANAGEMENT",          A.GREEN),
    "key teaching points":("⭐", "KEY TEACHING POINTS", A.CYAN),
}


def render_stream(stream) -> str:
    """Render streaming GGUF output with section-aware formatting."""
    full_response = ""
    buffer = ""

    print(f"{A.WHITE}", end="", flush=True)

    for chunk in stream:
        token = chunk["choices"][0]["delta"].get("content", "")
        full_response += token
        buffer += token

        for key, (icon, title, color) in SECTION_MAP.items():
            pattern = f"## {key.title()}"
            if pattern in buffer:
                pre_header = buffer.split(pattern)[0]
                print(pre_header.strip(), end="", flush=True)
                section_header(title, icon, color)
                buffer = ""
                print(f"{A.WHITE}", end="", flush=True)
                break

        if " " in token or "\n" in token:
            print(buffer, end="", flush=True)
            buffer = ""

    print(buffer)  # print remaining buffer
    return full_response


# ─── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"\n{A.YELLOW}  ⏳  Loading Medical GGUF model from HuggingFace cache…{A.RST}", end="", flush=True)

    try:
        llm = Llama.from_pretrained(
            repo_id      = MEDICAL_REPO_ID,
            filename     = MEDICAL_FILENAME,
            n_gpu_layers = -1,    # Full Metal GPU offload for M2
            n_ctx        = 4096,
            verbose      = False,
        )
        # Determine model size for banner
        try:
            from huggingface_hub import hf_hub_download
            import os as _os
            _path = hf_hub_download(repo_id=MEDICAL_REPO_ID, filename=MEDICAL_FILENAME)
            model_gb = _os.path.getsize(_path) / 1e9
        except Exception:
            model_gb = 2.4

        print(f"{A.GREEN} Done!{A.RST}\n")
    except Exception as e:
        print(f"\n{A.RED}  ❌ Error loading model: {e}{A.RST}")
        print(f"\n  Ensure the model is cached by running:")
        print(f"  python mecinal_model.py\n")
        sys.exit(1)

    banner(model_gb)

    # ── Interactive chat loop ──────────────────────────────────────────────────
    while True:
        try:
            q = input(f"{A.CYAN}{A.BOLD}  ❓  Question:{A.RST}  ").strip()
        except (EOFError, KeyboardInterrupt):
            print(f"\n{A.DIM}  Goodbye! Stay healthy 🏥{A.RST}\n")
            break

        if not q:
            continue
        if q.lower() in ("quit", "exit", "q"):
            print(f"\n{A.DIM}  Goodbye! Stay healthy 🏥{A.RST}\n")
            break

        print()
        hr("─", A.GREY)
        print(f"{A.BOLD}{A.WHITE}  🗣  {q}{A.RST}")
        hr("─", A.GREY)
        print()

        messages = [
            {"role": "system", "content": SYSTEM},
            {"role": "user",   "content": q},
        ]

        t0 = time.time()
        stream = llm.create_chat_completion(
            messages       = messages,
            stream         = True,
            max_tokens     = 1024,
            temperature    = 0.35,
            repeat_penalty = 1.10,
        )

        response_text = render_stream(stream)
        elapsed = time.time() - t0
        words   = len(response_text.split())

        # Footer stats
        print()
        hr("═", A.GREY)
        stats = f"  📝 {words} words  ·  ⏱  {elapsed:.1f}s  ·  ⚡ {words/elapsed:.1f} w/s  ·  M2 GPU Accelerated  "
        print(f"{A.DIM}{stats}{A.RST}")
        hr("═", A.GREY)
        print()