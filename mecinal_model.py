#!/usr/bin/env python3
"""
mecinal_model.py — Vitas AI | Medical GGUF Model Verification Script
=====================================================================
Verifies that the local Medical GGUF model loads correctly from the
HuggingFace Hub cache at ~/.cache/huggingface/hub/ and can produce a
valid medical response.

Usage:
    python mecinal_model.py                      # Quick load + sample inference
    python mecinal_model.py --check-only         # Load check only (no inference)
    python mecinal_model.py --query "headache"   # Custom test query

Run this to confirm the model is cached and working before starting
the Django server or the chatmedicainal.py terminal UI.
"""

import sys
import time
import argparse
from pathlib import Path

# ─── llama-cpp-python ─────────────────────────────────────────────────────────
try:
    from llama_cpp import Llama
    LLAMA_OK = True
except ImportError:
    LLAMA_OK = False

# ─── HuggingFace Hub ──────────────────────────────────────────────────────────
try:
    from huggingface_hub import hf_hub_download, try_to_load_from_cache
    HF_OK = True
except ImportError:
    HF_OK = False

# ─── Model identifiers ────────────────────────────────────────────────────────
REPO_ID  = "Shubh769/medical_phi3_q4km.gguf"
FILENAME = "medical_phi3_q4km.gguf"
HF_CACHE = Path.home() / ".cache" / "huggingface" / "hub"

# ─── ANSI colours ─────────────────────────────────────────────────────────────
GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
CYAN   = "\033[96m"
GREY   = "\033[90m"
BOLD   = "\033[1m"
RST    = "\033[0m"

SEP = f"{GREY}{'─' * 60}{RST}"


def _ok(msg):   print(f"{GREEN}  ✅  {msg}{RST}")
def _warn(msg): print(f"{YELLOW}  ⚠️   {msg}{RST}")
def _err(msg):  print(f"{RED}  ❌  {msg}{RST}")
def _info(msg): print(f"{CYAN}  ℹ️   {msg}{RST}")


def check_dependencies() -> bool:
    """Verify required packages are installed."""
    print(f"\n{BOLD}[1/4] Dependency Check{RST}")
    print(SEP)
    ok = True

    if LLAMA_OK:
        _ok("llama-cpp-python  — found")
    else:
        _err("llama-cpp-python  — NOT installed")
        _info("Fix:  pip install llama-cpp-python")
        ok = False

    if HF_OK:
        _ok("huggingface_hub   — found")
    else:
        _warn("huggingface_hub   — not installed (optional for cache check)")
        _info("Fix:  pip install huggingface_hub")

    return ok


def check_hf_cache() -> bool:
    """Check whether the GGUF model is already in the HF cache."""
    print(f"\n{BOLD}[2/4] HuggingFace Cache Check{RST}")
    print(SEP)
    _info(f"Cache dir:  {HF_CACHE}")

    if not HF_CACHE.exists():
        _warn("HuggingFace cache directory does not exist yet.")
        _info("Model will be downloaded on first load (~2.4 GB).")
        return False

    # Try to resolve via huggingface_hub
    if HF_OK:
        cached_path = try_to_load_from_cache(repo_id=REPO_ID, filename=FILENAME)
        if cached_path and cached_path != "no_blob_found":
            size_gb = Path(cached_path).stat().st_size / 1e9
            _ok(f"Model found in cache  ({size_gb:.2f} GB)")
            _info(f"Path:  {cached_path}")
            return True
        else:
            _warn("Model NOT found in cache — will be downloaded on first load.")
            return False
    else:
        # Fallback: search cache dir manually
        matches = list(HF_CACHE.rglob(FILENAME))
        if matches:
            size_gb = matches[0].stat().st_size / 1e9
            _ok(f"Model found in cache  ({size_gb:.2f} GB)")
            _info(f"Path:  {matches[0]}")
            return True
        else:
            _warn("Model NOT in cache — will be downloaded on first use.")
            return False


def load_model() -> "Llama | None":
    """Load the GGUF model using Llama.from_pretrained."""
    print(f"\n{BOLD}[3/4] Model Load Test{RST}")
    print(SEP)
    _info(f"Loading  {REPO_ID}/{FILENAME}  with M2 Metal offload…")

    t0 = time.time()
    try:
        llm = Llama.from_pretrained(
            repo_id      = REPO_ID,
            filename     = FILENAME,
            n_gpu_layers = -1,     # Full Metal GPU offload (Apple M2)
            n_ctx        = 4096,
            verbose      = False,
        )
        elapsed = time.time() - t0
        _ok(f"Model loaded in {elapsed:.1f}s")
        return llm
    except Exception as exc:
        _err(f"Model load failed: {exc}")
        return None


def run_inference(llm: "Llama", query: str) -> bool:
    """Run a sample inference call to verify end-to-end functionality."""
    print(f"\n{BOLD}[4/4] Inference Test{RST}")
    print(SEP)
    _info(f"Query:  {query}")
    print()

    system = (
        "You are a medical AI. Answer the question concisely in 3 sentences. "
        "End with: 'DISCLAIMER: Consult a physician.'"
    )

    t0 = time.time()
    try:
        resp = llm.create_chat_completion(
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": query},
            ],
            max_tokens     = 256,
            temperature    = 0.35,
            repeat_penalty = 1.10,
        )
        elapsed = time.time() - t0
        response_text = resp["choices"][0]["message"]["content"].strip()

        print(f"{CYAN}{'─' * 60}{RST}")
        print(response_text)
        print(f"{CYAN}{'─' * 60}{RST}")

        words = len(response_text.split())
        _ok(f"Inference complete — {words} words in {elapsed:.1f}s  ({words/elapsed:.1f} w/s)")
        return True
    except Exception as exc:
        _err(f"Inference failed: {exc}")
        return False


def check_db() -> bool:
    """Verify PostgreSQL connectivity using .env credentials."""
    print(f"\n{BOLD}[DB] PostgreSQL Connectivity Check{RST}")
    print(SEP)
    try:
        from dotenv import load_dotenv
        import os
        load_dotenv(Path(__file__).parent / ".env")
    except ImportError:
        _warn("python-dotenv not installed — skipping .env load")
        import os

    db_name = os.environ.get("DB_NAME", "vitas_db")
    db_user = os.environ.get("DB_USER", "vitas_user")
    db_pass = os.environ.get("DB_PASSWORD", "")
    db_host = os.environ.get("DB_HOST", "localhost")
    db_port = int(os.environ.get("DB_PORT", "5432"))

    try:
        import psycopg2
        conn = psycopg2.connect(
            dbname=db_name, user=db_user, password=db_pass,
            host=db_host, port=db_port, connect_timeout=3,
        )
        cur = conn.cursor()
        cur.execute("SELECT version();")
        ver = cur.fetchone()[0]
        conn.close()
        _ok(f"Connected to  {db_name}@{db_host}:{db_port}")
        _info(f"PostgreSQL:  {ver.split(',')[0]}")
        return True
    except ImportError:
        _warn("psycopg2 not installed — skipping DB check")
        _info("Fix:  pip install psycopg2-binary")
        return False
    except Exception as e:
        _err(f"DB connection failed: {e}")
        _info(f"Ensure PostgreSQL is running and .env has correct DB_* vars")
        return False


def main():
    parser = argparse.ArgumentParser(description="Vitas AI Medical GGUF Model Verification")
    parser.add_argument("--check-only", action="store_true",
                        help="Only check deps and cache — skip load and inference")
    parser.add_argument("--query", type=str, default="What is hypertension and how is it managed?",
                        help="Test query for inference check (default: hypertension)")
    parser.add_argument("--skip-db", action="store_true",
                        help="Skip PostgreSQL connectivity check")
    args = parser.parse_args()

    print(f"\n{BOLD}{CYAN}{'═' * 60}{RST}")
    print(f"{BOLD}{CYAN}  Vitas AI — Medical GGUF Model Verification{RST}")
    print(f"{BOLD}{CYAN}{'═' * 60}{RST}")

    all_ok = True

    # 1. Dependencies
    if not check_dependencies():
        print(f"\n{RED}  ⛔  Critical dependencies missing. Fix them and re-run.{RST}\n")
        sys.exit(1)

    # 2. Cache check
    check_hf_cache()

    # 3. DB check (before model load)
    if not args.skip_db:
        db_ok = check_db()
        if not db_ok:
            all_ok = False

    if args.check_only:
        print(f"\n{GREY}  (check-only mode — skipping model load & inference){RST}\n")
        sys.exit(0 if all_ok else 1)

    # 4. Model load
    llm = load_model()
    if llm is None:
        print(f"\n{RED}  ⛔  Model failed to load. See error above.{RST}\n")
        sys.exit(1)

    # 5. Inference
    if not run_inference(llm, args.query):
        all_ok = False

    # Summary
    print(f"\n{BOLD}{'═' * 60}{RST}")
    if all_ok:
        print(f"{GREEN}{BOLD}  ✅  All checks passed — Medical GGUF model is ready!{RST}")
        print(f"{GREY}  Run:  python chatmedicainal.py   (terminal UI){RST}")
        print(f"{GREY}  Run:  python manage.py runserver  (web UI){RST}")
    else:
        print(f"{YELLOW}{BOLD}  ⚠️   Some checks failed — review warnings above.{RST}")
    print(f"{BOLD}{'═' * 60}{RST}\n")

    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()