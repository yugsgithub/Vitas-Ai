"""
rag_engine.py — Vitas AI | Local Document RAG Engine
=====================================================
Implements a fully local Retrieval-Augmented Generation (RAG) pipeline
using ChromaDB for vector storage and sentence-transformers for embeddings.

Runs natively on Apple M2 without any external API calls.

Public API:
  index_document(text, conversation_id)          -> None
  query_documents(query, conversation_id, n=4)   -> str  (formatted context block)
  has_documents(conversation_id)                 -> bool
  delete_collection(conversation_id)             -> None

Storage: ./rag_storage/  (excluded from git via .gitignore)
Model:   sentence-transformers/all-MiniLM-L6-v2  (~80 MB, M2-friendly)
"""

import logging
import os
from typing import List, Optional

logger = logging.getLogger(__name__)

# ─── CONSTANTS ────────────────────────────────────────────────────────────────
RAG_STORAGE_PATH   = os.path.join(os.path.dirname(os.path.dirname(__file__)), "rag_storage")
EMBEDDING_MODEL    = "all-MiniLM-L6-v2"
CHUNK_SIZE         = 500    # target characters per chunk
CHUNK_OVERLAP      = 50     # overlap between consecutive chunks
DEFAULT_N_RESULTS  = 10     # Increased to 10 for longer, more comprehensive answers
MIN_CHUNK_LENGTH   = 50     # skip chunks shorter than this (noise)

# ─── LAZY SINGLETONS ─────────────────────────────────────────────────────────
_chroma_client    = None
_embedding_model  = None


def _get_client():
    """Lazy-init the ChromaDB PersistentClient. Thread-safe via Python GIL."""
    global _chroma_client
    if _chroma_client is None:
        try:
            import chromadb
            os.makedirs(RAG_STORAGE_PATH, exist_ok=True)
            _chroma_client = chromadb.PersistentClient(path=RAG_STORAGE_PATH)
            logger.info(f"[RAG] ChromaDB initialised at: {RAG_STORAGE_PATH}")
        except ImportError:
            logger.error("[RAG] chromadb not installed. Run: pip install chromadb")
            raise
    return _chroma_client


def _get_embedding_model():
    """Lazy-init the sentence-transformer embedding model."""
    global _embedding_model
    if _embedding_model is None:
        try:
            from sentence_transformers import SentenceTransformer
            logger.info(f"[RAG] Loading embedding model: {EMBEDDING_MODEL}")
            _embedding_model = SentenceTransformer(EMBEDDING_MODEL)
            logger.info("[RAG] ✅ Embedding model loaded.")
        except ImportError:
            logger.error("[RAG] sentence-transformers not installed. Run: pip install sentence-transformers")
            raise
    return _embedding_model


def _collection_name(conversation_id) -> str:
    """Convert a conversation ID to a valid ChromaDB collection name."""
    return f"conv_{conversation_id}"


# ─── CHUNKING ─────────────────────────────────────────────────────────────────

def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """
    Split text into overlapping chunks using a recursive character splitter.

    Tries to split on paragraph boundaries first (double newline), then
    single newlines, then spaces, and finally hard-cuts at 'size' chars.

    Args:
        text:    The full document text.
        size:    Target chunk size in characters.
        overlap: Number of characters to overlap between adjacent chunks.

    Returns:
        List of non-empty text chunks.
    """
    if not text or not text.strip():
        return []

    separators = ["\n\n", "\n", ". ", " ", ""]
    chunks: List[str] = []

    def _split(txt: str, sep_idx: int) -> List[str]:
        if len(txt) <= size:
            return [txt]
        sep = separators[sep_idx] if sep_idx < len(separators) else ""
        if sep == "":
            # Hard cut
            parts = []
            i = 0
            while i < len(txt):
                parts.append(txt[i: i + size])
                i += size - overlap
            return parts
        parts = txt.split(sep)
        result = []
        current = ""
        for part in parts:
            candidate = current + (sep if current else "") + part
            if len(candidate) <= size:
                current = candidate
            else:
                if current:
                    result.append(current)
                if len(part) > size:
                    result.extend(_split(part, sep_idx + 1))
                    current = ""
                else:
                    current = part
        if current:
            result.append(current)
        return result

    raw_chunks = _split(text.strip(), 0)

    # Apply overlap and filter noise
    for i, chunk in enumerate(raw_chunks):
        chunk = chunk.strip()
        if len(chunk) < MIN_CHUNK_LENGTH:
            continue
        # Prepend overlap from previous chunk for context continuity
        if i > 0 and overlap > 0:
            prev = raw_chunks[i - 1].strip()
            tail = prev[-overlap:] if len(prev) >= overlap else prev
            chunk = tail + " " + chunk
        chunks.append(chunk)

    return chunks


# ─── INDEXING ─────────────────────────────────────────────────────────────────

def index_document(text: str, conversation_id) -> None:
    """
    Chunk a document's text, embed all chunks, and upsert into ChromaDB.

    Uses the conversation_id as the collection namespace so each conversation
    has its own isolated vector store.

    Args:
        text:            Full extracted text from the uploaded document.
        conversation_id: The Django ChatConversation primary key (int or str).

    Returns:
        None. Raises on critical failures (caller should wrap in try/except).
    """
    if not text or not text.strip():
        logger.warning("[RAG] index_document called with empty text — skipping.")
        return

    chunks = chunk_text(text)
    if not chunks:
        logger.warning("[RAG] No chunks produced from document — skipping.")
        return

    logger.info(f"[RAG] Indexing {len(chunks)} chunks for conversation {conversation_id}")

    client     = _get_client()
    embedder   = _get_embedding_model()
    col_name   = _collection_name(conversation_id)

    # Get or create collection
    collection = client.get_or_create_collection(
        name     = col_name,
        metadata = {"hnsw:space": "cosine"},
    )

    # Generate embeddings (sentence-transformers returns a numpy array)
    embeddings = embedder.encode(chunks, show_progress_bar=False).tolist()

    # Build IDs — prefix with chunk index to allow upsert idempotency
    ids = [f"chunk_{i}" for i in range(len(chunks))]

    collection.upsert(
        ids        = ids,
        documents  = chunks,
        embeddings = embeddings,
    )

    logger.info(f"[RAG] ✅ Indexed {len(chunks)} chunks into collection '{col_name}'.")


# ─── RETRIEVAL ────────────────────────────────────────────────────────────────

def query_documents(
    query: str,
    conversation_id,
    n_results: int = DEFAULT_N_RESULTS,
) -> Optional[str]:
    """
    Retrieve the most semantically relevant text chunks for a query.

    Args:
        query:           The user's question / search string.
        conversation_id: The Django ChatConversation primary key.
        n_results:       Number of top chunks to retrieve (default: 4).

    Returns:
        A formatted context block string ready for injection into a model prompt,
        or None if no documents are indexed for this conversation.
    """
    if not has_documents(conversation_id):
        return None

    client   = _get_client()
    embedder = _get_embedding_model()
    col_name = _collection_name(conversation_id)

    try:
        collection = client.get_collection(name=col_name)
    except Exception:
        logger.info(f"[RAG] No collection found for conversation {conversation_id}")
        return None

    # Embed the query
    query_embedding = embedder.encode([query], show_progress_bar=False).tolist()[0]

    # Query ChromaDB
    results = collection.query(
        query_embeddings = [query_embedding],
        n_results        = min(n_results, collection.count()),
        include          = ["documents", "distances"],
    )

    docs = results.get("documents", [[]])[0]
    if not docs:
        return None

    logger.info(f"[RAG] Retrieved {len(docs)} chunks for query: '{query[:60]}...'")

    # Format into a structured context block for the model
    formatted_chunks = "\n\n".join(
        f"[Chunk {i + 1}]\n{chunk.strip()}" for i, chunk in enumerate(docs)
    )

    return (
        "--- RETRIEVED DOCUMENT CONTEXT ---\n"
        f"{formatted_chunks}\n"
        "----------------------------------"
    )


# ─── UTILITIES ────────────────────────────────────────────────────────────────

def has_documents(conversation_id) -> bool:
    """
    Check whether any documents have been indexed for a conversation.

    Args:
        conversation_id: The Django ChatConversation primary key.

    Returns:
        True if the collection exists and has at least one chunk.
    """
    try:
        client = _get_client()
        col = client.get_collection(name=_collection_name(conversation_id))
        return col.count() > 0
    except Exception:
        return False


def delete_collection(conversation_id) -> None:
    """
    Delete all indexed chunks for a conversation (call on conversation delete).

    Args:
        conversation_id: The Django ChatConversation primary key.
    """
    try:
        client = _get_client()
        client.delete_collection(name=_collection_name(conversation_id))
        logger.info(f"[RAG] Deleted collection for conversation {conversation_id}")
    except Exception as e:
        logger.warning(f"[RAG] Could not delete collection for {conversation_id}: {e}")
