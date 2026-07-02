# Implementation Specification: Local Document RAG

**Date:** 2026-05-05
**Topic:** Retrieval-Augmented Generation (RAG) Pipeline for Vitas AI

## Overview
This specification details the transition from a naive "context stuffing" file processing method to a robust, local Retrieval-Augmented Generation (RAG) architecture. The system will leverage ChromaDB and `sentence-transformers` running natively on Apple Silicon (M2) to index user-uploaded documents per conversation, enabling highly accurate Q&A over large documents without exceeding the context limits of the underlying Phi-3 GGUF models.

---

## Phase 1: Environment & Core Engine Setup

**Objective:** Install dependencies and build the standalone vector storage and retrieval engine.

1. **Dependency Management:**
   - Install `chromadb` for persistent vector storage.
   - Install `sentence-transformers` for local embedding generation (`all-MiniLM-L6-v2`).
   - Update `.gitignore` to exclude the newly created `rag_storage/` directory.

2. **Develop `rag_engine.py`:**
   - Create a new module: `chatbot/rag_engine.py`.
   - **Initialization:** Set up a ChromaDB `PersistentClient` pointing to `./rag_storage`. Load the `all-MiniLM-L6-v2` embedding model.
   - **Chunking Logic:** Implement a recursive character text splitter. Target chunk size: 500 characters, Overlap: 50 characters.
   - **Indexing Function (`index_document`):** Accepts extracted text and a `conversation_id`. Splits the text, generates embeddings, and upserts them into a ChromaDB collection uniquely named after the `conversation_id` (e.g., `conv_123`).
   - **Retrieval Function (`query_documents`):** Accepts a query string and a `conversation_id`. Returns the top 3-5 most relevant text chunks.

---

## Phase 2: Ingestion Pipeline Integration

**Objective:** Intercept file uploads and seamlessly index them in the background.

1. **Update `views.py` (`upload_file` endpoint):**
   - After a file is successfully saved to the `UploadedFile` model, trigger the text extraction process (currently housed in `ai_models.py`).
   - Pass the extracted text to `rag_engine.index_document(text, conversation.id)`.
   - **Error Handling:** Wrap the indexing call in a `try/except` block. If indexing fails, log the error but allow the file upload to succeed so the UI doesn't crash.

---

## Phase 3: Retrieval Pipeline Integration

**Objective:** Modify the AI routing layer to retrieve relevant context instead of injecting the entire document.

1. **Update `ai_models.py` (`generate_ai_response`):**
   - Locate the existing logic that extracts and truncates file content (`file_context = file_context[:10000]`).
   - Replace this with a call to `rag_engine.query_documents(message, conversation_id)`.
   - **Context Formatting:** Take the retrieved chunks and format them into a structured string. Example:
     ```
     --- RETRIEVED CONTEXT ---
     [Chunk 1]
     [Chunk 2]
     -------------------------
     ```
   - Pass this formatted context into the respective model wrappers (`get_medicinal_response` and `get_ayurvedic_response`).

---

## Phase 4: Prompt Engineering & Model Tuning

**Objective:** Instruct the local GGUF models to prioritize the retrieved context and prevent hallucination.

1. **Update `medical_model.py`:**
   - Modify `_run_medical_inference` or the prompt builder to inject the RAG context clearly.
   - Add specific instructions: *"Use the provided retrieved context to answer the user's question. If the answer is not contained within the context, state that you do not know based on the document."*

2. **Update `phi3_model.py` (Ayurvedic Chained Generation):**
   - The Ayurvedic model uses 4 chained sub-calls. The retrieved context must be injected into the base prompt for *each* of these sub-calls so that the Overview, Herbs, Diet, and Takeaway sections are all grounded in the uploaded document.
   - Adjust `_SUB_PROMPTS` to accommodate the context cleanly.

---

## Phase 5: Verification & Testing

**Objective:** Ensure the system is robust, performant, and error-free.

1. **Unit Testing (Manual via `manage.py shell`):**
   - Index a sample 10-page PDF.
   - Run sample queries and verify that the retrieved chunks directly answer the query.
2. **UI Testing:**
   - Upload a document via the chat interface.
   - Ask a highly specific question about page 8.
   - Verify the AI returns the correct answer and correctly applies the formatting (Medical vs. Ayurvedic).
3. **Performance Profiling:**
   - Verify that the `sentence-transformers` embedding generation completes quickly on the M2 chip without blocking the main Django thread excessively.
