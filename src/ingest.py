"""
ingest.py
---------
Loads pre-chunked JSON into a Chroma vector database using nomic-embed-text
via Ollama for embeddings.

This is a one-time setup step. Run it once to build the knowledge base;
the agent queries it at runtime.

Usage:
    python3 src/ingest.py
"""

import json
import sys
import urllib.request
from pathlib import Path

import chromadb
from chromadb.config import Settings
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma


# ---------------------------------------------------------------------------
# Configuration — edit these as needed
# ---------------------------------------------------------------------------

CHUNKS_PATH      = "documents/45_cfr_164514_chunks.json"
DB_PATH          = "./chroma_db"
COLLECTION_NAME  = "phi_regulations"
EMBEDDING_MODEL  = "nomic-embed-text"
LLM_HOST         = "http://172.25.240.1:11434"  # Ollama host (Windows host IP from WSL)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def check_llm_connection() -> None:
    """Verify the LLM host is reachable before doing any work."""
    try:
        urllib.request.urlopen(LLM_HOST, timeout=5)
    except Exception as e:
        print(f"ERROR: cannot reach LLM host at {LLM_HOST}: {e}", file=sys.stderr)
        print("Make sure Ollama is running (check with: curl http://localhost:11434)", file=sys.stderr)
        sys.exit(1)


def check_chroma_empty() -> bool:
    """Return True if the collection is empty and ingestion should proceed."""
    raw_db = chromadb.PersistentClient(
        path=DB_PATH,
        settings=Settings(anonymized_telemetry=False),
    )
    existing_collections = [c.name for c in raw_db.list_collections()]
    if COLLECTION_NAME in existing_collections:
        count = raw_db.get_collection(COLLECTION_NAME).count()
        if count > 0:
            print(f"Collection '{COLLECTION_NAME}' already has {count} documents.")
            print("Skipping — delete the chroma_db directory to re-ingest from scratch.")
            return False
    return True


# ---------------------------------------------------------------------------
# Ingestion
# ---------------------------------------------------------------------------

def ingest() -> None:
    # shorting function on validation
    check_llm_connection()
    if not check_chroma_empty():
        return

    # Load chunks
    chunks_path = Path(CHUNKS_PATH)
    if not chunks_path.exists():
        print(f"ERROR: file not found: {chunks_path}", file=sys.stderr)
        sys.exit(1)

    with open(chunks_path, encoding="utf-8") as f:
        chunks = json.load(f)

    if not chunks:
        print("ERROR: no chunks found in input file", file=sys.stderr)
        sys.exit(1)

    print(f"Loaded {len(chunks)} chunks from {chunks_path}")

    # LangChain embedding function — swap OllamaEmbeddings for any LangChain
    # Embeddings-compatible class here (e.g. OpenAIEmbeddings, HuggingFaceEmbeddings)
    embed_fn = OllamaEmbeddings(
        model=EMBEDDING_MODEL,
        base_url=LLM_HOST,
    )

    ids       = [c["id"] for c in chunks]
    texts     = [c["text"] for c in chunks]
    metadatas = [c["metadata"] for c in chunks]

    total = len(chunks)
    batch_size = 10

    print(f"Embedding {total} chunks using '{EMBEDDING_MODEL}'...")

    chroma_client = chromadb.PersistentClient(
        path=DB_PATH,
        settings=Settings(anonymized_telemetry=False),
    )

    for i in range(0, total, batch_size):
        batch_texts = texts[i : i + batch_size]
        batch_meta  = metadatas[i : i + batch_size]
        batch_ids   = ids[i : i + batch_size]

        # Explicitly embed the batch — returns a list of float vectors
        batch_vectors = embed_fn.embed_documents(batch_texts)

        print(f"  [{i + len(batch_texts)}/{total}] embedded {len(batch_texts)} chunks "
              f"-> {len(batch_vectors[0])}-dim vectors")

        # Store vectors + original text + metadata in Chroma
        if i == 0:
            vectorstore = Chroma(
                collection_name=COLLECTION_NAME,
                embedding_function=embed_fn,
                client=chroma_client,
            )

        vectorstore._collection.add(
            ids        = batch_ids,
            embeddings = batch_vectors,
            documents  = batch_texts,
            metadatas  = batch_meta,
        )

    print(f"\nDone. {total} chunks stored in Chroma at '{DB_PATH}' (collection: '{COLLECTION_NAME}')")


if __name__ == "__main__":
    ingest()
