#!/usr/bin/env python3
"""
Explore the osm_vectorstore_nomic Chroma vector store.

Shows a few example queries and the top-k retrieved documents,
so you can understand what the RAG baseline is actually retrieving.

Usage
-----
  python explore_vectorstore.py
  python explore_vectorstore.py --k 5 --queries 3
"""

import argparse
import json
import os
from pathlib import Path

from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

ROOT = Path(__file__).parent
STORE_DIR = ROOT / "osm_vectorstore_nomic"

EXAMPLE_QUERIES = [
    "What's the closest fast food restaurant in the vicinity of Schoolhouse Gallery, Provincetown, MA?",
    "What's the closest available gallery to Kenan House, Wilmington, NC?",
    "What's the closest garden you'd suggest near Facility For Advanced Spatial Technology, Denver, CO?",
    "Which fast food restaurant is nearest from White Memorial Foundation, Litchfield, CT?",
    "Which gallery is the closest one near Strawberry Hill, San Francisco, CA?",
]


def load_store(base_url: str) -> Chroma:
    embeddings = OllamaEmbeddings(
        model="nomic-embed-text",
        base_url=base_url,
    )
    return Chroma(
        collection_name="osm",
        embedding_function=embeddings,
        persist_directory=str(STORE_DIR),
    )


def pretty_doc(rank: int, doc) -> str:
    try:
        obj = json.loads(doc.page_content)
        content = json.dumps(obj, indent=2)
    except json.JSONDecodeError:
        content = doc.page_content
    return f"  [{rank}] {content}"


def main():
    parser = argparse.ArgumentParser(description="Explore osm_vectorstore_nomic")
    parser.add_argument("--k", type=int, default=3, help="Number of results per query (default: 3)")
    parser.add_argument("--queries", type=int, default=len(EXAMPLE_QUERIES),
                        help=f"Number of example queries to run (default: {len(EXAMPLE_QUERIES)})")
    parser.add_argument("--ollama-url", default=os.environ.get("OLLAMA_HOST", "http://localhost:11434"),
                        help="Ollama base URL (default: http://localhost:11434)")
    args = parser.parse_args()

    if not STORE_DIR.exists():
        print(f"[error] Vector store not found at: {STORE_DIR}")
        print("        Run the RAG baseline first:  python baselines.py --model sonnet4.6 --baseline rag --embeddings nomic")
        return

    print(f"Loading vector store from: {STORE_DIR}")
    store = load_store(args.ollama_url)
    total = store._collection.count()
    print(f"Total documents in store:  {total}\n")
    print("=" * 70)

    for i, query in enumerate(EXAMPLE_QUERIES[: args.queries]):
        print(f"\nQuery {i + 1}: {query!r}")
        print("-" * 70)
        docs = store.similarity_search(query, k=args.k)
        if not docs:
            print("  (no results)")
        for rank, doc in enumerate(docs, 1):
            print(pretty_doc(rank, doc))
        print()

    print("=" * 70)
    print("Done.")


if __name__ == "__main__":
    main()
