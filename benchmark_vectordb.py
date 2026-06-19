#!/usr/bin/env python3
"""
benchmark_vectordb.py
=====================

A standalone benchmark that compares LightRAG's *vector database* backends
(NanoVectorDBStorage, FaissVectorDBStorage, Qdrant, Postgres, ...) on the things
that actually matter for a vector store:

  * insert throughput      (how fast it ingests vectors)
  * query latency          (p50 / p95 / mean of a similarity search)
  * recall@k               (search accuracy vs. a brute-force ground truth)
  * on-disk index size     (for file-based backends)

This is NOT an answer-quality benchmark (that is what RAGAS / eval_rag_quality.py
do). The goal here is to measure the storage engine itself.

WHY FAKE EMBEDDINGS?
--------------------
Every LightRAG vector backend turns text into a vector *inside* its `upsert()` and
`query()` methods by calling an "embedding function". If we used a real embedding
model (e.g. OpenAI), we would mostly be timing the network + the model, not the
database. So by default this script uses a tiny, deterministic fake embedding
function (same text -> same vector, no network, no API key). That isolates the
database's own performance and keeps the benchmark free and reproducible.

If you want to benchmark with *real* embeddings instead, see the note next to
`make_embedding_func()` below.

HOW IT TALKS TO THE BACKENDS
----------------------------
It drives each backend through the exact same interface LightRAG itself uses --
the `BaseVectorStorage` contract (`upsert`, `query`, `index_done_callback`,
`drop`, ...). So the numbers reflect real LightRAG behaviour, not a side channel.

USAGE
-----
    # from the repo root, with the project venv activated
    pip install faiss-cpu          # only needed for the Faiss backend
    python benchmark_vectordb.py \
        --backends NanoVectorDBStorage,FaissVectorDBStorage \
        --num-vectors 2000 --num-queries 200 --top-k 10

Server-backed databases (Qdrant, Postgres, Milvus, Mongo, OpenSearch) also work:
install their client package, start the service, set the required environment
variables (see STORAGE_ENV_REQUIREMENTS in lightrag/kg/__init__.py), then add the
class name to --backends.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib
import json
import os
import shutil
import tempfile
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

# These come from the LightRAG package itself, so the script must be run from a
# place where `import lightrag` works (the repo root, with the venv activated).
from lightrag.utils import wrap_embedding_func_with_attrs
from lightrag.kg import STORAGES, STORAGE_IMPLEMENTATIONS, STORAGE_ENV_REQUIREMENTS
from lightrag.kg.shared_storage import initialize_share_data, finalize_share_data


# ---------------------------------------------------------------------------
# 1. The fake, deterministic embedding function
# ---------------------------------------------------------------------------
def text_to_vector(text: str, dim: int) -> np.ndarray:
    """Turn a string into a fixed, reproducible vector.

    We seed a random number generator with a hash of the text, so the *same*
    text always produces the *same* vector. This is what lets us compute a
    brute-force "ground truth" later that exactly matches what the database sees.
    """
    # Use the first 8 hex chars of the MD5 hash as an integer seed.
    seed = int(hashlib.md5(text.encode("utf-8")).hexdigest()[:8], 16)
    rng = np.random.default_rng(seed)
    return rng.standard_normal(dim).astype(np.float32)


def make_embedding_func(dim: int):
    """Build a LightRAG-compatible embedding function of the given dimension.

    `wrap_embedding_func_with_attrs(...)` returns a decorator; applying it to our
    async function produces an `EmbeddingFunc` object that the backends know how
    to call. The inner function accepts **kwargs so it harmlessly absorbs extra
    arguments the backends pass (like context="document" or _priority=5).

    To benchmark with REAL embeddings instead, replace the body below with a call
    to, e.g., `from lightrag.llm.openai import openai_embed` and return
    `await openai_embed.func(texts, model="text-embedding-3-small")` -- and set
    `dim` to that model's dimension.
    """

    async def _embed(texts: list[str], **kwargs) -> np.ndarray:
        return np.stack([text_to_vector(t, dim) for t in texts])

    return wrap_embedding_func_with_attrs(embedding_dim=dim)(_embed)


# ---------------------------------------------------------------------------
# 2. Resolving a backend name (a string) into its class
# ---------------------------------------------------------------------------
def resolve_backend_class(name: str):
    """Map a class name like "FaissVectorDBStorage" to the actual class object.

    LightRAG stores relative module paths (".kg.faiss_impl") in the STORAGES
    registry; we turn that into the absolute path "lightrag.kg.faiss_impl" and
    import it. Mirrors what LightRAG._get_storage_class does internally.
    """
    if name not in STORAGES:
        raise ValueError(
            f"Unknown backend '{name}'. Known vector backends: "
            + ", ".join(STORAGE_IMPLEMENTATIONS["VECTOR_STORAGE"]["implementations"])
        )
    relative_path = STORAGES[name]  # e.g. ".kg.faiss_impl"
    module = importlib.import_module("lightrag" + relative_path)
    return getattr(module, name)


def check_env_requirements(name: str) -> list[str]:
    """Return the list of *missing* environment variables required by a backend."""
    required = STORAGE_ENV_REQUIREMENTS.get(name, [])
    return [var for var in required if not os.getenv(var)]


# ---------------------------------------------------------------------------
# 3. The benchmark result container
# ---------------------------------------------------------------------------
@dataclass
class BackendResult:
    backend: str
    num_vectors: int = 0
    num_queries: int = 0
    top_k: int = 0
    insert_seconds: float = 0.0
    insert_vectors_per_sec: float = 0.0
    query_p50_ms: float = 0.0
    query_p95_ms: float = 0.0
    query_mean_ms: float = 0.0
    recall_at_k: float = 0.0
    index_size_bytes: int = 0
    error: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return self.__dict__


# ---------------------------------------------------------------------------
# 4. Helpers
# ---------------------------------------------------------------------------
def percentile_ms(samples_seconds: list[float], pct: float) -> float:
    """Return the given percentile of a list of durations, converted to ms."""
    if not samples_seconds:
        return 0.0
    return float(np.percentile(np.array(samples_seconds) * 1000.0, pct))


def directory_size_bytes(path: str) -> int:
    """Total size of all files under a directory (best-effort)."""
    total = 0
    for root, _dirs, files in os.walk(path):
        for fname in files:
            try:
                total += os.path.getsize(os.path.join(root, fname))
            except OSError:
                pass
    return total


def brute_force_ground_truth(
    doc_ids: list[str], doc_vectors: np.ndarray, query_vectors: np.ndarray, top_k: int
) -> list[set[str]]:
    """Compute the true top-k nearest neighbours by cosine similarity, in NumPy.

    Returns, for each query, the SET of the top_k document ids. We compare the
    backend's answers against this to measure recall.
    """
    # Normalize so that a dot product == cosine similarity.
    docs = doc_vectors / (np.linalg.norm(doc_vectors, axis=1, keepdims=True) + 1e-12)
    queries = query_vectors / (
        np.linalg.norm(query_vectors, axis=1, keepdims=True) + 1e-12
    )
    sims = queries @ docs.T  # shape: (num_queries, num_docs)
    truth: list[set[str]] = []
    for row in sims:
        # argsort ascending -> take the last top_k indices (highest similarity).
        top_idx = np.argsort(row)[-top_k:]
        truth.append({doc_ids[i] for i in top_idx})
    return truth


# ---------------------------------------------------------------------------
# 5. Benchmarking a single backend
# ---------------------------------------------------------------------------
async def benchmark_backend(
    name: str,
    args: argparse.Namespace,
    doc_ids: list[str],
    doc_texts: list[str],
    query_texts: list[str],
    ground_truth: list[set[str]],
) -> BackendResult:
    """Run the full insert / query / recall / size benchmark for one backend."""
    result = BackendResult(
        backend=name,
        num_vectors=args.num_vectors,
        num_queries=args.num_queries,
        top_k=args.top_k,
    )

    # Each backend gets its own temporary working directory so file-based
    # backends don't clash and we can measure their on-disk size in isolation.
    work_dir = tempfile.mkdtemp(prefix=f"vdbbench_{name}_")

    # Backends require a couple of config values via `global_config`. We set
    # cosine_better_than_threshold very low so the backend does NOT filter out
    # results -- random vectors can have negative cosine, and we want raw recall.
    global_config: dict[str, Any] = {
        "working_dir": work_dir,
        "embedding_batch_num": args.batch_size,
        "vector_db_storage_cls_kwargs": {
            "cosine_better_than_threshold": args.cosine_threshold
        },
    }

    embedding_func = make_embedding_func(args.dim)
    storage = None
    try:
        backend_cls = resolve_backend_class(name)
        # Build the backend object exactly the way LightRAG wires it up.
        storage = backend_cls(
            namespace="benchmark",
            workspace="",
            global_config=global_config,
            embedding_func=embedding_func,
            meta_fields=set(),
        )
        await storage.initialize()

        # ----- INSERT PHASE -----
        # The upsert payload maps id -> {"content": text, ...metadata...}.
        # The backend embeds each "content" string internally.
        insert_start = time.perf_counter()
        for start in range(0, len(doc_ids), args.batch_size):
            batch_ids = doc_ids[start : start + args.batch_size]
            batch_texts = doc_texts[start : start + args.batch_size]
            payload = {
                _id: {"content": _text}
                for _id, _text in zip(batch_ids, batch_texts)
            }
            await storage.upsert(payload)
        # Flush to disk so the timing includes persistence, like real indexing.
        await storage.index_done_callback()
        result.insert_seconds = time.perf_counter() - insert_start
        if result.insert_seconds > 0:
            result.insert_vectors_per_sec = args.num_vectors / result.insert_seconds

        # ----- QUERY PHASE -----
        latencies: list[float] = []
        hits = 0
        for q_text, truth_ids in zip(query_texts, ground_truth):
            q_start = time.perf_counter()
            returned = await storage.query(q_text, top_k=args.top_k)
            latencies.append(time.perf_counter() - q_start)
            # Count how many of the true top-k ids the backend actually returned.
            returned_ids = {r.get("id") for r in returned}
            hits += len(truth_ids & returned_ids)

        result.query_p50_ms = percentile_ms(latencies, 50)
        result.query_p95_ms = percentile_ms(latencies, 95)
        result.query_mean_ms = (sum(latencies) / len(latencies) * 1000.0) if latencies else 0.0
        # recall@k = (true neighbours found) / (true neighbours possible)
        denom = args.top_k * len(query_texts)
        result.recall_at_k = hits / denom if denom else 0.0

        # ----- DISK SIZE PHASE -----
        result.index_size_bytes = directory_size_bytes(work_dir)

    except Exception as exc:  # noqa: BLE001 - we want to report any backend failure
        result.error = f"{type(exc).__name__}: {exc}"
    finally:
        # Clean up: drop the data, finalize the backend, remove the temp dir.
        if storage is not None:
            try:
                await storage.drop()
            except Exception:
                pass
            try:
                await storage.finalize()
            except Exception:
                pass
        shutil.rmtree(work_dir, ignore_errors=True)

    return result


# ---------------------------------------------------------------------------
# 6. Orchestration + reporting
# ---------------------------------------------------------------------------
def build_dataset(args: argparse.Namespace):
    """Create synthetic documents, queries, and the brute-force ground truth."""
    doc_ids = [f"vec-{i}" for i in range(args.num_vectors)]
    doc_texts = [f"benchmark-document-{i}" for i in range(args.num_vectors)]
    # Queries use a different text prefix so they aren't identical to documents.
    query_texts = [f"benchmark-query-{i}" for i in range(args.num_queries)]

    doc_vectors = np.stack([text_to_vector(t, args.dim) for t in doc_texts])
    query_vectors = np.stack([text_to_vector(t, args.dim) for t in query_texts])
    ground_truth = brute_force_ground_truth(
        doc_ids, doc_vectors, query_vectors, args.top_k
    )
    return doc_ids, doc_texts, query_texts, ground_truth


def print_report(results: list[BackendResult]) -> None:
    header = (
        f"{'Backend':<26} {'Insert(s)':>10} {'Vec/s':>10} "
        f"{'p50(ms)':>9} {'p95(ms)':>9} {'mean(ms)':>9} "
        f"{'Recall@k':>9} {'Size(KB)':>10}"
    )
    print("\n" + "=" * len(header))
    print("VECTOR DATABASE BENCHMARK RESULTS")
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for r in results:
        if r.error:
            print(f"{r.backend:<26} ERROR: {r.error}")
            continue
        print(
            f"{r.backend:<26} {r.insert_seconds:>10.3f} "
            f"{r.insert_vectors_per_sec:>10.0f} "
            f"{r.query_p50_ms:>9.2f} {r.query_p95_ms:>9.2f} {r.query_mean_ms:>9.2f} "
            f"{r.recall_at_k:>9.3f} {r.index_size_bytes / 1024:>10.1f}"
        )
    print("=" * len(header) + "\n")


async def main_async(args: argparse.Namespace) -> None:
    backends = [b.strip() for b in args.backends.split(",") if b.strip()]

    # The storage backends rely on a shared lock/flag registry. Initialize it
    # once for the whole run (single-process mode).
    initialize_share_data(workers=1)

    print(
        f"Dataset: {args.num_vectors} vectors x {args.dim} dims | "
        f"{args.num_queries} queries | top_k={args.top_k}\n"
    )
    doc_ids, doc_texts, query_texts, ground_truth = build_dataset(args)

    results: list[BackendResult] = []
    try:
        for name in backends:
            missing = check_env_requirements(name)
            if missing:
                print(
                    f"[skip] {name}: missing required env vars: {', '.join(missing)}"
                )
                results.append(
                    BackendResult(backend=name, error=f"missing env: {missing}")
                )
                continue
            print(f"[run]  {name} ...")
            results.append(
                await benchmark_backend(
                    name, args, doc_ids, doc_texts, query_texts, ground_truth
                )
            )
    finally:
        finalize_share_data()

    print_report(results)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump([r.as_dict() for r in results], f, indent=2)
    print(f"Wrote detailed results to {args.output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark LightRAG vector database backends "
        "(insert speed, query latency, recall, index size)."
    )
    parser.add_argument(
        "--backends",
        default="NanoVectorDBStorage,FaissVectorDBStorage",
        help="Comma-separated backend class names to benchmark.",
    )
    parser.add_argument(
        "--num-vectors", type=int, default=2000, help="Number of vectors to insert."
    )
    parser.add_argument(
        "--num-queries", type=int, default=200, help="Number of search queries to run."
    )
    parser.add_argument(
        "--top-k", type=int, default=10, help="Neighbours to retrieve per query."
    )
    parser.add_argument(
        "--dim", type=int, default=768, help="Embedding dimension (fake embeddings)."
    )
    parser.add_argument(
        "--batch-size", type=int, default=100, help="Upsert/embedding batch size."
    )
    parser.add_argument(
        "--cosine-threshold",
        type=float,
        default=-1.0,
        help="Min cosine for a result to count. Keep very low so the backend "
        "does not filter results during the recall measurement.",
    )
    parser.add_argument(
        "--output",
        default="benchmark_results.json",
        help="Where to write the JSON results.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    asyncio.run(main_async(parse_args()))
