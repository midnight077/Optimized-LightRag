# Understanding LightRAG — A Beginner's Guide

This document explains three things about the LightRAG codebase, written for someone
who is **new to Python**:

1. **The full journey of a document** — from the moment you upload it (in the web UI or
   from a script) until it has been turned into a searchable *knowledge graph*.
2. **Which vector database LightRAG uses by default**, and how to switch it to another
   one (for example, Faiss).
3. **How to benchmark different vector databases** — measuring *storage* speed and
   accuracy (insert time, query latency, recall), **not** the quality of the final
   answers (which is what tools like RAGAS measure).

Throughout, file references look like [lightrag/lightrag.py:1237](lightrag/lightrag.py#L1237)
and are clickable in most editors.

---

## A few Python ideas you'll need

If you're new to Python, these five ideas appear constantly in this codebase. Skim them
now and refer back when needed.

- **`async` / `await`** — Normal code runs one line at a time and *waits* for slow things
  (network calls, disk reads). An `async def` function can be *paused* while it waits, so
  the program can do other useful work in the meantime. You "wait for" such a function
  with `await`. LightRAG uses this everywhere because it talks to LLMs and databases over
  the network, which is slow. You'll see things like `await rag.ainsert(...)`.

- **A "coroutine" and `asyncio.run(...)`** — Calling an `async` function does *not* run it;
  it returns a "to-do" object called a coroutine. Something must actually run it. At the
  top level you use `asyncio.run(main())`; inside other async code you use `await`.

- **Decorators (`@something`)** — A decorator is a line starting with `@` placed just above
  a function. It *wraps* the function to add behaviour. LightRAG uses
  `@wrap_embedding_func_with_attrs(...)` to attach extra info (like the embedding size) to
  an embedding function.

- **Dataclasses (`@dataclass`)** — A class is a blueprint for an object that bundles data
  and functions together. `@dataclass` is a shortcut that auto-writes the boring setup
  code, so the class definition just lists its fields. Most LightRAG classes are
  dataclasses.

- **Abstract base class (ABC) / abstract method** — An "abstract" class is a *contract*: it
  says "any real implementation must provide these methods" but doesn't implement them
  itself. LightRAG defines `BaseVectorStorage` as a contract; Faiss, Qdrant, etc. each
  *fulfil* that contract. This is exactly what lets you swap one database for another.

- **Environment variables** — Settings read from outside the program (from your shell or a
  `.env` file), e.g. `LIGHTRAG_VECTOR_STORAGE=FaissVectorDBStorage`. Code reads them with
  `os.getenv("NAME")`. They let you change behaviour without editing code.

---

## 0. Orientation — the big picture

**What LightRAG is.** Plain "RAG" (Retrieval-Augmented Generation) means: store your
documents, and when a user asks a question, *retrieve* the most relevant pieces and feed
them to an LLM so it can answer using your data. LightRAG adds a twist: while indexing, it
also asks an LLM to pull out **entities** (people, places, concepts) and the
**relationships** between them, building a **knowledge graph**. At query time it can search
both the raw text *and* this graph.

**The four kinds of storage.** Everything LightRAG saves falls into one of four buckets.
Each bucket has a pluggable "backend" (you choose the technology):

| Storage type        | What it holds                                         | Default backend        |
|---------------------|-------------------------------------------------------|------------------------|
| **KV_STORAGE**      | Raw document text, text chunks, LLM response cache    | `JsonKVStorage`        |
| **VECTOR_STORAGE**  | Embeddings (numeric vectors) for chunks/entities/rels | `NanoVectorDBStorage`  |
| **GRAPH_STORAGE**   | The entity↔relationship knowledge graph               | `NetworkXStorage`      |
| **DOC_STATUS_STORAGE** | Per-document processing status (pending/done/…)    | `JsonDocStatusStorage` |

These defaults are declared in [lightrag/lightrag.py:222-231](lightrag/lightrag.py#L222-L231).
This guide focuses on the **VECTOR_STORAGE** bucket (questions 2 and 3).

**Mental model in one sentence:** you push documents in → LightRAG chops them into chunks,
turns chunks into vectors (for similarity search) and into a graph of entities/relations
(via an LLM) → all of that is saved into the four stores → at query time it searches the
vectors and the graph to build context for the LLM.

**Repository map (the folders you care about):**

- [lightrag/](lightrag/) — the core Python library.
  - [lightrag/lightrag.py](lightrag/lightrag.py) — the main `LightRAG` class (the orchestrator).
  - [lightrag/operate.py](lightrag/operate.py) — the heavy lifting: chunking, entity extraction, graph merging, querying.
  - [lightrag/base.py](lightrag/base.py) — the *contracts* (abstract base classes) for each storage type.
  - [lightrag/kg/](lightrag/kg/) — concrete storage backends (one file per technology: Faiss, Qdrant, Postgres, …).
  - [lightrag/llm/](lightrag/llm/) — connectors to LLM providers (OpenAI, Ollama, Gemini, …).
  - [lightrag/api/](lightrag/api/) — the FastAPI web server and its HTTP endpoints.
- [lightrag_webui/](lightrag_webui/) — the React + TypeScript web front-end.
- [examples/](examples/) — runnable scripts showing how to use the library directly.

---

## 1. The full journey of a document

There are **two front doors** for getting a document into LightRAG, but they both lead to
the **same internal pipeline**. Here is the shape of the whole thing:

```
  ┌─────────────────────────┐        ┌──────────────────────────┐
  │  WEB UI                  │        │  SCRIPT / CLI            │
  │  drag-drop a file        │        │  await rag.ainsert(text) │
  │  → POST /documents/upload│        │                          │
  └───────────┬─────────────┘        └────────────┬─────────────┘
              │  (HTTP, runs in background)        │  (direct function call)
              ▼                                    ▼
      ┌───────────────────────────────────────────────────────┐
      │            THE SHARED INDEXING PIPELINE                │
      │  1. Enqueue   → create PENDING status, save raw text   │
      │  2. Chunk     → split text into overlapping pieces     │
      │  3. Embed     → chunks become vectors (→ VECTOR store) │
      │  4. Extract   → LLM finds entities + relationships     │
      │  5. Merge     → build/update the knowledge GRAPH       │
      │  6. Finalize  → persist everything; mark PROCESSED     │
      └───────────────────────────────────────────────────────┘
```

Let's walk each part.

### 1a. Front door #1 — the Web UI

When you drag a file into the web interface, a React component handles it:
[lightrag_webui/src/components/documents/UploadDocumentsDialog.tsx](lightrag_webui/src/components/documents/UploadDocumentsDialog.tsx),
in its `handleDocumentsUpload()` function. For each file it calls a small helper that makes
the actual HTTP request:

```ts
// lightrag_webui/src/api/lightrag.ts  (around line 792)
export const uploadDocument = async (file: File, onUploadProgress?) => {
  const formData = new FormData()
  formData.append('file', file)
  const response = await axiosInstance.post('/documents/upload', formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
    ...
  })
  return response.data   // { status: 'success' | 'duplicated' | 'failure', track_id }
}
```

See [lightrag_webui/src/api/lightrag.ts:792](lightrag_webui/src/api/lightrag.ts#L792). The
important part: the browser sends the file to the server's `POST /documents/upload`
endpoint, and gets back a **`track_id`** it can use to poll for progress.

### 1b. Front door #1 (server side) — receiving the upload

The server-side handler is `upload_to_input_dir()` at
[lightrag/api/routers/document_routes.py:2129](lightrag/api/routers/document_routes.py#L2129).
In plain terms it:

1. **Sanitizes the filename** (so a malicious name like `../../etc/passwd` can't escape the
   upload folder).
2. **Checks the file type and size** against allowed limits.
3. **Saves the file** into the server's input directory.
4. **Creates a `track_id`** so the UI can follow progress.
5. **Schedules processing in the background** so the HTTP request can return immediately:

   ```python
   background_tasks.add_task(pipeline_index_file, rag, file_path, track_id)
   ```

That `background_tasks.add_task(...)` is the hand-off: the web request finishes right away,
and the heavy work happens afterwards in `pipeline_index_file()`
([document_routes.py:1691](lightrag/api/routers/document_routes.py#L1691)).

Before indexing, the file's **text must be extracted**. That happens in
`pipeline_enqueue_file()` ([document_routes.py:1233](lightrag/api/routers/document_routes.py#L1233)),
which knows how to read many formats: plain text/markdown/JSON directly, PDFs via `pypdf`
or `docling`, Word/PowerPoint/Excel via `python-docx` / `python-pptx` / `openpyxl`. Once it
has plain text, it calls the same enqueue function the script path uses (below).

> There are also `POST /documents/text` and `/documents/texts` endpoints for inserting raw
> text without a file — they skip extraction and go straight to enqueueing.

### 1c. Front door #2 — a script / CLI

If you use LightRAG as a library, you skip the web server entirely. A typical script (see
[examples/lightrag_openai_demo.py](examples/lightrag_openai_demo.py)) looks like:

```python
rag = LightRAG(working_dir="./rag_storage",
               llm_model_func=gpt_4o_mini_complete,
               embedding_func=openai_embed)
await rag.initialize_storages()      # REQUIRED — see note below
await rag.ainsert("Your document text here")
```

The entry point is `ainsert()` at [lightrag/lightrag.py:1237](lightrag/lightrag.py#L1237).
It does two things in sequence: enqueue the document, then run the processing pipeline and
*wait* for it to finish (unlike the web path, which runs in the background).

> **The #1 beginner mistake:** forgetting `await rag.initialize_storages()`
> ([lightrag.py:757](lightrag/lightrag.py#L757)). The `LightRAG(...)` line only builds the
> object; `initialize_storages()` actually opens the databases. Skip it and you get errors
> like `AttributeError: __aenter__`.

### 1d. The shared pipeline, stage by stage

Both front doors converge here. The two key methods live in
[lightrag/lightrag.py](lightrag/lightrag.py).

**Stage 1 — Enqueue:** `apipeline_enqueue_documents()`
([lightrag.py:1344](lightrag/lightrag.py#L1344))
- Computes a content-based ID for each document (an MD5 hash) so identical content isn't
  indexed twice.
- Removes duplicates.
- Saves the **raw text** into the KV store (`full_docs`).
- Creates a status record marked **`PENDING`** in the doc-status store.

**Stage 2 — The processing loop:** `apipeline_process_enqueue_documents()`
([lightrag.py:1740](lightrag/lightrag.py#L1740))
- Finds all documents that still need work (`PENDING` or `FAILED`).
- Uses a *semaphore* — `asyncio.Semaphore(self.max_parallel_insert)` — which is just a
  limit on how many documents are processed at the same time (so you don't overload the
  LLM). For each document it runs the inner `process_document()` helper.

**Stage 3 — Chunking:** `chunking_by_token_size()`
([lightrag/operate.py:101](lightrag/operate.py#L101))
- Splits the document into pieces ("chunks") measured in **tokens** (sub-word units the LLM
  counts in), not characters. Defaults: ~1200 tokens per chunk with ~100 tokens of overlap
  between neighbours (overlap helps keep sentences that straddle a boundary intact).
- Each chunk records its text, token count, and position (`chunk_order_index`).

**Stage 4 — Embed the chunks (first vector-DB write):**
The chunks are *upserted* into the chunk vector store (`chunks_vdb`) and the text KV store
(`text_chunks`). "Upsert" = insert-or-update. **This is where embeddings happen:** the
vector backend calls your `embedding_func` on each chunk's text and stores the resulting
vector. (See the Faiss/Nano `upsert()` methods in section 2 for exactly how.)

**Stage 5 — Extract entities & relationships:** `extract_entities()`
([lightrag/operate.py:2883](lightrag/operate.py#L2883))
- For each chunk, builds a prompt and asks the **LLM** to list the entities and the
  relationships between them.
- Optionally does extra "gleaning" passes to catch things missed the first time.
- Caches LLM responses (so re-running is cheaper).
- Returns, per chunk, a pair `(maybe_nodes, maybe_edges)` — candidate graph nodes and edges.

**Stage 6 — Merge into the knowledge graph (more vector-DB writes):**
`merge_nodes_and_edges()` ([lightrag/operate.py:2501](lightrag/operate.py#L2501)) runs in
two phases:
- **Phase 1 — entities:** `_merge_nodes_then_upsert()`
  ([operate.py:1623](lightrag/operate.py#L1623)) combines descriptions of the same entity
  found across different chunks, writes the node into the **graph** store, *and* writes the
  entity's embedding into the entity vector store (`entity_vdb`).
- **Phase 2 — relationships:** `_merge_edges_then_upsert()`
  ([operate.py:1948](lightrag/operate.py#L1948)) does the same for edges, writing into the
  graph store and the relationship vector store (`relationships_vdb`).
- Per-entity/per-edge locks keep concurrent updates from corrupting each other.

**Stage 7 — Finalize:** When all documents in a batch are done, an internal `_insert_done()`
step calls `index_done_callback()` on **every** storage backend, which flushes in-memory
data to disk. The document's status is updated to **`PROCESSED`**.

### 1e. Where embeddings touch the vector database (summary)

Three different vector stores get populated during indexing:

| Vector store        | Filled in stage | Contains embeddings of…           |
|---------------------|-----------------|-----------------------------------|
| `chunks_vdb`        | Stage 4         | each text chunk                   |
| `entity_vdb`        | Stage 6 phase 1 | each entity's name + description   |
| `relationships_vdb` | Stage 6 phase 2 | each relationship's description    |

In every case the *embedding vector is computed inside the backend's `upsert()` method* by
calling the configured `embedding_func`. The backend never sees text without turning it into
a vector first. This matters for benchmarking (section 3): if you want to measure pure
*database* speed, you should hold the embedding function constant (or fake/cheap) so you're
not accidentally timing the LLM/embedding API.

### 1f. The document status lifecycle

Every document has a status, defined by the `DocStatus` enum at
[lightrag/base.py:752](lightrag/base.py#L752):

```python
class DocStatus(str, Enum):
    PENDING = "pending"           # enqueued, waiting
    PROCESSING = "processing"     # being chunked / extracted right now
    PREPROCESSED = "preprocessed" # partial (multimodal not finished)
    PROCESSED = "processed"       # fully indexed and queryable
    FAILED = "failed"             # something went wrong (error saved)
```

The full status record (`DocProcessingStatus`,
[base.py:763](lightrag/base.py#L763)) also stores a content summary, length, file path,
timestamps, chunk count, the `track_id`, and any error message. You can query it via
endpoints like `GET /documents` and `GET /documents/track_status/{track_id}` — that's how
the web UI shows you progress.

```
PENDING ──► PROCESSING ──► PROCESSED
                  │
                  └────────► FAILED   (error_msg explains why)
```

---

## 2. The default vector database & how to change it

### 2a. What's the default?

The default vector backend is **`NanoVectorDBStorage`**, declared at
[lightrag/lightrag.py:225](lightrag/lightrag.py#L225):

```python
vector_storage: str = field(default="NanoVectorDBStorage")
```

`NanoVectorDBStorage` (in [lightrag/kg/nano_vector_db_impl.py](lightrag/kg/nano_vector_db_impl.py))
is a tiny, file-based vector database that needs **no external server** — perfect for
getting started. It saves vectors to a JSON file in your `working_dir`.

### 2b. All the available vector backends

These are registered in [lightrag/kg/__init__.py](lightrag/kg/__init__.py) under
`STORAGE_IMPLEMENTATIONS["VECTOR_STORAGE"]`:

| Class name                   | File                                                      | Needs an external service? |
|------------------------------|-----------------------------------------------------------|----------------------------|
| `NanoVectorDBStorage` (default) | [kg/nano_vector_db_impl.py](lightrag/kg/nano_vector_db_impl.py) | No (local file)        |
| `FaissVectorDBStorage`       | [kg/faiss_impl.py](lightrag/kg/faiss_impl.py)             | No (local file; needs `faiss-cpu` package) |
| `MilvusVectorDBStorage`      | [kg/milvus_impl.py](lightrag/kg/milvus_impl.py)           | Yes (Milvus server)        |
| `QdrantVectorDBStorage`      | [kg/qdrant_impl.py](lightrag/kg/qdrant_impl.py)           | Yes (Qdrant server)        |
| `PGVectorStorage`            | [kg/postgres_impl.py](lightrag/kg/postgres_impl.py)       | Yes (PostgreSQL + pgvector)|
| `MongoVectorDBStorage`       | [kg/mongo_impl.py](lightrag/kg/mongo_impl.py)             | Yes (MongoDB)              |
| `OpenSearchVectorDBStorage`  | [kg/opensearch_impl.py](lightrag/kg/opensearch_impl.py)   | Yes (OpenSearch)           |

(Chroma exists but is deprecated.) Note that **Nano and Faiss are the only two that run
with zero external setup** — which is why section 3's benchmark uses them by default.

### 2c. How a name like `"FaissVectorDBStorage"` becomes a running object

You give LightRAG a *string* (the class name). It turns that string into the actual class
in `_get_storage_class()` at [lightrag/lightrag.py:1177](lightrag/lightrag.py#L1177):

- For the four built-in defaults, it imports them directly (fast path).
- For everything else, it looks the name up in the `STORAGES` dictionary
  (in [kg/__init__.py](lightrag/kg/__init__.py)), which maps each name to its module path
  (e.g. `"FaissVectorDBStorage" → ".kg.faiss_impl"`), then imports it lazily with
  `lazy_external_import(...)`. "Lazy" means the heavy library (e.g. `faiss`) is only
  imported if you actually use that backend — so you don't need every database installed.

LightRAG also validates your choice with `verify_storage_implementation()` and checks that
any required environment variables are present, using `STORAGE_ENV_REQUIREMENTS` (also in
[kg/__init__.py](lightrag/kg/__init__.py)). For Nano and Faiss this list is empty (no env
vars required); for Qdrant it requires `QDRANT_URL`, for Postgres `POSTGRES_USER` /
`POSTGRES_PASSWORD` / `POSTGRES_DATABASE`, and so on.

### 2d. The contract every vector backend must satisfy

This is *why* swapping backends works. The abstract class `BaseVectorStorage` at
[lightrag/base.py:218](lightrag/base.py#L218) defines the shared interface. The two methods
the whole system relies on are:

```python
# lightrag/base.py
@abstractmethod
async def query(self, query: str, top_k: int,
                query_embedding: list[float] = None) -> list[dict[str, Any]]:
    """Return the top_k most similar items."""

@abstractmethod
async def upsert(self, data: dict[str, dict[str, Any]]) -> None:
    """Insert or update vectors (embeddings are computed here)."""
```

Plus housekeeping methods: `get_by_id`, `get_by_ids`, `delete`, `delete_entity`,
`delete_entity_relation`, `get_vectors_by_ids`, `drop`, and `index_done_callback` (flush to
disk). Every backend file in `kg/` implements all of these. Because the *core* code only
ever calls these contract methods — never anything Faiss- or Qdrant-specific — you can drop
in any backend and the rest of LightRAG doesn't notice.

Each backend is constructed with a common set of fields (inherited from `StorageNameSpace`
at [base.py:172](lightrag/base.py#L172) and `BaseVectorStorage`): `namespace`, `workspace`,
`global_config` (a dict with `working_dir`, `embedding_batch_num`, and
`vector_db_storage_cls_kwargs`), `embedding_func`, and `meta_fields`. The benchmark in
section 3 builds these by hand — it's the clearest example of how the backend is wired up.

### 2e. How to switch to Faiss (two ways)

**Way 1 — in code (constructor parameter):**

```python
rag = LightRAG(
    working_dir="./rag_storage",
    vector_storage="FaissVectorDBStorage",     # <- the only change
    llm_model_func=...,
    embedding_func=...,
    vector_db_storage_cls_kwargs={"cosine_better_than_threshold": 0.2},
)
await rag.initialize_storages()
```

**Way 2 — via environment variable (for the API server):**

```bash
# in your .env file or shell
LIGHTRAG_VECTOR_STORAGE=FaissVectorDBStorage
```

The server reads this in [lightrag/api/config.py:427](lightrag/api/config.py#L427)
(default defined at [config.py:62](lightrag/api/config.py#L62)), and `env.example` documents
it at [env.example:452](env.example#L452) with commented examples for other backends below.

**Before you switch, three things to know about Faiss specifically**
(from [lightrag/kg/faiss_impl.py](lightrag/kg/faiss_impl.py)):

1. **Install the package:** `pip install faiss-cpu` (or `faiss-gpu`). Faiss is not bundled.
2. **It uses cosine similarity** via a `faiss.IndexFlatIP` index over L2-normalized vectors
   ([faiss_impl.py:65](lightrag/kg/faiss_impl.py#L65)). `IndexFlatIP` is *exact* (brute
   force) — it checks every vector, so recall is perfect but it gets slower as data grows.
3. **It requires `cosine_better_than_threshold`** to be set in
   `vector_db_storage_cls_kwargs` ([faiss_impl.py:34](lightrag/kg/faiss_impl.py#L34)) — this
   is the minimum similarity for a result to count. LightRAG passes this for you when
   configured normally.

> **Important caveat (applies to *any* backend switch):** embeddings are model-specific and
> have a fixed size (dimension). If you also change the embedding model, or switch backends
> in a way that changes the stored dimension, you must **clear your data directory** and
> re-index. Mixing vectors of different sizes/models produces garbage results. Faiss will
> even refuse to load an index whose dimension doesn't match
> ([faiss_impl.py:386](lightrag/kg/faiss_impl.py#L386)).

---

## 3. Benchmarking different vector databases

### 3a. What already exists in the repo (and why it isn't what you want)

It's worth being clear: **the repository does not ship a vector-database benchmark.** What
it *does* have is answer-quality and sanity tooling:

- [lightrag/evaluation/eval_rag_quality.py](lightrag/evaluation/eval_rag_quality.py) — a
  **RAGAS**-based evaluator. It measures *answer quality* (faithfulness, answer relevance,
  context precision/recall) by calling an LLM as a judge. This is the thing you said you're
  **not** interested in.
- [lightrag/evaluation/offline_retrieval_check.py](lightrag/evaluation/offline_retrieval_check.py)
  — an offline lexical (keyword/TF-IDF) sanity check computing recall@k and MRR. It doesn't
  touch the vector DB backends at all.
- [reproduce/batch_eval.py](reproduce/batch_eval.py) — compares two whole RAG systems on
  subjective qualities. Again, answer quality, not storage performance.
- `performance_timing_log()` in [lightrag/utils.py](lightrag/utils.py) — an opt-in timing
  logger (enabled by `LIGHTRAG_PERFORMANCE_TIMING_LOGS`) sprinkled inside some backends. It
  emits log lines but doesn't collect or compare metrics.

None of these measure *how fast a vector database inserts and searches vectors, or how
accurate its search is*. So this guide ships a new script that does.

### 3b. What you actually want to measure

For comparing vector *databases* (holding everything else equal), the meaningful metrics
are:

- **Insert throughput** — how many vectors per second the backend can ingest (and total
  time to load N vectors).
- **Query latency** — how long a single similarity search takes. Report the **median (p50)**
  and the **95th percentile (p95)**, because tail latency matters more than the average.
- **Recall@k** — accuracy. Some indexes are *approximate* (fast but may miss true
  neighbours). We measure recall by comparing each backend's top-k results against a
  **brute-force ground truth** computed directly in NumPy. Exact indexes (Nano, Faiss
  `IndexFlatIP`) should score ~1.0; approximate ones (e.g. Milvus/Qdrant with ANN indexes)
  may score lower in exchange for speed.
- **On-disk size** — how much space the index uses after saving (for file-based backends).

### 3c. The benchmark script: `benchmark_vectordb.py`

This guide adds a runnable script at the repo root:
[benchmark_vectordb.py](benchmark_vectordb.py). The key design idea is that it exercises
each backend **through the exact same `BaseVectorStorage` interface that LightRAG itself
uses** (`upsert` and `query`) — so the numbers reflect real LightRAG behaviour, not a
synthetic side-channel.

How it works, in plain terms:

1. **Fake, deterministic embeddings.** Instead of calling a real (slow, paid) embedding
   API, it uses a tiny function that turns text into a reproducible random vector
   (seeded by a hash of the text). This is wrapped with
   `@wrap_embedding_func_with_attrs(embedding_dim=...)` so it looks like a real LightRAG
   embedding function. **Why:** we want to time the *database*, not the embedding model. (A
   `--real-embeddings` style swap is documented in the script if you want true vectors.)
2. **Builds each backend by hand**, supplying the same fields LightRAG would: `namespace`,
   `workspace`, a `global_config` dict (with `working_dir`, `embedding_batch_num`, and
   `vector_db_storage_cls_kwargs={"cosine_better_than_threshold": ...}`), the fake
   `embedding_func`, and `meta_fields`. It calls `initialize_share_data()` once (the
   backends need the shared lock machinery) and `await storage.initialize()` per backend.
3. **Insert phase:** upserts `--num-vectors` synthetic items in batches and times it.
4. **Query phase:** runs `--num-queries` searches at `--top-k` and records each latency.
5. **Recall phase:** computes the true nearest neighbours with NumPy and compares.
6. **Disk phase:** calls `index_done_callback()` to flush, then measures the files on disk.
7. Prints a comparison table and writes `benchmark_results.json`. It cleans up by calling
   `drop()` / `finalize()` and deleting its temp working directories.

### 3d. Running it

```bash
# from the repo root
source .venv/bin/activate          # activate the project's virtual environment
pip install faiss-cpu              # only needed to benchmark the Faiss backend

python benchmark_vectordb.py \
    --backends NanoVectorDBStorage,FaissVectorDBStorage \
    --num-vectors 2000 \
    --num-queries 200 \
    --top-k 10
```

You'll get a table comparing insert time, query p50/p95, recall@10, and index size for each
backend, plus a `benchmark_results.json` file you can chart or diff later. No API keys are
required because the embeddings are faked.

**Adding server-backed databases.** To include Qdrant, Postgres, etc., install their client
package, start the service, set the required environment variables (see
`STORAGE_ENV_REQUIREMENTS` in [kg/__init__.py](lightrag/kg/__init__.py) — e.g. `QDRANT_URL`),
and add the class name to `--backends`. The script will instantiate it the same way; only
the configuration differs.

### 3e. How to read the results

- If recall@k for an exact backend (Nano, Faiss) is **not** ~1.0, something is wrong with
  the setup (e.g. dimension mismatch) — investigate before trusting timings.
- Compare **query p95**, not just the average — a backend with a great average but a terrible
  p95 will feel laggy in production.
- Insert throughput matters most when you re-index large corpora frequently; query latency
  matters most for serving users. Weigh them according to your workload.
- These synthetic numbers tell you about the *database engine*. Real-world performance also
  depends on your embedding model, network, and hardware — so treat this as an
  apples-to-apples engine comparison, not an absolute production prediction.

---

## Quick reference — key functions

| What | Where |
|------|-------|
| Storage defaults | [lightrag.py:222-231](lightrag/lightrag.py#L222-L231) |
| Open the databases (must call!) | `initialize_storages()` [lightrag.py:757](lightrag/lightrag.py#L757) |
| String → backend class | `_get_storage_class()` [lightrag.py:1177](lightrag/lightrag.py#L1177) |
| Insert (library entry point) | `ainsert()` [lightrag.py:1237](lightrag/lightrag.py#L1237) |
| Enqueue documents | `apipeline_enqueue_documents()` [lightrag.py:1344](lightrag/lightrag.py#L1344) |
| Processing loop | `apipeline_process_enqueue_documents()` [lightrag.py:1740](lightrag/lightrag.py#L1740) |
| Chunking | `chunking_by_token_size()` [operate.py:101](lightrag/operate.py#L101) |
| Entity/relation extraction | `extract_entities()` [operate.py:2883](lightrag/operate.py#L2883) |
| Merge into graph | `merge_nodes_and_edges()` [operate.py:2501](lightrag/operate.py#L2501) |
| Web upload (browser) | `uploadDocument()` [lightrag.ts:792](lightrag_webui/src/api/lightrag.ts#L792) |
| Web upload (server) | `upload_to_input_dir()` [document_routes.py:2129](lightrag/api/routers/document_routes.py#L2129) |
| File text extraction | `pipeline_enqueue_file()` [document_routes.py:1233](lightrag/api/routers/document_routes.py#L1233) |
| Vector storage contract | `BaseVectorStorage` [base.py:218](lightrag/base.py#L218) |
| Faiss backend | [kg/faiss_impl.py](lightrag/kg/faiss_impl.py) |
| Backend registry & env requirements | [kg/__init__.py](lightrag/kg/__init__.py) |
| Document status enum | `DocStatus` [base.py:752](lightrag/base.py#L752) |
| Benchmark script (new) | [benchmark_vectordb.py](benchmark_vectordb.py) |
