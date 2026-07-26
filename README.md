# TSL-RAG — EU Transport & Logistics Legal Assistant

A Retrieval-Augmented Generation (RAG) system designed to navigate, query, and cite European and Polish road transport laws.

## The Problem
Transport law spans multiple overlapping regulations (e.g., EC 561/2006, AETR, Dir. 2002/15, and national penalty tariffs). A compliance officer asking *"Can a driver extend their daily rest if they're on a ferry?"* requires an answer that is legally accurate, cites the correct article, and respects document hierarchy.

Generic RAG systems relying solely on flat vector search often fail to retrieve exact legal boundaries. This project implements a custom hybrid search and reranking pipeline to address these domain-specific challenges.

## Architecture

```text
     PDF Documents
           │
           ▼
┌─────────────────────────────────────────────┐
│              Ingestion Pipeline             │
│  LegalPDFParser → LegalChunker → Embedder   │
│  • pdfplumber (tables) + pymupdf (text)     │
│  • Article-aware chunking (400 tok max)     │
│  • Tables → atomic chunks (never split)     │
│  • nomic-embed-text → 768d vectors          │
└──────────────────────┬──────────────────────┘
                       │
                       ▼
              PostgreSQL + pgvector
              (HNSW index, m=16)
                       │
          ┌────────────┼────────────┐
          ▼            ▼            ▼
     Dense Search   BM25 Search  Metadata
     (cosine sim)  (rank-bm25)   Filters
          │            │
          └─────┬──────┘
                ▼
         RRF Fusion
    (Reciprocal Rank Fusion)
                │
                ▼
     Cross-Encoder Reranker
  (ms-marco-MiniLM-L-6-v2, CPU)
                │
                ▼
        RAG Generator
   (Ollama llama3.1 8B / mistral 7B)
   Polish system prompt + citation enforcement
                │
                ▼
     FastAPI  ←→  Streamlit UI

## Tech Stack

| Layer | Technology | Rationale |
|-------|------------|-----------|
| **Embedding** | `nomic-embed-text` (768d) | Free, local, strong multilingual support. |
| **Vector DB** | PostgreSQL + `pgvector` | Supports SQL joins, metadata filtering, and HNSW indexing. |
| **Keyword search** | `rank-bm25` (in-memory) | Low latency for smaller, highly specific legal corpora. |
| **Reranker** | `ms-marco-MiniLM-L-6-v2` | Lightweight cross-encoder (~90MB), runs efficiently on CPU. |
| **LLM** | `llama3.1:8b-instruct` | Runs locally via Ollama (~5GB VRAM required). |
| **API** | FastAPI + Uvicorn | Async handling, automatic OpenAPI documentation. |
| **UI** | Streamlit | Rapid prototyping with built-in retrieval debug panels. |
| **Eval** | Custom harness + Gemini 2.0 Flash | LLM-as-a-Judge for semantic answer scoring. |

**Zero paid APIs required** to run the full system. Gemini key is optional (eval only).

## Legal Corpus

| Document | Type | Description |
|----------|------|-------------|
| **EC_561_2006** | EU Regulation | Drivers' hours — the primary source. |
| **EU_2020_1054** | EU Regulation | Mobility Package amendments to EC 561/2006. |
| **DIRECTIVE_2002_15** | EU Directive | Working time for mobile road transport workers. |
| **DIRECTIVE_2020_1057** | EU Directive | Posting of drivers in the road transport sector. |
| **EU_165_2014** | EU Regulation | Rules regarding tachographs in road transport. |
| **EU_1071_2009** | EU Regulation | Conditions to pursue the occupation of road transport operator. |
| **EU_1072_2009** | EU Regulation | Common rules for access to the international road haulage market (cabotage). |
| **AETR** | Int. Agreement | European Agreement concerning Work of Crews of Vehicles in International Road Transport. |
| **PL_DRIVER_HOURS_ACT** | National Law | Polish Act on the working time of drivers. |
| **TARIFF_DRIVER_2022** | Penalty Tariff | Fines for driver violations (PL). |
| **TARIFF_COMPANY_2022** | Penalty Tariff | Fines for transport company violations (PL). |
| **TARIFF_MANAGER_2022** | Penalty Tariff | Fines for transport manager violations (PL). |

## Evaluation

A RAG system without an eval harness is an incomplete project. This system ships with a **golden dataset of 15 questions** across 6 categories, evaluated by two methods:

**Method 1 — Keyword match (fast, offline):**
```
Questions      : 15
Answer score   : 0.633   (key facts present)
Citation hit   : 0.667   (correct doc cited)
Refusal prec.  : 1.000   (out-of-scope correctly refused)
Avg latency    : ~8.3 s

Per category:
  numeric_fact    n=9   fact=0.83  cite=0.77  ✅
  out_of_scope    n=2   fact=1.00  cite=1.00  ✅
  penalty         n=1   fact=0.00  cite=0.00  ← complex table understanding
```

**Method 2 — LLM-as-a-Judge (Gemini 2.0 Flash):**

Keyword match penalises semantically correct answers phrased differently (*"dziewięć godzin"* ≠ *"9 godzin"*). Gemini evaluates meaning, not string overlap. The judge outputs a score (0.0–1.0) **and a reasoning string** stored in JSON — failures are auditable, not a black box.

```bash
# Keyword match (no API key needed)
uv run python -m evals.run_evals --output evals/results/run_001.json

# Semantic eval (requires GEMINI_API_KEY in .env)
uv run python -m evals.run_evals --use-judge --output evals/results/run_001_judge.json
```

**Known limitations (documented, not hidden):**
- Penalty tariff PDFs are scanned images — OCR required for full coverage. Parser correctly returns 0 table chunks; system falls back gracefully. (It've been done on used files)
- Latency ~10s on RTX 4060 with llama3.1 8B Q4. Acceptable for compliance tooling.

## Quick Start

**Prerequisites:** Docker, Ollama, Python 3.11+, `uv`.

**1. Pull local models:**
```bash
ollama pull llama3.1:8b-instruct-q4_K_M
ollama pull nomic-embed-text
```

**2. Clone & install:**
```bash
git clone https://github.com/yourname/tsl-rag
cd tsl-rag
uv sync
```

**3. Configure environment:**
```bash
cp env.example .env
# Defaults work out of the box
# Add GEMINI_API_KEY for LLM-as-a-Judge eval (optional)
```

**4. Start the database:**
```bash
docker compose up -d
```

**5. Ingest documents:**
```bash
# Place PDF files in data/raw/ first
uv run python -m tsl_rag.ingestion.cli ingest-all data/raw/

# Verify:
docker exec tsl_rag_postgres psql -U postgres -d tsl_rag \
  -c "SELECT * FROM corpus_stats;"
```

**6. Run the application:**
```bash
# Terminal 1 — Backend API
uv run python main.py
# → http://localhost:8000/docs

# Terminal 2 — Frontend UI
uv run streamlit run ui.py
# → http://localhost:8501

or

# Terminal 1 — Backend API
make api
# → http://localhost:8000/docs

# Terminal 2 — Frontend UI
make ui
# → http://localhost:8501
```



## Project Structure

```
tsl-rag/
├── src/tsl_rag/
│   ├── core/
│   │   ├── settings.py          # Pydantic Settings, Ollama/OpenAI provider switch
│   │   ├── models.py            # Chunk, DocumentMetadata, QueryResponse
│   │   └── llm_client.py        # OpenAI-compatible client for Ollama + OpenAI
│   ├── ingestion/
│   │   ├── parsers/legal_pdf_parser.py   # pdfplumber + pymupdf, hierarchy detection
│   │   ├── chunkers/legal_chunker.py     # Article-aware chunking, table isolation
│   │   ├── embedders/embedder.py         # Batch embed → pgvector upsert
│   │   └── cli.py                        # Typer CLI: ingest / ingest-all
│   ├── retrieval/
│   │   ├── retriever.py         # Hybrid BM25 + dense + RRF fusion
│   │   └── reranker.py          # CrossEncoder wrapper (lazy-loaded)
│   ├── generation/
│   │   └── generator.py         # System prompt, citation extraction
│   ├── api/
│   │   ├── main.py              # FastAPI factory and entrypoint
│   │   └── routers/query.py     # POST /query, GET /query/health
│   └── ui/
│       └── app.py               # Streamlit chat UI with retrieval debug panel
├── evals/
│   ├── golden_dataset/questions.py   # 15 questions × 6 categories
│   ├── judge.py                      # GeminiJudge — LLM-as-a-Judge
│   └── run_evals.py                  # Eval harness, --use-judge flag
├── tests/unit/                  # 11 unit tests, no external dependencies
├── docker/init.sql              # pgvector schema, HNSW index, corpus_stats view
├── docker-compose.yml
├── Makefile                     # Shortcut commands runner
└── pyproject.toml
```

## Design Decisions

**Custom pipeline vs. orchestration frameworks:** LangChain and LlamaIndex abstract away the retrieval mechanics. This project implements hybrid search, RRF fusion, and cross-encoder reranking explicitly — every component is independently testable and tunable.

**Article-boundary chunking:** Legal text loses meaning when split arbitrarily. The `LegalChunker` treats article boundaries as hard limits and never splits tabular data (penalty tariffs), ensuring the LLM receives complete, coherent legal provisions.

**PostgreSQL + pgvector:** Removes the need for external SaaS vector databases while enabling standard SQL joins and metadata pre-filtering (e.g., by `document_type` or `contains_penalty`).

**Gemini as eval judge, not the local model:** LLM-as-a-Judge requires a model stronger than the system under test. Using Ollama to judge Ollama outputs introduces self-evaluation bias. Gemini 2.0 Flash is independent, free-tier, and evaluates meaning — not string overlap.

## Running Tests

```bash
uv run pytest tests/unit/ -v
# 11 passed — no external dependencies required
```

## License

MIT
=======
