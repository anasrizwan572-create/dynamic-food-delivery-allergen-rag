# Dynamic Food Delivery Menu & Allergen RAG

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-green.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-red.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An academic and production-grade Retrieval-Augmented Generation (RAG) system built for food delivery platforms in Pakistan (Foodpanda PK context, local restaurant chains, and cloud kitchens). Features hybrid retrieval (BM25 + dense vectors), Reciprocal Rank Fusion (RRF), cross-encoder reranking, hard-constraint metadata filtering, an independent deterministic allergen safety verifier, and strictly grounded generation.

---

## 1. Project Overview
Navigating restaurant menus with dietary restrictions and severe food allergies (e.g., peanuts, dairy, gluten, eggs, shellfish) is high-stakes. Standard vector search can conflate dishes with and without allergens because their semantic descriptions are similar (e.g. "Chicken Karahi with cream" vs "Dairy-free Chicken Karahi"). 

**Dynamic Food Delivery Menu & Allergen RAG** solves this problem by decoupling semantic exploration from safety verification. Queries are parsed into hard constraints (allergens, price limits, Halal, location) and soft preferences (protein content, spice level, cuisine). Candidate dishes retrieved via hybrid search are passed through a deterministic **Allergen Safety Verifier** before being supplied as evidence to a grounded LLM generator.

---

## 2. Problem Statement
- **The Danger of Hallucination**: LLMs can hallucinate that a dish is "dairy-free" or "nut-free" simply because the brief description omitted mentioning butter or peanut oil.
- **Ambiguous & Missing Evidence**: Absence of allergen information must **never** be assumed to mean allergen-free.
- **Pakistani Food Nuances**: Ingredients like desi ghee, cream, nuts in gravies (korma, peshawari karahi), and Halal preparation require domain-specific extraction and normalization.

---

## 3. Architecture
```
                    CUSTOMER
                       │
                       ▼
                    WEB UI (Streamlit)
                       │
                       ▼
                 USER QUERY
                       │
                       ▼
              QUERY / CONSTRAINT PARSER
                       │
                       ▼
             METADATA PRE-FILTER (Price, Halal, Location, Excluded Allergens)
                       │
          ┌────────────┴────────────┐
          ▼                         ▼
    DENSE VECTOR SEARCH         BM25 SEARCH (Sparse Keyword)
          │                         │
          └────────────┬────────────┘
                       │
                       ▼
                 RRF FUSION (Reciprocal Rank Fusion, k=60)
                       │
                       ▼
                  RERANKER (Cross-Encoder)
                       │
                       ▼
             ALLERGEN SAFETY VERIFIER
             [REJECT | INSUFFICIENT_EVIDENCE | CONFLICTING_EVIDENCE | PASS]
                       │
                       ▼
              VERIFIED EVIDENCE CHUNKS
                       │
                       ▼
                     LLM (Strictly Grounded Prompting)
                       │
                       ▼
             STRUCTURED RESPONSE & CITATIONS
```

---

## 4. Key Features
- **Deterministic Constraint Parser**: Extracts hard constraints (max price, location, Halal, allergen exclusions) and soft preferences (protein, spice).
- **Hybrid Retrieval**: Combines BM25 lexical search with dense vector similarity via Reciprocal Rank Fusion (RRF).
- **Cross-Encoder Reranking**: Re-scores top fused candidates for deep query-document relevance.
- **Strict Allergen Safety Verifier**: Non-negotiable safety module. Explicit allergen presence causes instant rejection. Missing data flags `INSUFFICIENT_EVIDENCE`.
- **Grounded LLM Generation**: The LLM cannot invent prices, dishes, or allergen claims. Cites exact source IDs.
- **Standardized Failure States**:
  - `SUPPORTED`
  - `PARTIALLY_SUPPORTED`
  - `INSUFFICIENT_EVIDENCE`
  - `CONFLICTING_EVIDENCE`
  - `NO_MATCHING_ITEMS`
- **Full-Stack Implementation**: FastAPI REST backend + interactive Streamlit web dashboard.
- **Evaluation Suite**: 30+ gold-standard benchmark queries evaluating Recall@K, Precision@K, MRR, NDCG, Allergen Filter Recall, and Citation Accuracy.

---

## 5. Tech Stack
- **Backend API**: FastAPI, Uvicorn, Pydantic v2
- **Frontend Dashboard**: Streamlit
- **Search & Retrieval**: BM25 (`rank-bm25`), Dense Vector Search, Reciprocal Rank Fusion (RRF)
- **Embeddings & Reranking**: Sentence-Transformers / Configurable Local & OpenAI embeddings
- **Data Engineering**: Pandas, NumPy
- **Testing & Benchmarking**: Pytest, Scikit-learn

---

## 6. Dataset Schema
Menu items are structured with the following fields:
| Field | Type | Description | Example |
|---|---|---|---|
| `id` | string | Unique menu item ID | `dish_001` |
| `restaurant_name` | string | Restaurant or Cloud Kitchen | `Monal Express` |
| `dish_name` | string | Full name of the dish | `Grilled Chicken Breast` |
| `description` | string | Brief description | `Char-grilled chicken with sauteed greens` |
| `ingredients` | string | Listed raw ingredients | `chicken breast, olive oil, lemon, herbs` |
| `price_pkr` | float/int | Price in Pakistani Rupees | `1250` |
| `location` | string | City/Zone | `Gulberg` |
| `halal` | boolean | Halal certified status | `true` |
| `spice_level` | string | Spice intensity | `Mild`, `Medium`, `Spicy` |
| `allergens` | string/list | Normalized allergen tags | `["dairy"]` or `[]` |
| `category` | string | Category | `Chicken`, `BBQ`, `Desi`, etc. |
| `protein_g` | float/int | Approximate protein (g) | `42` |
| `availability` | boolean | Availability status | `true` |
| `cuisine` | string | Cuisine type | `Pakistani`, `Continental` |
| `source` | string | Source attribution | `Gulberg Dine-in Menu 2026` |
| `source_url` | string | Source reference link | `https://example.com/menu` |
| `timestamp` | string | ISO timestamp | `2026-09-28T00:00:00Z` |

---

## 7. Installation

### 1. Clone the repository
```bash
git clone https://github.com/example/dynamic-food-allergen-rag.git
cd dynamic-food-allergen-rag
```

### 2. Set up virtual environment
```bash
python -m venv venv
# Windows:
.\venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate
```

### 3. Install dependencies
```bash
python -m pip install -r requirements.txt
```

---

## 8. Environment Variables
Copy `.env.example` to `.env` and configure your settings:
```bash
cp .env.example .env
```
Key configuration parameters:
- `LLM_PROVIDER`: `mock` (default for offline/local evaluation), `openai`, or `gemini`
- `OPENAI_API_KEY`: API key for OpenAI (if using OpenAI LLM/Embeddings)
- `GEMINI_API_KEY`: API key for Gemini (if using Gemini LLM)
- `EMBEDDING_PROVIDER`: `local` (default) or `openai`
- `RERANKER_TYPE`: `cross_encoder` or `dummy`
- `DATA_CSV_PATH`: Path to menu dataset (default `data/menu.csv`)

---

## 9. How to Run Ingestion & Build Vector Index

### Clean, Normalize, and Index Menu into ChromaDB
```bash
# Ingest raw data, clean, normalize, and build vector embeddings
python -m ingestion.build_index

# Force full reset and rebuild of the vector index
python -m ingestion.build_index --rebuild
```

### Run Semantic Vector Search (CLI Demo)
```bash
# Run preset demonstration queries
python -m rag.search_cli

# Run a custom natural language query
python -m rag.search_cli "spicy grilled chicken in Gulberg" --top-k 5
```

---

## 9.1 Milestone 4 — Basic RAG Pipeline

The baseline RAG pipeline implements end-to-end menu question answering with strict grounding:

### Architecture Flow:
```
User Query -> Query Embedding -> ChromaDB Vector Retrieval -> Context Builder -> Grounded Generator -> Answer with [dish_xxx] Citations
```

### Components:
- **`rag/retriever.py`:** Retrieves top-k candidate dishes from ChromaDB using 384-dimensional dense vectors.
- **`rag/context_builder.py`:** Formats retrieved candidate dishes into distinct `[DOCUMENT i]` context blocks containing prices, location, Halal status, spice level, ingredients, and source IDs.
- **`rag/prompts.py`:** Enforces strict grounding rules (no hallucinations, mandatory source citations, no unverified allergen claims).
- **`rag/generator.py`:** Configurable generator interface supporting OpenAI chat models and deterministic offline mock fallback.
- **`rag/pipeline.py`:** Orchestrates query retrieval, context construction, and grounded generation with latency telemetry.

### Running Baseline RAG:
```bash
# Run demonstration benchmark queries
python -m rag.demo_rag

# Run Python code directly:
python -c "from rag.pipeline import answer_query; print(answer_query('What chicken dishes are available?', top_k=3)['answer'])"
```

### Known Limitations of Baseline RAG:
- Semantic retrieval alone cannot enforce numeric budget limits (e.g. `price <= 1500`).
- Semantic retrieval cannot enforce strict allergen exclusions.
- Exact keywords may be missed without BM25 sparse search (Milestone 6).

---

## 9.2 Milestone 5 — Constraint Parsing & Metadata Pre-Filtering

Milestone 5 addresses the numeric and structured filtering limitations by introducing deterministic query constraint parsing and metadata pre-filtering:

### Pipeline Flow:
```
User Query
    │
    ▼
Query Parser (rag/query_parser.py)
 ├── Hard Constraints: max_price_pkr, min_price_pkr, halal, location, excluded_allergens
 └── Soft Preferences: category, cuisine, spice_level, protein_preference
    │
    ▼
Metadata Pre-Filter (rag/metadata_filter.py)
 ├── Price Verification: price <= max_price_pkr
 ├── Halal Verification: halal == true
 ├── Location Verification: location == canonical_zone
 └── Allergen Dual-Scan: tag checking + ingredient string pattern matching
    │
    ├── 0 Matches ──► Status: NO_MATCHING_ITEMS (Bypasses LLM, 0 ms generation latency)
    │
    ▼
Passed Candidates ──► Context Builder ──► Grounded LLM Generation
```

### Key Modules:
- **`rag/query_parser.py`:** Extracts hard constraints and soft preferences using high-precision regexes, gazetteer matching, and context-aware masking for allergen exceptions (e.g. coconut milk, almond milk, peanut butter).
- **`rag/metadata_filter.py`:** Evaluates candidate documents against hard constraints, attaches warnings for unverified allergen statuses (`dish_009`: `ALLERGEN_INFORMATION_UNKNOWN`, `dish_033`: `CONFLICTING_ALLERGEN_EVIDENCE`), and returns structured `FilterResult`.
- **`rag/retriever.py`:** Implements `retrieve_filtered(query, constraints, top_k)` to pre-filter candidates before final candidate selection.
- **`rag/pipeline.py`:** Orchestrates query parsing, candidate filtering, context creation, and grounded generation with full latency telemetry.

### Running Constraint-Aware RAG Demo:
```bash
python -m rag.demo_rag
```

---

## 9.3 Milestone 6 — BM25 + Hybrid Retrieval + Reciprocal Rank Fusion (RRF)

Milestone 6 couples sparse lexical keyword matching with dense vector retrieval using Reciprocal Rank Fusion (RRF):

### Architecture Flow:
```
User Query
    │
    ▼
Query Parser (rag/query_parser.py)
    │
    ▼
Hard Metadata Pre-filter (rag/metadata_filter.py)
    │
    ├── 0 Matches ──► Status: NO_MATCHING_ITEMS (Immediate rejection)
    │
    ▼
Allowed Dish IDs
    │
    ├───────────────────────────────┐
    ▼                               ▼
Dense Retrieval (ChromaDB)      BM25 Index (rag/bm25.py)
    │                               │
    └───────────────┬───────────────┘
                    ▼
        Reciprocal Rank Fusion (RRF)
        Score = sum( 1 / (k + rank) )
                    │
                    ▼
       Top-K Candidates with Telemetry
                    │
                    ▼
             Context Builder
                    │
                    ▼
         Grounded LLM Generation
```

### Key Modules:
- **`rag/bm25.py`:** Deterministic lexical index using Okapi BM25 (`rank-bm25.BM25Okapi`, $k_1=1.5, b=0.75$) over menu tokens (dish names, restaurants, descriptions, ingredients, categories, cuisines).
- **`rag/hybrid_retriever.py`:** Runs parallel dense vector and BM25 searches restricted to pre-filtered allowed IDs, merges results via RRF ($k=60$), deduplicates by stable dish ID, and preserves all allergen metadata and safety warnings.
- **`rag/pipeline.py`:** Integrates `HybridRetriever` by default (configurable via `RETRIEVAL_MODE=hybrid|dense`).

### Running Hybrid RAG Demo:
```bash
python -m rag.demo_rag
```

---

## 9.4 Milestone 7 — Cross-Encoder Reranking

Milestone 7 integrates a deep neural cross-encoder reranker after the Hybrid RRF stage and prior to context construction:

### Pipeline Flow:
```
User Query
    │
    ▼
Query Parser (rag/query_parser.py)
    │
    ▼
Hard Metadata Pre-filter (rag/metadata_filter.py)
    │
    ▼
Dense + BM25 Retrieval (restricted to allowed IDs)
    │
    ▼
Reciprocal Rank Fusion (retrieves candidate pool: candidate_k = 15)
    │
    ▼
Cross-Encoder Reranker (rag/reranker.py)
 ├── Token-to-token cross-attention between query and candidate text
 └── Sorts by cross-encoder score descending (top_k = 5)
    │
    ▼
Context Builder ──► Grounded LLM Generation
```

### Key Modules:
- **`rag/reranker.py`:** Implements `SentenceTransformerReranker` using `cross-encoder/ms-marco-MiniLM-L-6-v2` with `DeterministicFallbackReranker` fallback. Ensures stable dish ID tracking, deterministic tie-breaking on dish ID, and preservation of all metadata and safety warnings.
- **`rag/pipeline.py`:** Integrates the cross-encoder reranker into `BasicRAGPipeline`, reporting separate `rerank_ms` latency metrics.

### Running End-to-End Reranked Demo:
```bash
python -m rag.demo_rag
```

---

## 9.5 Milestone 8 — Allergen Safety Verifier Module

Milestone 8 introduces an independent, deterministic Allergen Safety Verifier situated between cross-encoder reranking and context construction:

### Pipeline Flow:
```
User Query
    │
    ▼
Query Parser (rag/query_parser.py)
    │
    ▼
Hard Metadata Pre-filter (rag/metadata_filter.py)
    │
    ▼
Dense + BM25 Hybrid Retrieval + RRF (rag/hybrid_retriever.py)
    │
    ▼
Cross-Encoder Reranker (rag/reranker.py)
    │
    ▼
Allergen Safety Verifier (rag/allergen_verifier.py)
 ├── Strict Tag & Ingredient Regex Scan
 ├── VERIFIED_PASS ──────────► Passed to Context Builder
 ├── REJECT ────────────────► Omitted from prompt
 ├── INSUFFICIENT_EVIDENCE ──► Omitted + Exact Invariant Warning
 └── CONFLICTING_EVIDENCE ───► Omitted + Exact Invariant Warning
    │
    ├── 0 Matches ──► Return Status: NO_MATCHING_ITEMS (Immediate rejection)
    │
    ▼
Context Builder ──► Grounded LLM Generation
```

### Key Modules:
- **`rag/allergen_verifier.py`:** Enforces 4 definitive safety states (`VERIFIED_PASS`, `REJECT`, `INSUFFICIENT_EVIDENCE`, `CONFLICTING_EVIDENCE`). Implements context-aware regex masking for plant milk/non-dairy butter/gf flour.
- **Safety Invariant Warnings:**
  - `dish_009`: `"Warning for 'Mystery Special Daily Daal': ALLERGEN_INFORMATION_UNKNOWN. Cannot guarantee allergen-free status due to unverified allergen data."`
  - `dish_033`: `"Warning for 'Chefs Secret Karahi': CONFLICTING_ALLERGEN_EVIDENCE. Cannot guarantee allergen-free status due to unverified allergen data."`
- **`rag/pipeline.py`:** Integrates `AllergenSafetyVerifier`, deduplicating pre-filter and downstream warnings, short-circuiting with `NO_MATCHING_ITEMS` if all candidates fail verification, and recording `verify_ms` latency metrics.

---

## 9.6 Milestone 9 — Citation Engine & Grounded Synthesis

Milestone 9 couples deterministic grounded response synthesis with multi-point factual claim and citation verification:

### Pipeline Flow:
```
User Query
    │
    ▼
Query Parser (rag/query_parser.py)
    │
    ▼
Hard Metadata Pre-filter (rag/metadata_filter.py)
    │
    ▼
Dense + BM25 Hybrid Retrieval + RRF (rag/hybrid_retriever.py)
    │
    ▼
Cross-Encoder Reranker (rag/reranker.py)
    │
    ▼
Allergen Safety Verifier (rag/allergen_verifier.py)
    │
    ▼
Context Builder (rag/context_builder.py)
    │
    ▼
Grounded Generator (rag/generator.py)
    │
    ▼
Citation Engine & Grounding Verifier (rag/citation.py) ◄── [MILESTONE 9 GROUNDING GATE]
 ├── Extracts bracket citations: [dish_xxx]
 ├── Detects phantom / hallucinated source IDs
 ├── Validates price claims against verified menu prices
 ├── Prevents unsupported allergen-free claims on UNKNOWN / CONFLICT records
 └── Classifies GroundingStatus (SUPPORTED, INSUFFICIENT_EVIDENCE, CONFLICTING_EVIDENCE, etc.)
    │
    ▼
Structured JSON Response with Grounding Telemetry
```

### Key Modules:
- **`rag/citation.py`:** Implements `CitationEngine`, `Citation`, `GroundingStatus`, and `GroundingVerificationReport`. Verifies factual claims, extracts structured citations, prevents hallucinations, and enforces invariant allergen safety warnings.
- **`rag/generator.py`:** `MockGroundedGenerator` delegates to `CitationEngine.synthesize_grounded_answer()` ensuring deterministic offline grounded responses.
- **`rag/pipeline.py`:** Emits structured `citations`, `results`, and `grounding` fields alongside standard `answer`, `sources`, and `warnings`.

---

## 10. How to Start FastAPI
```bash
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```
Interactive API docs available at: `http://localhost:8000/docs`

### Available Endpoints:
- **`GET /health`**: Health status and vector index readiness.
- **`POST /query`**: Grounded search and recommendation engine. Accepts natural language query and optional constraint parameters (`max_price_pkr`, `location`, `halal`, `excluded_allergens`, `spice_level`, `top_k`).
- **`GET /menu`**: Catalog menu items with optional filtering (`location`, `halal`, `max_price`, `category`) and pagination (`skip`, `limit`).
- **`GET /metrics`**: Operational telemetry, collection item count, and index dimension.

### Example cURL Query:
```bash
curl -X POST "http://localhost:8000/query" \
     -H "Content-Type: application/json" \
     -d '{"query": "chicken in Gulberg", "excluded_allergens": ["dairy"], "max_price_pkr": 2000}'
```

---

## 11. How to Start Streamlit UI
```bash
python -m streamlit run ui/app.py
```

### UI Features:
- **Natural Language Search Bar**: Accepts conversational food requests.
- **Sidebar Constraint Overrides**: Interactive controls for Delivery Zone, Halal-only, Max Budget (PKR) slider, Spice Level, Strict Allergen Exclusion checkboxes, and Top-K results.
- **Quick-Prompt Buttons**: Pre-set queries for one-click testing of budget, allergens, and safety invariant edge cases (`dish_009` UNKNOWN, `dish_033` CONFLICT).
- **Structured Dish Cards**: Rich cards displaying exact prices, location, Halal badges, spice ratings, allergen verification status, and match explanations.
- **Active Safety Advisories**: Prominent warning banner displaying exact cautionary notices for unverified or conflicting allergen items.
- **Citations & Grounding Telemetry**: Interactive expander showing raw citations and end-to-end latency breakdown.

---

## 12. How to Run Tests
```bash
python -m pytest tests/ -v
```
The test suite consists of **155 unit, integration, and benchmark tests** across 16 test files covering schemas, data ingestion, embeddings, vector stores, basic retrieval, constraint parsing, metadata pre-filtering, BM25 indexing, hybrid RRF, cross-encoder reranking, allergen safety verification, citation engine, FastAPI REST API, Streamlit Web UI, and the Milestone 12 Evaluation Benchmark Suite.

---

## 13. How to Run Evaluation Benchmark Suite
The project includes a fully reproducible, deterministic evaluation benchmark suite ([`evaluation/`](file:///c:/Users/sheik/Downloads/Food%20Rag%20Application/evaluation)) evaluating retrieval architectures and end-to-end grounded generation across 15 representative scenarios:

```bash
# Run Master Benchmark Orchestrator (Retrieval + Generation + Master Audit)
python -m evaluation.run_benchmark

# Run Retrieval & Ranking Comparative Benchmark Only
python -m evaluation.evaluate_retrieval

# Run End-to-End Grounded Generation & Safety Benchmark Only
python -m evaluation.evaluate_generation

# Run Benchmark Unit & Integration Tests
python -m pytest tests/test_evaluation.py -v
```

Generated reports are persisted in `evaluation/reports/`:
- `evaluation/reports/retrieval_benchmark.json`: Architectural comparison (Dense, BM25, Hybrid RRF, Reranker, Pipeline).
- `evaluation/reports/generation_benchmark.json`: End-to-end safety, budget, dietary, citation, and latency metrics.
- `evaluation/reports/benchmark_summary.json`: Master executive audit report.

---

## 14. Example Queries
- *"Find high-protein Halal chicken meals under PKR 1,500 without dairy or peanuts available in Gulberg."*
- *"Gluten-free beef burger under 2000 in DHA"*
- *"Spicy biryani without shellfish in Saddar"*
- *"Vegetarian meal under 800 without dairy"*

---

## 15. Safety & Allergen Logic
The system enforces zero tolerance for allergenic risk:
1. **Hard Exclusion**: Dishes containing declared user allergens are purged.
2. **Ingredient Scanning**: Hidden allergen ingredients (e.g. whey, ghee, curd, cashew, satay) trigger exclusion.
3. **Missing Info Guard**: Dishes with empty or unverified allergen records are marked `INSUFFICIENT_EVIDENCE` and excluded from affirmative recommendations.
4. **LLM Sandboxing**: LLM response prompts only receive verified safe chunks.

---

## 16. Evaluation Results (Milestone 12 Live Audit)
Measured metrics across the 15 benchmark scenarios (`evaluation/reports/benchmark_summary.json`):

### Retrieval Architecture Comparison:
| Architecture | MRR | Recall@5 | NDCG@5 | Forbidden Leakage | Latency |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Dense Vector Search | 0.9111 | 0.4889 | 0.4920 | 0.1467 (14.7% Leak) | 28.20 ms |
| BM25 Lexical Search | 0.8148 | 0.4556 | 0.4412 | 0.1800 (18.0% Leak) | 0.45 ms |
| Hybrid (Dense + BM25 RRF) | 0.8222 | 0.5222 | 0.4735 | 0.2133 (21.3% Leak) | 28.52 ms |
| Hybrid + Cross-Encoder Reranker | 0.7407 | 0.5389 | 0.4592 | 0.2400 (24.0% Leak) | 29.31 ms |
| **Production Pipeline (Pre-filter + Hybrid + Rerank)** | **0.9444** | **0.8444** | **0.8318** | **0.0000 (0.0% Leak)** | **21.53 ms** |

### End-to-End Safety & Generation Quality:
- **Allergen Safety Recall (Zero Tolerance)**: **100.0%** (1.0000)
- **Budget Price Precision (Mathematical Inequality)**: **100.0%** (1.0000)
- **Halal Dietary Recall**: **100.0%** (1.0000)
- **Citation Accuracy (Non-hallucinated Sources)**: **100.0%** (1.0000)
- **Safety Invariant Compliance (`dish_009` UNKNOWN / `dish_033` CONFLICT)**: **100.0%** (1.0000)
- **Negative Query Rejection Accuracy (`NO_MATCHING_ITEMS`)**: **100.0%** (1.0000)
- **Mean End-to-End Pipeline Latency**: **18.96 ms**

---

## 17. Project Limitations
- Academic demonstration dataset is synthetic/representative and not real-time Foodpanda API data.
- Cross-contact/kitchen airborne contamination cannot be verified solely from digital menus.
- Not intended as a substitute for professional medical or allergist advice.

---

## 18. Future Improvements
- Multi-modal ingestion (OCR on photographed restaurant physical menus).
- Cross-contamination probability scoring based on restaurant cuisine profiles.
- Live vendor inventory and pricing synchronization.
