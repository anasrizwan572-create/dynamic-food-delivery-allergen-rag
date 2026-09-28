# Product Requirements Document (PRD)

## Project Title
**Dynamic Food Delivery Menu & Allergen RAG**

## Project Domain
Pakistani Food Delivery & Restaurant Aggregator Platforms (Foodpanda PK, Pakistani restaurant chains, cloud kitchens, and digital menu platforms).

---

## 1. Executive Summary & Problem Statement
In food delivery applications, food allergy safety is a critical matter. Traditional chatbot or generic search implementations often suffer from hallucinated ingredients, ambiguous allergen status, or missed dietary restrictions (e.g., cross-contamination, hidden dairy, nuts, shellfish, or non-Halal preparation).
A simple vector search or naive LLM pipeline is dangerous: vector embeddings may place "peanut butter cookies" and "peanut-free oatmeal cookies" close together in semantic space. If an LLM answers without rigid guardrails, it risks recommending allergenic foods to vulnerable customers.

**Dynamic Food Delivery Menu & Allergen RAG** addresses this by pairing hybrid search (Dense Vector + BM25 Sparse Search + Reciprocal Rank Fusion) and cross-encoder reranking with an **independent, deterministic Allergen Safety Verification engine** before grounded generation. Allergen exclusion is treated as a **hard constraint**, ensuring that no unverified or allergen-containing dish is ever recommended.

---

## 2. Target Context & Users
- **Target Platforms**: Food delivery platforms in Pakistan (Lahore, Karachi, Islamabad, Rawalpindi, etc.) covering local restaurant chains and cloud kitchens.
- **Target Users**:
  - Customers with strict dietary allergies (peanuts, tree nuts, dairy/lactose, gluten, shellfish, eggs, soy).
  - Customers seeking budget-conscious meals (PKR limits) with specific requirements (Halal verification, spice levels, high protein).
  - Restaurant aggregators requiring rigorous auditability, safety verification, and transparent citations.

---

## 3. System Architecture & Pipeline Flow
The architecture enforces strict separation of concerns across retrieval, safety verification, and grounded generation:

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
              QUERY / CONSTRAINT PARSER (Deterministic + Rule-based Synonyms)
                       │
                       ▼
             METADATA PRE-FILTER (Price, Halal, Location, Excluded Allergens)
                       │
          ┌────────────┴────────────┐
          ▼                         ▼
    DENSE VECTOR SEARCH         BM25 SEARCH (Sparse Keyword)
    (Cosine Similarity)         (Tokenized Text Corpus)
          │                         │
          └────────────┬────────────┘
                       │
                       ▼
                 RRF FUSION (Reciprocal Rank Fusion, k=60)
                       │
                       ▼
                  RERANKER (Cross-Encoder / Relevance Scorer)
                       │
                       ▼
             ALLERGEN SAFETY VERIFIER (Deterministic Rule Engine)
             [Match -> REJECT, Missing/Ambiguous -> INSUFFICIENT_EVIDENCE,
              Conflicting -> CONFLICTING_EVIDENCE, Verified -> PASS]
                       │
                       ▼
              VERIFIED EVIDENCE CHUNKS
                       │
                       ▼
                     LLM (Strict Grounding & Zero-Hallucination Prompt)
                       │
                       ▼
             STRUCTURED RESPONSE (Status, Results, Citations, Warnings)
                       │
                       ▼
                    WEB UI / REST API
```

---

## 4. Key Functional Requirements

### 4.1 Data Ingestion & Normalization
- Ingestion of structured menu items (CSV, JSON, extensible to PDF/unstructured).
- Data fields:
  - `id`: Unique identifier (e.g. `dish_001`)
  - `restaurant_name`: Restaurant name
  - `dish_name`: Name of dish
  - `description`: Textual summary
  - `ingredients`: Comma-separated or listed ingredients
  - `price_pkr`: Price in Pakistani Rupees
  - `location`: Delivery zone/city (e.g., Gulberg, DHA, F-7, Saddar)
  - `halal`: Boolean indicator (True/False)
  - `spice_level`: Mild, Medium, Spicy, Extra Spicy
  - `allergens`: Normalized list of allergens (e.g., `["dairy", "peanuts"]`)
  - `category`: Menu category (Chicken, Beef, Vegetarian, Seafood, Dessert, etc.)
  - `protein_g`: Protein content in grams
  - `availability`: Boolean stock status
  - `cuisine`: Cuisine type (Pakistani, Chinese, Continental, etc.)
  - `source`: Source document or menu reference
  - `source_url`: URL or file reference
  - `timestamp`: Ingestion/update timestamp
- Normalization:
  - Allergen synonym standardization (e.g. `milk` -> `dairy`, `peanut` -> `peanuts`, `wheat`/`flour` -> `gluten`).
  - Handling missing or empty fields with explicit `"unknown"` / missing flags to prevent false safety assumptions.

### 4.2 Query & Constraint Parser
- Extracts both **Hard Constraints** and **Soft Preferences**:
  - **Hard Constraints**: Maximum price (`max_price_pkr`), Halal requirement (`halal`), Location (`location`), Excluded Allergens (`excluded_allergens`).
  - **Soft Preferences**: Category preference, protein target (`high` / `> X g`), spice level, cuisine.
- Deterministic extraction: Regex and lookup dictionaries for allergy terms, prices (`PKR 1500`, `under 1200`), locations, and Halal indicators.

### 4.3 Hybrid Retrieval & Reranking
- **Dense Vector Search**: Embeddings computed over composite semantic texts (name, description, ingredients, category). Vector database with cosine distance.
- **Sparse BM25 Search**: Lexical BM25 retrieval across tokenized menu texts to guarantee exact keyword matches (e.g., specific dish names, ingredients).
- **Reciprocal Rank Fusion (RRF)**:
  $$RRF\_Score(d) = \sum_{m \in \{Dense, BM25\}} \frac{1}{k + rank_m(d)}$$
  with default $k = 60$.
- **Reranker**: Cross-Encoder relevance scoring of the fused top-N candidate pool.

### 4.4 Allergen Safety Verifier
- Dedicated module operating **prior to LLM generation**.
- Evaluates:
  1. **Explicit allergen match**: Dish or ingredient list includes user-excluded allergen $\rightarrow$ **REJECT**.
  2. **Reliable evidence of absence**: Verified allergen list does not contain excluded allergen, and no suspicious keywords in ingredients $\rightarrow$ **PASS**.
  3. **Missing or ambiguous allergen data**: If a dish has unverified or empty allergen metadata $\rightarrow$ Flagged as **INSUFFICIENT_EVIDENCE**, accompanied by mandatory cautionary warning.
  4. **Conflicting evidence**: Conflicting records across data sources $\rightarrow$ Flagged as **CONFLICTING_EVIDENCE**.
- **Crucial Rule**: The LLM is never allowed to override safety verification decisions.

### 4.5 Grounded LLM Generation
- System prompt enforcing strict evidence boundaries:
  - Answers strictly using provided verified evidence chunks.
  - Zero tolerance for fabricating prices, ingredients, or allergen statuses.
  - If no items qualify, cleanly returns `NO_MATCHING_ITEMS`.
  - Transparent inline citations linking each dish to its unique ID and source menu.

### 4.6 Failure & Verification States
Typed response statuses:
- `SUPPORTED`: High-confidence recommendations satisfying all hard and soft constraints.
- `PARTIALLY_SUPPORTED`: Matches hard constraints, but minor soft preferences partially unmet or minor non-critical warnings.
- `INSUFFICIENT_EVIDENCE`: Potential matches found, but allergen or safety data is unverified or ambiguous.
- `CONFLICTING_EVIDENCE`: Discrepancy detected between restaurant menu versions or ingredient lists.
- `NO_MATCHING_ITEMS`: No menu items satisfy the hard constraints (budget, location, Halal, or allergens).

---

## 5. Interface & API Specifications
- **REST API (FastAPI)**:
  - `GET /health`: Health and index status.
  - `POST /query`: Structured JSON input `{"query": "..."}` returning parsed constraints, results, warnings, citations, and execution latency.
  - `GET /menu`: Inspect menu dataset.
- **Web UI (Streamlit)**:
  - Natural-language search box with preset quick-test queries.
  - Real-time display of parsed query constraints.
  - Rich result cards displaying dish name, restaurant, price in PKR, location, spice level, Halal badge, allergens, match rationale, and source citation.
  - Dedicated callouts for safety warnings, insufficient evidence, and conflict states.

---

## 6. Evaluation Framework
- **Gold-Standard Evaluation Set**: 25–50 benchmark test cases covering:
  - Exact allergen exclusion
  - Multiple allergen combinations
  - Budget + Halal + Location intersections
  - Unknown/missing allergen edge cases
  - Non-Halal dishes
  - Conflicting data sources
  - Exact dish name queries and ingredient keyword lookups
- **Metrics**:
  - Retrieval: Recall@K, Precision@K, Mean Reciprocal Rank (MRR), NDCG.
  - Safety & Filtering: Allergen Filter Recall (100% target for safety), Dietary Filter Recall, Price Precision, Citation Accuracy.
  - Generation: Faithfulness, Groundedness, Hallucination Rate.
