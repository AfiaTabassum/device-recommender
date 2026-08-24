# milvus_tools.py
# ============================================================
# Milvus / Zilliz search tool registry for StarTech product DB.
#
# Wraps three search strategies as LLM-callable tools:
#   - dense_search          : pure semantic vector search
#   - hybrid_search         : dense + BM25 via WeightedRanker
#   - hybrid_search_rrf     : dense + BM25 via RRF (no weight tuning)
#
# Public interface (mirrors ragtools.py convention):
#   get_available_tools()   → list[dict]   (all tool schemas)
#   handle_tool_call(tool_call) → any      (dispatcher)
#   create_tool_response_message(tool_call, result) → dict
#
# Internal setup:
#   _init_milvus()   — called once at import time.
#                       Connects to Zilliz, loads the selected
#                       collection, and fits the embedding models.
#                       Safe to import even if credentials are
#                       missing (prints a warning, disables tools).
#
# ── INDEX SELECTION ──────────────────────────────────────────
# After benchmarking, set INDEX_NAME below to the winning
# Milvus collection name and leave it.  All three tool
# functions read this constant at call time — no per-call
# parameter needed.
# ============================================================
#!pip install pymilvus
#!pip install pymilvus.model



import os
import json

from dotenv import load_dotenv
load_dotenv()


# ============================================================
# CONFIGURATION  — edit these before running
# ============================================================

# ── Zilliz Cloud credentials ──────────────────────────────────
ZILLIZ_URI   = os.environ["ZILLIZ_URI"]
ZILLIZ_TOKEN = os.environ["ZILLIZ_TOKEN"]

# ── ★ SELECT YOUR INDEX HERE after benchmarking ★ ─────────────
# Replace with the winning collection name from your benchmark results.
# Options (from insertion notebook):
#   "startech_knn_flat"   (FLAT  — exact KNN, highest recall)
#   "startech_ivf_flat"   (IVF_FLAT)
#   "startech_pq"         (IVF_PQ nlist=1)
#   "startech_ivf_pq"     (IVF_PQ)
#   "startech_hnsw"       (HNSW  — best speed/recall trade-off)
INDEX_NAME = "startech_hnsw"

# ── Search params per collection (must match index type) ───────
_SEARCH_PARAMS_MAP = {
    "startech_knn_flat": {"metric_type": "COSINE", "params": {}},
    "startech_ivf_flat": {"metric_type": "COSINE", "params": {"nprobe": 2}},
    "startech_pq":       {"metric_type": "COSINE", "params": {"nprobe": 1}},
    "startech_ivf_pq":   {"metric_type": "COSINE", "params": {"nprobe": 2}},
    "startech_hnsw":     {"metric_type": "COSINE", "params": {"ef": 8}},
}

# ── Model & search defaults ────────────────────────────────────
_MODEL_NAME   = "BAAI/bge-small-en-v1.5"
_BM25_SAMPLE  = 2000      # corpus docs to fit BM25 IDF (2 000 is sufficient)
_DEFAULT_TOP_K = 5

# ── Fields returned by every search ───────────────────────────
_OUTPUT_FIELDS = [
    "id", "device_name", "device_category", "brand", "price", "price_currency",
    "cpu", "ram", "storage", "gpu", "display", "battery", "operating_system",
    "description", "review_text", "device_text",
    "source", "url", "embed_text",
]


# ============================================================
# MODULE-LEVEL SINGLETONS  — populated by _init_milvus()
# ============================================================

_client       = None    # MilvusClient (sole Milvus API)
_embed_model  = None    # SentenceTransformer for dense vectors
_bm25_ef      = None    # BM25EmbeddingFunction for sparse vectors
_dense_params = None    # search_params dict matching INDEX_NAME
_milvus_ok    = False   # False if init failed; all tools return error dicts


# ============================================================
# INITIALISATION
# ============================================================

def _init_milvus() -> None:
    """
    Connect to Zilliz Cloud, load INDEX_NAME collection into memory,
    and initialise the dense + BM25 embedding models.

    Called automatically at module import time.
    Sets _milvus_ok = True on success.
    On any failure, prints a warning and leaves tools disabled.
    """
    global _client, _embed_model, _bm25_ef, _dense_params, _milvus_ok

    try:
        from pymilvus import MilvusClient
        from sentence_transformers import SentenceTransformer
        from pymilvus.model.sparse import BM25EmbeddingFunction
        from pymilvus.model.sparse.bm25.tokenizers import build_default_analyzer

        # ── Connect via MilvusClient (no ORM connections.connect) ──
        _client = MilvusClient(uri=ZILLIZ_URI, token=ZILLIZ_TOKEN)
        print(f"[milvus_tools] Connected to Zilliz: {ZILLIZ_URI}")

        # ── Server version via MilvusClient ───────────────────────
        server_info = _client.get_server_version()
        print(f"[milvus_tools] Server version: {server_info}")

        # ── Check collection exists ───────────────────────────────
        existing = _client.list_collections()
        if INDEX_NAME not in existing:
            raise RuntimeError(
                f"Collection '{INDEX_NAME}' not found. "
                "Run the insertion notebook first, or update INDEX_NAME."
            )

        # ── Load collection ───────────────────────────────────────
        _client.load_collection(INDEX_NAME)
        stats = _client.get_collection_stats(INDEX_NAME)
        num_entities = stats.get("row_count", "unknown")
        print(f"[milvus_tools] Collection '{INDEX_NAME}' loaded "
              f"({num_entities} entities).")

        # ── Dense search params for this index ────────────────────
        _dense_params = _SEARCH_PARAMS_MAP.get(
            INDEX_NAME,
            {"metric_type": "COSINE", "params": {"ef": 64}},   # safe fallback
        )

        # ── Dense embedding model ──────────────────────────────────
        print(f"[milvus_tools] Loading dense model: {_MODEL_NAME} …")
        _embed_model = SentenceTransformer(_MODEL_NAME)
        print("[milvus_tools] Dense model ready.")

        # ── BM25 — fit on corpus sample ───────────────────────────
        print(f"[milvus_tools] Fetching {_BM25_SAMPLE} docs for BM25 fit …")
        sample_res = _client.query(
            collection_name=INDEX_NAME,
            filter="id >= 0",
            output_fields=["embed_text"],
            limit=_BM25_SAMPLE,
        )
        corpus = [r["embed_text"] for r in sample_res if r.get("embed_text")]
        analyzer = build_default_analyzer(language="en")
        _bm25_ef  = BM25EmbeddingFunction(analyzer, num_workers=1)
        _bm25_ef.fit(corpus)
        print(f"[milvus_tools] BM25 fitted on {len(corpus)} documents.")

        _milvus_ok = True
        print(f"[milvus_tools] ✅ Init complete. Active index: {INDEX_NAME}")

    except Exception as exc:
        print(f"[milvus_tools] WARNING: Init failed — all Milvus tools disabled.\n  Error: {exc}")
        _milvus_ok = False


# NOTE: _init_milvus() is NOT auto-called here.
# App.py calls laptop_SP_search_tools._init_milvus() explicitly
# AFTER injecting st.secrets into os.environ so credentials are available.


# ============================================================
# INTERNAL HELPERS
# ============================================================

def _get_dense_vec(query: str) -> list:
    """Encode query → normalised dense float vector."""
    return _embed_model.encode([query], normalize_embeddings=True)[0].tolist()


def _get_sparse_vec(query: str) -> dict:
    """
    Encode query → sparse BM25 {token_index: weight} dict.
    Handles both coo_array and csr_matrix depending on scipy version.
    """
    mat = _bm25_ef.encode_queries([query]).tocsr()
    return dict(zip(
        mat.indices[mat.indptr[0]:mat.indptr[1]].tolist(),
        mat.data   [mat.indptr[0]:mat.indptr[1]].tolist(),
    ))


def _build_filter(
    brand: str | None           = None,
    device_category: str | None = None,
    price_min: float | None     = None,
    price_max: float | None     = None,
    ram: str | None             = None,
    storage: str | None         = None,
) -> str:
    """Build a Milvus boolean expression from optional metadata filters."""
    clauses = []
    if brand:
        clauses.append(f'brand == "{brand}"')
    if device_category:
        clauses.append(f'device_category == "{device_category}"')
    if price_min is not None:
        clauses.append(f"price >= {price_min}")
    if price_max is not None:
        clauses.append(f"price <= {price_max}")
    if ram:
        clauses.append(f'ram like "%{ram}%"')
    if storage:
        clauses.append(f'storage like "%{storage}%"')
    return " && ".join(clauses)


def _hits_to_dicts(hits) -> list[dict]:
    """
    Convert a list of pymilvus Hit objects to plain dicts so results
    are JSON-serialisable for the LLM tool-response message.
    """
    results = []
    for hit in hits:
        entry = {"score": round(hit.score, 6)}
        entry.update({field: hit.entity.get(field) for field in _OUTPUT_FIELDS})
        results.append(entry)
    return results


def _client_results_to_dicts(results) -> list[dict]:
    """
    Convert MilvusClient hybrid_search results to plain dicts.
    MilvusClient.hybrid_search returns a list-of-lists (one inner list per
    query request); we always issue a single query so we take [0] to get
    the flat list of HybridHits, then access .distance and .entity as
    attributes (same as the ORM Hit objects).
    """
    output = []
    for hit in results[0]:
        score = getattr(hit, "distance", None) or getattr(hit, "score", 0)
        entry = {"score": round(score, 6)}
        entity = getattr(hit, "entity", {})
        entry.update({field: entity.get(field) for field in _OUTPUT_FIELDS})
        output.append(entry)
    return output


def _not_ready_error(tool_name: str) -> list[dict]:
    return [{"error": f"[{tool_name}] Milvus not initialised. Check credentials and INDEX_NAME."}]


# ============================================================
# TOOL FUNCTIONS
# Each function reads INDEX_NAME / _client from module scope.
# The `index_name` variable is intentionally a module constant
# (not a parameter) — set it once after selecting the best index.
# ============================================================

def dense_search(
    query: str,
    top_k: int                      = _DEFAULT_TOP_K,
    brand: str | None               = None,
    device_category: str | None     = None,
    price_min: float | None         = None,
    price_max: float | None         = None,
    ram: str | None                 = None,
    storage: str | None             = None,
) -> list[dict]:
    """
    Pure dense (semantic) vector search against INDEX_NAME.

    Embeds `query` with BAAI/bge-small-en-v1.5 and runs an ANN search
    over the `dense_vector` field using the index-specific search params.

    Args:
        query           : Natural-language search query.
        top_k           : Number of results to return (default 5).
        brand           : Optional brand filter, e.g. "Samsung".
        device_category : Optional category filter, e.g. "Laptop".
        price_min       : Optional minimum price filter.
        price_max       : Optional maximum price filter.
        ram             : Optional RAM substring filter matched against the `ram`
                          field (case-insensitive contains). Extract from the user
                          query, e.g. "16GB", "32GB", "8GB", "12GB".
        storage         : Optional storage substring filter matched against the
                          `storage` field (case-insensitive contains). Extract from
                          the user query, e.g. "512GB", "1TB", "256GB", "2TB".

    Returns:
        List of result dicts, each containing score + all OUTPUT_FIELDS.
    """
    # ── index_name constant (set at module level after benchmarking) ──
    index_name = INDEX_NAME

    if not _milvus_ok:
        return _not_ready_error("dense_search")

    filter_expr = _build_filter(brand, device_category, price_min, price_max, ram, storage)

    hits = _client.search(
        collection_name=INDEX_NAME,
        data=[_get_dense_vec(query)],
        anns_field="dense_vector",
        search_params=_dense_params,
        limit=top_k,
        filter=filter_expr or "",
        output_fields=_OUTPUT_FIELDS,
    )[0]

    return _hits_to_dicts(hits)


def hybrid_search(
    query: str,
    top_k: int                      = _DEFAULT_TOP_K,
    dense_weight: float             = 0.7,
    sparse_weight: float            = 0.3,
    brand: str | None               = None,
    device_category: str | None     = None,
    price_min: float | None         = None,
    price_max: float | None         = None,
    ram: str | None                 = None,
    storage: str | None             = None,
) -> list[dict]:
    """
    Weighted hybrid search — combines dense + BM25 sparse via WeightedRanker.

    Runs two ANN sub-searches (dense + sparse) against INDEX_NAME and
    re-ranks the merged candidate pool with weighted score fusion.

    Args:
        query         : Search query string.
        top_k         : Number of results to return (default 5).
        dense_weight  : Weight for the dense sub-search (0.0–1.0, default 0.7).
        sparse_weight : Weight for the BM25 sub-search (0.0–1.0, default 0.3).
        brand           : Optional brand filter.
        device_category : Optional category filter.
        price_min       : Optional minimum price filter.
        price_max       : Optional maximum price filter.
        ram             : Optional RAM substring filter matched against the `ram`
                          field (case-insensitive contains). Extract from the user
                          query, e.g. "16GB", "32GB", "8GB", "12GB".
        storage         : Optional storage substring filter matched against the
                          `storage` field (case-insensitive contains). Extract from
                          the user query, e.g. "512GB", "1TB", "256GB", "2TB".

    Returns:
        List of result dicts, each containing score + all OUTPUT_FIELDS.
    """
    # ── index_name constant (set at module level after benchmarking) ──
    index_name = INDEX_NAME

    if not _milvus_ok:
        return _not_ready_error("hybrid_search")

    from pymilvus import AnnSearchRequest, WeightedRanker

    filter_expr = _build_filter(brand, device_category, price_min, price_max, ram, storage)

    results = _client.hybrid_search(
        collection_name=INDEX_NAME,
        reqs=[
            AnnSearchRequest(
                data=[_get_dense_vec(query)],
                anns_field="dense_vector",
                param=_dense_params,
                limit=top_k * 2,
                expr=filter_expr or None,
            ),
            AnnSearchRequest(
                data=[_get_sparse_vec(query)],
                anns_field="sparse_vector",
                param={"metric_type": "IP", "params": {"drop_ratio_search": 0.2}},
                limit=top_k * 2,
                expr=filter_expr or None,
            ),
        ],
        ranker=WeightedRanker(dense_weight, sparse_weight),
        limit=top_k,
        output_fields=_OUTPUT_FIELDS,
    )

    return _client_results_to_dicts(results)


def hybrid_search_rrf(
    query: str,
    top_k: int                      = _DEFAULT_TOP_K,
    rrf_k: int                      = 60,
    brand: str | None               = None,
    device_category: str | None     = None,
    price_min: float | None         = None,
    price_max: float | None         = None,
    ram: str | None                 = None,
    storage: str | None             = None,
) -> list[dict]:
    """
    Reciprocal Rank Fusion (RRF) hybrid search — no weight tuning needed.

    Runs two ANN sub-searches (dense + sparse) against INDEX_NAME and
    re-ranks via RRF, which is robust to score-scale differences between
    the two vector spaces.

    Args:
        query           : Search query string.
        top_k           : Number of results to return (default 5).
        rrf_k           : RRF smoothing constant (default 60).
        brand           : Optional brand filter.
        device_category : Optional category filter.
        price_min       : Optional minimum price filter.
        price_max       : Optional maximum price filter.
        ram             : Optional RAM substring filter matched against the `ram`
                          field (case-insensitive contains). Extract from the user
                          query, e.g. "16GB", "32GB", "8GB", "12GB".
        storage         : Optional storage substring filter matched against the
                          `storage` field (case-insensitive contains). Extract from
                          the user query, e.g. "512GB", "1TB", "256GB", "2TB".

    Returns:
        List of result dicts, each containing score + all OUTPUT_FIELDS.
    """
    # ── index_name constant (set at module level after benchmarking) ──
    index_name = INDEX_NAME

    if not _milvus_ok:
        return _not_ready_error("hybrid_search_rrf")

    from pymilvus import AnnSearchRequest, RRFRanker

    filter_expr = _build_filter(brand, device_category, price_min, price_max, ram, storage)

    results = _client.hybrid_search(
        collection_name=INDEX_NAME,
        reqs=[
            AnnSearchRequest(
                data=[_get_dense_vec(query)],
                anns_field="dense_vector",
                param=_dense_params,
                limit=top_k * 2,
                expr=filter_expr or None,
            ),
            AnnSearchRequest(
                data=[_get_sparse_vec(query)],
                anns_field="sparse_vector",
                param={"metric_type": "IP", "params": {"drop_ratio_search": 0.2}},
                limit=top_k * 2,
                expr=filter_expr or None,
            ),
        ],
        ranker=RRFRanker(k=rrf_k),
        limit=top_k,
        output_fields=_OUTPUT_FIELDS,
    )

    return _client_results_to_dicts(results)


# ============================================================
# UNIFIED TOOL REGISTRY
# ============================================================

def get_available_tools() -> list[dict]:
    """
    Return the complete list of tool schemas for all three search tools.
    Pass the output directly to your LLM client's `tools` parameter.
    """
    _filter_props = {
        "brand": {
            "type": "string",
            "description": (
                "Optional brand name to filter results, e.g. 'Samsung', 'Apple', 'Asus', 'HP', 'Dell' etc. "
                "Case-sensitive; must match stored brand value exactly."
            ),
        },
        "device_category": {
            "type": "string",
            "description": (
                "Optional device category filter, all unique device categories are:" 
                "'Laptop', 'Smartphone', 'Feature Phone', 'Mobile Phone', 'Premium Ultrabook', 'Gaming Laptop' ."
            ),
        },
        "price_min": {
            "type": "number",
            "description": "Optional minimum price (inclusive) in BDT.",
        },
        "price_max": {
            "type": "number",
            "description": "Optional maximum price (inclusive) in BDT.",
        },
        "ram": {
            "type": "string",
            "description": (
                "Optional RAM substring filter matched against the `ram` field. "
                "Extract the size mentioned in the user query and pass it as-is. "
                "Examples: '8GB', '16GB', '32GB', '12GB', '6GB'."
            ),
        },
        "storage": {
            "type": "string",
            "description": (
                "Optional storage substring filter matched against the `storage` field. "
                "Extract the size mentioned in the user query and pass it as-is. "
                "Examples: '256GB', '512GB', '1TB', '2TB', '128GB', '1TB HDD', '256GB SSD', etc."
            ),
        },
    }

    return [

        # ── dense_search ──────────────────────────────────────
        {
            "type": "function",
            "function": {
                "name": "dense_search",
                "description": (
                    f"Pure semantic (dense) vector search over the StarTech product catalogue "
                    f"using the '{INDEX_NAME}' Milvus index. "
                    "Best for natural-language queries where meaning matters more than exact words: "
                    "'affordable laptop for students', 'phone with great camera for night shots'. "
                    "Optionally filter by brand, category, or price range."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Natural-language product search query.",
                        },
                        "top_k": {
                            "type": "integer",
                            "description": "Number of results to return. Default: 5. Default value is recommended for use. Should not exceed 10.",
                            "default": 5,
                        },
                        **_filter_props,
                    },
                    "required": ["query"],
                },
            },
        },

        # ── hybrid_search ─────────────────────────────────────
        {
            "type": "function",
            "function": {
                "name": "hybrid_search",
                "description": (
                    f"Weighted hybrid search (dense + BM25 sparse) over the StarTech catalogue "
                    f"using the '{INDEX_NAME}' Milvus index and WeightedRanker fusion. "
                    "Best when the query mixes semantic intent with specific brand or spec keywords: "
                    "'Asus ROG gaming laptop RTX 4060', 'Samsung Galaxy 108MP camera phone'. "
                    "Tune dense_weight / sparse_weight to shift emphasis between meaning and keywords."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Product search query (can include brand/spec keywords).",
                        },
                        "top_k": {
                            "type": "integer",
                            "description": "Number of results to return. Default: 5. Default value is recommended for use. Should not exceed 10.",
                            "default": 5,
                        },
                        "dense_weight": {
                            "type": "number",
                            "description": (
                                "Weight for the semantic (dense) sub-search. "
                                "Range 0.0–1.0. Default: 0.7."
                            ),
                            "default": 0.7,
                        },
                        "sparse_weight": {
                            "type": "number",
                            "description": (
                                "Weight for the BM25 (keyword) sub-search. "
                                "Range 0.0–1.0. Default: 0.3."
                            ),
                            "default": 0.3,
                        },
                        **_filter_props,
                    },
                    "required": ["query"],
                },
            },
        },

        # ── hybrid_search_rrf ─────────────────────────────────
        {
            "type": "function",
            "function": {
                "name": "hybrid_search_rrf",
                "description": (
                    f"Reciprocal Rank Fusion (RRF) hybrid search (dense + BM25 sparse) "
                    f"over the StarTech catalogue using the '{INDEX_NAME}' Milvus index. "
                    "Like hybrid_search but uses rank-based fusion instead of score weights — "
                    "more robust when dense and sparse score scales differ. "
                    "Recommended as the default hybrid method when weight tuning is not needed."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Product search query.",
                        },
                        "top_k": {
                            "type": "integer",
                            "description": "Number of results to return. Default: 5. Default value is recommended for use. Should not exceed 10.",
                            "default": 5,
                        },
                        "rrf_k": {
                            "type": "integer",
                            "description": (
                                "RRF smoothing constant. Higher values reduce the influence "
                                "of very high-ranked results. Default: 60. Default value is recommended for use."
                            ),
                            "default": 60,
                        },
                        **_filter_props,
                    },
                    "required": ["query"],
                },
            },
        },

    ]


# ── Master dispatch map ───────────────────────────────────────

_TOOLS_MAP = {
    "dense_search":      dense_search,
    "hybrid_search":     hybrid_search,
    "hybrid_search_rrf": hybrid_search_rrf,
}


def handle_tool_call(tool_call) -> any:
    """
    Dispatch a tool_call object (as returned by Groq / aisuite / OpenAI)
    to the correct Python function and return the raw result.

    Raises:
        KeyError   — if tool_call.function.name is not registered.
        Any exception the underlying search function raises.
    """
    function_name = tool_call.function.name
    arguments     = json.loads(tool_call.function.arguments)

    if function_name not in _TOOLS_MAP:
        raise KeyError(
            f"Unknown tool '{function_name}'. "
            f"Available: {list(_TOOLS_MAP.keys())}"
        )

    return _TOOLS_MAP[function_name](**arguments)


def create_tool_response_message(tool_call, tool_result) -> dict:
    """
    Build the 'tool' role message dict required by the Groq / OpenAI API
    to pass a tool result back into the conversation.
    """
    return {
        "role":         "tool",
        "tool_call_id": tool_call.id,
        "name":         tool_call.function.name,
        "content":      json.dumps(tool_result, default=str),
    }