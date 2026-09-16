"""Extract product metadata from imported document evidence."""

import json
import hashlib
import os
import re
import sys
from typing import List, Dict, Any, Tuple

from pymilvus import DataType
from langchain_core.messages import SystemMessage, HumanMessage

from app.import_process.agent.state import ImportGraphState
from app.clients.milvus_utils import get_milvus_client
from app.lm.lm_utils import get_llm_client
from app.lm.embedding_utils import generate_embeddings
from app.utils.normalize_sparse_vector import normalize_sparse_vector
from app.utils.task_utils import add_running_task
from app.core.logger import logger
from app.core.load_prompt import load_prompt

# Extraction defaults.
DEFAULT_ITEM_NAME_CHUNK_K = 5
SINGLE_CHUNK_CONTENT_MAX_LEN = 800
CONTEXT_TOTAL_MAX_CHARS = 2500
DOC_TITLE_SUFFIXES = [
    "用户手册",
    "使用说明书",
    "说明书",
    "安装指南",
    "快速入门",
    "产品手册",
    "维护手册",
    "配置指南",
    "产品说明书",
    "技术白皮书",
]
DOC_TITLE_ARTIFACT_PATTERNS = [
    r"[-_ ]*整本手册$",
    r"[-_ ]*\d+W\d+$",
    r"[-_ ]*R\d{3,5}$",
    r"[-_ ]*\(\d{2,}.*?\)$",
]
CATEGORY_KEYWORDS = [
    "室内无线网关",
    "无线网关",
    "无线控制器",
    "控制器",
    "万用表",
    "交换机",
    "路由器",
    "摄像机",
    "烫金机",
    "服务器",
]
MODEL_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9-]{3,}")
def _normalize_name_key(value: str) -> str:
    cleaned = _clean_product_source_text(value)
    compact = re.sub(r"\s+", "", cleaned)
    return compact.upper()


def _ordered_unique(values: List[str]) -> List[str]:
    seen = set()
    ordered = []
    for value in values:
        clean_value = (value or "").strip()
        if not clean_value or clean_value in seen:
            continue
        seen.add(clean_value)
        ordered.append(clean_value)
    return ordered


def _clean_product_source_text(text: str) -> str:
    value = (text or "").strip()
    value = re.sub(r"\.(pdf|docx?|md|txt|xlsx?)$", "", value, flags=re.IGNORECASE)
    for suffix in DOC_TITLE_SUFFIXES:
        value = re.sub(re.escape(suffix) + r"$", "", value)
    for _ in range(3):
        previous = value
        for pattern in DOC_TITLE_ARTIFACT_PATTERNS:
            value = re.sub(pattern, "", value, flags=re.IGNORECASE).strip()
        value = value.strip(" _-")
        if value == previous:
            break
    return value.strip(" _-")


def _extract_model_code(*texts: str) -> str:
    for text in texts:
        candidates: List[str] = []
        for token in MODEL_TOKEN_PATTERN.findall(text or ""):
            if any(ch.isalpha() for ch in token) and any(ch.isdigit() for ch in token):
                candidates.append(token.upper())
        if candidates:
            candidates.sort(key=len, reverse=True)
            return candidates[0]
    return ""


def _extract_brand(*texts: str, model_code: str = "") -> str:
    for text in texts:
        cleaned = _clean_product_source_text(text)
        if not cleaned:
            continue
        if model_code and model_code in cleaned:
            prefix = cleaned.split(model_code, 1)[0].strip(" _-")
            if 1 < len(prefix) <= 12:
                return prefix
    return ""


def _extract_category(*texts: str, brand: str = "", model_code: str = "") -> str:
    for text in texts:
        cleaned = _clean_product_source_text(text)
        if not cleaned:
            continue
        candidate = cleaned
        if brand and candidate.startswith(brand):
            candidate = candidate[len(brand):].strip(" _-")
        if model_code and model_code in candidate:
            candidate = candidate.split(model_code, 1)[-1].strip(" _-")
        for keyword in CATEGORY_KEYWORDS:
            if keyword and keyword in candidate:
                return keyword
        if 1 < len(candidate) <= 20:
            return candidate
    return ""


def _build_product_id(canonical_name: str) -> str:
    normalized = _normalize_name_key(canonical_name).lower()
    digest = hashlib.md5(normalized.encode("utf-8")).hexdigest()[:12]
    return f"prod_{digest}"


def _build_alias_row_id(product_id: str, alias_name: str) -> str:
    base = f"{product_id}:{_normalize_name_key(alias_name)}".lower()
    digest = hashlib.md5(base.encode("utf-8")).hexdigest()[:16]
    return f"alias_{digest}"


def _parse_llm_json(content: str) -> Dict[str, Any]:
    raw = (content or "").strip()
    if raw.startswith("```json"):
        raw = raw.replace("```json", "", 1).replace("```", "").strip()
    elif raw.startswith("```"):
        raw = raw.replace("```", "").strip()
    return json.loads(raw)


def _normalize_aliases(values: Any) -> List[str]:
    if not isinstance(values, list):
        return []
    return _ordered_unique([
        normalized
        for value in values
        if (normalized := _normalize_name_key(str(value or "")))
    ])


def _build_product_profile(file_title: str, llm_result: Any) -> Dict[str, Any]:
    cleaned_title = _clean_product_source_text(file_title)
    if isinstance(llm_result, dict):
        raw_canonical = (
            llm_result.get("canonical_name")
            or llm_result.get("item_name")
            or llm_result.get("product_name")
            or ""
        )
        llm_brand = _clean_product_source_text(str(llm_result.get("brand", "")))
        llm_model_code = str(llm_result.get("model_code", "")).strip().upper()
        llm_category = _clean_product_source_text(str(llm_result.get("category", "")))
        llm_aliases = _normalize_aliases(llm_result.get("aliases", []))
        confidence = llm_result.get("confidence")
        evidence = str(llm_result.get("evidence", "")).strip()
    else:
        raw_canonical = str(llm_result or "")
        llm_brand = ""
        llm_model_code = ""
        llm_category = ""
        llm_aliases = []
        confidence = None
        evidence = ""

    canonical_name = _normalize_name_key(raw_canonical) or _normalize_name_key(cleaned_title)
    model_code = _extract_model_code(cleaned_title, canonical_name)
    if llm_model_code:
        model_code = llm_model_code
    brand = _extract_brand(cleaned_title, canonical_name, model_code=model_code)
    if llm_brand:
        brand = llm_brand
    category = _extract_category(cleaned_title, canonical_name, brand=brand, model_code=model_code)
    if llm_category:
        category = llm_category
    assembled_name = ""
    if brand and model_code:
        assembled_name = f"{brand} {model_code}"
    elif model_code:
        assembled_name = model_code
    if category:
        assembled_name = f"{assembled_name}{category}" if assembled_name else category
    if assembled_name:
        canonical_name = _normalize_name_key(assembled_name)

    alias_pairs = [
        (canonical_name, "exact_full_name"),
        (cleaned_title, "title_variant"),
    ]
    for alias_name in llm_aliases:
        alias_pairs.append((alias_name, "llm_alias"))
    if model_code:
        alias_pairs.append((model_code, "model_only"))
    if brand and model_code:
        alias_pairs.append((f"{brand} {model_code}", "brand_plus_model"))
        alias_pairs.append((f"{brand}{model_code}", "brand_plus_model"))
    if model_code and category:
        alias_pairs.append((f"{model_code}{category}", "model_plus_category"))
        alias_pairs.append((f"{model_code} {category}", "model_plus_category"))

    alias_entries = []
    seen_aliases = set()
    product_id = _build_product_id(canonical_name)
    for alias_name, alias_type in alias_pairs:
        normalized_alias = _normalize_name_key(alias_name)
        if not normalized_alias or normalized_alias in seen_aliases:
            continue
        seen_aliases.add(normalized_alias)
        alias_entries.append({
            "product_id": product_id,
            "canonical_name": canonical_name,
            "alias_name": normalized_alias,
            "item_name": normalized_alias,
            "alias_type": alias_type,
            "brand": brand,
            "category": category,
            "model_code": model_code,
        })

    profile = {
        "product_id": product_id,
        "canonical_name": canonical_name,
        "brand": brand,
        "category": category,
        "model_code": model_code,
        "aliases": [entry["alias_name"] for entry in alias_entries],
        "alias_entries": alias_entries,
        "source_file_title": file_title,
        "confidence": confidence,
        "evidence": evidence,
    }
    logger.info(
        f"Product profile generated: canonical={canonical_name}, "
        f"brand={brand or '-'}, category={category or '-'}, model={model_code or '-'}, "
        f"aliases={profile['aliases']}"
    )
    return profile


def step_1_get_inputs(state: ImportGraphState) -> Tuple[str, List[Dict]]:
    """Load the source title and chunk list required for metadata extraction."""
    file_title = state.get("file_title", "") or state.get("file_name", "")
    chunks = state.get("chunks") or []

    if not file_title:
        if chunks and isinstance(chunks[0], dict):
            file_title = chunks[0].get("file_title", "")
            logger.warning("Source title recovered from the first chunk")

    if not file_title:
        logger.warning("Source title is unavailable; metadata extraction may be less precise")

    if not isinstance(chunks, list) or not chunks:
        logger.warning("Product metadata extraction skipped because chunks are empty or invalid")
        return file_title, []

    logger.info(f"Product metadata inputs validated: chunks={len(chunks)}")
    return file_title, chunks


def step_2_build_context(chunks: List[Dict], k: int = DEFAULT_ITEM_NAME_CHUNK_K, max_chars: int = CONTEXT_TOTAL_MAX_CHARS) -> str:
    """Build a bounded, structured context for product metadata extraction."""
    if not chunks:
        return ""

    parts: List[str] = []
    total_chars = 0

    for idx, chunk in enumerate(chunks[:k]):
        if not isinstance(chunk, dict):
            logger.debug(f"Chunk skipped because its shape is invalid: index={idx + 1}")
            continue

        chunk_title = chunk.get("title", "").strip()
        chunk_content = chunk.get("content", "").strip()

        if not (chunk_title or chunk_content):
            logger.debug(f"Chunk skipped because it is empty: index={idx + 1}")
            continue

        if len(chunk_content) > SINGLE_CHUNK_CONTENT_MAX_LEN:
            chunk_content = chunk_content[:SINGLE_CHUNK_CONTENT_MAX_LEN]
            logger.debug(f"Chunk content truncated: index={idx + 1}, limit={SINGLE_CHUNK_CONTENT_MAX_LEN}")

        piece = f"【切片{idx + 1}】\n标题：{chunk_title} \n内容：{chunk_content}"
        parts.append(piece)
        total_chars += len(piece)

        if total_chars > max_chars:
            logger.info(f"Metadata context limit reached: max_chars={max_chars}")
            break

    context = "\n\n".join(parts).strip()
    final_context = context[:max_chars]
    logger.info(f"Metadata context built: length={len(final_context)}")
    return final_context


def step_3_call_llm(file_title: str, context: str) -> Dict[str, Any]:
    """Extract a structured product profile from the source title and context."""
    logger.info("Product metadata extraction started")

    if not context:
        logger.warning("Metadata context is empty; using the source title as fallback")
        return {"canonical_name": file_title, "aliases": [], "confidence": 0, "evidence": "empty_context"}

    try:
        human_prompt = load_prompt("item_name_recognition", file_title=file_title, context=context)
        system_prompt = load_prompt("product_recognition_system")
        logger.debug(
            f"Metadata prompts assembled: system_length={len(system_prompt)}, "
            f"user_length={len(human_prompt)}"
        )

        llm = get_llm_client(json_mode=True)
        if not llm:
            logger.error("LLM client is unavailable; using the source title as fallback")
            return {"canonical_name": file_title, "aliases": [], "confidence": 0, "evidence": "llm_client_unavailable"}

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_prompt)
        ]
        resp = llm.invoke(messages)

        content = getattr(resp, "content", "").strip()
        result = _parse_llm_json(content)

        canonical_name = _normalize_name_key(str(result.get("canonical_name", "")))
        if not canonical_name:
            logger.warning("LLM response has no canonical_name; using the source title")
            result["canonical_name"] = file_title
        else:
            result["canonical_name"] = canonical_name

        if "aliases" not in result:
            result["aliases"] = []
        if "confidence" not in result:
            result["confidence"] = None

        logger.info(f"Product metadata extracted: canonical_name={result.get('canonical_name')}")
        return result

    except Exception as e:
        logger.error(f"Product metadata extraction failed: {e}", exc_info=True)
        return {"canonical_name": file_title, "aliases": [], "confidence": 0, "evidence": f"fallback: {str(e)}"}

def step_4_update_chunks(state: ImportGraphState, chunks: List[Dict], product_profile: Dict[str, Any]):
    """Write the normalized product profile to state and every chunk."""
    canonical_name = _normalize_name_key(product_profile.get("canonical_name", ""))
    item_name = canonical_name
    state["item_name"] = canonical_name
    product_profile["canonical_name"] = canonical_name
    state["product_profile"] = product_profile
    state["item_aliases"] = product_profile.get("aliases", [])
    for chunk in chunks:
        chunk["item_name"] = canonical_name
        chunk["canonical_name"] = canonical_name
        chunk["product_id"] = product_profile.get("product_id", "")
        chunk["brand"] = product_profile.get("brand", "")
        chunk["category"] = product_profile.get("category", "")
        chunk["model_code"] = product_profile.get("model_code", "")
        chunk["doc_id"] = state.get("doc_id", "")
    state["chunks"] = chunks
    logger.info(f"Product profile written to chunks: canonical_name={item_name}, chunks={len(chunks)}")

def _create_item_name_collection(client, collection_name: str):
    schema = client.create_schema(auto_id=False, enable_dynamic_field=True)
    schema.add_field(field_name="pk", datatype=DataType.VARCHAR, max_length=255, is_primary=True, auto_id=False)
    schema.add_field(field_name="file_title", datatype=DataType.VARCHAR, max_length=65535)
    schema.add_field(field_name="item_name", datatype=DataType.VARCHAR, max_length=65535)
    schema.add_field(field_name="alias_name", datatype=DataType.VARCHAR, max_length=65535)
    schema.add_field(field_name="canonical_name", datatype=DataType.VARCHAR, max_length=65535)
    schema.add_field(field_name="product_id", datatype=DataType.VARCHAR, max_length=255)
    schema.add_field(field_name="brand", datatype=DataType.VARCHAR, max_length=255)
    schema.add_field(field_name="category", datatype=DataType.VARCHAR, max_length=255)
    schema.add_field(field_name="model_code", datatype=DataType.VARCHAR, max_length=255)
    schema.add_field(field_name="alias_type", datatype=DataType.VARCHAR, max_length=255)
    schema.add_field(field_name="dense_vector", datatype=DataType.FLOAT_VECTOR, dim=1024)
    schema.add_field(field_name="sparse_vector", datatype=DataType.SPARSE_FLOAT_VECTOR)

    index_params = client.prepare_index_params()
    index_params.add_index(
        field_name="dense_vector",
        index_name="dense_vector_index",
        index_type="HNSW",
        metric_type="COSINE",
        params={"M": 16, "efConstruction": 200}
    )
    index_params.add_index(
        field_name="sparse_vector",
        index_name="sparse_vector_index",
        index_type="SPARSE_INVERTED_INDEX",
        metric_type="IP",
        params={"inverted_index_algo": "DAAT_MAXSCORE", "normalize": True, "quantization": "none"}
    )
    client.create_collection(collection_name=collection_name, schema=schema, index_params=index_params)


def _ensure_item_name_collection_ready(client, collection_name: str):
    if not client.has_collection(collection_name=collection_name):
        logger.info(f"Product alias collection not found; creating: {collection_name}")
        _create_item_name_collection(client, collection_name)
        logger.info(f"Product alias collection created: {collection_name}")
        return

    logger.info(f"Using existing product alias collection: {collection_name}")


def _write_item_name_rows(client, collection_name: str, rows: List[Dict[str, Any]]):
    if hasattr(client, "upsert"):
        return client.upsert(collection_name=collection_name, data=rows)
    return client.insert(collection_name=collection_name, data=rows)


def step_6_generate_alias_vectors(alias_entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Generate dense and sparse vectors for the normalized product aliases."""
    alias_texts = [
        entry.get("alias_name") or entry.get("item_name", "")
        for entry in alias_entries
        if entry.get("alias_name") or entry.get("item_name")
    ]
    logger.info(f"Alias embedding generation started: aliases={len(alias_texts)}")

    if not alias_texts:
        logger.warning("Alias embedding generation skipped because the alias list is empty")
        return []

    try:
        vector_result = generate_embeddings(alias_texts)
        if not vector_result or "dense" not in vector_result or "sparse" not in vector_result:
            logger.warning("Alias embedding generation returned no vectors")
            return []

        vector_rows = []
        for idx, entry in enumerate(alias_entries):
            alias_name = entry.get("alias_name") or entry.get("item_name", "")
            if not alias_name:
                continue
            vector_rows.append({
                **entry,
                "alias_name": alias_name,
                "item_name": alias_name,
                "dense_vector": vector_result["dense"][idx],
                "sparse_vector": vector_result["sparse"][idx],
            })

        logger.info(f"Alias embedding generation completed: rows={len(vector_rows)}")
        return vector_rows
    except Exception as e:
        logger.error(f"Alias embedding generation failed: {e}", exc_info=True)
        return []


def step_7_save_alias_rows_to_milvus(
    state: ImportGraphState,
    file_title: str,
    product_profile: Dict[str, Any],
    vector_rows: List[Dict[str, Any]]
):
    """Persist product aliases and their vectors in the configured Milvus collection."""
    milvus_uri = os.environ.get("MILVUS_URL")
    collection_name = os.environ.get("ITEM_NAME_COLLECTION")
    if not all([milvus_uri, collection_name]):
        logger.warning("Alias persistence skipped because Milvus configuration is incomplete")
        return

    canonical_name = product_profile.get("canonical_name", "")
    logger.info(f"Alias persistence started: canonical_name={canonical_name}, collection={collection_name}")

    try:
        client = get_milvus_client()
        if not client:
            logger.error("Milvus client is unavailable; alias persistence skipped")
            return

        _ensure_item_name_collection_ready(client, collection_name)

        payloads = []
        for row in vector_rows:
            alias_name = _normalize_name_key(row.get("alias_name") or row.get("item_name", ""))
            product_id = row.get("product_id") or product_profile.get("product_id", "")
            payload = {
                "pk": _build_alias_row_id(product_id, alias_name),
                "file_title": file_title,
                "alias_name": alias_name,
                "item_name": alias_name,
                "canonical_name": _normalize_name_key(row.get("canonical_name", canonical_name)),
                "product_id": product_id,
                "brand": row.get("brand", ""),
                "category": row.get("category", ""),
                "model_code": row.get("model_code", ""),
                "alias_type": row.get("alias_type", ""),
            }
            if row.get("dense_vector") is not None:
                payload["dense_vector"] = row["dense_vector"]
            if row.get("sparse_vector") is not None:
                payload["sparse_vector"] = normalize_sparse_vector(row["sparse_vector"])
            payloads.append(payload)

        if not payloads:
            logger.warning("Alias persistence skipped because no rows are available")
            return

        _write_item_name_rows(client, collection_name, payloads)
        state["item_name"] = canonical_name
        logger.info(
            f"Alias persistence completed: canonical_name={canonical_name}, "
            f"rows={len(payloads)}, collection={collection_name}"
        )
    except Exception as e:
        logger.error(f"Alias persistence failed: {e}", exc_info=True)


def node_item_name_recognition(state: ImportGraphState) -> ImportGraphState:
    """Extract product metadata, enrich chunks, and persist alias vectors."""
    node_name = sys._getframe().f_code.co_name
    logger.info(f"{node_name} started")
    add_running_task(state.get("task_id", ""), node_name)

    try:
        file_title, chunks = step_1_get_inputs(state)
        if not chunks:
            logger.warning(f"{node_name} skipped: no valid chunks")
            return state

        context = step_2_build_context(chunks)

        product_result = step_3_call_llm(file_title, context)
        product_profile = _build_product_profile(file_title, product_result)

        step_4_update_chunks(state, chunks, product_profile)

        vector_rows = step_6_generate_alias_vectors(product_profile.get("alias_entries", []))

        step_7_save_alias_rows_to_milvus(state, file_title, product_profile, vector_rows)

        logger.info(
            f"{node_name} completed: canonical_name={product_profile.get('canonical_name')}"
        )

    except Exception as e:
        logger.error(f"{node_name} failed: {e}", exc_info=True)
        state["item_name"] = "未知商品"

    return state
