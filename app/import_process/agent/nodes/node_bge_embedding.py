"""Batch dense-and-sparse embedding generation for imported chunks."""

import sys
import os
from typing import Any, List, Dict

from app.import_process.agent.state import ImportGraphState
from app.lm.embedding_utils import get_bge_m3_ef, generate_embeddings
from app.utils.task_utils import add_running_task,add_done_task
from app.core.logger import logger

def node_bge_embedding(state: ImportGraphState) -> ImportGraphState:
    """Generate embeddings for imported chunks and attach them to import state."""
    current_node = sys._getframe().f_code.co_name
    logger.info(f"{current_node} started")

    add_running_task(state.get("task_id", ""), current_node)
    logger.info("Embedding generation started")

    try:
        texts_to_embed = step_1_validate_input(state)
        bge_m3_ef = step_2_init_model()
        output_data = step_3_generate_embeddings(texts_to_embed, bge_m3_ef)
        state['chunks'] = output_data
        logger.info(f"Embedding generation completed: chunks={len(output_data)}")
        add_done_task(state.get("task_id", ""), current_node)
    except Exception as e:
        logger.error(f"Embedding generation failed: {e}", exc_info=True)

    return state

def step_1_validate_input(state: ImportGraphState) -> List[Dict[str, Any]]:
    """Validate and return the chunks that require embedding generation."""
    texts_to_embed = state.get("chunks")
    if not isinstance(texts_to_embed, list) or not texts_to_embed:
        logger.error("Embedding input validation failed: chunks is empty or invalid")
        raise ValueError("No valid chunks available for embedding")

    logger.info(f"Embedding input validated: chunks={len(texts_to_embed)}")
    return texts_to_embed

def step_2_init_model():
    """Load the shared BGE-M3 embedding function."""
    try:
        ef = get_bge_m3_ef()
        if ef is None:
            raise ValueError("BGE-M3 embedding function is unavailable")

        logger.info("BGE-M3 embedding function loaded")
        return ef
    except Exception as e:
        error_msg = f"BGE-M3 embedding initialization failed: {e}"
        logger.error(error_msg)
        raise ValueError(error_msg)

def step_3_generate_embeddings(texts_to_embed: List[Dict[str, Any]], bge_m3_ef: Any) -> List[Dict[str, Any]]:
    """Generate and attach dense and sparse vectors in bounded batches."""
    output_data = []
    batch_size = 5

    total = len(texts_to_embed)
    for i in range(0, total, batch_size):
        batch_texts = texts_to_embed[i:i + batch_size]
        start_idx, end_idx = i + 1, min(i + len(batch_texts), total)

        try:
            input_texts = []
            for doc in batch_texts:
                canonical_name = doc.get("canonical_name") or doc.get("item_name", "")
                content = doc["content"]
                text = f"商品：{canonical_name}，介绍：{content}" if canonical_name else content
                input_texts.append(text)

            docs_embeddings = generate_embeddings(input_texts)
            if not docs_embeddings:
                logger.warning(f"Embedding batch returned no vectors: range={start_idx}-{end_idx}")
                output_data.extend(batch_texts)
                continue

            for j, doc in enumerate(batch_texts):
                item = doc.copy()
                item["dense_vector"] = docs_embeddings["dense"][j]
                item["sparse_vector"] = docs_embeddings["sparse"][j]
                output_data.append(item)

            logger.info(f"Embedding batch completed: range={start_idx}-{end_idx}")

        except Exception as e:
            logger.error(
                f"Embedding batch failed; preserving input chunks: range={start_idx}-{end_idx}, error={e}",
                exc_info=True
            )
            output_data.extend(batch_texts)
            continue

    return output_data
