"""BGE-M3 embedding utilities with process-local model reuse."""

from pymilvus.model.hybrid import BGEM3EmbeddingFunction
import threading
from app.core.logger import logger
from app.conf.embedding_config import embedding_config

# Keep one model instance per process because model loading is expensive.
_bge_m3_ef = None
_bge_m3_init_lock = threading.Lock()

def get_bge_m3_ef():
    """Initialize the shared model once, including concurrent first requests."""
    if _bge_m3_ef is not None:
        return _bge_m3_ef
    with _bge_m3_init_lock:
        return _load_bge_m3_ef()


def _load_bge_m3_ef():
    """Return the cached BGE-M3 embedding function, loading it on first use."""
    global _bge_m3_ef
    if _bge_m3_ef is not None:
        logger.debug("BGE-M3 model cache hit")
        return _bge_m3_ef

    model_name = embedding_config.bge_m3_path or "BAAI/bge-m3"
    device = embedding_config.bge_device or "cpu"
    use_fp16 = embedding_config.bge_fp16 or False

    logger.info(
        "Initializing BGE-M3 model",
        extra={
            "model_name": model_name,
            "device": device,
            "use_fp16": use_fp16,
            "normalize_embeddings": True,
            "thread_name": threading.current_thread().name
        }
    )

    try:
        _bge_m3_ef = BGEM3EmbeddingFunction(
            model_name=model_name,
            device=device,
            use_fp16=use_fp16,
            normalize_embeddings=True,
        )
        logger.success(
            f"BGE-M3 model initialized with normalized embeddings, thread={threading.current_thread().name}"
        )
        return _bge_m3_ef
    except Exception as e:
        logger.error(f"BGE-M3 model initialization failed: {e}", exc_info=True)
        raise


def generate_embeddings(texts):
    """Generate normalized dense and sparse embeddings for a batch of texts."""
    if not isinstance(texts, list) or len(texts) == 0:
        logger.warning("Embedding input must be a non-empty list")
        raise ValueError("texts must be a non-empty list")

    logger.info(
        f"Generating hybrid embeddings: count={len(texts)}, thread={threading.current_thread().name}"
    )
    try:
        model = get_bge_m3_ef()
        embeddings = model.encode_documents(texts)
        logger.debug(f"Embedding generation completed; normalizing sparse output for {len(texts)} texts")

        processed_sparse = []
        for i in range(len(texts)):
            sparse_indices = embeddings["sparse"].indices[
                embeddings["sparse"].indptr[i]:embeddings["sparse"].indptr[i + 1]
            ].tolist()
            sparse_data = embeddings["sparse"].data[
                embeddings["sparse"].indptr[i]:embeddings["sparse"].indptr[i + 1]
            ].tolist()
            sparse_dict = {k: v for k, v in zip(sparse_indices, sparse_data)}
            processed_sparse.append(sparse_dict)

        result = {
            "dense": [emb.tolist() for emb in embeddings["dense"]],
            "sparse": processed_sparse,
        }
        logger.success(f"Hybrid embeddings generated: count={len(texts)}")
        return result

    except Exception as e:
        logger.error(f"Embedding generation failed: {e}", exc_info=True)
        raise
