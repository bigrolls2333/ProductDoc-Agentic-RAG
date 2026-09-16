"""Local cross-encoder reranking utilities."""

from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from app.conf.reranker_config import reranker_config

_reranker_model = None


class TransformersReranker:
    """Score query-document pairs with a Transformers sequence classifier."""

    def __init__(self, model_name_or_path: str, device: str = "cpu", use_fp16: bool = False):
        self.device = device or "cpu"
        dtype = torch.float16 if use_fp16 and self.device.startswith("cuda") else torch.float32
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name_or_path,
            dtype=dtype,
        )
        self.model.to(self.device)
        self.model.eval()

    def compute_score(self, sentence_pairs, batch_size: int = 8, max_length: int = 512):
        """Return model scores for sentence pairs in bounded batches."""
        scores = []
        with torch.no_grad():
            for start in range(0, len(sentence_pairs), batch_size):
                batch = sentence_pairs[start:start + batch_size]
                inputs = self.tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                    return_tensors="pt",
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                logits = self.model(**inputs, return_dict=True).logits.view(-1)
                scores.extend(logits.float().cpu().tolist())
        return scores


def _resolve_reranker_path(raw_path: str) -> str:
    """Resolve a local model path, including a cached Hugging Face snapshot."""
    if raw_path and Path(raw_path).exists():
        return raw_path

    if raw_path == "BAAI/bge-reranker-large":
        cache_root = Path.home() / ".cache" / "huggingface" / "hub" / "models--BAAI--bge-reranker-large"
        ref_path = cache_root / "refs" / "main"
        if ref_path.exists():
            snapshot_id = ref_path.read_text(encoding="utf-8").strip()
            snapshot_path = cache_root / "snapshots" / snapshot_id
            if snapshot_path.exists():
                return str(snapshot_path)

    return raw_path

def get_reranker_model():
    """Return the process-local reranker instance."""
    global _reranker_model
    if _reranker_model is None:
        _reranker_model = TransformersReranker(
            model_name_or_path=_resolve_reranker_path(reranker_config.bge_reranker_large),
            device=reranker_config.bge_reranker_device,
            use_fp16=reranker_config.bge_reranker_fp16
        )
    return _reranker_model
