"""Check the actual prompt boundary without initializing model/database clients."""

import ast
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
import threading
import time
from concurrent.futures import ThreadPoolExecutor


ROOT = Path(__file__).resolve().parents[1]


def load_function(relative, name, namespace):
    """Compile only the pure target function to avoid import-time service calls."""
    tree = ast.parse((ROOT / relative).read_text(encoding="utf-8-sig"))
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name)
    exec(compile(ast.Module(body=[function], type_ignores=[]), relative, "exec"), namespace)
    return namespace[name]


class GenerationEvidenceTests(unittest.TestCase):
    def test_prompt_selection_receives_actual_upstream_item_names(self):
        namespace = {"QueryGraphState": dict, "MAX_CONTEXT_CHARS": 500, "hashlib": hashlib,
                     "logger": SimpleNamespace(debug=lambda *args: None),
                     "load_prompt": lambda name, **kwargs: json.dumps(kwargs, ensure_ascii=False)}
        function = load_function('app/query_process/agent/nodes/node_answer_output.py', 'step_2_construct_prompt', namespace)
        docs = [{'chunk_id':cid,'canonical_name':name,'text':'source text'} for cid,name in [('a0','A'),('a1','A'),('b0','B')]]
        state = {'original_query':'这两个怎么配置','item_names':['A','B'], 'reranked_docs':docs[:1],
                 'context_candidate_docs':docs, 'context_selection_policy':'item_balanced','context_per_item_k':2}
        function(state)
        self.assertEqual([d['chunk_id'] for d in state['answer_context_docs']], ['a0','b0','a1'])
        self.assertEqual(state['answer_context_trace']['item_names'], ['A','B'])

    def test_concurrent_first_requests_initialize_one_embedding_model(self):
        calls = []
        sentinel = object()
        def factory(**kwargs):
            calls.append(kwargs)
            time.sleep(.01)
            return sentinel
        namespace = {"_bge_m3_ef": None, "_bge_m3_init_lock": threading.Lock(), "threading": threading,
                     "embedding_config": SimpleNamespace(bge_m3_path="fake", bge_device="cpu", bge_fp16=False),
                     "BGEM3EmbeddingFunction": factory,
                     "logger": SimpleNamespace(debug=lambda *a: None, info=lambda *a, **kw: None, success=lambda *a: None)}
        load_function("app/lm/embedding_utils.py", "_load_bge_m3_ef", namespace)
        getter = load_function("app/lm/embedding_utils.py", "get_bge_m3_ef", namespace)
        with ThreadPoolExecutor(max_workers=8) as executor:
            models = list(executor.map(lambda _: getter(), range(8)))
        self.assertEqual(len(calls), 1)
        self.assertTrue(all(model is sentinel for model in models))

    def test_trace_contains_only_documents_that_reached_the_prompt(self):
        namespace = {"QueryGraphState": dict, "MAX_CONTEXT_CHARS": 200,
                     "hashlib": hashlib, "logger": SimpleNamespace(debug=lambda *args: None),
                     "load_prompt": lambda name, **kwargs: json.dumps(kwargs, ensure_ascii=False)}
        function = load_function("app/query_process/agent/nodes/node_answer_output.py", "step_2_construct_prompt", namespace)
        state = {"original_query": "电源？", "reranked_docs": [
            {"chunk_id": "a", "doc_id": "document", "text": "12V", "source": "local"},
            {"chunk_id": "b", "doc_id": "document", "text": "x" * 300, "source": "local"}]}
        prompt = function(state)
        self.assertEqual([doc["chunk_id"] for doc in state["answer_context_docs"]], ["a"])
        self.assertNotIn("x" * 300, prompt)
        self.assertEqual(state["answer_context_trace"]["omitted_count"], 1)
        self.assertEqual(state["answer_context_trace"]["prompt_sha256"], hashlib.sha256(prompt.encode()).hexdigest())

    def test_rerank_normalization_preserves_document_and_chunk_identity(self):
        namespace = {"logger": SimpleNamespace(info=lambda *a: None, debug=lambda *a: None, warning=lambda *a: None)}
        function = load_function("app/query_process/agent/nodes/node_rerank.py", "step_1_merge_docs", namespace)
        row = function({"rrf_chunks": [{"chunk_id": "chunk", "doc_id": "document", "content": "text", "product_id": "p"}]})[0]
        self.assertEqual(row["doc_id"], "document")
        self.assertEqual(row["chunk_id"], "chunk")
        self.assertEqual(row["product_id"], "p")


if __name__ == "__main__":
    unittest.main()
