"""Targeted correctness checks for benchmark denominators and failure boundaries."""

import copy
import tempfile
import unittest
from pathlib import Path

from app.evaluation.dataset import load_benchmark, write_jsonl
from app.evaluation.diagnosis import diagnose
from app.evaluation.metrics import retrieval_metrics, evidence_metrics
from app.evaluation.runner import run_experiment
from app.evaluation.workflow import apply_reviews, collect_cases, prepare_legacy, build_pool


EXAMPLE = Path(__file__).resolve().parents[1] / "eval" / "examples" / "retrieval"


class MetricTests(unittest.TestCase):
    def test_fact_alternatives_are_or_and_joint_evidence_is_and(self):
        facts = [{'fact_id': 'one', 'evidence_ids': ['a', 'b'], 'evidence_sets': [['a'], ['b']]},
                 {'fact_id': 'two', 'evidence_ids': ['a', 'c']}]
        judgments = {'x': {'evidence_ids': ['b']}, 'y': {'evidence_ids': ['a', 'c']}}
        partial = evidence_metrics(['x'], ['a', 'b', 'c'], judgments, facts)
        self.assertEqual(partial['fact_recall'], .5)
        self.assertEqual(partial['covered_fact_ids'], ['one'])
        full = evidence_metrics(['y'], ['a', 'b', 'c'], judgments, facts)
        self.assertEqual(full['all_facts'], 1)
        self.assertLess(full['evidence_recall'], 1)

    def test_missing_gold_lowers_ndcg_even_when_observed_order_is_ideal(self):
        scores = retrieval_metrics(["a"], {"a": 2, "b": 2}, (1, 2))
        self.assertEqual(scores["recall@2"], .5)
        self.assertLess(scores["ndcg@2"], 1)
        self.assertEqual(scores["precision@2"], .5)

    def test_duplicates_do_not_inflate_hits_and_unjudged_remain_visible(self):
        scores = retrieval_metrics(["unknown", "a", "a"], {"a": 2, "b": 1}, (3,))
        self.assertEqual(scores["recall@3"], .5)
        self.assertEqual(scores["mrr@3"], .5)
        self.assertEqual(scores["unjudged@3"], 1)
        self.assertEqual(scores["judged_fraction@3"], .5)

    def test_no_answer_has_no_ir_target(self):
        self.assertIsNone(retrieval_metrics(["a"], {"a": 0}, (1,))["recall@1"])


class DiagnosisTests(unittest.TestCase):
    def test_distinguishes_filter_retrieval_selection_and_budget_loss(self):
        query = {"answerable": True, "evidence_ids": ["a", "b", "c", "d"]}
        judgments = {key: {"evidence_ids": [key]} for key in "abcd"}
        stages = {"eligible": list("abc"), "candidates": list("ab"), "reranked": ["a"], "context": []}
        self.assertEqual([d["type"] for d in diagnose(query, stages, judgments)],
                         ["product_filter_loss", "candidate_retrieval_miss", "ranking_or_selection_loss", "context_budget_loss"])

    def test_missing_intermediate_stage_is_not_invented(self):
        result = diagnose({"answerable": True, "evidence_ids": ["e"]}, {"context": []}, {})
        self.assertEqual(result[0]["type"], "no_observed_evidence_loss")


class DatasetAndExperimentTests(unittest.TestCase):
    def test_real_pipeline_retains_no_answer_separately(self):
        benchmark = load_benchmark(EXAMPLE)
        report = run_experiment(benchmark, ["bm25"], candidate_k=3, context_k=1, ks=(1, 3))
        summary = report["summary"]["bm25"]
        self.assertEqual(summary["completed"], 2)
        self.assertEqual(summary["metrics"]["candidates"]["recall@3"]["n"], 1)
        self.assertEqual(summary["metrics"]["candidates"]["recall@3"]["mean"], 1)
        self.assertEqual(summary["metrics"]["context_evidence"]["evidence_recall"]["mean"], .5)
        with tempfile.TemporaryDirectory() as directory:
            queue = Path(directory) / "cases.jsonl"
            self.assertEqual(collect_cases(report, queue)["added"], 1)
            self.assertEqual(collect_cases(report, queue)["added"], 0)

    def test_bad_query_is_execution_error_not_quality_failure(self):
        benchmark = copy.deepcopy(load_benchmark(EXAMPLE))
        benchmark.queries["DEMO-001"]["product_ids"] = []
        benchmark.queries["DEMO-001"]["canonical_names"] = []
        report = run_experiment(benchmark, ["bm25"], candidate_k=3, context_k=1)
        self.assertEqual(report["summary"]["bm25"]["errors"], 1)
        self.assertEqual(report["results"][0]["diagnosis"][0]["type"], "execution_error")

    def test_group_leakage_and_unmapped_gold_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for source in EXAMPLE.iterdir():
                (root / source.name).write_bytes(source.read_bytes())
            rows = list(load_benchmark(root).queries.values())
            rows[1]["split"] = "test"
            write_jsonl(root / "queries.jsonl", rows)
            with self.assertRaisesRegex(ValueError, "crosses splits"):
                load_benchmark(root)
            rows[1]["split"] = "dev"
            rows[0]["evidence_ids"].append("nonexistent")
            write_jsonl(root / "queries.jsonl", rows)
            with self.assertRaisesRegex(ValueError, "unknown evidence"):
                load_benchmark(root)

    def test_migration_does_not_infer_gold_or_overwrite_data(self):
        with tempfile.TemporaryDirectory() as directory:
            source, output = Path(directory) / "legacy.jsonl", Path(directory) / "draft"
            write_jsonl(source, [{"id": "q", "query": "question", "expected": {"source_hint": {"chunk_keywords": ["word"]}}}])
            prepare_legacy(source, output)
            benchmark = load_benchmark(output)
            self.assertEqual(benchmark.select(), [])
            self.assertIsNone(benchmark.queries["q"]["answerable"])
            with self.assertRaisesRegex(ValueError, "overwrite"):
                prepare_legacy(source, output)

    def test_test_split_cannot_be_collected(self):
        with self.assertRaisesRegex(ValueError, "test split"):
            collect_cases({"config": {"split": "test"}}, "unused.jsonl")

    def test_annotation_pool_can_bootstrap_pending_queries_without_scores(self):
        benchmark = copy.deepcopy(load_benchmark(EXAMPLE))
        for query in benchmark.queries.values():
            query["review_status"] = "pending"
        report = run_experiment(benchmark, ["bm25"], annotation_only=True, candidate_k=3, context_k=1)
        self.assertEqual(report["summary"]["bm25"]["metrics"]["candidates"], {})
        self.assertGreater(len(build_pool(benchmark, [report])), 0)
        with self.assertRaisesRegex(ValueError, "annotation-only"):
            collect_cases(report, "unused.jsonl")

    def test_review_publishes_new_version_and_keeps_original(self):
        from app.evaluation.dataset import read_jsonl
        benchmark = load_benchmark(EXAMPLE)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "v2"
            result = apply_reviews(benchmark, read_jsonl(EXAMPLE.parent / "reviews.jsonl"), output, "2.0")
            self.assertEqual(result["reviewed"], 1)
            self.assertEqual(load_benchmark(output).manifest["version"], "2.0")
            self.assertEqual(load_benchmark(EXAMPLE).manifest["version"], "1.0")


if __name__ == "__main__":
    unittest.main()
