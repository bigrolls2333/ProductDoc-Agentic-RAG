"""Reject unsupported judge evidence and distinguish abstention from faithfulness."""

import unittest
from pathlib import Path

from app.evaluation.answer_eval import summarize_judgment
from app.evaluation.dataset import load_benchmark


class AnswerEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.benchmark = load_benchmark(Path(__file__).resolve().parents[1] / "eval/examples/retrieval")
        self.record = {"query_id": "DEMO-001", "context_source": "actual_prompt", "abstained": False,
                       "answer": "DEMO-A 使用 12V 电源。[demo-a-power]",
                       "context_docs": [{"chunk_id": "demo-a-power", "text": self.benchmark.corpus["demo-a-power"]["content"]}]}
        self.judgment = {"correct": False, "reason": "Missing network information", "claims": [
            {"text": "使用 12V 电源", "verdict": "supported", "evidence": [{"chunk_id": "demo-a-power", "quote": "使用 12V 直流电源"}],
             "citations": ["demo-a-power"], "supported_citations": ["demo-a-power"]}]}

    def test_faithful_answer_can_still_be_incomplete(self):
        result = summarize_judgment(self.benchmark, self.record, self.judgment)
        self.assertEqual(result["faithfulness"], 1)
        self.assertIs(result["correct"], False)
        self.assertEqual(result["citation_completeness"], 1)

    def test_invented_evidence_quote_is_rejected(self):
        self.judgment["claims"][0]["evidence"][0]["quote"] = "支持 5G"
        with self.assertRaisesRegex(ValueError, "quote does not exist"):
            summarize_judgment(self.benchmark, self.record, self.judgment)

    def test_retrieved_but_not_used_document_cannot_support_claim(self):
        self.judgment["claims"][0]["evidence"][0] = {"chunk_id": "demo-a-network", "quote": "4G"}
        with self.assertRaisesRegex(ValueError, "actual context"):
            summarize_judgment(self.benchmark, self.record, self.judgment)

    def test_no_answer_abstention_has_no_fake_perfect_faithfulness(self):
        self.record.update(query_id="DEMO-002", answer="无法从文档确认。", abstained=True)
        result = summarize_judgment(self.benchmark, self.record, {"correct": None, "claims": []})
        self.assertTrue(result["correct"])
        self.assertIsNone(result["faithfulness"])

    def test_judge_cannot_ignore_bad_citation(self):
        self.record["answer"] += "[chunk_id=does-not-exist]"
        with self.assertRaisesRegex(ValueError, "omitted citations"):
            summarize_judgment(self.benchmark, self.record, self.judgment)

    def test_cli_command_prompts_are_not_citation_ids(self):
        self.record["answer"] += "\n```txt\n[LA2608]system-view\n[AC]radio enable\n```"
        result = summarize_judgment(self.benchmark, self.record, self.judgment)
        self.assertEqual(result["invalid_citation_ids"], [])

    def test_pre_generation_abstention_is_scored_without_inventing_a_prompt(self):
        self.record.update(query_id="DEMO-002", context_source="no_generation", context_docs=[],
                           answer="无法确认。", abstained=True)
        self.assertTrue(summarize_judgment(self.benchmark, self.record, {"correct": None, "claims": []})["correct"])


if __name__ == "__main__":
    unittest.main()
