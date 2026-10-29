"""Legacy routing corrections must not hide a missing required retrieval."""

import unittest
from app.evaluation.legacy import compare_doc_need, compare_plan


class LegacyScoringTests(unittest.TestCase):
    def test_unexecuted_non_applicable_router_is_skipped(self):
        result = compare_doc_need("not_applicable", "")
        self.assertTrue(result["passed"])
        self.assertTrue(result["skipped"])

    def test_unexecuted_required_router_still_fails(self):
        self.assertFalse(compare_doc_need("need_retrieval", "")["passed"])

    def test_absent_flags_mean_disabled_but_required_tools_still_fail(self):
        self.assertTrue(compare_plan({"use_embedding": False, "use_web": False}, {})[0])
        self.assertFalse(compare_plan({"use_embedding": True}, {})[0])


if __name__ == "__main__":
    unittest.main()
