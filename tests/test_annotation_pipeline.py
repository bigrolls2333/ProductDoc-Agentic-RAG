"""Evidence restoration must not turn invented or ambiguous text into a quote."""

import unittest
from app.evaluation.annotation import clean_content, verify_proposals
from app.lm.request_options import non_thinking_body


class AnnotationTests(unittest.TestCase):
    def test_unique_whitespace_difference_restores_exact_original_quote(self):
        chunks = [{"chunk_id": "c", "content": "端口为 A\n B，继续。"}]
        rows = verify_proposals({"judgments": [{"chunk_id": "c", "relevance": 2,
                                               "facts": [{"text": "端口", "quote": "端口为 A B"}]}]}, chunks)
        self.assertEqual(rows[0]["facts"][0]["quote"], "端口为 A\n B")

    def test_ambiguous_normalized_quote_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "uniquely"):
            verify_proposals({"judgments": [{"chunk_id": "c", "relevance": 2,
                                              "facts": [{"text": "fact", "quote": "a  b"}]}]},
                             [{"chunk_id": "c", "content": "a b, a b"}])

    def test_generated_image_caption_is_removed_from_annotation_input(self):
        self.assertEqual(clean_content("原文 ![生成的图像摘要](http://local/image.png) 末尾"), "原文  末尾")

    def test_deepseek_uses_its_documented_thinking_switch(self):
        self.assertEqual(non_thinking_body("https://api.deepseek.com/v1"), {"thinking": {"type": "disabled"}})
        self.assertEqual(non_thinking_body("https://api.openai.com/v1"), {})


if __name__ == "__main__":
    unittest.main()
