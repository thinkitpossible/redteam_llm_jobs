from __future__ import annotations

import unittest

from redteam_professions.llm.client import extract_message_content


class LlmClientParseTests(unittest.TestCase):
    def test_extract_message_content(self) -> None:
        body = {
            "choices": [
                {"message": {"content": "  hello world  "}},
            ]
        }
        self.assertEqual(extract_message_content(body), "hello world")

    def test_extract_missing_choices(self) -> None:
        with self.assertRaises(ValueError):
            extract_message_content({})

    def test_extract_empty_content(self) -> None:
        with self.assertRaises(ValueError):
            extract_message_content({"choices": [{"message": {"content": "   "}}]})


if __name__ == "__main__":
    unittest.main()
