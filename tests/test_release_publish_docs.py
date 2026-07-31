import re
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


class TestReleaseWorkflow(unittest.TestCase):
    def test_publish_docs_respects_false_input(self):
        workflow = REPO_ROOT / ".github" / "workflows" / "release.yml"
        text = workflow.read_text()
        match = re.search(r"^\s+publish-docs:\s*(.+)$", text, re.MULTILINE)
        self.assertIsNotNone(match)
        self.assertNotIn("|| true", match.group(1),
                         "publish-docs always evaluates to true; false input is ignored")

