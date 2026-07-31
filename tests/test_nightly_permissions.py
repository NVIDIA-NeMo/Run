import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


class TestNightlyWorkflow(unittest.TestCase):
    def test_tag_job_has_contents_write(self):
        workflow = REPO_ROOT / ".github" / "workflows" / "nightly.yml"
        text = workflow.read_text()
        self.assertIn("contents: write", text,
                      "nightly workflow needs contents:write to create/update the nightly tag")


