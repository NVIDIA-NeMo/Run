# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Verify the Guardwords caller with synthetic pull request metadata."""

import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = yaml.load(
    (ROOT / ".github/workflows/guardwords-check.yml").read_text(), Loader=yaml.BaseLoader
)


class GuardwordsWorkflow(unittest.TestCase):
    """Check mirror attestation and credential forwarding."""

    def test_scan_receives_only_catalog_key_after_verification(self) -> None:
        """Run only on mirror pushes and use the checked commit for scanning."""
        self.assertEqual(WORKFLOW["on"], {"push": {"branches": ["pull-request/[0-9]+"]}})
        self.assertEqual(WORKFLOW["permissions"], {"contents": "read", "pull-requests": "read"})
        job = WORKFLOW["jobs"]["guardwords-check"]
        self.assertEqual(job["needs"], "pr_info")
        self.assertEqual(job["with"]["head-sha"], "${{ github.sha }}")
        self.assertEqual(job["with"]["base-sha"], "${{ needs.pr_info.outputs.base_sha }}")
        self.assertEqual(
            job["with"]["pr-number"], "${{ fromJSON(needs.pr_info.outputs.pr_number) }}"
        )
        self.assertEqual(
            job["secrets"], {"GUARDWORDS_DEPLOY_KEY": "${{ secrets.GUARDWORDS_DEPLOY_KEY }}"}
        )

    def test_mirror_must_match_the_current_pr_head(self) -> None:
        """Release PR coordinates only after matching the push SHA to the GitHub response."""
        base, head = "a" * 40, "b" * 40
        (lookup,) = WORKFLOW["jobs"]["pr_info"]["steps"]
        for ref, sha, expected in (
            ("pull-request/42", head, True),
            ("pull-request/42", "c" * 40, False),
            ("pull-request/topic", head, False),
        ):
            with self.subTest(ref=ref, sha=sha), tempfile.TemporaryDirectory() as tmp:
                output = Path(tmp, "output")
                output.touch()
                stub = 'gh() { [[ "$*" == "api repos/NVIDIA-NeMo/Run/pulls/42" ]] && printf "%s" "$PR"; }'
                result = subprocess.run(
                    ["bash", "-eo", "pipefail", "-c", f"{stub}\n{lookup['run']}"],
                    env={
                        "PATH": os.environ["PATH"],
                        "PR": json.dumps({"base": {"sha": base}, "head": {"sha": head}}),
                        "GITHUB_REPOSITORY": "NVIDIA-NeMo/Run",
                        "GITHUB_REF_NAME": ref,
                        "GITHUB_SHA": sha,
                        "GITHUB_OUTPUT": str(output),
                    },
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(result.returncode == 0, expected, result.stderr)
                self.assertEqual(
                    output.read_text(), f"base_sha={base}\npr_number=42\n" if expected else ""
                )
