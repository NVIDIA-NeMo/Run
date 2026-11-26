# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# /// script
# dependencies = ["pyyaml>=6.0,<7"]
# ///
"""Exercise release guards without credentials, uploads, or package builds."""

import json
import os
import re
import subprocess
import tempfile
import unittest
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = yaml.safe_load(next((ROOT / ".github/workflows").glob("release.y*ml")).read_text())
JOBS = WORKFLOW["jobs"]


def evaluate(expression: str, context: dict[str, object]) -> bool:
    """Evaluate the boolean and equality subset used by release job guards."""
    source = expression.strip().removeprefix("${{").removesuffix("}}")
    source = source.replace("!cancelled()", "True").replace("!failure()", "True")
    source = re.sub(
        r"\b(?:github|inputs|needs)\.[\w.-]+", lambda match: repr(context[match[0]]), source
    )
    source = source.replace("&&", " and ").replace("||", " or ")
    source = re.sub(r"!(?!=)", "not ", source)
    return bool(eval(" ".join(source.split()), {"__builtins__": {}}, {}))


class ReleaseGuards(unittest.TestCase):
    """Cover publication eligibility and unsuccessful or skipped dependencies."""

    def setUp(self) -> None:
        """Create a successful manual release context for the canonical repo."""
        self.context = {
            "github.repository": "NVIDIA-NeMo/Run",
            "github.event_name": "workflow_dispatch",
            "inputs.dry-run": False,
            "needs.validate-inputs.result": "success",
            "needs.release-workflow-tests.result": "success",
            "needs.bump.result": "success",
            "needs.build-wheel.result": "success",
            "needs.publish-wheel.result": "success",
        }

    def test_publish_requires_manual_non_dry_run(self) -> None:
        """Pushes and dry runs cannot request the production upload job."""
        for event in ("push", "workflow_dispatch"):
            for dry_run in (False, True):
                with self.subTest(event=event, dry_run=dry_run):
                    context = dict(
                        self.context, **{"github.event_name": event, "inputs.dry-run": dry_run}
                    )
                    self.assertEqual(
                        evaluate(JOBS["publish-wheel"]["if"], context),
                        event == "workflow_dispatch" and not dry_run,
                    )
        self.context["github.repository"] = "example/fork"
        self.assertFalse(evaluate(JOBS["publish-wheel"]["if"], self.context))

    def test_missing_build_blocks_upload(self) -> None:
        """Failed, cancelled and skipped builds cannot reach publication."""
        for state in ("failure", "cancelled", "skipped"):
            with self.subTest(state=state):
                self.context["needs.build-wheel.result"] = state
                self.assertFalse(evaluate(JOBS["publish-wheel"]["if"], self.context))

    def test_regression_failures_block_version_preparation(self) -> None:
        """Require the regression job to succeed before version preparation."""
        self.assertIn("release-workflow-tests", JOBS["bump"]["needs"])
        self.assertTrue(evaluate(JOBS["bump"]["if"], self.context))
        for state in ("failure", "cancelled", "skipped"):
            with self.subTest(state=state):
                self.context["needs.release-workflow-tests.result"] = state
                self.assertFalse(evaluate(JOBS["bump"]["if"], self.context))

    def test_real_finalize_requires_upload(self) -> None:
        """A missing upload cannot finalize a real release; dry runs may continue."""
        for state in ("failure", "cancelled", "skipped"):
            with self.subTest(state=state):
                self.context["needs.publish-wheel.result"] = state
                self.context["inputs.dry-run"] = False
                self.assertFalse(evaluate(JOBS["finalize"]["if"], self.context))
                self.context["inputs.dry-run"] = True
                self.assertTrue(evaluate(JOBS["finalize"]["if"], self.context))

    def test_summary_rejects_missing_required_jobs(self) -> None:
        """Run the actual shell summary against failures and permitted skips."""
        script = JOBS["release-summary"]["steps"][0]["run"]
        for publish in (False, True):
            baseline = {name: {"result": "success"} for name in JOBS["release-summary"]["needs"]}
            baseline["publish-wheel"]["result"] = "success" if publish else "skipped"
            for job in (
                "release-workflow-tests",
                "validate-inputs",
                "bump",
                "build-wheel",
                "publish-wheel",
                "finalize",
            ):
                for state in ("success", "failure", "cancelled", "skipped"):
                    with self.subTest(publish=publish, job=job, state=state):
                        needs = {name: dict(value) for name, value in baseline.items()}
                        needs[job]["result"] = state
                        env = dict(
                            os.environ,
                            NEEDS=json.dumps(needs),
                            PUBLISH_EXPECTED=str(publish).lower(),
                            RELEASE_SKIPPED="false",
                        )
                        result = subprocess.run(
                            ["bash", "-euo", "pipefail", "-c", script], env=env, capture_output=True
                        )
                        expected = state == (
                            "skipped" if job == "publish-wheel" and not publish else "success"
                        )
                        self.assertEqual(result.returncode == 0, expected, result.stderr.decode())

    def test_validate_ref_treats_input_as_data(self) -> None:
        """Only full immutable SHAs pass; branch names and shell payloads fail."""
        script = JOBS["validate-inputs"]["steps"][0]["run"]
        for ref in ("a" * 40, "main", "abc1234", "$(exit 0)", "a" * 40 + "\n"):
            with self.subTest(ref=ref):
                result = subprocess.run(
                    ["bash", "-euo", "pipefail", "-c", script],
                    env=dict(os.environ, RELEASE_REF=ref),
                    capture_output=True,
                )
                self.assertEqual(result.returncode == 0, ref == "a" * 40)

    def test_protected_release_authentication(self) -> None:
        """Limit release credentials to the named environment in the called workflows."""
        freeze = yaml.safe_load((ROOT / ".github/workflows/release-freeze.yml").read_text())
        jobs = (JOBS["bump"], JOBS["finalize"], freeze["jobs"]["code-freeze"])
        for job in jobs:
            self.assertEqual(job["with"]["release-environment"], "run-release")
            self.assertNotIn("app-id", job["with"])
            self.assertNotIn("BOT_KEY", job.get("secrets", {}))
            self.assertNotIn("RELEASE_APP_PRIVATE_KEY", job.get("secrets", {}))
        self.assertEqual(jobs[2]["with"]["freeze-source-branch"], "main")
        self.assertEqual(jobs[0]["with"]["release-branch-pattern"], "[rv][0-9]*.[0-9]*.[0-9]*")
        self.assertEqual(JOBS["bump"]["permissions"], {"contents": "read"})

    def test_workflow_authority(self) -> None:
        """Require explicit job permissions and isolate the dual-organization App."""
        directory = ROOT / ".github/workflows"
        for path in sorted(directory.glob("*.yml")):
            text = path.read_text()
            workflow = yaml.safe_load(text)
            with self.subTest(workflow=path.name):
                for job in workflow["jobs"].values():
                    self.assertIsInstance(job.get("permissions", workflow.get("permissions")), dict)
                    self.assertNotEqual(job.get("secrets"), "inherit")
                if path.name != "community-bot.yml":
                    self.assertNotRegex(text, r"secrets\.(BOT_KEY|PAT)\b")
        changelog = yaml.safe_load((directory / "changelog-build.yml").read_text())
        nightly = yaml.safe_load((directory / "nightly.yml").read_text())
        for job in (changelog["jobs"]["changelog"], nightly["jobs"]["tag"]):
            self.assertEqual(job["if"], "github.ref == 'refs/heads/main'")

    def test_docs_rehearsal_and_production(self) -> None:
        """Build rehearsal documentation and withhold the publisher credentials."""
        finalize = JOBS["finalize"]
        rehearsal = JOBS["rehearse-docs"]
        docs = yaml.safe_load((ROOT / ".github/workflows/release-docs.yml").read_text())
        nightly = yaml.safe_load((ROOT / ".github/workflows/release-nightly-docs.yml").read_text())
        self.assertIn("rehearse-docs", JOBS["release-summary"]["needs"])
        self.assertNotIn("secrets", rehearsal)
        self.assertEqual(
            set(finalize["secrets"]), set(nightly["jobs"]["call-release-docs"]["secrets"])
        )
        for ref, kind, dry, allowed in (
            ("refs/heads/main", "branch", False, True),
            ("refs/tags/v0.13.0", "tag", False, True),
            ("refs/heads/main", "branch", True, False),
            ("refs/heads/topic", "branch", False, False),
        ):
            ctx = {"github.ref": ref, "github.ref_type": kind, "inputs.dry-run": dry}
            self.assertEqual(evaluate(docs["jobs"]["publish-docs"]["if"], ctx), allowed)
        for event, dry, requested, publish, build in (
            ("push", False, False, False, True),
            ("merge_group", False, False, False, True),
            ("workflow_dispatch", True, True, False, True),
            ("workflow_dispatch", False, True, True, False),
            ("workflow_dispatch", False, False, False, False),
        ):
            with self.subTest(event=event, dry=dry, requested=requested):
                ctx = {
                    "github.event_name": event,
                    "inputs.dry-run": dry,
                    "inputs.publish-docs": requested,
                }
                self.assertEqual(evaluate(finalize["with"]["publish-docs"], ctx), publish)
                self.assertEqual(evaluate(rehearsal["if"], ctx), build)
        gate = finalize["with"]["publish-docs"][4:-3]
        for key, value in finalize["secrets"].items():
            self.assertEqual(value, "${{ " + gate + " && secrets." + key + " || '' }}")

    def test_manual_source_and_target_checks(self) -> None:
        """Accept protected branch ancestry and Run's multi-digit release branches."""
        step = JOBS["validate-inputs"]["steps"][1]
        self.assertEqual(step["if"], "github.event_name == 'workflow_dispatch'")
        sha = "a" * 40
        with tempfile.TemporaryDirectory() as directory:
            gh = Path(directory) / "gh"
            gh.write_text(
                '#!/bin/sh\ncase "$2" in */branches/*) echo "$TIP" ;; '
                '*/compare/*) echo "$COMPARISON" ;; *) exit 1 ;; esac\n'
            )
            gh.chmod(0o755)
            for source, target, tip, comparison, allowed in (
                ("main", "r0.12.0", sha, "ahead", True),
                ("main", "v10.5.20", sha, "identical", True),
                ("main", "v10.5.20rc2", sha, "identical", False),
                ("main", "r0.13.0-unreviewed", sha, "identical", False),
                ("topic", "main", sha, "identical", False),
                ("main", "topic", sha, "identical", False),
                ("main", "main", "", "ahead", False),
                ("main", "main", sha, "diverged", False),
            ):
                with self.subTest(source=source, target=target, tip=tip, comparison=comparison):
                    env = dict(
                        os.environ,
                        PATH=directory + os.pathsep + os.environ["PATH"],
                        TARGET_BRANCH=target,
                        RELEASE_REF=sha,
                        TIP=tip,
                        COMPARISON=comparison,
                        GITHUB_REF="refs/heads/" + source,
                        GITHUB_REPOSITORY="NVIDIA-NeMo/Run",
                    )
                    result = subprocess.run(
                        ["bash", "-euo", "pipefail", "-c", step["run"]],
                        env=env,
                        capture_output=True,
                    )
                    self.assertEqual(result.returncode == 0, allowed, result.stderr)

    def test_upload_has_no_checkout_or_password(self) -> None:
        """Keep source execution and build credentials outside the OIDC job."""
        self.assertEqual(JOBS["build-wheel"]["permissions"], {"contents": "read"})
        self.assertNotIn("secrets", JOBS["build-wheel"])
        upload = JOBS["publish-wheel"]
        self.assertEqual(upload["permissions"], {"id-token": "write"})
        self.assertEqual(upload["environment"]["name"], "pypi")
        self.assertEqual(len(upload["steps"]), 2)
        for step in upload["steps"]:
            self.assertNotIn("run", step)
            self.assertNotIn("password", step.get("with", {}))
            self.assertNotIn("checkout", step["uses"])


if __name__ == "__main__":
    unittest.main()
