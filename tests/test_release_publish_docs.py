"""Regression test for release.yml publish-docs toggle."""
import pathlib

RELEASE_YML = pathlib.Path(".github/workflows/release.yml")


def test_publish_docs_input_allows_false():
    content = RELEASE_YML.read_text()
    forbidden = "publish-docs: ${{ inputs.publish-docs || true }}"
    assert forbidden not in content, (
        "publish-docs input is forced to true and cannot be disabled"
    )
