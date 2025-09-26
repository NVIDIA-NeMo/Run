"""
WCAG Diagrams Extension for Sphinx
Moves inline accessibility/interaction logic for diagrams into a reusable asset.
"""

from __future__ import annotations

import os
import shutil
from typing import Any

from sphinx.application import Sphinx
from sphinx.util import logging

logger = logging.getLogger(__name__)


def _copy_js_to_static(app: Sphinx) -> None:
    """Copy the wcag-diagrams.js asset into the built _static directory.

    We place the JS in _static so it can be referenced via app.add_js_file.
    """
    extension_dir = os.path.dirname(os.path.abspath(__file__))
    source_js = os.path.join(extension_dir, "wcag-diagrams.js")

    if not os.path.exists(source_js):
        logger.warning("wcag-diagrams.js not found: %s", source_js)
        return

    static_dir = os.path.join(app.outdir, "_static")
    os.makedirs(static_dir, exist_ok=True)

    dest_js = os.path.join(static_dir, "wcag-diagrams.js")
    try:
        shutil.copy2(source_js, dest_js)
        logger.info("WCAG diagrams JS copied to _static")
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to copy wcag-diagrams.js: %s", exc)


def _copy_css_to_static(app: Sphinx) -> None:
    """Copy the wcag-diagrams.css asset into the built _static directory."""
    extension_dir = os.path.dirname(os.path.abspath(__file__))
    source_css = os.path.join(extension_dir, "wcag-diagrams.css")

    if not os.path.exists(source_css):
        logger.warning("wcag-diagrams.css not found: %s", source_css)
        return

    static_dir = os.path.join(app.outdir, "_static")
    os.makedirs(static_dir, exist_ok=True)

    dest_css = os.path.join(static_dir, "wcag-diagrams.css")
    try:
        shutil.copy2(source_css, dest_css)
        logger.info("WCAG diagrams CSS copied to _static")
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to copy wcag-diagrams.css: %s", exc)


def _copy_assets_early(app: Sphinx, _docname: str, _source: list[str]) -> None:
    """Ensure assets are available early in the build process."""
    if hasattr(app, "_wcag_diagrams_assets_copied"):
        return
    _copy_js_to_static(app)
    _copy_css_to_static(app)
    app._wcag_diagrams_assets_copied = True  # type: ignore[attr-defined]  # noqa: SLF001


def _copy_assets_on_finish(app: Sphinx, exc: Exception | None) -> None:
    if exc is not None:
        return
    _copy_js_to_static(app)
    _copy_css_to_static(app)


def setup(app: Sphinx) -> dict[str, Any]:
    """Setup the WCAG diagrams extension."""
    # Ensure our JS is included in the built pages
    app.add_js_file("wcag-diagrams.js")
    # Ensure our CSS is included
    app.add_css_file("wcag-diagrams.css")

    # Copy asset early (so it's present when pages are rendered) and after build as backup
    app.connect("source-read", _copy_assets_early)
    app.connect("build-finished", _copy_assets_on_finish)

    return {
        "version": "1.0.0",
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }
