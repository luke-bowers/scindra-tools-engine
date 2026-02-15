"""Shared pytest configuration and fixtures."""
from __future__ import annotations

import pytest


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    """Auto-skip tests marked ``detector_optional`` when onnxruntime is missing."""
    try:
        import onnxruntime as _ort  # noqa: F401

        has_ort = True
    except ImportError:
        has_ort = False

    if has_ort:
        return

    skip_marker = pytest.mark.skip(
        reason="onnxruntime not installed (optional detector dependency)"
    )
    for item in items:
        if "detector_optional" in item.keywords:
            item.add_marker(skip_marker)
