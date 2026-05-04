"""Tests for license classification per paper §B.4."""

from __future__ import annotations

import pytest

from model_training.d2l_licenses import LicenseStatus, classify_license


@pytest.mark.parametrize(
    "spdx,expected",
    [
        ("MIT", LicenseStatus.permitted),
        ("Apache-2.0", LicenseStatus.permitted),
        ("BSD-2-Clause", LicenseStatus.permitted),
        ("BSD-3-Clause", LicenseStatus.permitted),
        ("GPL-2.0", LicenseStatus.attribution),
        ("GPL-3.0", LicenseStatus.attribution),
        ("LGPL-2.1", LicenseStatus.attribution),
        ("LGPL-3.0", LicenseStatus.attribution),
        ("AGPL-3.0", LicenseStatus.excluded),
        ("proprietary", LicenseStatus.excluded),
        ("NOASSERTION", LicenseStatus.excluded),
        ("", LicenseStatus.excluded),
    ],
)
def test_classify_license(spdx: str, expected: LicenseStatus) -> None:
    assert classify_license(spdx) is expected


def test_none_license_is_excluded() -> None:
    assert classify_license(None) is LicenseStatus.excluded


def test_classify_license_is_case_insensitive() -> None:
    assert classify_license("mit") is LicenseStatus.permitted
    assert classify_license("apache-2.0") is LicenseStatus.permitted
