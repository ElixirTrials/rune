"""SPDX license classification for mined repositories (paper §B.4)."""

from __future__ import annotations

from enum import Enum

__all__ = ["LicenseStatus", "classify_license"]


class LicenseStatus(str, Enum):
    """SPDX license classification status."""

    permitted = "permitted"
    attribution = "attribution"
    excluded = "excluded"


_PERMITTED = frozenset({"mit", "apache-2.0", "bsd-2-clause", "bsd-3-clause"})
_ATTRIBUTION = frozenset(
    {"gpl-2.0", "gpl-3.0", "lgpl-2.1", "lgpl-3.0"}
)


def classify_license(spdx: str | None) -> LicenseStatus:
    """Map an SPDX identifier to a license status."""
    if not spdx:
        return LicenseStatus.excluded
    key = spdx.strip().lower()
    if key in _PERMITTED:
        return LicenseStatus.permitted
    if key in _ATTRIBUTION:
        return LicenseStatus.attribution
    return LicenseStatus.excluded
