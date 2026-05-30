"""Extract code from truncated JSON model output."""

from __future__ import annotations

import re

_CODE_VALUE_RE = re.compile(r'"code"\s*:\s*"', re.DOTALL)
_ESCAPES: dict[str, str] = {
    "n": "\n",
    "t": "\t",
    "r": "\r",
    "\\": "\\",
    '"': '"',
    "/": "/",
}


def extract_code_value(raw: str) -> str:
    """Extract the ``code`` string from possibly-truncated JSON.

    Handles ``\\uXXXX`` escapes and returns ``""`` when the
    ``"code": "`` prefix isn't found.
    """
    m = _CODE_VALUE_RE.search(raw)
    if not m:
        return ""
    after = raw[m.end() :]
    chars: list[str] = []
    i = 0
    while i < len(after):
        ch = after[i]
        if ch == '"':
            break
        if ch == "\\":
            if i + 1 >= len(after):
                break
            nxt = after[i + 1]
            if nxt == "u" and i + 5 < len(after):
                hex_str = after[i + 2 : i + 6]
                try:
                    chars.append(chr(int(hex_str, 16)))
                    i += 6
                    continue
                except ValueError:
                    pass
            chars.append(_ESCAPES.get(nxt, nxt))
            i += 2
        else:
            chars.append(ch)
            i += 1
    return "".join(chars)
