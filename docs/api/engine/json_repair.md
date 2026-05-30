# JSON Repair

Extracts the `code` string value from truncated/malformed model JSON output by regex-locating the `"code":"` prefix and manually decoding escape sequences (including \uXXXX).

::: rune.engine.json_repair
