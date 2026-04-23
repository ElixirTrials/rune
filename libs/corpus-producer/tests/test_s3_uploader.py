"""Unit tests for the S3 manifest uploader.

boto3 is imported inside the uploader; tests stub the module via
``monkeypatch.setitem(sys.modules, "boto3", ...)`` so they run without
AWS credentials or the boto3 dependency.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest


def _make_manifest(tmp_path: Path, name: str = "decompose_humaneval.jsonl") -> Path:
    p = tmp_path / name
    p.write_text('{"task_id": "t1"}\n', encoding="utf-8")
    return p


def test_upload_manifest_missing_boto3_returns_false(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When boto3 is not importable, uploader logs and returns False."""
    monkeypatch.setitem(sys.modules, "boto3", None)

    from corpus_producer.s3_uploader import upload_manifest

    manifest = _make_manifest(tmp_path)
    ok = upload_manifest(manifest, bucket="my-bucket", prefix="oracles")
    assert ok is False


def test_upload_manifest_no_credentials_returns_false(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When boto3 raises NoCredentialsError, uploader logs and returns False."""
    fake_boto3 = MagicMock()
    # Exception classes live on botocore.exceptions; simulate both modules.
    fake_botocore_exc = type(
        "FakeNoCredentialsError",
        (Exception,),
        {},
    )

    # Client raises NoCredentialsError on upload_file
    client = MagicMock()
    client.upload_file.side_effect = fake_botocore_exc("no creds")
    fake_boto3.client.return_value = client

    # Point botocore.exceptions.NoCredentialsError at our fake class so the
    # uploader's except clause catches it.
    fake_exceptions_mod = MagicMock()
    fake_exceptions_mod.NoCredentialsError = fake_botocore_exc
    fake_exceptions_mod.ClientError = fake_botocore_exc
    fake_exceptions_mod.BotoCoreError = fake_botocore_exc
    monkeypatch.setitem(sys.modules, "boto3", fake_boto3)
    fake_botocore = MagicMock(exceptions=fake_exceptions_mod)
    monkeypatch.setitem(sys.modules, "botocore", fake_botocore)
    monkeypatch.setitem(sys.modules, "botocore.exceptions", fake_exceptions_mod)

    from corpus_producer.s3_uploader import upload_manifest

    manifest = _make_manifest(tmp_path)
    ok = upload_manifest(manifest, bucket="my-bucket", prefix="oracles")
    assert ok is False


def test_upload_manifest_happy_path_calls_boto3(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """On success, upload_manifest calls client.upload_file with the expected S3 key."""
    fake_boto3 = MagicMock()
    client = MagicMock()
    fake_boto3.client.return_value = client

    fake_exceptions_mod = MagicMock()
    fake_exceptions_mod.NoCredentialsError = type(
        "NoCredentialsError", (Exception,), {}
    )
    fake_exceptions_mod.ClientError = type("ClientError", (Exception,), {})
    fake_exceptions_mod.BotoCoreError = type("BotoCoreError", (Exception,), {})
    monkeypatch.setitem(sys.modules, "boto3", fake_boto3)
    fake_botocore = MagicMock(exceptions=fake_exceptions_mod)
    monkeypatch.setitem(sys.modules, "botocore", fake_botocore)
    monkeypatch.setitem(sys.modules, "botocore.exceptions", fake_exceptions_mod)

    from corpus_producer.s3_uploader import upload_manifest

    manifest = _make_manifest(tmp_path, "diagnose_pooled.jsonl")
    ok = upload_manifest(
        manifest, bucket="my-bucket", prefix="oracles/run-1"
    )
    assert ok is True
    fake_boto3.client.assert_called_once_with("s3")
    client.upload_file.assert_called_once_with(
        str(manifest), "my-bucket", "oracles/run-1/diagnose_pooled.jsonl"
    )


def test_upload_manifest_empty_prefix_uses_basename_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An empty prefix uploads to the bucket root with just the file basename."""
    fake_boto3 = MagicMock()
    client = MagicMock()
    fake_boto3.client.return_value = client

    fake_exceptions_mod = MagicMock()
    fake_exceptions_mod.NoCredentialsError = type(
        "NoCredentialsError", (Exception,), {}
    )
    fake_exceptions_mod.ClientError = type("ClientError", (Exception,), {})
    fake_exceptions_mod.BotoCoreError = type("BotoCoreError", (Exception,), {})
    monkeypatch.setitem(sys.modules, "boto3", fake_boto3)
    fake_botocore = MagicMock(exceptions=fake_exceptions_mod)
    monkeypatch.setitem(sys.modules, "botocore", fake_botocore)
    monkeypatch.setitem(sys.modules, "botocore.exceptions", fake_exceptions_mod)

    from corpus_producer.s3_uploader import upload_manifest

    manifest = _make_manifest(tmp_path, "plan_mbpp.jsonl")
    ok = upload_manifest(manifest, bucket="my-bucket", prefix="")
    assert ok is True
    client.upload_file.assert_called_once_with(
        str(manifest), "my-bucket", "plan_mbpp.jsonl"
    )


def test_upload_manifest_strips_trailing_slash_in_prefix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Trailing slash in prefix is normalized to avoid double-slash keys."""
    fake_boto3 = MagicMock()
    client = MagicMock()
    fake_boto3.client.return_value = client

    fake_exceptions_mod = MagicMock()
    fake_exceptions_mod.NoCredentialsError = type(
        "NoCredentialsError", (Exception,), {}
    )
    fake_exceptions_mod.ClientError = type("ClientError", (Exception,), {})
    fake_exceptions_mod.BotoCoreError = type("BotoCoreError", (Exception,), {})
    monkeypatch.setitem(sys.modules, "boto3", fake_boto3)
    fake_botocore = MagicMock(exceptions=fake_exceptions_mod)
    monkeypatch.setitem(sys.modules, "botocore", fake_botocore)
    monkeypatch.setitem(sys.modules, "botocore.exceptions", fake_exceptions_mod)

    from corpus_producer.s3_uploader import upload_manifest

    manifest = _make_manifest(tmp_path, "code_apps.jsonl")
    upload_manifest(manifest, bucket="b", prefix="runs/2026-04-23/")
    client.upload_file.assert_called_once_with(
        str(manifest), "b", "runs/2026-04-23/code_apps.jsonl"
    )


def test_s3_key_builder() -> None:
    """Pure helper ``build_s3_key`` normalizes prefix + basename."""
    from corpus_producer.s3_uploader import build_s3_key

    assert build_s3_key("oracles", "decompose_humaneval.jsonl") == (
        "oracles/decompose_humaneval.jsonl"
    )
    assert build_s3_key("oracles/", "decompose_humaneval.jsonl") == (
        "oracles/decompose_humaneval.jsonl"
    )
    assert build_s3_key("", "decompose_humaneval.jsonl") == (
        "decompose_humaneval.jsonl"
    )
    assert build_s3_key("a/b/c", "x.jsonl") == "a/b/c/x.jsonl"
