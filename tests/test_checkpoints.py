import hashlib
import io
from pathlib import Path

import pytest

from gui import checkpoints


class FakeResponse(io.BytesIO):
    def __init__(self, payload: bytes):
        super().__init__(payload)
        self.headers = {"Content-Length": str(len(payload))}


def test_official_checkpoint_metadata_matches_upstream_artifact():
    assert checkpoints.CHECKPOINT_URL.endswith("/MedSAM2_pretrain.pth")
    assert checkpoints.CHECKPOINT_SHA256 == (
        "059572b072eff2e41975bf85b0dcca96bc58889db60a89e9f5b1f075236735d7"
    )


def test_resolve_checkpoint_precedence(tmp_path, monkeypatch):
    explicit = tmp_path / "explicit.pth"
    environment = tmp_path / "environment.pth"
    legacy = tmp_path / checkpoints.CHECKPOINT_FILENAME
    managed = tmp_path / "cache" / checkpoints.CHECKPOINT_FILENAME
    for path in (explicit, environment, legacy, managed):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(path.name.encode())

    assert checkpoints.resolve_checkpoint(explicit, environ={}, cwd=tmp_path, cache_path=managed) == explicit
    assert (
        checkpoints.resolve_checkpoint(
            environ={checkpoints.CHECKPOINT_ENV: str(environment)},
            cwd=tmp_path,
            cache_path=managed,
        )
        == environment
    )
    assert checkpoints.resolve_checkpoint(environ={}, cwd=tmp_path, cache_path=managed) == legacy

    legacy.unlink()
    monkeypatch.setattr(checkpoints, "CHECKPOINT_SHA256", checkpoints.sha256_file(managed))
    assert checkpoints.resolve_checkpoint(environ={}, cwd=tmp_path, cache_path=managed) == managed


def test_resolve_missing_explicit_and_environment_paths(tmp_path):
    with pytest.raises(checkpoints.CheckpointError, match="--checkpoint"):
        checkpoints.resolve_checkpoint(tmp_path / "missing.pth", environ={}, cwd=tmp_path)

    with pytest.raises(checkpoints.CheckpointError, match=checkpoints.CHECKPOINT_ENV):
        checkpoints.resolve_checkpoint(
            environ={checkpoints.CHECKPOINT_ENV: str(tmp_path / "missing.pth")},
            cwd=tmp_path,
        )


def test_resolve_returns_none_when_no_checkpoint_exists(tmp_path):
    assert (
        checkpoints.resolve_checkpoint(
            environ={},
            cwd=tmp_path,
            cache_path=tmp_path / "missing-cache.pth",
        )
        is None
    )


def test_resolve_rejects_corrupt_managed_checkpoint(tmp_path, monkeypatch):
    managed = tmp_path / "managed.pth"
    managed.write_bytes(b"corrupt")
    monkeypatch.setattr(checkpoints, "CHECKPOINT_SHA256", "0" * 64)

    with pytest.raises(checkpoints.CheckpointError, match="checksum mismatch"):
        checkpoints.resolve_checkpoint(environ={}, cwd=tmp_path / "empty", cache_path=managed)


def test_download_checkpoint_verifies_and_installs_atomically(tmp_path):
    payload = b"official checkpoint bytes"
    expected = hashlib.sha256(payload).hexdigest()
    destination = tmp_path / "nested" / "checkpoint.pth"
    progress_calls = []

    result = checkpoints.download_checkpoint(
        destination,
        source_url="https://example.test/checkpoint",
        expected_sha256=expected,
        opener=lambda _url: FakeResponse(payload),
        progress=lambda downloaded, total: progress_calls.append((downloaded, total)),
        chunk_size=5,
    )

    assert result == destination.resolve()
    assert destination.read_bytes() == payload
    assert not destination.with_name(destination.name + ".part").exists()
    assert progress_calls[-1] == (len(payload), len(payload))


def test_download_checksum_failure_leaves_no_output(tmp_path):
    destination = tmp_path / "checkpoint.pth"

    with pytest.raises(checkpoints.CheckpointError, match="checksum mismatch"):
        checkpoints.download_checkpoint(
            destination,
            source_url="https://example.test/checkpoint",
            expected_sha256="0" * 64,
            opener=lambda _url: FakeResponse(b"wrong bytes"),
        )

    assert not destination.exists()
    assert not destination.with_name(destination.name + ".part").exists()


def test_download_failure_removes_partial_file(tmp_path):
    destination = tmp_path / "checkpoint.pth"

    def fail(_url):
        raise OSError("network unavailable")

    with pytest.raises(checkpoints.CheckpointError, match="network unavailable"):
        checkpoints.download_checkpoint(destination, opener=fail)

    assert not destination.exists()
    assert not destination.with_name(destination.name + ".part").exists()


def test_download_requires_force_for_mismatched_existing_file(tmp_path):
    destination = tmp_path / "checkpoint.pth"
    destination.write_bytes(b"old")
    payload = b"replacement"
    expected = hashlib.sha256(payload).hexdigest()

    with pytest.raises(checkpoints.CheckpointError, match="--force"):
        checkpoints.download_checkpoint(destination, expected_sha256=expected)

    result = checkpoints.download_checkpoint(
        destination,
        force=True,
        expected_sha256=expected,
        opener=lambda _url: FakeResponse(payload),
    )
    assert result.read_bytes() == payload


def test_forced_download_failure_preserves_existing_file(tmp_path):
    destination = tmp_path / "checkpoint.pth"
    destination.write_bytes(b"existing custom checkpoint")

    with pytest.raises(checkpoints.CheckpointError, match="network unavailable"):
        checkpoints.download_checkpoint(
            destination,
            force=True,
            opener=lambda _url: (_ for _ in ()).throw(OSError("network unavailable")),
        )

    assert destination.read_bytes() == b"existing custom checkpoint"
    assert not destination.with_name(destination.name + ".part").exists()


def test_gui_package_does_not_eagerly_import_napari():
    import subprocess
    import sys

    command = [
        sys.executable,
        "-c",
        "import sys, gui; assert 'napari' not in sys.modules",
    ]
    result = subprocess.run(command, check=False, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
