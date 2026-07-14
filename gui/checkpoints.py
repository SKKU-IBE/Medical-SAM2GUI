"""Checkpoint discovery and verified download helpers."""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import BinaryIO
from urllib.request import urlopen

from platformdirs import user_cache_dir


CHECKPOINT_FILENAME = "Medical_SAM2_pretrain.pth"
CHECKPOINT_ENV = "MEDICAL_SAM2_CHECKPOINT"
CHECKPOINT_URL = (
    "https://huggingface.co/jiayuanz3/MedSAM2_pretrain/resolve/main/"
    "MedSAM2_pretrain.pth"
)
CHECKPOINT_SHA256 = "059572b072eff2e41975bf85b0dcca96bc58889db60a89e9f5b1f075236735d7"
UPSTREAM_PAGE = "https://huggingface.co/jiayuanz3/MedSAM2_pretrain"


class CheckpointError(RuntimeError):
    """Raised when a checkpoint cannot be resolved or downloaded safely."""


def default_checkpoint_path() -> Path:
    """Return the platform-specific cache location for the official checkpoint."""
    cache_root = Path(user_cache_dir("medical-sam2-gui", "SKKU-IBE"))
    return cache_root / "checkpoints" / CHECKPOINT_FILENAME


def sha256_file(path: str | os.PathLike[str], chunk_size: int = 1024 * 1024) -> str:
    """Calculate a file SHA-256 without loading the checkpoint into memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _existing_file(path: Path, source: str) -> Path:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise CheckpointError(f"Checkpoint from {source} does not exist: {path}")
    return path


def resolve_checkpoint(
    explicit_path: str | os.PathLike[str] | None = None,
    *,
    environ: Mapping[str, str] | None = None,
    cwd: str | os.PathLike[str] | None = None,
    cache_path: str | os.PathLike[str] | None = None,
) -> Path | None:
    """Resolve a checkpoint using the documented precedence order.

    Explicit and environment-provided paths may point to custom weights and are
    therefore only checked for existence. The managed cache always contains the
    official artifact and is verified before use.
    """
    if explicit_path is not None:
        return _existing_file(Path(explicit_path), "--checkpoint")

    environ = os.environ if environ is None else environ
    environment_path = environ.get(CHECKPOINT_ENV)
    if environment_path:
        return _existing_file(Path(environment_path), CHECKPOINT_ENV)

    legacy_path = Path.cwd() if cwd is None else Path(cwd)
    legacy_path = legacy_path / CHECKPOINT_FILENAME
    if legacy_path.is_file():
        return legacy_path.resolve()

    managed_path = default_checkpoint_path() if cache_path is None else Path(cache_path)
    if not managed_path.is_file():
        return None

    actual_hash = sha256_file(managed_path)
    if actual_hash != CHECKPOINT_SHA256:
        raise CheckpointError(
            "Cached checkpoint checksum mismatch. Run "
            "`medical-sam2-download-checkpoint --force` to replace it. "
            f"Expected {CHECKPOINT_SHA256}, got {actual_hash}."
        )
    return managed_path.resolve()


def _content_length(response: BinaryIO) -> int | None:
    headers = getattr(response, "headers", None)
    if headers is None:
        return None
    value = headers.get("Content-Length")
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def download_checkpoint(
    destination: str | os.PathLike[str] | None = None,
    *,
    force: bool = False,
    source_url: str = CHECKPOINT_URL,
    expected_sha256: str = CHECKPOINT_SHA256,
    opener: Callable[[str], BinaryIO] = urlopen,
    progress: Callable[[int, int | None], None] | None = None,
    chunk_size: int = 1024 * 1024,
) -> Path:
    """Download and atomically install a checksum-verified checkpoint."""
    destination_path = default_checkpoint_path() if destination is None else Path(destination)
    destination_path = destination_path.expanduser().resolve()
    destination_path.parent.mkdir(parents=True, exist_ok=True)

    if destination_path.exists():
        current_hash = sha256_file(destination_path)
        if current_hash == expected_sha256:
            return destination_path
        if not force:
            raise CheckpointError(
                f"A different file already exists at {destination_path}. "
                "Use --force to replace it."
            )

    partial_path = destination_path.with_name(destination_path.name + ".part")
    partial_path.unlink(missing_ok=True)

    try:
        downloaded = 0
        with opener(source_url) as response, partial_path.open("wb") as output:
            total = _content_length(response)
            while True:
                chunk = response.read(chunk_size)
                if not chunk:
                    break
                output.write(chunk)
                downloaded += len(chunk)
                if progress is not None:
                    progress(downloaded, total)

        actual_hash = sha256_file(partial_path)
        if actual_hash != expected_sha256:
            raise CheckpointError(
                "Downloaded checkpoint checksum mismatch. "
                f"Expected {expected_sha256}, got {actual_hash}."
            )
        os.replace(partial_path, destination_path)
        return destination_path
    except Exception as exc:
        partial_path.unlink(missing_ok=True)
        if isinstance(exc, CheckpointError):
            raise
        raise CheckpointError(f"Checkpoint download failed: {exc}") from exc


def download_cli(argv: list[str] | None = None) -> int:
    """Console entry point for the verified checkpoint downloader."""
    parser = argparse.ArgumentParser(
        description="Download the official Medical-SAM2 checkpoint with SHA-256 verification."
    )
    parser.add_argument("--destination", type=Path, help="Override the user cache destination.")
    parser.add_argument("--force", action="store_true", help="Replace a mismatched existing file.")
    parser.add_argument("--yes", action="store_true", help="Skip the interactive confirmation.")
    args = parser.parse_args(argv)

    destination = args.destination or default_checkpoint_path()
    print("Medical-SAM2 checkpoint is supplied by the upstream project.")
    print(f"Source: {CHECKPOINT_URL}")
    print(f"Upstream terms and model information: {UPSTREAM_PAGE}")
    print(f"Destination: {destination}")
    if not args.yes:
        answer = input("Download after reviewing the upstream terms? [y/N] ").strip().lower()
        if answer not in {"y", "yes"}:
            print("Download cancelled.")
            return 0

    last_reported = -1

    def report(downloaded: int, total: int | None) -> None:
        nonlocal last_reported
        mib = downloaded // (1024 * 1024)
        if mib == last_reported:
            return
        last_reported = mib
        if total:
            print(f"\rDownloaded {downloaded / 1024 / 1024:.0f}/{total / 1024 / 1024:.0f} MiB", end="")
        else:
            print(f"\rDownloaded {downloaded / 1024 / 1024:.0f} MiB", end="")

    try:
        path = download_checkpoint(destination, force=args.force, progress=report)
    except CheckpointError as exc:
        print(f"\nError: {exc}", file=sys.stderr)
        return 2

    print(f"\nCheckpoint ready: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(download_cli())
