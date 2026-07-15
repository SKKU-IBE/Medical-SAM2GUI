"""Test-session setup shared across platforms."""

# PyTorch 2.9+ can fail to initialize c10.dll on Windows if Qt loads first.
import torch  # noqa: F401, E402
