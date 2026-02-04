"""nnUNetv2 inference helper for auto segmentation.

This wrapper keeps dependencies optional: if nnUNetv2 is not installed or the
API is unavailable, a clear RuntimeError is raised so the caller can surface the
message to the user.
"""
import os
from typing import Iterable, Optional, Sequence

import numpy as np
import torch


def _ensure_model_path(path: Optional[str]) -> str:
    if path is None or not str(path).strip():
        raise RuntimeError("nnUNetv2 model path is not set. Please select the model folder in Patient settings.")
    resolved = os.path.abspath(path)
    if not os.path.exists(resolved):
        raise RuntimeError(f"nnUNetv2 model path not found: {resolved}")
    return resolved


def run_nnunetv2_inference(
    imgs: torch.Tensor,
    meta: Optional[dict],
    model_path: Optional[str],
    device: str = "cuda",
    folds: Optional[Sequence[int]] = None,
    checkpoint_name: str = "checkpoint_final.pth",
    tile_step_size: float = 0.5,
) -> np.ndarray:
    """Run nnUNetv2 inference on a 3D volume.

    Args:
        imgs: Tensor shaped [T, C, H, W] or [T, H, W]; T == depth.
        meta: Optional metadata; if contains 'spacing', it will be passed to the predictor.
        model_path: Trained nnUNetv2 model folder.
        device: 'cuda' or 'cpu'.
        folds: folds to use; default (0,).
        checkpoint_name: which checkpoint to load from the model folder.
        tile_step_size: overlap for sliding window.

    Returns:
        seg_mask: np.ndarray of shape (T, H, W) with integer labels.
    """
    model_path = _ensure_model_path(model_path)
    if isinstance(device, str):
        try:
            device = torch.device(device)
        except Exception as exc:
            raise RuntimeError(f"Invalid device '{device}'. Use 'cuda' or 'cpu'.") from exc
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but torch.cuda.is_available() is False.")
    try:
        from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
    except Exception as exc:  # pragma: no cover - import error path
        raise RuntimeError(
            "nnUNetv2 package is not installed. Run 'pip install nnunetv2' and try again."
        ) from exc

    # nnUNet expects (C, Z, Y, X)
    if imgs.ndim == 4:
        # [T, C, H, W] -> [C, T, H, W]
        np_vol = imgs.cpu().numpy().transpose(1, 0, 2, 3)
    elif imgs.ndim == 3:
        # [T, H, W] -> [1, T, H, W]
        np_vol = imgs.cpu().numpy()[None, ...]
    else:
        raise RuntimeError(f"Unsupported input dimensions: {imgs.shape}")

    spacing = None
    if meta and isinstance(meta, dict):
        spacing = meta.get('spacing', None)
        if spacing is not None and len(spacing) >= 3:
            # SimpleITK spacing is (x, y, z); nnUNet expects (z, y, x)
            spacing = (float(spacing[2]), float(spacing[1]), float(spacing[0]))
    if spacing is None:
        spacing = (1.0, 1.0, 1.0)

    def _build_predictor(use_mirroring=True, perform_everything_on_device=True, step_size=tile_step_size):
        pred = nnUNetPredictor(
            tile_step_size=step_size,
            use_gaussian=True,
            use_mirroring=use_mirroring,
            device=device,
            perform_everything_on_device=perform_everything_on_device,
            verbose=False,
            verbose_preprocessing=False,
            allow_tqdm=False,
        )
        pred.initialize_from_trained_model_folder(
            model_path,
            use_folds=folds if folds is not None else (0,),
            checkpoint_name=checkpoint_name,
        )
        return pred

    def _align_channels(pred, arr):
        expected = None
        try:
            cfg = getattr(pred, 'configuration_manager', None)
            if cfg and hasattr(cfg, 'normalization_schemes'):
                expected = len(cfg.normalization_schemes)
        except Exception:
            expected = None
        if expected and arr.shape[0] != expected:
            if arr.shape[0] > expected:
                arr = arr[:expected]
            else:
                pad = expected - arr.shape[0]
                pad_block = np.repeat(arr[:1], pad, axis=0)
                arr = np.concatenate([arr, pad_block], axis=0)
        return arr, expected

    def _predict(pred, arr):
        if hasattr(pred, 'predict_from_ndarray'):
            return pred.predict_from_ndarray(arr, original_spacing=spacing, tiling=True)[0]
        return pred.predict_single_npy_array(arr, image_properties={'spacing': spacing}, save_or_return_probabilities=False)

    # First attempt: mirroring on, on-device processing (fast path)
    try:
        predictor = _build_predictor(use_mirroring=True, perform_everything_on_device=True, step_size=tile_step_size)
        np_vol_aligned, expected_channels = _align_channels(predictor, np_vol)
        print(f"[nnUNetv2] device={device}, np_vol shape={np_vol_aligned.shape}, spacing={spacing}, expected_channels={expected_channels}")
        seg = _predict(predictor, np_vol_aligned)
    except RuntimeError as exc:
        msg = str(exc).lower()
        if ('out of memory' in msg or 'cuda out of memory' in msg or 'cudnn error' in msg):
            # Retry with safer settings
            predictor = _build_predictor(use_mirroring=False, perform_everything_on_device=False, step_size=0.25)
            np_vol_aligned, expected_channels = _align_channels(predictor, np_vol)
            print(f"[nnUNetv2][retry-lowmem] device={device}, np_vol shape={np_vol_aligned.shape}, spacing={spacing}, expected_channels={expected_channels}")
            seg = _predict(predictor, np_vol_aligned)
        else:
            raise RuntimeError(f"nnUNetv2 inference failed: {exc}") from exc
    except Exception as exc:  # pragma: no cover - runtime error path
        raise RuntimeError(f"nnUNetv2 inference failed: {exc}") from exc

    if isinstance(seg, (list, tuple)):
        seg = seg[0]
    if seg.ndim == 4 and seg.shape[0] == 1:
        seg = seg[0]
    if seg.ndim != 3:
        raise RuntimeError(f"nnUNetv2 output shape unexpected: {seg.shape}")

    return seg.astype(np.int16)
