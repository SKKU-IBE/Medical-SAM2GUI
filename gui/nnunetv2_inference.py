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
        raise RuntimeError("nnUNetv2 모델 경로가 지정되지 않았습니다. Patient 설정에서 모델 폴더를 선택하세요.")
    resolved = os.path.abspath(path)
    if not os.path.exists(resolved):
        raise RuntimeError(f"nnUNetv2 모델 경로를 찾을 수 없습니다: {resolved}")
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
    try:
        from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
    except Exception as exc:  # pragma: no cover - import error path
        raise RuntimeError(
            "nnUNetv2 패키지가 설치되어 있지 않습니다. 'pip install nnunetv2' 후 다시 시도하세요."
        ) from exc

    # nnUNet expects (C, Z, Y, X)
    if imgs.ndim == 4:
        # [T, C, H, W] -> [C, T, H, W]
        np_vol = imgs.cpu().numpy().transpose(1, 0, 2, 3)
    elif imgs.ndim == 3:
        # [T, H, W] -> [1, T, H, W]
        np_vol = imgs.cpu().numpy()[None, ...]
    else:
        raise RuntimeError(f"지원하지 않는 입력 차원입니다: {imgs.shape}")

    spacing = None
    if meta and isinstance(meta, dict):
        spacing = meta.get('spacing', None)
        if spacing is not None and len(spacing) >= 3:
            # nnUNet expects (z, y, x)
            spacing = tuple(float(s) for s in spacing[:3])

    predictor = nnUNetPredictor(
        tile_step_size=tile_step_size,
        use_gaussian=True,
        use_mirroring=True,
        device=device,
        perform_everything_on_device=True,
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=True
    )
    predictor.initialize_from_trained_model_folder(
        model_path,
        use_folds=folds if folds is not None else (0,),
        checkpoint_name=checkpoint_name,
    )

    try:
        # API returns a tuple (seg, probs) where seg is int array (Z, Y, X)
        seg, _ = predictor.predict_from_ndarray(
            np_vol,
            original_spacing=spacing,
            tiling=True,
        )
    except Exception as exc:  # pragma: no cover - runtime error path
        raise RuntimeError(f"nnUNetv2 추론 중 오류가 발생했습니다: {exc}") from exc

    if isinstance(seg, (list, tuple)):
        seg = seg[0]
    if seg.ndim == 4 and seg.shape[0] == 1:
        seg = seg[0]
    if seg.ndim != 3:
        raise RuntimeError(f"nnUNetv2 출력 형태가 예상과 다릅니다: {seg.shape}")

    return seg.astype(np.int16)
