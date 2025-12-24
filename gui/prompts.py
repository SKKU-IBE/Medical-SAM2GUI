"""Prompt normalization and mask helpers used by GUI classes."""

from __future__ import annotations

from typing import Dict, Iterable, Tuple

import numpy as np
import torch


def normalize_box_prompts(box_prompts: Dict) -> Dict:
    """Convert heterogeneous box prompts into plain Python lists."""
    normalized: Dict = {}
    if not box_prompts:
        return normalized

    for frame_idx, objs in box_prompts.items():
        normalized[frame_idx] = {}
        for obj_id, box in objs.items():
            if isinstance(box, torch.Tensor):
                normalized[frame_idx][obj_id] = box.cpu().numpy().tolist()
            elif isinstance(box, np.ndarray):
                normalized[frame_idx][obj_id] = box.tolist()
            elif isinstance(box, (list, tuple)):
                normalized[frame_idx][obj_id] = list(box)
            else:
                normalized[frame_idx][obj_id] = box
    return normalized


def normalize_point_prompts(point_prompts: Dict) -> Dict:
    """Convert point prompts into list-of-tuples format (x, y, label)."""
    if not point_prompts:
        return {}

    normalized: Dict = {}
    for frame_idx, objs in point_prompts.items():
        normalized[frame_idx] = {}
        for obj_id, prompt_data in objs.items():
            if isinstance(prompt_data, dict):
                points_list = []
                points_data = prompt_data.get("points")
                if points_data is not None:
                    for pt_info in points_data:
                        if isinstance(pt_info, (list, tuple)) and len(pt_info) >= 3:
                            x, y, label = pt_info[:3]
                            if isinstance(x, torch.Tensor):
                                x = x.item()
                            if isinstance(y, torch.Tensor):
                                y = y.item()
                            if isinstance(label, torch.Tensor):
                                label = label.item()
                            points_list.append((x, y, label))
                        elif isinstance(pt_info, (list, tuple)) and len(pt_info) == 2:
                            x, y = pt_info[:2]
                            if isinstance(x, torch.Tensor):
                                x = x.item()
                            if isinstance(y, torch.Tensor):
                                y = y.item()
                            points_list.append((x, y, 1))
                normalized[frame_idx][obj_id] = points_list
            elif isinstance(prompt_data, list):
                points_list = []
                for pt_info in prompt_data:
                    if isinstance(pt_info, (list, tuple)) and len(pt_info) >= 3:
                        x, y, label = pt_info[:3]
                        if isinstance(x, torch.Tensor):
                            x = x.item()
                        if isinstance(y, torch.Tensor):
                            y = y.item()
                        if isinstance(label, torch.Tensor):
                            label = label.item()
                        points_list.append((x, y, label))
                normalized[frame_idx][obj_id] = points_list
            else:
                normalized[frame_idx][obj_id] = []
    return normalized


def make_initial_label_stack(
    video_segments: Dict[int, Dict[int, torch.Tensor]],
    obj_ids: Iterable[int],
    n_frames: int,
    spatial_shape: Tuple[int, int],
) -> np.ndarray:
    """Build initial label volume aligning masks with frame index."""
    stack = np.zeros((n_frames,) + spatial_shape, dtype=np.uint8)
    if not video_segments:
        return stack

    obj_id_map = {oid: i + 1 for i, oid in enumerate(obj_ids)}

    for frame_idx, segments in video_segments.items():
        for obj_id, logits in segments.items():
            if isinstance(logits, torch.Tensor):
                mask = (logits.sigmoid() > 0.5).cpu().numpy()
            else:
                mask = logits > 0.5

            if mask.ndim == 3 and mask.shape[0] == 1:
                mask = mask[0]

            stack[frame_idx][mask > 0] = obj_id_map.get(obj_id, obj_id)
    return stack
