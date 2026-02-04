"""Auto segmentation helper."""
import time
import numpy as np
import torch
import cv2
from typing import Optional

from gui.metrics import UsageMetricsRecorder


def auto_segmentation(
    pack,
    net,
    device,
    method='det',
    det_model='sam2_det',
    seg_model='sam2_seg',
    nnunet_model_path=None,
    metrics: Optional[UsageMetricsRecorder] = None,
):
    """Perform auto-segmentation.

    det_model/seg_model are chosen from the UI. The current det_model uses the SAM2 flow.
    If seg_model == "nnUNetv2", run nnUNetv2 inference directly to show masks.
    """
    net.eval()
    results = []

    with torch.no_grad():
        stage_start = time.time()
        # Use raw volume for nnUNet inference, but keep resized stack for viewer/prompts
        imgs = pack['images']
        imgs_nnunet = pack.get('images_nnunet', None)
        if imgs.ndim == 5:
            imgs = imgs.squeeze(0)
        patient_id = pack['meta']['patient']
        meta = pack['meta']
        imgs = imgs.to(device=device, dtype=torch.float32)

        video_segments = {}
        box_prompts = {}
        point_prompts = {}

        if method in ['cls-det', 'det']:
            cls_labels = pack['cls_labels']
            bboxes = pack['bboxes']
            positive_slices = [s for s, objs in cls_labels.items() if objs]
            if not positive_slices:
                results.append({
                    "patient_id": patient_id,
                    "meta": meta,
                    "start_idx": None,
                    "end_idx": None,
                    "imgs": imgs.cpu(),
                    "video_segments": {},
                    "box_prompts": {}
                })
            start_idx, end_idx = min(positive_slices), max(positive_slices)
            for s in range(start_idx, end_idx + 1):
                if s in bboxes and bboxes[s]:
                    box_prompts[s] = {oid: torch.tensor(box, device=device) for oid, box in bboxes[s].items()}
            sub_imgs = imgs[start_idx:end_idx + 1]
            state = net.val_init_state(imgs_tensor=sub_imgs)
            for frame_idx, objs in box_prompts.items():
                local_idx = frame_idx - start_idx
                for oid, box in objs.items():
                    net.train_add_new_bbox(
                        inference_state=state,
                        frame_idx=local_idx,
                        obj_id=oid,
                        bbox=box,
                        clear_old_points=False
                    )
            for out_local_idx, out_oids, out_logits in net.propagate_in_video(state, start_frame_idx=0):
                global_idx = start_idx + out_local_idx
                video_segments[global_idx] = {oid: mask.cpu() for oid, mask in zip(out_oids, out_logits)}
            net.reset_state(state)
            results.append({
                "patient_id": patient_id,
                "imgs": imgs.cpu(),
                "video_segments": video_segments,
                "box_prompts": box_prompts,
                "start_idx": start_idx,
                "end_idx": end_idx,
                "meta": meta,
                "det_model": det_model,
            })

        elif method == 'seg':
            # seg_model == 'sam2_seg' -> 기존 SAM2 전파 흐름 유지
            if seg_model == 'nnUNetv2':
                from gui.nnunetv2_inference import run_nnunetv2_inference
                seg_input = imgs_nnunet if imgs_nnunet is not None else imgs
                seg_mask = run_nnunetv2_inference(
                    imgs=seg_input,
                    meta=meta,
                    model_path=nnunet_model_path,
                    device=device,
                )
                # seg_mask: (Z, H_raw, W_raw) with integer labels
                start_idx, end_idx = 0, seg_mask.shape[0] - 1
                obj_ids = sorted(int(x) for x in np.unique(seg_mask) if x != 0)
                target_h, target_w = imgs.shape[-2], imgs.shape[-1]
                for z in range(seg_mask.shape[0]):
                    slice_map = seg_mask[z].astype(np.uint8)
                    # Resize mask to viewer resolution (nearest)
                    if slice_map.shape[0] != target_h or slice_map.shape[1] != target_w:
                        slice_map = cv2.resize(slice_map, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
                    per_obj = {}
                    for oid in obj_ids:
                        mask = (slice_map == oid).astype(np.float32)
                        if mask.any():
                            per_obj[oid] = torch.from_numpy(mask)
                    if per_obj:
                        video_segments[z] = per_obj
                results.append({
                    "patient_id": patient_id,
                    "imgs": imgs.cpu(),  # viewer stack (resized)
                    "video_segments": video_segments,
                    "prompts": {},
                    "start_idx": start_idx,
                    "end_idx": end_idx,
                    "meta": meta,
                    "seg_model": seg_model,
                })
            else:
                prompts = pack['prompts']
                valid_slices = [s for s, objs in prompts.items() if objs]
                if not valid_slices:
                    results.append({
                        "patient_id": patient_id,
                        "imgs": imgs.cpu(),
                        "video_segments": {},
                        "prompts": {},
                        "seg_model": seg_model,
                    })
                start_idx, end_idx = min(valid_slices), max(valid_slices)
                sel_prompts = {s: prompts[s] for s in valid_slices if start_idx <= s <= end_idx}
                sub_imgs = imgs[start_idx:end_idx + 1]
                state = net.val_init_state(imgs_tensor=sub_imgs)
                for frame_idx, objs in sel_prompts.items():
                    local_idx = frame_idx - start_idx
                    for oid, prm in objs.items():
                        if 'bboxes' in prm and prm['bboxes'] is not None:
                            bbox = torch.tensor(prm['bboxes'], device=device)
                            net.train_add_new_bbox(
                                inference_state=state,
                                frame_idx=local_idx,
                                obj_id=oid,
                                bbox=bbox,
                                clear_old_points=False
                            )
                        if 'points' in prm and prm['points'] is not None:
                            points_data = prm['points']
                            if isinstance(points_data[0], (list, tuple)) and len(points_data[0]) >= 2:
                                coords = [[pt[0], pt[1]] for pt in points_data]
                                labels = [pt[2] if len(pt) > 2 else 1 for pt in points_data]
                                points_tensor = torch.tensor(coords, device=device)
                                labels_tensor = torch.tensor(labels, dtype=torch.long, device=device)
                                net.train_add_new_points(
                                    inference_state=state,
                                    frame_idx=local_idx,
                                    obj_id=oid,
                                    points=points_tensor,
                                    labels=labels_tensor,
                                    clear_old_points=False
                                )
                for out_local_idx, out_oids, out_logits in net.propagate_in_video(state, start_frame_idx=0):
                    global_idx = start_idx + out_local_idx
                    video_segments[global_idx] = {oid: mask.cpu() for oid, mask in zip(out_oids, out_logits)}
                net.reset_state(state)
                results.append({
                    "patient_id": patient_id,
                    "imgs": imgs.cpu(),
                    "video_segments": video_segments,
                    "prompts": sel_prompts,
                    "start_idx": start_idx,
                    "end_idx": end_idx,
                    "meta": meta,
                    "seg_model": seg_model,
                })
        else:
            raise ValueError(f"Unsupported method: {method}")

        if metrics and metrics.is_active():
            metrics.record_stage(
                'auto_segmentation',
                stage_start,
                time.time(),
                method=method,
                slice_count=int(end_idx - start_idx + 1) if 'start_idx' in locals() and start_idx is not None else None,
                box_prompt_slices=len(box_prompts) if method in ['cls-det', 'det'] else len(sel_prompts) if 'sel_prompts' in locals() else 0,
                point_prompt_slices=0 if method in ['cls-det', 'det'] else len(sel_prompts) if 'sel_prompts' in locals() else 0,
            )

    return results
