"""Auto segmentation helper."""
import torch


def auto_segmentation(pack, net, device, method='det'):
    """Perform auto-segmentation using MedSAM2 network from pack data."""
    net.eval()
    results = []

    with torch.no_grad():
        imgs = pack['images']
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
                "meta": meta
            })

        elif method == 'seg':
            prompts = pack['prompts']
            valid_slices = [s for s, objs in prompts.items() if objs]
            if not valid_slices:
                results.append({
                    "patient_id": patient_id,
                    "imgs": imgs.cpu(),
                    "video_segments": {},
                    "prompts": {}
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
                "meta": meta
            })
        else:
            raise ValueError(f"Unsupported method: {method}")

    return results
