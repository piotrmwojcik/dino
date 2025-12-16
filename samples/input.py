#!/usr/bin/env python3
"""
DINO-style multi-crop pipeline with local inpainting + bbox tracking.

Key points:
- Each per-view box is [x_view, y_view, w_view, h_view, orig_x_rescaled, orig_y_rescaled, orig_w_rescaled, orig_h_rescaled].
  * The first 4 values are in the **view space**.
  * The trailing 4 values are the **original-image boxes rescaled to a fixed global_size×global_size** reference frame.
- DO NOT draw any boxes on the global crops.
- Draw the **original boxes** (using the rescaled tail) on the **original images panel**,
  which is resized (not padded) to S×S. We convert tail coords from global_size to S on the fly.
- **NEW**: For local views, the saved per-view grid includes a **patch strip** under every image,
  showing the first N extracted patches (ROI Align outputs).

Usage:
python3 run.py \
  --data-path /path/to/folder \
  --batch-size 2 \
  --local-crops-number 6 \
  --out-dir ./dino_views_demo \
  --show
"""

import argparse
from pathlib import Path
import math
from typing import List, Tuple, Sequence, Optional

import matplotlib.patches as patches
import torch
from torchvision.ops import roi_align
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import json
import torch.nn.functional as F  # for padding


# ----------------- DINO augmentations with LOCAL inpainting -----------------
class DataAugmentationDINO_InpaintLocals:
    """
    Returns:
      images_views: list[Tensor 3xHxW] length M = 2 + local_crops_number
      boxes_views:  list[list[Optional[list[float]]]]
                    per-view list aligned to original boxes; each entry is
                    [x,y,w,h, orig_x,orig_y,orig_w,orig_h] where:
                      - [x,y,w,h] are in the resized *view* space
                      - [orig_x,orig_y,orig_w,orig_h] are ORIGINAL boxes
                        RESCALED to a fixed global_size×global_size frame
                    or None depending on containment rule.
      patch_src_views: list[Tensor 3xHxW]  per-view tensors used to cut patches (locals: pre-inpaint)
      crop_rects_views: list[Tuple[left, top, width, height]] in ORIGINAL image space, one per view
    """
    def __init__(self,
                 global_crops_scale=(0.4, 1.0),
                 local_crops_scale=(0.05, 0.4),
                 local_crops_number=8,
                 global_size=224,
                 local_size=96,
                 gaussian_blur_p_global1=1.0,
                 gaussian_blur_p_global2=0.1,
                 locals_keep_fully_contained_only: bool = True):
        self.global_size = int(global_size)
        self.local_size = int(local_size)
        self.global_crops_scale = tuple(global_crops_scale)
        self.local_crops_scale = tuple(local_crops_scale)
        self.local_crops_number = int(local_crops_number)
        self.locals_keep_fully_contained_only = bool(locals_keep_fully_contained_only)

        self.flip_and_color = T.Compose([
            T.RandomApply([T.ColorJitter(0.3, 0.2, 0.3, 0.15)], p=0.8),
            T.RandomGrayscale(p=0.1),
        ])
        self.normalize = T.Compose([
            T.ToTensor(),
            T.Normalize((0.485, 0.456, 0.406),
                        (0.229, 0.224, 0.225)),
        ])
        self.gb_global1 = T.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0))
        self.gb_global2 = T.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0))
        self.gb_p1 = gaussian_blur_p_global1
        self.gb_p2 = gaussian_blur_p_global2

    @staticmethod
    def _random_resized_crop_params(img_size: Tuple[int, int], scale: Tuple[float, float]):
        """Square RandomResizedCrop params (aspect ratio = 1). img_size=(W,H)."""
        W, H = img_size
        for _ in range(10):
            target_area = np.random.uniform(scale[0], scale[1]) * H * W
            side = int(round(np.sqrt(target_area)))
            if 0 < side <= H and side <= W:
                i = np.random.randint(0, H - side + 1)  # top
                j = np.random.randint(0, W - side + 1)  # left
                return i, j, side, side
        side = min(H, W)
        i = (H - side) // 2
        j = (W - side) // 2
        return i, j, side, side

    @staticmethod
    def _transform_boxes_xywh(
        boxes_xywh: Sequence[Sequence[float]],
        crop_top: int, crop_left: int, crop_h: int, crop_w: int,
        out_size: int,
        do_hflip: bool,
        full_contained_only: bool = False,
    ) -> List[Optional[List[float]]]:
        """
        Apply crop->resize->optional hflip to [x,y,w,h] boxes.

        If full_contained_only=False: keep boxes that INTERSECT the crop (clipped).
        If full_contained_only=True:  keep ONLY boxes that are FULLY INSIDE the crop.

        Returns a list aligned to `boxes_xywh`:
          - Each element is [x,y,w,h, orig_x,orig_y,orig_w,orig_h], or None.
            The tail (orig_*) is initially in original pixels and will be rewritten.
        """
        out: List[Optional[List[float]]] = []
        if not boxes_xywh:
            return out

        crop_x0, crop_y0 = float(crop_left), float(crop_top)
        crop_x1, crop_y1 = crop_x0 + float(crop_w), crop_y0 + float(crop_h)
        scale = float(out_size) / float(crop_w)  # square crop

        for x, y, w, h in boxes_xywh:
            x0 = float(x); y0 = float(y); x1 = x0 + float(w); y1 = y0 + float(h)

            eps = 1e-6
            if full_contained_only:
                if not (x0 >= crop_x0 - eps and y0 >= crop_y0 - eps and
                        x1 <= crop_x1 + eps and y1 <= crop_y1 + eps):
                    out.append(None)
                    continue
                ix0, iy0, ix1, iy1 = x0, y0, x1, y1
            else:
                ix0 = max(x0, crop_x0); iy0 = max(y0, crop_y0)
                ix1 = min(x1, crop_x1); iy1 = min(y1, crop_y1)
                if ix1 <= ix0 or iy1 <= iy0:
                    out.append(None); continue

            cx0 = ix0 - crop_x0; cy0 = iy0 - crop_y0
            cw  = ix1 - ix0;     ch  = iy1 - iy0

            rx0 = cx0 * scale; ry0 = cy0 * scale
            rw  = cw  * scale;   rh  = ch  * scale

            if do_hflip:
                rx0 = float(out_size) - (rx0 + rw)

            # Tail temporarily in original pixels; will be rewritten after.
            out.append([rx0, ry0, rw, rh, float(x), float(y), float(w), float(h)])

        return out

    @staticmethod
    def _inpaint_with_mask(img_pil: Image.Image, mask_pil: Image.Image) -> Image.Image:
        """Set pixels to black where mask > 127 (white)."""
        img_np = np.array(img_pil).copy()        # H x W x 3
        m_np = np.array(mask_pil)                # H x W
        hole = m_np > 127
        if hole.any(): img_np[hole] = 0
        return Image.fromarray(img_np, mode="RGB")

    @staticmethod
    def _rescale_orig_tail_inplace(boxes_for_view, W_orig: int, H_orig: int, target: int):
        """Rewrite bb[4:8] in-place from original pixels to target×target space."""
        if not boxes_for_view or W_orig <= 0 or H_orig <= 0:
            return
        sx = float(target) / float(W_orig)
        sy = float(target) / float(H_orig)
        for bb in boxes_for_view:
            if bb is None or len(bb) < 8:
                continue
            bb[4] *= sx
            bb[5] *= sy
            bb[6] *= sx
            bb[7] *= sy

    def _apply_geom(self, img_pil: Image.Image, mask_pil: Image.Image,
                    out_size: int, scale_range: Tuple[float, float]):
        """Apply SAME crop/resize/flip to image and mask; return images + params."""
        i, j, h, w = self._random_resized_crop_params(img_pil.size, scale_range)  # img_pil.size=(W,H)

        img = TF.resized_crop(img_pil, top=i, left=j, height=h, width=w,
                              size=[out_size, out_size], interpolation=TF.InterpolationMode.BICUBIC)
        msk = TF.resized_crop(mask_pil, top=i, left=j, height=h, width=w,
                              size=[out_size, out_size], interpolation=TF.InterpolationMode.NEAREST)

        do_flip = (np.random.rand() < 0.5)
        if do_flip:
            img = TF.hflip(img)
            msk = TF.hflip(msk)

        return img, msk, (i, j, h, w, do_flip)

    def _photo_ops_pil(self, img: Image.Image, global_view_idx):
        img = T.Compose([
            T.RandomApply([T.ColorJitter(0.3, 0.2, 0.3, 0.15)], p=0.8),
            T.RandomGrayscale(p=0.1),
        ])(img)
        if global_view_idx == 0:
            if np.random.rand() < self.gb_p1: img = self.gb_global1(img)
        elif global_view_idx == 1:
            if np.random.rand() < self.gb_p2: img = self.gb_global2(img)
        else:
            if np.random.rand() < 0.5: img = self.gb_global1(img)
        return img

    def __call__(self, image_pil: Image.Image, mask_pil: Image.Image, boxes_xywh):
        images: List[torch.Tensor] = []
        boxes_per_view: List[List[Optional[List[float]]]] = []
        patch_src_views: List[torch.Tensor] = []
        crop_rects_views: List[Tuple[int, int, int, int]] = []

        W_orig, H_orig = image_pil.size  # PIL size=(W,H)
        orig_boxes_xywh = [[float(a) for a in bb] for bb in (boxes_xywh or [])]

        # GLOBAL views
        for gv in (0, 1):
            img_g, msk_g, (i, j, h, w, flip) = self._apply_geom(
                image_pil, mask_pil, self.global_size, self.global_crops_scale
            )
            img_g = self._photo_ops_pil(img_g, global_view_idx=gv)
            img_g_t = self.normalize(img_g)
            images.append(img_g_t)
            patch_src_views.append(img_g_t)

            b_g = self._transform_boxes_xywh(
                orig_boxes_xywh, i, j, h, w, self.global_size, flip,
                full_contained_only=True
            )
            self._rescale_orig_tail_inplace(b_g, W_orig, H_orig, self.global_size)

            boxes_per_view.append(b_g)
            crop_rects_views.append((j, i, w, h))  # left, top, width, height

        # LOCAL views
        for _ in range(self.local_crops_number):
            img_l, msk_l, (i, j, h, w, flip) = self._apply_geom(
                image_pil, mask_pil, self.local_size, self.local_crops_scale
            )
            img_l_src = self.normalize(self._photo_ops_pil(img_l.copy(), global_view_idx=None))
            patch_src_views.append(img_l_src)

            img_l_post = self._inpaint_with_mask(img_l, msk_l)
            img_l_post_t = self.normalize(self._photo_ops_pil(img_l_post, global_view_idx=None))
            images.append(img_l_post_t)

            b_l = self._transform_boxes_xywh(
                orig_boxes_xywh, i, j, h, w, self.local_size, flip,
                full_contained_only=self.locals_keep_fully_contained_only
            )
            self._rescale_orig_tail_inplace(b_l, W_orig, H_orig, self.global_size)

            boxes_per_view.append(b_l)
            crop_rects_views.append((j, i, w, h))

        return images, boxes_per_view, patch_src_views, crop_rects_views


# ----------------- Paired image+mask(+boxes) folder -----------------
class PairedMaskFolder(Dataset):
    """
    Returns:
      (images_views, boxes_views, patches_views, crop_rects_views,
       orig_image, orig_mask, orig_size_hw, patch_indices_views, orig_boxes_xywh)

      orig_boxes_xywh is the full list of original boxes from disk (unfiltered).
    """
    def __init__(self, root: str, transform: DataAugmentationDINO_InpaintLocals, patch_size: int = 32):
        self.root = Path(root)
        self.transform = transform
        self.patch_size = int(patch_size)
        all_pngs = sorted(self.root.glob("*.png"))

        pairs = []
        for p in all_pngs:
            n = p.name
            if "__overlay" in n or "__cell_" in n or "__boxed" in n:
                continue
            if n.endswith("__mask.png"):
                continue
            m = p.with_name(p.stem + "__mask.png")
            if m.exists():
                b = p.with_name(p.stem + "__bboxes.json")
                pairs.append((p, m, b if b.exists() else None))
        if not pairs:
            raise RuntimeError(f"No paired images found in {self.root}")
        self.pairs = pairs

    def __len__(self): return len(self.pairs)

    @staticmethod
    def _crop_patches_from_view(
        img_tensor: torch.Tensor,
        boxes_for_view: List[Optional[List[float]]],
        out_size: int
    ) -> Tuple[List[torch.Tensor], List[int]]:
        """
        Extract patches via ROI Align and return (patches, box_indices),
        where box_indices[k] is the index in `boxes_for_view` that produced patches[k].
        Tiny boxes are dropped and coarse duplicates skipped.
        """
        C, H, W = img_tensor.shape
        if not boxes_for_view:
            return [], []

        min_side = 3.0
        rois = []
        src_indices: List[int] = []
        coarse_seen = set()
        img_batched = img_tensor.unsqueeze(0)  # [1,C,H,W]

        for idx, bb in enumerate(boxes_for_view):
            if bb is None or len(bb) < 4:
                continue
            x, y, w, h = map(float, bb[:4])

            x0 = max(0.0, min(x, W))
            y0 = max(0.0, min(y, H))
            x1 = max(0.0, min(x + w, W))
            y1 = max(0.0, min(y + h, H))
            if x1 <= x0 or y1 <= y0:
                continue

            if (x1 - x0) < min_side or (y1 - y0) < min_side:
                continue

            key = (round(x0 * 2) / 2, round(y0 * 2) / 2, round(x1 * 2) / 2, round(y1 * 2) / 2)
            if key in coarse_seen:
                continue
            coarse_seen.add(key)

            rois.append([0.0, x0, y0, x1, y1])  # [batch_idx, x1, y1, x2, y2]
            src_indices.append(idx)

        if not rois:
            return [], []

        rois_t = torch.tensor(rois, dtype=img_tensor.dtype, device=img_tensor.device)
        crops = roi_align(
            img_batched, rois_t, output_size=(out_size, out_size),
            spatial_scale=1.0, sampling_ratio=-1, aligned=True
        )  # [N, C, out_size, out_size]

        patches_list = [c for c in crops]
        return patches_list, src_indices

    def __getitem__(self, idx):
        img_path, msk_path, box_path = self.pairs[idx]
        img = Image.open(img_path).convert("RGB")
        msk = Image.open(msk_path).convert("L")

        H_orig, W_orig = img.size[1], img.size[0]  # PIL size is (W,H)

        # Load original boxes (unfiltered, original pixels)
        orig_boxes_list = []
        if box_path is not None:
            try:
                data = json.loads(box_path.read_text())
                for b in data.get("boxes", []):
                    bb = b.get("bbox")
                    if bb and len(bb) == 4:
                        x, y, w, h = [float(v) for v in bb]
                        if w > 0 and h > 0:
                            orig_boxes_list.append([x, y, w, h])
            except Exception as e:
                print(f"[WARN] Could not read boxes from {box_path.name}: {e}")

        orig_image_t = self.transform.normalize(img)     # [3,H,W]
        orig_mask_t  = TF.to_tensor(msk)                 # [1,H,W], values in [0,1]

        images_views, boxes_views, patch_src_views, crop_rects_views = self.transform(img, msk, orig_boxes_list)

        boxes_views = [[(list(map(float, bb)) if bb is not None else None) for bb in view_boxes]
                       for view_boxes in boxes_views]

        patches_views: List[List[torch.Tensor]] = []
        patch_indices_views: List[List[int]] = []

        for v_idx, (src_img, view_boxes) in enumerate(zip(patch_src_views, boxes_views)):
            if v_idx >= 2:
                patches, idxs = self._crop_patches_from_view(src_img, view_boxes, out_size=self.patch_size)
            else:
                patches, idxs = [], []
            patches_views.append(patches)
            patch_indices_views.append(idxs)

        return (images_views, boxes_views, patches_views, crop_rects_views,
                orig_image_t, orig_mask_t, (H_orig, W_orig), patch_indices_views, orig_boxes_list)


def collate_views_ragged(
    batch: List[
        Tuple[
            List[torch.Tensor],                        # images_views
            List[List[Optional[List[float]]]],         # boxes_views (aligned w/ original, 8 values or None)
            List[List[torch.Tensor]],                  # patches_views
            List[Tuple[int, int, int, int]],           # crop_rects_views (per view)
            torch.Tensor,                              # orig_image (3,H,W)
            torch.Tensor,                              # orig_mask  (1,H,W)
            Tuple[int, int],                           # (H_orig, W_orig)
            List[List[int]],                           # patch_indices_views (ragged)
            List[List[float]],                         # orig_boxes_xywh (unfiltered, original pixels)
        ]
    ]
) -> Tuple[
    List[torch.Tensor],
    List[List[List[Optional[List[float]]]]],
    List[List[List[torch.Tensor]]],
    List[List[Tuple[int, int, int, int]]],
    torch.Tensor,
    torch.Tensor,
    List[Tuple[int, int]],
    List[List[List[int]]],
    List[List[List[float]]],                           # orig_boxes per-sample
]:
    """Collate ragged structures; pass-through lists for boxes/patches/crops/indices/orig_boxes."""
    assert len(batch) > 0
    images_batch  = [b[0] for b in batch]
    boxes_batch   = [b[1] for b in batch]
    patches_batch = [b[2] for b in batch]
    crops_batch   = [b[3] for b in batch]
    orig_imgs     = [b[4] for b in batch]
    orig_msks     = [b[5] for b in batch]
    orig_sizes    = [b[6] for b in batch]
    patch_idx_b   = [b[7] for b in batch]
    orig_boxes_b  = [b[8] for b in batch]
    M = len(images_batch[0])

    images_views:  List[torch.Tensor] = []
    boxes_views:   List[List[List[Optional[List[float]]]]] = []
    patches_views: List[List[List[torch.Tensor]]] = []
    crop_rects_views: List[List[Tuple[int, int, int, int]]] = []
    patch_indices_views: List[List[List[int]]] = []

    for v in range(M):
        imgs_v = [images_batch[b_idx][v] for b_idx in range(len(batch))]
        images_views.append(torch.stack(imgs_v, dim=0))  # [B,3,H,W]

        boxes_v  = [boxes_batch[b_idx][v]   for b_idx in range(len(batch))]
        patches_v= [patches_batch[b_idx][v] for b_idx in range(len(batch))]
        crops_v  = [crops_batch[b_idx][v]   for b_idx in range(len(batch))]
        idxs_v   = [patch_idx_b[b_idx][v]   for b_idx in range(len(batch))]

        boxes_views.append(boxes_v)
        patches_views.append(patches_v)
        crop_rects_views.append(crops_v)
        patch_indices_views.append(idxs_v)

    # ---- Pad & stack original images/masks to common size (top-left anchored) ----
    sizes = [img.shape[-2:] for img in orig_imgs]  # [(H,W), ...]
    Hmax = max(h for h, w in sizes)
    Wmax = max(w for h, w in sizes)

    def _pad_to_img(img: torch.Tensor, H: int, W: int) -> torch.Tensor:
        _, h, w = img.shape
        pad_right = W - w
        pad_bottom = H - h
        if pad_right == 0 and pad_bottom == 0:
            return img
        return F.pad(img, (0, pad_right, 0, pad_bottom), mode="constant", value=0.0)

    def _pad_to_mask(msk: torch.Tensor, H: int, W: int) -> torch.Tensor:
        _, h, w = msk.shape
        pad_right = W - w
        pad_bottom = H - h
        if pad_right == 0 and pad_bottom == 0:
            return msk
        return F.pad(msk, (0, pad_right, 0, pad_bottom), mode="constant", value=0.0)

    orig_imgs_padded = [_pad_to_img(t, Hmax, Wmax) for t in orig_imgs]
    orig_msks_padded = [_pad_to_mask(t, Hmax, Wmax) for t in orig_msks]
    orig_images = torch.stack(orig_imgs_padded, dim=0)   # [B,3,Hmax,Wmax]
    orig_masks  = torch.stack(orig_msks_padded, dim=0)   # [B,1,Hmax,Wmax]

    orig_boxes_out = [[bb for bb in orig_boxes_b[b_idx]] for b_idx in range(len(batch))]

    return (images_views, boxes_views, patches_views, crop_rects_views,
            orig_images, orig_masks, orig_sizes, patch_indices_views, orig_boxes_out)


# ----------------- viz utils (with PATCH STRIPS) -----------------
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)

def denorm_to_uint8(img_chw: torch.Tensor) -> np.ndarray:
    x = img_chw.detach().cpu()
    x = x * IMAGENET_STD + IMAGENET_MEAN
    x = x.clamp(0,1).numpy()
    return (np.transpose(x, (1,2,0)) * 255.0).round().astype(np.uint8)

def mask_to_float01(msk: torch.Tensor) -> np.ndarray:
    if msk.ndim == 3 and msk.shape[0] == 1:
        m = msk[0].detach().cpu().numpy().astype(np.float32)
    elif msk.ndim == 2:
        m = msk.detach().cpu().numpy().astype(np.float32)
    else:
        raise ValueError(f"mask_to_float01 expects [1,H,W] or [H,W], got {tuple(msk.shape)}")
    m = np.clip(m, 0.0, 1.0)
    return m

def _make_patch_strip(patches_list: List[torch.Tensor], gap: int = 2, max_patches: int = 10) -> np.ndarray:
    """
    Convert a list of CHW patch tensors (normalized) into a single HxW×3 strip image.
    """
    if not patches_list:
        # fallback tiny placeholder
        return np.zeros((32, 1, 3), dtype=np.uint8)
    patches_list = patches_list[:max_patches]
    imgs = [denorm_to_uint8(p) for p in patches_list]
    h = imgs[0].shape[0]
    gap_col = np.zeros((h, gap, 3), dtype=np.uint8)
    row = []
    for k, im in enumerate(imgs):
        if k > 0:
            row.append(gap_col)
        row.append(im)
    return np.concatenate(row, axis=1)

def save_grid(
    images_chw: List[torch.Tensor],
    cols: int,
    out_path: Path,
    title=None,
    boxes_per_image: List[List[Sequence[float]]] = None,
    masks_per_image: List[torch.Tensor] = None,
    rects_per_image: List[Optional[Sequence[float]]] = None,  # one extra rect per image (x,y,w,h)
    patches_per_image: List[List[torch.Tensor]] = None,       # NEW: list per image of list-of-patch-tensors
    max_patches: int = 10,                                     # NEW: max patches per strip
    box_edgecolor: str = "red",
    box_linewidth: float = 0.8,
    box_alpha: float = 0.6,
    mask_alpha: float = 0.35,
    mask_cmap: str = "spring",
    rect_edgecolor: str = "cyan",
    rect_linewidth: float = 1.0,
    rect_alpha: float = 0.9,
    rect_linestyle: str = "--",
):
    """
    Draw images + overlays (boxes/masks/rects) and SAVE them.
    If `patches_per_image` is provided, a second row is added under each image showing a patch strip.
    """
    n = len(images_chw)
    cols = max(1, cols)
    rows_img = math.ceil(n / cols)
    has_patch_rows = patches_per_image is not None

    sizes = [img.shape[-2:] for img in images_chw]
    Hmax = max(h for h, w in sizes)
    Wmax = max(w for h, w in sizes)
    px_per_in = 40.0

    # estimate patch strip height from first available patch
    patch_h_est = 32
    if has_patch_rows:
        for lst in patches_per_image:
            if lst:
                patch_h_est = denorm_to_uint8(lst[0]).shape[0]
                break

    cell_w_in = Wmax / px_per_in
    cell_h_in = Hmax / px_per_in
    patch_h_in = (patch_h_est / px_per_in) if has_patch_rows else 0.0

    fig_w_in = max(6.0, cols * cell_w_in)
    fig_h_in = max(4.0, rows_img * (cell_h_in + patch_h_in + 0.4))

    if has_patch_rows:
        fig, axes = plt.subplots(rows_img * 2, cols, figsize=(fig_w_in, fig_h_in), dpi=240)
    else:
        fig, axes = plt.subplots(rows_img, cols, figsize=(fig_w_in, fig_h_in), dpi=240)
    if not isinstance(axes, np.ndarray):
        axes = np.array([[axes]])
    axes = axes.reshape((-1, cols))

    # Draw images & optional patch strips
    for i in range(rows_img * cols):
        r, c = divmod(i, cols)
        if i >= n:
            if has_patch_rows:
                axes[2*r, c].axis("off"); axes[2*r+1, c].axis("off")
            else:
                axes[r, c].axis("off")
            continue

        arr = denorm_to_uint8(images_chw[i])
        if has_patch_rows:
            ax_img = axes[2*r, c]
            ax_img.imshow(arr); ax_img.axis("off")

            # patch strip beneath
            ax_strip = axes[2*r + 1, c]
            plist = patches_per_image[i] if (patches_per_image is not None and i < len(patches_per_image)) else []
            strip = _make_patch_strip(plist, gap=2, max_patches=max_patches) if plist is not None else np.zeros((patch_h_est, 1, 3), dtype=np.uint8)
            ax_strip.imshow(strip); ax_strip.axis("off")
        else:
            ax = axes[r, c]
            ax.imshow(arr); ax.axis("off")

    # Overlays (on the image axes)
    for i in range(min(n, rows_img * cols)):
        r, c = divmod(i, cols)
        ax_img = axes[2*r, c] if has_patch_rows else axes[r, c]
        arr_h, arr_w = denorm_to_uint8(images_chw[i]).shape[:2]

        # Masks
        if masks_per_image is not None and i < len(masks_per_image) and masks_per_image[i] is not None:
            m_arr = mask_to_float01(masks_per_image[i])
            ax_img.imshow(m_arr, alpha=mask_alpha, cmap=mask_cmap)

        # Boxes
        this_boxes = boxes_per_image[i] if (boxes_per_image is not None and i < len(boxes_per_image)) else []
        for bb in (this_boxes or []):
            if bb is None or len(bb) < 4: continue
            x, y, w, h = map(float, bb[:4])
            x = max(0.0, min(x, arr_w)); y = max(0.0, min(y, arr_h))
            w = max(0.0, min(w, arr_w - x)); h = max(0.0, min(h, arr_h - y))
            if w <= 0 or h <= 0: continue
            ax_img.add_patch(
                patches.Rectangle(
                    (x, y), w, h,
                    linewidth=box_linewidth,
                    edgecolor=box_edgecolor,
                    facecolor="none",
                    alpha=box_alpha,
                )
            )

        # Extra rect
        if rects_per_image is not None and i < len(rects_per_image) and rects_per_image[i] is not None:
            rx, ry, rw, rh = map(float, rects_per_image[i])
            rx = max(0.0, min(rx, arr_w)); ry = max(0.0, min(ry, arr_h))
            rw = max(0.0, min(rw, arr_w - rx)); rh = max(0.0, min(rh, arr_h - ry))
            if rw > 0 and rh > 0:
                ax_img.add_patch(
                    patches.Rectangle(
                        (rx, ry), rw, rh,
                        linewidth=rect_linewidth,
                        edgecolor=rect_edgecolor,
                        facecolor="none",
                        alpha=rect_alpha,
                        linestyle=rect_linestyle,
                    )
                )

    if title:
        fig.suptitle(title)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def summarize_batch(images, boxes, patches, max_views=8, max_batch=8, max_show_patches=1):
    assert isinstance(images, list), f"images is {type(images)}"
    M = len(images)
    B = images[0].shape[0]

    print(f"\n=== images ===")
    print(f"views M = {M}")
    for v, t in enumerate(images[:max_views]):
        print(f"  images[{v}]: tensor {tuple(t.shape)}  # [B,3,H,W]")

    print(f"\n=== boxes (view space + orig_tail rescaled to global_size) ===")
    assert isinstance(boxes, list) and len(boxes) == M
    for v in range(min(M, max_views)):
        bv = boxes[v]
        assert len(bv) == B, f"boxes[{v}] batch size {len(bv)} != {B}"
        counts_valid = [sum(1 for bb in bv[b] if bb is not None) for b in range(min(B, max_batch))]
        print(f"  boxes[{v}]: batch={len(bv)} | per-sample VALID box counts (first {max_batch}): {counts_valid}")

    print(f"\n=== patches (per view) ===")
    assert isinstance(patches, list) and len(patches) == M
    for v in range(min(M, max_views)):
        pv = patches[v]
        assert len(pv) == B, f"patches[{v}] batch size {len(pv)} != {B}"
        pcounts = [len(pv[b]) for b in range(min(B, max_batch))]
        print(f"  patches[{v}]: batch={len(pv)} | patches per sample (first {max_batch}): {pcounts}")
        shown = 0
        for b in range(min(B, max_batch)):
            for k in range(min(len(pv[b]), max_show_patches)):
                print(f"    example patches[{v}][{b}][{k}].shape = {tuple(pv[b][k].shape)}")
                shown += 1
                if shown >= max_show_patches:
                    break
            if shown >= max_show_patches:
                break


# ----------------- main -----------------
def main():
    ap = argparse.ArgumentParser("DINO inputs with LOCAL inpainting (mask→black) + bbox tracking. Original boxes are stored rescaled to global_size×global_size and drawn only on the resized original images.")
    ap.add_argument("--data-path", required=True, help="Folder with <name>.png + <name>__mask.png + (optional) <name>__bboxes.json")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--local-crops-number", type=int, default=8)
    ap.add_argument("--global-crops-scale", type=float, nargs=2, default=(0.4, 1.0))
    ap.add_argument("--local-crops-scale", type=float, nargs=2, default=(0.2, 0.8))
    ap.add_argument("--max-samples", type=int, default=4, help="How many samples from the batch to visualize")
    ap.add_argument("--out-dir", type=str, default="./dino_inpaint_locals_demo")
    ap.add_argument("--show", action="store_true", help="Overlays are saved to files; show just also pops a window if available.")
    ap.add_argument("--orig-display-size", type=int, default=214, help="Side length to which original images are resized for visualization (no padding).")
    ap.add_argument("--mask-alpha", type=float, default=0.35)
    ap.add_argument("--mask-cmap", type=str, default="spring")
    ap.add_argument("--locals-full-contained-only", action="store_true", default=True)
    ap.add_argument("--max-patches", type=int, default=10, help="Max patches per image strip in visualization")
    args = ap.parse_args()

    transform = DataAugmentationDINO_InpaintLocals(
        global_crops_scale=tuple(args.global_crops_scale),
        local_crops_scale=tuple(args.local_crops_scale),
        local_crops_number=args.local_crops_number,
        locals_keep_fully_contained_only=args.locals_full_contained_only,
    )
    dataset = PairedMaskFolder(args.data_path, transform=transform, patch_size=32)

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_views_ragged,
    )

    # one batch
    (images, boxes, patches, crop_rects,
     orig_images_pad, orig_masks_pad, orig_sizes,
     patch_indices, orig_boxes_all) = next(iter(loader))
    summarize_batch(images, boxes, patches)

    assert isinstance(images, list)
    M = len(images)
    B = images[0].shape[0]
    K = min(args.max_samples, B)

    print(f"\nTotal views M = 2 (global) + {args.local_crops_number} (local) = {M}")
    for i in range(M):
        tag = "global" if i < 2 else f"local{i-2:02d}"
        counts_boxes_valid = [sum(1 for bb in b if bb is not None) for b in boxes[i]]
        counts_patches = [len(p) for p in patches[i]]
        print(f"view[{i:02d}] {tag} | tensor = {tuple(images[i].shape)} | boxes per sample (first {K}): {counts_boxes_valid[:K]} | patches per sample: {counts_patches[:K]}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ============== Per-view grids ==============
    # Globals: no boxes, no patch strips.
    # Locals: draw per-view boxes AND show patch strips under each image.
    for i in range(M):
        tag = "global" if i < 2 else f"local{i - 2:02d}"
        imgs = [images[i][j] for j in range(K)]

        if i < 2:
            boxes_for_view = None
            masks_for_view = None
            patches_for_view = None
        else:
            boxes_for_view = [boxes[i][j] for j in range(K)]
            masks_for_view = None
            patches_for_view = [patches[i][j] for j in range(K)]

        save_grid(
            imgs, cols=min(K, 4),
            out_path=out_dir / f"view_{i:02d}_{tag}.png",
            title=f"View {i:02d} ({tag}) — first {K} samples",
            boxes_per_image=boxes_for_view,
            masks_per_image=masks_for_view,
            rects_per_image=None,
            patches_per_image=patches_for_view,     # <--- patch strips saved
            max_patches=args.max_patches,
            mask_alpha=args.mask_alpha,
            mask_cmap=args.mask_cmap,
        )

    # ============== Originals panel (RESIZED to S×S, draw ORIGINAL boxes) ==============
    S = int(args.orig_display_size)
    global_size = images[0].shape[-1]  # should equal transform.global_size (square)

    def crop_to_real_and_resize_img(img_t: torch.Tensor, H: int, W: int, S: int) -> torch.Tensor:
        cropped = img_t[:, :H, :W]  # [3,H,W]
        return TF.resize(cropped, [S, S], interpolation=TF.InterpolationMode.BICUBIC, antialias=True)

    def crop_to_real_and_resize_mask(msk_t: torch.Tensor, H: int, W: int, S: int) -> torch.Tensor:
        cropped = msk_t[:, :H, :W]  # [1,H,W]
        return TF.resize(cropped, [S, S], interpolation=TF.InterpolationMode.NEAREST)

    rescaled_orig_imgs: List[torch.Tensor] = []
    rescaled_orig_masks: List[torch.Tensor] = []
    boxes_for_originals_panel: List[List[Sequence[float]]] = []

    # Build boxes for original panel from tails (global_size coords) and scale to S.
    tail_to_S_scale = float(S) / float(global_size)

    for j in range(K):
        H_j, W_j = orig_sizes[j]
        rescaled_img = crop_to_real_and_resize_img(orig_images_pad[j], H_j, W_j, S)
        rescaled_msk = crop_to_real_and_resize_mask(orig_masks_pad[j], H_j, W_j, S)
        rescaled_orig_imgs.append(rescaled_img)
        rescaled_orig_masks.append(rescaled_msk)

        tail_boxes_global_size = []
        if len(boxes) >= 1 and len(boxes[0]) > j:
            for bb in boxes[0][j]:
                if bb is None or len(bb) < 8:
                    continue
                ox_g, oy_g, ow_g, oh_g = bb[4], bb[5], bb[6], bb[7]
                tail_boxes_global_size.append([ox_g, oy_g, ow_g, oh_g])

        # supplement from orig (if any missing), scaled orig->global_size
        if orig_boxes_all and j < len(orig_boxes_all):
            Worig = float(W_j); Horig = float(H_j)
            sx_g = float(global_size) / Worig if Worig > 0 else 1.0
            sy_g = float(global_size) / Horig if Horig > 0 else 1.0
            for ox, oy, ow, oh in orig_boxes_all[j]:
                b_g = [ox * sx_g, oy * sy_g, ow * sx_g, oh * sy_g]
                if not any(
                    abs(b_g[0]-t[0]) < 0.5 and abs(b_g[1]-t[1]) < 0.5 and
                    abs(b_g[2]-t[2]) < 0.5 and abs(b_g[3]-t[3]) < 0.5
                    for t in tail_boxes_global_size
                ):
                    tail_boxes_global_size.append(b_g)

        boxes_S = [[b[0]*tail_to_S_scale, b[1]*tail_to_S_scale, b[2]*tail_to_S_scale, b[3]*tail_to_S_scale]
                   for b in tail_boxes_global_size]
        boxes_for_originals_panel.append(boxes_S)

    save_grid(
        rescaled_orig_imgs, cols=min(K, 4),
        out_path=out_dir / "originals_with_boxes_SxS.png",
        title=f"Original images resized to {S}×{S} with ORIGINAL boxes (tail @ global_size → scaled to S)",
        boxes_per_image=boxes_for_originals_panel,
        masks_per_image=rescaled_orig_masks,
        rects_per_image=None,
    )

    # Optional: panel of ALL views for sample #0 (locals show view boxes + patch strips)
    panel_imgs = [v[0] for v in images]
    panel_boxes = [([] if i < 2 else boxes[i][0]) for i in range(M)]
    panel_patches = [([] if i < 2 else patches[i][0]) for i in range(M)]
    save_grid(
        panel_imgs, cols=6,
        out_path=out_dir / "all_views_sample0.png",
        title="All views for sample #0 (globals: no boxes; locals: view boxes + patch strips)",
        boxes_per_image=panel_boxes,
        masks_per_image=None,
        rects_per_image=None,
        patches_per_image=panel_patches,
        max_patches=args.max_patches,
    )

    if args.show:
        # In some environments, this may be headless; files are saved regardless.
        plt.show()


if __name__ == "__main__":
    main()
