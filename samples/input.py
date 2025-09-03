#!/usr/bin/env python3
import argparse
from pathlib import Path
import math
from typing import List, Tuple, Sequence, Optional

import matplotlib.patches as patches
import torch
import torchvision
from torchvision.ops import roi_align
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import json
import torch.nn.functional as F  # for padding
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self):
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        return self.pe.permute(1, 0, 2)


# ----------------- DINO augmentations with LOCAL inpainting -----------------
class DataAugmentationDINO_InpaintLocals:
    """
    Returns:
      images_views: list[Tensor 3xHxW] length M = 2 + local_crops_number
      boxes_views:  list[list[Optional[list[float]]]]
                    per-view list ALIGNED to original boxes; each entry is
                    [x,y,w,h, orig_x,orig_y,orig_w,orig_h] in floats, where:
                      - [x,y,w,h] are in the resized *view* space
                      - [orig_x,orig_y,orig_w,orig_h] are in the *input image* space
                    or None depending on containment rule:
                      * GLOBAL views: boxes intersecting the crop are kept (clipped).
                      * LOCAL  views: ONLY boxes FULLY INSIDE the crop are kept.
      patch_src_views: list[Tensor 3xHxW]     per-view tensors used to cut patches (locals: pre-inpaint)
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
                 solarization_p_global2=0.2,
                 # NEW: keep only fully-contained boxes in local views
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
        full_contained_only: bool = False,   # NEW
    ) -> List[Optional[List[float]]]:
        """
        Apply crop->resize->optional hflip to [x,y,w,h] boxes.

        If full_contained_only=False (default): keep boxes that INTERSECT the crop (clipped).
        If full_contained_only=True:  keep ONLY boxes that are FULLY INSIDE the crop.

        Returns an index-aligned list (same length/order as boxes_xywh):
          - Each element is [x,y,w,h, orig_x,orig_y,orig_w,orig_h] with:
               [x,y,w,h] in the resized view space
               [orig_*] in the input image space
          - Or None if filtered out by the rule above.
        """
        out: List[Optional[List[float]]] = []
        if not boxes_xywh:
            return out

        crop_x0, crop_y0 = float(crop_left), float(crop_top)
        crop_x1, crop_y1 = crop_x0 + float(crop_w), crop_y0 + float(crop_h)
        scale = float(out_size) / float(crop_w)  # crop is square (w==h), use w for scale

        for x, y, w, h in boxes_xywh:
            x0 = float(x); y0 = float(y); x1 = x0 + float(w); y1 = y0 + float(h)

            eps = 1e-6
            if full_contained_only:
                # require full containment (robust to float rounding)
                if not (x0 >= crop_x0 - eps and y0 >= crop_y0 - eps and
                        x1 <= crop_x1 + eps and y1 <= crop_y1 + eps):
                    out.append(None)
                    continue
                ix0, iy0, ix1, iy1 = x0, y0, x1, y1
            else:
                # keep any intersection (clip to crop)
                ix0 = max(x0, crop_x0)
                iy0 = max(y0, crop_y0)
                ix1 = min(x1, crop_x1)
                iy1 = min(y1, crop_y1)
                if ix1 <= ix0 or iy1 <= iy0:
                    out.append(None)
                    continue

            # shift to crop space
            cx0 = ix0 - crop_x0
            cy0 = iy0 - crop_y0
            cw  = ix1 - ix0
            ch  = iy1 - iy0

            # resize to out_size
            rx0 = cx0 * scale
            ry0 = cy0 * scale
            rw  = cw  * scale
            rh  = ch  * scale

            # optional horizontal flip in the resized viewport
            if do_hflip:
                rx0 = float(out_size) - (rx0 + rw)

            out.append([rx0, ry0, rw, rh, float(x), float(y), float(w), float(h)])

        return out

    def _apply_geom(self, img_pil: Image.Image, mask_pil: Image.Image,
                    out_size: int, scale_range: Tuple[float, float]):
        """Apply SAME crop/resize/flip to image and mask; return images + params."""
        i, j, h, w = self._random_resized_crop_params(img_pil.size, scale_range)  # img_pil.size = (W,H)

        img = TF.resized_crop(img_pil, top=i, left=j, height=h, width=w,
                              size=[out_size, out_size], interpolation=TF.InterpolationMode.BICUBIC)
        msk = TF.resized_crop(mask_pil, top=i, left=j, height=h, width=w,
                              size=[out_size, out_size], interpolation=TF.InterpolationMode.NEAREST)

        do_flip = (np.random.rand() < 0.5)
        if do_flip:
            img = TF.hflip(img)
            msk = TF.hflip(msk)

        params = (i, j, h, w, do_flip)
        return img, msk, params

    def _photo_ops_pil(self, img: Image.Image, global_view_idx):
        """Photometric ops on PIL image; return PIL (no normalization yet)."""
        img = self.flip_and_color(img)
        if global_view_idx == 0:
            if np.random.rand() < self.gb_p1:
                img = self.gb_global1(img)
        elif global_view_idx == 1:
            if np.random.rand() < self.gb_p2:
                img = self.gb_global2(img)
        else:
            if np.random.rand() < 0.5:
                img = self.gb_global1(img)
        return img

    @staticmethod
    def _inpaint_with_mask(img_pil: Image.Image, mask_pil: Image.Image) -> Image.Image:
        """Set pixels to black where mask > 127 (white)."""
        img_np = np.array(img_pil).copy()        # H x W x 3, uint8
        m_np = np.array(mask_pil)                # H x W, uint8
        hole = m_np > 127
        if hole.any():
            img_np[hole] = 0
        return Image.fromarray(img_np, mode="RGB")

    def __call__(self, image_pil: Image.Image, mask_pil: Image.Image, boxes_xywh):
        images: List[torch.Tensor] = []
        boxes_per_view: List[List[Optional[List[float]]]] = []
        patch_src_views: List[torch.Tensor] = []
        crop_rects_views: List[Tuple[int, int, int, int]] = []  # (left, top, width, height) in ORIGINAL space

        # Keep a safe copy of ORIGINAL boxes (pre-geom) as floats
        orig_boxes_xywh = [[float(a) for a in bb] for bb in (boxes_xywh or [])]

        # ---- two GLOBAL views (intersection allowed) ----
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
            boxes_per_view.append(b_g)
            crop_rects_views.append((j, i, w, h))  # left, top, width, height in ORIGINAL

        # ---- LOCAL views (ONLY fully-contained boxes kept) ----
        for _ in range(self.local_crops_number):
            img_l, msk_l, (i, j, h, w, flip) = self._apply_geom(
                image_pil, mask_pil, self.local_size, self.local_crops_scale
            )
            img_l = self._photo_ops_pil(img_l, global_view_idx=None)
            img_l_pre_t = self.normalize(img_l)
            patch_src_views.append(img_l_pre_t)

            img_l_post = self._inpaint_with_mask(img_l, msk_l)
            img_l_post_t = self.normalize(img_l_post)
            images.append(img_l_post_t)

            b_l = self._transform_boxes_xywh(
                orig_boxes_xywh, i, j, h, w, self.local_size, flip,
                full_contained_only=self.locals_keep_fully_contained_only
            )
            boxes_per_view.append(b_l)
            crop_rects_views.append((j, i, w, h))

        # lists
        return images, boxes_per_view, patch_src_views, crop_rects_views


# ----------------- Paired image+mask(+boxes) folder -----------------
class PairedMaskFolder(Dataset):
    """
    Pairs `<name>.png` with `<name>__mask.png` and optional `<name>__bboxes.json` in a flat folder.
    Returns (images_views, boxes_views, patches_views, crop_rects_views, orig_image, orig_mask, orig_size_hw) where:
      images_views: list[Tensor] length M (batched by custom collate)
      boxes_views:  list[list[Optional[[x,y,w,h,orig_x,orig_y,orig_w,orig_h]]]] length M
                    (ragged per-sample, in VIEW pixel space with attached original coords)
      patches_views: list[list[Tensor 3x32x32]] length M (ragged per-sample)
      crop_rects_views: list[Tuple[left, top, width, height]] length M (per view) in ORIGINAL space
      orig_image: Tensor[3,H,W] normalized original image (no crop/resize/flip)
      orig_mask:  Tensor[1,H,W] in {0..1} (no crop/resize/flip)
      orig_size_hw: (H_orig, W_orig) in pixels for the raw image
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
    def _crop_patches_from_view(img_tensor: torch.Tensor,
                                boxes_for_view: List[Optional[List[float]]],
                                out_size: int) -> List[torch.Tensor]:
        """
        Extract 32×32 (out_size) patches using float-precision boxes via ROI Align,
        drop very small boxes, and de-duplicate boxes that quantize to the same rect.
        """
        C, H, W = img_tensor.shape
        if not boxes_for_view:
            return []

        # 1) Filter & collect boxes in xyxy float (view space)
        min_side = 3.0  # pixels in the view; adjust as you like
        rois = []
        coarse_seen = set()  # for simple de-duplication
        patches_list = []

        # Build a 4D tensor for roi_align
        img_batched = img_tensor.unsqueeze(0)  # [1,C,H,W]

        for bb in boxes_for_view:
            if bb is None or len(bb) < 4:
                continue
            x, y, w, h = map(float, bb[:4])
            # Clamp to image bounds (float)
            x0 = max(0.0, min(x, W))
            y0 = max(0.0, min(y, H))
            x1 = max(0.0, min(x + w, W))
            y1 = max(0.0, min(y + h, H))
            if x1 <= x0 or y1 <= y0:
                continue

            # 2) Drop tiny boxes (they will all look alike after resize)
            if (x1 - x0) < min_side or (y1 - y0) < min_side:
                continue

            # 3) Coarse de-dup: round to 0.5px grid and skip repeats
            key = (round(x0 * 2) / 2, round(y0 * 2) / 2, round(x1 * 2) / 2, round(y1 * 2) / 2)
            if key in coarse_seen:
                continue
            coarse_seen.add(key)

            rois.append([0.0, x0, y0, x1, y1])  # roi_align wants [batch_idx, x1, y1, x2, y2]

        if not rois:
            return []

        rois_t = torch.tensor(rois, dtype=img_tensor.dtype, device=img_tensor.device)

        # 4) Float-precise cropping with bilinear sampling
        #    aligned=True gives less shift; spatial_scale=1.0 (coords are in pixels)
        crops = roi_align(
            img_batched, rois_t, output_size=(out_size, out_size),
            spatial_scale=1.0, sampling_ratio=-1, aligned=True
        )  # [N, C, out_size, out_size]

        # Split into list[Tensor[3, out_size, out_size]]
        patches_list = [c for c in crops]

        return patches_list

    def __getitem__(self, idx):
        img_path, msk_path, box_path = self.pairs[idx]
        img = Image.open(img_path).convert("RGB")
        msk = Image.open(msk_path).convert("L")

        H_orig, W_orig = img.size[1], img.size[0]  # PIL size is (W,H)

        boxes_xywh = []
        if box_path is not None:
            try:
                data = json.loads(box_path.read_text())
                for b in data.get("boxes", []):
                    bb = b.get("bbox")
                    if bb and len(bb) == 4:
                        x, y, w, h = [float(v) for v in bb]
                        if w > 0 and h > 0:
                            boxes_xywh.append([x, y, w, h])
            except Exception as e:
                print(f"[WARN] Could not read boxes from {box_path.name}: {e}")

        # Original tensors (no geometry)
        orig_image_t = self.transform.normalize(img)     # [3,H,W]
        orig_mask_t  = TF.to_tensor(msk)                 # [1,H,W], values in [0,1]

        # Augmented views with per-view boxes (with original coords embedded) and crop rects
        images_views, boxes_views, patch_src_views, crop_rects_views = self.transform(img, msk, boxes_xywh)

        # Keep boxes as python lists; preserve None placeholders
        boxes_views = [[(list(map(float, bb)) if bb is not None else None) for bb in view_boxes]
                       for view_boxes in boxes_views]

        # Cut patches from PRE-INPAINT source (locals only)
        patches_views: List[List[torch.Tensor]] = []
        for v_idx, (src_img, view_boxes) in enumerate(zip(patch_src_views, boxes_views)):
            if v_idx >= 2:
                patches = self._crop_patches_from_view(src_img, view_boxes, out_size=self.patch_size)
            else:
                patches = []
            patches_views.append(patches)

        return images_views, boxes_views, patches_views, crop_rects_views, orig_image_t, orig_mask_t, (H_orig, W_orig)


def collate_views_ragged(
    batch: List[
        Tuple[
            List[torch.Tensor],                        # images_views
            List[List[Optional[List[float]]]],         # boxes_views (aligned w/ original, 8 values or None)
            List[List[torch.Tensor]],                  # patches_views
            List[Tuple[int, int, int, int]],           # crop_rects_views (per view) in ORIGINAL space
            torch.Tensor,                              # orig_image (3,H,W)
            torch.Tensor,                              # orig_mask  (1,H,W)
            Tuple[int, int],                           # (H_orig, W_orig)
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
]:
    """
    Collate:
      images_views : list length M with Tensor [B,3,H,W] (stacked per view)
      boxes_views  : list length M with list length B of lists aligned to original boxes
                     (entries are either [x,y,w,h,orig_x,orig_y,orig_w,orig_h] in VIEW/INPUT spaces, or None)
      patches_views: list length M with list length B of ragged lists (NOT stacked)
      crop_rects   : list length M with list length B of (left, top, width, height) in ORIGINAL space
      orig_images  : Tensor [B,3,Hmax,Wmax] of originals (normalized), padded
      orig_masks   : Tensor [B,1,Hmax,Wmax] of masks (0..1), padded
      orig_sizes   : list length B of (H_orig, W_orig) (pre-padding)
    """
    assert len(batch) > 0
    images_batch  = [b[0] for b in batch]
    boxes_batch   = [b[1] for b in batch]
    patches_batch = [b[2] for b in batch]
    crops_batch   = [b[3] for b in batch]
    orig_imgs     = [b[4] for b in batch]
    orig_msks     = [b[5] for b in batch]
    orig_sizes    = [b[6] for b in batch]
    M = len(images_batch[0])

    images_views:  List[torch.Tensor] = []
    boxes_views:   List[List[List[Optional[List[float]]]]] = []
    patches_views: List[List[List[torch.Tensor]]] = []
    crop_rects_views: List[List[Tuple[int, int, int, int]]] = []

    for v in range(M):
        imgs_v = [images_batch[b_idx][v] for b_idx in range(len(batch))]
        images_views.append(torch.stack(imgs_v, dim=0))  # [B,3,H,W]
        boxes_v = [boxes_batch[b_idx][v] for b_idx in range(len(batch))]
        boxes_views.append(boxes_v)
        patches_v = [patches_batch[b_idx][v] for b_idx in range(len(batch))]
        patches_views.append(patches_v)
        crops_v = [crops_batch[b_idx][v] for b_idx in range(len(batch))]
        crop_rects_views.append(crops_v)

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

    return images_views, boxes_views, patches_views, crop_rects_views, orig_images, orig_masks, orig_sizes


# ----------------- viz utils (images only) -----------------
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)

def denorm_to_uint8(img_chw: torch.Tensor) -> np.ndarray:
    x = img_chw.detach().cpu()
    x = x * IMAGENET_STD + IMAGENET_MEAN
    x = x.clamp(0,1).numpy()
    return (np.transpose(x, (1,2,0)) * 255.0).round().astype(np.uint8)

def mask_to_float01(msk: torch.Tensor) -> np.ndarray:
    """Accept [1,H,W] or [H,W]; return numpy float32 HxW in [0,1]."""
    if msk.ndim == 3 and msk.shape[0] == 1:
        m = msk[0].detach().cpu().numpy().astype(np.float32)
    elif msk.ndim == 2:
        m = msk.detach().cpu().numpy().astype(np.float32)
    else:
        raise ValueError(f"mask_to_float01 expects [1,H,W] or [H,W], got {tuple(msk.shape)}")
    m = np.clip(m, 0.0, 1.0)
    return m

def _make_patch_strip(patches_list: List[torch.Tensor], gap: int = 2, max_patches: int = 10) -> np.ndarray:
    if not patches_list:
        return np.zeros((32, 1, 3), dtype=np.uint8)
    patches_list = patches_list[:max_patches]
    imgs = [denorm_to_uint8(p) for p in patches_list]
    h = imgs[0].shape[0]
    gap_col = np.zeros((h, gap, 3), dtype=np.uint8)
    row = []
    for k, im in enumerate(imgs):
        if k > 0: row.append(gap_col)
        row.append(im)
    return np.concatenate(row, axis=1)

def save_grid(
    images_chw: List[torch.Tensor],
    cols: int,
    out_path: Path,
    title=None,
    show=False,
    boxes_per_image: List[List[Sequence[float]]] = None,
    patches_per_image: List[List[torch.Tensor]] = None,
    masks_per_image: List[torch.Tensor] = None,
    rects_per_image: List[Optional[Sequence[float]]] = None,  # one extra rect per image (x,y,w,h)
    max_patches: int = 10,
    dpi: int = 240,
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
    Draw images (CHW tensors) in a grid. Optionally overlay:
      - per-image mask (single-channel, 0..1) with a colormap + alpha
      - per-image list of boxes
      - single crop rectangle per image
      - per-image strip of patches (under each image)
    """
    n = len(images_chw)
    cols = max(1, cols)
    rows_img = math.ceil(n / cols)
    has_patch_rows = patches_per_image is not None

    sizes = [img.shape[-2:] for img in images_chw]
    Hmax = max(h for h, w in sizes)
    Wmax = max(w for h, w in sizes)
    patch_h_est = 32
    px_per_in = 40.0

    cell_w_in = Wmax / px_per_in
    cell_h_in = Hmax / px_per_in
    patch_h_in = (patch_h_est / px_per_in) if has_patch_rows else 0.0

    fig_w_in = max(6.0, cols * cell_w_in)
    fig_h_in = max(4.0, rows_img * (cell_h_in + patch_h_in + 0.4))

    if has_patch_rows:
        fig, axes = plt.subplots(rows_img * 2, cols, figsize=(fig_w_in, fig_h_in), dpi=dpi)
    else:
        fig, axes = plt.subplots(rows_img, cols, figsize=(fig_w_in, fig_h_in), dpi=dpi)
    if not isinstance(axes, np.ndarray):
        axes = np.array([[axes]])
    axes = axes.reshape((-1, cols))

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

            # optional mask overlay
            if masks_per_image is not None and i < len(masks_per_image) and masks_per_image[i] is not None:
                m_arr = mask_to_float01(masks_per_image[i])
                ax_img.imshow(m_arr, alpha=mask_alpha, cmap=mask_cmap)

            ax_strip = axes[2*r + 1, c]
            plist = patches_per_image[i] if i < len(patches_per_image) else []
            strip = _make_patch_strip(plist, gap=2, max_patches=max_patches) if plist is not None else np.zeros((patch_h_est, 1, 3), dtype=np.uint8)
            ax_strip.imshow(strip); ax_strip.axis("off")
        else:
            ax = axes[r, c]
            ax.imshow(arr); ax.axis("off")
            # optional mask overlay
            if masks_per_image is not None and i < len(masks_per_image) and masks_per_image[i] is not None:
                m_arr = mask_to_float01(masks_per_image[i])
                ax.imshow(m_arr, alpha=mask_alpha, cmap=mask_cmap)

    if title:
        fig.suptitle(title)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)

    # Draw overlays only when show=True (same behavior as before)
    if show and (boxes_per_image is not None or rects_per_image is not None):
        for i in range(min(n, rows_img * cols)):
            r, c = divmod(i, cols)
            ax_img = axes[2*r, c] if has_patch_rows else axes[r, c]
            arr_h, arr_w = denorm_to_uint8(images_chw[i]).shape[:2]

            # boxes
            this_boxes = boxes_per_image[i] if (boxes_per_image is not None and i < len(boxes_per_image)) else []
            for bb in this_boxes:
                if bb is None or len(bb) < 4:
                    continue
                x, y, w, h = map(float, bb[:4])
                x = max(0.0, min(x, arr_w)); y = max(0.0, min(y, arr_h))
                w = max(0.0, min(w, arr_w - x)); h = max(0.0, min(h, arr_h - y))
                if w <= 0 or h <= 0:
                    continue
                ax_img.add_patch(
                    patches.Rectangle(
                        (x, y), w, h,
                        linewidth=box_linewidth,
                        edgecolor=box_edgecolor,
                        facecolor="none",
                        alpha=box_alpha,
                    )
                )

            # crop rectangle (one per image)
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

    if show:
        plt.show()
    plt.close(fig)


def summarize_batch(images, boxes, patches, max_views=8, max_batch=8, max_show_patches=1):
    """
    images : list[M] of Tensor[B,3,H,W]
    boxes  : list[M] of list[B] of lists aligned to original boxes
             (VIEW space; entries are [x,y,w,h,orig_x,orig_y,orig_w,orig_h] or None)
    patches: list[M] of list[B] of ragged lists of Tensor[3,32,32]
    """
    assert isinstance(images, list), f"images is {type(images)}"
    M = len(images)
    B = images[0].shape[0]

    print(f"\n=== images ===")
    print(f"views M = {M}")
    for v, t in enumerate(images[:max_views]):
        print(f"  images[{v}]: tensor {tuple(t.shape)}  # [B,3,H,W]")

    print(f"\n=== boxes (view space, aligned to original; 8-values or None) ===")
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
    ap = argparse.ArgumentParser("DINO inputs with LOCAL inpainting (mask→black) + bbox tracking + mask overlay on originals + crop rect overlay (locals keep only fully-contained boxes)")
    ap.add_argument("--data-path", required=True, help="Folder with <name>.png + <name>__mask.png + (optional) <name>__bboxes.json")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--local-crops-number", type=int, default=8)
    ap.add_argument("--global-crops-scale", type=float, nargs=2, default=(0.4, 1.0))
    ap.add_argument("--local-crops-scale", type=float, nargs=2, default=(0.05, 0.4))
    ap.add_argument("--max-samples", type=int, default=4, help="How many samples from the batch to visualize")
    ap.add_argument("--out-dir", type=str, default="./dino_inpaint_locals_demo")
    ap.add_argument("--show", action="store_true")
    # target side for rescaled original images (square)
    ap.add_argument("--orig-display-size", type=int, default=214,
                    help="Side length to which original images are resized for visualization; original boxes/masks are scaled to this size.")
    # mask overlay styling
    ap.add_argument("--mask-alpha", type=float, default=0.35)
    ap.add_argument("--mask-cmap", type=str, default="spring")
    # toggle: keep only fully-contained boxes in local views (default True)
    ap.add_argument("--locals-full-contained-only", action="store_true", default=True)
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
    images, boxes, patches, crop_rects, orig_images_pad, orig_masks_pad, orig_sizes = next(iter(loader))
    summarize_batch(images, boxes, patches)
    assert isinstance(images, list)
    M = len(images)
    B = images[0].shape[0]
    K = min(args.max_samples, B)

    print(f"\nTotal views M = 2 (global, NOT inpainted) + {args.local_crops_number} (local, INPAINTED) = {M}")
    for i in range(M):
        tag = "global" if i < 2 else f"local{i-2:02d}"
        counts_boxes_valid = [sum(1 for bb in b if bb is not None) for b in boxes[i]]
        counts_patches = [len(p) for p in patches[i]]
        print(f"view[{i:02d}] {tag} | tensor = {tuple(images[i].shape)} | boxes per sample (first {K}): {counts_boxes_valid[:K]} | patches per sample: {counts_patches[:K]}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Per-view grids (clean saves; boxes overlay only on-screen)
    for i in range(M):
        tag = "global" if i < 2 else f"local{i - 2:02d}"
        imgs = [images[i][j] for j in range(K)]
        boxes_for_view = None
        patches_for_view = None
        if i >= 2:  # local views: draw boxes on-screen + show patch row (saved & on-screen)
            boxes_for_view = [boxes[i][j] for j in range(K)]
            patches_for_view = [patches[i][j] for j in range(K)]
        save_grid(
            imgs, cols=min(K, 4),
            out_path=out_dir / f"view_{i:02d}_{tag}.png",
            title=f"View {i:02d} ({tag}) — first {K} samples",
            show=args.show,
            boxes_per_image=boxes_for_view,
            patches_per_image=patches_for_view,
        )

    # ---- Originals RESCALED + original boxes/masks SCALED to the same size + CROP RECT overlay ----
    S = int(args.orig_display_size)

    def crop_to_real_and_resize_img(img_t: torch.Tensor, H: int, W: int, S: int) -> torch.Tensor:
        cropped = img_t[:, :H, :W]  # [3,H,W]
        return TF.resize(cropped, [S, S], interpolation=TF.InterpolationMode.BICUBIC, antialias=True)

    def crop_to_real_and_resize_mask(msk_t: torch.Tensor, H: int, W: int, S: int) -> torch.Tensor:
        cropped = msk_t[:, :H, :W]  # [1,H,W]
        return TF.resize(cropped, [S, S], interpolation=TF.InterpolationMode.NEAREST)

    rescaled_orig_imgs: List[torch.Tensor] = []
    rescaled_orig_masks: List[torch.Tensor] = []
    rescaled_orig_boxes: List[List[Sequence[float]]] = []
    rescaled_crop_rects: List[Optional[Sequence[float]]] = []

    embed_C = 128
    img_size = 224
    pos_embed_x = PositionalEncoding(d_model=embed_C // 4, dropout=0., max_len=img_size)
    pos_embed_y = PositionalEncoding(d_model=embed_C // 4, dropout=0., max_len=img_size)
    pos_embed_h = PositionalEncoding(d_model=embed_C // 4, dropout=0., max_len=img_size)
    pos_embed_w = PositionalEncoding(d_model=embed_C // 4, dropout=0., max_len=img_size)

    print('!!! ', pos_embed_x)

    for j in range(K):
        H_j, W_j = orig_sizes[j]
        # crop out padding and resize to SxS
        rescaled_img = crop_to_real_and_resize_img(orig_images_pad[j], H_j, W_j, S)
        rescaled_msk = crop_to_real_and_resize_mask(orig_masks_pad[j], H_j, W_j, S)
        rescaled_orig_imgs.append(rescaled_img)
        rescaled_orig_masks.append(rescaled_msk)

        # Original boxes corresponding to FIRST GLOBAL CROP alignment (view 0)
        vb = boxes[0][j] if len(boxes) > 0 else []
        scaled_list = []
        if W_j > 0 and H_j > 0:
            sx = S / float(W_j)
            sy = S / float(H_j)
            for bb in vb:
                if bb is None or len(bb) < 8:
                    continue
                ox, oy, ow, oh = map(float, bb[4:8])  # original image space
                # scale to SxS resized original
                x_s = ox * sx
                y_s = oy * sy
                w_s = ow * sx
                h_s = oh * sy
                if w_s > 0 and h_s > 0:
                    scaled_list.append([x_s, y_s, w_s, h_s])
            # First GLOBAL crop rectangle (view 0) scaled to SxS
            if len(crop_rects) > 0 and j < len(crop_rects[0]):
                l0, t0, w0, h0 = crop_rects[0][j]
                rescaled_crop = [l0 * sx, t0 * sy, w0 * sx, h0 * sy]
            else:
                rescaled_crop = None
        else:
            rescaled_crop = None

        rescaled_orig_boxes.append(scaled_list)
        rescaled_crop_rects.append(rescaled_crop)

    save_grid(
        rescaled_orig_imgs, cols=min(K, 4),
        out_path=out_dir / "original_firstcrop_boxes_masks_rescaled.png",
        title=f"Rescaled originals to {S}×{S} with scaled original boxes + mask overlay + crop rectangle (first {K} samples)",
        show=args.show,  # overlays only on-screen
        boxes_per_image=rescaled_orig_boxes,
        masks_per_image=rescaled_orig_masks,
        rects_per_image=rescaled_crop_rects,
        patches_per_image=None,
        mask_alpha=args.mask_alpha,
        mask_cmap=args.mask_cmap,
        rect_edgecolor="cyan",
        rect_linewidth=1.0,
        rect_alpha=0.9,
        rect_linestyle="--",
    )

    # Panel of ALL views for sample #0 (no masks)
    panel_imgs = [v[0] for v in images]
    panel_boxes = [([] if i < 2 else boxes[i][0]) for i in range(M)]
    panel_patches = [([] if i < 2 else patches[i][0]) for i in range(M)]
    save_grid(
        panel_imgs, cols=6,
        out_path=out_dir / "all_views_sample0.png",
        title="All views for sample #0 (globals intact, locals inpainted)",
        show=args.show,
        boxes_per_image=panel_boxes,
        patches_per_image=panel_patches,
    )


if __name__ == "__main__":
    main()
