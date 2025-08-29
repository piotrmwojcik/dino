#!/usr/bin/env python3
import argparse
from pathlib import Path
import math
from typing import List, Tuple, Sequence

import matplotlib.patches as patches
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import json


# ----------------- DINO augmentations with LOCAL inpainting -----------------
class DataAugmentationDINO_InpaintLocals:
    """
    Returns:
      images_views: list[Tensor 3xHxW] length M = 2 + local_crops_number
      boxes_views:  list[list[list[float]]]   per-view list of [x,y,w,h] (floats) in the resized view space

    Geometric ops (square RandomResizedCrop + optional hflip) are applied
    IDENTICALLY to image, mask, and boxes.
    Photometric ops (jitter/blur/solarization) apply to image only.
    Local views are additionally inpainted using the aligned mask.
    """
    def __init__(self,
                 global_crops_scale=(0.4, 1.0),
                 local_crops_scale=(0.05, 0.4),
                 local_crops_number=8,
                 global_size=224,
                 local_size=96,
                 gaussian_blur_p_global1=1.0,
                 gaussian_blur_p_global2=0.1,
                 solarization_p_global2=0.2):
        self.global_size = int(global_size)
        self.local_size = int(local_size)
        self.global_crops_scale = tuple(global_crops_scale)
        self.local_crops_scale = tuple(local_crops_scale)
        self.local_crops_number = int(local_crops_number)

        # image-only photometric ops
        self.flip_and_color = T.Compose([
            T.RandomHorizontalFlip(p=0.5),  # (photometric) extra flip AFTER the shared geom? -> remove; we control flips
        ])
        # NOTE: we control all flips ourselves to keep boxes consistent,
        # so remove RandomHorizontalFlip from this stack:
        self.flip_and_color = T.Compose([
            T.RandomApply([T.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8),
            T.RandomGrayscale(p=0.2),
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
        self.solar = T.RandomSolarize(threshold=128, p=solarization_p_global2)

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
        do_hflip: bool
    ) -> List[List[float]]:
        """Apply crop->resize->optional hflip to [x,y,w,h] boxes."""
        out = []
        if not boxes_xywh:
            return out
        crop_x0, crop_y0 = float(crop_left), float(crop_top)
        crop_x1, crop_y1 = crop_x0 + float(crop_w), crop_y0 + float(crop_h)
        scale = float(out_size) / float(crop_w)  # crop is square (w==h), use w for scale

        for x, y, w, h in boxes_xywh:
            x0 = float(x); y0 = float(y); x1 = x0 + float(w); y1 = y0 + float(h)
            # intersect with crop
            ix0 = max(x0, crop_x0)
            iy0 = max(y0, crop_y0)
            ix1 = min(x1, crop_x1)
            iy1 = min(y1, crop_y1)
            if ix1 <= ix0 or iy1 <= iy0:
                continue  # box falls outside the crop

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

            out.append([rx0, ry0, rw, rh])
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
            img = self.solar(img)
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

    def __call__(self, image_pil: Image.Image, mask_pil: Image.Image,
                 boxes_xywh):
        images: List[torch.Tensor] = []
        boxes_per_view: List[List[List[float]]] = []

        # ---- two GLOBAL views (no inpainting) ----
        for gv in (0, 1):
            img_g, msk_g, (i, j, h, w, flip) = self._apply_geom(image_pil, mask_pil, self.global_size, self.global_crops_scale)
            img_g = self._photo_ops_pil(img_g, global_view_idx=gv)
            images.append(self.normalize(img_g))
            # transform boxes with the same geom
            b_g = self._transform_boxes_xywh(boxes_xywh or [], i, j, h, w, self.global_size, flip)
            boxes_per_view.append(b_g)

        # ---- LOCAL views (apply inpainting using mask) ----
        for _ in range(self.local_crops_number):
            img_l, msk_l, (i, j, h, w, flip) = self._apply_geom(image_pil, mask_pil, self.local_size, self.local_crops_scale)
            img_l = self._photo_ops_pil(img_l, global_view_idx=None)
            img_l = self._inpaint_with_mask(img_l, msk_l)  # black out mask regions
            images.append(self.normalize(img_l))
            b_l = self._transform_boxes_xywh(boxes_xywh or [], i, j, h, w, self.local_size, flip)
            boxes_per_view.append(b_l)

        return images, boxes_per_view  # lists, length M


# ----------------- Paired image+mask(+boxes) folder -----------------
class PairedMaskFolder(Dataset):
    """
    Pairs `<name>.png` with `<name>__mask.png` and optional `<name>__bboxes.json` in a flat folder.
    Returns (images_views, boxes_views) where:
      images_views: list[Tensor] length M (batched by default collate)
      boxes_views:  list[list[[x,y,w,h], ...]] length M (NOT stacked; ragged per-sample)
    """
    def __init__(self, root: str, transform: DataAugmentationDINO_InpaintLocals):
        self.root = Path(root)
        self.transform = transform
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
                # bbox json optional
                b = p.with_name(p.stem + "__bboxes.json")
                pairs.append((p, m, b if b.exists() else None))
        if not pairs:
            raise RuntimeError(f"No paired images found in {self.root}")
        self.pairs = pairs

    def __len__(self): return len(self.pairs)

    def __getitem__(self, idx):
        img_path, msk_path, box_path = self.pairs[idx]
        img = Image.open(img_path).convert("RGB")
        msk = Image.open(msk_path).convert("L")  # white (255) = region to blacken

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

        images_views, boxes_views = self.transform(img, msk, boxes_xywh)

        # IMPORTANT: make boxes python lists so default collate keeps ragged lists
        boxes_views = [[list(map(float, bb)) for bb in view_boxes] for view_boxes in boxes_views]
        return images_views, boxes_views


def collate_views_ragged(
    batch: List[Tuple[List[torch.Tensor], List[List[List[float]]]]]
) -> Tuple[List[torch.Tensor], List[List[List[List[float]]]]]:
    """
    batch: list of samples, each is (images_views, boxes_views) where
      images_views: list length M with Tensor [3,H,W]
      boxes_views : list length M with list[[x,y,w,h], ...] (ragged)
    Returns:
      images_views: list length M with Tensor [B,3,H,W] (stacked per view)
      boxes_views : list length M with list length B of ragged lists (NOT stacked)
    """
    assert len(batch) > 0
    images_batch = [b[0] for b in batch]  # per-sample images_views
    boxes_batch  = [b[1] for b in batch]  # per-sample boxes_views
    M = len(images_batch[0])

    images_views: List[torch.Tensor] = []
    boxes_views:  List[List[List[List[float]]]] = []

    for v in range(M):
        # Collect v-th view across the batch
        imgs_v  = [images_batch[b_idx][v] for b_idx in range(len(batch))]  # list of [3,H,W]
        # Stack images for this view (sizes are fixed by your pipeline)
        images_views.append(torch.stack(imgs_v, dim=0))  # [B,3,H,W]

        # Keep boxes ragged: list over batch; each item is list[[x,y,w,h], ...]
        boxes_v = [boxes_batch[b_idx][v] for b_idx in range(len(batch))]
        boxes_views.append(boxes_v)

    return images_views, boxes_views

# ----------------- viz utils (images only) -----------------
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)

def denorm_to_uint8(img_chw: torch.Tensor) -> np.ndarray:
    x = img_chw.detach().cpu()
    x = x * IMAGENET_STD + IMAGENET_MEAN
    x = x.clamp(0,1).numpy()
    return (np.transpose(x, (1,2,0)) * 255.0).round().astype(np.uint8)

def save_grid(
    images_chw: List[torch.Tensor],
    cols: int,
    out_path: Path,
    title=None,
    show=False,
    boxes_per_image: List[List[Sequence[float]]] = None,  # per-image list of [x,y,w,h]
):
    """
    Draws bounding boxes only on the on-screen display (when show=True).
    Saved images on disk remain WITHOUT boxes.
    """
    n = len(images_chw)
    cols = max(1, cols)
    rows = math.ceil(n / cols)

    sizes = [img.shape[-2:] for img in images_chw]
    Hmax = max(h for h, w in sizes)
    Wmax = max(w for h, w in sizes)

    fig_w = cols * Wmax / 100
    fig_h = rows * Hmax / 100
    fig, axes = plt.subplots(rows, cols, figsize=(max(4, fig_w/2.5), max(3, fig_h/2.5)))
    if not isinstance(axes, np.ndarray):
        axes = np.array([[axes]])
    axes = axes.reshape(rows, cols)

    # draw images only (no boxes yet)
    for i in range(rows * cols):
        r, c = divmod(i, cols)
        ax = axes[r, c]
        ax.axis("off")
        if i < n:
            arr = denorm_to_uint8(images_chw[i])
            ax.imshow(arr)

    if title:
        fig.suptitle(title)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # --- save a clean version without boxes ---
    fig.savefig(out_path, dpi=200)

    # --- if showing, overlay boxes now (so they appear only on-screen) ---
    if show and boxes_per_image is not None:
        for i in range(min(n, rows * cols)):
            r, c = divmod(i, cols)
            ax = axes[r, c]
            arr_h, arr_w = denorm_to_uint8(images_chw[i]).shape[:2]
            for bb in boxes_per_image[i]:
                if bb is None or len(bb) != 4:
                    continue
                x, y, w, h = map(float, bb)
                # clamp for safety
                x = max(0.0, min(x, arr_w))
                y = max(0.0, min(y, arr_h))
                w = max(0.0, min(w, arr_w - x))
                h = max(0.0, min(h, arr_h - y))
                if w <= 0 or h <= 0:
                    continue
                ax.add_patch(patches.Rectangle((x, y), w, h, linewidth=1.5, edgecolor="red", facecolor="none"))

    # finally show (with boxes if requested), then close
    if show:
        plt.show()
    plt.close(fig)


# ----------------- main -----------------
def main():
    ap = argparse.ArgumentParser("DINO inputs with LOCAL inpainting (mask→black) + bbox tracking")
    ap.add_argument("--data-path", required=True, help="Folder with <name>.png + <name>__mask.png + (optional) <name>__bboxes.json")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--local-crops-number", type=int, default=8)
    ap.add_argument("--global-crops-scale", type=float, nargs=2, default=(0.4, 1.0))
    ap.add_argument("--local-crops-scale", type=float, nargs=2, default=(0.05, 0.4))
    ap.add_argument("--max-samples", type=int, default=4, help="How many samples from the batch to visualize")
    ap.add_argument("--out-dir", type=str, default="./dino_inpaint_locals_demo")
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args()

    transform = DataAugmentationDINO_InpaintLocals(
        global_crops_scale=tuple(args.global_crops_scale),
        local_crops_scale=tuple(args.local_crops_scale),
        local_crops_number=args.local_crops_number,
    )
    dataset = PairedMaskFolder(args.data_path, transform=transform)

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_views_ragged,  # <— key change
    )

    # one batch
    images, boxes = next(iter(loader))  # images: list[tensor Bx3xHxW]; boxes: list[ length B ragged lists ]
    assert isinstance(images, list)
    M = len(images)
    B = images[0].shape[0]
    K = min(args.max_samples, B)

    print(f"\nTotal views M = 2 (global, NOT inpainted) + {args.local_crops_number} (local, INPAINTED) = {M}")
    for i in range(M):
        tag = "global" if i < 2 else f"local{i-2:02d}"
        H, W = images[i].shape[-2:]
        # boxes[i] is a list (batch) of lists (boxes for that sample/view)
        counts = [len(b) for b in boxes[i]]
        print(f"view[{i:02d}] {tag} | tensor = {tuple(images[i].shape)} | boxes per sample (first {K}): {counts[:K]}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for i in range(M):
        tag = "global" if i < 2 else f"local{i - 2:02d}"
        imgs = [images[i][j] for j in range(K)]
        boxes_for_view = None
        if i >= 2:  # only draw on locals (on-screen only)
            boxes_for_view = [boxes[i][j] for j in range(K)]

        save_grid(
            imgs, cols=min(K, 4),
            out_path=out_dir / f"view_{i:02d}_{tag}.png",
            title=f"View {i:02d} ({tag}) — first {K} samples",
            show=args.show,
            boxes_per_image=boxes_for_view,
        )

    # Panel of ALL views for sample #0 (boxes only visible on-screen for locals)
    panel_imgs = [v[0] for v in images]
    panel_boxes = [([] if i < 2 else boxes[i][0]) for i in range(M)]
    save_grid(
        panel_imgs, cols=6,
        out_path=out_dir / "all_views_sample0.png",
        title="All views for sample #0 (globals intact, locals inpainted)",
        show=args.show,
        boxes_per_image=panel_boxes,
    )

    print(f"\nSaved to: {out_dir.resolve()}")
    print("Note: Boxes were cropped/resized/flipped consistently with each view; only LOCAL views were inpainted.")


if __name__ == "__main__":
    main()
