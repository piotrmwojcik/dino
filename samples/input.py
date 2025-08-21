#!/usr/bin/env python3
import argparse
from pathlib import Path
import math
from typing import List, Tuple

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


# ----------------- DINO augmentations with LOCAL inpainting -----------------
class DataAugmentationDINO_InpaintLocals:
    """
    Returns ONLY images_views (a list of length M = 2 + local_crops_number).
    - Same random crop/flip is applied to image & mask for geometric consistency.
    - Color/blur/solarization are applied to the image.
    - Inpainting (set to black) is applied **only to local views** using the aligned mask.
    - Output images are normalized (ImageNet) tensors [3,H,W].
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
            T.RandomHorizontalFlip(p=0.5),
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
                i = np.random.randint(0, H - side + 1)
                j = np.random.randint(0, W - side + 1)
                return i, j, side, side
        side = min(H, W)
        i = (H - side) // 2
        j = (W - side) // 2
        return i, j, side, side

    def _apply_geom(self, img_pil: Image.Image, mask_pil: Image.Image,
                    out_size: int, scale_range: Tuple[float, float]):
        """Apply SAME crop/resize/flip to image and mask."""
        i, j, h, w = self._random_resized_crop_params(img_pil.size, scale_range)  # img_pil.size = (W,H)
        img = TF.resized_crop(img_pil, top=i, left=j, height=h, width=w,
                              size=[out_size, out_size], interpolation=TF.InterpolationMode.BICUBIC)
        msk = TF.resized_crop(mask_pil, top=i, left=j, height=h, width=w,
                              size=[out_size, out_size], interpolation=TF.InterpolationMode.NEAREST)
        # shared horizontal flip
        if np.random.rand() < 0.5:
            img = TF.hflip(img)
            msk = TF.hflip(msk)
        return img, msk

    def _photo_ops_pil(self, img: Image.Image, global_view_idx: int):
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
        """
        Set pixels to black where mask > 127 (white). Works on PIL, returns PIL.
        """
        img_np = np.array(img_pil).copy()        # H x W x 3, uint8
        m_np = np.array(mask_pil)                # H x W, uint8
        hole = m_np > 127
        if hole.any():
            img_np[hole] = 0
        return Image.fromarray(img_np, mode="RGB")

    def __call__(self, image_pil: Image.Image, mask_pil: Image.Image):
        images: List[torch.Tensor] = []

        # ---- two GLOBAL views (no inpainting) ----
        for gv in (0, 1):
            img_g, msk_g = self._apply_geom(image_pil, mask_pil, self.global_size, self.global_crops_scale)
            img_g = self._photo_ops_pil(img_g, global_view_idx=gv)
            images.append(self.normalize(img_g))

        # ---- LOCAL views (apply inpainting using mask) ----
        for _ in range(self.local_crops_number):
            img_l, msk_l = self._apply_geom(image_pil, mask_pil, self.local_size, self.local_crops_scale)
            img_l = self._photo_ops_pil(img_l, global_view_idx=None)
            img_l = self._inpaint_with_mask(img_l, msk_l)  # <- black out white mask regions
            images.append(self.normalize(img_l))

        return images  # list[Tensor 3xHxW], length M


# ----------------- Paired image+mask folder -----------------
class PairedMaskFolder(Dataset):
    """
    Pairs `<name>.png` with `<name>__mask.png` in a flat folder.
    Skips overlays/cell crops/etc.
    Returns (images_views, 0)  where images_views is list of tensors for DINO.
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
                pairs.append((p, m))
        if not pairs:
            raise RuntimeError(f"No paired images found in {self.root}")
        self.pairs = pairs

    def __len__(self): return len(self.pairs)

    def __getitem__(self, idx):
        img_path, msk_path = self.pairs[idx]
        img = Image.open(img_path).convert("RGB")
        msk = Image.open(msk_path).convert("L")  # white (255) = region to blacken
        images_views = self.transform(img, msk)  # list of tensors
        return images_views, 0


# ----------------- viz utils (images only) -----------------
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)

def denorm_to_uint8(img_chw: torch.Tensor) -> np.ndarray:
    x = img_chw.detach().cpu()
    x = x * IMAGENET_STD + IMAGENET_MEAN
    x = x.clamp(0,1).numpy()
    return (np.transpose(x, (1,2,0)) * 255.0).round().astype(np.uint8)

def save_grid(images_chw: List[torch.Tensor], cols: int, out_path: Path, title=None, show=False):
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

    for i in range(rows*cols):
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
    fig.savefig(out_path, dpi=200)
    if show:
        plt.show()
    plt.close(fig)


# ----------------- main -----------------
def main():
    ap = argparse.ArgumentParser("DINO inputs with LOCAL inpainting (mask→black)")
    ap.add_argument("--data-path", required=True, help="Folder with <name>.png and <name>__mask.png")
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
    loader = DataLoader(dataset,
                        batch_size=args.batch_size,
                        shuffle=True,
                        num_workers=args.num_workers,
                        pin_memory=True,
                        drop_last=True)

    # one batch
    images, _ = next(iter(loader))  # images is a LIST of tensors
    assert isinstance(images, list)
    M = len(images)
    B = images[0].shape[0]
    K = min(args.max_samples, B)

    print(f"\nTotal views M = 2 (global, NOT inpainted) + {args.local_crops_number} (local, INPAINTED) = {M}")
    for i in range(M):
        tag = "global" if i < 2 else f"local{i-2:02d}"
        print(f"view[{i:02d}] {tag} | tensor = {tuple(images[i].shape)}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save per-view grids (first K samples)
    for i in range(M):
        tag = "global" if i < 2 else f"local{i-2:02d}"
        imgs = [images[i][j] for j in range(K)]
        save_grid(
            imgs, cols=min(K, 4),
            out_path=out_dir / f"view_{i:02d}_{tag}.png",
            title=f"View {i:02d} ({tag}) — first {K} samples",
            show=args.show
        )

    # Save panel of ALL views for sample #0 (shows globals untouched, locals inpainted)
    panel_imgs = [v[0] for v in images]
    save_grid(
        panel_imgs, cols=6,
        out_path=out_dir / "all_views_sample0.png",
        title="All views for sample #0 (globals intact, locals inpainted)",
        show=args.show
    )

    print(f"\nSaved to: {out_dir.resolve()}")
    print("Note: Only LOCAL views were inpainted (mask white → black). Globals were left untouched.")

if __name__ == "__main__":
    main()
