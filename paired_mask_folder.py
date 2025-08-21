import os
from pathlib import Path
from typing import List, Tuple

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np

import torchvision.transforms as T
import torchvision.transforms.functional as TF


# ----------------------------
# Mask-aware DINO augmentations
# ----------------------------
class DataAugmentationDINOWithMask:
    """
    Returns two lists with the same length M = 2 + local_crops_number:
      - images:  [Tensor Bx3xHxW ...] (after normalization)
      - masks:   [Tensor Bx1xHxW ...] (0/1 uint8; only geom ops applied)

    For a single sample call, it returns:
      (images_views: List[3xHxW], masks_views: List[1xHxW])
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

        # color transforms (image only)
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
    def _random_resized_crop_params(img_size, scale):
        """Get params for RandomResizedCrop with aspect ratio=1.0 (square)."""
        H, W = img_size[1], img_size[0]
        # mimic torchvision's get_params but fix ratio to 1.0
        for _ in range(10):
            target_area = np.random.uniform(scale[0], scale[1]) * H * W
            side = int(round(np.sqrt(target_area)))
            if side <= H and side <= W:
                i = np.random.randint(0, H - side + 1)
                j = np.random.randint(0, W - side + 1)
                return i, j, side, side
        # fallback to center crop of min side
        side = min(H, W)
        i = (H - side) // 2
        j = (W - side) // 2
        return i, j, side, side

    def _apply_geom(self, img_pil: Image.Image, mask_pil: Image.Image,
                    out_size: int, scale_range: Tuple[float, float]):
        """Apply SAME random crop/resize/flip to image & mask."""
        # 1) Random square crop with scale
        i, j, h, w = self._random_resized_crop_params(img_pil.size, scale_range)
        img = TF.resized_crop(img_pil, top=i, left=j, height=h, width=w,
                              size=[out_size, out_size], interpolation=TF.InterpolationMode.BICUBIC)
        msk = TF.resized_crop(mask_pil, top=i, left=j, height=h, width=w,
                              size=[out_size, out_size], interpolation=TF.InterpolationMode.NEAREST)

        # 2) Random horizontal flip (shared coin)
        if np.random.rand() < 0.5:
            img = TF.hflip(img)
            msk = TF.hflip(msk)

        return img, msk

    def _finalize_img(self, img: Image.Image, global_view_idx: int | None):
        # color jitter / grayscale (image only)
        img = self.flip_and_color(img)
        # gaussian blur / solarization like DINO
        if global_view_idx == 0:
            if np.random.rand() < self.gb_p1:
                img = self.gb_global1(img)
        elif global_view_idx == 1:
            if np.random.rand() < self.gb_p2:
                img = self.gb_global2(img)
            img = self.solar(img)
        else:
            # local crops: light blur p=0.5 like original code
            if np.random.rand() < 0.5:
                img = self.gb_global1(img)
        # normalize to tensor
        img_t = self.normalize(img)
        return img_t

    @staticmethod
    def _finalize_mask(mask: Image.Image):
        # ensure binary 0/1 then to tensor with channel=1
        arr = np.array(mask)
        # masks are expected 0 or 255; binarize robustly
        bin01 = (arr > 127).astype(np.uint8)
        t = torch.from_numpy(bin01)[None, ...]  # 1xHxW
        return t

    def __call__(self, image_pil: Image.Image, mask_pil: Image.Image):
        images: List[torch.Tensor] = []
        masks:  List[torch.Tensor] = []

        # two globals
        for gv in (0, 1):
            img_g, msk_g = self._apply_geom(image_pil, mask_pil,
                                            out_size=self.global_size,
                                            scale_range=self.global_crops_scale)
            images.append(self._finalize_img(img_g, global_view_idx=gv))
            masks.append(self._finalize_mask(msk_g))

        # local crops
        for _ in range(self.local_crops_number):
            img_l, msk_l = self._apply_geom(image_pil, mask_pil,
                                            out_size=self.local_size,
                                            scale_range=self.local_crops_scale)
            images.append(self._finalize_img(img_l, global_view_idx=None))
            masks.append(self._finalize_mask(msk_l))

        return images, masks


# ----------------------------
# Paired image+mask dataset
# ----------------------------
class PairedMaskFolder(Dataset):
    """
    Scans a flat folder for PNGs and pairs each base image:
      <name>.png  with  <name>__mask.png
    Skips overlays / cells / bbox jsons automatically.

    Returns: ((images_views, masks_views), 0)
      - images_views: list of tensors [3xHxW], length M
      - masks_views:  list of tensors [1xHxW], length M
      - dummy target = 0 (keeps DINO's "(_, _)" pattern)
    """

    def __init__(self, root: str | Path, transform: DataAugmentationDINOWithMask):
        self.root = Path(root)
        self.transform = transform
        all_pngs = sorted(self.root.glob("*.png"))

        base_imgs = []
        for p in all_pngs:
            name = p.name
            if "__overlay" in name or "__cell_" in name or "__boxed" in name:
                continue
            if name.endswith("__mask.png"):
                continue
            mask_path = p.with_name(p.stem + "__mask.png")
            if mask_path.exists():
                base_imgs.append((p, mask_path))

        if not base_imgs:
            raise RuntimeError(f"No paired images found in {self.root} "
                               f"(expects <name>.png + <name>__mask.png)")

        self.pairs = base_imgs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_path, mask_path = self.pairs[idx]
        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")  # binary mask; cells white (255)

        images_views, masks_views = self.transform(img, mask)
        return (images_views, masks_views), 0  # dummy label for compatibility


# ----------------------------
# Example wiring
# ----------------------------
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-path", required=True, help="Folder with paired PNGs")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--local-crops-number", type=int, default=8)
    ap.add_argument("--global-crops-scale", type=float, nargs=2, default=(0.4, 1.0))
    ap.add_argument("--local-crops-scale", type=float, nargs=2, default=(0.05, 0.4))
    args = ap.parse_args()

    transform = DataAugmentationDINOWithMask(
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

    # Grab a batch
    (images, masks), _ = next(iter(loader))
    # images is a LIST of length M; each item is [B, 3, H, W]
    # masks  is a LIST of length M; each item is [B, 1, H, W]
    M = len(images)
    print(f"M views = {M} (2 globals + {args.local_crops_number} locals)")
    for i in range(M):
        print(f"view[{i:02d}] images: {tuple(images[i].shape)} | masks: {tuple(masks[i].shape)}")

    # Example for DINO usage:
    # teacher_out = teacher(images[:2])   # teacher sees only 2 global views
    # student_out = student(images)       # student sees all M views
    # Optional: you now also have masks[:2] aligned with those teacher inputs.
