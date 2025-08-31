#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


def load_feats_labels(feats_dir: Path) -> Tuple[torch.Tensor, torch.Tensor]:
    feats_path = feats_dir / "trainfeat.pth"
    labs_path  = feats_dir / "trainlabels.pth"
    if not feats_path.exists() or not labs_path.exists():
        raise FileNotFoundError(f"Missing features/labels in {feats_dir}. "
                                f"Expected trainfeat.pth & trainlabels.pth")

    feats = torch.load(feats_path, map_location="cpu")
    labels = torch.load(labs_path, map_location="cpu").long().view(-1)

    if feats.shape[0] != labels.shape[0]:
        raise ValueError(f"Feature/label count mismatch: {feats.shape[0]} vs {labels.shape[0]}")

    return feats, labels


def _infer_idx_to_class_from_dataset_root(dataset_root: Optional[Path]) -> Optional[List[str]]:
    """
    Try to reconstruct class index → class name using an ImageFolder-style tree.
    Accepts either the dataset root that contains 'train' or directly the 'train' directory.
    """
    if dataset_root is None:
        return None

    train_dir = dataset_root / "train"
    base = train_dir if train_dir.exists() else dataset_root
    if not base.exists() or not base.is_dir():
        return None

    classes = sorted([d.name for d in base.iterdir() if d.is_dir()])
    return classes if classes else None


def _load_idx_to_class_from_json(mapping_json: Optional[Path]) -> Optional[List[str]]:
    if mapping_json is None or not mapping_json.exists():
        return None
    with open(mapping_json, "r") as f:
        mapping = json.load(f)

    # Accept either {class_name: index} or {index(str or int): class_name} or a list
    if isinstance(mapping, dict):
        try:
            items = sorted(((int(v), str(k)) for k, v in mapping.items()), key=lambda x: x[0])  # name->idx
            return [name for _, name in items]
        except Exception:
            pass
        try:
            items = sorted(((int(k), str(v)) for k, v in mapping.items()), key=lambda x: x[0])  # idx->name
            return [name for _, name in items]
        except Exception:
            pass
    if isinstance(mapping, list) and all(isinstance(x, str) for x in mapping):
        return mapping
    raise ValueError(f"Unrecognized mapping format in {mapping_json}")


def build_idx_to_class(
    feats_dir: Path,
    dataset_root: Optional[Path],
    mapping_json: Optional[Path]
) -> List[str]:
    # 1) explicit JSON wins
    idx_to_class = _load_idx_to_class_from_json(mapping_json)
    if idx_to_class:
        return idx_to_class

    # 2) try to infer from dataset root
    idx_to_class = _infer_idx_to_class_from_dataset_root(dataset_root)
    if idx_to_class:
        return idx_to_class

    # 3) try common mapping files inside feats dir
    for cand in ["class_to_idx.json", "idx_to_class.json", "classes.json"]:
        p = feats_dir / cand
        if p.exists():
            return _load_idx_to_class_from_json(p)

    # 4) fallback: dummy names (will be expanded as needed)
    print("[WARN] Could not infer class names; using generic labels.")
    return []


def select_indices_all_types(
    labels: torch.Tensor,
    idx_to_class: List[str],
    per_class: int,
    max_total: Optional[int],
    seed: int = 0
) -> Tuple[np.ndarray, List[str]]:
    """
    Select indices from ALL classes.
      - per_class > 0: sample up to that many per class
      - per_class <= 0: take ALL samples per class
    Optionally cap by max_total after concatenation.
    """
    rng = np.random.default_rng(seed)
    labels_np = labels.cpu().numpy()

    # synthesize idx_to_class if missing/short
    n_classes = int(labels_np.max()) + 1
    if not idx_to_class or len(idx_to_class) < n_classes:
        idx_to_class = (idx_to_class + [f"class_{i}" for i in range(len(idx_to_class), n_classes)]) if idx_to_class else [f"class_{i}" for i in range(n_classes)]

    # group indices per class
    cls_to_indices: Dict[int, List[int]] = {}
    for idx, c in enumerate(labels_np):
        cls_to_indices.setdefault(int(c), []).append(idx)

    selected_indices: List[int] = []
    for c_idx, pool in sorted(cls_to_indices.items()):
        pool_arr = np.array(pool)
        if per_class is None or per_class <= 0 or len(pool_arr) <= per_class:
            pick = pool_arr
        else:
            pick = rng.choice(pool_arr, size=per_class, replace=False)
        if len(pick) > 0:
            selected_indices.extend(pick.tolist())

    selected_indices = np.array(selected_indices, dtype=np.int64)

    if max_total is not None and len(selected_indices) > max_total:
        selected_indices = rng.choice(selected_indices, size=max_total, replace=False)

    selected_indices.sort()
    return selected_indices, idx_to_class


def run_tsne(
    X: np.ndarray,
    y: np.ndarray,
    class_names: List[str],
    out_dir: Path,
    perplexity: float = 30.0,
    pca_dim: Optional[int] = 50,
    metric: str = "cosine",
    n_iter: int = 1000,
    learning_rate: float = 200.0,
    seed: int = 0,
    fname_prefix: str = "tsne",
):
    out_dir.mkdir(parents=True, exist_ok=True)

    # PCA (recommended before t-SNE)
    if pca_dim is not None and X.shape[1] > pca_dim:
        pca = PCA(n_components=pca_dim, random_state=seed)
        X_red = pca.fit_transform(X)
    else:
        X_red = X

    # Perplexity must be < n_samples
    n = X_red.shape[0]
    perp = min(perplexity, max(5, (n - 1) / 3))

    tsne = TSNE(
        n_components=2,
        perplexity=perp,
        learning_rate=learning_rate,
        n_iter=n_iter,
        metric=metric,
        init="pca",
        random_state=seed,
        verbose=1,
    )
    Z = tsne.fit_transform(X_red)

    # Save CSV
    import csv
    csv_path = out_dir / f"{fname_prefix}_points.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["x", "y", "label_idx", "class"])
        for (x, yv), li in zip(Z, y):
            cname = class_names[li] if li < len(class_names) else f"class_{li}"
            w.writerow([float(x), float(yv), int(li), cname])

    # Plot
    plt.figure(figsize=(10, 8), dpi=150)
    uniq = np.unique(y)
    remap = {c: i for i, c in enumerate(uniq)}
    y_compact = np.vectorize(remap.get)(y)

    scatter = plt.scatter(Z[:, 0], Z[:, 1], c=y_compact, s=8, alpha=0.8, cmap="tab20")
    handles, texts = [], []
    denom = max(1, len(uniq) - 1)
    for c in uniq:
        cname = class_names[c] if c < len(class_names) else f"class_{c}"
        color = scatter.cmap(remap[c] / denom)
        handles.append(plt.Line2D([], [], color=color, marker='o', linestyle='', markersize=6))
        texts.append(cname)
    plt.legend(handles, texts, loc="best", fontsize=8, frameon=True)
    plt.title("t-SNE of features (all classes)")
    plt.tight_layout()
    png_path = out_dir / f"{fname_prefix}.png"
    plt.savefig(png_path)
    plt.close()

    print(f"[OK] Saved: {png_path}")
    print(f"[OK] Saved: {csv_path}")


def main():
    ap = argparse.ArgumentParser("t-SNE from precomputed features (ALL classes)")
    ap.add_argument("--feats-dir", type=str, required=True,
                    help="Directory containing trainfeat.pth and trainlabels.pth")
    ap.add_argument("--dataset-root", type=str, default=None,
                    help="ImageFolder root (or its 'train' dir) to infer class names")
    ap.add_argument("--class-map-json", type=str, default=None,
                    help="Optional JSON mapping (idx->name or name->idx) to resolve class names")
    ap.add_argument("--per-class", type=int, default=-1,
                    help="Max samples per class; set <=0 to use ALL from each class")
    ap.add_argument("--max-total", type=int, default=None,
                    help="Optional global cap on total samples (applied after per-class sampling)")
    ap.add_argument("--pca-dim", type=int, default=50, help="PCA dimension before t-SNE (0 to disable)")
    ap.add_argument("--perplexity", type=float, default=30.0, help="t-SNE perplexity")
    ap.add_argument("--n-iter", type=int, default=1000, help="t-SNE iterations")
    ap.add_argument("--learning-rate", type=float, default=200.0, help="t-SNE learning rate")
    ap.add_argument("--metric", type=str, default="cosine", help="t-SNE metric (e.g., cosine, euclidean)")
    ap.add_argument("--seed", type=int, default=0, help="Random seed")
    ap.add_argument("--out", type=str, default=None, help="Output directory (default: <feats-dir>/tsne)")
    args = ap.parse_args()

    feats_dir = Path(args.feats_dir)
    out_dir = Path(args.out or (feats_dir / "tsne"))

    feats, labels = load_feats_labels(feats_dir)
    feats_np = feats.detach().cpu().numpy().astype(np.float32)
    labels_t = labels.detach().cpu()

    idx_to_class = build_idx_to_class(
        feats_dir=feats_dir,
        dataset_root=(Path(args.dataset_root) if args.dataset_root else None),
        mapping_json=(Path(args.class_map_json) if args.class_map_json else None),
    )
    # expand/pad class names if necessary
    n_classes = int(labels_t.max()) + 1
    if not idx_to_class or len(idx_to_class) < n_classes:
        idx_to_class = (idx_to_class + [f"class_{i}" for i in range(len(idx_to_class), n_classes)]) if idx_to_class else [f"class_{i}" for i in range(n_classes)]

    sel_idx, idx_to_class = select_indices_all_types(
        labels=labels_t,
        idx_to_class=idx_to_class,
        per_class=args.per_class,
        max_total=args.max_total,
        seed=args.seed
    )

    X = feats_np[sel_idx]
    y = labels_t.numpy()[sel_idx]

    pca_dim = args.pca_dim if args.pca_dim > 0 else None
    run_tsne(
        X=X,
        y=y,
        class_names=idx_to_class,
        out_dir=out_dir,
        perplexity=args.perplexity,
        pca_dim=pca_dim,
        metric=args.metric,
        n_iter=args.n_iter,
        learning_rate=args.learning_rate,
        seed=args.seed,
        fname_prefix="tsne",
    )


if __name__ == "__main__":
    main()
