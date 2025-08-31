#!/usr/bin/env python3
# Copyright (c) Facebook, Inc.
# Licensed under the Apache License, Version 2.0

import os
import sys
import argparse

import torch
from torch import nn
import torch.backends.cudnn as cudnn
from torchvision import datasets
from torchvision import transforms as pth_transforms
from torchvision import models as torchvision_models

import utils
import vision_transformer as vits


def extract_feature_pipeline(args):
    # ============ preparing data (single process) ... ============
    transform = pth_transforms.Compose([
        pth_transforms.Resize(32, interpolation=3),
        pth_transforms.CenterCrop(32),
        pth_transforms.ToTensor(),
        pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    train_dir = os.path.join(args.data_path, "train")
    dataset_train = ReturnIndexDataset(train_dir, transform=transform)
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    print(f"Data loaded with {len(dataset_train)} images from: {train_dir}")

    # ============ building network (single device) ... ============
    if "vit" in args.arch:
        model = vits.__dict__[args.arch](patch_size=args.patch_size, num_classes=0)
        print(f"Model {args.arch} {args.patch_size}x{args.patch_size} built.")
    elif "xcit" in args.arch:
        model = torch.hub.load('facebookresearch/xcit:main', args.arch, num_classes=0)
    elif args.arch in torchvision_models.__dict__.keys():
        model = torchvision_models.__dict__[args.arch](num_classes=0)
        model.fc = nn.Identity()
    else:
        print(f"Architecture {args.arch} not supported")
        sys.exit(1)

    device = torch.device("cuda") if (args.use_cuda and torch.cuda.is_available()) else torch.device("cpu")
    if device.type == "cuda":
        print(f"Using CUDA device 0 (visible count={torch.cuda.device_count()})")
    else:
        print("Using CPU")

    model.to(device)
    utils.load_pretrained_weights(model, args.pretrained_weights, args.checkpoint_key, args.arch, args.patch_size)
    model.eval()

    # ============ extract features ... ============
    print("Extracting features for train set...")
    train_features = extract_features(model, data_loader_train, device=device, keep_on_device=args.use_cuda)

    # L2-normalize
    if train_features is not None:
        train_features = nn.functional.normalize(train_features, dim=1, p=2)

    # labels from the underlying ImageFolder
    train_labels = torch.tensor([s[-1] for s in dataset_train.samples]).long()

    # save features and labels
    if args.dump_features:
        os.makedirs(args.dump_features, exist_ok=True)
        torch.save(train_features.cpu(), os.path.join(args.dump_features, "trainfeat.pth"))
        torch.save(train_labels.cpu(),  os.path.join(args.dump_features, "trainlabels.pth"))
        print(f"Saved train features/labels to: {args.dump_features}")

    return train_features, train_labels


@torch.no_grad()
def extract_features(model, data_loader, device, keep_on_device=True, multiscale=False):
    """
    Single-process feature extraction.
    - features are stored on `device` if keep_on_device=True, else on CPU.
    """
    features = None
    dst_device = device if keep_on_device else torch.device("cpu")

    for samples, index in data_loader:
        samples = samples.to(device, non_blocking=True)
        index = index.to(device, non_blocking=True)

        feats = utils.multi_scale(samples, model) if multiscale else model(samples).clone()

        # init storage feature matrix once we know feat dim
        if features is None:
            features = torch.zeros(len(data_loader.dataset), feats.shape[-1], device=dst_device)
            print(f"Storing features into tensor of shape {tuple(features.shape)} on {features.device}")

        # place feats on destination device (if needed)
        if feats.device != dst_device:
            feats_to_store = feats.to(dst_device, non_blocking=True)
        else:
            feats_to_store = feats

        # copy into preallocated matrix at the right indices
        features.index_copy_(0, index.to(dst_device), feats_to_store)

    return features


class ReturnIndexDataset(datasets.ImageFolder):
    def __getitem__(self, idx):
        img, _ = super(ReturnIndexDataset, self).__getitem__(idx)
        return img, torch.tensor(idx, dtype=torch.long)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Extract train features (single process, no distributed)')
    parser.add_argument('--batch_size', default=128, type=int, help='Batch size')
    parser.add_argument('--pretrained_weights', default='', type=str, help="Path to pretrained weights to evaluate.")
    parser.add_argument('--use_cuda', default=True, type=utils.bool_flag,
                        help="Keep features on GPU (set False to store on CPU).")
    parser.add_argument('--arch', default='vit_small', type=str, help='Architecture')
    parser.add_argument('--patch_size', default=16, type=int, help='Patch resolution of the model.')
    parser.add_argument("--checkpoint_key", default="teacher", type=str,
                        help='Key to use in the checkpoint (example: "teacher")')
    parser.add_argument('--dump_features', default=None,
                        help='Path to save computed train features/labels; empty for no saving')
    parser.add_argument('--load_features', default=None,
                        help='If features already computed, path to directory containing trainfeat.pth/trainlabels.pth')
    parser.add_argument('--num_workers', default=10, type=int, help='Number of data loading workers.')
    parser.add_argument('--data_path', default='/path/to/imagenet/', type=str,
                        help='Path to ImageNet root with a "train" subfolder.')
    args = parser.parse_args()

    print("git:\n  {}\n".format(utils.get_sha()))
    print("\n".join(f"{k}: {v}" for k, v in sorted(dict(vars(args)).items())))
    cudnn.benchmark = True

    if args.load_features:
        # load precomputed train features/labels
        train_features = torch.load(os.path.join(args.load_features, "trainfeat.pth"))
        train_labels   = torch.load(os.path.join(args.load_features, "trainlabels.pth"))
        print(f"Loaded train features/labels from: {args.load_features}")
    else:
        train_features, train_labels = extract_feature_pipeline(args)

    print("Done.")
