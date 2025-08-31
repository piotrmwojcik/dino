#!/usr/bin/env python3
# Copyright (c) Facebook, Inc.
# Licensed under the Apache License, Version 2.0

import os
import sys
import argparse

import torch
from torch import nn
import torch.distributed as dist
import torch.backends.cudnn as cudnn
from torchvision import datasets
from torchvision import transforms as pth_transforms
from torchvision import models as torchvision_models

import utils
import vision_transformer as vits


def extract_feature_pipeline(args):
    # ============ preparing data ... ============
    transform = pth_transforms.Compose([
        pth_transforms.Resize(32, interpolation=3),
        pth_transforms.CenterCrop(32),
        pth_transforms.ToTensor(),
        pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    train_dir = os.path.join(args.data_path, "train")
    dataset_train = ReturnIndexDataset(train_dir, transform=transform)
    sampler = torch.utils.data.DistributedSampler(dataset_train, shuffle=False)
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        sampler=sampler,
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    if utils.get_rank() == 0:
        print(f"Data loaded with {len(dataset_train)} images from: {train_dir}")

    # ============ building network ... ============
    if "vit" in args.arch:
        model = vits.__dict__[args.arch](patch_size=args.patch_size, num_classes=0)
        if utils.get_rank() == 0:
            print(f"Model {args.arch} {args.patch_size}x{args.patch_size} built.")
    elif "xcit" in args.arch:
        model = torch.hub.load('facebookresearch/xcit:main', args.arch, num_classes=0)
    elif args.arch in torchvision_models.__dict__.keys():
        model = torchvision_models.__dict__[args.arch](num_classes=0)
        model.fc = nn.Identity()
    else:
        print(f"Architecture {args.arch} not supported")
        sys.exit(1)

    model.cuda()
    utils.load_pretrained_weights(model, args.pretrained_weights, args.checkpoint_key, args.arch, args.patch_size)
    model.eval()

    # ============ extract features ... ============
    if utils.get_rank() == 0:
        print("Extracting features for train set...")
    train_features = extract_features(model, data_loader_train, args.use_cuda)

    # L2-normalize on rank 0
    if utils.get_rank() == 0 and train_features is not None:
        train_features = nn.functional.normalize(train_features, dim=1, p=2)

    # labels from the underlying ImageFolder
    train_labels = torch.tensor([s[-1] for s in dataset_train.samples]).long()

    # save features and labels (rank 0 only)
    if args.dump_features and dist.get_rank() == 0:
        os.makedirs(args.dump_features, exist_ok=True)
        torch.save(train_features.cpu(), os.path.join(args.dump_features, "trainfeat.pth"))
        torch.save(train_labels.cpu(),  os.path.join(args.dump_features, "trainlabels.pth"))
        print(f"Saved train features/labels to: {args.dump_features}")

    return train_features, train_labels


@torch.no_grad()
def extract_features(model, data_loader, use_cuda=True, multiscale=False):
    metric_logger = utils.MetricLogger(delimiter="  ")
    features = None
    for samples, index in metric_logger.log_every(data_loader, 10):
        samples = samples.cuda(non_blocking=True)
        index = index.cuda(non_blocking=True)
        feats = utils.multi_scale(samples, model) if multiscale else model(samples).clone()

        # init storage feature matrix on rank 0
        if dist.get_rank() == 0 and features is None:
            features = torch.zeros(len(data_loader.dataset), feats.shape[-1])
            if use_cuda:
                features = features.cuda(non_blocking=True)
            print(f"Storing features into tensor of shape {features.shape}")

        # gather indexes from all processes
        y_all = torch.empty(dist.get_world_size(), index.size(0), dtype=index.dtype, device=index.device)
        y_l = list(y_all.unbind(0))
        y_all_reduce = torch.distributed.all_gather(y_l, index, async_op=True)
        y_all_reduce.wait()
        index_all = torch.cat(y_l)

        # gather features from all processes
        feats_all = torch.empty(
            dist.get_world_size(), feats.size(0), feats.size(1),
            dtype=feats.dtype, device=feats.device
        )
        output_l = list(feats_all.unbind(0))
        output_all_reduce = torch.distributed.all_gather(output_l, feats, async_op=True)
        output_all_reduce.wait()

        # update storage on rank 0
        if dist.get_rank() == 0:
            if use_cuda:
                features.index_copy_(0, index_all, torch.cat(output_l))
            else:
                features.index_copy_(0, index_all.cpu(), torch.cat(output_l).cpu())
    return features


class ReturnIndexDataset(datasets.ImageFolder):
    def __getitem__(self, idx):
        img, _ = super(ReturnIndexDataset, self).__getitem__(idx)
        return img, idx


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Extract train features only')
    parser.add_argument('--batch_size_per_gpu', default=128, type=int, help='Per-GPU batch-size')
    parser.add_argument('--pretrained_weights', default='', type=str, help="Path to pretrained weights to evaluate.")
    parser.add_argument('--use_cuda', default=True, type=utils.bool_flag,
                        help="Store features on GPU (set False to avoid OOM).")
    parser.add_argument('--arch', default='vit_small', type=str, help='Architecture')
    parser.add_argument('--patch_size', default=16, type=int, help='Patch resolution of the model.')
    parser.add_argument("--checkpoint_key", default="teacher", type=str,
                        help='Key to use in the checkpoint (example: "teacher")')
    parser.add_argument('--dump_features', default=None,
                        help='Path to save computed train features/labels; empty for no saving')
    parser.add_argument('--load_features', default=None,
                        help='If features already computed, path to directory containing trainfeat.pth/trainlabels.pth')
    parser.add_argument('--num_workers', default=10, type=int, help='Number of data loading workers per GPU.')
    parser.add_argument("--dist_url", default="env://", type=str,
                        help="URL used to set up distributed training; see PyTorch docs.")
    parser.add_argument("--local_rank", default=0, type=int, help="Please ignore and do not set this argument.")
    parser.add_argument('--data_path', default='/path/to/imagenet/', type=str,
                        help='Path to ImageNet root with a "train" subfolder.')
    args = parser.parse_args()

    utils.init_distributed_mode(args)
    print("git:\n  {}\n".format(utils.get_sha()))
    print("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))
    cudnn.benchmark = True

    if args.load_features:
        # load precomputed train features/labels
        train_features = torch.load(os.path.join(args.load_features, "trainfeat.pth"))
        train_labels   = torch.load(os.path.join(args.load_features, "trainlabels.pth"))
        if utils.get_rank() == 0:
            print(f"Loaded train features/labels from: {args.load_features}")
    else:
        train_features, train_labels = extract_feature_pipeline(args)

    dist.barrier()
    if utils.get_rank() == 0:
        print("Done.")
