"""GT-box overlap targets, in the final augmented input coordinate system.

This is box geometry supervision, not fish masks or true occlusion boundaries.
The pixel convention follows the supplied demo: max of coverage-weighted pairs.
"""

import math

import torch
import torch.nn.functional as F


def intersection_over_ground_truth(gt_box, pre_box, eps=1e-6):
    """Broadcastable IoG with the first box as denominator, computed in float32."""
    gt_box, pre_box = gt_box.float(), pre_box.float()
    intersection_wh = (
        torch.minimum(gt_box[..., 2:], pre_box[..., 2:])
        - torch.maximum(gt_box[..., :2], pre_box[..., :2])
    ).clamp(min=0)
    gt_wh = (gt_box[..., 2:] - gt_box[..., :2]).clamp(min=eps)
    return intersection_wh.prod(-1) / gt_wh.prod(-1)


def _validate_target_options(image_hw, sigma, pair_chunk_size, pixel_chunk_size, geometry_chunk_size):
    height, width = map(int, image_hw)
    if height < 1 or width < 1 or not math.isfinite(sigma) or sigma < 0:
        raise ValueError("Expected positive image dimensions and finite nonnegative sigma")
    if any(not isinstance(n, int) or n < 1 for n in (pair_chunk_size, pixel_chunk_size, geometry_chunk_size)):
        raise ValueError("Overlap target chunk sizes must be positive integers")
    return height, width


def _rasterize_rectangles(raw, image_ids, left_top, right_bottom, strength, width,
                          pair_chunk_size, pixel_chunk_size):
    """Max-reduce at most pair_chunk_size x pixel_chunk_size coverage values."""
    if strength.shape[0] == 0:
        return
    pixels = raw.shape[1]
    for pixel_start in range(0, pixels, pixel_chunk_size):
        pixel_end = min(pixel_start + pixel_chunk_size, pixels)
        index = torch.arange(pixel_start, pixel_end, device=raw.device)
        px = index.remainder(width).float()[None]
        py = index.div(width, rounding_mode="floor").float()[None]
        region = raw[:, pixel_start:pixel_end]
        for pair_start in range(0, strength.shape[0], pair_chunk_size):
            stop = pair_start + pair_chunk_size
            lt = left_top[pair_start:stop]
            rb = right_bottom[pair_start:stop]
            cover_x = (torch.minimum(px + 1, rb[:, 0:1]) - torch.maximum(px, lt[:, 0:1])).clamp_(0, 1)
            cover_y = (torch.minimum(py + 1, rb[:, 1:2]) - torch.maximum(py, lt[:, 1:2])).clamp_(0, 1)
            patch = strength[pair_start:stop, None] * cover_y * cover_x
            if raw.shape[0] == 1:
                torch.maximum(region[0], patch.amax(0), out=region[0])
            else:
                ids = image_ids[pair_start:stop, None].expand_as(patch)
                region.scatter_reduce_(0, ids, patch, reduce="amax", include_self=True)


def _overlap_maps(boxes_xyxy, image_ids, batch_size, image_hw, sigma,
                  pair_chunk_size, pixel_chunk_size, geometry_chunk_size):
    """Generate a batch on-device without per-image CPU copies or per-pair loops.

    Ragged triangular pair indices avoid cross-image pairs and a B x N x N
    allocation. One scalar (the total pair count) is read per batch to schedule
    bounded chunks. Nonzero compaction happens per geometry chunk, not per pair;
    on CUDA it can synchronize to obtain the compacted length. Coordinates and
    raster maps never leave their device. Peak pair/pixel work is bounded by the
    chunk sizes rather than the total number of overlapping pairs.
    """
    height, width = _validate_target_options(
        image_hw, sigma, pair_chunk_size, pixel_chunk_size, geometry_chunk_size,
    )
    if boxes_xyxy.ndim != 2 or boxes_xyxy.shape[1] != 4:
        raise ValueError("Expected boxes with shape [N,4]")
    boxes = boxes_xyxy.detach().float().clone()
    finite = torch.isfinite(boxes).all()
    if boxes.device.type == "cuda":
        torch._assert_async(finite, "Box coordinates must be finite")
    elif not finite:
        raise ValueError("Box coordinates must be finite")
    image_ids = image_ids.to(device=boxes.device, dtype=torch.long).view(-1)
    if image_ids.shape[0] != boxes.shape[0]:
        raise ValueError("Each GT box must have one image index")
    boxes[:, 0::2].clamp_(0, width)
    boxes[:, 1::2].clamp_(0, height)
    wh = boxes[:, 2:] - boxes[:, :2]
    valid = (wh > 0).all(-1)
    areas = wh.clamp(min=1e-6).prod(-1)
    raw = torch.zeros(batch_size, height * width, device=boxes.device, dtype=torch.float32)

    if boxes.shape[0] > 1:
        counts = torch.zeros(batch_size, device=boxes.device, dtype=torch.long)
        counts.scatter_add_(0, image_ids, torch.ones_like(image_ids))
        pair_counts = counts * (counts - 1) // 2
        pair_ends = pair_counts.cumsum(0)
        pair_starts = pair_ends - pair_counts
        box_starts = counts.cumsum(0) - counts
        order = image_ids.argsort(stable=True)
        # A single control scalar per batch, never per box/pair/pixel.
        total_pairs = int(pair_counts.sum().item())
        for start in range(0, total_pairs, geometry_chunk_size):
            ordinal = torch.arange(start, min(start + geometry_chunk_size, total_pairs), device=boxes.device)
            images = torch.searchsorted(pair_ends, ordinal, right=True)
            local = ordinal - pair_starts[images]
            # Enumerate (i,j) as j=1..N-1, i=0..j-1 without triu_indices(N,N).
            j = ((1 + (1 + 8 * local.double()).sqrt()) * 0.5).floor().long()
            j -= (j * (j - 1) // 2 > local).long()
            j += (j * (j + 1) // 2 <= local).long()
            i = local - j * (j - 1) // 2
            first = order[box_starts[images] + i]
            second = order[box_starts[images] + j]
            lt = torch.maximum(boxes[first, :2], boxes[second, :2])
            rb = torch.minimum(boxes[first, 2:], boxes[second, 2:])
            intersection = (rb - lt).clamp(min=0).prod(-1)
            strength = intersection / torch.minimum(areas[first], areas[second])
            keep = (valid[first] & valid[second] & (strength > 0)).nonzero().flatten()
            _rasterize_rectangles(
                raw, images[keep], lt[keep], rb[keep], strength[keep], width,
                pair_chunk_size, pixel_chunk_size,
            )

    soft = raw.view(batch_size, 1, height, width)
    if sigma > 0:
        radius = math.ceil(3 * sigma)
        offsets = torch.arange(-radius, radius + 1, device=boxes.device, dtype=torch.float32)
        kernel = torch.exp(-0.5 * (offsets / sigma).square())
        kernel /= kernel.sum()
        # Smooth the batch together. Disabling AMP alone does not disable TF32;
        # full FP32 kernels keep the targets consistent with the CPU definition.
        cudnn = torch.backends.cudnn
        # flags() defaults enabled/benchmark/deterministic to False. Preserve
        # the caller's backend configuration and change only target precision.
        with cudnn.flags(
            enabled=cudnn.enabled,
            benchmark=cudnn.benchmark,
            benchmark_limit=cudnn.benchmark_limit,
            deterministic=cudnn.deterministic,
            allow_tf32=False,
        ):
            soft = F.conv2d(soft, kernel.view(1, 1, 1, -1), padding=(0, radius))
            soft = F.conv2d(soft, kernel.view(1, 1, -1, 1), padding=(radius, 0))
    return soft.clamp_(0, 1)


@torch.no_grad()
def build_overlap_prior(boxes_xyxy, image_hw, sigma=8.0, *, pair_chunk_size=32,
                        pixel_chunk_size=32768, geometry_chunk_size=4096):
    """Return an on-device [H,W] float32 map using bounded pair/pixel chunks.

    Preserve max(IoG(G_i,G_j), IoG(G_j,G_i)), fractional pixel coverage,
    pairwise maximum aggregation, and Gaussian smoothing in input pixels.
    Invalid/fully cropped boxes are ignored; there is no per-image normalization.
    """
    if boxes_xyxy.ndim != 2 or boxes_xyxy.shape[1] != 4:
        raise ValueError("Expected boxes with shape [N,4]")
    image_ids = torch.zeros(boxes_xyxy.shape[0], device=boxes_xyxy.device, dtype=torch.long)
    with torch.autocast(device_type=boxes_xyxy.device.type, enabled=False):
        return _overlap_maps(
            boxes_xyxy, image_ids, 1, image_hw, sigma,
            pair_chunk_size, pixel_chunk_size, geometry_chunk_size,
        )[0, 0]


@torch.no_grad()
def batch_overlap_targets(batch, target_hw, device, sigma=8.0, *, pair_chunk_size=32,
                          pixel_chunk_size=32768, geometry_chunk_size=4096):
    """Build on-device targets from final augmented normalized xywh labels.

    All geometry/rasterization/smoothing runs on device. Only pair-count control
    metadata is read by Python; GT coordinates and maps are never copied to CPU.
    """
    batch_size, _, height, width = batch["img"].shape
    device = torch.device(device)
    boxes = batch["bboxes"].detach().to(device=device, dtype=torch.float32)
    image_indices = batch["batch_idx"].to(device=device, dtype=torch.long).view(-1)
    with torch.autocast(device_type=device.type, enabled=False):
        center, half_wh = boxes[:, :2], boxes[:, 2:] / 2
        xyxy = torch.cat((center - half_wh, center + half_wh), -1)
        xyxy *= xyxy.new_tensor((width, height, width, height))
        targets = _overlap_maps(
            xyxy, image_indices, batch_size, (height, width), sigma,
            pair_chunk_size, pixel_chunk_size, geometry_chunk_size,
        )
        return F.interpolate(targets, size=target_hw, mode="area")


def balanced_prior_loss(logits, target, kind="bce"):
    """Equal mean weight for nonzero support and zero background, per image.

    Empty-support images still penalize false positives. Soft targets stay soft;
    no pos_weight changes the probability optimum. Regions absent in an image
    are omitted rather than reducing its total loss weight.
    """
    logits, target = logits.float(), target.float()
    if kind == "bce":
        error = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    elif kind == "l1":
        error = (logits.sigmoid() - target).abs()
    else:
        raise ValueError("Prior loss must be 'bce' or 'l1'")
    support = (target > 0).to(error.dtype)
    axes = (1, 2, 3)
    fg_count, bg_count = support.sum(axes), (1 - support).sum(axes)
    foreground = (error * support).sum(axes) / fg_count.clamp(min=1)
    background = (error * (1 - support)).sum(axes) / bg_count.clamp(min=1)
    regions = (fg_count > 0).to(error.dtype) + (bg_count > 0).to(error.dtype)
    return ((foreground + background) / regions.clamp(min=1)).mean()
