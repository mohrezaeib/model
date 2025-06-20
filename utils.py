import random
import torch
import os
from torchvision.utils import save_image
from PIL import Image, ImageDraw, ImageFont
from torchvision.transforms.functional import to_pil_image
from typing import Union, Sequence, Tuple

import time, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from torchvision.datasets import VOCDetection
from torchvision.utils import save_image
from typing import Union, Sequence, Tuple

DEVICE     = 'cuda' if torch.cuda.is_available() else 'cpu'
CLASSES    = [
    'aeroplane','bicycle','bird','boat','bottle',
    'bus','car','cat','chair','cow',
    'diningtable','dog','horse','motorbike','person',
    'pottedplant','sheep','sofa','train','tvmonitor'
]


def decode_labels(label_vec, class_names, threshold=0.5):
    """
    Decodes a multi-hot vector of shape (num_classes,)
    into a list of class names whose probability > threshold.
    """
    return [
        class_names[i] 
        for i in range(len(class_names)) 
        if label_vec[i] > threshold
    ]


def save_batch_images(save_path, imgs, combined_mask, x_masked):
    """
    Saves the original images, mask, and masked images as PNG files.
    """
    os.makedirs(save_path, exist_ok=True)
    save_image(imgs,          os.path.join(save_path, 'orig.png'),   nrow=8, normalize=True)
    save_image(combined_mask, os.path.join(save_path, 'mask.png'),   nrow=8, normalize=True)
    save_image(x_masked,      os.path.join(save_path, 'masked.png'), nrow=8, normalize=True)
    print(f'Images saved in {save_path}')


def save_label_info(save_path, gt_labels, pred_logits, area, class_names):
    """
    Saves a text file containing:
    - Ground truth labels
    - Predicted labels (with threshold=0.5)
    - Mask area per image
    """
    pred_probs = pred_logits.sigmoid()  # shape (B, num_classes)
    with open(os.path.join(save_path, 'label_info.txt'), 'w') as f:
        for i in range(len(gt_labels)):
            # Ground truth classes:
            gt_class_list = decode_labels(gt_labels[i], class_names, threshold=0.5)
            # Predicted classes:
            pred_class_list = decode_labels(pred_probs[i], class_names, threshold=0.5)

            f.write(f"Image {i}:\n")
            f.write(f"  Ground Truth  : {gt_class_list}\n")
            f.write(f"  Predicted (0.5): {pred_class_list}\n")
            f.write(f"  Mask Area      : {area[i].item():.2f}\n\n")

from PIL import Image, ImageDraw, ImageFont
from torchvision.transforms.functional import to_pil_image

def save_combined_images_with_labels(
    output_dir,
    imgs,
    combined_mask,
    x_masked,
    gt_labels,
    logits,
    area,
    class_names
):
    """
    Creates a single collage image for the entire batch, where each row is:
    [original | mask | masked].
    Then overlays text about GT labels, predicted labels, and mask area
    in the top-left corner of that row.

    Saves the final collage as 'combined.png' in output_dir.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Convert logits to probabilities
    pred_probs = logits.sigmoid()
    
    batch_size = imgs.size(0)

    # Convert the first sample to PIL just to measure width/height
    orig_pil_0 = to_pil_image(imgs[0].cpu())
    width, height = orig_pil_0.size

    # This collage will have B rows, each containing 3 images side-by-side.
    combined_width = width * 3
    combined_height = height * batch_size

    # Create a new blank canvas for the entire batch
    collage = Image.new('RGB', (combined_width, combined_height))

    # Prepare a drawing context
    draw = ImageDraw.Draw(collage)
    font = ImageFont.load_default()  # or supply your own TTF via ImageFont.truetype

    margin = 5

    for i in range(batch_size):
        # Convert each tensor in the batch to PIL images
        orig_pil   = to_pil_image(imgs[i].cpu())
        mask_pil   = to_pil_image(combined_mask[i].cpu())
        masked_pil = to_pil_image(x_masked[i].cpu())

        # Top-left corner of this row in the collage
        row_offset = i * height

        # Paste each image in the correct position
        collage.paste(orig_pil,   (0,             row_offset))
        collage.paste(mask_pil,   (width,         row_offset))
        collage.paste(masked_pil, (2 * width,     row_offset))

        # Decode the labels (GT, Pred)
        gt_labels_decoded   = decode_labels(gt_labels[i], class_names, threshold=0.5)
        pred_labels_decoded = decode_labels(pred_probs[i], class_names, threshold=0.5)

        # Gather text lines
        text_lines = [
            f"GT: {', '.join(gt_labels_decoded)}",
            f"Pred: {', '.join(pred_labels_decoded)}",
            f"Area: {area[i].item():.2f}"
        ]
        text_content = "\n".join(text_lines)
        
        # Measure text bounding box so we can draw a rectangle behind it
        # (Requires PIL >= 9.2 for multiline_textbbox)
        bbox = draw.multiline_textbbox((0, 0), text_content, font=font)
        if bbox is not None:
            x_min, y_min, x_max, y_max = bbox
            text_width = x_max - x_min
            text_height = y_max - y_min
        else:
            # Fallback if multiline_textbbox is not available
            text_width, text_height = draw.multiline_textsize(text_content, font=font)

        # Coordinates for the text background rectangle
        bg_x0 = margin
        bg_y0 = row_offset + margin
        bg_x1 = bg_x0 + text_width
        bg_y1 = bg_y0 + text_height

        # Draw a black rectangle as a background for text
        draw.rectangle([(bg_x0, bg_y0), (bg_x1, bg_y1)], fill=(0, 0, 0))

        # Now draw the text in white
        draw.multiline_text(
            (bg_x0, bg_y0),
            text_content,
            fill=(255, 255, 255),
            font=font
        )

    # Finally, save the entire collage
    collage.save(os.path.join(output_dir, "combined.png"))
    print(f"Batch collage saved as {os.path.join(output_dir, 'combined.png')}")


def run_visual_snapshot(net, imgs, gt_labels, output_dir='./output'):
    """
    Performs a forward pass on the given batch of images,
    decodes labels, and saves images & label info in a subfolder.
    """
    # Move to DEVICE
    imgs = imgs.to(DEVICE)
    gt_labels = gt_labels.to(DEVICE)

    # Forward pass = (logits, masked_img, combined_mask, area)
    logits, x_masked, combined_mask, area = net(imgs)

    # Create folder for snapshot
    os.makedirs(output_dir, exist_ok=True)

    # 1) Save separate images (original, mask, masked)
    save_batch_images(output_dir, imgs, combined_mask, x_masked)

    # 2) Save label info in a text file
    save_label_info(output_dir, gt_labels, logits, area, CLASSES)

    # 3) NEW: Save combined side-by-side images with text overlays
    save_combined_images_with_labels(
        output_dir,
        imgs,
        combined_mask,
        x_masked,
        gt_labels,
        logits,
        area,
        CLASSES
    )


def run_validation_snapshots(net, val_ld, ep, output_dir='./output'):
    """
    Runs two validation snapshots:
    1. 'fixed' snapshot on the next mini-batch from val_ld
    2. 'random' snapshot using 8 randomly sampled images from val_ld.dataset
    """
    # ----------------------------
    # FIXED SNAPSHOT
    # ----------------------------
    imgs_fixed, labels_fixed = next(iter(val_ld))  # get next mini-batch
    run_visual_snapshot(net, imgs_fixed[:8], labels_fixed[:8], 
                        output_dir=f'./{output_dir}/{ep}/fixed')

    # ----------------------------
    # RANDOM SNAPSHOT
    # ----------------------------
    idx = random.sample(range(len(val_ld.dataset)), 8)
    imgs_list = []
    gt_labels_list = []
    for i in idx:
        img, lab = val_ld.dataset[i]
        imgs_list.append(img)
        gt_labels_list.append(lab)

    # Combine into a single batch
    imgs_random = torch.stack(imgs_list, dim=0)
    labels_random = torch.stack(gt_labels_list, dim=0)

    run_visual_snapshot(net, imgs_random, labels_random, 
                        output_dir=f'./{output_dir}/{ep}/random')


# --------------------  UTILS  ----------------------
@torch.no_grad()
def evaluate(net, loader, threshold=0.5):
    """
    Evaluate model in a multi-label scenario:
      - y should be multi-hot, shape (B, num_classes).
      - 'pred' also has shape (B, num_classes).
      - We apply a sigmoid, then threshold predictions at 'threshold'.

    Metrics returned:
      1) Precision (micro)
      2) Recall    (micro)
      3) F1        (micro)
      4) Average   area
    """
    net.eval()
    total_area = 0.0

    # We'll accumulate counts for micro-averaged P/R/F1
    TP = 0.0
    FP = 0.0
    FN = 0.0

    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        # The network may return (pred, x_masked, mask, area) or something similar
        logits, x_masked, combined_mask, area = net(x)

        probs = torch.sigmoid(logits)
        # Binarize at threshold
        pred_bin = (probs > threshold).float()

        # Update confusion counts:
        TP += (pred_bin * y).sum().item()          # predicted=1, actual=1
        FP += (pred_bin * (1 - y)).sum().item()    # predicted=1, actual=0
        FN += ((1 - pred_bin) * y).sum().item()    # predicted=0, actual=1

        # Accumulate areas
        total_area +=  area.sum().item()

    # Micro-averaged Precision, Recall, F1
    precision = TP / (TP + FP + 1e-8)
    recall    = TP / (TP + FN + 1e-8)
    f1        = 2.0 * precision * recall / (precision + recall + 1e-8)

    # Per-sample average area
    dataset_size = len(loader.dataset)
    avg_area = total_area / dataset_size

    net.train()
    return precision, recall, f1, avg_area



# ---------------------  CUSTOM HELPERS  --------------------
def area_from_mask(mask: torch.Tensor) -> torch.Tensor:
    """
    Compute the fraction of the image area where mask is '1'.
    Assumes mask is in {0,1} or in [0,1].
    """
    # mask shape is (B,1,H,W) or (B,C,H,W)
    # return area fraction in [0,1]
    return mask.mean(dim=[1,2,3])  # mean over spatial dims (and possibly channel)


# -----------------  CIRCULAR SOFT-MASK  -----------------
def mask_circle(
    img: torch.Tensor,
    circle: Union[Sequence, torch.Tensor],   # (cx, cy, r) ‑- all ∈[0,1]
    *, hardness: float = 50.0, return_mask: bool = True
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """
    Soft mask of a circle in normalised image coordinates.

      img    : (C,H,W) or (B,C,H,W)
      circle : (3,)   or (B,3)  ––  (cx, cy, radius) in [0,1]
    """
    squeeze_batch = False
    if img.dim() == 3:           # (C,H,W) → (1,C,H,W)
        img = img.unsqueeze(0)
        squeeze_batch = True
    if img.dim() != 4:
        raise ValueError("img must be (C,H,W) or (B,C,H,W)")

    B, C, H, W = img.shape
    dev, dtype = img.device, img.dtype

    c = torch.as_tensor(circle, dtype=torch.float32, device=dev)
    if c.dim() == 1:
        c = c.unsqueeze(0)       # (1,3)
    if c.shape != (B, 3):
        raise ValueError("circle must be (3,) or (B,3) with (cx,cy,r)")

    cx, cy, r = c[:, 0], c[:, 1], c[:, 2]          # each shape (B,)
    # build normalised pixel grid
    y_lin = (torch.arange(H, device=dev, dtype=torch.float32) + 0.5) / H
    x_lin = (torch.arange(W, device=dev, dtype=torch.float32) + 0.5) / W
    yy = y_lin.view(1, 1, H, 1).expand(B, 1, H, W)   # (B,1,H,W)
    xx = x_lin.view(1, 1, 1, W).expand(B, 1, H, W)   # (B,1,H,W)

    # squared distance to centre
    dx2 = (xx - cx.view(B, 1, 1, 1)).pow(2)
    dy2 = (yy - cy.view(B, 1, 1, 1)).pow(2)
    dist = torch.sqrt(dx2 + dy2)                     # (B,1,H,W)

    # soft boundary: inside ⇔ dist < r
    mask = torch.sigmoid(hardness * (r.view(B, 1, 1, 1) - dist)).to(dtype)
    masked = img * mask

    if squeeze_batch:
        masked, mask = masked.squeeze(0), mask.squeeze(0)
    return (masked, mask) if return_mask else masked




def mask_polygon(
    img: torch.Tensor,
    vertices: Union[Sequence, torch.Tensor],
    *,
    hardness: float = 50.0,    # larger → sharper edge
    return_mask: bool = True,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """
    Given a batch of images (B,C,H,W) and corresponding polygon 
    vertices (B,N,2) in normalized [0,1] coords, create a soft mask 
    inside the polygon area. Return either the masked image or 
    (masked_image, mask).
    """
    squeeze_batch = False
    if img.dim() == 3:
        img = img.unsqueeze(0)
        squeeze_batch = True
    if not (img.dim() == 4):
        raise ValueError("img must be (C,H,W) or (B,C,H,W)")
    B, C, H, W = img.shape
    dev, dtype = img.device, img.dtype

    # vertices -> shape (B,N,2) with N >= 3
    v = torch.as_tensor(vertices, dtype=torch.float32, device=dev)
    if v.dim() == 2:
        v = v.unsqueeze(0)  # (1,N,2)
    if not (v.dim() == 3 and v.size(0) == B and v.size(-1) == 2 and v.size(1) >= 3):
        raise ValueError("vertices must be (N,2) or (B,N,2) with N≥3, matching batch")

    N = v.size(1)

    # 1. build pixel-center grid in [0,1]
    y_lin = (torch.arange(H, device=dev, dtype=torch.float32) + 0.5) / H
    x_lin = (torch.arange(W, device=dev, dtype=torch.float32) + 0.5) / W
    yy = y_lin.view(1, H, 1).expand(1, H, W)    # (1,H,W)
    xx = x_lin.view(1, 1, W).expand(1, H, W)    # (1,H,W)
    yy = yy.expand(B, 1, H, W)  # (B,1,H,W)
    xx = xx.expand(B, 1, H, W)  # (B,1,H,W)

    # 2. compute signed distance to each polygon edge
    vx = v[..., 0]  # (B,N)
    vy = v[..., 1]  # (B,N)
    vx_n = torch.roll(vx, -1, dims=1)  # (B,N)
    vy_n = torch.roll(vy, -1, dims=1)  # (B,N)

    ex = (vx_n - vx).view(B, N, 1, 1)
    ey = (vy_n - vy).view(B, N, 1, 1)
    px = xx - vx.view(B, N, 1, 1)  # (B,N,H,W)
    py = yy - vy.view(B, N, 1, 1)  # (B,N,H,W)

    # cross product -> signed parallelogram area
    signed_area = ex * py - ey * px   # (B,N,H,W)
    edge_len = torch.sqrt(ex*ex + ey*ey + 1e-8)
    signed_dist = signed_area / edge_len  # (B,N,H,W)

    # 3. determine polygon orientation per batch
    # sum over edges of (vx*vy_n - vx_n*vy) -> 2*signed area
    area2 = (vx * vy_n - vx_n * vy).sum(dim=1)   # (B,)
    orient = torch.where(area2 >= 0, 1.0, -1.0)  # (B,)
    orient = orient.view(B, 1, 1, 1)

    # 4. soft half-plane test and intersection across edges
    edge_prob = torch.sigmoid(hardness * orient * signed_dist)  # (B,N,H,W)
    mask = edge_prob.prod(dim=1, keepdim=True)                  # (B,1,H,W)
    mask = mask.to(dtype)

    # 5. mask the image & return
    masked = img * mask
    if squeeze_batch:
        masked = masked.squeeze(0)
        mask = mask.squeeze(0)
    return (masked, mask) if return_mask else masked

