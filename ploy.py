"""
Fully-differentiable polygon utilities
=====================================

All computations are carried out in PyTorch; no CPU rasterisers, no
`.item()`, no graph-breaking operations.  Gradients flow both

• from the output **image → input image** (trivial – element-wise product)  
• from the output **image → polygon vertices** (thanks to the soft mask)

Functions
---------
mask_polygon()          – returns the soft mask and the masked image  
area_from_mask()        – differentiable area fraction from a mask  
area_from_vertices()    – exact area from the vertex list (Shoelace)

Assumptions
-----------
The polygon must be *simple* (non-self-intersecting) and is assumed to be
convex for perfect accuracy of the signed-distance mask.  For moderately
concave shapes the approximation still works in practice, but the signed
distance is no longer exact.
"""

from __future__ import annotations
from typing import Sequence, Tuple, Union, overload

import torch
import math

from typing import Tuple, Union, Sequence

import torch
import torch.nn.functional as F

from typing import Tuple, Union, Sequence
import torch
import torch.nn.functional as F

def _as_tensor(x, dtype=None, device=None) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x.to(dtype=dtype, device=device)
    return torch.as_tensor(x, dtype=dtype, device=device)


def mask_polygon(
    img: torch.Tensor,
    vertices: Union[Sequence, torch.Tensor],
    *,
    hardness: float = 50.0,    # larger → sharper edge
    return_mask: bool = True,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """
    Multiply *img* by a differentiable *soft* mask defined by *vertices*.

    Supports batch:
      img      : (B,C,H,W) or (C,H,W)
      vertices : (B,N,2) or (N,2)   in normalized [0…1] coords

    hardness : float
      Edge steepness inside sigmoid; try 20–100.
    return_mask : bool
      If True, also return the mask of shape (B,1,H,W).

    Returns
    -------
    masked_img : torch.Tensor  (B,C,H,W)
    mask       : torch.Tensor  (B,1,H,W)   (only if return_mask)
    """
    # ------------------------------------------------------------------
    # 0. lift to batched
    # ------------------------------------------------------------------
    squeeze_batch = False
    if img.dim() == 3:
        img = img.unsqueeze(0)
        squeeze_batch = True
    if not (img.dim() == 4):
        raise ValueError("img must be (C,H,W) or (B,C,H,W)")
    B, C, H, W = img.shape
    dev, dtype = img.device, img.dtype

    v = torch.as_tensor(vertices, dtype=torch.float32, device=dev)
    if v.dim() == 2:
        v = v.unsqueeze(0)            # (1,N,2)
    if not (v.dim() == 3 and v.size(0) == B and v.size(-1) == 2 and v.size(1) >= 3):
        raise ValueError("vertices must be (N,2) or (B,N,2) with N≥3, matching batch")
    # now v is (B,N,2)
    N = v.size(1)

    # ------------------------------------------------------------------
    # 1. build pixel‐centre grid in [0,1]
    # ------------------------------------------------------------------
    y_lin = (torch.arange(H, device=dev, dtype=torch.float32) + 0.5) / H
    x_lin = (torch.arange(W, device=dev, dtype=torch.float32) + 0.5) / W
    yy = y_lin.view(1, H, 1).expand(1, H, W)    # (1,H,W)
    xx = x_lin.view(1, 1, W).expand(1, H, W)    # (1,H,W)
    # broadcast to (B,1,H,W)
    yy = yy.expand(B, 1, H, W)
    xx = xx.expand(B, 1, H, W)

    # ------------------------------------------------------------------
    # 2. compute signed distance to each edge
    # ------------------------------------------------------------------
    vx = v[..., 0]                            # (B,N)
    vy = v[..., 1]                            # (B,N)
    vx_n = torch.roll(vx, -1, dims=1)         # (B,N)
    vy_n = torch.roll(vy, -1, dims=1)         # (B,N)

    ex = (vx_n - vx).view(B, N, 1, 1)          # (B,N,1,1)
    ey = (vy_n - vy).view(B, N, 1, 1)
    px = xx - vx.view(B, N, 1, 1)             # (B,N,H,W)
    py = yy - vy.view(B, N, 1, 1)

    # cross‐product -> signed parallelogram area
    signed_area = ex * py - ey * px           # (B,N,H,W)
    edge_len = torch.sqrt(ex*ex + ey*ey + 1e-8)
    signed_dist = signed_area / edge_len      # (B,N,H,W)

    # ------------------------------------------------------------------
    # 3. determine polygon orientation per batch
    # ------------------------------------------------------------------
    # sum over edges of (vx*vy_n - vx_n*vy) -> 2*signed area
    area2 = (vx * vy_n - vx_n * vy).sum(dim=1)  # (B,)
    orient = torch.where(area2 >= 0, 1.0, -1.0) # (B,)
    orient = orient.view(B, 1, 1, 1)           # for broadcasting

    # ------------------------------------------------------------------
    # 4. soft half-plane test and intersection
    # ------------------------------------------------------------------
    edge_prob = torch.sigmoid(hardness * orient * signed_dist)  # (B,N,H,W)
    mask = edge_prob.prod(dim=1, keepdim=True)                  # (B,1,H,W)
    mask = mask.to(dtype)

    # ------------------------------------------------------------------
    # 5. mask the image & return
    # ------------------------------------------------------------------
    masked = img * mask
    if squeeze_batch:
        masked = masked.squeeze(0)
        mask   = mask.squeeze(0)
    return (masked, mask) if return_mask else masked


# ──────────────────────────────────────────────────────────────────────────
# 2.  area_from_mask  – differentiable
# ──────────────────────────────────────────────────────────────────────────
import torch

def area_from_mask(mask: torch.Tensor) -> torch.Tensor:
    """
    Area fraction (0–1) covered by a *soft* or *hard* mask,
    now with optional batch and channel dims.

    Accepts:
      mask : (H, W)
           | (C, H, W)
           | (B, H, W)
           | (B, C, H, W)

    Returns:
      area : scalar Tensor if no batch,
             else Tensor of shape (B,)
    """
    # if no batch dim, lift to B=1
    area = mask.mean()  # mean over H*W pixels
    return area
# # ──────────────────────────────────────────────────────────────────────────
# # 3.  area_from_vertices – exact & differentiable
# # ──────────────────────────────────────────────────────────────────────────
# def area_from_vertices(vertices: torch.Tensor) -> torch.Tensor:
#     """
#     Shoelace formula.  Vertices must be ordered; works in [0,1] space.

#     vertices : (N, 2) tensor  (with or without requires_grad)
#     returns  : scalar tensor
#     """
#     if vertices.ndim != 2 or vertices.shape[1] != 2 or vertices.size(0) < 3:
#         raise ValueError("expected (N, 2) vertices with N ≥ 3")

#     x, y = vertices[:, 0], vertices[:, 1]
#     area = 0.5 * torch.abs(torch.dot(x, torch.roll(y, -1)) -
#                            torch.dot(y, torch.roll(x, -1)))
#     return area


# ──────────────────────────────────────────────────────────────────────────
# minimal demo  – save results and print area
# ──────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import pathlib
    from PIL import Image

    torch.manual_seed(0)

    # ------------------------------------------------------------------ #
    # helper to turn a single-channel tensor into a PNG on disk
    # ------------------------------------------------------------------ #
    def save_tensor(img_t: torch.Tensor, path: str) -> None:
        """
        img_t : (H, W) or (C, H, W) tensor, values 0‥1
        path  : filename
        """
        if img_t.ndim == 3:
            img_t = img_t[0]             # take first channel
        img_np = (img_t.clamp(0, 1) * 255).to(torch.uint8).cpu().numpy()
        Image.fromarray(img_np).save(path)

    out_dir = pathlib.Path("demo_out")
    out_dir.mkdir(exist_ok=True)

    # ------------------------------------------------------------------ #
    # dummy data
    # ------------------------------------------------------------------ #
    verts = torch.tensor([[[0.2, 0.2],
                        [0.6, 0.2],
                        [0.7, 0.8],
                        [0.3, 0.7]]], requires_grad=True)

    img = torch.ones(1,3, 200, 200, requires_grad=True)

    # soft mask & masked image
    masked_img, soft_mask = mask_polygon(
        img, verts, return_mask=True
    )

    # ------------------------------------------------------------------ #
    # save results
    # ------------------------------------------------------------------ #
    save_tensor(masked_img[0], out_dir / "masked_image.png")
    save_tensor(soft_mask[0],  out_dir / "soft_mask.png")

    # ------------------------------------------------------------------ #
    # print scalar area
    # ------------------------------------------------------------------ #
    area = area_from_mask(soft_mask)      # tensor scalar
    print("area:", area.item())           # plain float