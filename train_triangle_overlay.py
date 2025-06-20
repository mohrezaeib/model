import time, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from torchvision.datasets import VOCDetection
from torchvision.utils import save_image
from typing import Union, Sequence, Tuple

# ---------------------  CUSTOM HELPERS  --------------------
def area_from_mask(mask: torch.Tensor) -> torch.Tensor:
    """
    Compute the fraction of the image area where mask is '1'.
    Assumes mask is in {0,1} or in [0,1].
    """
    # mask shape is (B,1,H,W) or (B,C,H,W)
    # return area fraction in [0,1]
    return mask.mean(dim=[1,2,3])  # mean over spatial dims (and possibly channel)

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

# ---------------------  CONFIG  --------------------
BATCH      = 32
EPOCHS     = 30
LR         = 3e-4
LAMBDA     = 1e-1       # area-penalty weight
IMG_SIZE   = 224
DEVICE     = 'cuda' if torch.cuda.is_available() else 'cpu'
CLASSES    = [
    'aeroplane','bicycle','bird','boat','bottle',
    'bus','car','cat','chair','cow',
    'diningtable','dog','horse','motorbike','person',
    'pottedplant','sheep','sofa','train','tvmonitor'
]
NUM_CLS    = len(CLASSES)

# Set N_BOX to the number of polygons you want per image:
N_BOX = 2   # e.g. 2 polygons per image

# ------------------  DATASET WRAPPER  --------------
class VOCDatasetCls(Dataset):
    def __init__(self, image_set: str, tfm):
        self.ds = VOCDetection(
            root='data', 
            year='2012',
            image_set=image_set, 
            download=False,
            transform=tfm
        )
        self.cls2idx = {c:i for i,c in enumerate(CLASSES)}

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        img, ann = self.ds[idx]
        obj = ann['annotation']['object']
        if not isinstance(obj, list):
            obj = [obj]
        label = self.cls2idx[obj[0]['name']]
        return img, label

def get_loaders():
    train_tfm = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomAffine(90, translate=(.3,.3), scale=(.8,1.2)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(.2,.2,.2,.1),
        transforms.RandomCrop((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor()
    ])
    val_tfm   = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor()
    ])
    train_ds = VOCDatasetCls('train', train_tfm)
    val_ds   = VOCDatasetCls('val',   val_tfm)
    train_ld = DataLoader(train_ds, BATCH, shuffle=True,  num_workers=2)
    val_ld   = DataLoader(val_ds,   BATCH, shuffle=False, num_workers=2)
    return train_ld, val_ld

# --------------------  MODEL  ----------------------
class BoxClassNet(nn.Module):
    def __init__(self, lam=LAMBDA):
        super().__init__()
        backbone = models.resnet50(weights='IMAGENET1K_V1')
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])  # GAP output
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.backbone.eval()

        feat_dim = 2048
        # For each requested polygon, we need 3 corners × 2 coords = 6 floats
        self.bbox_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(feat_dim, N_BOX * 6),
            nn.Sigmoid()
        )
        self.cls_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(feat_dim, NUM_CLS)
        )
        self.lam = lam

    def forward(self, x, y=None, save_imgs=False, save_path="."):
        # Extract features from fixed backbone
        with torch.no_grad():
            feat = self.backbone(x)

        # Predicted boxes: shape (B, N_BOX, 3, 2)
        boxes = self.bbox_head(feat).view(x.size(0), N_BOX, 3, 2)

        # -------------------------------------------------
        # Combine multiple polygons into a single “union” mask
        # -------------------------------------------------
        combined_mask = None
        for i in range(N_BOX):
            # Get mask from the i-th polygon
            _, m_i = mask_polygon(img=x, vertices=boxes[:, i], return_mask=True)
            if combined_mask is None:
                combined_mask = m_i
            else:
                # Pixelwise max --> union of all polygons
                combined_mask = torch.maximum(combined_mask, m_i)

        # Apply the combined mask once
        x_masked = x * combined_mask

        # Classify the masked image
        logits = self.cls_head(self.backbone(x_masked))

        if save_imgs:
            save_image(x,          f'{save_path}/orig.png',   nrow=8, normalize=True)
            save_image(combined_mask, f'{save_path}/mask.png', nrow=8, normalize=True)
            save_image(x_masked,   f'{save_path}/masked.png', nrow=8, normalize=True)
            print(f'Images saved in {save_path}')

        # If no label is provided, just return the logit predictions
        if y is None:
            return logits, x_masked, combined_mask

        # Compute loss = cross-entropy + penalty on area
        ce   = F.cross_entropy(logits, y)
        area = area_from_mask(combined_mask)  # (B,)
        loss = ce + self.lam * (area ** 2).mean()
        return loss, logits

# --------------------  UTILS  ----------------------
@torch.no_grad()
def evaluate(net, loader):
    net.eval()
    top1 = 0
    total = 0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        pred, _, _ = net(x) if isinstance(net(x), tuple) else net(x)
        top1  += (pred.argmax(1) == y).sum().item()
        total += y.size(0)
    net.train()
    return 100. * top1 / total

# ------------------  TRAIN LOOP  -------------------
def main():
    train_ld, val_ld = get_loaders()
    net = BoxClassNet().to(DEVICE)
    opt = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=LR)
    sched = torch.optim.lr_scheduler.StepLR(opt, 10, gamma=0.1)

    for ep in range(1, EPOCHS+1):
        t0 = time.time()
        run_loss = 0.0
        net.train()
        for x, y in train_ld:
            x, y = x.to(DEVICE), y.to(DEVICE)
            loss, _ = net(x, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            run_loss += loss.item() * x.size(0)
        sched.step()

        acc = evaluate(net, val_ld)
        print(f'E{ep:02d}/{EPOCHS}  '
              f'loss {run_loss/len(train_ld.dataset):.3f}  '
              f'acc {acc:5.2f}%  {(time.time()-t0):.1f}s')

        # quick visual snapshot
        if ep % 2 == 0:
            imgs, _ = next(iter(val_ld))
            net(imgs.to(DEVICE)[:8],
                save_imgs=True,
                save_path=f'./triangle_overlay/epoch_{ep}_fixed')

            # random visual snapshot
            idx = torch.randperm(len(val_ld.dataset))[:8]
            imgs = torch.stack([val_ld.dataset[i][0] for i in idx])
            net(imgs.to(DEVICE),
                save_imgs=True,
                save_path=f'./triangle_overlay/epoch_{ep}_random')

    torch.save(net.state_dict(), 'box_class_net.pth')

if __name__ == '__main__':
    main()
