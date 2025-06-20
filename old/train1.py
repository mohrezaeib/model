#!/usr/bin/env python3
# -------------------------------------------------------------
#   BOX + CLASSIFY  – ImageNet   (ResNet-18  +  ResNet-50)
# -------------------------------------------------------------
import os, time, math, random
from pathlib import Path
import torch, torch.nn as nn, torch.nn.functional as F
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader, distributed

# -------------------------  CONFIG  --------------------------
IMAGENET_ROOT = './ILSVRC2012_devkit_t12'   #  expects sub-dirs  train/  val/
BATCH         = 256                     # per GPU
EPOCHS        = 30
LR            = 1e-3
LAMBDA        = 2.0                     # strength of area penalty
IMG_SZ        = 224
SOFT_K        = 40.0                    # mask edge steepness
NUM_CLS       = 1000
DEVICE        = 'cuda' if torch.cuda.is_available() else 'cpu'

# -----------------  utils: differentiable box mask -----------
def soft_box_mask(boxes, H=IMG_SZ, W=IMG_SZ, k=SOFT_K):
    """
    boxes  : (B,4)  ->   cx,cy,w,h  ∈ [0,1]
    return : (B,1,H,W)  values in [0,1]
    """
    B = boxes.size(0)
    cx, cy, bw, bh = boxes.unbind(1)

    x_l = (cx - bw/2).clamp(0, 1); x_r = (cx + bw/2).clamp(0, 1)
    y_t = (cy - bh/2).clamp(0, 1); y_b = (cy + bh/2).clamp(0, 1)

    xs = torch.linspace(0,1,W,device=boxes.device).view(1,1,1,W).expand(B,1,H,W)
    ys = torch.linspace(0,1,H,device=boxes.device).view(1,1,H,1).expand(B,1,H,W)

    left   = torch.sigmoid(k*(xs - x_l.view(B,1,1,1)))
    right  = torch.sigmoid(k*(x_r.view(B,1,1,1) - xs))
    top    = torch.sigmoid(k*(ys - y_t.view(B,1,1,1)))
    bottom = torch.sigmoid(k*(y_b.view(B,1,1,1) - ys))
    return left*right*top*bottom


# -----------------------  MODEL PARTS ------------------------
def build_bbox_head():
    net = torchvision.models.resnet18(weights='IMAGENET1K_V1')
    net.fc = nn.Sequential(
        nn.Linear(net.fc.in_features, 4),
        nn.Sigmoid()
    )
    return net

def build_class_head():
    net = torchvision.models.resnet50(weights='IMAGENET1K_V1')
    # keep the usual 1 000-way fc
    return net

class BoxClassNet(nn.Module):
    def __init__(self, lam=LAMBDA):
        super().__init__()
        self.bbox_head = build_bbox_head()
        self.cls_head  = build_class_head()
        self.lam       = lam

        # freeze some layers if you don’t want to re-train from scratch
        # for m in self.cls_head.parameters():  m.requires_grad=False
        # self.cls_head.fc.weight.requires_grad = True
        # self.cls_head.fc.bias .requires_grad = True

    def forward(self, imgs, targets=None):
        B = imgs.size(0)
        boxes   = self.bbox_head(imgs)                 # (B,4)
        boxes[:,2:] = boxes[:,2:].clamp(0.05, 1.0)     # avoid degenerate w/h
        mask    = soft_box_mask(boxes, imgs.size(2), imgs.size(3)) # (B,1,H,W)
        x_mask  = imgs * mask
        logits  = self.cls_head(x_mask)                # (B,1000)

        if targets is None:
            return logits, boxes

        ce   = F.cross_entropy(logits, targets)
        area = (boxes[:,2] * boxes[:,3]).mean()
        loss = ce + self.lam*area
        return loss, logits, boxes


# ------------------------  DATA  ------------------------------
def get_loaders():
    norm = transforms.Normalize([0.485,0.456,0.406],
                                [0.229,0.224,0.225])

    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(IMG_SZ, scale=(0.08,1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(), norm])

    val_tf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(IMG_SZ),
        transforms.ToTensor(), norm])

    train_set = torchvision.datasets.ImageNet(IMAGENET_ROOT, split='train',
                                              transform=train_tf)
    val_set   = torchvision.datasets.ImageNet(IMAGENET_ROOT, split='val',
                                              transform=val_tf)

    train_loader = DataLoader(train_set, batch_size=BATCH, shuffle=True,
                              num_workers=8, pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_set,   batch_size=BATCH, shuffle=False,
                              num_workers=8, pin_memory=True)
    return train_loader, val_loader

# ----------------------  EVALUATION ---------------------------
@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    top1 = 0; total = 0
    for imgs, lbls in loader:
        imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
        logits, _  = model(imgs)
        pred = logits.argmax(1)
        top1 += (pred==lbls).sum().item()
        total += lbls.size(0)
    model.train()
    return 100*top1/total

# ------------------  TRAINING LOOP ----------------------------
def main():
    torch.manual_seed(0)
    train_loader, val_loader = get_loaders()

    model = BoxClassNet().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    lr_sched  = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    for epoch in range(1, EPOCHS+1):
        t0=time.time(); run_loss=0
        for imgs, lbls in train_loader:
            imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)

            loss, _, _ = model(imgs, lbls)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            run_loss += loss.item()*imgs.size(0)

        lr_sched.step()
        tr_loss = run_loss/len(train_loader.dataset)
        acc = evaluate(model, val_loader)
        dt=time.time()-t0
        print(f'E{epoch:02d}/{EPOCHS}  loss {tr_loss:.3f}  val-top1 {acc:5.2f}%  {dt/60:4.1f} min')

    # optional: save
    torch.save(model.state_dict(), 'boxclass_imagenet.pth')

if __name__ == '__main__':
    main()