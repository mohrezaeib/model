#!/usr/bin/env python
# Two-part CNN : BBox prediction + masked classification  on CIFAR-10
# ---------------------------------------------------------------
import math, time, os, random
import torch, torch.nn as nn, torch.nn.functional as F
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.utils import save_image
DEVICE  = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH   = 128
EPOCHS  = 50
LR      = 3e-4
LAMBDA  = 0.5              # weight for area-penalty term
IMG_H, IMG_W = 32, 32      # CIFAR-10 resolution
SOFT_K  = 30.0             # steepness of sigmoid edges for soft mask
NUM_CLS = 10

# ---------------------------------------------------------------
# Helper : build a differentiable soft box mask from (cx,cy,w,h)
# All coords are in [0,1] relative domain.
# ---------------------------------------------------------------
def soft_box_mask(boxes, H=IMG_H, W=IMG_W, k=SOFT_K):
    """
    boxes : (B,4) tensor with (cx,cy,w,h) in [0,1]
    return : (B,1,H,W) tensor with values in [0,1]
    """
    B = boxes.size(0)
    cx, cy, bw, bh = boxes.unbind(dim=1)            # (B,)
    # edges in [0,1]
    x_l = (cx - bw/2).clamp(0,1)
    x_r = (cx + bw/2).clamp(0,1)
    y_t = (cy - bh/2).clamp(0,1)
    y_b = (cy + bh/2).clamp(0,1)

    # coordinate grid – broadcast to (B,1,H,W)
    xs = torch.linspace(0,1,W,device=boxes.device).view(1,1,1,W)
    ys = torch.linspace(0,1,H,device=boxes.device).view(1,1,H,1)

    xs = xs.expand(B,1,H,W)         # (B,1,H,W)
    ys = ys.expand(B,1,H,W)

    # 1D sigmoid gates on each side, multiplied to form 2-D mask
    left   = torch.sigmoid( k * (xs - x_l.view(B,1,1,1)) )
    right  = torch.sigmoid( k * (x_r.view(B,1,1,1) - xs) )
    top    = torch.sigmoid( k * (ys - y_t.view(B,1,1,1)) )
    bottom = torch.sigmoid( k * (y_b.view(B,1,1,1) - ys) )

    mask = left * right * top * bottom          # (B,1,H,W)
    return mask


# ---------------------------------------------------------------
# 1. BBox head – very small CNN that outputs 4 numbers in (0,1)
# ---------------------------------------------------------------
class BBoxHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3,16,3,padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16,32,3,padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32,64,3,padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )
        self.fc = nn.Linear(64,4)

    def forward(self,x):
        x = self.conv(x).view(x.size(0),-1)
        # Use sigmoid to keep in [0,1]
        box = torch.sigmoid(self.fc(x))          # (B,4)
        # optional clamp small width/height to avoid degenerate boxes
        min_wh = 0.05
        w = box[:,2].clamp(min_wh,1)
        h = box[:,3].clamp(min_wh,1)
        box = torch.stack([box[:,0], box[:,1], w, h], dim=1)
        return box


# ---------------------------------------------------------------
# 2. Classifier head – ordinary CIFAR-10 CNN (masked input)
# ---------------------------------------------------------------
class ClassHead(nn.Module):
    def __init__(self,num_cls=NUM_CLS):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3,64,3,padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64,64,3,padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2),                              # 16x16
            nn.Conv2d(64,128,3,padding=1), nn.ReLU(),
            nn.MaxPool2d(2),                              # 8x8
            nn.Conv2d(128,256,3,padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),                      # 1x1
        )
        self.fc = nn.Linear(256,num_cls)

    def forward(self,x):
        x = self.net(x).view(x.size(0),-1)
        return self.fc(x)

# ---------------------------------------------------------------
# 3. Full model wrapper
# ---------------------------------------------------------------
class BoxClassNet(nn.Module):
    def __init__(self, lam=LAMBDA):
        super().__init__()
        self.bbox = BBoxHead()
        self.cls  = ClassHead()
        self.lam  = lam

    def forward(self, imgs, targets=None):
        """
        imgs : (B,3,H,W)
        targets : ground-truth labels or None
        returns (loss, logits, boxes) if targets given
                (logits, boxes)         otherwise
        """
        boxes  = self.bbox(imgs)                         # (B,4)
        mask   = soft_box_mask(boxes, imgs.size(2), imgs.size(3))
        x_mask = imgs * mask     
        images_copy = x_mask.clone()

        # Step 2: Save all images in a single image grid
        # save_image(images_copy, 'batch_output_croped.png', nrow=8, padding=2, normalize=True)                        # masked image
        # save_image(imgs, 'batch_output_original.png', nrow=8, padding=2, normalize=True)                        # masked image
        logits = self.cls(x_mask)                        # (B,10)

        if targets is None:
            return logits, boxes

        ce_loss   = F.cross_entropy(logits, targets)
        area      = (boxes[:,2] * boxes[:,3]).mean()     # mean relative area
        loss      = ce_loss + self.lam * area
        return loss, logits, boxes


# ---------------------------------------------------------------
# Data
# ---------------------------------------------------------------
transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor()
])
transform_test = transforms.ToTensor()

train_set = torchvision.datasets.CIFAR10(root='./data', train=True,
                                         download=True, transform=transform_train)
test_set  = torchvision.datasets.CIFAR10(root='./data', train=False,
                                         download=True, transform=transform_test)

train_loader = DataLoader(train_set,  batch_size=BATCH, shuffle=True,  num_workers=2, pin_memory=True)
test_loader  = DataLoader(test_set,   batch_size=BATCH, shuffle=False, num_workers=2, pin_memory=True)

# ---------------------------------------------------------------
# Training / evaluation utilities
# ---------------------------------------------------------------
def evaluate(model, loader):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            logits, _ = model(imgs)
            pred = logits.argmax(1)
            correct += (pred==labels).sum().item()
            total   += labels.size(0)
    model.train()
    return 100. * correct / total

# ---------------------------------------------------------------
# Main
# ---------------------------------------------------------------
def main():
    torch.manual_seed(0)
    model = BoxClassNet().to(DEVICE)
    optim_ = torch.optim.Adam(model.parameters(), lr=LR)

    for epoch in range(1,EPOCHS+1):
        t0 = time.time()
        running = 0.0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            loss, logits, boxes = model(imgs, labels)

            optim_.zero_grad()
            loss.backward()
            optim_.step()

            running += loss.item() * imgs.size(0)

        train_loss = running / len(train_loader.dataset)
        acc = evaluate(model, test_loader)
        dt  = time.time()-t0
        print(f'Epoch {epoch:2d}/{EPOCHS}  loss {train_loss:.4f}  test-acc {acc:5.2f}%  ({dt:3.1f}s)')

    # After training, we can visualise a couple of predicted boxes:
    try:
        import matplotlib.pyplot as plt
        model.eval()
        imgs, _ = next(iter(test_loader))
        imgs = imgs.to(DEVICE)[:8]
        with torch.no_grad():
            _, boxes = model(imgs)
        # convert to cpu and pixel coords
        boxes = boxes.cpu().numpy()
        imgs  = imgs.cpu().permute(0,2,3,1).numpy()

        fig,axs = plt.subplots(2,4, figsize=(8,4))
        for ax,img,box in zip(axs.flatten(), imgs, boxes):
            cx,cy,w,h = box
            x1 = (cx - w/2)*IMG_W; y1 = (cy - h/2)*IMG_H
            rect = plt.Rectangle((x1,y1), w*IMG_W, h*IMG_H, fill=False, color='red', lw=2)
            ax.imshow(img)
            ax.add_patch(rect)
            ax.axis('off')
        plt.tight_layout(); plt.show()
    except ImportError:
        pass


if __name__ == '__main__':
    main()