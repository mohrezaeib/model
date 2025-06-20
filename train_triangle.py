#!/usr/bin/env python3
# ---------------------------------------------
#  Bounding-box + masked-classification network
#  Dataset : PASCAL-VOC 2012
# ---------------------------------------------
import time, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from torchvision.datasets import VOCDetection
from torchvision.utils import save_image

from ploy import mask_polygon, area_from_mask        # custom helpers

# ---------------------  CONFIG  --------------------
BATCH      = 32
EPOCHS     = 30
LR         = 3e-4
LAMBDA     = 1e-1            # area-penalty weight
IMG_SIZE   = 224
DEVICE     = 'cuda' if torch.cuda.is_available() else 'cpu'
CLASSES    = ['aeroplane','bicycle','bird','boat','bottle',
              'bus','car','cat','chair','cow',
              'diningtable','dog','horse','motorbike','person',
              'pottedplant','sheep','sofa','train','tvmonitor']
NUM_CLS    = len(CLASSES)
N_BOX=1
# ------------------  DATASET WRAPPER  --------------
class VOCDatasetCls(Dataset):
    def __init__(self, image_set: str, tfm):
        self.ds   = VOCDetection(root='data', year='2012',
                                 image_set=image_set, download=False,
                                 transform=tfm)
        self.cls2idx = {c:i for i,c in enumerate(CLASSES)}

    def __len__(self):  return len(self.ds)

    def __getitem__(self, idx):
        img, ann = self.ds[idx]
        obj      = ann['annotation']['object']
        if not isinstance(obj, list): obj = [obj]
        label    = self.cls2idx[obj[0]['name']]
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
        backbone        = models.resnet50(weights='IMAGENET1K_V1')
        self.backbone   = nn.Sequential(*list(backbone.children())[:-1])  # GAP output
        for p in self.backbone.parameters(): p.requires_grad = False
        self.backbone.eval()

        feat_dim        = 2048
        self.bbox_head  = nn.Sequential(nn.Flatten(),
                                        nn.Linear(feat_dim, N_BOX*6),  # N_BOX×3×2
                                        nn.Sigmoid())
        self.cls_head   = nn.Sequential(nn.Flatten(),
                                        nn.Linear(feat_dim, NUM_CLS))
        self.lam        = lam

    def forward(self, x, y=None, save_imgs=False, save_path="."):
        with torch.no_grad():
            feat  = self.backbone(x)

        boxes      = self.bbox_head(feat).view(x.size(0), N_BOX, 3, 2)
        #TODO N_BOX for loop to overlay masks 
        x_masked, m = mask_polygon(img=x, vertices=boxes[:, 0], return_mask=True)

        logits     = self.cls_head(self.backbone(x_masked))

        if save_imgs:      # only for quick visual inspection
            save_image(x,         f'{save_path}/orig.png',   nrow=8, normalize=True)
            save_image(m,         f'{save_path}/mask.png',   nrow=8, normalize=True)
            save_image(x_masked,  f'{save_path}/masked.png', nrow=8, normalize=True)
            print(f'Images saved in {save_path}')

        if y is None:
            return logits, x_masked, m

        ce   = F.cross_entropy(logits, y)
        area = area_from_mask(m)          # (B,)
        loss = ce + self.lam * (area ** 2).mean()
        return loss, logits

# --------------------  UTILS  ----------------------
@torch.no_grad()
def evaluate(net, loader):
    net.eval(); top1=total=0
    for x,y in loader:
        x,y = x.to(DEVICE), y.to(DEVICE)
        pred, _, _ = net(x, save_imgs=False) if isinstance(net(x), tuple) else net(x)
        top1  += (pred.argmax(1) == y).sum().item()
        total += y.size(0)
    net.train()
    return 100. * top1 / total

# ------------------  TRAIN LOOP  -------------------
def main():
    train_ld, val_ld = get_loaders()
    net   = BoxClassNet().to(DEVICE)
    opt   = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=LR)
    sched = torch.optim.lr_scheduler.StepLR(opt, 10, gamma=0.1)

    for ep in range(1, EPOCHS+1):
        t0, run_loss = time.time(), 0.0
        for x,y in train_ld:
            x,y = x.to(DEVICE), y.to(DEVICE)
            loss, _ = net(x, y)
            opt.zero_grad(); loss.backward(); opt.step()
            run_loss += loss.item() * x.size(0)
        sched.step()

        acc = evaluate(net, val_ld)
        print(f'E{ep:02d}/{EPOCHS}  '
              f'loss {run_loss/len(train_ld.dataset):.3f}  '
              f'acc {acc:5.2f}%  {(time.time()-t0):.1f}s')

        # quick visual snapshot
        if ep % 2 == 0:
            imgs,_ = next(iter(val_ld))
            net(imgs.to(DEVICE)[:8],
                 save_imgs=True,
                        save_path=f'./triangle/epoch_{ep}_fixed')
        if ep % 2 == 0:
            idx = torch.randperm(len(val_ld.dataset))[:8]   # 8 random indices
            imgs = torch.stack([val_ld.dataset[i][0] for i in idx])
            net(imgs.to(DEVICE),
                save_imgs=True,
                save_path=f'./triangle/epoch_{ep}_random')

    torch.save(net.state_dict(), 'box_class_net.pth')

if __name__ == '__main__':
    main()
