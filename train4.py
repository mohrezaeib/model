#!/usr/bin/env python3
# ---------------------------------------------------------------
#  Two–part CNN (BBox + Masked Classifier) on Caltech-101
#  BBox-head  : ResNet-18
#  Class-head : ResNet-50
# ---------------------------------------------------------------
import numpy as np
import time, torch, torch.nn as nn, torch.nn.functional as F
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from ploy import mask_polygon, area_from_mask

# -----------------------  CONFIG  -----------------------------
BATCH     = 64
EPOCHS    = 30
LR        = 3e-4
LAMBDA    = 0.10               # weight for area penalty
IMG_SZ    = 224
SOFT_K    = 40.               # edge sharpness of soft mask
DEVICE    = 'cuda' if torch.cuda.is_available() else 'cpu'
CLASSES = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 
 'bus', 'car', 'cat', 'chair', 'cow', 
 'diningtable', 'dog', 'horse', 'motorbike', 'person', 
 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']
NUM_CLS   = len(CLASSES)               

# -----------------  differentiable soft mask ------------------
def soft_box_mask(boxes, H=IMG_SZ, W=IMG_SZ, k=SOFT_K):
    """
    boxes : (B,4)  – (cx,cy,w,h) relative coords in [0,1]
    return: (B,1,H,W) mask ∈ [0,1]
    """
    B = boxes.size(0)
    cx,cy,bw,bh = boxes.unbind(1)
    x_l, x_r = (cx-bw/2).clamp(0,1), (cx+bw/2).clamp(0,1)
    y_t, y_b = (cy-bh/2).clamp(0,1), (cy+bh/2).clamp(0,1)

    xs = torch.linspace(0,1,W,device=boxes.device).view(1,1,1,W).expand(B,1,H,W)
    ys = torch.linspace(0,1,H,device=boxes.device).view(1,1,H,1).expand(B,1,H,W)

    left   = torch.sigmoid(k*(xs-x_l.view(B,1,1,1)))
    right  = torch.sigmoid(k*(x_r.view(B,1,1,1)-xs))
    top    = torch.sigmoid(k*(ys-y_t.view(B,1,1,1)))
    bottom = torch.sigmoid(k*(y_b.view(B,1,1,1)-ys))
    return left*right*top*bottom   # (B,1,H,W)

# ---------------------  MODEL PARTS ---------------------------
def build_bbox_head():
    m = torchvision.models.resnet18(weights='IMAGENET1K_V1')
    m.fc = nn.Sequential(nn.Linear(m.fc.in_features,16), nn.Sigmoid())
    return m

def build_class_head():
    m = torchvision.models.resnet50(weights='IMAGENET1K_V1')
    m.fc = nn.Linear(m.fc.in_features, NUM_CLS)
    return m

class BoxClassNet(nn.Module):
    def __init__(self, lam=LAMBDA, mu=0.5, sigma=0.5):
        super().__init__()

        backbone = torchvision.models.resnet50(weights='IMAGENET1K_V1')
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])  # upto GAP
        feat_dim = 2048
        #
        # 2. heads
        #
        self.bbox = nn.Sequential(
            nn.Flatten(),                          # GAP output is B×2048×1×1
            nn.Linear(feat_dim, 14),
            nn.Sigmoid()
        )
        self.cls  = nn.Sequential(
            nn.Flatten(),
            nn.Linear(feat_dim, NUM_CLS)
        )
         
        self.lam  = lam
        self.mu = mu
        self.sigma = sigma
    def forward(self,x,targets=None, save=True):
        feat1 = self.backbone(x) 
        boxes = self.bbox(feat1)
        boxes = boxes.reshape(x.shape[0], 7,2)   # or x.view(2, 8)
        x_m, mask  = mask_polygon(img=x,vertices= boxes, return_mask=True)          
        feat2 = self.backbone(x_m) 
        logits= self.cls(feat2)
        logits2= self.cls(feat1)
        if save:
                save_image(x_m, 'batch_output_croped.png', nrow=8, padding=2, normalize=True)                        # masked image
                save_image(x, 'batch_output_original.png', nrow=8, padding=2, normalize=True)      
                save_image(mask, 'batch_output_mask.png', nrow=8, padding=2, normalize=True)   
                print("images saved!") 
        if targets is None:
            return logits, x_m, mask
  
        ce = F.cross_entropy(logits,targets)
        ce3 = F.cross_entropy(logits2,targets)
        area = area_from_mask(mask)
        loss=ce+ce3 + 1/(area * (1 - area)) + (area)
        #print(f"box loss {boxloss}, ce loss{ce}, area {area}")
        return loss,logits,x_m, mask

# ------------------------- DATA -------------------------------
# inside get_loaders() in your main file
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import VOCDetection
from torchvision import transforms
import os


from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import VOCDetection
import torch


# Augmentation for training data
train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomAffine(
        degrees=90,                # Rotate ±15 degrees
        translate=(0.3, 0.3),      # Translate up to 10% in both directions
        scale=(0.8, 1.2)           # Zoom out to 90%, zoom in to 110%
    ),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2,
        saturation=0.2,
        hue=0.1
    ),
    transforms.RandomCrop((224, 224)),
    transforms.ToTensor(),
])

# Simple transform for validation data
val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Wrapper dataset for classification
class VOCDatasetClassification(Dataset):
    def __init__(self, year='2012', image_set='train', transform=None):
        self.dataset = VOCDetection(
            root='data',
            year=year,
            image_set=image_set,
            download=True,
            transform=transform
        )
        self.classes = CLASSES
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, target = self.dataset[idx]

        objs = target['annotation'].get('object', [])
        if not isinstance(objs, list):
            objs = [objs]

        # Take the first object's class
        first_obj = objs[0]
        class_name = first_obj['name']
        class_idx = self.class_to_idx[class_name]

        return img, class_idx

# Set up train and validation DataLoaders
batch_size = 32

train_dataset = VOCDatasetClassification(image_set='train', transform=train_transform)
val_dataset = VOCDatasetClassification(image_set='val', transform=val_transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

# ---------------------  EVALUATION ----------------------------
@torch.no_grad()
def evaluate(model,loader):
    model.eval(); top1=total=0
    for x,y in loader:
        x,y = x.to(DEVICE),y.to(DEVICE)
        logits,_, _=model(x, save=False)
        top1 += (logits.argmax(1)==y).sum().item()
        total+=y.size(0)
    model.train()
    return 100*top1/total

# ----------------------  TRAIN LOOP ---------------------------
def main():
    torch.manual_seed(0)
    net = BoxClassNet().to(DEVICE)
    opt = torch.optim.AdamW(net.parameters(),lr=LR,weight_decay=1e-4)
    sched= torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=EPOCHS)

    for ep in range(1,EPOCHS+1):
        t0=time.time(); run=0
        for x,y in train_loader:
            x,y = x.to(DEVICE),y.to(DEVICE)
            loss,_,_ ,_= net(x,y, save=False)
            opt.zero_grad()
            loss.backward()
            opt.step()
            run += loss.item()*x.size(0)
        sched.step()
        acc = evaluate(net,val_loader)
        print(f'E{ep:02d}/{EPOCHS}  loss {run/len(train_loader.dataset):.3f}  '
              f'test acc {acc:5.2f}%  {(time.time()-t0):.1f}s')
        
         # After training, we can visualise a couple of predicted boxes:
        try:
            import matplotlib.pyplot as plt
            net.eval()
            imgs, labels = next(iter(val_loader))
            imgs = imgs.to(DEVICE)[:8]
            labels = labels[:8]  # ground truth labels
            with torch.no_grad():
                 logits, masked_img, mask = net(imgs, targets=None, save=True)
            #mask = torch.stack((mask, mask, mask), dim=1)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            labels = labels.cpu().numpy()

            imgs = imgs.cpu().permute(0, 2, 3, 1).numpy()
            mask = mask.cpu().permute(0, 2, 3, 1).numpy()
            masked_img = masked_img.cpu().permute(0, 2, 3, 1).numpy()
 
            # mean = np.array([0.485, 0.456, 0.406])
            # std = np.array([0.229, 0.224, 0.225])
            # imgs = imgs * std + mean
            # imgs = np.clip(imgs, 0, 1)

            # Suppose imgs, masks, masked_imgs, preds, labels, CLASSES, and ep are defined beforehand

            fig, axs = plt.subplots(len(imgs), 3, figsize=(9, 3 * len(imgs)))

            for i in range(len(imgs)):
                # Column 1: Original image
                axs[i, 0].imshow(imgs[i])
                axs[i, 0].set_title(f"Original\nGT: {CLASSES[labels[i]]}\nPred: {CLASSES[preds[i]]}", fontsize=8)
                axs[i, 0].axis('off')
                
                # Column 2: Mask
                axs[i, 1].imshow(mask[i])
                axs[i, 1].set_title("Mask", fontsize=8)
                axs[i, 1].axis('off')
                
                # Column 3: Masked image
                axs[i, 2].imshow(masked_img[i])
                axs[i, 2].set_title("Masked Image", fontsize=8)
                axs[i, 2].axis('off')

            plt.tight_layout()
            plt.savefig(f"boxing{ep}.png")
            
        except ImportError:
            pass
    torch.save(net.state_dict(),'boxmodel.pth')

if __name__ == '__main__':
    main()