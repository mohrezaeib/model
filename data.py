
import time, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from torchvision.datasets import VOCDetection

BATCH      = 32
IMG_SIZE   = 224

DEVICE     = 'cuda' if torch.cuda.is_available() else 'cpu'
CLASSES    = [
    'aeroplane','bicycle','bird','boat','bottle',
    'bus','car','cat','chair','cow',
    'diningtable','dog','horse','motorbike','person',
    'pottedplant','sheep','sofa','train','tvmonitor'
]
NUM_CLS    = len(CLASSES)# ------------------  DATASET WRAPPER  --------------
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
        # Create a multi-hot vector of length NUM_CLS, initialized to 0
        label_vec = torch.zeros(len(self.cls2idx), dtype=torch.float32)

        # For every object in this image, set the corresponding class index to 1
        for o in obj:
            cls_name = o['name']
            cls_index = self.cls2idx[cls_name]
            label_vec[cls_index] = 1.0

        return img, label_vec

def get_loaders():
    train_tfm = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomAffine(20, translate=(.05,.05), scale=(.5,1.1)),
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
