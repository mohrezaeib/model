# ────────────────────────────────────────────────────────────
# SmallObjectCoco – PyTorch Dataset
# Keeps only images that contain at least one object
# whose bounding-box area ≤ max_area_fraction
# AND whose box centre is at least min_offcentre away
# (IoU with image centre ≤ min_offcentre).
# Returns:
#   image   : tensor 3×224×224   (random crop around the object)
#   cls_lbl : int   (COCO category id 0…79, remapped to 0…C-1)
# ────────────────────────────────────────────────────────────
import json, random, pathlib
from torchvision.datasets.utils import download_url
from pycocotools.coco import COCO
from PIL import Image
import torchvision.transforms.functional as TF
import torch
from torch.utils.data import Dataset

class SmallObjectCoco(Dataset):
    def __init__(self, root, split='train', 
                 max_area_fraction=0.20,
                 min_offcentre=0.20,
                 out_size=224,
                 transform=None,
                 categories_keep=None):
        """
        root              : path that contains coco/{train2017,val2017,annotations}
        split             : 'train' | 'val'
        max_area_fraction : keep boxes whose (w*h)/(W*H) ≤ this value
        min_offcentre     : distance of box centre to image centre,
                            expressed as fraction of min(W,H).  0.5 = edge.
        categories_keep   : optional list of COCO category ids to keep
        """
        self.transform = transform
        self.out_size  = out_size
        ann_path = pathlib.Path(root, 'annotations',
                                f'instances_{split}2017.json')
        self.coco = COCO(str(ann_path))
        img_dir   = pathlib.Path(root, f'{split}2017')

        # build list of (img_id, box, cls) that satisfy constraints
        keep = []
        for img_id in self.coco.getImgIds():
            anns = self.coco.loadAnns(self.coco.getAnnIds(imgIds=img_id,
                                                          iscrowd=False))
            img_info = self.coco.loadImgs(img_id)[0]
            W, H = img_info['width'], img_info['height']
            cx_img, cy_img = 0.5*W, 0.5*H
            for a in anns:
                if categories_keep and a['category_id'] not in categories_keep:
                    continue
                x,y,w,h = a['bbox']
                area_frac = (w*h)/(W*H)
                if area_frac > max_area_fraction:
                    continue
                cx_box, cy_box = x + 0.5*w, y + 0.5*h
                dx = abs(cx_box - cx_img) / min(W,H)
                dy = abs(cy_box - cy_img) / min(W,H)
                offcentre = max(dx, dy)
                if offcentre < min_offcentre:
                    continue
                keep.append((img_id, (x,y,w,h), a['category_id']))
        print(f'Small-object subset ({split}): {len(keep)} samples')
        self.keep = keep
        self.img_dir = img_dir

        # map COCO category ids to 0…C-1
        cats = sorted({c for _,_,c in keep})
        self.cat2new = {c:i for i,c in enumerate(cats)}

    def __len__(self): return len(self.keep)

    def __getitem__(self, idx):
        img_id, bbox, cat = self.keep[idx]
        img_path = self.img_dir / self.coco.loadImgs(img_id)[0]['file_name']
        img = Image.open(img_path).convert('RGB')
        x,y,w,h = bbox

        # Random jitter around the small object, then Centre-Crop to out_size
        #  -> so the object is not guaranteed to be central in the *crop* either
        jitter = 0.15
        jx = random.uniform(-jitter, jitter)*w
        jy = random.uniform(-jitter, jitter)*h
        cx = x + 0.5*w + jx
        cy = y + 0.5*h + jy
        side = max(w, h) * random.uniform(2.0, 3.5)  # include plenty of background
        left = int(max(0, cx - 0.5*side))
        top  = int(max(0, cy - 0.5*side))
        right= int(min(img.width,  left + side))
        bottom=int(min(img.height, top + side))
        crop = img.crop((left, top, right, bottom))
        crop = crop.resize((self.out_size, self.out_size), Image.BILINEAR)

        if self.transform: crop = self.transform(crop)
        label = self.cat2new[cat]
        return crop, label