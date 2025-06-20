import time, numpy as np, torch, torch.nn as nn, torch.nn.functional as F



from data import *

from utils import *

# ---------------------  CONFIG  --------------------
BATCH      = 32
EPOCHS     = 30
LR         = 3e-4
LAMBDA     = 1e+0       # area-penalty weight
DEVICE     = 'cuda' if torch.cuda.is_available() else 'cpu'
# Set N_BOX to the number of polygons you want per image:
N_BOX = 1   # e.g. 2 polygons per image
N_EDGE = 4   # e.g. triangle, rectangular, etc

# --------------------  MODEL  ----------------------
class BoxClassNet(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = models.resnet50(weights='IMAGENET1K_V1')
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])  # GAP output
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.backbone.eval()

        feat_dim = 2048
        # For each requested polygon, we need N_EDGE corners × 2 coords 
        self.bbox_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(feat_dim, N_BOX * N_EDGE*2),
            nn.Sigmoid()
        )
        self.cls_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(feat_dim, NUM_CLS)
        )

    def forward(self, x):
        # Extract features from fixed backbone
        with torch.no_grad():
            feat = self.backbone(x)

        # Predicted boxes: shape (B, N_BOX, 3, 2)
        boxes = self.bbox_head(feat).view(x.size(0), N_BOX, N_EDGE, 2)

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
        area = area_from_mask(combined_mask)  # (B,)
        
        # Classify the masked image
        logits = self.cls_head(self.backbone(x_masked))


        return logits,x_masked, combined_mask, area



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
            logits,x_masked, combined_mask, area = net(x)
            ce = F.binary_cross_entropy_with_logits(logits, y)
            # add area penalty
            loss = ce + LAMBDA * (area ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            run_loss += loss.item() * x.size(0)
        sched.step()

        precision, recall, f1, avg_area = evaluate(net, val_ld)
        print(f'E{ep:02d}/{EPOCHS}  '
              f'loss {run_loss/len(train_ld.dataset):.3f}  '
              f'precision {precision:5.2f}  '
              f'recall {recall:5.2f}  '
              f'f1 {f1:5.2f}  '
              f'area {avg_area:5.2f} '
              f'{(time.time()-t0):.1f}s')
        run_validation_snapshots(net, val_ld, ep, output_dir="./triangle01")

    torch.save(net.state_dict(), 'box_class_net.pth')

if __name__ == '__main__':
    main()
