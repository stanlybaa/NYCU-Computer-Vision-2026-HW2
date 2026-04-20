import os
import json
import argparse
from PIL import Image
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from model import HW2DETR, compute_loss

class NYCUHW2Dataset(Dataset):
    def __init__(self, img_dir, json_path, is_train=True):
        self.img_dir = img_dir
        with open(json_path, 'r') as f:
            self.coco = json.load(f)
        self.images = {img['id']: img for img in self.coco['images']}
        self.annotations = {img['id']:[] for img in self.coco['images']}
        if 'annotations' in self.coco:
            for ann in self.coco['annotations']:
                self.annotations[ann['image_id']].append(ann)
        self.img_ids = list(self.images.keys())

    def __len__(self): return len(self.img_ids)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        img_info = self.images[img_id]
        img_path = os.path.join(self.img_dir, img_info['file_name'])
        image = Image.open(img_path).convert("RGB")
        orig_w, orig_h = image.size
        
        # padding
        scale = min(800 / orig_w, 800 / orig_h)
        new_w, new_h = int(orig_w * scale), int(orig_h * scale)
        image = image.resize((new_w, new_h), Image.BILINEAR)
        new_image = Image.new("RGB", (800, 800), (0, 0, 0))
        new_image.paste(image, (0, 0)) # upper-left
        
        image_tensor = T.Compose([
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
        ])(new_image)
        
        boxes, labels = [], []
        for ann in self.annotations[img_id]:
            x_min, y_min, w, h = ann['bbox']
            # relative point
            x_min_s, y_min_s, w_s, h_s = x_min * scale, y_min * scale, w * scale, h * scale
            cx = (x_min_s + w_s / 2) / 800
            cy = (y_min_s + h_s / 2) / 800
            nw = w_s / 800
            nh = h_s / 800
            
            boxes.append([cx, cy, nw, nh])
            labels.append(ann['category_id']) 
            
        target = {
            'boxes': torch.tensor(boxes, dtype=torch.float32) if boxes else torch.empty((0,4)),
            'labels': torch.tensor(labels, dtype=torch.int64) if labels else torch.empty((0,), dtype=torch.int64),
            'image_id': torch.tensor([img_id]),
            'orig_size': torch.tensor([orig_w, orig_h]),
            'scale': torch.tensor([scale])
        }
        return image_tensor, target

def collate_fn(batch):
    return torch.stack([item[0] for item in batch]), [item[1] for item in batch]

@torch.no_grad()
def evaluate(model, val_loader, coco_gt, device):
    model.eval()
    results =[]
    for images, targets in tqdm(val_loader, desc="Evaluating", leave=False):
        images = images.to(device)
        outputs = model(images)
        scores, labels = outputs['pred_logits'].softmax(-1)[..., 1:].max(-1)
        labels = labels + 1 
        boxes = outputs['pred_boxes']
        
        for i in range(len(targets)):
            img_id = targets[i]['image_id'].item()
            scale = targets[i]['scale'].item() 
            
            keep = scores[i] > 0.01
            for s, l, b in zip(scores[i][keep], labels[i][keep], boxes[i][keep]):
                cx, cy, nw, nh = b.cpu().numpy()
                
                # recover
                w_box_s, h_box_s = nw * 800, nh * 800
                x_min_s, y_min_s = (cx - nw / 2) * 800, (cy - nh / 2) * 800
                
                w_box = w_box_s / scale
                h_box = h_box_s / scale
                x_min = x_min_s / scale
                y_min = y_min_s / scale
                
                results.append({"image_id": img_id, "category_id": int(l), "bbox":[float(x_min), float(y_min), float(w_box), float(h_box)], "score": float(s)})
                
    if not results: return 0.0
    coco_dt = coco_gt.loadRes(results)
    coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    return coco_eval.stats[0]

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loader = DataLoader(NYCUHW2Dataset(f'{args.data_dir}/train', f'{args.data_dir}/train.json', True), args.batch_size, shuffle=True, collate_fn=collate_fn, num_workers=4)
    val_loader = DataLoader(NYCUHW2Dataset(f'{args.data_dir}/valid', f'{args.data_dir}/valid.json', False), args.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=4)
    coco_gt = COCO(f'{args.data_dir}/valid.json')
    
    model = HW2DETR().to(device)
    
    param_dicts = [
        {"params":[p for n, p in model.named_parameters() if "backbone" not in n and p.requires_grad]},
        {"params":[p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad], "lr": 1e-5},
    ]
    optimizer = torch.optim.AdamW(param_dicts, lr=1e-4, weight_decay=1e-4)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.1)
    
    os.makedirs('checkpoints', exist_ok=True)
    best_map = 0.0
    
    for epoch in range(args.epochs):
        model.train()
        progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for images, targets in progress:
            images = images.to(device)
            targets =[{k: v.to(device) for k, v in t.items()} for t in targets]
            
            outputs = model(images)
            loss_dict = compute_loss(outputs, targets)
            total_loss = loss_dict['ce'] + 5.0 * loss_dict['bbox'] + 2.0 * loss_dict['giou']
            
            # Auxiliary Loss 
            if 'aux_outputs' in outputs:
                for aux_out in outputs['aux_outputs']:
                    aux_loss = compute_loss(aux_out, targets)
                    total_loss += aux_loss['ce'] + 5.0 * aux_loss['bbox'] + 2.0 * aux_loss['giou']
            
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
            optimizer.step()
            
            progress.set_postfix({'ce': f"{loss_dict['ce'].item():.3f}", 'total': f"{total_loss.item():.3f}"})
            
        lr_scheduler.step()
        
        val_map = evaluate(model, val_loader, coco_gt, device)
        print(f"\nEpoch {epoch+1} | COCO mAP (0.5:0.95) = {val_map:.4f}\n")
        
        if val_map > best_map:
            best_map = val_map
            torch.save(model.state_dict(), 'checkpoints/best_model.pth')
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), f'checkpoints/model_epoch_{epoch+1}.pth')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='./nycu-hw2-data')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=4)
    main(parser.parse_args())
