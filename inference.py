import os
import json
import argparse
from PIL import Image
from tqdm import tqdm
import torch
import torchvision.transforms as T

from model import HW2DETR

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = HW2DETR().to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()
    
    test_dir = f'{args.data_dir}/test'
    predictions = []
    
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
    ])
    
    image_files = [f for f in os.listdir(test_dir) if f.lower().endswith('.png')]
    image_files = sorted(image_files, key=lambda x: int(os.path.splitext(x)[0]))
    
    print(f"Starting inference on {len(image_files)} test images...")
    
    with torch.no_grad():
        for file_name in tqdm(image_files, desc="Generating pred.json"):
            img_id = int(os.path.splitext(file_name)[0])
            img = Image.open(os.path.join(test_dir, file_name)).convert("RGB")
            orig_w, orig_h = img.size
            
            # Resize + Pad
            scale = min(800 / orig_w, 800 / orig_h)
            new_w, new_h = int(orig_w * scale), int(orig_h * scale)
            new_image = Image.new("RGB", (800, 800), (0, 0, 0))
            new_image.paste(img.resize((new_w, new_h), Image.BILINEAR), (0, 0))
            
            img_tensor = transform(new_image).unsqueeze(0).to(device)
            outputs = model(img_tensor)
            
            probs = outputs['pred_logits'].softmax(-1)[0]
            scores, labels = probs[:, 1:].max(-1) 
            labels = labels + 1 
            boxes = outputs['pred_boxes'][0]
            
            keep = scores > 0.01
            scores = scores[keep]
            labels = labels[keep]
            boxes = boxes[keep]
            
            for s, l, b in zip(scores, labels, boxes):
                cx, cy, nw, nh = b.cpu().numpy()
                
                # Recover
                w_box = (nw * 800) / scale
                h_box = (nh * 800) / scale
                x_min = ((cx - nw / 2) * 800) / scale
                y_min = ((cy - nh / 2) * 800) / scale
                
                x_max = x_min + w_box
                y_max = y_min + h_box
                
                x_min = max(0.0, float(x_min))
                y_min = max(0.0, float(y_min))
                x_max = min(float(orig_w), float(x_max))
                y_max = min(float(orig_h), float(y_max))
                
                final_w = x_max - x_min
                final_h = y_max - y_min
                
                if final_w > 0 and final_h > 0:
                    predictions.append({
                        "image_id": int(img_id),
                        "category_id": int(l.item()),
                        "bbox":[float(x_min), float(y_min), float(final_w), float(final_h)],
                        "score": float(s.item())
                    })

    # sorting
    predictions.sort(key=lambda x: (x["image_id"], -x["score"]))

    with open('pred.json', 'w', encoding='utf-8') as f:
        json.dump(predictions, f, indent=4)
        
    print("Done! Predictions saved to pred.json. Ready to submit!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='./nycu-hw2-data')
    parser.add_argument('--checkpoint', type=str, default='./checkpoints/best_model.pth')
    main(parser.parse_args())
