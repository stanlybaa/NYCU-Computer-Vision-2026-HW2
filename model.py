import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from scipy.optimize import linear_sum_assignment

def box_cxcywh_to_xyxy(x):
    cx, cy, w, h = x.unbind(-1)
    return torch.stack([cx - 0.5 * w, cy - 0.5 * h, cx + 0.5 * w, cy + 0.5 * h], dim=-1)

def box_iou(boxes1, boxes2):
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]
    union = area1[:, None] + area2 - inter
    return inter / union, union

def generalized_box_iou(boxes1, boxes2):
    iou, union = box_iou(boxes1, boxes2)
    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])
    wh = (rb - lt).clamp(min=0)
    area = wh[:, :, 0] * wh[:, :, 1]
    return iou - (area - union) / area

class PositionEmbeddingSine(nn.Module):
    def __init__(self, num_pos_feats=128, temperature=10000):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature

    def forward(self, x):
        mask = torch.zeros((x.size(0), x.size(2), x.size(3)), dtype=torch.bool, device=x.device)
        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        eps = 1e-6
        y_embed = y_embed / (y_embed[:, -1:, :] + eps) * (2 * math.pi)
        x_embed = x_embed / (x_embed[:, :, -1:] + eps) * (2 * math.pi)
        
        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * torch.div(dim_t, 2, rounding_mode='floor') / self.num_pos_feats)
        
        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        return torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)

def generate_sine_pe(points, d_model=256, temperature=10000):
    scale = 2 * math.pi
    dim_t = torch.arange(d_model // 2, dtype=torch.float32, device=points.device)
    dim_t = temperature ** (2 * torch.div(dim_t, 2, rounding_mode='floor') / (d_model // 2))
    
    pos_x = points[..., 0:1] * scale / dim_t
    pos_y = points[..., 1:2] * scale / dim_t
    
    pos_x = torch.stack((pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()), dim=-1).flatten(-2)
    pos_y = torch.stack((pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()), dim=-1).flatten(-2)
    return torch.cat((pos_y, pos_x), dim=-1)

class CustomEncoderLayer(nn.Module):
    def __init__(self, d_model=256, nhead=8, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src, pos):
        # Pre-Norm 
        src2 = self.norm1(src)
        q = k = src2 + pos
        src2 = self.self_attn(q, k, value=src2)[0]
        src = src + self.dropout1(src2)
        
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(F.relu(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src

class ConditionalDecoderLayer(nn.Module):
    def __init__(self, d_model=256, nhead=8, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.qpos_proj = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, d_model))
        
    def forward(self, tgt, memory, query_pos, memory_pos):
        # Pre-Norm
        tgt2 = self.norm1(tgt)
        q = k = tgt2 + query_pos
        tgt2 = self.self_attn(q, k, value=tgt2)[0]
        tgt = tgt + self.dropout1(tgt2)
        
        tgt2 = self.norm2(tgt)
        spatial_modulation = self.qpos_proj(tgt2) 
        q_spatial = query_pos * spatial_modulation
        tgt2 = self.cross_attn(tgt2 + q_spatial, memory + memory_pos, value=memory)[0]
        tgt = tgt + self.dropout2(tgt2)
        
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(F.relu(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

class HW2DETR(nn.Module):
    def __init__(self, num_classes=11, d_model=256, nhead=8, num_layers=6):
        super().__init__()
        resnet = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1)
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])
        self.conv = nn.Conv2d(2048, d_model, 1)
        self.pos_embed = PositionEmbeddingSine(d_model // 2)
        
        self.encoder_layers = nn.ModuleList([CustomEncoderLayer(d_model, nhead) for _ in range(num_layers)])
        self.decoder_layers = nn.ModuleList([ConditionalDecoderLayer(d_model, nhead) for _ in range(num_layers)])
        
        self.num_queries = 100
        self.query_content = nn.Parameter(torch.zeros(self.num_queries, d_model))
        self.reference_points = nn.Parameter(torch.rand(self.num_queries, 2)) 
        
        self.class_embed = nn.Linear(d_model, num_classes)
        self.bbox_embed = nn.Sequential(
            nn.Linear(d_model, d_model), nn.ReLU(),
            nn.Linear(d_model, d_model), nn.ReLU(),
            nn.Linear(d_model, 4)
        )
        
        self._reset_parameters()

    def _reset_parameters(self):
        for name, p in self.named_parameters():
            if 'backbone' not in name and 'reference_points' not in name and p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x):
        B = x.size(0)
        h = self.conv(self.backbone(x))
        pos = self.pos_embed(h)
        
        src = h.flatten(2).permute(0, 2, 1)
        pos = pos.flatten(2).permute(0, 2, 1)
        
        for layer in self.encoder_layers:
            src = layer(src, pos)
            
        tgt = self.query_content.unsqueeze(0).repeat(B, 1, 1)
        ref_points = self.reference_points.unsqueeze(0).repeat(B, 1, 1).sigmoid()
        query_pos = generate_sine_pe(ref_points, d_model=256)
        
        intermediate =[]
        for layer in self.decoder_layers:
            tgt = layer(tgt, src, query_pos, pos)
            intermediate.append(tgt)
            
        hs = torch.stack(intermediate) 
        
        outputs_class = self.class_embed(hs)
        outputs_coord = self.bbox_embed(hs).sigmoid()
        
        out = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]}
        if self.training:
            out['aux_outputs'] =[{'pred_logits': a, 'pred_boxes': b} for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]
        return out

@torch.no_grad()
def hungarian_match(outputs, targets):
    bs, num_queries = outputs["pred_logits"].shape[:2]
    out_prob = outputs["pred_logits"].flatten(0, 1).softmax(-1)
    out_bbox = outputs["pred_boxes"].flatten(0, 1)
    
    tgt_ids = torch.cat([v["labels"] for v in targets])
    tgt_bbox = torch.cat([v["boxes"] for v in targets])
    if len(tgt_ids) == 0:
        return[(torch.tensor([], dtype=torch.int64), torch.tensor([], dtype=torch.int64)) for _ in range(bs)]
        
    cost_class = -out_prob[:, tgt_ids]
    cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)
    cost_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_bbox))
    
    C = 5.0 * cost_bbox + 1.0 * cost_class + 2.0 * cost_giou
    C = C.view(bs, num_queries, -1).cpu()
    sizes = [len(v["boxes"]) for v in targets]
    indices =[linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
    return[(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]

def compute_loss(outputs, targets):
    indices = hungarian_match(outputs, targets)
    src_logits = outputs['pred_logits']
    idx = (torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)]),
           torch.cat([src for (src, _) in indices]))
    
    target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
    target_classes = torch.zeros(src_logits.shape[:2], dtype=torch.int64, device=src_logits.device)
    target_classes[idx] = target_classes_o
    
    empty_weight = torch.ones(11, device=src_logits.device)
    empty_weight[0] = 0.1 
    loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, weight=empty_weight)
    
    src_boxes = outputs['pred_boxes'][idx]
    target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)
    
    loss_dict = {'ce': loss_ce, 'bbox': loss_ce.new_tensor(0.0), 'giou': loss_ce.new_tensor(0.0)}
    if len(target_boxes) > 0:
        loss_dict['bbox'] = F.l1_loss(src_boxes, target_boxes, reduction='none').sum() / len(target_boxes)
        loss_dict['giou'] = (1 - torch.diag(generalized_box_iou(box_cxcywh_to_xyxy(src_boxes), box_cxcywh_to_xyxy(target_boxes)))).sum() / len(target_boxes)
        
    return loss_dict
