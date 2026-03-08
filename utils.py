"""
工具函数
"""
import os
import random
import numpy as np
import torch
import math
from copy import deepcopy


def setup_seed(seed):
    """
    设置随机种子
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class ModelEMA:
    """ 
    Model Exponential Moving Average from https://github.com/ultralytics/yolov5
    Keep a moving average of everything in the model state_dict (parameters and buffers).
    """
    def __init__(self, model, decay=0.9999, tau=2000, updates=0):
        # Create EMA
        self.ema = deepcopy(model.module if is_parallel(model) else model).eval()  # FP32 EMA
        self.updates = updates  # number of EMA updates
        
        # decay exponential ramp (to help early epochs)
        # decay 会随着 updates 增加逐渐从 0 增加到 0.9999
        self.decay = lambda x: decay * (1 - math.exp(-x / tau)) 

        for p in self.ema.parameters():
            p.requires_grad_(False)

    def update(self, model):
        # Update EMA parameters
        with torch.no_grad():
            self.updates += 1
            d = self.decay(self.updates)

            msd = model.module.state_dict() if is_parallel(model) else model.state_dict()  # model state_dict
            for k, v in self.ema.state_dict().items():
                if v.dtype.is_floating_point:
                    v *= d
                    v += (1 - d) * msd[k].detach()

    def update_attr(self, model, include=(), exclude=('process_group', 'reducer')):
        # Update EMA attributes
        copy_attr(self.ema, model, include, exclude)


def is_parallel(model):
    """Returns True if model is of type DP or DDP"""
    return type(model) in (torch.nn.parallel.DataParallel, torch.nn.parallel.DistributedDataParallel)


def copy_attr(a, b, include=(), exclude=()):
    # Copy attributes from b to a, options to only include() and to exclude()
    for k, v in b.__dict__.items():
        if (len(include) and k not in include) or k.startswith('_') or k in exclude:
            continue
        else:
            setattr(a, k, v)


def save_checkpoint(path, model, optimizer, epoch, loss, ema=None):
    """
    保存checkpoint
    Args:
        ema: 如果提供了EMA模型，也保存它
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'best_loss': loss
    }
    
    if ema:
        checkpoint['ema_state_dict'] = ema.ema.state_dict()
        checkpoint['ema_updates'] = ema.updates
        
    torch.save(checkpoint, path)


def load_checkpoint(path):
    """
    加载checkpoint
    """
    checkpoint = torch.load(path, map_location='cpu')
    return checkpoint


def non_max_suppression(detections, nms_threshold=0.65):
    """
    Non-Maximum Suppression
    Args:
        detections: [N, 6] tensor - [class_id, confidence, cx, cy, w, h]
        nms_threshold: IoU阈值
    Returns:
        keep_detections: [M, 6] tensor
    """
    if len(detections) == 0:
        return detections
    
    # 按置信度排序
    scores = detections[:, 1]
    sorted_indices = torch.argsort(scores, descending=True)
    detections = detections[sorted_indices]
    
    keep = []
    
    while len(detections) > 0:
        # 保留置信度最高的
        keep.append(detections[0].unsqueeze(0))
        
        if len(detections) == 1:
            break
        
        # 计算与其余box的IoU
        ious = compute_iou_batch(
            detections[0:1, 2:6], 
            detections[1:, 2:6]
        )
        
        # 过滤掉同类别且IoU大于阈值的box
        same_class = detections[0, 0] == detections[1:, 0]
        high_iou = ious[0] > nms_threshold
        remove_mask = same_class & high_iou
        
        # 保留其余的box
        keep_mask = ~remove_mask
        detections = detections[1:][keep_mask]
    
    if len(keep) > 0:
        keep_detections = torch.cat(keep, dim=0)
    else:
        keep_detections = torch.zeros((0, 6), device=detections.device)
    
    return keep_detections


def compute_iou_batch(boxes1, boxes2):
    """
    批量计算IoU
    Args:
        boxes1: [N, 4] - [cx, cy, w, h]
        boxes2: [M, 4] - [cx, cy, w, h]
    Returns:
        ious: [N, M]
    """
    # 转换为 [x1, y1, x2, y2]
    boxes1_x1 = boxes1[:, 0:1] - boxes1[:, 2:3] / 2
    boxes1_y1 = boxes1[:, 1:2] - boxes1[:, 3:4] / 2
    boxes1_x2 = boxes1[:, 0:1] + boxes1[:, 2:3] / 2
    boxes1_y2 = boxes1[:, 1:2] + boxes1[:, 3:4] / 2
    
    boxes2_x1 = boxes2[:, 0:1] - boxes2[:, 2:3] / 2
    boxes2_y1 = boxes2[:, 1:2] - boxes2[:, 3:4] / 2
    boxes2_x2 = boxes2[:, 0:1] + boxes2[:, 2:3] / 2
    boxes2_y2 = boxes2[:, 1:2] + boxes2[:, 3:4] / 2
    
    # 计算交集
    inter_x1 = torch.max(boxes1_x1, boxes2_x1.t())
    inter_y1 = torch.max(boxes1_y1, boxes2_y1.t())
    inter_x2 = torch.min(boxes1_x2, boxes2_x2.t())
    inter_y2 = torch.min(boxes1_y2, boxes2_y2.t())
    
    inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * \
                 torch.clamp(inter_y2 - inter_y1, min=0)
    
    # 计算并集
    boxes1_area = (boxes1_x2 - boxes1_x1) * (boxes1_y2 - boxes1_y1)
    boxes2_area = (boxes2_x2 - boxes2_x1) * (boxes2_y2 - boxes2_y1)
    union_area = boxes1_area + boxes2_area.t() - inter_area
    
    ious = inter_area / (union_area + 1e-6)
    
    return ious


def box_iou_numpy(box1, box2):
    """
    计算两个box的IoU (numpy版本)
    Args:
        box1, box2: [cx, cy, w, h]
    """
    # 转换为 [x1, y1, x2, y2]
    box1_x1 = box1[0] - box1[2] / 2
    box1_y1 = box1[1] - box1[3] / 2
    box1_x2 = box1[0] + box1[2] / 2
    box1_y2 = box1[1] + box1[3] / 2
    
    box2_x1 = box2[0] - box2[2] / 2
    box2_y1 = box2[1] - box2[3] / 2
    box2_x2 = box2[0] + box2[2] / 2
    box2_y2 = box2[1] + box2[3] / 2
    
    # 计算交集
    inter_x1 = max(box1_x1, box2_x1)
    inter_y1 = max(box1_y1, box2_y1)
    inter_x2 = min(box1_x2, box2_x2)
    inter_y2 = min(box1_y2, box2_y2)
    
    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    
    # 计算并集
    box1_area = (box1_x2 - box1_x1) * (box1_y2 - box1_y1)
    box2_area = (box2_x2 - box2_x1) * (box2_y2 - box2_y1)
    union_area = box1_area + box2_area - inter_area
    
    iou = inter_area / (union_area + 1e-6)
    
    return iou

