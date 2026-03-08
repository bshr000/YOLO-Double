"""
目标检测验证脚本 (mAP计算)
"""
import os
import torch
import numpy as np
import argparse
import yaml
from tqdm import tqdm
from torch.utils.data import DataLoader

# from model import build_model
from model_dual import DualModalDetector
from dataset import DualYOLODataset, collate_fn_dual
from utils import load_checkpoint, non_max_suppression

import warnings
warnings.filterwarnings("ignore")


def ap_per_class(tp, conf, pred_cls, target_cls, plot=False, save_dir='.', names=()):
    """
    Compute the average precision, given the recall and precision curves.
    Source: https://github.com/ultralytics/yolov5/blob/master/utils/metrics.py
    """
    
    # Sort by objectness
    i = np.argsort(-conf)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]

    # Find unique classes
    unique_classes = np.unique(target_cls)
    nc = unique_classes.shape[0]  # number of classes, number of detections

    # Create Precision-Recall curve and compute AP for each class
    px, py = np.linspace(0, 1, 1000), []  # for plotting
    ap, p, r = np.zeros((nc, tp.shape[1])), np.zeros((nc, 1000)), np.zeros((nc, 1000))
    for ci, c in enumerate(unique_classes):
        i = pred_cls == c
        n_l = (target_cls == c).sum()  # number of labels
        n_p = i.sum()  # number of predictions

        if n_p == 0 or n_l == 0:
            continue

        # Accumulate FPs and TPs
        fpc = (1 - tp[i]).cumsum(0)
        tpc = (tp[i]).cumsum(0)

        # Recall
        recall = tpc / (n_l + 1e-16)  # recall curve
        r[ci] = np.interp(-px, -conf[i], recall[:, 0], left=0)  # negative x, xp because xp decreases

        # Precision
        precision = tpc / (tpc + fpc)  # precision curve
        p[ci] = np.interp(-px, -conf[i], precision[:, 0], left=1)  # p at pr_score

        # AP from recall-precision curve
        for j in range(tp.shape[1]):
            ap[ci, j], mpre, mrec = compute_ap(recall[:, j], precision[:, j])
            if plot and j == 0:
                py.append(np.interp(px, mrec, mpre))  # precision at mAP@0.5

    # Compute F1 (harmonic mean of precision and recall)
    f1 = 2 * p * r / (p + r + 1e-16)

    return p, r, ap, f1, unique_classes.astype('int')


def compute_ap(recall, precision):
    """ Compute the average precision, given the recall and precision curves
    # Arguments
        recall:    The recall curve (list)
        precision: The precision curve (list)
    # Returns
        The average precision as computed in py-faster-rcnn.
    """

    # Append sentinel values to beginning and end
    mrec = np.concatenate(([0.], recall, [1.]))
    mpre = np.concatenate(([0.], precision, [0.]))

    # Compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # Integrate area under curve
    method = 'interp'  # methods: 'continuous', 'interp'
    if method == 'interp':
        x = np.linspace(0, 1, 101)  # 101-point interp (COCO)
        ap = np.trapz(np.interp(x, mrec, mpre), x)  # integrate
    else:  # 'continuous'
        i = np.where(mrec[1:] != mrec[:-1])[0]  # points where x axis (recall) changes
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])  # area under curve

    return ap, mpre, mrec


def box_iou(box1, box2):
    # https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
        box1 (Tensor[N, 4])
        box2 (Tensor[M, 4])
    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """
    def box_area(box):
        # box = 4xn
        return (box[2] - box[0]) * (box[3] - box[1])

    area1 = box_area(box1.T)
    area2 = box_area(box2.T)

    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    inter = (torch.min(box1[:, None, 2:], box2[:, 2:]) - torch.max(box1[:, None, :2], box2[:, :2])).clamp(0).prod(2)
    return inter / (area1[:, None] + area2 - inter)


def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2]
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y


def process_batch(detections, labels, iouv):
    """
    Return correct predictions matrix. Both sets of boxes are in (x1, y1, x2, y2) format.
    Arguments:
        detections (Array[N, 6]), x1, y1, x2, y2, conf, class
        labels (Array[M, 5]), class, x1, y1, x2, y2
    Returns:
        correct (Array[N, 10]), for 10 IoU levels
    """
    correct = np.zeros((detections.shape[0], iouv.shape[0])).astype(bool)
    iou = box_iou(torch.from_numpy(detections[:, :4]), torch.from_numpy(labels[:, 1:]))
    # IoU > threshold and classes match
    # iou: [N, M]
    # detections[:, 5]: [N] (class)
    # labels[:, 0]: [M] (class)
    
    # Broadcast class matching
    # det_cls: [N, 1]
    # lbl_cls: [1, M]
    det_cls = torch.from_numpy(detections[:, 5:6])
    lbl_cls = torch.from_numpy(labels[:, 0:1]).T
    
    # Check match: [N, M]
    class_match = det_cls == lbl_cls
    
    x = torch.where((iou >= iouv[0]) & class_match)
    
    if x[0].shape[0]:
        matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).numpy()  # [det_idx, gt_idx, iou]
        if x[0].shape[0] > 1:
            matches = matches[matches[:, 2].argsort()[::-1]]
            matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
            matches = matches[matches[:, 2].argsort()[::-1]]
            matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
        
        matches = torch.from_numpy(matches)
        correct[matches[:, 0].long()] = matches[:, 2:3] >= iouv
        
    return correct


def validate(model, dataloader, device, conf_thres=0.001, nms_thres=0.6, iou_thres=0.5, verbose=False):
    model.eval()
    
    # 验证指标配置
    iouv = torch.linspace(0.5, 0.95, 10).to(device)  # iou vector for mAP@0.5:0.95
    niou = iouv.numel()
    
    stats = []  # [(correct, conf, pcls, tcls)]
    seen_labels = []  # collect all target classes for label counting even if no TP
    
    print("Collecting predictions...")
    with torch.no_grad():
        for batch_i, (images_vis, images_ir, targets) in enumerate(tqdm(dataloader, desc="Val")):
            images_vis = images_vis.to(device)
            images_ir = images_ir.to(device)
            
            # 1. Inference
            # 使用混合精度
            with torch.amp.autocast('cuda', enabled=device.type == 'cuda'):
                predictions = model(images_vis, images_ir)
                # decode_predictions 返回 [B, N, 6] (cls, conf, cx, cy, w, h) (归一化)
                # 注意: decode_predictions 内部已经做了 threshold 过滤 (但通常是比较高的阈值)
                # 这里我们在 decode 时应该用较低阈值 (如0.001) 以计算 mAP
                detections_list = model.decode_predictions(predictions, conf_threshold=conf_thres)
            
            # 2. Process Batch
            for i, det in enumerate(detections_list):
                # det: [N, 6] (cls, conf, cx, cy, w, h)
                
                # 获取当前图像的 Targets
                # target['boxes']: [M, 5] (cls, cx, cy, w, h)
                target_boxes = targets[i]['boxes'].to(device)
                
                # 转换 Targets 为绝对坐标
                # 假设 Dataset 返回归一化坐标, 需乘以图像尺寸
                _, _, h, w = images_vis.shape
                
                tcls = target_boxes[:, 0]
                if len(tcls):
                    seen_labels.append(tcls.cpu().numpy())
                tbox = xywh2xyxy(target_boxes[:, 1:5]) 
                tbox[:, [0, 2]] *= w
                tbox[:, [1, 3]] *= h
                
                # 如果没有预测框
                if len(det) == 0:
                    if len(target_boxes):
                        stats.append((np.zeros((0, niou), dtype=bool), torch.Tensor(), torch.Tensor(), tcls.cpu().numpy()))
                    continue
                
                # NMS
                # non_max_suppression 输入期望: [N, 6]
                # 但这里的 det 已经是 [N, 6] (cls, conf, cx, cy, w, h)
                # 转换 det 为绝对坐标
                det_abs = det.clone()
                det_abs[:, 2] *= w  # cx
                det_abs[:, 3] *= h  # cy
                det_abs[:, 4] *= w  # w
                det_abs[:, 5] *= h  # h
                
                # 转换为 xyxy 供 NMS 使用 (utils.py 里的 NMS 期望输入格式可能需要调整)
                # 查看 utils.py, non_max_suppression 接收 [class, conf, cx, cy, w, h]
                # 并返回相同格式。
                # 为了 IoU 计算方便，我们这里先把 xywh 转 xyxy
                
                # 执行 NMS
                det_nms = non_max_suppression(det_abs, nms_threshold=nms_thres)
                
                if len(det_nms) == 0:
                    if len(target_boxes):
                        stats.append((np.zeros((0, niou), dtype=bool), torch.Tensor(), torch.Tensor(), tcls.cpu().numpy()))
                    continue
                
                # 准备 mAP 计算数据
                # det_nms: [M, 6] (cls, conf, cx, cy, w, h)
                # 转为 xyxy
                pred_xyxy = xywh2xyxy(det_nms[:, 2:6])
                pred_conf = det_nms[:, 1]
                pred_cls = det_nms[:, 0]
                
                # [x1, y1, x2, y2, conf, cls]
                pred_cat = torch.cat((pred_xyxy, pred_conf.unsqueeze(1), pred_cls.unsqueeze(1)), 1)
                
                # tcat: [cls, x1, y1, x2, y2]
                target_cat = torch.cat((tcls.unsqueeze(1), tbox), 1)
                
                # 计算 True Positive (Correct)
                # pred_cat: [N, 6] (x1, y1, x2, y2, conf, cls)
                # target_cat: [M, 5] (cls, x1, y1, x2, y2)
                
                correct = process_batch(pred_cat.cpu().numpy(), target_cat.cpu().numpy(), iouv.cpu())
                
                stats.append((correct, pred_conf.cpu().numpy(), pred_cls.cpu().numpy(), tcls.cpu().numpy()))
                
    # 3. Compute Metrics
    print("\nComputing statistics...")
    if len(stats):
        stats = [np.concatenate(x, 0) for x in zip(*stats)]
    else:
        # No predictions at all
        stats = [np.zeros((0, niou), dtype=bool), np.array([]), np.array([]), np.array([])]
    
    # label counting (independent from TP)
    if len(seen_labels):
        all_tcls = np.concatenate(seen_labels, 0).astype(int)
        nt = np.bincount(all_tcls.flatten(), minlength=model.num_classes)
    else:
        nt = np.zeros(model.num_classes, dtype=int)

    if len(stats) and stats[0].shape[0] and stats[3].shape[0]:
        p, r, ap, f1, ap_class = ap_per_class(*stats, plot=False, names=())
        ap50, ap = ap[:, 0], ap.mean(1)  # AP@0.5, AP@0.5:0.95
        mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()
    else:
        mp, mr, map50, map = 0.0, 0.0, 0.0, 0.0
        ap50, ap = [], []
        ap_class = []
    
    # Print results
    print(f"\n{'Class':<15} {'Images':<10} {'Labels':<10} {'P':<10} {'R':<10} {'mAP@.5':<10} {'mAP@.5:.95':<10}")
    print(f"{'all':<15} {len(dataloader.dataset):<10} {int(nt.sum()):<10} {mp:<10.3f} {mr:<10.3f} {map50:<10.3f} {map:<10.3f}")
    
    # Per-class results
    if verbose and len(ap_class):
        for i, c in enumerate(ap_class):
            # print(f"{c:<15} {len(dataloader.dataset):<10} {int(nt[c]):<10} {p[i]:<10.3f} {r[i]:<10.3f} {ap50[i]:<10.3f} {ap[i]:<10.3f}")
            
            # p, r, f1: shape [num_classes, 1000]
            bi = int(np.argmax(f1[i]))  # best F1 index
            p1 = float(p[i, bi])
            r1 = float(r[i, bi])

            print(f"{c:<15} {len(dataloader.dataset):<10} {int(nt[c]):<10} {p1:<10.3f} {r1:<10.3f} {float(ap50[i]):<10.3f} {float(ap[i]):<10.3f}")
            
    return map50, map


def main(args):
    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 加载配置
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
        
    # 加载数据
    val_dataset = DualYOLODataset(
        image_dir=config['data']['val_images'],
        label_dir=config['data']['val_labels'],
        input_size=tuple(config['model']['input_size']),
        augment=False
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['train']['batch_size'],
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn_dual,
        pin_memory=True
    )
    
    # 加载模型
    # model = build_model(config).to(device)
    model = DualModalDetector(num_classes=config['data']['num_classes'])
    model = model.to(device)
    
    # 加载权重
    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = load_checkpoint(args.checkpoint)
    
    # 优先加载 EMA
    if 'ema_state_dict' in ckpt:
        print("Using EMA weights...")
        state_dict = ckpt['ema_state_dict']
    else:
        print("Using standard weights...")
        state_dict = ckpt['model_state_dict']
    
    # 清理不必要的键 (如 thop 产生的)
    unwanted = [k for k in state_dict.keys() if 'total_ops' in k or 'total_params' in k]
    for k in unwanted:
        del state_dict[k]
        
    model.load_state_dict(state_dict)
    
    # 运行验证
    validate(
        model, 
        val_loader, 
        device,
        conf_thres=args.conf_thres,
        nms_thres=args.nms_thres,
        verbose=args.verbose
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.yaml', help='config file path')
    parser.add_argument('--checkpoint', type=str, default='checkpoints_vedai_8/best_model.pth', help='checkpoint path')
    parser.add_argument('--conf-thres', type=float, default=0.001, help='confidence threshold')
    parser.add_argument('--nms-thres', type=float, default=0.6, help='NMS threshold')
    parser.add_argument('--verbose', action='store_true', help='report mAP per class')
    args = parser.parse_args()
    
    main(args)

