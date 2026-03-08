"""
目标检测损失函数 (YOLOv8风格: CIoU + DFL + Soft-BCE + SimOTA Assigner)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def bbox_iou(box1, box2, xywh=True, GIoU=False, DIoU=False, CIoU=False, eps=1e-7):
    """
    计算IoU/GIoU/DIoU/CIoU
    Args:
        box1: [N, 4] 
        box2: [N, 4]
        xywh: 是否是(cx, cy, w, h)格式，False为(x1, y1, x2, y2)
    """
    # Get the coordinates of bounding boxes
    if xywh:  # transform from xywh to xyxy
        b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
        b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
        b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
        b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2
    else:  # x1, y1, x2, y2 coordinates
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

    # Intersection area
    inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
            (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)

    # Union Area
    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps
    union = w1 * h1 + w2 * h2 - inter + eps

    iou = inter / union
    if GIoU or DIoU or CIoU:
        cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)  # convex (smallest enclosing box) width
        ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)  # convex height
        if CIoU or DIoU:  # Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
            c2 = cw ** 2 + ch ** 2 + eps  # convex diagonal squared
            rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 +
                    (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4  # center distance squared
            if DIoU:
                return iou - rho2 / c2  # DIoU
            elif CIoU:  # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
                v = (4 / math.pi ** 2) * torch.pow(torch.atan(w2 / h2) - torch.atan(w1 / h1), 2)
                with torch.no_grad():
                    alpha = v / (v - iou + (1 + eps))
                return iou - (rho2 / c2 + v * alpha)  # CIoU
        else:  # GIoU https://arxiv.org/pdf/1902.09630.pdf
            c_area = cw * ch + eps  # convex area
            return iou - (c_area - union) / c_area  # GIoU
    return iou


class DFLLoss(nn.Module):
    """
    Distribution Focal Loss
    """
    def __init__(self, reg_max=16):
        super().__init__()
        self.reg_max = reg_max

    def forward(self, pred_dist, target):
        """
        Args:
            pred_dist: [N, reg_max+1] (logits)
            target: [N] (float values)
        """
        target = target.clamp(0, self.reg_max - 0.01)
        tl = target.long()  # target left
        tr = tl + 1  # target right
        wl = tr - target  # weight left
        wr = 1 - wl  # weight right
        
        return (F.cross_entropy(pred_dist, tl.view(-1), reduction="none") * wl +
                F.cross_entropy(pred_dist, tr.view(-1), reduction="none") * wr).mean()


class SimOTAAssigner(nn.Module):
    """
    SimOTA Assigner (Simplified OTA)
    """
    def __init__(self, center_radius=2.5, topk=10, iou_weight=3.0):
        super().__init__()
        self.center_radius = center_radius
        self.topk = topk
        self.iou_weight = iou_weight

    @torch.no_grad()
    def forward(self, pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, num_classes):
        """
    Args:
            pd_scores: [N_anchors, C] (Sigmoid applied)
            pd_bboxes: [N_anchors, 4] (xyxy)
            anc_points: [N_anchors, 2] (cx, cy)
            gt_labels: [N_gt, 1]
            gt_bboxes: [N_gt, 4] (xyxy)
    Returns:
            assigned_labels: [N_anchors]
            assigned_bboxes: [N_anchors, 4]
            assigned_scores: [N_anchors, C]
            mask_pos: [N_anchors] (bool)
        """
        num_gt = gt_bboxes.shape[0]
        num_anchors = pd_bboxes.shape[0]
        
        if num_gt == 0:
            device = pd_bboxes.device
            return (torch.full((num_anchors,), num_classes, device=device, dtype=torch.long),
                    torch.zeros_like(pd_bboxes),
                    torch.zeros_like(pd_scores),
                    torch.zeros(num_anchors, device=device, dtype=torch.bool))

        # 1. Preliminary Filtering (in_box or in_center)
        # valid_mask: [num_gt, num_anchors]
        valid_mask, is_in_boxes_and_center = self.get_in_gt_and_in_center_info(
            anc_points, gt_bboxes)
        
        # 2. Cost Matrix Calculation
        # pairwise_ious: [num_gt, num_anchors]
        pairwise_ious = self.compute_iou_matrix(gt_bboxes, pd_bboxes)
        # pairwise_ious_loss = -torch.log(pairwise_ious + 1e-8)
        
        # Classification cost
        # gt_labels: [num_gt, 1] -> one_hot -> [num_gt, C]
        gt_cls = F.one_hot(gt_labels.long().squeeze(-1), num_classes).to(pd_scores.dtype)
        gt_cls_idx = gt_labels.long().squeeze(-1) # [G]
        # scores for the corresponding GT class: [G, A]
        pairwise_pred_scores = pd_scores[:, gt_cls_idx].T # [G, A]
        
        cost_cls = F.binary_cross_entropy_with_logits(pairwise_pred_scores, torch.ones_like(pairwise_pred_scores), reduction='none')
        cost_iou = -torch.log(pairwise_ious + 1e-8)
        
        cost_matrix = cost_cls + self.iou_weight * cost_iou + 100000.0 * (~is_in_boxes_and_center)

        # 3. Dynamic K Estimation
        # K for each GT = sum(IoU of top 10 candidates)
        # topk_ious, _ = torch.topk(pairwise_ious, min(10, num_anchors), dim=1)
        # dynamic_ks = torch.clamp(topk_ious.sum(1).int(), min=1)
        
        # Simpler: just use sum of IoUs within candidate area
        ious_in_boxes_matrix = pairwise_ious
        ious_in_boxes_matrix[~is_in_boxes_and_center] = 0.0
        dynamic_ks = torch.clamp(ious_in_boxes_matrix.sum(1).int(), min=1)
        
        # 4. Assignment
        assigned_gt_inds = torch.full((num_anchors,), -1, device=pd_bboxes.device, dtype=torch.long)
        
        matching_matrix = torch.zeros_like(cost_matrix)
        
        for gt_idx in range(num_gt):
            k = dynamic_ks[gt_idx].item()
            _, pos_idx = torch.topk(cost_matrix[gt_idx], k, largest=False)
            matching_matrix[gt_idx, pos_idx] = 1.0

        # del topk_ious, dynamic_ks, pos_idx

        # Deal with conflicts (multiple GTs matched to same anchor)
        # If an anchor is matched by multiple GTs, assign to the one with min cost
        anchor_matching_gt = matching_matrix.sum(0) # [A]
        if (anchor_matching_gt > 1).sum() > 0:
            _, cost_argmin = torch.min(cost_matrix[:, anchor_matching_gt > 1], dim=0)
            matching_matrix[:, anchor_matching_gt > 1] *= 0.0
            matching_matrix[cost_argmin, anchor_matching_gt > 1] = 1.0
            
        fg_mask_inboxes = matching_matrix.sum(0) > 0.0
        
        # Result
        # target_labels: [N_pos]
        # target_bboxes: [N_pos, 4]
        # target_scores: [N_pos, C]
        
        # Initialize
        assigned_labels = torch.full((num_anchors,), num_classes, device=pd_bboxes.device, dtype=torch.long)
        assigned_bboxes = torch.zeros_like(pd_bboxes)
        assigned_scores = torch.zeros_like(pd_scores)
        
        matched_gt_inds = matching_matrix[:, fg_mask_inboxes].argmax(0)
        
        assigned_labels[fg_mask_inboxes] = gt_labels[matched_gt_inds].squeeze(-1).long()
        assigned_bboxes[fg_mask_inboxes] = gt_bboxes[matched_gt_inds]
        
        # Soft targets: IoU
        matched_ious = pairwise_ious[matched_gt_inds, torch.where(fg_mask_inboxes)[0]]
        
        # One-hot encoding with IoU
        one_hot_labels = F.one_hot(assigned_labels[fg_mask_inboxes], num_classes).to(assigned_scores.dtype)
        assigned_scores[fg_mask_inboxes] = one_hot_labels * matched_ious.unsqueeze(-1).to(assigned_scores.dtype)
        
        return assigned_labels, assigned_bboxes, assigned_scores, fg_mask_inboxes

    def compute_iou_matrix(self, box1, box2, eps=1e-9):
        # box1: [G, 4] xyxy, box2: [A, 4] xyxy
        # Returns: [G, A]
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]
        
        # intersection
        inter_x1 = torch.max(b1_x1.unsqueeze(1), b2_x1.unsqueeze(0))
        inter_y1 = torch.max(b1_y1.unsqueeze(1), b2_y1.unsqueeze(0))
        inter_x2 = torch.min(b1_x2.unsqueeze(1), b2_x2.unsqueeze(0))
        inter_y2 = torch.min(b1_y2.unsqueeze(1), b2_y2.unsqueeze(0))
        
        inter_area = (inter_x2 - inter_x1).clamp(0) * (inter_y2 - inter_y1).clamp(0)
        
        area1 = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
        area2 = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)
        
        union = area1.unsqueeze(1) + area2.unsqueeze(0) - inter_area + eps
        return inter_area / union

    def get_in_gt_and_in_center_info(self, anc_points, gt_bboxes):
        # anc_points: [A, 2], gt_bboxes: [G, 4]
        
        # is_in_boxes
        # gt_bboxes: [G, 4] -> [G, 1, 4]
        # anc_points: [A, 2] -> [1, A, 2]
        num_gt = gt_bboxes.size(0)
        num_anchors = anc_points.size(0)
        
        # expanded
        anchors_expanded = anc_points.unsqueeze(0).repeat(num_gt, 1, 1) # [G, A, 2]
        gt_bboxes_expanded = gt_bboxes.unsqueeze(1).repeat(1, num_anchors, 1) # [G, A, 4]
        
        # Check if anchor is inside GT box
        # l, t, r, b deltas
        # gt_x1, gt_y1, gt_x2, gt_y2
        l = anchors_expanded[..., 0] - gt_bboxes_expanded[..., 0]
        t = anchors_expanded[..., 1] - gt_bboxes_expanded[..., 1]
        r = gt_bboxes_expanded[..., 2] - anchors_expanded[..., 0]
        b = gt_bboxes_expanded[..., 3] - anchors_expanded[..., 1]
        
        is_in_gts = torch.stack([l, t, r, b], dim=-1).min(dim=-1)[0] > 0.01

        # Fallback for small objects: if a GT has no anchor inside it,
        # we select the nearest anchors to its center.
        # anchor_points: [A, 2] (cx, cy)
        # gt_bboxes: [G, 4]
        gt_cx = (gt_bboxes_expanded[..., 0] + gt_bboxes_expanded[..., 2]) / 2
        gt_cy = (gt_bboxes_expanded[..., 1] + gt_bboxes_expanded[..., 3]) / 2
        anc_cx = anchors_expanded[..., 0]
        anc_cy = anchors_expanded[..., 1]
        
        dists = (gt_cx - anc_cx)**2 + (gt_cy - anc_cy)**2
        valid_counts = is_in_gts.sum(dim=1)
        
        if (valid_counts == 0).any():
             problematic_gt_mask = (valid_counts == 0) # [G]
             _, nearest_indices = torch.topk(dists[problematic_gt_mask], k=1, dim=1, largest=False)
             _, nearest_indices = torch.topk(dists, k=1, dim=1, largest=False)
             nearest_mask = torch.zeros_like(is_in_gts)
             nearest_mask.scatter_(1, nearest_indices, True)
             
             is_in_gts = is_in_gts | nearest_mask
        
        return is_in_gts, is_in_gts # simplified


class DetectionLoss(nn.Module):
    """
    目标检测损失 (YOLOv8 Style: CIoU + DFL + Soft-BCE + SimOTA)
    """
    def __init__(self, num_classes, reg_max=16, lambda_box=7.5, lambda_cls=0.5, lambda_dfl=1.5):
        super(DetectionLoss, self).__init__()
        self.num_classes = num_classes
        self.reg_max = reg_max
        
        # Loss weights
        self.lambda_box = lambda_box 
        self.lambda_cls = lambda_cls 
        self.lambda_dfl = lambda_dfl 
        
        self.bce = nn.BCEWithLogitsLoss(reduction='none')
        self.dfl = DFLLoss(reg_max)
        self.assigner = SimOTAAssigner(num_classes)
    
    def forward(self, predictions, targets):
        """
        predictions: List of [B, H, W, C]
        targets: List of dict
        """
        device = predictions[0].device
        # 1. Prepare Predictions (Batch processing is complex for SimOTA, we loop over batch)
        # Flatten all scales
        pred_scores_list = []
        pred_regs_list = []
        anchors_list = []
        strides_list = []
        
        feature_strides = [8, 16, 32]
        
        # Process each scale
        for i, pred in enumerate(predictions):
            B, H, W, C = pred.shape
            stride = feature_strides[i]
            
            # [B, H, W, C] -> [B, HW, C]
            pred = pred.view(B, -1, 4*(self.reg_max+1) + self.num_classes)
            
            # Reg distribution
            pred_reg_dist = pred[..., :-self.num_classes] # [B, HW, 68]
            
            # Class scores
            pred_cls = pred[..., -self.num_classes:] # [B, HW, NC]
            pred_scores_list.append(pred_cls)
            
            # Decode Box Expectation
            # pred_reg_dist: [B, HW, 4, 17]
            pred_reg_dist = pred_reg_dist.view(B, -1, 4, self.reg_max + 1).softmax(dim=-1)
            project = torch.arange(self.reg_max + 1, dtype=torch.float, device=device)
            pred_ltrb = torch.matmul(pred_reg_dist, project) # [B, HW, 4]
            
            # Generate Grid
            y, x = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device), indexing='ij')
            x = x.flatten()
            y = y.flatten()
            
            # Anchor Points (absolute)
            ax = (x + 0.5) * stride
            ay = (y + 0.5) * stride
            anchor_points = torch.stack([ax, ay], dim=-1) # [HW, 2]
            anchors_list.append(anchor_points)
            strides_list.append(torch.full((H * W,), stride, device=device))
            
            # Decode xyxy
            # ltrb -> xyxy
            x1 = ax.unsqueeze(0) - pred_ltrb[..., 0] * stride
            y1 = ay.unsqueeze(0) - pred_ltrb[..., 1] * stride
            x2 = ax.unsqueeze(0) + pred_ltrb[..., 2] * stride
            y2 = ay.unsqueeze(0) + pred_ltrb[..., 3] * stride
            pred_xyxy = torch.stack([x1, y1, x2, y2], dim=-1) # [B, HW, 4]
            pred_regs_list.append(pred_xyxy)

        # Concat all scales
        # pd_scores: [B, N_all, C]
        pd_scores = torch.cat(pred_scores_list, dim=1)
        pd_bboxes = torch.cat(pred_regs_list, dim=1)
        anc_points = torch.cat(anchors_list, dim=0)
        strides_all = torch.cat(strides_list, dim=0) # [N_all]
        
        loss_box_sum = torch.tensor(0.0, device=device)
        loss_cls_sum = torch.tensor(0.0, device=device)
        loss_dfl_sum = torch.tensor(0.0, device=device)
        num_fg = 0
        
        # Loop over batch (SimOTA is easier per image)
        bs = len(targets)
        for b in range(bs):
            target = targets[b]
            boxes = target['boxes'].to(device) # [N, 5]
            
            p_score = pd_scores[b] # [N_anchors, C] (logits)
            p_bbox = pd_bboxes[b]  # [N_anchors, 4]
            
            if len(boxes) == 0:
                # No object
                loss_cls_sum += self.bce(p_score, torch.zeros_like(p_score)).sum()
                continue
                
            # Prepare GT
            gt_labels = boxes[:, 0:1] # [G, 1]
            # Norm xywh -> Abs xyxy
            h, w = 640, 640 # Should pass from input size
            gcx = boxes[:, 1] * w
            gcy = boxes[:, 2] * h
            gw = boxes[:, 3] * w
            gh = boxes[:, 4] * h
            gx1 = gcx - gw/2
            gy1 = gcy - gh/2
            gx2 = gcx + gw/2
            gy2 = gcy + gh/2
            gt_bboxes = torch.stack([gx1, gy1, gx2, gy2], dim=-1) # [G, 4]
            
            # SimOTA Assignment
            # p_score is logits, SimOTA needs sigmoid for cost calculation usually
            # My SimOTA implementation expects Sigmoid scores
            assigned_labels, assigned_bboxes, assigned_scores, fg_mask = self.assigner(
                p_score.sigmoid(), p_bbox, anc_points, gt_labels, gt_bboxes, self.num_classes
            )
            
            num_pos = fg_mask.sum()
            # print(f"DEBUG: Batch {b}, num_pos: {num_pos}")
            if num_pos > 0:
                num_fg += num_pos
                
                # 1. Classification Loss (Soft BCE)
                loss_cls_sum += self.bce(p_score, assigned_scores).sum()
                
                # 2. Box Loss (CIoU)
                # Only pos
                p_box_pos = p_bbox[fg_mask]
                t_box_pos = assigned_bboxes[fg_mask]
                iou = bbox_iou(p_box_pos, t_box_pos, xywh=False, CIoU=True)
                loss_box_sum += (1.0 - iou).sum()
                
                # 3. DFL Loss
                raw_pred_list = []
                for i, pred in enumerate(predictions):
                    B, H, W, C = pred.shape
                    # dist part: [B, HW, 4*17]
                    raw_pred_list.append(pred[b].view(-1, 4*(self.reg_max+1) + self.num_classes)[..., :-self.num_classes])
                raw_dist = torch.cat(raw_pred_list, dim=0) # [N_anchors, 68]
                
                pos_dist = raw_dist[fg_mask].view(-1, 4, self.reg_max + 1)
                
                # Target LTRB
                # t_box_pos is xyxy
                # anc_points_pos
                anc_pos = anc_points[fg_mask]
                t_l = anc_pos[:, 0] - t_box_pos[:, 0]
                t_t = anc_pos[:, 1] - t_box_pos[:, 1]
                t_r = t_box_pos[:, 2] - anc_pos[:, 0]
                t_b = t_box_pos[:, 3] - anc_pos[:, 1]
                t_ltrb = torch.stack([t_l, t_t, t_r, t_b], dim=-1)
                
                # Normalize t_ltrb by stride for DFL
                # DFL expects values in [0, reg_max]
                strides_pos = strides_all[fg_mask].unsqueeze(-1)
                t_ltrb = t_ltrb / strides_pos
                
                loss_dfl_sum += self.dfl(pos_dist.view(-1, self.reg_max + 1), t_ltrb.view(-1))
                
            else:
                loss_cls_sum += self.bce(p_score, torch.zeros_like(p_score)).sum()

        # Normalize
        num_fg = max(num_fg, 1)
        loss_cls = loss_cls_sum / num_fg * self.lambda_cls
        loss_box = loss_box_sum / num_fg * self.lambda_box
        loss_dfl = loss_dfl_sum / num_fg * self.lambda_dfl
        
        total_loss = loss_cls + loss_box + loss_dfl
        
        return total_loss, {
            'total': total_loss.item(),
            'cls': loss_cls.item(),
            'box': loss_box.item(),
            'dfl': loss_dfl.item()
        }

