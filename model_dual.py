import torch
import torch.nn as nn
import torch.nn.functional as F

# =================================================================
# 1. 基础组件 (来自 csp_backbone.py)
# =================================================================

def autopad(k, p=None, d=1):
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p

class Conv(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class Bottleneck(nn.Module):
    def __init__(self, c1, c2, shortcut=True, g=1, k=((3, 3), (3, 3)), e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))

class C2f(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

class SPPF(nn.Module):
    def __init__(self, c1, c2, k=5):
        super().__init__()
        c_ = c1 // 2
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        return self.cv2(torch.cat((x, y1, y2, self.m(y2)), 1))

class CSPDarknet(nn.Module):
    """CSPDarknet 骨干网络"""
    def __init__(self, depth_multiple=0.33, width_multiple=0.5):
        super().__init__()
        d, w = depth_multiple, width_multiple
        c = [int(x * w) for x in [64, 128, 256, 512, 1024]]
        n = [max(round(x * d), 1) if x > 1 else x for x in [3, 6, 6, 3]]

        self.stem = nn.Sequential(Conv(3, c[0], 3, 2))
        self.layer1 = nn.Sequential(Conv(c[0], c[1], 3, 2), C2f(c[1], c[1], n=n[0], shortcut=True))
        self.layer2 = nn.Sequential(Conv(c[1], c[2], 3, 2), C2f(c[2], c[2], n=n[1], shortcut=True))
        self.layer3 = nn.Sequential(Conv(c[2], c[3], 3, 2), C2f(c[3], c[3], n=n[2], shortcut=True))
        self.layer4 = nn.Sequential(Conv(c[3], c[4], 3, 2), C2f(c[4], c[4], n=n[3], shortcut=True), SPPF(c[4], c[4], 5))
        self.out_channels = [c[2], c[3], c[4]]

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        c3 = self.layer2(x)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)
        return [c3, c4, c5]

# =================================================================
# 2. 颈部与头部 (来自 model.py)
# =================================================================

class PANFPN(nn.Module):
    """PAN-FPN 结构"""
    def __init__(self, in_channels_list, out_channels=256):
        super(PANFPN, self).__init__()
        self.lateral_layers = nn.ModuleList([nn.Conv2d(in_channels, out_channels, 1) for in_channels in in_channels_list])
        self.fpn_output_layers = nn.ModuleList([nn.Conv2d(out_channels, out_channels, 3, padding=1) for _ in in_channels_list])
        self.pan_downsample_layers = nn.ModuleList([nn.Conv2d(out_channels, out_channels, 3, stride=2, padding=1) for _ in range(len(in_channels_list)-1)])
        self.pan_output_layers = nn.ModuleList([nn.Conv2d(out_channels, out_channels, 3, padding=1) for _ in in_channels_list])

    def forward(self, features):
        laterals = [lat(f) for lat, f in zip(self.lateral_layers, features)]
        for i in range(len(laterals) - 1, 0, -1):
            laterals[i - 1] += F.interpolate(laterals[i], size=laterals[i - 1].shape[2:], mode='nearest')
        fpn_feats = [out(l) for out, l in zip(self.fpn_output_layers, laterals)]
        pan_feats = [None] * len(fpn_feats)
        pan_feats[0] = self.pan_output_layers[0](fpn_feats[0])
        for i in range(1, len(fpn_feats)):
            down = self.pan_downsample_layers[i-1](pan_feats[i-1])
            pan_feats[i] = self.pan_output_layers[i](fpn_feats[i] + down)
        return pan_feats

class DetectionHead(nn.Module):
    """解耦检测头"""
    def __init__(self, in_channels, num_anchors, num_classes, reg_max=16):
        super(DetectionHead, self).__init__()
        self.num_classes = num_classes
        self.stem = nn.Conv2d(in_channels, in_channels, 1)
        self.cls_convs = nn.Sequential(nn.Conv2d(in_channels, in_channels, 3, padding=1), nn.BatchNorm2d(in_channels), nn.ReLU(inplace=True), nn.Conv2d(in_channels, in_channels, 3, padding=1), nn.BatchNorm2d(in_channels), nn.ReLU(inplace=True))
        self.cls_pred = nn.Conv2d(in_channels, num_classes, 1)
        self.reg_convs = nn.Sequential(nn.Conv2d(in_channels, in_channels, 3, padding=1), nn.BatchNorm2d(in_channels), nn.ReLU(inplace=True), nn.Conv2d(in_channels, in_channels, 3, padding=1), nn.BatchNorm2d(in_channels), nn.ReLU(inplace=True))
        self.reg_pred = nn.Conv2d(in_channels, 4 * (reg_max + 1), 1)

    def forward(self, x):
        x = self.stem(x)
        output = torch.cat([self.reg_pred(self.reg_convs(x)), self.cls_pred(self.cls_convs(x))], dim=1)
        return output.permute(0, 2, 3, 1).contiguous()

# =================================================================
# 3. 双分支整合模型 (核心修改)
# =================================================================

class DualModalDetector(nn.Module):
    def __init__(self, num_classes, backbone_config='cspdarknet_s'):
        super().__init__()
        self.num_classes = num_classes
        # 实例化两个相同的分支
        self.backbone_vis = CSPDarknet(0.33, 0.50)
        self.backbone_ir = CSPDarknet(0.33, 0.50)
        
        in_channels = self.backbone_vis.out_channels
        # 融合前的特征对齐层
        self.align_vis = nn.ModuleList([nn.Conv2d(c, c, 1) for c in in_channels])
        self.align_ir = nn.ModuleList([nn.Conv2d(c, c, 1) for c in in_channels])
        
        self.fpn = PANFPN(in_channels, 256)
        self.heads = nn.ModuleList([DetectionHead(256, 1, num_classes) for _ in range(3)])
        
        # Initialize biases for classification head
        self._init_cls_bias()

    def _init_cls_bias(self):
        # Initialize classification head bias to -log((1-p)/p) where p is small
        # This prevents instability at the beginning of training
        import math
        prior_prob = 1e-2
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        for head in self.heads:
            # head.cls_pred is a Conv2d(256, num_classes, 1)
            nn.init.constant_(head.cls_pred.bias, bias_value)

    def forward(self, x_vis, x_ir):
        """
        输入: x_vis (B,3,640,640), x_ir (B,3,640,640)
        """
        # 1. 在模型内部进行拼接
        x_concat = torch.cat([x_vis, x_ir], dim=1)  # (B, 6, 640, 640)
        
        # 2. 进行切片
        v_input = x_concat[:, :3, :, :]
        r_input = x_concat[:, 3:6, :, :]
        
        # 3. 分别通过骨干网络
        feat_v = self.backbone_vis(v_input)
        feat_r = self.backbone_ir(r_input)
        
        # 4. 在 P3, P4, P5 进行 Add 融合
        fused = []
        for i in range(len(feat_v)):
            fused.append(self.align_vis[i](feat_v[i]) + self.align_ir[i](feat_r[i]))
            
        # 5. 颈部与检测头
        fpn_outputs = self.fpn(fused)
        return [head(f) for head, f in zip(self.heads, fpn_outputs)]

    def decode_predictions(self, predictions, conf_threshold=0.5, feature_strides=(8, 16, 32)):
        """
        解码预测结果 (DFL版本, Anchor-Free)
        输出格式: List[Tensor], 每张图一个 Tensor: [N, 6] -> [cls, conf, cx, cy, w, h]
        其中 (cx,cy,w,h) 归一化到输入尺寸 (通常为 640x640 的 letterbox 后图)
        """
        device = predictions[0].device
        batch_size = predictions[0].size(0)
        all_detections = [[] for _ in range(batch_size)]

        reg_max = 16  # should match head / loss
        project = torch.arange(reg_max + 1, dtype=torch.float, device=device)

        for level, pred in enumerate(predictions):
            # pred: [B, H, W, 4*(reg_max+1) + num_classes]
            B, H, W, _ = pred.shape
            stride = feature_strides[level] if level < len(feature_strides) else feature_strides[-1]

            # 输入尺寸（letterbox 后）可由特征图尺寸 * stride 推出
            in_h = H * stride
            in_w = W * stride

            # Split: cls prob + reg dist
            pred_cls = pred[..., -self.num_classes:].sigmoid()  # [B, H, W, NC]
            pred_reg_dist = pred[..., :-self.num_classes]       # [B, H, W, 4*(reg_max+1)]
            pred_reg_dist = pred_reg_dist.view(B, H, W, 4, reg_max + 1).softmax(dim=-1)

            # DFL Expectation in bins: [B, H, W, 4]
            pred_ltrb = torch.matmul(pred_reg_dist, project)
            # Convert bins -> pixels
            pred_ltrb = pred_ltrb * float(stride)

            # Anchor points in pixels: (x+0.5, y+0.5) * stride
            grid_y, grid_x = torch.meshgrid(
                torch.arange(H, device=device),
                torch.arange(W, device=device),
                indexing='ij'
            )
            ax = (grid_x.float() + 0.5) * float(stride)
            ay = (grid_y.float() + 0.5) * float(stride)

            for b in range(B):
                scores = pred_cls[b]  # [H, W, NC]
                class_conf, class_id = torch.max(scores, dim=-1)  # [H, W]
                conf_mask = class_conf > conf_threshold
                if conf_mask.sum() == 0:
                    continue

                valid_ltrb = pred_ltrb[b][conf_mask]  # [N, 4] in pixels
                vy, vx = torch.where(conf_mask)

                anc_x = ax[vy, vx]
                anc_y = ay[vy, vx]

                x1 = anc_x - valid_ltrb[:, 0]
                y1 = anc_y - valid_ltrb[:, 1]
                x2 = anc_x + valid_ltrb[:, 2]
                y2 = anc_y + valid_ltrb[:, 3]

                # Clip to image bounds (optional, helps stability)
                x1 = x1.clamp(0, in_w)
                y1 = y1.clamp(0, in_h)
                x2 = x2.clamp(0, in_w)
                y2 = y2.clamp(0, in_h)

                w_box = (x2 - x1).clamp(min=0)
                h_box = (y2 - y1).clamp(min=0)
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2

                # Normalize to input size
                cx /= float(in_w)
                cy /= float(in_h)
                w_box /= float(in_w)
                h_box /= float(in_h)

                valid_conf = class_conf[conf_mask]
                valid_cls = class_id[conf_mask].float()

                detections = torch.stack([valid_cls, valid_conf, cx, cy, w_box, h_box], dim=1)
                all_detections[b].append(detections)

        # Merge across levels
        for b in range(batch_size):
            if len(all_detections[b]) > 0:
                all_detections[b] = torch.cat(all_detections[b], dim=0)
            else:
                all_detections[b] = torch.zeros((0, 6), device=device)

        return all_detections

if __name__ == "__main__":
    model = DualModalDetector(num_classes=8)
    # print(model) # 打印网络结构
    
    # 模拟数据
    v = torch.randn(2, 3, 640, 640)
    r = torch.randn(2, 3, 640, 640)
    
    outputs = model(v, r)
    for i, o in enumerate(outputs):
        print(f"Layer P{i+3} output shape: {o.shape}")