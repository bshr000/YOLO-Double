"""
推理脚本 (Dual Branch)
"""
import os
import cv2
import torch
import yaml
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm
import random

from model_dual import DualModalDetector
from utils import load_checkpoint, non_max_suppression

import warnings
warnings.filterwarnings("ignore")

def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2]
    y = np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y


def letterbox(image, new_shape=(640, 640), color=(114, 114, 114)):
    """
    Resize + pad to meet new_shape while keeping aspect ratio.
    Returns: img, ratio, (pad_w, pad_h)
    """
    shape = image.shape[:2]  # current shape [h, w]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    new_h, new_w = new_shape

    # Scale ratio (new / old)
    r = min(new_w / shape[1], new_h / shape[0])
    new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))  # w, h

    # Compute padding
    pad_w = new_w - new_unpad[0]
    pad_h = new_h - new_unpad[1]
    pad_w /= 2
    pad_h /= 2

    if shape[::-1] != new_unpad:
        image = cv2.resize(image, new_unpad, interpolation=cv2.INTER_LINEAR)

    top, bottom = int(round(pad_h - 0.1)), int(round(pad_h + 0.1))
    left, right = int(round(pad_w - 0.1)), int(round(pad_w + 0.1))
    image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)

    return image, r, (left, top)

def plot_one_box(x, img, color=None, label=None, line_thickness=3):
    # Plots one bounding box on image img
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)

def inference(args):
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load Config
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # Load Model
    print(f"Loading model from {args.checkpoint}...")
    # model = build_model(config).to(device)
    model = DualModalDetector(num_classes=config['data']['num_classes'])
    model = model.to(device)
    
    ckpt = load_checkpoint(args.checkpoint)
    
    if 'ema_state_dict' in ckpt:
        state_dict = ckpt['ema_state_dict']
    else:
        state_dict = ckpt['model_state_dict']
        
    # Clean keys
    unwanted = [k for k in state_dict.keys() if 'total_ops' in k or 'total_params' in k]
    for k in unwanted:
        del state_dict[k]
        
    model.load_state_dict(state_dict)
    model.eval()
    
    # Prepare Image List
    source = Path(args.source)
    if source.is_file() and source.suffix == '.txt':
        with open(source, 'r') as f:
            image_paths = [x.strip() for x in f.readlines()]
    elif source.is_dir():
        image_paths = sorted(list(source.glob('*.jpg')) + list(source.glob('*.png')) + list(source.glob('*.jpeg')))
    else:
        image_paths = [source]
        
    # Colors
    colors = [[np.random.randint(0, 255) for _ in range(3)] for _ in range(100)]
    
    # Inference Loop
    print(f"Starting inference on {len(image_paths)} images...")
    input_size = tuple(config['model']['input_size'])
    
    for img_path in tqdm(image_paths):
        img_path = str(img_path)
        
        # Derive IR path
        ir_path = img_path.replace('visible', 'infrared')
        
        # Read & Preprocess Visible
        img_vis_raw = cv2.imread(img_path)  # BGR
        if img_vis_raw is None:
            print(f"Failed to load visible {img_path}")
            continue
            
        # Read & Preprocess IR
        if not os.path.exists(ir_path):
            print(f"Infrared image not found: {ir_path}, skipping.")
            continue
        img_ir_raw = cv2.imread(ir_path)
        if img_ir_raw is None:
            print(f"Failed to load infrared {ir_path}")
            continue
        
        # Ensure IR is 3 channel
        if len(img_ir_raw.shape) == 2:
            img_ir_raw = cv2.cvtColor(img_ir_raw, cv2.COLOR_GRAY2BGR)  # read as BGR
        
        # Letterbox for VIS
        h0, w0 = img_vis_raw.shape[:2]
        img_vis, ratio, (pad_w, pad_h) = letterbox(img_vis_raw, new_shape=input_size)
        
        # Letterbox for IR
        img_ir, _, _ = letterbox(img_ir_raw, new_shape=input_size)
            
        # Normalize VIS
        img_vis = img_vis[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3xHxW
        img_vis = np.ascontiguousarray(img_vis)
        img_vis = torch.from_numpy(img_vis).to(device)
        img_vis = img_vis.float()
        img_vis /= 255.0
        if img_vis.ndimension() == 3:
            img_vis = img_vis.unsqueeze(0)
            
        # Normalize IR
        img_ir = img_ir[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3xHxW
        img_ir = np.ascontiguousarray(img_ir)
        img_ir = torch.from_numpy(img_ir).to(device)
        img_ir = img_ir.float()
        img_ir /= 255.0
        if img_ir.ndimension() == 3:
            img_ir = img_ir.unsqueeze(0)
            
        # Inference
        det = []
        with torch.no_grad():
            with torch.amp.autocast('cuda', enabled=device.type == 'cuda'):
                pred = model(img_vis, img_ir)
                # Decode
                detections = model.decode_predictions(pred, conf_threshold=args.conf_thres)
                
                det = detections[0]  # batch size 1
                if len(det):
                    det = non_max_suppression(det, nms_threshold=args.nms_thres)
                    
        # Post-process & Save
        name = Path(img_path).stem
        save_path_vis = output_dir / f"{name}_visible.png"
        save_path_ir = output_dir / f"{name}_infrared.png"
        txt_path = output_dir / f"{name}.txt"
        
        if len(det):
            # det is normalized to letterboxed input size [cls, conf, cx, cy, w, h]
            det = det.cpu().numpy()

            # Convert to letterbox pixel coords (x1,y1,x2,y2)
            inp_h, inp_w = input_size
            boxes_xyxy = xywh2xyxy(det[:, 2:6])
            boxes_xyxy[:, [0, 2]] *= inp_w
            boxes_xyxy[:, [1, 3]] *= inp_h

            # Undo letterbox to original image coords
            boxes_xyxy[:, [0, 2]] -= pad_w
            boxes_xyxy[:, [1, 3]] -= pad_h
            boxes_xyxy[:, :4] /= ratio

            # Clip to original image
            boxes_xyxy[:, [0, 2]] = boxes_xyxy[:, [0, 2]].clip(0, w0)
            boxes_xyxy[:, [1, 3]] = boxes_xyxy[:, [1, 3]].clip(0, h0)

            # Draw & save txt
            for i, box in enumerate(boxes_xyxy):
                cls_id = int(det[i, 0])
                conf = float(det[i, 1])

                label = f"{cls_id} {conf:.2f}"

                # 在可见光图像上画框
                plot_one_box(
                    box,
                    img_vis_raw,
                    label=label,
                    color=colors[cls_id % len(colors)],
                    line_thickness=2
                )

                # 在红外图像上画框
                plot_one_box(
                    box,
                    img_ir_raw,
                    label=label,
                    color=colors[cls_id % len(colors)],
                    line_thickness=2
                )

                if args.save_txt:
                    # 每次推理该图片前先清空旧结果，避免累计
                    if i == 0:
                        open(txt_path, 'w').close()
                    # Save YOLO normalized to ORIGINAL image (cx,cy,w,h,conf)
                    x1, y1, x2, y2 = box
                    cx = (x1 + x2) / 2 / w0
                    cy = (y1 + y2) / 2 / h0
                    bw = (x2 - x1) / w0
                    bh = (y2 - y1) / h0
                    with open(txt_path, 'a') as f:
                        f.write(f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f} {conf:.6f}\n")
                        
        cv2.imwrite(str(save_path_vis), img_vis_raw)
        cv2.imwrite(str(save_path_ir), img_ir_raw)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default="/mnt/data/zxy/dataset/vedai_8/test.txt", help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--config', type=str, default='config.yaml', help='model config')
    parser.add_argument('--checkpoint', type=str, default='checkpoints_vedai_8/best_model.pth', help='model checkpoint')
    parser.add_argument('--output', type=str, default='inference_output', help='output directory')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--nms-thres', type=float, default=0.45, help='NMS threshold')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    args = parser.parse_args()
    
    inference(args)