import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
import random

class DualYOLODataset(Dataset):
    """
    Dual branch YOLO Dataset (Visible + Infrared)
    """
    def __init__(self, image_dir, label_dir, input_size=(640, 640), 
                 augment=False, augment_params=None):
        """
        image_dir: Path to visible images directory OR .txt file list
        label_dir: Path to labels directory
        """
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.input_size = input_size
        self.augment = augment
        self.augment_params = augment_params or {}
        
        self.image_files = []
        
        # Check if image_dir is a txt file
        if image_dir.endswith('.txt') and os.path.isfile(image_dir):
            with open(image_dir, 'r') as f:
                self.image_files = [line.strip() for line in f.readlines() if line.strip()]
        else:
            # Directory mode
            if os.path.exists(image_dir):
                for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
                    self.image_files.extend(
                        [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.lower().endswith(ext)]
                    )
            self.image_files.sort()
            
        print(f"Found {len(self.image_files)} visible images in {image_dir}")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        vis_path = self.image_files[idx]
        
        # Derive IR path
        # Rule: images/visible -> images/infrared
        # We replace the last occurrence of 'visible' to be safe, or just 'visible'.
        # Assuming standard structure where 'visible' is a folder name.
        ir_path = vis_path.replace('visible', 'infrared')
        
        # Load Visible
        if not os.path.exists(vis_path):
             raise FileNotFoundError(f"Visible image not found: {vis_path}")
        img_vis = cv2.imread(vis_path)
        if img_vis is None:
            raise ValueError(f"Failed to read visible image: {vis_path}")
        img_vis = cv2.cvtColor(img_vis, cv2.COLOR_BGR2RGB)

        # Load Infrared
        if not os.path.exists(ir_path):
             raise FileNotFoundError(f"Infrared image not found: {ir_path}")
        img_ir = cv2.imread(ir_path)
        if img_ir is None:
             raise ValueError(f"Failed to read infrared image: {ir_path}")
        
        # Ensure IR is 3 channel (RGB) - Model expects 3 channels
        if len(img_ir.shape) == 2:
             img_ir = cv2.cvtColor(img_ir, cv2.COLOR_GRAY2RGB)
        else:
             img_ir = cv2.cvtColor(img_ir, cv2.COLOR_BGR2RGB)

        # Load Label
        # Derive label path from visible image path
        # Label path should be constructed from label_dir and image filename
        img_name = os.path.basename(vis_path)
        label_name = os.path.splitext(img_name)[0] + '.txt'
        label_path = os.path.join(self.label_dir, label_name)
        
        boxes = []
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f.readlines():
                    data = line.strip().split()
                    if len(data) == 5:
                        class_id = int(data[0])
                        cx, cy, w, h = map(float, data[1:5])
                        boxes.append([class_id, cx, cy, w, h])
        
        boxes = np.array(boxes, dtype=np.float32)
        if len(boxes) == 0:
            boxes = np.zeros((0, 5), dtype=np.float32)

        # Augmentation (Synchronous)
        if self.augment:
             img_vis, img_ir, boxes = self.augment_dual(img_vis, img_ir, boxes)

        # Resize / Letterbox
        h, w = img_vis.shape[:2]
        img_vis_resized, scale, pad = self.letterbox_resize(img_vis, self.input_size)
        img_ir_resized = self.apply_letterbox(img_ir, self.input_size, scale, pad)
        
        # Adjust boxes
        if len(boxes) > 0:
            boxes = self.adjust_boxes(boxes, (w, h), scale, pad)
            
        # Tensor
        tensor_vis = torch.from_numpy(img_vis_resized).permute(2, 0, 1).float() / 255.0
        tensor_ir = torch.from_numpy(img_ir_resized).permute(2, 0, 1).float() / 255.0
        
        target = {
            'boxes': torch.from_numpy(boxes),
            'image_id': idx,
            'orig_size': torch.tensor([h, w])
        }
        
        return tensor_vis, tensor_ir, target

    def letterbox_resize(self, image, target_size):
        h, w = image.shape[:2]
        target_h, target_w = target_size
        scale = min(target_w / w, target_h / h)
        new_w, new_h = int(w * scale), int(h * scale)
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        padded = np.full((target_h, target_w, 3), 114, dtype=np.uint8)
        pad_w = (target_w - new_w) // 2
        pad_h = (target_h - new_h) // 2
        padded[pad_h:pad_h+new_h, pad_w:pad_w+new_w] = resized
        return padded, scale, (pad_w, pad_h)

    def apply_letterbox(self, image, target_size, scale, pad):
        h, w = image.shape[:2]
        target_h, target_w = target_size
        new_w, new_h = int(w * scale), int(h * scale)
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        padded = np.full((target_h, target_w, 3), 114, dtype=np.uint8)
        pad_w, pad_h = pad
        padded[pad_h:pad_h+new_h, pad_w:pad_w+new_w] = resized
        return padded

    def adjust_boxes(self, boxes, orig_size, scale, pad):
        if len(boxes) == 0: return boxes
        orig_w, orig_h = orig_size
        pad_w, pad_h = pad
        target_h, target_w = self.input_size
        
        boxes_pixel = boxes.copy()
        boxes_pixel[:, 1] *= orig_w
        boxes_pixel[:, 2] *= orig_h
        boxes_pixel[:, 3] *= orig_w
        boxes_pixel[:, 4] *= orig_h
        
        boxes_pixel[:, 1] = boxes_pixel[:, 1] * scale + pad_w
        boxes_pixel[:, 2] = boxes_pixel[:, 2] * scale + pad_h
        boxes_pixel[:, 3] *= scale
        boxes_pixel[:, 4] *= scale
        
        boxes_pixel[:, 1] /= target_w
        boxes_pixel[:, 2] /= target_h
        boxes_pixel[:, 3] /= target_w
        boxes_pixel[:, 4] /= target_h
        
        boxes_pixel[:, 1:5] = np.clip(boxes_pixel[:, 1:5], 0, 1)
        return boxes_pixel

    def augment_dual(self, img_vis, img_ir, boxes):
        # HSV only on Visible
        if random.random() < 0.5:
             img_vis = self.augment_hsv(img_vis)
        
        # Flip (Synchronous)
        flip_lr = self.augment_params.get('flip_lr', 0.5)
        if random.random() < flip_lr:
            img_vis = np.fliplr(img_vis).copy()
            img_ir = np.fliplr(img_ir).copy()
            if len(boxes) > 0:
                boxes[:, 1] = 1.0 - boxes[:, 1]
        
        return img_vis, img_ir, boxes

    def augment_hsv(self, image):
        hsv_h = self.augment_params.get('hsv_h', 0.015)
        hsv_s = self.augment_params.get('hsv_s', 0.7)
        hsv_v = self.augment_params.get('hsv_v', 0.4)
        
        r = np.random.uniform(-1, 1, 3) * [hsv_h, hsv_s, hsv_v] + 1
        hue, sat, val = cv2.split(cv2.cvtColor(image, cv2.COLOR_RGB2HSV))
        dtype = image.dtype
        x = np.arange(0, 256, dtype=r.dtype)
        lut_hue = ((x * r[0]) % 180).astype(dtype)
        lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
        lut_val = np.clip(x * r[2], 0, 255).astype(dtype)
        im_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
        image = cv2.cvtColor(im_hsv, cv2.COLOR_HSV2RGB)
        return image

def collate_fn_dual(batch):
    images_vis = []
    images_ir = []
    targets = []
    for img_v, img_i, target in batch:
        images_vis.append(img_v)
        images_ir.append(img_i)
        targets.append(target)
    images_vis = torch.stack(images_vis, 0)
    images_ir = torch.stack(images_ir, 0)
    return images_vis, images_ir, targets

