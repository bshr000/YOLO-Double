import os
import cv2
import math
import glob
import numpy as np
from pathlib import Path


# =========================
# 路径配置（按你的实际数据修改）
# =========================
PRED_DIR = "/mnt/data/zxy/Det_double-true/inference_output"

# 原始可见光图像目录
VIS_DIR = "/mnt/data/zxy/dataset/vedai_8_double/images/visible/val"

# 原始红外图像目录
IR_DIR = "/mnt/data/zxy/dataset/vedai_8_double/images/infrared/val"

# GT标签目录（通常双模态共用同一套框）
LABEL_DIR = "/mnt/data/zxy/dataset/vedai_8_double/labels/visible/val"

# 输出对比图保存路径
SAVE_PATH = "/mnt/data/zxy/Det_double-true/top5_dual_compare_4x5.jpg"

# 可选：类别名，没有就设为 None
CLASS_NAMES = None
# 例如：
# CLASS_NAMES = ['car', 'truck', 'tractor', 'camping car', 'van', 'vehicle', 'pick-up', 'ship', 'plane']

TOPK = 5
IOU_THRES = 0.5

# 单张子图目标宽度
CELL_W = 360


# =========================
# 工具函数
# =========================
def load_yolo_labels(txt_path, img_w, img_h, with_conf=False):
    """
    读取 YOLO 标签
    GT:   cls cx cy w h
    Pred: cls cx cy w h conf
    返回格式：
        [cls_id, x1, y1, x2, y2, conf]
    """
    boxes = []
    if not os.path.exists(txt_path):
        return boxes

    with open(txt_path, 'r') as f:
        lines = [x.strip() for x in f.readlines() if x.strip()]

    for line in lines:
        parts = line.split()
        if with_conf:
            if len(parts) < 6:
                continue
            cls_id, cx, cy, bw, bh, conf = map(float, parts[:6])
        else:
            if len(parts) < 5:
                continue
            cls_id, cx, cy, bw, bh = map(float, parts[:5])
            conf = 1.0

        x1 = (cx - bw / 2) * img_w
        y1 = (cy - bh / 2) * img_h
        x2 = (cx + bw / 2) * img_w
        y2 = (cy + bh / 2) * img_h

        x1 = max(0, min(img_w - 1, x1))
        y1 = max(0, min(img_h - 1, y1))
        x2 = max(0, min(img_w - 1, x2))
        y2 = max(0, min(img_h - 1, y2))

        boxes.append([int(cls_id), x1, y1, x2, y2, conf])

    return boxes


def compute_iou(box1, box2):
    """
    box = [cls, x1, y1, x2, y2, conf]
    """
    xA = max(box1[1], box2[1])
    yA = max(box1[2], box2[2])
    xB = min(box1[3], box2[3])
    yB = min(box1[4], box2[4])

    inter_w = max(0, xB - xA)
    inter_h = max(0, yB - yA)
    inter = inter_w * inter_h

    area1 = max(0, box1[3] - box1[1]) * max(0, box1[4] - box1[2])
    area2 = max(0, box2[3] - box2[1]) * max(0, box2[4] - box2[2])
    union = area1 + area2 - inter + 1e-16
    return inter / union


def evaluate_image(gt_boxes, pred_boxes, iou_thres=0.5):
    """
    基于类别一致 + IoU阈值的贪心匹配
    返回:
        score, precision, recall, f1, mean_iou, tp, fp, fn
    """
    if len(gt_boxes) == 0 and len(pred_boxes) == 0:
        return 1.0, 1.0, 1.0, 1.0, 1.0, 0, 0, 0

    matched_gt = set()
    matched_pred = set()
    matches_iou = []

    candidates = []
    for pi, p in enumerate(pred_boxes):
        for gi, g in enumerate(gt_boxes):
            if p[0] != g[0]:
                continue
            iou = compute_iou(p, g)
            if iou >= iou_thres:
                candidates.append((iou, pi, gi))

    candidates.sort(key=lambda x: x[0], reverse=True)

    for iou, pi, gi in candidates:
        if pi not in matched_pred and gi not in matched_gt:
            matched_pred.add(pi)
            matched_gt.add(gi)
            matches_iou.append(iou)

    tp = len(matched_pred)
    fp = len(pred_boxes) - tp
    fn = len(gt_boxes) - tp

    precision = tp / (tp + fp + 1e-16)
    recall = tp / (tp + fn + 1e-16)
    f1 = 2 * precision * recall / (precision + recall + 1e-16)
    mean_iou = np.mean(matches_iou) if len(matches_iou) > 0 else 0.0

    score = 0.7 * f1 + 0.3 * mean_iou
    return score, precision, recall, f1, mean_iou, tp, fp, fn


def draw_boxes(image, boxes, class_names=None, color=(0, 255, 0), thickness=2, show_conf=False):
    img = image.copy()
    for box in boxes:
        cls_id, x1, y1, x2, y2, conf = box
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)

        if class_names is not None and 0 <= cls_id < len(class_names):
            cls_text = class_names[cls_id]
        else:
            cls_text = str(cls_id)

        label = f"{cls_text} {conf:.2f}" if show_conf else cls_text

        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        y_text = max(y1 - 4, th + 4)
        cv2.rectangle(img, (x1, y_text - th - 6), (x1 + tw + 6, y_text), color, -1)
        cv2.putText(img, label, (x1 + 3, y_text - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
    return img


def resize_keep_ratio(img, target_w):
    h, w = img.shape[:2]
    scale = target_w / w
    target_h = int(h * scale)
    return cv2.resize(img, (target_w, target_h))


def pad_to_size(img, target_h, target_w, value=255):
    h, w = img.shape[:2]
    canvas = np.full((target_h, target_w, 3), value, dtype=np.uint8)
    y = (target_h - h) // 2
    x = (target_w - w) // 2
    canvas[y:y+h, x:x+w] = img
    return canvas


def add_title(img, title, height=42, font_scale=0.7):
    canvas = np.full((img.shape[0] + height, img.shape[1], 3), 255, dtype=np.uint8)
    canvas[height:, :] = img
    cv2.putText(canvas, title, (8, 28), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), 2, cv2.LINE_AA)
    return canvas


def make_row_title(title, width, height=46):
    canvas = np.full((height, width, 3), 245, dtype=np.uint8)
    cv2.putText(canvas, title, (15, 31), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2, cv2.LINE_AA)
    return canvas


def find_image_by_stem(folder, stem):
    exts = [".jpg", ".png", ".jpeg", ".bmp", ".tif", ".tiff"]
    for ext in exts:
        p = os.path.join(folder, stem + ext)
        if os.path.exists(p):
            return p
    return None


# =========================
# 主流程
# =========================
def main():
    pred_dir = Path(PRED_DIR)
    pred_txt_list = sorted(pred_dir.glob("*.txt"))

    if len(pred_txt_list) == 0:
        print(f"未在 {PRED_DIR} 中找到预测 txt 文件。")
        return

    records = []

    for pred_txt_path in pred_txt_list:
        stem = pred_txt_path.stem

        pred_vis_path = pred_dir / f"{stem}_visible.png"
        pred_ir_path = pred_dir / f"{stem}_infrared.png"

        if not pred_vis_path.exists() or not pred_ir_path.exists():
            continue

        vis_path = find_image_by_stem(VIS_DIR, stem)
        ir_path = find_image_by_stem(IR_DIR, stem)
        gt_txt_path = os.path.join(LABEL_DIR, f"{stem}.txt")

        if vis_path is None or ir_path is None:
            continue

        vis_img = cv2.imread(vis_path)
        ir_img = cv2.imread(ir_path)

        if vis_img is None or ir_img is None:
            continue

        h, w = vis_img.shape[:2]

        gt_boxes = load_yolo_labels(gt_txt_path, w, h, with_conf=False)
        pred_boxes = load_yolo_labels(str(pred_txt_path), w, h, with_conf=True)

        score, precision, recall, f1, mean_iou, tp, fp, fn = evaluate_image(
            gt_boxes, pred_boxes, iou_thres=IOU_THRES
        )

        records.append({
            "stem": stem,
            "vis_path": vis_path,
            "ir_path": ir_path,
            "gt_txt_path": gt_txt_path,
            "pred_txt_path": str(pred_txt_path),
            "pred_vis_path": str(pred_vis_path),
            "pred_ir_path": str(pred_ir_path),
            "score": score,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "mean_iou": mean_iou,
            "tp": tp,
            "fp": fp,
            "fn": fn,
        })

    if len(records) == 0:
        print("没有找到可用于对比的样本，请检查目录配置。")
        return

    records.sort(key=lambda x: x["score"], reverse=True)
    top_records = records[:TOPK]

    # 4行：GT-VIS, GT-IR, Pred-VIS, Pred-IR
    row1_cells = []
    row2_cells = []
    row3_cells = []
    row4_cells = []

    for item in top_records:
        stem = item["stem"]

        vis_img = cv2.imread(item["vis_path"])
        ir_img = cv2.imread(item["ir_path"])
        pred_vis_img = cv2.imread(item["pred_vis_path"])
        pred_ir_img = cv2.imread(item["pred_ir_path"])

        if vis_img is None or ir_img is None or pred_vis_img is None or pred_ir_img is None:
            continue

        h, w = vis_img.shape[:2]
        gt_boxes = load_yolo_labels(item["gt_txt_path"], w, h, with_conf=False)

        gt_vis = draw_boxes(vis_img, gt_boxes, class_names=CLASS_NAMES, color=(0, 255, 0), thickness=2, show_conf=False)
        gt_ir = draw_boxes(ir_img, gt_boxes, class_names=CLASS_NAMES, color=(0, 255, 0), thickness=2, show_conf=False)

        col_title = (
            # f"{stem} | S={item['score']:.3f} "
            # f"F1={item['f1']:.3f} P={item['precision']:.3f} "
            # f"R={item['recall']:.3f} IoU={item['mean_iou']:.3f}"
            f"{stem} | IoU={item['mean_iou']:.3f}"
        )

        gt_vis = resize_keep_ratio(gt_vis, CELL_W)
        gt_ir = resize_keep_ratio(gt_ir, CELL_W)
        pred_vis_img = resize_keep_ratio(pred_vis_img, CELL_W)
        pred_ir_img = resize_keep_ratio(pred_ir_img, CELL_W)

        gt_vis = add_title(gt_vis, col_title, height=42, font_scale=0.65)
        # gt_ir = add_title(gt_ir, "GT Infrared", height=42, font_scale=0.65)
        # pred_vis_img = add_title(pred_vis_img, "Pred Visible", height=42, font_scale=0.65)
        # pred_ir_img = add_title(pred_ir_img, "Pred Infrared", height=42, font_scale=0.65)

        row1_cells.append(gt_vis)
        row2_cells.append(gt_ir)
        row3_cells.append(pred_vis_img)
        row4_cells.append(pred_ir_img)

    if len(row1_cells) == 0:
        print("没有成功生成可视化单元。")
        return

    # 每一行统一高度
    def build_row(cells):
        target_h = max(im.shape[0] for im in cells)
        target_w = max(im.shape[1] for im in cells)
        padded = [pad_to_size(im, target_h, target_w, value=255) for im in cells]
        return np.hstack(padded)

    row1 = build_row(row1_cells)
    row2 = build_row(row2_cells)
    row3 = build_row(row3_cells)
    row4 = build_row(row4_cells)

    row_title1 = make_row_title("Row 1: Original Visible + GT", row1.shape[1], height=46)
    row_title2 = make_row_title("Row 2: Original Infrared + GT", row2.shape[1], height=46)
    row_title3 = make_row_title("Row 3: Predicted Visible", row3.shape[1], height=46)
    row_title4 = make_row_title("Row 4: Predicted Infrared", row4.shape[1], height=46)

    row1 = np.vstack([row_title1, row1])
    row2 = np.vstack([row_title2, row2])
    row3 = np.vstack([row_title3, row3])
    row4 = np.vstack([row_title4, row4])

    max_w = max(row1.shape[1], row2.shape[1], row3.shape[1], row4.shape[1])

    def pad_row_width(row, width):
        if row.shape[1] == width:
            return row
        pad = np.full((row.shape[0], width - row.shape[1], 3), 255, dtype=np.uint8)
        return np.hstack([row, pad])

    row1 = pad_row_width(row1, max_w)
    row2 = pad_row_width(row2, max_w)
    row3 = pad_row_width(row3, max_w)
    row4 = pad_row_width(row4, max_w)

    final_img = np.vstack([row1, row2, row3, row4])

    header_h = 60
    header = np.full((header_h, final_img.shape[1], 3), 255, dtype=np.uint8)
    cv2.putText(
        header,
        "Top-5 Best Dual-Modal Inference Samples",
        (20, 38),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.1,
        (0, 0, 0),
        2,
        cv2.LINE_AA
    )

    final_img = np.vstack([header, final_img])

    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
    cv2.imwrite(SAVE_PATH, final_img)

    print(f"已保存对比图到: {SAVE_PATH}")
    print("\nTop-5 samples:")
    for i, item in enumerate(top_records, 1):
        print(
            f"{i}. {item['stem']} | "
            f"score={item['score']:.4f}, "
            f"f1={item['f1']:.4f}, "
            f"precision={item['precision']:.4f}, "
            f"recall={item['recall']:.4f}, "
            f"mean_iou={item['mean_iou']:.4f}, "
            f"tp={item['tp']}, fp={item['fp']}, fn={item['fn']}"
        )


if __name__ == "__main__":
    main()