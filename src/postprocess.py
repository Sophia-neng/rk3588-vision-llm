"""
标准 YOLOv8 单输出后处理
======================
输入: NPU 输出 (1, 84, 8400)
  通道 0:  cx (像素坐标)
  通道 1:  cy (像素坐标)
  通道 2:  w (像素)
  通道 3:  h (像素)
  通道 4-83: 80 类别置信度 (已过 sigmoid)
输出: xyxy 像素坐标 + 类别 + 置信度

数据证明：box[6988] cx=449.0 cy=216.2 w=67.2 h=62.5
  → 这是像素坐标（640×640 letterbox 空间内），不是网格单位。
"""

import numpy as np


def nms(boxes, scores, iou_thres=0.45):
    """纯 numpy NMS"""
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []

    while order.size > 0:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter)

        order = order[1:][iou <= iou_thres]

    return np.array(keep, dtype=np.int64)


def postprocess(output, conf_thres=0.25, iou_thres=0.45, img_size=(640, 640)):
    """
    输入: output = NPU 输出 (1, 84, 8400)
    返回: (boxes, classes, scores)
    """
    out = output[0].T  # (8400, 84)

    cx = out[:, 0]
    cy = out[:, 1]
    bw = out[:, 2]
    bh = out[:, 3]
    cls_raw = out[:, 4:]  # (8400, 80)

    # cxcywh → xyxy (已经是像素坐标，直接算)
    x1 = cx - bw / 2.0
    y1 = cy - bh / 2.0
    x2 = cx + bw / 2.0
    y2 = cy + bh / 2.0
    boxes = np.stack([x1, y1, x2, y2], axis=1)

    # 最高分类别
    scores = cls_raw.max(axis=1)
    class_ids = cls_raw.argmax(axis=1)

    # 置信度过滤
    mask = scores > conf_thres
    boxes = boxes[mask]
    scores = scores[mask]
    class_ids = class_ids[mask]

    if len(boxes) == 0:
        return np.array([]), np.array([]), np.array([])

    # 裁剪
    boxes[:, 0] = np.clip(boxes[:, 0], 0, img_size[0])
    boxes[:, 1] = np.clip(boxes[:, 1], 0, img_size[1])
    boxes[:, 2] = np.clip(boxes[:, 2], 0, img_size[0])
    boxes[:, 3] = np.clip(boxes[:, 3], 0, img_size[1])

    # NMS
    indices = nms(boxes, scores, iou_thres)
    return boxes[indices], class_ids[indices], scores[indices]
