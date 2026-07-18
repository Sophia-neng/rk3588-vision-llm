"""
测试自己的 yolov8n.rknn 模型推理
用法: python3 test_my_model.py
"""
import sys
sys.path.insert(0, '/home/lineng/rknn-yolo-demo/src')
from rknn_executor import RKNN_model_container
from postprocess import postprocess
import cv2
import numpy as np
import time

# 1. 加载模型
print("Loading model...")
model = RKNN_model_container('/home/lineng/rknn-yolo-demo/model/yolov8n.rknn', 0)
print("Model loaded.\n")

# 2. 读图片
img = cv2.imread('/home/lineng/rknn-yolo-demo/demo.png')
h, w = img.shape[:2]
print(f"Image: {w}x{h}")

# 3. 预处理（letterbox + BGR2RGB）
# 等比缩放，灰边补齐到 640x640
scale = min(640 / w, 640 / h)
new_w, new_h = int(w * scale), int(h * scale)
resized = cv2.resize(img, (new_w, new_h))
letterbox = np.full((640, 640, 3), 114, dtype=np.uint8)
letterbox[0:new_h, 0:new_w] = resized
input_tensor = cv2.cvtColor(letterbox, cv2.COLOR_BGR2RGB)
input_tensor = np.expand_dims(input_tensor, axis=0)  # HWC → NHWC (1,640,640,3)

# 4. NPU 推理
print("Running inference...")
start = time.time()
outputs = model.run([input_tensor])
infer_time = (time.time() - start) * 1000
print(f"Inference: {infer_time:.1f}ms")

# 5. 后处理
boxes, classes, scores = postprocess(outputs[0])
print(f"Detections: {len(boxes)}")

# 6. 把结果画回原图（注意要把坐标从 640x640 映射回原图）
scale_x = w / 640.0
scale_y = h / 640.0

for box, cls_id, score in zip(boxes, classes, scores):
    x1, y1, x2, y2 = box
    x1 = int(x1 * scale_x)
    y1 = int(y1 * scale_y)
    x2 = int(x2 * scale_x)
    y2 = int(y2 * scale_y)
    print(f"  [{cls_id}] ({x1},{y1})-({x2},{y2})  conf={score:.3f}")
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

cv2.imwrite('/home/lineng/rknn-yolo-demo/result_my_model.png', img)
print("\nResult saved to result_my_model.png")
