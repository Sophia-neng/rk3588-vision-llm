import sys
import numpy as np
import cv2

sys.path.insert(0, "/home/lineng/rknn-yolo-demo/src")
from rknn_executor import RKNN_model_container

model = RKNN_model_container("/home/lineng/rknn-yolo-demo/model/yolov8n.rknn", 0)
img = cv2.imread("/home/lineng/rknn-yolo-demo/demo.png")
h, w = img.shape[:2]
scale = min(640.0 / w, 640.0 / h)
nw = int(w * scale)
nh = int(h * scale)
resized = cv2.resize(img, (nw, nh))
lb = np.full((640, 640, 3), 114, dtype=np.uint8)
lb[0:nh, 0:nw] = resized
x = cv2.cvtColor(lb, cv2.COLOR_BGR2RGB)
x = np.expand_dims(x, 0)

out = model.run([x])[0]
scores = out[0, 4:, :].max(axis=0)
top5 = np.argsort(scores)[-5:]

for idx in top5[::-1]:
    cx = out[0, 0, idx]
    cy = out[0, 1, idx]
    bw = out[0, 2, idx]
    bh = out[0, 3, idx]
    sc = scores[idx]
    print("box[%d] cx=%.1f cy=%.1f w=%.1f h=%.1f sc=%.3f" % (idx, cx, cy, bw, bh, sc))
