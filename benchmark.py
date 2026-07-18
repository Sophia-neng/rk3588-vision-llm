import sys
sys.path.insert(0, 'src')
from rknn_executor import RKNN_model_container
import numpy as np
import time

model = RKNN_model_container('model/model.rknn', 0)
dummy = np.ones((1, 640, 640, 3), dtype=np.uint8)

for _ in range(5):
    model.run([dummy])

start = time.time()
for _ in range(30):
    model.run([dummy])
elapsed = time.time() - start
print(f'30次: {elapsed:.2f}s, 平均: {elapsed/30*1000:.1f}ms, FPS: {30/elapsed:.1f}')
