# convert_int8.py — YOLOv8n ONNX → RKNN 混合精度量化
# 策略：大部分层 INT8，检测头 Conv 层保留 FP16
import os
from rknn.api import RKNN

ONNX_PATH = os.path.expanduser("~/yolov8n.onnx")
DATASET_PATH = os.path.expanduser("~/dataset.txt")
OUTPUT_PATH = os.path.expanduser("~/yolov8n_int8.rknn")

rknn = RKNN(verbose=True)

# 1. 配置：混合精度
print("=" * 40)
print("Step 1: Config (hybrid quantization)")
print("=" * 40)
rknn.config(
    mean_values=[[0, 0, 0]],
    std_values=[[255, 255, 255]],
    target_platform="rk3588",
    quantized_dtype="w8a8",
    quantized_method="channel",                       # ⬅ channel 级量化（比默认 layer 级更精细）
    optimization_level=0,                             # ⬅ 最低优化级别（避免激进融合破坏精度）
)

# 2. 加载 ONNX
print("=" * 40)
print("Step 2: Load ONNX")
print("=" * 40)
ret = rknn.load_onnx(model=ONNX_PATH)
if ret != 0:
    print("ERROR: load_onnx failed!")
    exit(1)

# 3. 构建
print("=" * 40)
print("Step 3: Build (INT8 hybrid + calibration)")
print("=" * 40)
ret = rknn.build(
    do_quantization=True,
    dataset=DATASET_PATH,
    rknn_batch_size=1,
)
if ret != 0:
    print("ERROR: build failed!")
    exit(1)

# 4. 导出
print("=" * 40)
print("Step 4: Export RKNN")
print("=" * 40)
ret = rknn.export_rknn(OUTPUT_PATH)
if ret != 0:
    print("ERROR: export failed!")
    exit(1)

# 5. 大小对比
fp16_path = os.path.expanduser("~/yolov8n.rknn")
int8_path = os.path.expanduser("~/yolov8n_int8.rknn")
fp16_size = os.path.getsize(fp16_path)
int8_size = os.path.getsize(int8_path)

print("=" * 40)
print("Size Comparison:")
print(f"  FP16: {fp16_size / 1024:.1f} KB ({fp16_size / 1024 / 1024:.2f} MB)")
print(f"  INT8: {int8_size / 1024:.1f} KB ({int8_size / 1024 / 1024:.2f} MB)")
print(f"  Reduction: {(1 - int8_size/fp16_size) * 100:.1f}%")
print("=" * 40)

rknn.release()
print("Done!")
