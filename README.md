# RK3588 多模态视觉理解系统

在 Orange Pi 5 Pro (RK3588S, 6 TOPS NPU) 上部署 **YOLOv8n 目标检测 + Qwen2.5-0.5B 场景描述** 的双模型协同推理管线，端到端延迟 ~314ms。

![Platform](https://img.shields.io/badge/platform-Orange%20Pi%205%20Pro-orange)
![NPU](https://img.shields.io/badge/NPU-RK3588S%206%20TOPS-blue)
![Status](https://img.shields.io/badge/status-completed-brightgreen)
![License](https://img.shields.io/badge/license-MIT-lightgrey)

## 管线架构

```mermaid
graph TD
    A[🖼️ 输入图片] --> B[OpenCV 预处理<br/>resize → 640×640]
    B --> C[YOLOv8n NPU 推理<br/>FP16 · 33ms]
    C --> D[C++ 后处理<br/>NMS + decode]
    D --> E[构建 Prompt]
    E --> F[Qwen2.5-0.5B NPU 推理<br/>W8A8 · 281ms]
    F --> G[📝 场景描述文本]
    
    style C fill:#4CAF50,color:#fff
    style F fill:#2196F3,color:#fff
```

## 动机

**为什么是双模型而不是多模态大模型？** 端侧多模态模型（如 Qwen2.5-VL）在 0.5B 规模的视觉理解能力极弱。YOLOv8n（33ms）擅长目标检测，Qwen2.5-0.5B 擅长文本生成——两个小模型各司其职，比一个大模型勉强做两件事效果更好，且 NPU 分时调度即可，无需额外算力。

**为什么选这两个型号？** **YOLOv8n**：最小的 YOLOv8 变体，检测精度够用，推理 ~33ms。**Qwen2.5-0.5B**：端侧能跑的最小可用 LLM，W8A8 压缩后 764MB。两个模型加起来能在 8GB 板端内存共存。这个组合证明了"端侧视觉理解"的可行性下限——两三年后模型更小更快，方案会更成熟。

## 快速开始

```bash
# 1. 克隆仓库
git clone https://github.com/Sophia-neng/rk3588-vision-llm.git
cd rk3588-vision-llm

# 2. 编译（需要 RKNN Runtime + RKLLM Runtime 库）
g++ -std=c++17 src/vision_llm_demo.cpp -o vision_llm_demo \
  -I/usr/include/opencv4 -L/usr/lib/aarch64-linux-gnu \
  -lopencv_core -lopencv_imgproc -lopencv_imgcodecs -lopencv_highgui \
  -lrknnrt -lrkllmrt

# 3. 运行（必须参数）
export LD_LIBRARY_PATH=/path/to/libs:$LD_LIBRARY_PATH
./vision_llm_demo \
  --llm   /path/to/model.rkllm \
  --model /path/to/yolov8n.rknn \
  --image /path/to/image.jpg

# 完整参数
./vision_llm_demo \
  --llm     /path/to/model.rkllm \
  --model   /path/to/yolov8n.rknn \
  --image   /path/to/image.jpg \
  --tokens  128 \
  --context 2048 \
  --conf    0.25 \
  --iou     0.45
```

## 性能速查

| 指标                 | 数值         |
| -------------------- | ------------ |
| YOLOv8n NPU 推理     | 33 ms        |
| LLM NPU 推理         | 281 ms       |
| 端到端总延迟         | ~314 ms      |
| LLM TTFT（首 token） | ~90 ms       |
| LLM 生成速度         | ~35-42 tok/s |
| 双模型内存占用       | ~1.5 GiB     |
| 板端剩余内存         | ~5.5 GiB     |

> LLM 占端到端延迟 ~90%，YOLO 几乎不拖后腿。优化重点在 LLM 侧。LLM 推理数据基于 W8A8 量化模型（943MB→764MB）。

## 效果展示

| 输入              | YOLO 检出 | Qwen 场景描述              | 评价       |
| ----------------- | --------- | -------------------------- | ---------- |
| 🚌 bus.jpg         | 4人 1公交 | "一辆公交车上坐着四个人。" | ✅          |
| 👥 people.jpg      | 7人       | "有7个人。"                | ✅          |
| 🛋️ indoor.jpg      | 4人 2沙发 | "四个人坐在沙发上。"       | ⚠️ 漏小孩   |
| 🐕 cat_dog.jpg     | 3狗       | "一组狗图像。"             | ⚠️猫→狗混淆 |
| 🚗 cars_people.jpg | 19人 17车 | "有19个人、17辆汽车。"     | ⚠️ 密集漏检 |

**边界分析**：YOLOv8n 是 nano 版本（COCO mAP 37.3%），简单场景精准，密集遮挡场景存在漏检误检。这是 nano 版本的速度-精度 design tradeoff，不是 pipeline 缺陷。详见 [`docs/perf_baseline.md`](docs/perf_baseline.md)。

▶️ **[观看完整 Demo 视频](docs/demo.mp4)**（2.5 分钟，含实时推理过程）

## 技术栈

| 层级     | 技术                                     |
| -------- | ---------------------------------------- |
| 推理芯片 | RK3588S (6 TOPS NPU, triple-core)        |
| 目标检测 | YOLOv8n, PyTorch → ONNX → RKNN (FP16)    |
| 语言模型 | Qwen2.5-0.5B-Instruct, RKLLM (W8A8 量化) |
| 后处理   | C++ (NMS + decode, 与 Python 交叉验证)   |
| 图像处理 | OpenCV 4.x                               |
| 硬件平台 | Orange Pi 5 Pro 8GB                      |

## 踩坑精选

1. **NCHW vs anchor-major 数据布局陷阱**：NPU 输出 `(1,84,8400)` 是 `[channel][anchor]` 布局，但后处理需要 `[anchor][channel]`。不转置直接用，检出 64 个物体、score 飙到 637——内存排列全错。dump raw binary 逐位对齐才定位，Python 和 C++ 各踩一次。
2. **prompt 约束同时抑制幻觉和提升性能**：不加约束时 LLM 自行编造"玩手机""准备下车"等细节，TTFT 252ms / 17.5 tok/s。加一句"不要添加列表之外的细节"后零幻觉，TTFT 降到 96ms / 41.5 tok/s。端侧小模型管不住自己，用 prompt 替它管。

更多：见 [`docs/pitfalls.md`](docs/pitfalls.md)

## 目录结构

```
rk3588-vision-llm/
├── README.md
├── src/
│   ├── vision_llm_demo.cpp    # 双模型管线主程序
│   ├── yolo_pipeline.cpp       # YOLO 独立管线
│   └── rknn_minimal.c          # RKNN C API 最小示例
└── docs/
    ├── perf_baseline.md        # 性能基线（6 组）
    ├── pitfalls.md             # 踩坑记录
    └── interview_qa.md         # 面试背诵手册
```
