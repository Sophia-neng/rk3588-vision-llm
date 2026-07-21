# 踩坑记录

## 1. TF 卡烧录：.img.xz 需要先解压

**日期**：2026-07-08

**问题**：下载的 Ubuntu 镜像是 `ubuntu-22.04-orangepi5pro.img.xz`，直接用 balenaEtcher 烧录失败。

**原因**：`.xz` 是压缩格式，balenaEtcher 不支持直接烧录 `.xz` 文件。

**解决**：先用 `xz` 解压得到 `.img` 文件，再用 balenaEtcher 烧录 `.img` 到 TF 卡。

---

## 2. Wi-Fi 配置：热点连接不稳定，需要工具配合

**日期**：2026-07-08

**问题**：Orange Pi 连接手机热点时连接不稳定，且切换 Wi-Fi 时需要显示器+键盘操作。

**原因**：手机热点限制多，连接不稳定；板子没有固定显示器，Wi-Fi 切换成本高。

**解决**：配置多个常用 Wi-Fi 自动切换（`nmcli` 命令），确认 mDNS（`.local` 域名）可正常解析。

**教训**：上开发板之前先把 Wi-Fi 工具准备好，不要把时间耗在"连不上"上。

---

## 3. RKNN 转换默认 FP16，不需要显式指定量化参数

**日期**：2026-07-11

**现象**：RKNN Toolkit2 跑 ONNX→RKNN 转换时没有传任何量化参数，结果自动做了 FP16 量化。

**结论**：RKNN Toolkit2 默认 `do_quantization=False`，即 FP16 量化。

**验证**：`yolov8n.onnx`（FP32）12 MB → `yolov8n.rknn`（FP16）7.9 MB。权重减半 + RKNN 图元数据，完全符合 FP16 特征。

**三种量化模式**：

| 模式 | 精度 | NPU 速度 | 文件大小 | 需要 |
|------|------|------|------|------|
| FP16 | 几乎无损 | 快 | ~ONNX 50% | 零额外工作 |
| INT8 | 微小损失 | 更快 | ~ONNX 25% | 校准数据集（~100张图） |
| 混合精度 | 平衡 | 各层最优 | 介于中间 | 量化配置文件 |

**教训**：Week 1 目标是 FP16 转通，不需要纠结 INT8。INT8 是后续对比实验。

---

## 4. 输出格式双坑：张量数量不同 + 坐标格式不是 ltrb

**日期**：2026-07-11 ~ 07-12

**坑一：张量数量不同**。demo `model.rknn` 推理输出 9 个张量，自己转换的 `yolov8n.rknn` 只输出 1 个 `(1,84,8400)`。原因是 RKNN Toolkit 版本差异——2.2.0 输出 9 个（3 尺度 × 3 张量），2.3.2 合并成 1 个。后处理逻辑完全不同。

**坑二：坐标格式不是 ltrb**。Demo 后处理代码假设坐标是网格偏移量（需要 DFL 解码 + stride 倍乘），但 Ultralytics 导出的 ONNX 内部已做完 DFL + 坐标解码，输出的 4 个通道直接是 **cx, cy, w, h 像素坐标**。之前按 ltrb 网格偏移写后处理导致坐标异常（全图框覆盖）。

**定位方法**：dump 输出的前几列原始值——`cx=449.0 cy=216.2 w=67.2 h=62.5`，值范围 30~450，在 640×640 像素空间内合理。

**教训**：
- 同一份 ONNX 不同版本 Toolkit 转出来的 RKNN 输出格式可能完全不同
- 坐标格式不能凭文档或 demo 代码猜，必须 dump 实际数据验证
- 8400 = 80×80 + 40×40 + 20×20 三尺度，但输出已拼成单张量，无需手动拆

---

---

## 6. C++ 后处理重写：编译通过 ≠ 逻辑正确

**日期**：2026-07-12

**任务**：把 Python 版 `nms()` + `postprocess()` 翻译成 C++。

**踩坑清单**：

| 坑 | 根因 | 编译能查吗 |
|------|------|------|
| `return keep;` 写在 while 循环内 | 大括号对齐错了 | ❌ 反复编译都过，但只跑一轮 |
| `offset = i * NUM_ANCHORS`（应为 `NUM_FEATURES`） | 变量名混淆 | ❌ out-of-bounds 不报 |
| `all_boxes` 声明为空的，直接 `[i]` 访问 | vector 忘写大小 | ❌ 运行时 segfault |
| 出参 `&` 忘了 | 函数签名不熟 | ❌ 调用者拿到空数组 |
| `all_boxes[1]` 写成常量（应为 `[i]`） | 复制粘贴残留 | ❌ 全框相同 |
| `if (w>0 && h>0)` 代替 `max(0,w)` | 逻辑不等价 | ❌ 非重叠框被错杀 |

**核心教训**：
- Python 的布尔索引 `boxes[mask]` → C++ 用 `for + if + push_back`
- `const float*` + `offset = i * 84` 这种布局，偏移量写错编译器救不了
- vector 出参必加 `&`，C++ 初学者最容易无声翻车的地方

---

## 7. C++ 后处理交叉验证：数据布局对齐

**日期**：2026-07-13

**问题**：第一版 C++ 跑出来 64 个检测、score 高达 637——完全错误。

**定位**：NPU 输出 `(1,84,8400)` 是 NCHW 布局（channel-major），内存排列为 `[ch0:anchor0..8399] [ch1:anchor0..8399] ...`。但后处理需要 anchor-major 布局：`[anchor0:ch0..83] [anchor1:ch0..83] ...`。C++ 直接用 NCHW 原始索引去读，数据全对不上。

**解决**：dump 数据时先转置再存 binary。

**结果**：C++ 与 Python 全部 5 个检测框坐标、类别、置信度逐位一致。

**教训**：
- 模型输出内存布局 ≠ numpy 打印的 shape
- 跨语言验证先确认二进制数据布局一致
- `.npy` 有 header，C++ 不能直接读；用 `.tofile()` 存 raw binary

---

## 8. CPU 不支持 float16

**日期**：2026-07-15

**问题**：Colima x86 上执行 RKLLM 转换时输出警告 `WARNING: The cpu device not support float16! switch to float32!`

**原因**：x86 CPU 对 float16 的硬件支持不完整，RKLLM Toolkit 自动切换到 float32 做中间计算。

**影响**：最终 W8A8 RKLLM 文件不受影响，仅中间过程精度不同。看到这个 WARNING 不用慌。

---

## 9. C++ 后处理再现旧 bug：NCHW vs anchor-major 布局

**日期**：2026-07-16

**问题**：YOLO C++ 完整管线首次跑通时，检出 64 个物体、score 高达 637——和条目 7 一模一样的现象。

**原因**：和条目 7 一模一样——NPU 输出是 NCHW 布局，后处理要 anchor-major。

**解决**：在调 `postprocess` 之前加转置循环。

**教训**：同一个 bug 在 Python 和 C++ 两端各踩一次 = 概念没彻底消化。现在彻底懂了。NPU 输出的 `(1,84,8400)` 不是 `[anchor][channel]` 而是 `[channel][anchor]`。

---

---

## 11. 双模型合并：argv 冲突 + 搬家漏件

**日期**：2026-07-17

**任务**：把 `yolo_pipeline.cpp` 和 `llm_demo.cpp` 合并成 `vision_llm_demo.cpp`。

**踩坑清单**：

| 坑 | 现象 | 根因 |
|------|------|------|
| `argv[1]` 被两个模型抢占 | YOLO 和 LLM 的 `model_path` 都读 argv[1] | 两份代码参数表直接拼在一起 |
| 躲冲突改出 argv[4] 空位 | 参数表跳号 | "躲开冲突"≠"设计参数表" |
| `input_str` 未定义 | 抄了 `prompt_input = input_str.c_str()` | 抄代码要"移植"不要"复印" |
| 全局声明没搬 | `t0/t1/t2` 等全报 not declared | 搬了函数，漏了全局声明区 |
| `int ret` 重复定义 | redeclaration | 两份 main 各有一个 `int ret` |
| `ofstream` incomplete type | 缺 `#include <fstream>` | 头文件没搬全 |

**正确做法**：参数表先设计"用户怎么敲这条命令"，再倒推 argv 编号。编译器当免费检查员——合并完直接编译，报谁没声明就补谁。

---

## 12. c_str() 悬空指针：字符串定稿之后才取指针

**日期**：2026-07-17

**问题**：`rkllm_input.prompt_input = prompt.c_str()` 写在了拼接 prompt 的 for 循环之前——赋值时 prompt 还是空的。

**原因**：`c_str()` 返回 string **内部内存**的指针。之后每次 `+=` 都可能触发扩容搬家，旧指针变成悬空指针。

**解决**：`c_str()` 移到 for 循环结束之后、`rkllm_run` 之前。

**教训**：规矩记死——**字符串定稿之后才取 `c_str()`**。取内部指针的操作（`c_str()`、`data()`），后续任何修改都可能让它失效。

---

## 13. "没有报错" ≠ 成功：g++ 命令断行 + 库名拼错

**日期**：2026-07-17

**现象**：板端编译，终端只回了一句 `-O2: command not found`。以为没 error 就是过了，结果 `ls` 发现根本没产物。

**原因**：
- 从 history 复制命令时开头的 `g++ -` 被截掉了，shell 把 `-O2` 当命令名执行
- 还埋着第二个雷：`-lrkllmrk` 拼错（应为 `-lrkllmrt`）

**教训**：
- **验收标准是产物存在**，不是"没看到 error"
- `command not found` 也是报错
- `-L` 给目录、`-l` 给库名，各管各的

---

## 14. LLM 幻觉实录：prompt 不加约束，小模型自己加戏

**日期**：2026-07-17

**现象**：喂给 Qwen 的信息只有"4个person，1个bus"，它输出"其中一人正在使用手机进行社交互动，另外三人则坐在座位上准备下车"——全是编的。

**原因**：LLM 的本质是自回归续写最像人话的下一个词，不是在报告事实。temperature=0.1 也压不住——不是采样问题，是它手里没有事实。

**解决**：prompt = **指令 + 数据 + 约束**。加一句"只根据给出的物体列表描述，不要添加列表之外的细节"，输出变成"一辆公交车上坐着四个人。"零幻觉。

**教训**：
- 端侧小模型（0.5B）管不住自己，用 prompt 替它管
- 约束还顺带提升性能：无约束 TTFT 252ms → 加约束 96ms

---

## 15. INT8 量化 YOLOv8n：class score 全塌，5 种混合精度方案均无效

**日期**：2026-07-20

**问题**：YOLOv8n ONNX → RKNN W8A8 INT8 量化后，板端推理检出 0 objects。所有 class score 通道输出全为 0，仅坐标通道（cx/cy/w/h）值合理。

**定位**：dump FP16 vs INT8 输出逐位对比：
- 坐标通道值接近（cx~247 vs 5，cy~374 vs 18，不同 anchor 而已）
- FP16 最高 person score：0.8838（anchor 8227）
- INT8 最高 person score：0.0000（所有 8400 个 anchor 的 80 类 score 全塌）

**尝试的 5 种混合精度配置**：

| 配置 | 结果 |
|------|------|
| w8a8 全 INT8 | ❌ |
| `auto_hybrid_cos_thresh=0.95` | ❌ |
| `auto_hybrid_cos_thresh=0.80` | ❌ |
| `quantized_hybrid_level=3` | ❌ |
| `quantized_method="channel"` + `optimization_level=0` | ❌ |

全部在板端检出 0 objects。

**根因分析**：YOLO 检测头的 Sigmoid + class score 层对 INT8 量化极度敏感。Sigmoid 输出在 0-1 之间，INT8 只有 -128~127 的整数表达力，细粒度的小数被量化噪声淹没。RKNN Toolkit2 2.3.2 的自动混合精度（auto_hybrid_cos_thresh / quantized_hybrid_level）理论上应该识别这些层并保留 FP16，但实际未生效——可能因为 RK3588 NPU 的硬件限制或 SDK 2.3.2 的 bug。

**教训**：
- 检测模型的量化难点不在 backbone，在检测头。分类置信度比坐标回归敏感得多。
- RKNN Toolkit2 的自动混合精度不能完全信任——对关键模型需要手动验证。
- INT8 量化在 ResNet/MobileNet 分类模型上成熟，但在 YOLO 检测模型上仍需要更精细的手动混合精度或用更大的校准数据集（500-1000 张）。
- **面试加分项**：亲自踩过量化失败的所有坑，比直接跑通的人多一个维度——你知道"不什么不能做"。
