/*
 * RKNN C API 最小示例 — 概念参考
 * =================================
 * 编译（在板子上）:
 *   gcc -O2 rknn_minimal.c -o rknn_minimal -lrknnrt
 * 运行:
 *   ./rknn_minimal model.rknn image_640x640_rgb.bin
 *
 * 核心流程（6 步）:
 *   init → query → inputs_set → run → outputs_get → destroy
 *
 * 对应 Python 的:
 *   model = RKNN_model_container() → model.run() → del model
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "rknn_api.h"

int main(int argc, char **argv)
{
    if (argc < 3) {
        printf("Usage: %s <model.rknn> <input.bin>\n", argv[0]);
        printf("  input.bin: 640x640x3 RGB raw data (uint8)\n");
        return 1;
    }

    char *model_path = argv[1];
    char *input_path = argv[2];

    // ====== 1. 初始化: 加载模型到 NPU ======
    rknn_context ctx = 0;
    int ret = rknn_init(&ctx, (void *)model_path, 0, 0, NULL);
    if (ret < 0) {
        printf("rknn_init failed! ret=%d\n", ret);
        return -1;
    }
    printf("[1/6] rknn_init OK\n");

    // ====== 2. 查询模型输入/输出信息 ======
    rknn_input_output_num io_num;
    ret = rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
    if (ret < 0) {
        printf("rknn_query failed! ret=%d\n", ret);
        return -1;
    }
    printf("[2/6] rknn_query: n_input=%d, n_output=%d\n",
           io_num.n_input, io_num.n_output);

    // ====== 3. 准备输入数据 ======
    // 读二进制文件 (640x640x3 = 1,228,800 bytes)
    const int INPUT_SIZE = 640 * 640 * 3;
    unsigned char *input_data = (unsigned char *)malloc(INPUT_SIZE);
    FILE *fp = fopen(input_path, "rb");
    if (!fp) {
        printf("Cannot open input file: %s\n", input_path);
        return -1;
    }
    fread(input_data, 1, INPUT_SIZE, fp);
    fclose(fp);

    // 设置输入描述
    rknn_input inputs[1];
    memset(inputs, 0, sizeof(inputs));
    inputs[0].index = 0;
    inputs[0].type = RKNN_TENSOR_UINT8;     // 数据类型
    inputs[0].fmt = RKNN_TENSOR_NHWC;       // 布局: HWC
    inputs[0].size = INPUT_SIZE;
    inputs[0].buf = input_data;

    ret = rknn_inputs_set(ctx, io_num.n_input, inputs);
    if (ret < 0) {
        printf("rknn_inputs_set failed! ret=%d\n", ret);
        return -1;
    }
    printf("[3/6] rknn_inputs_set OK (fed %d bytes)\n", INPUT_SIZE);

    // ====== 4. 触发 NPU 推理 ======
    ret = rknn_run(ctx, NULL);
    if (ret < 0) {
        printf("rknn_run failed! ret=%d\n", ret);
        return -1;
    }
    printf("[4/6] rknn_run OK\n");

    // ====== 5. 获取输出 ======
    rknn_output outputs[1];
    memset(outputs, 0, sizeof(outputs));
    outputs[0].want_float = 1;  // 输出转 float32

    ret = rknn_outputs_get(ctx, 1, outputs, NULL);
    if (ret < 0) {
        printf("rknn_outputs_get failed! ret=%d\n", ret);
        return -1;
    }
    printf("[5/6] rknn_outputs_get OK, output size=%d bytes\n",
           outputs[0].size);

    // outputs[0].buf 现在指向 (1,84,8400) float32 数据
    float *output_data = (float *)outputs[0].buf;
    printf("  First 4 bbox channels of anchor 0: ");
    printf("cx=%.2f cy=%.2f w=%.2f h=%.2f\n",
           output_data[0], output_data[1],
           output_data[2], output_data[3]);

    // ====== 6. 释放资源 ======
    rknn_outputs_release(ctx, 1, outputs);  // 释放输出 buffer
    free(input_data);                         // 释放输入 buffer
    rknn_destroy(ctx);                        // 销毁 context
    printf("[6/6] rknn_destroy OK\n");

    return 0;
}
