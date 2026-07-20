#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <cstring>
#include <cstdlib>
#include <csignal>
#include <chrono>
#include <fstream>
#include <opencv2/opencv.hpp>
#include "rknn_api.h"
#include "rkllm.h"

using namespace std;

// ====== 全局状态（callback 需要访问） ======
LLMHandle llmHandle = nullptr;
int token_count = 0;
auto t0 = chrono::steady_clock::now();
auto t1 = t0;
auto t2 = t0;

// ====== COCO 类别名 ======
const char *COCO_CLASSES[] = {
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
    "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop",
    "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
    "toothbrush"};

// ====== 预处理：读图 → letterbox → 640×640 RGB ======
unsigned char *preprocess(const char *image_path, int &orig_w, int &orig_h)
{
    const int MODEL_W = 640;
    const int MODEL_H = 640;
    const int CHANNELS = 3;

    cv::Mat img = cv::imread(image_path);
    if (img.empty())
    {
        cerr << "Error: failed to read image " << image_path << endl;
        return nullptr;
    }

    orig_w = img.cols;
    orig_h = img.rows;

    // letterbox：长边缩到 640，短边等比缩放
    float scale = min((float)MODEL_W / orig_w, (float)MODEL_H / orig_h);
    int new_w = (int)(orig_w * scale);
    int new_h = (int)(orig_h * scale);

    cv::Mat resized;
    cv::resize(img, resized, cv::Size(new_w, new_h));

    // 补灰边（114 = YOLO 训练填充色）
    int dw = MODEL_W - new_w;
    int dh = MODEL_H - new_h;
    int top = dh / 2;
    int bottom = dh - top;
    int left = dw / 2;
    int right = dw - left;

    cv::Mat padded;
    cv::copyMakeBorder(resized, padded,
                       top, bottom, left, right,
                       cv::BORDER_CONSTANT,
                       cv::Scalar(114, 114, 114));

    // BGR → RGB
    cv::Mat rgb;
    cv::cvtColor(padded, rgb, cv::COLOR_BGR2RGB);

    unsigned char *output = (unsigned char *)malloc(MODEL_W * MODEL_H * CHANNELS);
    if (!output)
    {
        cerr << "Error: malloc failed in preprocess" << endl;
        return nullptr;
    }

    // 逐行拷贝：OpenCV Mat 行间可能有 padding，必须保证连续内存
    for (int r = 0; r < MODEL_H; r++)
    {
        memcpy(output + r * MODEL_W * CHANNELS,
               rgb.ptr<unsigned char>(r),
               MODEL_W * CHANNELS);
    }

    return output;
}

// ====== NMS（非极大值抑制） ======
vector<int> nms(const vector<vector<float>> &boxes,
                const vector<float> &scores,
                float iou_thres = 0.45)
{
    int N = boxes.size();
    vector<float> x1(N), y1(N), x2(N), y2(N), areas(N);
    for (int i = 0; i < N; i++)
    {
        x1[i] = boxes[i][0];
        y1[i] = boxes[i][1];
        x2[i] = boxes[i][2];
        y2[i] = boxes[i][3];
        areas[i] = (x2[i] - x1[i]) * (y2[i] - y1[i]);
    }

    vector<int> order(N);
    for (int i = 0; i < N; i++)
        order[i] = i;
    sort(order.begin(), order.end(), [&scores](int a, int b)
         { return scores[a] > scores[b]; });

    vector<int> keep;
    while (order.size() > 0)
    {
        int i = order[0];
        keep.push_back(i);
        if (order.size() == 1)
            break;

        vector<int> new_order;
        for (size_t k = 1; k < order.size(); k++)
        {
            int j = order[k];
            float xx1 = max(x1[i], x1[j]);
            float yy1 = max(y1[i], y1[j]);
            float xx2 = min(x2[i], x2[j]);
            float yy2 = min(y2[i], y2[j]);
            float w = max(0.0f, xx2 - xx1);
            float h = max(0.0f, yy2 - yy1);
            float iou_val = (w * h) / (areas[i] + areas[j] - w * h);
            if (iou_val <= iou_thres)
                new_order.push_back(j);
        }
        order = new_order;
    }
    return keep;
}

// ====== YOLO 后处理：decode + 置信度过滤 + NMS ======
void postprocess(const float *output,
                 vector<vector<float>> &boxes_out,
                 vector<int> &class_out,
                 vector<float> &score_out,
                 float conf_thres = 0.25,
                 float iou_thres = 0.45)
{
    const int NUM_ANCHORS = 8400;
    const int NUM_FEATURES = 84;

    vector<vector<float>> all_boxes(NUM_ANCHORS, vector<float>(4));
    vector<float> all_scores(NUM_ANCHORS);
    vector<int> all_classes(NUM_ANCHORS);

    for (int i = 0; i < NUM_ANCHORS; i++)
    {
        int offset = i * NUM_FEATURES;
        float cx = output[offset + 0];
        float cy = output[offset + 1];
        float bw = output[offset + 2];
        float bh = output[offset + 3];

        all_boxes[i][0] = cx - bw / 2.0f;
        all_boxes[i][1] = cy - bh / 2.0f;
        all_boxes[i][2] = cx + bw / 2.0f;
        all_boxes[i][3] = cy + bh / 2.0f;

        // 取 80 类中 score 最高的
        float max_score = 0.0f;
        int best_class = 0;
        for (int c = 0; c < 80; c++)
        {
            float score = output[offset + 4 + c];
            if (score > max_score)
            {
                max_score = score;
                best_class = c;
            }
        }
        all_scores[i] = max_score;
        all_classes[i] = best_class;
    }

    // 置信度过滤
    vector<vector<float>> filtered_boxes;
    vector<float> filtered_scores;
    vector<int> filtered_classes;
    for (int i = 0; i < NUM_ANCHORS; i++)
    {
        if (all_scores[i] > conf_thres)
        {
            filtered_boxes.push_back(all_boxes[i]);
            filtered_scores.push_back(all_scores[i]);
            filtered_classes.push_back(all_classes[i]);
        }
    }

    // clamp 到图像边界
    for (auto &box : filtered_boxes)
    {
        box[0] = clamp(box[0], 0.0f, 640.0f);
        box[1] = clamp(box[1], 0.0f, 640.0f);
        box[2] = clamp(box[2], 0.0f, 640.0f);
        box[3] = clamp(box[3], 0.0f, 640.0f);
    }

    // NMS
    vector<int> keep = nms(filtered_boxes, filtered_scores, iou_thres);
    for (auto v : keep)
    {
        boxes_out.push_back(filtered_boxes[v]);
        class_out.push_back(filtered_classes[v]);
        score_out.push_back(filtered_scores[v]);
    }
}

// ====== LLM 流式输出回调 ======
int callback(RKLLMResult *result, void *userdata, LLMCallState state)
{
    if (state == RKLLM_RUN_FINISH)
    {
        t2 = chrono::steady_clock::now();
        cout << endl;
    }
    else if (state == RKLLM_RUN_ERROR)
    {
        cerr << "Error: LLM run error" << endl;
    }
    else if (state == RKLLM_RUN_NORMAL)
    {
        token_count++;
        if (token_count == 1)
            t1 = chrono::steady_clock::now();

        cout << result->text << flush;
    }
    return 0;
}

void exit_handler(int signal)
{
    if (llmHandle != nullptr)
    {
        cout << "Exiting..." << endl;
        LLMHandle _tmp = llmHandle;
        llmHandle = nullptr;
        rkllm_destroy(_tmp);
    }
    exit(signal);
}

// ====== main ======
int main(int argc, char **argv)
{
    // ----- 参数解析 -----
    const char *llm_path = nullptr;
    const char *model_path = nullptr;
    const char *image_path = nullptr;
    int max_tokens = 128;
    int max_context = 2048;
    float conf_thres = 0.25;
    float iou_thres = 0.45;

    for (int i = 1; i < argc; i++)
    {
        if (argv[i][0] == '-' && i + 1 >= argc)
        {
            cerr << "Error: " << argv[i] << " requires an argument" << endl;
            return 1;
        }
        if (strcmp(argv[i], "--llm") == 0)    { llm_path   = argv[i + 1]; i++; }
        if (strcmp(argv[i], "--model") == 0)  { model_path = argv[i + 1]; i++; }
        if (strcmp(argv[i], "--image") == 0)  { image_path = argv[i + 1]; i++; }
        if (strcmp(argv[i], "--tokens") == 0) { max_tokens = atoi(argv[i + 1]); i++; }
        if (strcmp(argv[i], "--context") == 0){ max_context = atoi(argv[i + 1]); i++; }
        if (strcmp(argv[i], "--conf") == 0)   { conf_thres = atof(argv[i + 1]); i++; }
        if (strcmp(argv[i], "--iou") == 0)    { iou_thres  = atof(argv[i + 1]); i++; }
    }

    if (!llm_path || !model_path || !image_path)
    {
        cerr << "Usage: " << argv[0]
             << " --llm <path> --model <path> --image <path>"
             << " [--tokens 128] [--context 2048] [--conf 0.25] [--iou 0.45]"
             << endl;
        return 1;
    }

    // ====== 1. 加载 YOLO 模型 ======
    rknn_context ctx = 0;
    int ret = rknn_init(&ctx, (void *)model_path, 0, 0, NULL);
    if (ret < 0)
    {
        cerr << "Error: rknn_init failed, ret=" << ret << endl;
        return 1;
    }

    // ====== 2. 加载 LLM 模型 ======
    signal(SIGINT, exit_handler);

    RKLLMParam param = rkllm_createDefaultParam();
    param.model_path = llm_path;
    param.top_k = 1;
    param.top_p = 0.95;
    param.temperature = 0.1;
    param.repeat_penalty = 1.1;
    param.frequency_penalty = 0.0;
    param.presence_penalty = 0.0;
    param.max_new_tokens = max_tokens;
    param.max_context_len = max_context;
    param.skip_special_token = true;
    param.extend_param.base_domain_id = 0;
    param.extend_param.embed_flash = 1;

    RKLLMCallback rkllm_callback = {};
    rkllm_callback.result_callback = callback;
    ret = rkllm_init(&llmHandle, &param, &rkllm_callback);
    if (ret != 0)
    {
        cerr << "Error: rkllm_init failed, ret=" << ret << endl;
        rknn_destroy(ctx);
        return 1;
    }

    // ====== 3. 查询 YOLO 输入输出信息 ======
    rknn_input_output_num io_num;
    ret = rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
    if (ret < 0)
    {
        cerr << "Error: rknn_query failed, ret=" << ret << endl;
        rkllm_destroy(llmHandle);
        rknn_destroy(ctx);
        return 1;
    }
    cout << "[YOLO] n_input=" << io_num.n_input
         << ", n_output=" << io_num.n_output << endl;

    // ====== 4. 预处理 ======
    const int INPUT_SIZE = 640 * 640 * 3;
    int orig_w, orig_h;
    unsigned char *input_data = preprocess(image_path, orig_w, orig_h);
    if (!input_data)
    {
        cerr << "Error: preprocess failed" << endl;
        rkllm_destroy(llmHandle);
        rknn_destroy(ctx);
        return 1;
    }

    // ====== 5. 设置输入 + YOLO 推理 ======
    rknn_input inputs[1];
    memset(inputs, 0, sizeof(inputs));
    inputs[0].index = 0;
    inputs[0].type = RKNN_TENSOR_UINT8;
    inputs[0].fmt = RKNN_TENSOR_NHWC;
    inputs[0].size = INPUT_SIZE;
    inputs[0].buf = input_data;

    ret = rknn_inputs_set(ctx, io_num.n_input, inputs);
    if (ret < 0)
    {
        cerr << "Error: rknn_inputs_set failed, ret=" << ret << endl;
        free(input_data);
        rkllm_destroy(llmHandle);
        rknn_destroy(ctx);
        return 1;
    }

    auto yolo_start = chrono::high_resolution_clock::now();
    ret = rknn_run(ctx, NULL);
    auto yolo_end = chrono::high_resolution_clock::now();
    if (ret < 0)
    {
        cerr << "Error: rknn_run failed, ret=" << ret << endl;
        free(input_data);
        rkllm_destroy(llmHandle);
        rknn_destroy(ctx);
        return 1;
    }

    // ====== 6. 获取输出 + 后处理 ======
    rknn_output outputs[1];
    memset(outputs, 0, sizeof(outputs));
    outputs[0].want_float = 1;

    ret = rknn_outputs_get(ctx, 1, outputs, NULL);
    if (ret < 0)
    {
        cerr << "Error: rknn_outputs_get failed, ret=" << ret << endl;
        free(input_data);
        rkllm_destroy(llmHandle);
        rknn_destroy(ctx);
        return 1;
    }

    float *output_data = (float *)outputs[0].buf;
    const int NUM_ANCHORS = 8400;
    const int NUM_FEATURES = 84;

    // 数据布局转换：NCHW (84×8400) → anchor-major (8400×84)
    float *transposed = (float *)malloc(NUM_ANCHORS * NUM_FEATURES * sizeof(float));
    if (!transposed)
    {
        cerr << "Error: malloc failed for transposed buffer" << endl;
        rknn_outputs_release(ctx, 1, outputs);
        free(input_data);
        rkllm_destroy(llmHandle);
        rknn_destroy(ctx);
        return 1;
    }

    for (int i = 0; i < NUM_ANCHORS; i++)
        for (int c = 0; c < NUM_FEATURES; c++)
            transposed[i * NUM_FEATURES + c] = output_data[c * NUM_ANCHORS + i];

    vector<vector<float>> boxes;
    vector<float> scores;
    vector<int> classes;
    postprocess(transposed, boxes, classes, scores, conf_thres, iou_thres);
    free(transposed);

    cout << "[YOLO] detected " << boxes.size() << " objects" << endl;
    for (size_t i = 0; i < boxes.size(); i++)
        cout << "  " << COCO_CLASSES[classes[i]] << ", score=" << scores[i] << endl;

    // ====== 7. 构建 prompt + LLM 推理 ======
    int class_count[80] = {0};
    for (size_t i = 0; i < boxes.size(); i++)
        class_count[classes[i]]++;

    string prompt = "请用一句话描述这个场景,只根据给出的物体列表描述，不要添加列表之外的细节。图片中检测到：";
    for (int j = 0; j < 80; j++)
    {
        if (class_count[j] > 0)
        {
            prompt += to_string(class_count[j]) + "个" + COCO_CLASSES[j] + ",";
        }
    }

    RKLLMInput rkllm_input;
    memset(&rkllm_input, 0, sizeof(RKLLMInput));
    rkllm_input.input_type = RKLLM_INPUT_PROMPT;
    rkllm_input.prompt_input = (char *)prompt.c_str();

    RKLLMInferParam rkllm_infer_params;
    memset(&rkllm_infer_params, 0, sizeof(RKLLMInferParam));

    t0 = chrono::steady_clock::now();
    token_count = 0;

    auto llm_start = chrono::high_resolution_clock::now();
    rkllm_run(llmHandle, &rkllm_input, &rkllm_infer_params, NULL);
    auto llm_end = chrono::high_resolution_clock::now();

    // ====== 8. 性能统计 ======
    auto yolo_ms = chrono::duration_cast<chrono::milliseconds>(yolo_end - yolo_start).count();
    auto llm_ms = chrono::duration_cast<chrono::milliseconds>(llm_end - llm_start).count();
    auto ttft = chrono::duration_cast<chrono::milliseconds>(t1 - t0).count();
    auto decode = chrono::duration_cast<chrono::milliseconds>(t2 - t1).count();
    double tokens_per_s = token_count / (decode / 1000.0);

    cout << endl
         << "===== Performance =====" << endl
         << "YOLO:     " << yolo_ms << " ms" << endl
         << "LLM:      " << llm_ms << " ms" << endl
         << "TTFT:     " << ttft / 1000.0 << " s" << endl
         << "Decode:   " << decode / 1000.0 << " s (" << tokens_per_s << " tok/s)" << endl;

    // ====== 9. 释放资源 ======
    rknn_outputs_release(ctx, 1, outputs);
    free(input_data);
    rknn_destroy(ctx);
    rkllm_destroy(llmHandle);

    return 0;
}
