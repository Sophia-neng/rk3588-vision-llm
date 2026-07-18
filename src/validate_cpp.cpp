#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
using namespace std;

// ========== 从 postprocess.cpp 复制的函数（应保持一致）==========

std::vector<int> nms(
    const std::vector<std::vector<float>> &boxes,
    const std::vector<float> &scores,
    float iou_thres = 0.45)
{
    int N = boxes.size();
    std::vector<float> x1(N), y1(N), x2(N), y2(N);
    for (int i = 0; i < N; i++)
    {
        x1[i] = boxes[i][0];
        y1[i] = boxes[i][1];
        x2[i] = boxes[i][2];
        y2[i] = boxes[i][3];
    }
    std::vector<float> areas(N);
    for (int i = 0; i < N; i++)
    {
        areas[i] = (x2[i] - x1[i]) * (y2[i] - y1[i]);
    }
    std::vector<int> order(N);
    for (int i = 0; i < N; i++)
    {
        order[i] = i;
    }
    std::sort(order.begin(), order.end(), [&scores](int a, int b)
              { return scores[a] > scores[b]; });
    std::vector<int> keep;
    while (order.size() > 0)
    {
        int i = order[0];
        keep.push_back(i);
        if (order.size() == 1)
        {
            break;
        }
        std::vector<int> new_order;
        for (int k = 1; k < order.size(); k++)
        {
            int j = order[k];
            float xx1 = std::max(x1[i], x1[j]);
            float yy1 = std::max(y1[i], y1[j]);
            float xx2 = std::min(x2[i], x2[j]);
            float yy2 = std::min(y2[i], y2[j]);
            float w = max(0.0f, xx2 - xx1);
            float h = max(0.0f, yy2 - yy1);
            float inter = w * h;
            float iou_val = inter / (areas[i] + areas[j] - inter);
            if (iou_val <= iou_thres)
            {
                new_order.push_back(j);
            }
        }
        order = new_order;
    }
    return keep;
}

void postprocess(
    const float *output,
    std::vector<std::vector<float>> &boxes_out,
    std::vector<int> &class_out,
    std::vector<float> &score_out,
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
    float img_w = 640.0;
    float img_h = 640.0;
    for (auto &box : filtered_boxes)
    {
        box[0] = std::clamp(box[0], 0.0f, img_w);
        box[1] = std::clamp(box[1], 0.0f, img_h);
        box[2] = std::clamp(box[2], 0.0f, img_w);
        box[3] = std::clamp(box[3], 0.0f, img_h);
    }
    vector<int> keep = nms(filtered_boxes, filtered_scores, iou_thres);

    for (auto v : keep)
    {
        boxes_out.push_back(filtered_boxes[v]);
        class_out.push_back(filtered_classes[v]);
        score_out.push_back(filtered_scores[v]);
    }
}

// ========== 验证主程序 ==========

int main()
{
    // 1. 读取 NPU 原始输出
    const int NUM_FLOATS = 1 * 84 * 8400;
    vector<float> data(NUM_FLOATS);

    ifstream fin("src/npu_output_bus.bin", ios::binary);
    if (!fin)
    {
        cerr << "Error: cannot open src/npu_output_bus.bin" << endl;
        return 1;
    }
    fin.read(reinterpret_cast<char *>(data.data()), NUM_FLOATS * sizeof(float));
    fin.close();
    cout << "Loaded " << NUM_FLOATS << " floats from npu_output_bus.bin" << endl;

    // 2. 跑后处理
    vector<vector<float>> boxes;
    vector<int> classes;
    vector<float> scores;
    postprocess(data.data(), boxes, classes, scores);

    // 3. 输出
    cout << "Detections: " << boxes.size() << endl;
    for (size_t i = 0; i < boxes.size(); i++)
    {
        printf("  class=%d box=[%.1f %.1f %.1f %.1f] score=%.4f\n",
               classes[i],
               boxes[i][0], boxes[i][1], boxes[i][2], boxes[i][3],
               scores[i]);
    }

    // 4. 与 Python 参考结果对比
    cout << "\n========== 对比 Python 参考 ==========" << endl;
    cout << "Python: 5 detections" << endl;
    cout << "  class=0 box=[210.9 241.5 283.6 507.5] score=0.8838  (person)" << endl;
    cout << "  class=0 box=[109.6 235.5 224.9 536.5] score=0.8794  (person)" << endl;
    cout << "  class=0 box=[477.4 223.5 560.6 522.0] score=0.8735  (person)" << endl;
    cout << "  class=5 box=[99.4 135.8 550.6 456.8] score=0.8496  (bus)" << endl;
    cout << "  class=0 box=[80.2 327.0 116.3 514.0] score=0.3298  (person)" << endl;

    return 0;
}
