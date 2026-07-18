#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <opencv2/opencv.hpp>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "rknn_api.h"
using namespace std;

unsigned char *preprocess(const char *image_path, int &orig_w, int &orig_h)
{
	const int MODEL_W = 640;
	const int MODEL_H = 640;
	const int CHANNELS = 3;

	// 1. 读图（OpenCV 默认 BGR 顺序，UINT8）
	cv::Mat img = cv::imread(image_path);
	if (img.empty())
	{
		printf("Failed to read image: %s\n", image_path);
		return nullptr;
	}

	// 2. 记录原始尺寸（后处理还原坐标用）
	orig_w = img.cols;
	orig_h = img.rows;

	// 3. 算 letterbox 缩放比例
	//    scale = 长边缩到 640，短边等比缩放
	float scale = std::min(
		(float)MODEL_W / orig_w,
		(float)MODEL_H / orig_h);
	int new_w = (int)(orig_w * scale);
	int new_h = (int)(orig_h * scale);

	// 4. Resize：保持宽高比缩放到 new_w × new_h
	cv::Mat resized;
	cv::resize(img, resized, cv::Size(new_w, new_h));

	// 5. 算灰边宽度（整除不干净时，右边/下边多补一个像素）
	int dw = MODEL_W - new_w; // 总共有多少空余宽度
	int dh = MODEL_H - new_h; // 总共有多少空余高度
	int top = dh / 2;		  // 上方灰边
	int bottom = dh - top;	  // 下方灰边（可能比 top 多 1px）
	int left = dw / 2;		  // 左方灰边
	int right = dw - left;	  // 右方灰边

	// 6. 补灰边（114 是 YOLO 训练时的填充色）
	cv::Mat padded;
	cv::copyMakeBorder(resized, padded,
					   top, bottom, left, right,
					   cv::BORDER_CONSTANT,
					   cv::Scalar(114, 114, 114));

	// 现在 padded 是 640×640 BGR

	// 7. BGR → RGB（OpenCV 默认 BGR，YOLO 训练用 RGB）
	cv::Mat rgb;
	cv::cvtColor(padded, rgb, cv::COLOR_BGR2RGB);

	// 8. 分配输出 buffer（调用方负责 free）
	//    大小 = 640 × 640 × 3
	unsigned char *output = (unsigned char *)malloc(MODEL_W * MODEL_H * CHANNELS);
	if (!output)
	{
		printf("malloc failed!\n");
		return nullptr;
	}

	// 9. 拷贝到连续内存（OpenCV 的 Mat 可能每行有 padding 字节，
	//    必须用 memcpy 做逐行复制保证连续性）
	for (int r = 0; r < MODEL_H; r++)
	{
		memcpy(
			output + r * MODEL_W * CHANNELS, // 目的：第 r 行起始地址
			rgb.ptr<unsigned char>(r),		 // 源：  第 r 行起始地址
			MODEL_W * CHANNELS				 // 每行字节数
		);
	}

	return output;
}

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

int main(int argc, char **argv)
{
	if (argc < 3)
	{
		printf("Usage: %s <model.rknn> <image.jpg>\n", argv[0]);
		printf("  image: any JPEG/PNG image\n");
		return 1;
	}

	char *model_path = argv[1];
	char *input_path = argv[2];

	// ====== 1. 初始化: 加载模型到 NPU ======
	rknn_context ctx = 0;
	int ret = rknn_init(&ctx, (void *)model_path, 0, 0, NULL);
	if (ret < 0)
	{
		printf("rknn_init failed! ret=%d\n", ret);
		return -1;
	}
	printf("[1/6] rknn_init OK\n");

	// ====== 2. 查询模型输入/输出信息 ======
	rknn_input_output_num io_num;
	ret = rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
	if (ret < 0)
	{
		printf("rknn_query failed! ret=%d\n", ret);
		return -1;
	}
	printf("[2/6] rknn_query: n_input=%d, n_output=%d\n",
		   io_num.n_input, io_num.n_output);

	// ====== 3. 预处理，读图+letterbox->640*640RGB ======
	const int INPUT_SIZE = 640 * 640 * 3;
	int orig_w, orig_h;
	unsigned char *input_data = preprocess(input_path, orig_w, orig_h);
	if (!input_data)
	{
		printf("preprocess failed!\n");
		return -1;
	}

	// 设置输入描述
	rknn_input inputs[1];
	memset(inputs, 0, sizeof(inputs));
	inputs[0].index = 0;
	inputs[0].type = RKNN_TENSOR_UINT8; // 数据类型
	inputs[0].fmt = RKNN_TENSOR_NHWC;	// 布局: HWC
	inputs[0].size = INPUT_SIZE;
	inputs[0].buf = input_data;

	ret = rknn_inputs_set(ctx, io_num.n_input, inputs);
	if (ret < 0)
	{
		printf("rknn_inputs_set failed! ret=%d\n", ret);
		return -1;
	}
	printf("[3/6] rknn_inputs_set OK (fed %d bytes)\n", INPUT_SIZE);

	// ====== 4. 触发 NPU 推理 ======
	ret = rknn_run(ctx, NULL);
	if (ret < 0)
	{
		printf("rknn_run failed! ret=%d\n", ret);
		return -1;
	}
	printf("[4/6] rknn_run OK\n");

	// ====== 5. 获取输出 ======
	rknn_output outputs[1];
	memset(outputs, 0, sizeof(outputs));
	outputs[0].want_float = 1; // 输出转 float32

	ret = rknn_outputs_get(ctx, 1, outputs, NULL);
	if (ret < 0)
	{
		printf("rknn_outputs_get failed! ret=%d\n", ret);
		return -1;
	}
	printf("[5/6] rknn_outputs_get OK, output size=%d bytes\n",
		   outputs[0].size);

	// outputs[0].buf 现在指向 (1,84,8400) float32 数据
	float *output_data = (float *)outputs[0].buf;
	int NUM_ANCHORS = 8400;
	int NUM_FEATURES = 84;
	float *transposed = (float *)malloc(NUM_ANCHORS * NUM_FEATURES * sizeof(float));
	for (int i = 0; i < NUM_ANCHORS; i++)
	{ // 遍历每个 anchor
		for (int c = 0; c < NUM_FEATURES; c++)
		{ // 遍历 84 个通道
			transposed[i * NUM_FEATURES + c] = output_data[c * NUM_ANCHORS + i];
		}
	}
	vector<std::vector<float>> boxes;
	vector<float> scores;
	vector<int> classes;
	postprocess(transposed, boxes, classes, scores);
	free(transposed);
	printf("Detected %zu objects:\n", boxes.size());
	for (size_t i = 0; i < boxes.size(); i++)
	{
		printf("%s, score: %.2f\n", COCO_CLASSES[classes[i]], scores[i]);
	}

	// ====== 6. 释放资源 ======
	rknn_outputs_release(ctx, 1, outputs); // 释放输出 buffer
	free(input_data);					   // 释放输入 buffer
	rknn_destroy(ctx);					   // 销毁 context
	printf("[6/6] rknn_destroy OK\n");

	return 0;
}
