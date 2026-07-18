#include <iostream>
#include <vector>
#include <algorithm>
using namespace std;

std::vector<int> nms(
	const std::vector<std::vector<float>> &boxes,
	const std::vector<float> &scores,
	float iou_thres = 0.45)
{
	int N = boxes.size();
	std::vector<int> keep;
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
			float xx1 = max(x1[i], x1[j]);
			float yy1 = max(y1[i], y1[j]);
			float xx2 = min(x2[i], x2[j]);
			float yy2 = min(y2[i], y2[j]);
			float w = std::max(0.0f, xx2 - xx1);
			float h = std::max(0.0f, yy2 - yy1);
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
	std::vector<float> &scores_out,
	std::vector<int> &classes_out,
	float iou_thres = 0.45,
	float conf_thres = 0.25

)
{

	int NUM_TOTLE = 8400;
	int NUM_FEATURES = 84;
	std::vector<std::vector<float>> all_boxes(NUM_TOTLE, std::vector<float>(4));
	std::vector<float> all_scores(NUM_TOTLE);
	std::vector<int> all_classes(NUM_TOTLE);

	for (int i = 0; i < NUM_TOTLE; i++)
	{
		int offset = i * NUM_FEATURES;
		float cx = output[offset + 0];
		float cy = output[offset + 1];
		float w = output[offset + 2];
		float h = output[offset + 3];

		all_boxes[i][0] = cx - w / 2.0f;
		all_boxes[i][1] = cy - h / 2.0f;
		all_boxes[i][2] = cx + w / 2.0f;
		all_boxes[i][3] = cy + h / 2.0f;
		float max_score = 0.0;
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
	std::vector<std::vector<float>> filtered_boxes;
	std::vector<float> filtered_scores;
	std::vector<int> filtered_classes;
	for (int i = 0; i < NUM_TOTLE; i++)
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
		box[2] = std::
		clamp(box[2], 0.0f, img_w);
		box[3] = std::clamp(box[3], 0.0f, img_h);
	}
	std::vector<int> keep = nms(filtered_boxes, filtered_scores, iou_thres);
	for (auto v : keep)
	{
		boxes_out.push_back(filtered_boxes[v]);
		scores_out.push_back(filtered_scores[v]);
		classes_out.push_back(filtered_classes[v]);
	}
}