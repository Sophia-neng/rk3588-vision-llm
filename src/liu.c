rknn_context ctx = 0;
int ret = rknn_init(&ctx, "yolov8n.onnx", 0, 0, NULL);
rknn_input_output_num io_num;
ret = rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &io.num, sizeof(io_num));
rknn_input inputs[1];
memset(inputs, 0, sizeof(inputs));
inputs[0].index = 0;
inputs[0].type = RKNN_TENSOR_UNIT8;
inputs[0].size = 640 * 640 * 3;
inputs[0].fmt = RKNN_TENSOR_NCHW;
inputs[0].buf = img;
ret = rknn_inputs_set(ctx, 1, inputs);
rknn_run(ctx, NULL);
rknn_output outputs[1];
memset(outputs, 0, sizeof(outputs));
outputs[0].want_float = 1;
ret = rknn_ouutputs_get(ctx, 1, outputs, NULL)
	rknn_outputs_release(ctx, 1, outputs);
rknn_destroy(ctx);