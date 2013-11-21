void do_loop_filter()
{
	if (video.GOP_size < 2) return;

	int32_t stage, mb_size, plane_width;
	for (stage = 0; stage < (video.mb_width + (video.mb_height-1)*2); ++stage) 
	{
		device.state_gpu = clFlush(device.commandQueue_gpu);
		device.state_gpu = clSetKernelArg(device.normal_loop_filter_MBH, 5, sizeof(int32_t), &stage);
		device.state_gpu = clSetKernelArg(device.normal_loop_filter_MBV, 5, sizeof(int32_t), &stage);
		device.gpu_work_items_per_dim[0] = video.mb_height*16;

		mb_size = 16;
		plane_width = video.wrk_width;
		device.state_gpu = clSetKernelArg(device.normal_loop_filter_MBH, 4, sizeof(int32_t), &mb_size);
		device.state_gpu = clSetKernelArg(device.normal_loop_filter_MBH, 1, sizeof(int32_t), &plane_width);
		device.state_gpu = clSetKernelArg(device.normal_loop_filter_MBH, 0, sizeof(cl_mem), &device.reconstructed_frame_Y);
		device.state_gpu = clEnqueueNDRangeKernel(device.commandQueue_gpu, device.normal_loop_filter_MBH, 1, NULL, device.gpu_work_items_per_dim, NULL, 0, NULL, NULL);
		if (device.state_gpu != 0)  printf(">error while deblocking : %d", device.state_gpu);
		device.state_gpu = clFlush(device.commandQueue_gpu);
		if (device.state_gpu != 0)  printf(">error while deblocking : %d", device.state_gpu);
		device.gpu_work_items_per_dim[0] = video.mb_height*8;
		mb_size = 8;
		plane_width = video.wrk_width/2;
		device.state_gpu = clSetKernelArg(device.normal_loop_filter_MBH, 4, sizeof(int32_t), &mb_size);
		device.state_gpu = clSetKernelArg(device.normal_loop_filter_MBH, 1, sizeof(int32_t), &plane_width);
		device.state_gpu = clSetKernelArg(device.normal_loop_filter_MBH, 0, sizeof(cl_mem), &device.reconstructed_frame_U);
		device.state_gpu = clEnqueueNDRangeKernel(device.commandQueue_gpu, device.normal_loop_filter_MBH, 1, NULL, device.gpu_work_items_per_dim, NULL, 0, NULL, NULL);
		if (device.state_gpu != 0)  printf(">error while deblocking : %d", device.state_gpu);
		device.state_gpu = clFlush(device.commandQueue_gpu);
		if (device.state_gpu != 0)  printf(">error while deblocking : %d", device.state_gpu);
		device.state_gpu = clSetKernelArg(device.normal_loop_filter_MBH, 0, sizeof(cl_mem), &device.reconstructed_frame_V);
		device.state_gpu = clEnqueueNDRangeKernel(device.commandQueue_gpu, device.normal_loop_filter_MBH, 1, NULL, device.gpu_work_items_per_dim, NULL, 0, NULL, NULL);
		if (device.state_gpu != 0)  printf(">error while deblocking : %d", device.state_gpu);
		device.state_gpu = clFinish(device.commandQueue_gpu);
		if (device.state_gpu != 0)  printf(">error while deblocking : %d", device.state_gpu);

		device.gpu_work_items_per_dim[0] = video.mb_height*4;
		mb_size = 16;
		plane_width = video.wrk_width;
		device.state_gpu = clSetKernelArg(device.normal_loop_filter_MBV, 4, sizeof(int32_t), &mb_size);
		device.state_gpu = clSetKernelArg(device.normal_loop_filter_MBV, 1, sizeof(int32_t), &plane_width);
		device.state_gpu = clSetKernelArg(device.normal_loop_filter_MBV, 0, sizeof(cl_mem), &device.reconstructed_frame_Y);
		device.state_gpu = clEnqueueNDRangeKernel(device.commandQueue_gpu, device.normal_loop_filter_MBV, 1, NULL, device.gpu_work_items_per_dim, NULL, 0, NULL, NULL);
		if (device.state_gpu != 0)  printf(">error while deblocking : %d", device.state_gpu);
		device.state_gpu = clFlush(device.commandQueue_gpu);
		if (device.state_gpu != 0)  printf(">error while deblocking : %d", device.state_gpu);
		device.gpu_work_items_per_dim[0] = video.mb_height*2;
		mb_size = 8;
		plane_width = video.wrk_width/2;
		device.state_gpu = clSetKernelArg(device.normal_loop_filter_MBV, 4, sizeof(int32_t), &mb_size);
		device.state_gpu = clSetKernelArg(device.normal_loop_filter_MBV, 1, sizeof(int32_t), &plane_width);
		device.state_gpu = clSetKernelArg(device.normal_loop_filter_MBV, 0, sizeof(cl_mem), &device.reconstructed_frame_U);
		device.state_gpu = clEnqueueNDRangeKernel(device.commandQueue_gpu, device.normal_loop_filter_MBV, 1, NULL, device.gpu_work_items_per_dim, NULL, 0, NULL, NULL);
		if (device.state_gpu != 0)  printf(">error while deblocking : %d", device.state_gpu);
		device.state_gpu = clFlush(device.commandQueue_gpu);
		if (device.state_gpu != 0)  printf(">error while deblocking : %d", device.state_gpu);
		device.state_gpu = clSetKernelArg(device.normal_loop_filter_MBV, 0, sizeof(cl_mem), &device.reconstructed_frame_V);
		device.state_gpu = clEnqueueNDRangeKernel(device.commandQueue_gpu, device.normal_loop_filter_MBV, 1, NULL, device.gpu_work_items_per_dim, NULL, 0, NULL, NULL);
		if (device.state_gpu != 0)  printf(">error while deblocking : %d", device.state_gpu);
		device.state_gpu = clFinish(device.commandQueue_gpu);
		if (device.state_gpu != 0)  printf(">error while deblocking : %d", device.state_gpu);
	}
	
	return;
}
