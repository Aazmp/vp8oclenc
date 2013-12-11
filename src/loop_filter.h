static void prepare_on_gpu()
{
	int mb_num;
	frames.skip_prob = 0;

	// we need to grab transformed data from host memory
	device.state_gpu = clEnqueueWriteBuffer(device.commandQueue1_gpu, device.transformed_blocks_gpu, CL_TRUE, 0, video.mb_count*sizeof(macroblock), frames.transformed_blocks, 0, NULL, NULL);
	device.gpu_work_items_per_dim[0] = video.mb_count;
	device.state_gpu = clEnqueueNDRangeKernel(device.commandQueue1_gpu, device.prepare_filter_mask, 1, NULL, device.gpu_work_items_per_dim, NULL, 0, NULL, NULL);
	// need to return info about non_zero coeffs
	device.state_gpu = clEnqueueReadBuffer(device.commandQueue1_gpu, device.transformed_blocks_gpu ,CL_TRUE, 0, video.mb_count*sizeof(macroblock), frames.transformed_blocks, 0, NULL, NULL);
	
	for(mb_num = 0; mb_num < video.mb_count; ++mb_num) 
		if (frames.transformed_blocks[mb_num].non_zero_coeffs > 0) 
			++frames.skip_prob;

	frames.skip_prob *= 256;
	frames.skip_prob /= video.mb_count;
	frames.skip_prob = (frames.skip_prob > 254) ? 254 : frames.skip_prob;
	frames.skip_prob = (frames.skip_prob < 2) ? 2 : frames.skip_prob;
	//don't do this => frames.skip_prob = 255 - frames.skip_prob; incorrect desription of prob_skip_false
	return;
}

static void prepare_on_cpu()
{
	int mb_num;
	frames.skip_prob = 0;

	// we need to grab transformed data from host memory
	device.state_cpu = clEnqueueWriteBuffer(device.loopfilterY_commandQueue_cpu, device.transformed_blocks_cpu, CL_TRUE, 0, video.mb_count*sizeof(macroblock), frames.transformed_blocks, 0, NULL, NULL);
	device.gpu_work_items_per_dim[0] = 4;
	device.gpu_work_group_size_per_dim[0] = 1;
	device.state_cpu = clEnqueueNDRangeKernel(device.loopfilterY_commandQueue_cpu, device.prepare_filter_mask, 1, NULL, device.gpu_work_items_per_dim, device.gpu_work_group_size_per_dim, 0, NULL, NULL);
	device.state_cpu = clFinish(device.loopfilterY_commandQueue_cpu);
	// need to return info about non_zero coeffs
	device.state_cpu = clEnqueueReadBuffer(device.boolcoder_commandQueue_cpu, device.transformed_blocks_cpu ,CL_TRUE, 0, video.mb_count*sizeof(macroblock), frames.transformed_blocks, 0, NULL, NULL);

	for(mb_num = 0; mb_num < video.mb_count; ++mb_num) 
		if (frames.transformed_blocks[mb_num].non_zero_coeffs > 0) 
			++frames.skip_prob;

	frames.skip_prob *= 256;
	frames.skip_prob /= video.mb_count;
	frames.skip_prob = (frames.skip_prob > 254) ? 254 : frames.skip_prob;
	frames.skip_prob = (frames.skip_prob < 2) ? 2 : frames.skip_prob;
	return;
}

static void prepare_filter_mask_and_non_zero_coeffs()
{
	if (video.do_loop_filter_on_gpu) prepare_on_gpu();
	else prepare_on_cpu();
	return;
}

static void do_loop_filter_on_gpu()
{
	if (video.GOP_size < 2) return;

	if (frames.replaced > 0)
	{
		device.state_gpu = clEnqueueWriteBuffer(device.commandQueue1_gpu, device.reconstructed_frame_Y, CL_TRUE, 0, video.wrk_frame_size_luma, frames.reconstructed_Y, 0, NULL, NULL);
		device.state_gpu = clEnqueueWriteBuffer(device.commandQueue2_gpu, device.reconstructed_frame_U, CL_TRUE, 0, video.wrk_frame_size_chroma, frames.reconstructed_U, 0, NULL, NULL);
		device.state_gpu = clEnqueueWriteBuffer(device.commandQueue3_gpu, device.reconstructed_frame_V, CL_TRUE, 0, video.wrk_frame_size_chroma, frames.reconstructed_V, 0, NULL, NULL);
	}

	cl_int stage, mb_size, plane_width;
	for (stage = 0; stage < (video.mb_width + (video.mb_height-1)*2); ++stage) 
	{
		device.state_gpu = clSetKernelArg(device.normal_loop_filter_MBH, 5, sizeof(cl_int), &stage);
		device.state_gpu = clSetKernelArg(device.normal_loop_filter_MBV, 5, sizeof(cl_int), &stage);
		device.gpu_work_items_per_dim[0] = video.mb_height*16;

		mb_size = 16;
		plane_width = video.wrk_width;
		device.state_gpu = clSetKernelArg(device.normal_loop_filter_MBH, 4, sizeof(cl_int), &mb_size);
		device.state_gpu = clSetKernelArg(device.normal_loop_filter_MBH, 1, sizeof(cl_int), &plane_width);
		device.state_gpu = clSetKernelArg(device.normal_loop_filter_MBH, 0, sizeof(cl_mem), &device.reconstructed_frame_Y);
		device.state_gpu = clEnqueueNDRangeKernel(device.commandQueue1_gpu, device.normal_loop_filter_MBH, 1, NULL, device.gpu_work_items_per_dim, NULL, 0, NULL, NULL);
		if (device.state_gpu != 0)  printf(">error while deblocking : %d", device.state_gpu);
		device.state_gpu = clFlush(device.commandQueue1_gpu);
		if (device.state_gpu != 0)  printf(">error while deblocking : %d", device.state_gpu);
		device.gpu_work_items_per_dim[0] = video.mb_height*8;
		mb_size = 8;
		plane_width = video.wrk_width/2;
		device.state_gpu = clSetKernelArg(device.normal_loop_filter_MBH, 4, sizeof(cl_int), &mb_size);
		device.state_gpu = clSetKernelArg(device.normal_loop_filter_MBH, 1, sizeof(cl_int), &plane_width);
		device.state_gpu = clSetKernelArg(device.normal_loop_filter_MBH, 0, sizeof(cl_mem), &device.reconstructed_frame_U);
		device.state_gpu = clEnqueueNDRangeKernel(device.commandQueue2_gpu, device.normal_loop_filter_MBH, 1, NULL, device.gpu_work_items_per_dim, NULL, 0, NULL, NULL);
		if (device.state_gpu != 0)  printf(">error while deblocking : %d", device.state_gpu);
		device.state_gpu = clFlush(device.commandQueue2_gpu);
		if (device.state_gpu != 0)  printf(">error while deblocking : %d", device.state_gpu);
		device.state_gpu = clSetKernelArg(device.normal_loop_filter_MBH, 0, sizeof(cl_mem), &device.reconstructed_frame_V);
		device.state_gpu = clEnqueueNDRangeKernel(device.commandQueue3_gpu, device.normal_loop_filter_MBH, 1, NULL, device.gpu_work_items_per_dim, NULL, 0, NULL, NULL);
		if (device.state_gpu != 0)  printf(">error while deblocking : %d", device.state_gpu);
		device.state_gpu = clFlush(device.commandQueue3_gpu);
		if (device.state_gpu != 0)  printf(">error while deblocking : %d", device.state_gpu);

		device.gpu_work_items_per_dim[0] = video.mb_height*4;
		mb_size = 16;
		plane_width = video.wrk_width;
		device.state_gpu = clSetKernelArg(device.normal_loop_filter_MBV, 4, sizeof(cl_int), &mb_size);
		device.state_gpu = clSetKernelArg(device.normal_loop_filter_MBV, 1, sizeof(cl_int), &plane_width);
		device.state_gpu = clSetKernelArg(device.normal_loop_filter_MBV, 0, sizeof(cl_mem), &device.reconstructed_frame_Y);
		device.state_gpu = clEnqueueNDRangeKernel(device.commandQueue1_gpu, device.normal_loop_filter_MBV, 1, NULL, device.gpu_work_items_per_dim, NULL, 0, NULL, NULL);
		if (device.state_gpu != 0)  printf(">error while deblocking : %d", device.state_gpu);
		device.state_gpu = clFlush(device.commandQueue1_gpu);
		if (device.state_gpu != 0)  printf(">error while deblocking : %d", device.state_gpu);
		device.gpu_work_items_per_dim[0] = video.mb_height*2;
		mb_size = 8;
		plane_width = video.wrk_width/2;
		device.state_gpu = clSetKernelArg(device.normal_loop_filter_MBV, 4, sizeof(cl_int), &mb_size);
		device.state_gpu = clSetKernelArg(device.normal_loop_filter_MBV, 1, sizeof(cl_int), &plane_width);
		device.state_gpu = clSetKernelArg(device.normal_loop_filter_MBV, 0, sizeof(cl_mem), &device.reconstructed_frame_U);
		device.state_gpu = clEnqueueNDRangeKernel(device.commandQueue2_gpu, device.normal_loop_filter_MBV, 1, NULL, device.gpu_work_items_per_dim, NULL, 0, NULL, NULL);
		if (device.state_gpu != 0)  printf(">error while deblocking : %d", device.state_gpu);
		device.state_gpu = clFlush(device.commandQueue2_gpu);
		if (device.state_gpu != 0)  printf(">error while deblocking : %d", device.state_gpu);
		device.state_gpu = clSetKernelArg(device.normal_loop_filter_MBV, 0, sizeof(cl_mem), &device.reconstructed_frame_V);
		device.state_gpu = clEnqueueNDRangeKernel(device.commandQueue3_gpu, device.normal_loop_filter_MBV, 1, NULL, device.gpu_work_items_per_dim, NULL, 0, NULL, NULL);
		if (device.state_gpu != 0)  printf(">error while deblocking : %d", device.state_gpu);
		device.state_gpu = clFinish(device.commandQueue3_gpu);
		if (device.state_gpu != 0)  printf(">error while deblocking : %d", device.state_gpu);
	}
	
	return;
}

static void do_loop_filter_on_cpu()
{
	cl_int cwidth = video.wrk_width/2;
	cl_int cheight = video.wrk_height/2;
	cl_int mb_size = 16;

	device.state_cpu = clEnqueueWriteBuffer(device.loopfilterY_commandQueue_cpu, device.cpu_frame_Y, CL_TRUE, 0, video.wrk_frame_size_luma, frames.reconstructed_Y, 0, NULL, NULL);
	device.state_cpu = clEnqueueWriteBuffer(device.loopfilterU_commandQueue_cpu, device.cpu_frame_U, CL_TRUE, 0, video.wrk_frame_size_chroma, frames.reconstructed_U, 0, NULL, NULL);
	device.state_cpu = clEnqueueWriteBuffer(device.loopfilterV_commandQueue_cpu, device.cpu_frame_V, CL_TRUE, 0, video.wrk_frame_size_chroma, frames.reconstructed_V, 0, NULL, NULL);
	
	//after check SSIM we already have reconstructed buffers on device
	device.state_cpu = clFinish(device.loopfilterY_commandQueue_cpu);
	device.state_cpu = clFinish(device.loopfilterU_commandQueue_cpu);
	device.state_cpu = clFinish(device.loopfilterV_commandQueue_cpu);
	// Y
	device.state_cpu = clSetKernelArg(device.loop_filter_frame, 0, sizeof(cl_mem), &device.cpu_frame_Y);
	device.state_cpu = clSetKernelArg(device.loop_filter_frame, 4, sizeof(cl_int), &video.wrk_width);
	device.state_cpu = clSetKernelArg(device.loop_filter_frame, 5, sizeof(cl_int), &video.wrk_height);
	device.state_cpu = clSetKernelArg(device.loop_filter_frame, 6, sizeof(cl_int), &mb_size);
	device.cpu_work_items_per_dim[0] = 1;
	device.state_cpu = clEnqueueNDRangeKernel(device.loopfilterY_commandQueue_cpu, device.loop_filter_frame, 1, NULL, device.cpu_work_items_per_dim, NULL, 0, NULL, NULL);
	if (frames.threads_free > 1) {
		--frames.threads_free;
		device.state_cpu = clFlush(device.loopfilterY_commandQueue_cpu);
	}
	else {
		device.state_cpu = clFinish(device.loopfilterY_commandQueue_cpu);
		frames.threads_free =video.thread_limit;
	}
	if (device.state_cpu != 0)  printf(">error while deblocking : %d", device.state_cpu);
	// U
	mb_size = 8;
	device.state_cpu = clSetKernelArg(device.loop_filter_frame, 0, sizeof(cl_mem), &device.cpu_frame_U);
	device.state_cpu = clSetKernelArg(device.loop_filter_frame, 4, sizeof(cl_int), &cwidth);
	device.state_cpu = clSetKernelArg(device.loop_filter_frame, 5, sizeof(cl_int), &cheight);
	device.state_cpu = clSetKernelArg(device.loop_filter_frame, 6, sizeof(cl_int), &mb_size);
	device.state_cpu = clEnqueueNDRangeKernel(device.loopfilterU_commandQueue_cpu, device.loop_filter_frame, 1, NULL, device.cpu_work_items_per_dim, NULL, 0, NULL, NULL);
	if (frames.threads_free > 1) {
		--frames.threads_free;
		device.state_cpu = clFlush(device.loopfilterU_commandQueue_cpu);
	}
	else {
		device.state_cpu = clFinish(device.loopfilterU_commandQueue_cpu);
		frames.threads_free =video.thread_limit;
	}
	if (device.state_cpu != 0)  printf(">error while deblocking : %d", device.state_cpu);
	// V
	device.state_cpu = clSetKernelArg(device.loop_filter_frame, 0, sizeof(cl_mem), &device.cpu_frame_V);
	device.state_cpu = clEnqueueNDRangeKernel(device.loopfilterV_commandQueue_cpu, device.loop_filter_frame, 1, NULL, device.cpu_work_items_per_dim, NULL, 0, NULL, NULL);
	if (frames.threads_free > 1) {
		--frames.threads_free;
		device.state_cpu = clFlush(device.loopfilterV_commandQueue_cpu);
	}
	else {
		device.state_cpu = clFinish(device.loopfilterV_commandQueue_cpu);
		frames.threads_free =video.thread_limit;
	}
	if (device.state_cpu != 0)  printf(">error while deblocking : %d", device.state_cpu);

	return;
}

static void do_loop_filter()
{
	if (video.do_loop_filter_on_gpu) do_loop_filter_on_gpu();
	else do_loop_filter_on_cpu();
	return;
}