void prepare_GPU_buffers()
{
	cl_int width, height;

	//if filter was done on CPU we need to extract reconstructed frames from CPU device
	if (!video.do_loop_filter_on_gpu)
	{
		device.state_gpu = clFinish(device.loopfilterY_commandQueue_cpu);
		device.state_gpu = clFinish(device.loopfilterU_commandQueue_cpu);
		device.state_gpu = clFinish(device.loopfilterV_commandQueue_cpu);
		// cpu device --> host
		device.state_cpu = clEnqueueReadBuffer(device.loopfilterY_commandQueue_cpu, device.cpu_frame_Y ,CL_TRUE, 0, video.wrk_frame_size_luma, frames.reconstructed_Y, 0, NULL, NULL);
		device.state_cpu = clEnqueueReadBuffer(device.loopfilterU_commandQueue_cpu, device.cpu_frame_U ,CL_TRUE, 0, video.wrk_frame_size_chroma, frames.reconstructed_U, 0, NULL, NULL);
		device.state_cpu = clEnqueueReadBuffer(device.loopfilterV_commandQueue_cpu, device.cpu_frame_V ,CL_TRUE, 0, video.wrk_frame_size_chroma, frames.reconstructed_V, 0, NULL, NULL);
		// host --> gpu device
		device.state_gpu = clEnqueueWriteBuffer(device.commandQueue1_gpu, device.reconstructed_frame_Y, CL_TRUE, 0, video.wrk_frame_size_luma, frames.reconstructed_Y, 0, NULL, NULL);
		device.state_gpu = clEnqueueWriteBuffer(device.commandQueue2_gpu, device.reconstructed_frame_U, CL_TRUE, 0, video.wrk_frame_size_chroma, frames.reconstructed_U, 0, NULL, NULL);
		device.state_gpu = clEnqueueWriteBuffer(device.commandQueue3_gpu, device.reconstructed_frame_V, CL_TRUE, 0, video.wrk_frame_size_chroma, frames.reconstructed_V, 0, NULL, NULL);
	}
	frames.threads_free = video.thread_limit; // by this time boolcoder definetly already finished
	
	// first reset vector nets to zeros
	device.gpu_work_items_per_dim[0] = video.mb_count*4;
	device.state_gpu = clEnqueueNDRangeKernel(device.commandQueue1_gpu, device.reset_vectors, 1, NULL, device.gpu_work_items_per_dim, NULL, 0, NULL, NULL);
	device.state_gpu = clFinish(device.commandQueue1_gpu);

	// now prepare downsampled LAST buffers
	//prepare downsampled by 2
	device.state_gpu = clSetKernelArg(device.downsample, 0, sizeof(cl_mem), &device.reconstructed_frame_Y);
	device.state_gpu = clSetKernelArg(device.downsample, 1, sizeof(cl_mem), &device.last_frame_Y_downsampled_by2);
	device.state_gpu = clSetKernelArg(device.downsample, 2, sizeof(cl_int), &video.wrk_width);
	device.state_gpu = clSetKernelArg(device.downsample, 3, sizeof(cl_int), &video.wrk_height);
	device.gpu_work_items_per_dim[0] = video.wrk_width*video.wrk_height/4;
	device.state_gpu = clEnqueueNDRangeKernel(device.commandQueue1_gpu, device.downsample, 1, NULL, device.gpu_work_items_per_dim, NULL, 0, NULL, NULL);
	device.state_gpu = clFlush(device.commandQueue1_gpu);
	device.state_gpu = clSetKernelArg(device.downsample, 0, sizeof(cl_mem), &device.current_frame_Y);
	device.state_gpu = clSetKernelArg(device.downsample, 1, sizeof(cl_mem), &device.current_frame_Y_downsampled_by2);
	device.state_gpu = clEnqueueNDRangeKernel(device.commandQueue1_gpu, device.downsample, 1, NULL, device.gpu_work_items_per_dim, NULL, 0, NULL, NULL);
	device.state_gpu = clFinish(device.commandQueue1_gpu);
	//prepare downsampled by 4
	device.state_gpu = clSetKernelArg(device.downsample, 0, sizeof(cl_mem), &device.last_frame_Y_downsampled_by2);
	device.state_gpu = clSetKernelArg(device.downsample, 1, sizeof(cl_mem), &device.last_frame_Y_downsampled_by4);
	width = video.wrk_width/2;
	device.state_gpu = clSetKernelArg(device.downsample, 2, sizeof(cl_int), &width);
	height = video.wrk_height/2;
	device.state_gpu = clSetKernelArg(device.downsample, 3, sizeof(cl_int), &height);
	device.gpu_work_items_per_dim[0] = width*height/4;
	device.state_gpu = clEnqueueNDRangeKernel(device.commandQueue1_gpu, device.downsample, 1, NULL, device.gpu_work_items_per_dim, NULL, 0, NULL, NULL);
	device.state_gpu = clFlush(device.commandQueue1_gpu);
	device.state_gpu = clSetKernelArg(device.downsample, 0, sizeof(cl_mem), &device.current_frame_Y_downsampled_by2);
	device.state_gpu = clSetKernelArg(device.downsample, 1, sizeof(cl_mem), &device.current_frame_Y_downsampled_by4);
	device.state_gpu = clEnqueueNDRangeKernel(device.commandQueue1_gpu, device.downsample, 1, NULL, device.gpu_work_items_per_dim, NULL, 0, NULL, NULL);
	device.state_gpu = clFinish(device.commandQueue1_gpu);
	//prepare downsampled by 8
	device.state_gpu = clSetKernelArg(device.downsample, 0, sizeof(cl_mem), &device.last_frame_Y_downsampled_by4);
	device.state_gpu = clSetKernelArg(device.downsample, 1, sizeof(cl_mem), &device.last_frame_Y_downsampled_by8);
	width = video.wrk_width/4;
	device.state_gpu = clSetKernelArg(device.downsample, 2, sizeof(cl_int), &width);
	height = video.wrk_height/4;
	device.state_gpu = clSetKernelArg(device.downsample, 3, sizeof(cl_int), &height);
	device.gpu_work_items_per_dim[0] = width*height/4;
	device.state_gpu = clEnqueueNDRangeKernel(device.commandQueue1_gpu, device.downsample, 1, NULL, device.gpu_work_items_per_dim, NULL, 0, NULL, NULL);
	device.state_gpu = clFlush(device.commandQueue1_gpu);
	device.state_gpu = clSetKernelArg(device.downsample, 0, sizeof(cl_mem), &device.current_frame_Y_downsampled_by4);
	device.state_gpu = clSetKernelArg(device.downsample, 1, sizeof(cl_mem), &device.current_frame_Y_downsampled_by8);
	device.state_gpu = clEnqueueNDRangeKernel(device.commandQueue1_gpu, device.downsample, 1, NULL, device.gpu_work_items_per_dim, NULL, 0, NULL, NULL);
	device.state_gpu = clFinish(device.commandQueue1_gpu);
	//prepare downsampled by 16
	device.state_gpu = clSetKernelArg(device.downsample, 0, sizeof(cl_mem), &device.last_frame_Y_downsampled_by8);
	device.state_gpu = clSetKernelArg(device.downsample, 1, sizeof(cl_mem), &device.last_frame_Y_downsampled_by16);
	width = video.wrk_width/8;
	device.state_gpu = clSetKernelArg(device.downsample, 2, sizeof(cl_int), &width);
	height = video.wrk_height/8;
	device.state_gpu = clSetKernelArg(device.downsample, 3, sizeof(cl_int), &height);
	device.gpu_work_items_per_dim[0] = width*height/4;
	device.state_gpu = clEnqueueNDRangeKernel(device.commandQueue1_gpu, device.downsample, 1, NULL, device.gpu_work_items_per_dim, NULL, 0, NULL, NULL);
	device.state_gpu = clFlush(device.commandQueue1_gpu);
	device.state_gpu = clSetKernelArg(device.downsample, 0, sizeof(cl_mem), &device.current_frame_Y_downsampled_by8);
	device.state_gpu = clSetKernelArg(device.downsample, 1, sizeof(cl_mem), &device.current_frame_Y_downsampled_by16);
	device.state_gpu = clEnqueueNDRangeKernel(device.commandQueue1_gpu, device.downsample, 1, NULL, device.gpu_work_items_per_dim, NULL, 0, NULL, NULL);
	device.state_gpu = clFinish(device.commandQueue1_gpu);
	
	if (frames.prev_is_golden_frame)
	{
		device.state_gpu = clEnqueueCopyBuffer(device.commandQueue1_gpu, device.reconstructed_frame_Y, device.golden_frame_Y, 0, 0, video.wrk_frame_size_luma, 0, NULL, NULL);
		device.state_gpu = clEnqueueCopyBuffer(device.commandQueue1_gpu, device.last_frame_Y_downsampled_by2, device.golden_frame_Y_downsampled_by2, 0, 0, video.wrk_frame_size_luma/4, 0, NULL, NULL);
		device.state_gpu = clEnqueueCopyBuffer(device.commandQueue1_gpu, device.last_frame_Y_downsampled_by4, device.golden_frame_Y_downsampled_by4, 0, 0, video.wrk_frame_size_luma/16, 0, NULL, NULL);
		device.state_gpu = clEnqueueCopyBuffer(device.commandQueue1_gpu, device.last_frame_Y_downsampled_by8, device.golden_frame_Y_downsampled_by8, 0, 0, video.wrk_frame_size_luma/64, 0, NULL, NULL);
		device.state_gpu = clEnqueueCopyBuffer(device.commandQueue1_gpu, device.last_frame_Y_downsampled_by16, device.golden_frame_Y_downsampled_by16, 0, 0, video.wrk_frame_size_luma/256, 0, NULL, NULL);
	}
	if (frames.prev_is_altref_frame)
	{
		device.state_gpu = clEnqueueCopyBuffer(device.commandQueue1_gpu, device.reconstructed_frame_Y, device.altref_frame_Y, 0, 0, video.wrk_frame_size_luma, 0, NULL, NULL);
		device.state_gpu = clEnqueueCopyBuffer(device.commandQueue1_gpu, device.last_frame_Y_downsampled_by2, device.altref_frame_Y_downsampled_by2, 0, 0, video.wrk_frame_size_luma/4, 0, NULL, NULL);
		device.state_gpu = clEnqueueCopyBuffer(device.commandQueue1_gpu, device.last_frame_Y_downsampled_by4, device.altref_frame_Y_downsampled_by4, 0, 0, video.wrk_frame_size_luma/16, 0, NULL, NULL);
		device.state_gpu = clEnqueueCopyBuffer(device.commandQueue1_gpu, device.last_frame_Y_downsampled_by8, device.altref_frame_Y_downsampled_by8, 0, 0, video.wrk_frame_size_luma/64, 0, NULL, NULL);
		device.state_gpu = clEnqueueCopyBuffer(device.commandQueue1_gpu, device.last_frame_Y_downsampled_by16, device.altref_frame_Y_downsampled_by16, 0, 0, video.wrk_frame_size_luma/256, 0, NULL, NULL);
	}
	device.state_gpu = clFinish(device.commandQueue1_gpu);

	// prepare images (if they need to be renewed)
	const size_t origin[3] = {0, 0, 0};
	const size_t region_y[3] = {video.wrk_width, video.wrk_height, 1};
	const size_t region_uv[3] = {video.wrk_width/2, video.wrk_height/2, 1};

	device.state_gpu = clEnqueueReadBuffer(device.commandQueue1_gpu, device.reconstructed_frame_Y ,CL_TRUE, 0, video.wrk_frame_size_luma, frames.reconstructed_Y, 0, NULL, NULL);
	device.state_gpu = clEnqueueReadBuffer(device.commandQueue2_gpu, device.reconstructed_frame_U ,CL_TRUE, 0, video.wrk_frame_size_chroma, frames.reconstructed_U, 0, NULL, NULL);
	device.state_gpu = clEnqueueReadBuffer(device.commandQueue3_gpu, device.reconstructed_frame_V ,CL_TRUE, 0, video.wrk_frame_size_chroma, frames.reconstructed_V, 0, NULL, NULL);

	device.state_gpu = clEnqueueWriteImage(device.commandQueue1_gpu, device.last_frame_Y_image, CL_TRUE, origin, region_y, 0, 0, frames.reconstructed_Y, 0, NULL, NULL);
	device.state_gpu = clEnqueueWriteImage(device.commandQueue2_gpu, device.last_frame_U_image, CL_TRUE, origin, region_uv, 0, 0, frames.reconstructed_U, 0, NULL, NULL);
	device.state_gpu = clEnqueueWriteImage(device.commandQueue3_gpu, device.last_frame_V_image, CL_TRUE, origin, region_uv, 0, 0, frames.reconstructed_V, 0, NULL, NULL);

	if (frames.prev_is_golden_frame)
	{
		device.state_gpu = clEnqueueCopyImage(device.commandQueue1_gpu, device.last_frame_Y_image, device.golden_frame_Y_image, origin, origin, region_y, 0, NULL, NULL);
		device.state_gpu = clEnqueueCopyImage(device.commandQueue2_gpu, device.last_frame_U_image, device.golden_frame_U_image, origin, origin, region_uv, 0, NULL, NULL);
		device.state_gpu = clEnqueueCopyImage(device.commandQueue3_gpu, device.last_frame_V_image, device.golden_frame_V_image, origin, origin, region_uv, 0, NULL, NULL);
	}
	if (frames.prev_is_altref_frame)
	{
		device.state_gpu = clEnqueueCopyImage(device.commandQueue1_gpu, device.last_frame_Y_image, device.altref_frame_Y_image, origin, origin, region_y, 0, NULL, NULL);
		device.state_gpu = clEnqueueCopyImage(device.commandQueue2_gpu, device.last_frame_U_image, device.altref_frame_U_image, origin, origin, region_uv, 0, NULL, NULL);
		device.state_gpu = clEnqueueCopyImage(device.commandQueue3_gpu, device.last_frame_V_image, device.altref_frame_V_image, origin, origin, region_uv, 0, NULL, NULL);
	}
	
	return;
}

void inter_transform()
{
	clFinish(device.commandQueue1_gpu);
	clFinish(device.commandQueue2_gpu);
	clFinish(device.commandQueue3_gpu);

	cl_int val;
	// if golden and altref buffers represent different from last buffer frame
	// and altref is not the same as altref
	cl_int use_golden = !frames.prev_is_golden_frame;
	cl_int use_altref = (!frames.prev_is_altref_frame) && (frames.altref_frame_number != frames.golden_frame_number);
	//prepare downsampled frames and image objects
	prepare_GPU_buffers();
	
	//now search in downsampled by 16
	device.gpu_work_items_per_dim[0] = ((video.wrk_width/16)/8)*((video.wrk_height/16)/8);
	device.state_gpu = clSetKernelArg(device.luma_search_1step, 0, sizeof(cl_mem), &device.current_frame_Y_downsampled_by16);
	val = video.wrk_width/16;
	device.state_gpu = clSetKernelArg(device.luma_search_1step, 5, sizeof(cl_int), &val);
	val = video.wrk_height/16;
	device.state_gpu = clSetKernelArg(device.luma_search_1step, 6, sizeof(cl_int), &val);
	val = 16;
	device.state_gpu = clSetKernelArg(device.luma_search_1step, 7, sizeof(cl_int), &val);
	// LAST
	device.state_gpu = clSetKernelArg(device.luma_search_1step, 1, sizeof(cl_mem), &device.last_frame_Y_downsampled_by16);
	device.state_gpu = clSetKernelArg(device.luma_search_1step, 2, sizeof(cl_mem), &device.last_vnet1);
	device.state_gpu = clSetKernelArg(device.luma_search_1step, 3, sizeof(cl_mem), &device.last_vnet2);
	device.state_gpu = clEnqueueNDRangeKernel(device.commandQueue1_gpu, device.luma_search_1step, 1, NULL, device.gpu_work_items_per_dim, NULL, 0, NULL, NULL);
	device.state_gpu = clFlush(device.commandQueue1_gpu);
	// GOLDEN
	if (use_golden)
	{
		device.state_gpu = clSetKernelArg(device.luma_search_1step, 1, sizeof(cl_mem), &device.golden_frame_Y_downsampled_by16);
		device.state_gpu = clSetKernelArg(device.luma_search_1step, 2, sizeof(cl_mem), &device.golden_vnet1);
		device.state_gpu = clSetKernelArg(device.luma_search_1step, 3, sizeof(cl_mem), &device.golden_vnet2);
		device.state_gpu = clEnqueueNDRangeKernel(device.commandQueue2_gpu, device.luma_search_1step, 1, NULL, device.gpu_work_items_per_dim, NULL, 0, NULL, NULL);
		device.state_gpu = clFlush(device.commandQueue2_gpu);
	}
	// ALTREF
	if (use_altref)
	{
		device.state_gpu = clSetKernelArg(device.luma_search_1step, 1, sizeof(cl_mem), &device.altref_frame_Y_downsampled_by16);
		device.state_gpu = clSetKernelArg(device.luma_search_1step, 2, sizeof(cl_mem), &device.altref_vnet1);
		device.state_gpu = clSetKernelArg(device.luma_search_1step, 3, sizeof(cl_mem), &device.altref_vnet2);
		device.state_gpu = clEnqueueNDRangeKernel(device.commandQueue3_gpu, device.luma_search_1step, 1, NULL, device.gpu_work_items_per_dim, NULL, 0, NULL, NULL);
		device.state_gpu = clFlush(device.commandQueue3_gpu);
	}

	//now search in downsampled by 8
	device.gpu_work_items_per_dim[0] = ((video.wrk_width/8)/8)*((video.wrk_height/8)/8);
	device.state_gpu = clSetKernelArg(device.luma_search_1step, 0, sizeof(cl_mem), &device.current_frame_Y_downsampled_by8);
	val = video.wrk_width/8;
	device.state_gpu = clSetKernelArg(device.luma_search_1step, 5, sizeof(cl_int), &val);
	val = video.wrk_height/8;
	device.state_gpu = clSetKernelArg(device.luma_search_1step, 6, sizeof(cl_int), &val);
	val = 8;
	device.state_gpu = clSetKernelArg(device.luma_search_1step, 7, sizeof(cl_int), &val);
	// LAST
	device.state_gpu = clSetKernelArg(device.luma_search_1step, 1, sizeof(cl_mem), &device.last_frame_Y_downsampled_by8);
	device.state_gpu = clSetKernelArg(device.luma_search_1step, 2, sizeof(cl_mem), &device.last_vnet2);
	device.state_gpu = clSetKernelArg(device.luma_search_1step, 3, sizeof(cl_mem), &device.last_vnet1);
	device.state_gpu = clEnqueueNDRangeKernel(device.commandQueue1_gpu, device.luma_search_1step, 1, NULL, device.gpu_work_items_per_dim, NULL, 0, NULL, NULL);
	device.state_gpu = clFlush(device.commandQueue1_gpu);
	// GOLDEN
	if (use_golden)
	{
		device.state_gpu = clSetKernelArg(device.luma_search_1step, 1, sizeof(cl_mem), &device.golden_frame_Y_downsampled_by8);
		device.state_gpu = clSetKernelArg(device.luma_search_1step, 2, sizeof(cl_mem), &device.golden_vnet2);
		device.state_gpu = clSetKernelArg(device.luma_search_1step, 3, sizeof(cl_mem), &device.golden_vnet1);
		device.state_gpu = clEnqueueNDRangeKernel(device.commandQueue2_gpu, device.luma_search_1step, 1, NULL, device.gpu_work_items_per_dim, NULL, 0, NULL, NULL);
		device.state_gpu = clFlush(device.commandQueue2_gpu);
	}
	// ALTREF
	if (use_altref)
	{
		device.state_gpu = clSetKernelArg(device.luma_search_1step, 1, sizeof(cl_mem), &device.altref_frame_Y_downsampled_by8);
		device.state_gpu = clSetKernelArg(device.luma_search_1step, 2, sizeof(cl_mem), &device.altref_vnet2);
		device.state_gpu = clSetKernelArg(device.luma_search_1step, 3, sizeof(cl_mem), &device.altref_vnet1);
		device.state_gpu = clEnqueueNDRangeKernel(device.commandQueue3_gpu, device.luma_search_1step, 1, NULL, device.gpu_work_items_per_dim, NULL, 0, NULL, NULL);
		device.state_gpu = clFlush(device.commandQueue3_gpu);
	}	

	//now search in downsampled by 4
	device.gpu_work_items_per_dim[0] = ((video.wrk_width/4)/8)*((video.wrk_height/4)/8);
	device.state_gpu = clSetKernelArg(device.luma_search_1step, 0, sizeof(cl_mem), &device.current_frame_Y_downsampled_by4);
	val = video.wrk_width/4;
	device.state_gpu = clSetKernelArg(device.luma_search_1step, 5, sizeof(cl_int), &val);
	val = video.wrk_height/4;
	device.state_gpu = clSetKernelArg(device.luma_search_1step, 6, sizeof(cl_int), &val);
	val = 4;
	device.state_gpu = clSetKernelArg(device.luma_search_1step, 7, sizeof(cl_int), &val);
	// LAST
	device.state_gpu = clSetKernelArg(device.luma_search_1step, 1, sizeof(cl_mem), &device.last_frame_Y_downsampled_by4);
	device.state_gpu = clSetKernelArg(device.luma_search_1step, 2, sizeof(cl_mem), &device.last_vnet1);
	device.state_gpu = clSetKernelArg(device.luma_search_1step, 3, sizeof(cl_mem), &device.last_vnet2);
	device.state_gpu = clEnqueueNDRangeKernel(device.commandQueue1_gpu, device.luma_search_1step, 1, NULL, device.gpu_work_items_per_dim, NULL, 0, NULL, NULL);
	device.state_gpu = clFlush(device.commandQueue1_gpu);
	// GOLDEN
	if (use_golden)
	{
		device.state_gpu = clSetKernelArg(device.luma_search_1step, 1, sizeof(cl_mem), &device.golden_frame_Y_downsampled_by4);
		device.state_gpu = clSetKernelArg(device.luma_search_1step, 2, sizeof(cl_mem), &device.golden_vnet1);
		device.state_gpu = clSetKernelArg(device.luma_search_1step, 3, sizeof(cl_mem), &device.golden_vnet2);
		device.state_gpu = clEnqueueNDRangeKernel(device.commandQueue2_gpu, device.luma_search_1step, 1, NULL, device.gpu_work_items_per_dim, NULL, 0, NULL, NULL);
		device.state_gpu = clFlush(device.commandQueue2_gpu);
	}
	// ALTREF
	if (use_altref)
	{
		device.state_gpu = clSetKernelArg(device.luma_search_1step, 1, sizeof(cl_mem), &device.altref_frame_Y_downsampled_by4);
		device.state_gpu = clSetKernelArg(device.luma_search_1step, 2, sizeof(cl_mem), &device.altref_vnet1);
		device.state_gpu = clSetKernelArg(device.luma_search_1step, 3, sizeof(cl_mem), &device.altref_vnet2);
		device.state_gpu = clEnqueueNDRangeKernel(device.commandQueue3_gpu, device.luma_search_1step, 1, NULL, device.gpu_work_items_per_dim, NULL, 0, NULL, NULL);
		device.state_gpu = clFlush(device.commandQueue3_gpu);
	}

	//now search in downsampled by 2
	device.gpu_work_items_per_dim[0] = ((video.wrk_width/2)/8)*((video.wrk_height/2)/8);
	device.state_gpu = clSetKernelArg(device.luma_search_1step, 0, sizeof(cl_mem), &device.current_frame_Y_downsampled_by2);
	val = video.wrk_width/2;
	device.state_gpu = clSetKernelArg(device.luma_search_1step, 5, sizeof(cl_int), &val);
	val = video.wrk_height/2;
	device.state_gpu = clSetKernelArg(device.luma_search_1step, 6, sizeof(cl_int), &val);
	val = 2;
	device.state_gpu = clSetKernelArg(device.luma_search_1step, 7, sizeof(cl_int), &val);
	// LAST
	device.state_gpu = clSetKernelArg(device.luma_search_1step, 1, sizeof(cl_mem), &device.last_frame_Y_downsampled_by2);
	device.state_gpu = clSetKernelArg(device.luma_search_1step, 2, sizeof(cl_mem), &device.last_vnet2);
	device.state_gpu = clSetKernelArg(device.luma_search_1step, 3, sizeof(cl_mem), &device.last_vnet1);
	device.state_gpu = clEnqueueNDRangeKernel(device.commandQueue1_gpu, device.luma_search_1step, 1, NULL, device.gpu_work_items_per_dim, NULL, 0, NULL, NULL);
	device.state_gpu = clFlush(device.commandQueue1_gpu);
	// GOLDEN
	if (use_golden)
	{
		device.state_gpu = clSetKernelArg(device.luma_search_1step, 1, sizeof(cl_mem), &device.golden_frame_Y_downsampled_by2);
		device.state_gpu = clSetKernelArg(device.luma_search_1step, 2, sizeof(cl_mem), &device.golden_vnet2);
		device.state_gpu = clSetKernelArg(device.luma_search_1step, 3, sizeof(cl_mem), &device.golden_vnet1);
		device.state_gpu = clEnqueueNDRangeKernel(device.commandQueue2_gpu, device.luma_search_1step, 1, NULL, device.gpu_work_items_per_dim, NULL, 0, NULL, NULL);
		device.state_gpu = clFlush(device.commandQueue2_gpu);
	}
	// ALTREF
	if (use_altref)
	{
		device.state_gpu = clSetKernelArg(device.luma_search_1step, 1, sizeof(cl_mem), &device.altref_frame_Y_downsampled_by2);
		device.state_gpu = clSetKernelArg(device.luma_search_1step, 2, sizeof(cl_mem), &device.altref_vnet2);
		device.state_gpu = clSetKernelArg(device.luma_search_1step, 3, sizeof(cl_mem), &device.altref_vnet1);
		device.state_gpu = clEnqueueNDRangeKernel(device.commandQueue3_gpu, device.luma_search_1step, 1, NULL, device.gpu_work_items_per_dim, NULL, 0, NULL, NULL);
		device.state_gpu = clFlush(device.commandQueue3_gpu);
	}

	//now search in original size
	device.gpu_work_items_per_dim[0] = video.mb_count*4; 
	device.state_gpu = clSetKernelArg(device.luma_search_1step, 0, sizeof(cl_mem), &device.current_frame_Y);
	val = video.wrk_width;
	device.state_gpu = clSetKernelArg(device.luma_search_1step, 5, sizeof(cl_int), &val);
	val = video.wrk_height;
	device.state_gpu = clSetKernelArg(device.luma_search_1step, 6, sizeof(cl_int), &val);
	val = 1;
	device.state_gpu = clSetKernelArg(device.luma_search_1step, 7, sizeof(cl_int), &val);
	// LAST
	device.state_gpu = clSetKernelArg(device.luma_search_1step, 1, sizeof(cl_mem), &device.reconstructed_frame_Y);
	device.state_gpu = clSetKernelArg(device.luma_search_1step, 2, sizeof(cl_mem), &device.last_vnet1);
	device.state_gpu = clSetKernelArg(device.luma_search_1step, 3, sizeof(cl_mem), &device.last_vnet2);
	device.state_gpu = clEnqueueNDRangeKernel(device.commandQueue1_gpu, device.luma_search_1step, 1, NULL, device.gpu_work_items_per_dim, NULL, 0, NULL, NULL);
	device.state_gpu = clFlush(device.commandQueue1_gpu);
	// GOLDEN
	if (use_golden)
	{
		device.state_gpu = clSetKernelArg(device.luma_search_1step, 1, sizeof(cl_mem), &device.golden_frame_Y);
		device.state_gpu = clSetKernelArg(device.luma_search_1step, 2, sizeof(cl_mem), &device.golden_vnet1);
		device.state_gpu = clSetKernelArg(device.luma_search_1step, 3, sizeof(cl_mem), &device.golden_vnet2);
		device.state_gpu = clEnqueueNDRangeKernel(device.commandQueue2_gpu, device.luma_search_1step, 1, NULL, device.gpu_work_items_per_dim, NULL, 0, NULL, NULL);
		device.state_gpu = clFlush(device.commandQueue2_gpu);
	}
	// ALTREF
	if (use_altref)
	{
		device.state_gpu = clSetKernelArg(device.luma_search_1step, 1, sizeof(cl_mem), &device.altref_frame_Y);
		device.state_gpu = clSetKernelArg(device.luma_search_1step, 2, sizeof(cl_mem), &device.altref_vnet1);
		device.state_gpu = clSetKernelArg(device.luma_search_1step, 3, sizeof(cl_mem), &device.altref_vnet2);
		device.state_gpu = clEnqueueNDRangeKernel(device.commandQueue3_gpu, device.luma_search_1step, 1, NULL, device.gpu_work_items_per_dim, NULL, 0, NULL, NULL);
		device.state_gpu = clFlush(device.commandQueue3_gpu);
	}

	// search in image with interpolation on the run
	device.gpu_work_items_per_dim[0] = video.mb_count*4;
	// LAST
	device.state_gpu = clSetKernelArg(device.luma_search_2step, 1, sizeof(cl_mem), &device.last_frame_Y_image);
	device.state_gpu = clSetKernelArg(device.luma_search_2step, 2, sizeof(cl_mem), &device.last_vnet2);
	device.state_gpu = clSetKernelArg(device.luma_search_2step, 3, sizeof(cl_mem), &device.last_vnet1);
	device.state_gpu = clSetKernelArg(device.luma_search_2step, 4, sizeof(cl_mem), &device.last_metrics);
	device.state_gpu = clEnqueueNDRangeKernel(device.commandQueue1_gpu, device.luma_search_2step, 1, NULL, device.gpu_work_items_per_dim, NULL, 0, NULL, NULL);
	device.state_gpu = clFlush(device.commandQueue1_gpu);
	// GOLDEN
	if (use_golden)
	{
		device.state_gpu = clSetKernelArg(device.luma_search_2step, 1, sizeof(cl_mem), &device.golden_frame_Y_image);
		device.state_gpu = clSetKernelArg(device.luma_search_2step, 2, sizeof(cl_mem), &device.golden_vnet2);
		device.state_gpu = clSetKernelArg(device.luma_search_2step, 3, sizeof(cl_mem), &device.golden_vnet1);
		device.state_gpu = clSetKernelArg(device.luma_search_2step, 4, sizeof(cl_mem), &device.golden_metrics);
		device.state_gpu = clEnqueueNDRangeKernel(device.commandQueue2_gpu, device.luma_search_2step, 1, NULL, device.gpu_work_items_per_dim, NULL, 0, NULL, NULL);
		device.state_gpu = clFlush(device.commandQueue2_gpu);
	}
	// ALTREF
	if (use_altref)
	{
		device.state_gpu = clSetKernelArg(device.luma_search_2step, 1, sizeof(cl_mem), &device.altref_frame_Y_image);
		device.state_gpu = clSetKernelArg(device.luma_search_2step, 2, sizeof(cl_mem), &device.altref_vnet2);
		device.state_gpu = clSetKernelArg(device.luma_search_2step, 3, sizeof(cl_mem), &device.altref_vnet1);
		device.state_gpu = clSetKernelArg(device.luma_search_2step, 4, sizeof(cl_mem), &device.altref_metrics);
		device.state_gpu = clEnqueueNDRangeKernel(device.commandQueue3_gpu, device.luma_search_2step, 1, NULL, device.gpu_work_items_per_dim, NULL, 0, NULL, NULL);
		device.state_gpu = clFlush(device.commandQueue3_gpu);
	}

	device.state_gpu = clFinish(device.commandQueue1_gpu);
	device.state_gpu = clFinish(device.commandQueue2_gpu);
	device.state_gpu = clFinish(device.commandQueue3_gpu);

	// now set each MB with the best reference
	device.gpu_work_items_per_dim[0] = video.mb_count;
	device.state_gpu = clSetKernelArg(device.select_reference, 8, sizeof(cl_int), &use_golden);
	device.state_gpu = clSetKernelArg(device.select_reference, 9, sizeof(cl_int), &use_altref);
	device.state_gpu = clEnqueueNDRangeKernel(device.commandQueue1_gpu, device.select_reference, 1, NULL, device.gpu_work_items_per_dim, NULL, 0, NULL, NULL);

	// set 16x16 mode for macroblocks, whose blocks have identical vectors
	device.gpu_work_items_per_dim[0] = video.mb_count;
	device.state_gpu = clEnqueueNDRangeKernel(device.commandQueue1_gpu, device.pack_8x8_into_16x16, 1, NULL, device.gpu_work_items_per_dim, NULL, 0, NULL, NULL);
	device.state_gpu = clFinish(device.commandQueue1_gpu);

	// now for each plane and reference frame fill predictors and residual buffers
	cl_int plane, ref;
	cl_int cwidth = video.wrk_width/2;
	// Y
	plane = 0;
	device.state_gpu = clSetKernelArg(device.prepare_predictors_and_residual, 0, sizeof(cl_mem), &device.current_frame_Y);
	device.state_gpu = clSetKernelArg(device.prepare_predictors_and_residual, 2, sizeof(cl_mem), &device.predictors_Y);
	device.state_gpu = clSetKernelArg(device.prepare_predictors_and_residual, 3, sizeof(cl_mem), &device.residual_Y);
	device.state_gpu = clSetKernelArg(device.prepare_predictors_and_residual, 5, sizeof(cl_int), &video.wrk_width);
	device.state_gpu = clSetKernelArg(device.prepare_predictors_and_residual, 6, sizeof(cl_int), &plane);
	device.gpu_work_items_per_dim[0] = video.mb_count*16;
	// LAST
	ref = LAST;
	device.state_gpu = clSetKernelArg(device.prepare_predictors_and_residual, 1, sizeof(cl_mem), &device.last_frame_Y_image);
	device.state_gpu = clSetKernelArg(device.prepare_predictors_and_residual, 7, sizeof(cl_int), &ref);
	device.state_gpu = clEnqueueNDRangeKernel(device.commandQueue1_gpu, device.prepare_predictors_and_residual, 1, NULL, device.gpu_work_items_per_dim, NULL, 0, NULL, NULL);
	device.state_gpu = clFlush(device.commandQueue1_gpu);
	// GOLDEN
	if (use_golden)
	{
		ref = GOLDEN;
		device.state_gpu = clSetKernelArg(device.prepare_predictors_and_residual, 1, sizeof(cl_mem), &device.golden_frame_Y_image);
		device.state_gpu = clSetKernelArg(device.prepare_predictors_and_residual, 7, sizeof(cl_int), &ref);
		device.state_gpu = clEnqueueNDRangeKernel(device.commandQueue2_gpu, device.prepare_predictors_and_residual, 1, NULL, device.gpu_work_items_per_dim, NULL, 0, NULL, NULL);
		device.state_gpu = clFlush(device.commandQueue2_gpu);
	}
	// ALTREF
	if (use_altref)
	{
		ref = ALTREF;
		device.state_gpu = clSetKernelArg(device.prepare_predictors_and_residual, 1, sizeof(cl_mem), &device.altref_frame_Y_image);
		device.state_gpu = clSetKernelArg(device.prepare_predictors_and_residual, 7, sizeof(cl_int), &ref);
		device.state_gpu = clEnqueueNDRangeKernel(device.commandQueue3_gpu, device.prepare_predictors_and_residual, 1, NULL, device.gpu_work_items_per_dim, NULL, 0, NULL, NULL);
		device.state_gpu = clFlush(device.commandQueue3_gpu);
	}
	// U
	plane = 1;
	device.state_gpu = clSetKernelArg(device.prepare_predictors_and_residual, 0, sizeof(cl_mem), &device.current_frame_U);
	device.state_gpu = clSetKernelArg(device.prepare_predictors_and_residual, 2, sizeof(cl_mem), &device.predictors_U);
	device.state_gpu = clSetKernelArg(device.prepare_predictors_and_residual, 3, sizeof(cl_mem), &device.residual_U);
	device.state_gpu = clSetKernelArg(device.prepare_predictors_and_residual, 5, sizeof(cl_int), &cwidth);
	device.state_gpu = clSetKernelArg(device.prepare_predictors_and_residual, 6, sizeof(cl_int), &plane);
	device.gpu_work_items_per_dim[0] = video.mb_count*4;
	// LAST
	ref = LAST;
	device.state_gpu = clSetKernelArg(device.prepare_predictors_and_residual, 1, sizeof(cl_mem), &device.last_frame_U_image);
	device.state_gpu = clSetKernelArg(device.prepare_predictors_and_residual, 7, sizeof(cl_int), &ref);
	device.state_gpu = clEnqueueNDRangeKernel(device.commandQueue1_gpu, device.prepare_predictors_and_residual, 1, NULL, device.gpu_work_items_per_dim, NULL, 0, NULL, NULL);
	device.state_gpu = clFlush(device.commandQueue1_gpu);
	// GOLDEN
	if (use_golden)
	{
		ref = GOLDEN;
		device.state_gpu = clSetKernelArg(device.prepare_predictors_and_residual, 1, sizeof(cl_mem), &device.golden_frame_U_image);
		device.state_gpu = clSetKernelArg(device.prepare_predictors_and_residual, 7, sizeof(cl_int), &ref);
		device.state_gpu = clEnqueueNDRangeKernel(device.commandQueue2_gpu, device.prepare_predictors_and_residual, 1, NULL, device.gpu_work_items_per_dim, NULL, 0, NULL, NULL);
		device.state_gpu = clFlush(device.commandQueue2_gpu);
	}
	// ALTREF
	if (use_altref)
	{
		ref = ALTREF;
		device.state_gpu = clSetKernelArg(device.prepare_predictors_and_residual, 1, sizeof(cl_mem), &device.altref_frame_U_image);
		device.state_gpu = clSetKernelArg(device.prepare_predictors_and_residual, 7, sizeof(cl_int), &ref);
		device.state_gpu = clEnqueueNDRangeKernel(device.commandQueue3_gpu, device.prepare_predictors_and_residual, 1, NULL, device.gpu_work_items_per_dim, NULL, 0, NULL, NULL);
		device.state_gpu = clFlush(device.commandQueue3_gpu);
	}
	// V
	plane = 2;
	device.state_gpu = clSetKernelArg(device.prepare_predictors_and_residual, 0, sizeof(cl_mem), &device.current_frame_V);
	device.state_gpu = clSetKernelArg(device.prepare_predictors_and_residual, 2, sizeof(cl_mem), &device.predictors_V);
	device.state_gpu = clSetKernelArg(device.prepare_predictors_and_residual, 3, sizeof(cl_mem), &device.residual_V);
	device.state_gpu = clSetKernelArg(device.prepare_predictors_and_residual, 6, sizeof(cl_int), &plane);
	// LAST
	ref = LAST;
	device.state_gpu = clSetKernelArg(device.prepare_predictors_and_residual, 1, sizeof(cl_mem), &device.last_frame_V_image);
	device.state_gpu = clSetKernelArg(device.prepare_predictors_and_residual, 7, sizeof(cl_int), &ref);
	device.state_gpu = clEnqueueNDRangeKernel(device.commandQueue1_gpu, device.prepare_predictors_and_residual, 1, NULL, device.gpu_work_items_per_dim, NULL, 0, NULL, NULL);
	device.state_gpu = clFlush(device.commandQueue1_gpu);
	// GOLDEN
	if (use_golden)
	{
		ref = GOLDEN;
		device.state_gpu = clSetKernelArg(device.prepare_predictors_and_residual, 1, sizeof(cl_mem), &device.golden_frame_V_image);
		device.state_gpu = clSetKernelArg(device.prepare_predictors_and_residual, 7, sizeof(cl_int), &ref);
		device.state_gpu = clEnqueueNDRangeKernel(device.commandQueue2_gpu, device.prepare_predictors_and_residual, 1, NULL, device.gpu_work_items_per_dim, NULL, 0, NULL, NULL);
		device.state_gpu = clFlush(device.commandQueue2_gpu);
	}
	// ALTREF
	if (use_altref)
	{
		ref = ALTREF;
		device.state_gpu = clSetKernelArg(device.prepare_predictors_and_residual, 1, sizeof(cl_mem), &device.altref_frame_V_image);
		device.state_gpu = clSetKernelArg(device.prepare_predictors_and_residual, 7, sizeof(cl_int), &ref);
		device.state_gpu = clEnqueueNDRangeKernel(device.commandQueue3_gpu, device.prepare_predictors_and_residual, 1, NULL, device.gpu_work_items_per_dim, NULL, 0, NULL, NULL);
		device.state_gpu = clFlush(device.commandQueue3_gpu);
	}
	
	// now for each segment (begin with highest quantizer (last index))
	cl_int seg_id;
	for (seg_id = LQ_segment; seg_id >= UQ_segment; --seg_id)
	{
		device.state_gpu = clFinish(device.commandQueue1_gpu);
		device.state_gpu = clFinish(device.commandQueue2_gpu);
		device.state_gpu = clFinish(device.commandQueue3_gpu);

		device.state_gpu = clSetKernelArg(device.dct4x4, 4, sizeof(cl_int), &seg_id);
		device.state_gpu = clSetKernelArg(device.wht4x4_iwht4x4, 2, sizeof(cl_int), &seg_id);
		device.state_gpu = clSetKernelArg(device.idct4x4, 5, sizeof(cl_int), &seg_id);
		device.state_gpu = clSetKernelArg(device.count_SSIM_chroma, 4, sizeof(cl_int), &seg_id);
		device.state_gpu = clSetKernelArg(device.count_SSIM_luma, 4, sizeof(cl_int), &seg_id);
		//dct Y
		device.state_gpu = clSetKernelArg(device.dct4x4, 0, sizeof(cl_mem), &device.residual_Y);
		device.state_gpu = clSetKernelArg(device.dct4x4, 2, sizeof(cl_int), &video.wrk_width);
		plane = 0;
		device.state_gpu = clSetKernelArg(device.dct4x4, 6, sizeof(cl_int), &plane);
		device.gpu_work_items_per_dim[0] = video.mb_count*16;
		device.state_gpu = clEnqueueNDRangeKernel(device.commandQueue1_gpu, device.dct4x4, 1, NULL, device.gpu_work_items_per_dim, NULL, 0, NULL, NULL);
		device.state_gpu = clFlush(device.commandQueue1_gpu);	
		//dct U
		device.state_gpu = clSetKernelArg(device.dct4x4, 0, sizeof(cl_mem), &device.residual_U);
		device.state_gpu = clSetKernelArg(device.dct4x4, 2, sizeof(cl_int), &cwidth);
		plane = 1;
		device.state_gpu = clSetKernelArg(device.dct4x4, 6, sizeof(cl_int), &plane);
		device.gpu_work_items_per_dim[0] = video.mb_count*4;
		device.state_gpu = clEnqueueNDRangeKernel(device.commandQueue2_gpu, device.dct4x4, 1, NULL, device.gpu_work_items_per_dim, NULL, 0, NULL, NULL);
		device.state_gpu = clFlush(device.commandQueue2_gpu);	
		//dct V
		device.state_gpu = clSetKernelArg(device.dct4x4, 0, sizeof(cl_mem), &device.residual_V);
		plane = 2;
		device.state_gpu = clSetKernelArg(device.dct4x4, 6, sizeof(cl_int), &plane);
		device.state_gpu = clEnqueueNDRangeKernel(device.commandQueue3_gpu, device.dct4x4, 1, NULL, device.gpu_work_items_per_dim, NULL, 0, NULL, NULL);
		device.state_gpu = clFlush(device.commandQueue3_gpu);
		//wht and iwht
		device.gpu_work_items_per_dim[0] = video.mb_count;
		device.state_gpu = clEnqueueNDRangeKernel(device.commandQueue1_gpu, device.wht4x4_iwht4x4, 1, NULL, device.gpu_work_items_per_dim, NULL, 0, NULL, NULL);
		device.state_gpu = clFlush(device.commandQueue1_gpu);
		//idct Y
		device.state_gpu = clSetKernelArg(device.idct4x4, 0, sizeof(cl_mem), &device.reconstructed_frame_Y);
		device.state_gpu = clSetKernelArg(device.idct4x4, 1, sizeof(cl_mem), &device.predictors_Y);
		device.state_gpu = clSetKernelArg(device.idct4x4, 3, sizeof(cl_int), &video.wrk_width);
		plane = 0;
		device.state_gpu = clSetKernelArg(device.idct4x4, 7, sizeof(cl_int), &plane);
		device.gpu_work_items_per_dim[0] = video.mb_count*16;
		device.state_gpu = clEnqueueNDRangeKernel(device.commandQueue1_gpu, device.idct4x4, 1, NULL, device.gpu_work_items_per_dim, NULL, 0, NULL, NULL);
		device.state_gpu = clFlush(device.commandQueue1_gpu);
		//idct U
		device.state_gpu = clSetKernelArg(device.idct4x4, 0, sizeof(cl_mem), &device.reconstructed_frame_U);
		device.state_gpu = clSetKernelArg(device.idct4x4, 1, sizeof(cl_mem), &device.predictors_U);
		device.state_gpu = clSetKernelArg(device.idct4x4, 3, sizeof(cl_int), &cwidth);
		plane = 1;
		device.state_gpu = clSetKernelArg(device.idct4x4, 7, sizeof(cl_int), &plane);
		device.gpu_work_items_per_dim[0] = video.mb_count*4;
		device.state_gpu = clEnqueueNDRangeKernel(device.commandQueue2_gpu, device.idct4x4, 1, NULL, device.gpu_work_items_per_dim, NULL, 0, NULL, NULL);
		device.state_gpu = clFlush(device.commandQueue2_gpu);
		//idct V
		device.state_gpu = clSetKernelArg(device.idct4x4, 0, sizeof(cl_mem), &device.reconstructed_frame_V);
		device.state_gpu = clSetKernelArg(device.idct4x4, 1, sizeof(cl_mem), &device.predictors_V);
		plane = 2;
		device.state_gpu = clSetKernelArg(device.idct4x4, 7, sizeof(cl_int), &plane);
		device.state_gpu = clEnqueueNDRangeKernel(device.commandQueue3_gpu, device.idct4x4, 1, NULL, device.gpu_work_items_per_dim, NULL, 0, NULL, NULL);
		device.state_gpu = clFlush(device.commandQueue3_gpu);

		device.state_gpu = clFinish(device.commandQueue1_gpu);
		device.state_gpu = clFinish(device.commandQueue2_gpu);
		device.state_gpu = clFinish(device.commandQueue3_gpu);

		//count SSIM
		device.gpu_work_items_per_dim[0] = video.mb_count;
		//U
		cl_int reset = 1;
		device.state_gpu = clSetKernelArg(device.count_SSIM_chroma, 0, sizeof(cl_mem), &device.current_frame_U);
		device.state_gpu = clSetKernelArg(device.count_SSIM_chroma, 1, sizeof(cl_mem), &device.reconstructed_frame_U);
		device.state_gpu = clSetKernelArg(device.count_SSIM_chroma, 5, sizeof(cl_int), &reset);
		device.state_gpu = clEnqueueNDRangeKernel(device.commandQueue1_gpu, device.count_SSIM_chroma, 1, NULL, device.gpu_work_items_per_dim, NULL, 0, NULL, NULL);
		//V
		reset = 0;
		device.state_gpu = clSetKernelArg(device.count_SSIM_chroma, 0, sizeof(cl_mem), &device.current_frame_V);
		device.state_gpu = clSetKernelArg(device.count_SSIM_chroma, 1, sizeof(cl_mem), &device.reconstructed_frame_V);
		device.state_gpu = clSetKernelArg(device.count_SSIM_chroma, 5, sizeof(cl_int), &reset);
		device.state_gpu = clEnqueueNDRangeKernel(device.commandQueue1_gpu, device.count_SSIM_chroma, 1, NULL, device.gpu_work_items_per_dim, NULL, 0, NULL, NULL);
		//Y
		device.state_gpu = clEnqueueNDRangeKernel(device.commandQueue1_gpu, device.count_SSIM_luma, 1, NULL, device.gpu_work_items_per_dim, NULL, 0, NULL, NULL);
	}

	device.state_gpu = clFinish(device.commandQueue1_gpu);
	device.state_gpu = clFinish(device.commandQueue2_gpu);
	device.state_gpu = clFinish(device.commandQueue3_gpu);

	if (device.state_gpu != 0)
		printf("bad kernel %d",device.state_gpu);

    return;
}
