void prepare_GPU_buffers()
{
	int32_t width, height;
	
	// first reset vector nets to zeros
	device.state_gpu = clFinish(device.commandQueue_gpu);
	device.gpu_work_items_per_dim[0] = video.mb_count*4;
	device.state_gpu = clEnqueueNDRangeKernel(device.commandQueue_gpu, device.reset_vectors, 1, NULL, device.gpu_work_items_per_dim, NULL, 0, NULL, NULL);
	device.state_gpu = clFinish(device.commandQueue_gpu);

	// now prepare downsampled LAST buffers
	//prepare downsampled by 2
	device.state_gpu = clSetKernelArg(device.downsample, 0, sizeof(cl_mem), &device.reconstructed_frame_Y);
	device.state_gpu = clSetKernelArg(device.downsample, 1, sizeof(cl_mem), &device.last_frame_Y_downsampled_by2);
	device.state_gpu = clSetKernelArg(device.downsample, 2, sizeof(int32_t), &video.wrk_width);
	device.state_gpu = clSetKernelArg(device.downsample, 3, sizeof(int32_t), &video.wrk_height);
	device.gpu_work_items_per_dim[0] = video.wrk_width*video.wrk_height/4;
	device.state_gpu = clEnqueueNDRangeKernel(device.commandQueue_gpu, device.downsample, 1, NULL, device.gpu_work_items_per_dim, NULL, 0, NULL, NULL);
	device.state_gpu = clFlush(device.commandQueue_gpu);
	device.state_gpu = clSetKernelArg(device.downsample, 0, sizeof(cl_mem), &device.current_frame_Y);
	device.state_gpu = clSetKernelArg(device.downsample, 1, sizeof(cl_mem), &device.current_frame_Y_downsampled_by2);
	device.state_gpu = clEnqueueNDRangeKernel(device.commandQueue_gpu, device.downsample, 1, NULL, device.gpu_work_items_per_dim, NULL, 0, NULL, NULL);
	device.state_gpu = clFinish(device.commandQueue_gpu);
	//prepare downsampled by 4
	device.state_gpu = clSetKernelArg(device.downsample, 0, sizeof(cl_mem), &device.last_frame_Y_downsampled_by2);
	device.state_gpu = clSetKernelArg(device.downsample, 1, sizeof(cl_mem), &device.last_frame_Y_downsampled_by4);
	width = video.wrk_width/2;
	device.state_gpu = clSetKernelArg(device.downsample, 2, sizeof(int32_t), &width);
	height = video.wrk_height/2;
	device.state_gpu = clSetKernelArg(device.downsample, 3, sizeof(int32_t), &height);
	device.gpu_work_items_per_dim[0] = width*height/4;
	device.state_gpu = clEnqueueNDRangeKernel(device.commandQueue_gpu, device.downsample, 1, NULL, device.gpu_work_items_per_dim, NULL, 0, NULL, NULL);
	device.state_gpu = clFlush(device.commandQueue_gpu);
	device.state_gpu = clSetKernelArg(device.downsample, 0, sizeof(cl_mem), &device.current_frame_Y_downsampled_by2);
	device.state_gpu = clSetKernelArg(device.downsample, 1, sizeof(cl_mem), &device.current_frame_Y_downsampled_by4);
	device.state_gpu = clEnqueueNDRangeKernel(device.commandQueue_gpu, device.downsample, 1, NULL, device.gpu_work_items_per_dim, NULL, 0, NULL, NULL);
	device.state_gpu = clFinish(device.commandQueue_gpu);
	//prepare downsampled by 8
	device.state_gpu = clSetKernelArg(device.downsample, 0, sizeof(cl_mem), &device.last_frame_Y_downsampled_by4);
	device.state_gpu = clSetKernelArg(device.downsample, 1, sizeof(cl_mem), &device.last_frame_Y_downsampled_by8);
	width = video.wrk_width/4;
	device.state_gpu = clSetKernelArg(device.downsample, 2, sizeof(int32_t), &width);
	height = video.wrk_height/4;
	device.state_gpu = clSetKernelArg(device.downsample, 3, sizeof(int32_t), &height);
	device.gpu_work_items_per_dim[0] = width*height/4;
	device.state_gpu = clEnqueueNDRangeKernel(device.commandQueue_gpu, device.downsample, 1, NULL, device.gpu_work_items_per_dim, NULL, 0, NULL, NULL);
	device.state_gpu = clFlush(device.commandQueue_gpu);
	device.state_gpu = clSetKernelArg(device.downsample, 0, sizeof(cl_mem), &device.current_frame_Y_downsampled_by4);
	device.state_gpu = clSetKernelArg(device.downsample, 1, sizeof(cl_mem), &device.current_frame_Y_downsampled_by8);
	device.state_gpu = clEnqueueNDRangeKernel(device.commandQueue_gpu, device.downsample, 1, NULL, device.gpu_work_items_per_dim, NULL, 0, NULL, NULL);
	device.state_gpu = clFinish(device.commandQueue_gpu);
	//prepare downsampled by 16
	device.state_gpu = clSetKernelArg(device.downsample, 0, sizeof(cl_mem), &device.last_frame_Y_downsampled_by8);
	device.state_gpu = clSetKernelArg(device.downsample, 1, sizeof(cl_mem), &device.last_frame_Y_downsampled_by16);
	width = video.wrk_width/8;
	device.state_gpu = clSetKernelArg(device.downsample, 2, sizeof(int32_t), &width);
	height = video.wrk_height/8;
	device.state_gpu = clSetKernelArg(device.downsample, 3, sizeof(int32_t), &height);
	device.gpu_work_items_per_dim[0] = width*height/4;
	device.state_gpu = clEnqueueNDRangeKernel(device.commandQueue_gpu, device.downsample, 1, NULL, device.gpu_work_items_per_dim, NULL, 0, NULL, NULL);
	device.state_gpu = clFlush(device.commandQueue_gpu);
	device.state_gpu = clSetKernelArg(device.downsample, 0, sizeof(cl_mem), &device.current_frame_Y_downsampled_by8);
	device.state_gpu = clSetKernelArg(device.downsample, 1, sizeof(cl_mem), &device.current_frame_Y_downsampled_by16);
	device.state_gpu = clEnqueueNDRangeKernel(device.commandQueue_gpu, device.downsample, 1, NULL, device.gpu_work_items_per_dim, NULL, 0, NULL, NULL);
	device.state_gpu = clFinish(device.commandQueue_gpu);
	
	if (frames.prev_is_golden_frame)
	{
		device.state_gpu = clEnqueueCopyBuffer(device.commandQueue_gpu, device.reconstructed_frame_Y, device.golden_frame_Y, 0, 0, video.wrk_frame_size_luma, 0, NULL, NULL);
		device.state_gpu = clEnqueueCopyBuffer(device.commandQueue_gpu, device.reconstructed_frame_U, device.golden_frame_U, 0, 0, video.wrk_frame_size_chroma, 0, NULL, NULL);
		device.state_gpu = clEnqueueCopyBuffer(device.commandQueue_gpu, device.reconstructed_frame_V, device.golden_frame_V, 0, 0, video.wrk_frame_size_chroma, 0, NULL, NULL);
		device.state_gpu = clEnqueueCopyBuffer(device.commandQueue_gpu, device.last_frame_Y_downsampled_by2, device.golden_frame_Y_downsampled_by2, 0, 0, video.wrk_frame_size_luma/4, 0, NULL, NULL);
		device.state_gpu = clEnqueueCopyBuffer(device.commandQueue_gpu, device.last_frame_Y_downsampled_by4, device.golden_frame_Y_downsampled_by4, 0, 0, video.wrk_frame_size_luma/16, 0, NULL, NULL);
		device.state_gpu = clEnqueueCopyBuffer(device.commandQueue_gpu, device.last_frame_Y_downsampled_by8, device.golden_frame_Y_downsampled_by8, 0, 0, video.wrk_frame_size_luma/64, 0, NULL, NULL);
		device.state_gpu = clEnqueueCopyBuffer(device.commandQueue_gpu, device.last_frame_Y_downsampled_by16, device.golden_frame_Y_downsampled_by16, 0, 0, video.wrk_frame_size_luma/256, 0, NULL, NULL);
	}
	if (frames.prev_is_altref_frame)
	{
		device.state_gpu = clEnqueueCopyBuffer(device.commandQueue_gpu, device.reconstructed_frame_Y, device.altref_frame_Y, 0, 0, video.wrk_frame_size_luma, 0, NULL, NULL);
		device.state_gpu = clEnqueueCopyBuffer(device.commandQueue_gpu, device.reconstructed_frame_U, device.altref_frame_U, 0, 0, video.wrk_frame_size_chroma, 0, NULL, NULL);
		device.state_gpu = clEnqueueCopyBuffer(device.commandQueue_gpu, device.reconstructed_frame_V, device.altref_frame_V, 0, 0, video.wrk_frame_size_chroma, 0, NULL, NULL);
		device.state_gpu = clEnqueueCopyBuffer(device.commandQueue_gpu, device.last_frame_Y_downsampled_by2, device.altref_frame_Y_downsampled_by2, 0, 0, video.wrk_frame_size_luma/4, 0, NULL, NULL);
		device.state_gpu = clEnqueueCopyBuffer(device.commandQueue_gpu, device.last_frame_Y_downsampled_by4, device.altref_frame_Y_downsampled_by4, 0, 0, video.wrk_frame_size_luma/16, 0, NULL, NULL);
		device.state_gpu = clEnqueueCopyBuffer(device.commandQueue_gpu, device.last_frame_Y_downsampled_by8, device.altref_frame_Y_downsampled_by8, 0, 0, video.wrk_frame_size_luma/64, 0, NULL, NULL);
		device.state_gpu = clEnqueueCopyBuffer(device.commandQueue_gpu, device.last_frame_Y_downsampled_by16, device.altref_frame_Y_downsampled_by16, 0, 0, video.wrk_frame_size_luma/256, 0, NULL, NULL);
	}
	device.state_gpu = clFinish(device.commandQueue_gpu);

	// prepare images (if they need to be renewed)
	const size_t origin[3] = {0, 0, 0};
	const size_t region_y[3] = {video.wrk_width, video.wrk_height, 1};
	const size_t region_uv[3] = {video.wrk_width/2, video.wrk_height/2, 1};

	device.state_gpu = clEnqueueReadBuffer(device.commandQueue_gpu, device.reconstructed_frame_Y ,CL_TRUE, 0, video.wrk_frame_size_luma, frames.reconstructed_Y, 0, NULL, NULL);
	device.state_gpu = clEnqueueReadBuffer(device.commandQueue_gpu, device.reconstructed_frame_U ,CL_TRUE, 0, video.wrk_frame_size_chroma, frames.reconstructed_U, 0, NULL, NULL);
	device.state_gpu = clEnqueueReadBuffer(device.commandQueue_gpu, device.reconstructed_frame_V ,CL_TRUE, 0, video.wrk_frame_size_chroma, frames.reconstructed_V, 0, NULL, NULL);

	device.state_gpu = clEnqueueWriteImage(device.commandQueue_gpu, device.last_frame_Y_image, CL_TRUE, origin, region_y, 0, 0, frames.reconstructed_Y, 0, NULL, NULL);
	device.state_gpu = clEnqueueWriteImage(device.commandQueue_gpu, device.last_frame_U_image, CL_TRUE, origin, region_uv, 0, 0, frames.reconstructed_U, 0, NULL, NULL);
	device.state_gpu = clEnqueueWriteImage(device.commandQueue_gpu, device.last_frame_V_image, CL_TRUE, origin, region_uv, 0, 0, frames.reconstructed_V, 0, NULL, NULL);

	if (frames.prev_is_golden_frame)
	{
		device.state_gpu = clEnqueueCopyImage(device.commandQueue_gpu, device.last_frame_Y_image, device.golden_frame_Y_image, origin, origin, region_y, 0, NULL, NULL);
		device.state_gpu = clEnqueueCopyImage(device.commandQueue_gpu, device.last_frame_U_image, device.golden_frame_U_image, origin, origin, region_uv, 0, NULL, NULL);
		device.state_gpu = clEnqueueCopyImage(device.commandQueue_gpu, device.last_frame_V_image, device.golden_frame_V_image, origin, origin, region_uv, 0, NULL, NULL);
	}
	if (frames.prev_is_altref_frame)
	{
		device.state_gpu = clEnqueueCopyImage(device.commandQueue_gpu, device.last_frame_Y_image, device.altref_frame_Y_image, origin, origin, region_y, 0, NULL, NULL);
		device.state_gpu = clEnqueueCopyImage(device.commandQueue_gpu, device.last_frame_U_image, device.altref_frame_U_image, origin, origin, region_uv, 0, NULL, NULL);
		device.state_gpu = clEnqueueCopyImage(device.commandQueue_gpu, device.last_frame_V_image, device.altref_frame_V_image, origin, origin, region_uv, 0, NULL, NULL);
	}
	device.state_gpu = clFinish(device.commandQueue_gpu);
	
	return;
}

void inter_transform()
{
	clFinish(device.commandQueue_gpu);

	int32_t val;
	// if golden and altref buffers represent different from last buffer frame
	// and altref is not the same as altref
	int32_t use_golden = !frames.prev_is_golden_frame;
	int32_t use_altref = (!frames.prev_is_altref_frame) && (frames.altref_frame_number != frames.golden_frame_number);
	//prepare downsampled frames and image objects
	prepare_GPU_buffers();
	
	//now search in downsampled by 16
	device.gpu_work_items_per_dim[0] = ((video.wrk_width/16)/8)*((video.wrk_height/16)/8);
	device.state_gpu = clSetKernelArg(device.luma_search_1step, 0, sizeof(cl_mem), &device.current_frame_Y_downsampled_by16);
	val = video.wrk_width/16;
	device.state_gpu = clSetKernelArg(device.luma_search_1step, 5, sizeof(int32_t), &val);
	val = video.wrk_height/16;
	device.state_gpu = clSetKernelArg(device.luma_search_1step, 6, sizeof(int32_t), &val);
	val = 16;
	device.state_gpu = clSetKernelArg(device.luma_search_1step, 7, sizeof(int32_t), &val);
	// LAST
	device.state_gpu = clSetKernelArg(device.luma_search_1step, 1, sizeof(cl_mem), &device.last_frame_Y_downsampled_by16);
	device.state_gpu = clSetKernelArg(device.luma_search_1step, 2, sizeof(cl_mem), &device.last_vnet1);
	device.state_gpu = clSetKernelArg(device.luma_search_1step, 3, sizeof(cl_mem), &device.last_vnet2);
	device.state_gpu = clEnqueueNDRangeKernel(device.commandQueue_gpu, device.luma_search_1step, 1, NULL, device.gpu_work_items_per_dim, NULL, 0, NULL, NULL);
	device.state_gpu = clFlush(device.commandQueue_gpu);
	// GOLDEN
	if (use_golden)
	{
		device.state_gpu = clSetKernelArg(device.luma_search_1step, 1, sizeof(cl_mem), &device.golden_frame_Y_downsampled_by16);
		device.state_gpu = clSetKernelArg(device.luma_search_1step, 2, sizeof(cl_mem), &device.golden_vnet1);
		device.state_gpu = clSetKernelArg(device.luma_search_1step, 3, sizeof(cl_mem), &device.golden_vnet2);
		device.state_gpu = clEnqueueNDRangeKernel(device.commandQueue_gpu, device.luma_search_1step, 1, NULL, device.gpu_work_items_per_dim, NULL, 0, NULL, NULL);
		device.state_gpu = clFlush(device.commandQueue_gpu);
	}
	// ALTREF
	if (use_altref)
	{
		device.state_gpu = clSetKernelArg(device.luma_search_1step, 1, sizeof(cl_mem), &device.altref_frame_Y_downsampled_by16);
		device.state_gpu = clSetKernelArg(device.luma_search_1step, 2, sizeof(cl_mem), &device.altref_vnet1);
		device.state_gpu = clSetKernelArg(device.luma_search_1step, 3, sizeof(cl_mem), &device.altref_vnet2);
		device.state_gpu = clEnqueueNDRangeKernel(device.commandQueue_gpu, device.luma_search_1step, 1, NULL, device.gpu_work_items_per_dim, NULL, 0, NULL, NULL);
	}
	device.state_gpu = clFinish(device.commandQueue_gpu);

	//now search in downsampled by 8
	device.gpu_work_items_per_dim[0] = ((video.wrk_width/8)/8)*((video.wrk_height/8)/8);
	device.state_gpu = clSetKernelArg(device.luma_search_1step, 0, sizeof(cl_mem), &device.current_frame_Y_downsampled_by8);
	val = video.wrk_width/8;
	device.state_gpu = clSetKernelArg(device.luma_search_1step, 5, sizeof(int32_t), &val);
	val = video.wrk_height/8;
	device.state_gpu = clSetKernelArg(device.luma_search_1step, 6, sizeof(int32_t), &val);
	val = 8;
	device.state_gpu = clSetKernelArg(device.luma_search_1step, 7, sizeof(int32_t), &val);
	// LAST
	device.state_gpu = clSetKernelArg(device.luma_search_1step, 1, sizeof(cl_mem), &device.last_frame_Y_downsampled_by8);
	device.state_gpu = clSetKernelArg(device.luma_search_1step, 2, sizeof(cl_mem), &device.last_vnet2);
	device.state_gpu = clSetKernelArg(device.luma_search_1step, 3, sizeof(cl_mem), &device.last_vnet1);
	device.state_gpu = clEnqueueNDRangeKernel(device.commandQueue_gpu, device.luma_search_1step, 1, NULL, device.gpu_work_items_per_dim, NULL, 0, NULL, NULL);
	device.state_gpu = clFlush(device.commandQueue_gpu);
	// GOLDEN
	if (use_golden)
	{
		device.state_gpu = clSetKernelArg(device.luma_search_1step, 1, sizeof(cl_mem), &device.golden_frame_Y_downsampled_by8);
		device.state_gpu = clSetKernelArg(device.luma_search_1step, 2, sizeof(cl_mem), &device.golden_vnet2);
		device.state_gpu = clSetKernelArg(device.luma_search_1step, 3, sizeof(cl_mem), &device.golden_vnet1);
		device.state_gpu = clEnqueueNDRangeKernel(device.commandQueue_gpu, device.luma_search_1step, 1, NULL, device.gpu_work_items_per_dim, NULL, 0, NULL, NULL);
		device.state_gpu = clFlush(device.commandQueue_gpu);
	}
	// ALTREF
	if (use_altref)
	{
		device.state_gpu = clSetKernelArg(device.luma_search_1step, 1, sizeof(cl_mem), &device.altref_frame_Y_downsampled_by8);
		device.state_gpu = clSetKernelArg(device.luma_search_1step, 2, sizeof(cl_mem), &device.altref_vnet2);
		device.state_gpu = clSetKernelArg(device.luma_search_1step, 3, sizeof(cl_mem), &device.altref_vnet1);
		device.state_gpu = clEnqueueNDRangeKernel(device.commandQueue_gpu, device.luma_search_1step, 1, NULL, device.gpu_work_items_per_dim, NULL, 0, NULL, NULL);
	}
	device.state_gpu = clFinish(device.commandQueue_gpu);
	

	//now search in downsampled by 4
	device.gpu_work_items_per_dim[0] = ((video.wrk_width/4)/8)*((video.wrk_height/4)/8);
	device.state_gpu = clSetKernelArg(device.luma_search_1step, 0, sizeof(cl_mem), &device.current_frame_Y_downsampled_by4);
	val = video.wrk_width/4;
	device.state_gpu = clSetKernelArg(device.luma_search_1step, 5, sizeof(int32_t), &val);
	val = video.wrk_height/4;
	device.state_gpu = clSetKernelArg(device.luma_search_1step, 6, sizeof(int32_t), &val);
	val = 4;
	device.state_gpu = clSetKernelArg(device.luma_search_1step, 7, sizeof(int32_t), &val);
	// LAST
	device.state_gpu = clSetKernelArg(device.luma_search_1step, 1, sizeof(cl_mem), &device.last_frame_Y_downsampled_by4);
	device.state_gpu = clSetKernelArg(device.luma_search_1step, 2, sizeof(cl_mem), &device.last_vnet1);
	device.state_gpu = clSetKernelArg(device.luma_search_1step, 3, sizeof(cl_mem), &device.last_vnet2);
	device.state_gpu = clEnqueueNDRangeKernel(device.commandQueue_gpu, device.luma_search_1step, 1, NULL, device.gpu_work_items_per_dim, NULL, 0, NULL, NULL);
	device.state_gpu = clFlush(device.commandQueue_gpu);
	// GOLDEN
	if (use_golden)
	{
		device.state_gpu = clSetKernelArg(device.luma_search_1step, 1, sizeof(cl_mem), &device.golden_frame_Y_downsampled_by4);
		device.state_gpu = clSetKernelArg(device.luma_search_1step, 2, sizeof(cl_mem), &device.golden_vnet1);
		device.state_gpu = clSetKernelArg(device.luma_search_1step, 3, sizeof(cl_mem), &device.golden_vnet2);
		device.state_gpu = clEnqueueNDRangeKernel(device.commandQueue_gpu, device.luma_search_1step, 1, NULL, device.gpu_work_items_per_dim, NULL, 0, NULL, NULL);
		device.state_gpu = clFlush(device.commandQueue_gpu);
	}
	// ALTREF
	if (use_altref)
	{
		device.state_gpu = clSetKernelArg(device.luma_search_1step, 1, sizeof(cl_mem), &device.altref_frame_Y_downsampled_by4);
		device.state_gpu = clSetKernelArg(device.luma_search_1step, 2, sizeof(cl_mem), &device.altref_vnet1);
		device.state_gpu = clSetKernelArg(device.luma_search_1step, 3, sizeof(cl_mem), &device.altref_vnet2);
		device.state_gpu = clEnqueueNDRangeKernel(device.commandQueue_gpu, device.luma_search_1step, 1, NULL, device.gpu_work_items_per_dim, NULL, 0, NULL, NULL);
	}
	device.state_gpu = clFinish(device.commandQueue_gpu);

	//now search in downsampled by 2
	device.gpu_work_items_per_dim[0] = ((video.wrk_width/2)/8)*((video.wrk_height/2)/8);
	device.state_gpu = clSetKernelArg(device.luma_search_1step, 0, sizeof(cl_mem), &device.current_frame_Y_downsampled_by2);
	val = video.wrk_width/2;
	device.state_gpu = clSetKernelArg(device.luma_search_1step, 5, sizeof(int32_t), &val);
	val = video.wrk_height/2;
	device.state_gpu = clSetKernelArg(device.luma_search_1step, 6, sizeof(int32_t), &val);
	val = 2;
	device.state_gpu = clSetKernelArg(device.luma_search_1step, 7, sizeof(int32_t), &val);
	// LAST
	device.state_gpu = clSetKernelArg(device.luma_search_1step, 1, sizeof(cl_mem), &device.last_frame_Y_downsampled_by2);
	device.state_gpu = clSetKernelArg(device.luma_search_1step, 2, sizeof(cl_mem), &device.last_vnet2);
	device.state_gpu = clSetKernelArg(device.luma_search_1step, 3, sizeof(cl_mem), &device.last_vnet1);
	device.state_gpu = clEnqueueNDRangeKernel(device.commandQueue_gpu, device.luma_search_1step, 1, NULL, device.gpu_work_items_per_dim, NULL, 0, NULL, NULL);
	device.state_gpu = clFlush(device.commandQueue_gpu);
	// GOLDEN
	if (use_golden)
	{
		device.state_gpu = clSetKernelArg(device.luma_search_1step, 1, sizeof(cl_mem), &device.golden_frame_Y_downsampled_by2);
		device.state_gpu = clSetKernelArg(device.luma_search_1step, 2, sizeof(cl_mem), &device.golden_vnet2);
		device.state_gpu = clSetKernelArg(device.luma_search_1step, 3, sizeof(cl_mem), &device.golden_vnet1);
		device.state_gpu = clEnqueueNDRangeKernel(device.commandQueue_gpu, device.luma_search_1step, 1, NULL, device.gpu_work_items_per_dim, NULL, 0, NULL, NULL);
		device.state_gpu = clFlush(device.commandQueue_gpu);
	}
	// ALTREF
	if (use_altref)
	{
		device.state_gpu = clSetKernelArg(device.luma_search_1step, 1, sizeof(cl_mem), &device.altref_frame_Y_downsampled_by2);
		device.state_gpu = clSetKernelArg(device.luma_search_1step, 2, sizeof(cl_mem), &device.altref_vnet2);
		device.state_gpu = clSetKernelArg(device.luma_search_1step, 3, sizeof(cl_mem), &device.altref_vnet1);
		device.state_gpu = clEnqueueNDRangeKernel(device.commandQueue_gpu, device.luma_search_1step, 1, NULL, device.gpu_work_items_per_dim, NULL, 0, NULL, NULL);
	}
	device.state_gpu = clFinish(device.commandQueue_gpu);

	//now search in original size
	device.gpu_work_items_per_dim[0] = video.mb_count*4; 
	device.state_gpu = clSetKernelArg(device.luma_search_1step, 0, sizeof(cl_mem), &device.current_frame_Y);
	val = video.wrk_width;
	device.state_gpu = clSetKernelArg(device.luma_search_1step, 5, sizeof(int32_t), &val);
	val = video.wrk_height;
	device.state_gpu = clSetKernelArg(device.luma_search_1step, 6, sizeof(int32_t), &val);
	val = 1;
	device.state_gpu = clSetKernelArg(device.luma_search_1step, 7, sizeof(int32_t), &val);
	// LAST
	device.state_gpu = clSetKernelArg(device.luma_search_1step, 1, sizeof(cl_mem), &device.reconstructed_frame_Y);
	device.state_gpu = clSetKernelArg(device.luma_search_1step, 2, sizeof(cl_mem), &device.last_vnet1);
	device.state_gpu = clSetKernelArg(device.luma_search_1step, 3, sizeof(cl_mem), &device.last_vnet2);
	device.state_gpu = clEnqueueNDRangeKernel(device.commandQueue_gpu, device.luma_search_1step, 1, NULL, device.gpu_work_items_per_dim, NULL, 0, NULL, NULL);
	device.state_gpu = clFlush(device.commandQueue_gpu);
	// GOLDEN
	if (use_golden)
	{
		device.state_gpu = clSetKernelArg(device.luma_search_1step, 1, sizeof(cl_mem), &device.golden_frame_Y);
		device.state_gpu = clSetKernelArg(device.luma_search_1step, 2, sizeof(cl_mem), &device.golden_vnet1);
		device.state_gpu = clSetKernelArg(device.luma_search_1step, 3, sizeof(cl_mem), &device.golden_vnet2);
		device.state_gpu = clEnqueueNDRangeKernel(device.commandQueue_gpu, device.luma_search_1step, 1, NULL, device.gpu_work_items_per_dim, NULL, 0, NULL, NULL);
		device.state_gpu = clFlush(device.commandQueue_gpu);
	}
	// ALTREF
	if (use_altref)
	{
		device.state_gpu = clSetKernelArg(device.luma_search_1step, 1, sizeof(cl_mem), &device.altref_frame_Y);
		device.state_gpu = clSetKernelArg(device.luma_search_1step, 2, sizeof(cl_mem), &device.altref_vnet1);
		device.state_gpu = clSetKernelArg(device.luma_search_1step, 3, sizeof(cl_mem), &device.altref_vnet2);
		device.state_gpu = clEnqueueNDRangeKernel(device.commandQueue_gpu, device.luma_search_1step, 1, NULL, device.gpu_work_items_per_dim, NULL, 0, NULL, NULL);
	}
	device.state_gpu = clFinish(device.commandQueue_gpu);

	// search in image with interpolation on the run
	device.gpu_work_items_per_dim[0] = video.mb_count*4;
	// LAST
	device.state_gpu = clSetKernelArg(device.luma_search_2step, 1, sizeof(cl_mem), &device.last_frame_Y_image);
	device.state_gpu = clSetKernelArg(device.luma_search_2step, 2, sizeof(cl_mem), &device.last_vnet2);
	device.state_gpu = clSetKernelArg(device.luma_search_2step, 3, sizeof(cl_mem), &device.last_vnet1);
	device.state_gpu = clSetKernelArg(device.luma_search_2step, 4, sizeof(cl_mem), &device.last_metrics);
	device.state_gpu = clEnqueueNDRangeKernel(device.commandQueue_gpu, device.luma_search_2step, 1, NULL, device.gpu_work_items_per_dim, NULL, 0, NULL, NULL);
	device.state_gpu = clFlush(device.commandQueue_gpu);
	// GOLDEN
	if (use_golden)
	{
		device.state_gpu = clSetKernelArg(device.luma_search_2step, 1, sizeof(cl_mem), &device.golden_frame_Y_image);
		device.state_gpu = clSetKernelArg(device.luma_search_2step, 2, sizeof(cl_mem), &device.golden_vnet2);
		device.state_gpu = clSetKernelArg(device.luma_search_2step, 3, sizeof(cl_mem), &device.golden_vnet1);
		device.state_gpu = clSetKernelArg(device.luma_search_2step, 4, sizeof(cl_mem), &device.golden_metrics);
		device.state_gpu = clEnqueueNDRangeKernel(device.commandQueue_gpu, device.luma_search_2step, 1, NULL, device.gpu_work_items_per_dim, NULL, 0, NULL, NULL);
		device.state_gpu = clFlush(device.commandQueue_gpu);
	}
	// ALTREF
	if (use_altref)
	{
		device.state_gpu = clSetKernelArg(device.luma_search_2step, 1, sizeof(cl_mem), &device.altref_frame_Y_image);
		device.state_gpu = clSetKernelArg(device.luma_search_2step, 2, sizeof(cl_mem), &device.altref_vnet2);
		device.state_gpu = clSetKernelArg(device.luma_search_2step, 3, sizeof(cl_mem), &device.altref_vnet1);
		device.state_gpu = clSetKernelArg(device.luma_search_2step, 4, sizeof(cl_mem), &device.altref_metrics);
		device.state_gpu = clEnqueueNDRangeKernel(device.commandQueue_gpu, device.luma_search_2step, 1, NULL, device.gpu_work_items_per_dim, NULL, 0, NULL, NULL);
	}
	device.state_gpu = clFinish(device.commandQueue_gpu);

	// now set each MB with the best reference
	device.gpu_work_items_per_dim[0] = video.mb_count;
	device.state_gpu = clSetKernelArg(device.select_reference, 8, sizeof(int32_t), &use_golden);
	device.state_gpu = clSetKernelArg(device.select_reference, 9, sizeof(int32_t), &use_altref);
	device.state_gpu = clEnqueueNDRangeKernel(device.commandQueue_gpu, device.select_reference, 1, NULL, device.gpu_work_items_per_dim, NULL, 0, NULL, NULL);
	device.state_gpu = clFinish(device.commandQueue_gpu);

	// set 16x16 mode for macroblocks, whose blocks have identical vectors
	device.gpu_work_items_per_dim[0] = video.mb_count;
	device.state_gpu = clEnqueueNDRangeKernel(device.commandQueue_gpu, device.pack_8x8_into_16x16, 1, NULL, device.gpu_work_items_per_dim, NULL, 0, NULL, NULL);
	device.state_gpu = clFinish(device.commandQueue_gpu);

	// now for each plane and reference frame fill predictors and residual buffers
	int32_t plane, ref;
	int32_t cwidth = video.wrk_width/2;
	// Y
	plane = 0;
	device.state_gpu = clSetKernelArg(device.prepare_predictors_and_residual, 0, sizeof(cl_mem), &device.current_frame_Y);
	device.state_gpu = clSetKernelArg(device.prepare_predictors_and_residual, 2, sizeof(cl_mem), &device.predictors_Y);
	device.state_gpu = clSetKernelArg(device.prepare_predictors_and_residual, 3, sizeof(cl_mem), &device.residual_Y);
	device.state_gpu = clSetKernelArg(device.prepare_predictors_and_residual, 5, sizeof(int32_t), &video.wrk_width);
	device.state_gpu = clSetKernelArg(device.prepare_predictors_and_residual, 6, sizeof(int32_t), &plane);
	device.gpu_work_items_per_dim[0] = video.mb_count*16;
	// LAST
	ref = LAST;
	device.state_gpu = clSetKernelArg(device.prepare_predictors_and_residual, 1, sizeof(cl_mem), &device.last_frame_Y_image);
	device.state_gpu = clSetKernelArg(device.prepare_predictors_and_residual, 7, sizeof(int32_t), &ref);
	device.state_gpu = clEnqueueNDRangeKernel(device.commandQueue_gpu, device.prepare_predictors_and_residual, 1, NULL, device.gpu_work_items_per_dim, NULL, 0, NULL, NULL);
	device.state_gpu = clFlush(device.commandQueue_gpu);
	// GOLDEN
	if (use_golden)
	{
		ref = GOLDEN;
		device.state_gpu = clSetKernelArg(device.prepare_predictors_and_residual, 1, sizeof(cl_mem), &device.golden_frame_Y_image);
		device.state_gpu = clSetKernelArg(device.prepare_predictors_and_residual, 7, sizeof(int32_t), &ref);
		device.state_gpu = clEnqueueNDRangeKernel(device.commandQueue_gpu, device.prepare_predictors_and_residual, 1, NULL, device.gpu_work_items_per_dim, NULL, 0, NULL, NULL);
		device.state_gpu = clFlush(device.commandQueue_gpu);
	}
	// ALTREF
	if (use_altref)
	{
		ref = ALTREF;
		device.state_gpu = clSetKernelArg(device.prepare_predictors_and_residual, 1, sizeof(cl_mem), &device.altref_frame_Y_image);
		device.state_gpu = clSetKernelArg(device.prepare_predictors_and_residual, 7, sizeof(int32_t), &ref);
		device.state_gpu = clEnqueueNDRangeKernel(device.commandQueue_gpu, device.prepare_predictors_and_residual, 1, NULL, device.gpu_work_items_per_dim, NULL, 0, NULL, NULL);
		device.state_gpu = clFlush(device.commandQueue_gpu);
	}
	// U
	plane = 1;
	device.state_gpu = clSetKernelArg(device.prepare_predictors_and_residual, 0, sizeof(cl_mem), &device.current_frame_U);
	device.state_gpu = clSetKernelArg(device.prepare_predictors_and_residual, 2, sizeof(cl_mem), &device.predictors_U);
	device.state_gpu = clSetKernelArg(device.prepare_predictors_and_residual, 3, sizeof(cl_mem), &device.residual_U);
	device.state_gpu = clSetKernelArg(device.prepare_predictors_and_residual, 5, sizeof(int32_t), &cwidth);
	device.state_gpu = clSetKernelArg(device.prepare_predictors_and_residual, 6, sizeof(int32_t), &plane);
	device.gpu_work_items_per_dim[0] = video.mb_count*4;
	// LAST
	ref = LAST;
	device.state_gpu = clSetKernelArg(device.prepare_predictors_and_residual, 1, sizeof(cl_mem), &device.last_frame_U_image);
	device.state_gpu = clSetKernelArg(device.prepare_predictors_and_residual, 7, sizeof(int32_t), &ref);
	device.state_gpu = clEnqueueNDRangeKernel(device.commandQueue_gpu, device.prepare_predictors_and_residual, 1, NULL, device.gpu_work_items_per_dim, NULL, 0, NULL, NULL);
	device.state_gpu = clFlush(device.commandQueue_gpu);
	// GOLDEN
	if (use_golden)
	{
		ref = GOLDEN;
		device.state_gpu = clSetKernelArg(device.prepare_predictors_and_residual, 1, sizeof(cl_mem), &device.golden_frame_U_image);
		device.state_gpu = clSetKernelArg(device.prepare_predictors_and_residual, 7, sizeof(int32_t), &ref);
		device.state_gpu = clEnqueueNDRangeKernel(device.commandQueue_gpu, device.prepare_predictors_and_residual, 1, NULL, device.gpu_work_items_per_dim, NULL, 0, NULL, NULL);
		device.state_gpu = clFlush(device.commandQueue_gpu);
	}
	// ALTREF
	if (use_altref)
	{
		ref = ALTREF;
		device.state_gpu = clSetKernelArg(device.prepare_predictors_and_residual, 1, sizeof(cl_mem), &device.altref_frame_U_image);
		device.state_gpu = clSetKernelArg(device.prepare_predictors_and_residual, 7, sizeof(int32_t), &ref);
		device.state_gpu = clEnqueueNDRangeKernel(device.commandQueue_gpu, device.prepare_predictors_and_residual, 1, NULL, device.gpu_work_items_per_dim, NULL, 0, NULL, NULL);
		device.state_gpu = clFlush(device.commandQueue_gpu);
	}
	// V
	plane = 2;
	device.state_gpu = clSetKernelArg(device.prepare_predictors_and_residual, 0, sizeof(cl_mem), &device.current_frame_V);
	device.state_gpu = clSetKernelArg(device.prepare_predictors_and_residual, 2, sizeof(cl_mem), &device.predictors_V);
	device.state_gpu = clSetKernelArg(device.prepare_predictors_and_residual, 3, sizeof(cl_mem), &device.residual_V);
	device.state_gpu = clSetKernelArg(device.prepare_predictors_and_residual, 6, sizeof(int32_t), &plane);
	// LAST
	ref = LAST;
	device.state_gpu = clSetKernelArg(device.prepare_predictors_and_residual, 1, sizeof(cl_mem), &device.last_frame_V_image);
	device.state_gpu = clSetKernelArg(device.prepare_predictors_and_residual, 7, sizeof(int32_t), &ref);
	device.state_gpu = clEnqueueNDRangeKernel(device.commandQueue_gpu, device.prepare_predictors_and_residual, 1, NULL, device.gpu_work_items_per_dim, NULL, 0, NULL, NULL);
	device.state_gpu = clFlush(device.commandQueue_gpu);
	// GOLDEN
	if (use_golden)
	{
		ref = GOLDEN;
		device.state_gpu = clSetKernelArg(device.prepare_predictors_and_residual, 1, sizeof(cl_mem), &device.golden_frame_V_image);
		device.state_gpu = clSetKernelArg(device.prepare_predictors_and_residual, 7, sizeof(int32_t), &ref);
		device.state_gpu = clEnqueueNDRangeKernel(device.commandQueue_gpu, device.prepare_predictors_and_residual, 1, NULL, device.gpu_work_items_per_dim, NULL, 0, NULL, NULL);
		device.state_gpu = clFlush(device.commandQueue_gpu);
	}
	// ALTREF
	if (use_altref)
	{
		ref = ALTREF;
		device.state_gpu = clSetKernelArg(device.prepare_predictors_and_residual, 1, sizeof(cl_mem), &device.altref_frame_V_image);
		device.state_gpu = clSetKernelArg(device.prepare_predictors_and_residual, 7, sizeof(int32_t), &ref);
		device.state_gpu = clEnqueueNDRangeKernel(device.commandQueue_gpu, device.prepare_predictors_and_residual, 1, NULL, device.gpu_work_items_per_dim, NULL, 0, NULL, NULL);
	}
	device.state_gpu = clFinish(device.commandQueue_gpu);

	// now for each segment (begin with highest quantizer (last index))
	int32_t seg_id;
	for (seg_id = LQ_segment; seg_id >= UQ_segment; --seg_id)
	{
		device.state_gpu = clSetKernelArg(device.dct4x4, 4, sizeof(int32_t), &seg_id);
		device.state_gpu = clSetKernelArg(device.wht4x4_iwht4x4, 2, sizeof(int32_t), &seg_id);
		device.state_gpu = clSetKernelArg(device.idct4x4, 5, sizeof(int32_t), &seg_id);
		device.state_gpu = clSetKernelArg(device.count_SSIM_chroma, 4, sizeof(int32_t), &seg_id);
		device.state_gpu = clSetKernelArg(device.count_SSIM_luma, 4, sizeof(int32_t), &seg_id);
		//dct Y
		device.state_gpu = clSetKernelArg(device.dct4x4, 0, sizeof(cl_mem), &device.residual_Y);
		device.state_gpu = clSetKernelArg(device.dct4x4, 2, sizeof(int32_t), &video.wrk_width);
		plane = 0;
		device.state_gpu = clSetKernelArg(device.dct4x4, 6, sizeof(int32_t), &plane);
		device.gpu_work_items_per_dim[0] = video.mb_count*16;
		device.state_gpu = clEnqueueNDRangeKernel(device.commandQueue_gpu, device.dct4x4, 1, NULL, device.gpu_work_items_per_dim, NULL, 0, NULL, NULL);
		device.state_gpu = clFlush(device.commandQueue_gpu);	
		//dct U
		device.state_gpu = clSetKernelArg(device.dct4x4, 0, sizeof(cl_mem), &device.residual_U);
		device.state_gpu = clSetKernelArg(device.dct4x4, 2, sizeof(int32_t), &cwidth);
		plane = 1;
		device.state_gpu = clSetKernelArg(device.dct4x4, 6, sizeof(int32_t), &plane);
		device.gpu_work_items_per_dim[0] = video.mb_count*4;
		device.state_gpu = clEnqueueNDRangeKernel(device.commandQueue_gpu, device.dct4x4, 1, NULL, device.gpu_work_items_per_dim, NULL, 0, NULL, NULL);
		//dct V
		device.state_gpu = clSetKernelArg(device.dct4x4, 0, sizeof(cl_mem), &device.residual_V);
		plane = 2;
		device.state_gpu = clSetKernelArg(device.dct4x4, 6, sizeof(int32_t), &plane);
		device.state_gpu = clEnqueueNDRangeKernel(device.commandQueue_gpu, device.dct4x4, 1, NULL, device.gpu_work_items_per_dim, NULL, 0, NULL, NULL);
		device.state_gpu = clFinish(device.commandQueue_gpu);
		//wht and iwht
		device.gpu_work_items_per_dim[0] = video.mb_count;
		device.state_gpu = clEnqueueNDRangeKernel(device.commandQueue_gpu, device.wht4x4_iwht4x4, 1, NULL, device.gpu_work_items_per_dim, NULL, 0, NULL, NULL);
		device.state_gpu = clFinish(device.commandQueue_gpu);
		//idct Y
		device.state_gpu = clSetKernelArg(device.idct4x4, 0, sizeof(cl_mem), &device.reconstructed_frame_Y);
		device.state_gpu = clSetKernelArg(device.idct4x4, 1, sizeof(cl_mem), &device.predictors_Y);
		device.state_gpu = clSetKernelArg(device.idct4x4, 3, sizeof(int32_t), &video.wrk_width);
		plane = 0;
		device.state_gpu = clSetKernelArg(device.idct4x4, 7, sizeof(int32_t), &plane);
		device.gpu_work_items_per_dim[0] = video.mb_count*16;
		device.state_gpu = clEnqueueNDRangeKernel(device.commandQueue_gpu, device.idct4x4, 1, NULL, device.gpu_work_items_per_dim, NULL, 0, NULL, NULL);
		//idct U
		device.state_gpu = clSetKernelArg(device.idct4x4, 0, sizeof(cl_mem), &device.reconstructed_frame_U);
		device.state_gpu = clSetKernelArg(device.idct4x4, 1, sizeof(cl_mem), &device.predictors_U);
		device.state_gpu = clSetKernelArg(device.idct4x4, 3, sizeof(int32_t), &cwidth);
		plane = 1;
		device.state_gpu = clSetKernelArg(device.idct4x4, 7, sizeof(int32_t), &plane);
		device.gpu_work_items_per_dim[0] = video.mb_count*4;
		device.state_gpu = clEnqueueNDRangeKernel(device.commandQueue_gpu, device.idct4x4, 1, NULL, device.gpu_work_items_per_dim, NULL, 0, NULL, NULL);
		//idct V
		device.state_gpu = clSetKernelArg(device.idct4x4, 0, sizeof(cl_mem), &device.reconstructed_frame_V);
		device.state_gpu = clSetKernelArg(device.idct4x4, 1, sizeof(cl_mem), &device.predictors_V);
		plane = 2;
		device.state_gpu = clSetKernelArg(device.idct4x4, 7, sizeof(int32_t), &plane);
		device.state_gpu = clEnqueueNDRangeKernel(device.commandQueue_gpu, device.idct4x4, 1, NULL, device.gpu_work_items_per_dim, NULL, 0, NULL, NULL);
		device.state_gpu = clFinish(device.commandQueue_gpu);

		//count SSIM
		device.gpu_work_items_per_dim[0] = video.mb_count;
		//U
		int32_t reset = 1;
		device.state_gpu = clSetKernelArg(device.count_SSIM_chroma, 0, sizeof(cl_mem), &device.current_frame_U);
		device.state_gpu = clSetKernelArg(device.count_SSIM_chroma, 1, sizeof(cl_mem), &device.reconstructed_frame_U);
		device.state_gpu = clSetKernelArg(device.count_SSIM_chroma, 5, sizeof(int32_t), &reset);
		device.state_gpu = clEnqueueNDRangeKernel(device.commandQueue_gpu, device.count_SSIM_chroma, 1, NULL, device.gpu_work_items_per_dim, NULL, 0, NULL, NULL);
		device.state_gpu = clFinish(device.commandQueue_gpu);
		//V
		reset = 0;
		device.state_gpu = clSetKernelArg(device.count_SSIM_chroma, 0, sizeof(cl_mem), &device.current_frame_V);
		device.state_gpu = clSetKernelArg(device.count_SSIM_chroma, 1, sizeof(cl_mem), &device.reconstructed_frame_V);
		device.state_gpu = clSetKernelArg(device.count_SSIM_chroma, 5, sizeof(int32_t), &reset);
		device.state_gpu = clEnqueueNDRangeKernel(device.commandQueue_gpu, device.count_SSIM_chroma, 1, NULL, device.gpu_work_items_per_dim, NULL, 0, NULL, NULL);
		device.state_gpu = clFinish(device.commandQueue_gpu);
		//Y
		device.state_gpu = clEnqueueNDRangeKernel(device.commandQueue_gpu, device.count_SSIM_luma, 1, NULL, device.gpu_work_items_per_dim, NULL, 0, NULL, NULL);
		device.state_gpu = clFinish(device.commandQueue_gpu);
	}

	if (device.state_gpu != 0)
		printf("bad kernel %d",device.state_gpu);
	device.state_gpu = clFinish(device.commandQueue_gpu);
    return;
}
