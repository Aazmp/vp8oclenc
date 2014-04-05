static void prepare_GPU_buffers()
{	
	frames.threads_free = video.thread_limit; // by this time boolcoder definetly already finished
	// first reset vector nets to zeros
	device.gpu_work_items_per_dim[0] = video.mb_count*4;
	device.state_gpu = clEnqueueNDRangeKernel(device.commandQueue1_gpu, device.reset_vectors, 1, NULL, device.gpu_work_items_per_dim, NULL, 0, NULL, NULL);
	device.state_gpu = clFinish(device.commandQueue1_gpu);
	
	// now prepare downsampled LAST buffers
	//prepare downsampled by 2
	device.gpu_work_items_per_dim[0] = video.wrk_width*video.wrk_height/4;
	device.state_gpu = clEnqueueNDRangeKernel(device.commandQueue1_gpu, device.downsample_last_1x_to_2x, 1, NULL, device.gpu_work_items_per_dim, NULL, 0, NULL, NULL);
	device.state_gpu = ifFlush(device.commandQueue1_gpu);
	device.state_gpu = clEnqueueNDRangeKernel(device.commandQueue2_gpu, device.downsample_current_1x_to_2x, 1, NULL, device.gpu_work_items_per_dim, NULL, 0, NULL, NULL);
	device.state_gpu = ifFlush(device.commandQueue2_gpu);
	//prepare downsampled by 4
	device.gpu_work_items_per_dim[0] = video.wrk_width/2*video.wrk_height/2/4;
	device.state_gpu = clEnqueueNDRangeKernel(device.commandQueue1_gpu, device.downsample_last_2x_to_4x, 1, NULL, device.gpu_work_items_per_dim, NULL, 0, NULL, NULL);
	device.state_gpu = ifFlush(device.commandQueue1_gpu);
	device.state_gpu = clEnqueueNDRangeKernel(device.commandQueue2_gpu, device.downsample_current_2x_to_4x, 1, NULL, device.gpu_work_items_per_dim, NULL, 0, NULL, NULL);
	device.state_gpu = ifFlush(device.commandQueue2_gpu);
	//prepare downsampled by 8
	device.gpu_work_items_per_dim[0] = video.wrk_width/4*video.wrk_height/4/4;
	device.state_gpu = clEnqueueNDRangeKernel(device.commandQueue1_gpu, device.downsample_last_4x_to_8x, 1, NULL, device.gpu_work_items_per_dim, NULL, 0, NULL, NULL);
	device.state_gpu = ifFlush(device.commandQueue1_gpu);
	device.state_gpu = clEnqueueNDRangeKernel(device.commandQueue2_gpu, device.downsample_current_4x_to_8x, 1, NULL, device.gpu_work_items_per_dim, NULL, 0, NULL, NULL);
	device.state_gpu = ifFlush(device.commandQueue2_gpu);
	//prepare downsampled by 16
	device.gpu_work_items_per_dim[0] = video.wrk_width/8*video.wrk_height/8/4;
	device.state_gpu = clEnqueueNDRangeKernel(device.commandQueue1_gpu, device.downsample_last_8x_to_16x, 1, NULL, device.gpu_work_items_per_dim, NULL, 0, NULL, NULL);
	device.state_gpu = ifFlush(device.commandQueue1_gpu);
	device.state_gpu = clEnqueueNDRangeKernel(device.commandQueue2_gpu, device.downsample_current_8x_to_16x, 1, NULL, device.gpu_work_items_per_dim, NULL, 0, NULL, NULL);
	device.state_gpu = ifFlush(device.commandQueue2_gpu);
	
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
	
	// prepare images (if they need to be renewed)
	const size_t origin[3] = {0, 0, 0};
	const size_t region_y[3] = {video.wrk_width, video.wrk_height, 1};
	const size_t region_uv[3] = {video.wrk_width/2, video.wrk_height/2, 1};

	device.state_gpu = clEnqueueReadBuffer(device.commandQueue1_gpu, device.reconstructed_frame_Y ,CL_TRUE, 0, video.wrk_frame_size_luma, frames.reconstructed_Y, 0, NULL, NULL);
	device.state_gpu = clEnqueueReadBuffer(device.commandQueue2_gpu, device.reconstructed_frame_U ,CL_TRUE, 0, video.wrk_frame_size_chroma, frames.reconstructed_U, 0, NULL, NULL);
	device.state_gpu = clEnqueueReadBuffer(device.commandQueue3_gpu, device.reconstructed_frame_V ,CL_TRUE, 0, video.wrk_frame_size_chroma, frames.reconstructed_V, 0, NULL, NULL);

	device.state_gpu = clEnqueueWriteImage(device.commandQueue1_gpu, device.last_frame_Y_image, CL_FALSE, origin, region_y, 0, 0, frames.reconstructed_Y, 0, NULL, NULL);
	device.state_gpu = clEnqueueWriteImage(device.commandQueue2_gpu, device.last_frame_U_image, CL_FALSE, origin, region_uv, 0, 0, frames.reconstructed_U, 0, NULL, NULL);
	device.state_gpu = clEnqueueWriteImage(device.commandQueue3_gpu, device.last_frame_V_image, CL_FALSE, origin, region_uv, 0, 0, frames.reconstructed_V, 0, NULL, NULL);

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

static void inter_transform()
{
	const int width = video.wrk_width;
	const int height = video.wrk_height;
	
	// if golden and altref buffers represent different from last buffer frame
	// and altref is not the same as altref
	const cl_int use_golden = !frames.prev_is_golden_frame; 
	const cl_int use_altref = (!frames.prev_is_altref_frame) && (frames.altref_frame_number != frames.golden_frame_number);
	//prepare downsampled frames and image objects
	prepare_GPU_buffers();


	//now search in downsampled by 16
	device.gpu_work_items_per_dim[0] = ((video.wrk_width/16)/8)*((video.wrk_height/16)/8);
	device.gpu_work_items_per_dim[0] += (device.gpu_work_items_per_dim[0] % 256) > 0 ? 
										(256 - (device.gpu_work_items_per_dim[0]%256)) : 
										0;
	// LAST
	// we use local memory in kernel, so we have to explicitly set work group size
	//max work group size for this kernel is 256! (each work-group use 16kb (defined in kernel code) and each kernel-thread needs 64b => 16kb/64b == 256
	if (device.gpu_device_type == CL_DEVICE_TYPE_GPU)
		device.gpu_work_group_size_per_dim[0] = 256;
	else 
		// just for tests on cpu (useful to control memory). Some CPU won't work with 256 kernels in one hardware thread
		device.gpu_work_group_size_per_dim[0] = 8; 
	device.state_gpu = clEnqueueNDRangeKernel(device.commandQueue1_gpu, device.luma_search_last_16x, 1, NULL, device.gpu_work_items_per_dim, device.gpu_work_group_size_per_dim, 0, NULL, NULL);
	device.state_gpu = ifFlush(device.commandQueue1_gpu);
	// GOLDEN
	if (use_golden)
	{
		device.state_gpu = clEnqueueNDRangeKernel(device.commandQueue2_gpu, device.luma_search_golden_16x, 1, NULL, device.gpu_work_items_per_dim, device.gpu_work_group_size_per_dim, 0, NULL, NULL);
		device.state_gpu = ifFlush(device.commandQueue2_gpu);
	}
	// ALTREF
	if (use_altref)
	{
		device.state_gpu = clEnqueueNDRangeKernel(device.commandQueue3_gpu, device.luma_search_altref_16x, 1, NULL, device.gpu_work_items_per_dim, device.gpu_work_group_size_per_dim, 0, NULL, NULL);
		device.state_gpu = ifFlush(device.commandQueue3_gpu);
	}

	//now search in downsampled by 8
	device.gpu_work_items_per_dim[0] = ((video.wrk_width/8)/8)*((video.wrk_height/8)/8);
	device.gpu_work_items_per_dim[0] += (device.gpu_work_items_per_dim[0] % 256) > 0 ? 
										(256 - (device.gpu_work_items_per_dim[0]%256)) : 
										0;
	// LAST
	device.state_gpu = clEnqueueNDRangeKernel(device.commandQueue1_gpu, device.luma_search_last_8x, 1, NULL, device.gpu_work_items_per_dim, device.gpu_work_group_size_per_dim, 0, NULL, NULL);
	device.state_gpu = ifFlush(device.commandQueue1_gpu);
	// GOLDEN
	if (use_golden)
	{
		device.state_gpu = clEnqueueNDRangeKernel(device.commandQueue2_gpu, device.luma_search_golden_8x, 1, NULL, device.gpu_work_items_per_dim, device.gpu_work_group_size_per_dim, 0, NULL, NULL);
		device.state_gpu = ifFlush(device.commandQueue2_gpu);
	}
	// ALTREF
	if (use_altref)
	{
		device.state_gpu = clEnqueueNDRangeKernel(device.commandQueue3_gpu, device.luma_search_altref_8x, 1, NULL, device.gpu_work_items_per_dim, device.gpu_work_group_size_per_dim, 0, NULL, NULL);
		device.state_gpu = ifFlush(device.commandQueue3_gpu);
	}	

	//now search in downsampled by 4
	device.gpu_work_items_per_dim[0] = ((video.wrk_width/4)/8)*((video.wrk_height/4)/8);
	device.gpu_work_items_per_dim[0] += (device.gpu_work_items_per_dim[0] % 256) > 0 ? 
										(256 - (device.gpu_work_items_per_dim[0]%256)) : 
										0;
	// LAST
	device.state_gpu = clEnqueueNDRangeKernel(device.commandQueue1_gpu, device.luma_search_last_4x, 1, NULL, device.gpu_work_items_per_dim, device.gpu_work_group_size_per_dim, 0, NULL, NULL);
	device.state_gpu = ifFlush(device.commandQueue1_gpu);
	// GOLDEN
	if (use_golden)
	{
		device.state_gpu = clEnqueueNDRangeKernel(device.commandQueue2_gpu, device.luma_search_golden_4x, 1, NULL, device.gpu_work_items_per_dim, device.gpu_work_group_size_per_dim, 0, NULL, NULL);
		device.state_gpu = ifFlush(device.commandQueue2_gpu);
	}
	// ALTREF
	if (use_altref)
	{
		device.state_gpu = clEnqueueNDRangeKernel(device.commandQueue3_gpu, device.luma_search_altref_4x, 1, NULL, device.gpu_work_items_per_dim, device.gpu_work_group_size_per_dim, 0, NULL, NULL);
		device.state_gpu = ifFlush(device.commandQueue3_gpu);
	}

	//now search in downsampled by 2
	device.gpu_work_items_per_dim[0] = ((video.wrk_width/2)/8)*((video.wrk_height/2)/8);
	device.gpu_work_items_per_dim[0] += (device.gpu_work_items_per_dim[0] % 256) > 0 ? 
										(256 - (device.gpu_work_items_per_dim[0]%256)) : 
										0;
	// LAST
	device.state_gpu = clEnqueueNDRangeKernel(device.commandQueue1_gpu, device.luma_search_last_2x, 1, NULL, device.gpu_work_items_per_dim, device.gpu_work_group_size_per_dim, 0, NULL, NULL);
	device.state_gpu = ifFlush(device.commandQueue1_gpu);
	// GOLDEN
	if (use_golden)
	{
		device.state_gpu = clEnqueueNDRangeKernel(device.commandQueue2_gpu, device.luma_search_golden_2x, 1, NULL, device.gpu_work_items_per_dim, device.gpu_work_group_size_per_dim, 0, NULL, NULL);
		device.state_gpu = ifFlush(device.commandQueue2_gpu);
	}
	// ALTREF
	if (use_altref)
	{
		device.state_gpu = clEnqueueNDRangeKernel(device.commandQueue3_gpu, device.luma_search_altref_2x, 1, NULL, device.gpu_work_items_per_dim, device.gpu_work_group_size_per_dim, 0, NULL, NULL);
		device.state_gpu = ifFlush(device.commandQueue3_gpu);
	}

	//now search in original size
	device.gpu_work_items_per_dim[0] = video.mb_count*4; 
	device.gpu_work_items_per_dim[0] += (device.gpu_work_items_per_dim[0] % 256) > 0 ? 
										(256 - (device.gpu_work_items_per_dim[0]%256)) : 
										0;
	// LAST
	device.state_gpu = clEnqueueNDRangeKernel(device.commandQueue1_gpu, device.luma_search_last_1x, 1, NULL, device.gpu_work_items_per_dim, device.gpu_work_group_size_per_dim, 0, NULL, NULL);
	device.state_gpu = ifFlush(device.commandQueue1_gpu);
	// GOLDEN
	if (use_golden)
	{
		device.state_gpu = clEnqueueNDRangeKernel(device.commandQueue2_gpu, device.luma_search_golden_1x, 1, NULL, device.gpu_work_items_per_dim, device.gpu_work_group_size_per_dim, 0, NULL, NULL);
		device.state_gpu = ifFlush(device.commandQueue2_gpu);
	}
	// ALTREF
	if (use_altref)
	{
		device.state_gpu = clEnqueueNDRangeKernel(device.commandQueue3_gpu, device.luma_search_altref_1x, 1, NULL, device.gpu_work_items_per_dim, device.gpu_work_group_size_per_dim, 0, NULL, NULL);
		device.state_gpu = ifFlush(device.commandQueue3_gpu);
	}

	// search in image with interpolation on the run
	// LAST
	device.state_gpu = clEnqueueNDRangeKernel(device.commandQueue1_gpu, device.luma_search_last_d4x, 1, NULL, device.gpu_work_items_per_dim, device.gpu_work_group_size_per_dim, 0, NULL, NULL);
	device.state_gpu = ifFlush(device.commandQueue1_gpu);
	// GOLDEN
	if (use_golden)
	{
		device.state_gpu = clEnqueueNDRangeKernel(device.commandQueue2_gpu, device.luma_search_golden_d4x, 1, NULL, device.gpu_work_items_per_dim, device.gpu_work_group_size_per_dim, 0, NULL, NULL);
		device.state_gpu = ifFlush(device.commandQueue2_gpu);
	}
	// ALTREF
	if (use_altref)
	{
		device.state_gpu = clEnqueueNDRangeKernel(device.commandQueue3_gpu, device.luma_search_altref_d4x, 1, NULL, device.gpu_work_items_per_dim, device.gpu_work_group_size_per_dim, 0, NULL, NULL);
		device.state_gpu = ifFlush(device.commandQueue3_gpu);
	}

	//device.state_gpu = clFinish(device.commandQueue1_gpu);
	if (use_golden||use_altref)
		device.state_gpu = finalFlush(device.commandQueue1_gpu);
	if (use_golden)
		device.state_gpu = finalFlush(device.commandQueue2_gpu);
	if (use_altref)
		device.state_gpu = finalFlush(device.commandQueue3_gpu);
	if (use_golden)
		device.state_gpu = clFinish(device.commandQueue2_gpu);
	if (use_altref)
		device.state_gpu = clFinish(device.commandQueue3_gpu);

	// now set each MB with the best reference
	device.gpu_work_items_per_dim[0] = video.mb_count;
	device.state_gpu = clSetKernelArg(device.select_reference, 9, sizeof(cl_int), &use_golden);
	device.state_gpu = clSetKernelArg(device.select_reference, 10, sizeof(cl_int), &use_altref);
	device.state_gpu = clEnqueueNDRangeKernel(device.commandQueue1_gpu, device.select_reference, 1, NULL, device.gpu_work_items_per_dim, NULL, 0, NULL, NULL);
	device.state_gpu = ifFlush(device.commandQueue1_gpu);
	// set 16x16 mode for macroblocks, whose blocks have identical vectors
	device.gpu_work_items_per_dim[0] = video.mb_count;
	device.state_gpu = clEnqueueNDRangeKernel(device.commandQueue1_gpu, device.pack_8x8_into_16x16, 1, NULL, device.gpu_work_items_per_dim, NULL, 0, NULL, NULL);
	device.state_gpu = ifFlush(device.commandQueue1_gpu);
	
	device.state_gpu = clFinish(device.commandQueue1_gpu); //we need to finish packing before start preparing predictors for golden or altref and before reading buffers
	
	device.state_gpu = clEnqueueReadBuffer(device.dataCopy_gpu, device.macroblock_parts_gpu, CL_FALSE, 0, video.mb_count*sizeof(cl_int), frames.MB_parts, 0, NULL, NULL);
	device.state_gpu = clEnqueueReadBuffer(device.dataCopy_gpu, device.macroblock_reference_frame_gpu, CL_FALSE, 0, video.mb_count*sizeof(cl_int), frames.MB_reference_frame, 0, NULL, NULL);
	device.state_gpu = clEnqueueReadBuffer(device.dataCopy_gpu, device.macroblock_vectors_gpu, CL_FALSE, 0, video.mb_count*sizeof(macroblock_vectors_t), frames.MB_vectors, 0, NULL, NULL);
	device.state_gpu = clFlush(device.dataCopy_gpu);

	// now for each plane and reference frame fill predictors and residual buffers
	cl_int ref;
	cl_int cwidth = video.wrk_width/2;
	// Y
	device.gpu_work_items_per_dim[0] = video.mb_count*16;
	// LAST
	ref = LAST;
	device.state_gpu = clEnqueueNDRangeKernel(device.commandQueue1_gpu, device.prepare_predictors_and_residual_last_Y, 1, NULL, device.gpu_work_items_per_dim, NULL, 0, NULL, NULL);
	device.state_gpu = ifFlush(device.commandQueue1_gpu);
	// GOLDEN
	if (use_golden)
	{
		device.state_gpu = clEnqueueNDRangeKernel(device.commandQueue2_gpu, device.prepare_predictors_and_residual_golden_Y, 1, NULL, device.gpu_work_items_per_dim, NULL, 0, NULL, NULL);
		device.state_gpu = ifFlush(device.commandQueue2_gpu);
	}
	// ALTREF
	if (use_altref)
	{
		device.state_gpu = clEnqueueNDRangeKernel(device.commandQueue3_gpu, device.prepare_predictors_and_residual_altref_Y, 1, NULL, device.gpu_work_items_per_dim, NULL, 0, NULL, NULL);
		device.state_gpu = ifFlush(device.commandQueue3_gpu);
	}
	// U
	device.gpu_work_items_per_dim[0] = video.mb_count*4;
	// LAST
	device.state_gpu = clEnqueueNDRangeKernel(device.commandQueue1_gpu, device.prepare_predictors_and_residual_last_U, 1, NULL, device.gpu_work_items_per_dim, NULL, 0, NULL, NULL);
	device.state_gpu = ifFlush(device.commandQueue1_gpu);
	// GOLDEN
	if (use_golden)
	{
		device.state_gpu = clEnqueueNDRangeKernel(device.commandQueue2_gpu, device.prepare_predictors_and_residual_golden_U, 1, NULL, device.gpu_work_items_per_dim, NULL, 0, NULL, NULL);
		device.state_gpu = ifFlush(device.commandQueue2_gpu);
	}
	// ALTREF
	if (use_altref)
	{
		device.state_gpu = clEnqueueNDRangeKernel(device.commandQueue3_gpu, device.prepare_predictors_and_residual_altref_U, 1, NULL, device.gpu_work_items_per_dim, NULL, 0, NULL, NULL);
		device.state_gpu = ifFlush(device.commandQueue3_gpu);
	}
	// V
	// LAST
	device.state_gpu = clEnqueueNDRangeKernel(device.commandQueue1_gpu, device.prepare_predictors_and_residual_last_V, 1, NULL, device.gpu_work_items_per_dim, NULL, 0, NULL, NULL);
	device.state_gpu = ifFlush(device.commandQueue1_gpu);
	// GOLDEN
	if (use_golden)
	{
		device.state_gpu = clEnqueueNDRangeKernel(device.commandQueue2_gpu, device.prepare_predictors_and_residual_golden_V, 1, NULL, device.gpu_work_items_per_dim, NULL, 0, NULL, NULL);
		device.state_gpu = ifFlush(device.commandQueue2_gpu);
	}
	// ALTREF
	if (use_altref)
	{
		device.state_gpu = clEnqueueNDRangeKernel(device.commandQueue3_gpu, device.prepare_predictors_and_residual_altref_V, 1, NULL, device.gpu_work_items_per_dim, NULL, 0, NULL, NULL);
		device.state_gpu = ifFlush(device.commandQueue3_gpu);
	}
	
	device.state_gpu = finalFlush(device.commandQueue1_gpu);
	device.state_gpu = finalFlush(device.commandQueue3_gpu);
	device.state_gpu = clFinish(device.commandQueue2_gpu);
	device.state_gpu = clFinish(device.commandQueue3_gpu);

	// now for each segment (begin with highest quantizer (last index))
	for (cl_int seg_id = LQ_segment; seg_id >= UQ_segment; --seg_id)
	{
		device.state_gpu = clFinish(device.commandQueue1_gpu);

		//dct Y
		device.gpu_work_items_per_dim[0] = video.mb_count*16;
		device.state_gpu = clEnqueueNDRangeKernel(device.commandQueue1_gpu, device.dct4x4_Y[seg_id], 1, NULL, device.gpu_work_items_per_dim, NULL, 0, NULL, NULL);
		device.state_gpu = ifFlush(device.commandQueue1_gpu);	
		//dct U
		device.gpu_work_items_per_dim[0] = video.mb_count*4;
		device.state_gpu = clEnqueueNDRangeKernel(device.commandQueue2_gpu, device.dct4x4_U[seg_id], 1, NULL, device.gpu_work_items_per_dim, NULL, 0, NULL, NULL);
		device.state_gpu = ifFlush(device.commandQueue2_gpu);	
		//dct V
		device.state_gpu = clEnqueueNDRangeKernel(device.commandQueue3_gpu, device.dct4x4_V[seg_id], 1, NULL, device.gpu_work_items_per_dim, NULL, 0, NULL, NULL);
		device.state_gpu = ifFlush(device.commandQueue3_gpu);
		//wht and iwht
		device.gpu_work_items_per_dim[0] = video.mb_count;
		device.state_gpu = clEnqueueNDRangeKernel(device.commandQueue1_gpu, device.wht4x4_iwht4x4[seg_id], 1, NULL, device.gpu_work_items_per_dim, NULL, 0, NULL, NULL);
		device.state_gpu = ifFlush(device.commandQueue1_gpu);
		//idct Y
		device.gpu_work_items_per_dim[0] = video.mb_count*16;
		device.state_gpu = clEnqueueNDRangeKernel(device.commandQueue1_gpu, device.idct4x4_Y[seg_id], 1, NULL, device.gpu_work_items_per_dim, NULL, 0, NULL, NULL);
		device.state_gpu = ifFlush(device.commandQueue1_gpu);
		//idct U
		device.gpu_work_items_per_dim[0] = video.mb_count*4;
		device.state_gpu = clEnqueueNDRangeKernel(device.commandQueue2_gpu, device.idct4x4_U[seg_id], 1, NULL, device.gpu_work_items_per_dim, NULL, 0, NULL, NULL);
		device.state_gpu = ifFlush(device.commandQueue2_gpu);
		//idct V
		device.state_gpu = clEnqueueNDRangeKernel(device.commandQueue3_gpu, device.idct4x4_V[seg_id], 1, NULL, device.gpu_work_items_per_dim, NULL, 0, NULL, NULL);
		device.state_gpu = ifFlush(device.commandQueue3_gpu);

		//count SSIM
		device.gpu_work_items_per_dim[0] = video.mb_count;
		//Y
		device.state_gpu = clEnqueueNDRangeKernel(device.commandQueue1_gpu, device.count_SSIM_luma[seg_id], 1, NULL, device.gpu_work_items_per_dim, NULL, 0, NULL, NULL);
		device.state_gpu = ifFlush(device.commandQueue1_gpu);
		//U
		device.state_gpu = clEnqueueNDRangeKernel(device.commandQueue2_gpu, device.count_SSIM_chroma_U[seg_id], 1, NULL, device.gpu_work_items_per_dim, NULL, 0, NULL, NULL);
		device.state_gpu = ifFlush(device.commandQueue2_gpu);
		//V
		device.state_gpu = clEnqueueNDRangeKernel(device.commandQueue3_gpu, device.count_SSIM_chroma_V[seg_id], 1, NULL, device.gpu_work_items_per_dim, NULL, 0, NULL, NULL);
		device.state_gpu = ifFlush(device.commandQueue3_gpu);

		device.state_gpu = finalFlush(device.commandQueue1_gpu);
		device.state_gpu = finalFlush(device.commandQueue3_gpu);
		device.state_gpu = clFinish(device.commandQueue2_gpu);
		device.state_gpu = clFinish(device.commandQueue3_gpu);

		device.state_gpu = clEnqueueNDRangeKernel(device.commandQueue1_gpu, device.gather_SSIM, 1, NULL, device.gpu_work_items_per_dim, NULL, 0, NULL, NULL);
	}

	if (device.state_gpu != 0)
		printf("bad kernel %d",device.state_gpu);

    return;
}