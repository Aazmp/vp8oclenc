// string to print when "-h" option is met
char small_help[] = "\n"
					"-i\t:  input file path\n"
					"-o\t:  output file path\n"
					"-qmin\t:  min quantizer index (also the only index for key frames)\n"
					"-qmax\t:  max quantizer index\n"
					"-g\t:  Group of Pictures size\n"
					"-w\t:  amount of work-items on gpu launched simutaneosly\n"
					"-t\t:  amount of partitions for boolean encoding (also threads launched at once)\n"
					"-ls\t:  loop filter sharpness\n"
					"\n"
					"-SSIM-target\t:  tries to keep SSIM of encoded frame higher than this value;\n"
					"\t\t\t:  format 0.XX (just write XX digits without 0.)"
					"-altref-range\t:  amount of frames between using altref;\n"
					"\t\t\t:  altref are used with lower quantizer and reference previous altref"
					"\n\n"
					;

int init_all()
{
	device.cpu_work_items_per_dim[0] = video.number_of_partitions;
	device.cpu_work_group_size_per_dim[0] = 1;

	video.timestep = 1;
	video.timescale = 1;

    // init platform, device, context, program, kernels
    int i;
	cl_uint num_platforms;
	clGetPlatformIDs(2, NULL, &num_platforms);
	if (num_platforms < 1)
	{
		printf("no OpenCL platforms found \n");
		return -1;
	}
	device.platforms = (cl_platform_id*)malloc(num_platforms*sizeof(cl_platform_id));
	device.state_cpu = clGetPlatformIDs(num_platforms, device.platforms, NULL);
	device.device_cpu = (cl_device_id*)malloc(sizeof(cl_device_id));
	device.state_cpu = clGetDeviceIDs(device.platforms[0], CL_DEVICE_TYPE_CPU, 1, device.device_cpu, NULL);
	i = 1;
	while ((device.state_cpu != CL_SUCCESS) && (i < (int)num_platforms)) 
	{
		device.state_cpu = clGetDeviceIDs(device.platforms[i], CL_DEVICE_TYPE_CPU, 1, device.device_cpu, NULL);
		++i;
	}
	if (device.state_cpu != CL_SUCCESS)
	{
		printf("no CPU device found :) \n");
		return -1;
	}

	device.context_cpu = clCreateContext(NULL, 1, device.device_cpu, NULL, NULL, &device.state_cpu);
	if (video.GOP_size > 1) {
		device.device_gpu = (cl_device_id*)malloc(sizeof(cl_device_id));
		device.state_gpu = clGetDeviceIDs(device.platforms[0], CL_DEVICE_TYPE_GPU, 1, device.device_gpu, NULL);
		i = 1;
		while ((device.state_gpu != CL_SUCCESS) && (i < (int)num_platforms)) 
		{
			device.state_cpu = clGetDeviceIDs(device.platforms[i], CL_DEVICE_TYPE_GPU, 1, device.device_gpu, NULL);
			++i;
		}
		if (device.state_gpu != CL_SUCCESS)
		{
			printf("no GPU device found \n");
			return -1;
		}
		device.context_gpu = clCreateContext(NULL, 1, device.device_gpu, NULL, NULL, &device.state_gpu);
	}

	FILE *program_handle;
	uint32_t program_size;

	// program sources in text files
	// GPU:
	if (video.GOP_size > 1) {
		printf("reading GPU program...\n");
		const char gpu_options[] = "-cl-std=CL1.2 -cl-opt-disable";
		program_handle = fopen(GPUPATH, "rb");
		fseek(program_handle, 0, SEEK_END);
		program_size = ftell(program_handle);
		rewind(program_handle); // set to start
		char** device_program_source_gpu = (char**)malloc(sizeof(char*));
		*device_program_source_gpu = (char*)malloc(program_size+1);
		(*device_program_source_gpu)[program_size] = '\0';
		int deb = fread(*device_program_source_gpu, sizeof(char), program_size, program_handle);
		fclose(program_handle);
		device.program_gpu = clCreateProgramWithSource(device.context_gpu, 1, (const char**)device_program_source_gpu, NULL, &device.state_gpu);
		printf("building GPU program...\n");
		device.state_gpu = clBuildProgram(device.program_gpu, 1, device.device_gpu, gpu_options, NULL, NULL);
		if(device.state_gpu < 0)  //print log if there were mistakes during kernel building
		{
			error_file.handle = fopen(error_file.path,"w");
		    printf("\n -=kernel build fail=- \n");
			static char *program_log;
			size_t log_size;
			clGetProgramBuildInfo(device.program_gpu, *(device.device_gpu), CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
				program_log = (char*)malloc((log_size+1) * (sizeof(char)));
			clGetProgramBuildInfo(device.program_gpu, *(device.device_gpu), CL_PROGRAM_BUILD_LOG, log_size, program_log, NULL);
				fprintf(error_file.handle, "%s\n", program_log);
			free(program_log);
			fclose(error_file.handle);
			return device.state_gpu;
		}
		{ char kernel_name[] = "reset_vectors";
		device.reset_vectors = clCreateKernel(device.program_gpu, kernel_name, &device.state_gpu); }
		{ char kernel_name[] = "luma_search_1step";
		device.luma_search_1step = clCreateKernel(device.program_gpu, kernel_name, &device.state_gpu); }
		{ char kernel_name[] = "luma_search_2step";
		device.luma_search_2step = clCreateKernel(device.program_gpu, kernel_name, &device.state_gpu); }
		{ char kernel_name[] = "downsample_x2";
		device.downsample = clCreateKernel(device.program_gpu, kernel_name, &device.state_gpu); }
		{ char kernel_name[] = "try_another_reference";
		device.try_another_reference = clCreateKernel(device.program_gpu, kernel_name, &device.state_gpu); }
		{ char kernel_name[] = "luma_transform_16x16";
		device.luma_transform_16x16 = clCreateKernel(device.program_gpu, kernel_name, &device.state_gpu); }
		{ char kernel_name[] = "luma_transform_8x8";
		device.luma_transform_8x8 = clCreateKernel(device.program_gpu, kernel_name, &device.state_gpu); }
		{ char kernel_name[] = "chroma_transform";
		device.chroma_transform = clCreateKernel(device.program_gpu, kernel_name, &device.state_gpu); }
		{ char kernel_name[] = "prepare_filter_mask"; //and non zero coeffs count
		device.prepare_filter_mask = clCreateKernel(device.program_gpu, kernel_name, &device.state_gpu); }
		{ char kernel_name[] = "luma_interpolate_Hx4_bc";
		device.luma_interpolate_Hx4 = clCreateKernel(device.program_gpu, kernel_name, &device.state_gpu); }
		{ char kernel_name[] = "luma_interpolate_Vx4_bc";
		device.luma_interpolate_Vx4 = clCreateKernel(device.program_gpu, kernel_name, &device.state_gpu); }
		{ char kernel_name[] = "chroma_interpolate_Hx8_bc";
		device.chroma_interpolate_Hx8 = clCreateKernel(device.program_gpu, kernel_name, &device.state_gpu); }
		{ char kernel_name[] = "chroma_interpolate_Vx8_bc";
		device.chroma_interpolate_Vx8 = clCreateKernel(device.program_gpu, kernel_name, &device.state_gpu); }
		{ char kernel_name[] = "normal_loop_filter_MBH";
		device.normal_loop_filter_MBH = clCreateKernel(device.program_gpu, kernel_name, &device.state_gpu); }
		{ char kernel_name[] = "normal_loop_filter_MBV";
		device.normal_loop_filter_MBV = clCreateKernel(device.program_gpu, kernel_name, &device.state_gpu); }
		{ char kernel_name[] = "count_SSIM_luma";
		device.count_SSIM_luma = clCreateKernel(device.program_gpu, kernel_name, &device.state_gpu); }
		{ char kernel_name[] = "count_SSIM_chroma";
		device.count_SSIM_chroma = clCreateKernel(device.program_gpu, kernel_name, &device.state_gpu); }
	}
	// CPU:
	printf("reading CPU program...\n");
	const char cpu_options[] = "-cl-std=CL1.0";
	program_handle = fopen(CPUPATH, "rb");
	fseek(program_handle, 0, SEEK_END);
	program_size = ftell(program_handle);
	rewind(program_handle); // set to start
	char** device_program_source_cpu = (char**)malloc(sizeof(char*));
		*device_program_source_cpu = (char*)malloc(program_size+1);
	(*device_program_source_cpu)[program_size] = '\0';
	fread(*device_program_source_cpu, sizeof(char),	program_size, program_handle);
	fclose(program_handle);
	device.program_cpu = clCreateProgramWithSource(device.context_cpu, 1, (const char**)device_program_source_cpu, NULL, &device.state_cpu);
	printf("building CPU program...\n");
	device.state_cpu = clBuildProgram(device.program_cpu, 1, device.device_cpu, cpu_options, NULL, NULL);
	if(device.state_cpu < 0)  //print log if there were mistakes during kernel building
	{
		error_file.handle = fopen(error_file.path,"w");
	    printf("\n -=kernel build fail=- \n");
		static char *program_log;
		size_t log_size;
		clGetProgramBuildInfo(device.program_cpu, *(device.device_cpu), CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
		program_log = (char*) malloc((log_size+1) * (sizeof(char)));
		clGetProgramBuildInfo(device.program_cpu, *(device.device_cpu), CL_PROGRAM_BUILD_LOG, log_size, program_log, NULL);
		fprintf(error_file.handle, "%s\n", program_log);
		free(program_log);
		fclose(error_file.handle);
		return device.state_cpu;
	}
	{ char kernel_name[] = "encode_coefficients";
	device.encode_coefficients = clCreateKernel(device.program_cpu, kernel_name, &device.state_cpu); }
	{ char kernel_name[] = "count_probs";
	device.count_probs = clCreateKernel(device.program_cpu, kernel_name, &device.state_cpu); }
	{ char kernel_name[] = "num_div_denom";
	device.num_div_denom = clCreateKernel(device.program_cpu, kernel_name, &device.state_cpu); }


    video.src_frame_size_luma = video.src_height*video.src_width;
    video.src_frame_size_chroma = video.src_frame_size_luma >> 2;
    int src_frame_size_full = video.src_frame_size_luma + (video.src_frame_size_chroma << 1) ;

    video.wrk_height = video.dst_height;
    video.wrk_width = video.dst_width;
    // TODO: change to & 0x15
    if (video.dst_height % 16) // if non divisible by MB size - pad
        video.wrk_height = video.dst_height + (16 - (video.dst_height % 16));
    if (video.dst_width % 16) // if non divisible by MB size - pad
        video.wrk_width = video.dst_width + (16 - (video.dst_width % 16));

    video.wrk_frame_size_luma = video.wrk_height*video.wrk_width;
    video.wrk_frame_size_chroma = video.wrk_frame_size_luma >> 2;


    video.mb_height = video.wrk_height >> 4;
    video.mb_width = video.wrk_width >> 4;
    video.mb_count = video.mb_width*video.mb_height;

	frames.input_pack_size = 1;
    frames.input_pack = (uint8_t*)malloc(src_frame_size_full*frames.input_pack_size);
    // all buffers of previous frames must be padded
    frames.reconstructed_Y =(uint8_t*)malloc(video.wrk_frame_size_luma);
    frames.reconstructed_U =(uint8_t*)malloc(video.wrk_frame_size_chroma);
    frames.reconstructed_V =(uint8_t*)malloc(video.wrk_frame_size_chroma);
    frames.transformed_blocks =(macroblock*)malloc(sizeof(macroblock)*video.mb_count);
	frames.e_data = (macroblock_extra_data*)malloc(video.mb_count*sizeof(macroblock_extra_data));
    frames.encoded_frame = (uint8_t*)malloc(src_frame_size_full<<1); // extra size
	frames.partition_0 = (uint8_t*)malloc(64*video.mb_count + 128); 
	video.partition_step = sizeof(int16_t)*800*(video.mb_count)/2;
	frames.partitions = (uint8_t*)malloc(video.partition_step); 

	frames.last_U = (uint8_t*)malloc(video.wrk_frame_size_chroma);
	frames.last_V = (uint8_t*)malloc(video.wrk_frame_size_chroma);
    if (((video.src_height != video.dst_height) || (video.src_width != video.dst_width)) ||
        ((video.wrk_height != video.dst_height) || (video.wrk_width != video.dst_width)))
    {
        // then we need buffer to store resized input frame (and padded along the way, to avoid double-copy) - 1st if-line
        // we need to store padded data - 2nd if-line
        frames.current_Y = (uint8_t*)malloc(video.wrk_frame_size_luma);
        frames.current_U = (uint8_t*)malloc(video.wrk_frame_size_chroma);
        frames.current_V = (uint8_t*)malloc(video.wrk_frame_size_chroma);
    }
	else 
	{ // to avoid reading from unknown space (for example memcpy in get_yuv420_frame before first frame read)
		// also could be done by checking if frame is first (done too, both for safety)
		frames.current_U = frames.last_U;
		frames.current_V = frames.last_V;
	} // later these buffers are being reassigned

	if (video.GOP_size > 1) {
		device.current_frame_Y = clCreateBuffer(device.context_gpu, CL_MEM_READ_WRITE, video.wrk_frame_size_luma, NULL , &device.state_gpu);
		if (device.state_gpu != 0) { printf("GPU device memory problem %d with current_frame_Y\n", device.state_gpu); return -1; }
		device.current_frame_Y_downsampled_by2 = clCreateBuffer(device.context_gpu, CL_MEM_READ_WRITE, video.wrk_frame_size_luma/4, NULL , &device.state_gpu);
		if (device.state_gpu != 0) { printf("GPU device memory problem %d with current_frame_Y_downsampled_by2\n", device.state_gpu); return -1; }
		device.current_frame_Y_downsampled_by4 = clCreateBuffer(device.context_gpu, CL_MEM_READ_WRITE, video.wrk_frame_size_luma/16, NULL , &device.state_gpu);
		if (device.state_gpu != 0) { printf("GPU device memory problem %d with current_frame_Y_downsampled_by4\n", device.state_gpu); return -1; }
		device.current_frame_Y_downsampled_by8 = clCreateBuffer(device.context_gpu, CL_MEM_READ_WRITE, video.wrk_frame_size_luma/64, NULL , &device.state_gpu);
		if (device.state_gpu != 0) { printf("GPU device memory problem %d with current_frame_Y_downsampled_by8\n", device.state_gpu); return -1; }
		device.current_frame_Y_downsampled_by16 = clCreateBuffer(device.context_gpu, CL_MEM_READ_WRITE, video.wrk_frame_size_luma/256, NULL , &device.state_gpu);
		if (device.state_gpu != 0) { printf("GPU device memory problem %d with current_frame_Y_downsampled_by16\n", device.state_gpu); return -1; }
		device.current_frame_U = clCreateBuffer(device.context_gpu, CL_MEM_READ_ONLY, video.wrk_frame_size_chroma, NULL , &device.state_gpu);
		if (device.state_gpu != 0) { printf("GPU device memory problem %d with current_frame_U\n", device.state_gpu); return -1; }
		device.current_frame_V = clCreateBuffer(device.context_gpu, CL_MEM_READ_ONLY, video.wrk_frame_size_chroma, NULL , &device.state_gpu);
		if (device.state_gpu != 0) { printf("GPU device memory problem %d with current_frame_V\n", device.state_gpu); return -1; }
		device.ref_frame_Y_interpolated = clCreateBuffer(device.context_gpu, CL_MEM_READ_WRITE, 16*video.wrk_frame_size_luma, NULL , &device.state_gpu);
		if (device.state_gpu != 0) { printf("GPU device memory problem %d with last_frame_Y_interpolated\n", device.state_gpu); return -1; }
		device.ref_frame_Y_downsampled_by2 = clCreateBuffer(device.context_gpu, CL_MEM_READ_WRITE, video.wrk_frame_size_luma/4, NULL , &device.state_gpu);
		if (device.state_gpu != 0) { printf("GPU device memory problem %d with last_frame_Y_downsampled_by2\n", device.state_gpu); return -1; }
		device.ref_frame_Y_downsampled_by4 = clCreateBuffer(device.context_gpu, CL_MEM_READ_WRITE, video.wrk_frame_size_luma/16, NULL , &device.state_gpu);
		if (device.state_gpu != 0) { printf("GPU device memory problem %d with last_frame_Y_downsampled_by4\n", device.state_gpu); return -1; }
		device.ref_frame_Y_downsampled_by8 = clCreateBuffer(device.context_gpu, CL_MEM_READ_WRITE, video.wrk_frame_size_luma/64, NULL , &device.state_gpu);
		if (device.state_gpu != 0) { printf("GPU device memory problem %d with last_frame_Y_downsampled_by8\n", device.state_gpu); return -1; }
		device.ref_frame_Y_downsampled_by16 = clCreateBuffer(device.context_gpu, CL_MEM_READ_WRITE, video.wrk_frame_size_luma/256, NULL , &device.state_gpu);
		if (device.state_gpu != 0) { printf("GPU device memory problem %d with frame_Y_downsampled_by16\n", device.state_gpu); return -1; }
		device.ref_frame_U = clCreateBuffer(device.context_gpu, CL_MEM_READ_WRITE, 64*video.wrk_frame_size_chroma, NULL , &device.state_gpu);
		if (device.state_gpu != 0) { printf("GPU device memory problem %d with last_frame_U\n", device.state_gpu); return -1; }
		device.ref_frame_V = clCreateBuffer(device.context_gpu, CL_MEM_READ_WRITE, 64*video.wrk_frame_size_chroma, NULL , &device.state_gpu);
		if (device.state_gpu != 0) { printf("GPU device memory problem %d with last_frame_V\n", device.state_gpu); return -1; }
		device.reconstructed_frame_Y = clCreateBuffer(device.context_gpu, CL_MEM_READ_WRITE, video.wrk_frame_size_luma, NULL , &device.state_gpu);
		if (device.state_gpu != 0) { printf("GPU device memory problem %d with reconstructed_frame_Y\n", device.state_gpu); return -1; }
		device.reconstructed_frame_U = clCreateBuffer(device.context_gpu, CL_MEM_READ_WRITE, video.wrk_frame_size_chroma, NULL , &device.state_gpu);
		if (device.state_gpu != 0) { printf("GPU device memory problem %d with reconstructed_frame_U\n", device.state_gpu); return -1; }
		device.reconstructed_frame_V = clCreateBuffer(device.context_gpu, CL_MEM_READ_WRITE, video.wrk_frame_size_chroma, NULL , &device.state_gpu);
		if (device.state_gpu != 0) { printf("GPU device memory problem %d with reconstructed_frame_V\n", device.state_gpu); return -1; }
		device.golden_frame_Y = clCreateBuffer(device.context_gpu, CL_MEM_READ_WRITE, video.wrk_frame_size_luma, NULL , &device.state_gpu);
		if (device.state_gpu != 0) { printf("GPU device memory problem %d with golden_frame_Y\n", device.state_gpu); return -1; }
		device.golden_frame_U = clCreateBuffer(device.context_gpu, CL_MEM_READ_WRITE, video.wrk_frame_size_chroma, NULL , &device.state_gpu);
		if (device.state_gpu != 0) { printf("GPU device memory problem %d with golden_frame_U\n", device.state_gpu); return -1; }
		device.golden_frame_V = clCreateBuffer(device.context_gpu, CL_MEM_READ_WRITE, video.wrk_frame_size_chroma, NULL , &device.state_gpu);
		if (device.state_gpu != 0) { printf("GPU device memory problem %d with golden_frame_V\n", device.state_gpu); return -1; }
		device.altref_frame_Y = clCreateBuffer(device.context_gpu, CL_MEM_READ_WRITE, video.wrk_frame_size_luma, NULL , &device.state_gpu);
		if (device.state_gpu != 0) { printf("GPU device memory problem %d with altref_frame_Y\n", device.state_gpu); return -1; }
		device.altref_frame_U = clCreateBuffer(device.context_gpu, CL_MEM_READ_WRITE, video.wrk_frame_size_chroma, NULL , &device.state_gpu);
		if (device.state_gpu != 0) { printf("GPU device memory problem %d with altref_frame_U\n", device.state_gpu); return -1; }
		device.altref_frame_V = clCreateBuffer(device.context_gpu, CL_MEM_READ_WRITE, video.wrk_frame_size_chroma, NULL , &device.state_gpu);
		if (device.state_gpu != 0) { printf("GPU device memory problem %d with altref_frame_V\n", device.state_gpu); return -1; }
		device.transformed_blocks_gpu = clCreateBuffer(device.context_gpu, CL_MEM_READ_WRITE, sizeof(macroblock)*video.mb_count, NULL , &device.state_gpu);
		if (device.state_gpu != 0) { printf("GPU device memory problem %d with transformed_blocks_gpu\n", device.state_gpu); return -1; }
		device.vnet1 = clCreateBuffer(device.context_gpu, CL_MEM_READ_WRITE, sizeof(vector_net)*video.mb_count*4, NULL , &device.state_gpu);
		if (device.state_gpu != 0) { printf("GPU device memory problem %d with vnet1\n", device.state_gpu); return -1; }
		device.vnet2 = clCreateBuffer(device.context_gpu, CL_MEM_READ_WRITE, sizeof(vector_net)*video.mb_count*4, NULL , &device.state_gpu);
		if (device.state_gpu != 0) { printf("GPU device memory problem %d with vnet2\n", device.state_gpu); return -1; }
		device.mb_metrics = clCreateBuffer(device.context_gpu, CL_MEM_READ_WRITE, sizeof(int32_t)*video.mb_count*4, NULL , &device.state_gpu);
		if (device.state_gpu != 0) { printf("GPU device memory problem %d with mb_metrics\n", device.state_gpu); return -1; }
		device.mb_mask = clCreateBuffer(device.context_gpu, CL_MEM_READ_WRITE, sizeof(int32_t)*video.mb_count, NULL , &device.state_gpu);
		if (device.state_gpu != 0) { printf("GPU device memory problem %d with mb_mask\n", device.state_gpu); return -1; }
		device.segments_data = clCreateBuffer(device.context_gpu, CL_MEM_READ_ONLY, sizeof(segment_data)*4, NULL , &device.state_gpu);
		if (device.state_gpu != 0) { printf("GPU device memory problem %d with segments_data\n", device.state_gpu); return -1; }
	}
	device.transformed_blocks_cpu = clCreateBuffer(device.context_cpu, CL_MEM_READ_WRITE, sizeof(macroblock)*video.mb_count, NULL , &device.state_cpu);
	device.partitions = clCreateBuffer(device.context_cpu, CL_MEM_READ_WRITE, video.partition_step, NULL , &device.state_cpu);
	device.partitions_sizes = clCreateBuffer(device.context_cpu, CL_MEM_READ_WRITE, 8*sizeof(int32_t), NULL , &device.state_cpu);
	device.third_context = clCreateBuffer(device.context_cpu, CL_MEM_READ_WRITE, sizeof(uint8_t)*25*video.mb_count, NULL , &device.state_cpu);
	device.coeff_probs = clCreateBuffer(device.context_cpu, CL_MEM_READ_WRITE, 8*4*8*3*11*sizeof(uint32_t), NULL , &device.state_cpu);
	device.coeff_probs_denom = clCreateBuffer(device.context_cpu, CL_MEM_READ_WRITE, 8*4*8*3*11*sizeof(uint32_t), NULL , &device.state_cpu);

	if (video.GOP_size > 1) {
		/*__kernel void reset_vectors ( __global vector_net *const net1, //0
								__global vector_net *const net2, //1
								__global macroblock *const MBs, //2
								const int ref) //3*/
		device.state_gpu = clSetKernelArg(device.reset_vectors, 0, sizeof(cl_mem), &device.vnet1);
		device.state_gpu = clSetKernelArg(device.reset_vectors, 1, sizeof(cl_mem), &device.vnet2);
		device.state_gpu = clSetKernelArg(device.reset_vectors, 2, sizeof(cl_mem), &device.transformed_blocks_gpu);

		/*__kernel void downsample(__global uchar *const src_frame, //0
							__global uchar *const dst_frame, //1
							const signed int src_width, //2
							const signed int src_height) //3*/
		// all parameters variable

		/*__kernel void luma_search_1step //when looking into downsampled and original frames
						( 	__global uchar *const current_frame, //0
							__global uchar *const prev_frame, //1
							__global vector_net *const src_net, //2
							__global vector_net *const dst_net, //3
							const signed int net_width, //4 //in 8x8 blocks
							const signed int width, //5
							const signed int height, //6
							const signed int pixel_rate) //7 */

		//frames and nets are set according to downsampling rate
		int32_t net_width = video.mb_width*2;
		device.state_gpu = clSetKernelArg(device.luma_search_1step, 4, sizeof(int32_t), &net_width);
		// sizes, pixel_rate will be variable on kernel launch

		/*__kernel void luma_search_2step //searching in interpolated picture
						( 	__global uchar *const current_frame, //0
							__global uchar *const prev_frame, //1
							__global vector_net *const net, //2
							__global macroblock *const MBs, //3
							__global macroblock_metrics *const MBdiff, //4
							const signed int width, //5
							const signed int height) //6 */

		device.state_gpu = clSetKernelArg(device.luma_search_2step, 0, sizeof(cl_mem), &device.current_frame_Y);
	    device.state_gpu = clSetKernelArg(device.luma_search_2step, 1, sizeof(cl_mem), &device.ref_frame_Y_interpolated);
		device.state_gpu = clSetKernelArg(device.luma_search_2step, 2, sizeof(cl_mem), &device.vnet2);
		device.state_gpu = clSetKernelArg(device.luma_search_2step, 3, sizeof(cl_mem), &device.transformed_blocks_gpu);
		device.state_gpu = clSetKernelArg(device.luma_search_2step, 4, sizeof(cl_mem), &device.mb_metrics);
		device.state_gpu = clSetKernelArg(device.luma_search_2step, 5, sizeof(int32_t), &video.wrk_width);
		device.state_gpu = clSetKernelArg(device.luma_search_2step, 6, sizeof(int32_t), &video.wrk_height);

		/*__kernel void try_another_reference(__global uchar *const current_frame_Y, //0
								__global uchar *const ref_frame_Y, //1
								__global uchar *const recon_frame_Y, //2
								__global uchar *const current_frame_U, //3
								__global uchar *const ref_frame_U, //4
								__global uchar *const recon_frame_U, //5
								__global uchar *const current_frame_V, //6
								__global uchar *const ref_frame_V, //7
								__global uchar *const recon_frame_V, //8
								__global macroblock *const MBs, //9
								__global int *const MBdiff, //10
								const int width, //11
								__constant segment_data *const SD, //12
								const int ref) //13*/

		device.state_gpu = clSetKernelArg(device.try_another_reference, 0, sizeof(cl_mem), &device.current_frame_Y);
	    device.state_gpu = clSetKernelArg(device.try_another_reference, 2, sizeof(cl_mem), &device.reconstructed_frame_Y);
		device.state_gpu = clSetKernelArg(device.try_another_reference, 3, sizeof(cl_mem), &device.current_frame_U);
	    device.state_gpu = clSetKernelArg(device.try_another_reference, 5, sizeof(cl_mem), &device.reconstructed_frame_U);
		device.state_gpu = clSetKernelArg(device.try_another_reference, 6, sizeof(cl_mem), &device.current_frame_V);
	    device.state_gpu = clSetKernelArg(device.try_another_reference, 8, sizeof(cl_mem), &device.reconstructed_frame_V);
		device.state_gpu = clSetKernelArg(device.try_another_reference, 9, sizeof(cl_mem), &device.transformed_blocks_gpu);
		device.state_gpu = clSetKernelArg(device.try_another_reference, 10, sizeof(cl_mem), &device.mb_metrics);
		device.state_gpu = clSetKernelArg(device.try_another_reference, 11, sizeof(int32_t), &video.wrk_width);
		device.state_gpu = clSetKernelArg(device.try_another_reference, 12, sizeof(cl_mem), &device.segments_data);


		/*__kernel void luma_transform_16x16(__global uchar *const current_frame, //0
								__global uchar *const recon_frame, //1
								__global uchar *const prev_frame, //2
								__global macroblock *const MBs, //3
								const int width, //4
								__constant segment_data *const SD, //5
								const int segment_id, //6
								const float SSIM_target) //7*/

		device.state_gpu = clSetKernelArg(device.luma_transform_8x8, 0, sizeof(cl_mem), &device.current_frame_Y);
	    device.state_gpu = clSetKernelArg(device.luma_transform_8x8, 1, sizeof(cl_mem), &device.reconstructed_frame_Y);
		device.state_gpu = clSetKernelArg(device.luma_transform_8x8, 2, sizeof(cl_mem), &device.ref_frame_Y_interpolated);
		device.state_gpu = clSetKernelArg(device.luma_transform_8x8, 3, sizeof(cl_mem), &device.transformed_blocks_gpu);
		device.state_gpu = clSetKernelArg(device.luma_transform_8x8, 4, sizeof(int32_t), &video.wrk_width);
		device.state_gpu = clSetKernelArg(device.luma_transform_8x8, 5, sizeof(cl_mem), &device.segments_data);

		/*__kernel void luma_transform_8x8(__global uchar *const current_frame, //0
								__global uchar *const recon_frame, //1
								__global uchar *const prev_frame, //2
								__global macroblock *const MBs, //3
								const signed int width, //4
								__constant segment_data *const SD, //5
								const int segment_id, //6
								const float SSIM_target) //7*/

		device.state_gpu = clSetKernelArg(device.luma_transform_16x16, 0, sizeof(cl_mem), &device.current_frame_Y);
	    device.state_gpu = clSetKernelArg(device.luma_transform_16x16, 1, sizeof(cl_mem), &device.reconstructed_frame_Y);
		device.state_gpu = clSetKernelArg(device.luma_transform_16x16, 2, sizeof(cl_mem), &device.ref_frame_Y_interpolated);
		device.state_gpu = clSetKernelArg(device.luma_transform_16x16, 3, sizeof(cl_mem), &device.transformed_blocks_gpu);
		device.state_gpu = clSetKernelArg(device.luma_transform_16x16, 4, sizeof(int32_t), &video.wrk_width);
		device.state_gpu = clSetKernelArg(device.luma_transform_16x16, 5, sizeof(cl_mem), &device.segments_data);

		/*__kernel void chroma_transform( 	__global uchar *const current_frame, //0 
									__global uchar *const prev_frame, //1 
									__global uchar *const recon_frame, //2 
									__global macroblock *const MBs, //3
									const signed int chroma_width, //4 
									const signed int chroma_height, //5 
									__constant segments_data *const SD) //6
									const int block_place) //7*/
		
		int32_t chroma_width = video.wrk_width/2;
		int32_t chroma_height = video.wrk_height/2;
		// first 3 params and block_place a switched between U and V when kernels are being launched
		device.state_gpu = clSetKernelArg(device.chroma_transform, 3, sizeof(cl_mem), &device.transformed_blocks_gpu);
		device.state_gpu = clSetKernelArg(device.chroma_transform, 4, sizeof(int32_t), &chroma_width); // width
		device.state_gpu = clSetKernelArg(device.chroma_transform, 5, sizeof(int32_t), &chroma_height); // height
		device.state_gpu = clSetKernelArg(device.chroma_transform, 6, sizeof(cl_mem), &device.segments_data);

		/*__kernel luma_interpolate_Hx4(__global uchar *const src_frame, //0
										__global uchar *const dst_frame, //1
										const int width, //2
										const int height) //3*/

		device.state_gpu = clSetKernelArg(device.luma_interpolate_Hx4, 1, sizeof(cl_mem), &device.ref_frame_Y_interpolated);
		device.state_gpu = clSetKernelArg(device.luma_interpolate_Hx4, 2, sizeof(int32_t), &video.wrk_width);
		device.state_gpu = clSetKernelArg(device.luma_interpolate_Hx4, 3, sizeof(int32_t), &video.wrk_height);

		/*__kernel luma_interpolate_Vx4(__global uchar *const frame, //0
										const int width, //1
										const int height) //2*/

		device.state_gpu = clSetKernelArg(device.luma_interpolate_Vx4, 0, sizeof(cl_mem), &device.ref_frame_Y_interpolated);
		device.state_gpu = clSetKernelArg(device.luma_interpolate_Vx4, 1, sizeof(int32_t), &video.wrk_width);
		device.state_gpu = clSetKernelArg(device.luma_interpolate_Vx4, 2, sizeof(int32_t), &video.wrk_height);

		// interpolation functions have suffix "bc" for bicubic filter and "bl" for bilinear
		// interface exactlyy the same

		/*__kernel void chroma_interpolate_Hx8(	__global uchar *const src_frame, //0
												__global uchar *const dst_frame, //1
												const int width, //2
												const int height) //3*/

		device.state_gpu = clSetKernelArg(device.chroma_interpolate_Hx8, 2, sizeof(int32_t), &chroma_width);
		device.state_gpu = clSetKernelArg(device.chroma_interpolate_Hx8, 3, sizeof(int32_t), &chroma_height);

		/*__kernel void chroma_interpolate_Vx8( __global uchar *const frame, //0
												const int width, //1
												const int height) //2*/
		
		device.state_gpu = clSetKernelArg(device.chroma_interpolate_Vx8, 1, sizeof(int32_t), &chroma_width);
		device.state_gpu = clSetKernelArg(device.chroma_interpolate_Vx8, 2, sizeof(int32_t), &chroma_height);

		/*__kernel prepare_filter_mask(__global macroblock *const MBs,
								__global int *const mb_mask)*/

		device.state_gpu = clSetKernelArg(device.prepare_filter_mask, 0, sizeof(cl_mem), &device.transformed_blocks_gpu);
		device.state_gpu = clSetKernelArg(device.prepare_filter_mask, 1, sizeof(cl_mem), &device.mb_mask);

		/*__kernel void normal_loop_filter_MBH(__global uchar * const frame, //0
									const int width, //1
									__constant segment_data *const SD, //2
									__global macroblock *const MB, //3
									const int mb_size, //4
									const int stage, //5
									__global int *const mb_mask) //6*/

		device.state_gpu = clSetKernelArg(device.normal_loop_filter_MBH, 2, sizeof(cl_mem), &device.segments_data);
		device.state_gpu = clSetKernelArg(device.normal_loop_filter_MBH, 3, sizeof(cl_mem), &device.transformed_blocks_gpu);
		device.state_gpu = clSetKernelArg(device.normal_loop_filter_MBH, 6, sizeof(cl_mem), &device.mb_mask);

		/*__kernel void normal_loop_filter_MBV(__global uchar * const frame, //0
									const int width, //1
									__constant segment_data *const SD, //2
									__global macroblock *const MB, //3
									const int mb_size, //4
									const int stage, //5
									__global int *const mb_mask) //6*/
		device.state_gpu = clSetKernelArg(device.normal_loop_filter_MBV, 2, sizeof(cl_mem), &device.segments_data);
		device.state_gpu = clSetKernelArg(device.normal_loop_filter_MBV, 3, sizeof(cl_mem), &device.transformed_blocks_gpu);
		device.state_gpu = clSetKernelArg(device.normal_loop_filter_MBV, 6, sizeof(cl_mem), &device.mb_mask);

		/*__kernel void count_SSIM_luma
						(__global uchar *frame1, //0
						__global uchar *frame2, //1
						__global macroblock *MBs, //2
                        signed int width, //3
						const int segment_id)// 4	*/

		device.state_gpu = clSetKernelArg(device.count_SSIM_luma, 0, sizeof(cl_mem), &device.current_frame_Y);
		device.state_gpu = clSetKernelArg(device.count_SSIM_luma, 1, sizeof(cl_mem), &device.reconstructed_frame_Y);
		device.state_gpu = clSetKernelArg(device.count_SSIM_luma, 2, sizeof(cl_mem), &device.transformed_blocks_gpu);
		device.state_gpu = clSetKernelArg(device.count_SSIM_luma, 3, sizeof(int32_t), &video.wrk_width);

		/*__kernel void count_SSIM_chroma
						(__global uchar *frame1, //0
						__global uchar *frame2, //1
						__global macroblock *MBs, //2
                        signed int width, //3
						const int segment_id)// 4	*/

		device.state_gpu = clSetKernelArg(device.count_SSIM_chroma, 2, sizeof(cl_mem), &device.transformed_blocks_gpu);
		device.state_gpu = clSetKernelArg(device.count_SSIM_chroma, 3, sizeof(int32_t), &video.wrk_width);

		device.commandQueue_gpu = clCreateCommandQueue(device.context_gpu, device.device_gpu[0], 0, &device.state_gpu);
	}

	/*__kernel void encode_coefficients(	__global macroblock *MBs, - 0
											__global uchar *output, - 1
											__global int *partition_sizes, - 2
											__global uchar *third_context, - 3
											__global uint *coeff_probs, - 4
											__global uint *coeff_probs_denom, - 5
											int mb_height, - 6
											int mb_width, - 7
											int num_partitions, - 8
											int key_frame, - 9
											int partition_step, - 10
											int skip_prob) - 11 */
	
	device.state_cpu = clSetKernelArg(device.encode_coefficients, 0, sizeof(cl_mem), &device.transformed_blocks_cpu);
	device.state_cpu = clSetKernelArg(device.encode_coefficients, 1, sizeof(cl_mem), &device.partitions);
	device.state_cpu = clSetKernelArg(device.encode_coefficients, 2, sizeof(cl_mem), &device.partitions_sizes);
	device.state_cpu = clSetKernelArg(device.encode_coefficients, 3, sizeof(cl_mem), &device.third_context);
	device.state_cpu = clSetKernelArg(device.encode_coefficients, 4, sizeof(cl_mem), &device.coeff_probs);
	device.state_cpu = clSetKernelArg(device.encode_coefficients, 5, sizeof(cl_mem), &device.coeff_probs_denom);
	device.state_cpu = clSetKernelArg(device.encode_coefficients, 6, sizeof(int32_t), &video.mb_height);
	device.state_cpu = clSetKernelArg(device.encode_coefficients, 7, sizeof(int32_t), &video.mb_width);
	device.state_cpu = clSetKernelArg(device.encode_coefficients, 8, sizeof(int32_t), &video.number_of_partitions);
	// 9 before launch each time
	video.partition_step = video.partition_step / video.number_of_partitions;
	device.state_cpu = clSetKernelArg(device.encode_coefficients, 10, sizeof(int32_t), &video.partition_step);
	// 10 different for each frame

	/*__kernel void count_probs(	__global macroblock *MBs, - 0
									__global uint *coeff_probs, - 1
									__global uint *coeff_probs_denom, - 2
									__global uchar *third_context, - 3
									int mb_height, - 4
									int mb_width, - 5
									int num_partitions, - 6
									int key_frame, - 7
									int partition_step) - 8 */
	
	device.state_cpu = clSetKernelArg(device.count_probs, 0, sizeof(cl_mem), &device.transformed_blocks_cpu);
	device.state_cpu = clSetKernelArg(device.count_probs, 1, sizeof(cl_mem), &device.coeff_probs);
	device.state_cpu = clSetKernelArg(device.count_probs, 2, sizeof(cl_mem), &device.coeff_probs_denom);
	device.state_cpu = clSetKernelArg(device.count_probs, 3, sizeof(cl_mem), &device.third_context);
	device.state_cpu = clSetKernelArg(device.count_probs, 4, sizeof(int32_t), &video.mb_height);
	device.state_cpu = clSetKernelArg(device.count_probs, 5, sizeof(int32_t), &video.mb_width);
	device.state_cpu = clSetKernelArg(device.count_probs, 6, sizeof(int32_t), &video.number_of_partitions);
	// 7 before launch each time
	device.state_cpu = clSetKernelArg(device.count_probs, 8, sizeof(int32_t), &video.partition_step);

	/*__kernel void num_div_denom(	__global uint *coeff_probs, 
									__global uint *coeff_probs_denom,
									int num_partitions)*/
	device.state_cpu = clSetKernelArg(device.num_div_denom, 0, sizeof(cl_mem), &device.coeff_probs);
	device.state_cpu = clSetKernelArg(device.num_div_denom, 1, sizeof(cl_mem), &device.coeff_probs_denom);
	device.state_cpu = clSetKernelArg(device.num_div_denom, 2, sizeof(int32_t), &video.number_of_partitions);

	device.commandQueue_cpu = clCreateCommandQueue(device.context_cpu, device.device_cpu[0], 0, &device.state_cpu);
	return 1;
}

int string_to_value(char *str)
{
	int i = 0, new_digit, retval = 0;
	while ((str[i] != '\n') && (str[i] != '\0'))
	{
		new_digit = (int)(str[i]) - 48;
		if ((new_digit > 9) || (new_digit < 0))
			return -1;
		retval *= 10;
		retval += new_digit;
		++i;
	}
	return retval;
}

int ParseArgs(int argc, char *argv[])
{
    char f_o = 0, f_i = 0, f_qmax = 0, f_qmin = 0, f_qintra = 0, f_g = 0, f_w = 0, f_t = 0, f_ls = 0,f_SSIM_target=0,f_altref_range=0; 
	int i,ii;
    i = 1;
    while (i < argc)
    {
		ii = i;
        if (argv[i][0] == '-')
        {
			if ((argv[i][1] == 'h') && ((argv[i][2] == '\n') || (argv[i][2] == '\0')))
			{
				printf("%s\n", small_help);
				return -1;
			}
            if ((argv[i][1] == 'i') && ((argv[i][2] == '\n') || (argv[i][2] == '\0')))
            {
                ++i;
                if (i < argc)
                {
					input_file.path = argv[i];
                    f_i = 1;
                    printf("input file : %s;\n", input_file.path);
					if (++i >= argc) break;
                }
                else
                {
                    printf ("no value for YUV input;\n");
                    return -1;
                }
            }
            if ((argv[i][1] == 'o')&& ((argv[i][2] == '\n') || (argv[i][2] == '\0')))
            {
                ++i;
                if (i < argc)
                {
					output_file.path = argv[i];
                    f_o = 1;
                    printf("output file : %s;\n", output_file.path);
					if (++i >= argc) break;
                }
                else
                {
                    printf ("no destination for output;\n");
                    return -1;
                }
            }
            //if ((argv[i][1] == 'q') && (argv[i][2] == 'i') && ((argv[i][3] == '\n') || (argv[i][3] == '\0')))
			if (memcmp(&argv[i][1], "qmax", 4)==0)
            {
                ++i;
                if (i < argc)
                {
					video.qi_max = string_to_value(argv[i]);
					if (video.qi_max < 0)
					{
						printf ("wrong quantizer index format for intra i-frames! must be an integer from 0 to 127;\n");
						return -1;
					}
                    f_qmax = 1;
					if (++i >= argc) break;
                }
                else
                {
                    printf ("no value for quantizer;\n");
                    return -1;
                }
            }
			//if ((argv[i][1] == 'q') && (argv[i][2] == 'p') && ((argv[i][3] == '\n') || (argv[i][3] == '\0')))
			if (memcmp(&argv[i][1], "qmin", 4)==0)
            {
                ++i;
                if (i < argc)
                {
					video.qi_min = string_to_value(argv[i]);
					if (video.qi_min < 0)
					{
						printf ("wrong quantizer index format for inter p-frames! must be an integer from 0 to 127;\n");
						return -1;
					}
                    f_qmin = 1;
					if (++i >= argc) break;
                }
                else
                {
                    printf ("no value for quantizer;\n");
                    return -1;
                }
            }
			if (memcmp(&argv[i][1], "altref-range", 12)==0)
            {
                ++i;
                if (i < argc)
                {
					video.altref_range = string_to_value(argv[i]);
					if (video.altref_range < 0)
					{
						printf ("wrong altref range format\n");
						return -1;
					}
                    f_altref_range = 1;
					if (++i >= argc) break;
                }
                else
                {
                    printf ("no value for altref range;\n");
                    return -1;
                }
            }
            if ((argv[i][1] == 'g') && ((argv[i][2] == '\n') || (argv[i][2] == '\0')))
            {
                ++i;
                if (i < argc)
                {
					video.GOP_size = string_to_value(argv[i]);
					if (video.GOP_size < 1)
					{
						printf ("wrong GOP format! must be an integer from 1 to even more;\n");
						return -1;
					}
                    f_g = 1;
					if (++i >= argc) break;
                }
                else
                {
                    printf ("no value for group of pictures size;\n");
                    return -1;
                }
            }
            if ((argv[i][1] == 'w') && ((argv[i][2] == '\n') || (argv[i][2] == '\0')))
            {
                ++i;
                if (i < argc)
                {
					device.gpu_work_items_limit = (size_t)string_to_value(argv[i]);
					if (device.gpu_work_items_limit < 0)
					{
						printf ("wrong GPU work items format! must be an integer from 1, but better be a thousand, or two... or ten;\n");
						return -1;
					}
                    f_w = 1;
					if (++i >= argc) break;
                }
                else
                {
                    printf ("no value for GPU work items\n");
                    return -1;
                }
            }
            if ((argv[i][1] == 't') && ((argv[i][2] == '\n') || (argv[i][2] == '\0')))
            {
                ++i;
                if (i < argc)
                {
					video.number_of_partitions = (size_t)string_to_value(argv[i]);
					if (video.number_of_partitions < 0)
					{
						printf ("wrong CPU work items format! input values: 1 2 4 8;\n");
						return -1;
					}
					if ((video.number_of_partitions != 1) 
						&& (video.number_of_partitions != 2) 
						&& (video.number_of_partitions != 4) 
						&& (video.number_of_partitions != 8))
						video.number_of_partitions = 2;
                    f_t = 1;
					if (++i >= argc) break;
                }
                else
                {
                    printf ("no value for CPU work items;\n");
                    return -1;
                }
            }
			if ((argv[i][1] == 'l') && (argv[i][2] == 's') && ((argv[i][3] == '\n') || (argv[i][3] == '\0')))
            {
                ++i;
                if (i < argc)
                {
					video.loop_filter_sharpness = string_to_value(argv[i]);
					if (video.loop_filter_sharpness < 0)
					{
						printf ("wrong format for loop_filter_sharpness! must be an integer from 0 to 63;\n");
						return -1;
					} else 
						video.loop_filter_sharpness = (video.loop_filter_sharpness > 7) ? 7 : video.loop_filter_sharpness;
                    f_ls = 1;
					if (++i >= argc) break;
                }
                else
                {
                    printf ("no value for loop_filter_sharpness;\n");
                    return -1;
                }
            }
			if (memcmp(&argv[i][1], "SSIM-target", 11)==0)
            {
                ++i;
                if (i < argc)
                {
					int buf = string_to_value(argv[i]);
					if ((buf < 0) || (buf > 97))
					{
						printf ("wrong quantizer index format for inter p-frames! must be an integer from 0 to 97;\n");
						return -1;
					}
                    f_SSIM_target = 1;
					video.SSIM_target = ((float)buf)/100.0f;
					if (++i >= argc) break;
                }
                else
                {
                    printf ("no value for SSIM-target;\n");
                    return -1;
                }
            }
		}    
		if (i == ii)
		{
			printf("unknown option %s\n",argv[i]);
			return -1;
		}
	}
    if (f_i == 0)
    {
		printf("no input file specified;\n");
		return -1;
    }
    if (f_o == 0)
    {
		printf("no output file specified;\n");
		return -1;
    }
	if (f_qmax == 0)
    {
		printf("no min quantizer index -> set to 0;\n");
		video.qi_min = 0;
    }
	if (f_qmin == 0)
    {
		printf("no max quantizer index -> set to 48;\n");
		video.qi_max = 48;
    }
	if (f_g == 0)
    {
		printf("no size of group of pictures specified - set to 150;\n");
		video.GOP_size = 150;
    }
	if (f_w == 0)
    {
		printf("no amount of work items on GPU specified - set to 1024;\n");
		device.gpu_work_items_limit = 1024;
    }
	if (f_t == 0)
    {
		printf("no amount of work items on CPU specified - set to 4;\n");
		video.number_of_partitions = 4;
    }
	if (f_ls == 0)
    {
		printf("no loop filter sharpness is specified - set to 0;\n");
		video.loop_filter_sharpness = 0;
    }
	if (f_SSIM_target == 0)
	{
		printf("no SSIM intra-in-inter tuning");
		video.SSIM_target = -1.0f;
	}
	if (f_altref_range == 0)
	{
		printf("no altref range value, set to default %d\n",DEFAULT_ALTREF_RANGE);
		video.altref_range = DEFAULT_ALTREF_RANGE;
	}
	video.loop_filter_type = 0; //this is fixed now

	if (video.qi_max < video.qi_min) 
	{
		printf("wrong quantizer min-max range -> swap \n");
		int32_t buf = video.qi_max;
		video.qi_max = video.qi_min;
		video.qi_min = buf;
		
	}
	video.lastqi[UQ_segment] = (video.qi_max + video.qi_min*3 + 2)/4;
	video.lastqi[HQ_segment] = (video.qi_max + video.qi_min + 1)/2;
	video.lastqi[AQ_segment] = (video.qi_max*3 + video.qi_min + 2)/4;
	video.lastqi[LQ_segment] = video.qi_max;

	video.altrefqi[UQ_segment] = video.lastqi[UQ_segment]/4;
	video.altrefqi[HQ_segment] = video.lastqi[HQ_segment]/3;
	video.altrefqi[AQ_segment] = video.lastqi[AQ_segment]/3;
	video.altrefqi[LQ_segment] = video.lastqi[LQ_segment]/2;

	video.altrefqi[UQ_segment] = (video.altrefqi[UQ_segment] < video.qi_min) ? video.qi_min : video.altrefqi[UQ_segment];
    return 0;
}

int OpenYUV420FileAndParseHeader()
{
    // add framerate
	char magic_word[] = "YUV4MPEG2 ";
	char ch;
	int frame_start = 0;
	if (input_file.path[0] == '@') {
		input_file.handle = stdin;
		setmode(0, O_BINARY); //0x8000
	}
	else 
		input_file.handle = fopen(input_file.path,"rb");
	output_file.handle = fopen(output_file.path,"wb");
	int i = 0, j = 0;
	video.src_height = 0; video.src_width = 0;

	for (i = 0; i < 10; ++i)
	{
		if (!fread(&ch, sizeof(char), 1, input_file.handle)) 
			return -1;
		frames.header[j++] = ch;
		if (ch != magic_word[i]) 
			return -1;
	}
	for (i = 0; i < 3; ++i)
	{
		while ((ch != 'W') && (ch != 'H') && (ch != 'F')) {
			if (!fread(&ch, sizeof(char), 1, input_file.handle)) 
				return -1 ;
			frames.header[j++] = ch;
		}
		if (ch == 'W')
		{
			while (1)
			{
				if (!fread(&ch, sizeof(char), 1, input_file.handle)) 
					return -1 ;
				frames.header[j++] = ch;
				if (ch == 0x20) break;
				video.src_width*=10;
				video.src_width+=(ch - 0x30);
			}
		}
		if (ch == 'H')
		{
			while (1)
			{
				if (!fread(&ch, sizeof(char), 1, input_file.handle)) 
					return -1 ;
				frames.header[j++] = ch;
				if (ch == 0x20) break;
				video.src_height*=10;
				video.src_height+=(ch - 0x30);
			}
		}
		if (ch == 'F')
		{
			int denom = 0, num = 0;
			while (1)
			{
				if (!fread(&ch, sizeof(char), 1, input_file.handle)) 
					return -1 ;
				frames.header[j++] = ch;
				if (ch == ':') break;
				num*=10;
				num+=(ch - 0x30);
			}
			while (1)
			{
				if (!fread(&ch, sizeof(char), 1, input_file.handle)) 
					return -1 ;
				frames.header[j++] = ch;
				if (ch == 0x20) break;
				denom*=10;
				denom+=(ch - 0x30);
			}
			video.framerate = (num+denom/2)/denom;
		}
	}

	if ((video.src_width + video.src_height) == 0)
		return -1;

	while (1)
	{
		while (ch != 'F') 
		{
			if (!fread(&ch, sizeof(char), 1, input_file.handle)) 
				return -1 ;
			frames.header[j++] = ch;
		}
		if (!fread(&ch, sizeof(char), 1, input_file.handle)) 
			return -1 ;
		frames.header[j++] = ch;
		if (ch != 'R') 
			continue;
		if (!fread(&ch, sizeof(char), 1, input_file.handle)) 
			return -1 ;
		frames.header[j++] = ch;
		if (ch != 'A') 
			continue;
		if (!fread(&ch, sizeof(char), 1, input_file.handle)) 
			return -1 ;
		frames.header[j++] = ch;
		if (ch != 'M') 
			continue;
		if (!fread(&ch, sizeof(char), 1, input_file.handle)) 
			return -1 ;
		frames.header[j++] = ch;
		if (ch != 'E') 
			continue;
		if (!fread(&ch, sizeof(char), 1, input_file.handle)) 
			return -1 ;
		frames.header[j++] = ch;
		if (ch != 0x0A) 
			return -1;
		break;
	}

	video.dst_width = video.src_width; //output of the same size for now
	video.dst_height = video.src_height;
	frames.header_sz = j - 6; //exclude FRAME<0x0A> from header
	frames.header[frames.header_sz] = 0;
	printf("%d:%s", frames.header_sz, frames.header);
	return 0;
}
