// string to print when "-h" option is met
char small_help[] = "\n"
					"-i\t:\tinput file path\n"
					"-o\t:\toutput file path\n"
					"-qi\t:\tquantizer index, one for all intra-color planes\n"
					"-qp\t:\tquantizer index, one for all inter-color planes\n"
					"-g\t:\tGroup of Pictures size\n"
					"-w\t:\tamount of work-items on gpu launched simutaneosly\n"
					"-t\t:\tamount of partitions for boolean encoding (also threads launched at once)\n"
					"-vx\t:\tmaximum distance for X vector search (in 1/4 pixels)\n"
					"-vy\t:\tmaximum distance for Y vector search (in 1/4 pixels)\n"
					"-lt\t:\tloop filter type (0 - normal; 1 - simple)\n"
					"-ll\t:\tloop filter level (0 - disable)\n"
					"-ls\t:\tloop filter sharpness\n"
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
		const char gpu_options[] = "-cl-std=CL1.0";
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
		{ char kernel_name[] = "luma_search";
		device.luma_search = clCreateKernel(device.program_gpu, kernel_name, &device.state_gpu); }
		{ char kernel_name[] = "luma_transform";
		device.luma_transform = clCreateKernel(device.program_gpu, kernel_name, &device.state_gpu); }
		{ char kernel_name[] = "chroma_transform";
		device.chroma_transform = clCreateKernel(device.program_gpu, kernel_name, &device.state_gpu); }
		{ char kernel_name[] = "luma_interpolate_Hx4_bl";
		device.luma_interpolate_Hx4 = clCreateKernel(device.program_gpu, kernel_name, &device.state_gpu); }
		{ char kernel_name[] = "luma_interpolate_Vx4_bl";
		device.luma_interpolate_Vx4 = clCreateKernel(device.program_gpu, kernel_name, &device.state_gpu); }
		{ char kernel_name[] = "chroma_interpolate_Hx8_bl";
		device.chroma_interpolate_Hx8 = clCreateKernel(device.program_gpu, kernel_name, &device.state_gpu); }
		{ char kernel_name[] = "chroma_interpolate_Vx8_bl";
		device.chroma_interpolate_Vx8 = clCreateKernel(device.program_gpu, kernel_name, &device.state_gpu); }
		{ char kernel_name[] = "simple_loop_filter_MBH";
		device.simple_loop_filter_MBH = clCreateKernel(device.program_gpu, kernel_name, &device.state_gpu); }
		{ char kernel_name[] = "simple_loop_filter_MBV";
		device.simple_loop_filter_MBV = clCreateKernel(device.program_gpu, kernel_name, &device.state_gpu); }
		{ char kernel_name[] = "normal_loop_filter_MBH";
		device.normal_loop_filter_MBH = clCreateKernel(device.program_gpu, kernel_name, &device.state_gpu); }
		{ char kernel_name[] = "normal_loop_filter_MBV";
		device.normal_loop_filter_MBV = clCreateKernel(device.program_gpu, kernel_name, &device.state_gpu); }
		{ char kernel_name[] = "count_SSIM";
		device.count_SSIM = clCreateKernel(device.program_gpu, kernel_name, &device.state_gpu); }
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
		device.current_frame_U = clCreateBuffer(device.context_gpu, CL_MEM_READ_ONLY, video.wrk_frame_size_chroma, NULL , &device.state_gpu);
		device.current_frame_V = clCreateBuffer(device.context_gpu, CL_MEM_READ_ONLY, video.wrk_frame_size_chroma, NULL , &device.state_gpu);
		device.last_frame_Y = clCreateBuffer(device.context_gpu, CL_MEM_READ_WRITE, 16*video.wrk_frame_size_luma, NULL , &device.state_gpu);
		device.last_frame_U = clCreateBuffer(device.context_gpu, CL_MEM_READ_WRITE, 64*video.wrk_frame_size_chroma, NULL , &device.state_gpu);
		device.last_frame_V = clCreateBuffer(device.context_gpu, CL_MEM_READ_WRITE, 64*video.wrk_frame_size_chroma, NULL , &device.state_gpu);
		device.reconstructed_frame_Y = clCreateBuffer(device.context_gpu, CL_MEM_READ_WRITE, video.wrk_frame_size_luma, NULL , &device.state_gpu);
		device.reconstructed_frame_U = clCreateBuffer(device.context_gpu, CL_MEM_READ_WRITE, video.wrk_frame_size_chroma, NULL , &device.state_gpu);
		device.reconstructed_frame_V = clCreateBuffer(device.context_gpu, CL_MEM_READ_WRITE, video.wrk_frame_size_chroma, NULL , &device.state_gpu);
		device.transformed_blocks_gpu = clCreateBuffer(device.context_gpu, CL_MEM_READ_WRITE, sizeof(macroblock)*video.mb_count, NULL , &device.state_gpu);
	}
	device.transformed_blocks_cpu = clCreateBuffer(device.context_cpu, CL_MEM_READ_WRITE, sizeof(macroblock)*video.mb_count, NULL , &device.state_cpu);
	device.partitions = clCreateBuffer(device.context_cpu, CL_MEM_READ_WRITE, video.partition_step, NULL , &device.state_cpu);
	device.partitions_sizes = clCreateBuffer(device.context_cpu, CL_MEM_READ_WRITE, 8*sizeof(int32_t), NULL , &device.state_cpu);
	device.third_context = clCreateBuffer(device.context_cpu, CL_MEM_READ_WRITE, sizeof(uint8_t)*25*video.mb_count, NULL , &device.state_cpu);
	device.coeff_probs = clCreateBuffer(device.context_cpu, CL_MEM_READ_WRITE, 8*4*8*3*11*sizeof(uint32_t), NULL , &device.state_cpu);
	device.coeff_probs_denom = clCreateBuffer(device.context_cpu, CL_MEM_READ_WRITE, 8*4*8*3*11*sizeof(uint32_t), NULL , &device.state_cpu);

	if (video.GOP_size > 1) {
	/*__kernel __attribute__((reqd_work_group_size(GROUP_SIZE_FOR_SEARCH, 1, 1)))
		void luma_search( 	__global uchar *const current_frame, //0
							__global uchar *const prev_frame, //1
							__global macroblock *const MBs, //2
							const signed int width, //3
							const signed int height, //4
							const signed int first_MBlock_offset, //5
							const int deltaX, //6
							const int deltaY, //7
							const int dc_q, //8
							const int ac_q) //9*/

		device.state_gpu = clSetKernelArg(device.luma_search, 0, sizeof(cl_mem), &device.current_frame_Y);
	    device.state_gpu = clSetKernelArg(device.luma_search, 1, sizeof(cl_mem), &device.last_frame_Y);
		device.state_gpu = clSetKernelArg(device.luma_search, 2, sizeof(cl_mem), &device.transformed_blocks_gpu);
		device.state_gpu = clSetKernelArg(device.luma_search, 3, sizeof(int32_t), &video.wrk_width);
		device.state_gpu = clSetKernelArg(device.luma_search, 4, sizeof(int32_t), &video.wrk_height);
		// first_block_offset will be variable on kernel launch
		device.state_gpu = clSetKernelArg(device.luma_search, 6, sizeof(int32_t), &video.max_X_vector_length);
		device.state_gpu = clSetKernelArg(device.luma_search, 7, sizeof(int32_t), &video.max_Y_vector_length);

		/*___kernel void luma_transform(__global uchar *current_frame, //0
										__global uchar *recon_frame, //1
										__global uchar *prev_frame, //2
										__global macroblock *MBs, //3
										signed int width, //4
										signed int first_MBlock_offset, //5
										int dc_q, //6
										int ac_q) //7*/


		device.state_gpu = clSetKernelArg(device.luma_transform, 0, sizeof(cl_mem), &device.current_frame_Y);
	    device.state_gpu = clSetKernelArg(device.luma_transform, 1, sizeof(cl_mem), &device.reconstructed_frame_Y);
		device.state_gpu = clSetKernelArg(device.luma_transform, 2, sizeof(cl_mem), &device.last_frame_Y);
		device.state_gpu = clSetKernelArg(device.luma_transform, 3, sizeof(cl_mem), &device.transformed_blocks_gpu);
		device.state_gpu = clSetKernelArg(device.luma_transform, 4, sizeof(int32_t), &video.wrk_width);
		// first_block_offset will be variable on kernel launch

		/*__kernel void chroma_transform(	__global uchar *current_frame, - 0
											__global uchar *prev_frame, - 1
											__global uchar *recon_frame, - 2
											__global struct macroblock *MBs, - 3
											signed int chroma_width, - 4
											signed int chroma_height, - 5
											signed int first_block_offset, - 6
											int dc_q, - 7
											int ac_q, - 8 
											int block_place) - 9 */
		
		int32_t chroma_width = video.wrk_width/2;
		int32_t chroma_height = video.wrk_height/2;
		// first 3 params and block_place a switched between U and V when kernels are being launched
		device.state_gpu = clSetKernelArg(device.chroma_transform, 3, sizeof(cl_mem), &device.transformed_blocks_gpu);
		device.state_gpu = clSetKernelArg(device.chroma_transform, 4, sizeof(int32_t), &chroma_width); // width
		device.state_gpu = clSetKernelArg(device.chroma_transform, 5, sizeof(int32_t), &chroma_height); // height
		// first_block_offset will be variable on kernel launch

		/*__kernel luma_interpolate_Hx4(__global uchar *const src_frame, //0
										__global uchar *const dst_frame, //1
										const int width, //2
										const int height) //3*/

		device.state_gpu = clSetKernelArg(device.luma_interpolate_Hx4, 0, sizeof(cl_mem), &device.reconstructed_frame_Y);
		device.state_gpu = clSetKernelArg(device.luma_interpolate_Hx4, 1, sizeof(cl_mem), &device.last_frame_Y);
		device.state_gpu = clSetKernelArg(device.luma_interpolate_Hx4, 2, sizeof(int32_t), &video.wrk_width);
		device.state_gpu = clSetKernelArg(device.luma_interpolate_Hx4, 3, sizeof(int32_t), &video.wrk_height);

		/*__kernel luma_interpolate_Vx4(__global uchar *const frame, //0
										const int width, //1
										const int height) //2*/

		device.state_gpu = clSetKernelArg(device.luma_interpolate_Vx4, 0, sizeof(cl_mem), &device.last_frame_Y);
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


		/*__kernel void simple_loop_filter_MBH(	__global uchar * const frame, //0
												const int width, // 1
												const int mbedge_limit, // 2
												const int sub_edge_limit, // 3
												const int mb_col) // 4 */

		device.state_gpu = clSetKernelArg(device.simple_loop_filter_MBH, 0, sizeof(cl_mem), &device.reconstructed_frame_Y);
		device.state_gpu = clSetKernelArg(device.simple_loop_filter_MBH, 1, sizeof(int32_t), &video.wrk_width);
		// mb_col is incremented during launches, filter parameters are set before launch

		/*__kernel void simple_loop_filter_MBV(	__global uchar * const frame, //0
												const int width, // 1
												const int mbedge_limit, // 2
												const int sub_edge_limit, // 3
												const int mb_col) // 4 */

		device.state_gpu = clSetKernelArg(device.simple_loop_filter_MBV, 0, sizeof(cl_mem), &device.reconstructed_frame_Y);
		device.state_gpu = clSetKernelArg(device.simple_loop_filter_MBV, 1, sizeof(int32_t), &video.wrk_width);

		/*__kernel void normal_loop_filter_MBH(	__global uchar * const frame, //0
												const int width, //1
												const int mbedge_limit, //2
												const int sub_bedge_limit, //3
												const int interior_limit, //4
												const int hev_threshold, //5
												const int mb_size, //6
												const int mb_col) //7 */

		// no params at init time

		/*__kernel void normal_loop_filter_MBV(	__global uchar * const frame, //0
												const int width, //1
												const int mbedge_limit, //2
												const int sub_bedge_limit, //3
												const int interior_limit, //4
												const int hev_threshold, //5
												const int mb_size, //6
												const int mb_col) //7 */

		// no params at init time

		/*__kernel __attribute__((reqd_work_group_size(64, 1, 1)))
					void count_SSIM(__global uchar *current_frame, //0
									__global uchar *recon_frame, //1
									__global macroblock *MBs, //2
									signed int width, //3
									signed int mb_count)// 4*/

		device.state_gpu = clSetKernelArg(device.count_SSIM, 0, sizeof(cl_mem), &device.current_frame_Y);
		device.state_gpu = clSetKernelArg(device.count_SSIM, 1, sizeof(cl_mem), &device.reconstructed_frame_Y);
		device.state_gpu = clSetKernelArg(device.count_SSIM, 2, sizeof(cl_mem), &device.transformed_blocks_gpu);
		device.state_gpu = clSetKernelArg(device.count_SSIM, 3, sizeof(int32_t), &video.wrk_width);
		device.state_gpu = clSetKernelArg(device.count_SSIM, 4, sizeof(int32_t), &video.mb_count);

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
											int partition_step) - 10 */
	
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
    char f_o = 0, f_i = 0, f_qi = 0, f_qp = 0, f_g = 0, f_w = 0, f_t = 0, f_vx = 0, f_vy = 0, f_lt = 0, f_ll = 0, f_ls = 0; 
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
					++i;
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
					++i;
                }
                else
                {
                    printf ("no destination for output;\n");
                    return -1;
                }
            }
            if ((argv[i][1] == 'q') && (argv[i][2] == 'i') && ((argv[i][3] == '\n') || (argv[i][3] == '\0')))
            {
                ++i;
                if (i < argc)
                {
					video.quantizer_index_y_dc_i = string_to_value(argv[i]);
					if (video.quantizer_index_y_dc_i < 0)
					{
						printf ("wrong quantizer index format for intra i-frames! must be an integer from 0 to 127;\n");
						return -1;
					}
                    f_qi = 1;
					video.quantizer_index_y_ac_i = video.quantizer_index_y_dc_i;
					video.quantizer_index_uv_dc_i = video.quantizer_index_y_dc_i - 15;
					video.quantizer_index_uv_dc_i = (video.quantizer_index_uv_dc_i < 0) ? 0 : video.quantizer_index_uv_dc_i;
					video.quantizer_index_uv_ac_i = video.quantizer_index_y_dc_i - 15;
					video.quantizer_index_uv_ac_i = (video.quantizer_index_uv_ac_i < 0) ? 0 : video.quantizer_index_uv_ac_i;
					video.quantizer_index_y2_dc_i = video.quantizer_index_y_dc_i;
					video.quantizer_index_y2_ac_i = video.quantizer_index_y_dc_i;
					++i;
                }
                else
                {
                    printf ("no value for quantizer;\n");
                    return -1;
                }
            }
			if ((argv[i][1] == 'q') && (argv[i][2] == 'p') && ((argv[i][3] == '\n') || (argv[i][3] == '\0')))
            {
                ++i;
                if (i < argc)
                {
					video.quantizer_index_y_dc_p = string_to_value(argv[i]);
					if (video.quantizer_index_y_dc_p < 0)
					{
						printf ("wrong quantizer index format for inter p-frames! must be an integer from 0 to 127;\n");
						return -1;
					}
                    f_qp = 1;
					video.quantizer_index_y_ac_p = video.quantizer_index_y_dc_p;
					video.quantizer_index_uv_dc_p = video.quantizer_index_y_dc_p - 15;
					video.quantizer_index_uv_dc_p = (video.quantizer_index_uv_dc_p < 0) ? 0 : video.quantizer_index_uv_dc_p;
					video.quantizer_index_uv_ac_p = video.quantizer_index_y_dc_p - 15;
					video.quantizer_index_uv_ac_p = (video.quantizer_index_uv_ac_p < 0) ? 0 : video.quantizer_index_uv_ac_p;
					video.quantizer_index_y2_ac_p = video.quantizer_index_y_dc_p;
					video.quantizer_index_y2_dc_p = video.quantizer_index_y_dc_p;
					++i;
                }
                else
                {
                    printf ("no value for quantizer;\n");
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
					++i;
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
					++i;
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
					++i;
                }
                else
                {
                    printf ("no value for CPU work items;\n");
                    return -1;
                }
            }
            if ((argv[i][1] == 'v') && (argv[i][2] == 'x') && ((argv[i][3] == '\n') || (argv[i][3] == '\0')))
            {
                ++i;
                if (i < argc)
                {
					video.max_X_vector_length = string_to_value(argv[i]);
					if (video.max_X_vector_length < 0)
					{
						printf ("wrong max X vector length format! must be an integer from 0 to 511;\n");
						return -1;
					}
					video.max_X_vector_length = (video.max_X_vector_length > 511) ? 511 : video.max_X_vector_length;
                    f_vx = 1;
					++i;
                }
                else
                {
                    printf ("no value X vector length;\n");
                    return -1;
                }
            }
			if ((argv[i][1] == 'v') && (argv[i][2] == 'y') && ((argv[i][3] == '\n') || (argv[i][3] == '\0')))
            {
                ++i;
                if (i < argc)
                {
					video.max_Y_vector_length = string_to_value(argv[i]);
					if (video.max_Y_vector_length < 0)
					{
						printf ("wrong max X vector length format! must be an integer from 0 to 511;\n");
						return -1;
					}
					video.max_Y_vector_length = (video.max_Y_vector_length > 511) ? 511 : video.max_Y_vector_length;
                    f_vy = 1;
					++i;
                }
                else
                {
                    printf ("no value Y vector length;\n");
                    return -1;
                }
            }
			if ((argv[i][1] == 'l') && (argv[i][2] == 't') && ((argv[i][3] == '\n') || (argv[i][3] == '\0')))
            {
                ++i;
                if (i < argc)
                {
					video.loop_filter_type = string_to_value(argv[i]);
					if ((video.loop_filter_type < 0) || (video.loop_filter_type > 1))
					{
						printf ("wrong loop_filter_type! must be 0 - normal or 1 - simple;\n");
						return -1;
					} 
                    f_lt = 1;
					++i;
                }
                else
                {
                    printf ("no value for loop_filter_type;\n");
                    return -1;
                }
            }
			if ((argv[i][1] == 'l') && (argv[i][2] == 'l') && ((argv[i][3] == '\n') || (argv[i][3] == '\0')))
            {
                ++i;
                if (i < argc)
                {
					video.loop_filter_level = string_to_value(argv[i]);
					if (video.loop_filter_level < 0)
					{
						printf ("wrong format for loop_filter_level! must be an integer from 0 to 63;\n");
						return -1;
					} else 
						video.loop_filter_level = (video.loop_filter_level > 63) ? 63 : video.loop_filter_level;
                    f_ll = 1;
					++i;
                }
                else
                {
                    printf ("no value for loop_filter_level;\n");
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
					++i;
                }
                else
                {
                    printf ("no value for loop_filter_sharpness;\n");
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
	if (f_qi == 0)
    {
		printf("no quantizer index for intra i-frames specified - set to 0;\n");
		video.quantizer_index_y_dc_i = 0;
		video.quantizer_index_uv_ac_i = 0;
		video.quantizer_index_uv_dc_i = 0;
		video.quantizer_index_y2_ac_i = 0;
		video.quantizer_index_y2_dc_i = 0;
		video.quantizer_index_y_ac_i = 0;
    }
	if (f_qp == 0)
    {
		printf("no quantizer index for inter p-frames specified - set to 16;\n");
		video.quantizer_index_y_dc_p = 0;
		video.quantizer_index_uv_ac_p = 0;
		video.quantizer_index_uv_dc_p = 0;
		video.quantizer_index_y2_ac_p = 0;
		video.quantizer_index_y2_dc_p = 0;
		video.quantizer_index_y_ac_p = 0;
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
	if (f_vx == 0)
    {
		printf("no max X vector length specified - set to 0;\n");
		video.max_X_vector_length = 0;
    }
	if (f_vy == 0)
    {
		printf("no max Y vector length specified - set to 0;\n");
		video.max_Y_vector_length = 0;
    }
	if (f_lt == 0)
    {
		printf("no loop filter type is specified - set to 0 (no normal);\n");
		video.loop_filter_type = 0;
    }
	if (f_ll == 0)
    {
		printf("no loop filter level is specified - set to 0 (no filtering);\n");
		video.loop_filter_level = 0;
    }
	if (f_ls == 0)
    {
		printf("no loop filter sharpness is specified - set to 0;\n");
		video.loop_filter_sharpness = 0;
    }
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
	int32_t i = 0;
	video.src_height = 0; video.src_width = 0;

	for (i = 0; i < 10; ++i)
	{
		if (!fread(&ch, sizeof(char), 1, input_file.handle)) 
			return -1;
		printf("%c", ch);
		if (ch != magic_word[i]) 
			return -1;
	}
	printf("\n");

	for (i = 0; i < 3; ++i)
	{
		while ((ch != 'W') && (ch != 'H') && (ch != 'F'))
			if (!fread(&ch, sizeof(char), 1, input_file.handle)) 
				return -1 ;
		if (ch == 'W')
		{
			while (1)
			{
				if (!fread(&ch, sizeof(char), 1, input_file.handle)) 
					return -1 ;
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
				if (ch == ':') break;
				num*=10;
				num+=(ch - 0x30);
			}
			while (1)
			{
				if (!fread(&ch, sizeof(char), 1, input_file.handle)) 
					return -1 ;
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
			if (!fread(&ch, sizeof(char), 1, input_file.handle)) 
				return -1 ;
		if (!fread(&ch, sizeof(char), 1, input_file.handle)) 
			return -1 ;
		if (ch != 'R') 
			continue;
		if (!fread(&ch, sizeof(char), 1, input_file.handle)) 
			return -1 ;
		if (ch != 'A') 
			continue;
		if (!fread(&ch, sizeof(char), 1, input_file.handle)) 
			return -1 ;
		if (ch != 'M') 
			continue;
		if (!fread(&ch, sizeof(char), 1, input_file.handle)) 
			return -1 ;
		if (ch != 'E') 
			continue;
		if (!fread(&ch, sizeof(char), 1, input_file.handle)) 
			return -1 ;
		if (ch != 0x0A) 
			return -1;
		break;
	}

	video.dst_width = video.src_width; //output of the same size for now
	video.dst_height = video.src_height;
	return 0;
}
