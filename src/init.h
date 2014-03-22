// string to print when "-h" option is met
static char small_help[] = "\n"
					"-i\t\t:  input file path\n"
					"-o\t\t:  output file path\n"
					"-qmin\t\t:  min quantizer index (also the only index for key frames)\n"
					"-qmax\t\t:  max quantizer index\n"
					"-g\t\t:  Group of Pictures size\n"
					"-ls\t\t:  loop filter sharpness\n"
					"-partitions\t: amount of partitions for boolean encoding\n"
					"\t\t: (also threads for boolean coding)\n"
					"-threads\t: limit number of threads launched at once\n"
					"\t\t: (prevent overlapping of loop filters and bool coding)\n"
					"\t\t: all partitions are still encoded in parallel\n"
					"\t\t: (even if number of partitions > threads limit)\n"
					"\n"
					"-SSIM-target\t:  tries to keep SSIM of encoded frame higher than this value;\n"
					"\t\t:  format 0.XX (just write XX digits without 0.)\n"
					"-altref-range\t:  amount of frames between using altref;\n"
					"\t\t:  altref are used with lower quantizer and reference previous altref"
					"\n\n"
					;

static int cl_info()
{
	char* value;
	size_t valueSize;
    cl_uint platformCount;
    cl_platform_id* platforms;
    cl_uint deviceCount;
    cl_device_id* devices;
    cl_uint uintVal;
 
    // get all platforms
    clGetPlatformIDs(0, NULL, &platformCount);
    platforms = (cl_platform_id*) malloc(sizeof(cl_platform_id) * platformCount);
    clGetPlatformIDs(platformCount, platforms, NULL);
 
    for (cl_uint i = 0; i < platformCount; i++) 
	{
		clGetPlatformInfo(platforms[i], CL_PLATFORM_NAME, 0, NULL, &valueSize);
		value = (char*) malloc(valueSize);
		clGetPlatformInfo(platforms[i], CL_PLATFORM_NAME, valueSize, value, NULL);
        printf("%d. Platform: %s\n", i, value);
        free(value);

         // get all devices
        clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, 0, NULL, &deviceCount);
        devices = (cl_device_id*) malloc(sizeof(cl_device_id) * deviceCount);
        clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, deviceCount, devices, NULL);
 
        // for each device print critical attributes
        for (cl_uint j = 0; j < deviceCount; j++) 
		{
            // print device name
            clGetDeviceInfo(devices[j], CL_DEVICE_NAME, 0, NULL, &valueSize);
            value = (char*) malloc(valueSize);
            clGetDeviceInfo(devices[j], CL_DEVICE_NAME, valueSize, value, NULL);
            printf(" %d. Device: %s\n", j, value);
            free(value);
 
            // print hardware device version
            clGetDeviceInfo(devices[j], CL_DEVICE_VERSION, 0, NULL, &valueSize);
            value = (char*) malloc(valueSize);
            clGetDeviceInfo(devices[j], CL_DEVICE_VERSION, valueSize, value, NULL);
            printf("  %d.%d Hardware version: %s\n", j, 1, value);
            free(value);
 
            // print software driver version
            clGetDeviceInfo(devices[j], CL_DRIVER_VERSION, 0, NULL, &valueSize);
            value = (char*) malloc(valueSize);
            clGetDeviceInfo(devices[j], CL_DRIVER_VERSION, valueSize, value, NULL);
            printf("  %d.%d Software version: %s\n", j, 2, value);
            free(value);

            // print c version supported by compiler for device
            clGetDeviceInfo(devices[j], CL_DEVICE_OPENCL_C_VERSION, 0, NULL, &valueSize);
            value = (char*) malloc(valueSize);
            clGetDeviceInfo(devices[j], CL_DEVICE_OPENCL_C_VERSION, valueSize, value, NULL);
            printf("  %d.%d OpenCL C version: %s\n", j, 3, value);
            free(value);

            // print parallel compute units
            clGetDeviceInfo(devices[j], CL_DEVICE_MAX_COMPUTE_UNITS,
                    sizeof(uintVal), &uintVal, NULL);
            printf("  %d.%d Parallel compute units: %d\n", j, 4, uintVal);

			// print max work group size
            clGetDeviceInfo(devices[j], CL_DEVICE_MAX_WORK_GROUP_SIZE,
                    sizeof(uintVal), &uintVal, NULL);
            printf("  %d.%d Max work group size: %d\n", j, 4, uintVal);
 
        }
 
        free(devices);
 
    }
 
    free(platforms);
    return 0;
}

static int init_all()
{
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
	i = 0;
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
		device.gpu_device_type = CL_DEVICE_TYPE_GPU;
		device.gpu_preferred_platform_number = (device.gpu_preferred_platform_number >= num_platforms) ? 0 : device.gpu_preferred_platform_number;
		device.state_gpu = clGetDeviceIDs(device.platforms[device.gpu_preferred_platform_number], device.gpu_device_type, 1, device.device_gpu, NULL);
		i = 0;
		while ((device.state_gpu != CL_SUCCESS) && (i < (int)num_platforms)) 
		{
			device.state_cpu = clGetDeviceIDs(device.platforms[i], device.gpu_device_type, 1, device.device_gpu, NULL);
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
	cl_uint program_size;

	// program sources in text files
	// GPU:
	if (video.GOP_size > 1) {
		printf("reading GPU program...\n");
		program_handle = fopen(GPUPATH, "rb");
		if (program_handle == NULL)
		{
			printf("no file %s", GPUPATH);
			return -1;
		}
		fseek(program_handle, 0, SEEK_END);
		program_size = ftell(program_handle);
		if (program_size < 32)
		{
			printf("no program in file %s", GPUPATH);
			return -1;
		}
		rewind(program_handle); // set to start
		char** device_program_source_gpu = (char**)malloc(sizeof(char*));
		*device_program_source_gpu = (char*)malloc(program_size+1);
		(*device_program_source_gpu)[program_size] = '\0';
		size_t deb = fread(*device_program_source_gpu, sizeof(char), program_size, program_handle);
		fclose(program_handle);
		device.program_gpu = clCreateProgramWithSource(device.context_gpu, 1, (const char**)device_program_source_gpu, NULL, &device.state_gpu);
		printf("building GPU program...\n");
		if (video.do_loop_filter_on_gpu) {
			const char gpu_options[] = "-cl-std=CL1.0 -DLOOP_FILTER";
			device.state_gpu = clBuildProgram(device.program_gpu, 1, device.device_gpu, gpu_options, NULL, NULL);
		}
		else {
			const char gpu_options[] = "-cl-std=CL1.0";
			device.state_gpu = clBuildProgram(device.program_gpu, 1, device.device_gpu, gpu_options, NULL, NULL);
		}
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
			device.luma_search_last_16x = clCreateKernel(device.program_gpu, kernel_name, &device.state_gpu); 
			device.luma_search_last_8x = clCreateKernel(device.program_gpu, kernel_name, &device.state_gpu); 
			device.luma_search_last_4x = clCreateKernel(device.program_gpu, kernel_name, &device.state_gpu); 
			device.luma_search_last_2x = clCreateKernel(device.program_gpu, kernel_name, &device.state_gpu); 
			device.luma_search_last_1x = clCreateKernel(device.program_gpu, kernel_name, &device.state_gpu); 
			device.luma_search_golden_16x = clCreateKernel(device.program_gpu, kernel_name, &device.state_gpu); 
			device.luma_search_golden_8x = clCreateKernel(device.program_gpu, kernel_name, &device.state_gpu); 
			device.luma_search_golden_4x = clCreateKernel(device.program_gpu, kernel_name, &device.state_gpu); 
			device.luma_search_golden_2x = clCreateKernel(device.program_gpu, kernel_name, &device.state_gpu); 
			device.luma_search_golden_1x = clCreateKernel(device.program_gpu, kernel_name, &device.state_gpu); 
			device.luma_search_altref_16x = clCreateKernel(device.program_gpu, kernel_name, &device.state_gpu); 
			device.luma_search_altref_8x = clCreateKernel(device.program_gpu, kernel_name, &device.state_gpu); 
			device.luma_search_altref_4x = clCreateKernel(device.program_gpu, kernel_name, &device.state_gpu); 
			device.luma_search_altref_2x = clCreateKernel(device.program_gpu, kernel_name, &device.state_gpu); 
			device.luma_search_altref_1x = clCreateKernel(device.program_gpu, kernel_name, &device.state_gpu); 
		}
		{ char kernel_name[] = "luma_search_2step";
			device.luma_search_last_d4x = clCreateKernel(device.program_gpu, kernel_name, &device.state_gpu); 
			device.luma_search_golden_d4x = clCreateKernel(device.program_gpu, kernel_name, &device.state_gpu);
			device.luma_search_altref_d4x = clCreateKernel(device.program_gpu, kernel_name, &device.state_gpu); 
		}
		{ char kernel_name[] = "downsample_x2";
			device.downsample_current_1x_to_2x = clCreateKernel(device.program_gpu, kernel_name, &device.state_gpu); 
			device.downsample_current_2x_to_4x = clCreateKernel(device.program_gpu, kernel_name, &device.state_gpu); 
			device.downsample_current_4x_to_8x = clCreateKernel(device.program_gpu, kernel_name, &device.state_gpu); 
			device.downsample_current_8x_to_16x = clCreateKernel(device.program_gpu, kernel_name, &device.state_gpu); 
			device.downsample_last_1x_to_2x = clCreateKernel(device.program_gpu, kernel_name, &device.state_gpu); 
			device.downsample_last_2x_to_4x = clCreateKernel(device.program_gpu, kernel_name, &device.state_gpu); 
			device.downsample_last_4x_to_8x = clCreateKernel(device.program_gpu, kernel_name, &device.state_gpu); 
			device.downsample_last_8x_to_16x = clCreateKernel(device.program_gpu, kernel_name, &device.state_gpu); 
		}
		{ char kernel_name[] = "select_reference";
		device.select_reference = clCreateKernel(device.program_gpu, kernel_name, &device.state_gpu); }
		{ char kernel_name[] = "prepare_predictors_and_residual";
			device.prepare_predictors_and_residual_last_Y = clCreateKernel(device.program_gpu, kernel_name, &device.state_gpu); 
			device.prepare_predictors_and_residual_last_U = clCreateKernel(device.program_gpu, kernel_name, &device.state_gpu); 
			device.prepare_predictors_and_residual_last_V = clCreateKernel(device.program_gpu, kernel_name, &device.state_gpu); 
			device.prepare_predictors_and_residual_golden_Y = clCreateKernel(device.program_gpu, kernel_name, &device.state_gpu); 
			device.prepare_predictors_and_residual_golden_U = clCreateKernel(device.program_gpu, kernel_name, &device.state_gpu); 
			device.prepare_predictors_and_residual_golden_V = clCreateKernel(device.program_gpu, kernel_name, &device.state_gpu); 
			device.prepare_predictors_and_residual_altref_Y = clCreateKernel(device.program_gpu, kernel_name, &device.state_gpu); 
			device.prepare_predictors_and_residual_altref_U = clCreateKernel(device.program_gpu, kernel_name, &device.state_gpu); 
			device.prepare_predictors_and_residual_altref_V = clCreateKernel(device.program_gpu, kernel_name, &device.state_gpu); 
		}
		{ char kernel_name[] = "pack_8x8_into_16x16";
		device.pack_8x8_into_16x16 = clCreateKernel(device.program_gpu, kernel_name, &device.state_gpu); }
		{ char kernel_name[] = "dct4x4";
			device.dct4x4_Y[UQ_segment] = clCreateKernel(device.program_gpu, kernel_name, &device.state_gpu); 
			device.dct4x4_U[UQ_segment] = clCreateKernel(device.program_gpu, kernel_name, &device.state_gpu); 
			device.dct4x4_V[UQ_segment] = clCreateKernel(device.program_gpu, kernel_name, &device.state_gpu); 
			device.dct4x4_Y[HQ_segment] = clCreateKernel(device.program_gpu, kernel_name, &device.state_gpu); 
			device.dct4x4_U[HQ_segment] = clCreateKernel(device.program_gpu, kernel_name, &device.state_gpu); 
			device.dct4x4_V[HQ_segment] = clCreateKernel(device.program_gpu, kernel_name, &device.state_gpu); 
			device.dct4x4_Y[AQ_segment] = clCreateKernel(device.program_gpu, kernel_name, &device.state_gpu); 
			device.dct4x4_U[AQ_segment] = clCreateKernel(device.program_gpu, kernel_name, &device.state_gpu); 
			device.dct4x4_V[AQ_segment] = clCreateKernel(device.program_gpu, kernel_name, &device.state_gpu); 
			device.dct4x4_Y[LQ_segment] = clCreateKernel(device.program_gpu, kernel_name, &device.state_gpu); 
			device.dct4x4_U[LQ_segment] = clCreateKernel(device.program_gpu, kernel_name, &device.state_gpu); 
			device.dct4x4_V[LQ_segment] = clCreateKernel(device.program_gpu, kernel_name, &device.state_gpu); 
		}
		{ char kernel_name[] = "wht4x4_iwht4x4";
			device.wht4x4_iwht4x4[UQ_segment] = clCreateKernel(device.program_gpu, kernel_name, &device.state_gpu); 
			device.wht4x4_iwht4x4[HQ_segment] = clCreateKernel(device.program_gpu, kernel_name, &device.state_gpu); 
			device.wht4x4_iwht4x4[AQ_segment] = clCreateKernel(device.program_gpu, kernel_name, &device.state_gpu); 
			device.wht4x4_iwht4x4[LQ_segment] = clCreateKernel(device.program_gpu, kernel_name, &device.state_gpu); 
		}
		{ char kernel_name[] = "idct4x4";
			device.idct4x4_Y[UQ_segment] = clCreateKernel(device.program_gpu, kernel_name, &device.state_gpu); 
			device.idct4x4_U[UQ_segment] = clCreateKernel(device.program_gpu, kernel_name, &device.state_gpu); 
			device.idct4x4_V[UQ_segment] = clCreateKernel(device.program_gpu, kernel_name, &device.state_gpu); 
			device.idct4x4_Y[HQ_segment] = clCreateKernel(device.program_gpu, kernel_name, &device.state_gpu); 
			device.idct4x4_U[HQ_segment] = clCreateKernel(device.program_gpu, kernel_name, &device.state_gpu); 
			device.idct4x4_V[HQ_segment] = clCreateKernel(device.program_gpu, kernel_name, &device.state_gpu); 
			device.idct4x4_Y[AQ_segment] = clCreateKernel(device.program_gpu, kernel_name, &device.state_gpu); 
			device.idct4x4_U[AQ_segment] = clCreateKernel(device.program_gpu, kernel_name, &device.state_gpu); 
			device.idct4x4_V[AQ_segment] = clCreateKernel(device.program_gpu, kernel_name, &device.state_gpu); 
			device.idct4x4_Y[LQ_segment] = clCreateKernel(device.program_gpu, kernel_name, &device.state_gpu); 
			device.idct4x4_U[LQ_segment] = clCreateKernel(device.program_gpu, kernel_name, &device.state_gpu); 
			device.idct4x4_V[LQ_segment] = clCreateKernel(device.program_gpu, kernel_name, &device.state_gpu); 
		}
		if (video.do_loop_filter_on_gpu)
		{
			{ char kernel_name[] = "prepare_filter_mask"; //and non zero coeffs count
			device.prepare_filter_mask = clCreateKernel(device.program_gpu, kernel_name, &device.state_gpu); }
			{ char kernel_name[] = "normal_loop_filter_MBH";
			device.normal_loop_filter_MBH = clCreateKernel(device.program_gpu, kernel_name, &device.state_gpu); }
			{ char kernel_name[] = "normal_loop_filter_MBV";
			device.normal_loop_filter_MBV = clCreateKernel(device.program_gpu, kernel_name, &device.state_gpu); }
		}
		{ char kernel_name[] = "count_SSIM_luma";
			device.count_SSIM_luma[UQ_segment] = clCreateKernel(device.program_gpu, kernel_name, &device.state_gpu);
			device.count_SSIM_luma[HQ_segment] = clCreateKernel(device.program_gpu, kernel_name, &device.state_gpu);
			device.count_SSIM_luma[AQ_segment] = clCreateKernel(device.program_gpu, kernel_name, &device.state_gpu);
			device.count_SSIM_luma[LQ_segment] = clCreateKernel(device.program_gpu, kernel_name, &device.state_gpu);
		}
		{ char kernel_name[] = "count_SSIM_chroma";
			device.count_SSIM_chroma_U[UQ_segment] = clCreateKernel(device.program_gpu, kernel_name, &device.state_gpu);
			device.count_SSIM_chroma_U[HQ_segment] = clCreateKernel(device.program_gpu, kernel_name, &device.state_gpu);
			device.count_SSIM_chroma_U[AQ_segment] = clCreateKernel(device.program_gpu, kernel_name, &device.state_gpu);
			device.count_SSIM_chroma_U[LQ_segment] = clCreateKernel(device.program_gpu, kernel_name, &device.state_gpu);
			device.count_SSIM_chroma_V[UQ_segment] = clCreateKernel(device.program_gpu, kernel_name, &device.state_gpu);
			device.count_SSIM_chroma_V[HQ_segment] = clCreateKernel(device.program_gpu, kernel_name, &device.state_gpu);
			device.count_SSIM_chroma_V[AQ_segment] = clCreateKernel(device.program_gpu, kernel_name, &device.state_gpu);
			device.count_SSIM_chroma_V[LQ_segment] = clCreateKernel(device.program_gpu, kernel_name, &device.state_gpu);
		}
		{ char kernel_name[] = "gather_SSIM";
		device.gather_SSIM = clCreateKernel(device.program_gpu, kernel_name, &device.state_gpu); }
	}
	// CPU:
	printf("reading CPU program...\n");
	program_handle = fopen(CPUPATH, "rb");
	if (program_handle == NULL)
	{
		printf("no file %s", CPUPATH);
		return -1;
	}
	fseek(program_handle, 0, SEEK_END);
	program_size = ftell(program_handle);
	if (program_size < 32)
	{
		printf("no program in file %s", GPUPATH);
		return -1;
	}
	rewind(program_handle); // set to start
	char** device_program_source_cpu = (char**)malloc(sizeof(char*));
		*device_program_source_cpu = (char*)malloc(program_size+1);
	(*device_program_source_cpu)[program_size] = '\0';
	fread(*device_program_source_cpu, sizeof(char),	program_size, program_handle);
	fclose(program_handle);
	device.program_cpu = clCreateProgramWithSource(device.context_cpu, 1, (const char**)device_program_source_cpu, NULL, &device.state_cpu);
	printf("building CPU program...\n");
	if (!video.do_loop_filter_on_gpu) {
		const char cpu_options[] = "-cl-std=CL1.0 -DLOOP_FILTER";
		device.state_cpu = clBuildProgram(device.program_cpu, 1, device.device_cpu, cpu_options, NULL, NULL);
	}
	else {
		const char cpu_options[] = "-cl-std=CL1.0";
		device.state_cpu = clBuildProgram(device.program_cpu, 1, device.device_cpu, cpu_options, NULL, NULL);
	}
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
	if (!video.do_loop_filter_on_gpu)
	{
		{ char kernel_name[] = "prepare_filter_mask"; //and non zero coeffs count
		device.prepare_filter_mask = clCreateKernel(device.program_cpu, kernel_name, &device.state_gpu); }
		{ char kernel_name[] = "loop_filter_frame_luma";
		device.loop_filter_frame_luma = clCreateKernel(device.program_cpu, kernel_name, &device.state_cpu); }
		{ char kernel_name[] = "loop_filter_frame_chroma";
			device.loop_filter_frame_chroma_U = clCreateKernel(device.program_cpu, kernel_name, &device.state_cpu); 
			device.loop_filter_frame_chroma_V = clCreateKernel(device.program_cpu, kernel_name, &device.state_cpu); 
		}
	}


    video.src_frame_size_luma = video.src_height*video.src_width;
    video.src_frame_size_chroma = video.src_frame_size_luma >> 2;
    int src_frame_size_full = video.src_frame_size_luma + (video.src_frame_size_chroma << 1) ;

    video.wrk_height = video.dst_height;
    video.wrk_width = video.dst_width;
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
    frames.input_pack = (cl_uchar*)malloc(src_frame_size_full*frames.input_pack_size);
    // all buffers of previous frames must be padded
    frames.reconstructed_Y =(cl_uchar*)malloc(video.wrk_frame_size_luma);
    frames.reconstructed_U =(cl_uchar*)malloc(video.wrk_frame_size_chroma);
    frames.reconstructed_V =(cl_uchar*)malloc(video.wrk_frame_size_chroma);
    frames.transformed_blocks =(macroblock*)malloc(sizeof(macroblock)*video.mb_count);
	frames.e_data = (macroblock_extra_data*)malloc(video.mb_count*sizeof(macroblock_extra_data));
    frames.encoded_frame = (cl_uchar*)malloc(src_frame_size_full<<1); // extra size
	frames.partition_0 = (cl_uchar*)malloc(64*video.mb_count + 128); 
	video.partition_step = sizeof(cl_short)*800*(video.mb_count)/2;
	frames.partitions = (cl_uchar*)malloc(video.partition_step); 

	frames.last_U = (cl_uchar*)malloc(video.wrk_frame_size_chroma);
	frames.last_V = (cl_uchar*)malloc(video.wrk_frame_size_chroma);
    if (((video.src_height != video.dst_height) || (video.src_width != video.dst_width)) ||
        ((video.wrk_height != video.dst_height) || (video.wrk_width != video.dst_width)))
    {
        // then we need buffer to store resized input frame (and padded along the way, to avoid double-copy) - 1st if-line
        // we need to store padded data - 2nd if-line
        frames.current_Y = (cl_uchar*)malloc(video.wrk_frame_size_luma);
        frames.current_U = (cl_uchar*)malloc(video.wrk_frame_size_chroma);
        frames.current_V = (cl_uchar*)malloc(video.wrk_frame_size_chroma);
    }
	else 
	{ // to avoid reading from unknown space (for example memcpy in get_yuv420_frame before first frame read)
		// also could be done by checking if frame is first (done too, both for safety)
		frames.current_U = frames.last_U;
		frames.current_V = frames.last_V;
	} // later these buffers are being reassigned

	if (video.GOP_size > 1) {
		// downsampled sizes
		const int sz16 = (video.wrk_width / 16)*(video.wrk_height / 16);
		const int sz8 = (video.wrk_width / 8)*(video.wrk_height / 8);
		const int sz4 = (video.wrk_width / 4)*(video.wrk_height / 4);
		const int sz2 = (video.wrk_width / 2)*(video.wrk_height / 2);

		device.predictors_Y = clCreateBuffer(device.context_gpu, CL_MEM_READ_WRITE, video.wrk_frame_size_luma, NULL , &device.state_gpu);
		if (device.state_gpu != 0) { printf("GPU device memory problem %d with predictors_Y\n", device.state_gpu); return -1; }
		device.predictors_U = clCreateBuffer(device.context_gpu, CL_MEM_READ_WRITE, video.wrk_frame_size_chroma, NULL , &device.state_gpu);
		if (device.state_gpu != 0) { printf("GPU device memory problem %d with predictors_U\n", device.state_gpu); return -1; }
		device.predictors_V = clCreateBuffer(device.context_gpu, CL_MEM_READ_WRITE, video.wrk_frame_size_chroma, NULL , &device.state_gpu);
		if (device.state_gpu != 0) { printf("GPU device memory problem %d with predictors_V\n", device.state_gpu); return -1; }
		device.residual_Y = clCreateBuffer(device.context_gpu, CL_MEM_READ_WRITE, video.wrk_frame_size_luma*sizeof(short), NULL , &device.state_gpu);
		if (device.state_gpu != 0) { printf("GPU device memory problem %d with residual_Y\n", device.state_gpu); return -1; }
		device.residual_U = clCreateBuffer(device.context_gpu, CL_MEM_READ_WRITE, video.wrk_frame_size_chroma*sizeof(short), NULL , &device.state_gpu);
		if (device.state_gpu != 0) { printf("GPU device memory problem %d with residual_U\n", device.state_gpu); return -1; }
		device.residual_V = clCreateBuffer(device.context_gpu, CL_MEM_READ_WRITE, video.wrk_frame_size_chroma*sizeof(short), NULL , &device.state_gpu);
		if (device.state_gpu != 0) { printf("GPU device memory problem %d with residual_V\n", device.state_gpu); return -1; }
		device.current_frame_Y = clCreateBuffer(device.context_gpu, CL_MEM_READ_WRITE, video.wrk_frame_size_luma, NULL , &device.state_gpu);
		if (device.state_gpu != 0) { printf("GPU device memory problem %d with current_frame_Y\n", device.state_gpu); return -1; }
		device.current_frame_Y_downsampled_by2 = clCreateBuffer(device.context_gpu, CL_MEM_READ_WRITE, sz2, NULL , &device.state_gpu);
		if (device.state_gpu != 0) { printf("GPU device memory problem %d with current_frame_Y_downsampled_by2\n", device.state_gpu); return -1; }
		device.current_frame_Y_downsampled_by4 = clCreateBuffer(device.context_gpu, CL_MEM_READ_WRITE, sz4, NULL , &device.state_gpu);
		if (device.state_gpu != 0) { printf("GPU device memory problem %d with current_frame_Y_downsampled_by4\n", device.state_gpu); return -1; }
		device.current_frame_Y_downsampled_by8 = clCreateBuffer(device.context_gpu, CL_MEM_READ_WRITE, sz8, NULL , &device.state_gpu);
		if (device.state_gpu != 0) { printf("GPU device memory problem %d with current_frame_Y_downsampled_by8\n", device.state_gpu); return -1; }
		device.current_frame_Y_downsampled_by16 = clCreateBuffer(device.context_gpu, CL_MEM_READ_WRITE, sz16, NULL , &device.state_gpu);
		if (device.state_gpu != 0) { printf("GPU device memory problem %d with current_frame_Y_downsampled_by16\n", device.state_gpu); return -1; }
		device.current_frame_U = clCreateBuffer(device.context_gpu, CL_MEM_READ_ONLY, video.wrk_frame_size_chroma, NULL , &device.state_gpu);
		if (device.state_gpu != 0) { printf("GPU device memory problem %d with current_frame_U\n", device.state_gpu); return -1; }
		device.current_frame_V = clCreateBuffer(device.context_gpu, CL_MEM_READ_ONLY, video.wrk_frame_size_chroma, NULL , &device.state_gpu);
		if (device.state_gpu != 0) { printf("GPU device memory problem %d with current_frame_V\n", device.state_gpu); return -1; }
		device.last_frame_Y_downsampled_by2 = clCreateBuffer(device.context_gpu, CL_MEM_READ_WRITE, sz2, NULL , &device.state_gpu);
		if (device.state_gpu != 0) { printf("GPU device memory problem %d with last_frame_Y_downsampled_by2\n", device.state_gpu); return -1; }
		device.last_frame_Y_downsampled_by4 = clCreateBuffer(device.context_gpu, CL_MEM_READ_WRITE, sz4, NULL , &device.state_gpu);
		if (device.state_gpu != 0) { printf("GPU device memory problem %d with last_frame_Y_downsampled_by4\n", device.state_gpu); return -1; }
		device.last_frame_Y_downsampled_by8 = clCreateBuffer(device.context_gpu, CL_MEM_READ_WRITE, sz8, NULL , &device.state_gpu);
		if (device.state_gpu != 0) { printf("GPU device memory problem %d with last_frame_Y_downsampled_by8\n", device.state_gpu); return -1; }
		device.last_frame_Y_downsampled_by16 = clCreateBuffer(device.context_gpu, CL_MEM_READ_WRITE, sz16, NULL , &device.state_gpu);
		if (device.state_gpu != 0) { printf("GPU device memory problem %d with last_frame_Y_downsampled_by16\n", device.state_gpu); return -1; }
		device.golden_frame_Y_downsampled_by2 = clCreateBuffer(device.context_gpu, CL_MEM_READ_WRITE, sz2, NULL , &device.state_gpu);
		if (device.state_gpu != 0) { printf("GPU device memory problem %d with golden_frame_Y_downsampled_by2\n", device.state_gpu); return -1; }
		device.golden_frame_Y_downsampled_by4 = clCreateBuffer(device.context_gpu, CL_MEM_READ_WRITE, sz4, NULL , &device.state_gpu);
		if (device.state_gpu != 0) { printf("GPU device memory problem %d with golden_frame_Y_downsampled_by4\n", device.state_gpu); return -1; }
		device.golden_frame_Y_downsampled_by8 = clCreateBuffer(device.context_gpu, CL_MEM_READ_WRITE, sz8, NULL , &device.state_gpu);
		if (device.state_gpu != 0) { printf("GPU device memory problem %d with golden_frame_Y_downsampled_by8\n", device.state_gpu); return -1; }
		device.golden_frame_Y_downsampled_by16 = clCreateBuffer(device.context_gpu, CL_MEM_READ_WRITE, sz16, NULL , &device.state_gpu);
		if (device.state_gpu != 0) { printf("GPU device memory problem %d with golden_frame_Y_downsampled_by16\n", device.state_gpu); return -1; }
		device.altref_frame_Y_downsampled_by2 = clCreateBuffer(device.context_gpu, CL_MEM_READ_WRITE, sz2, NULL , &device.state_gpu);
		if (device.state_gpu != 0) { printf("GPU device memory problem %d with altref_frame_Y_downsampled_by2\n", device.state_gpu); return -1; }
		device.altref_frame_Y_downsampled_by4 = clCreateBuffer(device.context_gpu, CL_MEM_READ_WRITE, sz4, NULL , &device.state_gpu);
		if (device.state_gpu != 0) { printf("GPU device memory problem %d with altref_frame_Y_downsampled_by4\n", device.state_gpu); return -1; }
		device.altref_frame_Y_downsampled_by8 = clCreateBuffer(device.context_gpu, CL_MEM_READ_WRITE, sz8, NULL , &device.state_gpu);
		if (device.state_gpu != 0) { printf("GPU device memory problem %d with altref_frame_Y_downsampled_by8\n", device.state_gpu); return -1; }
		device.altref_frame_Y_downsampled_by16 = clCreateBuffer(device.context_gpu, CL_MEM_READ_WRITE, sz16, NULL , &device.state_gpu);
		if (device.state_gpu != 0) { printf("GPU device memory problem %d with altref_frame_Y_downsampled_by16\n", device.state_gpu); return -1; }
		device.reconstructed_frame_Y = clCreateBuffer(device.context_gpu, CL_MEM_READ_WRITE, video.wrk_frame_size_luma, NULL , &device.state_gpu);
		if (device.state_gpu != 0) { printf("GPU device memory problem %d with reconstructed_frame_Y\n", device.state_gpu); return -1; }
		device.reconstructed_frame_U = clCreateBuffer(device.context_gpu, CL_MEM_READ_WRITE, video.wrk_frame_size_chroma, NULL , &device.state_gpu);
		if (device.state_gpu != 0) { printf("GPU device memory problem %d with reconstructed_frame_U\n", device.state_gpu); return -1; }
		device.reconstructed_frame_V = clCreateBuffer(device.context_gpu, CL_MEM_READ_WRITE, video.wrk_frame_size_chroma, NULL , &device.state_gpu);
		if (device.state_gpu != 0) { printf("GPU device memory problem %d with reconstructed_frame_V\n", device.state_gpu); return -1; }
		device.golden_frame_Y = clCreateBuffer(device.context_gpu, CL_MEM_READ_WRITE, video.wrk_frame_size_luma, NULL , &device.state_gpu);
		if (device.state_gpu != 0) { printf("GPU device memory problem %d with golden_frame_Y\n", device.state_gpu); return -1; }
		device.altref_frame_Y = clCreateBuffer(device.context_gpu, CL_MEM_READ_WRITE, video.wrk_frame_size_luma, NULL , &device.state_gpu);
		if (device.state_gpu != 0) { printf("GPU device memory problem %d with altref_frame_Y\n", device.state_gpu); return -1; }
		device.transformed_blocks_gpu = clCreateBuffer(device.context_gpu, CL_MEM_READ_WRITE, sizeof(macroblock)*video.mb_count, NULL , &device.state_gpu);
		if (device.state_gpu != 0) { printf("GPU device memory problem %d with transformed_blocks_gpu\n", device.state_gpu); return -1; }	
		device.last_vnet1 = clCreateBuffer(device.context_gpu, CL_MEM_READ_WRITE, sizeof(vector_net)*video.mb_count*4, NULL , &device.state_gpu);
		if (device.state_gpu != 0) { printf("GPU device memory problem %d with last vnet1\n", device.state_gpu); return -1; }
		device.golden_vnet1 = clCreateBuffer(device.context_gpu, CL_MEM_READ_WRITE, sizeof(vector_net)*video.mb_count*4, NULL , &device.state_gpu);
		if (device.state_gpu != 0) { printf("GPU device memory problem %d with golden vnet1\n", device.state_gpu); return -1; }
		device.altref_vnet1 = clCreateBuffer(device.context_gpu, CL_MEM_READ_WRITE, sizeof(vector_net)*video.mb_count*4, NULL , &device.state_gpu);
		if (device.state_gpu != 0) { printf("GPU device memory problem %d with altref vnet1\n", device.state_gpu); return -1; }
		device.last_vnet2 = clCreateBuffer(device.context_gpu, CL_MEM_READ_WRITE, sizeof(vector_net)*video.mb_count*4, NULL , &device.state_gpu);
		if (device.state_gpu != 0) { printf("GPU device memory problem %d with last vnet2\n", device.state_gpu); return -1; }
		device.golden_vnet2 = clCreateBuffer(device.context_gpu, CL_MEM_READ_WRITE, sizeof(vector_net)*video.mb_count*4, NULL , &device.state_gpu);
		if (device.state_gpu != 0) { printf("GPU device memory problem %d with golden vnet2\n", device.state_gpu); return -1; }
		device.altref_vnet2 = clCreateBuffer(device.context_gpu, CL_MEM_READ_WRITE, sizeof(vector_net)*video.mb_count*4, NULL , &device.state_gpu);
		if (device.state_gpu != 0) { printf("GPU device memory problem %d with altref vnet2\n", device.state_gpu); return -1; }
		device.metrics1 = clCreateBuffer(device.context_gpu, CL_MEM_READ_WRITE, sizeof(cl_int)*video.mb_count*4, NULL , &device.state_gpu);
		if (device.state_gpu != 0) { printf("GPU device memory problem %d with last_metrics\n", device.state_gpu); return -1; }
		device.metrics2 = clCreateBuffer(device.context_gpu, CL_MEM_READ_WRITE, sizeof(cl_int)*video.mb_count*4, NULL , &device.state_gpu);
		if (device.state_gpu != 0) { printf("GPU device memory problem %d with golden_metrics\n", device.state_gpu); return -1; }
		device.metrics3 = clCreateBuffer(device.context_gpu, CL_MEM_READ_WRITE, sizeof(cl_int)*video.mb_count*4, NULL , &device.state_gpu);
		if (device.state_gpu != 0) { printf("GPU device memory problem %d with altref_metrics\n", device.state_gpu); return -1; }
		
		if (video.do_loop_filter_on_gpu)
		{
			device.mb_mask = clCreateBuffer(device.context_gpu, CL_MEM_READ_WRITE, sizeof(cl_int)*video.mb_count, NULL , &device.state_gpu);
			if (device.state_gpu != 0) { printf("GPU device memory problem %d with mb_mask\n", device.state_gpu); return -1; }
		}
		else
		{
			device.mb_mask = clCreateBuffer(device.context_cpu, CL_MEM_READ_WRITE, sizeof(cl_int)*video.mb_count, NULL , &device.state_cpu);
			if (device.state_cpu != 0) { printf("CPU device memory problem %d with mb_mask\n", device.state_cpu); return -1; }
			device.cpu_frame_Y = clCreateBuffer(device.context_cpu, CL_MEM_READ_WRITE, video.wrk_frame_size_luma, NULL , &device.state_cpu);
			if (device.state_gpu != 0) { printf("GPU device memory problem %d with cpu_frame_Y\n", device.state_gpu); return -1; }
			device.cpu_frame_U = clCreateBuffer(device.context_cpu, CL_MEM_READ_WRITE, video.wrk_frame_size_chroma, NULL , &device.state_cpu);
			if (device.state_gpu != 0) { printf("GPU device memory problem %d with cpu_frame_U\n", device.state_gpu); return -1; }
			device.cpu_frame_V = clCreateBuffer(device.context_cpu, CL_MEM_READ_WRITE, video.wrk_frame_size_chroma, NULL , &device.state_cpu);
			if (device.state_gpu != 0) { printf("GPU device memory problem %d with cpu_frame_V\n", device.state_gpu); return -1; }
		
		}
		device.segments_data_gpu = clCreateBuffer(device.context_gpu, CL_MEM_READ_ONLY, sizeof(segment_data)*4, NULL , &device.state_gpu);
		if (device.state_gpu != 0) { printf("GPU device memory problem %d with segments_data\n", device.state_gpu); return -1; }
		device.segments_data_cpu = clCreateBuffer(device.context_cpu, CL_MEM_READ_ONLY, sizeof(segment_data)*4, NULL , &device.state_cpu);
		if (device.state_cpu != 0) { printf("CPU device memory problem %d with segments_data\n", device.state_cpu); return -1; }

		// and now creating image obhects
		device.image_format.image_channel_order = CL_R;
		device.image_format.image_channel_data_type = CL_UNSIGNED_INT8;
		device.last_frame_Y_image = clCreateImage2D(device.context_gpu, CL_MEM_READ_ONLY, &device.image_format,
													video.wrk_width,video.wrk_height,0,NULL,&device.state_gpu);
		device.last_frame_U_image = clCreateImage2D(device.context_gpu, CL_MEM_READ_ONLY, &device.image_format,
													video.wrk_width/2,video.wrk_height/2,0,NULL,&device.state_gpu);
		device.last_frame_V_image = clCreateImage2D(device.context_gpu, CL_MEM_READ_ONLY, &device.image_format,
													video.wrk_width/2,video.wrk_height/2,0,NULL,&device.state_gpu);
		device.golden_frame_Y_image = clCreateImage2D(device.context_gpu, CL_MEM_READ_ONLY, &device.image_format,
													video.wrk_width,video.wrk_height,0,NULL,&device.state_gpu);
		device.golden_frame_U_image = clCreateImage2D(device.context_gpu, CL_MEM_READ_ONLY, &device.image_format,
													video.wrk_width/2,video.wrk_height/2,0,NULL,&device.state_gpu);
		device.golden_frame_V_image = clCreateImage2D(device.context_gpu, CL_MEM_READ_ONLY, &device.image_format,
													video.wrk_width/2,video.wrk_height/2,0,NULL,&device.state_gpu);
		device.altref_frame_Y_image = clCreateImage2D(device.context_gpu, CL_MEM_READ_ONLY, &device.image_format,
													video.wrk_width,video.wrk_height,0,NULL,&device.state_gpu);
		device.altref_frame_U_image = clCreateImage2D(device.context_gpu, CL_MEM_READ_ONLY, &device.image_format,
													video.wrk_width/2,video.wrk_height/2,0,NULL,&device.state_gpu);
		device.altref_frame_V_image = clCreateImage2D(device.context_gpu, CL_MEM_READ_ONLY, &device.image_format,
													video.wrk_width/2,video.wrk_height/2,0,NULL,&device.state_gpu);
		if (device.state_gpu != 0) 
			printf("=> create buffer problem!\n");
		
	}
	device.transformed_blocks_cpu = clCreateBuffer(device.context_cpu, CL_MEM_READ_WRITE, sizeof(macroblock)*video.mb_count, NULL , &device.state_cpu);
	device.partitions = clCreateBuffer(device.context_cpu, CL_MEM_READ_WRITE, video.partition_step, NULL , &device.state_cpu);
	device.partitions_sizes = clCreateBuffer(device.context_cpu, CL_MEM_READ_WRITE, 8*sizeof(cl_int), NULL , &device.state_cpu);
	device.third_context = clCreateBuffer(device.context_cpu, CL_MEM_READ_WRITE, sizeof(cl_uchar)*25*video.mb_count, NULL , &device.state_cpu);
	device.coeff_probs = clCreateBuffer(device.context_cpu, CL_MEM_READ_WRITE, 8*4*8*3*11*sizeof(cl_uint), NULL , &device.state_cpu);
	device.coeff_probs_denom = clCreateBuffer(device.context_cpu, CL_MEM_READ_WRITE, 8*4*8*3*11*sizeof(cl_uint), NULL , &device.state_cpu);

	const cl_int cwidth = video.wrk_width/2;
	const cl_int cheight = video.wrk_height/2;
	if (video.GOP_size > 1) {
		cl_int w, h, pixval, plane, ref, seg_id;
		const cl_int net_width = video.mb_width*2;

		/*__kernel void reset_vectors ( __global vector_net *const last_net1, //0
								__global vector_net *const last_net2, //1
								__global vector_net *const golden_net1, //2
								__global vector_net *const golden_net2, //3
								__global vector_net *const altref_net1, //4
								__global vector_net *const altref_net2, //5
								__global int *const last_Bdiff, //6
								__global int *const golden_Bdiff, //7
								__global int *const altref_Bdiff) //8*/
		device.state_gpu = clSetKernelArg(device.reset_vectors, 0, sizeof(cl_mem), &device.last_vnet1);
		device.state_gpu = clSetKernelArg(device.reset_vectors, 1, sizeof(cl_mem), &device.last_vnet2);
		device.state_gpu = clSetKernelArg(device.reset_vectors, 2, sizeof(cl_mem), &device.golden_vnet1);
		device.state_gpu = clSetKernelArg(device.reset_vectors, 3, sizeof(cl_mem), &device.golden_vnet2);
		device.state_gpu = clSetKernelArg(device.reset_vectors, 4, sizeof(cl_mem), &device.altref_vnet1);
		device.state_gpu = clSetKernelArg(device.reset_vectors, 5, sizeof(cl_mem), &device.altref_vnet2);
		device.state_gpu = clSetKernelArg(device.reset_vectors, 6, sizeof(cl_mem), &device.metrics1);
		device.state_gpu = clSetKernelArg(device.reset_vectors, 7, sizeof(cl_mem), &device.metrics2);
		device.state_gpu = clSetKernelArg(device.reset_vectors, 8, sizeof(cl_mem), &device.metrics3);

		/*__kernel void downsample(__global uchar *const src_frame, //0
									__global uchar *const dst_frame, //1
									const signed int src_width, //2
									const signed int src_height) //3*/
		device.state_gpu = clSetKernelArg(device.downsample_current_1x_to_2x, 0, sizeof(cl_mem), &device.current_frame_Y);
		device.state_gpu = clSetKernelArg(device.downsample_current_1x_to_2x, 1, sizeof(cl_mem), &device.current_frame_Y_downsampled_by2);
		device.state_gpu = clSetKernelArg(device.downsample_current_1x_to_2x, 2, sizeof(cl_int), &video.wrk_width);
		device.state_gpu = clSetKernelArg(device.downsample_current_1x_to_2x, 3, sizeof(cl_int), &video.wrk_height);
		device.state_gpu = clSetKernelArg(device.downsample_last_1x_to_2x, 0, sizeof(cl_mem), &device.reconstructed_frame_Y);
		device.state_gpu = clSetKernelArg(device.downsample_last_1x_to_2x, 1, sizeof(cl_mem), &device.last_frame_Y_downsampled_by2);
		device.state_gpu = clSetKernelArg(device.downsample_last_1x_to_2x, 2, sizeof(cl_int), &video.wrk_width);
		device.state_gpu = clSetKernelArg(device.downsample_last_1x_to_2x, 3, sizeof(cl_int), &video.wrk_height);
		w = video.wrk_width/2;
		h = video.wrk_height/2;
		device.state_gpu = clSetKernelArg(device.downsample_current_2x_to_4x, 0, sizeof(cl_mem), &device.current_frame_Y_downsampled_by2);
		device.state_gpu = clSetKernelArg(device.downsample_current_2x_to_4x, 1, sizeof(cl_mem), &device.current_frame_Y_downsampled_by4);
		device.state_gpu = clSetKernelArg(device.downsample_current_2x_to_4x, 2, sizeof(cl_int), &w);
		device.state_gpu = clSetKernelArg(device.downsample_current_2x_to_4x, 3, sizeof(cl_int), &h);
		device.state_gpu = clSetKernelArg(device.downsample_last_2x_to_4x, 0, sizeof(cl_mem), &device.last_frame_Y_downsampled_by2);
		device.state_gpu = clSetKernelArg(device.downsample_last_2x_to_4x, 1, sizeof(cl_mem), &device.last_frame_Y_downsampled_by4);
		device.state_gpu = clSetKernelArg(device.downsample_last_2x_to_4x, 2, sizeof(cl_int), &w);
		device.state_gpu = clSetKernelArg(device.downsample_last_2x_to_4x, 3, sizeof(cl_int), &h);
		w = video.wrk_width/4;
		h = video.wrk_height/4;
		device.state_gpu = clSetKernelArg(device.downsample_current_4x_to_8x, 0, sizeof(cl_mem), &device.current_frame_Y_downsampled_by4);
		device.state_gpu = clSetKernelArg(device.downsample_current_4x_to_8x, 1, sizeof(cl_mem), &device.current_frame_Y_downsampled_by8);
		device.state_gpu = clSetKernelArg(device.downsample_current_4x_to_8x, 2, sizeof(cl_int), &w);
		device.state_gpu = clSetKernelArg(device.downsample_current_4x_to_8x, 3, sizeof(cl_int), &h);
		device.state_gpu = clSetKernelArg(device.downsample_last_4x_to_8x, 0, sizeof(cl_mem), &device.last_frame_Y_downsampled_by4);
		device.state_gpu = clSetKernelArg(device.downsample_last_4x_to_8x, 1, sizeof(cl_mem), &device.last_frame_Y_downsampled_by8);
		device.state_gpu = clSetKernelArg(device.downsample_last_4x_to_8x, 2, sizeof(cl_int), &w);
		device.state_gpu = clSetKernelArg(device.downsample_last_4x_to_8x, 3, sizeof(cl_int), &h);
		w = video.wrk_width/8;
		h = video.wrk_height/8;
		device.state_gpu = clSetKernelArg(device.downsample_current_8x_to_16x, 0, sizeof(cl_mem), &device.current_frame_Y_downsampled_by8);
		device.state_gpu = clSetKernelArg(device.downsample_current_8x_to_16x, 1, sizeof(cl_mem), &device.current_frame_Y_downsampled_by16);
		device.state_gpu = clSetKernelArg(device.downsample_current_8x_to_16x, 2, sizeof(cl_int), &w);
		device.state_gpu = clSetKernelArg(device.downsample_current_8x_to_16x, 3, sizeof(cl_int), &h);
		device.state_gpu = clSetKernelArg(device.downsample_last_8x_to_16x, 0, sizeof(cl_mem), &device.last_frame_Y_downsampled_by8);
		device.state_gpu = clSetKernelArg(device.downsample_last_8x_to_16x, 1, sizeof(cl_mem), &device.last_frame_Y_downsampled_by16);
		device.state_gpu = clSetKernelArg(device.downsample_last_8x_to_16x, 2, sizeof(cl_int), &w);
		device.state_gpu = clSetKernelArg(device.downsample_last_8x_to_16x, 3, sizeof(cl_int), &h);

		/*__kernel void luma_search_1step //when looking into downsampled and original frames
									( 	__global uchar *const current_frame, //0
										__global uchar *const prev_frame, //1
										__global vector_net *const src_net, //2
										__global vector_net *const dst_net, //3
										const signed int net_width, //4 //in 8x8 blocks
										const signed int width, //5
										const signed int height, //6
										const signed int pixel_rate) //7 */
		//last_16x
		w = video.wrk_width/16;
		h = video.wrk_height/16;
		pixval = 16;
		device.state_gpu = clSetKernelArg(device.luma_search_last_16x, 0, sizeof(cl_mem), &device.current_frame_Y_downsampled_by16);
		device.state_gpu = clSetKernelArg(device.luma_search_last_16x, 1, sizeof(cl_mem), &device.last_frame_Y_downsampled_by16);
		device.state_gpu = clSetKernelArg(device.luma_search_last_16x, 2, sizeof(cl_mem), &device.last_vnet1);
		device.state_gpu = clSetKernelArg(device.luma_search_last_16x, 3, sizeof(cl_mem), &device.last_vnet2);
		device.state_gpu = clSetKernelArg(device.luma_search_last_16x, 4, sizeof(cl_int), &net_width);
		device.state_gpu = clSetKernelArg(device.luma_search_last_16x, 5, sizeof(cl_int), &w);
		device.state_gpu = clSetKernelArg(device.luma_search_last_16x, 6, sizeof(cl_int), &h);
		device.state_gpu = clSetKernelArg(device.luma_search_last_16x, 7, sizeof(cl_int), &pixval);
		//golden_16x
		device.state_gpu = clSetKernelArg(device.luma_search_golden_16x, 0, sizeof(cl_mem), &device.current_frame_Y_downsampled_by16);
		device.state_gpu = clSetKernelArg(device.luma_search_golden_16x, 1, sizeof(cl_mem), &device.golden_frame_Y_downsampled_by16);
		device.state_gpu = clSetKernelArg(device.luma_search_golden_16x, 2, sizeof(cl_mem), &device.golden_vnet1);
		device.state_gpu = clSetKernelArg(device.luma_search_golden_16x, 3, sizeof(cl_mem), &device.golden_vnet2);
		device.state_gpu = clSetKernelArg(device.luma_search_golden_16x, 4, sizeof(cl_int), &net_width);
		device.state_gpu = clSetKernelArg(device.luma_search_golden_16x, 5, sizeof(cl_int), &w);
		device.state_gpu = clSetKernelArg(device.luma_search_golden_16x, 6, sizeof(cl_int), &h);
		device.state_gpu = clSetKernelArg(device.luma_search_golden_16x, 7, sizeof(cl_int), &pixval);
		//altref_16x
		device.state_gpu = clSetKernelArg(device.luma_search_altref_16x, 0, sizeof(cl_mem), &device.current_frame_Y_downsampled_by16);
		device.state_gpu = clSetKernelArg(device.luma_search_altref_16x, 1, sizeof(cl_mem), &device.altref_frame_Y_downsampled_by16);
		device.state_gpu = clSetKernelArg(device.luma_search_altref_16x, 2, sizeof(cl_mem), &device.altref_vnet1);
		device.state_gpu = clSetKernelArg(device.luma_search_altref_16x, 3, sizeof(cl_mem), &device.altref_vnet2);
		device.state_gpu = clSetKernelArg(device.luma_search_altref_16x, 4, sizeof(cl_int), &net_width);
		device.state_gpu = clSetKernelArg(device.luma_search_altref_16x, 5, sizeof(cl_int), &w);
		device.state_gpu = clSetKernelArg(device.luma_search_altref_16x, 6, sizeof(cl_int), &h);
		device.state_gpu = clSetKernelArg(device.luma_search_altref_16x, 7, sizeof(cl_int), &pixval);
		//last_8x
		w = video.wrk_width/8;
		h = video.wrk_height/8;
		pixval = 8;
		device.state_gpu = clSetKernelArg(device.luma_search_last_8x, 0, sizeof(cl_mem), &device.current_frame_Y_downsampled_by8);
		device.state_gpu = clSetKernelArg(device.luma_search_last_8x, 1, sizeof(cl_mem), &device.last_frame_Y_downsampled_by8);
		device.state_gpu = clSetKernelArg(device.luma_search_last_8x, 2, sizeof(cl_mem), &device.last_vnet2);
		device.state_gpu = clSetKernelArg(device.luma_search_last_8x, 3, sizeof(cl_mem), &device.last_vnet1);
		device.state_gpu = clSetKernelArg(device.luma_search_last_8x, 4, sizeof(cl_int), &net_width);
		device.state_gpu = clSetKernelArg(device.luma_search_last_8x, 5, sizeof(cl_int), &w);
		device.state_gpu = clSetKernelArg(device.luma_search_last_8x, 6, sizeof(cl_int), &h);
		device.state_gpu = clSetKernelArg(device.luma_search_last_8x, 7, sizeof(cl_int), &pixval);
		//golden_8x
		device.state_gpu = clSetKernelArg(device.luma_search_golden_8x, 0, sizeof(cl_mem), &device.current_frame_Y_downsampled_by8);
		device.state_gpu = clSetKernelArg(device.luma_search_golden_8x, 1, sizeof(cl_mem), &device.golden_frame_Y_downsampled_by8);
		device.state_gpu = clSetKernelArg(device.luma_search_golden_8x, 2, sizeof(cl_mem), &device.golden_vnet2);
		device.state_gpu = clSetKernelArg(device.luma_search_golden_8x, 3, sizeof(cl_mem), &device.golden_vnet1);
		device.state_gpu = clSetKernelArg(device.luma_search_golden_8x, 4, sizeof(cl_int), &net_width);
		device.state_gpu = clSetKernelArg(device.luma_search_golden_8x, 5, sizeof(cl_int), &w);
		device.state_gpu = clSetKernelArg(device.luma_search_golden_8x, 6, sizeof(cl_int), &h);
		device.state_gpu = clSetKernelArg(device.luma_search_golden_8x, 7, sizeof(cl_int), &pixval);
		//altref_8x
		device.state_gpu = clSetKernelArg(device.luma_search_altref_8x, 0, sizeof(cl_mem), &device.current_frame_Y_downsampled_by8);
		device.state_gpu = clSetKernelArg(device.luma_search_altref_8x, 1, sizeof(cl_mem), &device.altref_frame_Y_downsampled_by8);
		device.state_gpu = clSetKernelArg(device.luma_search_altref_8x, 2, sizeof(cl_mem), &device.altref_vnet2);
		device.state_gpu = clSetKernelArg(device.luma_search_altref_8x, 3, sizeof(cl_mem), &device.altref_vnet1);
		device.state_gpu = clSetKernelArg(device.luma_search_altref_8x, 4, sizeof(cl_int), &net_width);
		device.state_gpu = clSetKernelArg(device.luma_search_altref_8x, 5, sizeof(cl_int), &w);
		device.state_gpu = clSetKernelArg(device.luma_search_altref_8x, 6, sizeof(cl_int), &h);
		device.state_gpu = clSetKernelArg(device.luma_search_altref_8x, 7, sizeof(cl_int), &pixval);
		//last_4x
		w = video.wrk_width/4;
		h = video.wrk_height/4;
		pixval = 4;
		device.state_gpu = clSetKernelArg(device.luma_search_last_4x, 0, sizeof(cl_mem), &device.current_frame_Y_downsampled_by4);
		device.state_gpu = clSetKernelArg(device.luma_search_last_4x, 1, sizeof(cl_mem), &device.last_frame_Y_downsampled_by4);
		device.state_gpu = clSetKernelArg(device.luma_search_last_4x, 2, sizeof(cl_mem), &device.last_vnet1);
		device.state_gpu = clSetKernelArg(device.luma_search_last_4x, 3, sizeof(cl_mem), &device.last_vnet2);
		device.state_gpu = clSetKernelArg(device.luma_search_last_4x, 4, sizeof(cl_int), &net_width);
		device.state_gpu = clSetKernelArg(device.luma_search_last_4x, 5, sizeof(cl_int), &w);
		device.state_gpu = clSetKernelArg(device.luma_search_last_4x, 6, sizeof(cl_int), &h);
		device.state_gpu = clSetKernelArg(device.luma_search_last_4x, 7, sizeof(cl_int), &pixval);
		//golden_4x
		device.state_gpu = clSetKernelArg(device.luma_search_golden_4x, 0, sizeof(cl_mem), &device.current_frame_Y_downsampled_by4);
		device.state_gpu = clSetKernelArg(device.luma_search_golden_4x, 1, sizeof(cl_mem), &device.golden_frame_Y_downsampled_by4);
		device.state_gpu = clSetKernelArg(device.luma_search_golden_4x, 2, sizeof(cl_mem), &device.golden_vnet1);
		device.state_gpu = clSetKernelArg(device.luma_search_golden_4x, 3, sizeof(cl_mem), &device.golden_vnet2);
		device.state_gpu = clSetKernelArg(device.luma_search_golden_4x, 4, sizeof(cl_int), &net_width);
		device.state_gpu = clSetKernelArg(device.luma_search_golden_4x, 5, sizeof(cl_int), &w);
		device.state_gpu = clSetKernelArg(device.luma_search_golden_4x, 6, sizeof(cl_int), &h);
		device.state_gpu = clSetKernelArg(device.luma_search_golden_4x, 7, sizeof(cl_int), &pixval);
		//altref_4x
		device.state_gpu = clSetKernelArg(device.luma_search_altref_4x, 0, sizeof(cl_mem), &device.current_frame_Y_downsampled_by4);
		device.state_gpu = clSetKernelArg(device.luma_search_altref_4x, 1, sizeof(cl_mem), &device.altref_frame_Y_downsampled_by4);
		device.state_gpu = clSetKernelArg(device.luma_search_altref_4x, 2, sizeof(cl_mem), &device.altref_vnet1);
		device.state_gpu = clSetKernelArg(device.luma_search_altref_4x, 3, sizeof(cl_mem), &device.altref_vnet2);
		device.state_gpu = clSetKernelArg(device.luma_search_altref_4x, 4, sizeof(cl_int), &net_width);
		device.state_gpu = clSetKernelArg(device.luma_search_altref_4x, 5, sizeof(cl_int), &w);
		device.state_gpu = clSetKernelArg(device.luma_search_altref_4x, 6, sizeof(cl_int), &h);
		device.state_gpu = clSetKernelArg(device.luma_search_altref_4x, 7, sizeof(cl_int), &pixval);
		//last_2x
		w = video.wrk_width/2;
		h = video.wrk_height/2;
		pixval = 2;
		device.state_gpu = clSetKernelArg(device.luma_search_last_2x, 0, sizeof(cl_mem), &device.current_frame_Y_downsampled_by2);
		device.state_gpu = clSetKernelArg(device.luma_search_last_2x, 1, sizeof(cl_mem), &device.last_frame_Y_downsampled_by2);
		device.state_gpu = clSetKernelArg(device.luma_search_last_2x, 2, sizeof(cl_mem), &device.last_vnet2);
		device.state_gpu = clSetKernelArg(device.luma_search_last_2x, 3, sizeof(cl_mem), &device.last_vnet1);
		device.state_gpu = clSetKernelArg(device.luma_search_last_2x, 4, sizeof(cl_int), &net_width);
		device.state_gpu = clSetKernelArg(device.luma_search_last_2x, 5, sizeof(cl_int), &w);
		device.state_gpu = clSetKernelArg(device.luma_search_last_2x, 6, sizeof(cl_int), &h);
		device.state_gpu = clSetKernelArg(device.luma_search_last_2x, 7, sizeof(cl_int), &pixval);
		//golden_2x
		device.state_gpu = clSetKernelArg(device.luma_search_golden_2x, 0, sizeof(cl_mem), &device.current_frame_Y_downsampled_by2);
		device.state_gpu = clSetKernelArg(device.luma_search_golden_2x, 1, sizeof(cl_mem), &device.golden_frame_Y_downsampled_by2);
		device.state_gpu = clSetKernelArg(device.luma_search_golden_2x, 2, sizeof(cl_mem), &device.golden_vnet2);
		device.state_gpu = clSetKernelArg(device.luma_search_golden_2x, 3, sizeof(cl_mem), &device.golden_vnet1);
		device.state_gpu = clSetKernelArg(device.luma_search_golden_2x, 4, sizeof(cl_int), &net_width);
		device.state_gpu = clSetKernelArg(device.luma_search_golden_2x, 5, sizeof(cl_int), &w);
		device.state_gpu = clSetKernelArg(device.luma_search_golden_2x, 6, sizeof(cl_int), &h);
		device.state_gpu = clSetKernelArg(device.luma_search_golden_2x, 7, sizeof(cl_int), &pixval);
		//altref_2x
		device.state_gpu = clSetKernelArg(device.luma_search_altref_2x, 0, sizeof(cl_mem), &device.current_frame_Y_downsampled_by2);
		device.state_gpu = clSetKernelArg(device.luma_search_altref_2x, 1, sizeof(cl_mem), &device.altref_frame_Y_downsampled_by2);
		device.state_gpu = clSetKernelArg(device.luma_search_altref_2x, 2, sizeof(cl_mem), &device.altref_vnet2);
		device.state_gpu = clSetKernelArg(device.luma_search_altref_2x, 3, sizeof(cl_mem), &device.altref_vnet1);
		device.state_gpu = clSetKernelArg(device.luma_search_altref_2x, 4, sizeof(cl_int), &net_width);
		device.state_gpu = clSetKernelArg(device.luma_search_altref_2x, 5, sizeof(cl_int), &w);
		device.state_gpu = clSetKernelArg(device.luma_search_altref_2x, 6, sizeof(cl_int), &h);
		device.state_gpu = clSetKernelArg(device.luma_search_altref_2x, 7, sizeof(cl_int), &pixval);
		//last_1x
		w = video.wrk_width;
		h = video.wrk_height;
		pixval = 1;
		device.state_gpu = clSetKernelArg(device.luma_search_last_1x, 0, sizeof(cl_mem), &device.current_frame_Y);
		device.state_gpu = clSetKernelArg(device.luma_search_last_1x, 1, sizeof(cl_mem), &device.reconstructed_frame_Y);
		device.state_gpu = clSetKernelArg(device.luma_search_last_1x, 2, sizeof(cl_mem), &device.last_vnet1);
		device.state_gpu = clSetKernelArg(device.luma_search_last_1x, 3, sizeof(cl_mem), &device.last_vnet2);
		device.state_gpu = clSetKernelArg(device.luma_search_last_1x, 4, sizeof(cl_int), &net_width);
		device.state_gpu = clSetKernelArg(device.luma_search_last_1x, 5, sizeof(cl_int), &w);
		device.state_gpu = clSetKernelArg(device.luma_search_last_1x, 6, sizeof(cl_int), &h);
		device.state_gpu = clSetKernelArg(device.luma_search_last_1x, 7, sizeof(cl_int), &pixval);
		//golden_1x
		device.state_gpu = clSetKernelArg(device.luma_search_golden_1x, 0, sizeof(cl_mem), &device.current_frame_Y);
		device.state_gpu = clSetKernelArg(device.luma_search_golden_1x, 1, sizeof(cl_mem), &device.golden_frame_Y);
		device.state_gpu = clSetKernelArg(device.luma_search_golden_1x, 2, sizeof(cl_mem), &device.golden_vnet1);
		device.state_gpu = clSetKernelArg(device.luma_search_golden_1x, 3, sizeof(cl_mem), &device.golden_vnet2);
		device.state_gpu = clSetKernelArg(device.luma_search_golden_1x, 4, sizeof(cl_int), &net_width);
		device.state_gpu = clSetKernelArg(device.luma_search_golden_1x, 5, sizeof(cl_int), &w);
		device.state_gpu = clSetKernelArg(device.luma_search_golden_1x, 6, sizeof(cl_int), &h);
		device.state_gpu = clSetKernelArg(device.luma_search_golden_1x, 7, sizeof(cl_int), &pixval);
		//altref_1x
		device.state_gpu = clSetKernelArg(device.luma_search_altref_1x, 0, sizeof(cl_mem), &device.current_frame_Y);
		device.state_gpu = clSetKernelArg(device.luma_search_altref_1x, 1, sizeof(cl_mem), &device.altref_frame_Y);
		device.state_gpu = clSetKernelArg(device.luma_search_altref_1x, 2, sizeof(cl_mem), &device.altref_vnet1);
		device.state_gpu = clSetKernelArg(device.luma_search_altref_1x, 3, sizeof(cl_mem), &device.altref_vnet2);
		device.state_gpu = clSetKernelArg(device.luma_search_altref_1x, 4, sizeof(cl_int), &net_width);
		device.state_gpu = clSetKernelArg(device.luma_search_altref_1x, 5, sizeof(cl_int), &w);
		device.state_gpu = clSetKernelArg(device.luma_search_altref_1x, 6, sizeof(cl_int), &h);
		device.state_gpu = clSetKernelArg(device.luma_search_altref_1x, 7, sizeof(cl_int), &pixval);

		/*__kernel void luma_search_2step //searching in interpolated picture
									( 	__global uchar *const current_frame, //0
										__read_only image2d_t ref_frame, //1
										__global vector_net *const net, //2
										__global vector_net *const ref_net, //3
										__global int *const ref_Bdiff, //4
										const int width, //5
										const int height) //6*/
		// last interpolated
		device.state_gpu = clSetKernelArg(device.luma_search_last_d4x, 0, sizeof(cl_mem), &device.current_frame_Y);
		device.state_gpu = clSetKernelArg(device.luma_search_last_d4x, 1, sizeof(cl_mem), &device.last_frame_Y_image);
		device.state_gpu = clSetKernelArg(device.luma_search_last_d4x, 2, sizeof(cl_mem), &device.last_vnet2);
		device.state_gpu = clSetKernelArg(device.luma_search_last_d4x, 3, sizeof(cl_mem), &device.last_vnet1);
		device.state_gpu = clSetKernelArg(device.luma_search_last_d4x, 4, sizeof(cl_mem), &device.metrics1);
		device.state_gpu = clSetKernelArg(device.luma_search_last_d4x, 5, sizeof(cl_int), &video.wrk_width);
		device.state_gpu = clSetKernelArg(device.luma_search_last_d4x, 6, sizeof(cl_int), &video.wrk_height);
		// golden interpolated
		device.state_gpu = clSetKernelArg(device.luma_search_golden_d4x, 0, sizeof(cl_mem), &device.current_frame_Y);
		device.state_gpu = clSetKernelArg(device.luma_search_golden_d4x, 1, sizeof(cl_mem), &device.golden_frame_Y_image);
		device.state_gpu = clSetKernelArg(device.luma_search_golden_d4x, 2, sizeof(cl_mem), &device.golden_vnet2);
		device.state_gpu = clSetKernelArg(device.luma_search_golden_d4x, 3, sizeof(cl_mem), &device.golden_vnet1);
		device.state_gpu = clSetKernelArg(device.luma_search_golden_d4x, 4, sizeof(cl_mem), &device.metrics2);
		device.state_gpu = clSetKernelArg(device.luma_search_golden_d4x, 5, sizeof(cl_int), &video.wrk_width);
		device.state_gpu = clSetKernelArg(device.luma_search_golden_d4x, 6, sizeof(cl_int), &video.wrk_height);
		// altref interpolated
		device.state_gpu = clSetKernelArg(device.luma_search_altref_d4x, 0, sizeof(cl_mem), &device.current_frame_Y);
		device.state_gpu = clSetKernelArg(device.luma_search_altref_d4x, 1, sizeof(cl_mem), &device.altref_frame_Y_image);
		device.state_gpu = clSetKernelArg(device.luma_search_altref_d4x, 2, sizeof(cl_mem), &device.altref_vnet2);
		device.state_gpu = clSetKernelArg(device.luma_search_altref_d4x, 3, sizeof(cl_mem), &device.altref_vnet1);
		device.state_gpu = clSetKernelArg(device.luma_search_altref_d4x, 4, sizeof(cl_mem), &device.metrics3);
		device.state_gpu = clSetKernelArg(device.luma_search_altref_d4x, 5, sizeof(cl_int), &video.wrk_width);
		device.state_gpu = clSetKernelArg(device.luma_search_altref_d4x, 6, sizeof(cl_int), &video.wrk_height);

		/*__kernel void select_reference(__global vector_net *const last_net, //0
										__global vector_net *const golden_net, //1
										__global vector_net *const altref_net, //2
										__global int *const last_Bdiff, //3
										__global int *const golden_Bdiff, //4
										__global int *const altref_Bdiff, //5
										__global macroblock *const MBs, //6
										const int width, //7
										const int use_golden, //8
										const int use_altref) //9*/
		device.state_gpu = clSetKernelArg(device.select_reference, 0, sizeof(cl_mem), &device.last_vnet1);
		device.state_gpu = clSetKernelArg(device.select_reference, 1, sizeof(cl_mem), &device.golden_vnet1);
		device.state_gpu = clSetKernelArg(device.select_reference, 2, sizeof(cl_mem), &device.altref_vnet1);
		device.state_gpu = clSetKernelArg(device.select_reference, 3, sizeof(cl_mem), &device.metrics1);
		device.state_gpu = clSetKernelArg(device.select_reference, 4, sizeof(cl_mem), &device.metrics2);
		device.state_gpu = clSetKernelArg(device.select_reference, 5, sizeof(cl_mem), &device.metrics3);
		device.state_gpu = clSetKernelArg(device.select_reference, 6, sizeof(cl_mem), &device.transformed_blocks_gpu);
		device.state_gpu = clSetKernelArg(device.select_reference, 7, sizeof(cl_int), &video.wrk_width);

		/*__kernel void pack_8x8_into_16x16(__global macroblock *const MBs) //0*/
		device.state_gpu = clSetKernelArg(device.pack_8x8_into_16x16, 0, sizeof(cl_mem), &device.transformed_blocks_gpu);

		/*__kernel void prepare_predictors_and_residual(__global uchar *const current_frame, //0
														__read_only image2d_t ref_frame, //1
														__global uchar *const predictor, //2
														__global short *const residual, //3
														__global macroblock *const MBs, //4
														const int width, //5
														const int plane, //6
														const int ref) //7*/
		plane = 0; // Y
		ref = LAST;
		device.state_gpu = clSetKernelArg(device.prepare_predictors_and_residual_last_Y, 0, sizeof(cl_mem), &device.current_frame_Y);
		device.state_gpu = clSetKernelArg(device.prepare_predictors_and_residual_last_Y, 1, sizeof(cl_mem), &device.last_frame_Y_image);
		device.state_gpu = clSetKernelArg(device.prepare_predictors_and_residual_last_Y, 2, sizeof(cl_mem), &device.predictors_Y);
		device.state_gpu = clSetKernelArg(device.prepare_predictors_and_residual_last_Y, 3, sizeof(cl_mem), &device.residual_Y);
		device.state_gpu = clSetKernelArg(device.prepare_predictors_and_residual_last_Y, 4, sizeof(cl_mem), &device.transformed_blocks_gpu);
		device.state_gpu = clSetKernelArg(device.prepare_predictors_and_residual_last_Y, 5, sizeof(cl_int), &video.wrk_width);
		device.state_gpu = clSetKernelArg(device.prepare_predictors_and_residual_last_Y, 6, sizeof(cl_int), &plane);
		device.state_gpu = clSetKernelArg(device.prepare_predictors_and_residual_last_Y, 7, sizeof(cl_int), &ref);
		ref = GOLDEN;
		device.state_gpu = clSetKernelArg(device.prepare_predictors_and_residual_golden_Y, 0, sizeof(cl_mem), &device.current_frame_Y);
		device.state_gpu = clSetKernelArg(device.prepare_predictors_and_residual_golden_Y, 1, sizeof(cl_mem), &device.golden_frame_Y_image);
		device.state_gpu = clSetKernelArg(device.prepare_predictors_and_residual_golden_Y, 2, sizeof(cl_mem), &device.predictors_Y);
		device.state_gpu = clSetKernelArg(device.prepare_predictors_and_residual_golden_Y, 3, sizeof(cl_mem), &device.residual_Y);
		device.state_gpu = clSetKernelArg(device.prepare_predictors_and_residual_golden_Y, 4, sizeof(cl_mem), &device.transformed_blocks_gpu);
		device.state_gpu = clSetKernelArg(device.prepare_predictors_and_residual_golden_Y, 5, sizeof(cl_int), &video.wrk_width);
		device.state_gpu = clSetKernelArg(device.prepare_predictors_and_residual_golden_Y, 6, sizeof(cl_int), &plane);
		device.state_gpu = clSetKernelArg(device.prepare_predictors_and_residual_golden_Y, 7, sizeof(cl_int), &ref);
		ref = ALTREF;
		device.state_gpu = clSetKernelArg(device.prepare_predictors_and_residual_altref_Y, 0, sizeof(cl_mem), &device.current_frame_Y);
		device.state_gpu = clSetKernelArg(device.prepare_predictors_and_residual_altref_Y, 1, sizeof(cl_mem), &device.altref_frame_Y_image);
		device.state_gpu = clSetKernelArg(device.prepare_predictors_and_residual_altref_Y, 2, sizeof(cl_mem), &device.predictors_Y);
		device.state_gpu = clSetKernelArg(device.prepare_predictors_and_residual_altref_Y, 3, sizeof(cl_mem), &device.residual_Y);
		device.state_gpu = clSetKernelArg(device.prepare_predictors_and_residual_altref_Y, 4, sizeof(cl_mem), &device.transformed_blocks_gpu);
		device.state_gpu = clSetKernelArg(device.prepare_predictors_and_residual_altref_Y, 5, sizeof(cl_int), &video.wrk_width);
		device.state_gpu = clSetKernelArg(device.prepare_predictors_and_residual_altref_Y, 6, sizeof(cl_int), &plane);
		device.state_gpu = clSetKernelArg(device.prepare_predictors_and_residual_altref_Y, 7, sizeof(cl_int), &ref);
		plane = 1; // U
		ref = LAST;
		device.state_gpu = clSetKernelArg(device.prepare_predictors_and_residual_last_U, 0, sizeof(cl_mem), &device.current_frame_U);
		device.state_gpu = clSetKernelArg(device.prepare_predictors_and_residual_last_U, 1, sizeof(cl_mem), &device.last_frame_U_image);
		device.state_gpu = clSetKernelArg(device.prepare_predictors_and_residual_last_U, 2, sizeof(cl_mem), &device.predictors_U);
		device.state_gpu = clSetKernelArg(device.prepare_predictors_and_residual_last_U, 3, sizeof(cl_mem), &device.residual_U);
		device.state_gpu = clSetKernelArg(device.prepare_predictors_and_residual_last_U, 4, sizeof(cl_mem), &device.transformed_blocks_gpu);
		device.state_gpu = clSetKernelArg(device.prepare_predictors_and_residual_last_U, 5, sizeof(cl_int), &cwidth);
		device.state_gpu = clSetKernelArg(device.prepare_predictors_and_residual_last_U, 6, sizeof(cl_int), &plane);
		device.state_gpu = clSetKernelArg(device.prepare_predictors_and_residual_last_U, 7, sizeof(cl_int), &ref);
		ref = GOLDEN;
		device.state_gpu = clSetKernelArg(device.prepare_predictors_and_residual_golden_U, 0, sizeof(cl_mem), &device.current_frame_U);
		device.state_gpu = clSetKernelArg(device.prepare_predictors_and_residual_golden_U, 1, sizeof(cl_mem), &device.golden_frame_U_image);
		device.state_gpu = clSetKernelArg(device.prepare_predictors_and_residual_golden_U, 2, sizeof(cl_mem), &device.predictors_U);
		device.state_gpu = clSetKernelArg(device.prepare_predictors_and_residual_golden_U, 3, sizeof(cl_mem), &device.residual_U);
		device.state_gpu = clSetKernelArg(device.prepare_predictors_and_residual_golden_U, 4, sizeof(cl_mem), &device.transformed_blocks_gpu);
		device.state_gpu = clSetKernelArg(device.prepare_predictors_and_residual_golden_U, 5, sizeof(cl_int), &cwidth);
		device.state_gpu = clSetKernelArg(device.prepare_predictors_and_residual_golden_U, 6, sizeof(cl_int), &plane);
		device.state_gpu = clSetKernelArg(device.prepare_predictors_and_residual_golden_U, 7, sizeof(cl_int), &ref);
		ref = ALTREF;
		device.state_gpu = clSetKernelArg(device.prepare_predictors_and_residual_altref_U, 0, sizeof(cl_mem), &device.current_frame_U);
		device.state_gpu = clSetKernelArg(device.prepare_predictors_and_residual_altref_U, 1, sizeof(cl_mem), &device.altref_frame_U_image);
		device.state_gpu = clSetKernelArg(device.prepare_predictors_and_residual_altref_U, 2, sizeof(cl_mem), &device.predictors_U);
		device.state_gpu = clSetKernelArg(device.prepare_predictors_and_residual_altref_U, 3, sizeof(cl_mem), &device.residual_U);
		device.state_gpu = clSetKernelArg(device.prepare_predictors_and_residual_altref_U, 4, sizeof(cl_mem), &device.transformed_blocks_gpu);
		device.state_gpu = clSetKernelArg(device.prepare_predictors_and_residual_altref_U, 5, sizeof(cl_int), &cwidth);
		device.state_gpu = clSetKernelArg(device.prepare_predictors_and_residual_altref_U, 6, sizeof(cl_int), &plane);
		device.state_gpu = clSetKernelArg(device.prepare_predictors_and_residual_altref_U, 7, sizeof(cl_int), &ref);
		plane = 2; // V
		ref = LAST;
		device.state_gpu = clSetKernelArg(device.prepare_predictors_and_residual_last_V, 0, sizeof(cl_mem), &device.current_frame_V);
		device.state_gpu = clSetKernelArg(device.prepare_predictors_and_residual_last_V, 1, sizeof(cl_mem), &device.last_frame_V_image);
		device.state_gpu = clSetKernelArg(device.prepare_predictors_and_residual_last_V, 2, sizeof(cl_mem), &device.predictors_V);
		device.state_gpu = clSetKernelArg(device.prepare_predictors_and_residual_last_V, 3, sizeof(cl_mem), &device.residual_V);
		device.state_gpu = clSetKernelArg(device.prepare_predictors_and_residual_last_V, 4, sizeof(cl_mem), &device.transformed_blocks_gpu);
		device.state_gpu = clSetKernelArg(device.prepare_predictors_and_residual_last_V, 5, sizeof(cl_int), &cwidth);
		device.state_gpu = clSetKernelArg(device.prepare_predictors_and_residual_last_V, 6, sizeof(cl_int), &plane);
		device.state_gpu = clSetKernelArg(device.prepare_predictors_and_residual_last_V, 7, sizeof(cl_int), &ref);
		ref = GOLDEN;
		device.state_gpu = clSetKernelArg(device.prepare_predictors_and_residual_golden_V, 0, sizeof(cl_mem), &device.current_frame_V);
		device.state_gpu = clSetKernelArg(device.prepare_predictors_and_residual_golden_V, 1, sizeof(cl_mem), &device.golden_frame_V_image);
		device.state_gpu = clSetKernelArg(device.prepare_predictors_and_residual_golden_V, 2, sizeof(cl_mem), &device.predictors_V);
		device.state_gpu = clSetKernelArg(device.prepare_predictors_and_residual_golden_V, 3, sizeof(cl_mem), &device.residual_V);
		device.state_gpu = clSetKernelArg(device.prepare_predictors_and_residual_golden_V, 4, sizeof(cl_mem), &device.transformed_blocks_gpu);
		device.state_gpu = clSetKernelArg(device.prepare_predictors_and_residual_golden_V, 5, sizeof(cl_int), &cwidth);
		device.state_gpu = clSetKernelArg(device.prepare_predictors_and_residual_golden_V, 6, sizeof(cl_int), &plane);
		device.state_gpu = clSetKernelArg(device.prepare_predictors_and_residual_golden_V, 7, sizeof(cl_int), &ref);
		ref = ALTREF;
		device.state_gpu = clSetKernelArg(device.prepare_predictors_and_residual_altref_V, 0, sizeof(cl_mem), &device.current_frame_V);
		device.state_gpu = clSetKernelArg(device.prepare_predictors_and_residual_altref_V, 1, sizeof(cl_mem), &device.altref_frame_V_image);
		device.state_gpu = clSetKernelArg(device.prepare_predictors_and_residual_altref_V, 2, sizeof(cl_mem), &device.predictors_V);
		device.state_gpu = clSetKernelArg(device.prepare_predictors_and_residual_altref_V, 3, sizeof(cl_mem), &device.residual_V);
		device.state_gpu = clSetKernelArg(device.prepare_predictors_and_residual_altref_V, 4, sizeof(cl_mem), &device.transformed_blocks_gpu);
		device.state_gpu = clSetKernelArg(device.prepare_predictors_and_residual_altref_V, 5, sizeof(cl_int), &cwidth);
		device.state_gpu = clSetKernelArg(device.prepare_predictors_and_residual_altref_V, 6, sizeof(cl_int), &plane);
		device.state_gpu = clSetKernelArg(device.prepare_predictors_and_residual_altref_V, 7, sizeof(cl_int), &ref);

		for (seg_id = UQ_segment; seg_id <= LQ_segment; ++seg_id)
		{
			/*__kernel void dct4x4(__global short *const residual, //0
									__global macroblock *const MBs, //1
									const int width, //2
									__constant segment_data *const SD, //3
									const int segment_id, //4
									const float SSIM_target, //5
									const int plane) //6*/

			plane = 0;
			device.state_gpu = clSetKernelArg(device.dct4x4_Y[seg_id], 0, sizeof(cl_mem), &device.residual_Y);
			device.state_gpu = clSetKernelArg(device.dct4x4_Y[seg_id], 1, sizeof(cl_mem), &device.transformed_blocks_gpu);
			device.state_gpu = clSetKernelArg(device.dct4x4_Y[seg_id], 2, sizeof(cl_int), &video.wrk_width);
			device.state_gpu = clSetKernelArg(device.dct4x4_Y[seg_id], 3, sizeof(cl_mem), &device.segments_data_gpu);
			device.state_gpu = clSetKernelArg(device.dct4x4_Y[seg_id], 4, sizeof(cl_int), &seg_id);
			device.state_gpu = clSetKernelArg(device.dct4x4_Y[seg_id], 5, sizeof(cl_int), &video.SSIM_target);
			device.state_gpu = clSetKernelArg(device.dct4x4_Y[seg_id], 6, sizeof(cl_int), &plane);
			plane = 1;
			device.state_gpu = clSetKernelArg(device.dct4x4_U[seg_id], 0, sizeof(cl_mem), &device.residual_U);
			device.state_gpu = clSetKernelArg(device.dct4x4_U[seg_id], 1, sizeof(cl_mem), &device.transformed_blocks_gpu);
			device.state_gpu = clSetKernelArg(device.dct4x4_U[seg_id], 2, sizeof(cl_int), &cwidth);
			device.state_gpu = clSetKernelArg(device.dct4x4_U[seg_id], 3, sizeof(cl_mem), &device.segments_data_gpu);
			device.state_gpu = clSetKernelArg(device.dct4x4_U[seg_id], 4, sizeof(cl_int), &seg_id);
			device.state_gpu = clSetKernelArg(device.dct4x4_U[seg_id], 5, sizeof(cl_int), &video.SSIM_target);
			device.state_gpu = clSetKernelArg(device.dct4x4_U[seg_id], 6, sizeof(cl_int), &plane);
			plane = 2;
			device.state_gpu = clSetKernelArg(device.dct4x4_V[seg_id], 0, sizeof(cl_mem), &device.residual_V);
			device.state_gpu = clSetKernelArg(device.dct4x4_V[seg_id], 1, sizeof(cl_mem), &device.transformed_blocks_gpu);
			device.state_gpu = clSetKernelArg(device.dct4x4_V[seg_id], 2, sizeof(cl_int), &cwidth);
			device.state_gpu = clSetKernelArg(device.dct4x4_V[seg_id], 3, sizeof(cl_mem), &device.segments_data_gpu);
			device.state_gpu = clSetKernelArg(device.dct4x4_V[seg_id], 4, sizeof(cl_int), &seg_id);
			device.state_gpu = clSetKernelArg(device.dct4x4_V[seg_id], 5, sizeof(cl_int), &video.SSIM_target);
			device.state_gpu = clSetKernelArg(device.dct4x4_V[seg_id], 6, sizeof(cl_int), &plane);

			/*__kernel void wht4x4_iwht4x4(__global macroblock *const MBs, //0
											__constant segment_data *const SD, //1
											const int segment_id, //2
											const float SSIM_target) //3*/

			device.state_gpu = clSetKernelArg(device.wht4x4_iwht4x4[seg_id], 0, sizeof(cl_mem), &device.transformed_blocks_gpu);
			device.state_gpu = clSetKernelArg(device.wht4x4_iwht4x4[seg_id], 1, sizeof(cl_mem), &device.segments_data_gpu);
			device.state_gpu = clSetKernelArg(device.wht4x4_iwht4x4[seg_id], 2, sizeof(cl_int), &seg_id);
			device.state_gpu = clSetKernelArg(device.wht4x4_iwht4x4[seg_id], 3, sizeof(cl_int), &video.SSIM_target);
		
			/*__kernel void idct4x4(__global uchar *const recon_frame, //0
									__global uchar *const predictor, //1
									__global macroblock *const MBs, //2
									const int width, //3
									__constant segment_data *const SD, //4
									const int segment_id, //5
									const float SSIM_target, //6
									const int plane) //7*/
			plane = 0;
			device.state_gpu = clSetKernelArg(device.idct4x4_Y[seg_id], 0, sizeof(cl_mem), &device.reconstructed_frame_Y);
			device.state_gpu = clSetKernelArg(device.idct4x4_Y[seg_id], 1, sizeof(cl_mem), &device.predictors_Y);
			device.state_gpu = clSetKernelArg(device.idct4x4_Y[seg_id], 2, sizeof(cl_mem), &device.transformed_blocks_gpu);
			device.state_gpu = clSetKernelArg(device.idct4x4_Y[seg_id], 3, sizeof(cl_int), &video.wrk_width);
			device.state_gpu = clSetKernelArg(device.idct4x4_Y[seg_id], 4, sizeof(cl_mem), &device.segments_data_gpu);
			device.state_gpu = clSetKernelArg(device.idct4x4_Y[seg_id], 5, sizeof(cl_int), &seg_id);
			device.state_gpu = clSetKernelArg(device.idct4x4_Y[seg_id], 6, sizeof(cl_int), &video.SSIM_target);
			device.state_gpu = clSetKernelArg(device.idct4x4_Y[seg_id], 7, sizeof(cl_int), &plane);
			plane = 1;
			device.state_gpu = clSetKernelArg(device.idct4x4_U[seg_id], 0, sizeof(cl_mem), &device.reconstructed_frame_U);
			device.state_gpu = clSetKernelArg(device.idct4x4_U[seg_id], 1, sizeof(cl_mem), &device.predictors_U);
			device.state_gpu = clSetKernelArg(device.idct4x4_U[seg_id], 2, sizeof(cl_mem), &device.transformed_blocks_gpu);
			device.state_gpu = clSetKernelArg(device.idct4x4_U[seg_id], 3, sizeof(cl_int), &cwidth);
			device.state_gpu = clSetKernelArg(device.idct4x4_U[seg_id], 4, sizeof(cl_mem), &device.segments_data_gpu);
			device.state_gpu = clSetKernelArg(device.idct4x4_U[seg_id], 5, sizeof(cl_int), &seg_id);
			device.state_gpu = clSetKernelArg(device.idct4x4_U[seg_id], 6, sizeof(cl_int), &video.SSIM_target);
			device.state_gpu = clSetKernelArg(device.idct4x4_U[seg_id], 7, sizeof(cl_int), &plane);
			plane = 2;
			device.state_gpu = clSetKernelArg(device.idct4x4_V[seg_id], 0, sizeof(cl_mem), &device.reconstructed_frame_V);
			device.state_gpu = clSetKernelArg(device.idct4x4_V[seg_id], 1, sizeof(cl_mem), &device.predictors_V);
			device.state_gpu = clSetKernelArg(device.idct4x4_V[seg_id], 2, sizeof(cl_mem), &device.transformed_blocks_gpu);
			device.state_gpu = clSetKernelArg(device.idct4x4_V[seg_id], 3, sizeof(cl_int), &cwidth);
			device.state_gpu = clSetKernelArg(device.idct4x4_V[seg_id], 4, sizeof(cl_mem), &device.segments_data_gpu);
			device.state_gpu = clSetKernelArg(device.idct4x4_V[seg_id], 5, sizeof(cl_int), &seg_id);
			device.state_gpu = clSetKernelArg(device.idct4x4_V[seg_id], 6, sizeof(cl_int), &video.SSIM_target);
			device.state_gpu = clSetKernelArg(device.idct4x4_V[seg_id], 7, sizeof(cl_int), &plane);

			/*__kernel void count_SSIM_luma(__global uchar *const frame1, //0
										__global uchar *const frame2, //1
										__global macroblock *const MBs, //2
										__global float *const metric, //3
										signed int width, //4
										const int segment_id)//5*/
			device.state_gpu = clSetKernelArg(device.count_SSIM_luma[seg_id], 0, sizeof(cl_mem), &device.current_frame_Y);
			device.state_gpu = clSetKernelArg(device.count_SSIM_luma[seg_id], 1, sizeof(cl_mem), &device.reconstructed_frame_Y);
			device.state_gpu = clSetKernelArg(device.count_SSIM_luma[seg_id], 2, sizeof(cl_mem), &device.transformed_blocks_gpu);
			device.state_gpu = clSetKernelArg(device.count_SSIM_luma[seg_id], 3, sizeof(cl_mem), &device.metrics1);
			device.state_gpu = clSetKernelArg(device.count_SSIM_luma[seg_id], 4, sizeof(cl_int), &video.wrk_width);
			device.state_gpu = clSetKernelArg(device.count_SSIM_luma[seg_id], 5, sizeof(cl_int), &seg_id);

			/*__kernel void count_SSIM_chroma(__global uchar *const frame1, //0
												__global uchar *const frame2, //1
												__global macroblock *const MBs, //2
												__global float *const metric, //3
												signed int cwidth, //4
												const int segment_id)// 5*/
			device.state_gpu = clSetKernelArg(device.count_SSIM_chroma_U[seg_id], 0, sizeof(cl_mem), &device.current_frame_U);
			device.state_gpu = clSetKernelArg(device.count_SSIM_chroma_U[seg_id], 1, sizeof(cl_mem), &device.reconstructed_frame_U);
			device.state_gpu = clSetKernelArg(device.count_SSIM_chroma_U[seg_id], 2, sizeof(cl_mem), &device.transformed_blocks_gpu);
			device.state_gpu = clSetKernelArg(device.count_SSIM_chroma_U[seg_id], 3, sizeof(cl_mem), &device.metrics2);
			device.state_gpu = clSetKernelArg(device.count_SSIM_chroma_U[seg_id], 4, sizeof(cl_int), &cwidth);
			device.state_gpu = clSetKernelArg(device.count_SSIM_chroma_U[seg_id], 5, sizeof(cl_int), &seg_id);
			device.state_gpu = clSetKernelArg(device.count_SSIM_chroma_V[seg_id], 0, sizeof(cl_mem), &device.current_frame_V);
			device.state_gpu = clSetKernelArg(device.count_SSIM_chroma_V[seg_id], 1, sizeof(cl_mem), &device.reconstructed_frame_V);
			device.state_gpu = clSetKernelArg(device.count_SSIM_chroma_V[seg_id], 2, sizeof(cl_mem), &device.transformed_blocks_gpu);
			device.state_gpu = clSetKernelArg(device.count_SSIM_chroma_V[seg_id], 3, sizeof(cl_mem), &device.metrics3);
			device.state_gpu = clSetKernelArg(device.count_SSIM_chroma_V[seg_id], 4, sizeof(cl_int), &cwidth);
			device.state_gpu = clSetKernelArg(device.count_SSIM_chroma_V[seg_id], 5, sizeof(cl_int), &seg_id);
		}
		
		if (video.do_loop_filter_on_gpu)
		{
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
			device.state_gpu = clSetKernelArg(device.normal_loop_filter_MBH, 2, sizeof(cl_mem), &device.segments_data_gpu);
			device.state_gpu = clSetKernelArg(device.normal_loop_filter_MBH, 3, sizeof(cl_mem), &device.transformed_blocks_gpu);
			device.state_gpu = clSetKernelArg(device.normal_loop_filter_MBH, 6, sizeof(cl_mem), &device.mb_mask);

			/*__kernel void normal_loop_filter_MBV(__global uchar * const frame, //0
													const int width, //1
													__constant segment_data *const SD, //2
													__global macroblock *const MB, //3
													const int mb_size, //4
													const int stage, //5
													__global int *const mb_mask) //6*/
			device.state_gpu = clSetKernelArg(device.normal_loop_filter_MBV, 2, sizeof(cl_mem), &device.segments_data_gpu);
			device.state_gpu = clSetKernelArg(device.normal_loop_filter_MBV, 3, sizeof(cl_mem), &device.transformed_blocks_gpu);
			device.state_gpu = clSetKernelArg(device.normal_loop_filter_MBV, 6, sizeof(cl_mem), &device.mb_mask);
		}
		
		/*void __kernel gather_SSIM(__global float *const metric1, //0
							__global float *const metric2, //1
							__global float *const metric3, //2
							__global macroblock *const MBs) //3*/
		device.state_gpu = clSetKernelArg(device.gather_SSIM, 0, sizeof(cl_mem), &device.metrics1);
		device.state_gpu = clSetKernelArg(device.gather_SSIM, 1, sizeof(cl_mem), &device.metrics2);
		device.state_gpu = clSetKernelArg(device.gather_SSIM, 2, sizeof(cl_mem), &device.metrics3);
		device.state_gpu = clSetKernelArg(device.gather_SSIM, 3, sizeof(cl_mem), &device.transformed_blocks_gpu);

		device.commandQueue1_gpu = clCreateCommandQueue(device.context_gpu, device.device_gpu[0], 0, &device.state_gpu);
		device.commandQueue2_gpu = clCreateCommandQueue(device.context_gpu, device.device_gpu[0], 0, &device.state_gpu);
		device.commandQueue3_gpu = clCreateCommandQueue(device.context_gpu, device.device_gpu[0], 0, &device.state_gpu);
	}

	/*__kernel void encode_coefficients(	__global macroblock *MBs, - 0
											__global uchar *output, - 1
											__global int *partition_sizes, - 2
											__global uchar *third_context, - 3
											__global uint *coeff_probs, - 4
											int mb_height, - 5
											int mb_width, - 6
											int num_partitions, - 7
											int partition_step) - 8 */
	
	device.state_cpu = clSetKernelArg(device.encode_coefficients, 0, sizeof(cl_mem), &device.transformed_blocks_cpu);
	device.state_cpu = clSetKernelArg(device.encode_coefficients, 1, sizeof(cl_mem), &device.partitions);
	device.state_cpu = clSetKernelArg(device.encode_coefficients, 2, sizeof(cl_mem), &device.partitions_sizes);
	device.state_cpu = clSetKernelArg(device.encode_coefficients, 3, sizeof(cl_mem), &device.third_context);
	device.state_cpu = clSetKernelArg(device.encode_coefficients, 4, sizeof(cl_mem), &device.coeff_probs);
	device.state_cpu = clSetKernelArg(device.encode_coefficients, 5, sizeof(cl_int), &video.mb_height);
	device.state_cpu = clSetKernelArg(device.encode_coefficients, 6, sizeof(cl_int), &video.mb_width);
	device.state_cpu = clSetKernelArg(device.encode_coefficients, 7, sizeof(cl_int), &video.number_of_partitions);
	video.partition_step = video.partition_step / video.number_of_partitions;
	device.state_cpu = clSetKernelArg(device.encode_coefficients, 8, sizeof(cl_int), &video.partition_step);

	/*__kernel void count_probs(	__global macroblock *MBs, - 0
									__global uint *coeff_probs, - 1
									__global uint *coeff_probs_denom, - 2
									__global uchar *third_context, - 3
									int mb_height, - 4
									int mb_width, - 5
									int num_partitions, - 6
									int partition_step) - 7 */
	
	device.state_cpu = clSetKernelArg(device.count_probs, 0, sizeof(cl_mem), &device.transformed_blocks_cpu);
	device.state_cpu = clSetKernelArg(device.count_probs, 1, sizeof(cl_mem), &device.coeff_probs);
	device.state_cpu = clSetKernelArg(device.count_probs, 2, sizeof(cl_mem), &device.coeff_probs_denom);
	device.state_cpu = clSetKernelArg(device.count_probs, 3, sizeof(cl_mem), &device.third_context);
	device.state_cpu = clSetKernelArg(device.count_probs, 4, sizeof(cl_int), &video.mb_height);
	device.state_cpu = clSetKernelArg(device.count_probs, 5, sizeof(cl_int), &video.mb_width);
	device.state_cpu = clSetKernelArg(device.count_probs, 6, sizeof(cl_int), &video.number_of_partitions);
	device.state_cpu = clSetKernelArg(device.count_probs, 7, sizeof(cl_int), &video.partition_step);

	/*__kernel void num_div_denom(	__global uint *coeff_probs, 
									__global uint *coeff_probs_denom,
									int num_partitions)*/
	device.state_cpu = clSetKernelArg(device.num_div_denom, 0, sizeof(cl_mem), &device.coeff_probs);
	device.state_cpu = clSetKernelArg(device.num_div_denom, 1, sizeof(cl_mem), &device.coeff_probs_denom);
	device.state_cpu = clSetKernelArg(device.num_div_denom, 2, sizeof(cl_int), &video.number_of_partitions);

	if (!video.do_loop_filter_on_gpu)
	{
		/*__kernel void prepare_filter_mask(__global macroblock *const MBs, //0
											__global int *const mb_mask, //1
											const int width, //2
											const int height, //3
											const int parts) //4*/
		cl_int parts = 4;
		device.state_cpu = clSetKernelArg(device.prepare_filter_mask, 0, sizeof(cl_mem), &device.transformed_blocks_cpu);
		device.state_cpu = clSetKernelArg(device.prepare_filter_mask, 1, sizeof(cl_mem), &device.mb_mask);
		device.state_cpu = clSetKernelArg(device.prepare_filter_mask, 2, sizeof(cl_int), &video.wrk_height);
		device.state_cpu = clSetKernelArg(device.prepare_filter_mask, 3, sizeof(cl_int), &video.wrk_width);
		device.state_cpu = clSetKernelArg(device.prepare_filter_mask, 4, sizeof(cl_int), &parts);

		/*__kernel void loop_filter_frame_luma(__global uchar *const frame, //0
												__global macroblock *const MBs, //1
												__global int *const mb_mask, //2
												__constant const segment_data *const SD, //3
												const int width, //4
												const int height) //5*/
		device.state_cpu = clSetKernelArg(device.loop_filter_frame_luma, 0, sizeof(cl_mem), &device.cpu_frame_Y);
		device.state_cpu = clSetKernelArg(device.loop_filter_frame_luma, 1, sizeof(cl_mem), &device.transformed_blocks_cpu);
		device.state_cpu = clSetKernelArg(device.loop_filter_frame_luma, 2, sizeof(cl_mem), &device.mb_mask);
		device.state_gpu = clSetKernelArg(device.loop_filter_frame_luma, 3, sizeof(cl_mem), &device.segments_data_cpu);
		device.state_cpu = clSetKernelArg(device.loop_filter_frame_luma, 4, sizeof(cl_int), &video.wrk_width);
		device.state_cpu = clSetKernelArg(device.loop_filter_frame_luma, 5, sizeof(cl_int), &video.wrk_height);

		/*__kernel void loop_filter_frame_chroma(__global uchar *const frame, //0
												__global macroblock *const MBs, //1
												__global int *const mb_mask, //2
												__constant const segment_data *const SD, //3
												const int width, //4
												const int height) //5*/
		device.state_cpu = clSetKernelArg(device.loop_filter_frame_chroma_U, 0, sizeof(cl_mem), &device.cpu_frame_U);
		device.state_cpu = clSetKernelArg(device.loop_filter_frame_chroma_U, 1, sizeof(cl_mem), &device.transformed_blocks_cpu);
		device.state_cpu = clSetKernelArg(device.loop_filter_frame_chroma_U, 2, sizeof(cl_mem), &device.mb_mask);
		device.state_gpu = clSetKernelArg(device.loop_filter_frame_chroma_U, 3, sizeof(cl_mem), &device.segments_data_cpu);
		device.state_cpu = clSetKernelArg(device.loop_filter_frame_chroma_U, 4, sizeof(cl_int), &cwidth);
		device.state_cpu = clSetKernelArg(device.loop_filter_frame_chroma_U, 5, sizeof(cl_int), &cheight);
		device.state_cpu = clSetKernelArg(device.loop_filter_frame_chroma_V, 0, sizeof(cl_mem), &device.cpu_frame_V);
		device.state_cpu = clSetKernelArg(device.loop_filter_frame_chroma_V, 1, sizeof(cl_mem), &device.transformed_blocks_cpu);
		device.state_cpu = clSetKernelArg(device.loop_filter_frame_chroma_V, 2, sizeof(cl_mem), &device.mb_mask);
		device.state_gpu = clSetKernelArg(device.loop_filter_frame_chroma_V, 3, sizeof(cl_mem), &device.segments_data_cpu);
		device.state_cpu = clSetKernelArg(device.loop_filter_frame_chroma_V, 4, sizeof(cl_int), &cwidth);
		device.state_cpu = clSetKernelArg(device.loop_filter_frame_chroma_V, 5, sizeof(cl_int), &cheight);
	}

	device.loopfilterY_commandQueue_cpu = clCreateCommandQueue(device.context_cpu, device.device_cpu[0], 0, &device.state_cpu);
	device.loopfilterU_commandQueue_cpu = clCreateCommandQueue(device.context_cpu, device.device_cpu[0], 0, &device.state_cpu);
	device.loopfilterV_commandQueue_cpu = clCreateCommandQueue(device.context_cpu, device.device_cpu[0], 0, &device.state_cpu);
	device.boolcoder_commandQueue_cpu = clCreateCommandQueue(device.context_cpu, device.device_cpu[0], 0, &device.state_cpu);
	return 1;
}

static int string_to_value(char *str)
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

static int ParseArgs(int argc, char *argv[])
{
    char f_o = 0, f_i = 0, f_qmax = 0, f_qmin = 0, f_qintra = 0, f_g = 0, f_partitions = 0, f_threads = 0, f_gpupn = 0,f_SSIM_target=0,f_altref_range=0; 
	int i,ii;
    i = 1;
	video.do_loop_filter_on_gpu = 0;
	video.print_info = 0;
	device.gpu_preferred_platform_number = 0;
    while (i < argc)
    {
		ii = i;
        if (argv[i][0] == '-')
        {
			if ((argv[i][1] == 'h') && ((argv[i][2] == '\n') || (argv[i][2] == '\0')))
			{
				printf("%s\n", small_help);
				cl_info();
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
			if (memcmp(&argv[i][1], "threads", 7)==0)
            {
                ++i;
                if (i < argc)
                {
					video.thread_limit = string_to_value(argv[i]);
					if (video.thread_limit < 1)
					{
						printf ("wrong maximum number of threads;\n");
						return -1;
					}
                    f_threads = 1;
					if (++i >= argc) break;
                }
                else
                {
                    printf ("no value for threads;\n");
                    return -1;
                }
            }
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
            if (memcmp(&argv[i][1], "partitions", 7)==0)
            {
                ++i;
                if (i < argc)
                {
					video.number_of_partitions = (size_t)string_to_value(argv[i]);
					if (video.number_of_partitions < 0)
					{
						printf ("wrong number of partitions;\n");
						return -1;
					}
					// real number to binary index for bool encoder
					if (video.number_of_partitions == 1) 
						video.number_of_partitions_ind = 0;
					else if (video.number_of_partitions == 2) 
						video.number_of_partitions_ind = 1;
					else if (video.number_of_partitions == 4) 
						video.number_of_partitions_ind = 2;
					else if (video.number_of_partitions == 8)
						video.number_of_partitions_ind = 3;
					else 
					{
						video.number_of_partitions = 1;
						video.number_of_partitions_ind = 0;
					}
                    f_partitions = 1;
					if (++i >= argc) break;
                }
                else
                {
                    printf ("no value for partitions;\n");
                    return -1;
                }
            }
			if (memcmp(&argv[i][1], "gpu-preferred-platform-number", 29) == 0)
            {
                ++i;
                if (i < argc)
                {
					device.gpu_preferred_platform_number = string_to_value(argv[i]);
					if (video.loop_filter_sharpness < 0)
					{
						printf ("wrong format for gpu-preferred-platform-number! must be a positive integer;\n");
						return -1;
					}
                    f_gpupn = 1;
					if (++i >= argc) break;
                }
                else
                {
                    printf ("no value for gpu-preferred-platform-number;\n");
                    return -1;
                }
            }
			if (memcmp(&argv[i][1], "SSIM-target", 11)==0)
            {
                ++i;
                if (i < argc)
                {
					int buf = string_to_value(argv[i]);
					if ((buf < 0) || (buf > 99))
					{
						printf ("wrong SSIM level! must be an integer from 0 to 99;\n");
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
			if (memcmp(&argv[i][1], "loop-filter-on-gpu", 11)==0)
            {
				video.do_loop_filter_on_gpu = 1;
				if (++i >= argc) break;
            }
			if (memcmp(&argv[i][1], "print-info", 10)==0)
            {
				video.print_info = 1;
				if (++i >= argc) break;
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
	if (f_partitions == 0)
    {
		printf("no number of partitions specified - set to 1;\n");
		video.number_of_partitions = 1;
    }
	if (f_threads == 0)
	{
		printf("no maximum number of threads specified - set to 2;\n");
		video.thread_limit = 2;
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
		cl_int buf = video.qi_max;
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

	frames.threads_free = video.thread_limit;

    return 0;
}

static int OpenYUV420FileAndParseHeader()
{
    // add framerate
	char magic_word[] = "YUV4MPEG2 ";
	char ch;
	int frame_start = 0;
	if (input_file.path[0] == '@') {
		input_file.handle = stdin;
#ifdef _WIN32
		setmode(0, O_BINARY); //0x8000
#endif
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
