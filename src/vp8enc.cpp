//a way to define path to OpenCL.lib
#pragma comment(lib,"C:\\Program Files (x86)\\AMD APP SDK\\2.9\\lib\\x86\\OpenCL.lib")

// all global structure definitions (fileContext, videoContext, deviceContext...)
#include "vp8enc.h"

//these are global variables used all over the encoder
struct fileContext	input_file, //YUV4MPEG2 
					output_file, //IVF
					error_file, //TXT for OpenCl compiler errors
					dump_file; //YUV4MPEG2 dump of reconstructed frames
struct deviceContext device; //both GPU and CPU OpenCL-devices (different handles, memory-objects, commlines...)
struct videoContext video; //properties of the video (sizes, indicies, vector limits...)
struct hostFrameBuffers frames; // host buffers, frame number, current/previous frame flags...

#include "encIO.h"
#include "init.h"
#include "intra_part.h"
#include "inter_part.h"
#include "loop_filter.h"
#include "debug.h"

////////////////// transforms are taken from multimedia mike's encoder version

extern void encode_header(cl_uchar* partition);

void entropy_encode()
{
	if (frames.threads_free < video.number_of_partitions) {
		device.state_cpu = clFinish(device.loopfilterY_commandQueue_cpu);
		device.state_cpu = clFinish(device.loopfilterU_commandQueue_cpu);
		device.state_cpu = clFinish(device.loopfilterV_commandQueue_cpu);
		frames.threads_free = video.thread_limit;
	}

	// here we start preparing DCT coefficient probabilities for frame
	//             by calculating average for all situations 
	// count_probs - accumulate numerators(num) and denominators(denom)
	// for each context
	// num[i][j][k][l] - amount of ZEROs which must be coded in i,j,k,l context
	// denom[i][j][k][l] - amount of bits(both 0 and 1) in i,j,k,l context to be coded
	device.cpu_work_items_per_dim[0] = video.number_of_partitions;
	device.cpu_work_group_size_per_dim[0] = 1;
	clSetKernelArg(device.count_probs, 7, sizeof(cl_int), &frames.current_is_key_frame);	
	clEnqueueNDRangeKernel(device.boolcoder_commandQueue_cpu, device.count_probs, 1, NULL, device.cpu_work_items_per_dim, device.cpu_work_group_size_per_dim, 0, NULL, NULL);
	frames.threads_free -= video.number_of_partitions;
	clFinish(device.boolcoder_commandQueue_cpu); 

	// just dividing nums by denoms and getting probability of bit being ZERO
	clEnqueueNDRangeKernel(device.boolcoder_commandQueue_cpu, device.num_div_denom, 1, NULL, device.cpu_work_items_per_dim, device.cpu_work_group_size_per_dim, 0, NULL, NULL);
	clFinish(device.boolcoder_commandQueue_cpu); // Blocking, we need prob-values before encoding coeffs and header

	// read calculated values
	clEnqueueReadBuffer(device.boolcoder_commandQueue_cpu, device.coeff_probs ,CL_TRUE, 0, 11*3*8*4*sizeof(cl_uint), frames.new_probs, 0, NULL, NULL);
	clEnqueueReadBuffer(device.boolcoder_commandQueue_cpu, device.coeff_probs_denom ,CL_TRUE, 0, 11*3*8*4*sizeof(cl_uint), frames.new_probs_denom, 0, NULL, NULL);
	{ int i,j,k,l;
	for (i = 0; i < 4; ++i)
		for (j = 0; j < 8; ++j)
			for (k = 0; k < 3; ++k)
				for (l = 0; l < 11; ++l)
					if (frames.new_probs_denom[i][j][k][l] < 2) // this situation never happened (no bit encoded with this context)
						frames.new_probs[i][j][k][l] = k_default_coeff_probs[i][j][k][l];
	}
	device.state_gpu = clEnqueueWriteBuffer(device.boolcoder_commandQueue_cpu, device.coeff_probs, CL_TRUE, 0, 11*3*8*4*sizeof(cl_uint), frames.new_probs, 0, NULL, NULL);

	// start of encoding coefficients 
	clSetKernelArg(device.encode_coefficients, 9, sizeof(cl_int), &frames.current_is_key_frame);
	device.state_cpu = clSetKernelArg(device.encode_coefficients, 11, sizeof(cl_int), &frames.skip_prob);
	clEnqueueNDRangeKernel(device.boolcoder_commandQueue_cpu, device.encode_coefficients, 1, NULL, device.cpu_work_items_per_dim, device.cpu_work_group_size_per_dim, 0, NULL, NULL);
	clFlush(device.boolcoder_commandQueue_cpu); // we don't need result until gather_frame(), so no block now

	// encoding header is done as a part of HOST code placed in entropy_host.c[pp]|entropy_host.h
	encode_header(frames.encoded_frame); 

    return;
}

void prepare_segments_data()
{
	int i,qi;
	int *refqi;
	if (frames.current_is_key_frame)
	{
		frames.segments_data[0].y_dc_idelta = 15; //these ones are equal for all segments
		frames.segments_data[0].y2_dc_idelta = 0; // but i am lazy to create new buffer for them
		frames.segments_data[0].y2_ac_idelta = 0; // because it's also should be copied to GPU
		frames.segments_data[0].uv_dc_idelta = 0;
		frames.segments_data[0].uv_ac_idelta = 0;
	}
	else
	{
		frames.segments_data[0].y_dc_idelta = 15; 
		frames.segments_data[0].y2_dc_idelta = 0; 
		frames.segments_data[0].y2_ac_idelta = 0;
		frames.segments_data[0].uv_dc_idelta = -15;
		frames.segments_data[0].uv_ac_idelta = -15;
	}
	if (frames.current_is_altref_frame)
		refqi = video.altrefqi;
	else refqi = video.lastqi;
	for (i = 0; i < 4; ++i)
	{
		frames.segments_data[i].y_ac_i = (frames.current_is_key_frame) ? video.qi_min : refqi[i];
		frames.y_ac_q[i] = vp8_ac_qlookup[frames.segments_data[i].y_ac_i];
		qi = frames.segments_data[i].y_ac_i + frames.segments_data[0].y_dc_idelta;
		qi = (qi > 127) ? 127 : ((qi < 0) ? 0 : qi);
		frames.y_dc_q[i] = vp8_dc_qlookup[qi];
		qi = frames.segments_data[i].y_ac_i + frames.segments_data[0].y2_dc_idelta;
		qi = (qi > 127) ? 127 : ((qi < 0) ? 0 : qi);
		frames.y2_dc_q[i] = (vp8_dc_qlookup[qi]) << 1; // *2
		qi = frames.segments_data[i].y_ac_i + frames.segments_data[0].y2_ac_idelta;
		qi = (qi > 127) ? 127 : ((qi < 0) ? 0 : qi);
		frames.y2_ac_q[i] = 31 * (vp8_ac_qlookup[qi]) / 20; // *155/100
		qi = frames.segments_data[i].y_ac_i + frames.segments_data[0].uv_dc_idelta;
		qi = (qi > 127) ? 127 : ((qi < 0) ? 0 : qi);
		frames.uv_dc_q[i] = vp8_dc_qlookup[qi];
		qi = frames.segments_data[i].y_ac_i + frames.segments_data[0].uv_ac_idelta;
		qi = (qi > 127) ? 127 : ((qi < 0) ? 0 : qi);
		frames.uv_ac_q[i] = vp8_ac_qlookup[qi];

		if (frames.y2_ac_q[i] < 8)
			frames.y2_ac_q[i] = 8;
		if (frames.uv_dc_q[i] > 132)
			frames.uv_dc_q[i] = 132;
		
		frames.segments_data[i].loop_filter_level = frames.y_dc_q[i]/QUANT_TO_FILTER_LEVEL;
		frames.segments_data[i].loop_filter_level = (frames.segments_data[i].loop_filter_level > 63) ? 63 : frames.segments_data[i].loop_filter_level;
		frames.segments_data[i].loop_filter_level = (frames.segments_data[i].loop_filter_level < 0) ? 0 : frames.segments_data[i].loop_filter_level;


		frames.segments_data[i].interior_limit = frames.segments_data[i].loop_filter_level;
		if (video.loop_filter_sharpness) {
			frames.segments_data[i].interior_limit >>= video.loop_filter_sharpness > 4 ? 2 : 1;
			if (frames.segments_data[i].interior_limit > 9 - video.loop_filter_sharpness)
				frames.segments_data[i].interior_limit = 9 - video.loop_filter_sharpness;
		}
		if (!frames.segments_data[i].interior_limit)
			frames.segments_data[i].interior_limit = 1;

		frames.segments_data[i].mbedge_limit = ((frames.segments_data[i].loop_filter_level + 2) * 2) + frames.segments_data[i].interior_limit;
		frames.segments_data[i].sub_bedge_limit = (frames.segments_data[i].loop_filter_level * 2) + frames.segments_data[i].interior_limit;

		frames.segments_data[i].hev_threshold = 0;
		if (frames.current_is_key_frame) 
		{
			if (frames.segments_data[i].loop_filter_level >= 40)
				frames.segments_data[i].hev_threshold = 2;
			else if (frames.segments_data[i].loop_filter_level >= 15)
				frames.segments_data[i].hev_threshold = 1;
		}
		else /* current frame is an interframe */
		{
			if (frames.segments_data[i].loop_filter_level >= 40)
				frames.segments_data[i].hev_threshold = 3;
			else if (frames.segments_data[i].loop_filter_level >= 20)
				frames.segments_data[i].hev_threshold = 2;
			else if (frames.segments_data[i].loop_filter_level >= 15)
				frames.segments_data[i].hev_threshold = 1;
		}
	}
	if (video.GOP_size < 2) return;
	//always to gpu
	device.state_gpu = clEnqueueWriteBuffer(device.commandQueue1_gpu, device.segments_data_gpu, CL_TRUE, 0, sizeof(segment_data)*4, frames.segments_data, 0, NULL, NULL); 
	// and for loop filter on cpu
	if (!video.do_loop_filter_on_gpu)
		device.state_cpu = clEnqueueWriteBuffer(device.loopfilterY_commandQueue_cpu, device.segments_data_cpu, CL_TRUE, 0, sizeof(segment_data)*4, frames.segments_data, 0, NULL, NULL); 
	return;
}

void check_SSIM()
{
	device.state_gpu = clEnqueueReadBuffer(device.commandQueue1_gpu, device.reconstructed_frame_Y ,CL_TRUE, 0, video.wrk_frame_size_luma, frames.reconstructed_Y, 0, NULL, NULL);
	device.state_gpu = clEnqueueReadBuffer(device.commandQueue2_gpu, device.reconstructed_frame_U ,CL_TRUE, 0, video.wrk_frame_size_chroma, frames.reconstructed_U, 0, NULL, NULL);
	device.state_gpu = clEnqueueReadBuffer(device.commandQueue3_gpu, device.reconstructed_frame_V ,CL_TRUE, 0, video.wrk_frame_size_chroma, frames.reconstructed_V, 0, NULL, NULL);

	frames.new_SSIM = 0;
	float min1 = 2, min2 = 2;
	int mb_num;
	frames.replaced = 0;
	for (mb_num = 0; mb_num < video.mb_count; ++mb_num)
	{
		min2 = (frames.transformed_blocks[mb_num].SSIM < min2) ? frames.transformed_blocks[mb_num].SSIM : min2;
		if (frames.transformed_blocks[mb_num].SSIM < video.SSIM_target) 
			frames.e_data[mb_num].is_inter_mb = (test_inter_on_intra(mb_num, AQ_segment) == 0) ? 0 : frames.e_data[mb_num].is_inter_mb;
		if (frames.transformed_blocks[mb_num].SSIM < video.SSIM_target) 
			frames.e_data[mb_num].is_inter_mb = (test_inter_on_intra(mb_num, HQ_segment) == 0) ? 0 : frames.e_data[mb_num].is_inter_mb;
		if (frames.transformed_blocks[mb_num].SSIM < video.SSIM_target) 
			frames.e_data[mb_num].is_inter_mb = (test_inter_on_intra(mb_num, UQ_segment) == 0) ? 0 : frames.e_data[mb_num].is_inter_mb;
		frames.replaced+=(frames.e_data[mb_num].is_inter_mb==0);

		frames.new_SSIM += frames.transformed_blocks[mb_num].SSIM;
		min1 = (frames.transformed_blocks[mb_num].SSIM < min1) ? frames.transformed_blocks[mb_num].SSIM : min1;
	}
	
	frames.new_SSIM /= (float)video.mb_count;
	//printf("Fr %d(P)=> Avg SSIM=%f; MinSSIM=%f(%f); replaced:%d\n", frames.frame_number,frames.new_SSIM,min1,min2,frames.replaced);
	return;
}

int scene_change()
{
	static int holdover = 0;
	int detect;
	// something more sophisticated is desirable
	int Udiff = 0, Vdiff = 0, diff, pix;
	for (pix = 0; pix < video.wrk_frame_size_chroma; ++pix)
	{
		diff = (int)frames.last_U[pix] - (int)frames.current_U[pix];
		diff = (diff < 0) ? -diff : diff;
		Udiff += diff;
	}
	Udiff /= video.wrk_frame_size_chroma;
	for (pix = 0; pix < video.wrk_frame_size_chroma; ++pix)
	{
		diff = (int)frames.last_V[pix] - (int)frames.current_V[pix];
		diff = (diff < 0) ? -diff : diff;
		Vdiff += diff;
	}
	Vdiff /= video.wrk_frame_size_chroma;
	detect = ((Udiff > 7) || (Vdiff > 7) || (Udiff+Vdiff > 10));
	//workaround to exclude serial intra_frames
	// could shrink V
	if ((detect) && ((frames.frame_number - frames.last_key_detect) < 4))
	{
		frames.last_key_detect = frames.frame_number;
		holdover = 1;
		return 0;
	}
	if ((detect) && ((frames.frame_number - frames.last_key_detect) >= 4))
	{
		//frames.last_key_detect will be set in intra_transform()
		return 1;
	}
	// then detect == 0
	if ((holdover) && ((frames.frame_number - frames.last_key_detect) < 4))
	{
		return 0;
	}
	if ((holdover) && ((frames.frame_number - frames.last_key_detect) >= 4))
	{
		holdover = 0;
		return 1;
	}

	return 0; //no detection and no hold over from previous detections
}

void finalize();

int main(int argc, char *argv[])
{
	cl_int mb_num;
	printf("\n");

	error_file.path = ERRORPATH;
	if (ParseArgs(argc, argv) < 0) 
	{
		printf("\npress any key to quit\n");
		return -1;
	}
	
    OpenYUV420FileAndParseHeader();
    
	printf("initialization started;\n");
    init_all();
	if ((device.state_cpu != 0) || (device.state_gpu != 0)) {
		printf("\npress any key to quit\n");
		return -1; 
	}
	printf("initialization complete;\n");

    frames.frames_until_key = 1;
	frames.frames_until_altref = 2;
    frames.frame_number = 0;
	frames.golden_frame_number = -1;
	frames.altref_frame_number = -1;
	
	write_output_header();
	//open_dump_file();

    while (get_yuv420_frame() > 0)
    {
		frames.prev_is_key_frame = frames.current_is_key_frame; 
		frames.prev_is_golden_frame = frames.current_is_golden_frame;
		frames.prev_is_altref_frame = frames.current_is_altref_frame;
		--frames.frames_until_key;
		--frames.frames_until_altref;
		frames.current_is_key_frame = (frames.frames_until_key < 1);
		frames.current_is_golden_frame = frames.current_is_key_frame;
		frames.current_is_altref_frame = (frames.frames_until_altref < 1) || frames.current_is_key_frame;
		frames.frames_until_altref = ((frames.frames_until_altref < 1) || frames.current_is_key_frame) ? video.altref_range :frames.frames_until_altref;
		frames.golden_frame_number = (frames.current_is_golden_frame) ? frames.frame_number : frames.golden_frame_number;
		frames.altref_frame_number = (frames.current_is_altref_frame) ? frames.frame_number : frames.altref_frame_number;

        if (frames.current_is_key_frame)
        {
			prepare_segments_data();
			intra_transform();       
		}
        else if (scene_change()) 
		{
			frames.current_is_key_frame = 1;
			prepare_segments_data(); // redo because loop filtering differs
			intra_transform();
			printf("key frame FORCED by chroma color difference!\n");
		} 
		else
        {
			device.state_gpu = clEnqueueWriteBuffer(device.commandQueue1_gpu, device.current_frame_Y, CL_TRUE, 0, video.wrk_frame_size_luma, frames.current_Y, 0, NULL, NULL);
			device.state_gpu = clEnqueueWriteBuffer(device.commandQueue2_gpu, device.current_frame_U, CL_TRUE, 0, video.wrk_frame_size_chroma, frames.current_U, 0, NULL, NULL);
			device.state_gpu = clEnqueueWriteBuffer(device.commandQueue3_gpu, device.current_frame_V, CL_TRUE, 0, video.wrk_frame_size_chroma, frames.current_V, 0, NULL, NULL);
			
			prepare_segments_data();
			inter_transform(); 
			// copy transformed_blocks to host
			device.state_gpu = clEnqueueReadBuffer(device.commandQueue1_gpu, device.transformed_blocks_gpu ,CL_TRUE, 0, video.mb_count*sizeof(macroblock), frames.transformed_blocks, 0, NULL, NULL);

			for(mb_num = 0; mb_num < video.mb_count; ++mb_num) 
				frames.e_data[mb_num].is_inter_mb = 1;

			check_SSIM();
			if ((frames.replaced > (video.mb_count/4)) || (frames.new_SSIM < video.SSIM_target))
			{
				// redo as intra
				frames.current_is_key_frame = 1;
				prepare_segments_data();
				intra_transform();
				printf("key frame FORCED by bad inter-result: replaced(%d) and SSIM(%f)!\n",frames.replaced,frames.new_SSIM);
			}

        }
		// searching for MBs to be skiped 
		prepare_filter_mask_and_non_zero_coeffs();
		do_loop_filter();
		//dump();

		if (video.do_loop_filter_on_gpu)
			device.state_cpu = clEnqueueWriteBuffer(device.boolcoder_commandQueue_cpu, device.transformed_blocks_cpu, CL_TRUE, 0, video.mb_count*sizeof(macroblock), frames.transformed_blocks, 0, NULL, NULL); 
		// else already there because were uploaded before loop filter
		entropy_encode();

		//if ((frames.frame_number % video.framerate) == 0) printf("second %d encoded\n", frames.frame_number/video.framerate);
		gather_frame();
		write_output_file();
        ++frames.frame_number;
    }
	write_output_header();
	//fclose(dump_file.handle);
	finalize();

	//getch();
	return 777;
}

void finalize()
{
	fclose(input_file.handle);
	fclose(output_file.handle);

	clReleaseMemObject(device.coeff_probs);
	clReleaseMemObject(device.coeff_probs_denom);
	clReleaseMemObject(device.partitions);
	clReleaseMemObject(device.partitions_sizes);
	clReleaseMemObject(device.third_context);
	clReleaseMemObject(device.transformed_blocks_cpu);
	if (video.GOP_size > 1) {
		clReleaseMemObject(device.current_frame_U);
		clReleaseMemObject(device.current_frame_V);
		clReleaseMemObject(device.current_frame_Y);
		clReleaseMemObject(device.current_frame_Y_downsampled_by2);
		clReleaseMemObject(device.current_frame_Y_downsampled_by4);
		clReleaseMemObject(device.current_frame_Y_downsampled_by8);
		clReleaseMemObject(device.current_frame_Y_downsampled_by16);
		clReleaseMemObject(device.reconstructed_frame_U);
		clReleaseMemObject(device.reconstructed_frame_V);
		clReleaseMemObject(device.reconstructed_frame_Y);
		clReleaseMemObject(device.last_frame_Y_image);
		clReleaseMemObject(device.last_frame_U_image);
		clReleaseMemObject(device.last_frame_V_image);
		clReleaseMemObject(device.last_frame_Y_downsampled_by2);
		clReleaseMemObject(device.last_frame_Y_downsampled_by4);
		clReleaseMemObject(device.last_frame_Y_downsampled_by8);
		clReleaseMemObject(device.last_frame_Y_downsampled_by16);
		clReleaseMemObject(device.golden_frame_Y);
		clReleaseMemObject(device.golden_frame_Y_image);
		clReleaseMemObject(device.golden_frame_U_image);
		clReleaseMemObject(device.golden_frame_V_image);
		clReleaseMemObject(device.golden_frame_Y_downsampled_by2);
		clReleaseMemObject(device.golden_frame_Y_downsampled_by4);
		clReleaseMemObject(device.golden_frame_Y_downsampled_by8);
		clReleaseMemObject(device.golden_frame_Y_downsampled_by16);
		clReleaseMemObject(device.altref_frame_Y);
		clReleaseMemObject(device.altref_frame_Y_image);
		clReleaseMemObject(device.altref_frame_U_image);
		clReleaseMemObject(device.altref_frame_V_image);
		clReleaseMemObject(device.altref_frame_Y_downsampled_by2);
		clReleaseMemObject(device.altref_frame_Y_downsampled_by4);
		clReleaseMemObject(device.altref_frame_Y_downsampled_by8);
		clReleaseMemObject(device.altref_frame_Y_downsampled_by16);
		clReleaseMemObject(device.predictors_Y);
		clReleaseMemObject(device.predictors_U);
		clReleaseMemObject(device.predictors_V);
		clReleaseMemObject(device.residual_Y);
		clReleaseMemObject(device.residual_U);
		clReleaseMemObject(device.residual_V);
		clReleaseMemObject(device.cpu_frame_Y);
		clReleaseMemObject(device.cpu_frame_U);
		clReleaseMemObject(device.cpu_frame_V);
		clReleaseMemObject(device.transformed_blocks_gpu);
		clReleaseMemObject(device.segments_data_gpu);
		clReleaseMemObject(device.segments_data_cpu);
		clReleaseMemObject(device.last_vnet1);
		clReleaseMemObject(device.last_vnet2);
		clReleaseMemObject(device.golden_vnet1);
		clReleaseMemObject(device.golden_vnet2);
		clReleaseMemObject(device.altref_vnet1);
		clReleaseMemObject(device.altref_vnet2);
		clReleaseMemObject(device.mb_mask);
		clReleaseMemObject(device.metrics1);
		clReleaseMemObject(device.metrics2);
		clReleaseMemObject(device.metrics3);
		clReleaseKernel(device.reset_vectors);
		clReleaseKernel(device.downsample);
		clReleaseKernel(device.luma_search_1step);
		clReleaseKernel(device.luma_search_2step);
		clReleaseKernel(device.select_reference);
		clReleaseKernel(device.prepare_predictors_and_residual);
		clReleaseKernel(device.pack_8x8_into_16x16);
		clReleaseKernel(device.dct4x4);
		clReleaseKernel(device.wht4x4_iwht4x4);
		clReleaseKernel(device.idct4x4);
		clReleaseKernel(device.count_SSIM_luma);
		clReleaseKernel(device.count_SSIM_chroma);
		clReleaseKernel(device.prepare_filter_mask);
		clReleaseKernel(device.normal_loop_filter_MBH);
		clReleaseKernel(device.normal_loop_filter_MBV);
		clReleaseKernel(device.loop_filter_frame);
		clReleaseCommandQueue(device.commandQueue1_gpu);
		clReleaseCommandQueue(device.commandQueue2_gpu);
		clReleaseCommandQueue(device.commandQueue3_gpu);
		clReleaseProgram(device.program_gpu);
		clReleaseContext(device.context_gpu);
		free(device.device_gpu);
	}

	clReleaseKernel(device.count_probs);
	clReleaseKernel(device.encode_coefficients);
	clReleaseKernel(device.num_div_denom);
	clReleaseCommandQueue(device.loopfilterY_commandQueue_cpu);
	clReleaseCommandQueue(device.loopfilterU_commandQueue_cpu);
	clReleaseCommandQueue(device.loopfilterV_commandQueue_cpu);
	clReleaseCommandQueue(device.boolcoder_commandQueue_cpu);
	clReleaseProgram(device.program_cpu);
	clReleaseContext(device.context_cpu);

	free(frames.input_pack);
	free(frames.reconstructed_Y);
    free(frames.reconstructed_U);
	free(frames.reconstructed_V);
	free(frames.last_U);
	free(frames.last_V);
	free(frames.transformed_blocks);
	free(frames.e_data);
	free(frames.encoded_frame);
	free(frames.partition_0);
	free(frames.partitions);

	if (((video.src_height != video.dst_height) || (video.src_width != video.dst_width)) ||
        ((video.wrk_height != video.dst_height) || (video.wrk_width != video.dst_width)))
    {
		free(frames.current_Y);
		free(frames.current_U);
		free(frames.current_V);
    }

	free(device.platforms);
	free(device.device_cpu);

	return;
}
