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
struct encoderStatistics encStat;

static cl_int ifFlush(cl_command_queue comm)
{
#ifdef ALLWAYS_FLUSH
	const cl_int ret = clFlush(comm);
	//const cl_int ret = clFinish(comm);
	if (ret < 0)
		printf("flush fail\n");
	return ret;
#endif //ALLWAYS_FLUSH
	return 0;
}

static cl_int finalFlush(cl_command_queue comm)
{
#ifndef ALLWAYS_FLUSH
	const cl_int ret = clFlush(comm);
	if (ret < 0)
		printf("flush fail\n");
	return ret;
#endif //ALLWAYS_FLUSH
	return 0;
}

#include "encIO.h"
#include "init.h"
#include "intra_part.h"
#include "inter_part.h"
#include "loop_filter.h"
#include "debug.h"

////////////////// transforms are taken from multimedia mike's encoder version

extern void encode_header(cl_uchar *const partition);

static void entropy_encode()
{
	if (frames.threads_free < video.number_of_partitions) {
		device.state_gpu = finalFlush(device.loopfilterU_commandQueue_cpu);
		device.state_gpu = finalFlush(device.loopfilterV_commandQueue_cpu);
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
	clEnqueueNDRangeKernel(device.boolcoder_commandQueue_cpu, device.count_probs, 1, NULL, device.cpu_work_items_per_dim, device.cpu_work_group_size_per_dim, 0, NULL, NULL);
	frames.threads_free -= video.number_of_partitions;

	// just dividing nums by denoms and getting probability of bit being ZERO
	clEnqueueNDRangeKernel(device.boolcoder_commandQueue_cpu, device.num_div_denom, 1, NULL, device.cpu_work_items_per_dim, device.cpu_work_group_size_per_dim, 0, NULL, NULL);

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
	device.state_gpu = clEnqueueWriteBuffer(device.boolcoder_commandQueue_cpu, device.coeff_probs, CL_FALSE, 0, 11*3*8*4*sizeof(cl_uint), frames.new_probs, 0, NULL, NULL);

	// start of encoding coefficients 
	clEnqueueNDRangeKernel(device.boolcoder_commandQueue_cpu, device.encode_coefficients, 1, NULL, device.cpu_work_items_per_dim, device.cpu_work_group_size_per_dim, 0, NULL, NULL);
	ifFlush(device.boolcoder_commandQueue_cpu); // we don't need result until gather_frame(), so no block now

	// encoding header is done as a part of HOST code placed in entropy_host.c[pp]|entropy_host.h
	encode_header(frames.encoded_frame); 

    return;
}

static void get_loopfilter_strength(int *const __restrict red, cl_int *const __restrict sh)
{
	int i,j, avg = 0, div = 0;
	for(i = 0; i < video.wrk_frame_size_luma; ++i)
		avg += frames.current_Y[i];
	avg += video.wrk_frame_size_luma/2;
	avg /= video.wrk_frame_size_luma;
	*red = (avg*5/255) + 3;

	for(i = 1; i < video.wrk_height - 1; ++i)
		for(j = 1; j < video.wrk_width - 1; ++j)
		{
			const int p = i*video.wrk_width + j;
			avg = frames.current_Y[p - video.wrk_width - 1] +
				frames.current_Y[p - video.wrk_width] +
				frames.current_Y[p - video.wrk_width + 1] +
				frames.current_Y[p - 1] +
				frames.current_Y[p + 1] +
				frames.current_Y[p + video.wrk_width - 1] +
				frames.current_Y[p + video.wrk_width] +
				frames.current_Y[p + video.wrk_width + 1];
			avg /= 8;
			div += (frames.current_Y[p] - avg)*(frames.current_Y[p] - avg);
		}
	div += (video.wrk_height - 1)*(video.wrk_width - 1)/2;
	div /= (video.wrk_height - 1)*(video.wrk_width - 1);

	*sh = div/8;
	*sh = (*sh > 7) ? 7 : *sh;

	return;
}

static void prepare_segments_data(const int update_filter = 0, const int shrpnss = 0)
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

	int reductor;
	get_loopfilter_strength(&reductor, &video.loop_filter_sharpness);
	if (update_filter)
	{
		reductor *= 2;
		video.loop_filter_sharpness = shrpnss;
	}

	
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
		
		frames.segments_data[i].loop_filter_level = frames.y_dc_q[i]/reductor;
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
	device.state_gpu = clEnqueueWriteBuffer(device.commandQueue1_gpu, device.segments_data_gpu, CL_FALSE, 0, sizeof(segment_data)*4, frames.segments_data, 0, NULL, NULL); 
	// and for loop filter on cpu
	if (!video.do_loop_filter_on_gpu)
		device.state_cpu = clEnqueueWriteBuffer(device.loopfilterY_commandQueue_cpu, device.segments_data_cpu, CL_FALSE, 0, sizeof(segment_data)*4, frames.segments_data, 0, NULL, NULL); 
	return;
}

static void check_SSIM()
{
	//device.state_gpu = clEnqueueReadBuffer(device.commandQueue1_gpu, device.reconstructed_frame_Y ,CL_TRUE, 0, video.wrk_frame_size_luma, frames.reconstructed_Y, 0, NULL, NULL);
	//device.state_gpu = clEnqueueReadBuffer(device.commandQueue2_gpu, device.reconstructed_frame_U ,CL_TRUE, 0, video.wrk_frame_size_chroma, frames.reconstructed_U, 0, NULL, NULL);
	//device.state_gpu = clEnqueueReadBuffer(device.commandQueue3_gpu, device.reconstructed_frame_V ,CL_TRUE, 0, video.wrk_frame_size_chroma, frames.reconstructed_V, 0, NULL, NULL);

	frames.new_SSIM = 0;
	float min1 = 2.0f, min2 = 2.0f;
	int mb_num;
	frames.replaced = 0;

	for (mb_num = 0; mb_num < video.mb_count; ++mb_num)
	{
		min2 = (frames.MB_SSIM[mb_num] < min2) ? frames.MB_SSIM[mb_num] : min2;
		if (frames.MB_SSIM[mb_num] < video.SSIM_target) 
			frames.e_data[mb_num].is_inter_mb = (test_inter_on_intra(mb_num, AQ_segment) == 0) ? 0 : frames.e_data[mb_num].is_inter_mb;
		if (frames.MB_SSIM[mb_num] < video.SSIM_target) 
			frames.e_data[mb_num].is_inter_mb = (test_inter_on_intra(mb_num, HQ_segment) == 0) ? 0 : frames.e_data[mb_num].is_inter_mb;
		if (frames.MB_SSIM[mb_num] < video.SSIM_target) 
			frames.e_data[mb_num].is_inter_mb = (test_inter_on_intra(mb_num, UQ_segment) == 0) ? 0 : frames.e_data[mb_num].is_inter_mb;
		frames.replaced+=(frames.e_data[mb_num].is_inter_mb==0);

		frames.new_SSIM += frames.MB_SSIM[mb_num];
		min1 = (frames.MB_SSIM[mb_num] < min1) ? frames.MB_SSIM[mb_num] : min1;
	}
	
	frames.new_SSIM /= (float)video.mb_count;
	if (video.print_info) 
		printf("%d>AvgSSIM=%f; MinSSIM=%f(%f); repl:%d ", frames.frame_number,frames.new_SSIM,min1,min2,frames.replaced);
	if (min1 > 0.95) 
		prepare_segments_data(1, 7);
	return;
}

static int scene_change()
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

static void finalize();

int main(int argc, char *argv[])
{
	cl_int mb_num;
	printf("\n");

	error_file.path = ERRORPATH;
	if (ParseArgs(argc, argv) < 0) 
	{
		return -1;
	}
	
    OpenYUV420FileAndParseHeader();
    
	printf("initialization started;\n");
    if ((init_all() < 0) || (device.state_cpu != 0) || (device.state_gpu != 0)) 
	{
		return -1; 
	}
	printf("initialization complete;\n");

	encStat.scene_changes_by_color = 0;
	encStat.scene_changes_by_ssim = 0;
	encStat.scene_changes_by_replaced = 0;
	encStat.scene_changes_by_bitrate = 0;

    frames.frames_until_key = 1;
	frames.frames_until_altref = 2;
    frames.frame_number = 0;
	frames.golden_frame_number = -1;
	frames.altref_frame_number = -1;
	
	write_output_header();
	//open_dump_file();

	frames.video_size = 0;
	frames.encoded_frame_size = 0;
    while (get_yuv420_frame() > 0)
    {
		//grab buffer for host (without copy, we will write it from scratch)
		clFinish(device.boolcoder_commandQueue_cpu);
		frames.MB = (macroblock_coeffs_t*)clEnqueueMapBuffer(device.loopfilterY_commandQueue_cpu, device.macroblock_coeffs_cpu, CL_TRUE, CL_MAP_WRITE_INVALIDATE_REGION, 0, sizeof(macroblock_coeffs_t)*video.mb_count, 0, NULL, NULL, &device.state_cpu);
		if (!video.do_loop_filter_on_gpu)
		{
			// we can use invalidate for key frames, but read/write will have the same speed (zero-copy) and so we can start memory transfer to GPU a little bit earlier (in case of inter frame)
			frames.reconstructed_Y = (unsigned char*)clEnqueueMapBuffer(device.loopfilterY_commandQueue_cpu, device.cpu_frame_Y, CL_TRUE, CL_MAP_READ|CL_MAP_WRITE, 0, sizeof(unsigned char)*video.wrk_frame_size_luma, 0, NULL, NULL, &device.state_cpu);
			frames.reconstructed_U = (unsigned char*)clEnqueueMapBuffer(device.loopfilterU_commandQueue_cpu, device.cpu_frame_U, CL_TRUE, CL_MAP_READ|CL_MAP_WRITE, 0, sizeof(unsigned char)*video.wrk_frame_size_chroma, 0, NULL, NULL, &device.state_cpu);
			frames.reconstructed_V = (unsigned char*)clEnqueueMapBuffer(device.loopfilterV_commandQueue_cpu, device.cpu_frame_V, CL_TRUE, CL_MAP_READ|CL_MAP_WRITE, 0, sizeof(unsigned char)*video.wrk_frame_size_chroma, 0, NULL, NULL, &device.state_cpu);
		}
				
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

		frames.prev_frame_size = frames.encoded_frame_size;
		frames.video_size += frames.encoded_frame_size;

        if (frames.current_is_key_frame)
        {
			prepare_segments_data();
			intra_transform();       
		}
        else 
		{
			device.state_gpu = clEnqueueWriteBuffer(device.commandQueue2_gpu, device.current_frame_Y, CL_FALSE, 0, video.wrk_frame_size_luma, frames.current_Y, 0, NULL, NULL);
			device.state_gpu = clEnqueueWriteBuffer(device.commandQueue3_gpu, device.current_frame_U, CL_FALSE, 0, video.wrk_frame_size_chroma, frames.current_U, 0, NULL, NULL);
			device.state_gpu = clEnqueueWriteBuffer(device.commandQueue3_gpu, device.current_frame_V, CL_FALSE, 0, video.wrk_frame_size_chroma, frames.current_V, 0, NULL, NULL);
			if (!video.do_loop_filter_on_gpu)
			{
				const size_t origin[3] = {0, 0, 0};
				const size_t region_y[3] = {video.wrk_width, video.wrk_height, 1};
				const size_t region_uv[3] = {video.wrk_width/2, video.wrk_height/2, 1};

				device.state_gpu = clEnqueueWriteBuffer(device.commandQueue1_gpu, device.reconstructed_frame_Y, CL_FALSE, 0, video.wrk_frame_size_luma, frames.reconstructed_Y, 0, NULL, NULL);
				device.state_gpu = clEnqueueWriteBuffer(device.commandQueue3_gpu, device.reconstructed_frame_U, CL_FALSE, 0, video.wrk_frame_size_chroma, frames.reconstructed_U, 0, NULL, NULL);
				device.state_gpu = clEnqueueWriteBuffer(device.commandQueue3_gpu, device.reconstructed_frame_V, CL_FALSE, 0, video.wrk_frame_size_chroma, frames.reconstructed_V, 0, NULL, NULL);

				device.state_gpu = clEnqueueWriteImage(device.commandQueue3_gpu, device.last_frame_Y_image, CL_FALSE, origin, region_y, 0, 0, frames.reconstructed_Y, 0, NULL, NULL);
				device.state_gpu = clEnqueueWriteImage(device.commandQueue3_gpu, device.last_frame_U_image, CL_FALSE, origin, region_uv, 0, 0, frames.reconstructed_U, 0, NULL, NULL);
				device.state_gpu = clEnqueueWriteImage(device.commandQueue3_gpu, device.last_frame_V_image, CL_FALSE, origin, region_uv, 0, 0, frames.reconstructed_V, 0, NULL, NULL);

				clFlush(device.commandQueue2_gpu);
			}
			clFlush(device.commandQueue1_gpu);
			clFlush(device.commandQueue3_gpu);

			const int new_scene = scene_change();
			if (new_scene) 
			{
				++encStat.scene_changes_by_color;
				frames.current_is_key_frame = 1;
				prepare_segments_data(); // redo because loop filtering differs
				intra_transform();
				printf("key frame FORCED by chroma color difference!\n");
			} 
			else
			{
				prepare_segments_data();
				inter_transform(); 
				// copy transformed_blocks to host
				device.state_gpu = clEnqueueReadBuffer(device.commandQueue1_gpu, device.macroblock_coeffs_gpu, CL_FALSE, 0, video.mb_count*sizeof(macroblock_coeffs_t), frames.MB, 0, NULL, NULL);
				device.state_gpu = clEnqueueReadBuffer(device.commandQueue1_gpu, device.macroblock_segment_id_gpu, CL_FALSE, 0, video.mb_count*sizeof(cl_int), frames.MB_segment_id, 0, NULL, NULL);
				device.state_gpu = clEnqueueReadBuffer(device.commandQueue1_gpu, device.macroblock_SSIM_gpu, CL_FALSE, 0, video.mb_count*sizeof(cl_float), frames.MB_SSIM, 0, NULL, NULL);

				// these buffers are copied inside inter_transform() as soon as they are ready
				//device.state_gpu = clEnqueueReadBuffer(device.commandQueue1_gpu, device.macroblock_parts_gpu, CL_FALSE, 0, video.mb_count*sizeof(cl_int), frames.MB_parts, 0, NULL, NULL);
				//device.state_gpu = clEnqueueReadBuffer(device.commandQueue1_gpu, device.macroblock_reference_frame_gpu, CL_FALSE, 0, video.mb_count*sizeof(cl_int), frames.MB_reference_frame, 0, NULL, NULL);
				//device.state_gpu = clEnqueueReadBuffer(device.commandQueue1_gpu, device.macroblock_vectors_gpu, CL_FALSE, 0, video.mb_count*sizeof(macroblock_vectors_t), frames.MB_vectors, 0, NULL, NULL);

				device.state_gpu = clEnqueueReadBuffer(device.commandQueue1_gpu, device.reconstructed_frame_Y ,CL_FALSE, 0, video.wrk_frame_size_luma, frames.reconstructed_Y, 0, NULL, NULL);
				device.state_gpu = clEnqueueReadBuffer(device.commandQueue1_gpu, device.reconstructed_frame_U ,CL_FALSE, 0, video.wrk_frame_size_chroma, frames.reconstructed_U, 0, NULL, NULL);
				device.state_gpu = clEnqueueReadBuffer(device.commandQueue1_gpu, device.reconstructed_frame_V ,CL_FALSE, 0, video.wrk_frame_size_chroma, frames.reconstructed_V, 0, NULL, NULL);
				clFlush(device.commandQueue1_gpu);

				for(mb_num = 0; mb_num < video.mb_count; ++mb_num) 
					frames.e_data[mb_num].is_inter_mb = 1;

				clFinish(device.dataCopy_gpu);
				clFinish(device.commandQueue1_gpu);

				check_SSIM();
				if ((frames.replaced > (video.mb_count/6)) || (frames.new_SSIM < video.SSIM_target))
				{
					if (frames.new_SSIM < video.SSIM_target) ++encStat.scene_changes_by_ssim;
					else ++encStat.scene_changes_by_replaced;
					// redo as intra
					frames.current_is_key_frame = 1;
					prepare_segments_data();
					intra_transform();
					if (video.print_info) 
						printf("\nkey frame FORCED by bad inter-result: replaced(%d) and SSIM(%f)!\n",frames.replaced,frames.new_SSIM);
				}

			}
		}
		// searching for MBs to be skiped 
		// copy (unmap) coefficients back to pinned cpu device memory
		// we will need it for loop filter(if on cpu) and for entropy encoding (always)
		clEnqueueUnmapMemObject(device.loopfilterY_commandQueue_cpu, device.macroblock_coeffs_cpu, frames.MB, 0, NULL, NULL);
		if (!video.do_loop_filter_on_gpu)
		{
			clEnqueueUnmapMemObject(device.loopfilterY_commandQueue_cpu, device.cpu_frame_Y, frames.reconstructed_Y, 0, NULL, NULL);
			clEnqueueUnmapMemObject(device.loopfilterY_commandQueue_cpu, device.cpu_frame_U, frames.reconstructed_U, 0, NULL, NULL);
			clEnqueueUnmapMemObject(device.loopfilterY_commandQueue_cpu, device.cpu_frame_V, frames.reconstructed_V, 0, NULL, NULL);
		}
		// we need not to block here, even if  two queues are using this object (prepare process will always wait for results)
		device.state_gpu = clEnqueueWriteBuffer(device.loopfilterY_commandQueue_cpu, device.macroblock_parts_cpu, CL_FALSE, 0, video.mb_count*sizeof(cl_int), frames.MB_parts, 0, NULL, NULL);
		device.state_gpu = clEnqueueWriteBuffer(device.loopfilterY_commandQueue_cpu, device.macroblock_segment_id_cpu, CL_FALSE, 0, video.mb_count*sizeof(cl_int), frames.MB_segment_id, 0, NULL, NULL);
		clFinish(device.loopfilterY_commandQueue_cpu);

		prepare_filter_mask_and_non_zero_coeffs();
		do_loop_filter();

//TODO		if (video.do_loop_filter_on_gpu)
//TODO			device.state_cpu = clEnqueueWriteBuffer(device.boolcoder_commandQueue_cpu, device.transformed_blocks_cpu, CL_FALSE, 0, video.mb_count*sizeof(macroblock), frames.transformed_blocks, 0, NULL, NULL); 
		// else already there because were uploaded before loop filter
		entropy_encode();
		
		//if ((frames.frame_number % video.framerate) == 0) printf("second %d encoded\n", frames.frame_number/video.framerate);
		gather_frame();
		if (video.print_info) 
			printf("br=%dk, frame~%dk\n", (int)(frames.video_size*video.framerate*8/(frames.frame_number+1)/1024), (frames.encoded_frame_size+512)/1024);

		//dump();
		write_output_file();
        ++frames.frame_number;
    }
	write_output_header();
	//fclose(dump_file.handle);
	finalize();

	printf("%d scene changes detected by color change\n", encStat.scene_changes_by_color);
	printf("%d scene changes detected by low ssim value\n", encStat.scene_changes_by_ssim);
	printf("%d scene changes detected by high amount of replaced blocks\n", encStat.scene_changes_by_replaced);
	printf("%d scene changes detected by bitrate raise\n", encStat.scene_changes_by_bitrate);
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
	clReleaseMemObject(device.macroblock_coeffs_cpu);
	clReleaseMemObject(device.macroblock_non_zero_coeffs_cpu);
	clReleaseMemObject(device.macroblock_parts_cpu);
	clReleaseMemObject(device.macroblock_segment_id_cpu);
	clReleaseMemObject(device.segments_data_cpu);
	if (video.GOP_size > 1) 
	{
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
		clReleaseMemObject(device.macroblock_coeffs_gpu);
		clReleaseMemObject(device.macroblock_non_zero_coeffs_gpu);
		clReleaseMemObject(device.macroblock_parts_gpu);
		clReleaseMemObject(device.macroblock_reference_frame_gpu);
		clReleaseMemObject(device.macroblock_segment_id_gpu);
		clReleaseMemObject(device.macroblock_SSIM_gpu);
		clReleaseMemObject(device.macroblock_vectors_gpu);
		clReleaseMemObject(device.segments_data_gpu);
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
		clReleaseKernel(device.downsample_last_1x_to_2x);
		clReleaseKernel(device.downsample_last_2x_to_4x);
		clReleaseKernel(device.downsample_last_4x_to_8x);
		clReleaseKernel(device.downsample_last_8x_to_16x);
		clReleaseKernel(device.downsample_current_1x_to_2x);
		clReleaseKernel(device.downsample_current_2x_to_4x);
		clReleaseKernel(device.downsample_current_4x_to_8x);
		clReleaseKernel(device.downsample_current_8x_to_16x);
		clReleaseKernel(device.luma_search_last_16x);
		clReleaseKernel(device.luma_search_golden_16x);
		clReleaseKernel(device.luma_search_altref_16x);
		clReleaseKernel(device.luma_search_last_8x);
		clReleaseKernel(device.luma_search_golden_8x);
		clReleaseKernel(device.luma_search_altref_8x);
		clReleaseKernel(device.luma_search_last_4x);
		clReleaseKernel(device.luma_search_golden_4x);
		clReleaseKernel(device.luma_search_altref_4x);
		clReleaseKernel(device.luma_search_last_2x);
		clReleaseKernel(device.luma_search_golden_2x);
		clReleaseKernel(device.luma_search_altref_2x);
		clReleaseKernel(device.luma_search_last_1x);
		clReleaseKernel(device.luma_search_golden_1x);
		clReleaseKernel(device.luma_search_altref_1x);
		clReleaseKernel(device.luma_search_last_d4x);
		clReleaseKernel(device.luma_search_golden_d4x);
		clReleaseKernel(device.luma_search_altref_d4x);
		clReleaseKernel(device.select_reference);
		clReleaseKernel(device.prepare_predictors_and_residual_last_Y);
		clReleaseKernel(device.prepare_predictors_and_residual_last_U);
		clReleaseKernel(device.prepare_predictors_and_residual_last_V);
		clReleaseKernel(device.prepare_predictors_and_residual_golden_Y);
		clReleaseKernel(device.prepare_predictors_and_residual_golden_U);
		clReleaseKernel(device.prepare_predictors_and_residual_golden_V);
		clReleaseKernel(device.prepare_predictors_and_residual_altref_Y);
		clReleaseKernel(device.prepare_predictors_and_residual_altref_U);
		clReleaseKernel(device.prepare_predictors_and_residual_altref_V);
		clReleaseKernel(device.pack_8x8_into_16x16);
		clReleaseKernel(device.dct4x4_Y[UQ_segment]);
		clReleaseKernel(device.dct4x4_U[UQ_segment]);
		clReleaseKernel(device.dct4x4_V[UQ_segment]);
		clReleaseKernel(device.dct4x4_Y[HQ_segment]);
		clReleaseKernel(device.dct4x4_U[HQ_segment]);
		clReleaseKernel(device.dct4x4_V[HQ_segment]);
		clReleaseKernel(device.dct4x4_Y[AQ_segment]);
		clReleaseKernel(device.dct4x4_U[AQ_segment]);
		clReleaseKernel(device.dct4x4_V[AQ_segment]);
		clReleaseKernel(device.dct4x4_Y[LQ_segment]);
		clReleaseKernel(device.dct4x4_U[LQ_segment]);
		clReleaseKernel(device.dct4x4_V[LQ_segment]);
		clReleaseKernel(device.wht4x4_iwht4x4[UQ_segment]);
		clReleaseKernel(device.wht4x4_iwht4x4[HQ_segment]);
		clReleaseKernel(device.wht4x4_iwht4x4[AQ_segment]);
		clReleaseKernel(device.wht4x4_iwht4x4[LQ_segment]);
		clReleaseKernel(device.idct4x4_Y[UQ_segment]);
		clReleaseKernel(device.idct4x4_U[UQ_segment]);
		clReleaseKernel(device.idct4x4_V[UQ_segment]);
		clReleaseKernel(device.idct4x4_Y[HQ_segment]);
		clReleaseKernel(device.idct4x4_U[HQ_segment]);
		clReleaseKernel(device.idct4x4_V[HQ_segment]);
		clReleaseKernel(device.idct4x4_Y[AQ_segment]);
		clReleaseKernel(device.idct4x4_U[AQ_segment]);
		clReleaseKernel(device.idct4x4_V[AQ_segment]);
		clReleaseKernel(device.idct4x4_Y[LQ_segment]);
		clReleaseKernel(device.idct4x4_U[LQ_segment]);
		clReleaseKernel(device.idct4x4_V[LQ_segment]);
		clReleaseKernel(device.count_SSIM_luma[UQ_segment]);
		clReleaseKernel(device.count_SSIM_luma[HQ_segment]);
		clReleaseKernel(device.count_SSIM_luma[AQ_segment]);
		clReleaseKernel(device.count_SSIM_luma[LQ_segment]);
		clReleaseKernel(device.count_SSIM_chroma_U[UQ_segment]);
		clReleaseKernel(device.count_SSIM_chroma_U[HQ_segment]);
		clReleaseKernel(device.count_SSIM_chroma_U[AQ_segment]);
		clReleaseKernel(device.count_SSIM_chroma_U[LQ_segment]);
		clReleaseKernel(device.count_SSIM_chroma_V[UQ_segment]);
		clReleaseKernel(device.count_SSIM_chroma_V[HQ_segment]);
		clReleaseKernel(device.count_SSIM_chroma_V[AQ_segment]);
		clReleaseKernel(device.count_SSIM_chroma_V[LQ_segment]);
		clReleaseKernel(device.gather_SSIM);
		clReleaseKernel(device.prepare_filter_mask);
		clReleaseKernel(device.normal_loop_filter_MBH);
		clReleaseKernel(device.normal_loop_filter_MBV);
		clReleaseKernel(device.loop_filter_frame_luma);
		clReleaseKernel(device.loop_filter_frame_chroma_U);
		clReleaseKernel(device.loop_filter_frame_chroma_V);
		clReleaseCommandQueue(device.commandQueue1_gpu);
		clReleaseCommandQueue(device.commandQueue2_gpu);
		clReleaseCommandQueue(device.commandQueue3_gpu);
		clReleaseCommandQueue(device.dataCopy_gpu);
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
	free(frames.last_U);
	free(frames.last_V);
	free(frames.MB_parts);
	free(frames.MB_non_zero_coeffs);
	free(frames.MB_reference_frame);
	free(frames.MB_segment_id);
	free(frames.MB_SSIM);
	free(frames.MB_vectors);
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