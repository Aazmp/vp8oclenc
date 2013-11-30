void open_dump_file()
{
	dump_file.path = DUMPPATH;
	dump_file.handle = fopen(dump_file.path, "wb");
	fwrite(frames.header, frames.header_sz, 1, dump_file.handle);
}

void dump()
{
	if (frames.frame_number > 1500) return; //disk space guard

	if (video.do_loop_filter_on_gpu)
	{
		device.state_gpu = clEnqueueReadBuffer(device.commandQueue1_gpu, device.reconstructed_frame_Y ,CL_TRUE, 0, video.wrk_frame_size_luma, frames.reconstructed_Y, 0, NULL, NULL);
		device.state_gpu = clEnqueueReadBuffer(device.commandQueue2_gpu, device.reconstructed_frame_U ,CL_TRUE, 0, video.wrk_frame_size_chroma, frames.reconstructed_U, 0, NULL, NULL);
		device.state_gpu = clEnqueueReadBuffer(device.commandQueue3_gpu, device.reconstructed_frame_V ,CL_TRUE, 0, video.wrk_frame_size_chroma, frames.reconstructed_V, 0, NULL, NULL);
	}
	else
	{
		device.state_cpu = clEnqueueReadBuffer(device.loopfilterY_commandQueue_cpu, device.cpu_frame_Y ,CL_TRUE, 0, video.wrk_frame_size_luma, frames.reconstructed_Y, 0, NULL, NULL);
		device.state_cpu = clEnqueueReadBuffer(device.loopfilterU_commandQueue_cpu, device.cpu_frame_U ,CL_TRUE, 0, video.wrk_frame_size_chroma, frames.reconstructed_U, 0, NULL, NULL);
		device.state_cpu = clEnqueueReadBuffer(device.loopfilterV_commandQueue_cpu, device.cpu_frame_V ,CL_TRUE, 0, video.wrk_frame_size_chroma, frames.reconstructed_V, 0, NULL, NULL);
	}
	char delimiter[] = "FRAME+";
	delimiter[5] = 0x0A;
	fwrite(delimiter, 6, 1, dump_file.handle);
	
	int i;
	for (i = 0; i < video.src_height; ++i)
		fwrite(&frames.reconstructed_Y[i*video.src_width], video.src_width, 1, dump_file.handle);
	for (i = 0; i < video.src_height/2; ++i)
		fwrite(&frames.reconstructed_U[i*video.src_width/2], video.src_width/2, 1, dump_file.handle);
	for (i = 0; i < video.src_height/2; ++i)
		fwrite(&frames.reconstructed_V[i*video.src_width/2], video.src_width/2, 1, dump_file.handle);	

	return;
}
