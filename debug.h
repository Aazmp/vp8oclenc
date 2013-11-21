void open_dump_file()
{
	dump_file.path = DUMPPATH;
	dump_file.handle = fopen(dump_file.path, "wb");
	fwrite(frames.header, frames.header_sz, 1, dump_file.handle);
}

void dump()
{
	if (frames.frame_number > 1500) return; //disk space guard

	device.state_gpu = clEnqueueReadBuffer(device.commandQueue_gpu, device.reconstructed_frame_Y ,CL_TRUE, 0, video.wrk_frame_size_luma, frames.reconstructed_Y, 0, NULL, NULL);
	device.state_gpu = clEnqueueReadBuffer(device.commandQueue_gpu, device.reconstructed_frame_U ,CL_TRUE, 0, video.wrk_frame_size_chroma, frames.reconstructed_U, 0, NULL, NULL);
	device.state_gpu = clEnqueueReadBuffer(device.commandQueue_gpu, device.reconstructed_frame_V ,CL_TRUE, 0, video.wrk_frame_size_chroma, frames.reconstructed_V, 0, NULL, NULL);

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
