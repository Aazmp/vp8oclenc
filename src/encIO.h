void gather_frame()
{
	// get info about partition sizes
	device.state_cpu = clEnqueueReadBuffer(device.boolcoder_commandQueue_cpu, device.partitions_sizes ,CL_TRUE, 0, 8*sizeof(cl_int), frames.partition_sizes, 0, NULL, NULL);
	// write partition size data
	// each size = 3 byte in little endian (LSB first)
	cl_int i;
	for (i = 0; i < (video.number_of_partitions-1); ++i) // size of last is not written
	{
		cl_uint psize = frames.partition_sizes[i];
		cl_uchar psize_b = (cl_uchar)(psize & 0xff);
		frames.encoded_frame[frames.encoded_frame_size] = psize_b;
		++frames.encoded_frame_size;
		psize_b = (cl_uchar)((psize >> 8) & 0xff);
		frames.encoded_frame[frames.encoded_frame_size] = psize_b;
		++frames.encoded_frame_size;
		psize_b = (cl_uchar)((psize >> 16) & 0xff);
		frames.encoded_frame[frames.encoded_frame_size] = psize_b;
		++frames.encoded_frame_size;
	}
	//copy coefficient-partitions
	for (i = 0; i < video.number_of_partitions; ++i)
	{
		device.state_gpu = clEnqueueReadBuffer(device.boolcoder_commandQueue_cpu, device.partitions, CL_TRUE, i*video.partition_step, frames.partition_sizes[i], 
																								&frames.encoded_frame[frames.encoded_frame_size], 0, NULL, NULL);	
		frames.encoded_frame_size += frames.partition_sizes[i];		
	}
	// now we got encoded frame
	return;
}

void write_output_file()
{
	// clock start in gather frame
	// write ivf frame header (12 bytes) LITTLE ENDIAN
	cl_uchar byte;
	cl_ulong timestamp;
	// 0-3 frame(vp8 frame) size
	byte = (cl_uchar)(frames.encoded_frame_size & 0xff);
	fwrite(&byte, 1, 1, output_file.handle);
	byte = (cl_uchar)((frames.encoded_frame_size >> 8) & 0xff);
	fwrite(&byte, 1, 1, output_file.handle);
	byte = (cl_uchar)((frames.encoded_frame_size >> 16) & 0xff);
	fwrite(&byte, 1, 1, output_file.handle);
	byte = (cl_uchar)((frames.encoded_frame_size >> 24) & 0xff);
	fwrite(&byte, 1, 1, output_file.handle);
	// 64bit timestamp
	timestamp = ((cl_ulong)(frames.frame_number))*((cl_ulong)video.timestep);
	byte = (cl_uchar)(timestamp & 0xff);
	fwrite(&byte, 1, 1, output_file.handle);
	byte = (cl_uchar)((timestamp >> 8) & 0xff);
	fwrite(&byte, 1, 1, output_file.handle);
	byte = (cl_uchar)((timestamp >> 16) & 0xff);
	fwrite(&byte, 1, 1, output_file.handle);
	byte = (cl_uchar)((timestamp >> 24) & 0xff);
	fwrite(&byte, 1, 1, output_file.handle);
	byte = (cl_uchar)((timestamp >> 32) & 0xff);
	fwrite(&byte, 1, 1, output_file.handle);
	byte = (cl_uchar)((timestamp >> 40) & 0xff);
	fwrite(&byte, 1, 1, output_file.handle);
	byte = (cl_uchar)((timestamp >> 48) & 0xff);
	fwrite(&byte, 1, 1, output_file.handle);
	byte = (cl_uchar)((timestamp >> 56) & 0xff);
	fwrite(&byte, 1, 1, output_file.handle);
	// now print frame
	fwrite(frames.encoded_frame, 1, frames.encoded_frame_size, output_file.handle);

	return;
}

void write_output_header()
{
	fseek (output_file.handle, 0, SEEK_SET);
	// header size 32bytes LITTLE ENDIAN
	cl_uchar byte;
	// 0-3 "DKIF"
	byte = 'D'; fwrite(&byte, 1, 1, output_file.handle);
	byte = 'K'; fwrite(&byte, 1, 1, output_file.handle);
	byte = 'I'; fwrite(&byte, 1, 1, output_file.handle);
	byte = 'F'; fwrite(&byte, 1, 1, output_file.handle);
	// 4-5 version (only 0 allowed)
	byte = 0; fwrite(&byte, 1, 1, output_file.handle);
			  fwrite(&byte, 1, 1, output_file.handle);
	// 6-7 header length in bytes 
	byte = 32; fwrite(&byte, 1, 1, output_file.handle);
	byte = 0; fwrite(&byte, 1, 1, output_file.handle);
	// 9-11 "VP80"
	byte = 'V'; fwrite(&byte, 1, 1, output_file.handle);
	byte = 'P'; fwrite(&byte, 1, 1, output_file.handle);
	byte = '8'; fwrite(&byte, 1, 1, output_file.handle);
	byte = '0'; fwrite(&byte, 1, 1, output_file.handle);
	// 12-13 width
	byte = (cl_uchar)(video.dst_width & 0xff);
	fwrite(&byte, 1, 1, output_file.handle);
	byte = (cl_uchar)((video.dst_width >> 8) & 0xff); 
	fwrite(&byte, 1, 1, output_file.handle);
	// 14-15 height
	byte = (cl_uchar)(video.dst_height & 0xff);
	fwrite(&byte, 1, 1, output_file.handle);
	byte = (cl_uchar)((video.dst_height >> 8) & 0xff); 
	fwrite(&byte, 1, 1, output_file.handle);
	// 16-19 framerate
	cl_uint fr = video.framerate;
	byte = (cl_uchar)(fr & 0xff);
	fwrite(&byte, 1, 1, output_file.handle);
	byte = (cl_uchar)((fr >> 8) & 0xff); 
	fwrite(&byte, 1, 1, output_file.handle);
	byte = (cl_uchar)((fr >> 16) & 0xff); 
	fwrite(&byte, 1, 1, output_file.handle);
	byte = (cl_uchar)((fr >> 24) & 0xff); 
	fwrite(&byte, 1, 1, output_file.handle);
	// 20-23 timescale
	byte = (cl_uchar)(video.timescale & 0xff);
	fwrite(&byte, 1, 1, output_file.handle);
	byte = (cl_uchar)((video.timescale >> 8) & 0xff); 
	fwrite(&byte, 1, 1, output_file.handle);
	byte = (cl_uchar)((video.timescale >> 16) & 0xff); 
	fwrite(&byte, 1, 1, output_file.handle);
	byte = (cl_uchar)((video.timescale >> 24) & 0xff); 
	fwrite(&byte, 1, 1, output_file.handle);
	// 24-27 frame count
	++frames.frame_number;
	byte = (cl_uchar)(frames.frame_number & 0xff);
	fwrite(&byte, 1, 1, output_file.handle);
	byte = (cl_uchar)((frames.frame_number >> 8) & 0xff); 
	fwrite(&byte, 1, 1, output_file.handle);
	byte = (cl_uchar)((frames.frame_number >> 16) & 0xff); 
	fwrite(&byte, 1, 1, output_file.handle);
	byte = (cl_uchar)((frames.frame_number >> 24) & 0xff); 
	fwrite(&byte, 1, 1, output_file.handle);
	--frames.frame_number;
	// 28-32 not using
	byte = 0;
	fwrite(&byte, 1, 1, output_file.handle);
	fwrite(&byte, 1, 1, output_file.handle);
	fwrite(&byte, 1, 1, output_file.handle);
	fwrite(&byte, 1, 1, output_file.handle);
	return; 
}

int copy_with_padding()
{
    	int i, j;
    	cl_uchar *srcY, *srcU, *srcV, *dstY, *dstU, *dstV;
    	cl_uchar ext_pixelY, ext_pixelU, ext_pixelV;
    	//first line copy
    	srcY = frames.tmp_Y;        srcU = frames.tmp_U;        srcV = frames.tmp_V;
    	dstY = frames.current_Y;    dstU = frames.current_U;    dstV = frames.current_V;
    	int wrk_width_chroma = video.wrk_width>>1;
    	int src_width_chroma = video.src_width>>1;
	
	for (i = 0; i < video.src_height; i+=2)
	{
        // two luma lines, one chroma and one chroma line at step
		memcpy(dstY, srcY, video.src_width);
		ext_pixelY = srcY[video.src_width-1];
		for (j = video.src_width; j < video.wrk_width; ++j) // extend to the right
			dstY[j] = ext_pixelY;
		srcY += video.src_width; // dst_width/height == src_width/height if this function called
		dstY += video.wrk_width;

		memcpy(dstY, srcY, video.src_width);
		ext_pixelY = srcY[video.src_width-1];
		for (j = video.src_width; j < video.wrk_width; ++j) // extend to the right
			dstY[j] = ext_pixelY;
		srcY += video.src_width;
		dstY += video.wrk_width;

		memcpy(dstU, srcU, src_width_chroma);
		ext_pixelU = srcU[src_width_chroma-1];	
		for (j = src_width_chroma; j < wrk_width_chroma; ++j) // extend to the right
			dstU[j] = ext_pixelU;
		srcU += (src_width_chroma);
        dstU += (wrk_width_chroma);

		memcpy(dstV, srcV, src_width_chroma);
		ext_pixelV = srcU[src_width_chroma-1];
		for (j = src_width_chroma; j < wrk_width_chroma; ++j) // extend to the right
			dstU[j] = ext_pixelV;
		srcV += (src_width_chroma);
		dstV += (wrk_width_chroma);
	}
	// now copy last line to all lower lines, so increment only for dst
	srcY = dstY - video.wrk_width;
	srcU = dstU - wrk_width_chroma;
	srcV = dstV - wrk_width_chroma;

	for (i = video.src_height; i < video.wrk_height; i+=2)
	{
		memcpy(dstY, srcY, video.wrk_width);
		dstY += video.wrk_width;
		memcpy(dstY, srcY, video.wrk_width);
		dstY += video.wrk_width;
		memcpy(dstU, srcU, wrk_width_chroma);
		dstU += (wrk_width_chroma);
		memcpy(dstV, srcV, wrk_width_chroma);
		dstV += (wrk_width_chroma);
	}

    	return 1;
}


int get_yuv420_frame()
{
	// if there is padding, could be just pointer switch
	if (frames.frame_number > 0) {
		memcpy(frames.last_U, frames.current_U, video.wrk_frame_size_chroma);
		memcpy(frames.last_V, frames.current_V, video.wrk_frame_size_chroma);
	}

	int src_frame_size_full = video.src_frame_size_luma + (video.src_frame_size_chroma << 1);
	int i, j, fragment_size = src_frame_size_full;

	i = fread(frames.input_pack, sizeof(cl_uchar), (src_frame_size_full % fragment_size), input_file.handle);
	while (i < src_frame_size_full)
	{
		j = fread(frames.input_pack + i, sizeof(cl_uchar), fragment_size, input_file.handle);
		if (j < fragment_size) 
			return 0;
		i += j;
	}

	frames.tmp_Y = frames.input_pack;
	frames.tmp_U = frames.tmp_Y + video.src_frame_size_luma;
	frames.tmp_V = frames.tmp_U + video.src_frame_size_chroma;

	if ((video.src_height == video.dst_height) && (video.src_width == video.dst_width)) //== no resize
	{
		if ((video.wrk_height == video.dst_height) && (video.wrk_width == video.dst_width)) //== no padding
		{
			// then our buffers already continious, no need in paddings, just assign raw data to current
			frames.current_Y = frames.tmp_Y;
			frames.current_U = frames.tmp_U;
			frames.current_V = frames.tmp_V;
		}
		else
		{
           // malloc for current_YUV was done at runtime
           copy_with_padding(); //from tmp_yuv to current_yuv
		}
	}
	char buf[6];
	i = fread(buf, sizeof(cl_uchar), 6, input_file.handle);
	if ((i > 0) && ((buf[0] != 'F') || (buf[4] != 'E'))) {
		printf("broken stream!\n");
		return -1;
	}

	//memset(frames.current_Y, 128, video.wrk_frame_size_luma);
	//memset(frames.current_U, 128, video.wrk_frame_size_chroma); //make black and white
	//memset(frames.current_V, 128, video.wrk_frame_size_chroma);
	return 1;
}