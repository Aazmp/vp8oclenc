//a method to define path to OpenCL.lib
#pragma comment(lib,"C:\\Program Files (x86)\\AMD APP\\lib\\x86\\OpenCL.lib")

// all global structure definitions (fileContext, videoContext, deviceContext...)
#include "vp8enc.h"

//these are global variables used all over the encoder
struct fileContext	input_file, //YUV4MPEG2 
					output_file, //IVF
					error_file; //TXT for OpenCl compiler errors
struct deviceContext device; //both GPU and CPU OpenCL-devices (different handles, memory-objects, commlines...)
struct videoContext video; //properties of the video (sizes, indicies, vector limits...)
struct hostFrameBuffers frames; // host buffers, frame number, current/previous frame flags...
struct times t; // times...

#include "IO.h"
#include "init.h"

void zigzag_block(int16_t *block)
{
    //      zigzag[16] = { 0, 1, 4, 8, 5, 2, 3,  6, 9, 12, 13, 10, 7, 11, 14, 15 };
    //  inv_zigzag[16] = { 0, 1, 5, 6, 2, 4, 7, 12, 3,  8, 11, 13, 9, 10, 14, 15 };

    int16_t tmp1, tmp2, tmp3;

    tmp1 = block[2];
    tmp2 = block[3];
    tmp3 = block[10];
    block[2] = block[4];
    block[3] = block[8];
    block[10] = block[13];
    block[4] = block[5];
    block[8] = block[9];
    block[13] = block[11];
    block[5] = tmp1;
    block[9] = block[12];
    block[11] = tmp3;
    block[12] = block[7];
    block[7] = block[6];
    block[6] = tmp2;

    return;
}

////////////////// transforms are taken from multimedia mike's encoder version

static const int cospi8sqrt2minus1=20091;
static const int sinpi8sqrt2 =35468;

extern void encode_header(uint8_t* partition);

void entropy_encode()
{
	// here we start preparing DCT coefficient probabilities for frame
	//             by calculating average for all situations 
	// count_probs - accumulate numerators(num) and denominators(denom)
	// for each context
	// num[i][j][k][l] - amount of ZEROs which must be coded in i,j,k,l context
	// denom[i][j][k][l] - amount of bits(both 0 and 1) in i,j,k,l context to be coded
	t.start = clock();
	clSetKernelArg(device.count_probs, 7, sizeof(int32_t), &frames.prev_is_key_frame);	
	clFinish(device.commandQueue_cpu);
	clEnqueueNDRangeKernel(device.commandQueue_cpu, device.count_probs, 1, NULL, device.cpu_work_items_per_dim, device.cpu_work_group_size_per_dim, 0, NULL, NULL);
	clFinish(device.commandQueue_cpu); 

	// just dividing nums by denoms and getting probability of bit being ZERO
	clEnqueueNDRangeKernel(device.commandQueue_cpu, device.num_div_denom, 1, NULL, device.cpu_work_items_per_dim, device.cpu_work_group_size_per_dim, 0, NULL, NULL);
	clFinish(device.commandQueue_cpu); // Blocking, we need prob-values before encoding coeffs and header
	t.count_probs += clock() - t.start;

	// read calculated values
	clEnqueueReadBuffer(device.commandQueue_cpu, device.coeff_probs ,CL_TRUE, 0, 11*3*8*4*sizeof(uint32_t), frames.new_probs, 0, NULL, NULL);

	t.start = clock();
	// start of encoding coefficients 
	clSetKernelArg(device.encode_coefficients, 9, sizeof(int32_t), &frames.prev_is_key_frame);	
	clEnqueueNDRangeKernel(device.commandQueue_cpu, device.encode_coefficients, 1, NULL, device.cpu_work_items_per_dim, device.cpu_work_group_size_per_dim, 0, NULL, NULL);
	clFlush(device.commandQueue_cpu); // we don't nedd result until gather_frame(), so no block now
	t.bool_encode_coeffs += clock() - t.start; // this time means something only if encode_coefficients is blocking (followed by clFinish, not clFlush)

	t.start = clock();
	// encoding header is done as a part of HOST code placed in entropy_host.c[pp]|entropy_host.h
	encode_header(frames.encoded_frame); 
	t.bool_encode_header += clock() - t.start;

    return;
}
void dequant_iDCT_depred_4x4(int16_t *input, uint8_t *output, int32_t width, int16_t dc_q, int16_t ac_q, int16_t *top_pred, int16_t *left_pred, int16_t top_left_pred)
{
	// fixed on TM_[B_]PRED
    int i;
    int a1, b1, c1, d1;
    int16_t ip0, ip4, ip8, ip12;
    int16_t tmp_block[16];
    short *ip=input;
    short *tp=tmp_block;
    int temp1, temp2;
    uint16_t q = dc_q;

    for (i = 0; i < 4; ++i)
    {
        ip0 = ip[0] * q;
        q = ac_q;
        ip4 = ip[4] * q;
        ip8 = ip[8] * q;
        ip12 = ip[12] * q;

        a1 = ip0+ip8;
        b1 = ip0-ip8;

        temp1 = (ip4 * sinpi8sqrt2)>>16;
        temp2 = ip12 + ((ip12 * cospi8sqrt2minus1)>>16);
        c1 = temp1 - temp2;

        temp1 = ip4 + ((ip4 * cospi8sqrt2minus1)>>16);
        temp2 = (ip12 * sinpi8sqrt2)>>16;
        d1 = temp1 + temp2;

        tp[0] = a1 + d1;
        tp[12] = a1 - d1;
        tp[4] = b1 + c1;
        tp[8] = b1 - c1;

        ++ip;
        ++tp;
    }

    uint8_t *op = output;
    tp = tmp_block;
    for(i = 0; i < 4; ++i)
    {
        a1 = tp[0]+tp[2];
        b1 = tp[0]-tp[2];

        temp1 = (tp[1] * sinpi8sqrt2)>>16;
        temp2 = tp[3]+((tp[3] * cospi8sqrt2minus1)>>16);
        c1 = temp1 - temp2;

        temp1 = tp[1] + ((tp[1] * cospi8sqrt2minus1)>>16);
        temp2 = (tp[3] * sinpi8sqrt2)>>16;
        d1 = temp1 + temp2;

        /* after adding this results to predictors - clamping maybe needed */
		tp[0] = (top_pred[0] - top_left_pred + left_pred[i]);
		tp[0] = (tp[0] < 0) ? 0 : ((tp[0] > 255) ? 255 : tp[0]);
        tp[0] = ((a1 + d1 + 4) >> 3) + tp[0];
		op[0] = (uint8_t)((tp[0] > 255) ? 255 : ((tp[0] < 0) ? 0 : tp[0] ));

        tp[3] = (top_pred[3] - top_left_pred + left_pred[i]);
		tp[3] = (tp[3] < 0) ? 0 : ((tp[3] > 255) ? 255 : tp[3]);
		tp[3] = ((a1 - d1 + 4) >> 3) + tp[3];
        op[3] = (uint8_t)((tp[3] > 255) ? 255 : ((tp[3] < 0) ? 0 : tp[3] ));

		tp[1] = (top_pred[1] - top_left_pred + left_pred[i]);
		tp[1] = (tp[1] < 0) ? 0 : ((tp[1] > 255) ? 255 : tp[1]);
        tp[1] = ((b1 + c1 + 4) >> 3) + tp[1];
        op[1] = (uint8_t)((tp[1] > 255) ? 255 : ((tp[1] < 0) ? 0 : tp[1] ));

		tp[2] = (top_pred[2] - top_left_pred + left_pred[i]);
		tp[2] = (tp[2] < 0) ? 0 : ((tp[2] > 255) ? 255 : tp[2]);
        tp[2] = ((b1 - c1 + 4) >> 3) + tp[2];
        op[2] = (uint8_t)((tp[2] > 255) ? 255 : ((tp[2] < 0) ? 0 : tp[2] ));

        tp+=4;
        op+=width;
    }

	return;
}


static void pred_DCT_quant_4x4(uint8_t *input, int16_t *output, int32_t width, int16_t dc_q, int16_t ac_q, int16_t *top_pred, int16_t *left_pred, int16_t top_left_pred)
{
	// fixed on TM_[B_]PRED
    // input - pointer to start of block in raw frame. I-line of the block will be input + I*width
    // output - pointer to encoded_macroblock.block[i] data.
    int32_t i;
    int32_t a1, b1, c1, d1;
    uint8_t *ip = input;
    int16_t *op = output;
    int32_t ip0, ip1, ip2, ip3;

    for (i = 0; i < 4; i++)
    {
		ip0 = (int32_t)(top_pred[0] + left_pred[i] - top_left_pred); ip0 = (ip0 < 0) ? 0 : ((ip0 > 255) ? 255 : ip0);
		ip1 = (int32_t)(top_pred[1] + left_pred[i] - top_left_pred); ip1 = (ip1 < 0) ? 0 : ((ip1 > 255) ? 255 : ip1);
		ip2 = (int32_t)(top_pred[2] + left_pred[i] - top_left_pred); ip2 = (ip2 < 0) ? 0 : ((ip2 > 255) ? 255 : ip2);
		ip3 = (int32_t)(top_pred[3] + left_pred[i] - top_left_pred); ip3 = (ip3 < 0) ? 0 : ((ip3 > 255) ? 255 : ip3);

        // subtract prediction
        ip0 = ((int32_t)ip[0] - ip0);
        ip1 = ((int32_t)ip[1] - ip1);
        ip2 = ((int32_t)ip[2] - ip2);
        ip3 = ((int32_t)ip[3] - ip3);

        a1 = ((ip0 + ip3)<<3);
        b1 = ((ip1 + ip2)<<3);
        c1 = ((ip1 - ip2)<<3);
        d1 = ((ip0 - ip3)<<3);

        op[0] = (int16_t)(a1 + b1);
        op[2] = (int16_t)(a1 - b1);

        op[1] = (int16_t)((c1 * 2217 + d1 * 5352 +  14500)>>12);
        op[3] = (int16_t)((d1 * 2217 - c1 * 5352 +   7500)>>12);

        ip += width; // because in's in the raw frame
        op += 4; // because it's in block-packed order

    }

    int32_t q = dc_q; // for the first coeff

    op = output;

    for (i = 0; i < 4; i++)
    {
        a1 = op[0] + op[12];
        b1 = op[4] + op[8];
        c1 = op[4] - op[8];
		d1 = op[0] - op[12];

        op[0] = (( a1 + b1 + 7)>>4) / q; // quant using dc_q only first time
        q = ac_q; // switch to ac_q for all others
        op[8] = (( a1 - b1 + 7)>>4) / q;

        op[4]  = (((c1 * 2217 + d1 * 5352 +  12000)>>16) + (d1!=0)) / q;
        op[12] = ((d1 * 2217 - c1 * 5352 +  51000)>>16) / q;

        ++op;
    }
	return;
}



static void dequant_iWHT_4x4(int32_t mb_num)
{
	//at this moment we never call this function, because of using only SPLITMV and B_PRED
    int i;
    int a1, b1, c1, d1;
    int a2, b2, c2, d2;
    int16_t *ip = frames.transformed_blocks[mb_num].coeffs[24];
    short *op = frames.transformed_blocks[mb_num].coeffs[0];
    int32_t q = video.quantizer_y2_dc_i;

    for (i = 0; i < 4; ++i)
    {
        a2 = ip[0]*q;
        q = video.quantizer_y2_ac_i;
        b2 = ip[4]*q;
        c2 = ip[8]*q;
        d2 = ip[12]*q;

        a1 = a2 + d2;
        b1 = b2 + c2;
        c1 = b2 - c2;
        d1 = a2 - d2;

        op[0] = a1 + b1;
        op[64] = c1 + d1;
        op[128] = a1 - b1;
        op[192] = d1 - c1;
        ++ip;
        op+=16;
    }

    ip = frames.transformed_blocks[mb_num].coeffs[0];
    op = ip;

    for (i = 0; i < 4; i++)
    {
        a1 = ip[0] + ip[48]; // 0 - 0; 1 - 16; 2 - 32; 3 - 48;
        b1 = ip[16] + ip[32];
        c1 = ip[16] - ip[32];
        d1 = ip[0] - ip[48];

        a2 = a1 + b1;
        b2 = c1 + d1;
        c2 = a1 - b1;
        d2 = d1 - c1;

        op[0] = (a2 + 3) >> 3;
        op[16] = (b2 + 3) >> 3;
        op[32] = (c2 + 3) >> 3;
        op[48] = (d2 + 3) >> 3;

        ip += 64;
        op += 64;
    }
	return;
}




static void WHT_quant_4x4(int32_t mb_num)
{
	//at this moment we never call this function, because of using only SPLITMV and B_PRED
    int32_t i;
    int32_t a1, b1, c1, d1;
    int32_t a2, b2, c2, d2;
    int16_t *ip = frames.transformed_blocks[mb_num].coeffs[0];
    int16_t *op = frames.transformed_blocks[mb_num].coeffs[24];

    for (i = 0; i < 4; i++)
    {
        // extracting dc coeffs from block-ordered macroblock
        // ip[0] - DCcoef of first mb in line, +16 coeffs and starts second, and so on...
        // ..every i element will be stored with 16*i offset from the start
        a1 = (ip[0] + ip[48]);
        b1 = (ip[16] + ip[32]);
        c1 = (ip[16] - ip[32]);
        d1 = (ip[0] - ip[48]);

        op[0] = a1 + b1;
        op[1] = c1 + d1;
        op[2] = a1 - b1;
        op[3] = d1 - c1;
        ip += 64; // input goes from [25][16] coeffs, so we skip line of 4 blocks (each 16 coeffs)
        op += 4;
    }

    ip = frames.transformed_blocks[mb_num].coeffs[24];
    op = ip;
    int32_t q = video.quantizer_y2_dc_i;

    for (i = 0; i < 4; i++)
    {
        a1 = ip[0] + ip[12];
        b1 = ip[4] + ip[8];
        c1 = ip[4] - ip[8];
        d1 = ip[0] - ip[12];

        a2 = a1 + b1;
        b2 = c1 + d1;
        c2 = a1 - b1;
        d2 = d1 - c1;

        a2 += (a2 > 0);
        b2 += (b2 > 0);
        c2 += (c2 > 0);
        d2 += (d2 > 0);

        op[0] = ((a2) >> 1)/q;
        q = video.quantizer_y2_ac_i;
        op[4] = ((b2) >> 1)/q;
        op[8] = ((c2) >> 1)/q;
        op[12] = ((d2) >> 1)/q;

        ++ip;
        ++op;
    }
	return;
}



void predict_and_transform_mb(int32_t mb_num)
{
	// prepare predictors (TM_B_PRED) and transform each block
    uint32_t i;
    uint32_t mb_row, mb_col, b_num, b_col, b_row;
    uint32_t Y_offset, UV_offset; // number of pixels from plane start
    uint32_t pred_ind_Y, pred_ind_UV;

    int16_t top_left_pred_Y, top_left_pred_U, top_left_pred_V;
    int16_t left_pred_Y[16], left_pred_U[8], left_pred_V[8];
    int16_t top_pred_Y[16], top_pred_U[8], top_pred_V[8];

    mb_row = mb_num / video.mb_width;
    mb_col = mb_num % video.mb_width;
    //Y_offset = (mb_row * 16)*video.wrk_width + (mb_col * 16);
    //UV_offset = (mb_row * 8)*(video.wrk_width / 2) + (mb_col * 8);
    Y_offset = ((mb_row * video.wrk_width) + mb_col) << 4;
    UV_offset = ((mb_row *video.wrk_width) << 2) + (mb_col << 3);

    // compute predictors
    if (mb_col == 0)
    {
        top_left_pred_Y = 129;
        top_left_pred_U = 129;
        top_left_pred_V = 129;
        for (i = 0; i < 16; i+=2)
        {
            left_pred_Y[i]=129;
            left_pred_Y[i+1]=129;
            left_pred_U[i>>1]=129;
            left_pred_V[i>>1]=129;
        }
    }
    else
    {
        pred_ind_Y = Y_offset - 1;
        pred_ind_UV = UV_offset - 1;
        for (i = 0; i < 16; i+=2)
        {
            left_pred_Y[i]= (int16_t)frames.reconstructed_Y[pred_ind_Y];
            pred_ind_Y += video.wrk_width;
            left_pred_Y[i+1]= (int16_t)frames.reconstructed_Y[pred_ind_Y];
            pred_ind_Y += video.wrk_width;
            left_pred_U[i/2]= (int16_t)frames.reconstructed_U[pred_ind_UV];
            left_pred_V[i/2]=  (int16_t)frames.reconstructed_V[pred_ind_UV];
            pred_ind_UV += (video.wrk_width>>1);
        }
    }

    if (mb_row == 0)
    {
        top_left_pred_Y = 127; // 127 beats 129 in the corner :)
        top_left_pred_U = 127;
        top_left_pred_V = 127;
        for (i = 0; i < 16; i+=2)
        {
            top_pred_Y[i]=127;
            top_pred_Y[i+1]=127;
            top_pred_U[i>>1]=127;
            top_pred_V[i>>1]=127;
        }
    }
    else
    {
        pred_ind_Y = Y_offset - video.wrk_width;
        pred_ind_UV = UV_offset - (video.wrk_width>>1);
        for (i = 0; i < 16; i+=2)
        {
            top_pred_Y[i]= (int16_t)frames.reconstructed_Y[pred_ind_Y];
            ++pred_ind_Y;
            top_pred_Y[i+1]= (int16_t)frames.reconstructed_Y[pred_ind_Y];
            ++pred_ind_Y;
            top_pred_U[i/2]= (int16_t)frames.reconstructed_U[pred_ind_UV];
            top_pred_V[i/2]=  (int16_t)frames.reconstructed_V[pred_ind_UV];
            ++pred_ind_UV;
        }
    }

    if ((mb_row != 0) && (mb_col != 0))
    {
        top_left_pred_Y = frames.reconstructed_Y[Y_offset - video.wrk_width - 1];
        top_left_pred_U = frames.reconstructed_U[UV_offset - (video.wrk_width>>1) - 1];
        top_left_pred_V = frames.reconstructed_V[UV_offset - (video.wrk_width>>1) - 1];
    }

    uint8_t *block_offset;
    for (b_row = 0; b_row < 4; ++b_row) // 4x4 luma blocks
    {
		int16_t buf_pred = left_pred_Y[(b_row<<2) + 3];
		for (b_col = 0; b_col < 4; ++b_col)
		{
			b_num = (b_row<<2) + b_col;

			block_offset = frames.current_Y + (Y_offset + (((b_row*video.wrk_width) + b_col ) <<2));

			pred_DCT_quant_4x4(block_offset, frames.transformed_blocks[mb_num].coeffs[b_num],
			                   video.wrk_width, video.quantizer_y_dc_i, video.quantizer_y_ac_i,
			                   &top_pred_Y[b_col<<2], &left_pred_Y[b_row<<2], top_left_pred_Y);

			block_offset = frames.reconstructed_Y + (Y_offset + (((b_row*video.wrk_width) + b_col ) <<2));

			dequant_iDCT_depred_4x4(frames.transformed_blocks[mb_num].coeffs[b_num], block_offset,
			                        video.wrk_width, video.quantizer_y_dc_i, video.quantizer_y_ac_i,
			                        &top_pred_Y[b_col<<2], &left_pred_Y[b_row<<2], top_left_pred_Y);

			zigzag_block(frames.transformed_blocks[mb_num].coeffs[b_num]);

			top_left_pred_Y = top_pred_Y[(b_col<<2) + 3];

			left_pred_Y[b_row<<2] = block_offset[3];
			block_offset += video.wrk_width;
			left_pred_Y[(b_row<<2) + 1] = block_offset[3];
			block_offset += video.wrk_width;
			left_pred_Y[(b_row<<2) + 2] = block_offset[3];
			block_offset += video.wrk_width;
			left_pred_Y[(b_row<<2) + 3] = block_offset[3];

			top_pred_Y[b_col<<2] = block_offset[0];
			top_pred_Y[(b_col<<2) + 1] = block_offset[1];
			top_pred_Y[(b_col<<2) + 2] = block_offset[2];
			top_pred_Y[(b_col<<2) + 3] = block_offset[3];
		}
		top_left_pred_Y = buf_pred;
    }
    for (b_num = 0; b_num < 4; ++b_num) // 2x2 U-chroma blocks
    {
        b_row = b_num >> 1; // /2
        b_col = b_num % 2;

        block_offset = frames.current_U + (UV_offset + (( b_row*video.wrk_width<<1) + (b_col<<2) ));

        pred_DCT_quant_4x4(block_offset, frames.transformed_blocks[mb_num].coeffs[b_num+16],
                           video.wrk_width>>1, video.quantizer_uv_dc_i, video.quantizer_uv_ac_i,
                           &top_pred_U[b_col<<2], &left_pred_U[b_row<<2], top_left_pred_U);

        // chroma blocks could be immediately reconstructed
        block_offset = frames.reconstructed_U + (UV_offset + (( b_row*video.wrk_width<<1) + (b_col<<2) ));

        dequant_iDCT_depred_4x4(frames.transformed_blocks[mb_num].coeffs[b_num+16], block_offset,
                                video.wrk_width>>1,video.quantizer_uv_dc_i, video.quantizer_uv_ac_i,
                                &top_pred_U[b_col<<2], &left_pred_U[b_row<<2], top_left_pred_U);

        // and zigzag
        zigzag_block(frames.transformed_blocks[mb_num].coeffs[b_num+16]);
    }
    for (b_num = 0; b_num < 4; ++b_num) // 2x2 V-chroma blocks
    {
        b_row = b_num >> 1; // /2
        b_col = b_num % 2;
        block_offset = frames.current_V + (UV_offset + (( b_row*video.wrk_width<<1) + (b_col<<2) ));

        pred_DCT_quant_4x4(block_offset, frames.transformed_blocks[mb_num].coeffs[b_num+20],
                           video.wrk_width>>1, video.quantizer_uv_dc_i, video.quantizer_uv_ac_i,
                           &top_pred_V[b_col<<2], &left_pred_V[b_row<<2], top_left_pred_V);

        block_offset = frames.reconstructed_V + (UV_offset + (( b_row*video.wrk_width<<1) + (b_col<<2) ));

        dequant_iDCT_depred_4x4(frames.transformed_blocks[mb_num].coeffs[b_num+20], block_offset,
                                video.wrk_width>>1,video.quantizer_uv_dc_i, video.quantizer_uv_ac_i,
                                &top_pred_V[b_col<<2], &left_pred_V[b_row<<2], top_left_pred_V);

        zigzag_block(frames.transformed_blocks[mb_num].coeffs[b_num+20]);
    }

    return;
}

float count_SSIM_16x16(uint8_t *frame1, int32_t width1, uint8_t *frame2, int32_t width2)
{
	int i,j,M1=0,M2=0,D1=0,D2=0,C=0,t1,t2;
	for(i = 0; i < 16; ++i)
		for(j = 0; j < 16; ++j) {
			M1 += (int)frame1[i*width1+j];
			M2 += (int)frame2[i*width2+j];
		}
	M1 >>= 8; M2 >>= 8;
	for(i = 0; i < 16; ++i)
		for(j = 0; j < 16; ++j) {
			t1 = ((int)frame1[i*width1+j]) - M1;
			t2 = ((int)frame2[i*width2+j]) - M2;
			D1 += t1*t1; 
			D2 += t2*t2;
			C += t1*t2;
		}
	D1 >>= 8; D2 >>= 8; C >>= 8;

	const float c1 = 0.01f*0.01f*255*255;
	const float c2 = 0.03f*0.03f*255*255;
	float num, denom, M1f, M2f;
	M1f = (float)M1;
	M2f = (float)M2;
	num = (M1f*M2f*2 + c1)*((float)C*2 + c2);
	denom = (M1f*M1f + M2f*M2f + c1)*((float)D1 + (float)D2 + c2);

	return (num/denom);
}

int chroma_corrupted(uint8_t* frame1, int32_t width1, uint8_t *frame2, int32_t width2)
{
	int32_t i,j, sum1 = 0, sum2 = 0;
	for(i = 0; i < 8; ++i)
		for(j = 0; j < 8; ++j)
			sum1 += (int)frame1[i*width1 + j];
	for(i = 0; i < 8; ++i)
		for(j = 0; j < 8; ++j)
			sum2 += (int)frame2[i*width2 + j];
	sum1 = sum1 - sum2;
	sum1 = (sum1 < 0) ? -sum1 : sum1;
	sum1 = (sum1 > 2560); // divide by 256 to get average
	return sum1; 
}

int test_inter_on_intra(int32_t mb_num)
{
	macroblock test_mb;
	uint8_t test_recon_Y[256];
	uint8_t test_recon_V[64];
	uint8_t test_recon_U[64];
	const int32_t test_width = 16;

	// prepare predictors (TM_B_PRED) and transform each block
    int32_t i,j, mb_row, mb_col, b_num, b_col, b_row, Y_offset, UV_offset, pred_ind_Y, pred_ind_UV;

    int16_t top_left_pred_Y, top_left_pred_U, top_left_pred_V;
    int16_t left_pred_Y[16], left_pred_U[8], left_pred_V[8];
    int16_t top_pred_Y[16], top_pred_U[8], top_pred_V[8];

    mb_row = mb_num / video.mb_width;  mb_col = mb_num % video.mb_width;
    Y_offset = ((mb_row * video.wrk_width) + mb_col) << 4;
    UV_offset = ((mb_row *video.wrk_width) << 2) + (mb_col << 3);

    if (mb_col == 0) {
        top_left_pred_Y = 129; top_left_pred_U = 129; top_left_pred_V = 129;
        for (i = 0; i < 16; i+=2) {
            left_pred_Y[i]=129; left_pred_Y[i+1]=129; left_pred_U[i>>1]=129; left_pred_V[i>>1]=129;
        }
    }
    else {
        pred_ind_Y = Y_offset - 1; pred_ind_UV = UV_offset - 1;
        for (i = 0; i < 16; i+=2) {
            left_pred_Y[i]= (int16_t)frames.reconstructed_Y[pred_ind_Y];
            pred_ind_Y += video.wrk_width;
            left_pred_Y[i+1]= (int16_t)frames.reconstructed_Y[pred_ind_Y];
            pred_ind_Y += video.wrk_width;
            left_pred_U[i/2]= (int16_t)frames.reconstructed_U[pred_ind_UV];
            left_pred_V[i/2]=  (int16_t)frames.reconstructed_V[pred_ind_UV];
            pred_ind_UV += (video.wrk_width>>1);
        }
    }

    if (mb_row == 0) {
        top_left_pred_Y = 127; top_left_pred_U = 127; top_left_pred_V = 127;
        for (i = 0; i < 16; i+=2) {
            top_pred_Y[i]=127; top_pred_Y[i+1]=127; top_pred_U[i>>1]=127; top_pred_V[i>>1]=127;
        }
    }
    else {
        pred_ind_Y = Y_offset - video.wrk_width; pred_ind_UV = UV_offset - (video.wrk_width>>1);
        for (i = 0; i < 16; i+=2) {
            top_pred_Y[i]= (int16_t)frames.reconstructed_Y[pred_ind_Y];
            ++pred_ind_Y;
            top_pred_Y[i+1]= (int16_t)frames.reconstructed_Y[pred_ind_Y];
            ++pred_ind_Y;
            top_pred_U[i/2]= (int16_t)frames.reconstructed_U[pred_ind_UV];
            top_pred_V[i/2]=  (int16_t)frames.reconstructed_V[pred_ind_UV];
            ++pred_ind_UV;
        }
    }

    if ((mb_row != 0) && (mb_col != 0))
	{
        top_left_pred_Y = frames.reconstructed_Y[Y_offset - video.wrk_width - 1];
        top_left_pred_U = frames.reconstructed_U[UV_offset - (video.wrk_width>>1) - 1];
        top_left_pred_V = frames.reconstructed_V[UV_offset - (video.wrk_width>>1) - 1];
    }

    uint8_t *block_offset, *test_offset;
    for (b_row = 0; b_row < 4; ++b_row) // 4x4 luma blocks
    {
		int16_t buf_pred = left_pred_Y[(b_row<<2) + 3];
		for (b_col = 0; b_col < 4; ++b_col)
		{
			b_num = (b_row<<2) + b_col;

			block_offset = frames.current_Y + (Y_offset + (((b_row*video.wrk_width) + b_col ) <<2));
			test_offset = test_recon_Y + (((b_row*test_width) + b_col ) <<2);

			pred_DCT_quant_4x4(block_offset, test_mb.coeffs[b_num],
			                   video.wrk_width, video.quantizer_y_dc_p, video.quantizer_y_ac_p,
			                   &top_pred_Y[b_col<<2], &left_pred_Y[b_row<<2], top_left_pred_Y);

			dequant_iDCT_depred_4x4(test_mb.coeffs[b_num], test_offset,
			                        test_width, video.quantizer_y_dc_p, video.quantizer_y_ac_p,
			                        &top_pred_Y[b_col<<2], &left_pred_Y[b_row<<2], top_left_pred_Y);

			top_left_pred_Y = top_pred_Y[(b_col<<2) + 3];

			left_pred_Y[b_row<<2] = test_offset[3];
			test_offset += test_width;
			left_pred_Y[(b_row<<2) + 1] = test_offset[3];
			test_offset += test_width;
			left_pred_Y[(b_row<<2) + 2] = test_offset[3];
			test_offset += test_width;
			left_pred_Y[(b_row<<2) + 3] = test_offset[3];

			top_pred_Y[b_col<<2] = test_offset[0];
			top_pred_Y[(b_col<<2) + 1] = test_offset[1];
			top_pred_Y[(b_col<<2) + 2] = test_offset[2];
			top_pred_Y[(b_col<<2) + 3] = test_offset[3];
		}
		top_left_pred_Y = buf_pred;
    }
    for (b_num = 0; b_num < 4; ++b_num) // 2x2 U-chroma blocks
    {
        b_row = b_num >> 1; b_col = b_num % 2;
        block_offset = frames.current_U + (UV_offset + (( b_row*video.wrk_width<<1) + (b_col<<2) ));
		test_offset = test_recon_U + ((b_row*test_width<<1) + (b_col<<2));
        pred_DCT_quant_4x4(block_offset, test_mb.coeffs[b_num+16],
                           video.wrk_width>>1, video.quantizer_uv_dc_p, video.quantizer_uv_ac_p,
                           &top_pred_U[b_col<<2], &left_pred_U[b_row<<2], top_left_pred_U);
        dequant_iDCT_depred_4x4(test_mb.coeffs[b_num+16], test_offset,
                                test_width>>1,video.quantizer_uv_dc_p, video.quantizer_uv_ac_p,
                                &top_pred_U[b_col<<2], &left_pred_U[b_row<<2], top_left_pred_U);

    }
    for (b_num = 0; b_num < 4; ++b_num) // 2x2 V-chroma blocks
    {
        b_row = b_num >> 1; b_col = b_num % 2;
        block_offset = frames.current_V + (UV_offset + (( b_row*video.wrk_width<<1) + (b_col<<2) ));
		test_offset = test_recon_V + ((b_row*test_width<<1) + (b_col<<2));
        pred_DCT_quant_4x4(block_offset, test_mb.coeffs[b_num+20],
                           video.wrk_width>>1, video.quantizer_uv_dc_p, video.quantizer_uv_ac_p,
                           &top_pred_V[b_col<<2], &left_pred_V[b_row<<2], top_left_pred_V);
        dequant_iDCT_depred_4x4(test_mb.coeffs[b_num+20], test_offset,
                                test_width>>1,video.quantizer_uv_dc_p, video.quantizer_uv_ac_p,
                                &top_pred_V[b_col<<2], &left_pred_V[b_row<<2], top_left_pred_V);
    }
	
	test_mb.SSIM = count_SSIM_16x16(test_recon_Y, test_width, frames.current_Y + Y_offset, video.wrk_width);
	if (test_mb.SSIM > frames.transformed_blocks[mb_num].SSIM)
	//	&& (!chroma_corrupted(test_recon_U, test_width>>1, frames.current_U  + UV_offset, video.wrk_width>>1)) 
	//	&& (!chroma_corrupted(test_recon_V, test_width>>1, frames.current_V  + UV_offset, video.wrk_width>>1)))
	{
		//then we replace inter encoded and reconstructed MB with intra
		++frames.replaced;
		// replace residual
		memcpy(&frames.transformed_blocks[mb_num], &test_mb, sizeof(macroblock));
		// replace reconstructed fragment
		for(i = 0; i < 16; ++i)
			for(j = 0; j < 16; ++j)
				frames.reconstructed_Y[Y_offset + i*video.wrk_width + j] = test_recon_Y[i*test_width + j];
		for(i = 0; i < 8; ++i)
			for(j = 0; j < 8; ++j)
				frames.reconstructed_U[UV_offset + i*(video.wrk_width>>1) + j] = test_recon_U[i*(test_width>>1) + j];
		for(i = 0; i < 8; ++i)
			for(j = 0; j < 8; ++j)
				frames.reconstructed_V[UV_offset + i*(video.wrk_width>>1) + j] = test_recon_V[i*(test_width>>1) + j];

		for (b_num = 0; b_num < 24; ++b_num)
			zigzag_block(frames.transformed_blocks[mb_num].coeffs[b_num]);
		return 0;
	}
    return 1;
}



void intra_transform()
{
	t.start = clock();

	frames.frames_until_key = video.GOP_size;

    video.quantizer_y_dc_i = vp8_dc_qlookup[video.quantizer_index_y_dc_i];
    video.quantizer_y_ac_i = vp8_ac_qlookup[video.quantizer_index_y_ac_i];
    video.quantizer_y2_dc_i = (vp8_dc_qlookup[video.quantizer_index_y2_dc_i]) << 1; // *2
    video.quantizer_y2_ac_i = 31 * (vp8_ac_qlookup[video.quantizer_index_y2_ac_i]) / 20; // *155/100
    video.quantizer_uv_dc_i = vp8_dc_qlookup[video.quantizer_index_uv_dc_i];
    video.quantizer_uv_ac_i = vp8_ac_qlookup[video.quantizer_index_uv_ac_i];

    if (video.quantizer_y2_ac_i < 8)
        video.quantizer_y2_ac_i = 8;
    if (video.quantizer_uv_dc_i > 132)
        video.quantizer_uv_dc_i = 132;

	// on key transform we wait for all other to end
    int32_t mb_num;

    // now foreach macroblock
    for(mb_num = 0; mb_num < video.mb_count; ++mb_num)
    {
        // input data - raw current frame, output - in block-packed order
        // prepare macroblock (predict and subbtract) and transform(dct, wht, quant and zigzag)
        // BONUS: reconstruct in here too
		predict_and_transform_mb(mb_num);
    }
	t.intra_transform += clock() - t.start;
    return;
}


void do_loop_filter()
{
	t.start = clock();

	video.interior_limit = video.loop_filter_level;
	if (video.loop_filter_sharpness) {
		video.interior_limit >>= video.loop_filter_sharpness > 4 ? 2 : 1;
		if (video.interior_limit > 9 - video.loop_filter_sharpness)
			video.interior_limit = 9 - video.loop_filter_sharpness;
	}
	if (!video.interior_limit)
		video.interior_limit = 1;

	video.mbedge_limit = ((video.loop_filter_level + 2) * 2) + video.interior_limit;
	video.sub_bedge_limit = (video.loop_filter_level * 2) + video.interior_limit;

	if (video.loop_filter_type == 0) // normal
	{
		video.hev_threshold = 0;
		if (frames.prev_is_key_frame) /* current frame is a key frame */
		{
			if (video.loop_filter_level >= 40)
				video.hev_threshold = 2;
			else if (video.loop_filter_level >= 15)
				video.hev_threshold = 1;
		}
		else /* current frame is an interframe */
		{
			if (video.loop_filter_level >= 40)
				video.hev_threshold = 3;
			else if (video.loop_filter_level >= 20)
				video.hev_threshold = 2;
			else if (video.loop_filter_level >= 15)
				video.hev_threshold = 1;
		}
		device.state_gpu = clSetKernelArg(device.normal_loop_filter_MBH, 2, sizeof(int32_t), &video.mbedge_limit);
		device.state_gpu = clSetKernelArg(device.normal_loop_filter_MBH, 3, sizeof(int32_t), &video.sub_bedge_limit);
		device.state_gpu = clSetKernelArg(device.normal_loop_filter_MBH, 4, sizeof(int32_t), &video.interior_limit);
		device.state_gpu = clSetKernelArg(device.normal_loop_filter_MBH, 5, sizeof(int32_t), &video.hev_threshold);
		device.state_gpu = clSetKernelArg(device.normal_loop_filter_MBV, 2, sizeof(int32_t), &video.mbedge_limit);
		device.state_gpu = clSetKernelArg(device.normal_loop_filter_MBV, 3, sizeof(int32_t), &video.sub_bedge_limit);
		device.state_gpu = clSetKernelArg(device.normal_loop_filter_MBV, 4, sizeof(int32_t), &video.interior_limit);
		device.state_gpu = clSetKernelArg(device.normal_loop_filter_MBV, 5, sizeof(int32_t), &video.hev_threshold);
		int32_t mb_col, mb_size, plane_width;
		for (mb_col = 0; mb_col < video.mb_width; ++mb_col) 
		{
			device.state_gpu = clSetKernelArg(device.normal_loop_filter_MBH, 7, sizeof(int32_t), &mb_col);
			device.state_gpu = clSetKernelArg(device.normal_loop_filter_MBV, 7, sizeof(int32_t), &mb_col);
			device.gpu_work_group_size_per_dim[0] = 64; 

			device.gpu_work_items_per_dim[0] = video.mb_height*16;
			mb_size = 16;
			plane_width = video.wrk_width;
			device.state_gpu = clSetKernelArg(device.normal_loop_filter_MBH, 6, sizeof(int32_t), &mb_size);
			device.state_gpu = clSetKernelArg(device.normal_loop_filter_MBH, 1, sizeof(int32_t), &plane_width);
			device.state_gpu = clSetKernelArg(device.normal_loop_filter_MBH, 0, sizeof(cl_mem), &device.reconstructed_frame_Y);
			device.state_gpu = clEnqueueNDRangeKernel(device.commandQueue_gpu, device.normal_loop_filter_MBH, 1, NULL, device.gpu_work_items_per_dim, device.gpu_work_group_size_per_dim, 0, NULL, NULL);
			if (device.state_gpu != 0)  printf(">error while deblocking : %d", device.state_gpu);
			device.state_gpu = clFlush(device.commandQueue_gpu);
			if (device.state_gpu != 0)  printf(">error while deblocking : %d", device.state_gpu);
			device.gpu_work_items_per_dim[0] = video.mb_height*8;
			mb_size = 8;
			plane_width = video.wrk_width/2;
			device.state_gpu = clSetKernelArg(device.normal_loop_filter_MBH, 6, sizeof(int32_t), &mb_size);
			device.state_gpu = clSetKernelArg(device.normal_loop_filter_MBH, 1, sizeof(int32_t), &plane_width);
			device.state_gpu = clSetKernelArg(device.normal_loop_filter_MBH, 0, sizeof(cl_mem), &device.reconstructed_frame_U);
			device.state_gpu = clEnqueueNDRangeKernel(device.commandQueue_gpu, device.normal_loop_filter_MBH, 1, NULL, device.gpu_work_items_per_dim, device.gpu_work_group_size_per_dim, 0, NULL, NULL);
			if (device.state_gpu != 0)  printf(">error while deblocking : %d", device.state_gpu);
			device.state_gpu = clFlush(device.commandQueue_gpu);
			if (device.state_gpu != 0)  printf(">error while deblocking : %d", device.state_gpu);
			device.state_gpu = clSetKernelArg(device.normal_loop_filter_MBH, 0, sizeof(cl_mem), &device.reconstructed_frame_V);
			device.state_gpu = clEnqueueNDRangeKernel(device.commandQueue_gpu, device.normal_loop_filter_MBH, 1, NULL, device.gpu_work_items_per_dim, device.gpu_work_group_size_per_dim, 0, NULL, NULL);
			if (device.state_gpu != 0)  printf(">error while deblocking : %d", device.state_gpu);
			device.state_gpu = clFinish(device.commandQueue_gpu);
			if (device.state_gpu != 0)  printf(">error while deblocking : %d", device.state_gpu);

			device.gpu_work_items_per_dim[0] = video.mb_height*4;
			mb_size = 16;
			plane_width = video.wrk_width;
			device.state_gpu = clSetKernelArg(device.normal_loop_filter_MBV, 6, sizeof(int32_t), &mb_size);
			device.state_gpu = clSetKernelArg(device.normal_loop_filter_MBV, 1, sizeof(int32_t), &plane_width);
			device.state_gpu = clSetKernelArg(device.normal_loop_filter_MBV, 0, sizeof(cl_mem), &device.reconstructed_frame_Y);
			device.state_gpu = clEnqueueNDRangeKernel(device.commandQueue_gpu, device.normal_loop_filter_MBV, 1, NULL, device.gpu_work_items_per_dim, device.gpu_work_group_size_per_dim, 0, NULL, NULL);
			if (device.state_gpu != 0)  printf(">error while deblocking : %d", device.state_gpu);
			device.state_gpu = clFlush(device.commandQueue_gpu);
			if (device.state_gpu != 0)  printf(">error while deblocking : %d", device.state_gpu);
			device.gpu_work_items_per_dim[0] = video.mb_height*2;
			mb_size = 8;
			plane_width = video.wrk_width/2;
			device.state_gpu = clSetKernelArg(device.normal_loop_filter_MBV, 6, sizeof(int32_t), &mb_size);
			device.state_gpu = clSetKernelArg(device.normal_loop_filter_MBV, 1, sizeof(int32_t), &plane_width);
			device.state_gpu = clSetKernelArg(device.normal_loop_filter_MBV, 0, sizeof(cl_mem), &device.reconstructed_frame_U);
			device.state_gpu = clEnqueueNDRangeKernel(device.commandQueue_gpu, device.normal_loop_filter_MBV, 1, NULL, device.gpu_work_items_per_dim, device.gpu_work_group_size_per_dim, 0, NULL, NULL);
			if (device.state_gpu != 0)  printf(">error while deblocking : %d", device.state_gpu);
			device.state_gpu = clFlush(device.commandQueue_gpu);
			if (device.state_gpu != 0)  printf(">error while deblocking : %d", device.state_gpu);
			device.state_gpu = clSetKernelArg(device.normal_loop_filter_MBV, 0, sizeof(cl_mem), &device.reconstructed_frame_V);
			device.state_gpu = clEnqueueNDRangeKernel(device.commandQueue_gpu, device.normal_loop_filter_MBV, 1, NULL, device.gpu_work_items_per_dim, device.gpu_work_group_size_per_dim, 0, NULL, NULL);
			if (device.state_gpu != 0)  printf(">error while deblocking : %d", device.state_gpu);
			device.state_gpu = clFinish(device.commandQueue_gpu);
			if (device.state_gpu != 0)  printf(">error while deblocking : %d", device.state_gpu);
		}
	}
	else
	{
		device.state_gpu = clSetKernelArg(device.simple_loop_filter_MBH, 2, sizeof(int32_t), &video.mbedge_limit);
		device.state_gpu = clSetKernelArg(device.simple_loop_filter_MBH, 3, sizeof(int32_t), &video.sub_bedge_limit);
		device.state_gpu = clSetKernelArg(device.simple_loop_filter_MBV, 2, sizeof(int32_t), &video.mbedge_limit);
		device.state_gpu = clSetKernelArg(device.simple_loop_filter_MBV, 3, sizeof(int32_t), &video.sub_bedge_limit);
		int32_t mb_col;
		for (mb_col = 0; mb_col < video.mb_width; ++mb_col) 
		{
			device.state_gpu = clSetKernelArg(device.simple_loop_filter_MBH, 4, sizeof(int32_t), &mb_col);
			device.state_gpu = clSetKernelArg(device.simple_loop_filter_MBV, 4, sizeof(int32_t), &mb_col);
			
			device.gpu_work_items_per_dim[0] = video.mb_height*16;
			device.gpu_work_group_size_per_dim[0] = 64; 
			device.state_gpu = clEnqueueNDRangeKernel(device.commandQueue_gpu, device.simple_loop_filter_MBH, 1, NULL, device.gpu_work_items_per_dim, device.gpu_work_group_size_per_dim, 0, NULL, NULL);
			if (device.state_gpu != 0)  printf(">error while deblocking : %d", device.state_gpu);
			device.state_gpu = clFinish(device.commandQueue_gpu);
			if (device.state_gpu != 0)  printf(">error while deblocking : %d", device.state_gpu);

			device.gpu_work_items_per_dim[0] = video.mb_height*4;
			device.state_gpu = clEnqueueNDRangeKernel(device.commandQueue_gpu, device.simple_loop_filter_MBV, 1, NULL, device.gpu_work_items_per_dim, device.gpu_work_group_size_per_dim, 0, NULL, NULL);
			if (device.state_gpu != 0)  printf(">error while deblocking : %d", device.state_gpu);
			device.state_gpu = clFinish(device.commandQueue_gpu);
			if (device.state_gpu != 0)  printf(">error while deblocking : %d", device.state_gpu);
		}
	}
	t.loop_filter += clock() - t.start;
	return;
}

void interpolate()
{
	t.start = clock();

	// interpolate last frame buffer for luma(Y) with bicubic filter
	// both HORIZONTALLY
	device.gpu_work_items_per_dim[0] = video.wrk_height;
	device.state_gpu = clEnqueueNDRangeKernel(device.commandQueue_gpu, device.luma_interpolate_Hx4, 1, NULL, device.gpu_work_items_per_dim, NULL, 0, NULL, NULL);
	if (device.state_gpu != 0)  printf(">error when interpolating luma horizontally : %d", device.state_gpu);
	device.state_gpu = clFinish(device.commandQueue_gpu);
	device.gpu_work_items_per_dim[0] = video.wrk_height/2;
	device.state_gpu = clSetKernelArg(device.chroma_interpolate_Hx8, 0, sizeof(cl_mem), &device.reconstructed_frame_U);
	device.state_gpu = clSetKernelArg(device.chroma_interpolate_Hx8, 1, sizeof(cl_mem), &device.last_frame_U);
	device.state_gpu = clEnqueueNDRangeKernel(device.commandQueue_gpu, device.chroma_interpolate_Hx8, 1, NULL, device.gpu_work_items_per_dim, NULL, 0, NULL, NULL);
	if (device.state_gpu != 0)  printf(">error when interpolating U-chroma horizontaly : %d", device.state_gpu);
	device.state_gpu = clFinish(device.commandQueue_gpu);
	device.state_gpu = clSetKernelArg(device.chroma_interpolate_Hx8, 0, sizeof(cl_mem), &device.reconstructed_frame_V);
	device.state_gpu = clSetKernelArg(device.chroma_interpolate_Hx8, 1, sizeof(cl_mem), &device.last_frame_V);
	device.state_gpu = clEnqueueNDRangeKernel(device.commandQueue_gpu, device.chroma_interpolate_Hx8, 1, NULL, device.gpu_work_items_per_dim, NULL, 0, NULL, NULL);
	if (device.state_gpu != 0)  printf(">error when interpolating V-chroma horizontaly : %d", device.state_gpu);
	device.state_gpu = clFinish(device.commandQueue_gpu);
	// and VERTICALLY
	device.gpu_work_items_per_dim[0] = video.wrk_width;
	device.state_gpu = clEnqueueNDRangeKernel(device.commandQueue_gpu, device.luma_interpolate_Vx4, 1, NULL, device.gpu_work_items_per_dim, NULL, 0, NULL, NULL);
	if (device.state_gpu != 0)  printf(">error when interpolating luma vertically : %d", device.state_gpu);
	device.state_gpu = clFinish(device.commandQueue_gpu);
	device.state_gpu = clSetKernelArg(device.chroma_interpolate_Vx8, 0, sizeof(cl_mem), &device.last_frame_U);
	device.state_gpu = clEnqueueNDRangeKernel(device.commandQueue_gpu, device.chroma_interpolate_Vx8, 1, NULL, device.gpu_work_items_per_dim, NULL, 0, NULL, NULL);
	if (device.state_gpu != 0)  printf(">error when interpolating U-chroma vertically : %d", device.state_gpu);
	device.state_gpu = clFinish(device.commandQueue_gpu);
	device.state_gpu = clSetKernelArg(device.chroma_interpolate_Vx8, 0, sizeof(cl_mem), &device.last_frame_V);
	device.state_gpu = clEnqueueNDRangeKernel(device.commandQueue_gpu, device.chroma_interpolate_Vx8, 1, NULL, device.gpu_work_items_per_dim, NULL, 0, NULL, NULL);
	if (device.state_gpu != 0)  printf(">error when interpolating V-chroma vertically : %d", device.state_gpu);
	device.state_gpu = clFinish(device.commandQueue_gpu);

	t.interpolate += clock() - t.start;
}

void correct_quant_indexes()
{
	video.quantizer_index_y_dc_p_l = video.quantizer_index_y_dc_p_c;
	video.quantizer_index_y_ac_p_l = video.quantizer_index_y_ac_p_c;
	video.quantizer_index_y2_dc_p_l = video.quantizer_index_y2_dc_p_c;
	video.quantizer_index_y2_ac_p_l = video.quantizer_index_y2_ac_p_c;
	video.quantizer_index_uv_dc_p_l = video.quantizer_index_uv_dc_p_c;
	video.quantizer_index_uv_ac_p_l = video.quantizer_index_uv_ac_p_c;

	video.quantizer_index_y_dc_p_c = video.quantizer_index_y_dc_p;
	video.quantizer_index_y_ac_p_c = video.quantizer_index_y_ac_p;
	video.quantizer_index_y2_dc_p_c = video.quantizer_index_y2_dc_p;
	video.quantizer_index_y2_ac_p_c = video.quantizer_index_y2_ac_p;
	video.quantizer_index_uv_dc_p_c = video.quantizer_index_uv_dc_p;
	video.quantizer_index_uv_ac_p_c = video.quantizer_index_uv_ac_p;

	// smoothing quant values in 3 steps frop P-frame to I-frame
	if (frames.frames_until_key < 4)
	{
		video.quantizer_index_y_dc_p_c = (video.quantizer_index_y_dc_p + video.quantizer_index_y_dc_i*3 + 2)/4;
		video.quantizer_index_y_ac_p_c = (video.quantizer_index_y_ac_p + video.quantizer_index_y_ac_i*3 + 2)/4;
		video.quantizer_index_y2_dc_p_c = (video.quantizer_index_y2_dc_p + video.quantizer_index_y2_dc_i*3 + 2)/4;
		video.quantizer_index_y2_ac_p_c = (video.quantizer_index_y2_ac_p + video.quantizer_index_y2_ac_i*3 + 2)/4;
		video.quantizer_index_uv_dc_p_c = (video.quantizer_index_uv_dc_p + video.quantizer_index_uv_dc_i*3 + 2)/4;
		video.quantizer_index_uv_ac_p_c = (video.quantizer_index_uv_ac_p + video.quantizer_index_uv_ac_i*3 + 2)/4;
	} 
	else if (frames.frames_until_key < 16)
	{
		video.quantizer_index_y_dc_p_c = (video.quantizer_index_y_dc_p + video.quantizer_index_y_dc_i + 1)/2;
		video.quantizer_index_y_ac_p_c = (video.quantizer_index_y_ac_p + video.quantizer_index_y_ac_i + 1)/2;
		video.quantizer_index_y2_dc_p_c = (video.quantizer_index_y2_dc_p + video.quantizer_index_y2_dc_i + 1)/2;
		video.quantizer_index_y2_ac_p_c = (video.quantizer_index_y2_ac_p + video.quantizer_index_y2_ac_i + 1)/2;
		video.quantizer_index_uv_dc_p_c = (video.quantizer_index_uv_dc_p + video.quantizer_index_uv_dc_i + 1)/2;
		video.quantizer_index_uv_ac_p_c = (video.quantizer_index_uv_ac_p + video.quantizer_index_uv_ac_i + 1)/2;
	} 
	else if (frames.frames_until_key < 36)
	{
		video.quantizer_index_y_dc_p_c = (video.quantizer_index_y_dc_p*3 + video.quantizer_index_y_dc_i + 2)/4;
		video.quantizer_index_y_ac_p_c = (video.quantizer_index_y_ac_p*3 + video.quantizer_index_y_ac_i + 2)/4;
		video.quantizer_index_y2_dc_p_c = (video.quantizer_index_y2_dc_p*3 + video.quantizer_index_y2_dc_i + 2)/4;
		video.quantizer_index_y2_ac_p_c = (video.quantizer_index_y2_ac_p*3 + video.quantizer_index_y2_ac_i + 2)/4;
		video.quantizer_index_uv_dc_p_c = (video.quantizer_index_uv_dc_p*3 + video.quantizer_index_uv_dc_i + 2)/4;
		video.quantizer_index_uv_ac_p_c = (video.quantizer_index_uv_ac_p*3 + video.quantizer_index_uv_ac_i + 2)/4;
	} 
	else 
	{
		video.quantizer_index_y_dc_p_c = video.quantizer_index_y_dc_p;
		video.quantizer_index_y_ac_p_c = video.quantizer_index_y_ac_p;
		video.quantizer_index_y2_dc_p_c = video.quantizer_index_y2_dc_p;
		video.quantizer_index_y2_ac_p_c = video.quantizer_index_y2_ac_p;
		video.quantizer_index_uv_dc_p_c = video.quantizer_index_uv_dc_p;
		video.quantizer_index_uv_ac_p_c = video.quantizer_index_uv_ac_p;
	} 

	//check for delta for chroma (because they are lowered as much as possible)
	video.quantizer_index_uv_dc_p_c = ((video.quantizer_index_y_ac_p_c - video.quantizer_index_uv_dc_p_c) > 15) 
										? video.quantizer_index_y_ac_p_c - 15 
										: video.quantizer_index_uv_dc_p_c;
	video.quantizer_index_uv_ac_p_c = ((video.quantizer_index_y_ac_p_c - video.quantizer_index_uv_ac_p_c) > 15) 
										? video.quantizer_index_y_ac_p_c - 15 
										: video.quantizer_index_uv_ac_p_c;
	return;
}

void inter_transform()
{
	clFinish(device.commandQueue_gpu);

	if (video.loop_filter_level > 0) 
		do_loop_filter();

	interpolate();

	clock_t start_i = clock();

	// set quant for inter-frames
    video.quantizer_y_dc_p = vp8_dc_qlookup[video.quantizer_index_y_dc_p_c];
    video.quantizer_y_ac_p = vp8_ac_qlookup[video.quantizer_index_y_ac_p_c];
	video.quantizer_y2_dc_p = (vp8_dc_qlookup[video.quantizer_index_y2_dc_p_c]) << 1; // *2
    video.quantizer_y2_ac_p = 31 * (vp8_ac_qlookup[video.quantizer_index_y2_ac_p_c]) / 20; // *155/100
    video.quantizer_uv_dc_p = vp8_dc_qlookup[video.quantizer_index_uv_dc_p_c];
    video.quantizer_uv_ac_p = vp8_ac_qlookup[video.quantizer_index_uv_ac_p_c];

	if (video.quantizer_y2_ac_p < 8)
        video.quantizer_y2_ac_p = 8;
    if (video.quantizer_uv_dc_p > 132)
        video.quantizer_uv_dc_p = 132;
	
	device.state_gpu = clSetKernelArg(device.luma_search, 8, sizeof(int32_t), &video.quantizer_y_dc_p);
	device.state_gpu = clSetKernelArg(device.luma_search, 9, sizeof(int32_t), &video.quantizer_y_ac_p);

	device.state_gpu = clSetKernelArg(device.luma_transform, 6, sizeof(int32_t), &video.quantizer_y_dc_p);
	device.state_gpu = clSetKernelArg(device.luma_transform, 7, sizeof(int32_t), &video.quantizer_y_ac_p);

	device.state_gpu = clSetKernelArg(device.chroma_transform, 7, sizeof(int32_t), &video.quantizer_uv_dc_p);
	device.state_gpu = clSetKernelArg(device.chroma_transform, 8, sizeof(int32_t), &video.quantizer_uv_ac_p);

	int32_t blocks_to_be_done = video.mb_count*4;
	device.gpu_work_group_size_per_dim[0] = 256; //fixed as kernel attribute
	blocks_to_be_done += ((blocks_to_be_done % device.gpu_work_group_size_per_dim[0]) > 0) ? (device.gpu_work_group_size_per_dim[0] - (blocks_to_be_done % device.gpu_work_group_size_per_dim[0])) : 0;
	int32_t blocks_done;
	int32_t first_block_offset;
	
	for (blocks_done = 0; blocks_done < blocks_to_be_done; /*counter increased inside cycle*/)
    {	
		first_block_offset = blocks_done;
		device.state_gpu = clSetKernelArg(device.luma_search, 5, sizeof(int32_t), &first_block_offset);
        if ((size_t)(blocks_to_be_done - blocks_done) > device.gpu_work_items_limit)
        {
            device.gpu_work_items_per_dim[0] = device.gpu_work_items_limit;
            blocks_done += device.gpu_work_items_limit;
        }
        else
        {
            device.gpu_work_items_per_dim[0] = blocks_to_be_done - blocks_done;
            blocks_done += blocks_to_be_done - blocks_done;
        }
		device.state_gpu = clFinish(device.commandQueue_gpu);
		device.state_gpu = clEnqueueNDRangeKernel(device.commandQueue_gpu, device.luma_search, 1, NULL, device.gpu_work_items_per_dim, device.gpu_work_group_size_per_dim, 0, NULL, NULL);
		device.state_gpu = clFlush(device.commandQueue_gpu);
    }

	entropy_encode(); // entropy coding of previous frame
	
	blocks_to_be_done = video.mb_count*4;
	for (blocks_done = 0; blocks_done < blocks_to_be_done; /*counter increased inside cycle*/)
    {	
		first_block_offset = blocks_done;
		device.state_gpu = clSetKernelArg(device.luma_transform, 5, sizeof(int32_t), &first_block_offset);
        if ((size_t)(blocks_to_be_done - blocks_done) > device.gpu_work_items_limit)
        {
            device.gpu_work_items_per_dim[0] = device.gpu_work_items_limit;
            blocks_done += device.gpu_work_items_limit;
        }
        else
        {
            device.gpu_work_items_per_dim[0] = blocks_to_be_done - blocks_done;
            blocks_done += blocks_to_be_done - blocks_done;
        }
		device.state_gpu = clFinish(device.commandQueue_gpu);
		device.state_gpu = clEnqueueNDRangeKernel(device.commandQueue_gpu, device.luma_transform, 1, NULL, device.gpu_work_items_per_dim, NULL, 0, NULL, NULL);
		device.state_gpu = clFlush(device.commandQueue_gpu);
    }

	blocks_to_be_done = video.mb_count<<2;
	int block_place = 16;
	device.state_gpu = clSetKernelArg(device.chroma_transform, 9, sizeof(int32_t), &block_place);

	for (blocks_done = 0; blocks_done < blocks_to_be_done; /*counter increased inside cycle*/)
    {
		first_block_offset = blocks_done;
		device.state_gpu = clSetKernelArg(device.chroma_transform, 6, sizeof(int32_t), &first_block_offset);
		if ((size_t)(blocks_to_be_done - blocks_done) > (device.gpu_work_items_limit)) // because two planes are
        {
            device.gpu_work_items_per_dim[0] = device.gpu_work_items_limit;
            blocks_done += device.gpu_work_items_limit;
        }
        else
        {
            device.gpu_work_items_per_dim[0] = blocks_to_be_done - blocks_done;
            blocks_done += blocks_to_be_done - blocks_done;
        }
		device.state_gpu = clFinish(device.commandQueue_gpu); // UV always should wait for Y vectors
		device.state_gpu = clSetKernelArg(device.chroma_transform, 0, sizeof(cl_mem), &device.current_frame_U);
		device.state_gpu = clSetKernelArg(device.chroma_transform, 1, sizeof(cl_mem), &device.last_frame_U);
		device.state_gpu = clSetKernelArg(device.chroma_transform, 2, sizeof(cl_mem), &device.reconstructed_frame_U);
		device.state_gpu = clEnqueueNDRangeKernel(device.commandQueue_gpu, device.chroma_transform, 1, NULL, device.gpu_work_items_per_dim, NULL, 0, NULL, NULL);
		if (device.state_gpu != 0)  printf(">error when transforming chroma blocks : %d", device.state_gpu);
		device.state_gpu = clFlush(device.commandQueue_gpu);
		if (device.state_gpu != 0)  printf(">error when transforming chroma blocks : %d", device.state_gpu);
	}	

	block_place = 20;
	device.state_gpu = clSetKernelArg(device.chroma_transform, 9, sizeof(int32_t), &block_place);

	for (blocks_done = 0; blocks_done < blocks_to_be_done; /*counter increased inside cycle*/)
    {
		first_block_offset = blocks_done;
		clSetKernelArg(device.chroma_transform, 6, sizeof(int32_t), &first_block_offset);
		if ((size_t)(blocks_to_be_done - blocks_done) > (device.gpu_work_items_limit)) // because two planes are
        {
            device.gpu_work_items_per_dim[0] = device.gpu_work_items_limit;
            blocks_done += device.gpu_work_items_limit;
        }
        else
        {
            device.gpu_work_items_per_dim[0] = blocks_to_be_done - blocks_done;
            blocks_done += blocks_to_be_done - blocks_done;
        }
		device.state_gpu = clFinish(device.commandQueue_gpu);
		device.state_gpu = clSetKernelArg(device.chroma_transform, 0, sizeof(cl_mem), &device.current_frame_V);
		device.state_gpu = clSetKernelArg(device.chroma_transform, 1, sizeof(cl_mem), &device.last_frame_V);
		device.state_gpu = clSetKernelArg(device.chroma_transform, 2, sizeof(cl_mem), &device.reconstructed_frame_V);
		device.state_gpu = device.state_gpu = clEnqueueNDRangeKernel(device.commandQueue_gpu, device.chroma_transform, 1, NULL, device.gpu_work_items_per_dim, NULL, 0, NULL, NULL);
		if (device.state_gpu != 0)  printf(">error when transforming chroma blocks : %d", device.state_gpu);
		device.state_gpu = clFlush(device.commandQueue_gpu);
		if (device.state_gpu != 0)  printf(">error when transforming chroma blocks : %d", device.state_gpu);
	}	

	t.inter_transform += clock() - start_i;

	blocks_to_be_done = video.mb_count*4;
	device.gpu_work_group_size_per_dim[0] = 256; //fixed as kernel attribute
	blocks_to_be_done += ((blocks_to_be_done % device.gpu_work_group_size_per_dim[0]) > 0) ? (device.gpu_work_group_size_per_dim[0] - (blocks_to_be_done % device.gpu_work_group_size_per_dim[0])) : 0;
	
	//count SSIM
	device.state_gpu = clFinish(device.commandQueue_gpu);
	device.gpu_work_group_size_per_dim[0] = 64;
	device.gpu_work_items_per_dim[0] = video.mb_count;
	device.gpu_work_items_per_dim[0] += ((video.mb_count % device.gpu_work_group_size_per_dim[0]) > 0) ?
										device.gpu_work_group_size_per_dim[0] - (video.mb_count % device.gpu_work_group_size_per_dim[0]) :
										0;
	device.state_gpu = clEnqueueNDRangeKernel(device.commandQueue_gpu, device.count_SSIM, 1, NULL, device.gpu_work_items_per_dim, device.gpu_work_group_size_per_dim, 0, NULL, NULL);
	device.state_gpu = clFlush(device.commandQueue_gpu);

    return;
}

void check_SSIM()
{
	device.state_gpu = clEnqueueReadBuffer(device.commandQueue_gpu, device.reconstructed_frame_Y ,CL_TRUE, 0, video.wrk_frame_size_luma, frames.reconstructed_Y, 0, NULL, NULL);
	device.state_gpu = clEnqueueReadBuffer(device.commandQueue_gpu, device.reconstructed_frame_U ,CL_TRUE, 0, video.wrk_frame_size_chroma, frames.reconstructed_U, 0, NULL, NULL);
	device.state_gpu = clEnqueueReadBuffer(device.commandQueue_gpu, device.reconstructed_frame_V ,CL_TRUE, 0, video.wrk_frame_size_chroma, frames.reconstructed_V, 0, NULL, NULL);

	frames.new_SSIM = 0;
	float min = 2;
	float max = -2;
	int mb_num;
	frames.replaced = 0;
	for (mb_num = 0; mb_num < video.mb_count; ++mb_num)
	{
		if (frames.transformed_blocks[mb_num].SSIM < 0.7f) 
			frames.e_data[mb_num].is_inter_mb = test_inter_on_intra(mb_num);
		frames.new_SSIM += frames.transformed_blocks[mb_num].SSIM;
		min = (frames.transformed_blocks[mb_num].SSIM < min) ? frames.transformed_blocks[mb_num].SSIM : min;
		max = (frames.transformed_blocks[mb_num].SSIM > max) ? frames.transformed_blocks[mb_num].SSIM : max;
	}
	if (frames.replaced > 0)
	{
		device.state_gpu = clEnqueueWriteBuffer(device.commandQueue_gpu, device.reconstructed_frame_Y, CL_TRUE, 0, video.wrk_frame_size_luma, frames.reconstructed_Y, 0, NULL, NULL);
		device.state_gpu = clEnqueueWriteBuffer(device.commandQueue_gpu, device.reconstructed_frame_U, CL_TRUE, 0, video.wrk_frame_size_chroma, frames.reconstructed_U, 0, NULL, NULL);
		device.state_gpu = clEnqueueWriteBuffer(device.commandQueue_gpu, device.reconstructed_frame_V, CL_TRUE, 0, video.wrk_frame_size_chroma, frames.reconstructed_V, 0, NULL, NULL);
	}

	frames.new_SSIM /= (float)video.mb_count;
	printf("Fr %d(P)=> Avg SSIM=%f; MinSSIM=%f; MaxSSIM=%f I:%d\n", frames.frame_number, frames.new_SSIM, min, max, frames.replaced);
	return;
}

int scene_change()
{
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
	return ((Udiff > 7) || (Vdiff > 7) || (Udiff+Vdiff > 10));
}

void finalize();

int main(int argc, char *argv[])
{
	int32_t mb_num;
	printf("\n");
	t.init = clock();
	t.inter_transform = 0;
	t.intra_transform = 0;
	t.count_probs = 0;
	t.bool_encode_header = 0;
	t.bool_encode_coeffs = 0;
	t.read = 0;
	t.write = 0;
	t.interpolate = 0;
	t.loop_filter = 0;

	error_file.path = ERRORPATH;
	if (ParseArgs(argc, argv) < 0) 
	{
		printf("\npress any key to quit\n");
		getch();
		return -1;
	}
	
    OpenYUV420FileAndParseHeader();
    
	printf("initialization started;\n");
    init_all();
	if ((device.state_cpu != 0) || (device.state_gpu != 0)) {
		printf("\npress any key to quit\n");
		getch();
		return -1; 
	}
	printf("initialization complete;\n");
	t.all = clock();
	t.init = t.all - t.init;

    frames.current_is_key_frame = 1;
    frames.frame_number = 0;
	if (!get_yuv420_frame()) {  // reads frame to host memory
		printf("first frame is incomplete;\n");
	}
	
	write_output_header();

    intra_transform();
	if (video.GOP_size > 1) {
		device.state_gpu = clEnqueueWriteBuffer(device.commandQueue_gpu, device.reconstructed_frame_Y, CL_TRUE, 0, video.wrk_frame_size_luma, frames.reconstructed_Y, 0, NULL, NULL);
		device.state_gpu = clEnqueueWriteBuffer(device.commandQueue_gpu, device.reconstructed_frame_U, CL_TRUE, 0, video.wrk_frame_size_chroma, frames.reconstructed_U, 0, NULL, NULL);
		device.state_gpu = clEnqueueWriteBuffer(device.commandQueue_gpu, device.reconstructed_frame_V, CL_TRUE, 0, video.wrk_frame_size_chroma, frames.reconstructed_V, 0, NULL, NULL);
	}
			
    ++frames.frame_number;
    while (get_yuv420_frame() > 0)
    {
		device.state_cpu = clEnqueueWriteBuffer(device.commandQueue_cpu, device.transformed_blocks_cpu, CL_TRUE, 0, video.mb_count*sizeof(macroblock), frames.transformed_blocks, 0, NULL, NULL); 
		frames.prev_is_key_frame = frames.current_is_key_frame; // for entropy coding of previous frame;
		--frames.frames_until_key;
		frames.current_is_key_frame = (frames.frames_until_key < 1);
        if (frames.current_is_key_frame)
        {
			entropy_encode(); // entropy coding of previous frame
            intra_transform(); 
			if (video.GOP_size > 1) {
				device.state_gpu = clEnqueueWriteBuffer(device.commandQueue_gpu, device.reconstructed_frame_Y, CL_TRUE, 0, video.wrk_frame_size_luma, frames.reconstructed_Y, 0, NULL, NULL);
				device.state_gpu = clEnqueueWriteBuffer(device.commandQueue_gpu, device.reconstructed_frame_U, CL_TRUE, 0, video.wrk_frame_size_chroma, frames.reconstructed_U, 0, NULL, NULL);
				device.state_gpu = clEnqueueWriteBuffer(device.commandQueue_gpu, device.reconstructed_frame_V, CL_TRUE, 0, video.wrk_frame_size_chroma, frames.reconstructed_V, 0, NULL, NULL);
			}        
		}
        else if (scene_change()) 
		{
			entropy_encode();
			intra_transform();
			device.state_gpu = clEnqueueWriteBuffer(device.commandQueue_gpu, device.reconstructed_frame_Y, CL_TRUE, 0, video.wrk_frame_size_luma, frames.reconstructed_Y, 0, NULL, NULL);
			device.state_gpu = clEnqueueWriteBuffer(device.commandQueue_gpu, device.reconstructed_frame_U, CL_TRUE, 0, video.wrk_frame_size_chroma, frames.reconstructed_U, 0, NULL, NULL);
			device.state_gpu = clEnqueueWriteBuffer(device.commandQueue_gpu, device.reconstructed_frame_V, CL_TRUE, 0, video.wrk_frame_size_chroma, frames.reconstructed_V, 0, NULL, NULL);
			frames.current_is_key_frame = 1;
			printf("key frame FORCED by chroma color difference!\n");
		} 
		else
        {
			device.state_gpu = clEnqueueWriteBuffer(device.commandQueue_gpu, device.current_frame_Y, CL_TRUE, 0, video.wrk_frame_size_luma, frames.current_Y, 0, NULL, NULL);
			device.state_gpu = clEnqueueWriteBuffer(device.commandQueue_gpu, device.current_frame_U, CL_TRUE, 0, video.wrk_frame_size_chroma, frames.current_U, 0, NULL, NULL);
			device.state_gpu = clEnqueueWriteBuffer(device.commandQueue_gpu, device.current_frame_V, CL_TRUE, 0, video.wrk_frame_size_chroma, frames.current_V, 0, NULL, NULL);
            
			correct_quant_indexes();
			inter_transform(); 
			device.state_gpu = clFinish(device.commandQueue_gpu);
			// copy transformed_blocks to host
			device.state_gpu = clEnqueueReadBuffer(device.commandQueue_gpu, device.transformed_blocks_gpu ,CL_TRUE, 0, video.mb_count*sizeof(macroblock), frames.transformed_blocks, 0, NULL, NULL);
			
			for(mb_num = 0; mb_num < video.mb_count; ++mb_num)
				frames.e_data[mb_num].is_inter_mb = 1;

			check_SSIM();
			if (frames.replaced > (video.mb_count/4))
			{
				// redo as intra
				intra_transform();
				device.state_gpu = clEnqueueWriteBuffer(device.commandQueue_gpu, device.reconstructed_frame_Y, CL_TRUE, 0, video.wrk_frame_size_luma, frames.reconstructed_Y, 0, NULL, NULL);
				device.state_gpu = clEnqueueWriteBuffer(device.commandQueue_gpu, device.reconstructed_frame_U, CL_TRUE, 0, video.wrk_frame_size_chroma, frames.reconstructed_U, 0, NULL, NULL);
				device.state_gpu = clEnqueueWriteBuffer(device.commandQueue_gpu, device.reconstructed_frame_V, CL_TRUE, 0, video.wrk_frame_size_chroma, frames.reconstructed_V, 0, NULL, NULL);
				frames.current_is_key_frame = 1;
				printf("key frame FORCED by replaced blocks!\n");
			}
			
			// we will move reconstructed buffers to last while interpolating later in inter_transform();
			device.state_gpu = clFinish(device.commandQueue_gpu);
        }
		if ((frames.frame_number % video.framerate) == 0) printf("second %d encoded\n", frames.frame_number/video.framerate);
		gather_frame();
		write_output_file();
        ++frames.frame_number;
    }
	device.state_cpu = clEnqueueWriteBuffer(device.commandQueue_cpu, device.transformed_blocks_cpu, CL_TRUE, 0, video.mb_count*sizeof(macroblock), frames.transformed_blocks, 0, NULL, NULL);
	frames.prev_is_key_frame = frames.current_is_key_frame;
	device.state_gpu = clFinish(device.commandQueue_cpu);
	entropy_encode();
	gather_frame();
	write_output_file();
	write_output_header();

	t.all = clock() - t.all;

	finalize();

	printf("frames done: %d.\n", frames.frame_number);
	printf("init time :%f seconds\n", ((float)t.init)/CLOCKS_PER_SEC);
	printf("encoding time :%f seconds\n", ((float)t.all)/CLOCKS_PER_SEC);
	printf("read input file time part :%f seconds\n", ((float)t.read)/CLOCKS_PER_SEC);
	printf("intra transforming time part :%f seconds\n", ((float)t.intra_transform)/CLOCKS_PER_SEC);
	printf("inter transforming time part :%f seconds\n", ((float)t.inter_transform)/CLOCKS_PER_SEC);
	printf("loop filtering time part :%f seconds\n", ((float)t.loop_filter)/CLOCKS_PER_SEC);
	printf("interpolating time part :%f seconds\n", ((float)t.interpolate)/CLOCKS_PER_SEC);
	printf("counting probs for coeffs :%f seconds\n", ((float)t.count_probs)/CLOCKS_PER_SEC);
	printf("header boolean encoding time part :%f seconds\n", ((float)t.bool_encode_header)/CLOCKS_PER_SEC);
	printf("write output file time part :%f seconds\n", ((float)t.write)/CLOCKS_PER_SEC);

	printf("\n FPS: %f.\n", (float)frames.frame_number/(((float)t.all)/CLOCKS_PER_SEC));

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
		clReleaseMemObject(device.reconstructed_frame_U);
		clReleaseMemObject(device.reconstructed_frame_V);
		clReleaseMemObject(device.reconstructed_frame_Y);
		clReleaseMemObject(device.last_frame_U);
		clReleaseMemObject(device.last_frame_V);
		clReleaseMemObject(device.last_frame_Y);
		clReleaseMemObject(device.transformed_blocks_gpu);
		clReleaseKernel(device.luma_search);
		clReleaseKernel(device.luma_transform);
		clReleaseKernel(device.chroma_transform);
		clReleaseCommandQueue(device.commandQueue_gpu);
		clReleaseProgram(device.program_gpu);
		clReleaseContext(device.context_gpu);
		device.device_gpu = (cl_device_id*)malloc(sizeof(cl_device_id));
		free(device.device_gpu);
	}

	clReleaseKernel(device.count_probs);
	clReleaseKernel(device.encode_coefficients);
	clReleaseKernel(device.num_div_denom);
	clReleaseCommandQueue(device.commandQueue_cpu);
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
}
