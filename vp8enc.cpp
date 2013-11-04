//a way to define path to OpenCL.lib
#pragma comment(lib,"C:\\Program Files (x86)\\AMD APP\\lib\\x86\\OpenCL.lib")

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

#include "IO.h"
#include "init.h"

typedef enum
{
	DC_PRED, V_PRED, H_PRED, TM_PRED, B_PRED, 
	num_uv_modes = B_PRED, num_ymodes
} intra_mbmode;
typedef enum
{
	B_DC_PRED, B_TM_PRED, B_VE_PRED, B_HE_PRED, B_LD_PRED, B_RD_PRED, B_VR_PRED, B_VL_PRED, B_HD_PRED, B_HU_PRED, 
	num_intra_bmodes
} intra_bmode;

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
	clSetKernelArg(device.count_probs, 7, sizeof(int32_t), &frames.current_is_key_frame);	
	clFinish(device.commandQueue_cpu);
	clEnqueueNDRangeKernel(device.commandQueue_cpu, device.count_probs, 1, NULL, device.cpu_work_items_per_dim, device.cpu_work_group_size_per_dim, 0, NULL, NULL);
	clFinish(device.commandQueue_cpu); 

	// just dividing nums by denoms and getting probability of bit being ZERO
	clEnqueueNDRangeKernel(device.commandQueue_cpu, device.num_div_denom, 1, NULL, device.cpu_work_items_per_dim, device.cpu_work_group_size_per_dim, 0, NULL, NULL);
	clFinish(device.commandQueue_cpu); // Blocking, we need prob-values before encoding coeffs and header

	// read calculated values
	clEnqueueReadBuffer(device.commandQueue_cpu, device.coeff_probs ,CL_TRUE, 0, 11*3*8*4*sizeof(uint32_t), frames.new_probs, 0, NULL, NULL);
	clEnqueueReadBuffer(device.commandQueue_cpu, device.coeff_probs_denom ,CL_TRUE, 0, 11*3*8*4*sizeof(uint32_t), frames.new_probs_denom, 0, NULL, NULL);
	{ int i,j,k,l;
	for (i = 0; i < 4; ++i)
		for (j = 0; j < 8; ++j)
			for (k = 0; k < 3; ++k)
				for (l = 0; l < 11; ++l)
					if (frames.new_probs_denom[i][j][k][l] < 2) // this situation never happened (no bit encoded with this context)
						frames.new_probs[i][j][k][l] = k_default_coeff_probs[i][j][k][l];
	}
	device.state_gpu = clEnqueueWriteBuffer(device.commandQueue_cpu, device.coeff_probs, CL_TRUE, 0, 11*3*8*4*sizeof(uint32_t), frames.new_probs, 0, NULL, NULL);

	// start of encoding coefficients 
	clSetKernelArg(device.encode_coefficients, 9, sizeof(int32_t), &frames.current_is_key_frame);
	device.state_cpu = clSetKernelArg(device.encode_coefficients, 11, sizeof(int32_t), &frames.skip_prob);
	clEnqueueNDRangeKernel(device.commandQueue_cpu, device.encode_coefficients, 1, NULL, device.cpu_work_items_per_dim, device.cpu_work_group_size_per_dim, 0, NULL, NULL);
	clFlush(device.commandQueue_cpu); // we don't need result until gather_frame(), so no block now

	// encoding header is done as a part of HOST code placed in entropy_host.c[pp]|entropy_host.h
	encode_header(frames.encoded_frame); 

    return;
}

void iDCT4x4(int16_t *input, uint8_t *output, uint8_t* predictor, int32_t dc_q, int32_t ac_q)
{
    int i;
    int a1, b1, c1, d1;
	int ip0, ip4, ip8, ip12, q;
    int16_t tmp_block[16];
    short *ip=input;
    short *tp=tmp_block;
    int temp1, temp2; 

	q = dc_q;
    for (i = 0; i < 4; ++i)
    {
		ip0 = ip[0]*q;
		q = ac_q;
		ip4 = ip[4]*q;
		ip8 = ip[8]*q;
		ip12 = ip[12]*q;

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
        tp[0] = ((a1 + d1 + 4) >> 3) + predictor[0];
		op[0] = (uint8_t)((tp[0] > 255) ? 255 : ((tp[0] < 0) ? 0 : tp[0] ));
		tp[3] = ((a1 - d1 + 4) >> 3) + predictor[3];
        op[3] = (uint8_t)((tp[3] > 255) ? 255 : ((tp[3] < 0) ? 0 : tp[3] ));
        tp[1] = ((b1 + c1 + 4) >> 3) + predictor[1];
        op[1] = (uint8_t)((tp[1] > 255) ? 255 : ((tp[1] < 0) ? 0 : tp[1] ));
        tp[2] = ((b1 - c1 + 4) >> 3) + predictor[2];
        op[2] = (uint8_t)((tp[2] > 255) ? 255 : ((tp[2] < 0) ? 0 : tp[2] ));

        op+=4;
		tp+=4;
		predictor += 4;
    }

	return;
}


static void DCT4x4(int16_t *input, int16_t *output)
{
    // input - pointer to start of block in raw frame. I-line of the block will be input + I*width
    // output - pointer to encoded_macroblock.block[i] data.
    int32_t i;
    int32_t a1, b1, c1, d1;
    int16_t *ip = input;
    int16_t *op = output;

    for (i = 0; i < 4; i++)
    {
        a1 = ((ip[0] + ip[3])<<3);
        b1 = ((ip[1] + ip[2])<<3);
        c1 = ((ip[1] - ip[2])<<3);
        d1 = ((ip[0] - ip[3])<<3);

        op[0] = (int16_t)(a1 + b1);
        op[2] = (int16_t)(a1 - b1);

        op[1] = (int16_t)((c1 * 2217 + d1 * 5352 +  14500)>>12);
        op[3] = (int16_t)((d1 * 2217 - c1 * 5352 +   7500)>>12);

        ip += 4; // because in's in the raw frame
        op += 4; // because it's in block-packed order

    }
    op = output;

    for (i = 0; i < 4; i++)
    {
        a1 = op[0] + op[12];
        b1 = op[4] + op[8];
        c1 = op[4] - op[8];
		d1 = op[0] - op[12];

        op[0] = (( a1 + b1 + 7)>>4); // quant using dc_q only first time
        op[8] = (( a1 - b1 + 7)>>4);
        op[4]  = (((c1 * 2217 + d1 * 5352 +  12000)>>16) + (d1!=0));
        op[12] = ((d1 * 2217 - c1 * 5352 +  51000)>>16);

        ++op;
    }
	return;
}

static int weight(int16_t * r) //r - residual to be weighted through WHT
{
    int32_t i;
    int32_t a1, b1, c1, d1;
    int32_t a2, b2, c2, d2;
    int16_t *ip = r;
	int16_t tmp[16];
	int16_t *t = tmp;

    for (i = 0; i < 4; i++)
    {
        // extracting dc coeffs from block-ordered macroblock
        // ip[0] - DCcoef of first mb in line, +16 coeffs and starts second, and so on...
        // ..every i element will be stored with 16*i offset from the start
        a1 = ((int32_t)ip[0] + (int32_t)ip[3]);
        b1 = ((int32_t)ip[1] + (int32_t)ip[2]);
        c1 = ((int32_t)ip[1] - (int32_t)ip[2]);
        d1 = ((int32_t)ip[0] - (int32_t)ip[3]);

        t[0] = a1 + b1;
        t[1] = c1 + d1;
        t[2] = a1 - b1;
        t[3] = d1 - c1;
        ip += 4; // input goes from [25][16] coeffs, so we skip line of 4 blocks (each 16 coeffs)
        t += 4;
    }

	t = tmp;
    for (i = 0; i < 4; i++)
    {
        a1 = t[0] + t[12];
        b1 = t[4] + t[8];
        c1 = t[4] - t[8];
        d1 = t[0] - t[12];

        a2 = a1 + b1;
        b2 = c1 + d1;
        c2 = a1 - b1;
        d2 = d1 - c1;

        a2 += (a2 > 0);
        b2 += (b2 > 0);
        c2 += (c2 > 0);
        d2 += (d2 > 0);

        t[0] = ((a2) >> 1);
        t[4] = ((b2) >> 1);
        t[8] = ((c2) >> 1);
        t[12] = ((d2) >> 1);

        ++t;
    }

	for (i = 0; i < 4; i++)
    {
		a1 = (t[0] < 0) ? -t[0] : t[0];
		a1 += (t[0] < 0) ? -t[1] : t[1];
		a1 += (t[0] < 0) ? -t[2] : t[2];
		a1 += (t[0] < 0) ? -t[3] : t[3];

        ++t;
    }

	return a1;
}

void quant4x4(int16_t *coeffs, int32_t dc_q, int32_t ac_q)
{
	// possible opt: val + (val>>16)*(q/2)
	coeffs[0] += (coeffs[0] < 0) ? (-dc_q/2) : (dc_q/2);
	coeffs[1] += (coeffs[1] < 0) ? (-ac_q/2) : (ac_q/2);
	coeffs[2] += (coeffs[2] < 0) ? (-ac_q/2) : (ac_q/2);
	coeffs[3] += (coeffs[3] < 0) ? (-ac_q/2) : (ac_q/2);
	coeffs[4] += (coeffs[4] < 0) ? (-ac_q/2) : (ac_q/2);
	coeffs[5] += (coeffs[5] < 0) ? (-ac_q/2) : (ac_q/2);
	coeffs[6] += (coeffs[6] < 0) ? (-ac_q/2) : (ac_q/2);
	coeffs[7] += (coeffs[7] < 0) ? (-ac_q/2) : (ac_q/2);
	coeffs[8] += (coeffs[8] < 0) ? (-ac_q/2) : (ac_q/2);
	coeffs[9] += (coeffs[9] < 0) ? (-ac_q/2) : (ac_q/2);
	coeffs[10] += (coeffs[10] < 0) ? (-ac_q/2) : (ac_q/2);
	coeffs[11] += (coeffs[10] < 0) ? (-ac_q/2) : (ac_q/2);
	coeffs[12] += (coeffs[12] < 0) ? (-ac_q/2) : (ac_q/2);
	coeffs[13] += (coeffs[13] < 0) ? (-ac_q/2) : (ac_q/2);
	coeffs[14] += (coeffs[14] < 0) ? (-ac_q/2) : (ac_q/2);
	coeffs[15] += (coeffs[15] < 0) ? (-ac_q/2) : (ac_q/2);

	coeffs[0] /= (int16_t)dc_q;
	coeffs[1] /= (int16_t)ac_q;
	coeffs[2] /= (int16_t)ac_q;
	coeffs[3] /= (int16_t)ac_q;
	coeffs[4] /= (int16_t)ac_q;
	coeffs[5] /= (int16_t)ac_q;
	coeffs[6] /= (int16_t)ac_q;
	coeffs[7] /= (int16_t)ac_q;
	coeffs[8] /= (int16_t)ac_q;
	coeffs[9] /= (int16_t)ac_q;
	coeffs[10] /= (int16_t)ac_q;
	coeffs[11] /= (int16_t)ac_q;
	coeffs[12] /= (int16_t)ac_q;
	coeffs[13] /= (int16_t)ac_q;
	coeffs[14] /= (int16_t)ac_q;
	coeffs[15] /= (int16_t)ac_q;
	return;
}



int32_t pick_luma_predictor(uint8_t *original, uint8_t *predictor, int16_t *residual, int16_t *top_pred, int16_t *left_pred, int16_t top_left_pred)
{
	int16_t MinWeight, bmode, val, buf, i, j, W;
	uint8_t pr_tmp[16];
	int16_t res_tmp[16];
// set B_DC_PRED as s start
	bmode = B_DC_PRED;
	val = 4;
	for (i = 0; i < 4; ++i)
		val += top_pred[i] + left_pred[i];
	val >>= 3;
	for (i = 0; i < 4; ++i) 
		for (j = 0; j < 4; ++j) {
		predictor[i*4 + j] = (uint8_t)val; 
		residual[i*4 + j] = (int16_t)original[i*4+j] - (int16_t)predictor[i*4+j];

	}
	MinWeight = weight(residual);
// try B_TM_PRED
	for (i = 0; i < 4; ++i)
		for (j = 0; j < 4; ++j) 
		{
			val = top_pred[j] + left_pred[i] - top_left_pred;
			val = (val < 0) ? 0 : val; val = (val > 255) ? 255 : val;
			pr_tmp[i*4+j] = (uint8_t)val;
			res_tmp[i*4 + j] = (int16_t)original[i*4+j] - (int16_t)pr_tmp[i*4+j];
		}
	W = weight(res_tmp);
	if (W < MinWeight) 
	{
		bmode = B_TM_PRED;
		MinWeight = W;
		for (i = 0; i < 4; ++i)
			for (j = 0; j < 4; ++j) 
			{
				predictor[i*4+j] = pr_tmp[i*4+j];
				residual[i*4+j] = res_tmp[i*4+j];
			}
	}
	if (MinWeight == 0) return bmode;
// try B_VE_PRED
	buf = top_left_pred;
	for (j = 0; j < 4; ++j) 
	{
		val = buf + top_pred[j]*2 + top_pred[j+1] + 2;
		val >>= 2;
		pr_tmp[j] = (uint8_t)val;
		pr_tmp[4+j] = (uint8_t)val;
		pr_tmp[8+j] = (uint8_t)val;
		pr_tmp[12+j] = (uint8_t)val;
		res_tmp[j] = (int16_t)original[j] - (int16_t)pr_tmp[j];
		res_tmp[4 + j] = (int16_t)original[4+j] - (int16_t)pr_tmp[4+j];
		res_tmp[8 + j] = (int16_t)original[8+j] - (int16_t)pr_tmp[8+j];
		res_tmp[12 + j] = (int16_t)original[12+j] - (int16_t)pr_tmp[12+j];
		buf = top_pred[j];
	}
	W = weight(res_tmp);
	if (W < MinWeight) 
	{
		bmode = B_VE_PRED;
		MinWeight = W;
		for (i = 0; i < 4; ++i)
			for (j = 0; j < 4; ++j) 
			{
				predictor[i*4+j] = pr_tmp[i*4+j];
				residual[i*4+j] = res_tmp[i*4+j];
			}
	}
// try B_HE_PRED
	buf = top_left_pred;
	for (i = 0; i < 3; ++i) 
	{
		val = buf + left_pred[i]*2 + left_pred[i+1] + 2;
		val >>= 2;
		pr_tmp[i*4] = (uint8_t)val;
		pr_tmp[i*4+1] = (uint8_t)val;
		pr_tmp[i*4+2] = (uint8_t)val;
		pr_tmp[i*4+3] = (uint8_t)val;
		res_tmp[i*4] = (int16_t)original[i*4] - (int16_t)pr_tmp[i*4];
		res_tmp[i*4 + 1] = (int16_t)original[i*4+1] - (int16_t)pr_tmp[i*4+1];
		res_tmp[i*4 + 2] = (int16_t)original[i*4+2] - (int16_t)pr_tmp[i*4+2];
		res_tmp[i*4 + 3] = (int16_t)original[i*4+3] - (int16_t)pr_tmp[i*4+3];
		buf = left_pred[i];
	} 
	// last row i==3
	val = left_pred[3]*3 + left_pred[i-1] + 2;
	val >>= 2;
	pr_tmp[12] = (uint8_t)val;
	pr_tmp[13] = (uint8_t)val;
	pr_tmp[14] = (uint8_t)val;
	pr_tmp[15] = (uint8_t)val;
	res_tmp[12] = (int16_t)original[12] - (int16_t)pr_tmp[12];
	res_tmp[13] = (int16_t)original[13] - (int16_t)pr_tmp[13];
	res_tmp[14] = (int16_t)original[14] - (int16_t)pr_tmp[14];
	res_tmp[15] = (int16_t)original[15] - (int16_t)pr_tmp[15];
	W = weight(res_tmp);
	if (W < MinWeight) 
	{
		bmode = B_HE_PRED;
		MinWeight = W;
		for (i = 0; i < 4; ++i)
			for (j = 0; j < 4; ++j) 
			{
				predictor[i*4+j] = pr_tmp[i*4+j];
				residual[i*4+j] = res_tmp[i*4+j];
			}
	} 
// try B_LD_PRED
	pr_tmp[0] = (uint8_t)((top_pred[0]+top_pred[1]*2+top_pred[2]+2)>>2);
	pr_tmp[1] = pr_tmp[4] = (uint8_t)((top_pred[1]+top_pred[2]*2+top_pred[3]+2)>>2);
	pr_tmp[2] = pr_tmp[5] = pr_tmp[8] = (uint8_t)((top_pred[2]+top_pred[3]*2+top_pred[4]+2)>>2);
	pr_tmp[3] = pr_tmp[6] = pr_tmp[9] = pr_tmp[12] = (uint8_t)((top_pred[3]+top_pred[4]*2+top_pred[5]+2)>>2);
	pr_tmp[7] = pr_tmp[10] = pr_tmp[13] = (uint8_t)((top_pred[4]+top_pred[5]*2+top_pred[6]+2)>>2);
	pr_tmp[11] = pr_tmp[14] = (uint8_t)((top_pred[5]+top_pred[6]*2+top_pred[7]+2)>>2);
	pr_tmp[15] = (uint8_t)((top_pred[6]+top_pred[7]*3+2)>>2);
	for(i = 0; i < 4; ++i)
		for(j = 0; j < 4; ++j)
			res_tmp[i*4 + j] = (int16_t)original[i*4+j] - (int16_t)pr_tmp[i*4+j];
	W = weight(res_tmp);
	if (W < MinWeight) 
	{
		bmode = B_LD_PRED;
		MinWeight = W;
		for (i = 0; i < 4; ++i)
			for (j = 0; j < 4; ++j) 
			{
				predictor[i*4+j] = pr_tmp[i*4+j];
				residual[i*4+j] = res_tmp[i*4+j];
			}
	} 
// try B_RD_PRED
	pr_tmp[12] = (uint8_t)((left_pred[3] + left_pred[2]*2 + left_pred[1] + 2)>>2);
	pr_tmp[13] = pr_tmp[8] = (uint8_t)((left_pred[2] + left_pred[1]*2 + left_pred[0] + 2)>>2);
	pr_tmp[14] = pr_tmp[9] = pr_tmp[4] = (uint8_t)((left_pred[1] + left_pred[0]*2 + top_left_pred + 2)>>2);
	pr_tmp[15] = pr_tmp[10] = pr_tmp[5] = pr_tmp[0] = (uint8_t)((left_pred[0]+top_left_pred*2 + top_pred[0] + 2)>>2);
	pr_tmp[11] = pr_tmp[6] = pr_tmp[1] = (uint8_t)((top_left_pred + top_pred[0]*2 + top_pred[1] + 2)>>2);
	pr_tmp[7] = pr_tmp[2] = (uint8_t)((top_pred[0] + top_pred[1]*2 + top_pred[2] + 2)>>2);
	pr_tmp[3] = (uint8_t)((top_pred[1] + top_pred[2]*2 + top_pred[3] + 2)>>2);
	for(i = 0; i < 4; ++i)
		for(j = 0; j < 4; ++j)
			res_tmp[i*4 + j] = (int16_t)original[i*4+j] - (int16_t)pr_tmp[i*4+j];
	W = weight(res_tmp);
	if (W < MinWeight) 
	{
		bmode = B_RD_PRED;
		MinWeight = W;
		for (i = 0; i < 4; ++i)
			for (j = 0; j < 4; ++j) 
			{
				predictor[i*4+j] = pr_tmp[i*4+j];
				residual[i*4+j] = res_tmp[i*4+j];
			}
	} 
// try B_VR_PRED
	pr_tmp[12] = (uint8_t)((left_pred[2] + left_pred[1]*2 + left_pred[0] + 2)>>2);
	pr_tmp[8] = (uint8_t)((left_pred[1] + left_pred[0]*2 + top_left_pred + 2)>>2);
	pr_tmp[13] = pr_tmp[4] = (uint8_t)((left_pred[0] + top_left_pred*2 + top_pred[0] + 2)>>2);
	pr_tmp[9] = pr_tmp[0] = (uint8_t)((top_left_pred + top_pred[0] + 1)>>1);
	pr_tmp[14] = pr_tmp[5] = (uint8_t)((top_left_pred + top_pred[0]*2 + top_pred[1] + 2)>>2);
	pr_tmp[10] = pr_tmp[1] = (uint8_t)((top_pred[0] + top_pred[1] + 1)>>1);
	pr_tmp[15] = pr_tmp[6] = (uint8_t)((top_pred[0] + top_pred[1]*2 + top_pred[2] + 2)>>2);
	pr_tmp[11] = pr_tmp[2] = (uint8_t)((top_pred[1] + top_pred[2] + 1)>>1);
	pr_tmp[7] = (uint8_t)((top_pred[1] + top_pred[2]*2 + top_pred[3] + 2)>>2);
	pr_tmp[3] = (uint8_t)((top_pred[2] + top_pred[3] + 1)>>1);
	for(i = 0; i < 4; ++i)
		for(j = 0; j < 4; ++j)
			res_tmp[i*4 + j] = (int16_t)original[i*4+j] - (int16_t)pr_tmp[i*4+j];
	W = weight(res_tmp);
	if (W < MinWeight) 
	{
		bmode = B_VR_PRED;
		MinWeight = W;
		for (i = 0; i < 4; ++i)
			for (j = 0; j < 4; ++j) 
			{
				predictor[i*4+j] = pr_tmp[i*4+j];
				residual[i*4+j] = res_tmp[i*4+j];
			}
	} 
// try B_VL_PRED
	pr_tmp[0] = (uint8_t)((top_pred[0] + top_pred[1] + 1)>>1);
	pr_tmp[4] = (uint8_t)((top_pred[0] + top_pred[1]*2 + top_pred[2] + 2)>>2);
	pr_tmp[8] = pr_tmp[1] = (uint8_t)((top_pred[1] + top_pred[2] + 1)>>1);
	pr_tmp[12] = pr_tmp[5] = (uint8_t)((top_pred[1] + top_pred[2]*2 + top_pred[3] + 2)>>2);
	pr_tmp[9] = pr_tmp[2] = (uint8_t)((top_pred[2] + top_pred[3] + 1)>>1);
	pr_tmp[13] = pr_tmp[6] = (uint8_t)((top_pred[2] + top_pred[3]*2 + top_pred[4] + 2)>>2);
	pr_tmp[10] = pr_tmp[3] = (uint8_t)((top_pred[3] + top_pred[4] + 1)>>1);
	pr_tmp[14] = pr_tmp[7] = (uint8_t)((top_pred[3] + top_pred[4]*2 + top_pred[5] + 2)>>2);
	/* Last two values do not strictly follow the pattern. */
	pr_tmp[11] = (uint8_t)((top_pred[4] + top_pred[5]*2 + top_pred[6] + 2)>>2);
	pr_tmp[15] = (uint8_t)((top_pred[5] + top_pred[6]*2 + top_pred[7] + 2)>>2);
	for(i = 0; i < 4; ++i)
		for(j = 0; j < 4; ++j)
			res_tmp[i*4 + j] = (int16_t)original[i*4+j] - (int16_t)pr_tmp[i*4+j];
	W = weight(res_tmp);
	if (W < MinWeight) 
	{
		bmode = B_VL_PRED;
		MinWeight = W;
		for (i = 0; i < 4; ++i)
			for (j = 0; j < 4; ++j) 
			{
				predictor[i*4+j] = pr_tmp[i*4+j];
				residual[i*4+j] = res_tmp[i*4+j];
			}
	} 
// try B_HD_PRED
	pr_tmp[12] = (uint8_t)((left_pred[3] + left_pred[2] + 1)>>1);
	pr_tmp[13] = (uint8_t)((left_pred[3] + left_pred[2]*2 + left_pred[1] + 2)>>2);
	pr_tmp[8] = pr_tmp[14] = (uint8_t)((left_pred[2] + left_pred[1] + 1)>>1);
	pr_tmp[9] = pr_tmp[15] = (uint8_t)((left_pred[2] + left_pred[1]*2 + left_pred[0] + 2)>>2);
	pr_tmp[4] = pr_tmp[10] = (uint8_t)((left_pred[1] + left_pred[0] + 1)>>1);
	pr_tmp[5] = pr_tmp[11] = (uint8_t)((left_pred[1] + left_pred[0]*2 + top_left_pred + 2)>>2);
	pr_tmp[0] = pr_tmp[6] = (uint8_t)((left_pred[0] + top_left_pred + 1)>>1);
	pr_tmp[1] = pr_tmp[7] = (uint8_t)((left_pred[0] + top_left_pred*2 + top_pred[0] + 2)>>2);
	pr_tmp[2] = (uint8_t)((top_left_pred + top_pred[0]*2 + top_pred[1] + 2)>>2);
	pr_tmp[3] = (uint8_t)((top_pred[0] + top_pred[1]*2 + top_pred[2] + 2)>>2);
	for(i = 0; i < 4; ++i)
		for(j = 0; j < 4; ++j)
			res_tmp[i*4 + j] = (int16_t)original[i*4+j] - (int16_t)pr_tmp[i*4+j];
	W = weight(res_tmp);
	if (W < MinWeight) 
	{
		bmode = B_HD_PRED;
		MinWeight = W;
		for (i = 0; i < 4; ++i)
			for (j = 0; j < 4; ++j) 
			{
				predictor[i*4+j] = pr_tmp[i*4+j];
				residual[i*4+j] = res_tmp[i*4+j];
			}
	} 
// try B_HU_PRED
	pr_tmp[0] = (uint8_t)((left_pred[0] + left_pred[1] + 1)>>1);
	pr_tmp[1] = (uint8_t)((left_pred[0] + left_pred[1]*2 + left_pred[2] + 2)>>2);
	pr_tmp[2] = pr_tmp[4] = (uint8_t)((left_pred[1] + left_pred[2] + 1)>>1);
	pr_tmp[3] = pr_tmp[5] = (uint8_t)((left_pred[1] + left_pred[2]*2 + left_pred[3] + 2)>>2);
	pr_tmp[6] = pr_tmp[8] = (uint8_t)((left_pred[2] + left_pred[3] + 1)>>1);
	pr_tmp[7] = pr_tmp[9] = (uint8_t)((left_pred[2] + left_pred[3]*3 + 2)>>2);
	/* Not possible to follow pattern for much of the bottom
	row because no (nearby) already-constructed pixels lie
	on the diagonals in question. */
	pr_tmp[10] = pr_tmp[11] = pr_tmp[12] = pr_tmp[13] = pr_tmp[14] = pr_tmp[15] = (uint8_t)left_pred[3];
	for(i = 0; i < 4; ++i)
		for(j = 0; j < 4; ++j)
			res_tmp[i*4 + j] = (int16_t)original[i*4+j] - (int16_t)pr_tmp[i*4+j];
	W = weight(res_tmp);
	if (W < MinWeight) 
	{
		bmode = B_HU_PRED;
		MinWeight = W;
		for (i = 0; i < 4; ++i)
			for (j = 0; j < 4; ++j) 
			{
				predictor[i*4+j] = pr_tmp[i*4+j];
				residual[i*4+j] = res_tmp[i*4+j];
			}
	}
	return bmode;
}


void predict_and_transform_mb(int32_t mb_num)
{
	frames.transformed_blocks[mb_num].parts = are4x4;
    int32_t i, mb_row, mb_col, b_num, b_col, b_row, Y_offset, UV_offset, pred_ind_Y, pred_ind_UV;

    int16_t top_left_pred_Y, top_left_pred_U, top_left_pred_V;
    int16_t left_pred_Y[16], left_pred_U[8], left_pred_V[8], top_pred_Y[20], top_pred_U[8], top_pred_V[8];

	uint8_t predictor[16], block_pixels[16];
	int16_t residual[16];

	int32_t y_dc_q,y_ac_q,uv_dc_q,uv_ac_q;
	frames.transformed_blocks[mb_num].segment_id = intra_segment;
	y_dc_q = frames.y_dc_q[intra_segment];
	y_ac_q = frames.y_ac_q[intra_segment];
	uv_dc_q = frames.uv_dc_q[intra_segment];
	uv_ac_q = frames.uv_ac_q[intra_segment];

    mb_row = mb_num / video.mb_width; mb_col = mb_num % video.mb_width;
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
		top_pred_Y[16] = 127;
		top_pred_Y[17] = 127;
		top_pred_Y[18] = 127;
		top_pred_Y[19] = 127;
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
		if (mb_col < (video.mb_width - 1)) {
			top_pred_Y[16] = (int16_t)frames.reconstructed_Y[pred_ind_Y];
			top_pred_Y[17] = (int16_t)frames.reconstructed_Y[pred_ind_Y + 1];
			top_pred_Y[18] = (int16_t)frames.reconstructed_Y[pred_ind_Y + 2];
			top_pred_Y[19] = (int16_t)frames.reconstructed_Y[pred_ind_Y + 3];
		} else {
			top_pred_Y[16] = top_pred_Y[15];
			top_pred_Y[17] = top_pred_Y[15];
			top_pred_Y[18] = top_pred_Y[15];
			top_pred_Y[19] = top_pred_Y[15];
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

			for(i = 0; i < 4; ++i)
			{
				block_pixels[i*4] = *(block_offset);
				block_pixels[i*4 + 1] = *(block_offset + 1);
				block_pixels[i*4 + 2] = *(block_offset + 2);
				block_pixels[i*4 + 3] = *(block_offset + 3);
				block_offset += video.wrk_width;
			}
			frames.e_data[mb_num].mode[b_num] = pick_luma_predictor(block_pixels, predictor, residual, &top_pred_Y[b_col<<2], &left_pred_Y[b_row<<2], top_left_pred_Y);
			DCT4x4(residual, frames.transformed_blocks[mb_num].coeffs[b_num]);
			quant4x4(frames.transformed_blocks[mb_num].coeffs[b_num], y_dc_q, y_ac_q);
			iDCT4x4(frames.transformed_blocks[mb_num].coeffs[b_num], block_pixels, predictor, y_dc_q, y_ac_q);
			block_offset = frames.reconstructed_Y + (Y_offset + (((b_row*video.wrk_width) + b_col ) <<2));
			for(i = 0; i < 4; ++i)
			{
				*(block_offset + i*video.wrk_width) = block_pixels[i*4];
				*(block_offset + i*video.wrk_width + 1) = block_pixels[i*4 + 1];
				*(block_offset + i*video.wrk_width + 2) = block_pixels[i*4 + 2];
				*(block_offset + i*video.wrk_width + 3) = block_pixels[i*4 + 3];
			}
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
	// all chromas will be stuck to TM_PRED
    for (b_num = 0; b_num < 4; ++b_num) // 2x2 U-chroma blocks
    {
        b_row = b_num >> 1; // /2
        b_col = b_num % 2;
        block_offset = frames.current_U + (UV_offset + (( b_row*video.wrk_width<<1) + (b_col<<2) ));
		for(i = 0; i < 4; ++i)
		{
			residual[i*4] = top_pred_U[b_col*4] + left_pred_U[b_row*4 + i] - top_left_pred_U;
			residual[i*4 + 1] = top_pred_U[b_col*4 + 1] + left_pred_U[b_row*4 + i] - top_left_pred_U;
			residual[i*4 + 2] = top_pred_U[b_col*4 + 2] + left_pred_U[b_row*4 + i] - top_left_pred_U;
			residual[i*4 + 3] = top_pred_U[b_col*4 + 3] + left_pred_U[b_row*4 + i] - top_left_pred_U;
			predictor[i*4] = (uint8_t)(residual[i*4] < 0) ? 0 : ((residual[i*4] > 255) ? 255 : residual[i*4]);
			predictor[i*4 + 1] = (uint8_t)(residual[i*4 + 1] < 0) ? 0 : ((residual[i*4 + 1] > 255) ? 255 : residual[i*4 + 1]);
			predictor[i*4 + 2] = (uint8_t)(residual[i*4 + 2] < 0) ? 0 : ((residual[i*4 + 2] > 255) ? 255 : residual[i*4 + 2]);
			predictor[i*4 + 3] = (uint8_t)(residual[i*4 + 3] < 0) ? 0 : ((residual[i*4 + 3] > 255) ? 255 : residual[i*4 + 3]);
			residual[i*4] = (int16_t)*(block_offset) - (int16_t)predictor[i*4];
			residual[i*4 + 1] = (int16_t)*(block_offset + 1) - (int16_t)predictor[i*4 + 1];
			residual[i*4 + 2] = (int16_t)*(block_offset + 2) - (int16_t)predictor[i*4 + 2];
			residual[i*4 + 3] = (int16_t)*(block_offset + 3) - (int16_t)predictor[i*4 + 3];
			block_offset += video.wrk_width/2;
		}
		DCT4x4(residual, frames.transformed_blocks[mb_num].coeffs[b_num+16]);
		quant4x4(frames.transformed_blocks[mb_num].coeffs[b_num+16], uv_dc_q, uv_ac_q);
		iDCT4x4(frames.transformed_blocks[mb_num].coeffs[b_num+16], block_pixels, predictor, uv_dc_q, uv_ac_q);
		block_offset = frames.reconstructed_U + (UV_offset + (( b_row*video.wrk_width<<1) + (b_col<<2) ));
		for(i = 0; i < 4; ++i)
		{
			*(block_offset + i*video.wrk_width/2) = block_pixels[i*4];
			*(block_offset + i*video.wrk_width/2 + 1) = block_pixels[i*4 + 1];
			*(block_offset + i*video.wrk_width/2 + 2) = block_pixels[i*4 + 2];
			*(block_offset + i*video.wrk_width/2 + 3) = block_pixels[i*4 + 3];
		}
        zigzag_block(frames.transformed_blocks[mb_num].coeffs[b_num+16]);
    }
    for (b_num = 0; b_num < 4; ++b_num) // 2x2 V-chroma blocks
	{	
		b_row = b_num >> 1; // /2
        b_col = b_num % 2;
        block_offset = frames.current_V + (UV_offset + (( b_row*video.wrk_width<<1) + (b_col<<2) ));
		for(i = 0; i < 4; ++i)
		{
			residual[i*4] = top_pred_V[b_col*4] + left_pred_V[b_row*4 + i] - top_left_pred_V;
			residual[i*4 + 1] = top_pred_V[b_col*4 + 1] + left_pred_V[b_row*4 + i] - top_left_pred_V;
			residual[i*4 + 2] = top_pred_V[b_col*4 + 2] + left_pred_V[b_row*4 + i] - top_left_pred_V;
			residual[i*4 + 3] = top_pred_V[b_col*4 + 3] + left_pred_V[b_row*4 + i] - top_left_pred_V;
			predictor[i*4] = (uint8_t)(residual[i*4] < 0) ? 0 : ((residual[i*4] > 255) ? 255 : residual[i*4]);
			predictor[i*4 + 1] = (uint8_t)(residual[i*4 + 1] < 0) ? 0 : ((residual[i*4 + 1] > 255) ? 255 : residual[i*4 + 1]);
			predictor[i*4 + 2] = (uint8_t)(residual[i*4 + 2] < 0) ? 0 : ((residual[i*4 + 2] > 255) ? 255 : residual[i*4 + 2]);
			predictor[i*4 + 3] = (uint8_t)(residual[i*4 + 3] < 0) ? 0 : ((residual[i*4 + 3] > 255) ? 255 : residual[i*4 + 3]);
			residual[i*4] = (int16_t)*(block_offset) - (int16_t)predictor[i*4];
			residual[i*4 + 1] = (int16_t)*(block_offset + 1) - (int16_t)predictor[i*4 + 1];
			residual[i*4 + 2] = (int16_t)*(block_offset + 2) - (int16_t)predictor[i*4 + 2];
			residual[i*4 + 3] = (int16_t)*(block_offset + 3) - (int16_t)predictor[i*4 + 3];
			block_offset += video.wrk_width/2;
		}
		DCT4x4(residual, frames.transformed_blocks[mb_num].coeffs[b_num+20]);
		quant4x4(frames.transformed_blocks[mb_num].coeffs[b_num+20], uv_dc_q, uv_ac_q);
		iDCT4x4(frames.transformed_blocks[mb_num].coeffs[b_num+20], block_pixels, predictor, uv_dc_q, uv_ac_q);
		block_offset = frames.reconstructed_V + (UV_offset + (( b_row*video.wrk_width<<1) + (b_col<<2) ));
		for(i = 0; i < 4; ++i)
		{
			*(block_offset + i*video.wrk_width/2) = block_pixels[i*4];
			*(block_offset + i*video.wrk_width/2 + 1) = block_pixels[i*4 + 1];
			*(block_offset + i*video.wrk_width/2 + 2) = block_pixels[i*4 + 2];
			*(block_offset + i*video.wrk_width/2 + 3) = block_pixels[i*4 + 3];
		}
        zigzag_block(frames.transformed_blocks[mb_num].coeffs[b_num+20]);
    }

    return;
}

float count_SSIM_16x16(uint8_t *const frame1Y, uint8_t *const frame1U, uint8_t *const frame1V, const int32_t width1, 
					   uint8_t *const frame2Y, uint8_t *const frame2U, uint8_t *const frame2V, const int32_t width2)
{
	int i,j,M1=0,M2=0,D1=0,D2=0,C=0,t1,t2;
	const float c1 = 0.01f*0.01f*255*255;
	const float c2 = 0.03f*0.03f*255*255;
	float num, denom, ssim, M1f, M2f;

	for(i = 0; i < 16; ++i)
		for(j = 0; j < 16; ++j) {
			M1 += (int)frame1Y[i*width1+j];
			M2 += (int)frame2Y[i*width2+j];
		}
	M1 += 128; M2 += 128;
	M1 /= 256; M2 /= 256;
	for(i = 0; i < 16; ++i)
		for(j = 0; j < 16; ++j) {
			t1 = ((int)frame1Y[i*width1+j]) - M1;
			t2 = ((int)frame2Y[i*width2+j]) - M2;
			D1 += t1*t1; 
			D2 += t2*t2;
			C += t1*t2;
		}
	D1 += 128; D2 += 128; C += 128;
	D1 /= 256; D2 /= 256; C /= 256;

	M1f = (float)M1;
	M2f = (float)M2;
	num = (M1f*M2f*2 + c1)*((float)C*2 + c2);
	denom = (M1f*M1f + M2f*M2f + c1)*((float)D1 + (float)D2 + c2);
	ssim = num/denom;

	// workround to exlude big DC difference when AC difference ~ 0
	M1 -= M2;
	M1 = (M1 < 0) ? -M1 : M1;
	M1f = (M1 > 4) ? (float)M1*0.02f : 0.0f;
	ssim -= M1f;

	const int cwidth1 = width1/2;
	const int cwidth2 = width2/2;
	for(i = 0; i < 8; ++i)
		for(j = 0; j < 8; ++j) {
			M1 += (int)frame1U[i*cwidth1+j];
			M2 += (int)frame2U[i*cwidth2+j];
		}
	M1 += 32; M2 += 32;
	M1 /= 64; M2 /= 64;
	for(i = 0; i < 8; ++i)
		for(j = 0; j < 8; ++j) {
			t1 = ((int)frame1U[i*cwidth1+j]) - M1;
			t2 = ((int)frame2U[i*cwidth2+j]) - M2;
			D1 += t1*t1; 
			D2 += t2*t2;
			C += t1*t2;
		}
	D1 += 32; D2 += 32; C += 32;
	D1 /= 64; D2 /= 64; C /= 64;

	M1f = (float)M1;
	M2f = (float)M2;
	num = (M1f*M2f*2 + c1)*((float)C*2 + c2);
	denom = (M1f*M1f + M2f*M2f + c1)*((float)D1 + (float)D2 + c2);
	ssim += num/denom;

	for(i = 0; i < 8; ++i)
		for(j = 0; j < 8; ++j) {
			M1 += (int)frame1V[i*cwidth1+j];
			M2 += (int)frame2V[i*cwidth2+j];
		}
	M1 += 32; M2 += 32;
	M1 /= 64; M2 /= 64;
	for(i = 0; i < 8; ++i)
		for(j = 0; j < 8; ++j) {
			t1 = ((int)frame1V[i*cwidth1+j]) - M1;
			t2 = ((int)frame2V[i*cwidth2+j]) - M2;
			D1 += t1*t1; 
			D2 += t2*t2;
			C += t1*t2;
		}
	D1 += 32; D2 += 32; C += 32;
	D1 /= 64; D2 /= 64; C /= 64;

	M1f = (float)M1;
	M2f = (float)M2;
	num = (M1f*M2f*2 + c1)*((float)C*2 + c2);
	denom = (M1f*M1f + M2f*M2f + c1)*((float)D1 + (float)D2 + c2);

	ssim += num/denom;
	ssim /= 3;

	return ssim;
}

int test_inter_on_intra(int32_t mb_num, segment_ids id)
{
	macroblock test_mb;
	static uint8_t test_recon_Y[256];
	static uint8_t test_recon_V[64];
	static uint8_t test_recon_U[64];
	const int32_t test_width = 16;

	uint8_t predictor[16], block_pixels[16];
	int16_t residual[16];

	int32_t y_dc_q,y_ac_q,uv_dc_q,uv_ac_q;
	test_mb.segment_id = id;
	y_dc_q = frames.y_dc_q[id];
	y_ac_q = frames.y_ac_q[id];
	uv_dc_q = frames.uv_dc_q[id];
	uv_ac_q = frames.uv_ac_q[id];

	// prepare predictors (TM_B_PRED) and transform each block
    int32_t i,j, mb_row, mb_col, b_num, b_col, b_row, Y_offset, UV_offset, pred_ind_Y, pred_ind_UV;

    int16_t top_left_pred_Y, top_left_pred_U, top_left_pred_V;
    int16_t left_pred_Y[16], left_pred_U[8], left_pred_V[8];
    int16_t top_pred_Y[20], top_pred_U[8], top_pred_V[8];

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
		top_pred_Y[16] = 127;
		top_pred_Y[17] = 127;
		top_pred_Y[18] = 127;
		top_pred_Y[19] = 127;
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
		if (mb_col < (video.mb_width - 1)) {
			top_pred_Y[16] = (int16_t)frames.reconstructed_Y[pred_ind_Y];
			top_pred_Y[17] = (int16_t)frames.reconstructed_Y[pred_ind_Y + 1];
			top_pred_Y[18] = (int16_t)frames.reconstructed_Y[pred_ind_Y + 2];
			top_pred_Y[19] = (int16_t)frames.reconstructed_Y[pred_ind_Y + 3];
		} else {
			top_pred_Y[16] = top_pred_Y[15];
			top_pred_Y[17] = top_pred_Y[15];
			top_pred_Y[18] = top_pred_Y[15];
			top_pred_Y[19] = top_pred_Y[15];
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

			for(i = 0; i < 4; ++i)
			{
				block_pixels[i*4] = *(block_offset);
				block_pixels[i*4 + 1] = *(block_offset + 1);
				block_pixels[i*4 + 2] = *(block_offset + 2);
				block_pixels[i*4 + 3] = *(block_offset + 3);
				block_offset += video.wrk_width;
			}
			frames.e_data[mb_num].mode[b_num] = pick_luma_predictor(block_pixels, predictor, residual, &top_pred_Y[b_col<<2], &left_pred_Y[b_row<<2], top_left_pred_Y);
			DCT4x4(residual, test_mb.coeffs[b_num]);
			quant4x4(test_mb.coeffs[b_num], y_dc_q, y_ac_q);
			iDCT4x4(test_mb.coeffs[b_num], block_pixels, predictor, y_dc_q, y_ac_q);
			for(i = 0; i < 4; ++i)
			{
				*(test_offset + i*test_width) = block_pixels[i*4];
				*(test_offset + i*test_width + 1) = block_pixels[i*4 + 1];
				*(test_offset + i*test_width + 2) = block_pixels[i*4 + 2];
				*(test_offset + i*test_width + 3) = block_pixels[i*4 + 3];
			}


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
		b_row = b_num >> 1;
        b_col = b_num % 2;
        block_offset = frames.current_U + (UV_offset + (( b_row*video.wrk_width<<1) + (b_col<<2) ));
		test_offset = test_recon_U + ((b_row*test_width<<1) + (b_col<<2));
		for(i = 0; i < 4; ++i)
		{
			residual[i*4] = top_pred_U[b_col*4] + left_pred_U[b_row*4 + i] - top_left_pred_U;
			residual[i*4 + 1] = top_pred_U[b_col*4 + 1] + left_pred_U[b_row*4 + i] - top_left_pred_U;
			residual[i*4 + 2] = top_pred_U[b_col*4 + 2] + left_pred_U[b_row*4 + i] - top_left_pred_U;
			residual[i*4 + 3] = top_pred_U[b_col*4 + 3] + left_pred_U[b_row*4 + i] - top_left_pred_U;
			predictor[i*4] = (uint8_t)(residual[i*4] < 0) ? 0 : ((residual[i*4] > 255) ? 255 : residual[i*4]);
			predictor[i*4 + 1] = (uint8_t)(residual[i*4 + 1] < 0) ? 0 : ((residual[i*4 + 1] > 255) ? 255 : residual[i*4 + 1]);
			predictor[i*4 + 2] = (uint8_t)(residual[i*4 + 2] < 0) ? 0 : ((residual[i*4 + 2] > 255) ? 255 : residual[i*4 + 2]);
			predictor[i*4 + 3] = (uint8_t)(residual[i*4 + 3] < 0) ? 0 : ((residual[i*4 + 3] > 255) ? 255 : residual[i*4 + 3]);
			residual[i*4] = (int16_t)*(block_offset) - (int16_t)predictor[i*4];
			residual[i*4 + 1] = (int16_t)*(block_offset + 1) - (int16_t)predictor[i*4 + 1];
			residual[i*4 + 2] = (int16_t)*(block_offset + 2) - (int16_t)predictor[i*4 + 2];
			residual[i*4 + 3] = (int16_t)*(block_offset + 3) - (int16_t)predictor[i*4 + 3];
			block_offset += video.wrk_width/2;
		}
		DCT4x4(residual, test_mb.coeffs[b_num+16]);
		quant4x4(test_mb.coeffs[b_num+16], uv_dc_q, uv_ac_q);
		iDCT4x4(test_mb.coeffs[b_num+16], block_pixels, predictor, uv_dc_q, uv_ac_q);
		for(i = 0; i < 4; ++i)
		{
			*(test_offset + i*test_width/2) = block_pixels[i*4];
			*(test_offset + i*test_width/2 + 1) = block_pixels[i*4 + 1];
			*(test_offset + i*test_width/2 + 2) = block_pixels[i*4 + 2];
			*(test_offset + i*test_width/2 + 3) = block_pixels[i*4 + 3];
		}
    }
    for (b_num = 0; b_num < 4; ++b_num) // 2x2 V-chroma blocks
    {
		b_row = b_num >> 1;
        b_col = b_num % 2;
        block_offset = frames.current_V + (UV_offset + (( b_row*video.wrk_width<<1) + (b_col<<2) ));
		test_offset = test_recon_V + ((b_row*test_width<<1) + (b_col<<2));
		for(i = 0; i < 4; ++i)
		{
			residual[i*4] = top_pred_V[b_col*4] + left_pred_V[b_row*4 + i] - top_left_pred_V;
			residual[i*4 + 1] = top_pred_V[b_col*4 + 1] + left_pred_V[b_row*4 + i] - top_left_pred_V;
			residual[i*4 + 2] = top_pred_V[b_col*4 + 2] + left_pred_V[b_row*4 + i] - top_left_pred_V;
			residual[i*4 + 3] = top_pred_V[b_col*4 + 3] + left_pred_V[b_row*4 + i] - top_left_pred_V;
			predictor[i*4] = (uint8_t)(residual[i*4] < 0) ? 0 : ((residual[i*4] > 255) ? 255 : residual[i*4]);
			predictor[i*4 + 1] = (uint8_t)(residual[i*4 + 1] < 0) ? 0 : ((residual[i*4 + 1] > 255) ? 255 : residual[i*4 + 1]);
			predictor[i*4 + 2] = (uint8_t)(residual[i*4 + 2] < 0) ? 0 : ((residual[i*4 + 2] > 255) ? 255 : residual[i*4 + 2]);
			predictor[i*4 + 3] = (uint8_t)(residual[i*4 + 3] < 0) ? 0 : ((residual[i*4 + 3] > 255) ? 255 : residual[i*4 + 3]);
			residual[i*4] = (int16_t)*(block_offset) - (int16_t)predictor[i*4];
			residual[i*4 + 1] = (int16_t)*(block_offset + 1) - (int16_t)predictor[i*4 + 1];
			residual[i*4 + 2] = (int16_t)*(block_offset + 2) - (int16_t)predictor[i*4 + 2];
			residual[i*4 + 3] = (int16_t)*(block_offset + 3) - (int16_t)predictor[i*4 + 3];
			block_offset += video.wrk_width/2;
		}
		DCT4x4(residual, test_mb.coeffs[b_num+20]);
		quant4x4(test_mb.coeffs[b_num+20], uv_dc_q, uv_ac_q);
		iDCT4x4(test_mb.coeffs[b_num+20], block_pixels, predictor, uv_dc_q, uv_ac_q);
		for(i = 0; i < 4; ++i)
		{
			*(test_offset + i*test_width/2) = block_pixels[i*4];
			*(test_offset + i*test_width/2 + 1) = block_pixels[i*4 + 1];
			*(test_offset + i*test_width/2 + 2) = block_pixels[i*4 + 2];
			*(test_offset + i*test_width/2 + 3) = block_pixels[i*4 + 3];
		}
    }
	
	test_mb.SSIM = count_SSIM_16x16(test_recon_Y, test_recon_U, test_recon_V, test_width, 
									frames.current_Y + Y_offset, frames.current_U + UV_offset, frames.current_V + UV_offset, video.wrk_width);
	if (test_mb.SSIM > frames.transformed_blocks[mb_num].SSIM)
	{
		test_mb.parts = are4x4;
		//then we replace inter encoded and reconstructed MB with intra
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
	frames.frames_until_key = video.GOP_size;
	frames.frames_until_altref = video.altref_range;
	frames.last_key_detect = frames.frame_number;

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
	if (video.GOP_size > 1) {
		device.state_gpu = clEnqueueWriteBuffer(device.commandQueue_gpu, device.reconstructed_frame_Y, CL_TRUE, 0, video.wrk_frame_size_luma, frames.reconstructed_Y, 0, NULL, NULL);
		device.state_gpu = clEnqueueWriteBuffer(device.commandQueue_gpu, device.reconstructed_frame_U, CL_TRUE, 0, video.wrk_frame_size_chroma, frames.reconstructed_U, 0, NULL, NULL);
		device.state_gpu = clEnqueueWriteBuffer(device.commandQueue_gpu, device.reconstructed_frame_V, CL_TRUE, 0, video.wrk_frame_size_chroma, frames.reconstructed_V, 0, NULL, NULL);
	}  

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
		
		frames.segments_data[i].loop_filter_level = frames.y_dc_q[i]/QUANT_TO_FILTER_LEVEL - 1;
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
	device.state_gpu = 
		clEnqueueWriteBuffer(device.commandQueue_gpu, device.segments_data, CL_TRUE, 0, sizeof(segment_data)*4, frames.segments_data, 0, NULL, NULL); 
	return;
}

void do_loop_filter()
{
	if (video.GOP_size < 2) return;

	int32_t stage, mb_size, plane_width;
	for (stage = 0; stage < (video.mb_width + (video.mb_height-1)*2); ++stage) 
	{
		device.state_gpu = clFlush(device.commandQueue_gpu);
		device.state_gpu = clSetKernelArg(device.normal_loop_filter_MBH, 5, sizeof(int32_t), &stage);
		device.state_gpu = clSetKernelArg(device.normal_loop_filter_MBV, 5, sizeof(int32_t), &stage);
		device.gpu_work_items_per_dim[0] = video.mb_height*16;

		mb_size = 16;
		plane_width = video.wrk_width;
		device.state_gpu = clSetKernelArg(device.normal_loop_filter_MBH, 4, sizeof(int32_t), &mb_size);
		device.state_gpu = clSetKernelArg(device.normal_loop_filter_MBH, 1, sizeof(int32_t), &plane_width);
		device.state_gpu = clSetKernelArg(device.normal_loop_filter_MBH, 0, sizeof(cl_mem), &device.reconstructed_frame_Y);
		device.state_gpu = clEnqueueNDRangeKernel(device.commandQueue_gpu, device.normal_loop_filter_MBH, 1, NULL, device.gpu_work_items_per_dim, NULL, 0, NULL, NULL);
		if (device.state_gpu != 0)  printf(">error while deblocking : %d", device.state_gpu);
		device.state_gpu = clFlush(device.commandQueue_gpu);
		if (device.state_gpu != 0)  printf(">error while deblocking : %d", device.state_gpu);
		device.gpu_work_items_per_dim[0] = video.mb_height*8;
		mb_size = 8;
		plane_width = video.wrk_width/2;
		device.state_gpu = clSetKernelArg(device.normal_loop_filter_MBH, 4, sizeof(int32_t), &mb_size);
		device.state_gpu = clSetKernelArg(device.normal_loop_filter_MBH, 1, sizeof(int32_t), &plane_width);
		device.state_gpu = clSetKernelArg(device.normal_loop_filter_MBH, 0, sizeof(cl_mem), &device.reconstructed_frame_U);
		device.state_gpu = clEnqueueNDRangeKernel(device.commandQueue_gpu, device.normal_loop_filter_MBH, 1, NULL, device.gpu_work_items_per_dim, NULL, 0, NULL, NULL);
		if (device.state_gpu != 0)  printf(">error while deblocking : %d", device.state_gpu);
		device.state_gpu = clFlush(device.commandQueue_gpu);
		if (device.state_gpu != 0)  printf(">error while deblocking : %d", device.state_gpu);
		device.state_gpu = clSetKernelArg(device.normal_loop_filter_MBH, 0, sizeof(cl_mem), &device.reconstructed_frame_V);
		device.state_gpu = clEnqueueNDRangeKernel(device.commandQueue_gpu, device.normal_loop_filter_MBH, 1, NULL, device.gpu_work_items_per_dim, NULL, 0, NULL, NULL);
		if (device.state_gpu != 0)  printf(">error while deblocking : %d", device.state_gpu);
		device.state_gpu = clFinish(device.commandQueue_gpu);
		if (device.state_gpu != 0)  printf(">error while deblocking : %d", device.state_gpu);

		device.gpu_work_items_per_dim[0] = video.mb_height*4;
		mb_size = 16;
		plane_width = video.wrk_width;
		device.state_gpu = clSetKernelArg(device.normal_loop_filter_MBV, 4, sizeof(int32_t), &mb_size);
		device.state_gpu = clSetKernelArg(device.normal_loop_filter_MBV, 1, sizeof(int32_t), &plane_width);
		device.state_gpu = clSetKernelArg(device.normal_loop_filter_MBV, 0, sizeof(cl_mem), &device.reconstructed_frame_Y);
		device.state_gpu = clEnqueueNDRangeKernel(device.commandQueue_gpu, device.normal_loop_filter_MBV, 1, NULL, device.gpu_work_items_per_dim, NULL, 0, NULL, NULL);
		if (device.state_gpu != 0)  printf(">error while deblocking : %d", device.state_gpu);
		device.state_gpu = clFlush(device.commandQueue_gpu);
		if (device.state_gpu != 0)  printf(">error while deblocking : %d", device.state_gpu);
		device.gpu_work_items_per_dim[0] = video.mb_height*2;
		mb_size = 8;
		plane_width = video.wrk_width/2;
		device.state_gpu = clSetKernelArg(device.normal_loop_filter_MBV, 4, sizeof(int32_t), &mb_size);
		device.state_gpu = clSetKernelArg(device.normal_loop_filter_MBV, 1, sizeof(int32_t), &plane_width);
		device.state_gpu = clSetKernelArg(device.normal_loop_filter_MBV, 0, sizeof(cl_mem), &device.reconstructed_frame_U);
		device.state_gpu = clEnqueueNDRangeKernel(device.commandQueue_gpu, device.normal_loop_filter_MBV, 1, NULL, device.gpu_work_items_per_dim, NULL, 0, NULL, NULL);
		if (device.state_gpu != 0)  printf(">error while deblocking : %d", device.state_gpu);
		device.state_gpu = clFlush(device.commandQueue_gpu);
		if (device.state_gpu != 0)  printf(">error while deblocking : %d", device.state_gpu);
		device.state_gpu = clSetKernelArg(device.normal_loop_filter_MBV, 0, sizeof(cl_mem), &device.reconstructed_frame_V);
		device.state_gpu = clEnqueueNDRangeKernel(device.commandQueue_gpu, device.normal_loop_filter_MBV, 1, NULL, device.gpu_work_items_per_dim, NULL, 0, NULL, NULL);
		if (device.state_gpu != 0)  printf(">error while deblocking : %d", device.state_gpu);
		device.state_gpu = clFinish(device.commandQueue_gpu);
		if (device.state_gpu != 0)  printf(">error while deblocking : %d", device.state_gpu);
	}
	//if is key - renew GOLDEN buffer
	if (frames.current_is_key_frame) 
	{
		device.state_gpu = clEnqueueCopyBuffer(device.commandQueue_gpu, device.reconstructed_frame_Y, device.golden_frame_Y, 0, 0, video.wrk_frame_size_luma, 0, NULL, NULL);
		device.state_gpu = clEnqueueCopyBuffer(device.commandQueue_gpu, device.reconstructed_frame_U, device.golden_frame_U, 0, 0, video.wrk_frame_size_chroma, 0, NULL, NULL);
		device.state_gpu = clEnqueueCopyBuffer(device.commandQueue_gpu, device.reconstructed_frame_V, device.golden_frame_V, 0, 0, video.wrk_frame_size_chroma, 0, NULL, NULL);
		device.state_gpu = clFinish(device.commandQueue_gpu);
	}
	// if key or forced altref - renew altref buffer
	if ((frames.current_is_key_frame)  || (frames.current_is_altref_frame))
	{
		device.state_gpu = clEnqueueCopyBuffer(device.commandQueue_gpu, device.reconstructed_frame_Y, device.altref_frame_Y, 0, 0, video.wrk_frame_size_luma, 0, NULL, NULL);
		device.state_gpu = clEnqueueCopyBuffer(device.commandQueue_gpu, device.reconstructed_frame_U, device.altref_frame_U, 0, 0, video.wrk_frame_size_chroma, 0, NULL, NULL);
		device.state_gpu = clEnqueueCopyBuffer(device.commandQueue_gpu, device.reconstructed_frame_V, device.altref_frame_V, 0, 0, video.wrk_frame_size_chroma, 0, NULL, NULL);
		device.state_gpu = clFinish(device.commandQueue_gpu);
		frames.frames_until_altref = video.altref_range;
	}

	return;
}

void interpolate()
{
	// interpolate last frame buffer for luma(Y) with bicubic filter
	// both HORIZONTALLY
	device.gpu_work_items_per_dim[0] = video.wrk_height;
	if (frames.current_is_altref_frame)
		device.state_gpu = clSetKernelArg(device.luma_interpolate_Hx4, 0, sizeof(cl_mem), &device.altref_frame_Y);
	else
		device.state_gpu = clSetKernelArg(device.luma_interpolate_Hx4, 0, sizeof(cl_mem), &device.reconstructed_frame_Y);
	device.state_gpu = clEnqueueNDRangeKernel(device.commandQueue_gpu, device.luma_interpolate_Hx4, 1, NULL, device.gpu_work_items_per_dim, NULL, 0, NULL, NULL);
	if (device.state_gpu != 0)  printf(">error when interpolating luma horizontally : %d", device.state_gpu);
	device.state_gpu = clFlush(device.commandQueue_gpu);
	device.gpu_work_items_per_dim[0] = video.wrk_height/2;
	if (frames.current_is_altref_frame)
		device.state_gpu = clSetKernelArg(device.chroma_interpolate_Hx8, 0, sizeof(cl_mem), &device.altref_frame_U);
	else
		device.state_gpu = clSetKernelArg(device.chroma_interpolate_Hx8, 0, sizeof(cl_mem), &device.reconstructed_frame_U);
	device.state_gpu = clSetKernelArg(device.chroma_interpolate_Hx8, 1, sizeof(cl_mem), &device.ref_frame_U);
	device.state_gpu = clEnqueueNDRangeKernel(device.commandQueue_gpu, device.chroma_interpolate_Hx8, 1, NULL, device.gpu_work_items_per_dim, NULL, 0, NULL, NULL);
	if (device.state_gpu != 0)  printf(">error when interpolating U-chroma horizontaly : %d", device.state_gpu);
	device.state_gpu = clFlush(device.commandQueue_gpu);
	if (frames.current_is_altref_frame)
		device.state_gpu = clSetKernelArg(device.chroma_interpolate_Hx8, 0, sizeof(cl_mem), &device.altref_frame_V);
	else
		device.state_gpu = clSetKernelArg(device.chroma_interpolate_Hx8, 0, sizeof(cl_mem), &device.reconstructed_frame_V);
	device.state_gpu = clSetKernelArg(device.chroma_interpolate_Hx8, 1, sizeof(cl_mem), &device.ref_frame_V);
	device.state_gpu = clEnqueueNDRangeKernel(device.commandQueue_gpu, device.chroma_interpolate_Hx8, 1, NULL, device.gpu_work_items_per_dim, NULL, 0, NULL, NULL);
	if (device.state_gpu != 0)  printf(">error when interpolating V-chroma horizontaly : %d", device.state_gpu);
	device.state_gpu = clFinish(device.commandQueue_gpu);
	// and VERTICALLY
	device.gpu_work_items_per_dim[0] = video.wrk_width;
	device.state_gpu = clEnqueueNDRangeKernel(device.commandQueue_gpu, device.luma_interpolate_Vx4, 1, NULL, device.gpu_work_items_per_dim, NULL, 0, NULL, NULL);
	if (device.state_gpu != 0)  printf(">error when interpolating luma vertically : %d", device.state_gpu);
	device.state_gpu = clFlush(device.commandQueue_gpu);
	device.state_gpu = clSetKernelArg(device.chroma_interpolate_Vx8, 0, sizeof(cl_mem), &device.ref_frame_U);
	device.state_gpu = clEnqueueNDRangeKernel(device.commandQueue_gpu, device.chroma_interpolate_Vx8, 1, NULL, device.gpu_work_items_per_dim, NULL, 0, NULL, NULL);
	if (device.state_gpu != 0)  printf(">error when interpolating U-chroma vertically : %d", device.state_gpu);
	device.state_gpu = clFlush(device.commandQueue_gpu);
	device.state_gpu = clSetKernelArg(device.chroma_interpolate_Vx8, 0, sizeof(cl_mem), &device.ref_frame_V);
	device.state_gpu = clEnqueueNDRangeKernel(device.commandQueue_gpu, device.chroma_interpolate_Vx8, 1, NULL, device.gpu_work_items_per_dim, NULL, 0, NULL, NULL);
	if (device.state_gpu != 0)  printf(">error when interpolating V-chroma vertically : %d", device.state_gpu);
	device.state_gpu = clFinish(device.commandQueue_gpu);
	return;
}

void downsample()
{
	int32_t width, height;
	
	// first reset vector nets to zeros
	device.state_gpu = clFinish(device.commandQueue_gpu);
	int ref = (frames.current_is_altref_frame) ? ALTREF : LAST;
	device.state_gpu = clSetKernelArg(device.reset_vectors, 3, sizeof(int32_t), &ref);
	device.gpu_work_items_per_dim[0] = video.mb_count;
	device.state_gpu = clEnqueueNDRangeKernel(device.commandQueue_gpu, device.reset_vectors, 1, NULL, device.gpu_work_items_per_dim, NULL, 0, NULL, NULL);
	device.state_gpu = clFinish(device.commandQueue_gpu);

	//prepare downsampled by 2
	if (frames.current_is_altref_frame)
		device.state_gpu = clSetKernelArg(device.downsample, 0, sizeof(cl_mem), &device.altref_frame_Y);
	else
		device.state_gpu = clSetKernelArg(device.downsample, 0, sizeof(cl_mem), &device.reconstructed_frame_Y);// last one
	device.state_gpu = clSetKernelArg(device.downsample, 1, sizeof(cl_mem), &device.ref_frame_Y_downsampled_by2);
	device.state_gpu = clSetKernelArg(device.downsample, 2, sizeof(int32_t), &video.wrk_width);
	device.state_gpu = clSetKernelArg(device.downsample, 3, sizeof(int32_t), &video.wrk_height);
	device.gpu_work_items_per_dim[0] = video.wrk_width*video.wrk_height/4;
	device.state_gpu = clEnqueueNDRangeKernel(device.commandQueue_gpu, device.downsample, 1, NULL, device.gpu_work_items_per_dim, NULL, 0, NULL, NULL);
	device.state_gpu = clFlush(device.commandQueue_gpu);
	device.state_gpu = clSetKernelArg(device.downsample, 0, sizeof(cl_mem), &device.current_frame_Y);
	device.state_gpu = clSetKernelArg(device.downsample, 1, sizeof(cl_mem), &device.current_frame_Y_downsampled_by2);
	device.state_gpu = clEnqueueNDRangeKernel(device.commandQueue_gpu, device.downsample, 1, NULL, device.gpu_work_items_per_dim, NULL, 0, NULL, NULL);
	device.state_gpu = clFinish(device.commandQueue_gpu);

	//prepare downsampled by 4
	device.state_gpu = clSetKernelArg(device.downsample, 0, sizeof(cl_mem), &device.ref_frame_Y_downsampled_by2);
	device.state_gpu = clSetKernelArg(device.downsample, 1, sizeof(cl_mem), &device.ref_frame_Y_downsampled_by4);
	width = video.wrk_width/2;
	device.state_gpu = clSetKernelArg(device.downsample, 2, sizeof(int32_t), &width);
	height = video.wrk_height/2;
	device.state_gpu = clSetKernelArg(device.downsample, 3, sizeof(int32_t), &height);
	device.gpu_work_items_per_dim[0] = width*height/4;
	device.state_gpu = clEnqueueNDRangeKernel(device.commandQueue_gpu, device.downsample, 1, NULL, device.gpu_work_items_per_dim, NULL, 0, NULL, NULL);
	device.state_gpu = clFlush(device.commandQueue_gpu);
	device.state_gpu = clSetKernelArg(device.downsample, 0, sizeof(cl_mem), &device.current_frame_Y_downsampled_by2);
	device.state_gpu = clSetKernelArg(device.downsample, 1, sizeof(cl_mem), &device.current_frame_Y_downsampled_by4);
	device.state_gpu = clEnqueueNDRangeKernel(device.commandQueue_gpu, device.downsample, 1, NULL, device.gpu_work_items_per_dim, NULL, 0, NULL, NULL);
	device.state_gpu = clFinish(device.commandQueue_gpu);

	//prepare downsampled by 8
	device.state_gpu = clSetKernelArg(device.downsample, 0, sizeof(cl_mem), &device.ref_frame_Y_downsampled_by4);
	device.state_gpu = clSetKernelArg(device.downsample, 1, sizeof(cl_mem), &device.ref_frame_Y_downsampled_by8);
	width = video.wrk_width/4;
	device.state_gpu = clSetKernelArg(device.downsample, 2, sizeof(int32_t), &width);
	height = video.wrk_height/4;
	device.state_gpu = clSetKernelArg(device.downsample, 3, sizeof(int32_t), &height);
	device.gpu_work_items_per_dim[0] = width*height/4;
	device.state_gpu = clEnqueueNDRangeKernel(device.commandQueue_gpu, device.downsample, 1, NULL, device.gpu_work_items_per_dim, NULL, 0, NULL, NULL);
	device.state_gpu = clFlush(device.commandQueue_gpu);
	device.state_gpu = clSetKernelArg(device.downsample, 0, sizeof(cl_mem), &device.current_frame_Y_downsampled_by4);
	device.state_gpu = clSetKernelArg(device.downsample, 1, sizeof(cl_mem), &device.current_frame_Y_downsampled_by8);
	device.state_gpu = clEnqueueNDRangeKernel(device.commandQueue_gpu, device.downsample, 1, NULL, device.gpu_work_items_per_dim, NULL, 0, NULL, NULL);
	device.state_gpu = clFinish(device.commandQueue_gpu);

	//prepare downsampled by 16
	device.state_gpu = clSetKernelArg(device.downsample, 0, sizeof(cl_mem), &device.ref_frame_Y_downsampled_by8);
	device.state_gpu = clSetKernelArg(device.downsample, 1, sizeof(cl_mem), &device.ref_frame_Y_downsampled_by16);
	width = video.wrk_width/8;
	device.state_gpu = clSetKernelArg(device.downsample, 2, sizeof(int32_t), &width);
	height = video.wrk_height/8;
	device.state_gpu = clSetKernelArg(device.downsample, 3, sizeof(int32_t), &height);
	device.gpu_work_items_per_dim[0] = width*height/4;
	device.state_gpu = clEnqueueNDRangeKernel(device.commandQueue_gpu, device.downsample, 1, NULL, device.gpu_work_items_per_dim, NULL, 0, NULL, NULL);
	device.state_gpu = clFlush(device.commandQueue_gpu);
	device.state_gpu = clSetKernelArg(device.downsample, 0, sizeof(cl_mem), &device.current_frame_Y_downsampled_by8);
	device.state_gpu = clSetKernelArg(device.downsample, 1, sizeof(cl_mem), &device.current_frame_Y_downsampled_by16);
	device.state_gpu = clEnqueueNDRangeKernel(device.commandQueue_gpu, device.downsample, 1, NULL, device.gpu_work_items_per_dim, NULL, 0, NULL, NULL);
	device.state_gpu = clFinish(device.commandQueue_gpu);
	
	return;
}

void inter_transform()
{
	clFinish(device.commandQueue_gpu);
	// if we are encoding altref frame > then we search in previous altref frame
	// if we are encoding regular frame -> then in last

	int32_t val;

	//prepare downsampled frames
	downsample();
	// prepare interpolated
	interpolate();

	//now search in downsampled by 16
	device.gpu_work_items_per_dim[0] = ((video.wrk_width/16)/8)*((video.wrk_height/16)/8);
	device.state_gpu = clSetKernelArg(device.luma_search_1step, 0, sizeof(cl_mem), &device.current_frame_Y_downsampled_by16);
	device.state_gpu = clSetKernelArg(device.luma_search_1step, 1, sizeof(cl_mem), &device.ref_frame_Y_downsampled_by16);
	device.state_gpu = clSetKernelArg(device.luma_search_1step, 2, sizeof(cl_mem), &device.vnet1);
	device.state_gpu = clSetKernelArg(device.luma_search_1step, 3, sizeof(cl_mem), &device.vnet2);
	val = video.wrk_width/16;
	device.state_gpu = clSetKernelArg(device.luma_search_1step, 5, sizeof(int32_t), &val);
	val = video.wrk_height/16;
	device.state_gpu = clSetKernelArg(device.luma_search_1step, 6, sizeof(int32_t), &val);
	val = 16;
	device.state_gpu = clSetKernelArg(device.luma_search_1step, 7, sizeof(int32_t), &val);
	device.state_gpu = clEnqueueNDRangeKernel(device.commandQueue_gpu, device.luma_search_1step, 1, NULL, device.gpu_work_items_per_dim, NULL, 0, NULL, NULL);
	device.state_gpu = clFinish(device.commandQueue_gpu);

	//now search in downsampled by 8
	device.gpu_work_items_per_dim[0] = ((video.wrk_width/8)/8)*((video.wrk_height/8)/8);
	device.state_gpu = clSetKernelArg(device.luma_search_1step, 0, sizeof(cl_mem), &device.current_frame_Y_downsampled_by8);
	device.state_gpu = clSetKernelArg(device.luma_search_1step, 1, sizeof(cl_mem), &device.ref_frame_Y_downsampled_by8);
	device.state_gpu = clSetKernelArg(device.luma_search_1step, 2, sizeof(cl_mem), &device.vnet2);
	device.state_gpu = clSetKernelArg(device.luma_search_1step, 3, sizeof(cl_mem), &device.vnet1);
	val = video.wrk_width/8;
	device.state_gpu = clSetKernelArg(device.luma_search_1step, 5, sizeof(int32_t), &val);
	val = video.wrk_height/8;
	device.state_gpu = clSetKernelArg(device.luma_search_1step, 6, sizeof(int32_t), &val);
	val = 8;
	device.state_gpu = clSetKernelArg(device.luma_search_1step, 7, sizeof(int32_t), &val);
	device.state_gpu = clEnqueueNDRangeKernel(device.commandQueue_gpu, device.luma_search_1step, 1, NULL, device.gpu_work_items_per_dim, NULL, 0, NULL, NULL);
	device.state_gpu = clFinish(device.commandQueue_gpu);

	//now search in downsampled by 4
	device.gpu_work_items_per_dim[0] = ((video.wrk_width/4)/8)*((video.wrk_height/4)/8);
	device.state_gpu = clSetKernelArg(device.luma_search_1step, 0, sizeof(cl_mem), &device.current_frame_Y_downsampled_by4);
	device.state_gpu = clSetKernelArg(device.luma_search_1step, 1, sizeof(cl_mem), &device.ref_frame_Y_downsampled_by4);
	device.state_gpu = clSetKernelArg(device.luma_search_1step, 2, sizeof(cl_mem), &device.vnet1);
	device.state_gpu = clSetKernelArg(device.luma_search_1step, 3, sizeof(cl_mem), &device.vnet2);
	val = video.wrk_width/4;
	device.state_gpu = clSetKernelArg(device.luma_search_1step, 5, sizeof(int32_t), &val);
	val = video.wrk_height/4;
	device.state_gpu = clSetKernelArg(device.luma_search_1step, 6, sizeof(int32_t), &val);
	val = 4;
	device.state_gpu = clSetKernelArg(device.luma_search_1step, 7, sizeof(int32_t), &val);
	device.state_gpu = clEnqueueNDRangeKernel(device.commandQueue_gpu, device.luma_search_1step, 1, NULL, device.gpu_work_items_per_dim, NULL, 0, NULL, NULL);
	device.state_gpu = clFinish(device.commandQueue_gpu);

	//now search in downsampled by 2
	device.gpu_work_items_per_dim[0] = ((video.wrk_width/2)/8)*((video.wrk_height/2)/8);
	device.state_gpu = clSetKernelArg(device.luma_search_1step, 0, sizeof(cl_mem), &device.current_frame_Y_downsampled_by2);
	device.state_gpu = clSetKernelArg(device.luma_search_1step, 1, sizeof(cl_mem), &device.ref_frame_Y_downsampled_by2);
	device.state_gpu = clSetKernelArg(device.luma_search_1step, 2, sizeof(cl_mem), &device.vnet2);
	device.state_gpu = clSetKernelArg(device.luma_search_1step, 3, sizeof(cl_mem), &device.vnet1);
	val = video.wrk_width/2;
	device.state_gpu = clSetKernelArg(device.luma_search_1step, 5, sizeof(int32_t), &val);
	val = video.wrk_height/2;
	device.state_gpu = clSetKernelArg(device.luma_search_1step, 6, sizeof(int32_t), &val);
	val = 2;
	device.state_gpu = clSetKernelArg(device.luma_search_1step, 7, sizeof(int32_t), &val);
	device.state_gpu = clEnqueueNDRangeKernel(device.commandQueue_gpu, device.luma_search_1step, 1, NULL, device.gpu_work_items_per_dim, NULL, 0, NULL, NULL);
	device.state_gpu = clFinish(device.commandQueue_gpu);

	//now search in original size
	device.gpu_work_items_per_dim[0] = video.mb_count*4; 
	device.state_gpu = clSetKernelArg(device.luma_search_1step, 0, sizeof(cl_mem), &device.current_frame_Y);
	if (frames.current_is_altref_frame)
		device.state_gpu = clSetKernelArg(device.luma_search_1step, 1, sizeof(cl_mem), &device.altref_frame_Y);
	else
		device.state_gpu = clSetKernelArg(device.luma_search_1step, 1, sizeof(cl_mem), &device.reconstructed_frame_Y);
	device.state_gpu = clSetKernelArg(device.luma_search_1step, 2, sizeof(cl_mem), &device.vnet1);
	device.state_gpu = clSetKernelArg(device.luma_search_1step, 3, sizeof(cl_mem), &device.vnet2);
	val = video.wrk_width;
	device.state_gpu = clSetKernelArg(device.luma_search_1step, 5, sizeof(int32_t), &val);
	val = video.wrk_height;
	device.state_gpu = clSetKernelArg(device.luma_search_1step, 6, sizeof(int32_t), &val);
	val = 1;
	device.state_gpu = clSetKernelArg(device.luma_search_1step, 7, sizeof(int32_t), &val);
	device.state_gpu = clEnqueueNDRangeKernel(device.commandQueue_gpu, device.luma_search_1step, 1, NULL, device.gpu_work_items_per_dim, NULL, 0, NULL, NULL);
	device.state_gpu = clFinish(device.commandQueue_gpu);

	//now search in interpolated frame
	// all kernel arguments vave already been set
	device.gpu_work_items_per_dim[0] = video.mb_count*4; //again. amount of blocks the same as with original size
	device.state_gpu = clEnqueueNDRangeKernel(device.commandQueue_gpu, device.luma_search_2step, 1, NULL, device.gpu_work_items_per_dim, NULL, 0, NULL, NULL);
	device.state_gpu = clFinish(device.commandQueue_gpu);

	//now check if it's possible to make (0;0)-reference to other frames 
	if (!frames.prev_is_key_frame) 
	{
		device.gpu_work_items_per_dim[0] = video.mb_count;
		//first try golden
		int32_t reffr = GOLDEN;
		device.state_gpu = clSetKernelArg(device.try_another_reference, 1, sizeof(cl_mem), &device.golden_frame_Y);
		device.state_gpu = clSetKernelArg(device.try_another_reference, 4, sizeof(cl_mem), &device.golden_frame_U);
		device.state_gpu = clSetKernelArg(device.try_another_reference, 7, sizeof(cl_mem), &device.golden_frame_V);
		device.state_gpu = clSetKernelArg(device.try_another_reference, 13, sizeof(int32_t), &reffr);
		device.state_gpu = clEnqueueNDRangeKernel(device.commandQueue_gpu, device.try_another_reference, 1, NULL, device.gpu_work_items_per_dim, NULL, 0, NULL, NULL);
		device.state_gpu = clFinish(device.commandQueue_gpu);
	}
	if (0)//((!frames.prev_is_altref_frame) && (!frames.current_is_altref_frame))
	{
		//not working with current frames being new altref now
		int32_t reffr = ALTREF;
		device.state_gpu = clSetKernelArg(device.try_another_reference, 1, sizeof(cl_mem), &device.altref_frame_Y);
		device.state_gpu = clSetKernelArg(device.try_another_reference, 4, sizeof(cl_mem), &device.altref_frame_U);
		device.state_gpu = clSetKernelArg(device.try_another_reference, 7, sizeof(cl_mem), &device.altref_frame_V);
		device.state_gpu = clSetKernelArg(device.try_another_reference, 13, sizeof(int32_t), &reffr);
		device.state_gpu = clEnqueueNDRangeKernel(device.commandQueue_gpu, device.try_another_reference, 1, NULL, device.gpu_work_items_per_dim, NULL, 0, NULL, NULL);
		device.state_gpu = clFinish(device.commandQueue_gpu);
	}

	// now for each segment (begin with highest quantizer (last index))
	int32_t seg_id;
	float SSIM_targ = video.SSIM_target;
	device.state_gpu = clSetKernelArg(device.luma_transform_16x16, 7, sizeof(float), &SSIM_targ);
	device.state_gpu = clSetKernelArg(device.luma_transform_8x8, 7, sizeof(float), &SSIM_targ);
	for (seg_id = LQ_segment; seg_id >= UQ_segment; --seg_id)
	{
		device.state_gpu = clSetKernelArg(device.luma_transform_16x16, 6, sizeof(int32_t), &seg_id);
		device.state_gpu = clSetKernelArg(device.luma_transform_8x8, 6, sizeof(int32_t), &seg_id);
		device.state_gpu = clSetKernelArg(device.count_SSIM_luma, 4, sizeof(int32_t), &seg_id);
		device.state_gpu = clSetKernelArg(device.count_SSIM_chroma, 4, sizeof(int32_t), &seg_id);

		device.gpu_work_items_per_dim[0] = video.mb_count;
		device.state_gpu = clEnqueueNDRangeKernel(device.commandQueue_gpu, device.luma_transform_16x16, 1, NULL, device.gpu_work_items_per_dim, NULL, 0, NULL, NULL);
		device.state_gpu = clFinish(device.commandQueue_gpu);

		device.gpu_work_items_per_dim[0] = video.mb_count*4;
		device.state_gpu = clEnqueueNDRangeKernel(device.commandQueue_gpu, device.luma_transform_8x8, 1, NULL, device.gpu_work_items_per_dim, NULL, 0, NULL, NULL);
		device.state_gpu = clFinish(device.commandQueue_gpu);

		device.gpu_work_items_per_dim[0] = video.mb_count*4;
		int block_place = 16;
		device.state_gpu = clSetKernelArg(device.chroma_transform, 7, sizeof(int32_t), &block_place);
		device.state_gpu = clSetKernelArg(device.chroma_transform, 0, sizeof(cl_mem), &device.current_frame_U);
		device.state_gpu = clSetKernelArg(device.chroma_transform, 1, sizeof(cl_mem), &device.ref_frame_U);
		device.state_gpu = clSetKernelArg(device.chroma_transform, 2, sizeof(cl_mem), &device.reconstructed_frame_U);
		device.state_gpu = clEnqueueNDRangeKernel(device.commandQueue_gpu, device.chroma_transform, 1, NULL, device.gpu_work_items_per_dim, NULL, 0, NULL, NULL);
		device.state_gpu = clFinish(device.commandQueue_gpu);
		
		block_place = 20;
		device.state_gpu = clSetKernelArg(device.chroma_transform, 7, sizeof(int32_t), &block_place);
		device.state_gpu = clSetKernelArg(device.chroma_transform, 0, sizeof(cl_mem), &device.current_frame_V);
		device.state_gpu = clSetKernelArg(device.chroma_transform, 1, sizeof(cl_mem), &device.ref_frame_V);
		device.state_gpu = clSetKernelArg(device.chroma_transform, 2, sizeof(cl_mem), &device.reconstructed_frame_V);
		device.state_gpu = device.state_gpu = clEnqueueNDRangeKernel(device.commandQueue_gpu, device.chroma_transform, 1, NULL, device.gpu_work_items_per_dim, NULL, 0, NULL, NULL);
		device.state_gpu = clFinish(device.commandQueue_gpu);

		//count SSIM
		device.gpu_work_items_per_dim[0] = video.mb_count;
		//U
		device.state_gpu = clSetKernelArg(device.count_SSIM_chroma, 0, sizeof(cl_mem), &device.current_frame_U);
		device.state_gpu = clSetKernelArg(device.count_SSIM_chroma, 1, sizeof(cl_mem), &device.reconstructed_frame_U);
		device.state_gpu = clEnqueueNDRangeKernel(device.commandQueue_gpu, device.count_SSIM_chroma, 1, NULL, device.gpu_work_items_per_dim, NULL, 0, NULL, NULL);
		device.state_gpu = clFinish(device.commandQueue_gpu);
		//V
		device.state_gpu = clSetKernelArg(device.count_SSIM_chroma, 0, sizeof(cl_mem), &device.current_frame_V);
		device.state_gpu = clSetKernelArg(device.count_SSIM_chroma, 1, sizeof(cl_mem), &device.reconstructed_frame_V);
		device.state_gpu = clEnqueueNDRangeKernel(device.commandQueue_gpu, device.count_SSIM_chroma, 1, NULL, device.gpu_work_items_per_dim, NULL, 0, NULL, NULL);
		device.state_gpu = clFinish(device.commandQueue_gpu);
		//Y
		device.state_gpu = clEnqueueNDRangeKernel(device.commandQueue_gpu, device.count_SSIM_luma, 1, NULL, device.gpu_work_items_per_dim, NULL, 0, NULL, NULL);
		device.state_gpu = clFinish(device.commandQueue_gpu);
	}

	device.state_gpu = clFinish(device.commandQueue_gpu);
    return;
}

void prepare_filter_mask_and_non_zero_coeffs()
{
	int mb_num;
	frames.skip_prob = 0;

	if (video.GOP_size > 1)
	{
		// we need to grab transformed data from host memory
		device.state_gpu = clEnqueueWriteBuffer(device.commandQueue_gpu, device.transformed_blocks_gpu, CL_TRUE, 0, video.mb_count*sizeof(macroblock), frames.transformed_blocks, 0, NULL, NULL);
		device.gpu_work_items_per_dim[0] = video.mb_count;
		device.state_gpu = clEnqueueNDRangeKernel(device.commandQueue_gpu, device.prepare_filter_mask, 1, NULL, device.gpu_work_items_per_dim, NULL, 0, NULL, NULL);
		// need to return info about non_zero coeffs
		device.state_gpu = clEnqueueReadBuffer(device.commandQueue_gpu, device.transformed_blocks_gpu ,CL_TRUE, 0, video.mb_count*sizeof(macroblock), frames.transformed_blocks, 0, NULL, NULL);
	}
	else //have to count on CPU
	{
		int b_num, coeff_num, b_last;
		frames.skip_prob = 0;
		for(mb_num = 0; mb_num < video.mb_count; ++mb_num) 
		{
			frames.transformed_blocks[mb_num].non_zero_coeffs = 0;
			b_last = (frames.transformed_blocks[mb_num].parts == are16x16) ? 25 : 24;

			for (b_num = 0; b_num < b_last; ++b_num) {
				int first_coeff = (frames.transformed_blocks[mb_num].parts == are16x16) ? 1 : 0;
				first_coeff = (b_num >= 16) ? 0 : first_coeff;
				for (coeff_num = first_coeff; coeff_num < 16; ++coeff_num)
					frames.transformed_blocks[mb_num].non_zero_coeffs += (frames.transformed_blocks[mb_num].coeffs[b_num][coeff_num] < 0) ?
																			-frames.transformed_blocks[mb_num].coeffs[b_num][coeff_num] :	
																			frames.transformed_blocks[mb_num].coeffs[b_num][coeff_num];
			}
		}
	}
	for(mb_num = 0; mb_num < video.mb_count; ++mb_num) 
		if (frames.transformed_blocks[mb_num].non_zero_coeffs > 0) 
			++frames.skip_prob;

	frames.skip_prob *= 256;
	frames.skip_prob /= video.mb_count;
	frames.skip_prob = (frames.skip_prob > 254) ? 254 : frames.skip_prob;
	frames.skip_prob = (frames.skip_prob < 2) ? 2 : frames.skip_prob;
	//don't do this => frames.skip_prob = 255 - frames.skip_prob; incorrect desription of prob_skip_false
	return;
}

void check_SSIM()
{
	device.state_gpu = clEnqueueReadBuffer(device.commandQueue_gpu, device.reconstructed_frame_Y ,CL_TRUE, 0, video.wrk_frame_size_luma, frames.reconstructed_Y, 0, NULL, NULL);
	device.state_gpu = clEnqueueReadBuffer(device.commandQueue_gpu, device.reconstructed_frame_U ,CL_TRUE, 0, video.wrk_frame_size_chroma, frames.reconstructed_U, 0, NULL, NULL);
	device.state_gpu = clEnqueueReadBuffer(device.commandQueue_gpu, device.reconstructed_frame_V ,CL_TRUE, 0, video.wrk_frame_size_chroma, frames.reconstructed_V, 0, NULL, NULL);

	frames.new_SSIM = 0;
	float min1 = 2, min2 = 2;
	int mb_num;
	frames.replaced = 0;
	for (mb_num = 0; mb_num < video.mb_count; ++mb_num)
	{
		min2 = (frames.transformed_blocks[mb_num].SSIM < min2) ? frames.transformed_blocks[mb_num].SSIM : min2;
		//if (frames.transformed_blocks[mb_num].SSIM < video.SSIM_target) 
		//	frames.e_data[mb_num].is_inter_mb = (test_inter_on_intra(mb_num, AQ_segment) == 0) ? 0 : frames.e_data[mb_num].is_inter_mb;
		//if (frames.transformed_blocks[mb_num].SSIM < video.SSIM_target) 
		//	frames.e_data[mb_num].is_inter_mb = (test_inter_on_intra(mb_num, HQ_segment) == 0) ? 0 : frames.e_data[mb_num].is_inter_mb;
		if (frames.transformed_blocks[mb_num].SSIM < video.SSIM_target) 
			frames.e_data[mb_num].is_inter_mb = (test_inter_on_intra(mb_num, UQ_segment) == 0) ? 0 : frames.e_data[mb_num].is_inter_mb;
		frames.replaced+=(frames.e_data[mb_num].is_inter_mb==0);

		frames.new_SSIM += frames.transformed_blocks[mb_num].SSIM;
		min1 = (frames.transformed_blocks[mb_num].SSIM < min1) ? frames.transformed_blocks[mb_num].SSIM : min1;
	}
	if (frames.replaced > 0)
	{
		device.state_gpu = clEnqueueWriteBuffer(device.commandQueue_gpu, device.reconstructed_frame_Y, CL_TRUE, 0, video.wrk_frame_size_luma, frames.reconstructed_Y, 0, NULL, NULL);
		device.state_gpu = clEnqueueWriteBuffer(device.commandQueue_gpu, device.reconstructed_frame_U, CL_TRUE, 0, video.wrk_frame_size_chroma, frames.reconstructed_U, 0, NULL, NULL);
		device.state_gpu = clEnqueueWriteBuffer(device.commandQueue_gpu, device.reconstructed_frame_V, CL_TRUE, 0, video.wrk_frame_size_chroma, frames.reconstructed_V, 0, NULL, NULL);
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

int main(int argc, char *argv[])
{
	int32_t mb_num;
	printf("\n");

	error_file.path = ERRORPATH;
	if (ParseArgs(argc, argv) < 0) 
	{
		printf("\npress any key to quit\n");
		getch(); return -1;
	}
	
    OpenYUV420FileAndParseHeader();
    
	printf("initialization started;\n");
    init_all();
	if ((device.state_cpu != 0) || (device.state_gpu != 0)) {
		printf("\npress any key to quit\n");
		getch(); return -1; 
	}
	printf("initialization complete;\n");

    frames.frames_until_key = 1;
	frames.frames_until_altref = 2;
    frames.frame_number = 0;
	
	write_output_header();
	//open_dump_file();

    while (get_yuv420_frame() > 0)
    {
		frames.prev_is_key_frame = frames.current_is_key_frame; 
		frames.prev_is_altref_frame = frames.current_is_altref_frame;
		--frames.frames_until_key;
		--frames.frames_until_altref;
		frames.current_is_key_frame = (frames.frames_until_key < 1);
		frames.current_is_altref_frame = (frames.frames_until_altref < 1);

		prepare_segments_data();

        if (frames.current_is_key_frame)
        {
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
			device.state_gpu = clEnqueueWriteBuffer(device.commandQueue_gpu, device.current_frame_Y, CL_TRUE, 0, video.wrk_frame_size_luma, frames.current_Y, 0, NULL, NULL);
			device.state_gpu = clEnqueueWriteBuffer(device.commandQueue_gpu, device.current_frame_U, CL_TRUE, 0, video.wrk_frame_size_chroma, frames.current_U, 0, NULL, NULL);
			device.state_gpu = clEnqueueWriteBuffer(device.commandQueue_gpu, device.current_frame_V, CL_TRUE, 0, video.wrk_frame_size_chroma, frames.current_V, 0, NULL, NULL);
			inter_transform(); 
			device.state_gpu = clFinish(device.commandQueue_gpu);
			// copy transformed_blocks to host
			device.state_gpu = clEnqueueReadBuffer(device.commandQueue_gpu, device.transformed_blocks_gpu ,CL_TRUE, 0, video.mb_count*sizeof(macroblock), frames.transformed_blocks, 0, NULL, NULL);

			for(mb_num = 0; mb_num < video.mb_count; ++mb_num) 
				frames.e_data[mb_num].is_inter_mb = 1;

			check_SSIM();
			if ((frames.replaced > (video.mb_count/4)) || (frames.new_SSIM < video.SSIM_target))
			{
				// redo as intra
				frames.current_is_key_frame = 1;
				prepare_segments_data();
				intra_transform();
				device.state_gpu = clEnqueueWriteBuffer(device.commandQueue_gpu, device.reconstructed_frame_Y, CL_TRUE, 0, video.wrk_frame_size_luma, frames.reconstructed_Y, 0, NULL, NULL);
				device.state_gpu = clEnqueueWriteBuffer(device.commandQueue_gpu, device.reconstructed_frame_U, CL_TRUE, 0, video.wrk_frame_size_chroma, frames.reconstructed_U, 0, NULL, NULL);
				device.state_gpu = clEnqueueWriteBuffer(device.commandQueue_gpu, device.reconstructed_frame_V, CL_TRUE, 0, video.wrk_frame_size_chroma, frames.reconstructed_V, 0, NULL, NULL);
				printf("key frame FORCED by bad inter-result!\n");
			}

			// we will move reconstructed buffers to last while interpolating later in inter_transform();
			device.state_gpu = clFinish(device.commandQueue_gpu);
        }
		// searching for MBs to be skiped 
		prepare_filter_mask_and_non_zero_coeffs();
		do_loop_filter();
		//dump();

		device.state_cpu = clEnqueueWriteBuffer(device.commandQueue_cpu, device.transformed_blocks_cpu, CL_TRUE, 0, video.mb_count*sizeof(macroblock), frames.transformed_blocks, 0, NULL, NULL); 
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
		clReleaseMemObject(device.ref_frame_U);
		clReleaseMemObject(device.ref_frame_V);
		clReleaseMemObject(device.ref_frame_Y_interpolated);
		clReleaseMemObject(device.ref_frame_Y_downsampled_by2);
		clReleaseMemObject(device.ref_frame_Y_downsampled_by4);
		clReleaseMemObject(device.ref_frame_Y_downsampled_by8);
		clReleaseMemObject(device.ref_frame_Y_downsampled_by16);
		clReleaseMemObject(device.transformed_blocks_gpu);
		clReleaseMemObject(device.segments_data);
		clReleaseMemObject(device.vnet1);
		clReleaseMemObject(device.vnet2);
		clReleaseMemObject(device.mb_mask);
		clReleaseMemObject(device.mb_metrics);
		clReleaseKernel(device.reset_vectors);
		clReleaseKernel(device.downsample);
		clReleaseKernel(device.luma_search_1step);
		clReleaseKernel(device.luma_search_2step);
		clReleaseKernel(device.luma_transform_8x8);
		clReleaseKernel(device.luma_transform_16x16);
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
