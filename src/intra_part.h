
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

static void zigzag_block(block_t *const block)
{
    //      zigzag[16] = { 0, 1, 4, 8, 5, 2, 3,  6, 9, 12, 13, 10, 7, 11, 14, 15 };
    //  inv_zigzag[16] = { 0, 1, 5, 6, 2, 4, 7, 12, 3,  8, 11, 13, 9, 10, 14, 15 };

    cl_short tmp1, tmp2, tmp3;

	tmp1 = block->coeff[2];
    tmp2 = block->coeff[3];
    tmp3 = block->coeff[10];
    block->coeff[2] = block->coeff[4];
    block->coeff[3] = block->coeff[8];
    block->coeff[10] = block->coeff[13];
    block->coeff[4] = block->coeff[5];
    block->coeff[8] = block->coeff[9];
    block->coeff[13] = block->coeff[11];
    block->coeff[5] = tmp1;
    block->coeff[9] = block->coeff[12];
    block->coeff[11] = tmp3;
    block->coeff[12] = block->coeff[7];
    block->coeff[7] = block->coeff[6];
    block->coeff[6] = tmp2;

    return;
}

static const int cospi8sqrt2minus1=20091;
static const int sinpi8sqrt2 =35468;

static void iDCT4x4(const block_t *const __restrict input, cl_uchar *const __restrict output, const cl_uchar *const __restrict predictor, const cl_int dc_q, const cl_int ac_q)
{
    int i;
    int a1, b1, c1, d1;
	int ip0, ip4, ip8, ip12, q;
    cl_short tmp_block[16];
	const short *ip = &input->coeff[0];
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

    cl_uchar *op = output;
	const cl_uchar *pr = predictor;
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
        tp[0] = ((a1 + d1 + 4) >> 3) + pr[0];
		op[0] = (cl_uchar)((tp[0] > 255) ? 255 : ((tp[0] < 0) ? 0 : tp[0] ));
		tp[3] = ((a1 - d1 + 4) >> 3) + pr[3];
        op[3] = (cl_uchar)((tp[3] > 255) ? 255 : ((tp[3] < 0) ? 0 : tp[3] ));
        tp[1] = ((b1 + c1 + 4) >> 3) + pr[1];
        op[1] = (cl_uchar)((tp[1] > 255) ? 255 : ((tp[1] < 0) ? 0 : tp[1] ));
        tp[2] = ((b1 - c1 + 4) >> 3) + pr[2];
        op[2] = (cl_uchar)((tp[2] > 255) ? 255 : ((tp[2] < 0) ? 0 : tp[2] ));

        op += 4;
		tp += 4;
		pr += 4;
    }

	return;
}


static void DCT4x4(const cl_short *const __restrict input, block_t *const __restrict output)
{
    // input - pointer to start of block in raw frame. I-line of the block will be input + I*width
    // output - pointer to encoded_macroblock.block[i] data.
    cl_int i;
    cl_int a1, b1, c1, d1;
    const cl_short *ip = input;
	cl_short *op = &output->coeff[0];

    for (i = 0; i < 4; i++)
    {
        a1 = ((ip[0] + ip[3])<<3);
        b1 = ((ip[1] + ip[2])<<3);
        c1 = ((ip[1] - ip[2])<<3);
        d1 = ((ip[0] - ip[3])<<3);

        op[0] = (cl_short)(a1 + b1);
        op[2] = (cl_short)(a1 - b1);

        op[1] = (cl_short)((c1 * 2217 + d1 * 5352 +  14500)>>12);
        op[3] = (cl_short)((d1 * 2217 - c1 * 5352 +   7500)>>12);

        ip += 4; 
        op += 4; 

    }
	op = &output->coeff[0];

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

static int weight(const cl_short *const r) //r - 4x4 residual to be weighted through vp8 DCT
{
    // input - pointer to start of block in raw frame. I-line of the block will be input + I*width
    // output - pointer to encoded_macroblock.block[i] data.

	int i;
    int a1, b1, c1, d1;
	const short *ip = r;
	short tmp[16];
    short *op = tmp;

    for (i = 0; i < 4; i++)
    {
        a1 = ((ip[0] + ip[3])<<3);
        b1 = ((ip[1] + ip[2])<<3);
        c1 = ((ip[1] - ip[2])<<3);
        d1 = ((ip[0] - ip[3])<<3);

        op[0] = (cl_short)(a1 + b1);
        op[2] = (cl_short)(a1 - b1);

        op[1] = (cl_short)((c1 * 2217 + d1 * 5352 +  14500)>>12);
        op[3] = (cl_short)((d1 * 2217 - c1 * 5352 +   7500)>>12);

        ip += 4; 
        op += 4; 

    }
    op = tmp;

    for (i = 0; i < 4; i++)
    {
        a1 = op[0] + op[12];
        b1 = op[4] + op[8];
        c1 = op[4] - op[8];
		d1 = op[0] - op[12];

        op[0] = (( a1 + b1 + 7)>>4);
        op[8] = (( a1 - b1 + 7)>>4);
        op[4]  = (((c1 * 2217 + d1 * 5352 +  12000)>>16) + (d1!=0));
        op[12] = ((d1 * 2217 - c1 * 5352 +  51000)>>16);

        ++op;
    }

	tmp[0] /= 4;
	a1 = 0;
	for (i = 0; i < 16; ++i)
		a1 += (tmp[i] < 0) ? -tmp[i] : tmp[i];

	return a1;
}

static void quant4x4(block_t *const block, const cl_int dc_q, const cl_int ac_q)
{
	cl_short *const coeffs = &block->coeff[0];
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

	coeffs[0] /= (cl_short)dc_q;
	coeffs[1] /= (cl_short)ac_q;
	coeffs[2] /= (cl_short)ac_q;
	coeffs[3] /= (cl_short)ac_q;
	coeffs[4] /= (cl_short)ac_q;
	coeffs[5] /= (cl_short)ac_q;
	coeffs[6] /= (cl_short)ac_q;
	coeffs[7] /= (cl_short)ac_q;
	coeffs[8] /= (cl_short)ac_q;
	coeffs[9] /= (cl_short)ac_q;
	coeffs[10] /= (cl_short)ac_q;
	coeffs[11] /= (cl_short)ac_q;
	coeffs[12] /= (cl_short)ac_q;
	coeffs[13] /= (cl_short)ac_q;
	coeffs[14] /= (cl_short)ac_q;
	coeffs[15] /= (cl_short)ac_q;
	return;
}

static cl_int pick_luma_predictor(const cl_uchar *const __restrict original, 
								  cl_uchar *const __restrict predictor, 
								  cl_short *const __restrict residual, 
								  const cl_short *const __restrict top_pred, 
								  const cl_short *const __restrict left_pred, 
								  const cl_short top_left_pred)
{
	cl_short MinWeight, bmode, val, buf, i, j, W;
	cl_uchar pr_tmp[16];
	cl_short res_tmp[16];
// set B_DC_PRED as s start
	bmode = B_DC_PRED;
	val = 4;
	for (i = 0; i < 4; ++i)
		val += top_pred[i] + left_pred[i];
	val >>= 3;
	for (i = 0; i < 4; ++i) 
		for (j = 0; j < 4; ++j) {
		predictor[i*4 + j] = (cl_uchar)val; 
		residual[i*4 + j] = (cl_short)original[i*4+j] - (cl_short)predictor[i*4+j];
	}
	MinWeight = weight(residual);
// try B_TM_PRED
	for (i = 0; i < 4; ++i)
		for (j = 0; j < 4; ++j) 
		{
			val = top_pred[j] + left_pred[i] - top_left_pred;
			val = (val < 0) ? 0 : val; val = (val > 255) ? 255 : val;
			pr_tmp[i*4+j] = (cl_uchar)val;
			res_tmp[i*4 + j] = (cl_short)original[i*4+j] - (cl_short)pr_tmp[i*4+j];
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
		pr_tmp[j] = (cl_uchar)val;
		pr_tmp[4+j] = (cl_uchar)val;
		pr_tmp[8+j] = (cl_uchar)val;
		pr_tmp[12+j] = (cl_uchar)val;
		res_tmp[j] = (cl_short)original[j] - (cl_short)pr_tmp[j];
		res_tmp[4 + j] = (cl_short)original[4+j] - (cl_short)pr_tmp[4+j];
		res_tmp[8 + j] = (cl_short)original[8+j] - (cl_short)pr_tmp[8+j];
		res_tmp[12 + j] = (cl_short)original[12+j] - (cl_short)pr_tmp[12+j];
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
		pr_tmp[i*4] = (cl_uchar)val;
		pr_tmp[i*4+1] = (cl_uchar)val;
		pr_tmp[i*4+2] = (cl_uchar)val;
		pr_tmp[i*4+3] = (cl_uchar)val;
		res_tmp[i*4] = (cl_short)original[i*4] - (cl_short)pr_tmp[i*4];
		res_tmp[i*4 + 1] = (cl_short)original[i*4+1] - (cl_short)pr_tmp[i*4+1];
		res_tmp[i*4 + 2] = (cl_short)original[i*4+2] - (cl_short)pr_tmp[i*4+2];
		res_tmp[i*4 + 3] = (cl_short)original[i*4+3] - (cl_short)pr_tmp[i*4+3];
		buf = left_pred[i];
	} 
	// last row i==3
	val = left_pred[3]*3 + left_pred[i-1] + 2;
	val >>= 2;
	pr_tmp[12] = (cl_uchar)val;
	pr_tmp[13] = (cl_uchar)val;
	pr_tmp[14] = (cl_uchar)val;
	pr_tmp[15] = (cl_uchar)val;
	res_tmp[12] = (cl_short)original[12] - (cl_short)pr_tmp[12];
	res_tmp[13] = (cl_short)original[13] - (cl_short)pr_tmp[13];
	res_tmp[14] = (cl_short)original[14] - (cl_short)pr_tmp[14];
	res_tmp[15] = (cl_short)original[15] - (cl_short)pr_tmp[15];
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
	pr_tmp[0] = (cl_uchar)((top_pred[0]+top_pred[1]*2+top_pred[2]+2)>>2);
	pr_tmp[1] = pr_tmp[4] = (cl_uchar)((top_pred[1]+top_pred[2]*2+top_pred[3]+2)>>2);
	pr_tmp[2] = pr_tmp[5] = pr_tmp[8] = (cl_uchar)((top_pred[2]+top_pred[3]*2+top_pred[4]+2)>>2);
	pr_tmp[3] = pr_tmp[6] = pr_tmp[9] = pr_tmp[12] = (cl_uchar)((top_pred[3]+top_pred[4]*2+top_pred[5]+2)>>2);
	pr_tmp[7] = pr_tmp[10] = pr_tmp[13] = (cl_uchar)((top_pred[4]+top_pred[5]*2+top_pred[6]+2)>>2);
	pr_tmp[11] = pr_tmp[14] = (cl_uchar)((top_pred[5]+top_pred[6]*2+top_pred[7]+2)>>2);
	pr_tmp[15] = (cl_uchar)((top_pred[6]+top_pred[7]*3+2)>>2);
	for(i = 0; i < 4; ++i)
		for(j = 0; j < 4; ++j)
			res_tmp[i*4 + j] = (cl_short)original[i*4+j] - (cl_short)pr_tmp[i*4+j];
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
	pr_tmp[12] = (cl_uchar)((left_pred[3] + left_pred[2]*2 + left_pred[1] + 2)>>2);
	pr_tmp[13] = pr_tmp[8] = (cl_uchar)((left_pred[2] + left_pred[1]*2 + left_pred[0] + 2)>>2);
	pr_tmp[14] = pr_tmp[9] = pr_tmp[4] = (cl_uchar)((left_pred[1] + left_pred[0]*2 + top_left_pred + 2)>>2);
	pr_tmp[15] = pr_tmp[10] = pr_tmp[5] = pr_tmp[0] = (cl_uchar)((left_pred[0]+top_left_pred*2 + top_pred[0] + 2)>>2);
	pr_tmp[11] = pr_tmp[6] = pr_tmp[1] = (cl_uchar)((top_left_pred + top_pred[0]*2 + top_pred[1] + 2)>>2);
	pr_tmp[7] = pr_tmp[2] = (cl_uchar)((top_pred[0] + top_pred[1]*2 + top_pred[2] + 2)>>2);
	pr_tmp[3] = (cl_uchar)((top_pred[1] + top_pred[2]*2 + top_pred[3] + 2)>>2);
	for(i = 0; i < 4; ++i)
		for(j = 0; j < 4; ++j)
			res_tmp[i*4 + j] = (cl_short)original[i*4+j] - (cl_short)pr_tmp[i*4+j];
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
	pr_tmp[12] = (cl_uchar)((left_pred[2] + left_pred[1]*2 + left_pred[0] + 2)>>2);
	pr_tmp[8] = (cl_uchar)((left_pred[1] + left_pred[0]*2 + top_left_pred + 2)>>2);
	pr_tmp[13] = pr_tmp[4] = (cl_uchar)((left_pred[0] + top_left_pred*2 + top_pred[0] + 2)>>2);
	pr_tmp[9] = pr_tmp[0] = (cl_uchar)((top_left_pred + top_pred[0] + 1)>>1);
	pr_tmp[14] = pr_tmp[5] = (cl_uchar)((top_left_pred + top_pred[0]*2 + top_pred[1] + 2)>>2);
	pr_tmp[10] = pr_tmp[1] = (cl_uchar)((top_pred[0] + top_pred[1] + 1)>>1);
	pr_tmp[15] = pr_tmp[6] = (cl_uchar)((top_pred[0] + top_pred[1]*2 + top_pred[2] + 2)>>2);
	pr_tmp[11] = pr_tmp[2] = (cl_uchar)((top_pred[1] + top_pred[2] + 1)>>1);
	pr_tmp[7] = (cl_uchar)((top_pred[1] + top_pred[2]*2 + top_pred[3] + 2)>>2);
	pr_tmp[3] = (cl_uchar)((top_pred[2] + top_pred[3] + 1)>>1);
	for(i = 0; i < 4; ++i)
		for(j = 0; j < 4; ++j)
			res_tmp[i*4 + j] = (cl_short)original[i*4+j] - (cl_short)pr_tmp[i*4+j];
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
	pr_tmp[0] = (cl_uchar)((top_pred[0] + top_pred[1] + 1)>>1);
	pr_tmp[4] = (cl_uchar)((top_pred[0] + top_pred[1]*2 + top_pred[2] + 2)>>2);
	pr_tmp[8] = pr_tmp[1] = (cl_uchar)((top_pred[1] + top_pred[2] + 1)>>1);
	pr_tmp[12] = pr_tmp[5] = (cl_uchar)((top_pred[1] + top_pred[2]*2 + top_pred[3] + 2)>>2);
	pr_tmp[9] = pr_tmp[2] = (cl_uchar)((top_pred[2] + top_pred[3] + 1)>>1);
	pr_tmp[13] = pr_tmp[6] = (cl_uchar)((top_pred[2] + top_pred[3]*2 + top_pred[4] + 2)>>2);
	pr_tmp[10] = pr_tmp[3] = (cl_uchar)((top_pred[3] + top_pred[4] + 1)>>1);
	pr_tmp[14] = pr_tmp[7] = (cl_uchar)((top_pred[3] + top_pred[4]*2 + top_pred[5] + 2)>>2);
	/* Last two values do not strictly follow the pattern. */
	pr_tmp[11] = (cl_uchar)((top_pred[4] + top_pred[5]*2 + top_pred[6] + 2)>>2);
	pr_tmp[15] = (cl_uchar)((top_pred[5] + top_pred[6]*2 + top_pred[7] + 2)>>2);
	for(i = 0; i < 4; ++i)
		for(j = 0; j < 4; ++j)
			res_tmp[i*4 + j] = (cl_short)original[i*4+j] - (cl_short)pr_tmp[i*4+j];
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
	pr_tmp[12] = (cl_uchar)((left_pred[3] + left_pred[2] + 1)>>1);
	pr_tmp[13] = (cl_uchar)((left_pred[3] + left_pred[2]*2 + left_pred[1] + 2)>>2);
	pr_tmp[8] = pr_tmp[14] = (cl_uchar)((left_pred[2] + left_pred[1] + 1)>>1);
	pr_tmp[9] = pr_tmp[15] = (cl_uchar)((left_pred[2] + left_pred[1]*2 + left_pred[0] + 2)>>2);
	pr_tmp[4] = pr_tmp[10] = (cl_uchar)((left_pred[1] + left_pred[0] + 1)>>1);
	pr_tmp[5] = pr_tmp[11] = (cl_uchar)((left_pred[1] + left_pred[0]*2 + top_left_pred + 2)>>2);
	pr_tmp[0] = pr_tmp[6] = (cl_uchar)((left_pred[0] + top_left_pred + 1)>>1);
	pr_tmp[1] = pr_tmp[7] = (cl_uchar)((left_pred[0] + top_left_pred*2 + top_pred[0] + 2)>>2);
	pr_tmp[2] = (cl_uchar)((top_left_pred + top_pred[0]*2 + top_pred[1] + 2)>>2);
	pr_tmp[3] = (cl_uchar)((top_pred[0] + top_pred[1]*2 + top_pred[2] + 2)>>2);
	for(i = 0; i < 4; ++i)
		for(j = 0; j < 4; ++j)
			res_tmp[i*4 + j] = (cl_short)original[i*4+j] - (cl_short)pr_tmp[i*4+j];
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
	pr_tmp[0] = (cl_uchar)((left_pred[0] + left_pred[1] + 1)>>1);
	pr_tmp[1] = (cl_uchar)((left_pred[0] + left_pred[1]*2 + left_pred[2] + 2)>>2);
	pr_tmp[2] = pr_tmp[4] = (cl_uchar)((left_pred[1] + left_pred[2] + 1)>>1);
	pr_tmp[3] = pr_tmp[5] = (cl_uchar)((left_pred[1] + left_pred[2]*2 + left_pred[3] + 2)>>2);
	pr_tmp[6] = pr_tmp[8] = (cl_uchar)((left_pred[2] + left_pred[3] + 1)>>1);
	pr_tmp[7] = pr_tmp[9] = (cl_uchar)((left_pred[2] + left_pred[3]*3 + 2)>>2);
	/* Not possible to follow pattern for much of the bottom
	row because no (nearby) already-constructed pixels lie
	on the diagonals in question. */
	pr_tmp[10] = pr_tmp[11] = pr_tmp[12] = pr_tmp[13] = pr_tmp[14] = pr_tmp[15] = (cl_uchar)left_pred[3];
	for(i = 0; i < 4; ++i)
		for(j = 0; j < 4; ++j)
			res_tmp[i*4 + j] = (cl_short)original[i*4+j] - (cl_short)pr_tmp[i*4+j];
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

static void predict_and_transform_mb(const cl_int mb_num)
{
	frames.MB_parts[mb_num] = are4x4;
    cl_int i, b_num, b_col, b_row, pred_ind_Y, pred_ind_UV;

    cl_short top_left_pred_Y, top_left_pred_U, top_left_pred_V;
    cl_short left_pred_Y[16], left_pred_U[8], left_pred_V[8], top_pred_Y[20], top_pred_U[8], top_pred_V[8];

	cl_uchar predictor[16], block_pixels[16];
	cl_short residual[16];

	frames.MB_segment_id[mb_num] = intra_segment;
	const cl_int y_dc_q = frames.y_dc_q[intra_segment];
	const cl_int y_ac_q = frames.y_ac_q[intra_segment];
	const cl_int uv_dc_q = frames.uv_dc_q[intra_segment];
	const cl_int uv_ac_q = frames.uv_ac_q[intra_segment];

    const cl_int mb_row = mb_num / video.mb_width; 
	const cl_int mb_col = mb_num % video.mb_width;
    const cl_int Y_offset = ((mb_row * video.wrk_width) + mb_col) << 4;
    const cl_int UV_offset = ((mb_row *video.wrk_width) << 2) + (mb_col << 3);

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
            left_pred_Y[i]= (cl_short)frames.reconstructed_Y[pred_ind_Y];
            pred_ind_Y += video.wrk_width;
            left_pred_Y[i+1]= (cl_short)frames.reconstructed_Y[pred_ind_Y];
            pred_ind_Y += video.wrk_width;
            left_pred_U[i/2]= (cl_short)frames.reconstructed_U[pred_ind_UV];
            left_pred_V[i/2]=  (cl_short)frames.reconstructed_V[pred_ind_UV];
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
            top_pred_Y[i]= (cl_short)frames.reconstructed_Y[pred_ind_Y];
            ++pred_ind_Y;
            top_pred_Y[i+1]= (cl_short)frames.reconstructed_Y[pred_ind_Y];
            ++pred_ind_Y;
            top_pred_U[i/2]= (cl_short)frames.reconstructed_U[pred_ind_UV];
            top_pred_V[i/2]=  (cl_short)frames.reconstructed_V[pred_ind_UV];
            ++pred_ind_UV;
        }
		if (mb_col < (video.mb_width - 1)) {
			top_pred_Y[16] = (cl_short)frames.reconstructed_Y[pred_ind_Y];
			top_pred_Y[17] = (cl_short)frames.reconstructed_Y[pred_ind_Y + 1];
			top_pred_Y[18] = (cl_short)frames.reconstructed_Y[pred_ind_Y + 2];
			top_pred_Y[19] = (cl_short)frames.reconstructed_Y[pred_ind_Y + 3];
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

    cl_uchar *block_offset;

    for (b_row = 0; b_row < 4; ++b_row) // 4x4 luma blocks
    {
		cl_short buf_pred = left_pred_Y[(b_row<<2) + 3];
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
			DCT4x4(residual, &frames.MB[mb_num].block[b_num]);
			quant4x4(&frames.MB[mb_num].block[b_num], y_dc_q, y_ac_q);
			iDCT4x4(&frames.MB[mb_num].block[b_num], block_pixels, predictor, y_dc_q, y_ac_q);
			block_offset = frames.reconstructed_Y + (Y_offset + (((b_row*video.wrk_width) + b_col ) <<2));
			for(i = 0; i < 4; ++i)
			{
				*(block_offset + i*video.wrk_width) = block_pixels[i*4];
				*(block_offset + i*video.wrk_width + 1) = block_pixels[i*4 + 1];
				*(block_offset + i*video.wrk_width + 2) = block_pixels[i*4 + 2];
				*(block_offset + i*video.wrk_width + 3) = block_pixels[i*4 + 3];
			}
			zigzag_block(&frames.MB[mb_num].block[b_num]);

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
			predictor[i*4] = (cl_uchar)(residual[i*4] < 0) ? 0 : ((residual[i*4] > 255) ? 255 : residual[i*4]);
			predictor[i*4 + 1] = (cl_uchar)(residual[i*4 + 1] < 0) ? 0 : ((residual[i*4 + 1] > 255) ? 255 : residual[i*4 + 1]);
			predictor[i*4 + 2] = (cl_uchar)(residual[i*4 + 2] < 0) ? 0 : ((residual[i*4 + 2] > 255) ? 255 : residual[i*4 + 2]);
			predictor[i*4 + 3] = (cl_uchar)(residual[i*4 + 3] < 0) ? 0 : ((residual[i*4 + 3] > 255) ? 255 : residual[i*4 + 3]);
			residual[i*4] = (cl_short)*(block_offset) - (cl_short)predictor[i*4];
			residual[i*4 + 1] = (cl_short)*(block_offset + 1) - (cl_short)predictor[i*4 + 1];
			residual[i*4 + 2] = (cl_short)*(block_offset + 2) - (cl_short)predictor[i*4 + 2];
			residual[i*4 + 3] = (cl_short)*(block_offset + 3) - (cl_short)predictor[i*4 + 3];
			block_offset += video.wrk_width/2;
		}
		DCT4x4(residual, &frames.MB[mb_num].block[b_num+16]);
		quant4x4(&frames.MB[mb_num].block[b_num+16], uv_dc_q, uv_ac_q);
		iDCT4x4(&frames.MB[mb_num].block[b_num+16], block_pixels, predictor, uv_dc_q, uv_ac_q);
		block_offset = frames.reconstructed_U + (UV_offset + (( b_row*video.wrk_width<<1) + (b_col<<2) ));
		for(i = 0; i < 4; ++i)
		{
			*(block_offset + i*video.wrk_width/2) = block_pixels[i*4];
			*(block_offset + i*video.wrk_width/2 + 1) = block_pixels[i*4 + 1];
			*(block_offset + i*video.wrk_width/2 + 2) = block_pixels[i*4 + 2];
			*(block_offset + i*video.wrk_width/2 + 3) = block_pixels[i*4 + 3];
		}
        zigzag_block(&frames.MB[mb_num].block[b_num+16]);
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
			predictor[i*4] = (cl_uchar)(residual[i*4] < 0) ? 0 : ((residual[i*4] > 255) ? 255 : residual[i*4]);
			predictor[i*4 + 1] = (cl_uchar)(residual[i*4 + 1] < 0) ? 0 : ((residual[i*4 + 1] > 255) ? 255 : residual[i*4 + 1]);
			predictor[i*4 + 2] = (cl_uchar)(residual[i*4 + 2] < 0) ? 0 : ((residual[i*4 + 2] > 255) ? 255 : residual[i*4 + 2]);
			predictor[i*4 + 3] = (cl_uchar)(residual[i*4 + 3] < 0) ? 0 : ((residual[i*4 + 3] > 255) ? 255 : residual[i*4 + 3]);
			residual[i*4] = (cl_short)*(block_offset) - (cl_short)predictor[i*4];
			residual[i*4 + 1] = (cl_short)*(block_offset + 1) - (cl_short)predictor[i*4 + 1];
			residual[i*4 + 2] = (cl_short)*(block_offset + 2) - (cl_short)predictor[i*4 + 2];
			residual[i*4 + 3] = (cl_short)*(block_offset + 3) - (cl_short)predictor[i*4 + 3];
			block_offset += video.wrk_width/2;
		}
		DCT4x4(residual, &frames.MB[mb_num].block[b_num+20]);
		quant4x4(&frames.MB[mb_num].block[b_num+20], uv_dc_q, uv_ac_q);
		iDCT4x4(&frames.MB[mb_num].block[b_num+20], block_pixels, predictor, uv_dc_q, uv_ac_q);
		block_offset = frames.reconstructed_V + (UV_offset + (( b_row*video.wrk_width<<1) + (b_col<<2) ));
		for(i = 0; i < 4; ++i)
		{
			*(block_offset + i*video.wrk_width/2) = block_pixels[i*4];
			*(block_offset + i*video.wrk_width/2 + 1) = block_pixels[i*4 + 1];
			*(block_offset + i*video.wrk_width/2 + 2) = block_pixels[i*4 + 2];
			*(block_offset + i*video.wrk_width/2 + 3) = block_pixels[i*4 + 3];
		}
        zigzag_block(&frames.MB[mb_num].block[b_num+20]);
    }

    return;
}


static float count_SSIM_16x16(const cl_uchar *const __restrict frame1Y, 
							  const cl_uchar *const __restrict frame1U, 
							  const cl_uchar *const __restrict frame1V, 
							  const cl_int width1, 
							  const cl_uchar *const __restrict frame2Y, 
							  const cl_uchar *const __restrict frame2U, 
							  const cl_uchar *const __restrict frame2V, 
							  const cl_int width2)
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

	M1 -= M2;
	M1 = (M1 < 0) ? -M1 : M1;
	M1f = (M1 > 4) ? (float)M1*0.02f : 0.0f;
	ssim -= M1f;

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

	M1 -= M2;
	M1 = (M1 < 0) ? -M1 : M1;
	M1f = (M1 > 4) ? (float)M1*0.02f : 0.0f;
	ssim -= M1f;

	ssim /= 3;

	return ssim;
}


static int test_inter_on_intra(const cl_int mb_num, const segment_id_t id)
{
	//macroblock test_mb;
	macroblock_coeffs_t test_mb;
	float test_SSIM;
	static cl_uchar test_recon_Y[256];
	static cl_uchar test_recon_V[64];
	static cl_uchar test_recon_U[64];
	const cl_int test_width = 16;

	cl_uchar predictor[16], block_pixels[16];
	cl_short residual[16];

	cl_int y_dc_q,y_ac_q,uv_dc_q,uv_ac_q;
	y_dc_q = frames.y_dc_q[id];
	y_ac_q = frames.y_ac_q[id];
	uv_dc_q = frames.uv_dc_q[id];
	uv_ac_q = frames.uv_ac_q[id];

	// prepare predictors (TM_B_PRED) and transform each block
    cl_int i,j, mb_row, mb_col, b_num, b_col, b_row, Y_offset, UV_offset, pred_ind_Y, pred_ind_UV;

    cl_short top_left_pred_Y, top_left_pred_U, top_left_pred_V;
    cl_short left_pred_Y[16], left_pred_U[8], left_pred_V[8];
    cl_short top_pred_Y[20], top_pred_U[8], top_pred_V[8];

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
            left_pred_Y[i]= (cl_short)frames.reconstructed_Y[pred_ind_Y];
            pred_ind_Y += video.wrk_width;
            left_pred_Y[i+1]= (cl_short)frames.reconstructed_Y[pred_ind_Y];
            pred_ind_Y += video.wrk_width;
            left_pred_U[i/2]= (cl_short)frames.reconstructed_U[pred_ind_UV];
            left_pred_V[i/2]=  (cl_short)frames.reconstructed_V[pred_ind_UV];
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
            top_pred_Y[i]= (cl_short)frames.reconstructed_Y[pred_ind_Y];
            ++pred_ind_Y;
            top_pred_Y[i+1]= (cl_short)frames.reconstructed_Y[pred_ind_Y];
            ++pred_ind_Y;
            top_pred_U[i/2]= (cl_short)frames.reconstructed_U[pred_ind_UV];
            top_pred_V[i/2]=  (cl_short)frames.reconstructed_V[pred_ind_UV];
            ++pred_ind_UV;
        }
		if (mb_col < (video.mb_width - 1)) {
			top_pred_Y[16] = (cl_short)frames.reconstructed_Y[pred_ind_Y];
			top_pred_Y[17] = (cl_short)frames.reconstructed_Y[pred_ind_Y + 1];
			top_pred_Y[18] = (cl_short)frames.reconstructed_Y[pred_ind_Y + 2];
			top_pred_Y[19] = (cl_short)frames.reconstructed_Y[pred_ind_Y + 3];
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

    cl_uchar *block_offset, *test_offset;
    for (b_row = 0; b_row < 4; ++b_row) // 4x4 luma blocks
    {
		cl_short buf_pred = left_pred_Y[(b_row<<2) + 3];
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
			DCT4x4(residual, &test_mb.block[b_num]);
			quant4x4(&test_mb.block[b_num], y_dc_q, y_ac_q);
			iDCT4x4(&test_mb.block[b_num], block_pixels, predictor, y_dc_q, y_ac_q);
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
			predictor[i*4] = (cl_uchar)(residual[i*4] < 0) ? 0 : ((residual[i*4] > 255) ? 255 : residual[i*4]);
			predictor[i*4 + 1] = (cl_uchar)(residual[i*4 + 1] < 0) ? 0 : ((residual[i*4 + 1] > 255) ? 255 : residual[i*4 + 1]);
			predictor[i*4 + 2] = (cl_uchar)(residual[i*4 + 2] < 0) ? 0 : ((residual[i*4 + 2] > 255) ? 255 : residual[i*4 + 2]);
			predictor[i*4 + 3] = (cl_uchar)(residual[i*4 + 3] < 0) ? 0 : ((residual[i*4 + 3] > 255) ? 255 : residual[i*4 + 3]);
			residual[i*4] = (cl_short)*(block_offset) - (cl_short)predictor[i*4];
			residual[i*4 + 1] = (cl_short)*(block_offset + 1) - (cl_short)predictor[i*4 + 1];
			residual[i*4 + 2] = (cl_short)*(block_offset + 2) - (cl_short)predictor[i*4 + 2];
			residual[i*4 + 3] = (cl_short)*(block_offset + 3) - (cl_short)predictor[i*4 + 3];
			block_offset += video.wrk_width/2;
		}
		DCT4x4(residual, &test_mb.block[b_num+16]);
		quant4x4(&test_mb.block[b_num+16], uv_dc_q, uv_ac_q);
		iDCT4x4(&test_mb.block[b_num+16], block_pixels, predictor, uv_dc_q, uv_ac_q);
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
			predictor[i*4] = (cl_uchar)(residual[i*4] < 0) ? 0 : ((residual[i*4] > 255) ? 255 : residual[i*4]);
			predictor[i*4 + 1] = (cl_uchar)(residual[i*4 + 1] < 0) ? 0 : ((residual[i*4 + 1] > 255) ? 255 : residual[i*4 + 1]);
			predictor[i*4 + 2] = (cl_uchar)(residual[i*4 + 2] < 0) ? 0 : ((residual[i*4 + 2] > 255) ? 255 : residual[i*4 + 2]);
			predictor[i*4 + 3] = (cl_uchar)(residual[i*4 + 3] < 0) ? 0 : ((residual[i*4 + 3] > 255) ? 255 : residual[i*4 + 3]);
			residual[i*4] = (cl_short)*(block_offset) - (cl_short)predictor[i*4];
			residual[i*4 + 1] = (cl_short)*(block_offset + 1) - (cl_short)predictor[i*4 + 1];
			residual[i*4 + 2] = (cl_short)*(block_offset + 2) - (cl_short)predictor[i*4 + 2];
			residual[i*4 + 3] = (cl_short)*(block_offset + 3) - (cl_short)predictor[i*4 + 3];
			block_offset += video.wrk_width/2;
		}
		DCT4x4(residual, &test_mb.block[b_num+20]);
		quant4x4(&test_mb.block[b_num+20], uv_dc_q, uv_ac_q);
		iDCT4x4(&test_mb.block[b_num+20], block_pixels, predictor, uv_dc_q, uv_ac_q);
		for(i = 0; i < 4; ++i)
		{
			*(test_offset + i*test_width/2) = block_pixels[i*4];
			*(test_offset + i*test_width/2 + 1) = block_pixels[i*4 + 1];
			*(test_offset + i*test_width/2 + 2) = block_pixels[i*4 + 2];
			*(test_offset + i*test_width/2 + 3) = block_pixels[i*4 + 3];
		}
    }
	
	test_SSIM = count_SSIM_16x16(test_recon_Y, test_recon_U, test_recon_V, test_width, 
									frames.current_Y + Y_offset, frames.current_U + UV_offset, frames.current_V + UV_offset, video.wrk_width);
	if (test_SSIM > frames.MB_SSIM[mb_num])
	{
		frames.MB_parts[mb_num] = are4x4;
		frames.MB_SSIM[mb_num] = test_SSIM;
		frames.MB_segment_id[mb_num] = id;
		//then we replace inter encoded and reconstructed MB with intra
		// replace residual
		memcpy(&frames.MB[mb_num], &test_mb, sizeof(macroblock_coeffs_t));
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
			zigzag_block(&frames.MB[mb_num].block[b_num]);
		return 0;
	}
    return 1;
}

static void intra_transform()
{
	frames.frames_until_key = video.GOP_size;
	frames.frames_until_altref = video.altref_range;
	frames.last_key_detect = frames.frame_number;

	frames.current_is_golden_frame = 1;
	frames.current_is_altref_frame = 1;
	frames.golden_frame_number = frames.frame_number;
	frames.altref_frame_number = frames.frame_number;


	// on key transform we wait for all other to end
    cl_int mb_num;

    // now foreach macroblock
    for(mb_num = 0; mb_num < video.mb_count; ++mb_num)
    {
        // input data - raw current frame, output - in block-packed order
        // prepare macroblock (predict and subbtract) and transform(dct, wht, quant and zigzag)
        // BONUS: reconstruct in here too
		predict_and_transform_mb(mb_num);
    }
	if (video.GOP_size > 1) 
	{
		if (video.do_loop_filter_on_gpu)
		{
			device.state_gpu = clEnqueueWriteBuffer(device.commandQueue1_gpu, device.reconstructed_frame_Y, CL_FALSE, 0, video.wrk_frame_size_luma, frames.reconstructed_Y, 0, NULL, NULL);
			device.state_gpu = clEnqueueWriteBuffer(device.commandQueue2_gpu, device.reconstructed_frame_U, CL_FALSE, 0, video.wrk_frame_size_chroma, frames.reconstructed_U, 0, NULL, NULL);
			device.state_gpu = clEnqueueWriteBuffer(device.commandQueue3_gpu, device.reconstructed_frame_V, CL_FALSE, 0, video.wrk_frame_size_chroma, frames.reconstructed_V, 0, NULL, NULL);
		}  
		else
		{
			device.state_cpu = clEnqueueWriteBuffer(device.loopfilterY_commandQueue_cpu, device.cpu_frame_Y, CL_FALSE, 0, video.wrk_frame_size_luma, frames.reconstructed_Y, 0, NULL, NULL);
			device.state_cpu = clEnqueueWriteBuffer(device.loopfilterU_commandQueue_cpu, device.cpu_frame_U, CL_FALSE, 0, video.wrk_frame_size_chroma, frames.reconstructed_U, 0, NULL, NULL);
			device.state_cpu = clEnqueueWriteBuffer(device.loopfilterV_commandQueue_cpu, device.cpu_frame_V, CL_FALSE, 0, video.wrk_frame_size_chroma, frames.reconstructed_V, 0, NULL, NULL);
		}
	}

    return;
}