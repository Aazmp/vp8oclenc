
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

static const int cospi8sqrt2minus1=20091;
static const int sinpi8sqrt2 =35468;

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

	frames.current_is_golden_frame = 1;
	frames.current_is_altref_frame = 1;
	frames.golden_frame_number = frames.frame_number;
	frames.altref_frame_number = frames.frame_number;


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
