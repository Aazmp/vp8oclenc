#pragma OPENCL EXTENSION cl_khr_byte_addressable_store : enable
//#pragma OPENCL EXTENSION cl_amd_printf : enable

typedef short int16_t;
typedef int int32_t;
typedef unsigned int uint32_t;
typedef unsigned char uint8_t;
typedef signed char int8_t;
typedef unsigned short uint16_t;
typedef uint8_t Prob;
typedef int8_t tree_index;
typedef const tree_index Tree[];

typedef enum {
	are16x16 = 0,
	are8x8 = 1,
	are4x4 = 2
} partition_mode;

typedef struct {
    int16_t coeffs[25][16];
    int32_t vector_x[4];
    int32_t vector_y[4];
	float SSIM;
	int non_zero_coeffs;
	int parts;
} macroblock;

typedef struct {
	__global uint8_t *output; /* ptr to next byte to be written */
	uint32_t range; /* 128 <= range <= 255 */
	uint32_t bottom; /* minimum value of remaining output */
	int32_t bit_count; /* # of shifts before an output byte is available */
	uint32_t count;
} vp8_bool_encoder;

void init_bool_encoder(vp8_bool_encoder *e, __global uint8_t *start_partition)
{
    e->output = start_partition;
    e->range = 255;
    e->bottom = 0;
    e->bit_count = 24;
    e->count = 0;
}

void add_one_to_output(__global uint8_t *q)
{
    while( *--q == 255)
        *q = 0;
    ++*q;
}

void write_bool(vp8_bool_encoder *e, int prob, int bool_value)
{
    /* split is approximately (range * prob) / 256 and, crucially,
    is strictly bigger than zero and strictly smaller than range */
    uint32_t split = 1 + ( ((e->range - 1) * prob) >> 8);
    if( bool_value) {
        e->bottom += split; /* move up bottom of interval */
        e->range -= split; /* with corresponding decrease in range */
    } else
        e->range = split;
    while( e->range < 128)
    {
        e->range <<= 1;
        if( e->bottom & ((uint32_t)1 << 31)) {/* detect carry */
            add_one_to_output(e->output);
		}
        e->bottom <<= 1;
        if( !--e->bit_count) {
            *e->output++ = (uint8_t) (e->bottom >> 24);
            e->count++;
            e->bottom &= (1 << 24) - 1;
            e->bit_count = 8;
        }
    }
}

void write_flag(vp8_bool_encoder *e, int b)
{
    write_bool(e, 128, (b)?1:0);
}

void write_literal(vp8_bool_encoder *e, int i, int size)
{
    int mask = 1 << (size - 1);
    while (mask)
    {
        write_flag(e, !((i & mask) == 0));
        mask >>= 1;
    }
}

void flush_bool_encoder(vp8_bool_encoder *e)
{
    int c = e->bit_count;
    uint32_t v = e->bottom;
    if( v & (1 << (32 - c)))
        add_one_to_output(e->output);
    v <<= c & 7;
    c >>= 3;
    while( --c >= 0)
        v <<= 8;
    c = 4;
    while( --c >= 0) {
        /* write remaining data, possibly padded */
        *e->output++ = (uint8_t) (v >> 24);
        e->count++;
        v <<= 8;
    }
}

typedef enum
{	DCT_0, /* value 0 */
	DCT_1, /* 1 */
	DCT_2, /* 2 */
	DCT_3, /* 3 */
	DCT_4, /* 4 */
	dct_cat1, /* range 5 - 6 (size 2) */
	dct_cat2, /* 7 - 10 (4) */
	dct_cat3, /* 11 - 18 (8) */
	dct_cat4, /* 19 - 34 (16) */
	dct_cat5, /* 35 - 66 (32) */
	dct_cat6, /* 67 - 2048 (1982) */
	dct_eob, /* end of block */
	num_dct_tokens /* 12 */
} dct_token;
typedef struct {
	int sign;
	int bits;
	int size;
	int extra_bits;
	int extra_size;
	__constant Prob* pcat;
} token;
__constant tree_index coeff_tree [2 * (num_dct_tokens - 1)] = {	-dct_eob, 2, /* eob = "0" */
																-DCT_0, 4, /* 0 = "10" */
																-DCT_1, 6, /* 1 = "110" */
																8, 12,
																-DCT_2, 10, /* 2 = "11100" */
																-DCT_3, -DCT_4, /* 3 = "111010", 4 = "111011" */
																14, 16,
																-dct_cat1, -dct_cat2, /* cat1 = "111100", cat2 = "111101" */
																18, 20,
																-dct_cat3, -dct_cat4, /* cat3 = "1111100", cat4 = "1111101" */
																-dct_cat5, -dct_cat6 /* cat5 = "1111110", cat6 = "1111111" */
															  };
__constant Prob Pcat1[] = { 159, 0};
__constant Prob Pcat2[] = { 165, 145, 0};
__constant Prob Pcat3[] = { 173, 148, 140, 0};
__constant Prob Pcat4[] = { 176, 155, 140, 135, 0};
__constant Prob Pcat5[] = { 180, 157, 141, 134, 130, 0};
__constant Prob Pcat6[] = { 254, 254, 243, 230, 196, 177, 153, 140, 133, 130, 129, 0};
__constant int coeff_bands[16] = { 0, 1, 2, 3, 6, 4, 5, 6, 6, 6, 6, 6, 6, 6, 6, 7};

void encode_block(vp8_bool_encoder *vbe, __global uint *coeff_probs, int mb_num, int b_num, token *tokens, int ctx1, uchar ctx3)
{
	// ctx1 = 0 for Y beggining at coefficient 1 (when y2 exists)
	//		= 1 for Y2
	//		= 2 for U or V
	//		= 3 for Y beggining at coefficient 0 (when Y2 is absent)
	// ctx2 = coefficient position in block {(0), 1, 2, 3, ... 15} 
	//										chooses value from coeff_bands[16] = { 0, 1, 2, 3, 6, 4, 5, 6, 6, 6, 6, 6, 6, 6, 6, 7};
	// ctx3 = for the first(second when ctx1 = 0) coefficient it is equal to number of nearby(above and left only counts) blocks
	//																										with non-zero coefficients
	// 		= for next coefficients it equals: to 0, when previous is zero: to 1, when previous is +1 or -1; to 2 in other cases
	// ctx4 = token tree position
	int ctx2;
	tree_index ctx4;
	
	int i = ((ctx1 == 0) ? 1 : 0); // maybe (!ctx1)
	int prev_is_zero = 0;
	
	for (; i < 16; ++i)
	{
		ctx2=coeff_bands[i];
		
		// if previous coefficient was DCT_0, then current can't be EOB (inneficient to have 0,0,0,eob)
		// since EOB the only token, that has ZERO as highest bit
		// then ONE in first bit becomes implicit and doesn't require encoding
		
		// to handle this we must lower encoding bits size by 1
		// and tree_index at 2, instead of 0 (the route is tree[0+1]==2 when we encode "1")
		if (prev_is_zero) {
			ctx4 = 2;
			--(tokens[i].size);
		} else ctx4 = 0;

		do {
			const int b = (tokens[i].bits >> (--(tokens[i].size))) & 1;
			write_bool(vbe, (uchar)coeff_probs[(((ctx1<<3) + ctx2)*3 + ctx3)*11 + (ctx4>>1)], b);
			ctx4 = coeff_tree[ctx4+b];
		} while (tokens[i].size);

		if (tokens[i].bits == 0) return; // EOB == "0"
		
		//now we maybe we have extra bits to encode (if previously dct_catx was encoded)
		if (tokens[i].extra_size > 0)
		{
			int mask = 1 << (tokens[i].extra_size-1);
            int j = 0;
            while (tokens[i].pcat[j])
            {
                write_bool(vbe, tokens[i].pcat[j], (tokens[i].extra_bits & mask) ? 1 : 0);
                ++j;
                mask >>= 1;	
			}
		}
		
		ctx3 = 2;
		if (tokens[i].bits == 6) ctx3 = 1; /* DCT__1 = "110" */
		if (tokens[i].bits == 2) { //DCT__0 == "10"
			prev_is_zero = 1;
			ctx3 = 0;
		} 
		else {
			write_bool(vbe, 128, tokens[i].sign); //sign
			prev_is_zero = 0;
		}

	}
	
	return;
}
														 
void tokenize_block(__global macroblock *MBs, int mb_num, int b_num, token tokens[16])
{
	int next = 0; // imaginary 17th element
	int i;
	for (i = 15; i >= 0; --i) // tokenize block
	{
		int coeff = (int)MBs[mb_num].coeffs[b_num][i];		
		tokens[i].sign = (coeff < 0) ? 1 : 0;
		coeff = (coeff < 0) ? -coeff : coeff;
		tokens[i].extra_bits = 0;
		tokens[i].extra_size = 0;
		tokens[i].pcat = Pcat1;
		if (coeff == 0) {
			if (next == 0) {
				tokens[i].bits = 0; //dct_eob = "0"
				tokens[i].size = 1;
			} else {
				tokens[i].bits = 2; /* 0 = "10" */
				tokens[i].size = 2; 
			}
		}
		else if (coeff == 1) {
			tokens[i].bits = 6; /* 1 = "110" */
			tokens[i].size = 3;
		}
		else if (coeff == 2) {
			tokens[i].bits = 28; /* 2 = "11100" */
			tokens[i].size = 5;
		}
		else if (coeff == 3) {
			tokens[i].bits = 58; /* 3 = "111010" */
			tokens[i].size = 6;
		}
		else if (coeff == 4) {
			tokens[i].bits = 59; /* 4 = "111011" */
			tokens[i].size = 6;
		}
		else if (coeff <= 6) {
			tokens[i].bits = 60; /* cat1 = "111100" */
			tokens[i].size = 6; /* range 5 - 6 (size 2) */
			tokens[i].extra_bits = coeff - 5;
			tokens[i].extra_size = 1;
			//Pcat1 already assigned
		}
		else if (coeff <= 10) {
			tokens[i].bits = 61; /* cat2 = "111101" */
			tokens[i].size = 6; /* 7 - 10 (4) */
			tokens[i].extra_bits = coeff - 7;
			tokens[i].extra_size = 2;
			tokens[i].pcat = Pcat2;
		}
		else if (coeff <= 18) {
			tokens[i].bits = 124; /* cat3 = "1111100" */
			tokens[i].size = 7; /* 11 - 18 (8) */
			tokens[i].extra_bits = coeff - 11;
			tokens[i].extra_size = 3;
			tokens[i].pcat = Pcat3;
		}
		else if (coeff <= 34) {
			tokens[i].bits = 125; /* cat4 = "1111101" */
			tokens[i].size = 7; /* 19 - 34 (16) */
			tokens[i].extra_bits = coeff - 19;
			tokens[i].extra_size = 4;
			tokens[i].pcat = Pcat4;
		}
		else if (coeff <= 66) {
			tokens[i].bits = 126; /* cat5 = "1111110" */
			tokens[i].size = 7; /* 35 - 66 (32) */
			tokens[i].extra_bits = coeff - 35;
			tokens[i].extra_size = 5;
			tokens[i].pcat = Pcat5;
		}
		else {
			tokens[i].bits = 127; /* cat6 = "1111111" */
			tokens[i].size = 7; /* 67 - 2048 (1982) */
			tokens[i].extra_bits = coeff - 67;
			tokens[i].extra_size = 11;
			tokens[i].pcat = Pcat6;
		}		
		next = tokens[i].bits;
	}
	return;
}

__kernel void encode_coefficients(	__global macroblock *MBs, 
									__global uchar *output, 
									__global int *partition_sizes,
									__global uchar *third_context,
									__global uint *coeff_probs,
									__global uint *coeff_probs_denom,
									int mb_height, 
									int mb_width, 
									int num_partitions,
									int key_frame,
									int partition_step,
									int skip_prob)
{
	int part_num = get_global_id(0);
	int mb_row, mb_num, mb_col, b_num;
	int first_context;
	vp8_bool_encoder vbe[1];
	
	token tokens[16];
	
	init_bool_encoder(vbe, output + partition_step*part_num);

	for (mb_row = part_num; mb_row < mb_height; mb_row+= num_partitions)
	{
		for (mb_col = 0; mb_col < mb_width; ++mb_col)
		{
			mb_num = mb_col + mb_row * mb_width;
			if (MBs[mb_num].non_zero_coeffs == 0) 
				continue;
			if (MBs[mb_num].parts == are16x16)
			{
				first_context = 1; // for Y2
				tokenize_block(MBs, mb_num, 24, tokens); 
				encode_block(vbe, coeff_probs, mb_num, 24, tokens, first_context, *(third_context + mb_num*25 + 24));
				first_context = 0; //for Y, when Y2 exists
			} else {
				first_context = 3; //for Y, when Y2 is absent
			}
			// then always goes Y
			// 16 of them
			for (b_num = 0; b_num < 16; ++b_num)
			{
				tokenize_block(MBs, mb_num, b_num, tokens);
				encode_block(vbe, coeff_probs, mb_num, b_num, tokens, first_context, *(third_context + mb_num*25 + b_num));
			}
			//now 8 U-blocks
			first_context = 2; // for all chromas
			for (b_num = 16; b_num < 20; ++b_num)
			{
				tokenize_block(MBs, mb_num, b_num, tokens);
				encode_block(vbe, coeff_probs, mb_num, b_num, tokens, first_context, *(third_context + mb_num*25 + b_num));
			}
			//now 8 V-blocks
			for (b_num = 20; b_num < 24; ++b_num)
			{
				tokenize_block(MBs, mb_num, b_num, tokens);
				encode_block(vbe, coeff_probs, mb_num, b_num, tokens, first_context, *(third_context + mb_num*25 + b_num));
			}
		}
	}
	flush_bool_encoder(vbe);
	partition_sizes[part_num] = vbe->count;	
	
	return;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void tokenize_block_cut(__global macroblock *MBs, int mb_num, int b_num, token tokens[16])
{
	int next = 0; // imaginary 17th element
	int i;
	for (i = 15; i >= 0; --i) // tokenize block
	{
		int coeff = (int)MBs[mb_num].coeffs[b_num][i];
		coeff = (coeff < 0) ? -coeff : coeff;
		if (coeff == 0) {
			if (next == 0) {
				tokens[i].bits = 0; //dct_eob = "0"
				tokens[i].size = 1;
			} else {
				tokens[i].bits = 2; /* 0 = "10" */
				tokens[i].size = 2; 
			}
		}
		else if (coeff == 1) {
			tokens[i].bits = 6; /* 1 = "110" */
			tokens[i].size = 3;
		}
		else if (coeff == 2) {
			tokens[i].bits = 28; /* 2 = "11100" */
			tokens[i].size = 5;
		}
		else if (coeff == 3) {
			tokens[i].bits = 58; /* 3 = "111010" */
			tokens[i].size = 6;
		}
		else if (coeff == 4) {
			tokens[i].bits = 59; /* 4 = "111011" */
			tokens[i].size = 6;
		}
		else if (coeff <= 6) {
			tokens[i].bits = 60; /* cat1 = "111100" */
			tokens[i].size = 6; /* range 5 - 6 (size 2) */
		}
		else if (coeff <= 10) {
			tokens[i].bits = 61; /* cat2 = "111101" */
			tokens[i].size = 6; /* 7 - 10 (4) */
		}
		else if (coeff <= 18) {
			tokens[i].bits = 124; /* cat3 = "1111100" */
			tokens[i].size = 7; /* 11 - 18 (8) */
		}
		else if (coeff <= 34) {
			tokens[i].bits = 125; /* cat4 = "1111101" */
			tokens[i].size = 7; /* 19 - 34 (16) */
		}
		else if (coeff <= 66) {
			tokens[i].bits = 126; /* cat5 = "1111110" */
			tokens[i].size = 7; /* 35 - 66 (32) */
		}
		else {
			tokens[i].bits = 127; /* cat6 = "1111111" */
			tokens[i].size = 7; /* 67 - 2048 (1982) */
		}		
		next = tokens[i].bits;
	}
	return;
}


void count_probs_in_block(	__global uint *const coeff_probs,
							__global uint *const coeff_probs_denom,
							const int part_num,
							const int mb_num, const int b_num, 
							token *tokens, 
							const int ctx1,const int in_ctx3)
{
	// ctx1 = 0 for Y beggining at coefficient 1 (when y2 exists)
	//		= 1 for Y2
	//		= 2 for U or V
	//		= 3 for Y beggining at coefficient 0 (when Y2 is absent)
	// ctx2 = coefficient position in block {(0), 1, 2, 3, ... 15} 
	//										chooses value from coeff_bands[16] = { 0, 1, 2, 3, 6, 4, 5, 6, 6, 6, 6, 6, 6, 6, 6, 7};
	// ctx3 = for the first(second when ctx1 = 0) coefficient it is equal to number of nearby(above and left only counts) blocks
	//																										with non-zero coefficients
	// 		= for next coefficients it equals: to 0, when previous is zero: to 1, when previous is +1 or -1; to 2 in other cases
	// ctx4 = token tree position
	int ctx2;
	int ctx3 = in_ctx3;
	tree_index ctx4;
	
	int i = ((ctx1 == 0) ? 1 : 0); // maybe (!ctx1)
	int prev_is_zero = 0;
	
	for (; i < 16; ++i)
	{
		ctx2=coeff_bands[i];
		
		// if previous coefficient was DCT_0, then current can't be EOB (inneficient to have 0,0,0,eob)
		// since EOB the only token, that has ZERO as highest bit
		// then ONE in first bit becomes implicit and doesn't require encoding
		
		// to handle this we must lower encoding bits size by 1
		// and tree_index at 2, instead of 0 (the route is tree[0+1]==2 when we encode "1")
		if (prev_is_zero) {
			ctx4 = 2;
			--(tokens[i].size);
		} else ctx4 = 0;

		//__constant Prob *const p = default_coeff_probs[ctx1][ctx2][ctx3];
		
		do {
			const uchar b = (tokens[i].bits >> (--(tokens[i].size))) & 1;
			coeff_probs[((((part_num<<5) + (ctx1<<3)) + ctx2)*3 + ctx3)*11 + (ctx4>>1)] += (1 - b); //increase numerator when b == 0
			++(coeff_probs_denom[((((part_num<<5) + (ctx1<<3)) + ctx2)*3 + ctx3)*11 + (ctx4>>1)]); // increase denominator
			ctx4 = coeff_tree[ctx4+b];
		} while (tokens[i].size);

		ctx3 = 2;
		if (tokens[i].bits == 6) ctx3 = 1; /* DCT__1 = "110" */
		if (tokens[i].bits == 2) { //DCT__0 == "10"
			prev_is_zero = 1;
			ctx3 = 0;
		} 
		else {
			prev_is_zero = 0;
		}
	}
	return;
}

__kernel void count_probs(	__global macroblock *MBs, 
							__global uint *coeff_probs, 
							__global uint *coeff_probs_denom,
							__global uchar *third_context,
							int mb_height, 
							int mb_width, 
							int num_partitions,
							int key_frame,
							int partition_step)
{
	int part_num = get_global_id(0);
	int mb_row, mb_num, mb_col, b_num;
	int prev_mb, prev_b;
	int i;
	int first_context, firstCoeff;
	token tokens[16];
	
	{ 
		// we have to work with 1-dimensional array in global memory, so
		// coeff_probs[p][ctx1][ctx2][ctx3][ctx4] => *(coeff_probs + p*11*3*8*4 + ctx1*11*3*8 + ctx2*11*3 + ctx3*11 + ctx4)
		// or coeff_probs[((((p<<5) + (ctx1<<3)) + ctx2)*3 + ctx3)*11 + ctx4]
		int ctx1, ctx2, ctx3, ctx4;
		for (ctx1 = 0; ctx1 < 4; ++ctx1)
			for (ctx2 = 0; ctx2 < 8; ++ctx2)
				for (ctx3 = 0; ctx3 < 3; ++ctx3)
					for (ctx4 = 0; ctx4 < 11; ++ctx4) {
						coeff_probs[((((part_num<<5) + (ctx1<<3)) + ctx2)*3 + ctx3)*11 + ctx4] = 0;
						coeff_probs_denom[((((part_num<<5) + (ctx1<<3)) + ctx2)*3 + ctx3)*11 + ctx4] = 1;
					}
	}

	for (mb_row = part_num; mb_row < mb_height; mb_row+= num_partitions)
	{
		for (mb_col = 0; mb_col < mb_width; ++mb_col)
		{
			mb_num = mb_col + mb_row * mb_width;
			if (MBs[mb_num].non_zero_coeffs == 0) 
				continue;
			if (MBs[mb_num].parts == are16x16)
			{ 
				first_context = 1; // for Y2
				*(third_context + mb_num*25 + 24) = 0;
				if (mb_row > 0) { // check if "above" Y2 has non zero
					// we go up until we find MB with Y2 mode enabled
					prev_mb = mb_num-mb_width;
					while(prev_mb>=0) {
						if (MBs[prev_mb].parts == are16x16) break;
						prev_mb-=mb_width;
					}
					if (prev_mb >= 0)
						for (i = 0; i < 16; ++i) {
							if (MBs[prev_mb].coeffs[24][i] != 0) {
								++(*(third_context + mb_num*25 + 24));
								break;
							}
						}
				} 
				if (mb_col > 0) { // check if "left" Y2 has non zero
					prev_mb = mb_num-1;
					while (prev_mb >= (mb_row*mb_width)) {
						if (MBs[prev_mb].parts == are16x16) break;
						--prev_mb;
					}
					if (prev_mb >= (mb_row*mb_width))
						for (i = 0; i < 16; ++i) { 
							if (MBs[prev_mb].coeffs[24][i] != 0) {
								++(*(third_context + mb_num*25 + 24));
								break;
							}
						}
				}
				tokenize_block_cut(MBs, mb_num, 24, tokens); 
				count_probs_in_block(coeff_probs, coeff_probs_denom, part_num, mb_num, 24, tokens, first_context, *(third_context + mb_num*25 + 24));
				first_context = 0; //for Y, when Y2 exists
			} else first_context = 3; //for Y, when Y2 is absent
			// then always goes Y
			// 16 of them
			for (b_num = 0; b_num < 16; ++b_num)
			{
				*(third_context + mb_num*25 + b_num) = 0;
				// look above:
				prev_mb = -1; // as flag, that above is empty
				if ((b_num >> 2) > 0) { // /4
					prev_mb = mb_num;
					prev_b = b_num - 4;
				} 
				else if (mb_row > 0) {
					prev_mb = mb_num - mb_width;
					prev_b = b_num + 12;
				}
				if (prev_mb >= 0) {
					firstCoeff = (MBs[prev_mb].parts == are16x16) ? 1 : 0;
					for (i = firstCoeff; i < 16; ++i) {
						if (MBs[prev_mb].coeffs[prev_b][i] != 0) {
							++(*(third_context + mb_num*25 + b_num));
							break;
						}
					}
				}
				// look to the left
				prev_mb = -1;
				if ((b_num & 3) > 0) { // %4
					prev_mb = mb_num;
					prev_b = b_num - 1;
				} 
				else if (mb_col > 0) {
					prev_mb = mb_num - 1;
					prev_b = b_num + 3;
				}
				if (prev_mb >= 0) {
					firstCoeff = (MBs[prev_mb].parts == are16x16) ? 1 : 0;
					for (i = firstCoeff; i < 16; ++i) {
						if (MBs[prev_mb].coeffs[prev_b][i] != 0) {
							++(*(third_context + mb_num*25 + b_num));
							break;
						}
					}
				}
				tokenize_block_cut(MBs, mb_num, b_num, tokens);
				count_probs_in_block(coeff_probs, coeff_probs_denom, part_num, mb_num, b_num, tokens, first_context, *(third_context + mb_num*25 + b_num));
			}
			//now 8 U-blocks
			first_context = 2; // for all chromas
			for (b_num = 16; b_num < 20; ++b_num)
			{
				*(third_context + mb_num*25 + b_num) = 0;
				// look above:
				prev_mb = -1; // as flag, that above is empty
				if (((b_num-16) >> 1) > 0) { // /2
					prev_mb = mb_num;
					prev_b = b_num - 2;
				} 
				else if (mb_row > 0) {
					prev_mb = mb_num - mb_width;
					prev_b = b_num + 2;
				}
				if (prev_mb >= 0) {
					for (i = 0; i < 16; ++i) {
						if (MBs[prev_mb].coeffs[prev_b][i] != 0) {
							++(*(third_context + mb_num*25 + b_num));
							break;
						}
					}
				}
				// look to the left
				prev_mb = -1;
				if (((b_num-16) & 1) > 0) { // %2
					prev_mb = mb_num;
					prev_b = b_num - 1;
				} 
				else if (mb_col > 0) {
					prev_mb = mb_num - 1;
					prev_b = b_num + 1;
				}
				if (prev_mb >= 0) {
					for (i = 0; i < 16; ++i) {
						if (MBs[prev_mb].coeffs[prev_b][i] != 0) {
							++(*(third_context + mb_num*25 + b_num));
							break;
						}
					}
				}
				tokenize_block_cut(MBs, mb_num, b_num, tokens);
				count_probs_in_block(coeff_probs, coeff_probs_denom, part_num, mb_num, b_num, tokens, first_context, *(third_context + mb_num*25 + b_num));
			}
			//now 8 V-blocks
			for (b_num = 20; b_num < 24; ++b_num)
			{
				*(third_context + mb_num*25 + b_num) = 0;
				// look above:
				prev_mb = -1; // as flag, that above is empty
				if (((b_num-20) >> 1) > 0) { // /2
					prev_mb = mb_num;
					prev_b = b_num - 2;
				} 
				else if (mb_row > 0) {
					prev_mb = mb_num - mb_width;
					prev_b = b_num + 2;
				}
				if (prev_mb >= 0) {
					for (i = 0; i < 16; ++i) {
						if (MBs[prev_mb].coeffs[prev_b][i] != 0) {
							++(*(third_context + mb_num*25 + b_num));
							break;
						}
					}
				}
				// look to the left
				prev_mb = -1;
				if (((b_num-20) & 1) > 0) { // %2
					prev_mb = mb_num;
					prev_b = b_num - 1;
				} 
				else if (mb_col > 0) {
					prev_mb = mb_num - 1;
					prev_b = b_num + 1;
				}
				if (prev_mb >= 0) {
					for (i = 0; i < 16; ++i) {
						if (MBs[prev_mb].coeffs[prev_b][i] != 0) {
							++(*(third_context + mb_num*25 + b_num));
							break;
						}
					}
				}
				tokenize_block_cut(MBs, mb_num, b_num, tokens);
				count_probs_in_block(coeff_probs, coeff_probs_denom, part_num, mb_num, b_num, tokens, first_context, *(third_context + mb_num*25 + b_num));
			}
		}

	}
	
	return;
}

__kernel void num_div_denom(__global uint *coeff_probs, 
							__global uint *coeff_probs_denom,
							int num_partitions)
{
	int part_num = get_global_id(0);
	int ctx1, ctx2, ctx3, ctx4, p;
	uint num, denom;
	for (ctx1 = 0; ctx1 < 4; ++ctx1)
		for (ctx2 = part_num; ctx2 < 8; ctx2 += num_partitions)
			for (ctx3 = 0; ctx3 < 3; ++ctx3)
				for (ctx4 = 0; ctx4 < 11; ++ctx4) {
					num = 0;
					denom = 0;
					for (p = 0; p < num_partitions; ++p) {
						num += coeff_probs[((((p<<5) + (ctx1<<3)) + ctx2)*3 + ctx3)*11 + ctx4];
						denom += coeff_probs_denom[((((p<<5) + (ctx1<<3)) + ctx2)*3 + ctx3)*11 + ctx4];
					}
					num = (num << 8) / denom;
					coeff_probs[(((ctx1<<3) + ctx2)*3 + ctx3)*11 + ctx4] = (num > 255) ? 255 : ((num == 0) ? 1 : num);
				}
	return;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////
