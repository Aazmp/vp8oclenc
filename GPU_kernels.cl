#pragma OPENCL EXTENSION cl_khr_byte_addressable_store : enable

typedef enum {
	are16x16 = 0,
	are8x8 = 1,
	are4x4 = 2
} partition_mode;

typedef enum {
	LAST = 0,
	GOLDEN = 1,
	ALTREF = 2
} ref_frame;

typedef enum {
	intra_segment = 0,
	HQ_segment = 1,
	AQ_segment = 2,
	LQ_segment = 3
} segment_ids;

typedef struct {
	int y_ac_i; 
	int y_dc_idelta;
	int y2_dc_idelta;
	int y2_ac_idelta;
	int uv_dc_idelta;
	int uv_ac_idelta;
	int loop_filter_level;
	int mbedge_limit;
	int sub_bedge_limit;
	int interior_limit;
	int hev_threshold;
} segment_data;

typedef struct {
    short coeffs[25][16];
    int vector_x[4];
    int vector_y[4];
	float SSIM;
	int non_zero_coeffs;
	int parts;
	int reference_frame;
	segment_ids segment_id;
} macroblock;

typedef struct {
	int vector_x;
	int vector_y;
} vector_net;

static const int vp8_dc_qlookup[128] =
{
      4,   5,   6,   7,   8,   9,  10,  10,  11,  12,  13,  14,  15,  16,  17,  17,
     18,  19,  20,  20,  21,  21,  22,  22,  23,  23,  24,  25,  25,  26,  27,  28,
     29,  30,  31,  32,  33,  34,  35,  36,  37,  37,  38,  39,  40,  41,  42,  43,
     44,  45,  46,  46,  47,  48,  49,  50,  51,  52,  53,  54,  55,  56,  57,  58,
     59,  60,  61,  62,  63,  64,  65,  66,  67,  68,  69,  70,  71,  72,  73,  74,
     75,  76,  76,  77,  78,  79,  80,  81,  82,  83,  84,  85,  86,  87,  88,  89,
     91,  93,  95,  96,  98, 100, 101, 102, 104, 106, 108, 110, 112, 114, 116, 118,
    122, 124, 126, 128, 130, 132, 134, 136, 138, 140, 143, 145, 148, 151, 154, 157,
};

static const int vp8_ac_qlookup[128] =
{
      4,   5,   6,   7,   8,   9,  10,  11,  12,  13,  14,  15,  16,  17,  18,  19,
     20,  21,  22,  23,  24,  25,  26,  27,  28,  29,  30,  31,  32,  33,  34,  35,
     36,  37,  38,  39,  40,  41,  42,  43,  44,  45,  46,  47,  48,  49,  50,  51,
     52,  53,  54,  55,  56,  57,  58,  60,  62,  64,  66,  68,  70,  72,  74,  76,
     78,  80,  82,  84,  86,  88,  90,  92,  94,  96,  98, 100, 102, 104, 106, 108,
    110, 112, 114, 116, 119, 122, 125, 128, 131, 134, 137, 140, 143, 146, 149, 152,
    155, 158, 161, 164, 167, 170, 173, 177, 181, 185, 189, 193, 197, 201, 205, 209,
    213, 217, 221, 225, 229, 234, 239, 245, 249, 254, 259, 264, 269, 274, 279, 284,
};

void weight(int4 *__L0, int4 *__L1, int4 *__L2, int4 *__L3) 
{
	int4 L0 = *__L0;
	int4 L1 = *__L1;
	int4 L2 = *__L2;
	int4 L3 = *__L3; // <========================================================================

	*__L0 = ((L0 + L3) << 3);	// a1 = ((ip[0] + ip[3])<<3);
	*__L1 = ((L1 + L2) << 3);	// b1 = ((ip[1] + ip[2])<<3);
	*__L2 = ((L1 - L2) << 3);	// c1 = ((ip[1] - ip[2])<<3);
	*__L3 = ((L0 - L3) << 3);	// d1 = ((ip[0] - ip[3])<<3);
	

	L0 = *__L0 + *__L1;				// op[0] = (a1 + b1); 
	L2 = *__L0 - *__L1;				// op[2] = (a1 - b1);
	
	L1 = (((*__L2 * 2217) + (*__L3 * 5352) + 14500) >> 12);
														// op[1] = (c1 * 2217 + d1 * 5352 +  14500)>>12;
	L3 = (((*__L3 * 2217) - (*__L2 * 5352) + 7500) >> 12);
														// op[3] = (d1 * 2217 - c1 * 5352 +   7500)>>12;

	*__L0 = (int4)(L0.x, L1.x, L2.x, L3.x);
	*__L1 = (int4)(L0.y, L1.y, L2.y, L3.y);
	*__L2 = (int4)(L0.z, L1.z, L2.z, L3.z);
	*__L3 = (int4)(L0.w, L1.w, L2.w, L3.w);

	L0 = *__L0 + *__L3;				// a1 = op[0] + op[3];	
	L1 = *__L1 + *__L2;				// b1 = op[1] + op[2];
	L2 = *__L1 - *__L2;				// c1 = op[1] - op[2];
	L3 = *__L0 - *__L3;				// d1 = op[0] - op[3];
	
		
	*__L0 = ((L0 + L1 + 7) >> 4);	// op[0] = (( a1 + b1 + 7)>>4);
	*__L2 = ((L0 - L1 + 7) >> 4);	// op[2] = (( a1 - b1 + 7)>>4);
	
	*__L1 = ((L2 * 2217) + (L3 * 5352) + 12000) >> 16;
	(*__L1).x += (L3.x != 0);
	(*__L1).y += (L3.y != 0);
	(*__L1).z += (L3.z != 0);
	(*__L1).w += (L3.w != 0);				// op[1]  = (((c1 * 2217 + d1 * 5352 +  12000)>>16) + (d1!=0));
	
	*__L3 = (((L3 * 2217) - (L2 * 5352) + 51000) >>16 );
														// op[3] = ((d1 * 2217 - c1 * 5352 +  51000)>>16);
	
	*__L0 = convert_int4(abs(*__L0));
	*__L1 = convert_int4(abs(*__L1));
	*__L2 = convert_int4(abs(*__L2));
	*__L3 = convert_int4(abs(*__L3));
	
	(*__L0).x /= 4;
	// just a SATD
	(*__L0).x +=(*__L0).y + (*__L0).z + (*__L0).w +
	(*__L1).x + (*__L1).y + (*__L1).z + (*__L1).w +
	(*__L2).x + (*__L2).y + (*__L2).z + (*__L2).w +
	(*__L3).x + (*__L3).y + (*__L3).z + (*__L3).w;
	
	
	return;
}

void DCT_and_quant(int4 *const __Line0, int4 *const __Line1, int4 *const __Line2, int4 *const __Line3, const int dc_q, const int ac_q) 
{
	__private int4 Line0, Line1, Line2, Line3;
	Line0 = (int4)((*__Line0).x, (*__Line1).x, (*__Line2).x, (*__Line3).x);
	Line1 = (int4)((*__Line0).y, (*__Line1).y, (*__Line2).y, (*__Line3).y);
	Line2 = (int4)((*__Line0).z, (*__Line1).z, (*__Line2).z, (*__Line3).z);
	Line3 = (int4)((*__Line0).w, (*__Line1).w, (*__Line2).w, (*__Line3).w);

	*__Line0 = ((Line0 + Line3) << 3);	// a1 = ((ip[0] + ip[3])<<3);
	*__Line1 = ((Line1 + Line2) << 3);	// b1 = ((ip[1] + ip[2])<<3);
	*__Line2 = ((Line1 - Line2) << 3);	// c1 = ((ip[1] - ip[2])<<3);
	*__Line3 = ((Line0 - Line3) << 3);	// d1 = ((ip[0] - ip[3])<<3);
	

	Line0 = *__Line0 + *__Line1;				// op[0] = (a1 + b1); 
	Line2 = *__Line0 - *__Line1;				// op[2] = (a1 - b1);
	
	Line1 = (((*__Line2 * 2217) + (*__Line3 * 5352) + 14500) >> 12);
														// op[1] = (c1 * 2217 + d1 * 5352 +  14500)>>12;
	Line3 = (((*__Line3 * 2217) - (*__Line2 * 5352) + 7500) >> 12);
														// op[3] = (d1 * 2217 - c1 * 5352 +   7500)>>12;

	*__Line0 = (int4)(Line0.x, Line1.x, Line2.x, Line3.x);
	*__Line1 = (int4)(Line0.y, Line1.y, Line2.y, Line3.y);
	*__Line2 = (int4)(Line0.z, Line1.z, Line2.z, Line3.z);
	*__Line3 = (int4)(Line0.w, Line1.w, Line2.w, Line3.w);

	Line0 = *__Line0 + *__Line3;				// a1 = op[0] + op[3];	
	Line1 = *__Line1 + *__Line2;				// b1 = op[1] + op[2];
	Line2 = *__Line1 - *__Line2;				// c1 = op[1] - op[2];
	Line3 = *__Line0 - *__Line3;				// d1 = op[0] - op[3];
	
		
	*__Line0 = ((Line0 + Line1 + 7) >> 4);	// op[0] = (( a1 + b1 + 7)>>4);
	*__Line2 = ((Line0 - Line1 + 7) >> 4);	// op[2] = (( a1 - b1 + 7)>>4);
	
	*__Line1 = ((Line2 * 2217) + (Line3 * 5352) + 12000) >> 16;
	(*__Line1).x += (Line3.x != 0);
	(*__Line1).y += (Line3.y != 0);
	(*__Line1).z += (Line3.z != 0);
	(*__Line1).w += (Line3.w != 0);				// op[1]  = (((c1 * 2217 + d1 * 5352 +  12000)>>16) + (d1!=0));
	
	*__Line3 = (((Line3 * 2217) - (Line2 * 5352) + 51000) >>16 );
														// op[3] = ((d1 * 2217 - c1 * 5352 +  51000)>>16);
														
	*__Line0 /= (int4)(dc_q, ac_q, ac_q, ac_q);
	*__Line1 /= ac_q;
	*__Line2 /= ac_q;
	*__Line3 /= ac_q;
	
	return;
}

void dequant_and_iDCT(int4 *const __Line0, int4 *const __Line1, int4 *const __Line2, int4 *const __Line3, const int dc_q, const int ac_q) // <- input DCT lines
{
	__private int4 Line0, Line1, Line2, Line3;
	// dequant
	Line0 = (*__Line0)*((int4)(dc_q, ac_q, ac_q, ac_q));
	Line1 = (*__Line1)*ac_q;
	Line2 = (*__Line2)*ac_q;
	Line3 = (*__Line3)*ac_q;	
		
	*__Line0 = Line0 + Line2;				// a1 = ip[0]+ip[2];
	*__Line1 = Line0 - Line2;				// b1 = ip[0]-ip[2];
								
	*__Line2 = ((Line1 * 35468) >> 16) - (Line3 + ((Line3 * 20091)>>16));
														// temp1 = (ip[1] * sinpi8sqrt2)>>16;
														// temp2 = ip[3] + ((ip[3] * cospi8sqrt2minus1)>>16);
														// c1 = temp1 - temp2;
	*__Line3 = (Line1 + ((Line1 * 20091)>>16)) + ((Line3 * 35468) >> 16);
														// temp1 = ip[1] + ((ip[1] * cospi8sqrt2minus1)>>16);
														// temp2 = (ip[3] * sinpi8sqrt2)>>16;
														// d1 = temp1 + temp2;
														
	Line0 = *__Line0 + *__Line3;			// op[0] = a1 + d1;
	Line3 = *__Line0 - *__Line3;			// op[3] = a1 - d1;
	Line1 = *__Line1 + *__Line2;			// op[1] = b1 + c1;
	Line2 = *__Line1 - *__Line2;			// op[2] = b1 - c1;
	
	// now transpose
	*__Line0 = (int4)(Line0.x, Line1.x, Line2.x, Line3.x);
	*__Line1 = (int4)(Line0.y, Line1.y, Line2.y, Line3.y);
	*__Line2 = (int4)(Line0.z, Line1.z, Line2.z, Line3.z);
	*__Line3 = (int4)(Line0.w, Line1.w, Line2.w, Line3.w);
	
	// second iDCT
	Line0 = *__Line0 + *__Line2;			// a1 = op[0]+op[2];
	Line1 = *__Line0 - *__Line2;			// b1 = tp[0]-tp[2];
	
	Line2 = ((*__Line1 * 35468) >> 16) - (*__Line3 + ((*__Line3 * 20091)>>16));
														// temp1 = (ip[1] * sinpi8sqrt2)>>16;
														// temp2 = ip[3] + ((ip[3] * cospi8sqrt2minus1)>>16);
														// c1 = temp1 - temp2;
	Line3 = (*__Line1 + ((*__Line1 * 20091)>>16)) + ((*__Line3 * 35468) >> 16);
														// temp1 = ip[1] + ((ip[1] * cospi8sqrt2minus1)>>16);
														// temp2 = (ip[3] * sinpi8sqrt2)>>16;
														// d1 = temp1 + temp2;
	*__Line0 = ((Line0 + Line3 + 4) >> 3);
														// op[0] = ((a1 + d1 + 4) >> 3) + pred[0,i]
	*__Line1 = ((Line1 + Line2 + 4) >> 3);
														// op[1] = ((b1 + c1 + 4) >> 3) + pred[1,i]
	*__Line2 = ((Line1 - Line2 + 4) >> 3);
														// op[2] = ((b1 - c1 + 4) >> 3) + pred[2,i]
	*__Line3 = ((Line0 - Line3 + 4) >> 3);
														// op[3] = ((a1 - d1 + 4) >> 3) + pred[3,i]

	Line0 = *__Line0;
	Line1 = *__Line1;
	Line2 = *__Line2;
	Line3 = *__Line3; // <========================================================================	
	*__Line0 = (int4)(Line0.x, Line1.x, Line2.x, Line3.x);
	*__Line1 = (int4)(Line0.y, Line1.y, Line2.y, Line3.y);
	*__Line2 = (int4)(Line0.z, Line1.z, Line2.z, Line3.z);
	*__Line3 = (int4)(Line0.w, Line1.w, Line2.w, Line3.w);

	return;
}

void WHT_and_quant(int4 *const __Line0, int4 *const __Line1, int4 *const __Line2, int4 *const __Line3, const int dc_q, const int ac_q)
{
	int4 Line0, Line1, Line2, Line3;
	
	Line0 = *__Line0 + *__Line3;
    Line1 = *__Line1 + *__Line2; 
    Line2 = *__Line1 - *__Line2;
    Line3 = *__Line0 - *__Line3;

    *__Line0 = Line0 + Line1; 
    *__Line1 = Line2 + Line3; 
	*__Line2 = Line0 - Line1;
    *__Line3 = Line3 - Line2; 
	
	Line0.x = (*__Line0).x + (*__Line0).w;
	Line1.x = (*__Line1).x + (*__Line1).w;	
	Line2.x = (*__Line2).x + (*__Line2).w;
	Line3.x = (*__Line3).x + (*__Line3).w;
	
    Line0.y = (*__Line0).y + (*__Line0).z;
	Line1.y = (*__Line1).y + (*__Line1).z;	
	Line2.y = (*__Line2).y + (*__Line2).z;
	Line3.y = (*__Line3).y + (*__Line3).z;   
	
    Line0.z = (*__Line0).y - (*__Line0).z;
	Line1.z = (*__Line1).y - (*__Line1).z;	
	Line2.z = (*__Line2).y - (*__Line2).z;
	Line3.z = (*__Line3).y - (*__Line3).z;
    
	Line0.w = (*__Line0).x - (*__Line0).w;
	Line1.w = (*__Line1).x - (*__Line1).w;	
	Line2.w = (*__Line2).x - (*__Line2).w;
	Line3.w = (*__Line3).x - (*__Line3).w;

    (*__Line0).x = Line0.x + Line0.y; 
	(*__Line1).x = Line1.x + Line1.y;
	(*__Line2).x = Line2.x + Line2.y;
	(*__Line3).x = Line3.x + Line3.y;
	
	(*__Line0).y = Line0.z + Line0.w;
	(*__Line1).y = Line1.z + Line1.w;
	(*__Line2).y = Line2.z + Line2.w;
	(*__Line3).y = Line3.z + Line3.w;
	
	(*__Line0).z = Line0.x - Line0.y;
	(*__Line1).z = Line1.x - Line1.y;
	(*__Line2).z = Line2.x - Line2.y;
	(*__Line3).z = Line3.x - Line3.y;
        
	(*__Line0).w = Line0.w - Line0.z;
	(*__Line1).w = Line1.w - Line1.z;
	(*__Line2).w = Line2.w - Line2.z;
	(*__Line3).w = Line3.w - Line3.z;

    (*__Line0).x += ((*__Line0).x > 0);
	(*__Line1).x += ((*__Line1).x > 0);
	(*__Line2).x += ((*__Line2).x > 0);
	(*__Line3).x += ((*__Line3).x > 0);
    (*__Line0).y += ((*__Line0).y > 0);
	(*__Line1).y += ((*__Line1).y > 0);
	(*__Line2).y += ((*__Line2).y > 0);
	(*__Line3).y += ((*__Line3).y > 0); 
    (*__Line0).z += ((*__Line0).z > 0);
	(*__Line1).z += ((*__Line1).z > 0);
	(*__Line2).z += ((*__Line2).z > 0);
	(*__Line3).z += ((*__Line3).z > 0);
    (*__Line0).w += ((*__Line0).w > 0);
	(*__Line1).w += ((*__Line1).w > 0);
	(*__Line2).w += ((*__Line2).w > 0);
	(*__Line3).w += ((*__Line3).w > 0);
	
	*__Line0 >>= 1;
    *__Line1 >>= 1;
    *__Line2 >>= 1;
    *__Line3 >>= 1;
	
	*__Line0 /= (int4)(dc_q, ac_q, ac_q, ac_q);
	*__Line1 /= ac_q;
	*__Line2 /= ac_q;
	*__Line3 /= ac_q;
	
	return;
}

void dequant_and_iWHT(int4 *const __Line0, int4 *const __Line1, int4 *const __Line2, int4 *const __Line3, const int dc_q, const int ac_q)
{
	int4 Line0, Line1, Line2, Line3;
	*__Line0 *= (int4)(dc_q, ac_q, ac_q, ac_q);
	*__Line1 *= ac_q;
	*__Line2 *= ac_q;
	*__Line3 *= ac_q;

	Line0.x = (*__Line0).x + (*__Line0).w;
	Line1.x = (*__Line1).x + (*__Line1).w;	
	Line2.x = (*__Line2).x + (*__Line2).w;
	Line3.x = (*__Line3).x + (*__Line3).w;
	
    Line0.y = (*__Line0).y + (*__Line0).z; 
	Line1.y = (*__Line1).y + (*__Line1).z;	
	Line2.y = (*__Line2).y + (*__Line2).z;
	Line3.y = (*__Line3).y + (*__Line3).z;   
	
    Line0.z = (*__Line0).y - (*__Line0).z; 
	Line1.z = (*__Line1).y - (*__Line1).z;	
	Line2.z = (*__Line2).y - (*__Line2).z;
	Line3.z = (*__Line3).y - (*__Line3).z;
    
	Line0.w = (*__Line0).x - (*__Line0).w; 
	Line1.w = (*__Line1).x - (*__Line1).w;	
	Line2.w = (*__Line2).x - (*__Line2).w;
	Line3.w = (*__Line3).x - (*__Line3).w;

	(*__Line0).x = Line0.x + Line0.y; 
	(*__Line1).x = Line1.x + Line1.y;
	(*__Line2).x = Line2.x + Line2.y;
	(*__Line3).x = Line3.x + Line3.y;
	
	(*__Line0).y = Line0.z + Line0.w; 
	(*__Line1).y = Line1.z + Line1.w;
	(*__Line2).y = Line2.z + Line2.w;
	(*__Line3).y = Line3.z + Line3.w;
	
	(*__Line0).z = Line0.x - Line0.y;
	(*__Line1).z = Line1.x - Line1.y;
	(*__Line2).z = Line2.x - Line2.y;
	(*__Line3).z = Line3.x - Line3.y;
        
	(*__Line0).w = Line0.w - Line0.z; 
	(*__Line1).w = Line1.w - Line1.z;
	(*__Line2).w = Line2.w - Line2.z;
	(*__Line3).w = Line3.w - Line3.z;

    Line0 = *__Line0 + *__Line3;
    Line1 = *__Line1 + *__Line2;
    Line2 = *__Line1 - *__Line2;
    Line3 = *__Line0 - *__Line3;

    *__Line0 = Line0 + Line1; *__Line1 = Line2 + Line3;
	*__Line2 = Line0 - Line1; *__Line3 = Line3 - Line2;

    *__Line0 += 3; 	*__Line1 += 3;	*__Line2 += 3; *__Line3 += 3;	
	*__Line0 >>= 3;  *__Line1 >>= 3; *__Line2 >>= 3; *__Line3 >>= 3;	

	return;
}


__kernel void reset_vectors ( __global vector_net *const net1, //0
								__global vector_net *const net2, //1
								__global macroblock *const MBs) //2
{
	int mb_num = get_global_id(0);
	net1[mb_num + 0].vector_x = 0;
	net1[mb_num + 0].vector_y = 0;
	net1[mb_num + 1].vector_x = 0;
	net1[mb_num + 1].vector_y = 0;
	net1[mb_num + 2].vector_x = 0;
	net1[mb_num + 2].vector_y = 0;
	net1[mb_num + 3].vector_x = 0;
	net1[mb_num + 3].vector_y = 0;
	net2[mb_num + 0].vector_x = 0;
	net2[mb_num + 0].vector_y = 0;
	net2[mb_num + 1].vector_x = 0;
	net2[mb_num + 1].vector_y = 0;
	net2[mb_num + 2].vector_x = 0;
	net2[mb_num + 2].vector_y = 0;
	net2[mb_num + 3].vector_x = 0;
	net2[mb_num + 3].vector_y = 0;
	MBs[mb_num].reference_frame = LAST;
	MBs[mb_num].segment_id = LQ_segment;
	
	return;
}

__kernel void downsample_x2(__global uchar *const src_frame, //0
							__global uchar *const dst_frame, //1
							const signed int src_width, //2
							const signed int src_height) //3
{
	//each thread takes 2x2 pixl block and downsample it to 1 pixel by average-value
	int b2x2_num = get_global_id(0);
	int x = (b2x2_num % (src_width/2))*2;
	int y = (b2x2_num / (src_width/2))*2;
	int i = y*src_width + x;
	int2 L;
	
	L = convert_int2(vload2(0, src_frame + i)); i += src_width;
	L += convert_int2(vload2(0, src_frame + i)); 
	L.x += L.y + 2;
	L.x /= 4;
	
	x /= 2; y /= 2;
	i = y*(src_width/2) + x;
	
	dst_frame[i] = L.x;
	return;
}

__kernel void luma_search_1step //when looking into downsampled and original frames
						( 	__global uchar *const current_frame, //0
							__global uchar *const prev_frame, //1
							__global vector_net *const src_net, //2
							__global vector_net *const dst_net, //3
							const signed int net_width, //4 //in 8x8 blocks
							const signed int width, //5
							const signed int height, //6
							const signed int pixel_rate) //7
{   
	__private uchar8 CL0, CL1, CL2, CL3, CL4, CL5, CL6, CL7;
	__private uchar8 PL0, PL1, PL2, PL3, PL4, PL5, PL6, PL7;
	
	__private int4 DL0, DL1, DL2, DL3;
	
	__private int start_x, end_x, start_y, end_y; 
	__private unsigned int MinDiff, Diff; 
	__private int px, py, cx, cy;
	__private int vector_x, vector_y, vector_x0, vector_y0;   
	__private int ci, pi;   
	__private int b8x8_num; 
	__private int cut_width;
	
	cut_width = (width / 8) * 8;
	
	//first we determine parameters of current stage
	b8x8_num = get_global_id(0);
	//b8x8_num == net_index
	cx = (b8x8_num % (cut_width/8))*8;
	cy = (b8x8_num / (cut_width/8))*8;
	
	vector_x = 0; vector_y = 0;
	if (pixel_rate < 16) //then we need to read previous stage vector
	{
		// we have to determine corresponding parameters of previous stage
		cx /= 2; cy /=2;
		b8x8_num = (cy/8)*net_width + (cx/8);
		// and read vectors
		vector_x = src_net[b8x8_num].vector_x;
		vector_y = src_net[b8x8_num].vector_y;
		vector_x /= pixel_rate;
		vector_y /= pixel_rate;
		cx *= 2; cy *=2; //return to current stage position
	} 
	//now block numbers in current stage
	vector_x0 = vector_x; vector_y0 = vector_y;

	b8x8_num = (cy/8)*net_width + (cx/8);
	ci = cy*width + cx;
	
	CL0 = vload8(0, current_frame + ci); ci += width;
	CL1 = vload8(0, current_frame + ci); ci += width;
	CL2 = vload8(0, current_frame + ci); ci += width;
	CL3 = vload8(0, current_frame + ci); ci += width;
	CL4 = vload8(0, current_frame + ci); ci += width;
	CL5 = vload8(0, current_frame + ci); ci += width;
	CL6 = vload8(0, current_frame + ci); ci += width;
	CL7 = vload8(0, current_frame + ci); 

	MinDiff = 0xffff;

	int delta = 2;
	//delta = (pixel_rate < 16) ? (delta+1) : delta;
	//delta = (pixel_rate < 4) ? (delta+1) : delta;
	
	start_x	= cx + vector_x - delta;	end_x = cx + vector_x + delta;
	start_y = cy + vector_y - delta;	end_y = cy + vector_y + delta;
	
	start_x = (start_x < 0) ? 0 : start_x;
	end_x = (end_x > (width - 8)) ? (width - 8) : end_x;
	start_y = (start_y < 0) ? 0 : start_y;
	end_y = (end_y > (height - 8)) ? (height - 8) : end_y;
	
	//if ((cx + vector_x) > end_x) printf ((__constant char*)"b=%d\tvx=%d\tsx=%d\tex=%d\tw=%d\n",b8x8_num,vector_x,start_x,end_x,width);
	
	#pragma unroll 1
	for (px = start_x; px <= end_x; ++px )
	{
		#pragma unroll 1
		for (py = start_y; py <= end_y; ++py)
		{ 
			// 1 full pixel step (fpel)
			pi = py*width + px;
			PL0 = vload8(0, prev_frame + pi); pi += width; 
			PL1 = vload8(0, prev_frame + pi); pi += width;
			PL2 = vload8(0, prev_frame + pi); pi += width;
			PL3 = vload8(0, prev_frame + pi); pi += width;
			PL4 = vload8(0, prev_frame + pi); pi += width;
			PL5 = vload8(0, prev_frame + pi); pi += width;
			PL6 = vload8(0, prev_frame + pi); pi += width;
			PL7 = vload8(0, prev_frame + pi); 
			// block 00
			DL0 = convert_int4(CL0.s0123) - convert_int4(PL0.s0123);
			DL1 = convert_int4(CL1.s0123) - convert_int4(PL1.s0123);
			DL2 = convert_int4(CL2.s0123) - convert_int4(PL2.s0123);
			DL3 = convert_int4(CL3.s0123) - convert_int4(PL3.s0123);
			weight(&DL0, &DL1, &DL2, &DL3);	Diff = DL0.x;
			// block 10			
			DL0 = convert_int4(CL4.s0123) - convert_int4(PL4.s0123);
			DL1 = convert_int4(CL5.s0123) - convert_int4(PL5.s0123);
			DL2 = convert_int4(CL6.s0123) - convert_int4(PL6.s0123);
			DL3 = convert_int4(CL7.s0123) - convert_int4(PL7.s0123);
			weight(&DL0, &DL1, &DL2, &DL3);	Diff += DL0.x;
			// block 01
			DL0 = convert_int4(CL0.s4567) - convert_int4(PL0.s4567);
			DL1 = convert_int4(CL1.s4567) - convert_int4(PL1.s4567);
			DL2 = convert_int4(CL2.s4567) - convert_int4(PL2.s4567);
			DL3 = convert_int4(CL3.s4567) - convert_int4(PL3.s4567);
			weight(&DL0, &DL1, &DL2, &DL3);	Diff += DL0.x;
			// block 11
			DL0 = convert_int4(CL4.s4567) - convert_int4(PL4.s4567);
			DL1 = convert_int4(CL5.s4567) - convert_int4(PL5.s4567);
			DL2 = convert_int4(CL6.s4567) - convert_int4(PL6.s4567);
			DL3 = convert_int4(CL7.s4567) - convert_int4(PL7.s4567);
			weight(&DL0, &DL1, &DL2, &DL3);	Diff += DL0.x;
			
			//Diff += ((int)abs(px-cx) + (int)abs(py-cy))*4;
			Diff += (int)(abs((int)abs(px-cx)-vector_x0) + abs((int)abs(py-cy)-vector_y0))*32;
						
			vector_x = (Diff < MinDiff) ? (px - cx) : vector_x;
			vector_y = (Diff < MinDiff) ? (py - cy) : vector_y;
			MinDiff = (Diff < MinDiff) ? Diff : MinDiff;			
    	} 
    } 
	
	
	dst_net[b8x8_num].vector_x = (vector_x*pixel_rate);
	dst_net[b8x8_num].vector_y = (vector_y*pixel_rate);
	
	return;
	
}
	
__kernel void luma_search_2step //searching in interpolated picture
						( 	__global uchar *const current_frame, //0
							__global uchar *const prev_frame, //1
							__global vector_net *const net, //2
							__global macroblock *const MBs, //3
							__global int *const MBdiff, //4
							const signed int width, //5
							const signed int height) //6
{
	__private uchar8 CL0, CL1, CL2, CL3, CL4, CL5, CL6, CL7;
	
	__private uchar4 UC00, UC01, UC02, UC03,
							UC10, UC11, UC12, UC13,
							UC20, UC21, UC22, UC23,
							UC30, UC31, UC32, UC33;
	__private int4 DL0, DL1, DL2, DL3;
	
	__private int start_x, end_x, start_y, end_y, vector_x, vector_y,vector_x0,vector_y0; 
	__private unsigned int MinDiff, Diff0, Diff1, Diff2, Diff3;	
	__private int px, py, cx, cy, ci, pi;
	__private int width_x4 = width*4;
	__private int width_x4_x4 = width_x4*4;
	__private int mb_num, b8x8_num;  
	
	// now b8x8_num represents absolute number of 8x8 block (net_index)
	b8x8_num = get_global_id(0);
	vector_x = net[b8x8_num].vector_x;
	vector_y = net[b8x8_num].vector_y;
	vector_x *= 4;	vector_y *= 4;
	vector_x0 = vector_x;
	vector_y0 = vector_y;

	cx = (b8x8_num % (width/8))*8;
	cy = (b8x8_num / (width/8))*8;
	
	mb_num = (cy/16)*(width/16) + (cx/16);
	// and now b8x8_num represents number of 8x8 block in 16x16 macroblock
	b8x8_num = ((cy%16)/8)*2 + (cx%16)/8;
	ci = cy*width + cx;
	
	CL0 = vload8(0, current_frame + ci); ci += width;
	CL1 = vload8(0, current_frame + ci); ci += width;
	CL2 = vload8(0, current_frame + ci); ci += width;
	CL3 = vload8(0, current_frame + ci); ci += width;
	CL4 = vload8(0, current_frame + ci); ci += width;
	CL5 = vload8(0, current_frame + ci); ci += width;
	CL6 = vload8(0, current_frame + ci); ci += width;
	CL7 = vload8(0, current_frame + ci); 

	MinDiff = 0xffff;

	//printf((__constant char*)"[%d]=%d; %d\n",b8x8_num,vector_x,vector_y);
	
	cx *= 4; cy *=4; //into qpel
	
	start_x = cx - 4 + vector_x;	end_x = cx + 3 + vector_x;
	start_y = cy + vector_y - 3;	end_y = cy + vector_y + 3;
	
	vector_x = 0; //in case previous iteration vectors fall out of frame and
	vector_y = 0; // loops never entered (this should not happen though)
	
	start_x = (start_x < 0) ? 0 : start_x;
	end_x = (end_x > (width_x4 - 32)) ? (width_x4 - 32) : end_x;
	start_y = (start_y < 0) ? 0 : start_y;
	end_y = (end_y > ((height*4) - 32)) ? ((height*4) - 32) : end_y;

	start_x &= ~0x3;
	#pragma unroll 1
	for (px = start_x; px <= end_x; px+=4 )
	{
		#pragma unroll 1
		for (py = start_y; py <= end_y; py+=1)
		{ 
			// 1 full pixel step (fpel)
			pi = py*width_x4 + px;
			// block 00
			UC00 = vload4(0,prev_frame+pi); UC01 = vload4(0,prev_frame+pi+4); UC02 = vload4(0,prev_frame+pi+8); UC03 = vload4(0,prev_frame+pi+12); pi += width_x4_x4;
			UC10 = vload4(0,prev_frame+pi); UC11 = vload4(0,prev_frame+pi+4); UC12 = vload4(0,prev_frame+pi+8); UC13 = vload4(0,prev_frame+pi+12); pi += width_x4_x4;
			UC20 = vload4(0,prev_frame+pi); UC21 = vload4(0,prev_frame+pi+4); UC22 = vload4(0,prev_frame+pi+8); UC23 = vload4(0,prev_frame+pi+12); pi += width_x4_x4;
			UC30 = vload4(0,prev_frame+pi); UC31 = vload4(0,prev_frame+pi+4); UC32 = vload4(0,prev_frame+pi+8); UC33 = vload4(0,prev_frame+pi+12); pi += width_x4_x4;			
			DL0 = convert_int4(CL0.s0123) - convert_int4((uchar4)(UC00.x, UC01.x, UC02.x, UC03.x));
			DL1 = convert_int4(CL1.s0123) - convert_int4((uchar4)(UC10.x, UC11.x, UC12.x, UC13.x));
			DL2 = convert_int4(CL2.s0123) - convert_int4((uchar4)(UC20.x, UC21.x, UC22.x, UC23.x));
			DL3 = convert_int4(CL3.s0123) - convert_int4((uchar4)(UC30.x, UC31.x, UC32.x, UC33.x));
			weight(&DL0, &DL1, &DL2, &DL3);	Diff0 = DL0.x;
			DL0 = convert_int4(CL0.s0123) - convert_int4((uchar4)(UC00.y, UC01.y, UC02.y, UC03.y));
			DL1 = convert_int4(CL1.s0123) - convert_int4((uchar4)(UC10.y, UC11.y, UC12.y, UC13.y));
			DL2 = convert_int4(CL2.s0123) - convert_int4((uchar4)(UC20.y, UC21.y, UC22.y, UC23.y));
			DL3 = convert_int4(CL3.s0123) - convert_int4((uchar4)(UC30.y, UC31.y, UC32.y, UC33.y));
			weight(&DL0, &DL1, &DL2, &DL3);	Diff1 = DL0.x;
			DL0 = convert_int4(CL0.s0123) - convert_int4((uchar4)(UC00.z, UC01.z, UC02.z, UC03.z));
			DL1 = convert_int4(CL1.s0123) - convert_int4((uchar4)(UC10.z, UC11.z, UC12.z, UC13.z));
			DL2 = convert_int4(CL2.s0123) - convert_int4((uchar4)(UC20.z, UC21.z, UC22.z, UC23.z));
			DL3 = convert_int4(CL3.s0123) - convert_int4((uchar4)(UC30.z, UC31.z, UC32.z, UC33.z));
			weight(&DL0, &DL1, &DL2, &DL3);	Diff2 = DL0.x;
			DL0 = convert_int4(CL0.s0123) - convert_int4((uchar4)(UC00.w, UC01.w, UC02.w, UC03.w));
			DL1 = convert_int4(CL1.s0123) - convert_int4((uchar4)(UC10.w, UC11.w, UC12.w, UC13.w));
			DL2 = convert_int4(CL2.s0123) - convert_int4((uchar4)(UC20.w, UC21.w, UC22.w, UC23.w));
			DL3 = convert_int4(CL3.s0123) - convert_int4((uchar4)(UC30.w, UC31.w, UC32.w, UC33.w));
			weight(&DL0, &DL1, &DL2, &DL3);	Diff3 = DL0.x;
			// block 10			
			UC00 = vload4(0,prev_frame+pi); UC01 = vload4(0,prev_frame+pi+4); UC02 = vload4(0,prev_frame+pi+8); UC03 = vload4(0,prev_frame+pi+12); pi += width_x4_x4;
			UC10 = vload4(0,prev_frame+pi); UC11 = vload4(0,prev_frame+pi+4); UC12 = vload4(0,prev_frame+pi+8); UC13 = vload4(0,prev_frame+pi+12); pi += width_x4_x4;
			UC20 = vload4(0,prev_frame+pi); UC21 = vload4(0,prev_frame+pi+4); UC22 = vload4(0,prev_frame+pi+8); UC23 = vload4(0,prev_frame+pi+12); pi += width_x4_x4;
			UC30 = vload4(0,prev_frame+pi); UC31 = vload4(0,prev_frame+pi+4); UC32 = vload4(0,prev_frame+pi+8); UC33 = vload4(0,prev_frame+pi+12); pi += width_x4_x4;
			DL0 = convert_int4(CL4.s0123) - convert_int4((uchar4)(UC00.x, UC01.x, UC02.x, UC03.x));
			DL1 = convert_int4(CL5.s0123) - convert_int4((uchar4)(UC10.x, UC11.x, UC12.x, UC13.x));
			DL2 = convert_int4(CL6.s0123) - convert_int4((uchar4)(UC20.x, UC21.x, UC22.x, UC23.x));
			DL3 = convert_int4(CL7.s0123) - convert_int4((uchar4)(UC30.x, UC31.x, UC32.x, UC33.x));
			weight(&DL0, &DL1, &DL2, &DL3);	Diff0 += DL0.x;
			DL0 = convert_int4(CL4.s0123) - convert_int4((uchar4)(UC00.y, UC01.y, UC02.y, UC03.y));
			DL1 = convert_int4(CL5.s0123) - convert_int4((uchar4)(UC10.y, UC11.y, UC12.y, UC13.y));
			DL2 = convert_int4(CL6.s0123) - convert_int4((uchar4)(UC20.y, UC21.y, UC22.y, UC23.y));
			DL3 = convert_int4(CL7.s0123) - convert_int4((uchar4)(UC30.y, UC31.y, UC32.y, UC33.y));
			weight(&DL0, &DL1, &DL2, &DL3);	Diff1 += DL0.x;
			DL0 = convert_int4(CL4.s0123) - convert_int4((uchar4)(UC00.z, UC01.z, UC02.z, UC03.z));
			DL1 = convert_int4(CL5.s0123) - convert_int4((uchar4)(UC10.z, UC11.z, UC12.z, UC13.z));
			DL2 = convert_int4(CL6.s0123) - convert_int4((uchar4)(UC20.z, UC21.z, UC22.z, UC23.z));
			DL3 = convert_int4(CL7.s0123) - convert_int4((uchar4)(UC30.z, UC31.z, UC32.z, UC33.z));
			weight(&DL0, &DL1, &DL2, &DL3);	Diff2 += DL0.x;
			DL0 = convert_int4(CL4.s0123) - convert_int4((uchar4)(UC00.w, UC01.w, UC02.w, UC03.w));
			DL1 = convert_int4(CL5.s0123) - convert_int4((uchar4)(UC10.w, UC11.w, UC12.w, UC13.w));
			DL2 = convert_int4(CL6.s0123) - convert_int4((uchar4)(UC20.w, UC21.w, UC22.w, UC23.w));
			DL3 = convert_int4(CL7.s0123) - convert_int4((uchar4)(UC30.w, UC31.w, UC32.w, UC33.w));
			weight(&DL0, &DL1, &DL2, &DL3);	Diff3 += DL0.x;
			pi -= (width_x4<<5);
			// block 01
			pi += 16;
			UC00 = vload4(0,prev_frame+pi); UC01 = vload4(0,prev_frame+pi+4); UC02 = vload4(0,prev_frame+pi+8); UC03 = vload4(0,prev_frame+pi+12); pi += width_x4_x4;
			UC10 = vload4(0,prev_frame+pi); UC11 = vload4(0,prev_frame+pi+4); UC12 = vload4(0,prev_frame+pi+8); UC13 = vload4(0,prev_frame+pi+12); pi += width_x4_x4;
			UC20 = vload4(0,prev_frame+pi); UC21 = vload4(0,prev_frame+pi+4); UC22 = vload4(0,prev_frame+pi+8); UC23 = vload4(0,prev_frame+pi+12); pi += width_x4_x4;
			UC30 = vload4(0,prev_frame+pi); UC31 = vload4(0,prev_frame+pi+4); UC32 = vload4(0,prev_frame+pi+8); UC33 = vload4(0,prev_frame+pi+12); pi += width_x4_x4;
			DL0 = convert_int4(CL0.s4567) - convert_int4((uchar4)(UC00.x, UC01.x, UC02.x, UC03.x));
			DL1 = convert_int4(CL1.s4567) - convert_int4((uchar4)(UC10.x, UC11.x, UC12.x, UC13.x));
			DL2 = convert_int4(CL2.s4567) - convert_int4((uchar4)(UC20.x, UC21.x, UC22.x, UC23.x));
			DL3 = convert_int4(CL3.s4567) - convert_int4((uchar4)(UC30.x, UC31.x, UC32.x, UC33.x));
			weight(&DL0, &DL1, &DL2, &DL3);	Diff0 += DL0.x;
			DL0 = convert_int4(CL0.s4567) - convert_int4((uchar4)(UC00.y, UC01.y, UC02.y, UC03.y));
			DL1 = convert_int4(CL1.s4567) - convert_int4((uchar4)(UC10.y, UC11.y, UC12.y, UC13.y));
			DL2 = convert_int4(CL2.s4567) - convert_int4((uchar4)(UC20.y, UC21.y, UC22.y, UC23.y));
			DL3 = convert_int4(CL3.s4567) - convert_int4((uchar4)(UC30.y, UC31.y, UC32.y, UC33.y));
			weight(&DL0, &DL1, &DL2, &DL3);	Diff1 += DL0.x;
			DL0 = convert_int4(CL0.s4567) - convert_int4((uchar4)(UC00.z, UC01.z, UC02.z, UC03.z));
			DL1 = convert_int4(CL1.s4567) - convert_int4((uchar4)(UC10.z, UC11.z, UC12.z, UC13.z));
			DL2 = convert_int4(CL2.s4567) - convert_int4((uchar4)(UC20.z, UC21.z, UC22.z, UC23.z));
			DL3 = convert_int4(CL3.s4567) - convert_int4((uchar4)(UC30.z, UC31.z, UC32.z, UC33.z));
			weight(&DL0, &DL1, &DL2, &DL3);	Diff2 += DL0.x;
			DL0 = convert_int4(CL0.s4567) - convert_int4((uchar4)(UC00.w, UC01.w, UC02.w, UC03.w));
			DL1 = convert_int4(CL1.s4567) - convert_int4((uchar4)(UC10.w, UC11.w, UC12.w, UC13.w));
			DL2 = convert_int4(CL2.s4567) - convert_int4((uchar4)(UC20.w, UC21.w, UC22.w, UC23.w));
			DL3 = convert_int4(CL3.s4567) - convert_int4((uchar4)(UC30.w, UC31.w, UC32.w, UC33.w));
			weight(&DL0, &DL1, &DL2, &DL3);	Diff3 += DL0.x;
			// block 11
			UC00 = vload4(0,prev_frame+pi); UC01 = vload4(0,prev_frame+pi+4); UC02 = vload4(0,prev_frame+pi+8); UC03 = vload4(0,prev_frame+pi+12); pi += width_x4_x4;
			UC10 = vload4(0,prev_frame+pi); UC11 = vload4(0,prev_frame+pi+4); UC12 = vload4(0,prev_frame+pi+8); UC13 = vload4(0,prev_frame+pi+12); pi += width_x4_x4;
			UC20 = vload4(0,prev_frame+pi); UC21 = vload4(0,prev_frame+pi+4); UC22 = vload4(0,prev_frame+pi+8); UC23 = vload4(0,prev_frame+pi+12); pi += width_x4_x4;
			UC30 = vload4(0,prev_frame+pi); UC31 = vload4(0,prev_frame+pi+4); UC32 = vload4(0,prev_frame+pi+8); UC33 = vload4(0,prev_frame+pi+12); pi += width_x4_x4;
			DL0 = convert_int4(CL4.s4567) - convert_int4((uchar4)(UC00.x, UC01.x, UC02.x, UC03.x));
			DL1 = convert_int4(CL5.s4567) - convert_int4((uchar4)(UC10.x, UC11.x, UC12.x, UC13.x));
			DL2 = convert_int4(CL6.s4567) - convert_int4((uchar4)(UC20.x, UC21.x, UC22.x, UC23.x));
			DL3 = convert_int4(CL7.s4567) - convert_int4((uchar4)(UC30.x, UC31.x, UC32.x, UC33.x));
			weight(&DL0, &DL1, &DL2, &DL3);	Diff0 += DL0.x;
			DL0 = convert_int4(CL4.s4567) - convert_int4((uchar4)(UC00.y, UC01.y, UC02.y, UC03.y));
			DL1 = convert_int4(CL5.s4567) - convert_int4((uchar4)(UC10.y, UC11.y, UC12.y, UC13.y));
			DL2 = convert_int4(CL6.s4567) - convert_int4((uchar4)(UC20.y, UC21.y, UC22.y, UC23.y));
			DL3 = convert_int4(CL7.s4567) - convert_int4((uchar4)(UC30.y, UC31.y, UC32.y, UC33.y));
			weight(&DL0, &DL1, &DL2, &DL3);	Diff1 += DL0.x;
			DL0 = convert_int4(CL4.s4567) - convert_int4((uchar4)(UC00.z, UC01.z, UC02.z, UC03.z));
			DL1 = convert_int4(CL5.s4567) - convert_int4((uchar4)(UC10.z, UC11.z, UC12.z, UC13.z));
			DL2 = convert_int4(CL6.s4567) - convert_int4((uchar4)(UC20.z, UC21.z, UC22.z, UC23.z));
			DL3 = convert_int4(CL7.s4567) - convert_int4((uchar4)(UC30.z, UC31.z, UC32.z, UC33.z));
			weight(&DL0, &DL1, &DL2, &DL3);	Diff2 += DL0.x;
			DL0 = convert_int4(CL4.s4567) - convert_int4((uchar4)(UC00.w, UC01.w, UC02.w, UC03.w));
			DL1 = convert_int4(CL5.s4567) - convert_int4((uchar4)(UC10.w, UC11.w, UC12.w, UC13.w));
			DL2 = convert_int4(CL6.s4567) - convert_int4((uchar4)(UC20.w, UC21.w, UC22.w, UC23.w));
			DL3 = convert_int4(CL7.s4567) - convert_int4((uchar4)(UC30.w, UC31.w, UC32.w, UC33.w));
			weight(&DL0, &DL1, &DL2, &DL3);	Diff3 += DL0.x;
			
			//Diff0 += abs(px - cx) + abs(py - cy);
			Diff0 += (int)(abs(px-cx-vector_x0) + abs(py-cy-vector_y0))*16;
			//Diff1 += abs(px+1 - cx) + abs(py - cy);
			Diff1 += (int)(abs(px+1-cx-vector_x0) + abs(py-cy-vector_y0))*16;
			//Diff2 += abs(px+2 - cx) + abs(py - cy);
			Diff2 += (int)(abs(px+2-cx-vector_x0) + abs(py-cy-vector_y0))*16;
			//Diff3 += abs(px+3 - cx) + abs(py - cy);
			Diff3 += (int)(abs(px+3-cx-vector_x0) + abs(py-cy-vector_y0))*16;

			vector_x = (Diff0 < MinDiff) ? (px - cx) : vector_x;
			vector_y = (Diff0 < MinDiff) ? (py - cy) : vector_y;
			MinDiff = (Diff0 < MinDiff) ? Diff0 : MinDiff;
			
			vector_x = (Diff1 < MinDiff) ? (px+1 - cx) : vector_x;
			vector_y = (Diff1 < MinDiff) ? (py - cy) : vector_y;
			MinDiff = (Diff1 < MinDiff) ? Diff1 : MinDiff;
			
			vector_x = (Diff2 < MinDiff) ? (px+2 - cx) : vector_x;
			vector_y = (Diff2 < MinDiff) ? (py - cy) : vector_y;
			MinDiff = (Diff2 < MinDiff) ? Diff2 : MinDiff;
			
			vector_x = (Diff3 < MinDiff) ? (px+3 - cx) : vector_x;
			vector_y = (Diff3 < MinDiff) ? (py - cy) : vector_y;
			MinDiff = (Diff3 < MinDiff) ? Diff3 : MinDiff;
    	} 
    } 
	
	MBs[mb_num].vector_x[b8x8_num] = vector_x;
	MBs[mb_num].vector_y[b8x8_num] = vector_y;
	
	MBdiff[mb_num*4+b8x8_num] = (MinDiff - ((int)abs(vector_x - vector_x0) + (int)abs(vector_y - vector_y0))*16);
	
	return;
}
	
__kernel void try_golden_reference(__global uchar *const current_frame_Y, //0
								__global uchar *const golden_frame_Y, //1
								__global uchar *const recon_frame_Y, //2
								__global uchar *const current_frame_U, //3
								__global uchar *const golden_frame_U, //4
								__global uchar *const recon_frame_U, //5
								__global uchar *const current_frame_V, //6
								__global uchar *const golden_frame_V, //7
								__global uchar *const recon_frame_V, //8
								__global macroblock *const MBs, //9
								__global int *const MBdiff, //10
								const int width, //11
								__constant segment_data *const SD) //12
{
	//this kernel try to compare difference from search  to difference resulting from vector(0;0) into golden frame
	//if golden reference is better kernel does DCT and WHT on luma and chroma also

	__private int mb_num,x,y,i,y_ac_q,y2_dc_q,y2_ac_q,uv_dc_q,uv_ac_q;
	__private int4 DL0,DL1,DL2,DL3;
	__private short16 L0,L1,L2,L3,L4,L5,L6,L7,L8,L9,L10,L11,L12,L13,L14,L15;
	
	mb_num = get_global_id(0);
	x = (mb_num % (width/16))*16;
	y = (mb_num / (width/16))*16;
	
	i = y*width + x;
	L0 = convert_short16(vload16(0, current_frame_Y+i)); i+= width;
	L1 = convert_short16(vload16(0, current_frame_Y+i)); i+= width;
	L2 = convert_short16(vload16(0, current_frame_Y+i)); i+= width;
	L3 = convert_short16(vload16(0, current_frame_Y+i)); i+= width;
	L4 = convert_short16(vload16(0, current_frame_Y+i)); i+= width;
	L5 = convert_short16(vload16(0, current_frame_Y+i)); i+= width;
	L6 = convert_short16(vload16(0, current_frame_Y+i)); i+= width;
	L7 = convert_short16(vload16(0, current_frame_Y+i)); i+= width;
	L8 = convert_short16(vload16(0, current_frame_Y+i)); i+= width;
	L9 = convert_short16(vload16(0, current_frame_Y+i)); i+= width;
	L10 = convert_short16(vload16(0, current_frame_Y+i)); i+= width;
	L11 = convert_short16(vload16(0, current_frame_Y+i)); i+= width;
	L12 = convert_short16(vload16(0, current_frame_Y+i)); i+= width;
	L13 = convert_short16(vload16(0, current_frame_Y+i)); i+= width;
	L14 = convert_short16(vload16(0, current_frame_Y+i)); i+= width;
	L15 = convert_short16(vload16(0, current_frame_Y+i));
	
	i = y*width + x;
	//now read reference and do transform
	//but with 1,1 quantizer, because we need DCT metrics
	L0 -= convert_short16(vload16(0, golden_frame_Y+i)); i+=width;
	L1 -= convert_short16(vload16(0, golden_frame_Y+i)); i+=width;
	L2 -= convert_short16(vload16(0, golden_frame_Y+i)); i+=width;
	L3 -= convert_short16(vload16(0, golden_frame_Y+i)); i+=width;
	DL0 = convert_int4(L0.s0123); DL1 = convert_int4(L1.s0123); DL2 = convert_int4(L2.s0123); DL3 = convert_int4(L3.s0123);
	DCT_and_quant(&DL0, &DL1, &DL2, &DL3, 1, 1);
	L0.s0123 = convert_short4(DL0);	L1.s0123 = convert_short4(DL1);	L2.s0123 = convert_short4(DL2);	L3.s0123 = convert_short4(DL3);
	DL0 = convert_int4(L0.s4567); DL1 = convert_int4(L1.s4567); DL2 = convert_int4(L2.s4567); DL3 = convert_int4(L3.s4567);
	DCT_and_quant(&DL0, &DL1, &DL2, &DL3, 1, 1);
	L0.s4567 = convert_short4(DL0);	L1.s4567 = convert_short4(DL1);	L2.s4567 = convert_short4(DL2);	L3.s4567 = convert_short4(DL3);
	DL0 = convert_int4(L0.s89AB); DL1 = convert_int4(L1.s89AB); DL2 = convert_int4(L2.s89AB); DL3 = convert_int4(L3.s89AB);
	DCT_and_quant(&DL0, &DL1, &DL2, &DL3, 1, 1);
	L0.s89AB = convert_short4(DL0);	L1.s89AB = convert_short4(DL1);	L2.s89AB = convert_short4(DL2);	L3.s89AB = convert_short4(DL3);
	DL0 = convert_int4(L0.sCDEF); DL1 = convert_int4(L1.sCDEF); DL2 = convert_int4(L2.sCDEF); DL3 = convert_int4(L3.sCDEF);
	DCT_and_quant(&DL0, &DL1, &DL2, &DL3, 1, 1);
	L0.sCDEF = convert_short4(DL0);	L1.sCDEF = convert_short4(DL1);	L2.sCDEF = convert_short4(DL2);	L3.sCDEF = convert_short4(DL3);
	L4 -= convert_short16(vload16(0, golden_frame_Y+i)); i+=width;
	L5 -= convert_short16(vload16(0, golden_frame_Y+i)); i+=width;
	L6 -= convert_short16(vload16(0, golden_frame_Y+i)); i+=width;
	L7 -= convert_short16(vload16(0, golden_frame_Y+i)); i+=width;
	DL0 = convert_int4(L4.s0123); DL1 = convert_int4(L5.s0123); DL2 = convert_int4(L6.s0123); DL3 = convert_int4(L7.s0123);
	DCT_and_quant(&DL0, &DL1, &DL2, &DL3, 1, 1);
	L4.s0123 = convert_short4(DL0);	L5.s0123 = convert_short4(DL1);	L6.s0123 = convert_short4(DL2);	L7.s0123 = convert_short4(DL3);
	DL0 = convert_int4(L4.s4567); DL1 = convert_int4(L5.s4567); DL2 = convert_int4(L6.s4567); DL3 = convert_int4(L7.s4567);
	DCT_and_quant(&DL0, &DL1, &DL2, &DL3, 1, 1);
	L4.s4567 = convert_short4(DL0);	L5.s4567 = convert_short4(DL1);	L6.s4567 = convert_short4(DL2);	L7.s4567 = convert_short4(DL3);
	DL0 = convert_int4(L4.s89AB); DL1 = convert_int4(L5.s89AB); DL2 = convert_int4(L6.s89AB); DL3 = convert_int4(L7.s89AB);
	DCT_and_quant(&DL0, &DL1, &DL2, &DL3, 1, 1);
	L4.s89AB = convert_short4(DL0);	L5.s89AB = convert_short4(DL1);	L6.s89AB = convert_short4(DL2);	L7.s89AB = convert_short4(DL3);
	DL0 = convert_int4(L4.sCDEF); DL1 = convert_int4(L5.sCDEF); DL2 = convert_int4(L6.sCDEF); DL3 = convert_int4(L7.sCDEF);
	DCT_and_quant(&DL0, &DL1, &DL2, &DL3, 1, 1);
	L4.sCDEF = convert_short4(DL0);	L5.sCDEF = convert_short4(DL1);	L6.sCDEF = convert_short4(DL2);	L7.sCDEF = convert_short4(DL3);
	L8 -= convert_short16(vload16(0, golden_frame_Y+i)); i+=width;
	L9 -= convert_short16(vload16(0, golden_frame_Y+i)); i+=width;
	L10 -= convert_short16(vload16(0, golden_frame_Y+i)); i+=width;
	L11 -= convert_short16(vload16(0, golden_frame_Y+i)); i+=width;
	DL0 = convert_int4(L8.s0123); DL1 = convert_int4(L9.s0123); DL2 = convert_int4(L10.s0123); DL3 = convert_int4(L11.s0123);
	DCT_and_quant(&DL0, &DL1, &DL2, &DL3, 1, 1);
	L8.s0123 = convert_short4(DL0);	L9.s0123 = convert_short4(DL1);	L10.s0123 = convert_short4(DL2);	L11.s0123 = convert_short4(DL3);
	DL0 = convert_int4(L8.s4567); DL1 = convert_int4(L9.s4567); DL2 = convert_int4(L10.s4567); DL3 = convert_int4(L11.s4567);
	DCT_and_quant(&DL0, &DL1, &DL2, &DL3, 1, 1);
	L8.s4567 = convert_short4(DL0);	L9.s4567 = convert_short4(DL1);	L10.s4567 = convert_short4(DL2);	L11.s4567 = convert_short4(DL3);
	DL0 = convert_int4(L8.s89AB); DL1 = convert_int4(L9.s89AB); DL2 = convert_int4(L10.s89AB); DL3 = convert_int4(L11.s89AB);
	DCT_and_quant(&DL0, &DL1, &DL2, &DL3, 1, 1);
	L8.s89AB = convert_short4(DL0);	L9.s89AB = convert_short4(DL1);	L10.s89AB = convert_short4(DL2);	L11.s89AB = convert_short4(DL3);
	DL0 = convert_int4(L8.sCDEF); DL1 = convert_int4(L9.sCDEF); DL2 = convert_int4(L10.sCDEF); DL3 = convert_int4(L11.sCDEF);
	DCT_and_quant(&DL0, &DL1, &DL2, &DL3, 1, 1);
	L8.sCDEF = convert_short4(DL0);	L9.sCDEF = convert_short4(DL1);	L10.sCDEF = convert_short4(DL2);	L11.sCDEF = convert_short4(DL3);
	L12 -= convert_short16(vload16(0, golden_frame_Y+i)); i+=width;
	L13 -= convert_short16(vload16(0, golden_frame_Y+i)); i+=width;
	L14 -= convert_short16(vload16(0, golden_frame_Y+i)); i+=width;
	L15 -= convert_short16(vload16(0, golden_frame_Y+i)); i+=width;
	DL0 = convert_int4(L12.s0123); DL1 = convert_int4(L13.s0123); DL2 = convert_int4(L14.s0123); DL3 = convert_int4(L15.s0123);
	DCT_and_quant(&DL0, &DL1, &DL2, &DL3, 1, 1);
	L12.s0123 = convert_short4(DL0);	L13.s0123 = convert_short4(DL1);	L14.s0123 = convert_short4(DL2);	L15.s0123 = convert_short4(DL3);
	DL0 = convert_int4(L12.s4567); DL1 = convert_int4(L13.s4567); DL2 = convert_int4(L14.s4567); DL3 = convert_int4(L15.s4567);
	DCT_and_quant(&DL0, &DL1, &DL2, &DL3, 1, 1);
	L12.s4567 = convert_short4(DL0);	L13.s4567 = convert_short4(DL1);	L14.s4567 = convert_short4(DL2);	L15.s4567 = convert_short4(DL3);
	DL0 = convert_int4(L12.s89AB); DL1 = convert_int4(L13.s89AB); DL2 = convert_int4(L14.s89AB); DL3 = convert_int4(L15.s89AB);
	DCT_and_quant(&DL0, &DL1, &DL2, &DL3, 1, 1);
	L12.s89AB = convert_short4(DL0);	L13.s89AB = convert_short4(DL1);	L14.s89AB = convert_short4(DL2);	L15.s89AB = convert_short4(DL3);
	DL0 = convert_int4(L12.sCDEF); DL1 = convert_int4(L13.sCDEF); DL2 = convert_int4(L14.sCDEF); DL3 = convert_int4(L15.sCDEF);
	DCT_and_quant(&DL0, &DL1, &DL2, &DL3, 1, 1);
	L12.sCDEF = convert_short4(DL0);	L13.sCDEF = convert_short4(DL1);	L14.sCDEF = convert_short4(DL2);	L15.sCDEF = convert_short4(DL3);	
	
	//now metrics which is sum of DCT coefficients (DC divided by 4);
	DL0 = convert_int4(abs(convert_int4(L0.s0123)) + abs(convert_int4(L0.s4567)) + abs(convert_int4(L0.s89AB)) + abs(convert_int4(L0.sCDEF)));
	DL1 = convert_int4(abs(convert_int4(L1.s0123)) + abs(convert_int4(L1.s4567)) + abs(convert_int4(L1.s89AB)) + abs(convert_int4(L1.sCDEF)));
	DL2 = convert_int4(abs(convert_int4(L2.s0123)) + abs(convert_int4(L2.s4567)) + abs(convert_int4(L2.s89AB)) + abs(convert_int4(L2.sCDEF)));
	DL3 = convert_int4(abs(convert_int4(L3.s0123)) + abs(convert_int4(L3.s4567)) + abs(convert_int4(L3.s89AB)) + abs(convert_int4(L3.sCDEF)));
	DL0 += convert_int4(abs(convert_int4(L4.s0123)) + abs(convert_int4(L4.s4567)) + abs(convert_int4(L4.s89AB)) + abs(convert_int4(L4.sCDEF)));
	DL1 += convert_int4(abs(convert_int4(L5.s0123)) + abs(convert_int4(L5.s4567)) + abs(convert_int4(L5.s89AB)) + abs(convert_int4(L5.sCDEF)));
	DL2 += convert_int4(abs(convert_int4(L6.s0123)) + abs(convert_int4(L6.s4567)) + abs(convert_int4(L6.s89AB)) + abs(convert_int4(L6.sCDEF)));
	DL3 += convert_int4(abs(convert_int4(L7.s0123)) + abs(convert_int4(L7.s4567)) + abs(convert_int4(L7.s89AB)) + abs(convert_int4(L7.sCDEF)));
	DL0 += convert_int4(abs(convert_int4(L8.s0123)) + abs(convert_int4(L8.s4567)) + abs(convert_int4(L8.s89AB)) + abs(convert_int4(L8.sCDEF)));
	DL1 += convert_int4(abs(convert_int4(L9.s0123)) + abs(convert_int4(L9.s4567)) + abs(convert_int4(L9.s89AB)) + abs(convert_int4(L9.sCDEF)));
	DL2 += convert_int4(abs(convert_int4(L10.s0123)) + abs(convert_int4(L10.s4567)) + abs(convert_int4(L10.s89AB)) + abs(convert_int4(L10.sCDEF)));
	DL3 += convert_int4(abs(convert_int4(L11.s0123)) + abs(convert_int4(L11.s4567)) + abs(convert_int4(L11.s89AB)) + abs(convert_int4(L11.sCDEF)));
	DL0 += convert_int4(abs(convert_int4(L12.s0123)) + abs(convert_int4(L12.s4567)) + abs(convert_int4(L12.s89AB)) + abs(convert_int4(L12.sCDEF)));
	DL1 += convert_int4(abs(convert_int4(L13.s0123)) + abs(convert_int4(L13.s4567)) + abs(convert_int4(L13.s89AB)) + abs(convert_int4(L13.sCDEF)));
	DL2 += convert_int4(abs(convert_int4(L14.s0123)) + abs(convert_int4(L14.s4567)) + abs(convert_int4(L14.s89AB)) + abs(convert_int4(L14.sCDEF)));
	DL3 += convert_int4(abs(convert_int4(L15.s0123)) + abs(convert_int4(L15.s4567)) + abs(convert_int4(L15.s89AB)) + abs(convert_int4(L15.sCDEF)));
	DL0.x /= 4; DL0+=DL1+DL2+DL3; 
	DL0.x+=DL0.y+DL0.z+DL0.w+1; //golden reference metrics
	DL1.x = MBdiff[mb_num*4];
	DL1.x += MBdiff[mb_num*4+1];
	DL1.x += MBdiff[mb_num*4+2];
	DL1.x += MBdiff[mb_num*4+3]; //best last reference metrics minus vector part
	
	if (DL0.x > DL1.x) return; //last reference is better
	//else
	y_ac_q = MBs[mb_num].segment_id;
	y_ac_q = SD[y_ac_q].y_ac_i;
	y2_dc_q = SD[0].y2_dc_idelta; y2_dc_q += y_ac_q;
	y2_ac_q = SD[0].y2_ac_idelta; y2_ac_q += y_ac_q;
	uv_dc_q = SD[0].uv_dc_idelta; uv_dc_q += y_ac_q;
	uv_ac_q = SD[0].uv_ac_idelta; uv_ac_q += y_ac_q;
	y2_dc_q = select(y2_dc_q,0,y2_dc_q<0); y2_dc_q = select(y2_dc_q,127,y2_dc_q>127);
	y2_ac_q = select(y2_ac_q,0,y2_ac_q<0); y2_ac_q = select(y2_ac_q,127,y2_ac_q>127);
	uv_dc_q = select(uv_dc_q,0,uv_dc_q<0); uv_dc_q = select(uv_dc_q,127,uv_dc_q>127);
	uv_ac_q = select(uv_ac_q,0,uv_ac_q<0); uv_ac_q = select(uv_ac_q,127,uv_ac_q>127);
	y_ac_q = vp8_ac_qlookup[y_ac_q];
	y2_dc_q = (vp8_dc_qlookup[y2_dc_q])*2;
	y2_ac_q = 31*(vp8_ac_qlookup[y2_ac_q])/20;
	uv_dc_q = vp8_dc_qlookup[uv_dc_q];
	uv_ac_q = vp8_ac_qlookup[uv_ac_q];
	y2_ac_q = select(y2_ac_q,8,y2_ac_q<8);
	uv_dc_q = select(uv_dc_q,132,uv_dc_q>132);
	
	MBs[mb_num].reference_frame = GOLDEN;
	MBs[mb_num].parts = are16x16; 
	MBs[mb_num].vector_x[0] = 0;
	MBs[mb_num].vector_x[1] = 0;
	MBs[mb_num].vector_x[2] = 0;
	MBs[mb_num].vector_x[3] = 0;
	MBs[mb_num].vector_y[0] = 0;
	MBs[mb_num].vector_y[1] = 0;
	MBs[mb_num].vector_y[2] = 0;
	MBs[mb_num].vector_y[3] = 0;
	
	//we need to quant AC values
	DL0 = (int4)(1, y_ac_q, y_ac_q, y_ac_q);
	L0.s0123 /= convert_short4(DL0); L0.s4567 /= convert_short4(DL0); L0.s89AB /= convert_short4(DL0); L0.sCDEF /= convert_short4(DL0);
	L1 /= (short16)y_ac_q; L2 /= (short16)y_ac_q; L3 /= (short16)y_ac_q;
	L4.s0123 /= convert_short4(DL0); L4.s4567 /= convert_short4(DL0); L4.s89AB /= convert_short4(DL0); L4.sCDEF /= convert_short4(DL0);
	L5 /= (short16)y_ac_q; L6  /= (short16)y_ac_q; L7 /= (short16)y_ac_q;
	L8.s0123 /= convert_short4(DL0); L8.s4567 /= convert_short4(DL0); L8.s89AB /= convert_short4(DL0); L8.sCDEF /= convert_short4(DL0);
	L9 /= (short16)y_ac_q; L10 /= (short16)y_ac_q; L11 /= (short16)y_ac_q;
	L12.s0123 /= convert_short4(DL0); L12.s4567 /= convert_short4(DL0); L12.s89AB /= convert_short4(DL0); L12.sCDEF /= convert_short4(DL0);
	L13 /= (short16)y_ac_q; L14 /= (short16)y_ac_q; L15 /= (short16)y_ac_q;
	
	//now we need to WHTransform DC-coefficients
	DL0=convert_int4((short4)(L0.s0, L0.s4, L0.s8, L0.sC));
	DL1=convert_int4((short4)(L4.s0, L4.s4, L4.s8, L4.sC));
	DL2=convert_int4((short4)(L8.s0, L8.s4, L8.s8, L8.sC));
	DL3=convert_int4((short4)(L12.s0,L12.s4,L12.s8,L12.sC));
	WHT_and_quant(&DL0, &DL1, &DL2, &DL3, y2_dc_q, y2_ac_q);
	
	//now fill transformed bloks data
	const int inv_zigzag[16] = { 0, 1, 5, 6, 2, 4, 7, 12, 3,  8, 11, 13, 9, 10, 14, 15 };
	//mb_row 0
	                                             MBs[mb_num].coeffs[0][inv_zigzag[1]]=L0.s1;  MBs[mb_num].coeffs[0][inv_zigzag[2]]=L0.s2;  MBs[mb_num].coeffs[0][inv_zigzag[3]]=L0.s3;
	MBs[mb_num].coeffs[0][inv_zigzag[4]]=L1.s0;	 MBs[mb_num].coeffs[0][inv_zigzag[5]]=L1.s1;  MBs[mb_num].coeffs[0][inv_zigzag[6]]=L1.s2;  MBs[mb_num].coeffs[0][inv_zigzag[7]]=L1.s3;
	MBs[mb_num].coeffs[0][inv_zigzag[8]]=L2.s0;	 MBs[mb_num].coeffs[0][inv_zigzag[9]]=L2.s1;  MBs[mb_num].coeffs[0][inv_zigzag[10]]=L2.s2; MBs[mb_num].coeffs[0][inv_zigzag[11]]=L2.s3;
	MBs[mb_num].coeffs[0][inv_zigzag[12]]=L3.s0; MBs[mb_num].coeffs[0][inv_zigzag[13]]=L3.s1; MBs[mb_num].coeffs[0][inv_zigzag[14]]=L3.s2; MBs[mb_num].coeffs[0][inv_zigzag[15]]=L3.s3;
	//block 01
	                                             MBs[mb_num].coeffs[1][inv_zigzag[1]]=L0.s5;  MBs[mb_num].coeffs[1][inv_zigzag[2]]=L0.s6;  MBs[mb_num].coeffs[1][inv_zigzag[3]]=L0.s7;
	MBs[mb_num].coeffs[1][inv_zigzag[4]]=L1.s4;	 MBs[mb_num].coeffs[1][inv_zigzag[5]]=L1.s5;  MBs[mb_num].coeffs[1][inv_zigzag[6]]=L1.s6;  MBs[mb_num].coeffs[1][inv_zigzag[7]]=L1.s7;
	MBs[mb_num].coeffs[1][inv_zigzag[8]]=L2.s4;	 MBs[mb_num].coeffs[1][inv_zigzag[9]]=L2.s5;  MBs[mb_num].coeffs[1][inv_zigzag[10]]=L2.s6; MBs[mb_num].coeffs[1][inv_zigzag[11]]=L2.s7;
	MBs[mb_num].coeffs[1][inv_zigzag[12]]=L3.s4; MBs[mb_num].coeffs[1][inv_zigzag[13]]=L3.s5; MBs[mb_num].coeffs[1][inv_zigzag[14]]=L3.s6; MBs[mb_num].coeffs[1][inv_zigzag[15]]=L3.s7;
	//block 02
	                                             MBs[mb_num].coeffs[2][inv_zigzag[1]]=L0.s9;  MBs[mb_num].coeffs[2][inv_zigzag[2]]=L0.sA;  MBs[mb_num].coeffs[2][inv_zigzag[3]]=L0.sB;
	MBs[mb_num].coeffs[2][inv_zigzag[4]]=L1.s8;	 MBs[mb_num].coeffs[2][inv_zigzag[5]]=L1.s9;  MBs[mb_num].coeffs[2][inv_zigzag[6]]=L1.sA;  MBs[mb_num].coeffs[2][inv_zigzag[7]]=L1.sB;
	MBs[mb_num].coeffs[2][inv_zigzag[8]]=L2.s8;	 MBs[mb_num].coeffs[2][inv_zigzag[9]]=L2.s9;  MBs[mb_num].coeffs[2][inv_zigzag[10]]=L2.sA; MBs[mb_num].coeffs[2][inv_zigzag[11]]=L2.sB;
	MBs[mb_num].coeffs[2][inv_zigzag[12]]=L3.s8; MBs[mb_num].coeffs[2][inv_zigzag[13]]=L3.s9; MBs[mb_num].coeffs[2][inv_zigzag[14]]=L3.sA; MBs[mb_num].coeffs[2][inv_zigzag[15]]=L3.sB;
	//block 03
	                                             MBs[mb_num].coeffs[3][inv_zigzag[1]]=L0.sD;  MBs[mb_num].coeffs[3][inv_zigzag[2]]=L0.sE;  MBs[mb_num].coeffs[3][inv_zigzag[3]]=L0.sF;
	MBs[mb_num].coeffs[3][inv_zigzag[4]]=L1.sC;	 MBs[mb_num].coeffs[3][inv_zigzag[5]]=L1.sD;  MBs[mb_num].coeffs[3][inv_zigzag[6]]=L1.sE;  MBs[mb_num].coeffs[3][inv_zigzag[7]]=L1.sF;
	MBs[mb_num].coeffs[3][inv_zigzag[8]]=L2.sC;	 MBs[mb_num].coeffs[3][inv_zigzag[9]]=L2.sD;  MBs[mb_num].coeffs[3][inv_zigzag[10]]=L2.sE; MBs[mb_num].coeffs[3][inv_zigzag[11]]=L2.sF;
	MBs[mb_num].coeffs[3][inv_zigzag[12]]=L3.sC; MBs[mb_num].coeffs[3][inv_zigzag[13]]=L3.sD; MBs[mb_num].coeffs[3][inv_zigzag[14]]=L3.sE; MBs[mb_num].coeffs[3][inv_zigzag[15]]=L3.sF;
	//mb_row 1
	//block 10
	                                             MBs[mb_num].coeffs[4][inv_zigzag[1]]=L4.s1;  MBs[mb_num].coeffs[4][inv_zigzag[2]]=L4.s2;  MBs[mb_num].coeffs[4][inv_zigzag[3]]=L4.s3;
	MBs[mb_num].coeffs[4][inv_zigzag[4]]=L5.s0;	 MBs[mb_num].coeffs[4][inv_zigzag[5]]=L5.s1;  MBs[mb_num].coeffs[4][inv_zigzag[6]]=L5.s2;  MBs[mb_num].coeffs[4][inv_zigzag[7]]=L5.s3;
	MBs[mb_num].coeffs[4][inv_zigzag[8]]=L6.s0;	 MBs[mb_num].coeffs[4][inv_zigzag[9]]=L6.s1;  MBs[mb_num].coeffs[4][inv_zigzag[10]]=L6.s2; MBs[mb_num].coeffs[4][inv_zigzag[11]]=L6.s3;
	MBs[mb_num].coeffs[4][inv_zigzag[12]]=L7.s0; MBs[mb_num].coeffs[4][inv_zigzag[13]]=L7.s1; MBs[mb_num].coeffs[4][inv_zigzag[14]]=L7.s2; MBs[mb_num].coeffs[4][inv_zigzag[15]]=L7.s3;
	//block 11
	                                             MBs[mb_num].coeffs[5][inv_zigzag[1]]=L4.s5;  MBs[mb_num].coeffs[5][inv_zigzag[2]]=L4.s6;  MBs[mb_num].coeffs[5][inv_zigzag[3]]=L4.s7;
	MBs[mb_num].coeffs[5][inv_zigzag[4]]=L5.s4;	 MBs[mb_num].coeffs[5][inv_zigzag[5]]=L5.s5;  MBs[mb_num].coeffs[5][inv_zigzag[6]]=L5.s6;  MBs[mb_num].coeffs[5][inv_zigzag[7]]=L5.s7;
	MBs[mb_num].coeffs[5][inv_zigzag[8]]=L6.s4;	 MBs[mb_num].coeffs[5][inv_zigzag[9]]=L6.s5;  MBs[mb_num].coeffs[5][inv_zigzag[10]]=L6.s6; MBs[mb_num].coeffs[5][inv_zigzag[11]]=L6.s7;
	MBs[mb_num].coeffs[5][inv_zigzag[12]]=L7.s4; MBs[mb_num].coeffs[5][inv_zigzag[13]]=L7.s5; MBs[mb_num].coeffs[5][inv_zigzag[14]]=L7.s6; MBs[mb_num].coeffs[5][inv_zigzag[15]]=L7.s7;
	//block 12
	                                             MBs[mb_num].coeffs[6][inv_zigzag[1]]=L4.s9;  MBs[mb_num].coeffs[6][inv_zigzag[2]]=L4.sA;  MBs[mb_num].coeffs[6][inv_zigzag[3]]=L4.sB;
	MBs[mb_num].coeffs[6][inv_zigzag[4]]=L5.s8;	 MBs[mb_num].coeffs[6][inv_zigzag[5]]=L5.s9;  MBs[mb_num].coeffs[6][inv_zigzag[6]]=L5.sA;  MBs[mb_num].coeffs[6][inv_zigzag[7]]=L5.sB;
	MBs[mb_num].coeffs[6][inv_zigzag[8]]=L6.s8;	 MBs[mb_num].coeffs[6][inv_zigzag[9]]=L6.s9;  MBs[mb_num].coeffs[6][inv_zigzag[10]]=L6.sA; MBs[mb_num].coeffs[6][inv_zigzag[11]]=L6.sB;
	MBs[mb_num].coeffs[6][inv_zigzag[12]]=L7.s8; MBs[mb_num].coeffs[6][inv_zigzag[13]]=L7.s9; MBs[mb_num].coeffs[6][inv_zigzag[14]]=L7.sA; MBs[mb_num].coeffs[6][inv_zigzag[15]]=L7.sB;
	//block 13
	                                             MBs[mb_num].coeffs[7][inv_zigzag[1]]=L4.sD;  MBs[mb_num].coeffs[7][inv_zigzag[2]]=L4.sE;  MBs[mb_num].coeffs[7][inv_zigzag[3]]=L4.sF;
	MBs[mb_num].coeffs[7][inv_zigzag[4]]=L5.sC;	 MBs[mb_num].coeffs[7][inv_zigzag[5]]=L5.sD;  MBs[mb_num].coeffs[7][inv_zigzag[6]]=L5.sE;  MBs[mb_num].coeffs[7][inv_zigzag[7]]=L5.sF;
	MBs[mb_num].coeffs[7][inv_zigzag[8]]=L6.sC;	 MBs[mb_num].coeffs[7][inv_zigzag[9]]=L6.sD;  MBs[mb_num].coeffs[7][inv_zigzag[10]]=L6.sE; MBs[mb_num].coeffs[7][inv_zigzag[11]]=L6.sF;
	MBs[mb_num].coeffs[7][inv_zigzag[12]]=L7.sC; MBs[mb_num].coeffs[7][inv_zigzag[13]]=L7.sD; MBs[mb_num].coeffs[7][inv_zigzag[14]]=L7.sE; MBs[mb_num].coeffs[7][inv_zigzag[15]]=L7.sF;
	//mb_row 2
	//block 20
	                                             MBs[mb_num].coeffs[8][inv_zigzag[1]]=L8.s1;  MBs[mb_num].coeffs[8][inv_zigzag[2]]=L8.s2;  MBs[mb_num].coeffs[8][inv_zigzag[3]]=L8.s3;
	MBs[mb_num].coeffs[8][inv_zigzag[4]]=L9.s0;	 MBs[mb_num].coeffs[8][inv_zigzag[5]]=L9.s1;  MBs[mb_num].coeffs[8][inv_zigzag[6]]=L9.s2;  MBs[mb_num].coeffs[8][inv_zigzag[7]]=L9.s3;
	MBs[mb_num].coeffs[8][inv_zigzag[8]]=L10.s0; MBs[mb_num].coeffs[8][inv_zigzag[9]]=L10.s1; MBs[mb_num].coeffs[8][inv_zigzag[10]]=L10.s2;MBs[mb_num].coeffs[8][inv_zigzag[11]]=L10.s3;
	MBs[mb_num].coeffs[8][inv_zigzag[12]]=L11.s0;MBs[mb_num].coeffs[8][inv_zigzag[13]]=L11.s1;MBs[mb_num].coeffs[8][inv_zigzag[14]]=L11.s2;MBs[mb_num].coeffs[8][inv_zigzag[15]]=L11.s3;
	//block 21
	                                             MBs[mb_num].coeffs[9][inv_zigzag[1]]=L8.s5;  MBs[mb_num].coeffs[9][inv_zigzag[2]]=L8.s6;  MBs[mb_num].coeffs[9][inv_zigzag[3]]=L8.s7;
	MBs[mb_num].coeffs[9][inv_zigzag[4]]=L9.s4;	 MBs[mb_num].coeffs[9][inv_zigzag[5]]=L9.s5;  MBs[mb_num].coeffs[9][inv_zigzag[6]]=L9.s6;  MBs[mb_num].coeffs[9][inv_zigzag[7]]=L9.s7;
	MBs[mb_num].coeffs[9][inv_zigzag[8]]=L10.s4; MBs[mb_num].coeffs[9][inv_zigzag[9]]=L10.s5; MBs[mb_num].coeffs[9][inv_zigzag[10]]=L10.s6;MBs[mb_num].coeffs[9][inv_zigzag[11]]=L10.s7;
	MBs[mb_num].coeffs[9][inv_zigzag[12]]=L11.s4;MBs[mb_num].coeffs[9][inv_zigzag[13]]=L11.s5;MBs[mb_num].coeffs[9][inv_zigzag[14]]=L11.s6;MBs[mb_num].coeffs[9][inv_zigzag[15]]=L11.s7;
	//block 22
	                                              MBs[mb_num].coeffs[10][inv_zigzag[1]]=L8.s9;  MBs[mb_num].coeffs[10][inv_zigzag[2]]=L8.sA;  MBs[mb_num].coeffs[10][inv_zigzag[3]]=L8.sB;
	MBs[mb_num].coeffs[10][inv_zigzag[4]]=L9.s8;  MBs[mb_num].coeffs[10][inv_zigzag[5]]=L9.s9;  MBs[mb_num].coeffs[10][inv_zigzag[6]]=L9.sA;  MBs[mb_num].coeffs[10][inv_zigzag[7]]=L9.sB;
	MBs[mb_num].coeffs[10][inv_zigzag[8]]=L10.s8; MBs[mb_num].coeffs[10][inv_zigzag[9]]=L10.s9; MBs[mb_num].coeffs[10][inv_zigzag[10]]=L10.sA;MBs[mb_num].coeffs[10][inv_zigzag[11]]=L10.sB;
	MBs[mb_num].coeffs[10][inv_zigzag[12]]=L11.s8;MBs[mb_num].coeffs[10][inv_zigzag[13]]=L11.s9;MBs[mb_num].coeffs[10][inv_zigzag[14]]=L11.sA;MBs[mb_num].coeffs[10][inv_zigzag[15]]=L11.sB;
	//block 23
	                                              MBs[mb_num].coeffs[11][inv_zigzag[1]]=L8.sD;  MBs[mb_num].coeffs[11][inv_zigzag[2]]=L8.sE;  MBs[mb_num].coeffs[11][inv_zigzag[3]]=L8.sF;
	MBs[mb_num].coeffs[11][inv_zigzag[4]]=L9.sC;  MBs[mb_num].coeffs[11][inv_zigzag[5]]=L9.sD;  MBs[mb_num].coeffs[11][inv_zigzag[6]]=L9.sE;  MBs[mb_num].coeffs[11][inv_zigzag[7]]=L9.sF;
	MBs[mb_num].coeffs[11][inv_zigzag[8]]=L10.sC; MBs[mb_num].coeffs[11][inv_zigzag[9]]=L10.sD; MBs[mb_num].coeffs[11][inv_zigzag[10]]=L10.sE;MBs[mb_num].coeffs[11][inv_zigzag[11]]=L10.sF;
	MBs[mb_num].coeffs[11][inv_zigzag[12]]=L11.sC;MBs[mb_num].coeffs[11][inv_zigzag[13]]=L11.sD;MBs[mb_num].coeffs[11][inv_zigzag[14]]=L11.sE;MBs[mb_num].coeffs[11][inv_zigzag[15]]=L11.sF;
	//mb_row 3
	//block 30
	                                              MBs[mb_num].coeffs[12][inv_zigzag[1]]=L12.s1; MBs[mb_num].coeffs[12][inv_zigzag[2]]=L12.s2; MBs[mb_num].coeffs[12][inv_zigzag[3]]=L12.s3;
	MBs[mb_num].coeffs[12][inv_zigzag[4]]=L13.s0; MBs[mb_num].coeffs[12][inv_zigzag[5]]=L13.s1; MBs[mb_num].coeffs[12][inv_zigzag[6]]=L13.s2; MBs[mb_num].coeffs[12][inv_zigzag[7]]=L13.s3;
	MBs[mb_num].coeffs[12][inv_zigzag[8]]=L14.s0; MBs[mb_num].coeffs[12][inv_zigzag[9]]=L14.s1; MBs[mb_num].coeffs[12][inv_zigzag[10]]=L14.s2;MBs[mb_num].coeffs[12][inv_zigzag[11]]=L14.s3;
	MBs[mb_num].coeffs[12][inv_zigzag[12]]=L15.s0;MBs[mb_num].coeffs[12][inv_zigzag[13]]=L15.s1;MBs[mb_num].coeffs[12][inv_zigzag[14]]=L15.s2;MBs[mb_num].coeffs[12][inv_zigzag[15]]=L15.s3;
	//block 31
	                                              MBs[mb_num].coeffs[13][inv_zigzag[1]]=L12.s5; MBs[mb_num].coeffs[13][inv_zigzag[2]]=L12.s6; MBs[mb_num].coeffs[13][inv_zigzag[3]]=L12.s7;
	MBs[mb_num].coeffs[13][inv_zigzag[4]]=L13.s4; MBs[mb_num].coeffs[13][inv_zigzag[5]]=L13.s5; MBs[mb_num].coeffs[13][inv_zigzag[6]]=L13.s6; MBs[mb_num].coeffs[13][inv_zigzag[7]]=L13.s7;
	MBs[mb_num].coeffs[13][inv_zigzag[8]]=L14.s4; MBs[mb_num].coeffs[13][inv_zigzag[9]]=L14.s5; MBs[mb_num].coeffs[13][inv_zigzag[10]]=L14.s6;MBs[mb_num].coeffs[13][inv_zigzag[11]]=L14.s7;
	MBs[mb_num].coeffs[13][inv_zigzag[12]]=L15.s4;MBs[mb_num].coeffs[13][inv_zigzag[13]]=L15.s5;MBs[mb_num].coeffs[13][inv_zigzag[14]]=L15.s6;MBs[mb_num].coeffs[13][inv_zigzag[15]]=L15.s7;
	//block 32
	                                              MBs[mb_num].coeffs[14][inv_zigzag[1]]=L12.s9; MBs[mb_num].coeffs[14][inv_zigzag[2]]=L12.sA; MBs[mb_num].coeffs[14][inv_zigzag[3]]=L12.sB;
	MBs[mb_num].coeffs[14][inv_zigzag[4]]=L13.s8; MBs[mb_num].coeffs[14][inv_zigzag[5]]=L13.s9; MBs[mb_num].coeffs[14][inv_zigzag[6]]=L13.sA; MBs[mb_num].coeffs[14][inv_zigzag[7]]=L13.sB;
	MBs[mb_num].coeffs[14][inv_zigzag[8]]=L14.s8; MBs[mb_num].coeffs[14][inv_zigzag[9]]=L14.s9; MBs[mb_num].coeffs[14][inv_zigzag[10]]=L14.sA;MBs[mb_num].coeffs[14][inv_zigzag[11]]=L14.sB;
	MBs[mb_num].coeffs[14][inv_zigzag[12]]=L15.s8;MBs[mb_num].coeffs[14][inv_zigzag[13]]=L15.s9;MBs[mb_num].coeffs[14][inv_zigzag[14]]=L15.sA;MBs[mb_num].coeffs[14][inv_zigzag[15]]=L15.sB;
	//block 33
	                                              MBs[mb_num].coeffs[15][inv_zigzag[1]]=L12.sD; MBs[mb_num].coeffs[15][inv_zigzag[2]]=L12.sE; MBs[mb_num].coeffs[15][inv_zigzag[3]]=L12.sF;
	MBs[mb_num].coeffs[15][inv_zigzag[4]]=L13.sC; MBs[mb_num].coeffs[15][inv_zigzag[5]]=L13.sD; MBs[mb_num].coeffs[15][inv_zigzag[6]]=L13.sE; MBs[mb_num].coeffs[15][inv_zigzag[7]]=L13.sF;
	MBs[mb_num].coeffs[15][inv_zigzag[8]]=L14.sC; MBs[mb_num].coeffs[15][inv_zigzag[9]]=L14.sD; MBs[mb_num].coeffs[15][inv_zigzag[10]]=L14.sE;MBs[mb_num].coeffs[15][inv_zigzag[11]]=L14.sF;
	MBs[mb_num].coeffs[15][inv_zigzag[12]]=L15.sC;MBs[mb_num].coeffs[15][inv_zigzag[13]]=L15.sD;MBs[mb_num].coeffs[15][inv_zigzag[14]]=L15.sE;MBs[mb_num].coeffs[15][inv_zigzag[15]]=L15.sF;
	
	//block Y2
	MBs[mb_num].coeffs[24][inv_zigzag[0]]=DL0.x;  MBs[mb_num].coeffs[24][inv_zigzag[1]]=DL0.y;  MBs[mb_num].coeffs[24][inv_zigzag[2]]=DL0.z;  MBs[mb_num].coeffs[24][inv_zigzag[3]]=DL0.w;
	MBs[mb_num].coeffs[24][inv_zigzag[4]]=DL1.x;  MBs[mb_num].coeffs[24][inv_zigzag[5]]=DL1.y;  MBs[mb_num].coeffs[24][inv_zigzag[6]]=DL1.z;  MBs[mb_num].coeffs[24][inv_zigzag[7]]=DL1.w;
	MBs[mb_num].coeffs[24][inv_zigzag[8]]=DL2.x;  MBs[mb_num].coeffs[24][inv_zigzag[9]]=DL2.y;  MBs[mb_num].coeffs[24][inv_zigzag[10]]=DL2.z; MBs[mb_num].coeffs[24][inv_zigzag[11]]=DL2.w;
	MBs[mb_num].coeffs[24][inv_zigzag[12]]=DL3.x; MBs[mb_num].coeffs[24][inv_zigzag[13]]=DL3.y; MBs[mb_num].coeffs[24][inv_zigzag[14]]=DL3.z; MBs[mb_num].coeffs[24][inv_zigzag[15]]=DL3.w;
	
	//and now inverse-WHT
    dequant_and_iWHT(&DL0, &DL1, &DL2, &DL3, y2_dc_q, y2_ac_q);
	L0.s0=DL0.x;  L0.s4=DL0.y; L0.s8=DL0.z; L0.sC=DL0.w;
	L4.s0=DL1.x;  L4.s4=DL1.y; L4.s8=DL1.z; L4.sC=DL1.w;
	L8.s0=DL2.x;  L8.s4=DL2.y; L8.s8=DL2.z; L8.sC=DL2.w;
	L12.s0=DL3.x; L12.s4=DL3.y;L12.s8=DL3.z;L12.sC=DL3.w;
	
	//iDCT and reconstruct
	i = y*width + x;
	//blocks 00-03
	DL0 = convert_int4(L0.s0123); DL1 = convert_int4(L1.s0123); DL2 = convert_int4(L2.s0123); DL3 = convert_int4(L3.s0123);
	dequant_and_iDCT(&DL0, &DL1, &DL2, &DL3, 1, y_ac_q);
	L0.s0123 = convert_short4(DL0);	L1.s0123 = convert_short4(DL1);	L2.s0123 = convert_short4(DL2);	L3.s0123 = convert_short4(DL3);
	DL0 = convert_int4(L0.s4567); DL1 = convert_int4(L1.s4567); DL2 = convert_int4(L2.s4567); DL3 = convert_int4(L3.s4567);
	dequant_and_iDCT(&DL0, &DL1, &DL2, &DL3, 1, y_ac_q);
	L0.s4567 = convert_short4(DL0);	L1.s4567 = convert_short4(DL1);	L2.s4567 = convert_short4(DL2);	L3.s4567 = convert_short4(DL3);
	DL0 = convert_int4(L0.s89AB); DL1 = convert_int4(L1.s89AB); DL2 = convert_int4(L2.s89AB); DL3 = convert_int4(L3.s89AB);
	dequant_and_iDCT(&DL0, &DL1, &DL2, &DL3, 1, y_ac_q);
	L0.s89AB = convert_short4(DL0);	L1.s89AB = convert_short4(DL1);	L2.s89AB = convert_short4(DL2);	L3.s89AB = convert_short4(DL3);
	DL0 = convert_int4(L0.sCDEF); DL1 = convert_int4(L1.sCDEF); DL2 = convert_int4(L2.sCDEF); DL3 = convert_int4(L3.sCDEF);
	dequant_and_iDCT(&DL0, &DL1, &DL2, &DL3, 1, y_ac_q);
	L0.sCDEF = convert_short4(DL0);	L1.sCDEF = convert_short4(DL1);	L2.sCDEF = convert_short4(DL2);	L3.sCDEF = convert_short4(DL3);
	L0 += convert_short16(vload16(0, golden_frame_Y+i)); i+=width;
	L1 += convert_short16(vload16(0, golden_frame_Y+i)); i+=width;
	L2 += convert_short16(vload16(0, golden_frame_Y+i)); i+=width;
	L3 += convert_short16(vload16(0, golden_frame_Y+i)); i+=width;
	//blocks 10-13
	DL0 = convert_int4(L4.s0123); DL1 = convert_int4(L5.s0123); DL2 = convert_int4(L6.s0123); DL3 = convert_int4(L7.s0123);
	dequant_and_iDCT(&DL0, &DL1, &DL2, &DL3, 1, y_ac_q);
	L4.s0123 = convert_short4(DL0);	L5.s0123 = convert_short4(DL1);	L6.s0123 = convert_short4(DL2);	L7.s0123 = convert_short4(DL3);
	DL0 = convert_int4(L4.s4567); DL1 = convert_int4(L5.s4567); DL2 = convert_int4(L6.s4567); DL3 = convert_int4(L7.s4567);
	dequant_and_iDCT(&DL0, &DL1, &DL2, &DL3, 1, y_ac_q);
	L4.s4567 = convert_short4(DL0);	L5.s4567 = convert_short4(DL1);	L6.s4567 = convert_short4(DL2);	L7.s4567 = convert_short4(DL3);
	DL0 = convert_int4(L4.s89AB); DL1 = convert_int4(L5.s89AB); DL2 = convert_int4(L6.s89AB); DL3 = convert_int4(L7.s89AB);
	dequant_and_iDCT(&DL0, &DL1, &DL2, &DL3, 1, y_ac_q);
	L4.s89AB = convert_short4(DL0);	L5.s89AB = convert_short4(DL1);	L6.s89AB = convert_short4(DL2);	L7.s89AB = convert_short4(DL3);
	DL0 = convert_int4(L4.sCDEF); DL1 = convert_int4(L5.sCDEF); DL2 = convert_int4(L6.sCDEF); DL3 = convert_int4(L7.sCDEF);
	dequant_and_iDCT(&DL0, &DL1, &DL2, &DL3, 1, y_ac_q);
	L4.sCDEF = convert_short4(DL0);	L5.sCDEF = convert_short4(DL1);	L6.sCDEF = convert_short4(DL2);	L7.sCDEF = convert_short4(DL3);
	L4 += convert_short16(vload16(0, golden_frame_Y+i)); i+=width;
	L5 += convert_short16(vload16(0, golden_frame_Y+i)); i+=width;
	L6 += convert_short16(vload16(0, golden_frame_Y+i)); i+=width;
	L7 += convert_short16(vload16(0, golden_frame_Y+i)); i+=width;
	//blocks 20-23
	DL0 = convert_int4(L8.s0123); DL1 = convert_int4(L9.s0123); DL2 = convert_int4(L10.s0123); DL3 = convert_int4(L11.s0123);
	dequant_and_iDCT(&DL0, &DL1, &DL2, &DL3, 1, y_ac_q);
	L8.s0123 = convert_short4(DL0);	L9.s0123 = convert_short4(DL1);	L10.s0123 = convert_short4(DL2);	L11.s0123 = convert_short4(DL3);
	DL0 = convert_int4(L8.s4567); DL1 = convert_int4(L9.s4567); DL2 = convert_int4(L10.s4567); DL3 = convert_int4(L11.s4567);
	dequant_and_iDCT(&DL0, &DL1, &DL2, &DL3, 1, y_ac_q);
	L8.s4567 = convert_short4(DL0);	L9.s4567 = convert_short4(DL1);	L10.s4567 = convert_short4(DL2);	L11.s4567 = convert_short4(DL3);
	DL0 = convert_int4(L8.s89AB); DL1 = convert_int4(L9.s89AB); DL2 = convert_int4(L10.s89AB); DL3 = convert_int4(L11.s89AB);
	dequant_and_iDCT(&DL0, &DL1, &DL2, &DL3, 1, y_ac_q);
	L8.s89AB = convert_short4(DL0);	L9.s89AB = convert_short4(DL1);	L10.s89AB = convert_short4(DL2);	L11.s89AB = convert_short4(DL3);
	DL0 = convert_int4(L8.sCDEF); DL1 = convert_int4(L9.sCDEF); DL2 = convert_int4(L10.sCDEF); DL3 = convert_int4(L11.sCDEF);
	dequant_and_iDCT(&DL0, &DL1, &DL2, &DL3, 1, y_ac_q);
	L8.sCDEF = convert_short4(DL0);	L9.sCDEF = convert_short4(DL1);	L10.sCDEF = convert_short4(DL2);	L11.sCDEF = convert_short4(DL3);
	L8 += convert_short16(vload16(0, golden_frame_Y+i)); i+=width;
	L9 += convert_short16(vload16(0, golden_frame_Y+i)); i+=width;
	L10 += convert_short16(vload16(0, golden_frame_Y+i)); i+=width;
	L11 += convert_short16(vload16(0, golden_frame_Y+i)); i+=width;
	//blocks 30-33
	DL0 = convert_int4(L12.s0123); DL1 = convert_int4(L13.s0123); DL2 = convert_int4(L14.s0123); DL3 = convert_int4(L15.s0123);
	dequant_and_iDCT(&DL0, &DL1, &DL2, &DL3, 1, y_ac_q);
	L12.s0123 = convert_short4(DL0);	L13.s0123 = convert_short4(DL1);	L14.s0123 = convert_short4(DL2);	L15.s0123 = convert_short4(DL3);
	DL0 = convert_int4(L12.s4567); DL1 = convert_int4(L13.s4567); DL2 = convert_int4(L14.s4567); DL3 = convert_int4(L15.s4567);
	dequant_and_iDCT(&DL0, &DL1, &DL2, &DL3, 1, y_ac_q);
	L12.s4567 = convert_short4(DL0);	L13.s4567 = convert_short4(DL1);	L14.s4567 = convert_short4(DL2);	L15.s4567 = convert_short4(DL3);
	DL0 = convert_int4(L12.s89AB); DL1 = convert_int4(L13.s89AB); DL2 = convert_int4(L14.s89AB); DL3 = convert_int4(L15.s89AB);
	dequant_and_iDCT(&DL0, &DL1, &DL2, &DL3, 1, y_ac_q);
	L12.s89AB = convert_short4(DL0);	L13.s89AB = convert_short4(DL1);	L14.s89AB = convert_short4(DL2);	L15.s89AB = convert_short4(DL3);
	DL0 = convert_int4(L12.sCDEF); DL1 = convert_int4(L13.sCDEF); DL2 = convert_int4(L14.sCDEF); DL3 = convert_int4(L15.sCDEF);
	dequant_and_iDCT(&DL0, &DL1, &DL2, &DL3, 1, y_ac_q);
	L12.sCDEF = convert_short4(DL0);	L13.sCDEF = convert_short4(DL1);	L14.sCDEF = convert_short4(DL2);	L15.sCDEF = convert_short4(DL3);
	L12 += convert_short16(vload16(0, golden_frame_Y+i)); i+=width;
	L13 += convert_short16(vload16(0, golden_frame_Y+i)); i+=width;
	L14 += convert_short16(vload16(0, golden_frame_Y+i)); i+=width;
	L15 += convert_short16(vload16(0, golden_frame_Y+i));
	
	//now place results in reconstruction frame
	i = y*width + x;
	vstore16(convert_uchar16_sat(L0), 0, recon_frame_Y+i); i+= width;
	vstore16(convert_uchar16_sat(L1), 0, recon_frame_Y+i); i+= width;
	vstore16(convert_uchar16_sat(L2), 0, recon_frame_Y+i); i+= width;
	vstore16(convert_uchar16_sat(L3), 0, recon_frame_Y+i); i+= width;
	vstore16(convert_uchar16_sat(L4), 0, recon_frame_Y+i); i+= width;
	vstore16(convert_uchar16_sat(L5), 0, recon_frame_Y+i); i+= width;
	vstore16(convert_uchar16_sat(L6), 0, recon_frame_Y+i); i+= width;
	vstore16(convert_uchar16_sat(L7), 0, recon_frame_Y+i); i+= width;
	vstore16(convert_uchar16_sat(L8), 0, recon_frame_Y+i); i+= width;
	vstore16(convert_uchar16_sat(L9), 0, recon_frame_Y+i); i+= width;
	vstore16(convert_uchar16_sat(L10), 0, recon_frame_Y+i); i+= width;
	vstore16(convert_uchar16_sat(L11), 0, recon_frame_Y+i); i+= width;
	vstore16(convert_uchar16_sat(L12), 0, recon_frame_Y+i); i+= width;
	vstore16(convert_uchar16_sat(L13), 0, recon_frame_Y+i); i+= width;
	vstore16(convert_uchar16_sat(L14), 0, recon_frame_Y+i); i+= width;
	vstore16(convert_uchar16_sat(L15), 0, recon_frame_Y+i);	

	//and now the same for chromas, but without weighting
	x /= 2; y /= 2;
	int chroma_width = width/2;
	// U-plane current
	i = y*chroma_width + x;
	L0.s01234567 = convert_short8(vload8(0, current_frame_U+i)); i+= chroma_width;
	L1.s01234567 = convert_short8(vload8(0, current_frame_U+i)); i+= chroma_width;
	L2.s01234567 = convert_short8(vload8(0, current_frame_U+i)); i+= chroma_width;
	L3.s01234567 = convert_short8(vload8(0, current_frame_U+i)); i+= chroma_width;
	L4.s01234567 = convert_short8(vload8(0, current_frame_U+i)); i+= chroma_width;
	L5.s01234567 = convert_short8(vload8(0, current_frame_U+i)); i+= chroma_width;
	L6.s01234567 = convert_short8(vload8(0, current_frame_U+i)); i+= chroma_width;
	L7.s01234567 = convert_short8(vload8(0, current_frame_U+i));
	// V-plane current
	i = y*chroma_width + x;
	L0.s89ABCDEF = convert_short8(vload8(0, current_frame_V+i)); i+= chroma_width;
	L1.s89ABCDEF = convert_short8(vload8(0, current_frame_V+i)); i+= chroma_width;
	L2.s89ABCDEF = convert_short8(vload8(0, current_frame_V+i)); i+= chroma_width;
	L3.s89ABCDEF = convert_short8(vload8(0, current_frame_V+i)); i+= chroma_width;
	L4.s89ABCDEF = convert_short8(vload8(0, current_frame_V+i)); i+= chroma_width;
	L5.s89ABCDEF = convert_short8(vload8(0, current_frame_V+i)); i+= chroma_width;
	L6.s89ABCDEF = convert_short8(vload8(0, current_frame_V+i)); i+= chroma_width;
	L7.s89ABCDEF = convert_short8(vload8(0, current_frame_V+i));
	// U-plane golden
	i = y*chroma_width + x;
	L8.s01234567 = convert_short8(vload8(0, golden_frame_U+i)); i+= chroma_width;
	L9.s01234567 = convert_short8(vload8(0, golden_frame_U+i)); i+= chroma_width;
	L10.s01234567 = convert_short8(vload8(0, golden_frame_U+i)); i+= chroma_width;
	L11.s01234567 = convert_short8(vload8(0, golden_frame_U+i)); i+= chroma_width;
	L12.s01234567 = convert_short8(vload8(0, golden_frame_U+i)); i+= chroma_width;
	L13.s01234567 = convert_short8(vload8(0, golden_frame_U+i)); i+= chroma_width;
	L14.s01234567 = convert_short8(vload8(0, golden_frame_U+i)); i+= chroma_width;
	L15.s01234567 = convert_short8(vload8(0, golden_frame_U+i));
	// V-plane golden
	i = y*chroma_width + x;
	L8.s89ABCDEF = convert_short8(vload8(0, golden_frame_V+i)); i+= chroma_width;
	L9.s89ABCDEF = convert_short8(vload8(0, golden_frame_V+i)); i+= chroma_width;
	L10.s89ABCDEF = convert_short8(vload8(0, golden_frame_V+i)); i+= chroma_width;
	L11.s89ABCDEF = convert_short8(vload8(0, golden_frame_V+i)); i+= chroma_width;
	L12.s89ABCDEF = convert_short8(vload8(0, golden_frame_V+i)); i+= chroma_width;
	L13.s89ABCDEF = convert_short8(vload8(0, golden_frame_V+i)); i+= chroma_width;
	L14.s89ABCDEF = convert_short8(vload8(0, golden_frame_V+i)); i+= chroma_width;
	L15.s89ABCDEF = convert_short8(vload8(0, golden_frame_V+i));
	
	//residual
	L0 -= L8; L1 -= L9; L2 -= L10; L3 -= L11; L4-= L12; L5 -= L13; L6 -= L14; L7 -= L15;
	//transform U
	DL0 = convert_int4(L0.s0123); DL1 = convert_int4(L1.s0123); DL2 = convert_int4(L2.s0123); DL3 = convert_int4(L3.s0123);
	DCT_and_quant(&DL0, &DL1, &DL2, &DL3, uv_dc_q, uv_ac_q);
	L0.s0123 = convert_short4(DL0);	L1.s0123 = convert_short4(DL1);	L2.s0123 = convert_short4(DL2);	L3.s0123 = convert_short4(DL3);
	DL0 = convert_int4(L0.s4567); DL1 = convert_int4(L1.s4567); DL2 = convert_int4(L2.s4567); DL3 = convert_int4(L3.s4567);
	DCT_and_quant(&DL0, &DL1, &DL2, &DL3, uv_dc_q, uv_ac_q);
	L0.s4567 = convert_short4(DL0);	L1.s4567 = convert_short4(DL1);	L2.s4567 = convert_short4(DL2);	L3.s4567 = convert_short4(DL3);
	DL0 = convert_int4(L4.s0123); DL1 = convert_int4(L5.s0123); DL2 = convert_int4(L6.s0123); DL3 = convert_int4(L7.s0123);
	DCT_and_quant(&DL0, &DL1, &DL2, &DL3, uv_dc_q, uv_ac_q);
	L4.s0123 = convert_short4(DL0);	L5.s0123 = convert_short4(DL1);	L6.s0123 = convert_short4(DL2);	L7.s0123 = convert_short4(DL3);
	DL0 = convert_int4(L4.s4567); DL1 = convert_int4(L5.s4567); DL2 = convert_int4(L6.s4567); DL3 = convert_int4(L7.s4567);
	DCT_and_quant(&DL0, &DL1, &DL2, &DL3, uv_dc_q, uv_ac_q);
	L4.s4567 = convert_short4(DL0);	L5.s4567 = convert_short4(DL1);	L6.s4567 = convert_short4(DL2);	L7.s4567 = convert_short4(DL3);
	//transform V
	DL0 = convert_int4(L0.s89AB); DL1 = convert_int4(L1.s89AB); DL2 = convert_int4(L2.s89AB); DL3 = convert_int4(L3.s89AB);
	DCT_and_quant(&DL0, &DL1, &DL2, &DL3, uv_dc_q, uv_ac_q);
	L0.s89AB = convert_short4(DL0);	L1.s89AB = convert_short4(DL1);	L2.s89AB = convert_short4(DL2);	L3.s89AB = convert_short4(DL3);
	DL0 = convert_int4(L0.sCDEF); DL1 = convert_int4(L1.sCDEF); DL2 = convert_int4(L2.sCDEF); DL3 = convert_int4(L3.sCDEF);
	DCT_and_quant(&DL0, &DL1, &DL2, &DL3, uv_dc_q, uv_ac_q);
	L0.sCDEF = convert_short4(DL0);	L1.sCDEF = convert_short4(DL1);	L2.sCDEF = convert_short4(DL2);	L3.sCDEF = convert_short4(DL3);
	DL0 = convert_int4(L4.s89AB); DL1 = convert_int4(L5.s89AB); DL2 = convert_int4(L6.s89AB); DL3 = convert_int4(L7.s89AB);
	DCT_and_quant(&DL0, &DL1, &DL2, &DL3, uv_dc_q, uv_ac_q);
	L4.s89AB = convert_short4(DL0);	L5.s89AB = convert_short4(DL1);	L6.s89AB = convert_short4(DL2);	L7.s89AB = convert_short4(DL3);
	DL0 = convert_int4(L4.sCDEF); DL1 = convert_int4(L5.sCDEF); DL2 = convert_int4(L6.sCDEF); DL3 = convert_int4(L7.sCDEF);
	DCT_and_quant(&DL0, &DL1, &DL2, &DL3, uv_dc_q, uv_ac_q);
	L4.sCDEF = convert_short4(DL0);	L5.sCDEF = convert_short4(DL1);	L6.sCDEF = convert_short4(DL2);	L7.sCDEF = convert_short4(DL3);
	
	//write that to transformed data
	//block U00
	MBs[mb_num].coeffs[16][inv_zigzag[0]]=L0.s0;  MBs[mb_num].coeffs[16][inv_zigzag[1]]=L0.s1;  MBs[mb_num].coeffs[16][inv_zigzag[2]]=L0.s2;  MBs[mb_num].coeffs[16][inv_zigzag[3]]=L0.s3;
	MBs[mb_num].coeffs[16][inv_zigzag[4]]=L1.s0;  MBs[mb_num].coeffs[16][inv_zigzag[5]]=L1.s1;  MBs[mb_num].coeffs[16][inv_zigzag[6]]=L1.s2;  MBs[mb_num].coeffs[16][inv_zigzag[7]]=L1.s3;
	MBs[mb_num].coeffs[16][inv_zigzag[8]]=L2.s0;  MBs[mb_num].coeffs[16][inv_zigzag[9]]=L2.s1;  MBs[mb_num].coeffs[16][inv_zigzag[10]]=L2.s2; MBs[mb_num].coeffs[16][inv_zigzag[11]]=L2.s3;
	MBs[mb_num].coeffs[16][inv_zigzag[12]]=L3.s0; MBs[mb_num].coeffs[16][inv_zigzag[13]]=L3.s1; MBs[mb_num].coeffs[16][inv_zigzag[14]]=L3.s2; MBs[mb_num].coeffs[16][inv_zigzag[15]]=L3.s3;
	//block U01
	MBs[mb_num].coeffs[17][inv_zigzag[0]]=L0.s4;  MBs[mb_num].coeffs[17][inv_zigzag[1]]=L0.s5;  MBs[mb_num].coeffs[17][inv_zigzag[2]]=L0.s6;  MBs[mb_num].coeffs[17][inv_zigzag[3]]=L0.s7;
	MBs[mb_num].coeffs[17][inv_zigzag[4]]=L1.s4;  MBs[mb_num].coeffs[17][inv_zigzag[5]]=L1.s5;  MBs[mb_num].coeffs[17][inv_zigzag[6]]=L1.s6;  MBs[mb_num].coeffs[17][inv_zigzag[7]]=L1.s7;
	MBs[mb_num].coeffs[17][inv_zigzag[8]]=L2.s4;  MBs[mb_num].coeffs[17][inv_zigzag[9]]=L2.s5;  MBs[mb_num].coeffs[17][inv_zigzag[10]]=L2.s6; MBs[mb_num].coeffs[17][inv_zigzag[11]]=L2.s7;
	MBs[mb_num].coeffs[17][inv_zigzag[12]]=L3.s4; MBs[mb_num].coeffs[17][inv_zigzag[13]]=L3.s5; MBs[mb_num].coeffs[17][inv_zigzag[14]]=L3.s6; MBs[mb_num].coeffs[17][inv_zigzag[15]]=L3.s7;
	//block U10
	MBs[mb_num].coeffs[18][inv_zigzag[0]]=L4.s0;  MBs[mb_num].coeffs[18][inv_zigzag[1]]=L4.s1;  MBs[mb_num].coeffs[18][inv_zigzag[2]]=L4.s2;  MBs[mb_num].coeffs[18][inv_zigzag[3]]=L4.s3;
	MBs[mb_num].coeffs[18][inv_zigzag[4]]=L5.s0;  MBs[mb_num].coeffs[18][inv_zigzag[5]]=L5.s1;  MBs[mb_num].coeffs[18][inv_zigzag[6]]=L5.s2;  MBs[mb_num].coeffs[18][inv_zigzag[7]]=L5.s3;
	MBs[mb_num].coeffs[18][inv_zigzag[8]]=L6.s0;  MBs[mb_num].coeffs[18][inv_zigzag[9]]=L6.s1;  MBs[mb_num].coeffs[18][inv_zigzag[10]]=L6.s2; MBs[mb_num].coeffs[18][inv_zigzag[11]]=L6.s3;
	MBs[mb_num].coeffs[18][inv_zigzag[12]]=L7.s0; MBs[mb_num].coeffs[18][inv_zigzag[13]]=L7.s1; MBs[mb_num].coeffs[18][inv_zigzag[14]]=L7.s2; MBs[mb_num].coeffs[18][inv_zigzag[15]]=L7.s3;
	//block U11
	MBs[mb_num].coeffs[19][inv_zigzag[0]]=L4.s4;  MBs[mb_num].coeffs[19][inv_zigzag[1]]=L4.s5;  MBs[mb_num].coeffs[19][inv_zigzag[2]]=L4.s6;  MBs[mb_num].coeffs[19][inv_zigzag[3]]=L4.s7;
	MBs[mb_num].coeffs[19][inv_zigzag[4]]=L5.s4;  MBs[mb_num].coeffs[19][inv_zigzag[5]]=L5.s5;  MBs[mb_num].coeffs[19][inv_zigzag[6]]=L5.s6;  MBs[mb_num].coeffs[19][inv_zigzag[7]]=L5.s7;
	MBs[mb_num].coeffs[19][inv_zigzag[8]]=L6.s4;  MBs[mb_num].coeffs[19][inv_zigzag[9]]=L6.s5;  MBs[mb_num].coeffs[19][inv_zigzag[10]]=L6.s6; MBs[mb_num].coeffs[19][inv_zigzag[11]]=L6.s7;
	MBs[mb_num].coeffs[19][inv_zigzag[12]]=L7.s4; MBs[mb_num].coeffs[19][inv_zigzag[13]]=L7.s5; MBs[mb_num].coeffs[19][inv_zigzag[14]]=L7.s6; MBs[mb_num].coeffs[19][inv_zigzag[15]]=L7.s7;
	
	//block V00
	MBs[mb_num].coeffs[20][inv_zigzag[0]]=L0.s8;  MBs[mb_num].coeffs[20][inv_zigzag[1]]=L0.s9;  MBs[mb_num].coeffs[20][inv_zigzag[2]]=L0.sA;  MBs[mb_num].coeffs[20][inv_zigzag[3]]=L0.sB;
	MBs[mb_num].coeffs[20][inv_zigzag[4]]=L1.s8;  MBs[mb_num].coeffs[20][inv_zigzag[5]]=L1.s9;  MBs[mb_num].coeffs[20][inv_zigzag[6]]=L1.sA;  MBs[mb_num].coeffs[20][inv_zigzag[7]]=L1.sB;
	MBs[mb_num].coeffs[20][inv_zigzag[8]]=L2.s8;  MBs[mb_num].coeffs[20][inv_zigzag[9]]=L2.s9;  MBs[mb_num].coeffs[20][inv_zigzag[10]]=L2.sA; MBs[mb_num].coeffs[20][inv_zigzag[11]]=L2.sB;
	MBs[mb_num].coeffs[20][inv_zigzag[12]]=L3.s8; MBs[mb_num].coeffs[20][inv_zigzag[13]]=L3.s9; MBs[mb_num].coeffs[20][inv_zigzag[14]]=L3.sA; MBs[mb_num].coeffs[20][inv_zigzag[15]]=L3.sB;
	//block V01
	MBs[mb_num].coeffs[21][inv_zigzag[0]]=L0.sC;  MBs[mb_num].coeffs[21][inv_zigzag[1]]=L0.sD;  MBs[mb_num].coeffs[21][inv_zigzag[2]]=L0.sE;  MBs[mb_num].coeffs[21][inv_zigzag[3]]=L0.sF;
	MBs[mb_num].coeffs[21][inv_zigzag[4]]=L1.sC;  MBs[mb_num].coeffs[21][inv_zigzag[5]]=L1.sD;  MBs[mb_num].coeffs[21][inv_zigzag[6]]=L1.sE;  MBs[mb_num].coeffs[21][inv_zigzag[7]]=L1.sF;
	MBs[mb_num].coeffs[21][inv_zigzag[8]]=L2.sC;  MBs[mb_num].coeffs[21][inv_zigzag[9]]=L2.sD;  MBs[mb_num].coeffs[21][inv_zigzag[10]]=L2.sE; MBs[mb_num].coeffs[21][inv_zigzag[11]]=L2.sF;
	MBs[mb_num].coeffs[21][inv_zigzag[12]]=L3.sC; MBs[mb_num].coeffs[21][inv_zigzag[13]]=L3.sD; MBs[mb_num].coeffs[21][inv_zigzag[14]]=L3.sE; MBs[mb_num].coeffs[21][inv_zigzag[15]]=L3.sF;
	//block V10
	MBs[mb_num].coeffs[22][inv_zigzag[0]]=L4.s8;  MBs[mb_num].coeffs[22][inv_zigzag[1]]=L4.s9;  MBs[mb_num].coeffs[22][inv_zigzag[2]]=L4.sA;  MBs[mb_num].coeffs[22][inv_zigzag[3]]=L4.sB;
	MBs[mb_num].coeffs[22][inv_zigzag[4]]=L5.s8;  MBs[mb_num].coeffs[22][inv_zigzag[5]]=L5.s9;  MBs[mb_num].coeffs[22][inv_zigzag[6]]=L5.sA;  MBs[mb_num].coeffs[22][inv_zigzag[7]]=L5.sB;
	MBs[mb_num].coeffs[22][inv_zigzag[8]]=L6.s8;  MBs[mb_num].coeffs[22][inv_zigzag[9]]=L6.s9;  MBs[mb_num].coeffs[22][inv_zigzag[10]]=L6.sA; MBs[mb_num].coeffs[22][inv_zigzag[11]]=L6.sB;
	MBs[mb_num].coeffs[22][inv_zigzag[12]]=L7.s8; MBs[mb_num].coeffs[22][inv_zigzag[13]]=L7.s9; MBs[mb_num].coeffs[22][inv_zigzag[14]]=L7.sA; MBs[mb_num].coeffs[22][inv_zigzag[15]]=L7.sB;
	//block V11
	MBs[mb_num].coeffs[23][inv_zigzag[0]]=L4.sC;  MBs[mb_num].coeffs[23][inv_zigzag[1]]=L4.sD;  MBs[mb_num].coeffs[23][inv_zigzag[2]]=L4.sE;  MBs[mb_num].coeffs[23][inv_zigzag[3]]=L4.sF;
	MBs[mb_num].coeffs[23][inv_zigzag[4]]=L5.sC;  MBs[mb_num].coeffs[23][inv_zigzag[5]]=L5.sD;  MBs[mb_num].coeffs[23][inv_zigzag[6]]=L5.sE;  MBs[mb_num].coeffs[23][inv_zigzag[7]]=L5.sF;
	MBs[mb_num].coeffs[23][inv_zigzag[8]]=L6.sC;  MBs[mb_num].coeffs[23][inv_zigzag[9]]=L6.sD;  MBs[mb_num].coeffs[23][inv_zigzag[10]]=L6.sE; MBs[mb_num].coeffs[23][inv_zigzag[11]]=L6.sF;
	MBs[mb_num].coeffs[23][inv_zigzag[12]]=L7.sC; MBs[mb_num].coeffs[23][inv_zigzag[13]]=L7.sD; MBs[mb_num].coeffs[23][inv_zigzag[14]]=L7.sE; MBs[mb_num].coeffs[23][inv_zigzag[15]]=L7.sF;
	
	// now reconstruction process
	// iDCT
	DL0 = convert_int4(L0.s0123); DL1 = convert_int4(L1.s0123); DL2 = convert_int4(L2.s0123); DL3 = convert_int4(L3.s0123);
	dequant_and_iDCT(&DL0, &DL1, &DL2, &DL3, uv_dc_q, uv_ac_q);
	L0.s0123 = convert_short4(DL0);	L1.s0123 = convert_short4(DL1);	L2.s0123 = convert_short4(DL2);	L3.s0123 = convert_short4(DL3);
	DL0 = convert_int4(L0.s4567); DL1 = convert_int4(L1.s4567); DL2 = convert_int4(L2.s4567); DL3 = convert_int4(L3.s4567);
	dequant_and_iDCT(&DL0, &DL1, &DL2, &DL3, uv_dc_q, uv_ac_q);
	L0.s4567 = convert_short4(DL0);	L1.s4567 = convert_short4(DL1);	L2.s4567 = convert_short4(DL2);	L3.s4567 = convert_short4(DL3);
	DL0 = convert_int4(L4.s0123); DL1 = convert_int4(L5.s0123); DL2 = convert_int4(L6.s0123); DL3 = convert_int4(L7.s0123);
	dequant_and_iDCT(&DL0, &DL1, &DL2, &DL3, uv_dc_q, uv_ac_q);
	L4.s0123 = convert_short4(DL0);	L5.s0123 = convert_short4(DL1);	L6.s0123 = convert_short4(DL2);	L7.s0123 = convert_short4(DL3);
	DL0 = convert_int4(L4.s4567); DL1 = convert_int4(L5.s4567); DL2 = convert_int4(L6.s4567); DL3 = convert_int4(L7.s4567);
	dequant_and_iDCT(&DL0, &DL1, &DL2, &DL3, uv_dc_q, uv_ac_q);
	L4.s4567 = convert_short4(DL0);	L5.s4567 = convert_short4(DL1);	L6.s4567 = convert_short4(DL2);	L7.s4567 = convert_short4(DL3);	
	//transform V
	DL0 = convert_int4(L0.s89AB); DL1 = convert_int4(L1.s89AB); DL2 = convert_int4(L2.s89AB); DL3 = convert_int4(L3.s89AB);
	dequant_and_iDCT(&DL0, &DL1, &DL2, &DL3, uv_dc_q, uv_ac_q);
	L0.s89AB = convert_short4(DL0);	L1.s89AB = convert_short4(DL1);	L2.s89AB = convert_short4(DL2);	L3.s89AB = convert_short4(DL3);
	DL0 = convert_int4(L0.sCDEF); DL1 = convert_int4(L1.sCDEF); DL2 = convert_int4(L2.sCDEF); DL3 = convert_int4(L3.sCDEF);
	dequant_and_iDCT(&DL0, &DL1, &DL2, &DL3, uv_dc_q, uv_ac_q);
	L0.sCDEF = convert_short4(DL0);	L1.sCDEF = convert_short4(DL1);	L2.sCDEF = convert_short4(DL2);	L3.sCDEF = convert_short4(DL3);
	DL0 = convert_int4(L4.s89AB); DL1 = convert_int4(L5.s89AB); DL2 = convert_int4(L6.s89AB); DL3 = convert_int4(L7.s89AB);
	dequant_and_iDCT(&DL0, &DL1, &DL2, &DL3, uv_dc_q, uv_ac_q);
	L4.s89AB = convert_short4(DL0);	L5.s89AB = convert_short4(DL1);	L6.s89AB = convert_short4(DL2);	L7.s89AB = convert_short4(DL3);
	DL0 = convert_int4(L4.sCDEF); DL1 = convert_int4(L5.sCDEF); DL2 = convert_int4(L6.sCDEF); DL3 = convert_int4(L7.sCDEF);
	dequant_and_iDCT(&DL0, &DL1, &DL2, &DL3, uv_dc_q, uv_ac_q);
	L4.sCDEF = convert_short4(DL0);	L5.sCDEF = convert_short4(DL1);	L6.sCDEF = convert_short4(DL2);	L7.sCDEF = convert_short4(DL3);
	
	//we still have predictor in L8..L15 lines
	L0 += L8; L1 += L9; L2 += L10; L3 += L11; L4 += L12; L5 += L13; L6 += L14; L7 += L15;

	//and store this into reconstructed frame
	// U-plane
	i = y*chroma_width + x;
	vstore8(convert_uchar8_sat(L0.s01234567), 0, recon_frame_U+i); i+= chroma_width;
	vstore8(convert_uchar8_sat(L1.s01234567), 0, recon_frame_U+i); i+= chroma_width;
	vstore8(convert_uchar8_sat(L2.s01234567), 0, recon_frame_U+i); i+= chroma_width;
	vstore8(convert_uchar8_sat(L3.s01234567), 0, recon_frame_U+i); i+= chroma_width;
	vstore8(convert_uchar8_sat(L4.s01234567), 0, recon_frame_U+i); i+= chroma_width;
	vstore8(convert_uchar8_sat(L5.s01234567), 0, recon_frame_U+i); i+= chroma_width;
	vstore8(convert_uchar8_sat(L6.s01234567), 0, recon_frame_U+i); i+= chroma_width;
	vstore8(convert_uchar8_sat(L7.s01234567), 0, recon_frame_U+i);
	// V-plane 
	i = y*chroma_width + x;
	vstore8(convert_uchar8_sat(L0.s89ABCDEF), 0, recon_frame_V+i); i+= chroma_width;
	vstore8(convert_uchar8_sat(L1.s89ABCDEF), 0, recon_frame_V+i); i+= chroma_width;
	vstore8(convert_uchar8_sat(L2.s89ABCDEF), 0, recon_frame_V+i); i+= chroma_width;
	vstore8(convert_uchar8_sat(L3.s89ABCDEF), 0, recon_frame_V+i); i+= chroma_width;
	vstore8(convert_uchar8_sat(L4.s89ABCDEF), 0, recon_frame_V+i); i+= chroma_width;
	vstore8(convert_uchar8_sat(L5.s89ABCDEF), 0, recon_frame_V+i); i+= chroma_width;
	vstore8(convert_uchar8_sat(L6.s89ABCDEF), 0, recon_frame_V+i); i+= chroma_width;
	vstore8(convert_uchar8_sat(L7.s89ABCDEF), 0, recon_frame_V+i);
	
	return;
}

	
__kernel void luma_transform_16x16(__global uchar *const current_frame, //0
								__global uchar *const recon_frame, //1
								__global uchar *const prev_frame, //2
								__global macroblock *const MBs, //3
								const int width, //4
								__constant segment_data *const SD) //5
{	
	// possible optimization - LOCAL memory for storing predictors until reconstruction
	// but it's very device specific (HD6000-HD7000 has 32kb per work_group, older may has less)

	__private int mb_num, vector_x,vector_y,x,y,condition,i,y_ac_q,y2_dc_q,y2_ac_q;
	__private int4 DL0,DL1,DL2,DL3;
	__private short16 L0,L1,L2,L3,L4,L5,L6,L7,L8,L9,L10,L11,L12,L13,L14,L15;
	__private uchar4 buf;

	mb_num = get_global_id(0);

	if (MBs[mb_num].reference_frame != LAST) return;
	MBs[mb_num].parts = are8x8; 

	vector_x = MBs[mb_num].vector_x[0];
	vector_y = MBs[mb_num].vector_y[0];
	x = MBs[mb_num].vector_x[1];
	y = MBs[mb_num].vector_y[1];
	condition = ((vector_x != x) || (vector_y != y)); //XOr optimization possible
	if (condition) return;
	x = MBs[mb_num].vector_x[2];
	y = MBs[mb_num].vector_y[2];
	condition = ((vector_x != x) || (vector_y != y)); 
	if (condition) return;
	x = MBs[mb_num].vector_x[3];
	y = MBs[mb_num].vector_y[3];
	condition = ((vector_x != x) || (vector_y != y)); 
	if (condition) return; 
	MBs[mb_num].parts = are16x16;
	
	i = MBs[mb_num].segment_id;
	i = SD[i].y_ac_i;
	
	y2_dc_q = SD[0].y2_dc_idelta; y2_dc_q += i;
	y2_ac_q = SD[0].y2_ac_idelta; y2_ac_q += i;
	y2_dc_q = select(y2_dc_q,0,y2_dc_q<0); y2_dc_q = select(y2_dc_q,127,y2_dc_q>127);
	y2_ac_q = select(y2_ac_q,0,y2_ac_q<0); y2_ac_q = select(y2_ac_q,127,y2_ac_q>127);
	
	y_ac_q = vp8_ac_qlookup[i];
	y2_dc_q = (vp8_dc_qlookup[y2_dc_q])*2;
	y2_ac_q = 31*(vp8_ac_qlookup[y2_ac_q])/20;
	y2_ac_q = select(y2_ac_q,8,y2_ac_q<8);
	
	x = (mb_num % (width/16))*16;
	y = (mb_num / (width/16))*16;
	
	// now read 16 lines of 16 char elements 
	i = y*width + x;
	L0 = convert_short16(vload16(0, current_frame+i)); i+= width;
	L1 = convert_short16(vload16(0, current_frame+i)); i+= width;
	L2 = convert_short16(vload16(0, current_frame+i)); i+= width;
	L3 = convert_short16(vload16(0, current_frame+i)); i+= width;
	L4 = convert_short16(vload16(0, current_frame+i)); i+= width;
	L5 = convert_short16(vload16(0, current_frame+i)); i+= width;
	L6 = convert_short16(vload16(0, current_frame+i)); i+= width;
	L7 = convert_short16(vload16(0, current_frame+i)); i+= width;
	L8 = convert_short16(vload16(0, current_frame+i)); i+= width;
	L9 = convert_short16(vload16(0, current_frame+i)); i+= width;
	L10 = convert_short16(vload16(0, current_frame+i)); i+= width;
	L11 = convert_short16(vload16(0, current_frame+i)); i+= width;
	L12 = convert_short16(vload16(0, current_frame+i)); i+= width;
	L13 = convert_short16(vload16(0, current_frame+i)); i+= width;
	L14 = convert_short16(vload16(0, current_frame+i)); i+= width;
	L15 = convert_short16(vload16(0, current_frame+i));
	
	
	// now read 16 lines of 16 char elements (unaligned and with 3 char ranges between)
	//and immediately count the residual into L-lines
	//also for each four 16pixel-lines we have 4 blocks of 4x4
	// immediately DCTransform them and place coefficients into L-lines
	
	//blocks 00-03
	//i = (y*4 + vector_y)*width*4 + (x*4 + vector_x);
	i = mad24(y,4,vector_y)*mul24(width,4) + mad24(x,4,vector_x);
	buf.x = prev_frame[i]; buf.y = prev_frame[i+4]; buf.z = prev_frame[i+8]; buf.w = prev_frame[i+12]; L0.s0123 -= convert_short4(buf); i += 16;
	buf.x = prev_frame[i]; buf.y = prev_frame[i+4]; buf.z = prev_frame[i+8]; buf.w = prev_frame[i+12]; L0.s4567 -= convert_short4(buf); i += 16;
	buf.x = prev_frame[i]; buf.y = prev_frame[i+4]; buf.z = prev_frame[i+8]; buf.w = prev_frame[i+12]; L0.s89AB -= convert_short4(buf); i += 16;
	buf.x = prev_frame[i]; buf.y = prev_frame[i+4]; buf.z = prev_frame[i+8]; buf.w = prev_frame[i+12]; L0.sCDEF -= convert_short4(buf); i = mad24(width,16,i-48);
	buf.x = prev_frame[i]; buf.y = prev_frame[i+4]; buf.z = prev_frame[i+8]; buf.w = prev_frame[i+12]; L1.s0123 -= convert_short4(buf); i += 16;
	buf.x = prev_frame[i]; buf.y = prev_frame[i+4]; buf.z = prev_frame[i+8]; buf.w = prev_frame[i+12]; L1.s4567 -= convert_short4(buf); i += 16;
	buf.x = prev_frame[i]; buf.y = prev_frame[i+4]; buf.z = prev_frame[i+8]; buf.w = prev_frame[i+12]; L1.s89AB -= convert_short4(buf); i += 16;
	buf.x = prev_frame[i]; buf.y = prev_frame[i+4]; buf.z = prev_frame[i+8]; buf.w = prev_frame[i+12]; L1.sCDEF -= convert_short4(buf); i = mad24(width,16,i-48);
	buf.x = prev_frame[i]; buf.y = prev_frame[i+4]; buf.z = prev_frame[i+8]; buf.w = prev_frame[i+12]; L2.s0123 -= convert_short4(buf); i += 16;
	buf.x = prev_frame[i]; buf.y = prev_frame[i+4]; buf.z = prev_frame[i+8]; buf.w = prev_frame[i+12]; L2.s4567 -= convert_short4(buf); i += 16;
	buf.x = prev_frame[i]; buf.y = prev_frame[i+4]; buf.z = prev_frame[i+8]; buf.w = prev_frame[i+12]; L2.s89AB -= convert_short4(buf); i += 16;
	buf.x = prev_frame[i]; buf.y = prev_frame[i+4]; buf.z = prev_frame[i+8]; buf.w = prev_frame[i+12]; L2.sCDEF -= convert_short4(buf); i = mad24(width,16,i-48);
	buf.x = prev_frame[i]; buf.y = prev_frame[i+4]; buf.z = prev_frame[i+8]; buf.w = prev_frame[i+12]; L3.s0123 -= convert_short4(buf); i += 16;
	buf.x = prev_frame[i]; buf.y = prev_frame[i+4]; buf.z = prev_frame[i+8]; buf.w = prev_frame[i+12]; L3.s4567 -= convert_short4(buf); i += 16;
	buf.x = prev_frame[i]; buf.y = prev_frame[i+4]; buf.z = prev_frame[i+8]; buf.w = prev_frame[i+12]; L3.s89AB -= convert_short4(buf); i += 16;
	buf.x = prev_frame[i]; buf.y = prev_frame[i+4]; buf.z = prev_frame[i+8]; buf.w = prev_frame[i+12]; L3.sCDEF -= convert_short4(buf); i = mad24(width,16,i-48);
	DL0 = convert_int4(L0.s0123); DL1 = convert_int4(L1.s0123); DL2 = convert_int4(L2.s0123); DL3 = convert_int4(L3.s0123);
	DCT_and_quant(&DL0, &DL1, &DL2, &DL3, 1, y_ac_q);
	L0.s0123 = convert_short4(DL0);	L1.s0123 = convert_short4(DL1);	L2.s0123 = convert_short4(DL2);	L3.s0123 = convert_short4(DL3);
	DL0 = convert_int4(L0.s4567); DL1 = convert_int4(L1.s4567); DL2 = convert_int4(L2.s4567); DL3 = convert_int4(L3.s4567);
	DCT_and_quant(&DL0, &DL1, &DL2, &DL3, 1, y_ac_q);
	L0.s4567 = convert_short4(DL0);	L1.s4567 = convert_short4(DL1);	L2.s4567 = convert_short4(DL2);	L3.s4567 = convert_short4(DL3);
	DL0 = convert_int4(L0.s89AB); DL1 = convert_int4(L1.s89AB); DL2 = convert_int4(L2.s89AB); DL3 = convert_int4(L3.s89AB);
	DCT_and_quant(&DL0, &DL1, &DL2, &DL3, 1, y_ac_q);
	L0.s89AB = convert_short4(DL0);	L1.s89AB = convert_short4(DL1);	L2.s89AB = convert_short4(DL2);	L3.s89AB = convert_short4(DL3);
	DL0 = convert_int4(L0.sCDEF); DL1 = convert_int4(L1.sCDEF); DL2 = convert_int4(L2.sCDEF); DL3 = convert_int4(L3.sCDEF);
	DCT_and_quant(&DL0, &DL1, &DL2, &DL3, 1, y_ac_q);
	L0.sCDEF = convert_short4(DL0);	L1.sCDEF = convert_short4(DL1);	L2.sCDEF = convert_short4(DL2);	L3.sCDEF = convert_short4(DL3);
	//blocks 10-13
	buf.x = prev_frame[i]; buf.y = prev_frame[i+4]; buf.z = prev_frame[i+8]; buf.w = prev_frame[i+12]; L4.s0123 -= convert_short4(buf); i += 16;
	buf.x = prev_frame[i]; buf.y = prev_frame[i+4]; buf.z = prev_frame[i+8]; buf.w = prev_frame[i+12]; L4.s4567 -= convert_short4(buf); i += 16;
	buf.x = prev_frame[i]; buf.y = prev_frame[i+4]; buf.z = prev_frame[i+8]; buf.w = prev_frame[i+12]; L4.s89AB -= convert_short4(buf); i += 16;
	buf.x = prev_frame[i]; buf.y = prev_frame[i+4]; buf.z = prev_frame[i+8]; buf.w = prev_frame[i+12]; L4.sCDEF -= convert_short4(buf); i = mad24(width,16,i-48);
	buf.x = prev_frame[i]; buf.y = prev_frame[i+4]; buf.z = prev_frame[i+8]; buf.w = prev_frame[i+12]; L5.s0123 -= convert_short4(buf); i += 16;
	buf.x = prev_frame[i]; buf.y = prev_frame[i+4]; buf.z = prev_frame[i+8]; buf.w = prev_frame[i+12]; L5.s4567 -= convert_short4(buf); i += 16;
	buf.x = prev_frame[i]; buf.y = prev_frame[i+4]; buf.z = prev_frame[i+8]; buf.w = prev_frame[i+12]; L5.s89AB -= convert_short4(buf); i += 16;
	buf.x = prev_frame[i]; buf.y = prev_frame[i+4]; buf.z = prev_frame[i+8]; buf.w = prev_frame[i+12]; L5.sCDEF -= convert_short4(buf); i = mad24(width,16,i-48);
	buf.x = prev_frame[i]; buf.y = prev_frame[i+4]; buf.z = prev_frame[i+8]; buf.w = prev_frame[i+12]; L6.s0123 -= convert_short4(buf); i += 16;
	buf.x = prev_frame[i]; buf.y = prev_frame[i+4]; buf.z = prev_frame[i+8]; buf.w = prev_frame[i+12]; L6.s4567 -= convert_short4(buf); i += 16;
	buf.x = prev_frame[i]; buf.y = prev_frame[i+4]; buf.z = prev_frame[i+8]; buf.w = prev_frame[i+12]; L6.s89AB -= convert_short4(buf); i += 16;
	buf.x = prev_frame[i]; buf.y = prev_frame[i+4]; buf.z = prev_frame[i+8]; buf.w = prev_frame[i+12]; L6.sCDEF -= convert_short4(buf); i = mad24(width,16,i-48);
	buf.x = prev_frame[i]; buf.y = prev_frame[i+4]; buf.z = prev_frame[i+8]; buf.w = prev_frame[i+12]; L7.s0123 -= convert_short4(buf); i += 16;
	buf.x = prev_frame[i]; buf.y = prev_frame[i+4]; buf.z = prev_frame[i+8]; buf.w = prev_frame[i+12]; L7.s4567 -= convert_short4(buf); i += 16;
	buf.x = prev_frame[i]; buf.y = prev_frame[i+4]; buf.z = prev_frame[i+8]; buf.w = prev_frame[i+12]; L7.s89AB -= convert_short4(buf); i += 16;
	buf.x = prev_frame[i]; buf.y = prev_frame[i+4]; buf.z = prev_frame[i+8]; buf.w = prev_frame[i+12]; L7.sCDEF -= convert_short4(buf); i = mad24(width,16,i-48);
	DL0 = convert_int4(L4.s0123); DL1 = convert_int4(L5.s0123); DL2 = convert_int4(L6.s0123); DL3 = convert_int4(L7.s0123);
	DCT_and_quant(&DL0, &DL1, &DL2, &DL3, 1, y_ac_q);
	L4.s0123 = convert_short4(DL0);	L5.s0123 = convert_short4(DL1);	L6.s0123 = convert_short4(DL2);	L7.s0123 = convert_short4(DL3);
	DL0 = convert_int4(L4.s4567); DL1 = convert_int4(L5.s4567); DL2 = convert_int4(L6.s4567); DL3 = convert_int4(L7.s4567);
	DCT_and_quant(&DL0, &DL1, &DL2, &DL3, 1, y_ac_q);
	L4.s4567 = convert_short4(DL0);	L5.s4567 = convert_short4(DL1);	L6.s4567 = convert_short4(DL2);	L7.s4567 = convert_short4(DL3);
	DL0 = convert_int4(L4.s89AB); DL1 = convert_int4(L5.s89AB); DL2 = convert_int4(L6.s89AB); DL3 = convert_int4(L7.s89AB);
	DCT_and_quant(&DL0, &DL1, &DL2, &DL3, 1, y_ac_q);
	L4.s89AB = convert_short4(DL0);	L5.s89AB = convert_short4(DL1);	L6.s89AB = convert_short4(DL2);	L7.s89AB = convert_short4(DL3);
	DL0 = convert_int4(L4.sCDEF); DL1 = convert_int4(L5.sCDEF); DL2 = convert_int4(L6.sCDEF); DL3 = convert_int4(L7.sCDEF);
	DCT_and_quant(&DL0, &DL1, &DL2, &DL3, 1, y_ac_q);
	L4.sCDEF = convert_short4(DL0);	L5.sCDEF = convert_short4(DL1);	L6.sCDEF = convert_short4(DL2);	L7.sCDEF = convert_short4(DL3);
	//blocks 20-23
	buf.x = prev_frame[i]; buf.y = prev_frame[i+4]; buf.z = prev_frame[i+8]; buf.w = prev_frame[i+12]; L8.s0123 -= convert_short4(buf); i += 16;
	buf.x = prev_frame[i]; buf.y = prev_frame[i+4]; buf.z = prev_frame[i+8]; buf.w = prev_frame[i+12]; L8.s4567 -= convert_short4(buf); i += 16;
	buf.x = prev_frame[i]; buf.y = prev_frame[i+4]; buf.z = prev_frame[i+8]; buf.w = prev_frame[i+12]; L8.s89AB -= convert_short4(buf); i += 16;
	buf.x = prev_frame[i]; buf.y = prev_frame[i+4]; buf.z = prev_frame[i+8]; buf.w = prev_frame[i+12]; L8.sCDEF -= convert_short4(buf); i = mad24(width,16,i-48);
	buf.x = prev_frame[i]; buf.y = prev_frame[i+4]; buf.z = prev_frame[i+8]; buf.w = prev_frame[i+12]; L9.s0123 -= convert_short4(buf); i += 16;
	buf.x = prev_frame[i]; buf.y = prev_frame[i+4]; buf.z = prev_frame[i+8]; buf.w = prev_frame[i+12]; L9.s4567 -= convert_short4(buf); i += 16;
	buf.x = prev_frame[i]; buf.y = prev_frame[i+4]; buf.z = prev_frame[i+8]; buf.w = prev_frame[i+12]; L9.s89AB -= convert_short4(buf); i += 16;
	buf.x = prev_frame[i]; buf.y = prev_frame[i+4]; buf.z = prev_frame[i+8]; buf.w = prev_frame[i+12]; L9.sCDEF -= convert_short4(buf); i = mad24(width,16,i-48);
	buf.x = prev_frame[i]; buf.y = prev_frame[i+4]; buf.z = prev_frame[i+8]; buf.w = prev_frame[i+12]; L10.s0123 -= convert_short4(buf); i += 16;
	buf.x = prev_frame[i]; buf.y = prev_frame[i+4]; buf.z = prev_frame[i+8]; buf.w = prev_frame[i+12]; L10.s4567 -= convert_short4(buf); i += 16;
	buf.x = prev_frame[i]; buf.y = prev_frame[i+4]; buf.z = prev_frame[i+8]; buf.w = prev_frame[i+12]; L10.s89AB -= convert_short4(buf); i += 16;
	buf.x = prev_frame[i]; buf.y = prev_frame[i+4]; buf.z = prev_frame[i+8]; buf.w = prev_frame[i+12]; L10.sCDEF -= convert_short4(buf); i = mad24(width,16,i-48);
	buf.x = prev_frame[i]; buf.y = prev_frame[i+4]; buf.z = prev_frame[i+8]; buf.w = prev_frame[i+12]; L11.s0123 -= convert_short4(buf); i += 16;
	buf.x = prev_frame[i]; buf.y = prev_frame[i+4]; buf.z = prev_frame[i+8]; buf.w = prev_frame[i+12]; L11.s4567 -= convert_short4(buf); i += 16;
	buf.x = prev_frame[i]; buf.y = prev_frame[i+4]; buf.z = prev_frame[i+8]; buf.w = prev_frame[i+12]; L11.s89AB -= convert_short4(buf); i += 16;
	buf.x = prev_frame[i]; buf.y = prev_frame[i+4]; buf.z = prev_frame[i+8]; buf.w = prev_frame[i+12]; L11.sCDEF -= convert_short4(buf); i = mad24(width,16,i-48);
	DL0 = convert_int4(L8.s0123); DL1 = convert_int4(L9.s0123); DL2 = convert_int4(L10.s0123); DL3 = convert_int4(L11.s0123);
	DCT_and_quant(&DL0, &DL1, &DL2, &DL3, 1, y_ac_q);
	L8.s0123 = convert_short4(DL0);	L9.s0123 = convert_short4(DL1);	L10.s0123 = convert_short4(DL2);	L11.s0123 = convert_short4(DL3);
	DL0 = convert_int4(L8.s4567); DL1 = convert_int4(L9.s4567); DL2 = convert_int4(L10.s4567); DL3 = convert_int4(L11.s4567);
	DCT_and_quant(&DL0, &DL1, &DL2, &DL3, 1, y_ac_q);
	L8.s4567 = convert_short4(DL0);	L9.s4567 = convert_short4(DL1);	L10.s4567 = convert_short4(DL2);	L11.s4567 = convert_short4(DL3);
	DL0 = convert_int4(L8.s89AB); DL1 = convert_int4(L9.s89AB); DL2 = convert_int4(L10.s89AB); DL3 = convert_int4(L11.s89AB);
	DCT_and_quant(&DL0, &DL1, &DL2, &DL3, 1, y_ac_q);
	L8.s89AB = convert_short4(DL0);	L9.s89AB = convert_short4(DL1);	L10.s89AB = convert_short4(DL2);	L11.s89AB = convert_short4(DL3);
	DL0 = convert_int4(L8.sCDEF); DL1 = convert_int4(L9.sCDEF); DL2 = convert_int4(L10.sCDEF); DL3 = convert_int4(L11.sCDEF);
	DCT_and_quant(&DL0, &DL1, &DL2, &DL3, 1, y_ac_q);
	L8.sCDEF = convert_short4(DL0);	L9.sCDEF = convert_short4(DL1);	L10.sCDEF = convert_short4(DL2);	L11.sCDEF = convert_short4(DL3);
	//blocks 30-33
	buf.x = prev_frame[i]; buf.y = prev_frame[i+4]; buf.z = prev_frame[i+8]; buf.w = prev_frame[i+12]; L12.s0123 -= convert_short4(buf); i += 16;
	buf.x = prev_frame[i]; buf.y = prev_frame[i+4]; buf.z = prev_frame[i+8]; buf.w = prev_frame[i+12]; L12.s4567 -= convert_short4(buf); i += 16;
	buf.x = prev_frame[i]; buf.y = prev_frame[i+4]; buf.z = prev_frame[i+8]; buf.w = prev_frame[i+12]; L12.s89AB -= convert_short4(buf); i += 16;
	buf.x = prev_frame[i]; buf.y = prev_frame[i+4]; buf.z = prev_frame[i+8]; buf.w = prev_frame[i+12]; L12.sCDEF -= convert_short4(buf); i = mad24(width,16,i-48);
	buf.x = prev_frame[i]; buf.y = prev_frame[i+4]; buf.z = prev_frame[i+8]; buf.w = prev_frame[i+12]; L13.s0123 -= convert_short4(buf); i += 16;
	buf.x = prev_frame[i]; buf.y = prev_frame[i+4]; buf.z = prev_frame[i+8]; buf.w = prev_frame[i+12]; L13.s4567 -= convert_short4(buf); i += 16;
	buf.x = prev_frame[i]; buf.y = prev_frame[i+4]; buf.z = prev_frame[i+8]; buf.w = prev_frame[i+12]; L13.s89AB -= convert_short4(buf); i += 16;
	buf.x = prev_frame[i]; buf.y = prev_frame[i+4]; buf.z = prev_frame[i+8]; buf.w = prev_frame[i+12]; L13.sCDEF -= convert_short4(buf); i = mad24(width,16,i-48);
	buf.x = prev_frame[i]; buf.y = prev_frame[i+4]; buf.z = prev_frame[i+8]; buf.w = prev_frame[i+12]; L14.s0123 -= convert_short4(buf); i += 16;
	buf.x = prev_frame[i]; buf.y = prev_frame[i+4]; buf.z = prev_frame[i+8]; buf.w = prev_frame[i+12]; L14.s4567 -= convert_short4(buf); i += 16;
	buf.x = prev_frame[i]; buf.y = prev_frame[i+4]; buf.z = prev_frame[i+8]; buf.w = prev_frame[i+12]; L14.s89AB -= convert_short4(buf); i += 16;
	buf.x = prev_frame[i]; buf.y = prev_frame[i+4]; buf.z = prev_frame[i+8]; buf.w = prev_frame[i+12]; L14.sCDEF -= convert_short4(buf); i = mad24(width,16,i-48);
	buf.x = prev_frame[i]; buf.y = prev_frame[i+4]; buf.z = prev_frame[i+8]; buf.w = prev_frame[i+12]; L15.s0123 -= convert_short4(buf); i += 16;
	buf.x = prev_frame[i]; buf.y = prev_frame[i+4]; buf.z = prev_frame[i+8]; buf.w = prev_frame[i+12]; L15.s4567 -= convert_short4(buf); i += 16;
	buf.x = prev_frame[i]; buf.y = prev_frame[i+4]; buf.z = prev_frame[i+8]; buf.w = prev_frame[i+12]; L15.s89AB -= convert_short4(buf); i += 16;
	buf.x = prev_frame[i]; buf.y = prev_frame[i+4]; buf.z = prev_frame[i+8]; buf.w = prev_frame[i+12]; L15.sCDEF -= convert_short4(buf); 
	DL0 = convert_int4(L12.s0123); DL1 = convert_int4(L13.s0123); DL2 = convert_int4(L14.s0123); DL3 = convert_int4(L15.s0123);
	DCT_and_quant(&DL0, &DL1, &DL2, &DL3, 1, y_ac_q);
	L12.s0123 = convert_short4(DL0);	L13.s0123 = convert_short4(DL1);	L14.s0123 = convert_short4(DL2);	L15.s0123 = convert_short4(DL3);
	DL0 = convert_int4(L12.s4567); DL1 = convert_int4(L13.s4567); DL2 = convert_int4(L14.s4567); DL3 = convert_int4(L15.s4567);
	DCT_and_quant(&DL0, &DL1, &DL2, &DL3, 1, y_ac_q);
	L12.s4567 = convert_short4(DL0);	L13.s4567 = convert_short4(DL1);	L14.s4567 = convert_short4(DL2);	L15.s4567 = convert_short4(DL3);
	DL0 = convert_int4(L12.s89AB); DL1 = convert_int4(L13.s89AB); DL2 = convert_int4(L14.s89AB); DL3 = convert_int4(L15.s89AB);
	DCT_and_quant(&DL0, &DL1, &DL2, &DL3, 1, y_ac_q);
	L12.s89AB = convert_short4(DL0);	L13.s89AB = convert_short4(DL1);	L14.s89AB = convert_short4(DL2);	L15.s89AB = convert_short4(DL3);
	DL0 = convert_int4(L12.sCDEF); DL1 = convert_int4(L13.sCDEF); DL2 = convert_int4(L14.sCDEF); DL3 = convert_int4(L15.sCDEF);
	DCT_and_quant(&DL0, &DL1, &DL2, &DL3, 1, y_ac_q);
	L12.sCDEF = convert_short4(DL0);	L13.sCDEF = convert_short4(DL1);	L14.sCDEF = convert_short4(DL2);	L15.sCDEF = convert_short4(DL3);
	
	//now we need to WHTransform DC-coefficients
	DL0=convert_int4((short4)(L0.s0, L0.s4, L0.s8, L0.sC));
	DL1=convert_int4((short4)(L4.s0, L4.s4, L4.s8, L4.sC));
	DL2=convert_int4((short4)(L8.s0, L8.s4, L8.s8, L8.sC));
	DL3=convert_int4((short4)(L12.s0,L12.s4,L12.s8,L12.sC));
	WHT_and_quant(&DL0, &DL1, &DL2, &DL3, y2_dc_q, y2_ac_q);
	
	//now fill transformed bloks data
	const int inv_zigzag[16] = { 0, 1, 5, 6, 2, 4, 7, 12, 3,  8, 11, 13, 9, 10, 14, 15 };
	//mb_row 0
	//block 00
	                                             MBs[mb_num].coeffs[0][inv_zigzag[1]]=L0.s1;  MBs[mb_num].coeffs[0][inv_zigzag[2]]=L0.s2;  MBs[mb_num].coeffs[0][inv_zigzag[3]]=L0.s3;
	MBs[mb_num].coeffs[0][inv_zigzag[4]]=L1.s0;	 MBs[mb_num].coeffs[0][inv_zigzag[5]]=L1.s1;  MBs[mb_num].coeffs[0][inv_zigzag[6]]=L1.s2;  MBs[mb_num].coeffs[0][inv_zigzag[7]]=L1.s3;
	MBs[mb_num].coeffs[0][inv_zigzag[8]]=L2.s0;	 MBs[mb_num].coeffs[0][inv_zigzag[9]]=L2.s1;  MBs[mb_num].coeffs[0][inv_zigzag[10]]=L2.s2; MBs[mb_num].coeffs[0][inv_zigzag[11]]=L2.s3;
	MBs[mb_num].coeffs[0][inv_zigzag[12]]=L3.s0; MBs[mb_num].coeffs[0][inv_zigzag[13]]=L3.s1; MBs[mb_num].coeffs[0][inv_zigzag[14]]=L3.s2; MBs[mb_num].coeffs[0][inv_zigzag[15]]=L3.s3;
	//block 01
	                                             MBs[mb_num].coeffs[1][inv_zigzag[1]]=L0.s5;  MBs[mb_num].coeffs[1][inv_zigzag[2]]=L0.s6;  MBs[mb_num].coeffs[1][inv_zigzag[3]]=L0.s7;
	MBs[mb_num].coeffs[1][inv_zigzag[4]]=L1.s4;	 MBs[mb_num].coeffs[1][inv_zigzag[5]]=L1.s5;  MBs[mb_num].coeffs[1][inv_zigzag[6]]=L1.s6;  MBs[mb_num].coeffs[1][inv_zigzag[7]]=L1.s7;
	MBs[mb_num].coeffs[1][inv_zigzag[8]]=L2.s4;	 MBs[mb_num].coeffs[1][inv_zigzag[9]]=L2.s5;  MBs[mb_num].coeffs[1][inv_zigzag[10]]=L2.s6; MBs[mb_num].coeffs[1][inv_zigzag[11]]=L2.s7;
	MBs[mb_num].coeffs[1][inv_zigzag[12]]=L3.s4; MBs[mb_num].coeffs[1][inv_zigzag[13]]=L3.s5; MBs[mb_num].coeffs[1][inv_zigzag[14]]=L3.s6; MBs[mb_num].coeffs[1][inv_zigzag[15]]=L3.s7;
	//block 02
	                                             MBs[mb_num].coeffs[2][inv_zigzag[1]]=L0.s9;  MBs[mb_num].coeffs[2][inv_zigzag[2]]=L0.sA;  MBs[mb_num].coeffs[2][inv_zigzag[3]]=L0.sB;
	MBs[mb_num].coeffs[2][inv_zigzag[4]]=L1.s8;	 MBs[mb_num].coeffs[2][inv_zigzag[5]]=L1.s9;  MBs[mb_num].coeffs[2][inv_zigzag[6]]=L1.sA;  MBs[mb_num].coeffs[2][inv_zigzag[7]]=L1.sB;
	MBs[mb_num].coeffs[2][inv_zigzag[8]]=L2.s8;	 MBs[mb_num].coeffs[2][inv_zigzag[9]]=L2.s9;  MBs[mb_num].coeffs[2][inv_zigzag[10]]=L2.sA; MBs[mb_num].coeffs[2][inv_zigzag[11]]=L2.sB;
	MBs[mb_num].coeffs[2][inv_zigzag[12]]=L3.s8; MBs[mb_num].coeffs[2][inv_zigzag[13]]=L3.s9; MBs[mb_num].coeffs[2][inv_zigzag[14]]=L3.sA; MBs[mb_num].coeffs[2][inv_zigzag[15]]=L3.sB;
	//block 03
	                                             MBs[mb_num].coeffs[3][inv_zigzag[1]]=L0.sD;  MBs[mb_num].coeffs[3][inv_zigzag[2]]=L0.sE;  MBs[mb_num].coeffs[3][inv_zigzag[3]]=L0.sF;
	MBs[mb_num].coeffs[3][inv_zigzag[4]]=L1.sC;	 MBs[mb_num].coeffs[3][inv_zigzag[5]]=L1.sD;  MBs[mb_num].coeffs[3][inv_zigzag[6]]=L1.sE;  MBs[mb_num].coeffs[3][inv_zigzag[7]]=L1.sF;
	MBs[mb_num].coeffs[3][inv_zigzag[8]]=L2.sC;	 MBs[mb_num].coeffs[3][inv_zigzag[9]]=L2.sD;  MBs[mb_num].coeffs[3][inv_zigzag[10]]=L2.sE; MBs[mb_num].coeffs[3][inv_zigzag[11]]=L2.sF;
	MBs[mb_num].coeffs[3][inv_zigzag[12]]=L3.sC; MBs[mb_num].coeffs[3][inv_zigzag[13]]=L3.sD; MBs[mb_num].coeffs[3][inv_zigzag[14]]=L3.sE; MBs[mb_num].coeffs[3][inv_zigzag[15]]=L3.sF;
	//mb_row 1
	//block 10
	                                             MBs[mb_num].coeffs[4][inv_zigzag[1]]=L4.s1;  MBs[mb_num].coeffs[4][inv_zigzag[2]]=L4.s2;  MBs[mb_num].coeffs[4][inv_zigzag[3]]=L4.s3;
	MBs[mb_num].coeffs[4][inv_zigzag[4]]=L5.s0;	 MBs[mb_num].coeffs[4][inv_zigzag[5]]=L5.s1;  MBs[mb_num].coeffs[4][inv_zigzag[6]]=L5.s2;  MBs[mb_num].coeffs[4][inv_zigzag[7]]=L5.s3;
	MBs[mb_num].coeffs[4][inv_zigzag[8]]=L6.s0;	 MBs[mb_num].coeffs[4][inv_zigzag[9]]=L6.s1;  MBs[mb_num].coeffs[4][inv_zigzag[10]]=L6.s2; MBs[mb_num].coeffs[4][inv_zigzag[11]]=L6.s3;
	MBs[mb_num].coeffs[4][inv_zigzag[12]]=L7.s0; MBs[mb_num].coeffs[4][inv_zigzag[13]]=L7.s1; MBs[mb_num].coeffs[4][inv_zigzag[14]]=L7.s2; MBs[mb_num].coeffs[4][inv_zigzag[15]]=L7.s3;
	//block 11
	                                             MBs[mb_num].coeffs[5][inv_zigzag[1]]=L4.s5;  MBs[mb_num].coeffs[5][inv_zigzag[2]]=L4.s6;  MBs[mb_num].coeffs[5][inv_zigzag[3]]=L4.s7;
	MBs[mb_num].coeffs[5][inv_zigzag[4]]=L5.s4;	 MBs[mb_num].coeffs[5][inv_zigzag[5]]=L5.s5;  MBs[mb_num].coeffs[5][inv_zigzag[6]]=L5.s6;  MBs[mb_num].coeffs[5][inv_zigzag[7]]=L5.s7;
	MBs[mb_num].coeffs[5][inv_zigzag[8]]=L6.s4;	 MBs[mb_num].coeffs[5][inv_zigzag[9]]=L6.s5;  MBs[mb_num].coeffs[5][inv_zigzag[10]]=L6.s6; MBs[mb_num].coeffs[5][inv_zigzag[11]]=L6.s7;
	MBs[mb_num].coeffs[5][inv_zigzag[12]]=L7.s4; MBs[mb_num].coeffs[5][inv_zigzag[13]]=L7.s5; MBs[mb_num].coeffs[5][inv_zigzag[14]]=L7.s6; MBs[mb_num].coeffs[5][inv_zigzag[15]]=L7.s7;
	//block 12
	                                             MBs[mb_num].coeffs[6][inv_zigzag[1]]=L4.s9;  MBs[mb_num].coeffs[6][inv_zigzag[2]]=L4.sA;  MBs[mb_num].coeffs[6][inv_zigzag[3]]=L4.sB;
	MBs[mb_num].coeffs[6][inv_zigzag[4]]=L5.s8;	 MBs[mb_num].coeffs[6][inv_zigzag[5]]=L5.s9;  MBs[mb_num].coeffs[6][inv_zigzag[6]]=L5.sA;  MBs[mb_num].coeffs[6][inv_zigzag[7]]=L5.sB;
	MBs[mb_num].coeffs[6][inv_zigzag[8]]=L6.s8;	 MBs[mb_num].coeffs[6][inv_zigzag[9]]=L6.s9;  MBs[mb_num].coeffs[6][inv_zigzag[10]]=L6.sA; MBs[mb_num].coeffs[6][inv_zigzag[11]]=L6.sB;
	MBs[mb_num].coeffs[6][inv_zigzag[12]]=L7.s8; MBs[mb_num].coeffs[6][inv_zigzag[13]]=L7.s9; MBs[mb_num].coeffs[6][inv_zigzag[14]]=L7.sA; MBs[mb_num].coeffs[6][inv_zigzag[15]]=L7.sB;
	//block 13
	                                             MBs[mb_num].coeffs[7][inv_zigzag[1]]=L4.sD;  MBs[mb_num].coeffs[7][inv_zigzag[2]]=L4.sE;  MBs[mb_num].coeffs[7][inv_zigzag[3]]=L4.sF;
	MBs[mb_num].coeffs[7][inv_zigzag[4]]=L5.sC;	 MBs[mb_num].coeffs[7][inv_zigzag[5]]=L5.sD;  MBs[mb_num].coeffs[7][inv_zigzag[6]]=L5.sE;  MBs[mb_num].coeffs[7][inv_zigzag[7]]=L5.sF;
	MBs[mb_num].coeffs[7][inv_zigzag[8]]=L6.sC;	 MBs[mb_num].coeffs[7][inv_zigzag[9]]=L6.sD;  MBs[mb_num].coeffs[7][inv_zigzag[10]]=L6.sE; MBs[mb_num].coeffs[7][inv_zigzag[11]]=L6.sF;
	MBs[mb_num].coeffs[7][inv_zigzag[12]]=L7.sC; MBs[mb_num].coeffs[7][inv_zigzag[13]]=L7.sD; MBs[mb_num].coeffs[7][inv_zigzag[14]]=L7.sE; MBs[mb_num].coeffs[7][inv_zigzag[15]]=L7.sF;
	//mb_row 2
	//block 20
	                                             MBs[mb_num].coeffs[8][inv_zigzag[1]]=L8.s1;  MBs[mb_num].coeffs[8][inv_zigzag[2]]=L8.s2;  MBs[mb_num].coeffs[8][inv_zigzag[3]]=L8.s3;
	MBs[mb_num].coeffs[8][inv_zigzag[4]]=L9.s0;	 MBs[mb_num].coeffs[8][inv_zigzag[5]]=L9.s1;  MBs[mb_num].coeffs[8][inv_zigzag[6]]=L9.s2;  MBs[mb_num].coeffs[8][inv_zigzag[7]]=L9.s3;
	MBs[mb_num].coeffs[8][inv_zigzag[8]]=L10.s0; MBs[mb_num].coeffs[8][inv_zigzag[9]]=L10.s1; MBs[mb_num].coeffs[8][inv_zigzag[10]]=L10.s2;MBs[mb_num].coeffs[8][inv_zigzag[11]]=L10.s3;
	MBs[mb_num].coeffs[8][inv_zigzag[12]]=L11.s0;MBs[mb_num].coeffs[8][inv_zigzag[13]]=L11.s1;MBs[mb_num].coeffs[8][inv_zigzag[14]]=L11.s2;MBs[mb_num].coeffs[8][inv_zigzag[15]]=L11.s3;
	//block 21
	                                             MBs[mb_num].coeffs[9][inv_zigzag[1]]=L8.s5;  MBs[mb_num].coeffs[9][inv_zigzag[2]]=L8.s6;  MBs[mb_num].coeffs[9][inv_zigzag[3]]=L8.s7;
	MBs[mb_num].coeffs[9][inv_zigzag[4]]=L9.s4;	 MBs[mb_num].coeffs[9][inv_zigzag[5]]=L9.s5;  MBs[mb_num].coeffs[9][inv_zigzag[6]]=L9.s6;  MBs[mb_num].coeffs[9][inv_zigzag[7]]=L9.s7;
	MBs[mb_num].coeffs[9][inv_zigzag[8]]=L10.s4; MBs[mb_num].coeffs[9][inv_zigzag[9]]=L10.s5; MBs[mb_num].coeffs[9][inv_zigzag[10]]=L10.s6;MBs[mb_num].coeffs[9][inv_zigzag[11]]=L10.s7;
	MBs[mb_num].coeffs[9][inv_zigzag[12]]=L11.s4;MBs[mb_num].coeffs[9][inv_zigzag[13]]=L11.s5;MBs[mb_num].coeffs[9][inv_zigzag[14]]=L11.s6;MBs[mb_num].coeffs[9][inv_zigzag[15]]=L11.s7;
	//block 22
	                                              MBs[mb_num].coeffs[10][inv_zigzag[1]]=L8.s9;  MBs[mb_num].coeffs[10][inv_zigzag[2]]=L8.sA;  MBs[mb_num].coeffs[10][inv_zigzag[3]]=L8.sB;
	MBs[mb_num].coeffs[10][inv_zigzag[4]]=L9.s8;  MBs[mb_num].coeffs[10][inv_zigzag[5]]=L9.s9;  MBs[mb_num].coeffs[10][inv_zigzag[6]]=L9.sA;  MBs[mb_num].coeffs[10][inv_zigzag[7]]=L9.sB;
	MBs[mb_num].coeffs[10][inv_zigzag[8]]=L10.s8; MBs[mb_num].coeffs[10][inv_zigzag[9]]=L10.s9; MBs[mb_num].coeffs[10][inv_zigzag[10]]=L10.sA;MBs[mb_num].coeffs[10][inv_zigzag[11]]=L10.sB;
	MBs[mb_num].coeffs[10][inv_zigzag[12]]=L11.s8;MBs[mb_num].coeffs[10][inv_zigzag[13]]=L11.s9;MBs[mb_num].coeffs[10][inv_zigzag[14]]=L11.sA;MBs[mb_num].coeffs[10][inv_zigzag[15]]=L11.sB;
	//block 23
	                                              MBs[mb_num].coeffs[11][inv_zigzag[1]]=L8.sD;  MBs[mb_num].coeffs[11][inv_zigzag[2]]=L8.sE;  MBs[mb_num].coeffs[11][inv_zigzag[3]]=L8.sF;
	MBs[mb_num].coeffs[11][inv_zigzag[4]]=L9.sC;  MBs[mb_num].coeffs[11][inv_zigzag[5]]=L9.sD;  MBs[mb_num].coeffs[11][inv_zigzag[6]]=L9.sE;  MBs[mb_num].coeffs[11][inv_zigzag[7]]=L9.sF;
	MBs[mb_num].coeffs[11][inv_zigzag[8]]=L10.sC; MBs[mb_num].coeffs[11][inv_zigzag[9]]=L10.sD; MBs[mb_num].coeffs[11][inv_zigzag[10]]=L10.sE;MBs[mb_num].coeffs[11][inv_zigzag[11]]=L10.sF;
	MBs[mb_num].coeffs[11][inv_zigzag[12]]=L11.sC;MBs[mb_num].coeffs[11][inv_zigzag[13]]=L11.sD;MBs[mb_num].coeffs[11][inv_zigzag[14]]=L11.sE;MBs[mb_num].coeffs[11][inv_zigzag[15]]=L11.sF;
	//mb_row 3
	//block 30
	                                              MBs[mb_num].coeffs[12][inv_zigzag[1]]=L12.s1; MBs[mb_num].coeffs[12][inv_zigzag[2]]=L12.s2; MBs[mb_num].coeffs[12][inv_zigzag[3]]=L12.s3;
	MBs[mb_num].coeffs[12][inv_zigzag[4]]=L13.s0; MBs[mb_num].coeffs[12][inv_zigzag[5]]=L13.s1; MBs[mb_num].coeffs[12][inv_zigzag[6]]=L13.s2; MBs[mb_num].coeffs[12][inv_zigzag[7]]=L13.s3;
	MBs[mb_num].coeffs[12][inv_zigzag[8]]=L14.s0; MBs[mb_num].coeffs[12][inv_zigzag[9]]=L14.s1; MBs[mb_num].coeffs[12][inv_zigzag[10]]=L14.s2;MBs[mb_num].coeffs[12][inv_zigzag[11]]=L14.s3;
	MBs[mb_num].coeffs[12][inv_zigzag[12]]=L15.s0;MBs[mb_num].coeffs[12][inv_zigzag[13]]=L15.s1;MBs[mb_num].coeffs[12][inv_zigzag[14]]=L15.s2;MBs[mb_num].coeffs[12][inv_zigzag[15]]=L15.s3;
	//block 31
	                                              MBs[mb_num].coeffs[13][inv_zigzag[1]]=L12.s5; MBs[mb_num].coeffs[13][inv_zigzag[2]]=L12.s6; MBs[mb_num].coeffs[13][inv_zigzag[3]]=L12.s7;
	MBs[mb_num].coeffs[13][inv_zigzag[4]]=L13.s4; MBs[mb_num].coeffs[13][inv_zigzag[5]]=L13.s5; MBs[mb_num].coeffs[13][inv_zigzag[6]]=L13.s6; MBs[mb_num].coeffs[13][inv_zigzag[7]]=L13.s7;
	MBs[mb_num].coeffs[13][inv_zigzag[8]]=L14.s4; MBs[mb_num].coeffs[13][inv_zigzag[9]]=L14.s5; MBs[mb_num].coeffs[13][inv_zigzag[10]]=L14.s6;MBs[mb_num].coeffs[13][inv_zigzag[11]]=L14.s7;
	MBs[mb_num].coeffs[13][inv_zigzag[12]]=L15.s4;MBs[mb_num].coeffs[13][inv_zigzag[13]]=L15.s5;MBs[mb_num].coeffs[13][inv_zigzag[14]]=L15.s6;MBs[mb_num].coeffs[13][inv_zigzag[15]]=L15.s7;
	//block 32
	                                              MBs[mb_num].coeffs[14][inv_zigzag[1]]=L12.s9; MBs[mb_num].coeffs[14][inv_zigzag[2]]=L12.sA; MBs[mb_num].coeffs[14][inv_zigzag[3]]=L12.sB;
	MBs[mb_num].coeffs[14][inv_zigzag[4]]=L13.s8; MBs[mb_num].coeffs[14][inv_zigzag[5]]=L13.s9; MBs[mb_num].coeffs[14][inv_zigzag[6]]=L13.sA; MBs[mb_num].coeffs[14][inv_zigzag[7]]=L13.sB;
	MBs[mb_num].coeffs[14][inv_zigzag[8]]=L14.s8; MBs[mb_num].coeffs[14][inv_zigzag[9]]=L14.s9; MBs[mb_num].coeffs[14][inv_zigzag[10]]=L14.sA;MBs[mb_num].coeffs[14][inv_zigzag[11]]=L14.sB;
	MBs[mb_num].coeffs[14][inv_zigzag[12]]=L15.s8;MBs[mb_num].coeffs[14][inv_zigzag[13]]=L15.s9;MBs[mb_num].coeffs[14][inv_zigzag[14]]=L15.sA;MBs[mb_num].coeffs[14][inv_zigzag[15]]=L15.sB;
	//block 33
	                                              MBs[mb_num].coeffs[15][inv_zigzag[1]]=L12.sD; MBs[mb_num].coeffs[15][inv_zigzag[2]]=L12.sE; MBs[mb_num].coeffs[15][inv_zigzag[3]]=L12.sF;
	MBs[mb_num].coeffs[15][inv_zigzag[4]]=L13.sC; MBs[mb_num].coeffs[15][inv_zigzag[5]]=L13.sD; MBs[mb_num].coeffs[15][inv_zigzag[6]]=L13.sE; MBs[mb_num].coeffs[15][inv_zigzag[7]]=L13.sF;
	MBs[mb_num].coeffs[15][inv_zigzag[8]]=L14.sC; MBs[mb_num].coeffs[15][inv_zigzag[9]]=L14.sD; MBs[mb_num].coeffs[15][inv_zigzag[10]]=L14.sE;MBs[mb_num].coeffs[15][inv_zigzag[11]]=L14.sF;
	MBs[mb_num].coeffs[15][inv_zigzag[12]]=L15.sC;MBs[mb_num].coeffs[15][inv_zigzag[13]]=L15.sD;MBs[mb_num].coeffs[15][inv_zigzag[14]]=L15.sE;MBs[mb_num].coeffs[15][inv_zigzag[15]]=L15.sF;
	
	//block Y2
	MBs[mb_num].coeffs[24][inv_zigzag[0]]=DL0.x;  MBs[mb_num].coeffs[24][inv_zigzag[1]]=DL0.y;  MBs[mb_num].coeffs[24][inv_zigzag[2]]=DL0.z;  MBs[mb_num].coeffs[24][inv_zigzag[3]]=DL0.w;
	MBs[mb_num].coeffs[24][inv_zigzag[4]]=DL1.x;  MBs[mb_num].coeffs[24][inv_zigzag[5]]=DL1.y;  MBs[mb_num].coeffs[24][inv_zigzag[6]]=DL1.z;  MBs[mb_num].coeffs[24][inv_zigzag[7]]=DL1.w;
	MBs[mb_num].coeffs[24][inv_zigzag[8]]=DL2.x;  MBs[mb_num].coeffs[24][inv_zigzag[9]]=DL2.y;  MBs[mb_num].coeffs[24][inv_zigzag[10]]=DL2.z; MBs[mb_num].coeffs[24][inv_zigzag[11]]=DL2.w;
	MBs[mb_num].coeffs[24][inv_zigzag[12]]=DL3.x; MBs[mb_num].coeffs[24][inv_zigzag[13]]=DL3.y; MBs[mb_num].coeffs[24][inv_zigzag[14]]=DL3.z; MBs[mb_num].coeffs[24][inv_zigzag[15]]=DL3.w;
	
	//and now inverse
	//iWHT
    dequant_and_iWHT(&DL0, &DL1, &DL2, &DL3, y2_dc_q, y2_ac_q);
	L0.s0=DL0.x;  L0.s4=DL0.y; L0.s8=DL0.z; L0.sC=DL0.w;
	L4.s0=DL1.x;  L4.s4=DL1.y; L4.s8=DL1.z; L4.sC=DL1.w;
	L8.s0=DL2.x;  L8.s4=DL2.y; L8.s8=DL2.z; L8.sC=DL2.w;
	L12.s0=DL3.x; L12.s4=DL3.y;L12.s8=DL3.z;L12.sC=DL3.w;

	//iDCT and reconstruct
	//blocks 00-03
	i = mad24(y,4,vector_y)*mul24(width,4) + mad24(x,4,vector_x);
	DL0 = convert_int4(L0.s0123); DL1 = convert_int4(L1.s0123); DL2 = convert_int4(L2.s0123); DL3 = convert_int4(L3.s0123);
	dequant_and_iDCT(&DL0, &DL1, &DL2, &DL3, 1, y_ac_q);
	L0.s0123 = convert_short4(DL0);	L1.s0123 = convert_short4(DL1);	L2.s0123 = convert_short4(DL2);	L3.s0123 = convert_short4(DL3);
	DL0 = convert_int4(L0.s4567); DL1 = convert_int4(L1.s4567); DL2 = convert_int4(L2.s4567); DL3 = convert_int4(L3.s4567);
	dequant_and_iDCT(&DL0, &DL1, &DL2, &DL3, 1, y_ac_q);
	L0.s4567 = convert_short4(DL0);	L1.s4567 = convert_short4(DL1);	L2.s4567 = convert_short4(DL2);	L3.s4567 = convert_short4(DL3);
	DL0 = convert_int4(L0.s89AB); DL1 = convert_int4(L1.s89AB); DL2 = convert_int4(L2.s89AB); DL3 = convert_int4(L3.s89AB);
	dequant_and_iDCT(&DL0, &DL1, &DL2, &DL3, 1, y_ac_q);
	L0.s89AB = convert_short4(DL0);	L1.s89AB = convert_short4(DL1);	L2.s89AB = convert_short4(DL2);	L3.s89AB = convert_short4(DL3);
	DL0 = convert_int4(L0.sCDEF); DL1 = convert_int4(L1.sCDEF); DL2 = convert_int4(L2.sCDEF); DL3 = convert_int4(L3.sCDEF);
	dequant_and_iDCT(&DL0, &DL1, &DL2, &DL3, 1, y_ac_q);
	L0.sCDEF = convert_short4(DL0);	L1.sCDEF = convert_short4(DL1);	L2.sCDEF = convert_short4(DL2);	L3.sCDEF = convert_short4(DL3);
	buf.x = prev_frame[i]; buf.y = prev_frame[i+4]; buf.z = prev_frame[i+8]; buf.w = prev_frame[i+12]; L0.s0123 += convert_short4(buf); i += 16;
	buf.x = prev_frame[i]; buf.y = prev_frame[i+4]; buf.z = prev_frame[i+8]; buf.w = prev_frame[i+12]; L0.s4567 += convert_short4(buf); i += 16;
	buf.x = prev_frame[i]; buf.y = prev_frame[i+4]; buf.z = prev_frame[i+8]; buf.w = prev_frame[i+12]; L0.s89AB += convert_short4(buf); i += 16;
	buf.x = prev_frame[i]; buf.y = prev_frame[i+4]; buf.z = prev_frame[i+8]; buf.w = prev_frame[i+12]; L0.sCDEF += convert_short4(buf); i = mad24(width,16,i-48);
	buf.x = prev_frame[i]; buf.y = prev_frame[i+4]; buf.z = prev_frame[i+8]; buf.w = prev_frame[i+12]; L1.s0123 += convert_short4(buf); i += 16;
	buf.x = prev_frame[i]; buf.y = prev_frame[i+4]; buf.z = prev_frame[i+8]; buf.w = prev_frame[i+12]; L1.s4567 += convert_short4(buf); i += 16;
	buf.x = prev_frame[i]; buf.y = prev_frame[i+4]; buf.z = prev_frame[i+8]; buf.w = prev_frame[i+12]; L1.s89AB += convert_short4(buf); i += 16;
	buf.x = prev_frame[i]; buf.y = prev_frame[i+4]; buf.z = prev_frame[i+8]; buf.w = prev_frame[i+12]; L1.sCDEF += convert_short4(buf); i = mad24(width,16,i-48);
	buf.x = prev_frame[i]; buf.y = prev_frame[i+4]; buf.z = prev_frame[i+8]; buf.w = prev_frame[i+12]; L2.s0123 += convert_short4(buf); i += 16;
	buf.x = prev_frame[i]; buf.y = prev_frame[i+4]; buf.z = prev_frame[i+8]; buf.w = prev_frame[i+12]; L2.s4567 += convert_short4(buf); i += 16;
	buf.x = prev_frame[i]; buf.y = prev_frame[i+4]; buf.z = prev_frame[i+8]; buf.w = prev_frame[i+12]; L2.s89AB += convert_short4(buf); i += 16;
	buf.x = prev_frame[i]; buf.y = prev_frame[i+4]; buf.z = prev_frame[i+8]; buf.w = prev_frame[i+12]; L2.sCDEF += convert_short4(buf); i = mad24(width,16,i-48);
	buf.x = prev_frame[i]; buf.y = prev_frame[i+4]; buf.z = prev_frame[i+8]; buf.w = prev_frame[i+12]; L3.s0123 += convert_short4(buf); i += 16;
	buf.x = prev_frame[i]; buf.y = prev_frame[i+4]; buf.z = prev_frame[i+8]; buf.w = prev_frame[i+12]; L3.s4567 += convert_short4(buf); i += 16;
	buf.x = prev_frame[i]; buf.y = prev_frame[i+4]; buf.z = prev_frame[i+8]; buf.w = prev_frame[i+12]; L3.s89AB += convert_short4(buf); i += 16;
	buf.x = prev_frame[i]; buf.y = prev_frame[i+4]; buf.z = prev_frame[i+8]; buf.w = prev_frame[i+12]; L3.sCDEF += convert_short4(buf); i = mad24(width,16,i-48);
	//blocks 10-13
	DL0 = convert_int4(L4.s0123); DL1 = convert_int4(L5.s0123); DL2 = convert_int4(L6.s0123); DL3 = convert_int4(L7.s0123);
	dequant_and_iDCT(&DL0, &DL1, &DL2, &DL3, 1, y_ac_q);
	L4.s0123 = convert_short4(DL0);	L5.s0123 = convert_short4(DL1);	L6.s0123 = convert_short4(DL2);	L7.s0123 = convert_short4(DL3);
	DL0 = convert_int4(L4.s4567); DL1 = convert_int4(L5.s4567); DL2 = convert_int4(L6.s4567); DL3 = convert_int4(L7.s4567);
	dequant_and_iDCT(&DL0, &DL1, &DL2, &DL3, 1, y_ac_q);
	L4.s4567 = convert_short4(DL0);	L5.s4567 = convert_short4(DL1);	L6.s4567 = convert_short4(DL2);	L7.s4567 = convert_short4(DL3);
	DL0 = convert_int4(L4.s89AB); DL1 = convert_int4(L5.s89AB); DL2 = convert_int4(L6.s89AB); DL3 = convert_int4(L7.s89AB);
	dequant_and_iDCT(&DL0, &DL1, &DL2, &DL3, 1, y_ac_q);
	L4.s89AB = convert_short4(DL0);	L5.s89AB = convert_short4(DL1);	L6.s89AB = convert_short4(DL2);	L7.s89AB = convert_short4(DL3);
	DL0 = convert_int4(L4.sCDEF); DL1 = convert_int4(L5.sCDEF); DL2 = convert_int4(L6.sCDEF); DL3 = convert_int4(L7.sCDEF);
	dequant_and_iDCT(&DL0, &DL1, &DL2, &DL3, 1, y_ac_q);
	L4.sCDEF = convert_short4(DL0);	L5.sCDEF = convert_short4(DL1);	L6.sCDEF = convert_short4(DL2);	L7.sCDEF = convert_short4(DL3);
	buf.x = prev_frame[i]; buf.y = prev_frame[i+4]; buf.z = prev_frame[i+8]; buf.w = prev_frame[i+12]; L4.s0123 += convert_short4(buf); i += 16;
	buf.x = prev_frame[i]; buf.y = prev_frame[i+4]; buf.z = prev_frame[i+8]; buf.w = prev_frame[i+12]; L4.s4567 += convert_short4(buf); i += 16;
	buf.x = prev_frame[i]; buf.y = prev_frame[i+4]; buf.z = prev_frame[i+8]; buf.w = prev_frame[i+12]; L4.s89AB += convert_short4(buf); i += 16;
	buf.x = prev_frame[i]; buf.y = prev_frame[i+4]; buf.z = prev_frame[i+8]; buf.w = prev_frame[i+12]; L4.sCDEF += convert_short4(buf); i = mad24(width,16,i-48);
	buf.x = prev_frame[i]; buf.y = prev_frame[i+4]; buf.z = prev_frame[i+8]; buf.w = prev_frame[i+12]; L5.s0123 += convert_short4(buf); i += 16;
	buf.x = prev_frame[i]; buf.y = prev_frame[i+4]; buf.z = prev_frame[i+8]; buf.w = prev_frame[i+12]; L5.s4567 += convert_short4(buf); i += 16;
	buf.x = prev_frame[i]; buf.y = prev_frame[i+4]; buf.z = prev_frame[i+8]; buf.w = prev_frame[i+12]; L5.s89AB += convert_short4(buf); i += 16;
	buf.x = prev_frame[i]; buf.y = prev_frame[i+4]; buf.z = prev_frame[i+8]; buf.w = prev_frame[i+12]; L5.sCDEF += convert_short4(buf); i = mad24(width,16,i-48);
	buf.x = prev_frame[i]; buf.y = prev_frame[i+4]; buf.z = prev_frame[i+8]; buf.w = prev_frame[i+12]; L6.s0123 += convert_short4(buf); i += 16;
	buf.x = prev_frame[i]; buf.y = prev_frame[i+4]; buf.z = prev_frame[i+8]; buf.w = prev_frame[i+12]; L6.s4567 += convert_short4(buf); i += 16;
	buf.x = prev_frame[i]; buf.y = prev_frame[i+4]; buf.z = prev_frame[i+8]; buf.w = prev_frame[i+12]; L6.s89AB += convert_short4(buf); i += 16;
	buf.x = prev_frame[i]; buf.y = prev_frame[i+4]; buf.z = prev_frame[i+8]; buf.w = prev_frame[i+12]; L6.sCDEF += convert_short4(buf); i = mad24(width,16,i-48);
	buf.x = prev_frame[i]; buf.y = prev_frame[i+4]; buf.z = prev_frame[i+8]; buf.w = prev_frame[i+12]; L7.s0123 += convert_short4(buf); i += 16;
	buf.x = prev_frame[i]; buf.y = prev_frame[i+4]; buf.z = prev_frame[i+8]; buf.w = prev_frame[i+12]; L7.s4567 += convert_short4(buf); i += 16;
	buf.x = prev_frame[i]; buf.y = prev_frame[i+4]; buf.z = prev_frame[i+8]; buf.w = prev_frame[i+12]; L7.s89AB += convert_short4(buf); i += 16;
	buf.x = prev_frame[i]; buf.y = prev_frame[i+4]; buf.z = prev_frame[i+8]; buf.w = prev_frame[i+12]; L7.sCDEF += convert_short4(buf); i = mad24(width,16,i-48);
	//blocks 20-23
	DL0 = convert_int4(L8.s0123); DL1 = convert_int4(L9.s0123); DL2 = convert_int4(L10.s0123); DL3 = convert_int4(L11.s0123);
	dequant_and_iDCT(&DL0, &DL1, &DL2, &DL3, 1, y_ac_q);
	L8.s0123 = convert_short4(DL0);	L9.s0123 = convert_short4(DL1);	L10.s0123 = convert_short4(DL2);	L11.s0123 = convert_short4(DL3);
	DL0 = convert_int4(L8.s4567); DL1 = convert_int4(L9.s4567); DL2 = convert_int4(L10.s4567); DL3 = convert_int4(L11.s4567);
	dequant_and_iDCT(&DL0, &DL1, &DL2, &DL3, 1, y_ac_q);
	L8.s4567 = convert_short4(DL0);	L9.s4567 = convert_short4(DL1);	L10.s4567 = convert_short4(DL2);	L11.s4567 = convert_short4(DL3);
	DL0 = convert_int4(L8.s89AB); DL1 = convert_int4(L9.s89AB); DL2 = convert_int4(L10.s89AB); DL3 = convert_int4(L11.s89AB);
	dequant_and_iDCT(&DL0, &DL1, &DL2, &DL3, 1, y_ac_q);
	L8.s89AB = convert_short4(DL0);	L9.s89AB = convert_short4(DL1);	L10.s89AB = convert_short4(DL2);	L11.s89AB = convert_short4(DL3);
	DL0 = convert_int4(L8.sCDEF); DL1 = convert_int4(L9.sCDEF); DL2 = convert_int4(L10.sCDEF); DL3 = convert_int4(L11.sCDEF);
	dequant_and_iDCT(&DL0, &DL1, &DL2, &DL3, 1, y_ac_q);
	L8.sCDEF = convert_short4(DL0);	L9.sCDEF = convert_short4(DL1);	L10.sCDEF = convert_short4(DL2);	L11.sCDEF = convert_short4(DL3);
	buf.x = prev_frame[i]; buf.y = prev_frame[i+4]; buf.z = prev_frame[i+8]; buf.w = prev_frame[i+12]; L8.s0123 += convert_short4(buf); i += 16;
	buf.x = prev_frame[i]; buf.y = prev_frame[i+4]; buf.z = prev_frame[i+8]; buf.w = prev_frame[i+12]; L8.s4567 += convert_short4(buf); i += 16;
	buf.x = prev_frame[i]; buf.y = prev_frame[i+4]; buf.z = prev_frame[i+8]; buf.w = prev_frame[i+12]; L8.s89AB += convert_short4(buf); i += 16;
	buf.x = prev_frame[i]; buf.y = prev_frame[i+4]; buf.z = prev_frame[i+8]; buf.w = prev_frame[i+12]; L8.sCDEF += convert_short4(buf); i = mad24(width,16,i-48);
	buf.x = prev_frame[i]; buf.y = prev_frame[i+4]; buf.z = prev_frame[i+8]; buf.w = prev_frame[i+12]; L9.s0123 += convert_short4(buf); i += 16;
	buf.x = prev_frame[i]; buf.y = prev_frame[i+4]; buf.z = prev_frame[i+8]; buf.w = prev_frame[i+12]; L9.s4567 += convert_short4(buf); i += 16;
	buf.x = prev_frame[i]; buf.y = prev_frame[i+4]; buf.z = prev_frame[i+8]; buf.w = prev_frame[i+12]; L9.s89AB += convert_short4(buf); i += 16;
	buf.x = prev_frame[i]; buf.y = prev_frame[i+4]; buf.z = prev_frame[i+8]; buf.w = prev_frame[i+12]; L9.sCDEF += convert_short4(buf); i = mad24(width,16,i-48);
	buf.x = prev_frame[i]; buf.y = prev_frame[i+4]; buf.z = prev_frame[i+8]; buf.w = prev_frame[i+12]; L10.s0123 += convert_short4(buf); i += 16;
	buf.x = prev_frame[i]; buf.y = prev_frame[i+4]; buf.z = prev_frame[i+8]; buf.w = prev_frame[i+12]; L10.s4567 += convert_short4(buf); i += 16;
	buf.x = prev_frame[i]; buf.y = prev_frame[i+4]; buf.z = prev_frame[i+8]; buf.w = prev_frame[i+12]; L10.s89AB += convert_short4(buf); i += 16;
	buf.x = prev_frame[i]; buf.y = prev_frame[i+4]; buf.z = prev_frame[i+8]; buf.w = prev_frame[i+12]; L10.sCDEF += convert_short4(buf); i = mad24(width,16,i-48);
	buf.x = prev_frame[i]; buf.y = prev_frame[i+4]; buf.z = prev_frame[i+8]; buf.w = prev_frame[i+12]; L11.s0123 += convert_short4(buf); i += 16;
	buf.x = prev_frame[i]; buf.y = prev_frame[i+4]; buf.z = prev_frame[i+8]; buf.w = prev_frame[i+12]; L11.s4567 += convert_short4(buf); i += 16;
	buf.x = prev_frame[i]; buf.y = prev_frame[i+4]; buf.z = prev_frame[i+8]; buf.w = prev_frame[i+12]; L11.s89AB += convert_short4(buf); i += 16;
	buf.x = prev_frame[i]; buf.y = prev_frame[i+4]; buf.z = prev_frame[i+8]; buf.w = prev_frame[i+12]; L11.sCDEF += convert_short4(buf); i = mad24(width,16,i-48);
	//blocks 30-33
	DL0 = convert_int4(L12.s0123); DL1 = convert_int4(L13.s0123); DL2 = convert_int4(L14.s0123); DL3 = convert_int4(L15.s0123);
	dequant_and_iDCT(&DL0, &DL1, &DL2, &DL3, 1, y_ac_q);
	L12.s0123 = convert_short4(DL0);	L13.s0123 = convert_short4(DL1);	L14.s0123 = convert_short4(DL2);	L15.s0123 = convert_short4(DL3);
	DL0 = convert_int4(L12.s4567); DL1 = convert_int4(L13.s4567); DL2 = convert_int4(L14.s4567); DL3 = convert_int4(L15.s4567);
	dequant_and_iDCT(&DL0, &DL1, &DL2, &DL3, 1, y_ac_q);
	L12.s4567 = convert_short4(DL0);	L13.s4567 = convert_short4(DL1);	L14.s4567 = convert_short4(DL2);	L15.s4567 = convert_short4(DL3);
	DL0 = convert_int4(L12.s89AB); DL1 = convert_int4(L13.s89AB); DL2 = convert_int4(L14.s89AB); DL3 = convert_int4(L15.s89AB);
	dequant_and_iDCT(&DL0, &DL1, &DL2, &DL3, 1, y_ac_q);
	L12.s89AB = convert_short4(DL0);	L13.s89AB = convert_short4(DL1);	L14.s89AB = convert_short4(DL2);	L15.s89AB = convert_short4(DL3);
	DL0 = convert_int4(L12.sCDEF); DL1 = convert_int4(L13.sCDEF); DL2 = convert_int4(L14.sCDEF); DL3 = convert_int4(L15.sCDEF);
	dequant_and_iDCT(&DL0, &DL1, &DL2, &DL3, 1, y_ac_q);
	L12.sCDEF = convert_short4(DL0);	L13.sCDEF = convert_short4(DL1);	L14.sCDEF = convert_short4(DL2);	L15.sCDEF = convert_short4(DL3);
	buf.x = prev_frame[i]; buf.y = prev_frame[i+4]; buf.z = prev_frame[i+8]; buf.w = prev_frame[i+12]; L12.s0123 += convert_short4(buf); i += 16;
	buf.x = prev_frame[i]; buf.y = prev_frame[i+4]; buf.z = prev_frame[i+8]; buf.w = prev_frame[i+12]; L12.s4567 += convert_short4(buf); i += 16;
	buf.x = prev_frame[i]; buf.y = prev_frame[i+4]; buf.z = prev_frame[i+8]; buf.w = prev_frame[i+12]; L12.s89AB += convert_short4(buf); i += 16;
	buf.x = prev_frame[i]; buf.y = prev_frame[i+4]; buf.z = prev_frame[i+8]; buf.w = prev_frame[i+12]; L12.sCDEF += convert_short4(buf); i = mad24(width,16,i-48);
	buf.x = prev_frame[i]; buf.y = prev_frame[i+4]; buf.z = prev_frame[i+8]; buf.w = prev_frame[i+12]; L13.s0123 += convert_short4(buf); i += 16;
	buf.x = prev_frame[i]; buf.y = prev_frame[i+4]; buf.z = prev_frame[i+8]; buf.w = prev_frame[i+12]; L13.s4567 += convert_short4(buf); i += 16;
	buf.x = prev_frame[i]; buf.y = prev_frame[i+4]; buf.z = prev_frame[i+8]; buf.w = prev_frame[i+12]; L13.s89AB += convert_short4(buf); i += 16;
	buf.x = prev_frame[i]; buf.y = prev_frame[i+4]; buf.z = prev_frame[i+8]; buf.w = prev_frame[i+12]; L13.sCDEF += convert_short4(buf); i = mad24(width,16,i-48);
	buf.x = prev_frame[i]; buf.y = prev_frame[i+4]; buf.z = prev_frame[i+8]; buf.w = prev_frame[i+12]; L14.s0123 += convert_short4(buf); i += 16;
	buf.x = prev_frame[i]; buf.y = prev_frame[i+4]; buf.z = prev_frame[i+8]; buf.w = prev_frame[i+12]; L14.s4567 += convert_short4(buf); i += 16;
	buf.x = prev_frame[i]; buf.y = prev_frame[i+4]; buf.z = prev_frame[i+8]; buf.w = prev_frame[i+12]; L14.s89AB += convert_short4(buf); i += 16;
	buf.x = prev_frame[i]; buf.y = prev_frame[i+4]; buf.z = prev_frame[i+8]; buf.w = prev_frame[i+12]; L14.sCDEF += convert_short4(buf); i = mad24(width,16,i-48);
	buf.x = prev_frame[i]; buf.y = prev_frame[i+4]; buf.z = prev_frame[i+8]; buf.w = prev_frame[i+12]; L15.s0123 += convert_short4(buf); i += 16;
	buf.x = prev_frame[i]; buf.y = prev_frame[i+4]; buf.z = prev_frame[i+8]; buf.w = prev_frame[i+12]; L15.s4567 += convert_short4(buf); i += 16;
	buf.x = prev_frame[i]; buf.y = prev_frame[i+4]; buf.z = prev_frame[i+8]; buf.w = prev_frame[i+12]; L15.s89AB += convert_short4(buf); i += 16;
	buf.x = prev_frame[i]; buf.y = prev_frame[i+4]; buf.z = prev_frame[i+8]; buf.w = prev_frame[i+12]; L15.sCDEF += convert_short4(buf); 
	
	//now place results in reconstruction frame
	i = y*width + x;
	buf = convert_uchar4_sat(L0.s0123); vstore4(buf, 0, recon_frame+i);
	buf = convert_uchar4_sat(L0.s4567); vstore4(buf, 0, recon_frame+i+4);
	buf = convert_uchar4_sat(L0.s89AB); vstore4(buf, 0, recon_frame+i+8);
	buf = convert_uchar4_sat(L0.sCDEF); vstore4(buf, 0, recon_frame+i+12); i+=width;
	buf = convert_uchar4_sat(L1.s0123); vstore4(buf, 0, recon_frame+i);
	buf = convert_uchar4_sat(L1.s4567); vstore4(buf, 0, recon_frame+i+4);
	buf = convert_uchar4_sat(L1.s89AB); vstore4(buf, 0, recon_frame+i+8);
	buf = convert_uchar4_sat(L1.sCDEF); vstore4(buf, 0, recon_frame+i+12); i+=width;
	buf = convert_uchar4_sat(L2.s0123); vstore4(buf, 0, recon_frame+i);
	buf = convert_uchar4_sat(L2.s4567); vstore4(buf, 0, recon_frame+i+4);
	buf = convert_uchar4_sat(L2.s89AB); vstore4(buf, 0, recon_frame+i+8);
	buf = convert_uchar4_sat(L2.sCDEF); vstore4(buf, 0, recon_frame+i+12); i+=width;
	buf = convert_uchar4_sat(L3.s0123); vstore4(buf, 0, recon_frame+i);
	buf = convert_uchar4_sat(L3.s4567); vstore4(buf, 0, recon_frame+i+4);
	buf = convert_uchar4_sat(L3.s89AB); vstore4(buf, 0, recon_frame+i+8);
	buf = convert_uchar4_sat(L3.sCDEF); vstore4(buf, 0, recon_frame+i+12); i+=width;
	buf = convert_uchar4_sat(L4.s0123); vstore4(buf, 0, recon_frame+i);
	buf = convert_uchar4_sat(L4.s4567); vstore4(buf, 0, recon_frame+i+4);
	buf = convert_uchar4_sat(L4.s89AB); vstore4(buf, 0, recon_frame+i+8);
	buf = convert_uchar4_sat(L4.sCDEF); vstore4(buf, 0, recon_frame+i+12); i+=width;
	buf = convert_uchar4_sat(L5.s0123); vstore4(buf, 0, recon_frame+i);
	buf = convert_uchar4_sat(L5.s4567); vstore4(buf, 0, recon_frame+i+4);
	buf = convert_uchar4_sat(L5.s89AB); vstore4(buf, 0, recon_frame+i+8);
	buf = convert_uchar4_sat(L5.sCDEF); vstore4(buf, 0, recon_frame+i+12); i+=width;
	buf = convert_uchar4_sat(L6.s0123); vstore4(buf, 0, recon_frame+i);
	buf = convert_uchar4_sat(L6.s4567); vstore4(buf, 0, recon_frame+i+4);
	buf = convert_uchar4_sat(L6.s89AB); vstore4(buf, 0, recon_frame+i+8);
	buf = convert_uchar4_sat(L6.sCDEF); vstore4(buf, 0, recon_frame+i+12); i+=width;
	buf = convert_uchar4_sat(L7.s0123); vstore4(buf, 0, recon_frame+i);
	buf = convert_uchar4_sat(L7.s4567); vstore4(buf, 0, recon_frame+i+4);
	buf = convert_uchar4_sat(L7.s89AB); vstore4(buf, 0, recon_frame+i+8);
	buf = convert_uchar4_sat(L7.sCDEF); vstore4(buf, 0, recon_frame+i+12); i+=width;
	buf = convert_uchar4_sat(L8.s0123); vstore4(buf, 0, recon_frame+i);
	buf = convert_uchar4_sat(L8.s4567); vstore4(buf, 0, recon_frame+i+4);
	buf = convert_uchar4_sat(L8.s89AB); vstore4(buf, 0, recon_frame+i+8);
	buf = convert_uchar4_sat(L8.sCDEF); vstore4(buf, 0, recon_frame+i+12); i+=width;
	buf = convert_uchar4_sat(L9.s0123); vstore4(buf, 0, recon_frame+i);
	buf = convert_uchar4_sat(L9.s4567); vstore4(buf, 0, recon_frame+i+4);
	buf = convert_uchar4_sat(L9.s89AB); vstore4(buf, 0, recon_frame+i+8);
	buf = convert_uchar4_sat(L9.sCDEF); vstore4(buf, 0, recon_frame+i+12); i+=width;
	buf = convert_uchar4_sat(L10.s0123); vstore4(buf, 0, recon_frame+i);
	buf = convert_uchar4_sat(L10.s4567); vstore4(buf, 0, recon_frame+i+4);
	buf = convert_uchar4_sat(L10.s89AB); vstore4(buf, 0, recon_frame+i+8);
	buf = convert_uchar4_sat(L10.sCDEF); vstore4(buf, 0, recon_frame+i+12); i+=width;
	buf = convert_uchar4_sat(L11.s0123); vstore4(buf, 0, recon_frame+i);
	buf = convert_uchar4_sat(L11.s4567); vstore4(buf, 0, recon_frame+i+4);
	buf = convert_uchar4_sat(L11.s89AB); vstore4(buf, 0, recon_frame+i+8);
	buf = convert_uchar4_sat(L11.sCDEF); vstore4(buf, 0, recon_frame+i+12); i+=width;
	buf = convert_uchar4_sat(L12.s0123); vstore4(buf, 0, recon_frame+i);
	buf = convert_uchar4_sat(L12.s4567); vstore4(buf, 0, recon_frame+i+4);
	buf = convert_uchar4_sat(L12.s89AB); vstore4(buf, 0, recon_frame+i+8);
	buf = convert_uchar4_sat(L12.sCDEF); vstore4(buf, 0, recon_frame+i+12); i+=width;
	buf = convert_uchar4_sat(L13.s0123); vstore4(buf, 0, recon_frame+i);
	buf = convert_uchar4_sat(L13.s4567); vstore4(buf, 0, recon_frame+i+4);
	buf = convert_uchar4_sat(L13.s89AB); vstore4(buf, 0, recon_frame+i+8);
	buf = convert_uchar4_sat(L13.sCDEF); vstore4(buf, 0, recon_frame+i+12); i+=width;
	buf = convert_uchar4_sat(L14.s0123); vstore4(buf, 0, recon_frame+i);
	buf = convert_uchar4_sat(L14.s4567); vstore4(buf, 0, recon_frame+i+4);
	buf = convert_uchar4_sat(L14.s89AB); vstore4(buf, 0, recon_frame+i+8);
	buf = convert_uchar4_sat(L14.sCDEF); vstore4(buf, 0, recon_frame+i+12); i+=width;
	buf = convert_uchar4_sat(L15.s0123); vstore4(buf, 0, recon_frame+i);
	buf = convert_uchar4_sat(L15.s4567); vstore4(buf, 0, recon_frame+i+4);
	buf = convert_uchar4_sat(L15.s89AB); vstore4(buf, 0, recon_frame+i+8);
	buf = convert_uchar4_sat(L15.sCDEF); vstore4(buf, 0, recon_frame+i+12);

	return;
}

	
__kernel void luma_transform_8x8(__global uchar *const current_frame, //0
								__global uchar *const recon_frame, //1
								__global uchar *const prev_frame, //2
								__global macroblock *const MBs, //3
								const signed int width, //4
								__constant segment_data *const SD) //5
	
{
	__private int4 DL0, DL1, DL2, DL3;

	__private short8 DCTLine0, DCTLine1, DCTLine2, DCTLine3,
					  DCTLine4, DCTLine5, DCTLine6, DCTLine7;
	__private uchar4 CL, PL;
	
   	__private int cx, cy, px, py, vector_x, vector_y;	
	__private int ci, pi; 
	__private int mb_num, b8x8_num, b4x4_in_mb,b8x8_in_mb,dc_q,ac_q;
	__private int width_x4 = width*4;		
	
	b8x8_num = get_global_id(0); 
	cx = (b8x8_num % (width/8))*8;
	cy = (b8x8_num / (width/8))*8;
	mb_num = (cy/16)*(width/16) + (cx/16);
	
	if (MBs[mb_num].parts != are8x8) return;
	if (MBs[mb_num].reference_frame != LAST) return;
	
	ac_q = MBs[mb_num].segment_id;
	ac_q = SD[ac_q].y_ac_i;
	dc_q = SD[0].y_dc_idelta; dc_q += ac_q;
	dc_q = select(dc_q,0,dc_q<0); dc_q = select(dc_q,127,dc_q>127);
	ac_q = select(ac_q,0,ac_q<0); ac_q = select(ac_q,127,ac_q>127);
	dc_q = vp8_dc_qlookup[dc_q];
	ac_q = vp8_ac_qlookup[ac_q];
	
	ci = cy*width + cx; 
	b4x4_in_mb = ((cy%16)/4)*4 + (cx%16)/4;
	b8x8_in_mb = ((cy%16)/8)*2 + (cx%16)/8;

	vector_x = MBs[mb_num].vector_x[b8x8_in_mb];
	vector_y = MBs[mb_num].vector_y[b8x8_in_mb];
		
	//now go to qpel
	cx*=4; cy*=4;
	py = cy + vector_y;
	px = cx + vector_x;
	
	pi = py*width_x4+px;

	const int inv_zigzag[16] = { 0, 1, 5, 6, 2, 4, 7, 12, 3,  8, 11, 13, 9, 10, 14, 15 };
	
	// block 00
	CL = vload4(0, &current_frame[ci]); ci += width;
	PL.x = prev_frame[pi]; PL.y = prev_frame[pi+4]; PL.z = prev_frame[pi+8]; PL.w = prev_frame[pi+12]; pi += width_x4*4;
	DL0 = convert_int4(CL) - convert_int4(PL);
	CL = vload4(0, &current_frame[ci]); ci += width;
	PL.x = prev_frame[pi]; PL.y = prev_frame[pi+4]; PL.z = prev_frame[pi+8]; PL.w = prev_frame[pi+12]; pi += width_x4*4;
	DL1 = convert_int4(CL) - convert_int4(PL);
	CL = vload4(0, &current_frame[ci]); ci += width;
	PL.x = prev_frame[pi]; PL.y = prev_frame[pi+4]; PL.z = prev_frame[pi+8]; PL.w = prev_frame[pi+12]; pi += width_x4*4;
	DL2 = convert_int4(CL) - convert_int4(PL);
	CL = vload4(0, &current_frame[ci]); ci += width;
	PL.x = prev_frame[pi]; PL.y = prev_frame[pi+4]; PL.z = prev_frame[pi+8]; PL.w = prev_frame[pi+12]; pi += width_x4*4;
	DL3 = convert_int4(CL) - convert_int4(PL);
	ci -= (width*4); pi -= (width_x4*16);
	DCT_and_quant(&DL0, &DL1, &DL2, &DL3, dc_q, ac_q);
	DCTLine0.s0123 = convert_short4(DL0);
	DCTLine1.s0123 = convert_short4(DL1);
	DCTLine2.s0123 = convert_short4(DL2);
	DCTLine3.s0123 = convert_short4(DL3);
	// block 01
	ci += 4; pi += 16;
	CL = vload4(0, &current_frame[ci]); ci += width;
	PL.x = prev_frame[pi]; PL.y = prev_frame[pi+4]; PL.z = prev_frame[pi+8]; PL.w = prev_frame[pi+12]; pi += width_x4*4;
	DL0 = convert_int4(CL) - convert_int4(PL);
	CL = vload4(0, &current_frame[ci]); ci += width;
	PL.x = prev_frame[pi]; PL.y = prev_frame[pi+4]; PL.z = prev_frame[pi+8]; PL.w = prev_frame[pi+12]; pi += width_x4*4;
	DL1 = convert_int4(CL) - convert_int4(PL);
	CL = vload4(0, &current_frame[ci]); ci += width;
	PL.x = prev_frame[pi]; PL.y = prev_frame[pi+4]; PL.z = prev_frame[pi+8]; PL.w = prev_frame[pi+12]; pi += width_x4*4;
	DL2 = convert_int4(CL) - convert_int4(PL);
	CL = vload4(0, &current_frame[ci]); ci += width;
	PL.x = prev_frame[pi]; PL.y = prev_frame[pi+4]; PL.z = prev_frame[pi+8]; PL.w = prev_frame[pi+12]; pi += width_x4*4;
	DL3 = convert_int4(CL) - convert_int4(PL);
	ci -= 4; pi -= 16;
	DCT_and_quant(&DL0, &DL1, &DL2, &DL3, dc_q, ac_q);
	DCTLine0.s4567 = convert_short4(DL0);
	DCTLine1.s4567 = convert_short4(DL1);
	DCTLine2.s4567 = convert_short4(DL2);
	DCTLine3.s4567 = convert_short4(DL3);
	// block 10
	CL = vload4(0, &current_frame[ci]); ci += width;
	PL.x = prev_frame[pi]; PL.y = prev_frame[pi+4]; PL.z = prev_frame[pi+8]; PL.w = prev_frame[pi+12]; pi += width_x4*4;
	DL0 = convert_int4(CL) - convert_int4(PL);
	CL = vload4(0, &current_frame[ci]); ci += width;
	PL.x = prev_frame[pi]; PL.y = prev_frame[pi+4]; PL.z = prev_frame[pi+8]; PL.w = prev_frame[pi+12]; pi += width_x4*4;
	DL1 = convert_int4(CL) - convert_int4(PL);
	CL = vload4(0, &current_frame[ci]); ci += width;
	PL.x = prev_frame[pi]; PL.y = prev_frame[pi+4]; PL.z = prev_frame[pi+8]; PL.w = prev_frame[pi+12]; pi += width_x4*4;
	DL2 = convert_int4(CL) - convert_int4(PL);
	CL = vload4(0, &current_frame[ci]); ci += width;
	PL.x = prev_frame[pi]; PL.y = prev_frame[pi+4]; PL.z = prev_frame[pi+8]; PL.w = prev_frame[pi+12]; pi += width_x4*4;
	DL3 = convert_int4(CL) - convert_int4(PL);
	ci -= (width*4); pi -= (width_x4*16);
	DCT_and_quant(&DL0, &DL1, &DL2, &DL3, dc_q, ac_q);
	DCTLine4.s0123 = convert_short4(DL0);
	DCTLine5.s0123 = convert_short4(DL1);
	DCTLine6.s0123 = convert_short4(DL2);
	DCTLine7.s0123 = convert_short4(DL3);
	// block 11
	ci += 4; pi += 16;
	CL = vload4(0, &current_frame[ci]); ci += width;
	PL.x = prev_frame[pi]; PL.y = prev_frame[pi+4]; PL.z = prev_frame[pi+8]; PL.w = prev_frame[pi+12]; pi += width_x4*4;
	DL0 = convert_int4(CL) - convert_int4(PL);
	CL = vload4(0, &current_frame[ci]); ci += width;
	PL.x = prev_frame[pi]; PL.y = prev_frame[pi+4]; PL.z = prev_frame[pi+8]; PL.w = prev_frame[pi+12]; pi += width_x4*4;
	DL1 = convert_int4(CL) - convert_int4(PL);
	CL = vload4(0, &current_frame[ci]); ci += width;
	PL.x = prev_frame[pi]; PL.y = prev_frame[pi+4]; PL.z = prev_frame[pi+8]; PL.w = prev_frame[pi+12]; pi += width_x4*4;
	DL2 = convert_int4(CL) - convert_int4(PL);
	CL = vload4(0, &current_frame[ci]); ci += width;
	PL.x = prev_frame[pi]; PL.y = prev_frame[pi+4]; PL.z = prev_frame[pi+8]; PL.w = prev_frame[pi+12]; pi += width_x4*4;
	DL3 = convert_int4(CL) - convert_int4(PL);
	DCT_and_quant(&DL0, &DL1, &DL2, &DL3, dc_q, ac_q);
	DCTLine4.s4567 = convert_short4(DL0);
	DCTLine5.s4567 = convert_short4(DL1);
	DCTLine6.s4567 = convert_short4(DL2);
	DCTLine7.s4567 = convert_short4(DL3);
	ci -= 4; pi -= 16;
	ci -= (width*8); pi -= (width_x4*32);

	// block 00
	MBs[mb_num].coeffs[b4x4_in_mb][inv_zigzag[0]]=DCTLine0.s0;
	MBs[mb_num].coeffs[b4x4_in_mb][inv_zigzag[1]]=DCTLine0.s1; 
	MBs[mb_num].coeffs[b4x4_in_mb][inv_zigzag[2]]=DCTLine0.s2; 
	MBs[mb_num].coeffs[b4x4_in_mb][inv_zigzag[3]]=DCTLine0.s3;
	MBs[mb_num].coeffs[b4x4_in_mb][inv_zigzag[4]]=DCTLine1.s0; 
	MBs[mb_num].coeffs[b4x4_in_mb][inv_zigzag[5]]=DCTLine1.s1; 
	MBs[mb_num].coeffs[b4x4_in_mb][inv_zigzag[6]]=DCTLine1.s2; 
	MBs[mb_num].coeffs[b4x4_in_mb][inv_zigzag[7]]=DCTLine1.s3;
	MBs[mb_num].coeffs[b4x4_in_mb][inv_zigzag[8]]=DCTLine2.s0; 
	MBs[mb_num].coeffs[b4x4_in_mb][inv_zigzag[9]]=DCTLine2.s1; 
	MBs[mb_num].coeffs[b4x4_in_mb][inv_zigzag[10]]=DCTLine2.s2; 
	MBs[mb_num].coeffs[b4x4_in_mb][inv_zigzag[11]]=DCTLine2.s3;
	MBs[mb_num].coeffs[b4x4_in_mb][inv_zigzag[12]]=DCTLine3.s0; 
	MBs[mb_num].coeffs[b4x4_in_mb][inv_zigzag[13]]=DCTLine3.s1; 
	MBs[mb_num].coeffs[b4x4_in_mb][inv_zigzag[14]]=DCTLine3.s2; 
	MBs[mb_num].coeffs[b4x4_in_mb][inv_zigzag[15]]=DCTLine3.s3;
	// block 01
	MBs[mb_num].coeffs[b4x4_in_mb + 1][inv_zigzag[0]]=DCTLine0.s4;
	MBs[mb_num].coeffs[b4x4_in_mb + 1][inv_zigzag[1]]=DCTLine0.s5; 
	MBs[mb_num].coeffs[b4x4_in_mb + 1][inv_zigzag[2]]=DCTLine0.s6; 
	MBs[mb_num].coeffs[b4x4_in_mb + 1][inv_zigzag[3]]=DCTLine0.s7;
	MBs[mb_num].coeffs[b4x4_in_mb + 1][inv_zigzag[4]]=DCTLine1.s4; 
	MBs[mb_num].coeffs[b4x4_in_mb + 1][inv_zigzag[5]]=DCTLine1.s5; 
	MBs[mb_num].coeffs[b4x4_in_mb + 1][inv_zigzag[6]]=DCTLine1.s6; 
	MBs[mb_num].coeffs[b4x4_in_mb + 1][inv_zigzag[7]]=DCTLine1.s7;
	MBs[mb_num].coeffs[b4x4_in_mb + 1][inv_zigzag[8]]=DCTLine2.s4; 
	MBs[mb_num].coeffs[b4x4_in_mb + 1][inv_zigzag[9]]=DCTLine2.s5; 
	MBs[mb_num].coeffs[b4x4_in_mb + 1][inv_zigzag[10]]=DCTLine2.s6; 
	MBs[mb_num].coeffs[b4x4_in_mb + 1][inv_zigzag[11]]=DCTLine2.s7;
	MBs[mb_num].coeffs[b4x4_in_mb + 1][inv_zigzag[12]]=DCTLine3.s4; 
	MBs[mb_num].coeffs[b4x4_in_mb + 1][inv_zigzag[13]]=DCTLine3.s5; 
	MBs[mb_num].coeffs[b4x4_in_mb + 1][inv_zigzag[14]]=DCTLine3.s6; 
	MBs[mb_num].coeffs[b4x4_in_mb + 1][inv_zigzag[15]]=DCTLine3.s7;
	// block 10
	MBs[mb_num].coeffs[b4x4_in_mb + 4][inv_zigzag[0]]=DCTLine4.s0;
	MBs[mb_num].coeffs[b4x4_in_mb + 4][inv_zigzag[1]]=DCTLine4.s1; 
	MBs[mb_num].coeffs[b4x4_in_mb + 4][inv_zigzag[2]]=DCTLine4.s2; 
	MBs[mb_num].coeffs[b4x4_in_mb + 4][inv_zigzag[3]]=DCTLine4.s3;
	MBs[mb_num].coeffs[b4x4_in_mb + 4][inv_zigzag[4]]=DCTLine5.s0; 
	MBs[mb_num].coeffs[b4x4_in_mb + 4][inv_zigzag[5]]=DCTLine5.s1; 
	MBs[mb_num].coeffs[b4x4_in_mb + 4][inv_zigzag[6]]=DCTLine5.s2; 
	MBs[mb_num].coeffs[b4x4_in_mb + 4][inv_zigzag[7]]=DCTLine5.s3;
	MBs[mb_num].coeffs[b4x4_in_mb + 4][inv_zigzag[8]]=DCTLine6.s0; 
	MBs[mb_num].coeffs[b4x4_in_mb + 4][inv_zigzag[9]]=DCTLine6.s1; 
	MBs[mb_num].coeffs[b4x4_in_mb + 4][inv_zigzag[10]]=DCTLine6.s2; 
	MBs[mb_num].coeffs[b4x4_in_mb + 4][inv_zigzag[11]]=DCTLine6.s3;
	MBs[mb_num].coeffs[b4x4_in_mb + 4][inv_zigzag[12]]=DCTLine7.s0; 
	MBs[mb_num].coeffs[b4x4_in_mb + 4][inv_zigzag[13]]=DCTLine7.s1; 
	MBs[mb_num].coeffs[b4x4_in_mb + 4][inv_zigzag[14]]=DCTLine7.s2; 
	MBs[mb_num].coeffs[b4x4_in_mb + 4][inv_zigzag[15]]=DCTLine7.s3;
	// block 11
	MBs[mb_num].coeffs[b4x4_in_mb + 5][inv_zigzag[0]]=DCTLine4.s4;
	MBs[mb_num].coeffs[b4x4_in_mb + 5][inv_zigzag[1]]=DCTLine4.s5; 
	MBs[mb_num].coeffs[b4x4_in_mb + 5][inv_zigzag[2]]=DCTLine4.s6; 
	MBs[mb_num].coeffs[b4x4_in_mb + 5][inv_zigzag[3]]=DCTLine4.s7;
	MBs[mb_num].coeffs[b4x4_in_mb + 5][inv_zigzag[4]]=DCTLine5.s4; 
	MBs[mb_num].coeffs[b4x4_in_mb + 5][inv_zigzag[5]]=DCTLine5.s5; 
	MBs[mb_num].coeffs[b4x4_in_mb + 5][inv_zigzag[6]]=DCTLine5.s6; 
	MBs[mb_num].coeffs[b4x4_in_mb + 5][inv_zigzag[7]]=DCTLine5.s7;
	MBs[mb_num].coeffs[b4x4_in_mb + 5][inv_zigzag[8]]=DCTLine6.s4; 
	MBs[mb_num].coeffs[b4x4_in_mb + 5][inv_zigzag[9]]=DCTLine6.s5; 
	MBs[mb_num].coeffs[b4x4_in_mb + 5][inv_zigzag[10]]=DCTLine6.s6; 
	MBs[mb_num].coeffs[b4x4_in_mb + 5][inv_zigzag[11]]=DCTLine6.s7;
	MBs[mb_num].coeffs[b4x4_in_mb + 5][inv_zigzag[12]]=DCTLine7.s4; 
	MBs[mb_num].coeffs[b4x4_in_mb + 5][inv_zigzag[13]]=DCTLine7.s5; 
	MBs[mb_num].coeffs[b4x4_in_mb + 5][inv_zigzag[14]]=DCTLine7.s6; 
	MBs[mb_num].coeffs[b4x4_in_mb + 5][inv_zigzag[15]]=DCTLine7.s7;
	
	// block 00
	DL0 = convert_int4(DCTLine0.s0123);
	DL1 = convert_int4(DCTLine1.s0123);
	DL2 = convert_int4(DCTLine2.s0123);
	DL3 = convert_int4(DCTLine3.s0123);	
	dequant_and_iDCT(&DL0, &DL1, &DL2, &DL3, dc_q, ac_q);
	PL.x = prev_frame[pi]; PL.y = prev_frame[pi+4]; PL.z = prev_frame[pi+8]; PL.w = prev_frame[pi+12]; pi += width_x4*4;
	DL0 += convert_int4(PL);
	PL.x = prev_frame[pi]; PL.y = prev_frame[pi+4]; PL.z = prev_frame[pi+8]; PL.w = prev_frame[pi+12]; pi += width_x4*4;	
	DL1 += convert_int4(PL);
	PL.x = prev_frame[pi]; PL.y = prev_frame[pi+4]; PL.z = prev_frame[pi+8]; PL.w = prev_frame[pi+12]; pi += width_x4*4;
	DL2 += convert_int4(PL);
	PL.x = prev_frame[pi]; PL.y = prev_frame[pi+4]; PL.z = prev_frame[pi+8]; PL.w = prev_frame[pi+12]; pi += width_x4*4;
	DL3 += convert_int4(PL);
	*(__global uchar4*)(&recon_frame[ci]) = (convert_uchar4_sat(DL0)); ci+=width;
	*(__global uchar4*)(&recon_frame[ci]) = (convert_uchar4_sat(DL1)); ci+=width;
	*(__global uchar4*)(&recon_frame[ci]) = (convert_uchar4_sat(DL2)); ci+=width;
	*(__global uchar4*)(&recon_frame[ci]) = (convert_uchar4_sat(DL3)); ci+=width;
	ci-=(width*4); pi -= (width_x4*16);
	// block 01
	ci+=4; pi += 16;
	DL0 = convert_int4(DCTLine0.s4567);
	DL1 = convert_int4(DCTLine1.s4567);
	DL2 = convert_int4(DCTLine2.s4567);
	DL3 = convert_int4(DCTLine3.s4567);	
	dequant_and_iDCT(&DL0, &DL1, &DL2, &DL3, dc_q, ac_q);
	PL.x = prev_frame[pi]; PL.y = prev_frame[pi+4]; PL.z = prev_frame[pi+8]; PL.w = prev_frame[pi+12]; pi += width_x4*4;
	DL0 += convert_int4(PL);
	PL.x = prev_frame[pi]; PL.y = prev_frame[pi+4]; PL.z = prev_frame[pi+8]; PL.w = prev_frame[pi+12]; pi += width_x4*4;
	DL1 += convert_int4(PL);
	PL.x = prev_frame[pi]; PL.y = prev_frame[pi+4]; PL.z = prev_frame[pi+8]; PL.w = prev_frame[pi+12]; pi += width_x4*4;
	DL2 += convert_int4(PL);
	PL.x = prev_frame[pi]; PL.y = prev_frame[pi+4]; PL.z = prev_frame[pi+8]; PL.w = prev_frame[pi+12]; pi += width_x4*4;
	DL3 += convert_int4(PL);
	vstore4(convert_uchar4_sat(DL0), 0 , recon_frame + ci); ci+=width;
	*(__global uchar4*)(&recon_frame[ci]) = (convert_uchar4_sat(DL1)); ci+=width;
	*(__global uchar4*)(&recon_frame[ci]) = (convert_uchar4_sat(DL2)); ci+=width;
	*(__global uchar4*)(&recon_frame[ci]) = (convert_uchar4_sat(DL3)); ci+=width;
	ci-=4; pi -= 16;
	// block 10
	DL0 = convert_int4(DCTLine4.s0123);
	DL1 = convert_int4(DCTLine5.s0123);
	DL2 = convert_int4(DCTLine6.s0123);
	DL3 = convert_int4(DCTLine7.s0123);	
	dequant_and_iDCT(&DL0, &DL1, &DL2, &DL3, dc_q, ac_q);
	PL.x = prev_frame[pi]; PL.y = prev_frame[pi+4]; PL.z = prev_frame[pi+8]; PL.w = prev_frame[pi+12]; pi += width_x4*4;
	DL0 += convert_int4(PL);
	PL.x = prev_frame[pi]; PL.y = prev_frame[pi+4]; PL.z = prev_frame[pi+8]; PL.w = prev_frame[pi+12]; pi += width_x4*4;
	DL1 += convert_int4(PL);
	PL.x = prev_frame[pi]; PL.y = prev_frame[pi+4]; PL.z = prev_frame[pi+8]; PL.w = prev_frame[pi+12]; pi += width_x4*4;
	DL2 += convert_int4(PL);
	PL.x = prev_frame[pi]; PL.y = prev_frame[pi+4]; PL.z = prev_frame[pi+8]; PL.w = prev_frame[pi+12]; pi += width_x4*4;
	DL3 += convert_int4(PL);
	*(__global uchar4*)(&recon_frame[ci]) = (convert_uchar4_sat(DL0)); ci+=width;
	*(__global uchar4*)(&recon_frame[ci]) = (convert_uchar4_sat(DL1)); ci+=width;
	*(__global uchar4*)(&recon_frame[ci]) = (convert_uchar4_sat(DL2)); ci+=width;
	*(__global uchar4*)(&recon_frame[ci]) = (convert_uchar4_sat(DL3)); ci+=width;
	ci-=(width*4); pi -= (width_x4*16);
	// block 01
	ci+=4; pi += 16;
	DL0 = convert_int4(DCTLine4.s4567);
	DL1 = convert_int4(DCTLine5.s4567);
	DL2 = convert_int4(DCTLine6.s4567);
	DL3 = convert_int4(DCTLine7.s4567);	
	dequant_and_iDCT(&DL0, &DL1, &DL2, &DL3, dc_q, ac_q);
	PL.x = prev_frame[pi]; PL.y = prev_frame[pi+4]; PL.z = prev_frame[pi+8]; PL.w = prev_frame[pi+12]; pi += width_x4*4;
	DL0 += convert_int4(PL);
	PL.x = prev_frame[pi]; PL.y = prev_frame[pi+4]; PL.z = prev_frame[pi+8]; PL.w = prev_frame[pi+12]; pi += width_x4*4;
	DL1 += convert_int4(PL);
	PL.x = prev_frame[pi]; PL.y = prev_frame[pi+4]; PL.z = prev_frame[pi+8]; PL.w = prev_frame[pi+12]; pi += width_x4*4;
	DL2 += convert_int4(PL);
	PL.x = prev_frame[pi]; PL.y = prev_frame[pi+4]; PL.z = prev_frame[pi+8]; PL.w = prev_frame[pi+12]; pi += width_x4*4;
	DL3 += convert_int4(PL);
	*(__global uchar4*)(&recon_frame[ci]) = (convert_uchar4_sat(DL0)); ci+=width;
	*(__global uchar4*)(&recon_frame[ci]) = (convert_uchar4_sat(DL1)); ci+=width;
	*(__global uchar4*)(&recon_frame[ci]) = (convert_uchar4_sat(DL2)); ci+=width;
	*(__global uchar4*)(&recon_frame[ci]) = (convert_uchar4_sat(DL3)); 

	return;
}

__kernel void chroma_transform( 	__global uchar *const current_frame, //0 
									__global uchar *const prev_frame, //1 
									__global uchar *const recon_frame, //2 
									__global macroblock *const MBs, //3 
									const signed int chroma_width, //4 
									const signed int chroma_height, //5 
									__constant segment_data *const SD, //6
									const int block_place) //7
{  
	__private int chroma_block_num;
	__private int chroma_width_x8 = chroma_width*8;
	__private int cx, cy, px, py;
	__private int mb_num, block_in_mb,dc_q,ac_q,qi;
	
	chroma_block_num = get_global_id(0); 

	cx = (chroma_block_num % (chroma_width/4))*4;
	cy = (chroma_block_num / (chroma_width/4))*4; 

	mb_num = (cy/8)*(chroma_width/8) + (cx/8);
	block_in_mb = ((cy/4)%2)*2 + ((cx/4)%2);
	if (MBs[mb_num].reference_frame != LAST) return;
	
	qi = MBs[mb_num].segment_id;
	qi = SD[qi].y_ac_i;
	dc_q = SD[0].uv_dc_idelta; dc_q += qi;
	ac_q = SD[0].uv_ac_idelta; ac_q += qi;
	dc_q = select(dc_q,0,dc_q<0); dc_q = select(dc_q,127,dc_q>127);
	ac_q = select(ac_q,0,ac_q<0); ac_q = select(ac_q,127,ac_q>127);
	
	dc_q = vp8_dc_qlookup[dc_q];
	ac_q = vp8_ac_qlookup[ac_q];
	dc_q = select(dc_q,132,dc_q>132);

	__private int ci00, ci10, ci20, ci30;
	ci00 = cy*chroma_width + cx; 
	ci10 = ci00+chroma_width; 
	ci20 = ci10+chroma_width; 
	ci30 = ci20+chroma_width; 

	__private uchar4 CurrentLine0, CurrentLine1, CurrentLine2, CurrentLine3;
	CurrentLine0 = vload4(0, current_frame + ci00);
	CurrentLine1 = vload4(0, current_frame + ci10);
	CurrentLine2 = vload4(0, current_frame + ci20);
	CurrentLine3 = vload4(0, current_frame + ci30);
	
	__private int vector_x, vector_y;
		
	vector_x = MBs[mb_num].vector_x[block_in_mb];
	vector_y = MBs[mb_num].vector_y[block_in_mb];	
	
	block_in_mb += block_place; // 16 for U, 20 for V

	__private uchar4 PredictorLine0, PredictorLine1, 
					PredictorLine2, PredictorLine3; 

	px = (cx*8) + vector_x;
	py = (cy*8) + vector_y;
	
	PredictorLine0.x = prev_frame[py*chroma_width_x8 + px]; 
	PredictorLine0.y = prev_frame[py*chroma_width_x8 + px + 8];
	PredictorLine0.z = prev_frame[py*chroma_width_x8 + px + 16];
	PredictorLine0.w = prev_frame[py*chroma_width_x8 + px + 24];
	PredictorLine1.x = prev_frame[(py + 8)*chroma_width_x8 + px];
	PredictorLine1.y = prev_frame[(py + 8)*chroma_width_x8 + px + 8];
	PredictorLine1.z = prev_frame[(py + 8)*chroma_width_x8 + px + 16];
	PredictorLine1.w = prev_frame[(py + 8)*chroma_width_x8 + px + 24];
	PredictorLine2.x = prev_frame[(py + 16)*chroma_width_x8 + px]; 
	PredictorLine2.y = prev_frame[(py + 16)*chroma_width_x8 + px + 8];
	PredictorLine2.z = prev_frame[(py + 16)*chroma_width_x8 + px + 16];
	PredictorLine2.w = prev_frame[(py + 16)*chroma_width_x8 + px + 24];
	PredictorLine3.x = prev_frame[(py + 24)*chroma_width_x8 + px]; 
	PredictorLine3.y = prev_frame[(py + 24)*chroma_width_x8 + px + 8];
	PredictorLine3.z = prev_frame[(py + 24)*chroma_width_x8 + px + 16];
	PredictorLine3.w = prev_frame[(py + 24)*chroma_width_x8 + px + 24];
	
	__private int4 DL0, DL1, DL2, DL3;
	
	DL0 = convert_int4(CurrentLine0) - convert_int4(PredictorLine0);
	DL1 = convert_int4(CurrentLine1) - convert_int4(PredictorLine1);
	DL2 = convert_int4(CurrentLine2) - convert_int4(PredictorLine2);
	DL3 = convert_int4(CurrentLine3) - convert_int4(PredictorLine3);
	
	DCT_and_quant(&DL0, &DL1, &DL2, &DL3, dc_q, ac_q);
	
	const int inv_zigzag[16] = { 0, 1, 5, 6, 2, 4, 7, 12, 3,  8, 11, 13, 9, 10, 14, 15 };
	MBs[mb_num].coeffs[block_in_mb][inv_zigzag[0]]=(short)DL0.x;
	MBs[mb_num].coeffs[block_in_mb][inv_zigzag[1]]=(short)DL0.y;
	MBs[mb_num].coeffs[block_in_mb][inv_zigzag[2]]=(short)DL0.z;
	MBs[mb_num].coeffs[block_in_mb][inv_zigzag[3]]=(short)DL0.w;
	MBs[mb_num].coeffs[block_in_mb][inv_zigzag[4]]=(short)DL1.x;
	MBs[mb_num].coeffs[block_in_mb][inv_zigzag[5]]=(short)DL1.y;
	MBs[mb_num].coeffs[block_in_mb][inv_zigzag[6]]=(short)DL1.z;
	MBs[mb_num].coeffs[block_in_mb][inv_zigzag[7]]=(short)DL1.w;
	MBs[mb_num].coeffs[block_in_mb][inv_zigzag[8]]=(short)DL2.x;
	MBs[mb_num].coeffs[block_in_mb][inv_zigzag[9]]=(short)DL2.y;
	MBs[mb_num].coeffs[block_in_mb][inv_zigzag[10]]=(short)DL2.z;
	MBs[mb_num].coeffs[block_in_mb][inv_zigzag[11]]=(short)DL2.w;
	MBs[mb_num].coeffs[block_in_mb][inv_zigzag[12]]=(short)DL3.x;
	MBs[mb_num].coeffs[block_in_mb][inv_zigzag[13]]=(short)DL3.y;
	MBs[mb_num].coeffs[block_in_mb][inv_zigzag[14]]=(short)DL3.z;
	MBs[mb_num].coeffs[block_in_mb][inv_zigzag[15]]=(short)DL3.w;
	
	dequant_and_iDCT(&DL0, &DL1, &DL2, &DL3, dc_q, ac_q);
	
	DL0 += convert_int4(PredictorLine0);
	DL1 += convert_int4(PredictorLine1);
	DL2 += convert_int4(PredictorLine2);
	DL3 += convert_int4(PredictorLine3);
	
	CurrentLine0 = (convert_uchar4_sat(DL0));
	CurrentLine1 = (convert_uchar4_sat(DL1));
	CurrentLine2 = (convert_uchar4_sat(DL2));
	CurrentLine3 = (convert_uchar4_sat(DL3));
	vstore4(CurrentLine0, 0 , recon_frame + ci00);
	vstore4(CurrentLine1, 0 , recon_frame + ci10);
	vstore4(CurrentLine2, 0 , recon_frame + ci20);
	vstore4(CurrentLine3, 0 , recon_frame + ci30);
	
	return;
}

__kernel void prepare_filter_mask(__global macroblock *const MBs,
								__global int *const mb_mask)
{
	__private int mb_num, b_num, i, mask, coeffs, split_mode;
	
	mb_num = get_global_id(0);
	mask = 0; coeffs = 0; split_mode = MBs[mb_num].parts;
	
	// divide loops into several to enable unlooping for compiler
	for (b_num = 0; b_num < 16; ++b_num) {
		for (i = 1; i < 16; ++i) {
			coeffs += (int)abs(MBs[mb_num].coeffs[b_num][i]);
		}
	}
	for (b_num = 16; b_num < 24; ++b_num) {
		for (i = 0; i < 16; ++i) {
			coeffs += (int)abs(MBs[mb_num].coeffs[b_num][i]);
		}
	}
	if (split_mode == are16x16) {
		for (i = 0; i < 16; ++i) {
			coeffs += (int)abs(MBs[mb_num].coeffs[24][i]);
		}
	}
	else {
		for (b_num = 0; b_num < 16; ++b_num) {
			coeffs += (int)abs(MBs[mb_num].coeffs[b_num][0]);
		}
	}
			
	MBs[mb_num].non_zero_coeffs = coeffs;
	mask = ((split_mode != are16x16) || (coeffs > 0)) ? -1 : 0;
	mb_mask[mb_num] = mask;
	
	return;
}

__kernel void normal_loop_filter_MBH(__global uchar * const frame, //0
									const int width, //1
									__constant const segment_data *const SD, //2
									__global macroblock *const MB, //3
									const int mb_size, //4
									const int stage, //5
									__global int *const mb_mask) //6
{
	// here we write R, then rewrite it as L
	// may be reduced, but different mb_size and mask should be accounted
	int x, y, i, mb_col, mb_row, mb_num;
	uchar4 L, R;
	int4 P, Q;
	int a, b, w, mask, hev, subblock_mask;
	int interior_limit, mbedge_limit, sub_bedge_limit, hev_threshold, id;

	mb_row = get_global_id(0)/mb_size;
	mb_col = stage - (2*mb_row);

	if (mb_col < 0) return;
	if (mb_col >= (width/mb_size)) return;
	
	mb_num = mb_row*(width/mb_size)+mb_col;
	id = MB[mb_num].segment_id;
	if (SD[id].loop_filter_level == 0) return;
	interior_limit = SD[id].interior_limit;
	mbedge_limit = SD[id].mbedge_limit;
	sub_bedge_limit = SD[id].sub_bedge_limit;
	hev_threshold = SD[id].hev_threshold;
	subblock_mask = mb_mask[mb_num];
	
	y = get_global_id(0);
	x = mb_col * mb_size;
	
	i = y*width + x;
	
	R = vload4(0, frame + i);
	if ( x > 0) 
	{
		L = vload4(0, frame + i - 4);
		P = convert_int4(L) - 128;
		Q = convert_int4(R) - 128;
		mask = ((int)abs(P.x - P.y) > interior_limit);
		mask |= ((int)abs(P.y - P.z) > interior_limit);
		mask |= ((int)abs(P.z - P.w) > interior_limit);
		mask |= ((int)abs(Q.y - Q.x) > interior_limit);
		mask |= ((int)abs(Q.z - Q.y) > interior_limit);
		mask |= ((int)abs(Q.w - Q.z) > interior_limit);
		mask |= (((int)abs(P.w - Q.x) * 2 + (int)abs(P.z - Q.y) / 2) > mbedge_limit);
		mask -= 1; //sets 0 to  1111...111 for enabling filtering
		hev = ((int)abs(P.z - P.w) > hev_threshold);
		hev |= ((int)abs(Q.y - Q.x) > hev_threshold);
		hev *= -1; //OpenCL for scalar gives TRUE == +1
		//w = clamp128(clamp128(p1 - q1) + 3*(q0 - p0));
		w = P.z - Q.y;
		w = select(w,-128,w<-128);
		w = select(w,127,w>127);
		w = mad24(Q.x-P.w, 3, w);
		w = select(w,-128,w<-128);
		w = select(w,127,w>127);
		w &= mask;
		a = w & hev;
		// b = clamp128(a+3) >> 3
		b = a + 3;
		b = select(b,-128,b<-128);
		b = select(b,127,b>127);
		b >>= 3;
		// a = clamp128(a+4) >> 3
		a = a + 4;
		a = select(a,-128,a<-128);
		a = select(a,127,a>127);
		a >>= 3;
		Q.x -= a; P.w += b;
		w &= ~hev;
		//a = clamp128((27*w + 63) >> 7);
		a = mad24(w, 27, 63) >> 7;
		a = select(a,-128,a<-128);
		a = select(a,127,a>127);
		Q.x -= a; P.w += a;
		//a = clamp128((18*w + 63) >> 7);
		a = mad24(w, 18, 63) >> 7;
		a = select(a,-128,a<-128);
		a = select(a,127,a>127);
		Q.y -= a; P.z += a;
		//a = clamp128((9*w + 63) >> 7);
		a = mad24(w, 9, 63) >> 7;
		a = select(a,-128,a<-128);
		a = select(a,127,a>127);	
		Q.z -= a; P.y += a;
		R = convert_uchar4_sat(Q + 128);
		L = convert_uchar4_sat(P + 128);
		vstore4(L, 0, frame + i - 4);
		//vstore4(R, 0, frame + i); it will be stored as L later
	}
	if (subblock_mask == 0) {
		vstore4(R, 0, frame + i);
		return;
	}
	
	// and 3 more times for edges between blocks in MB
	L = R;
	i += 4;
	R = vload4(0, frame + i);
	P = convert_int4(L) - 128;
	Q = convert_int4(R) - 128;
	mask = ((int)abs(P.x - P.y) > interior_limit);
	mask |= ((int)abs(P.y - P.z) > interior_limit);
	mask |= ((int)abs(P.z - P.w) > interior_limit);
	mask |= ((int)abs(Q.y - Q.x) > interior_limit);
	mask |= ((int)abs(Q.z - Q.y) > interior_limit);
	mask |= ((int)abs(Q.w - Q.z) > interior_limit);
	mask |= (((int)abs(P.w - Q.x) * 2 + (int)abs(P.z - Q.y) / 2) > sub_bedge_limit);
	mask -= 1;
	hev = ((int)abs(P.z - P.w) > hev_threshold);
	hev |= ((int)abs(Q.y - Q.x) > hev_threshold);
	hev *= -1;
	//a = clamp128((use_outer_taps? clamp128(p1 - q1) : 0) + 3*(q0 - p0));
	a = P.z - Q.y; 
	a = select(a,-128,a<-128);
	a = select(a,127,a>127);
	a &= hev;
	a = mad24((Q.x - P.w), 3, a);
	a = select(a,-128,a<-128);
	a = select(a,127,a>127);
	a &= mask;
	// b = clamp128(a+3) >> 3
	b = a + 3;
	b = select(b,-128,b<-128);
	b = select(b,127,b>127);
	b >>= 3;
	// a = clamp128(a+4) >> 3
	a = a + 4;
	a = select(a,-128,a<-128);
	a = select(a,127,a>127);
	a >>= 3;
	Q.x -= a; P.w += b;
	a = (a + 1) >> 1;
	a &= ~hev;
	Q.y -= a; P.z += a; 
	R = convert_uchar4_sat(Q + 128);
	L = convert_uchar4_sat(P + 128);
	vstore4(L, 0, frame + i - 4);
	//vstore4(R, 0, frame + i);
	
	if (mb_size < 16) { // we were doing chroma
		vstore4(R, 0, frame + i);
		return;
	}
	
	L = R;
	i += 4;
	R = vload4(0, frame + i);
	P = convert_int4(L) - 128;
	Q = convert_int4(R) - 128;
	mask = ((int)abs(P.x - P.y) > interior_limit);
	mask |= ((int)abs(P.y - P.z) > interior_limit);
	mask |= ((int)abs(P.z - P.w) > interior_limit);
	mask |= ((int)abs(Q.y - Q.x) > interior_limit);
	mask |= ((int)abs(Q.z - Q.y) > interior_limit);
	mask |= ((int)abs(Q.w - Q.z) > interior_limit);
	mask |= (((int)abs(P.w - Q.x) * 2 + (int)abs(P.z - Q.y) / 2) > sub_bedge_limit);
	mask -= 1;
	hev = ((int)abs(P.z - P.w) > hev_threshold);
	hev |= ((int)abs(Q.y - Q.x) > hev_threshold);
	hev *= -1; 
	//a = clamp128((use_outer_taps? clamp128(p1 - q1) : 0) + 3*(q0 - p0));
	a = P.z - Q.y; 
	a = select(a,-128,a<-128);
	a = select(a,127,a>127);
	a &= hev;
	a = mad24((Q.x - P.w), 3, a);
	a = select(a,-128,a<-128);
	a = select(a,127,a>127);
	a &= mask;
	// b = clamp128(a+3) >> 3
	b = a + 3;
	b = select(b,-128,b<-128);
	b = select(b,127,b>127);
	b >>= 3;
	// a = clamp128(a+4) >> 3
	a = a + 4;
	a = select(a,-128,a<-128);
	a = select(a,127,a>127);
	a >>= 3;
	Q.x -= a; P.w += b;
	a = (a + 1) >> 1;
	a &= ~hev;
	Q.y -= a; P.z += a; 
	R = convert_uchar4_sat(Q + 128);
	L = convert_uchar4_sat(P + 128);
	vstore4(L, 0, frame + i - 4);
	//vstore4(R, 0, frame + i);
	
	L = R;
	i += 4;
	R = vload4(0, frame + i);
	P = convert_int4(L) - 128;
	Q = convert_int4(R) - 128;
	mask = ((int)abs(P.x - P.y) > interior_limit);
	mask |= ((int)abs(P.y - P.z) > interior_limit);
	mask |= ((int)abs(P.z - P.w) > interior_limit);
	mask |= ((int)abs(Q.y - Q.x) > interior_limit);
	mask |= ((int)abs(Q.z - Q.y) > interior_limit);
	mask |= ((int)abs(Q.w - Q.z) > interior_limit);
	mask |= (((int)abs(P.w - Q.x) * 2 + (int)abs(P.z - Q.y) / 2) > sub_bedge_limit);
	mask -= 1;
	hev = ((int)abs(P.z - P.w) > hev_threshold);
	hev |= ((int)abs(Q.y - Q.x) > hev_threshold);
	hev *= -1; 
	//a = clamp128((use_outer_taps? clamp128(p1 - q1) : 0) + 3*(q0 - p0));
	a = P.z - Q.y; 
	a = select(a,-128,a<-128);
	a = select(a,127,a>127);
	a &= hev;
	a = mad24((Q.x - P.w), 3, a);
	a = select(a,-128,a<-128);
	a = select(a,127,a>127);
	a &= mask;
	// b = clamp128(a+3) >> 3
	b = a + 3;
	b = select(b,-128,b<-128);
	b = select(b,127,b>127);
	b >>= 3;
	// a = clamp128(a+4) >> 3
	a = a + 4;
	a = select(a,-128,a<-128);
	a = select(a,127,a>127);
	a >>= 3;
	Q.x -= a; P.w += b;
	a = (a + 1) >> 1;
	a &= ~hev;
	Q.y -= a; P.z += a; 
	R = convert_uchar4_sat(Q + 128);
	L = convert_uchar4_sat(P + 128);
	vstore4(L, 0, frame + i - 4);
	vstore4(R, 0, frame + i);
	
	return;	
}

__kernel void normal_loop_filter_MBV(__global uchar * const frame, //0
									const int width, //1
									__constant segment_data *const SD, //2
									__global macroblock *const MB, //3
									const int mb_size, //4
									const int stage, //5
									__global int *const mb_mask) //6
{
	int x, y, i, mb_row, mb_col, mb_num;
	uchar4 ucP3, ucP2, ucP1, ucP0, /*edge*/ ucQ0, ucQ1, ucQ2, ucQ3;
	int4 p3, p2, p1, p0, /*edge*/ q0, q1, q2, q3;
	int4 a, b, w, mask, hev;
	int interior_limit, mbedge_limit, sub_bedge_limit, hev_threshold, id, subblock_mask;

	mb_row = get_global_id(0)/(mb_size/4);
	mb_col = stage - (2*mb_row);
	
	if (mb_col < 0) return;
	if (mb_col >= (width/mb_size)) return;
	
	mb_num = mb_row*(width/mb_size)+mb_col;
	id = MB[mb_num].segment_id;
	if (SD[id].loop_filter_level == 0) return;
	interior_limit = SD[id].interior_limit;
	mbedge_limit = SD[id].mbedge_limit;
	sub_bedge_limit = SD[id].sub_bedge_limit;
	hev_threshold = SD[id].hev_threshold;
	
	subblock_mask = mb_mask[mb_num];
	
	//printf((__constant char*)"%d",subblock_mask);
	
	y = (get_global_id(0)/(mb_size/4))*mb_size;
	x = mad24((int)mb_col, mb_size, (int)(get_global_id(0)%(mb_size/4))*4);
	
	i = y*width + x;
	
	ucQ0 = vload4(0, frame + i); i+= width;
	ucQ1 = vload4(0, frame + i); i+= width;
	ucQ2 = vload4(0, frame + i); i+= width;
	ucQ3 = vload4(0, frame + i); i+= width;
	i -= width*4;
	
	if (y > 0) 
	{
		i -= width*4;
		ucP3 = vload4(0, frame + i); i+= width;
		ucP2 = vload4(0, frame + i); i+= width;
		ucP1 = vload4(0, frame + i); i+= width;
		ucP0 = vload4(0, frame + i); i+= width;
		p3 = convert_int4(ucP3) - 128;
		p2 = convert_int4(ucP2) - 128;
		p1 = convert_int4(ucP1) - 128;
		p0 = convert_int4(ucP0) - 128;
		q0 = convert_int4(ucQ0) - 128;
		q1 = convert_int4(ucQ1) - 128;
		q2 = convert_int4(ucQ2) - 128;
		q3 = convert_int4(ucQ3) - 128;
		mask = (convert_int4(abs(p3 - p2)) > interior_limit);
		mask |= (convert_int4(abs(p2 - p1)) > interior_limit);
		mask |= (convert_int4(abs(p1 - p0)) > interior_limit);
		mask |= (convert_int4(abs(q1 - q0)) > interior_limit);
		mask |= (convert_int4(abs(q2 - q1)) > interior_limit);
		mask |= (convert_int4(abs(q3 - q2)) > interior_limit);
		mask |= ((convert_int4(abs(p0 - q0)) * 2 + convert_int4(abs(p1 - q1)) / 2)  > mbedge_limit);
		mask = ~mask; // for vectors in OpenCL TRUE means -1 (all bits set)
		hev = (convert_int4(abs(p1 - p0)) > hev_threshold);
		hev |= (convert_int4(abs(q1 - q0)) > hev_threshold);
		//w = clamp128(clamp128(p1 - q1) + 3*(q0 - p0));
		w = p1 - q1;
		w = select(w,-128,w<-128);
		w = select(w,127,w>127);
		w = mad24(q0-p0, 3, w);
		w = select(w,-128,w<-128);
		w = select(w,127,w>127);
		w &= mask;
		a = w & hev;
		// b = clamp128(a+3) >> 3
		b = a + 3;
		b = select(b,-128,b<-128);
		b = select(b,127,b>127);
		b >>= 3;
		// a = clamp128(a+4) >> 3
		a = a + 4;
		a = select(a,-128,a<-128);
		a = select(a,127,a>127);
		a >>= 3;
		q0 -= a; p0 += b;
		w &= ~hev;		
		//a = clamp128((27*w + 63) >> 7);
		a = mad24(w, 27, 63) >> 7;
		a = select(a,-128,a<-128);
		a = select(a,127,a>127);
		q0 -= a; p0 += a;
		//a = clamp128((18*w + 63) >> 7);
		a = mad24(w, 18, 63) >> 7;
		a = select(a,-128,a<-128);
		a = select(a,127,a>127);
		q1 -= a; p1 += a;
		//a = clamp128((9*w + 63) >> 7);
		a = mad24(w, 9, 63) >> 7;
		a = select(a,-128,a<-128);
		a = select(a,127,a>127);
		q2 -= a; p2 += a;
		ucP2 = convert_uchar4_sat(p2 + 128);
		ucP1 = convert_uchar4_sat(p1 + 128);
		ucP0 = convert_uchar4_sat(p0 + 128);
		ucQ0 = convert_uchar4_sat(q0 + 128);
		ucQ1 = convert_uchar4_sat(q1 + 128);
		ucQ2 = convert_uchar4_sat(q2 + 128);		
		
		vstore4(ucP2, 0, frame + i - 3*width);
		vstore4(ucP1, 0, frame + i - 2*width);
		vstore4(ucP0, 0, frame + i - width);
		vstore4(ucQ0, 0, frame + i);
		vstore4(ucQ1, 0, frame + i + width);
		vstore4(ucQ2, 0, frame + i + 2*width);
	}
	
	if (subblock_mask == 0) return;
	
	// and 3 more times for edges between blocks in MB
	i += width*4;
	ucP3 = ucQ0;
	ucP2 = ucQ1;
	ucP1 = ucQ2;
	ucP0 = ucQ3;
	ucQ0 = vload4(0, frame + i); i+= width;
	ucQ1 = vload4(0, frame + i); i+= width;
	ucQ2 = vload4(0, frame + i); i+= width;
	ucQ3 = vload4(0, frame + i); i+= width;
	i -= width*4;	
	p3 = convert_int4(ucP3) - 128;
	p2 = convert_int4(ucP2) - 128;
	p1 = convert_int4(ucP1) - 128;
	p0 = convert_int4(ucP0) - 128;
	q0 = convert_int4(ucQ0) - 128;
	q1 = convert_int4(ucQ1) - 128;
	q2 = convert_int4(ucQ2) - 128;
	q3 = convert_int4(ucQ3) - 128;
	mask = (convert_int4(abs(p3 - p2)) > interior_limit);
	mask |= (convert_int4(abs(p2 - p1)) > interior_limit);
	mask |= (convert_int4(abs(p1 - p0)) > interior_limit);
	mask |= (convert_int4(abs(q1 - q0)) > interior_limit);
	mask |= (convert_int4(abs(q2 - q1)) > interior_limit);
	mask |= (convert_int4(abs(q3 - q2)) > interior_limit);
	mask |= ((convert_int4(abs(p0 - q0)) * 2 + convert_int4(abs(p1 - q1)) / 2)  > sub_bedge_limit);
	mask = ~mask; // for vectors in OpenCL TRUE means -1 (all bits set)
	hev = (convert_int4(abs(p1 - p0)) > hev_threshold);
	hev |= (convert_int4(abs(q1 - q0)) > hev_threshold);
	//a = clamp128((use_outer_taps? clamp128(p1 - q1) : 0) + 3*(q0 - p0));
	a = p1 - q1;
	a = select(a,-128,a<-128);
	a = select(a,127,a>127);
	a &= hev;
	a = mad24((q0 - p0), 3, a);
	a = select(a,-128,a<-128);
	a = select(a,127,a>127);
	a &= mask;
	// b = clamp128(a+3) >> 3
	b = a + 3;
	b = select(b,-128,b<-128);
	b = select(b,127,b>127);
	b >>= 3;
	// a = clamp128(a+4) >> 3
	a = a + 4;
	a = select(a,-128,a<-128);
	a = select(a,127,a>127);
	a >>= 3;
	q0 -= a; p0 += b;
	a = (a + 1) >> 1;
	a &= ~hev;
	q1 -= a; p1 += a;
	ucP1 = convert_uchar4_sat(p1 + 128);
	ucP0 = convert_uchar4_sat(p0 + 128);
	ucQ0 = convert_uchar4_sat(q0 + 128);
	ucQ1 = convert_uchar4_sat(q1 + 128);
	vstore4(ucP1, 0, frame + i - 2*width);
	vstore4(ucP0, 0, frame + i - width);
	vstore4(ucQ0, 0, frame + i);
	vstore4(ucQ1, 0, frame + i + width);

	if (mb_size < 16) return; //we were doing chroma
	
	i += width*4;
	ucP3 = ucQ0;
	ucP2 = ucQ1;
	ucP1 = ucQ2;
	ucP0 = ucQ3;
	ucQ0 = vload4(0, frame + i); i+= width;
	ucQ1 = vload4(0, frame + i); i+= width;
	ucQ2 = vload4(0, frame + i); i+= width;
	ucQ3 = vload4(0, frame + i); i+= width;
	i -= width*4;	
	p3 = convert_int4(ucP3) - 128;
	p2 = convert_int4(ucP2) - 128;
	p1 = convert_int4(ucP1) - 128;
	p0 = convert_int4(ucP0) - 128;
	q0 = convert_int4(ucQ0) - 128;
	q1 = convert_int4(ucQ1) - 128;
	q2 = convert_int4(ucQ2) - 128;
	q3 = convert_int4(ucQ3) - 128;
	mask = (convert_int4(abs(p3 - p2)) > interior_limit);
	mask |= (convert_int4(abs(p2 - p1)) > interior_limit);
	mask |= (convert_int4(abs(p1 - p0)) > interior_limit);
	mask |= (convert_int4(abs(q1 - q0)) > interior_limit);
	mask |= (convert_int4(abs(q2 - q1)) > interior_limit);
	mask |= (convert_int4(abs(q3 - q2)) > interior_limit);
	mask |= ((convert_int4(abs(p0 - q0)) * 2 + convert_int4(abs(p1 - q1)) / 2)  > sub_bedge_limit);
	mask = ~mask; // for vectors in OpenCL TRUE means -1 (all bits set)
	hev = (convert_int4(abs(p1 - p0)) > hev_threshold);
	hev |= (convert_int4(abs(q1 - q0)) > hev_threshold);
	//a = clamp128((use_outer_taps? clamp128(p1 - q1) : 0) + 3*(q0 - p0));
	a = p1 - q1;
	a = select(a,-128,a<-128);
	a = select(a,127,a>127);
	a &= hev;
	a = mad24((q0 - p0), 3, a);
	a = select(a,-128,a<-128);
	a = select(a,127,a>127);
	a &= mask;
	// b = clamp128(a+3) >> 3
	b = a + 3;
	b = select(b,-128,b<-128);
	b = select(b,127,b>127);
	b >>= 3;
	// a = clamp128(a+4) >> 3
	a = a + 4;
	a = select(a,-128,a<-128);
	a = select(a,127,a>127);
	a >>= 3;
	q0 -= a; p0 += b;
	a = (a + 1) >> 1;
	a &= ~hev;
	q1 -= a; p1 += a;
	ucP1 = convert_uchar4_sat(p1 + 128);
	ucP0 = convert_uchar4_sat(p0 + 128);
	ucQ0 = convert_uchar4_sat(q0 + 128);
	ucQ1 = convert_uchar4_sat(q1 + 128);
	vstore4(ucP1, 0, frame + i - 2*width);
	vstore4(ucP0, 0, frame + i - width);
	vstore4(ucQ0, 0, frame + i);
	vstore4(ucQ1, 0, frame + i + width);
	
	i += width*4;
	ucP3 = ucQ0;
	ucP2 = ucQ1;
	ucP1 = ucQ2;
	ucP0 = ucQ3;
	ucQ0 = vload4(0, frame + i); i+= width;
	ucQ1 = vload4(0, frame + i); i+= width;
	ucQ2 = vload4(0, frame + i); i+= width;
	ucQ3 = vload4(0, frame + i); i+= width;
	i -= width*4;	
	p3 = convert_int4(ucP3) - 128;
	p2 = convert_int4(ucP2) - 128;
	p1 = convert_int4(ucP1) - 128;
	p0 = convert_int4(ucP0) - 128;
	q0 = convert_int4(ucQ0) - 128;
	q1 = convert_int4(ucQ1) - 128;
	q2 = convert_int4(ucQ2) - 128;
	q3 = convert_int4(ucQ3) - 128;
	mask = (convert_int4(abs(p3 - p2)) > interior_limit);
	mask |= (convert_int4(abs(p2 - p1)) > interior_limit);
	mask |= (convert_int4(abs(p1 - p0)) > interior_limit);
	mask |= (convert_int4(abs(q1 - q0)) > interior_limit);
	mask |= (convert_int4(abs(q2 - q1)) > interior_limit);
	mask |= (convert_int4(abs(q3 - q2)) > interior_limit);
	mask |= ((convert_int4(abs(p0 - q0)) * 2 + convert_int4(abs(p1 - q1)) / 2)  > sub_bedge_limit);
	mask = ~mask; // for vectors in OpenCL TRUE means -1 (all bits set)
	hev = (convert_int4(abs(p1 - p0)) > hev_threshold);
	hev |= (convert_int4(abs(q1 - q0)) > hev_threshold);
	//a = clamp128((use_outer_taps? clamp128(p1 - q1) : 0) + 3*(q0 - p0));
	a = p1 - q1;
	a = select(a,-128,a<-128);
	a = select(a,127,a>127);
	a &= hev;
	a = mad24((q0 - p0), 3, a);
	a = select(a,-128,a<-128);
	a = select(a,127,a>127);
	a &= mask;
	// b = clamp128(a+3) >> 3
	b = a + 3;
	b = select(b,-128,b<-128);
	b = select(b,127,b>127);
	b >>= 3;
	// a = clamp128(a+4) >> 3
	a = a + 4;
	a = select(a,-128,a<-128);
	a = select(a,127,a>127);
	a >>= 3;
	q0 -= a; p0 += b;
	a = (a + 1) >> 1;
	a &= ~hev;
	q1 -= a; p1 += a;
	ucP1 = convert_uchar4_sat(p1 + 128);
	ucP0 = convert_uchar4_sat(p0 + 128);
	ucQ0 = convert_uchar4_sat(q0 + 128);
	ucQ1 = convert_uchar4_sat(q1 + 128);
	vstore4(ucP1, 0, frame + i - 2*width);
	vstore4(ucP0, 0, frame + i - width);
	vstore4(ucQ0, 0, frame + i);
	vstore4(ucQ1, 0, frame + i + width);
	
	return;	
}

__kernel void luma_interpolate_Hx4_bc( __global uchar *const src_frame, //0
									__global uchar *const dst_frame, //1
									const int width, //2
									const int height) //3
{
	if (get_global_id(0) > (height-1)) return;
	
	//bicubic filter:
	/*
	const int filters [8] [6] = { // indexed by displacement
								{ 0, 0, 128, 0, 0, 0 }, 
								{ 0, -6, 123, 12, -1, 0 }, // 1/8 
								{ 2, -11, 108, 36, -8, 1 }, // 1/4 
								{ 0, -9, 93, 50, -6, 0 }, // 3/8 
								{ 3, -16, 77, 77, -16, 3 }, // 1/2 is symmetric 
								{ 0, -6, 50, 93, -9, 0 }, // 5/8 = reverse of 3/8 
								{ 1, -8, 36, 108, -11, 2 }, // 3/4 = reverse of 1/4 
								{ 0, -1, 12, 123, -6, 0 } // 7/8 = reverse of 1/8 
								}; */

	__private uchar4 M4, R4, L4;
	__private uchar16 M16;
	
	int i, ind, buf;
	int width_x4 = width*4;
	
	
	ind = (get_global_id(0) + 1)*width - 4;
	M4 = vload4(0, src_frame + ind);
	R4.s0 = M4.s3;
	R4.s1 = M4.s3;
	R4.s2 = M4.s3;
	
	for (i = width-4; i >= 4; i -= 4)
	{
		ind = get_global_id(0)*width + i;
		L4 = vload4(0, src_frame + ind - 4);
		
		M16.s0 = M4.s0;
		buf = mad24((int)L4.s2,2,mad24((int)L4.s3,-11,mad24((int)M4.s0,108,mad24((int)M4.s1,36,mad24((int)M4.s2,-8, (int)M4.s3 + 64)))))/128;
		buf = (buf < 0) ? 0 : buf; 
		buf = (buf > 255) ? 255 : buf; 
		M16.s1 = (uchar)buf;
		buf = mad24((int)L4.s2,3,mad24((int)L4.s3,-16,mad24((int)M4.s0,77,mad24((int)M4.s1,77,mad24((int)M4.s2,-16, mad24((int)M4.s3,3,64))))))/128;
		buf = (buf < 0) ? 0 : buf; 
		buf = (buf > 255) ? 255 : buf; 
		M16.s2 = (uchar)buf;
		buf = ((int)L4.s2 + mad24((int)L4.s3,-8,mad24((int)M4.s0,36,mad24((int)M4.s1,108,mad24((int)M4.s2,-11, mad24((int)M4.s3,2,64))))))/128;		
		buf = (buf < 0) ? 0 : buf; 
		buf = (buf > 255) ? 255 : buf; 
		M16.s3 = (uchar)buf;
		
		M16.s4 = M4.s1;
		buf = mad24((int)L4.s3,2,mad24((int)M4.s0,-11,mad24((int)M4.s1,108,mad24((int)M4.s2,36,mad24((int)M4.s3,-8, (int)R4.s0 + 64)))))/128;
		buf = (buf < 0) ? 0 : buf; 
		buf = (buf > 255) ? 255 : buf; 
		M16.s5 = (uchar)buf;
		buf = mad24((int)L4.s3,3,mad24((int)M4.s0,-16,mad24((int)M4.s1,77,mad24((int)M4.s2,77,mad24((int)M4.s3,-16, mad24((int)R4.s0,3,64))))))/128;
		buf = (buf < 0) ? 0 : buf; 
		buf = (buf > 255) ? 255 : buf; 
		M16.s6 = (uchar)buf;
		buf = ((int)L4.s3 + mad24((int)M4.s0,-8,mad24((int)M4.s1,36,mad24((int)M4.s2,108,mad24((int)M4.s3,-11, mad24((int)R4.s0,2,64))))))/128;		
		buf = (buf < 0) ? 0 : buf; 
		buf = (buf > 255) ? 255 : buf; 
		M16.s7 = (uchar)buf;
		
		M16.s8 = M4.s2;
		buf = mad24((int)M4.s0,2,mad24((int)M4.s1,-11,mad24((int)M4.s2,108,mad24((int)M4.s3,36,mad24((int)R4.s0,-8, (int)R4.s1 + 64)))))/128;
		buf = (buf < 0) ? 0 : buf; 
		buf = (buf > 255) ? 255 : buf; 
		M16.s9 = (uchar)buf;
		buf = mad24((int)M4.s0,3,mad24((int)M4.s1,-16,mad24((int)M4.s2,77,mad24((int)M4.s3,77,mad24((int)R4.s0,-16, mad24((int)R4.s1,3,64))))))/128;
		buf = (buf < 0) ? 0 : buf; 
		buf = (buf > 255) ? 255 : buf; 
		M16.sA = (uchar)buf;
		buf = ((int)M4.s0 + mad24((int)M4.s1,-8,mad24((int)M4.s2,36,mad24((int)M4.s3,108,mad24((int)R4.s0,-11, mad24((int)R4.s1,2,64))))))/128;		
		buf = (buf < 0) ? 0 : buf; 
		buf = (buf > 255) ? 255 : buf; 
		M16.sB = (uchar)buf;
		
		M16.sC = M4.s3;
		buf = mad24((int)M4.s1,2,mad24((int)M4.s2,-11,mad24((int)M4.s3,108,mad24((int)R4.s0,36,mad24((int)R4.s1,-8, (int)R4.s2 + 64)))))/128;
		buf = (buf < 0) ? 0 : buf; 
		buf = (buf > 255) ? 255 : buf; 
		M16.sD = (uchar)buf;
		buf = mad24((int)M4.s1,3,mad24((int)M4.s2,-16,mad24((int)M4.s3,77,mad24((int)R4.s0,77,mad24((int)R4.s1,-16, mad24((int)R4.s2,3,64))))))/128;
		buf = (buf < 0) ? 0 : buf; 
		buf = (buf > 255) ? 255 : buf; 
		M16.sE = (uchar)buf;
		buf = ((int)M4.s1 + mad24((int)M4.s2,-8,mad24((int)M4.s3,36,mad24((int)R4.s0,108,mad24((int)R4.s1,-11, mad24((int)R4.s2,2,64))))))/128;		
		buf = (buf < 0) ? 0 : buf; 
		buf = (buf > 255) ? 255 : buf; 
		M16.sF = (uchar)buf;

		ind = get_global_id(0)*width_x4 + (i*4);
		vstore16(M16, 0, dst_frame + ind);
		
		R4 = M4;
		M4 = L4;
	}
	
	L4.s2 = M4.s0; L4.s3 = M4.s0;
		
	M16.s0 = M4.s0;
	buf = mad24((int)L4.s2,2,mad24((int)L4.s3,-11,mad24((int)M4.s0,108,mad24((int)M4.s1,36,mad24((int)M4.s2,-8, (int)M4.s3 + 64)))))/128;
	buf = (buf < 0) ? 0 : buf; 
	buf = (buf > 255) ? 255 : buf; 
	M16.s1 = (uchar)buf;
	buf = mad24((int)L4.s2,3,mad24((int)L4.s3,-16,mad24((int)M4.s0,77,mad24((int)M4.s1,77,mad24((int)M4.s2,-16, mad24((int)M4.s3,3,64))))))/128;
	buf = (buf < 0) ? 0 : buf; 
	buf = (buf > 255) ? 255 : buf; 
	M16.s2 = (uchar)buf;
	buf = ((int)L4.s2 + mad24((int)L4.s3,-8,mad24((int)M4.s0,36,mad24((int)M4.s1,108,mad24((int)M4.s2,-11, mad24((int)M4.s3,2,64))))))/128;		
	buf = (buf < 0) ? 0 : buf; 
	buf = (buf > 255) ? 255 : buf; 
	M16.s3 = (uchar)buf;
		
	M16.s4 = M4.s1;
	buf = mad24((int)L4.s3,2,mad24((int)M4.s0,-11,mad24((int)M4.s1,108,mad24((int)M4.s2,36,mad24((int)M4.s3,-8, (int)R4.s0 + 64)))))/128;
	buf = (buf < 0) ? 0 : buf; 
	buf = (buf > 255) ? 255 : buf; 
	M16.s5 = (uchar)buf;
	buf = mad24((int)L4.s3,3,mad24((int)M4.s0,-16,mad24((int)M4.s1,77,mad24((int)M4.s2,77,mad24((int)M4.s3,-16, mad24((int)R4.s0,3,64))))))/128;
	buf = (buf < 0) ? 0 : buf; 
	buf = (buf > 255) ? 255 : buf; 
	M16.s6 = (uchar)buf;
	buf = ((int)L4.s3 + mad24((int)M4.s0,-8,mad24((int)M4.s1,36,mad24((int)M4.s2,108,mad24((int)M4.s3,-11, mad24((int)R4.s0,2,64))))))/128;		
	buf = (buf < 0) ? 0 : buf; 
	buf = (buf > 255) ? 255 : buf; 
	M16.s7 = (uchar)buf;
		
	M16.s8 = M4.s2;
	buf = mad24((int)M4.s0,2,mad24((int)M4.s1,-11,mad24((int)M4.s2,108,mad24((int)M4.s3,36,mad24((int)R4.s0,-8, (int)R4.s1 + 64)))))/128;
	buf = (buf < 0) ? 0 : buf; 
	buf = (buf > 255) ? 255 : buf; 
	M16.s9 = (uchar)buf;
	buf = mad24((int)M4.s0,3,mad24((int)M4.s1,-16,mad24((int)M4.s2,77,mad24((int)M4.s3,77,mad24((int)R4.s0,-16, mad24((int)R4.s1,3,64))))))/128;
	buf = (buf < 0) ? 0 : buf; 
	buf = (buf > 255) ? 255 : buf; 
	M16.sA = (uchar)buf;
	buf = ((int)M4.s0 + mad24((int)M4.s1,-8,mad24((int)M4.s2,36,mad24((int)M4.s3,108,mad24((int)R4.s0,-11, mad24((int)R4.s1,2,64))))))/128;		
	buf = (buf < 0) ? 0 : buf; 
	buf = (buf > 255) ? 255 : buf; 
	M16.sB = (uchar)buf;
		
	M16.sC = M4.s3;
	buf = mad24((int)M4.s1,2,mad24((int)M4.s2,-11,mad24((int)M4.s3,108,mad24((int)R4.s0,36,mad24((int)R4.s1,-8, (int)R4.s2 + 64)))))/128;
	buf = (buf < 0) ? 0 : buf; 
	buf = (buf > 255) ? 255 : buf; 
	M16.sD = (uchar)buf;
	buf = mad24((int)M4.s1,3,mad24((int)M4.s2,-16,mad24((int)M4.s3,77,mad24((int)R4.s0,77,mad24((int)R4.s1,-16, mad24((int)R4.s2,3,64))))))/128;
	buf = (buf < 0) ? 0 : buf; 
	buf = (buf > 255) ? 255 : buf; 
	M16.sE = (uchar)buf;
	buf = ((int)M4.s1 + mad24((int)M4.s2,-8,mad24((int)M4.s3,36,mad24((int)R4.s0,108,mad24((int)R4.s1,-11, mad24((int)R4.s2,2,64))))))/128;		
	buf = (buf < 0) ? 0 : buf; 
	buf = (buf > 255) ? 255 : buf; 
	M16.sF = (uchar)buf;

	ind = get_global_id(0)*width_x4;
	vstore16(M16, 0, dst_frame + ind);

	return;	
}

__kernel void luma_interpolate_Vx4_bc( __global uchar *const frame, //0
									const int width, //1
									const int height) //2
{
	__private uchar4 A0, A1, M0, M1, M2, M3, U0, U1, U2;
	__private int4 buf;

	int width_x4 = width*4;
	if (4*get_global_id(0) > (width_x4-1)) return;
	int i, ind;
	
	ind = (height-1)*width_x4 + (get_global_id(0)*4);
	M0 = vload4(0, frame + ind);
	A1 = vload4(0, frame + ind - width_x4);
	U0 = M0; U1 = M0; U2 = M0;
	
	for (i = height-1; i >= 2; --i)
	{
		ind = i*width_x4 + (get_global_id(0)*4);
		A0 = vload4(0, frame + ind - width_x4*2);
	
		buf = mad24(convert_int4(A0),2,mad24(convert_int4(A1),-11,mad24(convert_int4(M0),108,mad24(convert_int4(U0),36,mad24(convert_int4(U1),-8, convert_int4(U2)+64)))))/128;
		M1 = convert_uchar4_sat(buf);
		buf = mad24(convert_int4(A0),3,mad24(convert_int4(A1),-16,mad24(convert_int4(M0),77,mad24(convert_int4(U0),77,mad24(convert_int4(U1),-16, mad24(convert_int4(U2),3,64))))))/128;
		M2 = convert_uchar4_sat(buf);
		buf = (convert_int4(A0) + mad24(convert_int4(A1),-8,mad24(convert_int4(M0),36,mad24(convert_int4(U0),108,mad24(convert_int4(U1),-11, mad24(convert_int4(U2),2,64))))))/128;		
		M3 = convert_uchar4_sat(buf);
		
		ind = (i*4)*width_x4 + (get_global_id(0)*4);
		vstore4(M0, 0, frame + ind); ind += width_x4;
		vstore4(M1, 0, frame + ind); ind += width_x4;
		vstore4(M2, 0, frame + ind); ind += width_x4;
		vstore4(M3, 0, frame + ind);
		
		U2 = U1;
		U1 = U0;
		U0 = M0;
		M0 = A1;
		A1 = A0;
	}
	
	buf = mad24(convert_int4(A0),2,mad24(convert_int4(A1),-11,mad24(convert_int4(M0),108,mad24(convert_int4(U0),36,mad24(convert_int4(U1),-8, convert_int4(U2)+64)))))/128;
	M1 = convert_uchar4_sat(buf);
	buf = mad24(convert_int4(A0),3,mad24(convert_int4(A1),-16,mad24(convert_int4(M0),77,mad24(convert_int4(U0),77,mad24(convert_int4(U1),-16, mad24(convert_int4(U2),3,64))))))/128;
	M2 = convert_uchar4_sat(buf);
	buf = (convert_int4(A0) + mad24(convert_int4(A1),-8,mad24(convert_int4(M0),36,mad24(convert_int4(U0),108,mad24(convert_int4(U1),-11, mad24(convert_int4(U2),2,64))))))/128;		
	M3 = convert_uchar4_sat(buf);
		
	ind = 4*width_x4 + (get_global_id(0)*4);
	vstore4(M0, 0, frame + ind); ind += width_x4;
	vstore4(M1, 0, frame + ind); ind += width_x4;
	vstore4(M2, 0, frame + ind); ind += width_x4;
	vstore4(M3, 0, frame + ind);
		
	U2 = U1;
	U1 = U0;
	U0 = M0;
	M0 = A1;
	A1 = A0;
	
	buf = mad24(convert_int4(A0),2,mad24(convert_int4(A1),-11,mad24(convert_int4(M0),108,mad24(convert_int4(U0),36,mad24(convert_int4(U1),-8, convert_int4(U2)+64)))))/128;
	M1 = convert_uchar4_sat(buf);
	buf = mad24(convert_int4(A0),3,mad24(convert_int4(A1),-16,mad24(convert_int4(M0),77,mad24(convert_int4(U0),77,mad24(convert_int4(U1),-16, mad24(convert_int4(U2),3,64))))))/128;
	M2 = convert_uchar4_sat(buf);
	buf = (convert_int4(A0) + mad24(convert_int4(A1),-8,mad24(convert_int4(M0),36,mad24(convert_int4(U0),108,mad24(convert_int4(U1),-11, mad24(convert_int4(U2),2,64))))))/128;		
	M3 = convert_uchar4_sat(buf);
		
	ind = (get_global_id(0)*4);
	vstore4(M0, 0, frame + ind); ind += width_x4;
	vstore4(M1, 0, frame + ind); ind += width_x4;
	vstore4(M2, 0, frame + ind); ind += width_x4;
	vstore4(M3, 0, frame + ind);

		
	return;	
}



__kernel void chroma_interpolate_Hx8_bc(__global uchar *const src_frame, //0
										__global uchar *const dst_frame, //1
										const int width, //2
										const int height) //3
{
	if (get_global_id(0) > (height-1)) return; 

	__private int4 M, R, L, O; // medium 4 pixels, right, left, output
	// buffer to deal with 8 bit pixel reading/writing
	__private uchar4 buf;
	
	int i, ind;
	int width_x8 = width*8;
	
	
	ind = (get_global_id(0) + 1)*width - 4;
	buf = vload4(0, src_frame + ind);
	M = convert_int4(buf);
	R.s0 = M.s3; R.s1 = M.s3; R.s2 = M.s3; R.s3 = M.s3;
	
	for (i = width-4; i >= 0; i -= 4)
	{
		// index for reading not-interpolated pixels (4 pixels to the left)
		// medium M pixels are set as left L pixels from previous step
		// and previous M -> new right(R)
		ind = get_global_id(0)*width + i - 4;
		if (i >= 4) {
			buf = vload4(0, src_frame + ind);
			L = convert_int4(buf);
		}
		else L = M.s0;
		
		//index for writing interpolated pixels
		ind = get_global_id(0)*width_x8 + (i*8);
		
		//using this: a*b + c*d + e*f + 64 == mad24(a,b,mad24(c,d,mad24(e,f,64))); 
		O.s0 =                                   M.s0;																//	{ 0, 0, 128, 0, 0, 0 }, 
		O.s1 =              mad24(L.s3, -6,mad24(M.s0,123,mad24(M.s1, 12,mad24(M.s2, -1, 64))))>>7;					//	{ 0, -6, 123, 12, -1, 0 }
		O.s2 = mad24(L.s2,2,mad24(L.s3,-11,mad24(M.s0,108,mad24(M.s1, 36,mad24(M.s2, -8, M.s3 + 64)))))>>7;			//	{ 2, -11, 108, 36, -8, 1 }
		O.s3 =              mad24(L.s3, -9,mad24(M.s0, 93,mad24(M.s1, 50,mad24(M.s2, -6, 64))))>>7;					//	{ 0, -9, 93, 50, -6, 0 }
			buf = convert_uchar4_sat(O); vstore4(buf, 0, dst_frame + ind);
		O.s0 = mad24(L.s2,3,mad24(L.s3,-16,mad24(M.s0, 77,mad24(M.s1, 77,mad24(M.s2,-16, mad24(M.s3,3,64))))))>>7;	//	{ 3, -16, 77, 77, -16, 3 }
		O.s1 =              mad24(L.s3, -6,mad24(M.s0, 50,mad24(M.s1, 93,mad24(M.s2, -9, 64))))>>7;					//	{ 0, -6, 50, 93, -9, 0 }
		O.s2 =      (L.s2 + mad24(L.s3, -8,mad24(M.s0, 36,mad24(M.s1,108,mad24(M.s2,-11, mad24(M.s3,2,64))))))>>7;	//	{ 1, -8, 36, 108, -11, 2 }
		O.s3 =              mad24(L.s3, -1,mad24(M.s0, 12,mad24(M.s1,123,mad24(M.s2, -6, 64))))>>7;					//	{ 0, -1, 12, 123, -6, 0 }
			ind += 4; buf = convert_uchar4_sat(O); vstore4(buf, 0, dst_frame + ind);
		O.s0 =                                   M.s1;																//	{ 0, 0, 128, 0, 0, 0 }, 
		O.s1 =              mad24(M.s0, -6,mad24(M.s1,123,mad24(M.s2, 12,mad24(M.s3, -1, 64))))>>7;					//	{ 0, -6, 123, 12, -1, 0 }
		O.s2 = mad24(L.s3,2,mad24(M.s0,-11,mad24(M.s1,108,mad24(M.s2, 36,mad24(M.s3, -8, R.s0 + 64)))))>>7;			//	{ 2, -11, 108, 36, -8, 1 }
		O.s3 =              mad24(M.s0, -9,mad24(M.s1, 93,mad24(M.s2, 50,mad24(M.s3, -6, 64))))>>7;					//	{ 0, -9, 93, 50, -6, 0 }
			ind += 4; buf = convert_uchar4_sat(O); vstore4(buf, 0, dst_frame + ind);
		O.s0 = mad24(L.s3,3,mad24(M.s0,-16,mad24(M.s1, 77,mad24(M.s2, 77,mad24(M.s3,-16, mad24(R.s0,3,64))))))>>7;	//	{ 3, -16, 77, 77, -16, 3 }
		O.s1 =              mad24(M.s0, -6,mad24(M.s1, 50,mad24(M.s2, 93,mad24(M.s3, -9, 64))))>>7;					//	{ 0, -6, 50, 93, -9, 0 }
		O.s2 =      (L.s3 + mad24(M.s0, -8,mad24(M.s1, 36,mad24(M.s2,108,mad24(M.s3,-11, mad24(R.s0,2,64))))))>>7;	//	{ 1, -8, 36, 108, -11, 2 }
		O.s3 =              mad24(M.s0, -1,mad24(M.s1, 12,mad24(M.s2,123,mad24(M.s3, -6, 64))))>>7;					//	{ 0, -1, 12, 123, -6, 0 }
			ind += 4; buf = convert_uchar4_sat(O); vstore4(buf, 0, dst_frame + ind);
		O.s0 =                                   M.s2;																//	{ 0, 0, 128, 0, 0, 0 }, 
		O.s1 =              mad24(M.s1, -6,mad24(M.s2,123,mad24(M.s3, 12,mad24(R.s0, -1, 64))))>>7;					//	{ 0, -6, 123, 12, -1, 0 }
		O.s2 = mad24(M.s0,2,mad24(M.s1,-11,mad24(M.s2,108,mad24(M.s3, 36,mad24(R.s0, -8, R.s1 + 64)))))>>7;			//	{ 2, -11, 108, 36, -8, 1 }
		O.s3 =              mad24(M.s1, -9,mad24(M.s2, 93,mad24(M.s3, 50,mad24(R.s0, -6, 64))))>>7;					//	{ 0, -9, 93, 50, -6, 0 }
			ind += 4; buf = convert_uchar4_sat(O); vstore4(buf, 0, dst_frame + ind);
		O.s0 = mad24(M.s0,3,mad24(M.s1,-16,mad24(M.s2, 77,mad24(M.s3, 77,mad24(R.s0,-16, mad24(R.s1,3,64))))))>>7;	//	{ 3, -16, 77, 77, -16, 3 }
		O.s1 =              mad24(M.s1, -6,mad24(M.s2, 50,mad24(M.s3, 93,mad24(R.s0, -9, 64))))>>7;					//	{ 0, -6, 50, 93, -9, 0 }
		O.s2 =      (M.s0 + mad24(M.s1, -8,mad24(M.s2, 36,mad24(M.s3,108,mad24(R.s0,-11, mad24(R.s1,2,64))))))>>7;	//	{ 1, -8, 36, 108, -11, 2 }
		O.s3 =              mad24(M.s1, -1,mad24(M.s2, 12,mad24(M.s3,123,mad24(R.s0, -6, 64))))>>7;					//	{ 0, -1, 12, 123, -6, 0 }
			ind += 4; buf = convert_uchar4_sat(O); vstore4(buf, 0, dst_frame + ind);
		O.s0 =                                   M.s3;																//	{ 0, 0, 128, 0, 0, 0 }, 
		O.s1 =              mad24(M.s2, -6,mad24(M.s3,123,mad24(R.s0, 12,mad24(R.s1, -1, 64))))>>7;					//	{ 0, -6, 123, 12, -1, 0 }
		O.s2 = mad24(M.s1,2,mad24(M.s2,-11,mad24(M.s3,108,mad24(R.s0, 36,mad24(R.s1, -8, R.s2 + 64)))))>>7;			//	{ 2, -11, 108, 36, -8, 1 }
		O.s3 =              mad24(M.s2, -9,mad24(M.s3, 93,mad24(R.s0, 50,mad24(R.s1, -6, 64))))>>7;					//	{ 0, -9, 93, 50, -6, 0 }
			ind += 4; buf = convert_uchar4_sat(O); vstore4(buf, 0, dst_frame + ind);
		O.s0 = mad24(M.s1,3,mad24(M.s2,-16,mad24(M.s3, 77,mad24(R.s0, 77,mad24(R.s1,-16, mad24(R.s2,3,64))))))>>7;	//	{ 3, -16, 77, 77, -16, 3 }
		O.s1 =              mad24(M.s2, -6,mad24(M.s3, 50,mad24(R.s0, 93,mad24(R.s1, -9, 64))))>>7;					//	{ 0, -6, 50, 93, -9, 0 }
		O.s2 =      (M.s1 + mad24(M.s2, -8,mad24(M.s3, 36,mad24(R.s0,108,mad24(R.s1,-11, mad24(R.s2,2,64))))))>>7;	//	{ 1, -8, 36, 108, -11, 2 }
		O.s3 =              mad24(M.s2, -1,mad24(M.s3, 12,mad24(R.s0,123,mad24(R.s1, -6, 64))))>>7;					//	{ 0, -1, 12, 123, -6, 0 }
			ind += 4; buf = convert_uchar4_sat(O); vstore4(buf, 0, dst_frame + ind);

		R = M;
		M = L;
	}
	
	return;	
}

__kernel void chroma_interpolate_Vx8_bc(__global uchar *const frame, //0
										const int width, //1
										const int height) //2
{
	__private int4 A0, A1, M, O, U0, U1, U2; //2 rows above, 1 row above, middle, output, 1 row under, 2 rows under, 3 rows under; 
	__private uchar4 buf;

	int width_x8 = width*8;
	int i, ind;
	
	if ((get_global_id(0)*4) > (width_x8-1)) return;
	
	ind = (height-1)*width_x8 + (get_global_id(0)*4);
	buf = vload4(0, frame + ind);
	M = convert_int4(buf);
	buf = vload4(0, frame + (ind - width_x8));
	A1 = convert_int4(buf);
	U0 = M; U1 = M; U2 = M;
	
	for (i = height-1; i >= 0; --i)
	{
		ind = (i-2)*width_x8 + (get_global_id(0)*4);
		if (i >= 2) { //checking (i > 2) gives compile-error o_O 
			buf = vload4(0, frame + ind);
			A0 = convert_int4(buf);
		}
		
		//	{ 0,   0, 128,   0,   0,  0 }, 
		//	{ 0,  -6, 123,  12,  -1,  0 }
		//	{ 2, -11, 108,  36,  -8,  1 }
		//	{ 0,  -9,  93,  50,  -6,  0 }
		//	{ 3, -16,  77,  77, -16,  3 }
		//	{ 0,  -6,  50,  93,  -9,  0 }
		//	{ 1,  -8,  36, 108, -11,  2 }
		//	{ 0,  -1,  12, 123,  -6,  0 }	
																		ind = (i*8)*width_x8 + (get_global_id(0)*4); buf = convert_uchar4_sat(M); vstore4(buf, 0, frame + ind);
		O =            mad24(A1, -6,mad24(M,123,mad24(U0, 12,mad24(U1, -1, 64))))/128;	            ind += width_x8; buf = convert_uchar4_sat(O); vstore4(buf, 0, frame + ind);
		O = mad24(A0,2,mad24(A1,-11,mad24(M,108,mad24(U0, 36,mad24(U1, -8, U2 + 64)))))/128;        ind += width_x8; buf = convert_uchar4_sat(O); vstore4(buf, 0, frame + ind);
		O =            mad24(A1, -9,mad24(M, 93,mad24(U0, 50,mad24(U1, -6, 64))))/128;	            ind += width_x8; buf = convert_uchar4_sat(O); vstore4(buf, 0, frame + ind);
		O = mad24(A0,3,mad24(A1,-16,mad24(M, 77,mad24(U0, 77,mad24(U1,-16, mad24(U2,3,64))))))/128; ind += width_x8; buf = convert_uchar4_sat(O); vstore4(buf, 0, frame + ind);
		O =            mad24(A1, -6,mad24(M, 50,mad24(U0, 93,mad24(U1, -9, 64))))/128;              ind += width_x8; buf = convert_uchar4_sat(O); vstore4(buf, 0, frame + ind);
		O =      (A0 + mad24(A1, -8,mad24(M, 36,mad24(U0,108,mad24(U1,-11, mad24(U2,2,64))))))/128; ind += width_x8; buf = convert_uchar4_sat(O); vstore4(buf, 0, frame + ind);
		O =            mad24(A1, -1,mad24(M, 12,mad24(U0,123,mad24(U1, -6, 64))))/128;              ind += width_x8; buf = convert_uchar4_sat(O); vstore4(buf, 0, frame + ind);
		
		U2 = U1;
		U1 = U0;
		U0 = M;
		M = A1;
		A1 = A0;
	}
	
	return;
}

__kernel __attribute__((reqd_work_group_size(64, 1, 1)))
		void count_SSIM(__global uchar *frame1, //0
						__global uchar *frame2, //1
						__global macroblock *MBs, //2
						signed int width, //3
						signed int mb_count)// 4
{
	__private int mb_num, i;
	__private uchar16 FL0, FL1, FL2, FL3, FL4, FL5, FL6, FL7, FL8, FL9, FL10, FL11, FL12, FL13, FL14, FL15;
	__private uchar16 SL0, SL1, SL2, SL3, SL4, SL5, SL6, SL7, SL8, SL9, SL10, SL11, SL12, SL13, SL14, SL15;
	__private float16 IL, IL1;
	__private float M1, M2, D1, D2, C;
	mb_num = get_global_id(0);
	if (mb_num >= mb_count) return;
	i = ((mb_num / (width/16))*16)*width + ((mb_num % (width/16))*16);
	
	FL0 = vload16(0, frame1 + i); i += width; IL = convert_float16(FL0);
	FL1 = vload16(0, frame1 + i); i += width; IL += convert_float16(FL1);
	FL2 = vload16(0, frame1 + i); i += width; IL += convert_float16(FL2);
	FL3 = vload16(0, frame1 + i); i += width; IL += convert_float16(FL3);
	FL4 = vload16(0, frame1 + i); i += width; IL += convert_float16(FL4);
	FL5 = vload16(0, frame1 + i); i += width; IL += convert_float16(FL5);
	FL6 = vload16(0, frame1 + i); i += width; IL += convert_float16(FL6);
	FL7 = vload16(0, frame1 + i); i += width; IL += convert_float16(FL7);
	FL8 = vload16(0, frame1 + i); i += width; IL += convert_float16(FL8);
	FL9 = vload16(0, frame1 + i); i += width; IL += convert_float16(FL9);
	FL10 = vload16(0, frame1 + i); i += width; IL += convert_float16(FL10);
	FL11 = vload16(0, frame1 + i); i += width; IL += convert_float16(FL11);
	FL12 = vload16(0, frame1 + i); i += width; IL += convert_float16(FL12);
	FL13 = vload16(0, frame1 + i); i += width; IL += convert_float16(FL13);
	FL14 = vload16(0, frame1 + i); i += width; IL += convert_float16(FL14);
	FL15 = vload16(0, frame1 + i); i -= 15*width; IL += convert_float16(FL15);
	M1 = (IL.s0 + IL.s1 + IL.s2 + IL.s3 + IL.s4 + IL.s5 + IL.s6 + IL.s7 + IL.s8 + IL.s9 + IL.sA + IL.sB + IL.sC + IL.sD + IL.sE + IL.sF)/256;
	IL = convert_float16(FL0) - M1; IL *= IL;
	IL1 = convert_float16(FL1) - M1; IL = mad(IL1,IL1,IL);
	IL1 = convert_float16(FL2) - M1; IL = mad(IL1,IL1,IL);
	IL1 = convert_float16(FL3) - M1; IL = mad(IL1,IL1,IL);
	IL1 = convert_float16(FL4) - M1; IL = mad(IL1,IL1,IL);
	IL1 = convert_float16(FL5) - M1; IL = mad(IL1,IL1,IL);
	IL1 = convert_float16(FL6) - M1; IL = mad(IL1,IL1,IL);
	IL1 = convert_float16(FL7) - M1; IL = mad(IL1,IL1,IL);
	IL1 = convert_float16(FL8) - M1; IL = mad(IL1,IL1,IL);
	IL1 = convert_float16(FL9) - M1; IL = mad(IL1,IL1,IL);
	IL1 = convert_float16(FL10) - M1; IL = mad(IL1,IL1,IL);
	IL1 = convert_float16(FL11) - M1; IL = mad(IL1,IL1,IL);
	IL1 = convert_float16(FL12) - M1; IL = mad(IL1,IL1,IL);
	IL1 = convert_float16(FL13) - M1; IL = mad(IL1,IL1,IL);
	IL1 = convert_float16(FL14) - M1; IL = mad(IL1,IL1,IL);
	IL1 = convert_float16(FL15) - M1; IL = mad(IL1,IL1,IL);
	D1 = (IL.s0 + IL.s1 + IL.s2 + IL.s3 + IL.s4 + IL.s5 + IL.s6 + IL.s7 + IL.s8 + IL.s9 + IL.sA + IL.sB + IL.sC + IL.sD + IL.sE + IL.sF)/256;
	
	SL0 = vload16(0, frame2 + i); i += width; IL = convert_float16(SL0);
	SL1 = vload16(0, frame2 + i); i += width; IL += convert_float16(SL1);
	SL2 = vload16(0, frame2 + i); i += width; IL += convert_float16(SL2);
	SL3 = vload16(0, frame2 + i); i += width; IL += convert_float16(SL3);
	SL4 = vload16(0, frame2 + i); i += width; IL += convert_float16(SL4); 
	SL5 = vload16(0, frame2 + i); i += width; IL += convert_float16(SL5);
	SL6 = vload16(0, frame2 + i); i += width; IL += convert_float16(SL6);
	SL7 = vload16(0, frame2 + i); i += width; IL += convert_float16(SL7);
	SL8 = vload16(0, frame2 + i); i += width; IL += convert_float16(SL8);
	SL9 = vload16(0, frame2 + i); i += width; IL += convert_float16(SL9);
	SL10 = vload16(0, frame2 + i); i += width; IL += convert_float16(SL10);
	SL11 = vload16(0, frame2 + i); i += width; IL += convert_float16(SL11);
	SL12 = vload16(0, frame2 + i); i += width; IL += convert_float16(SL12);
	SL13 = vload16(0, frame2 + i); i += width; IL += convert_float16(SL13);
	SL14 = vload16(0, frame2 + i); i += width; IL += convert_float16(SL14);
	SL15 = vload16(0, frame2 + i); IL += convert_float16(SL15);
	M2 = (IL.s0 + IL.s1 + IL.s2 + IL.s3 + IL.s4 + IL.s5 + IL.s6 + IL.s7 + IL.s8 + IL.s9 + IL.sA + IL.sB + IL.sC + IL.sD + IL.sE + IL.sF)/256;
	IL = convert_float16(SL0) - M2; IL *= IL;
	IL1 = convert_float16(SL1) - M2; IL = mad(IL1,IL1,IL);
	IL1 = convert_float16(SL2) - M2; IL = mad(IL1,IL1,IL);
	IL1 = convert_float16(SL3) - M2; IL = mad(IL1,IL1,IL);
	IL1 = convert_float16(SL4) - M2; IL = mad(IL1,IL1,IL);
	IL1 = convert_float16(SL5) - M2; IL = mad(IL1,IL1,IL);
	IL1 = convert_float16(SL6) - M2; IL = mad(IL1,IL1,IL);
	IL1 = convert_float16(SL7) - M2; IL = mad(IL1,IL1,IL);
	IL1 = convert_float16(SL8) - M2; IL = mad(IL1,IL1,IL);
	IL1 = convert_float16(SL9) - M2; IL = mad(IL1,IL1,IL);
	IL1 = convert_float16(SL10) - M2; IL = mad(IL1,IL1,IL);
	IL1 = convert_float16(SL11) - M2; IL = mad(IL1,IL1,IL);
	IL1 = convert_float16(SL12) - M2; IL = mad(IL1,IL1,IL);
	IL1 = convert_float16(SL13) - M2; IL = mad(IL1,IL1,IL);
	IL1 = convert_float16(SL14) - M2; IL = mad(IL1,IL1,IL);
	IL1 = convert_float16(SL15) - M2; IL = mad(IL1,IL1,IL);
	D2 = (IL.s0 + IL.s1 + IL.s2 + IL.s3 + IL.s4 + IL.s5 + IL.s6 + IL.s7 + IL.s8 + IL.s9 + IL.sA + IL.sB + IL.sC + IL.sD + IL.sE + IL.sF)/256;
	
	IL = (convert_float16(FL0) - M1)*(convert_float16(SL0) - M2);
	IL += (convert_float16(FL1) - M1)*(convert_float16(SL1) - M2);
	IL += (convert_float16(FL2) - M1)*(convert_float16(SL2) - M2);
	IL += (convert_float16(FL3) - M1)*(convert_float16(SL3) - M2);
	IL += (convert_float16(FL4) - M1)*(convert_float16(SL4) - M2);
	IL += (convert_float16(FL5) - M1)*(convert_float16(SL5) - M2);
	IL += (convert_float16(FL6) - M1)*(convert_float16(SL6) - M2);
	IL += (convert_float16(FL7) - M1)*(convert_float16(SL7) - M2);
	IL += (convert_float16(FL8) - M1)*(convert_float16(SL8) - M2);
	IL += (convert_float16(FL9) - M1)*(convert_float16(SL9) - M2);
	IL += (convert_float16(FL10) - M1)*(convert_float16(SL10) - M2);
	IL += (convert_float16(FL11) - M1)*(convert_float16(SL11) - M2);
	IL += (convert_float16(FL12) - M1)*(convert_float16(SL12) - M2);
	IL += (convert_float16(FL13) - M1)*(convert_float16(SL13) - M2);
	IL += (convert_float16(FL14) - M1)*(convert_float16(SL14) - M2);
	IL += (convert_float16(FL15) - M1)*(convert_float16(SL15) - M2);
	C = (IL.s0 + IL.s1 + IL.s2 + IL.s3 + IL.s4 + IL.s5 + IL.s6 + IL.s7 + IL.s8 + IL.s9 + IL.sA + IL.sB + IL.sC + IL.sD + IL.sE + IL.sF)/256;
	
	const float c1 = 0.01f*0.01f*255*255;
	const float c2 = 0.03f*0.03f*255*255;
	C = mad(M1,M2*2,c1)*mad(C,2,c2)/(mad(M1,M1,mad(M2,M2,c1))*(D1 + D2 + c2));
	
	MBs[mb_num].SSIM = C;
	
	return;
}
