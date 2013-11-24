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
	UQ_segment = 0,
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

__constant int vector_diff_weight = 16;

void weight(int4 *const __L0, int4 *const __L1, int4 *const __L2, int4 *const __L3) 
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

void weight_opt(int4 *const XX, int16 *const L) 
{
	*XX = (*L).s0123;
	(*L).s0123 = (((*L).s0123 + (*L).sCDEF) << 3);	// a1 = ((ip[0] + ip[3])<<3);
	(*L).sCDEF = ((*XX - (*L).sCDEF) << 3);	// d1 = ((ip[0] - ip[3])<<3);
	*XX = (*L).s4567;
	(*L).s4567 = (((*L).s4567 + (*L).s89AB) << 3);	// b1 = ((ip[1] + ip[2])<<3);
	(*L).s89AB = ((*XX - (*L).s89AB) << 3);	// c1 = ((ip[1] - ip[2])<<3);

	*XX = (*L).s89AB;
	(*L).s89AB = (*L).s0123 - (*L).s4567;				// op[2] = (a1 - b1);
	(*L).s0123 = (*L).s0123 + (*L).s4567;				// op[0] = (a1 + b1); 
	
	(*L).s4567 = (((*XX * 2217) + ((*L).sCDEF * 5352) + 14500) >> 12);
														// op[1] = (c1 * 2217 + d1 * 5352 +  14500)>>12;
	(*L).sCDEF = ((((*L).sCDEF * 2217) - (*XX * 5352) + 7500) >> 12);
														// op[3] = (d1 * 2217 - c1 * 5352 +   7500)>>12;

	(*XX).x = (*L).s0; (*XX).y = (*L).s4; (*XX).z = (*L).s8; (*XX).w = (*L).sC;
	// a1 = op[0] + op[3];
	(*L).s0 = (*L).s0 + (*L).s3;
	(*L).s4 = (*L).s4 + (*L).s7;
	(*L).s8 = (*L).s8 + (*L).sB;
	(*L).sC = (*L).sC + (*L).sF;
	// d1 = op[0] - op[3];	
	(*L).s3 = (*XX).x - (*L).s3;
	(*L).s7 = (*XX).y - (*L).s7;
	(*L).sB = (*XX).z - (*L).sB;
	(*L).sF = (*XX).w - (*L).sF;
	(*XX).x = (*L).s1; (*XX).y = (*L).s5; (*XX).z = (*L).s9; (*XX).w = (*L).sD;
	// b1 = op[1] + op[2];
	(*L).s1 = (*L).s1 + (*L).s2;
	(*L).s5 = (*L).s5 + (*L).s6;
	(*L).s9 = (*L).s9 + (*L).sA;
	(*L).sD = (*L).sD + (*L).sE;
	// c1 = op[1] - op[2];
	(*L).s2 = (*XX).x - (*L).s2;
	(*L).s6 = (*XX).y - (*L).s6;
	(*L).sA = (*XX).z - (*L).sA;
	(*L).sE = (*XX).w - (*L).sE;
	
	(*XX).x = (*L).s2; (*XX).y = (*L).s6; (*XX).z = (*L).sA; (*XX).w = (*L).sE;
	// op[2] = (( a1 - b1 + 7)>>4);
	(*L).s2 = (((*L).s0 - (*L).s1 + 7) >> 4);
	(*L).s6 = (((*L).s4 - (*L).s5 + 7) >> 4);
	(*L).sA = (((*L).s8 - (*L).s9 + 7) >> 4);
	(*L).sE = (((*L).sC - (*L).sD + 7) >> 4);
	// op[0] = (( a1 + b1 + 7)>>4);	
	(*L).s0 = (((*L).s0 + (*L).s1 + 7) >> 4);
	(*L).s4 = (((*L).s4 + (*L).s5 + 7) >> 4);
	(*L).s8 = (((*L).s8 + (*L).s9 + 7) >> 4);
	(*L).sC = (((*L).sC + (*L).sD + 7) >> 4);
	
	// op[1]  = (((c1 * 2217 + d1 * 5352 +  12000)>>16) + (d1!=0));
	(*L).s1 = ((((*XX).x * 2217) + ((*L).s3 *5352) + 12000) >> 16) + ((*L).s3!=0);
	(*L).s5 = ((((*XX).y * 2217) + ((*L).s7 *5352) + 12000) >> 16) + ((*L).s7!=0);
	(*L).s9 = ((((*XX).z * 2217) + ((*L).sB *5352) + 12000) >> 16) + ((*L).sB!=0);
	(*L).sD = ((((*XX).w * 2217) + ((*L).sF *5352) + 12000) >> 16) + ((*L).sF!=0);
	
	// op[3] = ((d1 * 2217 - c1 * 5352 +  51000)>>16);
	(*L).s3 = ((((*L).s3 * 2217) - ((*XX).x *5352) + 51000) >> 16);
	(*L).s7 = ((((*L).s7 * 2217) - ((*XX).y *5352) + 51000) >> 16);
	(*L).sB = ((((*L).sB * 2217) - ((*XX).z *5352) + 51000) >> 16);
	(*L).sF = ((((*L).sF * 2217) - ((*XX).w *5352) + 51000) >> 16);
	
	*L = convert_int16(abs(*L));
	(*XX).x = (*L).s0/4 + (*L).s1 + (*L).s2 + (*L).s3 +
				(*L).s4 + (*L).s5 + (*L).s6 + (*L).s7 +
				(*L).s8 + (*L).s9 + (*L).sA + (*L).sB +
				(*L).sC + (*L).sD + (*L).sE + (*L).sF;
	
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


__kernel void reset_vectors ( __global vector_net *const last_net, //0
								__global vector_net *const golden_net, //1
								__global vector_net *const altref_net, //2
								__global int *const last_Bdiff, //3
								__global int *const golden_Bdiff, //4
								__global int *const altref_Bdiff) //5
{
	int b8x8_num = get_global_id(0);
	last_net[b8x8_num].vector_x = 0; 
	last_net[b8x8_num].vector_y = 0;
	golden_net[b8x8_num].vector_x = 0; 
	golden_net[b8x8_num].vector_y = 0;
	altref_net[b8x8_num].vector_x = 0; 
	altref_net[b8x8_num].vector_y = 0;
	last_Bdiff[b8x8_num] = 0x7fffffff;
	golden_Bdiff[b8x8_num] = 0x7fffffff;
	altref_Bdiff[b8x8_num] = 0x7fffffff;
	
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
			
			Diff += (int)(abs((int)abs(px-cx)-vector_x0) + abs((int)abs(py-cy)-vector_y0))*vector_diff_weight*2;
						
			vector_x = (Diff < MinDiff) ? (px - cx) : vector_x;
			vector_y = (Diff < MinDiff) ? (py - cy) : vector_y;
			MinDiff = (Diff < MinDiff) ? Diff : MinDiff;			
    	} 
    } 
	
	
	dst_net[b8x8_num].vector_x = (vector_x*pixel_rate);
	dst_net[b8x8_num].vector_y = (vector_y*pixel_rate);
	
	return;
	
}


__constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;
__constant int f[8][6] = { /* indexed by displacement */
	{ 0, 0, 128, 0, 0, 0 }, /* degenerate whole-pixel */
	{ 0, -6, 123, 12, -1, 0 }, /* 1/8 */
	{ 2, -11, 108, 36, -8, 1 }, /* 1/4 */
	{ 0, -9, 93, 50, -6, 0 }, /* 3/8 */
	{ 3, -16, 77, 77, -16, 3 }, /* 1/2 is symmetric */
	{ 0, -6, 50, 93, -9, 0 }, /* 5/8 = reverse of 3/8 */
	{ 1, -8, 36, 108, -11, 2 }, /* 3/4 = reverse of 1/4 */
	{ 0, -1, 12, 123, -6, 0 } /* 7/8 = reverse of 1/8 */
};

void construct(__read_only image2d_t ref_frame, const int2 coords, const int2 d, int4 *const DL0, int4 *const DL1, int4 *const DL2, int4 *const DL3)
{
	__private int2 c;
	__private uint4 buf4ui;
	__private int4 l;
	__private uchar4 p4;
	__private uchar2 p2;
	__private uchar4 l0,l1,l2,l3,la0,la1;
	
	// read line of 9 pixels and interpolate line of 4 pixels from it
	// we need 9 of these 4-pixel lines
	// line -2
	c.x = coords.x - 2; c.y = coords.y - 2;
	buf4ui = read_imageui(ref_frame,sampler,c); p4.s0 = (uchar)buf4ui.s0; ++c.x;
	buf4ui = read_imageui(ref_frame,sampler,c); p4.s1 = (uchar)buf4ui.s0; ++c.x;
	buf4ui = read_imageui(ref_frame,sampler,c); p4.s2 = (uchar)buf4ui.s0; ++c.x;
	buf4ui = read_imageui(ref_frame,sampler,c); p4.s3 = (uchar)buf4ui.s0; ++c.x;
	buf4ui = read_imageui(ref_frame,sampler,c); p2.s0 = (uchar)buf4ui.s0; ++c.x;
	buf4ui = read_imageui(ref_frame,sampler,c); p2.s1 = (uchar)buf4ui.s0; ++c.x;
	l.s0 = (mad24((int)p4.s0,f[d.x][0],mad24((int)p4.s1,f[d.x][1],mad24((int)p4.s2,f[d.x][2],mad24((int)p4.s3,f[d.x][3],mad24((int)p2.s0,f[d.x][4],mad24((int)p2.s1,f[d.x][5],64))))))/128);
	p4.s0=p4.s1;p4.s1=p4.s2;p4.s2=p4.s3;p4.s3=p2.s0;p2.s0=p2.s1; buf4ui = read_imageui(ref_frame,sampler,c); p2.s1 = (uchar)buf4ui.s0; ++c.x;
	l.s1 = (mad24((int)p4.s0,f[d.x][0],mad24((int)p4.s1,f[d.x][1],mad24((int)p4.s2,f[d.x][2],mad24((int)p4.s3,f[d.x][3],mad24((int)p2.s0,f[d.x][4],mad24((int)p2.s1,f[d.x][5],64))))))/128);
	p4.s0=p4.s1;p4.s1=p4.s2;p4.s2=p4.s3;p4.s3=p2.s0;p2.s0=p2.s1; buf4ui = read_imageui(ref_frame,sampler,c); p2.s1 = (uchar)buf4ui.s0; ++c.x;
	l.s2 = (mad24((int)p4.s0,f[d.x][0],mad24((int)p4.s1,f[d.x][1],mad24((int)p4.s2,f[d.x][2],mad24((int)p4.s3,f[d.x][3],mad24((int)p2.s0,f[d.x][4],mad24((int)p2.s1,f[d.x][5],64))))))/128);
	p4.s0=p4.s1;p4.s1=p4.s2;p4.s2=p4.s3;p4.s3=p2.s0;p2.s0=p2.s1; buf4ui = read_imageui(ref_frame,sampler,c); p2.s1 = (uchar)buf4ui.s0;
	l.s3 = (mad24((int)p4.s0,f[d.x][0],mad24((int)p4.s1,f[d.x][1],mad24((int)p4.s2,f[d.x][2],mad24((int)p4.s3,f[d.x][3],mad24((int)p2.s0,f[d.x][4],mad24((int)p2.s1,f[d.x][5],64))))))/128);
	la0 = convert_uchar4_sat(l);
	// line -1
	c.x = coords.x - 2; c.y = coords.y - 1;
	buf4ui = read_imageui(ref_frame,sampler,c); p4.s0 = (uchar)buf4ui.s0; ++c.x;
	buf4ui = read_imageui(ref_frame,sampler,c); p4.s1 = (uchar)buf4ui.s0; ++c.x;
	buf4ui = read_imageui(ref_frame,sampler,c); p4.s2 = (uchar)buf4ui.s0; ++c.x;
	buf4ui = read_imageui(ref_frame,sampler,c); p4.s3 = (uchar)buf4ui.s0; ++c.x;
	buf4ui = read_imageui(ref_frame,sampler,c); p2.s0 = (uchar)buf4ui.s0; ++c.x;
	buf4ui = read_imageui(ref_frame,sampler,c); p2.s1 = (uchar)buf4ui.s0; ++c.x;
	l.s0 = (mad24((int)p4.s0,f[d.x][0],mad24((int)p4.s1,f[d.x][1],mad24((int)p4.s2,f[d.x][2],mad24((int)p4.s3,f[d.x][3],mad24((int)p2.s0,f[d.x][4],mad24((int)p2.s1,f[d.x][5],64))))))/128);
	p4.s0=p4.s1;p4.s1=p4.s2;p4.s2=p4.s3;p4.s3=p2.s0;p2.s0=p2.s1; buf4ui = read_imageui(ref_frame,sampler,c); p2.s1 = (uchar)buf4ui.s0; ++c.x;
	l.s1 = (mad24((int)p4.s0,f[d.x][0],mad24((int)p4.s1,f[d.x][1],mad24((int)p4.s2,f[d.x][2],mad24((int)p4.s3,f[d.x][3],mad24((int)p2.s0,f[d.x][4],mad24((int)p2.s1,f[d.x][5],64))))))/128);
	p4.s0=p4.s1;p4.s1=p4.s2;p4.s2=p4.s3;p4.s3=p2.s0;p2.s0=p2.s1; buf4ui = read_imageui(ref_frame,sampler,c); p2.s1 = (uchar)buf4ui.s0; ++c.x;
	l.s2 = (mad24((int)p4.s0,f[d.x][0],mad24((int)p4.s1,f[d.x][1],mad24((int)p4.s2,f[d.x][2],mad24((int)p4.s3,f[d.x][3],mad24((int)p2.s0,f[d.x][4],mad24((int)p2.s1,f[d.x][5],64))))))/128);
	p4.s0=p4.s1;p4.s1=p4.s2;p4.s2=p4.s3;p4.s3=p2.s0;p2.s0=p2.s1; buf4ui = read_imageui(ref_frame,sampler,c); p2.s1 = (uchar)buf4ui.s0;
	l.s3 = (mad24((int)p4.s0,f[d.x][0],mad24((int)p4.s1,f[d.x][1],mad24((int)p4.s2,f[d.x][2],mad24((int)p4.s3,f[d.x][3],mad24((int)p2.s0,f[d.x][4],mad24((int)p2.s1,f[d.x][5],64))))))/128);
	la1 = convert_uchar4_sat(l);
	// line 0
	c.x = coords.x - 2; c.y = coords.y;
	buf4ui = read_imageui(ref_frame,sampler,c); p4.s0 = (uchar)buf4ui.s0; ++c.x;
	buf4ui = read_imageui(ref_frame,sampler,c); p4.s1 = (uchar)buf4ui.s0; ++c.x;
	buf4ui = read_imageui(ref_frame,sampler,c); p4.s2 = (uchar)buf4ui.s0; ++c.x;
	buf4ui = read_imageui(ref_frame,sampler,c); p4.s3 = (uchar)buf4ui.s0; ++c.x;
	buf4ui = read_imageui(ref_frame,sampler,c); p2.s0 = (uchar)buf4ui.s0; ++c.x;
	buf4ui = read_imageui(ref_frame,sampler,c); p2.s1 = (uchar)buf4ui.s0; ++c.x;
	l.s0 = (mad24((int)p4.s0,f[d.x][0],mad24((int)p4.s1,f[d.x][1],mad24((int)p4.s2,f[d.x][2],mad24((int)p4.s3,f[d.x][3],mad24((int)p2.s0,f[d.x][4],mad24((int)p2.s1,f[d.x][5],64))))))/128);
	p4.s0=p4.s1;p4.s1=p4.s2;p4.s2=p4.s3;p4.s3=p2.s0;p2.s0=p2.s1; buf4ui = read_imageui(ref_frame,sampler,c); p2.s1 = (uchar)buf4ui.s0; ++c.x;
	l.s1 = (mad24((int)p4.s0,f[d.x][0],mad24((int)p4.s1,f[d.x][1],mad24((int)p4.s2,f[d.x][2],mad24((int)p4.s3,f[d.x][3],mad24((int)p2.s0,f[d.x][4],mad24((int)p2.s1,f[d.x][5],64))))))/128);
	p4.s0=p4.s1;p4.s1=p4.s2;p4.s2=p4.s3;p4.s3=p2.s0;p2.s0=p2.s1; buf4ui = read_imageui(ref_frame,sampler,c); p2.s1 = (uchar)buf4ui.s0; ++c.x;
	l.s2 = (mad24((int)p4.s0,f[d.x][0],mad24((int)p4.s1,f[d.x][1],mad24((int)p4.s2,f[d.x][2],mad24((int)p4.s3,f[d.x][3],mad24((int)p2.s0,f[d.x][4],mad24((int)p2.s1,f[d.x][5],64))))))/128);
	p4.s0=p4.s1;p4.s1=p4.s2;p4.s2=p4.s3;p4.s3=p2.s0;p2.s0=p2.s1; buf4ui = read_imageui(ref_frame,sampler,c); p2.s1 = (uchar)buf4ui.s0;
	l.s3 = (mad24((int)p4.s0,f[d.x][0],mad24((int)p4.s1,f[d.x][1],mad24((int)p4.s2,f[d.x][2],mad24((int)p4.s3,f[d.x][3],mad24((int)p2.s0,f[d.x][4],mad24((int)p2.s1,f[d.x][5],64))))))/128);
	l0 = convert_uchar4_sat(l);
	// line 1
	c.x = coords.x - 2; c.y = coords.y + 1;
	buf4ui = read_imageui(ref_frame,sampler,c); p4.s0 = (uchar)buf4ui.s0; ++c.x;
	buf4ui = read_imageui(ref_frame,sampler,c); p4.s1 = (uchar)buf4ui.s0; ++c.x;
	buf4ui = read_imageui(ref_frame,sampler,c); p4.s2 = (uchar)buf4ui.s0; ++c.x;
	buf4ui = read_imageui(ref_frame,sampler,c); p4.s3 = (uchar)buf4ui.s0; ++c.x;
	buf4ui = read_imageui(ref_frame,sampler,c); p2.s0 = (uchar)buf4ui.s0; ++c.x;
	buf4ui = read_imageui(ref_frame,sampler,c); p2.s1 = (uchar)buf4ui.s0; ++c.x;
	l.s0 = (mad24((int)p4.s0,f[d.x][0],mad24((int)p4.s1,f[d.x][1],mad24((int)p4.s2,f[d.x][2],mad24((int)p4.s3,f[d.x][3],mad24((int)p2.s0,f[d.x][4],mad24((int)p2.s1,f[d.x][5],64))))))/128);
	p4.s0=p4.s1;p4.s1=p4.s2;p4.s2=p4.s3;p4.s3=p2.s0;p2.s0=p2.s1; buf4ui = read_imageui(ref_frame,sampler,c); p2.s1 = (uchar)buf4ui.s0; ++c.x;
	l.s1 = (mad24((int)p4.s0,f[d.x][0],mad24((int)p4.s1,f[d.x][1],mad24((int)p4.s2,f[d.x][2],mad24((int)p4.s3,f[d.x][3],mad24((int)p2.s0,f[d.x][4],mad24((int)p2.s1,f[d.x][5],64))))))/128);
	p4.s0=p4.s1;p4.s1=p4.s2;p4.s2=p4.s3;p4.s3=p2.s0;p2.s0=p2.s1; buf4ui = read_imageui(ref_frame,sampler,c); p2.s1 = (uchar)buf4ui.s0; ++c.x;
	l.s2 = (mad24((int)p4.s0,f[d.x][0],mad24((int)p4.s1,f[d.x][1],mad24((int)p4.s2,f[d.x][2],mad24((int)p4.s3,f[d.x][3],mad24((int)p2.s0,f[d.x][4],mad24((int)p2.s1,f[d.x][5],64))))))/128);
	p4.s0=p4.s1;p4.s1=p4.s2;p4.s2=p4.s3;p4.s3=p2.s0;p2.s0=p2.s1; buf4ui = read_imageui(ref_frame,sampler,c); p2.s1 = (uchar)buf4ui.s0;
	l.s3 = (mad24((int)p4.s0,f[d.x][0],mad24((int)p4.s1,f[d.x][1],mad24((int)p4.s2,f[d.x][2],mad24((int)p4.s3,f[d.x][3],mad24((int)p2.s0,f[d.x][4],mad24((int)p2.s1,f[d.x][5],64))))))/128);
	l1 = convert_uchar4_sat(l);
	// line 2
	c.x = coords.x - 2; c.y = coords.y + 2;
	buf4ui = read_imageui(ref_frame,sampler,c); p4.s0 = (uchar)buf4ui.s0; ++c.x;
	buf4ui = read_imageui(ref_frame,sampler,c); p4.s1 = (uchar)buf4ui.s0; ++c.x;
	buf4ui = read_imageui(ref_frame,sampler,c); p4.s2 = (uchar)buf4ui.s0; ++c.x;
	buf4ui = read_imageui(ref_frame,sampler,c); p4.s3 = (uchar)buf4ui.s0; ++c.x;
	buf4ui = read_imageui(ref_frame,sampler,c); p2.s0 = (uchar)buf4ui.s0; ++c.x;
	buf4ui = read_imageui(ref_frame,sampler,c); p2.s1 = (uchar)buf4ui.s0; ++c.x;
	l.s0 = (mad24((int)p4.s0,f[d.x][0],mad24((int)p4.s1,f[d.x][1],mad24((int)p4.s2,f[d.x][2],mad24((int)p4.s3,f[d.x][3],mad24((int)p2.s0,f[d.x][4],mad24((int)p2.s1,f[d.x][5],64))))))/128);
	p4.s0=p4.s1;p4.s1=p4.s2;p4.s2=p4.s3;p4.s3=p2.s0;p2.s0=p2.s1; buf4ui = read_imageui(ref_frame,sampler,c); p2.s1 = (uchar)buf4ui.s0; ++c.x;
	l.s1 = (mad24((int)p4.s0,f[d.x][0],mad24((int)p4.s1,f[d.x][1],mad24((int)p4.s2,f[d.x][2],mad24((int)p4.s3,f[d.x][3],mad24((int)p2.s0,f[d.x][4],mad24((int)p2.s1,f[d.x][5],64))))))/128);
	p4.s0=p4.s1;p4.s1=p4.s2;p4.s2=p4.s3;p4.s3=p2.s0;p2.s0=p2.s1; buf4ui = read_imageui(ref_frame,sampler,c); p2.s1 = (uchar)buf4ui.s0; ++c.x;
	l.s2 = (mad24((int)p4.s0,f[d.x][0],mad24((int)p4.s1,f[d.x][1],mad24((int)p4.s2,f[d.x][2],mad24((int)p4.s3,f[d.x][3],mad24((int)p2.s0,f[d.x][4],mad24((int)p2.s1,f[d.x][5],64))))))/128);
	p4.s0=p4.s1;p4.s1=p4.s2;p4.s2=p4.s3;p4.s3=p2.s0;p2.s0=p2.s1; buf4ui = read_imageui(ref_frame,sampler,c); p2.s1 = (uchar)buf4ui.s0;
	l.s3 = (mad24((int)p4.s0,f[d.x][0],mad24((int)p4.s1,f[d.x][1],mad24((int)p4.s2,f[d.x][2],mad24((int)p4.s3,f[d.x][3],mad24((int)p2.s0,f[d.x][4],mad24((int)p2.s1,f[d.x][5],64))))))/128);
	l2 = convert_uchar4_sat(l);
	// line 3
	c.x = coords.x - 2; c.y = coords.y + 3; 
	buf4ui = read_imageui(ref_frame,sampler,c); p4.s0 = (uchar)buf4ui.s0; ++c.x;
	buf4ui = read_imageui(ref_frame,sampler,c); p4.s1 = (uchar)buf4ui.s0; ++c.x;
	buf4ui = read_imageui(ref_frame,sampler,c); p4.s2 = (uchar)buf4ui.s0; ++c.x;
	buf4ui = read_imageui(ref_frame,sampler,c); p4.s3 = (uchar)buf4ui.s0; ++c.x;
	buf4ui = read_imageui(ref_frame,sampler,c); p2.s0 = (uchar)buf4ui.s0; ++c.x;
	buf4ui = read_imageui(ref_frame,sampler,c); p2.s1 = (uchar)buf4ui.s0; ++c.x;
	l.s0 = (mad24((int)p4.s0,f[d.x][0],mad24((int)p4.s1,f[d.x][1],mad24((int)p4.s2,f[d.x][2],mad24((int)p4.s3,f[d.x][3],mad24((int)p2.s0,f[d.x][4],mad24((int)p2.s1,f[d.x][5],64))))))/128);
	p4.s0=p4.s1;p4.s1=p4.s2;p4.s2=p4.s3;p4.s3=p2.s0;p2.s0=p2.s1; buf4ui = read_imageui(ref_frame,sampler,c); p2.s1 = (uchar)buf4ui.s0; ++c.x;
	l.s1 = (mad24((int)p4.s0,f[d.x][0],mad24((int)p4.s1,f[d.x][1],mad24((int)p4.s2,f[d.x][2],mad24((int)p4.s3,f[d.x][3],mad24((int)p2.s0,f[d.x][4],mad24((int)p2.s1,f[d.x][5],64))))))/128);
	p4.s0=p4.s1;p4.s1=p4.s2;p4.s2=p4.s3;p4.s3=p2.s0;p2.s0=p2.s1; buf4ui = read_imageui(ref_frame,sampler,c); p2.s1 = (uchar)buf4ui.s0; ++c.x;
	l.s2 = (mad24((int)p4.s0,f[d.x][0],mad24((int)p4.s1,f[d.x][1],mad24((int)p4.s2,f[d.x][2],mad24((int)p4.s3,f[d.x][3],mad24((int)p2.s0,f[d.x][4],mad24((int)p2.s1,f[d.x][5],64))))))/128);
	p4.s0=p4.s1;p4.s1=p4.s2;p4.s2=p4.s3;p4.s3=p2.s0;p2.s0=p2.s1; buf4ui = read_imageui(ref_frame,sampler,c); p2.s1 = (uchar)buf4ui.s0;
	l.s3 = (mad24((int)p4.s0,f[d.x][0],mad24((int)p4.s1,f[d.x][1],mad24((int)p4.s2,f[d.x][2],mad24((int)p4.s3,f[d.x][3],mad24((int)p2.s0,f[d.x][4],mad24((int)p2.s1,f[d.x][5],64))))))/128);
	l3 = convert_uchar4_sat(l);
	
	// now we interpolate in collumns
	// row 0
	p4.s0=la0.s0;p4.s1=la1.s0;p4.s2=l0.s0;p4.s3=l1.s0;p2.s0=l2.s0;p2.s1=l3.s0;
	(*DL0).s0 = (mad24((int)p4.s0,f[d.y][0],mad24((int)p4.s1,f[d.y][1],mad24((int)p4.s2,f[d.y][2],mad24((int)p4.s3,f[d.y][3],mad24((int)p2.s0,f[d.y][4],mad24((int)p2.s1,f[d.y][5],64))))))/128);
	p4.s0=la0.s1;p4.s1=la1.s1;p4.s2=l0.s1;p4.s3=l1.s1;p2.s0=l2.s1;p2.s1=l3.s1;
	(*DL0).s1 = (mad24((int)p4.s0,f[d.y][0],mad24((int)p4.s1,f[d.y][1],mad24((int)p4.s2,f[d.y][2],mad24((int)p4.s3,f[d.y][3],mad24((int)p2.s0,f[d.y][4],mad24((int)p2.s1,f[d.y][5],64))))))/128);
	p4.s0=la0.s2;p4.s1=la1.s2;p4.s2=l0.s2;p4.s3=l1.s2;p2.s0=l2.s2;p2.s1=l3.s2;
	(*DL0).s2 = (mad24((int)p4.s0,f[d.y][0],mad24((int)p4.s1,f[d.y][1],mad24((int)p4.s2,f[d.y][2],mad24((int)p4.s3,f[d.y][3],mad24((int)p2.s0,f[d.y][4],mad24((int)p2.s1,f[d.y][5],64))))))/128);
	p4.s0=la0.s3;p4.s1=la1.s3;p4.s2=l0.s3;p4.s3=l1.s3;p2.s0=l2.s3;p2.s1=l3.s3;
	(*DL0).s3 = (mad24((int)p4.s0,f[d.y][0],mad24((int)p4.s1,f[d.y][1],mad24((int)p4.s2,f[d.y][2],mad24((int)p4.s3,f[d.y][3],mad24((int)p2.s0,f[d.y][4],mad24((int)p2.s1,f[d.y][5],64))))))/128);
	// row 1 
	la0 = la1; la1 = l0; l0 = l1; l1 = l2; l2 = l3;
	// first produce line l3
	c.x = coords.x - 2; c.y = coords.y + 4; 
	buf4ui = read_imageui(ref_frame,sampler,c); p4.s0 = (uchar)buf4ui.s0; ++c.x;
	buf4ui = read_imageui(ref_frame,sampler,c); p4.s1 = (uchar)buf4ui.s0; ++c.x;
	buf4ui = read_imageui(ref_frame,sampler,c); p4.s2 = (uchar)buf4ui.s0; ++c.x;
	buf4ui = read_imageui(ref_frame,sampler,c); p4.s3 = (uchar)buf4ui.s0; ++c.x;
	buf4ui = read_imageui(ref_frame,sampler,c); p2.s0 = (uchar)buf4ui.s0; ++c.x;
	buf4ui = read_imageui(ref_frame,sampler,c); p2.s1 = (uchar)buf4ui.s0; ++c.x;
	l3.s0 = (uchar)(mad24((int)p4.s0,f[d.x][0],mad24((int)p4.s1,f[d.x][1],mad24((int)p4.s2,f[d.x][2],mad24((int)p4.s3,f[d.x][3],mad24((int)p2.s0,f[d.x][4],mad24((int)p2.s1,f[d.x][5],64))))))/128);
	p4.s0=p4.s1;p4.s1=p4.s2;p4.s2=p4.s3;p4.s3=p2.s0;p2.s0=p2.s1; buf4ui = read_imageui(ref_frame,sampler,c); p2.s1 = (uchar)buf4ui.s0; ++c.x;
	l3.s1 = (uchar)(mad24((int)p4.s0,f[d.x][0],mad24((int)p4.s1,f[d.x][1],mad24((int)p4.s2,f[d.x][2],mad24((int)p4.s3,f[d.x][3],mad24((int)p2.s0,f[d.x][4],mad24((int)p2.s1,f[d.x][5],64))))))/128);
	p4.s0=p4.s1;p4.s1=p4.s2;p4.s2=p4.s3;p4.s3=p2.s0;p2.s0=p2.s1; buf4ui = read_imageui(ref_frame,sampler,c); p2.s1 = (uchar)buf4ui.s0; ++c.x;
	l3.s2 = (uchar)(mad24((int)p4.s0,f[d.x][0],mad24((int)p4.s1,f[d.x][1],mad24((int)p4.s2,f[d.x][2],mad24((int)p4.s3,f[d.x][3],mad24((int)p2.s0,f[d.x][4],mad24((int)p2.s1,f[d.x][5],64))))))/128);
	p4.s0=p4.s1;p4.s1=p4.s2;p4.s2=p4.s3;p4.s3=p2.s0;p2.s0=p2.s1; buf4ui = read_imageui(ref_frame,sampler,c); p2.s1 = (uchar)buf4ui.s0;
	l3.s3 = (uchar)(mad24((int)p4.s0,f[d.x][0],mad24((int)p4.s1,f[d.x][1],mad24((int)p4.s2,f[d.x][2],mad24((int)p4.s3,f[d.x][3],mad24((int)p2.s0,f[d.x][4],mad24((int)p2.s1,f[d.x][5],64))))))/128);
	p4.s0=la0.s0;p4.s1=la1.s0;p4.s2=l0.s0;p4.s3=l1.s0;p2.s0=l2.s0;p2.s1=l3.s0;
	// and produce resulting row
	(*DL1).s0 = (mad24((int)p4.s0,f[d.y][0],mad24((int)p4.s1,f[d.y][1],mad24((int)p4.s2,f[d.y][2],mad24((int)p4.s3,f[d.y][3],mad24((int)p2.s0,f[d.y][4],mad24((int)p2.s1,f[d.y][5],64))))))/128);
	p4.s0=la0.s1;p4.s1=la1.s1;p4.s2=l0.s1;p4.s3=l1.s1;p2.s0=l2.s1;p2.s1=l3.s1;
	(*DL1).s1 = (mad24((int)p4.s0,f[d.y][0],mad24((int)p4.s1,f[d.y][1],mad24((int)p4.s2,f[d.y][2],mad24((int)p4.s3,f[d.y][3],mad24((int)p2.s0,f[d.y][4],mad24((int)p2.s1,f[d.y][5],64))))))/128);
	p4.s0=la0.s2;p4.s1=la1.s2;p4.s2=l0.s2;p4.s3=l1.s2;p2.s0=l2.s2;p2.s1=l3.s2;
	(*DL1).s2 = (mad24((int)p4.s0,f[d.y][0],mad24((int)p4.s1,f[d.y][1],mad24((int)p4.s2,f[d.y][2],mad24((int)p4.s3,f[d.y][3],mad24((int)p2.s0,f[d.y][4],mad24((int)p2.s1,f[d.y][5],64))))))/128);
	p4.s0=la0.s3;p4.s1=la1.s3;p4.s2=l0.s3;p4.s3=l1.s3;p2.s0=l2.s3;p2.s1=l3.s3;
	(*DL1).s3 = (mad24((int)p4.s0,f[d.y][0],mad24((int)p4.s1,f[d.y][1],mad24((int)p4.s2,f[d.y][2],mad24((int)p4.s3,f[d.y][3],mad24((int)p2.s0,f[d.y][4],mad24((int)p2.s1,f[d.y][5],64))))))/128);
	// row 2
	la0 = la1; la1 = l0; l0 = l1; l1 = l2; l2 = l3;
	c.x = coords.x - 2; c.y = coords.y + 5; 
	buf4ui = read_imageui(ref_frame,sampler,c); p4.s0 = (uchar)buf4ui.s0; ++c.x;
	buf4ui = read_imageui(ref_frame,sampler,c); p4.s1 = (uchar)buf4ui.s0; ++c.x;
	buf4ui = read_imageui(ref_frame,sampler,c); p4.s2 = (uchar)buf4ui.s0; ++c.x;
	buf4ui = read_imageui(ref_frame,sampler,c); p4.s3 = (uchar)buf4ui.s0; ++c.x;
	buf4ui = read_imageui(ref_frame,sampler,c); p2.s0 = (uchar)buf4ui.s0; ++c.x;
	buf4ui = read_imageui(ref_frame,sampler,c); p2.s1 = (uchar)buf4ui.s0; ++c.x;
	l3.s0 = (uchar)(mad24((int)p4.s0,f[d.x][0],mad24((int)p4.s1,f[d.x][1],mad24((int)p4.s2,f[d.x][2],mad24((int)p4.s3,f[d.x][3],mad24((int)p2.s0,f[d.x][4],mad24((int)p2.s1,f[d.x][5],64))))))/128);
	p4.s0=p4.s1;p4.s1=p4.s2;p4.s2=p4.s3;p4.s3=p2.s0;p2.s0=p2.s1; buf4ui = read_imageui(ref_frame,sampler,c); p2.s1 = (uchar)buf4ui.s0; ++c.x;
	l3.s1 = (uchar)(mad24((int)p4.s0,f[d.x][0],mad24((int)p4.s1,f[d.x][1],mad24((int)p4.s2,f[d.x][2],mad24((int)p4.s3,f[d.x][3],mad24((int)p2.s0,f[d.x][4],mad24((int)p2.s1,f[d.x][5],64))))))/128);
	p4.s0=p4.s1;p4.s1=p4.s2;p4.s2=p4.s3;p4.s3=p2.s0;p2.s0=p2.s1; buf4ui = read_imageui(ref_frame,sampler,c); p2.s1 = (uchar)buf4ui.s0; ++c.x;
	l3.s2 = (uchar)(mad24((int)p4.s0,f[d.x][0],mad24((int)p4.s1,f[d.x][1],mad24((int)p4.s2,f[d.x][2],mad24((int)p4.s3,f[d.x][3],mad24((int)p2.s0,f[d.x][4],mad24((int)p2.s1,f[d.x][5],64))))))/128);
	p4.s0=p4.s1;p4.s1=p4.s2;p4.s2=p4.s3;p4.s3=p2.s0;p2.s0=p2.s1; buf4ui = read_imageui(ref_frame,sampler,c); p2.s1 = (uchar)buf4ui.s0;
	l3.s3 = (uchar)(mad24((int)p4.s0,f[d.x][0],mad24((int)p4.s1,f[d.x][1],mad24((int)p4.s2,f[d.x][2],mad24((int)p4.s3,f[d.x][3],mad24((int)p2.s0,f[d.x][4],mad24((int)p2.s1,f[d.x][5],64))))))/128);
	p4.s0=la0.s0;p4.s1=la1.s0;p4.s2=l0.s0;p4.s3=l1.s0;p2.s0=l2.s0;p2.s1=l3.s0;
	(*DL2).s0 = (mad24((int)p4.s0,f[d.y][0],mad24((int)p4.s1,f[d.y][1],mad24((int)p4.s2,f[d.y][2],mad24((int)p4.s3,f[d.y][3],mad24((int)p2.s0,f[d.y][4],mad24((int)p2.s1,f[d.y][5],64))))))/128);
	p4.s0=la0.s1;p4.s1=la1.s1;p4.s2=l0.s1;p4.s3=l1.s1;p2.s0=l2.s1;p2.s1=l3.s1;
	(*DL2).s1 = (mad24((int)p4.s0,f[d.y][0],mad24((int)p4.s1,f[d.y][1],mad24((int)p4.s2,f[d.y][2],mad24((int)p4.s3,f[d.y][3],mad24((int)p2.s0,f[d.y][4],mad24((int)p2.s1,f[d.y][5],64))))))/128);
	p4.s0=la0.s2;p4.s1=la1.s2;p4.s2=l0.s2;p4.s3=l1.s2;p2.s0=l2.s2;p2.s1=l3.s2;
	(*DL2).s2 = (mad24((int)p4.s0,f[d.y][0],mad24((int)p4.s1,f[d.y][1],mad24((int)p4.s2,f[d.y][2],mad24((int)p4.s3,f[d.y][3],mad24((int)p2.s0,f[d.y][4],mad24((int)p2.s1,f[d.y][5],64))))))/128);
	p4.s0=la0.s3;p4.s1=la1.s3;p4.s2=l0.s3;p4.s3=l1.s3;p2.s0=l2.s3;p2.s1=l3.s3;
	(*DL2).s3 = (mad24((int)p4.s0,f[d.y][0],mad24((int)p4.s1,f[d.y][1],mad24((int)p4.s2,f[d.y][2],mad24((int)p4.s3,f[d.y][3],mad24((int)p2.s0,f[d.y][4],mad24((int)p2.s1,f[d.y][5],64))))))/128);

	// row 3
	la0 = la1; la1 = l0; l0 = l1; l1 = l2; l2 = l3;
	c.x = coords.x - 2; c.y = coords.y + 6;
	buf4ui = read_imageui(ref_frame,sampler,c); p4.s0 = (uchar)buf4ui.s0; ++c.x;
	buf4ui = read_imageui(ref_frame,sampler,c); p4.s1 = (uchar)buf4ui.s0; ++c.x;
	buf4ui = read_imageui(ref_frame,sampler,c); p4.s2 = (uchar)buf4ui.s0; ++c.x;
	buf4ui = read_imageui(ref_frame,sampler,c); p4.s3 = (uchar)buf4ui.s0; ++c.x;
	buf4ui = read_imageui(ref_frame,sampler,c); p2.s0 = (uchar)buf4ui.s0; ++c.x;
	buf4ui = read_imageui(ref_frame,sampler,c); p2.s1 = (uchar)buf4ui.s0; ++c.x;
	l3.s0 = (uchar)(mad24((int)p4.s0,f[d.x][0],mad24((int)p4.s1,f[d.x][1],mad24((int)p4.s2,f[d.x][2],mad24((int)p4.s3,f[d.x][3],mad24((int)p2.s0,f[d.x][4],mad24((int)p2.s1,f[d.x][5],64))))))/128);
	p4.s0=p4.s1;p4.s1=p4.s2;p4.s2=p4.s3;p4.s3=p2.s0;p2.s0=p2.s1; buf4ui = read_imageui(ref_frame,sampler,c); p2.s1 = (uchar)buf4ui.s0; ++c.x;
	l3.s1 = (uchar)(mad24((int)p4.s0,f[d.x][0],mad24((int)p4.s1,f[d.x][1],mad24((int)p4.s2,f[d.x][2],mad24((int)p4.s3,f[d.x][3],mad24((int)p2.s0,f[d.x][4],mad24((int)p2.s1,f[d.x][5],64))))))/128);
	p4.s0=p4.s1;p4.s1=p4.s2;p4.s2=p4.s3;p4.s3=p2.s0;p2.s0=p2.s1; buf4ui = read_imageui(ref_frame,sampler,c); p2.s1 = (uchar)buf4ui.s0; ++c.x;
	l3.s2 = (uchar)(mad24((int)p4.s0,f[d.x][0],mad24((int)p4.s1,f[d.x][1],mad24((int)p4.s2,f[d.x][2],mad24((int)p4.s3,f[d.x][3],mad24((int)p2.s0,f[d.x][4],mad24((int)p2.s1,f[d.x][5],64))))))/128);
	p4.s0=p4.s1;p4.s1=p4.s2;p4.s2=p4.s3;p4.s3=p2.s0;p2.s0=p2.s1; buf4ui = read_imageui(ref_frame,sampler,c); p2.s1 = (uchar)buf4ui.s0;
	l3.s3 = (uchar)(mad24((int)p4.s0,f[d.x][0],mad24((int)p4.s1,f[d.x][1],mad24((int)p4.s2,f[d.x][2],mad24((int)p4.s3,f[d.x][3],mad24((int)p2.s0,f[d.x][4],mad24((int)p2.s1,f[d.x][5],64))))))/128);
	p4.s0=la0.s0;p4.s1=la1.s0;p4.s2=l0.s0;p4.s3=l1.s0;p2.s0=l2.s0;p2.s1=l3.s0;
	(*DL3).s0 = (mad24((int)p4.s0,f[d.y][0],mad24((int)p4.s1,f[d.y][1],mad24((int)p4.s2,f[d.y][2],mad24((int)p4.s3,f[d.y][3],mad24((int)p2.s0,f[d.y][4],mad24((int)p2.s1,f[d.y][5],64))))))/128);
	p4.s0=la0.s1;p4.s1=la1.s1;p4.s2=l0.s1;p4.s3=l1.s1;p2.s0=l2.s1;p2.s1=l3.s1;
	(*DL3).s1 = (mad24((int)p4.s0,f[d.y][0],mad24((int)p4.s1,f[d.y][1],mad24((int)p4.s2,f[d.y][2],mad24((int)p4.s3,f[d.y][3],mad24((int)p2.s0,f[d.y][4],mad24((int)p2.s1,f[d.y][5],64))))))/128);
	p4.s0=la0.s2;p4.s1=la1.s2;p4.s2=l0.s2;p4.s3=l1.s2;p2.s0=l2.s2;p2.s1=l3.s2;
	(*DL3).s2 = (mad24((int)p4.s0,f[d.y][0],mad24((int)p4.s1,f[d.y][1],mad24((int)p4.s2,f[d.y][2],mad24((int)p4.s3,f[d.y][3],mad24((int)p2.s0,f[d.y][4],mad24((int)p2.s1,f[d.y][5],64))))))/128);
	p4.s0=la0.s3;p4.s1=la1.s3;p4.s2=l0.s3;p4.s3=l1.s3;p2.s0=l2.s3;p2.s1=l3.s3;
	(*DL3).s3 = (mad24((int)p4.s0,f[d.y][0],mad24((int)p4.s1,f[d.y][1],mad24((int)p4.s2,f[d.y][2],mad24((int)p4.s3,f[d.y][3],mad24((int)p2.s0,f[d.y][4],mad24((int)p2.s1,f[d.y][5],64))))))/128);
	
	*DL0 = convert_int4(convert_uchar4_sat(*DL0));
	*DL1 = convert_int4(convert_uchar4_sat(*DL1));
	*DL2 = convert_int4(convert_uchar4_sat(*DL2));
	*DL3 = convert_int4(convert_uchar4_sat(*DL3));
	
	return;
}

void construct_opt(__read_only image2d_t ref_frame, const short2 coords, const short2 d, int16 *const DL,
					uchar16 *const l, uchar16 *const lap4p2, int4 *const li, uint4 *const buf4ui)
{
	__private int2 c;
	
	// read line of 9 pixels and interpolate line of 4 pixels from it
	// we need 9 of these 4-pixel lines
	// line -2
	c.x = coords.x - 2; c.y = coords.y - 2;
	(*buf4ui) = read_imageui(ref_frame,sampler,c); (*lap4p2).s8 = (uchar)(*buf4ui).s0; ++c.x;
	(*buf4ui) = read_imageui(ref_frame,sampler,c); (*lap4p2).s9 = (uchar)(*buf4ui).s0; ++c.x;
	(*buf4ui) = read_imageui(ref_frame,sampler,c); (*lap4p2).sA = (uchar)(*buf4ui).s0; ++c.x;
	(*buf4ui) = read_imageui(ref_frame,sampler,c); (*lap4p2).sB = (uchar)(*buf4ui).s0; ++c.x;
	(*buf4ui) = read_imageui(ref_frame,sampler,c); (*lap4p2).sC = (uchar)(*buf4ui).s0; ++c.x;
	(*buf4ui) = read_imageui(ref_frame,sampler,c); (*lap4p2).sD = (uchar)(*buf4ui).s0; ++c.x;
	(*li).s0 = (mad24((int)(*lap4p2).s8,f[d.x][0],mad24((int)(*lap4p2).s9,f[d.x][1],mad24((int)(*lap4p2).sA,f[d.x][2],mad24((int)(*lap4p2).sB,f[d.x][3],mad24((int)(*lap4p2).sC,f[d.x][4],mad24((int)(*lap4p2).sD,f[d.x][5],64))))))/128);
	(*lap4p2).s8=(*lap4p2).s9;(*lap4p2).s9=(*lap4p2).sA;(*lap4p2).sA=(*lap4p2).sB;(*lap4p2).sB=(*lap4p2).sC;(*lap4p2).sC=(*lap4p2).sD; (*buf4ui) = read_imageui(ref_frame,sampler,c); (*lap4p2).sD = (uchar)(*buf4ui).s0; ++c.x;
	(*li).s1 = (mad24((int)(*lap4p2).s8,f[d.x][0],mad24((int)(*lap4p2).s9,f[d.x][1],mad24((int)(*lap4p2).sA,f[d.x][2],mad24((int)(*lap4p2).sB,f[d.x][3],mad24((int)(*lap4p2).sC,f[d.x][4],mad24((int)(*lap4p2).sD,f[d.x][5],64))))))/128);
	(*lap4p2).s8=(*lap4p2).s9;(*lap4p2).s9=(*lap4p2).sA;(*lap4p2).sA=(*lap4p2).sB;(*lap4p2).sB=(*lap4p2).sC;(*lap4p2).sC=(*lap4p2).sD; (*buf4ui) = read_imageui(ref_frame,sampler,c); (*lap4p2).sD = (uchar)(*buf4ui).s0; ++c.x;
	(*li).s2 = (mad24((int)(*lap4p2).s8,f[d.x][0],mad24((int)(*lap4p2).s9,f[d.x][1],mad24((int)(*lap4p2).sA,f[d.x][2],mad24((int)(*lap4p2).sB,f[d.x][3],mad24((int)(*lap4p2).sC,f[d.x][4],mad24((int)(*lap4p2).sD,f[d.x][5],64))))))/128);
	(*lap4p2).s8=(*lap4p2).s9;(*lap4p2).s9=(*lap4p2).sA;(*lap4p2).sA=(*lap4p2).sB;(*lap4p2).sB=(*lap4p2).sC;(*lap4p2).sC=(*lap4p2).sD; (*buf4ui) = read_imageui(ref_frame,sampler,c); (*lap4p2).sD = (uchar)(*buf4ui).s0;
	(*li).s3 = (mad24((int)(*lap4p2).s8,f[d.x][0],mad24((int)(*lap4p2).s9,f[d.x][1],mad24((int)(*lap4p2).sA,f[d.x][2],mad24((int)(*lap4p2).sB,f[d.x][3],mad24((int)(*lap4p2).sC,f[d.x][4],mad24((int)(*lap4p2).sD,f[d.x][5],64))))))/128);
	(*lap4p2).s0123 = convert_uchar4_sat((*li));
	// line -1
	c.x = coords.x - 2; c.y = coords.y - 1;
	(*buf4ui) = read_imageui(ref_frame,sampler,c); (*lap4p2).s8 = (uchar)(*buf4ui).s0; ++c.x;
	(*buf4ui) = read_imageui(ref_frame,sampler,c); (*lap4p2).s9 = (uchar)(*buf4ui).s0; ++c.x;
	(*buf4ui) = read_imageui(ref_frame,sampler,c); (*lap4p2).sA = (uchar)(*buf4ui).s0; ++c.x;
	(*buf4ui) = read_imageui(ref_frame,sampler,c); (*lap4p2).sB = (uchar)(*buf4ui).s0; ++c.x;
	(*buf4ui) = read_imageui(ref_frame,sampler,c); (*lap4p2).sC = (uchar)(*buf4ui).s0; ++c.x;
	(*buf4ui) = read_imageui(ref_frame,sampler,c); (*lap4p2).sD = (uchar)(*buf4ui).s0; ++c.x;
	(*li).s0 = (mad24((int)(*lap4p2).s8,f[d.x][0],mad24((int)(*lap4p2).s9,f[d.x][1],mad24((int)(*lap4p2).sA,f[d.x][2],mad24((int)(*lap4p2).sB,f[d.x][3],mad24((int)(*lap4p2).sC,f[d.x][4],mad24((int)(*lap4p2).sD,f[d.x][5],64))))))/128);
	(*lap4p2).s8=(*lap4p2).s9;(*lap4p2).s9=(*lap4p2).sA;(*lap4p2).sA=(*lap4p2).sB;(*lap4p2).sB=(*lap4p2).sC;(*lap4p2).sC=(*lap4p2).sD; (*buf4ui) = read_imageui(ref_frame,sampler,c); (*lap4p2).sD = (uchar)(*buf4ui).s0; ++c.x;
	(*li).s1 = (mad24((int)(*lap4p2).s8,f[d.x][0],mad24((int)(*lap4p2).s9,f[d.x][1],mad24((int)(*lap4p2).sA,f[d.x][2],mad24((int)(*lap4p2).sB,f[d.x][3],mad24((int)(*lap4p2).sC,f[d.x][4],mad24((int)(*lap4p2).sD,f[d.x][5],64))))))/128);
	(*lap4p2).s8=(*lap4p2).s9;(*lap4p2).s9=(*lap4p2).sA;(*lap4p2).sA=(*lap4p2).sB;(*lap4p2).sB=(*lap4p2).sC;(*lap4p2).sC=(*lap4p2).sD; (*buf4ui) = read_imageui(ref_frame,sampler,c); (*lap4p2).sD = (uchar)(*buf4ui).s0; ++c.x;
	(*li).s2 = (mad24((int)(*lap4p2).s8,f[d.x][0],mad24((int)(*lap4p2).s9,f[d.x][1],mad24((int)(*lap4p2).sA,f[d.x][2],mad24((int)(*lap4p2).sB,f[d.x][3],mad24((int)(*lap4p2).sC,f[d.x][4],mad24((int)(*lap4p2).sD,f[d.x][5],64))))))/128);
	(*lap4p2).s8=(*lap4p2).s9;(*lap4p2).s9=(*lap4p2).sA;(*lap4p2).sA=(*lap4p2).sB;(*lap4p2).sB=(*lap4p2).sC;(*lap4p2).sC=(*lap4p2).sD; (*buf4ui) = read_imageui(ref_frame,sampler,c); (*lap4p2).sD = (uchar)(*buf4ui).s0;
	(*li).s3 = (mad24((int)(*lap4p2).s8,f[d.x][0],mad24((int)(*lap4p2).s9,f[d.x][1],mad24((int)(*lap4p2).sA,f[d.x][2],mad24((int)(*lap4p2).sB,f[d.x][3],mad24((int)(*lap4p2).sC,f[d.x][4],mad24((int)(*lap4p2).sD,f[d.x][5],64))))))/128);
	(*lap4p2).s4567 = convert_uchar4_sat((*li));
	// line 0
	c.x = coords.x - 2; c.y = coords.y;
	(*buf4ui) = read_imageui(ref_frame,sampler,c); (*lap4p2).s8 = (uchar)(*buf4ui).s0; ++c.x;
	(*buf4ui) = read_imageui(ref_frame,sampler,c); (*lap4p2).s9 = (uchar)(*buf4ui).s0; ++c.x;
	(*buf4ui) = read_imageui(ref_frame,sampler,c); (*lap4p2).sA = (uchar)(*buf4ui).s0; ++c.x;
	(*buf4ui) = read_imageui(ref_frame,sampler,c); (*lap4p2).sB = (uchar)(*buf4ui).s0; ++c.x;
	(*buf4ui) = read_imageui(ref_frame,sampler,c); (*lap4p2).sC = (uchar)(*buf4ui).s0; ++c.x;
	(*buf4ui) = read_imageui(ref_frame,sampler,c); (*lap4p2).sD = (uchar)(*buf4ui).s0; ++c.x;
	(*li).s0 = (mad24((int)(*lap4p2).s8,f[d.x][0],mad24((int)(*lap4p2).s9,f[d.x][1],mad24((int)(*lap4p2).sA,f[d.x][2],mad24((int)(*lap4p2).sB,f[d.x][3],mad24((int)(*lap4p2).sC,f[d.x][4],mad24((int)(*lap4p2).sD,f[d.x][5],64))))))/128);
	(*lap4p2).s8=(*lap4p2).s9;(*lap4p2).s9=(*lap4p2).sA;(*lap4p2).sA=(*lap4p2).sB;(*lap4p2).sB=(*lap4p2).sC;(*lap4p2).sC=(*lap4p2).sD; (*buf4ui) = read_imageui(ref_frame,sampler,c); (*lap4p2).sD = (uchar)(*buf4ui).s0; ++c.x;
	(*li).s1 = (mad24((int)(*lap4p2).s8,f[d.x][0],mad24((int)(*lap4p2).s9,f[d.x][1],mad24((int)(*lap4p2).sA,f[d.x][2],mad24((int)(*lap4p2).sB,f[d.x][3],mad24((int)(*lap4p2).sC,f[d.x][4],mad24((int)(*lap4p2).sD,f[d.x][5],64))))))/128);
	(*lap4p2).s8=(*lap4p2).s9;(*lap4p2).s9=(*lap4p2).sA;(*lap4p2).sA=(*lap4p2).sB;(*lap4p2).sB=(*lap4p2).sC;(*lap4p2).sC=(*lap4p2).sD; (*buf4ui) = read_imageui(ref_frame,sampler,c); (*lap4p2).sD = (uchar)(*buf4ui).s0; ++c.x;
	(*li).s2 = (mad24((int)(*lap4p2).s8,f[d.x][0],mad24((int)(*lap4p2).s9,f[d.x][1],mad24((int)(*lap4p2).sA,f[d.x][2],mad24((int)(*lap4p2).sB,f[d.x][3],mad24((int)(*lap4p2).sC,f[d.x][4],mad24((int)(*lap4p2).sD,f[d.x][5],64))))))/128);
	(*lap4p2).s8=(*lap4p2).s9;(*lap4p2).s9=(*lap4p2).sA;(*lap4p2).sA=(*lap4p2).sB;(*lap4p2).sB=(*lap4p2).sC;(*lap4p2).sC=(*lap4p2).sD; (*buf4ui) = read_imageui(ref_frame,sampler,c); (*lap4p2).sD = (uchar)(*buf4ui).s0;
	(*li).s3 = (mad24((int)(*lap4p2).s8,f[d.x][0],mad24((int)(*lap4p2).s9,f[d.x][1],mad24((int)(*lap4p2).sA,f[d.x][2],mad24((int)(*lap4p2).sB,f[d.x][3],mad24((int)(*lap4p2).sC,f[d.x][4],mad24((int)(*lap4p2).sD,f[d.x][5],64))))))/128);
	(*l).s0123 = convert_uchar4_sat((*li));
	// line 1
	c.x = coords.x - 2; c.y = coords.y + 1;
	(*buf4ui) = read_imageui(ref_frame,sampler,c); (*lap4p2).s8 = (uchar)(*buf4ui).s0; ++c.x;
	(*buf4ui) = read_imageui(ref_frame,sampler,c); (*lap4p2).s9 = (uchar)(*buf4ui).s0; ++c.x;
	(*buf4ui) = read_imageui(ref_frame,sampler,c); (*lap4p2).sA = (uchar)(*buf4ui).s0; ++c.x;
	(*buf4ui) = read_imageui(ref_frame,sampler,c); (*lap4p2).sB = (uchar)(*buf4ui).s0; ++c.x;
	(*buf4ui) = read_imageui(ref_frame,sampler,c); (*lap4p2).sC = (uchar)(*buf4ui).s0; ++c.x;
	(*buf4ui) = read_imageui(ref_frame,sampler,c); (*lap4p2).sD = (uchar)(*buf4ui).s0; ++c.x;
	(*li).s0 = (mad24((int)(*lap4p2).s8,f[d.x][0],mad24((int)(*lap4p2).s9,f[d.x][1],mad24((int)(*lap4p2).sA,f[d.x][2],mad24((int)(*lap4p2).sB,f[d.x][3],mad24((int)(*lap4p2).sC,f[d.x][4],mad24((int)(*lap4p2).sD,f[d.x][5],64))))))/128);
	(*lap4p2).s8=(*lap4p2).s9;(*lap4p2).s9=(*lap4p2).sA;(*lap4p2).sA=(*lap4p2).sB;(*lap4p2).sB=(*lap4p2).sC;(*lap4p2).sC=(*lap4p2).sD; (*buf4ui) = read_imageui(ref_frame,sampler,c); (*lap4p2).sD = (uchar)(*buf4ui).s0; ++c.x;
	(*li).s1 = (mad24((int)(*lap4p2).s8,f[d.x][0],mad24((int)(*lap4p2).s9,f[d.x][1],mad24((int)(*lap4p2).sA,f[d.x][2],mad24((int)(*lap4p2).sB,f[d.x][3],mad24((int)(*lap4p2).sC,f[d.x][4],mad24((int)(*lap4p2).sD,f[d.x][5],64))))))/128);
	(*lap4p2).s8=(*lap4p2).s9;(*lap4p2).s9=(*lap4p2).sA;(*lap4p2).sA=(*lap4p2).sB;(*lap4p2).sB=(*lap4p2).sC;(*lap4p2).sC=(*lap4p2).sD; (*buf4ui) = read_imageui(ref_frame,sampler,c); (*lap4p2).sD = (uchar)(*buf4ui).s0; ++c.x;
	(*li).s2 = (mad24((int)(*lap4p2).s8,f[d.x][0],mad24((int)(*lap4p2).s9,f[d.x][1],mad24((int)(*lap4p2).sA,f[d.x][2],mad24((int)(*lap4p2).sB,f[d.x][3],mad24((int)(*lap4p2).sC,f[d.x][4],mad24((int)(*lap4p2).sD,f[d.x][5],64))))))/128);
	(*lap4p2).s8=(*lap4p2).s9;(*lap4p2).s9=(*lap4p2).sA;(*lap4p2).sA=(*lap4p2).sB;(*lap4p2).sB=(*lap4p2).sC;(*lap4p2).sC=(*lap4p2).sD; (*buf4ui) = read_imageui(ref_frame,sampler,c); (*lap4p2).sD = (uchar)(*buf4ui).s0;
	(*li).s3 = (mad24((int)(*lap4p2).s8,f[d.x][0],mad24((int)(*lap4p2).s9,f[d.x][1],mad24((int)(*lap4p2).sA,f[d.x][2],mad24((int)(*lap4p2).sB,f[d.x][3],mad24((int)(*lap4p2).sC,f[d.x][4],mad24((int)(*lap4p2).sD,f[d.x][5],64))))))/128);
	(*l).s4567 = convert_uchar4_sat((*li));
	// line 2
	c.x = coords.x - 2; c.y = coords.y + 2;
	(*buf4ui) = read_imageui(ref_frame,sampler,c); (*lap4p2).s8 = (uchar)(*buf4ui).s0; ++c.x;
	(*buf4ui) = read_imageui(ref_frame,sampler,c); (*lap4p2).s9 = (uchar)(*buf4ui).s0; ++c.x;
	(*buf4ui) = read_imageui(ref_frame,sampler,c); (*lap4p2).sA = (uchar)(*buf4ui).s0; ++c.x;
	(*buf4ui) = read_imageui(ref_frame,sampler,c); (*lap4p2).sB = (uchar)(*buf4ui).s0; ++c.x;
	(*buf4ui) = read_imageui(ref_frame,sampler,c); (*lap4p2).sC = (uchar)(*buf4ui).s0; ++c.x;
	(*buf4ui) = read_imageui(ref_frame,sampler,c); (*lap4p2).sD = (uchar)(*buf4ui).s0; ++c.x;
	(*li).s0 = (mad24((int)(*lap4p2).s8,f[d.x][0],mad24((int)(*lap4p2).s9,f[d.x][1],mad24((int)(*lap4p2).sA,f[d.x][2],mad24((int)(*lap4p2).sB,f[d.x][3],mad24((int)(*lap4p2).sC,f[d.x][4],mad24((int)(*lap4p2).sD,f[d.x][5],64))))))/128);
	(*lap4p2).s8=(*lap4p2).s9;(*lap4p2).s9=(*lap4p2).sA;(*lap4p2).sA=(*lap4p2).sB;(*lap4p2).sB=(*lap4p2).sC;(*lap4p2).sC=(*lap4p2).sD; (*buf4ui) = read_imageui(ref_frame,sampler,c); (*lap4p2).sD = (uchar)(*buf4ui).s0; ++c.x;
	(*li).s1 = (mad24((int)(*lap4p2).s8,f[d.x][0],mad24((int)(*lap4p2).s9,f[d.x][1],mad24((int)(*lap4p2).sA,f[d.x][2],mad24((int)(*lap4p2).sB,f[d.x][3],mad24((int)(*lap4p2).sC,f[d.x][4],mad24((int)(*lap4p2).sD,f[d.x][5],64))))))/128);
	(*lap4p2).s8=(*lap4p2).s9;(*lap4p2).s9=(*lap4p2).sA;(*lap4p2).sA=(*lap4p2).sB;(*lap4p2).sB=(*lap4p2).sC;(*lap4p2).sC=(*lap4p2).sD; (*buf4ui) = read_imageui(ref_frame,sampler,c); (*lap4p2).sD = (uchar)(*buf4ui).s0; ++c.x;
	(*li).s2 = (mad24((int)(*lap4p2).s8,f[d.x][0],mad24((int)(*lap4p2).s9,f[d.x][1],mad24((int)(*lap4p2).sA,f[d.x][2],mad24((int)(*lap4p2).sB,f[d.x][3],mad24((int)(*lap4p2).sC,f[d.x][4],mad24((int)(*lap4p2).sD,f[d.x][5],64))))))/128);
	(*lap4p2).s8=(*lap4p2).s9;(*lap4p2).s9=(*lap4p2).sA;(*lap4p2).sA=(*lap4p2).sB;(*lap4p2).sB=(*lap4p2).sC;(*lap4p2).sC=(*lap4p2).sD; (*buf4ui) = read_imageui(ref_frame,sampler,c); (*lap4p2).sD = (uchar)(*buf4ui).s0;
	(*li).s3 = (mad24((int)(*lap4p2).s8,f[d.x][0],mad24((int)(*lap4p2).s9,f[d.x][1],mad24((int)(*lap4p2).sA,f[d.x][2],mad24((int)(*lap4p2).sB,f[d.x][3],mad24((int)(*lap4p2).sC,f[d.x][4],mad24((int)(*lap4p2).sD,f[d.x][5],64))))))/128);
	(*l).s89AB = convert_uchar4_sat((*li));
	// line 3
	c.x = coords.x - 2; c.y = coords.y + 3; 
	(*buf4ui) = read_imageui(ref_frame,sampler,c); (*lap4p2).s8 = (uchar)(*buf4ui).s0; ++c.x;
	(*buf4ui) = read_imageui(ref_frame,sampler,c); (*lap4p2).s9 = (uchar)(*buf4ui).s0; ++c.x;
	(*buf4ui) = read_imageui(ref_frame,sampler,c); (*lap4p2).sA = (uchar)(*buf4ui).s0; ++c.x;
	(*buf4ui) = read_imageui(ref_frame,sampler,c); (*lap4p2).sB = (uchar)(*buf4ui).s0; ++c.x;
	(*buf4ui) = read_imageui(ref_frame,sampler,c); (*lap4p2).sC = (uchar)(*buf4ui).s0; ++c.x;
	(*buf4ui) = read_imageui(ref_frame,sampler,c); (*lap4p2).sD = (uchar)(*buf4ui).s0; ++c.x;
	(*li).s0 = (mad24((int)(*lap4p2).s8,f[d.x][0],mad24((int)(*lap4p2).s9,f[d.x][1],mad24((int)(*lap4p2).sA,f[d.x][2],mad24((int)(*lap4p2).sB,f[d.x][3],mad24((int)(*lap4p2).sC,f[d.x][4],mad24((int)(*lap4p2).sD,f[d.x][5],64))))))/128);
	(*lap4p2).s8=(*lap4p2).s9;(*lap4p2).s9=(*lap4p2).sA;(*lap4p2).sA=(*lap4p2).sB;(*lap4p2).sB=(*lap4p2).sC;(*lap4p2).sC=(*lap4p2).sD; (*buf4ui) = read_imageui(ref_frame,sampler,c); (*lap4p2).sD = (uchar)(*buf4ui).s0; ++c.x;
	(*li).s1 = (mad24((int)(*lap4p2).s8,f[d.x][0],mad24((int)(*lap4p2).s9,f[d.x][1],mad24((int)(*lap4p2).sA,f[d.x][2],mad24((int)(*lap4p2).sB,f[d.x][3],mad24((int)(*lap4p2).sC,f[d.x][4],mad24((int)(*lap4p2).sD,f[d.x][5],64))))))/128);
	(*lap4p2).s8=(*lap4p2).s9;(*lap4p2).s9=(*lap4p2).sA;(*lap4p2).sA=(*lap4p2).sB;(*lap4p2).sB=(*lap4p2).sC;(*lap4p2).sC=(*lap4p2).sD; (*buf4ui) = read_imageui(ref_frame,sampler,c); (*lap4p2).sD = (uchar)(*buf4ui).s0; ++c.x;
	(*li).s2 = (mad24((int)(*lap4p2).s8,f[d.x][0],mad24((int)(*lap4p2).s9,f[d.x][1],mad24((int)(*lap4p2).sA,f[d.x][2],mad24((int)(*lap4p2).sB,f[d.x][3],mad24((int)(*lap4p2).sC,f[d.x][4],mad24((int)(*lap4p2).sD,f[d.x][5],64))))))/128);
	(*lap4p2).s8=(*lap4p2).s9;(*lap4p2).s9=(*lap4p2).sA;(*lap4p2).sA=(*lap4p2).sB;(*lap4p2).sB=(*lap4p2).sC;(*lap4p2).sC=(*lap4p2).sD; (*buf4ui) = read_imageui(ref_frame,sampler,c); (*lap4p2).sD = (uchar)(*buf4ui).s0;
	(*li).s3 = (mad24((int)(*lap4p2).s8,f[d.x][0],mad24((int)(*lap4p2).s9,f[d.x][1],mad24((int)(*lap4p2).sA,f[d.x][2],mad24((int)(*lap4p2).sB,f[d.x][3],mad24((int)(*lap4p2).sC,f[d.x][4],mad24((int)(*lap4p2).sD,f[d.x][5],64))))))/128);
	(*l).sCDEF = convert_uchar4_sat((*li));
	
	// now we interpolate in collumns
	// row 0
	(*lap4p2).s8=(*lap4p2).s0;(*lap4p2).s9=(*lap4p2).s4;(*lap4p2).sA=(*l).s0;(*lap4p2).sB=(*l).s4;(*lap4p2).sC=(*l).s8;(*lap4p2).sD=(*l).sC;
	(*DL).s0 = (mad24((int)(*lap4p2).s8,f[d.y][0],mad24((int)(*lap4p2).s9,f[d.y][1],mad24((int)(*lap4p2).sA,f[d.y][2],mad24((int)(*lap4p2).sB,f[d.y][3],mad24((int)(*lap4p2).sC,f[d.y][4],mad24((int)(*lap4p2).sD,f[d.y][5],64))))))/128);
	(*lap4p2).s8=(*lap4p2).s1;(*lap4p2).s9=(*lap4p2).s5;(*lap4p2).sA=(*l).s1;(*lap4p2).sB=(*l).s5;(*lap4p2).sC=(*l).s9;(*lap4p2).sD=(*l).sD;
	(*DL).s1 = (mad24((int)(*lap4p2).s8,f[d.y][0],mad24((int)(*lap4p2).s9,f[d.y][1],mad24((int)(*lap4p2).sA,f[d.y][2],mad24((int)(*lap4p2).sB,f[d.y][3],mad24((int)(*lap4p2).sC,f[d.y][4],mad24((int)(*lap4p2).sD,f[d.y][5],64))))))/128);
	(*lap4p2).s8=(*lap4p2).s2;(*lap4p2).s9=(*lap4p2).s6;(*lap4p2).sA=(*l).s2;(*lap4p2).sB=(*l).s6;(*lap4p2).sC=(*l).sA;(*lap4p2).sD=(*l).sE;
	(*DL).s2 = (mad24((int)(*lap4p2).s8,f[d.y][0],mad24((int)(*lap4p2).s9,f[d.y][1],mad24((int)(*lap4p2).sA,f[d.y][2],mad24((int)(*lap4p2).sB,f[d.y][3],mad24((int)(*lap4p2).sC,f[d.y][4],mad24((int)(*lap4p2).sD,f[d.y][5],64))))))/128);
	(*lap4p2).s8=(*lap4p2).s3;(*lap4p2).s9=(*lap4p2).s7;(*lap4p2).sA=(*l).s3;(*lap4p2).sB=(*l).s7;(*lap4p2).sC=(*l).sB;(*lap4p2).sD=(*l).sF;
	(*DL).s3 = (mad24((int)(*lap4p2).s8,f[d.y][0],mad24((int)(*lap4p2).s9,f[d.y][1],mad24((int)(*lap4p2).sA,f[d.y][2],mad24((int)(*lap4p2).sB,f[d.y][3],mad24((int)(*lap4p2).sC,f[d.y][4],mad24((int)(*lap4p2).sD,f[d.y][5],64))))))/128);
	// row 1 
	(*lap4p2).s0123 = (*lap4p2).s4567; (*lap4p2).s4567 = (*l).s0123; (*l).s0123 = (*l).s4567; (*l).s4567 = (*l).s89AB; (*l).s89AB = (*l).sCDEF;
	// first produce line l3
	c.x = coords.x - 2; c.y = coords.y + 4; 
	(*buf4ui) = read_imageui(ref_frame,sampler,c); (*lap4p2).s8 = (uchar)(*buf4ui).s0; ++c.x;
	(*buf4ui) = read_imageui(ref_frame,sampler,c); (*lap4p2).s9 = (uchar)(*buf4ui).s0; ++c.x;
	(*buf4ui) = read_imageui(ref_frame,sampler,c); (*lap4p2).sA = (uchar)(*buf4ui).s0; ++c.x;
	(*buf4ui) = read_imageui(ref_frame,sampler,c); (*lap4p2).sB = (uchar)(*buf4ui).s0; ++c.x;
	(*buf4ui) = read_imageui(ref_frame,sampler,c); (*lap4p2).sC = (uchar)(*buf4ui).s0; ++c.x;
	(*buf4ui) = read_imageui(ref_frame,sampler,c); (*lap4p2).sD = (uchar)(*buf4ui).s0; ++c.x;
	(*li).s0 = (mad24((int)(*lap4p2).s8,f[d.x][0],mad24((int)(*lap4p2).s9,f[d.x][1],mad24((int)(*lap4p2).sA,f[d.x][2],mad24((int)(*lap4p2).sB,f[d.x][3],mad24((int)(*lap4p2).sC,f[d.x][4],mad24((int)(*lap4p2).sD,f[d.x][5],64))))))/128);
	(*lap4p2).s8=(*lap4p2).s9;(*lap4p2).s9=(*lap4p2).sA;(*lap4p2).sA=(*lap4p2).sB;(*lap4p2).sB=(*lap4p2).sC;(*lap4p2).sC=(*lap4p2).sD; (*buf4ui) = read_imageui(ref_frame,sampler,c); (*lap4p2).sD = (uchar)(*buf4ui).s0; ++c.x;
	(*li).s1 = (mad24((int)(*lap4p2).s8,f[d.x][0],mad24((int)(*lap4p2).s9,f[d.x][1],mad24((int)(*lap4p2).sA,f[d.x][2],mad24((int)(*lap4p2).sB,f[d.x][3],mad24((int)(*lap4p2).sC,f[d.x][4],mad24((int)(*lap4p2).sD,f[d.x][5],64))))))/128);
	(*lap4p2).s8=(*lap4p2).s9;(*lap4p2).s9=(*lap4p2).sA;(*lap4p2).sA=(*lap4p2).sB;(*lap4p2).sB=(*lap4p2).sC;(*lap4p2).sC=(*lap4p2).sD; (*buf4ui) = read_imageui(ref_frame,sampler,c); (*lap4p2).sD = (uchar)(*buf4ui).s0; ++c.x;
	(*li).s2 = (mad24((int)(*lap4p2).s8,f[d.x][0],mad24((int)(*lap4p2).s9,f[d.x][1],mad24((int)(*lap4p2).sA,f[d.x][2],mad24((int)(*lap4p2).sB,f[d.x][3],mad24((int)(*lap4p2).sC,f[d.x][4],mad24((int)(*lap4p2).sD,f[d.x][5],64))))))/128);
	(*lap4p2).s8=(*lap4p2).s9;(*lap4p2).s9=(*lap4p2).sA;(*lap4p2).sA=(*lap4p2).sB;(*lap4p2).sB=(*lap4p2).sC;(*lap4p2).sC=(*lap4p2).sD; (*buf4ui) = read_imageui(ref_frame,sampler,c); (*lap4p2).sD = (uchar)(*buf4ui).s0;
	(*li).s3 = (mad24((int)(*lap4p2).s8,f[d.x][0],mad24((int)(*lap4p2).s9,f[d.x][1],mad24((int)(*lap4p2).sA,f[d.x][2],mad24((int)(*lap4p2).sB,f[d.x][3],mad24((int)(*lap4p2).sC,f[d.x][4],mad24((int)(*lap4p2).sD,f[d.x][5],64))))))/128);
	(*l).sCDEF = convert_uchar4_sat((*li));
	// and produce resulting row
	(*lap4p2).s8=(*lap4p2).s0;(*lap4p2).s9=(*lap4p2).s4;(*lap4p2).sA=(*l).s0;(*lap4p2).sB=(*l).s4;(*lap4p2).sC=(*l).s8;(*lap4p2).sD=(*l).sC;
	(*DL).s4 = (mad24((int)(*lap4p2).s8,f[d.y][0],mad24((int)(*lap4p2).s9,f[d.y][1],mad24((int)(*lap4p2).sA,f[d.y][2],mad24((int)(*lap4p2).sB,f[d.y][3],mad24((int)(*lap4p2).sC,f[d.y][4],mad24((int)(*lap4p2).sD,f[d.y][5],64))))))/128);
	(*lap4p2).s8=(*lap4p2).s1;(*lap4p2).s9=(*lap4p2).s5;(*lap4p2).sA=(*l).s1;(*lap4p2).sB=(*l).s5;(*lap4p2).sC=(*l).s9;(*lap4p2).sD=(*l).sD;
	(*DL).s5 = (mad24((int)(*lap4p2).s8,f[d.y][0],mad24((int)(*lap4p2).s9,f[d.y][1],mad24((int)(*lap4p2).sA,f[d.y][2],mad24((int)(*lap4p2).sB,f[d.y][3],mad24((int)(*lap4p2).sC,f[d.y][4],mad24((int)(*lap4p2).sD,f[d.y][5],64))))))/128);
	(*lap4p2).s8=(*lap4p2).s2;(*lap4p2).s9=(*lap4p2).s6;(*lap4p2).sA=(*l).s2;(*lap4p2).sB=(*l).s6;(*lap4p2).sC=(*l).sA;(*lap4p2).sD=(*l).sE;
	(*DL).s6 = (mad24((int)(*lap4p2).s8,f[d.y][0],mad24((int)(*lap4p2).s9,f[d.y][1],mad24((int)(*lap4p2).sA,f[d.y][2],mad24((int)(*lap4p2).sB,f[d.y][3],mad24((int)(*lap4p2).sC,f[d.y][4],mad24((int)(*lap4p2).sD,f[d.y][5],64))))))/128);
	(*lap4p2).s8=(*lap4p2).s3;(*lap4p2).s9=(*lap4p2).s7;(*lap4p2).sA=(*l).s3;(*lap4p2).sB=(*l).s7;(*lap4p2).sC=(*l).sB;(*lap4p2).sD=(*l).sF;
	(*DL).s7 = (mad24((int)(*lap4p2).s8,f[d.y][0],mad24((int)(*lap4p2).s9,f[d.y][1],mad24((int)(*lap4p2).sA,f[d.y][2],mad24((int)(*lap4p2).sB,f[d.y][3],mad24((int)(*lap4p2).sC,f[d.y][4],mad24((int)(*lap4p2).sD,f[d.y][5],64))))))/128);
	// row 2
	(*lap4p2).s0123 = (*lap4p2).s4567; (*lap4p2).s4567 = (*l).s0123; (*l).s0123 = (*l).s4567; (*l).s4567 = (*l).s89AB; (*l).s89AB = (*l).sCDEF;
	c.x = coords.x - 2; c.y = coords.y + 5; 
	(*buf4ui) = read_imageui(ref_frame,sampler,c); (*lap4p2).s8 = (uchar)(*buf4ui).s0; ++c.x;
	(*buf4ui) = read_imageui(ref_frame,sampler,c); (*lap4p2).s9 = (uchar)(*buf4ui).s0; ++c.x;
	(*buf4ui) = read_imageui(ref_frame,sampler,c); (*lap4p2).sA = (uchar)(*buf4ui).s0; ++c.x;
	(*buf4ui) = read_imageui(ref_frame,sampler,c); (*lap4p2).sB = (uchar)(*buf4ui).s0; ++c.x;
	(*buf4ui) = read_imageui(ref_frame,sampler,c); (*lap4p2).sC = (uchar)(*buf4ui).s0; ++c.x;
	(*buf4ui) = read_imageui(ref_frame,sampler,c); (*lap4p2).sD = (uchar)(*buf4ui).s0; ++c.x;
	(*li).s0 = (mad24((int)(*lap4p2).s8,f[d.x][0],mad24((int)(*lap4p2).s9,f[d.x][1],mad24((int)(*lap4p2).sA,f[d.x][2],mad24((int)(*lap4p2).sB,f[d.x][3],mad24((int)(*lap4p2).sC,f[d.x][4],mad24((int)(*lap4p2).sD,f[d.x][5],64))))))/128);
	(*lap4p2).s8=(*lap4p2).s9;(*lap4p2).s9=(*lap4p2).sA;(*lap4p2).sA=(*lap4p2).sB;(*lap4p2).sB=(*lap4p2).sC;(*lap4p2).sC=(*lap4p2).sD; (*buf4ui) = read_imageui(ref_frame,sampler,c); (*lap4p2).sD = (uchar)(*buf4ui).s0; ++c.x;
	(*li).s1 = (mad24((int)(*lap4p2).s8,f[d.x][0],mad24((int)(*lap4p2).s9,f[d.x][1],mad24((int)(*lap4p2).sA,f[d.x][2],mad24((int)(*lap4p2).sB,f[d.x][3],mad24((int)(*lap4p2).sC,f[d.x][4],mad24((int)(*lap4p2).sD,f[d.x][5],64))))))/128);
	(*lap4p2).s8=(*lap4p2).s9;(*lap4p2).s9=(*lap4p2).sA;(*lap4p2).sA=(*lap4p2).sB;(*lap4p2).sB=(*lap4p2).sC;(*lap4p2).sC=(*lap4p2).sD; (*buf4ui) = read_imageui(ref_frame,sampler,c); (*lap4p2).sD = (uchar)(*buf4ui).s0; ++c.x;
	(*li).s2 = (mad24((int)(*lap4p2).s8,f[d.x][0],mad24((int)(*lap4p2).s9,f[d.x][1],mad24((int)(*lap4p2).sA,f[d.x][2],mad24((int)(*lap4p2).sB,f[d.x][3],mad24((int)(*lap4p2).sC,f[d.x][4],mad24((int)(*lap4p2).sD,f[d.x][5],64))))))/128);
	(*lap4p2).s8=(*lap4p2).s9;(*lap4p2).s9=(*lap4p2).sA;(*lap4p2).sA=(*lap4p2).sB;(*lap4p2).sB=(*lap4p2).sC;(*lap4p2).sC=(*lap4p2).sD; (*buf4ui) = read_imageui(ref_frame,sampler,c); (*lap4p2).sD = (uchar)(*buf4ui).s0;
	(*li).s3 = (mad24((int)(*lap4p2).s8,f[d.x][0],mad24((int)(*lap4p2).s9,f[d.x][1],mad24((int)(*lap4p2).sA,f[d.x][2],mad24((int)(*lap4p2).sB,f[d.x][3],mad24((int)(*lap4p2).sC,f[d.x][4],mad24((int)(*lap4p2).sD,f[d.x][5],64))))))/128);
	(*l).sCDEF = convert_uchar4_sat((*li));
	(*lap4p2).s8=(*lap4p2).s0;(*lap4p2).s9=(*lap4p2).s4;(*lap4p2).sA=(*l).s0;(*lap4p2).sB=(*l).s4;(*lap4p2).sC=(*l).s8;(*lap4p2).sD=(*l).sC;
	(*DL).s8 = (mad24((int)(*lap4p2).s8,f[d.y][0],mad24((int)(*lap4p2).s9,f[d.y][1],mad24((int)(*lap4p2).sA,f[d.y][2],mad24((int)(*lap4p2).sB,f[d.y][3],mad24((int)(*lap4p2).sC,f[d.y][4],mad24((int)(*lap4p2).sD,f[d.y][5],64))))))/128);
	(*lap4p2).s8=(*lap4p2).s1;(*lap4p2).s9=(*lap4p2).s5;(*lap4p2).sA=(*l).s1;(*lap4p2).sB=(*l).s5;(*lap4p2).sC=(*l).s9;(*lap4p2).sD=(*l).sD;
	(*DL).s9 = (mad24((int)(*lap4p2).s8,f[d.y][0],mad24((int)(*lap4p2).s9,f[d.y][1],mad24((int)(*lap4p2).sA,f[d.y][2],mad24((int)(*lap4p2).sB,f[d.y][3],mad24((int)(*lap4p2).sC,f[d.y][4],mad24((int)(*lap4p2).sD,f[d.y][5],64))))))/128);
	(*lap4p2).s8=(*lap4p2).s2;(*lap4p2).s9=(*lap4p2).s6;(*lap4p2).sA=(*l).s2;(*lap4p2).sB=(*l).s6;(*lap4p2).sC=(*l).sA;(*lap4p2).sD=(*l).sE;
	(*DL).sA = (mad24((int)(*lap4p2).s8,f[d.y][0],mad24((int)(*lap4p2).s9,f[d.y][1],mad24((int)(*lap4p2).sA,f[d.y][2],mad24((int)(*lap4p2).sB,f[d.y][3],mad24((int)(*lap4p2).sC,f[d.y][4],mad24((int)(*lap4p2).sD,f[d.y][5],64))))))/128);
	(*lap4p2).s8=(*lap4p2).s3;(*lap4p2).s9=(*lap4p2).s7;(*lap4p2).sA=(*l).s3;(*lap4p2).sB=(*l).s7;(*lap4p2).sC=(*l).sB;(*lap4p2).sD=(*l).sF;
	(*DL).sB = (mad24((int)(*lap4p2).s8,f[d.y][0],mad24((int)(*lap4p2).s9,f[d.y][1],mad24((int)(*lap4p2).sA,f[d.y][2],mad24((int)(*lap4p2).sB,f[d.y][3],mad24((int)(*lap4p2).sC,f[d.y][4],mad24((int)(*lap4p2).sD,f[d.y][5],64))))))/128);

	// row 3
	(*lap4p2).s0123 = (*lap4p2).s4567; (*lap4p2).s4567 = (*l).s0123; (*l).s0123 = (*l).s4567; (*l).s4567 = (*l).s89AB; (*l).s89AB = (*l).sCDEF;
	c.x = coords.x - 2; c.y = coords.y + 6;
	(*buf4ui) = read_imageui(ref_frame,sampler,c); (*lap4p2).s8 = (uchar)(*buf4ui).s0; ++c.x;
	(*buf4ui) = read_imageui(ref_frame,sampler,c); (*lap4p2).s9 = (uchar)(*buf4ui).s0; ++c.x;
	(*buf4ui) = read_imageui(ref_frame,sampler,c); (*lap4p2).sA = (uchar)(*buf4ui).s0; ++c.x;
	(*buf4ui) = read_imageui(ref_frame,sampler,c); (*lap4p2).sB = (uchar)(*buf4ui).s0; ++c.x;
	(*buf4ui) = read_imageui(ref_frame,sampler,c); (*lap4p2).sC = (uchar)(*buf4ui).s0; ++c.x;
	(*buf4ui) = read_imageui(ref_frame,sampler,c); (*lap4p2).sD = (uchar)(*buf4ui).s0; ++c.x;
	(*li).s0 = (mad24((int)(*lap4p2).s8,f[d.x][0],mad24((int)(*lap4p2).s9,f[d.x][1],mad24((int)(*lap4p2).sA,f[d.x][2],mad24((int)(*lap4p2).sB,f[d.x][3],mad24((int)(*lap4p2).sC,f[d.x][4],mad24((int)(*lap4p2).sD,f[d.x][5],64))))))/128);
	(*lap4p2).s8=(*lap4p2).s9;(*lap4p2).s9=(*lap4p2).sA;(*lap4p2).sA=(*lap4p2).sB;(*lap4p2).sB=(*lap4p2).sC;(*lap4p2).sC=(*lap4p2).sD; (*buf4ui) = read_imageui(ref_frame,sampler,c); (*lap4p2).sD = (uchar)(*buf4ui).s0; ++c.x;
	(*li).s1 = (mad24((int)(*lap4p2).s8,f[d.x][0],mad24((int)(*lap4p2).s9,f[d.x][1],mad24((int)(*lap4p2).sA,f[d.x][2],mad24((int)(*lap4p2).sB,f[d.x][3],mad24((int)(*lap4p2).sC,f[d.x][4],mad24((int)(*lap4p2).sD,f[d.x][5],64))))))/128);
	(*lap4p2).s8=(*lap4p2).s9;(*lap4p2).s9=(*lap4p2).sA;(*lap4p2).sA=(*lap4p2).sB;(*lap4p2).sB=(*lap4p2).sC;(*lap4p2).sC=(*lap4p2).sD; (*buf4ui) = read_imageui(ref_frame,sampler,c); (*lap4p2).sD = (uchar)(*buf4ui).s0; ++c.x;
	(*li).s2 = (mad24((int)(*lap4p2).s8,f[d.x][0],mad24((int)(*lap4p2).s9,f[d.x][1],mad24((int)(*lap4p2).sA,f[d.x][2],mad24((int)(*lap4p2).sB,f[d.x][3],mad24((int)(*lap4p2).sC,f[d.x][4],mad24((int)(*lap4p2).sD,f[d.x][5],64))))))/128);
	(*lap4p2).s8=(*lap4p2).s9;(*lap4p2).s9=(*lap4p2).sA;(*lap4p2).sA=(*lap4p2).sB;(*lap4p2).sB=(*lap4p2).sC;(*lap4p2).sC=(*lap4p2).sD; (*buf4ui) = read_imageui(ref_frame,sampler,c); (*lap4p2).sD = (uchar)(*buf4ui).s0;
	(*li).s3 = (mad24((int)(*lap4p2).s8,f[d.x][0],mad24((int)(*lap4p2).s9,f[d.x][1],mad24((int)(*lap4p2).sA,f[d.x][2],mad24((int)(*lap4p2).sB,f[d.x][3],mad24((int)(*lap4p2).sC,f[d.x][4],mad24((int)(*lap4p2).sD,f[d.x][5],64))))))/128);
	(*l).sCDEF = convert_uchar4_sat((*li));
	(*lap4p2).s8=(*lap4p2).s0;(*lap4p2).s9=(*lap4p2).s4;(*lap4p2).sA=(*l).s0;(*lap4p2).sB=(*l).s4;(*lap4p2).sC=(*l).s8;(*lap4p2).sD=(*l).sC;
	(*DL).sC = (mad24((int)(*lap4p2).s8,f[d.y][0],mad24((int)(*lap4p2).s9,f[d.y][1],mad24((int)(*lap4p2).sA,f[d.y][2],mad24((int)(*lap4p2).sB,f[d.y][3],mad24((int)(*lap4p2).sC,f[d.y][4],mad24((int)(*lap4p2).sD,f[d.y][5],64))))))/128);
	(*lap4p2).s8=(*lap4p2).s1;(*lap4p2).s9=(*lap4p2).s5;(*lap4p2).sA=(*l).s1;(*lap4p2).sB=(*l).s5;(*lap4p2).sC=(*l).s9;(*lap4p2).sD=(*l).sD;
	(*DL).sD = (mad24((int)(*lap4p2).s8,f[d.y][0],mad24((int)(*lap4p2).s9,f[d.y][1],mad24((int)(*lap4p2).sA,f[d.y][2],mad24((int)(*lap4p2).sB,f[d.y][3],mad24((int)(*lap4p2).sC,f[d.y][4],mad24((int)(*lap4p2).sD,f[d.y][5],64))))))/128);
	(*lap4p2).s8=(*lap4p2).s2;(*lap4p2).s9=(*lap4p2).s6;(*lap4p2).sA=(*l).s2;(*lap4p2).sB=(*l).s6;(*lap4p2).sC=(*l).sA;(*lap4p2).sD=(*l).sE;
	(*DL).sE = (mad24((int)(*lap4p2).s8,f[d.y][0],mad24((int)(*lap4p2).s9,f[d.y][1],mad24((int)(*lap4p2).sA,f[d.y][2],mad24((int)(*lap4p2).sB,f[d.y][3],mad24((int)(*lap4p2).sC,f[d.y][4],mad24((int)(*lap4p2).sD,f[d.y][5],64))))))/128);
	(*lap4p2).s8=(*lap4p2).s3;(*lap4p2).s9=(*lap4p2).s7;(*lap4p2).sA=(*l).s3;(*lap4p2).sB=(*l).s7;(*lap4p2).sC=(*l).sB;(*lap4p2).sD=(*l).sF;
	(*DL).sF = (mad24((int)(*lap4p2).s8,f[d.y][0],mad24((int)(*lap4p2).s9,f[d.y][1],mad24((int)(*lap4p2).sA,f[d.y][2],mad24((int)(*lap4p2).sB,f[d.y][3],mad24((int)(*lap4p2).sC,f[d.y][4],mad24((int)(*lap4p2).sD,f[d.y][5],64))))))/128);
	
	*DL = convert_int16(convert_uchar16_sat(*DL));
	
	return;
}

/*__kernel void luma_search_2step_unopt //searching in interpolated picture
						( 	__global uchar *const current_frame, //0
							__read_only image2d_t ref_frame, //1
							__global vector_net *const net, //2
							__global vector_net *const ref_net, //3
							__global int *const ref_Bdiff, //4
							const int width, //5
							const int height) //6
{
	__private uchar8 CL0, CL1, CL2, CL3, CL4, CL5, CL6, CL7;
	__private int4 DL0, DL1, DL2, DL3;
	
	__private int start_x, end_x, start_y, end_y, vector_x, vector_y,vector_x0,vector_y0; 
	__private int MinDiff, Diff, b8x8_num;  
	__private int cx, cy, px, py; //coords
	__private int2 coords,d;
		
	// now b8x8_num represents absolute number of 8x8 block (net_index)
	b8x8_num = get_global_id(0);
	vector_x = net[b8x8_num].vector_x;
	vector_y = net[b8x8_num].vector_y;
	vector_x *= 4;	vector_y *= 4;
	vector_x0 = vector_x;
	vector_y0 = vector_y;

	cx = (b8x8_num % (width/8))*8;
	cy = (b8x8_num / (width/8))*8;
		
	Diff = cy*width + cx;
	CL0 = vload8(0, current_frame + Diff); Diff += width;
	CL1 = vload8(0, current_frame + Diff); Diff += width;
	CL2 = vload8(0, current_frame + Diff); Diff += width;
	CL3 = vload8(0, current_frame + Diff); Diff += width;
	CL4 = vload8(0, current_frame + Diff); Diff += width;
	CL5 = vload8(0, current_frame + Diff); Diff += width;
	CL6 = vload8(0, current_frame + Diff); Diff += width;
	CL7 = vload8(0, current_frame + Diff);
	
	MinDiff = 0xffff;
	cx *= 4; cy *= 4;
	
	start_x = cx + vector_x - 2;	end_x = cx + vector_x + 2;
	start_y = cy + vector_y - 2;	end_y = cy + vector_y + 2;
	
	vector_x = 0; vector_y = 0; 
	
	start_x = (start_x < 0) ? 0 : start_x;
	end_x = (end_x > (width*4 - 32)) ? (width*4 - 32) : end_x;
	start_y = (start_y < 0) ? 0 : start_y;
	end_y = (end_y > (height*4 - 32)) ? (height*4 - 32) : end_y;

	#pragma unroll 1
	for (px = start_x; px <= end_x; ++px)
	{
		#pragma unroll 1
		for (py = start_y; py <= end_y; ++py)
		{
			d.x=px%4; d.y=py%4;
			d *= 2;
			coords.x = px/4; coords.y = py/4;
			construct(ref_frame, coords, d, &DL0, &DL1, &DL2, &DL3);
			DL0 = convert_int4(CL0.s0123) - DL0;
			DL1 = convert_int4(CL1.s0123) - DL1;
			DL2 = convert_int4(CL2.s0123) - DL2;
			DL3 = convert_int4(CL3.s0123) - DL3;
			weight(&DL0, &DL1, &DL2, &DL3);	
			Diff = DL0.x;
			coords.x = (px + 16)/4; coords.y = py/4; 
			construct(ref_frame, coords, d, &DL0, &DL1, &DL2, &DL3);
			DL0 = convert_int4(CL0.s4567) - DL0;
			DL1 = convert_int4(CL1.s4567) - DL1;
			DL2 = convert_int4(CL2.s4567) - DL2;
			DL3 = convert_int4(CL3.s4567) - DL3;
			weight(&DL0, &DL1, &DL2, &DL3);	
			Diff += DL0.x;
			coords.x = px/4; coords.y = (py + 16)/4; 
			construct(ref_frame, coords, d, &DL0, &DL1, &DL2, &DL3);
			DL0 = convert_int4(CL4.s0123) - DL0;
			DL1 = convert_int4(CL5.s0123) - DL1;
			DL2 = convert_int4(CL6.s0123) - DL2;
			DL3 = convert_int4(CL7.s0123) - DL3;
			weight(&DL0, &DL1, &DL2, &DL3);	
			Diff += DL0.x;
			coords.x = (px + 16)/4; coords.y = (py + 16)/4;
			construct(ref_frame, coords, d, &DL0, &DL1, &DL2, &DL3);
			DL0 = convert_int4(CL4.s4567) - DL0;
			DL1 = convert_int4(CL5.s4567) - DL1;
			DL2 = convert_int4(CL6.s4567) - DL2;
			DL3 = convert_int4(CL7.s4567) - DL3;
			weight(&DL0, &DL1, &DL2, &DL3);	
			Diff += DL0.x;
			
			d /= 2;
			Diff += (int)(abs(px-cx-vector_x0) + abs(py-cy-vector_y0))*vector_diff_weight;
			
			vector_x = (Diff < MinDiff) ? (px - cx) : vector_x;
			vector_y = (Diff < MinDiff) ? (py - cy) : vector_y;
			MinDiff = (Diff < MinDiff) ? Diff : MinDiff;
    	} 
    } 

	MinDiff -= ((int)abs(vector_x - vector_x0) + (int)abs(vector_y - vector_y0))*vector_diff_weight;
	
	ref_net[b8x8_num].vector_x = vector_x;
	ref_net[b8x8_num].vector_y = vector_y;
	ref_Bdiff[b8x8_num] = MinDiff;
	
	return;
}  */

__kernel void luma_search_2step //searching in interpolated picture
						( 	__global uchar *const current_frame, //0
							__read_only image2d_t ref_frame, //1
							__global vector_net *const net, //2
							__global vector_net *const ref_net, //3
							__global int *const ref_Bdiff, //4
							const int width, //5
							const int height) //6
{
	//this version of kernel uses almost all variables packed to vectors (AMD compiler then use less VGPR)
	// code unreadable (use commented unpacked version of kernel)
	__private int16 DL;
	
	__private short8 pdata1;
	__private short8 pdata2;
	
	uchar16 l, lap4p2;
	int4 li;
	uint4 buf4ui;
	
	__private int MinDiff, Diff, b8x8_num;  
	
	__local uchar4 LDS[8*1024]; //32kb
	
	// the code has become unreadable, but it's better not to use defines (compiler probably go mad)
		
	// now b8x8_num represents absolute number of 8x8 block (net_index)
	b8x8_num = get_global_id(0);
	pdata1.s4 = net[b8x8_num].vector_x;
	pdata1.s5 = net[b8x8_num].vector_y;
	pdata1.s4 *= 4;	pdata1.s5 *= 4;
	pdata1.s6 = pdata1.s4;
	pdata1.s7 = pdata1.s5;

	pdata2.s0 = (b8x8_num % (width/8))*8;
	pdata2.s1 = (b8x8_num / (width/8))*8;
		
	Diff = pdata2.s1*width + pdata2.s0;
	LDS[(int)get_local_id(0)] = vload4(0, current_frame + Diff);; 
	LDS[(int)get_local_size(0) + (int)get_local_id(0)] = vload4(0, current_frame + Diff + 4);  Diff += width;
	LDS[mad24((int)get_local_size(0),2,(int)get_local_id(0))] = vload4(0, current_frame + Diff);
	LDS[mad24((int)get_local_size(0),3,(int)get_local_id(0))] = vload4(0, current_frame + Diff + 4); Diff += width;
	LDS[mad24((int)get_local_size(0),4,(int)get_local_id(0))] = vload4(0, current_frame + Diff);
	LDS[mad24((int)get_local_size(0),5,(int)get_local_id(0))] = vload4(0, current_frame + Diff + 4); Diff += width;
	LDS[mad24((int)get_local_size(0),6,(int)get_local_id(0))] = vload4(0, current_frame + Diff);
	LDS[mad24((int)get_local_size(0),7,(int)get_local_id(0))] = vload4(0, current_frame + Diff + 4); Diff += width;
	LDS[mad24((int)get_local_size(0),8,(int)get_local_id(0))] = vload4(0, current_frame + Diff);
	LDS[mad24((int)get_local_size(0),9,(int)get_local_id(0))] = vload4(0, current_frame + Diff + 4); Diff += width;
	LDS[mad24((int)get_local_size(0),10,(int)get_local_id(0))] = vload4(0, current_frame + Diff);
	LDS[mad24((int)get_local_size(0),11,(int)get_local_id(0))] = vload4(0, current_frame + Diff + 4); Diff += width;
	LDS[mad24((int)get_local_size(0),12,(int)get_local_id(0))] = vload4(0, current_frame + Diff);
	LDS[mad24((int)get_local_size(0),13,(int)get_local_id(0))] = vload4(0, current_frame + Diff + 4); Diff += width;
	LDS[mad24((int)get_local_size(0),14,(int)get_local_id(0))] = vload4(0, current_frame + Diff);
	LDS[mad24((int)get_local_size(0),15,(int)get_local_id(0))] = vload4(0, current_frame + Diff + 4); 
	
	MinDiff = 0x7fff;
	pdata2.s0 *= 4; pdata2.s1 *= 4;
	
	pdata1.s0 = pdata2.s0 + pdata1.s4 - 2;	pdata1.s1 = pdata2.s0 + pdata1.s4 + 2;
	pdata1.s2 = pdata2.s1 + pdata1.s5 - 2;	pdata1.s3 = pdata2.s1 + pdata1.s5 + 2;
	
	pdata1.s4 = 0; pdata1.s5 = 0; 
	
	pdata1.s0 = (pdata1.s0 < 0) ? 0 : pdata1.s0;
	pdata1.s1 = (pdata1.s1 > (width*4 - 32)) ? (width*4 - 32) : pdata1.s1;
	pdata1.s2 = (pdata1.s2 < 0) ? 0 : pdata1.s2;
	pdata1.s3 = (pdata1.s3 > (height*4 - 32)) ? (height*4 - 32) : pdata1.s3;

	#pragma unroll 1
	for (pdata2.s2 = pdata1.s0; pdata2.s2 <= pdata1.s1; ++pdata2.s2)
	{
		#pragma unroll 1
		for (pdata2.s3 = pdata1.s2; pdata2.s3 <= pdata1.s3; ++pdata2.s3)
		{
			pdata2.s6=pdata2.s2%4; pdata2.s7=pdata2.s3%4;
			pdata2.s67 *= 2;
			pdata2.s4 = pdata2.s2/4; pdata2.s5 = pdata2.s3/4;
			construct_opt(ref_frame, pdata2.s45, pdata2.s67, &DL, &l, &lap4p2, &li, &buf4ui);
			DL.s0123 = convert_int4(LDS[(int)get_local_id(0)]) - DL.s0123;
			DL.s4567 = convert_int4(LDS[mad24((int)get_local_size(0),2,(int)get_local_id(0))]) - DL.s4567;
			DL.s89AB = convert_int4(LDS[mad24((int)get_local_size(0),4,(int)get_local_id(0))]) - DL.s89AB;
			DL.sCDEF = convert_int4(LDS[mad24((int)get_local_size(0),6,(int)get_local_id(0))]) - DL.sCDEF;
			weight_opt(&li, &DL);	
			Diff = li.s0;
			pdata2.s4 = (pdata2.s2 + 16)/4; pdata2.s5 = pdata2.s3/4; 
			construct_opt(ref_frame, pdata2.s45, pdata2.s67, &DL, &l, &lap4p2, &li, &buf4ui);
			DL.s0123 = convert_int4(LDS[(int)get_local_size(0)+(int)get_local_id(0)]) - DL.s0123;
			DL.s4567 = convert_int4(LDS[mad24((int)get_local_size(0),3,(int)get_local_id(0))]) - DL.s4567;
			DL.s89AB = convert_int4(LDS[mad24((int)get_local_size(0),5,(int)get_local_id(0))]) - DL.s89AB;
			DL.sCDEF = convert_int4(LDS[mad24((int)get_local_size(0),7,(int)get_local_id(0))]) - DL.sCDEF;
			weight_opt(&li, &DL);	
			Diff += li.s0;
			pdata2.s4 = pdata2.s2/4; pdata2.s5 = (pdata2.s3 + 16)/4; 
			construct_opt(ref_frame, pdata2.s45, pdata2.s67, &DL, &l, &lap4p2, &li, &buf4ui);
			DL.s0123 = convert_int4(LDS[mad24((int)get_local_size(0),8,(int)get_local_id(0))]) - DL.s0123;
			DL.s4567 = convert_int4(LDS[mad24((int)get_local_size(0),10,(int)get_local_id(0))]) - DL.s4567;
			DL.s89AB = convert_int4(LDS[mad24((int)get_local_size(0),12,(int)get_local_id(0))]) - DL.s89AB;
			DL.sCDEF = convert_int4(LDS[mad24((int)get_local_size(0),14,(int)get_local_id(0))]) - DL.sCDEF;
			weight_opt(&li, &DL);	
			Diff += li.s0;
			pdata2.s4 = (pdata2.s2 + 16)/4; pdata2.s5 = (pdata2.s3 + 16)/4;
			construct_opt(ref_frame, pdata2.s45, pdata2.s67, &DL, &l, &lap4p2, &li, &buf4ui);
			DL.s0123 = convert_int4(LDS[mad24((int)get_local_size(0),9,(int)get_local_id(0))]) - DL.s0123;
			DL.s4567 = convert_int4(LDS[mad24((int)get_local_size(0),11,(int)get_local_id(0))]) - DL.s4567;
			DL.s89AB = convert_int4(LDS[mad24((int)get_local_size(0),13,(int)get_local_id(0))]) - DL.s89AB;
			DL.sCDEF = convert_int4(LDS[mad24((int)get_local_size(0),15,(int)get_local_id(0))]) - DL.sCDEF;
			weight_opt(&li, &DL);	
			Diff += li.s0;
			
			pdata2.s67 /= 2;
			Diff += (int)(abs(pdata2.s2-pdata2.s0-pdata1.s6) + abs(pdata2.s3-pdata2.s1-pdata1.s7))*vector_diff_weight;
			
			pdata1.s4 = (Diff < MinDiff) ? (pdata2.s2 - pdata2.s0) : pdata1.s4;
			pdata1.s5 = (Diff < MinDiff) ? (pdata2.s3 - pdata2.s1) : pdata1.s5;
			MinDiff = (Diff < MinDiff) ? Diff : MinDiff;
    	} 
    } 

	MinDiff -= ((int)abs(pdata1.s4 - pdata1.s6) + (int)abs(pdata1.s5 - pdata1.s7))*vector_diff_weight;
	
	ref_net[b8x8_num].vector_x = pdata1.s4;
	ref_net[b8x8_num].vector_y = pdata1.s5;
	ref_Bdiff[b8x8_num] = MinDiff;
	
	return;
}


__kernel void select_reference(__global vector_net *const last_net, //0
								__global vector_net *const golden_net, //1
								__global vector_net *const altref_net, //2
								__global int *const last_Bdiff, //3
								__global int *const golden_Bdiff, //4
								__global int *const altref_Bdiff, //5
								__global macroblock *const MBs, //6
								const int width, //7
								const int use_golden, //8
								const int use_altref) //9
{
	__private int diff1, diff2, mb_num, b8x8_num, mb_width, b8x8_width, ref, vx0, vy0, vx1, vy1, vx2, vy2, vx3, vy3;
	mb_num = get_global_id(0);
	mb_width = width/16;
	b8x8_width = mb_width*2;
	// each mb contain 2 8x8 blocks in row or in collumn
	b8x8_num = ((mb_num / mb_width)*2)*(b8x8_width) + ((mb_num % mb_width)*2);
	
	//load LAST diff
	diff1 = last_Bdiff[b8x8_num];
	diff1 += last_Bdiff[b8x8_num + 1];
	diff1 += last_Bdiff[b8x8_num + b8x8_width];
	diff1 += last_Bdiff[b8x8_num + b8x8_width + 1];
	
	//load ALTREF diff
	diff2 = 0x7fffffff;
	if (use_altref == 1)
	{
		diff2 = altref_Bdiff[b8x8_num];
		diff2 += altref_Bdiff[b8x8_num + 1];
		diff2 += altref_Bdiff[b8x8_num + b8x8_width];
		diff2 += altref_Bdiff[b8x8_num + b8x8_width + 1];
	}
	ref = select(ALTREF,LAST,diff1<=diff2);
	diff1 = select(diff2,diff1,diff1<=diff2);
	
	//load GOLDEN diff
	diff2 = 0x7fffffff;
	if (use_golden == 1)
	{
		diff2 = golden_Bdiff[b8x8_num];
		diff2 += golden_Bdiff[b8x8_num + 1];
		diff2 += golden_Bdiff[b8x8_num + b8x8_width];
		diff2 += golden_Bdiff[b8x8_num + b8x8_width + 1];
	}
	ref = select(GOLDEN,ref,diff1<=diff2);
	
	if (ref == LAST)
	{
		vx0 = last_net[b8x8_num].vector_x;
		vy0 = last_net[b8x8_num].vector_y;
		vx1 = last_net[b8x8_num + 1].vector_x;
		vy1 = last_net[b8x8_num + 1].vector_y;
		vx2 = last_net[b8x8_num + b8x8_width].vector_x;
		vy2 = last_net[b8x8_num + b8x8_width].vector_y;
		vx3 = last_net[b8x8_num + b8x8_width + 1].vector_x;
		vy3 = last_net[b8x8_num + b8x8_width + 1].vector_y;
	}
	if (ref == GOLDEN)
	{
		vx0 = golden_net[b8x8_num].vector_x;
		vy0 = golden_net[b8x8_num].vector_y;
		vx1 = golden_net[b8x8_num + 1].vector_x;
		vy1 = golden_net[b8x8_num + 1].vector_y;
		vx2 = golden_net[b8x8_num + b8x8_width].vector_x;
		vy2 = golden_net[b8x8_num + b8x8_width].vector_y;
		vx3 = golden_net[b8x8_num + b8x8_width + 1].vector_x;
		vy3 = golden_net[b8x8_num + b8x8_width + 1].vector_y;
	}
	if (ref == ALTREF)
	{
		vx0 = altref_net[b8x8_num].vector_x;
		vy0 = altref_net[b8x8_num].vector_y;
		vx1 = altref_net[b8x8_num + 1].vector_x;
		vy1 = altref_net[b8x8_num + 1].vector_y;
		vx2 = altref_net[b8x8_num + b8x8_width].vector_x;
		vy2 = altref_net[b8x8_num + b8x8_width].vector_y;
		vx3 = altref_net[b8x8_num + b8x8_width + 1].vector_x;
		vy3 = altref_net[b8x8_num + b8x8_width + 1].vector_y;
	}
	
	MBs[mb_num].reference_frame = ref;
	MBs[mb_num].vector_x[0] = vx0;
	MBs[mb_num].vector_y[0] = vy0;
	MBs[mb_num].vector_x[1] = vx1;
	MBs[mb_num].vector_y[1] = vy1;
	MBs[mb_num].vector_x[2] = vx2;
	MBs[mb_num].vector_y[2] = vy2;
	MBs[mb_num].vector_x[3] = vx3;
	MBs[mb_num].vector_y[3] = vy3;
	
	return;
}

__kernel void prepare_predictors_and_residual(__global uchar *const current_frame, //0
												__read_only image2d_t ref_frame, //1
												__global uchar *const predictor, //2
												__global short *const residual, //3
												__global macroblock *const MBs, //4
												const int width, //5
												const int plane, //6
												const int ref) //7
{
	__private int b_num, mb_num, g;
	__private int i;
	__private int4 DL0,DL1,DL2,DL3;
	__private uchar4 L0,L1,L2,L3;
	__private int2 coords,d,v,pos;
	__private int mb_size = (plane == 0) ? 16 : 8;

	b_num = get_global_id(0);
	pos.x = (b_num % (width/4))*4;
	pos.y = (b_num / (width/4))*4;
	mb_num = (pos.y/mb_size)*(width/mb_size) + (pos.x/mb_size);
	
	if (MBs[mb_num].reference_frame != ref) return;
	
	v = (pos%mb_size)/(mb_size/2);
	i = mad24(v.y,2,v.x);
	v.x = MBs[mb_num].vector_x[i];
	v.y = MBs[mb_num].vector_y[i];
	
	//diplacement fractional part
	g = (plane == 0) ? 4 : 8;
	d = mad24(pos,g,v)%g;
	//^ d is displacement in 1/g pixels
	// d*2 because 1/4 luma filter is 2/8 record in array
	d *= (plane == 0) ? 2 : 1;		
	coords = mad24(pos,g,v)/g; 
	i = pos.y*width + pos.x;
	construct(ref_frame, coords, d, &DL0, &DL1, &DL2, &DL3);
	L0 = convert_uchar4_sat(DL0);
	L1 = convert_uchar4_sat(DL1);
	L2 = convert_uchar4_sat(DL2);
	L3 = convert_uchar4_sat(DL3);
	vstore4(L0, 0, predictor+i); i+= width;
	vstore4(L1, 0, predictor+i); i+= width;
	vstore4(L2, 0, predictor+i); i+= width;
	vstore4(L3, 0, predictor+i); i-= 3*width;
	L0 = vload4(0, current_frame+i); i+= width;
	L1 = vload4(0, current_frame+i); i+= width;
	L2 = vload4(0, current_frame+i); i+= width;
	L3 = vload4(0, current_frame+i); i-= 3*width;
	DL0 = convert_int4(L0) - DL0; 
	DL1 = convert_int4(L1) - DL1; 
	DL2 = convert_int4(L2) - DL2; 
	DL3 = convert_int4(L3) - DL3;
	vstore4(convert_short4(DL0), 0, residual+i); i+= width;
	vstore4(convert_short4(DL1), 0, residual+i); i+= width;
	vstore4(convert_short4(DL2), 0, residual+i); i+= width;
	vstore4(convert_short4(DL3), 0, residual+i);
	
	return;	
}


__kernel void pack_8x8_into_16x16(__global macroblock *const MBs) //0
{
	__private int mb_num, vector_x,vector_y,x,y,condition;
	mb_num = get_global_id(0);
	MBs[mb_num].SSIM = -2.0f;
	MBs[mb_num].parts = are8x8; 
	vector_x = MBs[mb_num].vector_x[0];	vector_y = MBs[mb_num].vector_y[0];
	x = MBs[mb_num].vector_x[1];		y = MBs[mb_num].vector_y[1];
	condition = ((vector_x != x) || (vector_y != y)); //XOr optimization possible
	if (condition) return;
	x = MBs[mb_num].vector_x[2];		y = MBs[mb_num].vector_y[2];
	condition = ((vector_x != x) || (vector_y != y)); 
	if (condition) return;
	x = MBs[mb_num].vector_x[3];		y = MBs[mb_num].vector_y[3];
	condition = ((vector_x != x) || (vector_y != y)); 
	if (condition) return; 
	MBs[mb_num].parts = are16x16;
	return;
}

__kernel void dct4x4(__global short *const residual, //0
					__global macroblock *const MBs, //1
					const int width, //2
					__constant segment_data *const SD, //3
					const int segment_id, //4
					const float SSIM_target, //5
					const int plane) //6
{	
	__private int b_num, mb_num, ac_q,dc_q,uv_dc_q,uv_ac_q;
	__private int i;
	__private int4 DL0,DL1,DL2,DL3;
	__private int2 pos,v;
	__private int mb_size = (plane == 0) ? 16 : 8;

	b_num = get_global_id(0);
	pos.x = (b_num % (width/4))*4;
	pos.y = (b_num / (width/4))*4;
	mb_num = (pos.y/mb_size)*(width/mb_size) + (pos.x/mb_size);

	if (MBs[mb_num].SSIM > SSIM_target) return;
	
	MBs[mb_num].segment_id = segment_id;
	i = SD[segment_id].y_ac_i;
	
	dc_q = SD[0].y_dc_idelta; dc_q += i;
	dc_q = select((int)dc_q,0,dc_q<0); dc_q = select((int)dc_q,127,dc_q>127);
	ac_q = vp8_ac_qlookup[i];
	dc_q = (MBs[mb_num].parts == are16x16) ? 1 : vp8_dc_qlookup[dc_q];
	uv_dc_q = SD[0].uv_dc_idelta; uv_dc_q += i;
	uv_ac_q = SD[0].uv_ac_idelta; uv_ac_q += i;
	uv_dc_q = select(uv_dc_q,0,uv_dc_q<0); uv_dc_q = select(uv_dc_q,127,uv_dc_q>127);
	uv_ac_q = select(uv_ac_q,0,uv_ac_q<0); uv_ac_q = select(uv_ac_q,127,uv_ac_q>127);
	uv_dc_q = vp8_dc_qlookup[uv_dc_q];
	uv_ac_q = vp8_ac_qlookup[uv_ac_q];
	uv_dc_q = select(uv_dc_q,132,uv_dc_q>132);
	dc_q = select(uv_dc_q,dc_q,plane==0);
	ac_q = select(uv_ac_q,ac_q,plane==0);

	i = pos.y*width + pos.x;
	DL0 = convert_int4(vload4(0, residual+i)); i+= width;	
	DL1 = convert_int4(vload4(0, residual+i)); i+= width;
	DL2 = convert_int4(vload4(0, residual+i)); i+= width;	
	DL3 = convert_int4(vload4(0, residual+i));
	DCT_and_quant(&DL0, &DL1, &DL2, &DL3, dc_q, ac_q);
	
	v.x = (pos.x%mb_size)/4;
	v.y = (pos.y%mb_size)/4;
	b_num = v.y*(mb_size/4) + v.x;
	b_num += (plane == 1) ? 16 : 0;
	b_num += (plane == 2) ? 20 : 0;
	const int inv_zigzag[16] = { 0, 1, 5, 6, 2, 4, 7, 12, 3,  8, 11, 13, 9, 10, 14, 15 };
	MBs[mb_num].coeffs[b_num][inv_zigzag[0]]=(short)DL0.s0;  MBs[mb_num].coeffs[b_num][inv_zigzag[1]]=(short)DL0.s1;  MBs[mb_num].coeffs[b_num][inv_zigzag[2]]=(short)DL0.s2;  MBs[mb_num].coeffs[b_num][inv_zigzag[3]]=(short)DL0.s3;
	MBs[mb_num].coeffs[b_num][inv_zigzag[4]]=(short)DL1.s0;  MBs[mb_num].coeffs[b_num][inv_zigzag[5]]=(short)DL1.s1;  MBs[mb_num].coeffs[b_num][inv_zigzag[6]]=(short)DL1.s2;  MBs[mb_num].coeffs[b_num][inv_zigzag[7]]=(short)DL1.s3;
	MBs[mb_num].coeffs[b_num][inv_zigzag[8]]=(short)DL2.s0;  MBs[mb_num].coeffs[b_num][inv_zigzag[9]]=(short)DL2.s1;  MBs[mb_num].coeffs[b_num][inv_zigzag[10]]=(short)DL2.s2; MBs[mb_num].coeffs[b_num][inv_zigzag[11]]=(short)DL2.s3;
	MBs[mb_num].coeffs[b_num][inv_zigzag[12]]=(short)DL3.s0; MBs[mb_num].coeffs[b_num][inv_zigzag[13]]=(short)DL3.s1; MBs[mb_num].coeffs[b_num][inv_zigzag[14]]=(short)DL3.s2; MBs[mb_num].coeffs[b_num][inv_zigzag[15]]=(short)DL3.s3;

	return;
}

__kernel void wht4x4_iwht4x4(__global macroblock *const MBs, //0
							__constant segment_data *const SD, //1
							const int segment_id, //2
							const float SSIM_target) //3
{
	__private int mb_num,y2_dc_q,y2_ac_q;
	__private int i;
	__private int4 DL0,DL1,DL2,DL3;

	mb_num = get_global_id(0);

	if (MBs[mb_num].SSIM > SSIM_target) return;
	if (MBs[mb_num].parts != are16x16) return;
	
	MBs[mb_num].segment_id = segment_id;
	i = SD[segment_id].y_ac_i;
	
	y2_dc_q = SD[0].y2_dc_idelta; y2_dc_q += i;
	y2_ac_q = SD[0].y2_ac_idelta; y2_ac_q += i;
	y2_dc_q = select((int)y2_dc_q,0,y2_dc_q<0); y2_dc_q = select((int)y2_dc_q,127,y2_dc_q>127);
	y2_ac_q = select((int)y2_ac_q,0,y2_ac_q<0); y2_ac_q = select((int)y2_ac_q,127,y2_ac_q>127);
	
	y2_dc_q = (vp8_dc_qlookup[y2_dc_q])*2;
	y2_ac_q = 31*(vp8_ac_qlookup[y2_ac_q])/20;
	y2_ac_q = select((int)y2_ac_q,8,y2_ac_q<8);

	const int inv_zigzag[16] = { 0, 1, 5, 6, 2, 4, 7, 12, 3,  8, 11, 13, 9, 10, 14, 15 };
	DL0.s0=(int)MBs[mb_num].coeffs[0][0];  DL0.s1=(int)MBs[mb_num].coeffs[1][0];  DL0.s2=(int)MBs[mb_num].coeffs[2][0];  DL0.s3=(int)MBs[mb_num].coeffs[3][0];
	DL1.s0=(int)MBs[mb_num].coeffs[4][0];  DL1.s1=(int)MBs[mb_num].coeffs[5][0];  DL1.s2=(int)MBs[mb_num].coeffs[6][0];  DL1.s3=(int)MBs[mb_num].coeffs[7][0];
	DL2.s0=(int)MBs[mb_num].coeffs[8][0];  DL2.s1=(int)MBs[mb_num].coeffs[9][0];  DL2.s2=(int)MBs[mb_num].coeffs[10][0]; DL2.s3=(int)MBs[mb_num].coeffs[11][0];
	DL3.s0=(int)MBs[mb_num].coeffs[12][0]; DL3.s1=(int)MBs[mb_num].coeffs[13][0]; DL3.s2=(int)MBs[mb_num].coeffs[14][0]; DL3.s3=(int)MBs[mb_num].coeffs[15][0];
	WHT_and_quant(&DL0, &DL1, &DL2, &DL3, y2_dc_q, y2_ac_q);	
	MBs[mb_num].coeffs[24][inv_zigzag[0]]=(short)DL0.s0;  MBs[mb_num].coeffs[24][inv_zigzag[1]]=(short)DL0.s1;  MBs[mb_num].coeffs[24][inv_zigzag[2]]=(short)DL0.s2;  MBs[mb_num].coeffs[24][inv_zigzag[3]]=(short)DL0.s3;
	MBs[mb_num].coeffs[24][inv_zigzag[4]]=(short)DL1.s0;  MBs[mb_num].coeffs[24][inv_zigzag[5]]=(short)DL1.s1;  MBs[mb_num].coeffs[24][inv_zigzag[6]]=(short)DL1.s2;  MBs[mb_num].coeffs[24][inv_zigzag[7]]=(short)DL1.s3;
	MBs[mb_num].coeffs[24][inv_zigzag[8]]=(short)DL2.s0;  MBs[mb_num].coeffs[24][inv_zigzag[9]]=(short)DL2.s1;  MBs[mb_num].coeffs[24][inv_zigzag[10]]=(short)DL2.s2; MBs[mb_num].coeffs[24][inv_zigzag[11]]=(short)DL2.s3;
	MBs[mb_num].coeffs[24][inv_zigzag[12]]=(short)DL3.s0; MBs[mb_num].coeffs[24][inv_zigzag[13]]=(short)DL3.s1; MBs[mb_num].coeffs[24][inv_zigzag[14]]=(short)DL3.s2; MBs[mb_num].coeffs[24][inv_zigzag[15]]=(short)DL3.s3;
	dequant_and_iWHT(&DL0, &DL1, &DL2, &DL3, y2_dc_q, y2_ac_q);
	MBs[mb_num].coeffs[0][0]=(short)DL0.s0;  MBs[mb_num].coeffs[1][0]=(short)DL0.s1;  MBs[mb_num].coeffs[2][0]=(short)DL0.s2;  MBs[mb_num].coeffs[3][0]=(short)DL0.s3;
	MBs[mb_num].coeffs[4][0]=(short)DL1.s0;  MBs[mb_num].coeffs[5][0]=(short)DL1.s1;  MBs[mb_num].coeffs[6][0]=(short)DL1.s2;  MBs[mb_num].coeffs[7][0]=(short)DL1.s3;
	MBs[mb_num].coeffs[8][0]=(short)DL2.s0;  MBs[mb_num].coeffs[9][0]=(short)DL2.s1;  MBs[mb_num].coeffs[10][0]=(short)DL2.s2; MBs[mb_num].coeffs[11][0]=(short)DL2.s3;
	MBs[mb_num].coeffs[12][0]=(short)DL3.s0; MBs[mb_num].coeffs[13][0]=(short)DL3.s1; MBs[mb_num].coeffs[14][0]=(short)DL3.s2; MBs[mb_num].coeffs[15][0]=(short)DL3.s3;

	return;
}

__kernel void idct4x4(__global uchar *const recon_frame, //0
					__global uchar *const predictor, //1
					__global macroblock *const MBs, //2
					const int width, //3
					__constant segment_data *const SD, //4
					const int segment_id, //5
					const float SSIM_target, //6
					const int plane) //7
{	
	__private int b_num,mb_num, vx,vy,x,y,ac_q,dc_q,uv_dc_q,uv_ac_q;
	__private int i;
	__private int4 DL0,DL1,DL2,DL3;
	__private int mb_size = (plane == 0) ? 16 : 8;

	b_num = get_global_id(0);
	x = (b_num % (width/4))*4;
	y = (b_num / (width/4))*4;
	mb_num = (y/mb_size)*(width/mb_size) + (x/mb_size);

	if (MBs[mb_num].SSIM > SSIM_target) return;
	
	MBs[mb_num].segment_id = segment_id;
	i = SD[segment_id].y_ac_i;
	
	dc_q = SD[0].y_dc_idelta; dc_q += i;
	dc_q = select((int)dc_q,0,dc_q<0); dc_q = select((int)dc_q,127,dc_q>127);
	ac_q = vp8_ac_qlookup[i];
	dc_q = (MBs[mb_num].parts == are16x16) ? 1 : vp8_dc_qlookup[dc_q];
	uv_dc_q = SD[0].uv_dc_idelta; uv_dc_q += i;
	uv_ac_q = SD[0].uv_ac_idelta; uv_ac_q += i;
	uv_dc_q = select(uv_dc_q,0,uv_dc_q<0); uv_dc_q = select(uv_dc_q,127,uv_dc_q>127);
	uv_ac_q = select(uv_ac_q,0,uv_ac_q<0); uv_ac_q = select(uv_ac_q,127,uv_ac_q>127);
	uv_dc_q = vp8_dc_qlookup[uv_dc_q];
	uv_ac_q = vp8_ac_qlookup[uv_ac_q];
	uv_dc_q = select(uv_dc_q,132,uv_dc_q>132);
	dc_q = select(uv_dc_q,dc_q,plane==0);
	ac_q = select(uv_ac_q,ac_q,plane==0);

	vx = (x%mb_size)/4;
	vy = (y%mb_size)/4;
	b_num = vy*(mb_size/4) + vx;
	b_num += (plane == 1) ? 16 : 0;
	b_num += (plane == 2) ? 20 : 0;
	const int inv_zigzag[16] = { 0, 1, 5, 6, 2, 4, 7, 12, 3,  8, 11, 13, 9, 10, 14, 15 };
	DL0.s0=MBs[mb_num].coeffs[b_num][inv_zigzag[0]];   DL0.s1=MBs[mb_num].coeffs[b_num][inv_zigzag[1]];   DL0.s2=MBs[mb_num].coeffs[b_num][inv_zigzag[2]];   DL0.s3=MBs[mb_num].coeffs[b_num][inv_zigzag[3]];
	DL1.s0=MBs[mb_num].coeffs[b_num][inv_zigzag[4]];   DL1.s1=MBs[mb_num].coeffs[b_num][inv_zigzag[5]];   DL1.s2=MBs[mb_num].coeffs[b_num][inv_zigzag[6]];   DL1.s3=MBs[mb_num].coeffs[b_num][inv_zigzag[7]];
	DL2.s0=MBs[mb_num].coeffs[b_num][inv_zigzag[8]];   DL2.s1=MBs[mb_num].coeffs[b_num][inv_zigzag[9]];   DL2.s2=MBs[mb_num].coeffs[b_num][inv_zigzag[10]];  DL2.s3=MBs[mb_num].coeffs[b_num][inv_zigzag[11]];
	DL3.s0=MBs[mb_num].coeffs[b_num][inv_zigzag[12]];  DL3.s1=MBs[mb_num].coeffs[b_num][inv_zigzag[13]];  DL3.s2=MBs[mb_num].coeffs[b_num][inv_zigzag[14]];  DL3.s3=MBs[mb_num].coeffs[b_num][inv_zigzag[15]];
	dequant_and_iDCT(&DL0, &DL1, &DL2, &DL3, dc_q, ac_q);
	i = y*width + x;
	DL0 += convert_int4(vload4(0, predictor+i)); i+= width;
	DL1 += convert_int4(vload4(0, predictor+i)); i+= width;
	DL2 += convert_int4(vload4(0, predictor+i)); i+= width;
	DL3 += convert_int4(vload4(0, predictor+i)); i-= 3*width;
	vstore4(convert_uchar4_sat(DL0), 0, recon_frame+i); i+= width;
	vstore4(convert_uchar4_sat(DL1), 0, recon_frame+i); i+= width;
	vstore4(convert_uchar4_sat(DL2), 0, recon_frame+i); i+= width;
	vstore4(convert_uchar4_sat(DL3), 0, recon_frame+i);

	return;
}

__kernel void count_SSIM_luma
						(__global uchar *frame1, //0
						__global uchar *frame2, //1
						__global macroblock *MBs, //2
                        signed int width, //3
						const int segment_id)// 4														
{
    __private int mb_num, i;
    __private uchar16 FL0, FL1, FL2, FL3, FL4, FL5, FL6, FL7, FL8, FL9, FL10, FL11, FL12, FL13, FL14, FL15;
    __private uchar16 SL0, SL1, SL2, SL3, SL4, SL5, SL6, SL7, SL8, SL9, SL10, SL11, SL12, SL13, SL14, SL15;
    __private float16 IL, IL1;
    __private float M1, M2, D1, D2, C;
    mb_num = get_global_id(0);
	if (MBs[mb_num].segment_id != segment_id) return;
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
		
	// and now small workaround
	// SSIM sometimes gives high structural similarity for block with largely different average value
	// but similar structure: AC coeffs are close or variance is low
	//example M1 == 60; M2 == 67 and D1 == D2 == C (only DC coeff differs)
	// SSIM would be 0.993 while 7 points of luma difference would look like horrible unmatched square
	
	// so we lower SSIM (cheat) for each 1 point(over 4) by 0.02 
	D1 = (M1 - M2);
	D1 = select(D1,-D1,D1 < 0);
	D2 = select(0.0f,0.02f*D1,D1 > 4);
	C -= D2;
	
	D1 = MBs[mb_num].SSIM;
	C += D1;
	C /= 3;
	MBs[mb_num].SSIM = C;

	return;
}												

__kernel void count_SSIM_chroma
						(__global uchar *frame1, //0
						__global uchar *frame2, //1
						__global macroblock *MBs, //2
                        signed int cwidth, //3
						const int segment_id, //4
						const int reset)// 5	
{
	const float c1 = 0.01f*0.01f*255*255;
	const float c2 = 0.03f*0.03f*255*255;
	__private int mb_num, i;
	__private float M1, M2, D1, D2, C;
	mb_num = get_global_id(0);
	if (MBs[mb_num].segment_id != segment_id) return;
	
	__private uchar8 cFL0, cFL1, cFL2, cFL3, cFL4, cFL5, cFL6, cFL7;
	__private uchar8 cSL0, cSL1, cSL2, cSL3, cSL4, cSL5, cSL6, cSL7;
	__private float8 cIL, cIL1;
	
	i = ((mb_num / (cwidth/8))*8)*cwidth + ((mb_num % (cwidth/8))*8);

	cFL0 = cFL1 = cFL2 = cFL3 = cFL4 = cFL5 = cFL6 = cFL7 = 0;
	cSL0 = cSL1 = cSL2 = cSL3 = cSL4 = cSL5 = cSL6 = cSL7 = 0;
	// U
	cFL0 = vload8(0, frame1 + i); i += cwidth; cIL = convert_float8(cFL0); 
	cFL1 = vload8(0, frame1 + i); i += cwidth; cIL += convert_float8(cFL1);
	cFL2 = vload8(0, frame1 + i); i += cwidth; cIL += convert_float8(cFL2);
	cFL3 = vload8(0, frame1 + i); i += cwidth; cIL += convert_float8(cFL3);
	cFL4 = vload8(0, frame1 + i); i += cwidth; cIL += convert_float8(cFL4);
	cFL5 = vload8(0, frame1 + i); i += cwidth; cIL += convert_float8(cFL5);
	cFL6 = vload8(0, frame1 + i); i += cwidth; cIL += convert_float8(cFL6);
	cFL7 = vload8(0, frame1 + i); i -= 7*cwidth; cIL += convert_float8(cFL7); 
	M1 = (cIL.s0 + cIL.s1 + cIL.s2 + cIL.s3 + cIL.s4 + cIL.s5 + cIL.s6 + cIL.s7)/64;
	cIL = convert_float8(cFL0) - M1; cIL *= cIL;
	cIL1 = convert_float8(cFL1) - M1; cIL = mad(cIL1,cIL1,cIL);
	cIL1 = convert_float8(cFL2) - M1; cIL = mad(cIL1,cIL1,cIL);
	cIL1 = convert_float8(cFL3) - M1; cIL = mad(cIL1,cIL1,cIL);
	cIL1 = convert_float8(cFL4) - M1; cIL = mad(cIL1,cIL1,cIL);
	cIL1 = convert_float8(cFL5) - M1; cIL = mad(cIL1,cIL1,cIL);
	cIL1 = convert_float8(cFL6) - M1; cIL = mad(cIL1,cIL1,cIL);
	cIL1 = convert_float8(cFL7) - M1; cIL = mad(cIL1,cIL1,cIL);
	D1 = (cIL.s0 + cIL.s1 + cIL.s2 + cIL.s3 + cIL.s4 + cIL.s5 + cIL.s6 + cIL.s7)/64;
	cSL0 = vload8(0, frame2 + i); i += cwidth; cIL = convert_float8(cSL0);
	cSL1 = vload8(0, frame2 + i); i += cwidth; cIL += convert_float8(cSL1);
	cSL2 = vload8(0, frame2 + i); i += cwidth; cIL += convert_float8(cSL2);
	cSL3 = vload8(0, frame2 + i); i += cwidth; cIL += convert_float8(cSL3);
	cSL4 = vload8(0, frame2 + i); i += cwidth; cIL += convert_float8(cSL4); 
	cSL5 = vload8(0, frame2 + i); i += cwidth; cIL += convert_float8(cSL5);
	cSL6 = vload8(0, frame2 + i); i += cwidth; cIL += convert_float8(cSL6);
	cSL7 = vload8(0, frame2 + i); i -= 7*cwidth; cIL += convert_float8(cSL7); 
	M2 = (cIL.s0 + cIL.s1 + cIL.s2 + cIL.s3 + cIL.s4 + cIL.s5 + cIL.s6 + cIL.s7)/64;
	cIL = convert_float8(cSL0) - M2; cIL *= cIL; 
	cIL1 = convert_float8(cSL1) - M2; cIL = mad(cIL1,cIL1,cIL); 
	cIL1 = convert_float8(cSL2) - M2; cIL = mad(cIL1,cIL1,cIL); 
	cIL1 = convert_float8(cSL3) - M2; cIL = mad(cIL1,cIL1,cIL); 
	cIL1 = convert_float8(cSL4) - M2; cIL = mad(cIL1,cIL1,cIL); 
	cIL1 = convert_float8(cSL5) - M2; cIL = mad(cIL1,cIL1,cIL); 
	cIL1 = convert_float8(cSL5) - M2; cIL = mad(cIL1,cIL1,cIL); 
	cIL1 = convert_float8(cSL6) - M2; cIL = mad(cIL1,cIL1,cIL); 
	cIL1 = convert_float8(cSL7) - M2; cIL = mad(cIL1,cIL1,cIL); 
	D2 = (cIL.s0 + cIL.s1 + cIL.s2 + cIL.s3 + cIL.s4 + cIL.s5 + cIL.s6 + cIL.s7)/64;
	cIL = (convert_float8(cFL0) - M1)*(convert_float8(cSL0) - M2);
	cIL += (convert_float8(cFL1) - M1)*(convert_float8(cSL1) - M2);
	cIL += (convert_float8(cFL2) - M1)*(convert_float8(cSL2) - M2);
	cIL += (convert_float8(cFL3) - M1)*(convert_float8(cSL3) - M2);
	cIL += (convert_float8(cFL4) - M1)*(convert_float8(cSL4) - M2);
	cIL += (convert_float8(cFL5) - M1)*(convert_float8(cSL5) - M2);
	cIL += (convert_float8(cFL6) - M1)*(convert_float8(cSL6) - M2);
	cIL += (convert_float8(cFL7) - M1)*(convert_float8(cSL7) - M2);
	C = (cIL.s0 + cIL.s1 + cIL.s2 + cIL.s3 + cIL.s4 + cIL.s5 + cIL.s6 + cIL.s7)/64;
	C = mad(M1,M2*2,c1)*mad(C,2,c2)/(mad(M1,M1,mad(M2,M2,c1))*(D1 + D2 + c2));
	
	D1 = MBs[mb_num].SSIM;
	C += (reset == 1) ? 0 : D1;
	MBs[mb_num].SSIM = C;
	return;
}


#ifdef LOOP_FILTER
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
#endif
