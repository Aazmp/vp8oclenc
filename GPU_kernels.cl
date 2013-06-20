#pragma OPENCL EXTENSION cl_khr_byte_addressable_store : enable

typedef struct {
    short coeffs[25][16];
    int vector_x[4];
    int vector_y[4];
} macroblock;

__kernel void luma_interpolate_Hx4( __global uchar *const src_frame, //0
  								__global uchar *const dst_frame, //1
									const int width, //2
									const int height) //3
{
	if (get_global_id(0) > (height-1)) return;

	__private uchar4 UC4;
	__private uchar16 UC16;
	
	int i, ind;
	int width_x4 = width*4;
	uchar right;
	
	ind = (get_global_id(0) + 1)*width - 1;
	right  = src_frame[ind];
	
	for (i = width-4; i >= 0; i -= 4)
	{
		ind = get_global_id(0)*width + i;
		UC4 = vload4(0, src_frame + ind);
		
		UC16.s0 = UC4.s0;
		UC16.s1 = (uchar)(mad24((int)UC4.s0, 3, (int)UC4.s1 + 2)/4);
		UC16.s2 = (uchar)(((int)UC4.s0 + (int)UC4.s1 + 1)/2);
		UC16.s3 = (uchar)(mad24((int)UC4.s1, 3, (int)UC4.s0 + 2)/4);
		UC16.s4 = UC4.s1;
		UC16.s5 = (uchar)(mad24((int)UC4.s1, 3, (int)UC4.s2 + 2)/4);
		UC16.s6 = (uchar)(((int)UC4.s1 + (int)UC4.s2 + 1)/2);
		UC16.s7 = (uchar)(mad24((int)UC4.s2, 3, (int)UC4.s1 + 2)/4);
		UC16.s8 = UC4.s2;
		UC16.s9 = (uchar)(mad24((int)UC4.s2, 3, (int)UC4.s3 + 2)/4);
		UC16.sA = (uchar)(((int)UC4.s2 + (int)UC4.s3 + 1)/2);
		UC16.sB = (uchar)(mad24((int)UC4.s3, 3, (int)UC4.s2 + 2)/4);
		UC16.sC = UC4.s3;
		UC16.sD = (uchar)(mad24((int)UC4.s3, 3, (int)right + 2)/4);
		UC16.sE = (uchar)(((int)UC4.s3 + (int)right + 1)/2);
		UC16.sF = (uchar)(mad24((int)right, 3, (int)UC4.s3 + 2)/4);
		
		ind = get_global_id(0)*width_x4 + (i*4);
		vstore16(UC16, 0, dst_frame + ind);
		
		right = UC4.s0;
	}
	return;	
}

__kernel void luma_interpolate_Vx4( __global uchar *const frame, //0
									const int width, //1
									const int height) //2
{
	if (get_global_id(0) > (width-1)) return;

	__private uchar4 UC0, UC1, UC2, UC3, below;

	int width_x4 = width*4;
	int i, ind;
	
	ind = (height-1)*width_x4 + (get_global_id(0)*4);
	below = vload4(0, frame + ind);
	
	for (i = height-1; i >= 0; --i)
	{
		ind = i*width_x4 + (get_global_id(0)*4);
		UC0 = vload4(0, frame + ind);
		
		UC1 = convert_uchar4(mad24(convert_int4(UC0), 3, convert_int4(below) + 2)/4);
		UC2 = convert_uchar4((convert_int4(UC0) + convert_int4(below) + 1)/2);
		UC3 = convert_uchar4(mad24(convert_int4(below), 3, convert_int4(UC0) + 2)/4);

		ind = (i*4)*width_x4 + (get_global_id(0)*4);
		vstore4(UC0, 0, frame + ind); ind += width_x4;
		vstore4(UC1, 0, frame + ind); ind += width_x4;
		vstore4(UC2, 0, frame + ind); ind += width_x4;
		vstore4(UC3, 0, frame + ind);
		
		below = UC0;
	}
	return;	
}


void weight(int4 * const __L0, int4 * const __L1, int4 * const __L2, int4 * const __L3, const int * const dc_q, const int * const ac_q) 
{
	int4 L0, L1, L2, L3;
	L0 = *__L0 + *__L3;	//a1 = (ip[0] + ip[3]);
    L1 = *__L1 + *__L2;	//b1 = (ip[1] + ip[2]);
    L2 = *__L1 - *__L2;	//c1 = (ip[1] - ip[2]);
    L3 = *__L0 - *__L3;	//d1 = (ip[0] - ip[3]);

    *__L0 = L0 + L1;	//op[0] = a1 + b1;
    *__L1 = L2 + L3;	//op[1] = c1 + d1;
	*__L2 = L0 - L1;	//op[2] = a1 - b1;
    *__L3 = L3 - L2;	//op[3] = d1 - c1;
	
	L0.x = (*__L0).x + (*__L0).w;   //a1 = ip[0] + ip[3];
	L1.x = (*__L1).x + (*__L1).w;	
	L2.x = (*__L2).x + (*__L2).w;
	L3.x = (*__L3).x + (*__L3).w;
	
    L0.y = (*__L0).y + (*__L0).z;   //b1 = ip[1] + ip[2];
	L1.y = (*__L1).y + (*__L1).z;	
	L2.y = (*__L2).y + (*__L2).z;
	L3.y = (*__L3).y + (*__L3).z;   
	
    L0.z = (*__L0).y - (*__L0).z;   //c1 = ip[1] - ip[2];
	L1.z = (*__L1).y - (*__L1).z;	
	L2.z = (*__L2).y - (*__L2).z;
	L3.z = (*__L3).y - (*__L3).z;
    
	L0.w = (*__L0).x - (*__L0).w;   //d1 = ip[0] - ip[3];
	L1.w = (*__L1).x - (*__L1).w;	
	L2.w = (*__L2).x - (*__L2).w;
	L3.w = (*__L3).x - (*__L3).w;

    (*__L0).x = L0.x + L0.y;   //a2 = a1 + b1;
	(*__L1).x = L1.x + L1.y;
	(*__L2).x = L2.x + L2.y;
	(*__L3).x = L3.x + L3.y;
	
	(*__L0).y = L0.z + L0.w;   //b2 = c1 + d1;
	(*__L1).y = L1.z + L1.w;
	(*__L2).y = L2.z + L2.w;
	(*__L3).y = L3.z + L3.w;
	
	(*__L0).z = L0.x - L0.y;   //c2 = a1 - b1;
	(*__L1).z = L1.x - L1.y;
	(*__L2).z = L2.x - L2.y;
	(*__L3).z = L3.x - L3.y;
        
	(*__L0).w = L0.w - L0.z;   //d2 = d1 - c1;
	(*__L1).w = L1.w - L1.z;
	(*__L2).w = L2.w - L2.z;
	(*__L3).w = L3.w - L3.z;

    (*__L0).x += ((*__L0).x > 0);   //a2 += (a2 > 0);
	(*__L1).x += ((*__L1).x > 0);
	(*__L2).x += ((*__L2).x > 0);
	(*__L3).x += ((*__L3).x > 0);
    (*__L0).y += ((*__L0).y > 0);   //b2 += (b2 > 0);
	(*__L1).y += ((*__L1).y > 0);
	(*__L2).y += ((*__L2).y > 0);
	(*__L3).y += ((*__L3).y > 0); 
    (*__L0).z += ((*__L0).z > 0);   //c2 += (c2 > 0);
	(*__L1).z += ((*__L1).z > 0);
	(*__L2).z += ((*__L2).z > 0);
	(*__L3).z += ((*__L3).z > 0);
    (*__L0).w += ((*__L0).w > 0);   //d2 += (d2 > 0);
	(*__L1).w += ((*__L1).w > 0);
	(*__L2).w += ((*__L2).w > 0);
	(*__L3).w += ((*__L3).w > 0);
	
	*__L0 >>= 1; //op[0] = ((a2) >> 1);
    *__L1 >>= 1; //op[4] = ((b2) >> 1);
    *__L2 >>= 1; //op[8] = ((c2) >> 1);
    *__L3 >>= 1; //op[12] = ((d2) >> 1);
	
	*__L0 = convert_int4(abs(*__L0));
	*__L1 = convert_int4(abs(*__L1));
	*__L2 = convert_int4(abs(*__L2));
	*__L3 = convert_int4(abs(*__L3));
	
	*__L0 += *__L1 + *__L2 + *__L3;
	(*__L0).x += (*__L0).y + (*__L0).z + (*__L0).w;
	
	(*__L0).x *= 16;
	
	return;
}


void weight_SSD(int4 * const L0, int4 * const L1, int4 * const L2, int4 * const L3, const int * const dc_q, const int * const ac_q) // best results, but why?
{
	*L0 = mad24(*L0, *L0, mad24(*L1, *L1, mad24(*L2, *L2, mul24(*L3, *L3))));
	(*L0).x += (*L0).y + (*L0).z + (*L0).w;
}

/*
void weight_toyRD(int4 * const Line0, int4 * const Line1, int4 * const Line2, int4 * const Line3, const int * const dc_q, const int * const ac_q) // bad results
{
	private const uchar bit_weights[32] = { 0, 1, 3, 4, 4, 5, 5, 6,
											6, 6, 6, 8,	8, 8, 8, 9, 
											8, 8, 8, 9, 9, 9, 9, 9,
											9, 9, 9, 9, 9, 9, 9, 10};
	
	private int4 LineA; 
	
	LineA = *Line0; 
	*Line0 = (LineA + *Line3) >> 3;	
	*Line3 = (LineA - *Line3); 
	LineA = *Line1; 
	*Line1 = (LineA + *Line2) >> 3;	
	*Line2 = (LineA - *Line2); 
	
	LineA = *Line2; 
	
	*Line2 = *Line0 - *Line1;
	*Line0 = *Line0 + *Line1; 
				
	*Line1 = (mad24(LineA, 2217, mad24(*Line3, 5352, 1813))) >> 9;
	*Line3 = (mad24(*Line3, 2217, mad24(LineA, -5352, 938))) >> 9;
	
	//--
	
	LineA.x = (*Line0).x + (*Line0).w;
	LineA.y = (*Line1).x + (*Line1).w;
	LineA.z = (*Line2).x + (*Line2).w;
	LineA.w = (*Line3).x + (*Line3).w;
	
	(*Line0).w = (*Line0).x - (*Line0).w;
	(*Line1).w = (*Line1).x - (*Line1).w;
	(*Line2).w = (*Line2).x - (*Line2).w;
	(*Line3).w = (*Line3).x - (*Line3).w;
	
	(*Line0).x = (*Line0).y + (*Line0).z;
	(*Line1).x = (*Line1).y + (*Line1).z;
	(*Line2).x = (*Line2).y + (*Line2).z;
	(*Line3).x = (*Line3).y + (*Line3).z;
	
	(*Line0).z = (*Line0).y - (*Line0).z;
	(*Line1).z = (*Line1).y - (*Line1).z;
	(*Line2).z = (*Line2).y - (*Line2).z;
	(*Line3).z = (*Line3).y - (*Line3).z;
		
	//--
	
	(*Line0).y = ((mad24((*Line0).z, 554, mad24((*Line0).w, 1338, 3000))) /16384) + ((*Line0).w != 0);
	(*Line1).y = ((mad24((*Line1).z, 554, mad24((*Line1).w, 1338, 3000))) /16384) + ((*Line1).w != 0);
	(*Line2).y = ((mad24((*Line2).z, 554, mad24((*Line2).w, 1338, 3000))) /16384) + ((*Line2).w != 0);
	(*Line3).y = ((mad24((*Line3).z, 554, mad24((*Line3).w, 1338, 3000))) /16384) + ((*Line3).w != 0);
	
	(*Line0).w = (mad24((*Line0).w, 554, mad24((*Line0).z, -1338, 12750))) >> 14;
	(*Line1).w = (mad24((*Line1).w, 554, mad24((*Line1).z, -1338, 12750))) >> 14;
	(*Line2).w = (mad24((*Line2).w, 554, mad24((*Line2).z, -1338, 12750))) >> 14;
	(*Line3).w = (mad24((*Line3).w, 554, mad24((*Line3).z, -1338, 12750))) >> 14;
	
	(*Line0).z = ((LineA.x - (*Line0).x + 7) >> 4);
	(*Line1).z = ((LineA.y - (*Line1).x + 7) >> 4);
	(*Line2).z = ((LineA.z - (*Line2).x + 7) >> 4);
	(*Line3).z = ((LineA.w - (*Line3).x + 7) >> 4);
	
	(*Line0).x = ((LineA.x + (*Line0).x + 7) >> 4);
	(*Line1).x = ((LineA.y + (*Line1).x + 7) >> 4);
	(*Line2).x = ((LineA.z + (*Line2).x + 7) >> 4);
	(*Line3).x = ((LineA.w + (*Line3).x + 7) >> 4);
	
	*Line0 = convert_int4(abs(*Line0));	
	*Line1 = convert_int4(abs(*Line1));
	*Line2 = convert_int4(abs(*Line2));	
	*Line3 = convert_int4(abs(*Line3));
	
	LineA = (*Line0 % (int4)(*dc_q, *ac_q, *ac_q, *ac_q)) + (*Line1 % *ac_q) + (*Line2 % *ac_q) + (*Line3 % *ac_q);
	LineA.x += LineA.y + LineA.z + LineA.w;
	
	LineA.x /= 2;
	
	*Line0 /= (int4)(*dc_q, *ac_q, *ac_q, *ac_q); 
	*Line1 /= *ac_q; *Line2 /= *ac_q; *Line3 /= *ac_q;
	
	*Line0 = select(*Line0,31,*Line0>=31);
	*Line1 = select(*Line1,31,*Line1>=31);
	*Line2 = select(*Line2,31,*Line2>=31);
	*Line3 = select(*Line3,31,*Line3>=31);
		
	(*Line0).x = bit_weights[(*Line0).x]; (*Line0).y = bit_weights[(*Line0).y]; (*Line0).z = bit_weights[(*Line0).z]; (*Line0).w = bit_weights[(*Line0).w];
	(*Line1).x = bit_weights[(*Line1).x]; (*Line1).y = bit_weights[(*Line1).y]; (*Line0).z = bit_weights[(*Line1).z]; (*Line1).w = bit_weights[(*Line1).w];
	(*Line2).x = bit_weights[(*Line2).x]; (*Line2).y = bit_weights[(*Line2).y]; (*Line0).z = bit_weights[(*Line2).z]; (*Line2).w = bit_weights[(*Line2).w];
	(*Line3).x = bit_weights[(*Line3).x]; (*Line3).y = bit_weights[(*Line3).y]; (*Line0).z = bit_weights[(*Line3).z]; (*Line3).w = bit_weights[(*Line3).w];
	
	*Line0 += (*Line1) + (*Line2) + (*Line3);
	(*Line0).x += (*Line0).y + (*Line0).z + (*Line0).w + LineA.x;
	
	return ;
}
*/

void DCT_and_quant(int4 Line0, int4 Line1, int4 Line2, int4 Line3, // <- input differences
					int4 *__Line0, int4 *__Line1, int4 *__Line2, int4 *__Line3, const int dc_q, const int ac_q) // -> output DCT Lines
{

	*__Line0 = (Line0 + Line3) *8;	// a1 = ((ip[0] + ip[3])<<3);
	*__Line1 = (Line1 + Line2) *8;	// b1 = ((ip[1] + ip[2])<<3);
	*__Line2 = (Line1 - Line2) *8;	// c1 = ((ip[1] - ip[2])<<3);
	*__Line3 = (Line0 - Line3) *8;	// d1 = ((ip[0] - ip[3])<<3);
	

	Line0 = *__Line0 + *__Line1;				// op[0] = (a1 + b1); 
	Line2 = *__Line0 - *__Line1;				// op[2] = (a1 - b1);
	
	Line1 = ((*__Line2 * 2217) + (*__Line3 * 5352) + 14500) /4096;
														// op[1] = (c1 * 2217 + d1 * 5352 +  14500)>>12;
	Line3 = ((*__Line3 * 2217) - (*__Line2 * 5352) + 7500) /4096;
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
	
	*__Line1 = ((Line2 * 2217) + (Line3 * 5352) + 12000) /65536;
	(*__Line1).x += (Line3.x != 0);
	(*__Line1).y += (Line3.y != 0);
	(*__Line1).z += (Line3.z != 0);
	(*__Line1).w += (Line3.w != 0);				// op[1]  = (((c1 * 2217 + d1 * 5352 +  12000)>>16) + (d1!=0));
	
	*__Line3 = (((Line3 * 2217) - (Line2 * 5352) + 51000) /65536);
														// op[3] = ((d1 * 2217 - c1 * 5352 +  51000)>>16);
														
	*__Line0 /= (int4)(dc_q, ac_q, ac_q, ac_q);
	*__Line1 /= ac_q;
	*__Line2 /= ac_q;
	*__Line3 /= ac_q;
	
	return;
}

void dequant_and_iDCT(int4 *__Line0, int4 *__Line1, int4 *__Line2, int4 *__Line3,
					  int4 Line0, int4 Line1, int4 Line2, int4 Line3, const int dc_q, const int ac_q) // <- input DCT lines
{
	// dequant
	Line0 *= (int4)(dc_q, ac_q, ac_q, ac_q);
	Line1 *= ac_q;
	Line2 *= ac_q;
	Line3 *= ac_q;
	
		
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
	return;
}

#define GROUP_SIZE_FOR_SEARCH 256
__kernel __attribute__((reqd_work_group_size(GROUP_SIZE_FOR_SEARCH, 1, 1)))
		void luma_search( 	__global uchar *const current_frame, //0
							__global uchar *const prev_frame, //1
							__global macroblock *const MBs, //2
							const signed int width, //3
							const signed int height, //4
							const signed int first_MBlock_offset, //5
							const int deltaX, //6
							const int deltaY,//7
							const int dc_q, //8
							const int ac_q) //9
{
	// do not use <<>> instead of */ on signed integer
	// even if operands are greater than 0
	// ...some mysterious shit happens	

	__private uchar8 CL0, CL1, CL2, CL3, CL4, CL5, CL6, CL7;
	
	__private uchar4 UC00, UC01, UC02, UC03,
							UC10, UC11, UC12, UC13,
							UC20, UC21, UC22, UC23,
							UC30, UC31, UC32, UC33;
	__private int4 DL0, DL1, DL2, DL3;
	
	__private int start_x, end_x, start_y, end_y; 
	__private unsigned int MinDiff, Diff0, Diff1, Diff2, Diff3;	
	__private int px, py;
    __private int cx, cy;
    __private int vector_x, vector_y;   
	__private int ci; 
	__private int pi;   
	__private int width_x4 = width*4;
	__private int width_x4_x4 = width_x4*4;
	__private int mb_num;
	__private int b8x8_num;  
	
	// now b8x8_num represents absolute number of 8x8 block
	b8x8_num = (first_MBlock_offset + get_global_id(0));
	if (b8x8_num > ((width*height/64)-1)) return;
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

	cx *= 4; cy *= 4; //into qpel
	
	start_x = 0; end_x = width_x4 - 20; start_y = 0; end_y = (height*4) - 32;
	
	vector_x = 0;
	vector_y = 0; 

	start_x	= cx - deltaX;
	end_x = cx + deltaX;
	start_y = cy - deltaY;
	end_y = cy + deltaY;
	
	start_x = (start_x < 0) ? 0 : start_x;
	end_x = (end_x > (width_x4 - 32)) ? (width_x4 - 32) : end_x;
	start_y = (start_y < 0) ? 0 : start_y;
	end_y = (end_y > ((height*4) - 32)) ? ((height*4) - 32) : end_y;
	
	
	start_x &= ~0x3;
	start_y &= ~0x3;
	#pragma unroll 1
	for (px = start_x; px <= end_x; px+=4 )
	{
		#pragma unroll 1
		for (py = start_y; py <= end_y; py+=4)
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
			weight(&DL0, &DL1, &DL2, &DL3, &dc_q, &ac_q);	Diff1 = DL0.x;
			// block 10			
			UC00 = vload4(0,prev_frame+pi); UC01 = vload4(0,prev_frame+pi+4); UC02 = vload4(0,prev_frame+pi+8); UC03 = vload4(0,prev_frame+pi+12); pi += width_x4_x4;
			UC10 = vload4(0,prev_frame+pi); UC11 = vload4(0,prev_frame+pi+4); UC12 = vload4(0,prev_frame+pi+8); UC13 = vload4(0,prev_frame+pi+12); pi += width_x4_x4;
			UC20 = vload4(0,prev_frame+pi); UC21 = vload4(0,prev_frame+pi+4); UC22 = vload4(0,prev_frame+pi+8); UC23 = vload4(0,prev_frame+pi+12); pi += width_x4_x4;
			UC30 = vload4(0,prev_frame+pi); UC31 = vload4(0,prev_frame+pi+4); UC32 = vload4(0,prev_frame+pi+8); UC33 = vload4(0,prev_frame+pi+12); pi += width_x4_x4;
			DL0 = convert_int4(CL4.s0123) - convert_int4((uchar4)(UC00.x, UC01.x, UC02.x, UC03.x));
			DL1 = convert_int4(CL5.s0123) - convert_int4((uchar4)(UC10.x, UC11.x, UC12.x, UC13.x));
			DL2 = convert_int4(CL6.s0123) - convert_int4((uchar4)(UC20.x, UC21.x, UC22.x, UC23.x));
			DL3 = convert_int4(CL7.s0123) - convert_int4((uchar4)(UC30.x, UC31.x, UC32.x, UC33.x));
			weight(&DL0, &DL1, &DL2, &DL3, &dc_q, &ac_q);	Diff1 += DL0.x;
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
			weight(&DL0, &DL1, &DL2, &DL3, &dc_q, &ac_q);	Diff1 += DL0.x;
			// block 11
			UC00 = vload4(0,prev_frame+pi); UC01 = vload4(0,prev_frame+pi+4); UC02 = vload4(0,prev_frame+pi+8); UC03 = vload4(0,prev_frame+pi+12); pi += width_x4_x4;
			UC10 = vload4(0,prev_frame+pi); UC11 = vload4(0,prev_frame+pi+4); UC12 = vload4(0,prev_frame+pi+8); UC13 = vload4(0,prev_frame+pi+12); pi += width_x4_x4;
			UC20 = vload4(0,prev_frame+pi); UC21 = vload4(0,prev_frame+pi+4); UC22 = vload4(0,prev_frame+pi+8); UC23 = vload4(0,prev_frame+pi+12); pi += width_x4_x4;
			UC30 = vload4(0,prev_frame+pi); UC31 = vload4(0,prev_frame+pi+4); UC32 = vload4(0,prev_frame+pi+8); UC33 = vload4(0,prev_frame+pi+12); pi += width_x4_x4;
			DL0 = convert_int4(CL4.s4567) - convert_int4((uchar4)(UC00.x, UC01.x, UC02.x, UC03.x));
			DL1 = convert_int4(CL5.s4567) - convert_int4((uchar4)(UC10.x, UC11.x, UC12.x, UC13.x));
			DL2 = convert_int4(CL6.s4567) - convert_int4((uchar4)(UC20.x, UC21.x, UC22.x, UC23.x));
			DL3 = convert_int4(CL7.s4567) - convert_int4((uchar4)(UC30.x, UC31.x, UC32.x, UC33.x));
			weight(&DL0, &DL1, &DL2, &DL3, &dc_q, &ac_q);	Diff1 += DL0.x;

			Diff2 = abs(px - cx);
			Diff1 += (Diff2 > 8) ? 10 : ((Diff2 > 0) ? 3 : 0);
			Diff2 = abs(py - cy);
			Diff1 += (Diff2 > 8) ? 10 : ((Diff2 > 0) ? 3 : 0);
			
			if (Diff1 < MinDiff)
			{  
				MinDiff = Diff1;
				vector_x = px - cx;
				vector_y = py - cy;
			} 
       } 
    }  

	start_x = cx - 4 + vector_x;
	end_x = cx + 3 + vector_x;
	start_y = cy - 4 + vector_y;
	end_y = cy + 4 + vector_y;
	
	start_x = (start_x < 1) ? 1 : start_x;
	end_x = (end_x > (width_x4 - 33)) ? (width_x4 - 33) : end_x;
	start_y = (start_y < 1) ? 1 : start_y;
	end_y = (end_y > ((height*4) - 33)) ? ((height*4) - 33) : end_y;

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
			weight(&DL0, &DL1, &DL2, &DL3, &dc_q, &ac_q);	Diff0 += DL0.x;
			DL0 = convert_int4(CL0.s0123) - convert_int4((uchar4)(UC00.y, UC01.y, UC02.y, UC03.y));
			DL1 = convert_int4(CL1.s0123) - convert_int4((uchar4)(UC10.y, UC11.y, UC12.y, UC13.y));
			DL2 = convert_int4(CL2.s0123) - convert_int4((uchar4)(UC20.y, UC21.y, UC22.y, UC23.y));
			DL3 = convert_int4(CL3.s0123) - convert_int4((uchar4)(UC30.y, UC31.y, UC32.y, UC33.y));
			weight(&DL0, &DL1, &DL2, &DL3, &dc_q, &ac_q);	Diff1 = DL0.x;
			DL0 = convert_int4(CL0.s0123) - convert_int4((uchar4)(UC00.z, UC01.z, UC02.z, UC03.z));
			DL1 = convert_int4(CL1.s0123) - convert_int4((uchar4)(UC10.z, UC11.z, UC12.z, UC13.z));
			DL2 = convert_int4(CL2.s0123) - convert_int4((uchar4)(UC20.z, UC21.z, UC22.z, UC23.z));
			DL3 = convert_int4(CL3.s0123) - convert_int4((uchar4)(UC30.z, UC31.z, UC32.z, UC33.z));
			weight(&DL0, &DL1, &DL2, &DL3, &dc_q, &ac_q);	Diff2 = DL0.x;
			DL0 = convert_int4(CL0.s0123) - convert_int4((uchar4)(UC00.w, UC01.w, UC02.w, UC03.w));
			DL1 = convert_int4(CL1.s0123) - convert_int4((uchar4)(UC10.w, UC11.w, UC12.w, UC13.w));
			DL2 = convert_int4(CL2.s0123) - convert_int4((uchar4)(UC20.w, UC21.w, UC22.w, UC23.w));
			DL3 = convert_int4(CL3.s0123) - convert_int4((uchar4)(UC30.w, UC31.w, UC32.w, UC33.w));
			weight(&DL0, &DL1, &DL2, &DL3, &dc_q, &ac_q);	Diff3 = DL0.x;
			// block 10			
			UC00 = vload4(0,prev_frame+pi); UC01 = vload4(0,prev_frame+pi+4); UC02 = vload4(0,prev_frame+pi+8); UC03 = vload4(0,prev_frame+pi+12); pi += width_x4_x4;
			UC10 = vload4(0,prev_frame+pi); UC11 = vload4(0,prev_frame+pi+4); UC12 = vload4(0,prev_frame+pi+8); UC13 = vload4(0,prev_frame+pi+12); pi += width_x4_x4;
			UC20 = vload4(0,prev_frame+pi); UC21 = vload4(0,prev_frame+pi+4); UC22 = vload4(0,prev_frame+pi+8); UC23 = vload4(0,prev_frame+pi+12); pi += width_x4_x4;
			UC30 = vload4(0,prev_frame+pi); UC31 = vload4(0,prev_frame+pi+4); UC32 = vload4(0,prev_frame+pi+8); UC33 = vload4(0,prev_frame+pi+12); pi += width_x4_x4;
			DL0 = convert_int4(CL4.s0123) - convert_int4((uchar4)(UC00.x, UC01.x, UC02.x, UC03.x));
			DL1 = convert_int4(CL5.s0123) - convert_int4((uchar4)(UC10.x, UC11.x, UC12.x, UC13.x));
			DL2 = convert_int4(CL6.s0123) - convert_int4((uchar4)(UC20.x, UC21.x, UC22.x, UC23.x));
			DL3 = convert_int4(CL7.s0123) - convert_int4((uchar4)(UC30.x, UC31.x, UC32.x, UC33.x));
			weight(&DL0, &DL1, &DL2, &DL3, &dc_q, &ac_q);	Diff0 += DL0.x;
			DL0 = convert_int4(CL4.s0123) - convert_int4((uchar4)(UC00.y, UC01.y, UC02.y, UC03.y));
			DL1 = convert_int4(CL5.s0123) - convert_int4((uchar4)(UC10.y, UC11.y, UC12.y, UC13.y));
			DL2 = convert_int4(CL6.s0123) - convert_int4((uchar4)(UC20.y, UC21.y, UC22.y, UC23.y));
			DL3 = convert_int4(CL7.s0123) - convert_int4((uchar4)(UC30.y, UC31.y, UC32.y, UC33.y));
			weight(&DL0, &DL1, &DL2, &DL3, &dc_q, &ac_q);	Diff1 += DL0.x;
			DL0 = convert_int4(CL4.s0123) - convert_int4((uchar4)(UC00.z, UC01.z, UC02.z, UC03.z));
			DL1 = convert_int4(CL5.s0123) - convert_int4((uchar4)(UC10.z, UC11.z, UC12.z, UC13.z));
			DL2 = convert_int4(CL6.s0123) - convert_int4((uchar4)(UC20.z, UC21.z, UC22.z, UC23.z));
			DL3 = convert_int4(CL7.s0123) - convert_int4((uchar4)(UC30.z, UC31.z, UC32.z, UC33.z));
			weight(&DL0, &DL1, &DL2, &DL3, &dc_q, &ac_q);	Diff2 += DL0.x;
			DL0 = convert_int4(CL4.s0123) - convert_int4((uchar4)(UC00.w, UC01.w, UC02.w, UC03.w));
			DL1 = convert_int4(CL5.s0123) - convert_int4((uchar4)(UC10.w, UC11.w, UC12.w, UC13.w));
			DL2 = convert_int4(CL6.s0123) - convert_int4((uchar4)(UC20.w, UC21.w, UC22.w, UC23.w));
			DL3 = convert_int4(CL7.s0123) - convert_int4((uchar4)(UC30.w, UC31.w, UC32.w, UC33.w));
			weight(&DL0, &DL1, &DL2, &DL3, &dc_q, &ac_q);	Diff3 += DL0.x;
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
			weight(&DL0, &DL1, &DL2, &DL3, &dc_q, &ac_q);	Diff0 += DL0.x;
			DL0 = convert_int4(CL0.s4567) - convert_int4((uchar4)(UC00.y, UC01.y, UC02.y, UC03.y));
			DL1 = convert_int4(CL1.s4567) - convert_int4((uchar4)(UC10.y, UC11.y, UC12.y, UC13.y));
			DL2 = convert_int4(CL2.s4567) - convert_int4((uchar4)(UC20.y, UC21.y, UC22.y, UC23.y));
			DL3 = convert_int4(CL3.s4567) - convert_int4((uchar4)(UC30.y, UC31.y, UC32.y, UC33.y));
			weight(&DL0, &DL1, &DL2, &DL3, &dc_q, &ac_q);	Diff1 += DL0.x;
			DL0 = convert_int4(CL0.s4567) - convert_int4((uchar4)(UC00.z, UC01.z, UC02.z, UC03.z));
			DL1 = convert_int4(CL1.s4567) - convert_int4((uchar4)(UC10.z, UC11.z, UC12.z, UC13.z));
			DL2 = convert_int4(CL2.s4567) - convert_int4((uchar4)(UC20.z, UC21.z, UC22.z, UC23.z));
			DL3 = convert_int4(CL3.s4567) - convert_int4((uchar4)(UC30.z, UC31.z, UC32.z, UC33.z));
			weight(&DL0, &DL1, &DL2, &DL3, &dc_q, &ac_q);	Diff2 += DL0.x;
			DL0 = convert_int4(CL0.s4567) - convert_int4((uchar4)(UC00.w, UC01.w, UC02.w, UC03.w));
			DL1 = convert_int4(CL1.s4567) - convert_int4((uchar4)(UC10.w, UC11.w, UC12.w, UC13.w));
			DL2 = convert_int4(CL2.s4567) - convert_int4((uchar4)(UC20.w, UC21.w, UC22.w, UC23.w));
			DL3 = convert_int4(CL3.s4567) - convert_int4((uchar4)(UC30.w, UC31.w, UC32.w, UC33.w));
			weight(&DL0, &DL1, &DL2, &DL3, &dc_q, &ac_q);	Diff3 += DL0.x;
			// block 11
			UC00 = vload4(0,prev_frame+pi); UC01 = vload4(0,prev_frame+pi+4); UC02 = vload4(0,prev_frame+pi+8); UC03 = vload4(0,prev_frame+pi+12); pi += width_x4_x4;
			UC10 = vload4(0,prev_frame+pi); UC11 = vload4(0,prev_frame+pi+4); UC12 = vload4(0,prev_frame+pi+8); UC13 = vload4(0,prev_frame+pi+12); pi += width_x4_x4;
			UC20 = vload4(0,prev_frame+pi); UC21 = vload4(0,prev_frame+pi+4); UC22 = vload4(0,prev_frame+pi+8); UC23 = vload4(0,prev_frame+pi+12); pi += width_x4_x4;
			UC30 = vload4(0,prev_frame+pi); UC31 = vload4(0,prev_frame+pi+4); UC32 = vload4(0,prev_frame+pi+8); UC33 = vload4(0,prev_frame+pi+12); pi += width_x4_x4;
			DL0 = convert_int4(CL4.s4567) - convert_int4((uchar4)(UC00.x, UC01.x, UC02.x, UC03.x));
			DL1 = convert_int4(CL5.s4567) - convert_int4((uchar4)(UC10.x, UC11.x, UC12.x, UC13.x));
			DL2 = convert_int4(CL6.s4567) - convert_int4((uchar4)(UC20.x, UC21.x, UC22.x, UC23.x));
			DL3 = convert_int4(CL7.s4567) - convert_int4((uchar4)(UC30.x, UC31.x, UC32.x, UC33.x));
			weight(&DL0, &DL1, &DL2, &DL3, &dc_q, &ac_q);	Diff0 += DL0.x;
			DL0 = convert_int4(CL4.s4567) - convert_int4((uchar4)(UC00.y, UC01.y, UC02.y, UC03.y));
			DL1 = convert_int4(CL5.s4567) - convert_int4((uchar4)(UC10.y, UC11.y, UC12.y, UC13.y));
			DL2 = convert_int4(CL6.s4567) - convert_int4((uchar4)(UC20.y, UC21.y, UC22.y, UC23.y));
			DL3 = convert_int4(CL7.s4567) - convert_int4((uchar4)(UC30.y, UC31.y, UC32.y, UC33.y));
			weight(&DL0, &DL1, &DL2, &DL3, &dc_q, &ac_q);	Diff1 += DL0.x;
			DL0 = convert_int4(CL4.s4567) - convert_int4((uchar4)(UC00.z, UC01.z, UC02.z, UC03.z));
			DL1 = convert_int4(CL5.s4567) - convert_int4((uchar4)(UC10.z, UC11.z, UC12.z, UC13.z));
			DL2 = convert_int4(CL6.s4567) - convert_int4((uchar4)(UC20.z, UC21.z, UC22.z, UC23.z));
			DL3 = convert_int4(CL7.s4567) - convert_int4((uchar4)(UC30.z, UC31.z, UC32.z, UC33.z));
			weight(&DL0, &DL1, &DL2, &DL3, &dc_q, &ac_q);	Diff2 += DL0.x;
			DL0 = convert_int4(CL4.s4567) - convert_int4((uchar4)(UC00.w, UC01.w, UC02.w, UC03.w));
			DL1 = convert_int4(CL5.s4567) - convert_int4((uchar4)(UC10.w, UC11.w, UC12.w, UC13.w));
			DL2 = convert_int4(CL6.s4567) - convert_int4((uchar4)(UC20.w, UC21.w, UC22.w, UC23.w));
			DL3 = convert_int4(CL7.s4567) - convert_int4((uchar4)(UC30.w, UC31.w, UC32.w, UC33.w));
			weight(&DL0, &DL1, &DL2, &DL3, &dc_q, &ac_q);	Diff3 += DL0.x;
			
			Diff1 += 6;
			Diff2 += 6;
			Diff3 += 6;

			if (Diff0 < MinDiff)
			{  
				MinDiff = Diff0;
				vector_x = px - cx;
				vector_y = py - cy;
			} 
			if (Diff1 < MinDiff)
			{  
				MinDiff = Diff1;
				vector_x = px+1 - cx;
				vector_y = py - cy;
			} 
			if (Diff2 < MinDiff)
			{  
				MinDiff = Diff2;
				vector_x = px+2 - cx;
				vector_y = py - cy;
			} 
			if (Diff3 < MinDiff)
			{  
				MinDiff = Diff3;
				vector_x = px+3 - cx;
				vector_y = py - cy;
			} 
       } 
    } 
	
	MBs[mb_num].vector_x[b8x8_num] = vector_x;
	MBs[mb_num].vector_y[b8x8_num] = vector_y;

	return;
	
}
	
	
__kernel void luma_transform( 	__global uchar *current_frame, //0
								__global uchar *recon_frame, //1
								__global uchar *prev_frame, //2
								__global macroblock *MBs, //3
								signed int width, //4
								signed int first_MBlock_offset, //5
								int dc_q, //6
								int ac_q) //7
	
{
	__private int4 DL0, DL1, DL2, DL3;
	__private int4 BDL0, BDL1, BDL2, BDL3;
	__private short8 DCTLine0, DCTLine1, DCTLine2, DCTLine3,
					  DCTLine4, DCTLine5, DCTLine6, DCTLine7;
	__private uchar4 CL, PL;
	
    __private int cx, cy, px, py, vector_x, vector_y;	
	__private int ci, pi; 
	__private int mb_num, b8x8_num, b4x4_in_mb,b8x8_in_mb;
	__private int width_x4 = width<<2;		
	
	b8x8_num = first_MBlock_offset + get_global_id(0); 
	cx = (b8x8_num % (width>>3))<<3; //in fpel
	cy = (b8x8_num / (width>>3))<<3;   
	mb_num = (cy>>4)*(width>>4) + (cx>>4);
	
	ci = cy*width + cx; 
	b4x4_in_mb = ((cy%16)/4)*4 + (cx%16)/4;
	b8x8_in_mb = ((cy%16)/8)*2 + (cx%16)/8;

	vector_x = MBs[mb_num].vector_x[b8x8_in_mb];
	vector_y = MBs[mb_num].vector_y[b8x8_in_mb];
	//if (mb_num == 7) printf((__constant char*)"b8x8_in_mb = %d, vy = %d\n", b8x8_in_mb, vector_y);

	//printf((__constant char*)"mb_num = %d : b8x8_num = %d : b4x4_in_mb = %d : b8x8_in_mb = %d\n", mb_num, b8x8_num, b4x4_in_mb, b8x8_in_mb);
	//printf((__constant char*)"vx = %d : vy = %d \n", vector_x, vector_y);
	
	//now go to qpel
	cx<<=2; cy<<=2;
	py = cy + vector_y;
	px = cx + vector_x;
	
	pi = py*width_x4+px;

	const int inv_zigzag[16] = { 0, 1, 5, 6, 2, 4, 7, 12, 3,  8, 11, 13, 9, 10, 14, 15 };
	
	// block 00
	//printf((__constant char*)"read ci = %d : pi = %d\n", ci, pi);
	CL = vload4(0, &current_frame[ci]); ci += width;
	PL.x = prev_frame[pi]; PL.y = prev_frame[pi+4]; PL.z = prev_frame[pi+8]; PL.w = prev_frame[pi+12]; pi += width_x4<<2;
	BDL0 = convert_int4(CL) - convert_int4(PL);
	//printf((__constant char*)"read ci = %d : pi = %d\n", ci, pi);
	CL = vload4(0, &current_frame[ci]); ci += width;
	PL.x = prev_frame[pi]; PL.y = prev_frame[pi+4]; PL.z = prev_frame[pi+8]; PL.w = prev_frame[pi+12]; pi += width_x4<<2;
	BDL1 = convert_int4(CL) - convert_int4(PL);
	//printf((__constant char*)"read ci = %d : pi = %d\n", ci, pi);
	CL = vload4(0, &current_frame[ci]); ci += width;
	PL.x = prev_frame[pi]; PL.y = prev_frame[pi+4]; PL.z = prev_frame[pi+8]; PL.w = prev_frame[pi+12]; pi += width_x4<<2;
	BDL2 = convert_int4(CL) - convert_int4(PL);
	//printf((__constant char*)"read ci = %d : pi = %d\n", ci, pi);
	CL = vload4(0, &current_frame[ci]); ci += width;
	PL.x = prev_frame[pi]; PL.y = prev_frame[pi+4]; PL.z = prev_frame[pi+8]; PL.w = prev_frame[pi+12]; pi += width_x4<<2;
	BDL3 = convert_int4(CL) - convert_int4(PL);
	ci -= (width<<2); pi -= (width_x4<<4);
	DCT_and_quant(BDL0, BDL1, BDL2, BDL3, &DL0, &DL1, &DL2, &DL3, dc_q, ac_q);
	DCTLine0.s0123 = convert_short4(DL0);
	DCTLine1.s0123 = convert_short4(DL1);
	DCTLine2.s0123 = convert_short4(DL2);
	DCTLine3.s0123 = convert_short4(DL3);
	// block 01
	ci += 4; pi += 16;
	//printf((__constant char*)"read ci = %d : pi = %d\n", ci, pi);
	CL = vload4(0, &current_frame[ci]); ci += width;
	PL.x = prev_frame[pi]; PL.y = prev_frame[pi+4]; PL.z = prev_frame[pi+8]; PL.w = prev_frame[pi+12]; pi += width_x4<<2;
	BDL0 = convert_int4(CL) - convert_int4(PL);
	//printf((__constant char*)"read ci = %d : pi = %d\n", ci, pi);
	CL = vload4(0, &current_frame[ci]); ci += width;
	PL.x = prev_frame[pi]; PL.y = prev_frame[pi+4]; PL.z = prev_frame[pi+8]; PL.w = prev_frame[pi+12]; pi += width_x4<<2;
	BDL1 = convert_int4(CL) - convert_int4(PL);
	//printf((__constant char*)"read ci = %d : pi = %d\n", ci, pi);
	CL = vload4(0, &current_frame[ci]); ci += width;
	PL.x = prev_frame[pi]; PL.y = prev_frame[pi+4]; PL.z = prev_frame[pi+8]; PL.w = prev_frame[pi+12]; pi += width_x4<<2;
	BDL2 = convert_int4(CL) - convert_int4(PL);
	//printf((__constant char*)"read ci = %d : pi = %d\n", ci, pi);
	CL = vload4(0, &current_frame[ci]); ci += width;
	PL.x = prev_frame[pi]; PL.y = prev_frame[pi+4]; PL.z = prev_frame[pi+8]; PL.w = prev_frame[pi+12]; pi += width_x4<<2;
	BDL3 = convert_int4(CL) - convert_int4(PL);
	ci -= 4; pi -= 16;
	DCT_and_quant(BDL0, BDL1, BDL2, BDL3, &DL0, &DL1, &DL2, &DL3, dc_q, ac_q);
	DCTLine0.s4567 = convert_short4(DL0);
	DCTLine1.s4567 = convert_short4(DL1);
	DCTLine2.s4567 = convert_short4(DL2);
	DCTLine3.s4567 = convert_short4(DL3);
	// block 10
	//printf((__constant char*)"read ci = %d : pi = %d\n", ci, pi);
	CL = vload4(0, &current_frame[ci]); ci += width;
	PL.x = prev_frame[pi]; PL.y = prev_frame[pi+4]; PL.z = prev_frame[pi+8]; PL.w = prev_frame[pi+12]; pi += width_x4<<2;
	BDL0 = convert_int4(CL) - convert_int4(PL);
	//printf((__constant char*)"read ci = %d : pi = %d\n", ci, pi);
	CL = vload4(0, &current_frame[ci]); ci += width;
	PL.x = prev_frame[pi]; PL.y = prev_frame[pi+4]; PL.z = prev_frame[pi+8]; PL.w = prev_frame[pi+12]; pi += width_x4<<2;
	BDL1 = convert_int4(CL) - convert_int4(PL);
	//printf((__constant char*)"read ci = %d : pi = %d\n", ci, pi);
	CL = vload4(0, &current_frame[ci]); ci += width;
	PL.x = prev_frame[pi]; PL.y = prev_frame[pi+4]; PL.z = prev_frame[pi+8]; PL.w = prev_frame[pi+12]; pi += width_x4<<2;
	BDL2 = convert_int4(CL) - convert_int4(PL);
	//printf((__constant char*)"read ci = %d : pi = %d\n", ci, pi);
	CL = vload4(0, &current_frame[ci]); ci += width;
	PL.x = prev_frame[pi]; PL.y = prev_frame[pi+4]; PL.z = prev_frame[pi+8]; PL.w = prev_frame[pi+12]; pi += width_x4<<2;
	BDL3 = convert_int4(CL) - convert_int4(PL);
	ci -= (width<<2); pi -= (width_x4<<4);
	DCT_and_quant(BDL0, BDL1, BDL2, BDL3, &DL0, &DL1, &DL2, &DL3, dc_q, ac_q);
	DCTLine4.s0123 = convert_short4(DL0);
	DCTLine5.s0123 = convert_short4(DL1);
	DCTLine6.s0123 = convert_short4(DL2);
	DCTLine7.s0123 = convert_short4(DL3);
	// block 11
	ci += 4; pi += 16;
	//printf((__constant char*)"read ci = %d : pi = %d\n", ci, pi);
	CL = vload4(0, &current_frame[ci]); ci += width;
	PL.x = prev_frame[pi]; PL.y = prev_frame[pi+4]; PL.z = prev_frame[pi+8]; PL.w = prev_frame[pi+12]; pi += width_x4<<2;
	BDL0 = convert_int4(CL) - convert_int4(PL);
	//printf((__constant char*)"read ci = %d : pi = %d\n", ci, pi);
	CL = vload4(0, &current_frame[ci]); ci += width;
	PL.x = prev_frame[pi]; PL.y = prev_frame[pi+4]; PL.z = prev_frame[pi+8]; PL.w = prev_frame[pi+12]; pi += width_x4<<2;
	BDL1 = convert_int4(CL) - convert_int4(PL);
	//printf((__constant char*)"read ci = %d : pi = %d\n", ci, pi);
	CL = vload4(0, &current_frame[ci]); ci += width;
	PL.x = prev_frame[pi]; PL.y = prev_frame[pi+4]; PL.z = prev_frame[pi+8]; PL.w = prev_frame[pi+12]; pi += width_x4<<2;
	BDL2 = convert_int4(CL) - convert_int4(PL);
	//printf((__constant char*)"read ci = %d : pi = %d\n", ci, pi);
	CL = vload4(0, &current_frame[ci]); ci += width;
	PL.x = prev_frame[pi]; PL.y = prev_frame[pi+4]; PL.z = prev_frame[pi+8]; PL.w = prev_frame[pi+12]; pi += width_x4<<2;
	BDL3 = convert_int4(CL) - convert_int4(PL);
	DCT_and_quant(BDL0, BDL1, BDL2, BDL3, &DL0, &DL1, &DL2, &DL3, dc_q, ac_q);
	DCTLine4.s4567 = convert_short4(DL0);
	DCTLine5.s4567 = convert_short4(DL1);
	DCTLine6.s4567 = convert_short4(DL2);
	DCTLine7.s4567 = convert_short4(DL3);
	ci -= 4; pi -= 16;
	ci -= (width<<3); pi -= (width_x4<<5);

	// block 00
	MBs[mb_num].coeffs[b4x4_in_mb][inv_zigzag[0]]=DCTLine0.s0;
	MBs[mb_num].coeffs[b4x4_in_mb][inv_zigzag[1]]=DCTLine1.s0; 
	MBs[mb_num].coeffs[b4x4_in_mb][inv_zigzag[2]]=DCTLine2.s0; 
	MBs[mb_num].coeffs[b4x4_in_mb][inv_zigzag[3]]=DCTLine3.s0;
	MBs[mb_num].coeffs[b4x4_in_mb][inv_zigzag[4]]=DCTLine0.s1; 
	MBs[mb_num].coeffs[b4x4_in_mb][inv_zigzag[5]]=DCTLine1.s1; 
	MBs[mb_num].coeffs[b4x4_in_mb][inv_zigzag[6]]=DCTLine2.s1; 
	MBs[mb_num].coeffs[b4x4_in_mb][inv_zigzag[7]]=DCTLine3.s1;
	MBs[mb_num].coeffs[b4x4_in_mb][inv_zigzag[8]]=DCTLine0.s2; 
	MBs[mb_num].coeffs[b4x4_in_mb][inv_zigzag[9]]=DCTLine1.s2; 
	MBs[mb_num].coeffs[b4x4_in_mb][inv_zigzag[10]]=DCTLine2.s2; 
	MBs[mb_num].coeffs[b4x4_in_mb][inv_zigzag[11]]=DCTLine3.s2;
	MBs[mb_num].coeffs[b4x4_in_mb][inv_zigzag[12]]=DCTLine0.s3; 
	MBs[mb_num].coeffs[b4x4_in_mb][inv_zigzag[13]]=DCTLine1.s3; 
	MBs[mb_num].coeffs[b4x4_in_mb][inv_zigzag[14]]=DCTLine2.s3; 
	MBs[mb_num].coeffs[b4x4_in_mb][inv_zigzag[15]]=DCTLine3.s3;
	// block 01
	MBs[mb_num].coeffs[b4x4_in_mb + 1][inv_zigzag[0]]=DCTLine0.s4;
	MBs[mb_num].coeffs[b4x4_in_mb + 1][inv_zigzag[1]]=DCTLine1.s4; 
	MBs[mb_num].coeffs[b4x4_in_mb + 1][inv_zigzag[2]]=DCTLine2.s4; 
	MBs[mb_num].coeffs[b4x4_in_mb + 1][inv_zigzag[3]]=DCTLine3.s4;
	MBs[mb_num].coeffs[b4x4_in_mb + 1][inv_zigzag[4]]=DCTLine0.s5; 
	MBs[mb_num].coeffs[b4x4_in_mb + 1][inv_zigzag[5]]=DCTLine1.s5; 
	MBs[mb_num].coeffs[b4x4_in_mb + 1][inv_zigzag[6]]=DCTLine2.s5; 
	MBs[mb_num].coeffs[b4x4_in_mb + 1][inv_zigzag[7]]=DCTLine3.s5;
	MBs[mb_num].coeffs[b4x4_in_mb + 1][inv_zigzag[8]]=DCTLine0.s6; 
	MBs[mb_num].coeffs[b4x4_in_mb + 1][inv_zigzag[9]]=DCTLine1.s6; 
	MBs[mb_num].coeffs[b4x4_in_mb + 1][inv_zigzag[10]]=DCTLine2.s6; 
	MBs[mb_num].coeffs[b4x4_in_mb + 1][inv_zigzag[11]]=DCTLine3.s6;
	MBs[mb_num].coeffs[b4x4_in_mb + 1][inv_zigzag[12]]=DCTLine0.s7; 
	MBs[mb_num].coeffs[b4x4_in_mb + 1][inv_zigzag[13]]=DCTLine1.s7; 
	MBs[mb_num].coeffs[b4x4_in_mb + 1][inv_zigzag[14]]=DCTLine2.s7; 
	MBs[mb_num].coeffs[b4x4_in_mb + 1][inv_zigzag[15]]=DCTLine3.s7;
	// block 10
	MBs[mb_num].coeffs[b4x4_in_mb + 4][inv_zigzag[0]]=DCTLine4.s0;
	MBs[mb_num].coeffs[b4x4_in_mb + 4][inv_zigzag[1]]=DCTLine5.s0; 
	MBs[mb_num].coeffs[b4x4_in_mb + 4][inv_zigzag[2]]=DCTLine6.s0; 
	MBs[mb_num].coeffs[b4x4_in_mb + 4][inv_zigzag[3]]=DCTLine7.s0;
	MBs[mb_num].coeffs[b4x4_in_mb + 4][inv_zigzag[4]]=DCTLine4.s1; 
	MBs[mb_num].coeffs[b4x4_in_mb + 4][inv_zigzag[5]]=DCTLine5.s1; 
	MBs[mb_num].coeffs[b4x4_in_mb + 4][inv_zigzag[6]]=DCTLine6.s1; 
	MBs[mb_num].coeffs[b4x4_in_mb + 4][inv_zigzag[7]]=DCTLine7.s1;
	MBs[mb_num].coeffs[b4x4_in_mb + 4][inv_zigzag[8]]=DCTLine4.s2; 
	MBs[mb_num].coeffs[b4x4_in_mb + 4][inv_zigzag[9]]=DCTLine5.s2; 
	MBs[mb_num].coeffs[b4x4_in_mb + 4][inv_zigzag[10]]=DCTLine6.s2; 
	MBs[mb_num].coeffs[b4x4_in_mb + 4][inv_zigzag[11]]=DCTLine7.s2;
	MBs[mb_num].coeffs[b4x4_in_mb + 4][inv_zigzag[12]]=DCTLine4.s3; 
	MBs[mb_num].coeffs[b4x4_in_mb + 4][inv_zigzag[13]]=DCTLine5.s3; 
	MBs[mb_num].coeffs[b4x4_in_mb + 4][inv_zigzag[14]]=DCTLine6.s3; 
	MBs[mb_num].coeffs[b4x4_in_mb + 4][inv_zigzag[15]]=DCTLine7.s3;
	// block 11
	MBs[mb_num].coeffs[b4x4_in_mb + 5][inv_zigzag[0]]=DCTLine4.s4;
	MBs[mb_num].coeffs[b4x4_in_mb + 5][inv_zigzag[1]]=DCTLine5.s4; 
	MBs[mb_num].coeffs[b4x4_in_mb + 5][inv_zigzag[2]]=DCTLine6.s4; 
	MBs[mb_num].coeffs[b4x4_in_mb + 5][inv_zigzag[3]]=DCTLine7.s4;
	MBs[mb_num].coeffs[b4x4_in_mb + 5][inv_zigzag[4]]=DCTLine4.s5; 
	MBs[mb_num].coeffs[b4x4_in_mb + 5][inv_zigzag[5]]=DCTLine5.s5; 
	MBs[mb_num].coeffs[b4x4_in_mb + 5][inv_zigzag[6]]=DCTLine6.s5; 
	MBs[mb_num].coeffs[b4x4_in_mb + 5][inv_zigzag[7]]=DCTLine7.s5;
	MBs[mb_num].coeffs[b4x4_in_mb + 5][inv_zigzag[8]]=DCTLine4.s6; 
	MBs[mb_num].coeffs[b4x4_in_mb + 5][inv_zigzag[9]]=DCTLine5.s6; 
	MBs[mb_num].coeffs[b4x4_in_mb + 5][inv_zigzag[10]]=DCTLine6.s6; 
	MBs[mb_num].coeffs[b4x4_in_mb + 5][inv_zigzag[11]]=DCTLine7.s6;
	MBs[mb_num].coeffs[b4x4_in_mb + 5][inv_zigzag[12]]=DCTLine4.s7; 
	MBs[mb_num].coeffs[b4x4_in_mb + 5][inv_zigzag[13]]=DCTLine5.s7; 
	MBs[mb_num].coeffs[b4x4_in_mb + 5][inv_zigzag[14]]=DCTLine6.s7; 
	MBs[mb_num].coeffs[b4x4_in_mb + 5][inv_zigzag[15]]=DCTLine7.s7;
	
	//printf((__constant char*)"write ci = %d : pi = %d\n", ci, pi);
	// block 00
	dequant_and_iDCT(&DL0, &DL1, &DL2, &DL3, convert_int4(DCTLine0.s0123), 
											 convert_int4(DCTLine1.s0123),
											 convert_int4(DCTLine2.s0123), 
											 convert_int4(DCTLine3.s0123), dc_q, ac_q);
	PL.x = prev_frame[pi]; PL.y = prev_frame[pi+4]; PL.z = prev_frame[pi+8]; PL.w = prev_frame[pi+12]; pi += width_x4<<2;
	DL0 += convert_int4(PL);
	PL.x = prev_frame[pi]; PL.y = prev_frame[pi+4]; PL.z = prev_frame[pi+8]; PL.w = prev_frame[pi+12]; pi += width_x4<<2;	
	DL1 += convert_int4(PL);
	PL.x = prev_frame[pi]; PL.y = prev_frame[pi+4]; PL.z = prev_frame[pi+8]; PL.w = prev_frame[pi+12]; pi += width_x4<<2;
	DL2 += convert_int4(PL);
	PL.x = prev_frame[pi]; PL.y = prev_frame[pi+4]; PL.z = prev_frame[pi+8]; PL.w = prev_frame[pi+12]; pi += width_x4<<2;
	DL3 += convert_int4(PL);
	*(__global uchar4*)(&recon_frame[ci]) = (convert_uchar4_sat(DL0)); ci+=width;
	*(__global uchar4*)(&recon_frame[ci]) = (convert_uchar4_sat(DL1)); ci+=width;
	*(__global uchar4*)(&recon_frame[ci]) = (convert_uchar4_sat(DL2)); ci+=width;
	*(__global uchar4*)(&recon_frame[ci]) = (convert_uchar4_sat(DL3)); ci+=width;
	ci-=(width<<2); pi -= (width_x4<<4);
	// block 01
	ci+=4; pi += 16;
	dequant_and_iDCT(&DL0, &DL1, &DL2, &DL3, convert_int4(DCTLine0.s4567), 
											 convert_int4(DCTLine1.s4567),
											 convert_int4(DCTLine2.s4567), 
											 convert_int4(DCTLine3.s4567), dc_q, ac_q);
	PL.x = prev_frame[pi]; PL.y = prev_frame[pi+4]; PL.z = prev_frame[pi+8]; PL.w = prev_frame[pi+12]; pi += width_x4<<2;
	DL0 += convert_int4(PL);
	PL.x = prev_frame[pi]; PL.y = prev_frame[pi+4]; PL.z = prev_frame[pi+8]; PL.w = prev_frame[pi+12]; pi += width_x4<<2;
	DL1 += convert_int4(PL);
	PL.x = prev_frame[pi]; PL.y = prev_frame[pi+4]; PL.z = prev_frame[pi+8]; PL.w = prev_frame[pi+12]; pi += width_x4<<2;
	DL2 += convert_int4(PL);
	PL.x = prev_frame[pi]; PL.y = prev_frame[pi+4]; PL.z = prev_frame[pi+8]; PL.w = prev_frame[pi+12]; pi += width_x4<<2;
	DL3 += convert_int4(PL);
	vstore4(convert_uchar4_sat(DL0), 0 , recon_frame + ci); ci+=width;
	*(__global uchar4*)(&recon_frame[ci]) = (convert_uchar4_sat(DL1)); ci+=width;
	*(__global uchar4*)(&recon_frame[ci]) = (convert_uchar4_sat(DL2)); ci+=width;
	*(__global uchar4*)(&recon_frame[ci]) = (convert_uchar4_sat(DL3)); ci+=width;
	ci-=4; pi -= 16;
	// block 10
	dequant_and_iDCT(&DL0, &DL1, &DL2, &DL3, convert_int4(DCTLine4.s0123), 
											 convert_int4(DCTLine5.s0123),
											 convert_int4(DCTLine6.s0123), 
											 convert_int4(DCTLine7.s0123), dc_q, ac_q);
	PL.x = prev_frame[pi]; PL.y = prev_frame[pi+4]; PL.z = prev_frame[pi+8]; PL.w = prev_frame[pi+12]; pi += width_x4<<2;
	DL0 += convert_int4(PL);
	PL.x = prev_frame[pi]; PL.y = prev_frame[pi+4]; PL.z = prev_frame[pi+8]; PL.w = prev_frame[pi+12]; pi += width_x4<<2;
	DL1 += convert_int4(PL);
	PL.x = prev_frame[pi]; PL.y = prev_frame[pi+4]; PL.z = prev_frame[pi+8]; PL.w = prev_frame[pi+12]; pi += width_x4<<2;
	DL2 += convert_int4(PL);
	PL.x = prev_frame[pi]; PL.y = prev_frame[pi+4]; PL.z = prev_frame[pi+8]; PL.w = prev_frame[pi+12]; pi += width_x4<<2;
	DL3 += convert_int4(PL);
	*(__global uchar4*)(&recon_frame[ci]) = (convert_uchar4_sat(DL0)); ci+=width;
	*(__global uchar4*)(&recon_frame[ci]) = (convert_uchar4_sat(DL1)); ci+=width;
	*(__global uchar4*)(&recon_frame[ci]) = (convert_uchar4_sat(DL2)); ci+=width;
	*(__global uchar4*)(&recon_frame[ci]) = (convert_uchar4_sat(DL3)); ci+=width;
	ci-=(width<<2); pi -= (width_x4<<4);
	// block 01
	ci+=4; pi += 16;
	dequant_and_iDCT(&DL0, &DL1, &DL2, &DL3, convert_int4(DCTLine4.s4567), 
											 convert_int4(DCTLine5.s4567),
											 convert_int4(DCTLine6.s4567), 
											 convert_int4(DCTLine7.s4567), dc_q, ac_q);
	PL.x = prev_frame[pi]; PL.y = prev_frame[pi+4]; PL.z = prev_frame[pi+8]; PL.w = prev_frame[pi+12]; pi += width_x4<<2;
	DL0 += convert_int4(PL);
	PL.x = prev_frame[pi]; PL.y = prev_frame[pi+4]; PL.z = prev_frame[pi+8]; PL.w = prev_frame[pi+12]; pi += width_x4<<2;
	DL1 += convert_int4(PL);
	PL.x = prev_frame[pi]; PL.y = prev_frame[pi+4]; PL.z = prev_frame[pi+8]; PL.w = prev_frame[pi+12]; pi += width_x4<<2;
	DL2 += convert_int4(PL);
	PL.x = prev_frame[pi]; PL.y = prev_frame[pi+4]; PL.z = prev_frame[pi+8]; PL.w = prev_frame[pi+12]; pi += width_x4<<2;
	DL3 += convert_int4(PL);
	*(__global uchar4*)(&recon_frame[ci]) = (convert_uchar4_sat(DL0)); ci+=width;
	*(__global uchar4*)(&recon_frame[ci]) = (convert_uchar4_sat(DL1)); ci+=width;
	*(__global uchar4*)(&recon_frame[ci]) = (convert_uchar4_sat(DL2)); ci+=width;
	*(__global uchar4*)(&recon_frame[ci]) = (convert_uchar4_sat(DL3)); ci+=width;
	//printf((__constant char*)"end\n");
	return;
}

__kernel void chroma_transform( 	__global uchar *current_frame, //0 input frame (the one being encoded)
									__global uchar *prev_frame, //1 reference frame for predictors
									__global uchar *recon_frame, //2 reconstructed current frame after DCT-quant/dequant-iDCT
									__global macroblock *MBs, //3 here it's an input
									signed int chroma_width, //4 
									signed int chroma_height, //5 
									signed int first_block_offset, //6
									int dc_q, //7
									int ac_q, //8
									int block_place) //9
{  
	__private int chroma_block_num;                                                       
	chroma_block_num = first_block_offset + get_global_id(0); 
	
	__private int cx, cy;
	cx = (chroma_block_num % (chroma_width/4))*4;
	cy = (chroma_block_num / (chroma_width/4))*4; 

	__private int ci00, ci10, ci20, ci30;
	ci00 = cy*chroma_width + cx; 
	ci10 = ci00+chroma_width; 
	ci20 = ci10+chroma_width; 
	ci30 = ci20+chroma_width; 

	__private int4 CurrentLine0, CurrentLine1, CurrentLine2, CurrentLine3;
    CurrentLine0 = convert_int4(vload4(0, current_frame + ci00));
    CurrentLine1 = convert_int4(vload4(0, current_frame + ci10));
    CurrentLine2 = convert_int4(vload4(0, current_frame + ci20));
    CurrentLine3 = convert_int4(vload4(0, current_frame + ci30));

	__private short2 vector;
		
	__private int mb_num, block_in_mb;
	mb_num = (cy/8)*(chroma_width/8) + (cx/8);
	__private int Vnum, Hnum;
	Vnum = (cy/4)%2;			
	Hnum = (cx/4)%2;
	block_in_mb = Vnum*2 + Hnum;
	
	vector.x = MBs[mb_num].vector_x[block_in_mb];
	vector.y = MBs[mb_num].vector_y[block_in_mb];
	
	block_in_mb += block_place; // 16 for U, 20 for V

	__private int ddx, ddy, nddx, nddy;
	ddy = (int)(abs(vector.y)&7); // values indicating fractional part
	ddx = (int)(abs(vector.x)&7); // in 8th-pel
	
	ddy = (vector.y < 0) ? (8 - ddy) : ddy;
	ddx = (vector.x < 0) ? (8 - ddx) : ddx;
	nddx = 8 - ddx;
	nddy = 8 - ddy;

	vector.x = (((vector.x < 0)&&((ddx&7)>0)) ? (vector.x - 8): vector.x);
	vector.y = (((vector.y < 0)&&((ddy&7)>0)) ? (vector.y - 8): vector.y);
	vector /= 8;
	
	__private int4 PredictorLine0, PredictorLine1, 
					PredictorLine2, PredictorLine3, ExtraLine; 

	__private int extra_collumn[5];
	
	__private int px0, px1, px2, px3, px4, py0, py1, py2, py3, py4;
	
	
	px0 = cx + (int)vector.x;
	py0 = cy + (int)vector.y;
	px0 = (px0 < 0) ? 0 : px0;
	px1 = (px0 < -1) ? 0 : px0 + 1;
	px2 = (px0 < -2) ? 0 : px0 + 2;
	px3 = (px0 < -3) ? 0 : px0 + 3;
	px4 = (px0 < -4) ? 0 : px0 + 4;
	px0 = (px0 >= chroma_width) ? (chroma_width - 1) : px0;
	px1 = (px1 >= chroma_width) ? (chroma_width - 1) : px1;
	px2 = (px2 >= chroma_width) ? (chroma_width - 1) : px2;
	px3 = (px3 >= chroma_width) ? (chroma_width - 1) : px3;
	px4 = (px4 >= chroma_width) ? (chroma_width - 1) : px4;	
	
	py0 = (py0 < 0) ? 0 : py0;
	py1 = (py0 < -1) ? 0 : py0 + 1;
	py2 = (py0 < -2) ? 0 : py0 + 2;
	py3 = (py0 < -3) ? 0 : py0 + 3;
	py4 = (py0 < -4) ? 0 : py0 + 4;
	py0 = (py0 >= chroma_height) ? (chroma_height - 1) : py0;
	py1 = (py1 >= chroma_height) ? (chroma_height - 1) : py1;
	py2 = (py2 >= chroma_height) ? (chroma_height - 1) : py2;
	py3 = (py3 >= chroma_height) ? (chroma_height - 1) : py3;
	py4 = (py4 >= chroma_height) ? (chroma_height - 1) : py4;
	
	PredictorLine0.x = prev_frame[py0*chroma_width + px0]; 
	PredictorLine0.y = prev_frame[py0*chroma_width + px1];
	PredictorLine0.z = prev_frame[py0*chroma_width + px2];
	PredictorLine0.w = prev_frame[py0*chroma_width + px3];
	PredictorLine1.x = prev_frame[py1*chroma_width + px0];
	PredictorLine1.y = prev_frame[py1*chroma_width + px1];
	PredictorLine1.z = prev_frame[py1*chroma_width + px2];
	PredictorLine1.w = prev_frame[py1*chroma_width + px3];
	PredictorLine2.x = prev_frame[py2*chroma_width + px0]; 
	PredictorLine2.y = prev_frame[py2*chroma_width + px1];
	PredictorLine2.z = prev_frame[py2*chroma_width + px2];
	PredictorLine2.w = prev_frame[py2*chroma_width + px3];
	PredictorLine3.x = prev_frame[py3*chroma_width + px0]; 
	PredictorLine3.y = prev_frame[py3*chroma_width + px1];
	PredictorLine3.z = prev_frame[py3*chroma_width + px2];
	PredictorLine3.w = prev_frame[py3*chroma_width + px3];
	
	extra_collumn[0] = 0; 
	extra_collumn[1] = 0;
	extra_collumn[2] = 0;
	extra_collumn[3] = 0;
	extra_collumn[4] = 0;
	
	if ((ddx&7) != 0)
	{
		extra_collumn[0] = prev_frame[py0*chroma_width + px4];
		extra_collumn[1] = prev_frame[py1*chroma_width + px4];
		extra_collumn[2] = prev_frame[py2*chroma_width + px4];
		extra_collumn[3] = prev_frame[py3*chroma_width + px4];
		extra_collumn[4] = prev_frame[py4*chroma_width + px4];
	}
	if ((ddy&7) != 0)
	{
		ExtraLine.x = prev_frame[py4*chroma_width + px0]; 
		ExtraLine.y = prev_frame[py4*chroma_width + px1];
		ExtraLine.z = prev_frame[py4*chroma_width + px2];
		ExtraLine.w = prev_frame[py4*chroma_width + px3];

		PredictorLine0 = PredictorLine0*nddy + PredictorLine1*ddy;
			PredictorLine0 += 4;
				PredictorLine0 >>= 3;
		PredictorLine1 = PredictorLine1*nddy + PredictorLine2*ddy;
			PredictorLine1 += 4;
				PredictorLine1 >>= 3;
		PredictorLine2 = PredictorLine2*nddy + PredictorLine3*ddy;
			PredictorLine2 += 4;
				PredictorLine2 >>= 3;
		PredictorLine3 = PredictorLine3*nddy + ExtraLine*ddy;
			PredictorLine3 += 4;
				PredictorLine3 >>= 3;
		extra_collumn[0] = extra_collumn[0]*nddy + extra_collumn[1]*ddy + 4; 
			extra_collumn[0] >>= 3;
		extra_collumn[1] = extra_collumn[1]*nddy + extra_collumn[2]*ddy + 4;
			extra_collumn[1] >>= 3;
		extra_collumn[2] = extra_collumn[2]*nddy + extra_collumn[3]*ddy + 4;
			extra_collumn[2] >>= 3;
		extra_collumn[3] = extra_collumn[3]*nddy + extra_collumn[4]*ddy + 4;
			extra_collumn[3] >>= 3;
	}
	if ((ddx&7) != 0)
	{
		PredictorLine0.x = PredictorLine0.x*nddx + PredictorLine0.y*ddx + 4;
			PredictorLine1.x = PredictorLine1.x*nddx + PredictorLine1.y*ddx + 4;
				PredictorLine2.x = PredictorLine2.x*nddx + PredictorLine2.y*ddx + 4;
					PredictorLine3.x = PredictorLine3.x*nddx + PredictorLine3.y*ddx + 4;
		PredictorLine0.y = PredictorLine0.y*nddx + PredictorLine0.z*ddx + 4;
			PredictorLine1.y = PredictorLine1.y*nddx + PredictorLine1.z*ddx + 4;
				PredictorLine2.y = PredictorLine2.y*nddx + PredictorLine2.z*ddx + 4;
					PredictorLine3.y = PredictorLine3.y*nddx + PredictorLine3.z*ddx + 4;
		PredictorLine0.z = PredictorLine0.z*nddx + PredictorLine0.w*ddx + 4;
			PredictorLine1.z = PredictorLine1.z*nddx + PredictorLine1.w*ddx + 4;
				PredictorLine2.z = PredictorLine2.z*nddx + PredictorLine2.w*ddx + 4;
					PredictorLine3.z = PredictorLine3.z*nddx + PredictorLine3.w*ddx + 4;
		PredictorLine0.w = PredictorLine0.w*nddx + extra_collumn[0]*ddx + 4;
			PredictorLine1.w = PredictorLine1.w*nddx + extra_collumn[1]*ddx + 4;
				PredictorLine2.w = PredictorLine2.w*nddx + extra_collumn[2]*ddx + 4;
					PredictorLine3.w = PredictorLine3.w*nddx + extra_collumn[3]*ddx + 4;
					
		PredictorLine0 >>= 3;
		PredictorLine1 >>= 3;
		PredictorLine2 >>= 3;
		PredictorLine3 >>= 3;
	}

	__private int4 BestDiffLine0, BestDiffLine1, BestDiffLine2, BestDiffLine3;
	__private int4 DiffLine0=0, DiffLine1=1, DiffLine2=2, DiffLine3=3;

	BestDiffLine0 = CurrentLine0 - PredictorLine0;
	BestDiffLine1 = CurrentLine1 - PredictorLine1;
	BestDiffLine2 = CurrentLine2 - PredictorLine2;
	BestDiffLine3 = CurrentLine3 - PredictorLine3;

	DiffLine0 = (BestDiffLine0 + BestDiffLine3) << 3;	// a1 = ((ip[0] + ip[3])<<3);
	DiffLine1 = (BestDiffLine1 + BestDiffLine2) << 3;	// b1 = ((ip[1] + ip[2])<<3);
	DiffLine2 = (BestDiffLine1 - BestDiffLine2) << 3;	// c1 = ((ip[1] - ip[2])<<3);
	DiffLine3 = (BestDiffLine0 - BestDiffLine3) << 3;	// d1 = ((ip[0] - ip[3])<<3);
	
	BestDiffLine0 = DiffLine0 + DiffLine1;				// op[0] = (a1 + b1); 
	BestDiffLine2 = DiffLine0 - DiffLine1;				// op[2] = (a1 - b1);
	
	BestDiffLine1 = ((DiffLine2 * 2217) + (DiffLine3 * 5352) + 14500) >> 12;
														// op[1] = (c1 * 2217 + d1 * 5352 +  14500)>>12;
	BestDiffLine3 = ((DiffLine3 * 2217) - (DiffLine2 * 5352) + 7500) >> 12;
														// op[3] = (d1 * 2217 - c1 * 5352 +   7500)>>12;

	DiffLine0 = (int4)(BestDiffLine0.x, BestDiffLine1.x, BestDiffLine2.x, BestDiffLine3.x);
	DiffLine1 = (int4)(BestDiffLine0.y, BestDiffLine1.y, BestDiffLine2.y, BestDiffLine3.y);
	DiffLine2 = (int4)(BestDiffLine0.z, BestDiffLine1.z, BestDiffLine2.z, BestDiffLine3.z);
	DiffLine3 = (int4)(BestDiffLine0.w, BestDiffLine1.w, BestDiffLine2.w, BestDiffLine3.w);

	BestDiffLine0 = DiffLine0 + DiffLine3;				// a1 = op[0] + op[3];	
	BestDiffLine1 = DiffLine1 + DiffLine2;				// b1 = op[1] + op[2];
	BestDiffLine2 = DiffLine1 - DiffLine2;				// c1 = op[1] - op[2];
	BestDiffLine3 = DiffLine0 - DiffLine3;				// d1 = op[0] - op[3];

	
	DiffLine0 = ((BestDiffLine0 + BestDiffLine1 + 7) >> 4); 
														// op[0] = (( a1 + b1 + 7)>>4) / q;
	DiffLine2 = ((BestDiffLine0 - BestDiffLine1 + 7) >> 4);
														// op[2] = (( a1 - b1 + 7)>>4) / q;
	
	DiffLine1 = ((BestDiffLine2 * 2217) + (BestDiffLine3 * 5352) + 12000) >> 16;
	DiffLine1.x += (BestDiffLine3.x != 0);
	DiffLine1.y += (BestDiffLine3.y != 0);
	DiffLine1.z += (BestDiffLine3.z != 0);
	DiffLine1.w += (BestDiffLine3.w != 0); 	// op[1]  = (((c1 * 2217 + d1 * 5352 +  12000)>>16) + (d1!=0)) / q;
	
	DiffLine3 = (((BestDiffLine3 * 2217) - (BestDiffLine2 * 5352) + 51000) >> 16);
											// op[3] = ((d1 * 2217 - c1 * 5352 +  51000)>>16) / q;
														
	DiffLine0 /= (int4)(dc_q, ac_q, ac_q, ac_q);
	DiffLine1 /= ac_q;
	DiffLine2 /= ac_q;
	DiffLine3 /= ac_q;
	
	const int inv_zigzag[16] = { 0, 1, 5, 6, 2, 4, 7, 12, 3,  8, 11, 13, 9, 10, 14, 15 };
	MBs[mb_num].coeffs[block_in_mb][inv_zigzag[0]]=(short)DiffLine0.x;
	MBs[mb_num].coeffs[block_in_mb][inv_zigzag[1]]=(short)DiffLine1.x;
	MBs[mb_num].coeffs[block_in_mb][inv_zigzag[2]]=(short)DiffLine2.x;
	MBs[mb_num].coeffs[block_in_mb][inv_zigzag[3]]=(short)DiffLine3.x;
	MBs[mb_num].coeffs[block_in_mb][inv_zigzag[4]]=(short)DiffLine0.y;
	MBs[mb_num].coeffs[block_in_mb][inv_zigzag[5]]=(short)DiffLine1.y;
	MBs[mb_num].coeffs[block_in_mb][inv_zigzag[6]]=(short)DiffLine2.y;
	MBs[mb_num].coeffs[block_in_mb][inv_zigzag[7]]=(short)DiffLine3.y;
	MBs[mb_num].coeffs[block_in_mb][inv_zigzag[8]]=(short)DiffLine0.z;
	MBs[mb_num].coeffs[block_in_mb][inv_zigzag[9]]=(short)DiffLine1.z;
	MBs[mb_num].coeffs[block_in_mb][inv_zigzag[10]]=(short)DiffLine2.z;
	MBs[mb_num].coeffs[block_in_mb][inv_zigzag[11]]=(short)DiffLine3.z;
	MBs[mb_num].coeffs[block_in_mb][inv_zigzag[12]]=(short)DiffLine0.w;
	MBs[mb_num].coeffs[block_in_mb][inv_zigzag[13]]=(short)DiffLine1.w;
	MBs[mb_num].coeffs[block_in_mb][inv_zigzag[14]]=(short)DiffLine2.w;
	MBs[mb_num].coeffs[block_in_mb][inv_zigzag[15]]=(short)DiffLine3.w;
	
	DiffLine0 *= (int4)(dc_q, ac_q, ac_q, ac_q);
	DiffLine1 *= ac_q;
	DiffLine2 *= ac_q;
	DiffLine3 *= ac_q;
		
	BestDiffLine0 = DiffLine0 + DiffLine2;				// a1 = ip[0]+ip[2];
	BestDiffLine1 = DiffLine0 - DiffLine2;				// b1 = ip[0]-ip[2];
								
	BestDiffLine2 = ((DiffLine1 * 35468) >> 16) - (DiffLine3 + ((DiffLine3 * 20091)>>16));
														// temp1 = (ip[1] * sinpi8sqrt2)>>16;
														// temp2 = ip[3] + ((ip[3] * cospi8sqrt2minus1)>>16);
														// c1 = temp1 - temp2;
	BestDiffLine3 = (DiffLine1 + ((DiffLine1 * 20091)>>16)) + ((DiffLine3 * 35468) >> 16);
														// temp1 = ip[1] + ((ip[1] * cospi8sqrt2minus1)>>16);
														// temp2 = (ip[3] * sinpi8sqrt2)>>16;
														// d1 = temp1 + temp2;

	
	DiffLine0 = BestDiffLine0 + BestDiffLine3;			// op[0] = a1 + d1;
	DiffLine3 = BestDiffLine0 - BestDiffLine3;			// op[3] = a1 - d1;
	DiffLine1 = BestDiffLine1 + BestDiffLine2;			// op[1] = b1 + c1;
	DiffLine2 = BestDiffLine1 - BestDiffLine2;			// op[2] = b1 - c1;

	BestDiffLine0 = (int4)(DiffLine0.x, DiffLine1.x, DiffLine2.x, DiffLine3.x);
	BestDiffLine1 = (int4)(DiffLine0.y, DiffLine1.y, DiffLine2.y, DiffLine3.y);
	BestDiffLine2 = (int4)(DiffLine0.z, DiffLine1.z, DiffLine2.z, DiffLine3.z);
	BestDiffLine3 = (int4)(DiffLine0.w, DiffLine1.w, DiffLine2.w, DiffLine3.w);

	DiffLine0 = BestDiffLine0 + BestDiffLine2;			// a1 = op[0]+op[2];
	DiffLine1 = BestDiffLine0 - BestDiffLine2;			// b1 = tp[0]-tp[2];
	
	DiffLine2 = ((BestDiffLine1 * 35468) >> 16) - (BestDiffLine3 + ((BestDiffLine3 * 20091)>>16));
														// temp1 = (ip[1] * sinpi8sqrt2)>>16;
														// temp2 = ip[3] + ((ip[3] * cospi8sqrt2minus1)>>16);
														// c1 = temp1 - temp2;
	DiffLine3 = (BestDiffLine1 + ((BestDiffLine1 * 20091)>>16)) + ((BestDiffLine3 * 35468) >> 16);
														// temp1 = ip[1] + ((ip[1] * cospi8sqrt2minus1)>>16);
														// temp2 = (ip[3] * sinpi8sqrt2)>>16;
														// d1 = temp1 + temp2;

	BestDiffLine0 = 1*(((DiffLine0 + DiffLine3 + 4) >> 3) + PredictorLine0);
														// op[0] = ((a1 + d1 + 4) >> 3) + pred[0,i]
	BestDiffLine1 = 1*(((DiffLine1 + DiffLine2 + 4) >> 3) + PredictorLine1);
														// op[1] = ((b1 + c1 + 4) >> 3) + pred[1,i]
	BestDiffLine2 = 1*(((DiffLine1 - DiffLine2 + 4) >> 3) + PredictorLine2);
														// op[2] = ((b1 - c1 + 4) >> 3) + pred[2,i]
	BestDiffLine3 = 1*(((DiffLine0 - DiffLine3 + 4) >> 3) + PredictorLine3);
														// op[3] = ((a1 - d1 + 4) >> 3) + pred[3,i]
	
	*(__global uchar4*)(&recon_frame[ci00]) = (convert_uchar4_sat(BestDiffLine0));
	*(__global uchar4*)(&recon_frame[ci10]) = (convert_uchar4_sat(BestDiffLine1));
	*(__global uchar4*)(&recon_frame[ci20]) = (convert_uchar4_sat(BestDiffLine2));
	*(__global uchar4*)(&recon_frame[ci30]) = (convert_uchar4_sat(BestDiffLine3));

	return;
}

__kernel __attribute__((reqd_work_group_size(64, 1, 1)))
		void filter_MB_col_H( 	__global uchar * const frame,
								const int width,
								const int mbedge_limit,
								const int sub_bedge_limit,
								const int mb_col)
{
	int x, y, i, nf;
	uchar4 L, R;
	int a, b;
	
	y = get_global_id(0);
	x = mb_col * 16;
	
	i = y*width + x;
	
	R = vload4(0, frame + i);
	if ( x > 0) 
	{
		L = vload4(0, frame + i - 4);
	
		//macroblock edge filtering
		// flag for Need Filtering
		nf = ((abs(L.w - R.x) * 2 + abs(L.z - R.y)/2) <= mbedge_limit) ? 1 : 0;
		// filtering
		// a = clamp128(clamp128(p1 - q1) + 3*(q0 - p0))
		a = ((L.z - 128) - (R.y - 128)); 
		a = (a < -128) ? -128 : a;
		a = (a > 127) ? 127 : a;
		a = mad24(((R.x - 128) - (L.w - 128)), 3, a);
		a = (a < -128) ? -128 : a;
		a = (a > 127) ? 127 : a;
		// zeroing if no filtering needed
		a *= nf;
		// b = clamp128(a+3) >> 3
		b = a + 3;
		b = (b < -128) ? -128 : b;
		b = (b > 127) ? 127 : b;
		b >>= 3;
		// a = clamp128(a+4) >> 3
		a = a + 4;
		a = (a < -128) ? -128 : a;
		a = (a > 127) ? 127 : a;
		a >>= 3;
		// Q0 = s2u(q0 - a)
		R.x = ((R.x - 128) - a) + 128;
		// P0 = s2u(p0 + b)
		L.w = ((L.w - 128) + b) + 128;
		vstore4(L, 0, frame + i - 4);
		vstore4(R, 0, frame + i);		
	}
	
	// and 3 more times for edges between blocks in MB
	
	L = R;
	i += 4;
	R = vload4(0, frame + i);
	nf = ((abs(L.w - R.x) * 2 + abs(L.z - R.y)/2) <= sub_bedge_limit) ? 1 : 0;
	// a = clamp128(clamp128(p1 - q1) + 3*(q0 - p0))
	a = ((L.z - 128) - (R.y - 128)); 
	a = (a < -128) ? -128 : a;
	a = (a > 127) ? 127 : a;
	a = mad24(((R.x - 128) - (L.w - 128)), 3, a);
	a = (a < -128) ? -128 : a;
	a = (a > 127) ? 127 : a;
	a *= nf;
	// b = clamp128(a+3) >> 3
	b = a + 3;
	b = (b < -128) ? -128 : b;
	b = (b > 127) ? 127 : b;
	b >>= 3;
	// a = clamp128(a+4) >> 3
	a = a + 4;
	a = (a < -128) ? -128 : a;
	a = (a > 127) ? 127 : a;
	a >>= 3;
	// Q0 = s2u(q0 - a)
	R.x = ((R.x - 128) - a) + 128;
	// P0 = s2u(p0 + b)
	L.w = ((L.w - 128) + b) + 128;
	vstore4(L, 0, frame + i - 4);
	vstore4(R, 0, frame + i);
	
	L = R;
	i += 4;
	R = vload4(0, frame + i);
	nf = ((abs(L.w - R.x) * 2 + abs(L.z - R.y)/2) <= sub_bedge_limit) ? 1 : 0;
	// a = clamp128(clamp128(p1 - q1) + 3*(q0 - p0))
	a = ((L.z - 128) - (R.y - 128)); 
	a = (a < -128) ? -128 : a;
	a = (a > 127) ? 127 : a;
	a = mad24(((R.x - 128) - (L.w - 128)), 3, a);
	a = (a < -128) ? -128 : a;
	a = (a > 127) ? 127 : a;
	a *= nf;
	// b = clamp128(a+3) >> 3
	b = a + 3;
	b = (b < -128) ? -128 : b;
	b = (b > 127) ? 127 : b;
	b >>= 3;
	// a = clamp128(a+4) >> 3
	a = a + 4;
	a = (a < -128) ? -128 : a;
	a = (a > 127) ? 127 : a;
	a >>= 3;
	// Q0 = s2u(q0 - a)
	R.x = ((R.x - 128) - a) + 128;
	// P0 = s2u(p0 + b)
	L.w = ((L.w - 128) + b) + 128;
	vstore4(L, 0, frame + i - 4);
	vstore4(R, 0, frame + i);
	
	L = R;
	i += 4;
	R = vload4(0, frame + i);
	nf = ((abs(L.w - R.x) * 2 + abs(L.z - R.y)/2) <= sub_bedge_limit) ? 1 : 0;
	// a = clamp128(clamp128(p1 - q1) + 3*(q0 - p0))
	a = ((L.z - 128) - (R.y - 128)); 
	a = (a < -128) ? -128 : a;
	a = (a > 127) ? 127 : a;
	a = mad24(((R.x - 128) - (L.w - 128)), 3, a);
	a = (a < -128) ? -128 : a;
	a = (a > 127) ? 127 : a;
	a *= nf;
	// b = clamp128(a+3) >> 3
	b = a + 3;
	b = (b < -128) ? -128 : b;
	b = (b > 127) ? 127 : b;
	b >>= 3;
	// a = clamp128(a+4) >> 3
	a = a + 4;
	a = (a < -128) ? -128 : a;
	a = (a > 127) ? 127 : a;
	a >>= 3;
	// Q0 = s2u(q0 - a)
	R.x = (uchar)(((R.x - 128) - a) + 128);
	// P0 = s2u(p0 + b)
	L.w = (uchar)(((L.w - 128) + b) + 128);
	vstore4(L, 0, frame + i - 4);
	vstore4(R, 0, frame + i);
	
	return;	
}

__kernel __attribute__((reqd_work_group_size(64, 1, 1)))
		void filter_MB_col_V( 	__global uchar * const frame,
								const int width,
								const int mbedge_limit,
								const int sub_bedge_limit,
								const int mb_col)
{
	int x, y, i;
	uchar4 U2, U3, D0, D1; 
	int4 a, b, nf, one = 1, zero = 0;
	
	y = (get_global_id(0)/4)*16;
	x = mad24((int)mb_col, (int)16, (int)(get_global_id(0)%4)*4);
	
	i = y*width + x;
	
	D0 = vload4(0, frame + i); i += width;
	D1 = vload4(0, frame + i); i += width;
	i -= width*2;
	
	if ( y > 0) 
	{
		i -= width*2;
		U2 = vload4(0, frame + i); i += width;
		U3 = vload4(0, frame + i); i += width;
		
		//macroblock edge filtering
		// flag for Need Filtering
		// if ((abs(*P0 - *Q0)*2 + abs(*P1 - *Q1)/2) <= edge_limit))
		nf = select(zero,one,((abs(convert_int4(U3)  - convert_int4(D0)) * 2 + abs(convert_int4(U2) - convert_int4(D1))/2) <= mbedge_limit));
		// filtering
		// a = clamp128(clamp128(p1 - q1) + 3*(q0 - p0))
		a = ((convert_int4(U2) - 128) - (convert_int4(D1) - 128)); 
		a = (a < -128) ? -128 : a;
		a = (a > 127) ? 127 : a;
		a = mad24(((convert_int4(D0) - 128) - (convert_int4(U3) - 128)), 3, a);
		a = (a < -128) ? -128 : a;
		a = (a > 127) ? 127 : a;
		// zeroing if no filtering needed
		a *= nf;
		// b = clamp128(a+3) >> 3
		b = a + 3;
		b = (b < -128) ? -128 : b;
		b = (b > 127) ? 127 : b;
		b >>= 3;
		// a = clamp128(a+4) >> 3
		a = a + 4;
		a = (a < -128) ? -128 : a;
		a = (a > 127) ? 127 : a;
		a >>= 3;
		// Q0 = s2u(q0 - a)
		D0 = convert_uchar4(((convert_int4(D0) - 128) - a) + 128);
		// P0 = s2u(p0 + b)
		U3 = convert_uchar4(((convert_int4(U3) - 128) + b) + 128);
		vstore4(U3, 0, frame + i - width);
		vstore4(D0, 0, frame + i);		
	}
	
	i += width*2;
	
	U2 = vload4(0, frame + i); i += width;
	U3 = vload4(0, frame + i); i += width;
	D0 = vload4(0, frame + i); i += width;
	D1 = vload4(0, frame + i); i += width;
	
	i -= width*2;
	
	// if ((abs(*P0 - *Q0)*2 + abs(*P1 - *Q1)/2) <= edge_limit))
	nf = select(zero,one,((abs(convert_int4(U3)  - convert_int4(D0)) * 2 + abs(convert_int4(U2) - convert_int4(D1))/2) <= sub_bedge_limit));
	// filtering
	// a = clamp128(clamp128(p1 - q1) + 3*(q0 - p0))
	a = ((convert_int4(U2) - 128) - (convert_int4(D1) - 128)); 
	a = (a < -128) ? -128 : a;
	a = (a > 127) ? 127 : a;
	a = mad24(((convert_int4(D0) - 128) - (convert_int4(U3) - 128)), 3, a);
	a = (a < -128) ? -128 : a;
	a = (a > 127) ? 127 : a;
	a *= nf;
	// b = clamp128(a+3) >> 3
	b = a + 3;
	b = (b < -128) ? -128 : b;
	b = (b > 127) ? 127 : b;
	b >>= 3;
	// a = clamp128(a+4) >> 3
	a = a + 4;
	a = (a < -128) ? -128 : a;
	a = (a > 127) ? 127 : a;
	a >>= 3;
	// Q0 = s2u(q0 - a)
	D0 = convert_uchar4(((convert_int4(D0) - 128) - a) + 128);
	// P0 = s2u(p0 + b)
	U3 = convert_uchar4(((convert_int4(U3) - 128) + b) + 128);
	vstore4(U3, 0, frame + i - width);
	vstore4(D0, 0, frame + i);	

	i += width*2;
	
	U2 = vload4(0, frame + i); i += width;
	U3 = vload4(0, frame + i); i += width;
	D0 = vload4(0, frame + i); i += width;
	D1 = vload4(0, frame + i); i += width;
	
	i -= width*2;
	
	// if ((abs(*P0 - *Q0)*2 + abs(*P1 - *Q1)/2) <= edge_limit))
	nf = select(zero,one,((abs(convert_int4(U3)  - convert_int4(D0)) * 2 + abs(convert_int4(U2) - convert_int4(D1))/2) <= sub_bedge_limit));
	nf *= -1; //in OpenCL for vectors: 0 - false; -1 - true
	// filtering
	// a = clamp128(clamp128(p1 - q1) + 3*(q0 - p0))
	a = ((convert_int4(U2) - 128) - (convert_int4(D1) - 128)); 
	a = (a < -128) ? -128 : a;
	a = (a > 127) ? 127 : a;
	a = mad24(((convert_int4(D0) - 128) - (convert_int4(U3) - 128)), 3, a);
	a = (a < -128) ? -128 : a;
	a = (a > 127) ? 127 : a;
	a *= nf;
	// b = clamp128(a+3) >> 3
	b = a + 3;
	b = (b < -128) ? -128 : b;
	b = (b > 127) ? 127 : b;
	b >>= 3;
	// a = clamp128(a+4) >> 3
	a = a + 4;
	a = (a < -128) ? -128 : a;
	a = (a > 127) ? 127 : a;
	a >>= 3;
	// Q0 = s2u(q0 - a)
	D0 = convert_uchar4(((convert_int4(D0) - 128) - a) + 128);
	// P0 = s2u(p0 + b)
	U3 = convert_uchar4(((convert_int4(U3) - 128) + b) + 128);
	vstore4(U3, 0, frame + i - width);
	vstore4(D0, 0, frame + i);	
	
	i += width*2;
	
	U2 = vload4(0, frame + i); i += width;
	U3 = vload4(0, frame + i); i += width;
	D0 = vload4(0, frame + i); i += width;
	D1 = vload4(0, frame + i); i += width;
	
	i -= width*2;
	
	// if ((abs(*P0 - *Q0)*2 + abs(*P1 - *Q1)/2) <= edge_limit))
	nf = select(zero,one,((abs(convert_int4(U3)  - convert_int4(D0)) * 2 + abs(convert_int4(U2) - convert_int4(D1))/2) <= sub_bedge_limit));
	nf *= -1; //in OpenCL for vectors: 0 - false; -1 - true
	// filtering
	// a = clamp128(clamp128(p1 - q1) + 3*(q0 - p0))
	a = ((convert_int4(U2) - 128) - (convert_int4(D1) - 128)); 
	a = (a < -128) ? -128 : a;
	a = (a > 127) ? 127 : a;
	a = mad24(((convert_int4(D0) - 128) - (convert_int4(U3) - 128)), 3, a);
	a = (a < -128) ? -128 : a;
	a = (a > 127) ? 127 : a;
	a *= nf;
	// b = clamp128(a+3) >> 3
	b = a + 3;
	b = (b < -128) ? -128 : b;
	b = (b > 127) ? 127 : b;
	b >>= 3;
	// a = clamp128(a+4) >> 3
	a = a + 4;
	a = (a < -128) ? -128 : a;
	a = (a > 127) ? 127 : a;
	a >>= 3;
	// Q0 = s2u(q0 - a)
	D0 = convert_uchar4(((convert_int4(D0) - 128) - a) + 128);
	// P0 = s2u(p0 + b)
	U3 = convert_uchar4(((convert_int4(U3) - 128) + b) + 128);
	vstore4(U3, 0, frame + i - width);
	vstore4(D0, 0, frame + i);		
	
	return;	
}




