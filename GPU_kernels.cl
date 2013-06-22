#pragma OPENCL EXTENSION cl_khr_byte_addressable_store : enable

typedef struct {
    short coeffs[25][16];
    int vector_x[4];
    int vector_y[4];
} macroblock;

void weight(int4 * const __L0, int4 * const __L1, int4 * const __L2, int4 * const __L3, const int * const dc_q, const int * const ac_q) //Hadamard
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
	__private int chroma_width_x8 = chroma_width*8;
	chroma_block_num = first_block_offset + get_global_id(0); 
	
	__private int cx, cy, px, py;
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

	__private int4 PredictorLine0, PredictorLine1, 
					PredictorLine2, PredictorLine3; 

	px = (cx*8) + (int)vector.x;
	py = (cy*8) + (int)vector.y;

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

	BestDiffLine0 = (((DiffLine0 + DiffLine3 + 4) >> 3) + PredictorLine0);
														// op[0] = ((a1 + d1 + 4) >> 3) + pred[0,i]
	BestDiffLine1 = (((DiffLine1 + DiffLine2 + 4) >> 3) + PredictorLine1);
														// op[1] = ((b1 + c1 + 4) >> 3) + pred[1,i]
	BestDiffLine2 = (((DiffLine1 - DiffLine2 + 4) >> 3) + PredictorLine2);
														// op[2] = ((b1 - c1 + 4) >> 3) + pred[2,i]
	BestDiffLine3 = (((DiffLine0 - DiffLine3 + 4) >> 3) + PredictorLine3);
														// op[3] = ((a1 - d1 + 4) >> 3) + pred[3,i]
	
	*(__global uchar4*)(&recon_frame[ci00]) = (convert_uchar4_sat(BestDiffLine0));
	*(__global uchar4*)(&recon_frame[ci10]) = (convert_uchar4_sat(BestDiffLine1));
	*(__global uchar4*)(&recon_frame[ci20]) = (convert_uchar4_sat(BestDiffLine2));
	*(__global uchar4*)(&recon_frame[ci30]) = (convert_uchar4_sat(BestDiffLine3));

	return;
}

__kernel __attribute__((reqd_work_group_size(64, 1, 1)))
		void simple_loop_filter_MBH(__global uchar * const frame,
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
		nf = ((abs((int)L.w - (int)R.x) * 2 + abs((int)L.z - (int)R.y)/2) <= mbedge_limit) ? 1 : 0;
		// filtering
		// a = clamp128(clamp128(p1 - q1) + 3*(q0 - p0))
		a = (((int)L.z - 128) - ((int)R.y - 128)); 
		a = (a < -128) ? -128 : a;
		a = (a > 127) ? 127 : a;
		a = mad24((((int)R.x - 128) - ((int)L.w - 128)), 3, a);
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
		R.x = (uchar)((((int)R.x - 128) - a) + 128);
		// P0 = s2u(p0 + b)
		L.w = (uchar)((((int)L.w - 128) + b) + 128);
		vstore4(L, 0, frame + i - 4);
		vstore4(R, 0, frame + i);		
	}
	
	// and 3 more times for edges between blocks in MB
	
	L = R;
	i += 4;
	R = vload4(0, frame + i);
	nf = ((abs((int)L.w - (int)R.x) * 2 + abs((int)L.z - (int)R.y)/2) <= sub_bedge_limit) ? 1 : 0;
	// a = clamp128(clamp128(p1 - q1) + 3*(q0 - p0))
	a = (((int)L.z - 128) - ((int)R.y - 128)); 
	a = (a < -128) ? -128 : a;
	a = (a > 127) ? 127 : a;
	a = mad24((((int)R.x - 128) - ((int)L.w - 128)), 3, a);
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
	R.x = (uchar)((((int)R.x - 128) - a) + 128);
	// P0 = s2u(p0 + b)
	L.w = (uchar)((((int)L.w - 128) + b) + 128);
	vstore4(L, 0, frame + i - 4);
	vstore4(R, 0, frame + i);
	
	L = R;
	i += 4;
	R = vload4(0, frame + i);
	nf = ((abs((int)L.w - (int)R.x) * 2 + abs((int)L.z - (int)R.y)/2) <= sub_bedge_limit) ? 1 : 0;
	// a = clamp128(clamp128(p1 - q1) + 3*(q0 - p0))
	a = (((int)L.z - 128) - ((int)R.y - 128)); 
	a = (a < -128) ? -128 : a;
	a = (a > 127) ? 127 : a;
	a = mad24((((int)R.x - 128) - ((int)L.w - 128)), 3, a);
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
	R.x = (uchar)((((int)R.x - 128) - a) + 128);
	// P0 = s2u(p0 + b)
	L.w = (uchar)((((int)L.w - 128) + b) + 128);
	vstore4(L, 0, frame + i - 4);
	vstore4(R, 0, frame + i);
	
	L = R;
	i += 4;
	R = vload4(0, frame + i);
	nf = ((abs((int)L.w - (int)R.x) * 2 + abs((int)L.z - (int)R.y)/2) <= sub_bedge_limit) ? 1 : 0;
	// a = clamp128(clamp128(p1 - q1) + 3*(q0 - p0))
	a = (((int)L.z - 128) - ((int)R.y - 128)); 
	a = (a < -128) ? -128 : a;
	a = (a > 127) ? 127 : a;
	a = mad24((((int)R.x - 128) - ((int)L.w - 128)), 3, a);
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
	R.x = (uchar)((((int)R.x - 128) - a) + 128);
	// P0 = s2u(p0 + b)
	L.w = (uchar)((((int)L.w - 128) + b) + 128);
	vstore4(L, 0, frame + i - 4);
	vstore4(R, 0, frame + i);
	
	return;	
}

__kernel __attribute__((reqd_work_group_size(64, 1, 1)))
		void simple_loop_filter_MBV(__global uchar * const frame,
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
	
	if ( y > 0) 
	{
		D0 = vload4(0, frame + i); i += width;
		D1 = vload4(0, frame + i); i += width;
		i -= width*4;
	
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

__kernel __attribute__((reqd_work_group_size(64, 1, 1)))
		void normal_loop_filter_MBH(__global uchar * const frame, //0
									const int width, //1
									const int mbedge_limit, //2
									const int sub_bedge_limit, //3
									const int interior_limit, //4
									const int hev_threshold, //5
									const int mb_size, //6
									const int mb_col) //7
{
	int x, y, i;
	uchar4 L, R;
	int p3, p2, p1, p0, q0, q1, q2, q3;
	int a, b, w;
	int filter_yes;
	int hev;
	
	y = get_global_id(0);
	x = mb_col * mb_size;
	
	i = y*width + x;
	
	R = vload4(0, frame + i);
	if ( x > 0) 
	{
		L = vload4(0, frame + i - 4);
		p3 = (int)L.x - 128; p2 = (int)L.y - 128; p1 = (int)L.z - 128; p0 = (int)L.w - 128;
		q0 = (int)R.x - 128; q1 = (int)R.y - 128; q2 = (int)R.z - 128; q3 = (int)R.y - 128;
		filter_yes = ((int)abs(p3 - p2) <= interior_limit)
						&& ((int)abs(p2 - p1) <= interior_limit)
						&& ((int)abs(p1 - p0) <= interior_limit)
						&& ((int)abs(q1 - q0) <= interior_limit)
						&& ((int)abs(q2 - q1) <= interior_limit)
						&& ((int)abs(q3 - q2) <= interior_limit)
						&& ((int)abs(p0 - q0) * 2 + abs(p1 - q1) / 2  <= mbedge_limit);
		hev = ((int)abs(p1 - p0) <= hev_threshold) && ((int)abs(q1 - q0) <= hev_threshold);
		//w = clamp128(clamp128(p1 - q1) + 3*(q0 - p0));
		w = p1 - q1;
		w = (w < -128) ? -128 : w;
		w = (w > 127) ? 127 : w;
		w = w +  mad24(q0-p0, 3, w);
		w = (w < -128) ? -128 : w;
		w = (w > 127) ? 127 : w;
		//a = clamp128((27*w + 63) >> 7);
		a = mad24(w, 27, 63) >> 7;
		a = (a < -128) ? -128 : a;
		a = (a > 127) ? 127 : a;
		a = a*filter_yes*hev;
		R.x = (uchar)((q0 - a) + 128); L.w = (uchar)((p0 + a) + 128);
		//a = clamp128((18*w + 63) >> 7);
		a = mad24(w, 18, 63) >> 7;
		a = (a < -128) ? -128 : a;
		a = (a > 127) ? 127 : a;
		a = a*filter_yes*hev;
		R.y = (uchar)((q1 - a) + 128); L.z = (uchar)((p1 + a) + 128);
		//a = clamp128((9*w + 63) >> 7);
		a = mad24(w, 9, 63) >> 7;
		a = (a < -128) ? -128 : a;
		a = (a > 127) ? 127 : a;
		a = a*filter_yes*hev;			
		R.z = (uchar)((q2 - a) + 128); L.y = (uchar)((p2 + a) + 128);
		hev = !hev;
		// a = clamp128(clamp128(p1 - q1) + 3*(q0 - p0))
		a = (p1 - q1); 
		a = (a < -128) ? -128 : a;
		a = (a > 127) ? 127 : a;
		a = mad24((q0 - p0), 3, a);
		a = (a < -128) ? -128 : a;
		a = (a > 127) ? 127 : a;
		a *= filter_yes*hev;
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
		R.x = (uchar)((q0 - a) + 128);
		L.w = (uchar)((p0 + b) + 128);
		vstore4(L, 0, frame + i - 4);
		vstore4(R, 0, frame + i);
	}
	
	// and 3 more times for edges between blocks in MB
	L = R;
	i += 4;
	R = vload4(0, frame + i);
	p3 = (int)L.x - 128; p2 = (int)L.y - 128; p1 = (int)L.z - 128; p0 = (int)L.w - 128;
	q0 = (int)R.x - 128; q1 = (int)R.y - 128; q2 = (int)R.z - 128; q3 = (int)R.y - 128;
	filter_yes = ((int)abs(p3 - p2) <= interior_limit)
					&& ((int)abs(p2 - p1) <= interior_limit)
					&& ((int)abs(p1 - p0) <= interior_limit)
					&& ((int)abs(q1 - q0) <= interior_limit)
					&& ((int)abs(q2 - q1) <= interior_limit)
					&& ((int)abs(q3 - q2) <= interior_limit)
					&& ((int)abs(p0 - q0) * 2 + abs(p1 - q1) / 2  <= mbedge_limit);
	hev = ((int)abs(p1 - p0) <= hev_threshold) && ((int)abs(q1 - q0) <= hev_threshold);
	//a = clamp128((use_outer_taps? clamp128(p1 - q1) : 0) + 3*(q0 - p0));
	a = (!hev) ? (p1 - q1) : 0; 
	a = (a < -128) ? -128 : a;
	a = (a > 127) ? 127 : a;
	a = mad24((q0 - p0), 3, a);
	a = (a < -128) ? -128 : a;
	a = (a > 127) ? 127 : a;
	a *= filter_yes;
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
	R.x = (uchar)((q0 - a) + 128);
	L.w = (uchar)((p0 + b) + 128);
	a = (a + 1) >> 1;
	a *= hev;
	R.y = (uchar)((q1 - a) + 128);
	L.z = (uchar)((p1 + a) + 128);
	vstore4(L, 0, frame + i - 4);
	vstore4(R, 0, frame + i);
	
	if (mb_size < 16) return; // we were doing chroma
	
	L = R;
	i += 4;
	R = vload4(0, frame + i);
	p3 = (int)L.x - 128; p2 = (int)L.y - 128; p1 = (int)L.z - 128; p0 = (int)L.w - 128;
	q0 = (int)R.x - 128; q1 = (int)R.y - 128; q2 = (int)R.z - 128; q3 = (int)R.y - 128;
	filter_yes = ((int)abs(p3 - p2) <= interior_limit)
					&& ((int)abs(p2 - p1) <= interior_limit)
					&& ((int)abs(p1 - p0) <= interior_limit)
					&& ((int)abs(q1 - q0) <= interior_limit)
					&& ((int)abs(q2 - q1) <= interior_limit)
					&& ((int)abs(q3 - q2) <= interior_limit)
					&& ((int)abs(p0 - q0) * 2 + abs(p1 - q1) / 2  <= mbedge_limit);
	hev = ((int)abs(p1 - p0) <= hev_threshold) && ((int)abs(q1 - q0) <= hev_threshold);
	//a = clamp128((use_outer_taps? clamp128(p1 - q1) : 0) + 3*(q0 - p0));
	a = (!hev) ? (p1 - q1) : 0; 
	a = (a < -128) ? -128 : a;
	a = (a > 127) ? 127 : a;
	a = mad24((q0 - p0), 3, a);
	a = (a < -128) ? -128 : a;
	a = (a > 127) ? 127 : a;
	a *= filter_yes;
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
	R.x = (uchar)((q0 - a) + 128);
	L.w = (uchar)((p0 + b) + 128);
	a = (a + 1) >> 1;
	a *= hev;
	R.y = (uchar)((q1 - a) + 128);
	L.z = (uchar)((p1 + a) + 128);
	vstore4(L, 0, frame + i - 4);
	vstore4(R, 0, frame + i);
	
	L = R;
	i += 4;
	R = vload4(0, frame + i);
	p3 = (int)L.x - 128; p2 = (int)L.y - 128; p1 = (int)L.z - 128; p0 = (int)L.w - 128;
	q0 = (int)R.x - 128; q1 = (int)R.y - 128; q2 = (int)R.z - 128; q3 = (int)R.y - 128;
	filter_yes = ((int)abs(p3 - p2) <= interior_limit)
					&& ((int)abs(p2 - p1) <= interior_limit)
					&& ((int)abs(p1 - p0) <= interior_limit)
					&& ((int)abs(q1 - q0) <= interior_limit)
					&& ((int)abs(q2 - q1) <= interior_limit)
					&& ((int)abs(q3 - q2) <= interior_limit)
					&& ((int)abs(p0 - q0) * 2 + abs(p1 - q1) / 2  <= mbedge_limit);
	hev = ((int)abs(p1 - p0) <= hev_threshold) && ((int)abs(q1 - q0) <= hev_threshold);
	//a = clamp128((use_outer_taps? clamp128(p1 - q1) : 0) + 3*(q0 - p0));
	a = (!hev) ? (p1 - q1) : 0; 
	a = (a < -128) ? -128 : a;
	a = (a > 127) ? 127 : a;
	a = mad24((q0 - p0), 3, a);
	a = (a < -128) ? -128 : a;
	a = (a > 127) ? 127 : a;
	a *= filter_yes;
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
	R.x = (uchar)((q0 - a) + 128);
	L.w = (uchar)((p0 + b) + 128);
	a = (a + 1) >> 1;
	a *= hev;
	R.y = (uchar)((q1 - a) + 128);
	L.z = (uchar)((p1 + a) + 128);
	vstore4(L, 0, frame + i - 4);
	vstore4(R, 0, frame + i);
	
	return;	
}

__kernel __attribute__((reqd_work_group_size(64, 1, 1)))
		void normal_loop_filter_MBV(__global uchar * const frame, //0
									const int width, //1
									const int mbedge_limit, //2
									const int sub_bedge_limit, //3
									const int interior_limit, //4
									const int hev_threshold, //5
									const int mb_size, //6
									const int mb_col) //7
{
	int x, y, i;
	// these in usual forward order (U1 lower than U0):
	uchar4 U0, U1, U2, U3, /*edge*/ D0, D1, D2, D3;
	// these in vp8spec order:
	int4 p3, p2, p1, p0, /*edge*/ q0, q1, q2, q3;
	int4 a, b, w;
	int4 filter_yes;
	int4 hev;
	
	y = (get_global_id(0)/(mb_size/4))*mb_size;
	x = mad24((int)mb_col, mb_size, (int)(get_global_id(0)%(mb_size/4))*4);
	
	i = y*width + x;
	
	D0 = vload4(0, frame + i); i+= width;
	D1 = vload4(0, frame + i); i+= width;
	D2 = vload4(0, frame + i); i+= width;
	D3 = vload4(0, frame + i); i+= width;
	i -= width*4;
	
	if ( y > 0) 
	{
		i -= width*4;
		U0 = vload4(0, frame + i); i+= width;
		U1 = vload4(0, frame + i); i+= width;
		U2 = vload4(0, frame + i); i+= width;
		U3 = vload4(0, frame + i); i+= width;
	
		p3 = convert_int4(U0) - 128;
		p2 = convert_int4(U1) - 128;
		p1 = convert_int4(U2) - 128;
		p0 = convert_int4(U3) - 128;
		q0 = convert_int4(D0) - 128;
		q1 = convert_int4(D1) - 128;
		q2 = convert_int4(D2) - 128;
		q3 = convert_int4(D3) - 128;
		
		filter_yes = (convert_int4(abs(p3 - p2)) <= interior_limit)
						&& (convert_int4(abs(p2 - p1)) <= interior_limit)
						&& (convert_int4(abs(p1 - p0)) <= interior_limit)
						&& (convert_int4(abs(q1 - q0)) <= interior_limit)
						&& (convert_int4(abs(q2 - q1)) <= interior_limit)
						&& (convert_int4(abs(q3 - q2)) <= interior_limit)
						&& ((convert_int4(abs(p0 - q0)) * 2 + convert_int4(abs(p1 - q1)) / 2)  <= mbedge_limit);
		hev = (convert_int4(abs(p1 - p0)) <= hev_threshold) && (convert_int4(abs(q1 - q0)) <= hev_threshold);
		filter_yes = convert_int4(abs(filter_yes)); hev = convert_int4(abs(hev));
		
		//w = clamp128(clamp128(p1 - q1) + 3*(q0 - p0));
		w = p1 - q1;
		w = (w < -128) ? -128 : w;
		w = (w > 127) ? 127 : w;
		w = w +  mad24(q0-p0, 3, w);
		w = (w < -128) ? -128 : w;
		w = (w > 127) ? 127 : w;
		//a = clamp128((27*w + 63) >> 7);
		a = mad24(w, 27, 63) >> 7;
		a = (a < -128) ? -128 : a;
		a = (a > 127) ? 127 : a;
		a = a*filter_yes*hev;
		D0 = convert_uchar4((q0 - a) + 128); U3 = convert_uchar4((p0 + a) + 128);
		//a = clamp128((18*w + 63) >> 7);
		a = mad24(w, 18, 63) >> 7;
		a = (a < -128) ? -128 : a;
		a = (a > 127) ? 127 : a;
		a = a*filter_yes*hev;
		D1 = convert_uchar4((q1 - a) + 128); U2 = convert_uchar4((p1 + a) + 128);
		//a = clamp128((9*w + 63) >> 7);
		a = mad24(w, 9, 63) >> 7;
		a = (a < -128) ? -128 : a;
		a = (a > 127) ? 127 : a;
		a = a*filter_yes*hev;			
		D2 = convert_uchar4((q2 - a) + 128); U1 = convert_uchar4((p2 + a) + 128);

		hev = 1 - hev;
		// a = clamp128(clamp128(p1 - q1) + 3*(q0 - p0))
		a = (p1 - q1); 
		a = (a < -128) ? -128 : a;
		a = (a > 127) ? 127 : a;
		a = mad24((q0 - p0), 3, a);
		a = (a < -128) ? -128 : a;
		a = (a > 127) ? 127 : a;
		a *= filter_yes*hev;
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
		D0 = convert_uchar4((q0 - a) + 128); U3 = convert_uchar4((p0 + b) + 128);
		vstore4(U1, 0, frame + i - 3*width);
		vstore4(U2, 0, frame + i - 2*width);
		vstore4(U3, 0, frame + i - width);
		vstore4(D0, 0, frame + i);
		vstore4(D1, 0, frame + i + width);
		vstore4(D2, 0, frame + i + 2*width);
	}
	
	// and 3 more times for edges between blocks in MB

	i += width*4;
	U0 = D0;
	U1 = D1;
	U2 = D2;
	U3 = D3;
	D0 = vload4(0, frame + i); i+= width;
	D1 = vload4(0, frame + i); i+= width;
	D2 = vload4(0, frame + i); i+= width;
	D3 = vload4(0, frame + i); i+= width;
	i -= width*4;
	p3 = convert_int4(U0) - 128;
	p2 = convert_int4(U1) - 128;
	p1 = convert_int4(U2) - 128;
	p0 = convert_int4(U3) - 128;
	q0 = convert_int4(D0) - 128;
	q1 = convert_int4(D1) - 128;
	q2 = convert_int4(D2) - 128;
	q3 = convert_int4(D3) - 128;
	filter_yes = (convert_int4(abs(p3 - p2)) <= interior_limit)
					&& (convert_int4(abs(p2 - p1)) <= interior_limit)
					&& (convert_int4(abs(p1 - p0)) <= interior_limit)
					&& (convert_int4(abs(q1 - q0)) <= interior_limit)
					&& (convert_int4(abs(q2 - q1)) <= interior_limit)
					&& (convert_int4(abs(q3 - q2)) <= interior_limit)
					&& ((convert_int4(abs(p0 - q0)) * 2 + convert_int4(abs(p1 - q1)) / 2)  <= mbedge_limit);
	hev = (convert_int4(abs(p1 - p0)) <= hev_threshold) && (convert_int4(abs(q1 - q0)) <= hev_threshold);
	filter_yes = convert_int4(abs(filter_yes)); hev = convert_int4(abs(hev));
	//a = clamp128((use_outer_taps? clamp128(p1 - q1) : 0) + 3*(q0 - p0));
	a = select(0,p1-q1,hev==0);
	a = select(a,-128,a<-128);
	a = select(a,127,a>127);
	a = mad24((q0 - p0), 3, a);
	a = select(a,-128,a<-128);
	a = select(a,127,a>127);
	a *= filter_yes;
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
	D0 = convert_uchar4((q0 - a) + 128);
	U3 = convert_uchar4((p0 + b) + 128);
	a = (a + 1) >> 1;
	a *= hev;
	D1 = convert_uchar4((q1 - a) + 128);
	U2 = convert_uchar4((p1 + a) + 128);
	vstore4(U1, 0, frame + i - 3*width);
	vstore4(U2, 0, frame + i - 2*width);
	vstore4(U3, 0, frame + i - width);
	vstore4(D0, 0, frame + i);
	vstore4(D1, 0, frame + i + width);
	vstore4(D2, 0, frame + i + 2*width);
	
	if (mb_size < 16) return; //we were doing chroma
	
	i += width*4;
	U0 = D0;
	U1 = D1;
	U2 = D2;
	U3 = D3;
	D0 = vload4(0, frame + i); i+= width;
	D1 = vload4(0, frame + i); i+= width;
	D2 = vload4(0, frame + i); i+= width;
	D3 = vload4(0, frame + i); i+= width;
	i -= width*4;
	p3 = convert_int4(U0) - 128;
	p2 = convert_int4(U1) - 128;
	p1 = convert_int4(U2) - 128;
	p0 = convert_int4(U3) - 128;
	q0 = convert_int4(D0) - 128;
	q1 = convert_int4(D1) - 128;
	q2 = convert_int4(D2) - 128;
	q3 = convert_int4(D3) - 128;
	filter_yes = (convert_int4(abs(p3 - p2)) <= interior_limit)
					&& (convert_int4(abs(p2 - p1)) <= interior_limit)
					&& (convert_int4(abs(p1 - p0)) <= interior_limit)
					&& (convert_int4(abs(q1 - q0)) <= interior_limit)
					&& (convert_int4(abs(q2 - q1)) <= interior_limit)
					&& (convert_int4(abs(q3 - q2)) <= interior_limit)
					&& ((convert_int4(abs(p0 - q0)) * 2 + convert_int4(abs(p1 - q1)) / 2)  <= mbedge_limit);
	hev = (convert_int4(abs(p1 - p0)) <= hev_threshold) && (convert_int4(abs(q1 - q0)) <= hev_threshold);
	filter_yes = convert_int4(abs(filter_yes)); hev = convert_int4(abs(hev));
	//a = clamp128((use_outer_taps? clamp128(p1 - q1) : 0) + 3*(q0 - p0));
	a = select(0,p1-q1,hev==0);
	a = select(a,-128,a<-128);
	a = select(a,127,a>127);
	a = mad24((q0 - p0), 3, a);
	a = select(a,-128,a<-128);
	a = select(a,127,a>127);
	a *= filter_yes;
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
	D0 = convert_uchar4((q0 - a) + 128);
	U3 = convert_uchar4((p0 + b) + 128);
	a = (a + 1) >> 1;
	a *= hev;
	D1 = convert_uchar4((q1 - a) + 128);
	U2 = convert_uchar4((p1 + a) + 128);
	vstore4(U1, 0, frame + i - 3*width);
	vstore4(U2, 0, frame + i - 2*width);
	vstore4(U3, 0, frame + i - width);
	vstore4(D0, 0, frame + i);
	vstore4(D1, 0, frame + i + width);
	vstore4(D2, 0, frame + i + 2*width);
	
	i += width*4;
	U0 = D0;
	U1 = D1;
	U2 = D2;
	U3 = D3;
	D0 = vload4(0, frame + i); i+= width;
	D1 = vload4(0, frame + i); i+= width;
	D2 = vload4(0, frame + i); i+= width;
	D3 = vload4(0, frame + i); i+= width;
	i -= width*4;
	p3 = convert_int4(U0) - 128;
	p2 = convert_int4(U1) - 128;
	p1 = convert_int4(U2) - 128;
	p0 = convert_int4(U3) - 128;
	q0 = convert_int4(D0) - 128;
	q1 = convert_int4(D1) - 128;
	q2 = convert_int4(D2) - 128;
	q3 = convert_int4(D3) - 128;
	filter_yes = (convert_int4(abs(p3 - p2)) <= interior_limit)
					&& (convert_int4(abs(p2 - p1)) <= interior_limit)
					&& (convert_int4(abs(p1 - p0)) <= interior_limit)
					&& (convert_int4(abs(q1 - q0)) <= interior_limit)
					&& (convert_int4(abs(q2 - q1)) <= interior_limit)
					&& (convert_int4(abs(q3 - q2)) <= interior_limit)
					&& ((convert_int4(abs(p0 - q0)) * 2 + convert_int4(abs(p1 - q1)) / 2)  <= mbedge_limit);
	hev = (convert_int4(abs(p1 - p0)) <= hev_threshold) && (convert_int4(abs(q1 - q0)) <= hev_threshold);
	filter_yes = convert_int4(abs(filter_yes)); hev = convert_int4(abs(hev));
	//a = clamp128((use_outer_taps? clamp128(p1 - q1) : 0) + 3*(q0 - p0));
	a = select(0,p1-q1,hev==0);
	a = select(a,-128,a<-128);
	a = select(a,127,a>127);
	a = mad24((q0 - p0), 3, a);
	a = select(a,-128,a<-128);
	a = select(a,127,a>127);
	a *= filter_yes;
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
	D0 = convert_uchar4((q0 - a) + 128);
	U3 = convert_uchar4((p0 + b) + 128);
	a = (a + 1) >> 1;
	a *= hev;
	D1 = convert_uchar4((q1 - a) + 128);
	U2 = convert_uchar4((p1 + a) + 128);
	vstore4(U1, 0, frame + i - 3*width);
	vstore4(U2, 0, frame + i - 2*width);
	vstore4(U3, 0, frame + i - width);
	vstore4(D0, 0, frame + i);
	vstore4(D1, 0, frame + i + width);
	vstore4(D2, 0, frame + i + 2*width);	
	
	return;	
}

__kernel void luma_interpolate_Hx4( __global uchar *const src_frame, //0
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

__kernel void luma_interpolate_Vx4( __global uchar *const frame, //0
									const int width, //1
									const int height) //2
{
	if (get_global_id(0) > (width-1)) return;

	__private uchar4 A0, A1, M0, M1, M2, M3, U0, U1, U2;
	__private int4 buf;

	int width_x4 = width*4;
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

__kernel void chroma_interpolate_Hx8( 	__global uchar *const src_frame, //0
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
	__private uchar16 M16l, M16h;
	
	int i, ind, buf;
	int width_x8 = width*8;
	
	
	ind = (get_global_id(0) + 1)*width - 4;
	M4 = vload4(0, src_frame + ind);
	R4.s0 = M4.s3;	R4.s1 = M4.s3;	R4.s2 = M4.s3;
	
	for (i = width-4; i >= 4; i -= 4)
	{
		ind = get_global_id(0)*width + i;
		L4 = vload4(0, src_frame + ind - 4);
		
		M16l.s0 = M4.s0;
		buf = mad24((int)L4.s3,-6,mad24((int)M4.s0,123,mad24((int)M4.s1,12,mad24((int)M4.s2,-1,64))))/128;
		buf = (buf < 0) ? 0 : buf; 
		buf = (buf > 255) ? 255 : buf; 
		M16l.s1 = (uchar)buf;
		buf = mad24((int)L4.s2,2,mad24((int)L4.s3,-11,mad24((int)M4.s0,108,mad24((int)M4.s1,36,mad24((int)M4.s2,-8, (int)M4.s3 + 64)))))/128;
		buf = (buf < 0) ? 0 : buf; 
		buf = (buf > 255) ? 255 : buf; 
		M16l.s2 = (uchar)buf;
		buf = mad24((int)L4.s3,-9,mad24((int)M4.s0,93,mad24((int)M4.s1,50,mad24((int)M4.s2,-6,64))))/128;
		buf = (buf < 0) ? 0 : buf; 
		buf = (buf > 255) ? 255 : buf; 
		M16l.s3 = (uchar)buf;
		buf = mad24((int)L4.s2,3,mad24((int)L4.s3,-16,mad24((int)M4.s0,77,mad24((int)M4.s1,77,mad24((int)M4.s2,-16, mad24((int)M4.s3,3,64))))))/128;
		buf = (buf < 0) ? 0 : buf; 
		buf = (buf > 255) ? 255 : buf; 
		M16l.s4 = (uchar)buf;
		buf = mad24((int)L4.s3,-6,mad24((int)M4.s0,50,mad24((int)M4.s1,93,mad24((int)M4.s2,-9,64))))/128;
		buf = (buf < 0) ? 0 : buf; 
		buf = (buf > 255) ? 255 : buf; 
		M16l.s5 = (uchar)buf;
		buf = ((int)L4.s2 + mad24((int)L4.s3,-8,mad24((int)M4.s0,36,mad24((int)M4.s1,108,mad24((int)M4.s2,-11, mad24((int)M4.s3,2,64))))))/128;		
		buf = (buf < 0) ? 0 : buf; 
		buf = (buf > 255) ? 255 : buf; 
		M16l.s6 = (uchar)buf;
		buf = mad24((int)L4.s3,-1,mad24((int)M4.s0,12,mad24((int)M4.s1,123,mad24((int)M4.s2,-6,64))))/128;
		buf = (buf < 0) ? 0 : buf; 
		buf = (buf > 255) ? 255 : buf; 
		M16l.s7 = (uchar)buf;
		
		M16l.s8 = M4.s1;
		buf = mad24((int)M4.s0,-6,mad24((int)M4.s1,123,mad24((int)M4.s2,12,mad24((int)M4.s3,-1,64))))/128;
		buf = (buf < 0) ? 0 : buf; 
		buf = (buf > 255) ? 255 : buf; 
		M16l.s9 = (uchar)buf;
		buf = mad24((int)L4.s3,2,mad24((int)M4.s0,-11,mad24((int)M4.s1,108,mad24((int)M4.s2,36,mad24((int)M4.s3,-8, (int)R4.s0 + 64)))))/128;
		buf = (buf < 0) ? 0 : buf; 
		buf = (buf > 255) ? 255 : buf; 
		M16l.sA = (uchar)buf;
		buf = mad24((int)M4.s0,-9,mad24((int)M4.s1,93,mad24((int)M4.s2,50,mad24((int)M4.s3,-6,64))))/128;
		buf = (buf < 0) ? 0 : buf; 
		buf = (buf > 255) ? 255 : buf; 
		M16l.sB = (uchar)buf;
		buf = mad24((int)L4.s3,3,mad24((int)M4.s0,-16,mad24((int)M4.s1,77,mad24((int)M4.s2,77,mad24((int)M4.s3,-16, mad24((int)R4.s0,3,64))))))/128;
		buf = (buf < 0) ? 0 : buf; 
		buf = (buf > 255) ? 255 : buf; 
		M16l.sC = (uchar)buf;
		buf = mad24((int)M4.s0,-6,mad24((int)M4.s1,50,mad24((int)M4.s2,93,mad24((int)M4.s3,-9,64))))/128;
		buf = (buf < 0) ? 0 : buf; 
		buf = (buf > 255) ? 255 : buf; 
		M16l.sD = (uchar)buf;
		buf = ((int)L4.s3 + mad24((int)M4.s0,-8,mad24((int)M4.s1,36,mad24((int)M4.s2,108,mad24((int)M4.s3,-11, mad24((int)R4.s0,2,64))))))/128;		
		buf = (buf < 0) ? 0 : buf; 
		buf = (buf > 255) ? 255 : buf; 
		M16l.sE = (uchar)buf;
		buf = mad24((int)M4.s0,-1,mad24((int)M4.s1,12,mad24((int)M4.s2,123,mad24((int)M4.s3,-6,64))))/128;
		buf = (buf < 0) ? 0 : buf; 
		buf = (buf > 255) ? 255 : buf; 
		M16l.sF = (uchar)buf;
		
		M16h.s0 = M4.s2;
		buf = mad24((int)M4.s1,-6,mad24((int)M4.s2,123,mad24((int)M4.s3,12,mad24((int)R4.s0,-1,64))))/128;
		buf = (buf < 0) ? 0 : buf; 
		buf = (buf > 255) ? 255 : buf; 
		M16h.s1 = (uchar)buf;
		buf = mad24((int)M4.s0,2,mad24((int)M4.s1,-11,mad24((int)M4.s2,108,mad24((int)M4.s3,36,mad24((int)R4.s0,-8, (int)R4.s1 + 64)))))/128;
		buf = (buf < 0) ? 0 : buf; 
		buf = (buf > 255) ? 255 : buf; 
		M16h.s2 = (uchar)buf;
		buf = mad24((int)M4.s1,-9,mad24((int)M4.s2,93,mad24((int)M4.s3,50,mad24((int)R4.s0,-6,64))))/128;
		buf = (buf < 0) ? 0 : buf; 
		buf = (buf > 255) ? 255 : buf; 
		M16h.s3 = (uchar)buf;
		buf = mad24((int)M4.s0,3,mad24((int)M4.s1,-16,mad24((int)M4.s2,77,mad24((int)M4.s3,77,mad24((int)R4.s0,-16, mad24((int)R4.s1,3,64))))))/128;
		buf = (buf < 0) ? 0 : buf; 
		buf = (buf > 255) ? 255 : buf; 
		M16h.s4 = (uchar)buf;
		buf = mad24((int)M4.s1,-6,mad24((int)M4.s2,50,mad24((int)M4.s3,93,mad24((int)R4.s0,-9,64))))/128;
		buf = (buf < 0) ? 0 : buf; 
		buf = (buf > 255) ? 255 : buf; 
		M16h.s5 = (uchar)buf;
		buf = ((int)M4.s0 + mad24((int)M4.s1,-8,mad24((int)M4.s2,36,mad24((int)M4.s3,108,mad24((int)R4.s0,-11, mad24((int)R4.s1,2,64))))))/128;		
		buf = (buf < 0) ? 0 : buf; 
		buf = (buf > 255) ? 255 : buf; 
		M16h.s6 = (uchar)buf;
		buf = mad24((int)M4.s1,-1,mad24((int)M4.s2,12,mad24((int)M4.s3,123,mad24((int)R4.s0,-6,64))))/128;
		buf = (buf < 0) ? 0 : buf; 
		buf = (buf > 255) ? 255 : buf; 
		M16h.s7 = (uchar)buf;
		
		M16h.s8 = M4.s3;
		buf = mad24((int)M4.s2,-6,mad24((int)M4.s3,123,mad24((int)R4.s0,12,mad24((int)R4.s1,-1,64))))/128;
		buf = (buf < 0) ? 0 : buf; 
		buf = (buf > 255) ? 255 : buf; 
		M16h.s9 = (uchar)buf;
		buf = mad24((int)M4.s1,2,mad24((int)M4.s2,-11,mad24((int)M4.s3,108,mad24((int)R4.s0,36,mad24((int)R4.s1,-8, (int)R4.s2 + 64)))))/128;
		buf = (buf < 0) ? 0 : buf; 
		buf = (buf > 255) ? 255 : buf; 
		M16h.sA = (uchar)buf;
		buf = mad24((int)M4.s2,-9,mad24((int)M4.s3,93,mad24((int)R4.s0,50,mad24((int)R4.s1,-6,64))))/128;
		buf = (buf < 0) ? 0 : buf; 
		buf = (buf > 255) ? 255 : buf; 
		M16h.sB = (uchar)buf;
		buf = mad24((int)M4.s1,3,mad24((int)M4.s2,-16,mad24((int)M4.s3,77,mad24((int)R4.s0,77,mad24((int)R4.s1,-16, mad24((int)R4.s2,3,64))))))/128;
		buf = (buf < 0) ? 0 : buf; 
		buf = (buf > 255) ? 255 : buf; 
		M16h.sC = (uchar)buf;
		buf = mad24((int)M4.s2,-6,mad24((int)M4.s3,50,mad24((int)R4.s0,93,mad24((int)R4.s1,-9,64))))/128;
		buf = (buf < 0) ? 0 : buf; 
		buf = (buf > 255) ? 255 : buf; 
		M16h.sD = (uchar)buf;
		buf = ((int)M4.s1 + mad24((int)M4.s2,-8,mad24((int)M4.s3,36,mad24((int)R4.s0,108,mad24((int)R4.s1,-11, mad24((int)R4.s2,2,64))))))/128;		
		buf = (buf < 0) ? 0 : buf; 
		buf = (buf > 255) ? 255 : buf; 
		M16h.sE = (uchar)buf;
		buf = mad24((int)M4.s2,-1,mad24((int)M4.s3,12,mad24((int)R4.s0,123,mad24((int)R4.s1,-6,64))))/128;
		buf = (buf < 0) ? 0 : buf; 
		buf = (buf > 255) ? 255 : buf; 
		M16h.sF = (uchar)buf;

		ind = get_global_id(0)*width_x8 + (i*8);

		vstore16(M16l, 0, dst_frame + ind);
		vstore16(M16h, 0, dst_frame + ind + 16);
		
		R4 = M4;
		M4 = L4;
	}
	
	L4.s2 = M4.s0; L4.s3 = M4.s0;

	M16l.s0 = M4.s0;
	buf = mad24((int)L4.s3,-6,mad24((int)M4.s0,123,mad24((int)M4.s1,12,mad24((int)M4.s2,-1,64))))/128;
	buf = (buf < 0) ? 0 : buf; 
	buf = (buf > 255) ? 255 : buf; 
	M16l.s1 = (uchar)buf;
	buf = mad24((int)L4.s2,2,mad24((int)L4.s3,-11,mad24((int)M4.s0,108,mad24((int)M4.s1,36,mad24((int)M4.s2,-8, (int)M4.s3 + 64)))))/128;
	buf = (buf < 0) ? 0 : buf; 
	buf = (buf > 255) ? 255 : buf; 
	M16l.s2 = (uchar)buf;
	buf = mad24((int)L4.s3,-9,mad24((int)M4.s0,93,mad24((int)M4.s1,50,mad24((int)M4.s2,-6,64))))/128;
	buf = (buf < 0) ? 0 : buf; 
	buf = (buf > 255) ? 255 : buf; 
	M16l.s3 = (uchar)buf;
	buf = mad24((int)L4.s2,3,mad24((int)L4.s3,-16,mad24((int)M4.s0,77,mad24((int)M4.s1,77,mad24((int)M4.s2,-16, mad24((int)M4.s3,3,64))))))/128;
	buf = (buf < 0) ? 0 : buf; 
	buf = (buf > 255) ? 255 : buf; 
	M16l.s4 = (uchar)buf;
	buf = mad24((int)L4.s3,-6,mad24((int)M4.s0,50,mad24((int)M4.s1,93,mad24((int)M4.s2,-9,64))))/128;
	buf = (buf < 0) ? 0 : buf; 
	buf = (buf > 255) ? 255 : buf; 
	M16l.s5 = (uchar)buf;
	buf = ((int)L4.s2 + mad24((int)L4.s3,-8,mad24((int)M4.s0,36,mad24((int)M4.s1,108,mad24((int)M4.s2,-11, mad24((int)M4.s3,2,64))))))/128;		
	buf = (buf < 0) ? 0 : buf; 
	buf = (buf > 255) ? 255 : buf; 
	M16l.s6 = (uchar)buf;
	buf = mad24((int)L4.s3,-1,mad24((int)M4.s0,12,mad24((int)M4.s1,123,mad24((int)M4.s2,-6,64))))/128;
	buf = (buf < 0) ? 0 : buf; 
	buf = (buf > 255) ? 255 : buf; 
	M16l.s7 = (uchar)buf;
		
	M16l.s8 = M4.s1;
	buf = mad24((int)M4.s0,-6,mad24((int)M4.s1,123,mad24((int)M4.s2,12,mad24((int)M4.s3,-1,64))))/128;
	buf = (buf < 0) ? 0 : buf; 
	buf = (buf > 255) ? 255 : buf; 
	M16l.s9 = (uchar)buf;
	buf = mad24((int)L4.s3,2,mad24((int)M4.s0,-11,mad24((int)M4.s1,108,mad24((int)M4.s2,36,mad24((int)M4.s3,-8, (int)R4.s0 + 64)))))/128;
	buf = (buf < 0) ? 0 : buf; 
	buf = (buf > 255) ? 255 : buf; 
	M16l.sA = (uchar)buf;
	buf = mad24((int)M4.s0,-9,mad24((int)M4.s1,93,mad24((int)M4.s2,50,mad24((int)M4.s3,-6,64))))/128;
	buf = (buf < 0) ? 0 : buf; 
	buf = (buf > 255) ? 255 : buf; 
	M16l.sB = (uchar)buf;
	buf = mad24((int)L4.s3,3,mad24((int)M4.s0,-16,mad24((int)M4.s1,77,mad24((int)M4.s2,77,mad24((int)M4.s3,-16, mad24((int)R4.s0,3,64))))))/128;
	buf = (buf < 0) ? 0 : buf; 
	buf = (buf > 255) ? 255 : buf; 
	M16l.sC = (uchar)buf;
	buf = mad24((int)M4.s0,-6,mad24((int)M4.s1,50,mad24((int)M4.s2,93,mad24((int)M4.s3,-9,64))))/128;
	buf = (buf < 0) ? 0 : buf; 
	buf = (buf > 255) ? 255 : buf; 
	M16l.sD = (uchar)buf;
	buf = ((int)L4.s3 + mad24((int)M4.s0,-8,mad24((int)M4.s1,36,mad24((int)M4.s2,108,mad24((int)M4.s3,-11, mad24((int)R4.s0,2,64))))))/128;		
	buf = (buf < 0) ? 0 : buf; 
	buf = (buf > 255) ? 255 : buf; 
	M16l.sE = (uchar)buf;
	buf = mad24((int)M4.s0,-1,mad24((int)M4.s1,12,mad24((int)M4.s2,123,mad24((int)M4.s3,-6,64))))/128;
	buf = (buf < 0) ? 0 : buf; 
	buf = (buf > 255) ? 255 : buf; 
	M16l.sF = (uchar)buf;
		
	M16h.s0 = M4.s2;
	buf = mad24((int)M4.s1,-6,mad24((int)M4.s2,123,mad24((int)M4.s3,12,mad24((int)R4.s0,-1,64))))/128;
	buf = (buf < 0) ? 0 : buf; 
	buf = (buf > 255) ? 255 : buf; 
	M16h.s1 = (uchar)buf;
	buf = mad24((int)M4.s0,2,mad24((int)M4.s1,-11,mad24((int)M4.s2,108,mad24((int)M4.s3,36,mad24((int)R4.s0,-8, (int)R4.s1 + 64)))))/128;
	buf = (buf < 0) ? 0 : buf; 
	buf = (buf > 255) ? 255 : buf; 
	M16h.s2 = (uchar)buf;
	buf = mad24((int)M4.s1,-9,mad24((int)M4.s2,93,mad24((int)M4.s3,50,mad24((int)R4.s0,-6,64))))/128;
	buf = (buf < 0) ? 0 : buf; 
	buf = (buf > 255) ? 255 : buf; 
	M16h.s3 = (uchar)buf;
	buf = mad24((int)M4.s0,3,mad24((int)M4.s1,-16,mad24((int)M4.s2,77,mad24((int)M4.s3,77,mad24((int)R4.s0,-16, mad24((int)R4.s1,3,64))))))/128;
	buf = (buf < 0) ? 0 : buf; 
	buf = (buf > 255) ? 255 : buf; 
	M16h.s4 = (uchar)buf;
	buf = mad24((int)M4.s1,-6,mad24((int)M4.s2,50,mad24((int)M4.s3,93,mad24((int)R4.s0,-9,64))))/128;
	buf = (buf < 0) ? 0 : buf; 
	buf = (buf > 255) ? 255 : buf; 
	M16h.s5 = (uchar)buf;
	buf = ((int)M4.s0 + mad24((int)M4.s1,-8,mad24((int)M4.s2,36,mad24((int)M4.s3,108,mad24((int)R4.s0,-11, mad24((int)R4.s1,2,64))))))/128;		
	buf = (buf < 0) ? 0 : buf; 
	buf = (buf > 255) ? 255 : buf; 
	M16h.s6 = (uchar)buf;
	buf = mad24((int)M4.s1,-1,mad24((int)M4.s2,12,mad24((int)M4.s3,123,mad24((int)R4.s0,-6,64))))/128;
	buf = (buf < 0) ? 0 : buf; 
	buf = (buf > 255) ? 255 : buf; 
	M16h.s7 = (uchar)buf;
	
	M16h.s8 = M4.s3;
	buf = mad24((int)M4.s2,-6,mad24((int)M4.s3,123,mad24((int)R4.s0,12,mad24((int)R4.s1,-1,64))))/128;
	buf = (buf < 0) ? 0 : buf; 
	buf = (buf > 255) ? 255 : buf; 
	M16h.s9 = (uchar)buf;
	buf = mad24((int)M4.s1,2,mad24((int)M4.s2,-11,mad24((int)M4.s3,108,mad24((int)R4.s0,36,mad24((int)R4.s1,-8, (int)R4.s2 + 64)))))/128;
	buf = (buf < 0) ? 0 : buf; 
	buf = (buf > 255) ? 255 : buf; 
	M16h.sA = (uchar)buf;
	buf = mad24((int)M4.s2,-9,mad24((int)M4.s3,93,mad24((int)R4.s0,50,mad24((int)R4.s1,-6,64))))/128;
	buf = (buf < 0) ? 0 : buf; 
	buf = (buf > 255) ? 255 : buf; 
	M16h.sB = (uchar)buf;
	buf = mad24((int)M4.s1,3,mad24((int)M4.s2,-16,mad24((int)M4.s3,77,mad24((int)R4.s0,77,mad24((int)R4.s1,-16, mad24((int)R4.s2,3,64))))))/128;
	buf = (buf < 0) ? 0 : buf; 
	buf = (buf > 255) ? 255 : buf; 
	M16h.sC = (uchar)buf;
	buf = mad24((int)M4.s2,-6,mad24((int)M4.s3,50,mad24((int)R4.s0,93,mad24((int)R4.s1,-9,64))))/128;
	buf = (buf < 0) ? 0 : buf; 
	buf = (buf > 255) ? 255 : buf; 
	M16h.sD = (uchar)buf;
	buf = ((int)M4.s1 + mad24((int)M4.s2,-8,mad24((int)M4.s3,36,mad24((int)R4.s0,108,mad24((int)R4.s1,-11, mad24((int)R4.s2,2,64))))))/128;		
	buf = (buf < 0) ? 0 : buf; 
	buf = (buf > 255) ? 255 : buf; 
	M16h.sE = (uchar)buf;
	buf = mad24((int)M4.s2,-1,mad24((int)M4.s3,12,mad24((int)R4.s0,123,mad24((int)R4.s1,-6,64))))/128;
	buf = (buf < 0) ? 0 : buf; 
	buf = (buf > 255) ? 255 : buf; 
	M16h.sF = (uchar)buf;

	ind = get_global_id(0)*width_x8;
	vstore16(M16l, 0, dst_frame + ind);
	vstore16(M16h, 0, dst_frame + ind + 16);

	return;	
}

__kernel void chroma_interpolate_Vx8( __global uchar *const frame, //0
									const int width, //1
									const int height) //2
{
	__private uchar4 A0, A1, M0, M1, M2, M3, M4, M5, M6, M7, U0, U1, U2;
	__private int4 buf;

	int width_x8 = width*8;
	int i, ind;
	
	ind = (height-1)*width_x8 + (get_global_id(0)*4);
	M0 = vload4(0, frame + ind);
	A1 = vload4(0, frame + ind - width_x8);
	U0 = M0; U1 = M0; U2 = M0;
	
	for (i = height-1; i >= 2; --i)
	{
		ind = i*width_x8 + (get_global_id(0)*4);
		A0 = vload4(0, frame + ind - width_x8*2);
	
		buf = mad24(convert_int4(A1),-6,mad24(convert_int4(M0),123,mad24(convert_int4(U0),12,mad24(convert_int4(U1),-1,64))))/128;
		M1 = convert_uchar4_sat(buf);
		buf = mad24(convert_int4(A0),2,mad24(convert_int4(A1),-11,mad24(convert_int4(M0),108,mad24(convert_int4(U0),36,mad24(convert_int4(U1),-8, convert_int4(U2) + 64)))))/128;
		M2 = convert_uchar4_sat(buf);
		buf = mad24(convert_int4(A1),-9,mad24(convert_int4(M0),93,mad24(convert_int4(U0),50,mad24(convert_int4(U1),-6,64))))/128;
		M3 = convert_uchar4_sat(buf);
		buf = mad24(convert_int4(A0),3,mad24(convert_int4(A1),-16,mad24(convert_int4(M0),77,mad24(convert_int4(U0),77,mad24(convert_int4(U1),-16, mad24(convert_int4(U2),3,64))))))/128;
		M4 = convert_uchar4_sat(buf);
		buf = mad24(convert_int4(A1),-6,mad24(convert_int4(M0),50,mad24(convert_int4(U0),93,mad24(convert_int4(U1),-9,64))))/128;
		M5 = convert_uchar4_sat(buf);
		buf = (convert_int4(A0) + mad24(convert_int4(A1),-8,mad24(convert_int4(M0),36,mad24(convert_int4(U0),108,mad24(convert_int4(U1),-11, mad24(convert_int4(U2),2,64))))))/128;		
		M6 = convert_uchar4_sat(buf);
		buf = mad24(convert_int4(A1),-1,mad24(convert_int4(M0),12,mad24(convert_int4(U0),123,mad24(convert_int4(U1),-6,64))))/128;
		M7 = convert_uchar4_sat(buf);
		
		ind = (i*8)*width_x8 + (get_global_id(0)*4);
		vstore4(M0, 0, frame + ind); ind += width_x8;
		vstore4(M1, 0, frame + ind); ind += width_x8;
		vstore4(M2, 0, frame + ind); ind += width_x8;
		vstore4(M3, 0, frame + ind); ind += width_x8;
		vstore4(M4, 0, frame + ind); ind += width_x8;
		vstore4(M5, 0, frame + ind); ind += width_x8;
		vstore4(M6, 0, frame + ind); ind += width_x8;
		vstore4(M7, 0, frame + ind);
		
		U2 = U1;
		U1 = U0;
		U0 = M0;
		M0 = A1;
		A1 = A0;
	}
	
	buf = mad24(convert_int4(A1),-6,mad24(convert_int4(M0),123,mad24(convert_int4(U0),12,mad24(convert_int4(U1),-1,64))))/128;
	M1 = convert_uchar4_sat(buf);
	buf = mad24(convert_int4(A0),2,mad24(convert_int4(A1),-11,mad24(convert_int4(M0),108,mad24(convert_int4(U0),36,mad24(convert_int4(U1),-8, convert_int4(U2) + 64)))))/128;
	M2 = convert_uchar4_sat(buf);
	buf = mad24(convert_int4(A1),-9,mad24(convert_int4(M0),93,mad24(convert_int4(U0),50,mad24(convert_int4(U1),-6,64))))/128;
	M3 = convert_uchar4_sat(buf);
	buf = mad24(convert_int4(A0),3,mad24(convert_int4(A1),-16,mad24(convert_int4(M0),77,mad24(convert_int4(U0),77,mad24(convert_int4(U1),-16, mad24(convert_int4(U2),3,64))))))/128;
	M4 = convert_uchar4_sat(buf);
	buf = mad24(convert_int4(A1),-6,mad24(convert_int4(M0),50,mad24(convert_int4(U0),93,mad24(convert_int4(U1),-9,64))))/128;
	M5 = convert_uchar4_sat(buf);
	buf = (convert_int4(A0) + mad24(convert_int4(A1),-8,mad24(convert_int4(M0),36,mad24(convert_int4(U0),108,mad24(convert_int4(U1),-11, mad24(convert_int4(U2),2,64))))))/128;		
	M6 = convert_uchar4_sat(buf);
	buf = mad24(convert_int4(A1),-1,mad24(convert_int4(M0),12,mad24(convert_int4(U0),123,mad24(convert_int4(U1),-6,64))))/128;
	M7 = convert_uchar4_sat(buf);
		
	ind = 8*width_x8 + (get_global_id(0)*4);
	vstore4(M0, 0, frame + ind); ind += width_x8;
	vstore4(M1, 0, frame + ind); ind += width_x8;
	vstore4(M2, 0, frame + ind); ind += width_x8;
	vstore4(M3, 0, frame + ind); ind += width_x8;
	vstore4(M4, 0, frame + ind); ind += width_x8;
	vstore4(M5, 0, frame + ind); ind += width_x8;
	vstore4(M6, 0, frame + ind); ind += width_x8;
	vstore4(M7, 0, frame + ind);
		
	U2 = U1;
	U1 = U0;
	U0 = M0;
	M0 = A1;
	A1 = A0;
	
	buf = mad24(convert_int4(A1),-6,mad24(convert_int4(M0),123,mad24(convert_int4(U0),12,mad24(convert_int4(U1),-1,64))))/128;
	M1 = convert_uchar4_sat(buf);
	buf = mad24(convert_int4(A0),2,mad24(convert_int4(A1),-11,mad24(convert_int4(M0),108,mad24(convert_int4(U0),36,mad24(convert_int4(U1),-8, convert_int4(U2) + 64)))))/128;
	M2 = convert_uchar4_sat(buf);
	buf = mad24(convert_int4(A1),-9,mad24(convert_int4(M0),93,mad24(convert_int4(U0),50,mad24(convert_int4(U1),-6,64))))/128;
	M3 = convert_uchar4_sat(buf);
	buf = mad24(convert_int4(A0),3,mad24(convert_int4(A1),-16,mad24(convert_int4(M0),77,mad24(convert_int4(U0),77,mad24(convert_int4(U1),-16, mad24(convert_int4(U2),3,64))))))/128;
	M4 = convert_uchar4_sat(buf);
	buf = mad24(convert_int4(A1),-6,mad24(convert_int4(M0),50,mad24(convert_int4(U0),93,mad24(convert_int4(U1),-9,64))))/128;
	M5 = convert_uchar4_sat(buf);
	buf = (convert_int4(A0) + mad24(convert_int4(A1),-8,mad24(convert_int4(M0),36,mad24(convert_int4(U0),108,mad24(convert_int4(U1),-11, mad24(convert_int4(U2),2,64))))))/128;		
	M6 = convert_uchar4_sat(buf);
	buf = mad24(convert_int4(A1),-1,mad24(convert_int4(M0),12,mad24(convert_int4(U0),123,mad24(convert_int4(U1),-6,64))))/128;
	M7 = convert_uchar4_sat(buf);
		
	ind = (get_global_id(0)*4);
	vstore4(M0, 0, frame + ind); ind += width_x8;
	vstore4(M1, 0, frame + ind); ind += width_x8;
	vstore4(M2, 0, frame + ind); ind += width_x8;
	vstore4(M3, 0, frame + ind); ind += width_x8;
	vstore4(M4, 0, frame + ind); ind += width_x8;
	vstore4(M5, 0, frame + ind); ind += width_x8;
	vstore4(M6, 0, frame + ind); ind += width_x8;
	vstore4(M7, 0, frame + ind);

		
	return;	
}



