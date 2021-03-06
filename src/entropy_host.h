typedef cl_uchar Prob;
typedef cl_char tree_index;
typedef const tree_index Tree[];

typedef struct {
  int bits;
  int size;
} encoding_symbol;

//--------------------------------------------------------------------------------------------------------------------

typedef enum
{
	DC_PRED, /* predict DC using row above and column to the left */
	V_PRED, /* predict rows using row above */
	H_PRED, /* predict columns using column to the left */
	TM_PRED, /* propagate second differences a la "True Motion" */
	B_PRED, /* each Y subblock is independently predicted */
	num_uv_modes = B_PRED, /* first four modes apply to chroma */
	num_ymodes /* all modes apply to luma */
} intra_mbmode;
typedef enum
{
	B_DC_PRED, /* predict DC using row above and column to the left */
	B_TM_PRED, /* propagate second differences a la "True Motion" */
	B_VE_PRED, /* predict rows using row above */
	B_HE_PRED, /* predict columns using column to the left */
	B_LD_PRED, /* southwest (left and down) 45 degree diagonal prediction */
	B_RD_PRED, /* southeast (right and down) "" */
	B_VR_PRED, /* SSE (vertical right) diagonal prediction */
	B_VL_PRED, /* SSW (vertical left) "" */
	B_HD_PRED, /* ESE (horizontal down) "" */
	B_HU_PRED, /* ENE (horizontal up) "" */
	num_intra_bmodes
}
intra_bmode;
const tree_index mb_segment_tree [2 * (4-1)] = { 2, 4, /* root: "0", "1" subtrees */
												-0, -1, /* "00" = 0th value, "01" = 1st value */
												-2, -3 /* "10" = 2nd value, "11" = 3rd value */
												};
//const tree_index ymode_tree [2 * (num_ymodes - 1)] = {	-DC_PRED, 2, /* root: DC_PRED = "0", "1" subtree */
//														4, 6, /* "1" subtree has 2 descendant subtrees */
//														-V_PRED, -H_PRED, /* "10" subtree: V_PRED = "100", H_PRED = "101" */
//														-TM_PRED, -B_PRED /* "11" subtree: TM_PRED = "110", B_PRED = "111" */
//													 };
const tree_index kf_ymode_tree[2*(num_ymodes-1)] = {-B_PRED, 2, /* root: B_PRED = "0", "1" subtree */
													4, 6, /* "1" subtree has 2 descendant subtrees */
													-DC_PRED, -V_PRED, /* "10" subtree: DC_PRED = "100", V_PRED = "101" */
													-H_PRED, -TM_PRED /* "11" subtree: H_PRED = "110", TM_PRED = "111" */
												   };
const tree_index ymode_tree [2 * (num_ymodes - 1)] = {	-DC_PRED, 2, /* root: DC_PRED = "0", "1" subtree */
														4, 6, /* "1" subtree has 2 descendant subtrees */
														-V_PRED, -H_PRED, /* "10" subtree: V_PRED = "100",H_PRED = "101" */
														-TM_PRED, -B_PRED /* "11" subtree: TM_PRED = "110",B_PRED = "111" */
													 };
const tree_index uv_mode_tree [2 * (num_uv_modes - 1)] = {	-DC_PRED, 2, /* root: DC_PRED = "0", "1" subtree */
															-V_PRED, 4, /* "1" subtree: V_PRED = "10", "11" subtree */
															-H_PRED, -TM_PRED /* "11" subtree: H_PRED = "110", TM_PRED = "111" */
														 };
const tree_index bmode_tree [2 * (num_intra_bmodes - 1)] = {-B_DC_PRED, 2, /* B_DC_PRED = "0" */
															-B_TM_PRED, 4, /* B_TM_PRED = "10" */
															-B_VE_PRED, 6, /* B_VE_PRED = "110" */
															8, 12,
															-B_HE_PRED, 10, /* B_HE_PRED = "11100" */
															-B_RD_PRED, -B_VR_PRED, /* B_RD_PRED = "111010", B_VR_PRED = "111011" */
															-B_LD_PRED, 14, /* B_LD_PRED = "111110" */
															-B_VL_PRED, 16, /* B_VL_PRED = "1111110" */
															-B_HD_PRED, -B_HU_PRED /* HD = "11111110", HU = "11111111" */
														   };
Prob new_segment_prob[4] = { 128, 128, 128, 128 };
const Prob kf_ymode_prob [num_ymodes - 1] = { 145, 156, 163, 128};
const Prob ymode_prob [num_ymodes - 1] = { 112, 86, 140, 37}; //default
const Prob B_ymode_prob [num_ymodes - 1] = { 0, 0, 0, 0}; //adapted fo B_PRED = "111"
const Prob kf_uv_mode_prob [num_uv_modes - 1] = { 142, 114, 183};
const Prob uv_mode_prob [num_uv_modes - 1] = { 162, 101, 204}; // default
const Prob TM_uv_mode_prob [num_uv_modes - 1] = { 0, 0, 0}; // adapted for TM_PRED = "111"
const Prob kf_bmode_prob [num_intra_bmodes][num_intra_bmodes][num_intra_bmodes-1] =
{
	{
		{ 231, 120, 48, 89, 115, 113, 120, 152, 112},
		{ 152, 179, 64, 126, 170, 118, 46, 70, 95},
		{ 175, 69, 143, 80, 85, 82, 72, 155, 103},
		{ 56, 58, 10, 171, 218, 189, 17, 13, 152},
		{ 144, 71, 10, 38, 171, 213, 144, 34, 26},
		{ 114, 26, 17, 163, 44, 195, 21, 10, 173},
		{ 121, 24, 80, 195, 26, 62, 44, 64, 85},
		{ 170, 46, 55, 19, 136, 160, 33, 206, 71},
		{ 63, 20, 8, 114, 114, 208, 12, 9, 226},
		{ 81, 40, 11, 96, 182, 84, 29, 16, 36}
	},
	{
		{ 134, 183, 89, 137, 98, 101, 106, 165, 148},
		{ 72, 187, 100, 130, 157, 111, 32, 75, 80},
		{ 66, 102, 167, 99, 74, 62, 40, 234, 128},
		{ 41, 53, 9, 178, 241, 141, 26, 8, 107},
		{ 104, 79, 12, 27, 217, 255, 87, 17, 7},
		{ 74, 43, 26, 146, 73, 166, 49, 23, 157},
		{ 65, 38, 105, 160, 51, 52, 31, 115, 128},
		{ 87, 68, 71, 44, 114, 51, 15, 186, 23},
		{ 47, 41, 14, 110, 182, 183, 21, 17, 194},
		{ 66, 45, 25, 102, 197, 189, 23, 18, 22}
	},
	{
		{ 88, 88, 147, 150, 42, 46, 45, 196, 205},
		{ 43, 97, 183, 117, 85, 38, 35, 179, 61},
		{ 39, 53, 200, 87, 26, 21, 43, 232, 171},
		{ 56, 34, 51, 104, 114, 102, 29, 93, 77},
		{ 107, 54, 32, 26, 51, 1, 81, 43, 31},
		{ 39, 28, 85, 171, 58, 165, 90, 98, 64},
		{ 34, 22, 116, 206, 23, 34, 43, 166, 73},
		{ 68, 25, 106, 22, 64, 171, 36, 225, 114},
		{ 34, 19, 21, 102, 132, 188, 16, 76, 124},
		{ 62, 18, 78, 95, 85, 57, 50, 48, 51}
	},
	{
		{ 193, 101, 35, 159, 215, 111, 89, 46, 111},
		{ 60, 148, 31, 172, 219, 228, 21, 18, 111},
		{ 112, 113, 77, 85, 179, 255, 38, 120, 114},
		{ 40, 42, 1, 196, 245, 209, 10, 25, 109},
		{ 100, 80, 8, 43, 154, 1, 51, 26, 71},
		{ 88, 43, 29, 140, 166, 213, 37, 43, 154},
		{ 61, 63, 30, 155, 67, 45, 68, 1, 209},
		{ 142, 78, 78, 16, 255, 128, 34, 197, 171},
		{ 41, 40, 5, 102, 211, 183, 4, 1, 221},
		{ 51, 50, 17, 168, 209, 192, 23, 25, 82}
	},
	{
		{ 125, 98, 42, 88, 104, 85, 117, 175, 82},
		{ 95, 84, 53, 89, 128, 100, 113, 101, 45},
		{ 75, 79, 123, 47, 51, 128, 81, 171, 1},
		{ 57, 17, 5, 71, 102, 57, 53, 41, 49},
		{ 115, 21, 2, 10, 102, 255, 166, 23, 6},
		{ 38, 33, 13, 121, 57, 73, 26, 1, 85},
		{ 41, 10, 67, 138, 77, 110, 90, 47, 114},
		{ 101, 29, 16, 10, 85, 128, 101, 196, 26},
		{ 57, 18, 10, 102, 102, 213, 34, 20, 43},
		{ 117, 20, 15, 36, 163, 128, 68, 1, 26}
	},
	{
		{ 138, 31, 36, 171, 27, 166, 38, 44, 229},
		{ 67, 87, 58, 169, 82, 115, 26, 59, 179},
		{ 63, 59, 90, 180, 59, 166, 93, 73, 154},
		{ 40, 40, 21, 116, 143, 209, 34, 39, 175},
		{ 57, 46, 22, 24, 128, 1, 54, 17, 37},
		{ 47, 15, 16, 183, 34, 223, 49, 45, 183},
		{ 46, 17, 33, 183, 6, 98, 15, 32, 183},
		{ 65, 32, 73, 115, 28, 128, 23, 128, 205},
		{ 40, 3, 9, 115, 51, 192, 18, 6, 223},
		{ 87, 37, 9, 115, 59, 77, 64, 21, 47}
	},
	{
		{ 104, 55, 44, 218, 9, 54, 53, 130, 226},
		{ 64, 90, 70, 205, 40, 41, 23, 26, 57},
		{ 54, 57, 112, 184, 5, 41, 38, 166, 213},
		{ 30, 34, 26, 133, 152, 116, 10, 32, 134},
		{ 75, 32, 12, 51, 192, 255, 160, 43, 51},
		{ 39, 19, 53, 221, 26, 114, 32, 73, 255},
		{ 31, 9, 65, 234, 2, 15, 1, 118, 73},
		{ 88, 31, 35, 67, 102, 85, 55, 186, 85},
		{ 56, 21, 23, 111, 59, 205, 45, 37, 192},
		{ 55, 38, 70, 124, 73, 102, 1, 34, 98}
	},
	{
		{ 102, 61, 71, 37, 34, 53, 31, 243, 192},
		{ 69, 60, 71, 38, 73, 119, 28, 222, 37},
		{ 68, 45, 128, 34, 1, 47, 11, 245, 171},
		{ 62, 17, 19, 70, 146, 85, 55, 62, 70},
		{ 75, 15, 9, 9, 64, 255, 184, 119, 16},
		{ 37, 43, 37, 154, 100, 163, 85, 160, 1},
		{ 63, 9, 92, 136, 28, 64, 32, 201, 85},
		{ 86, 6, 28, 5, 64, 255, 25, 248, 1},
		{ 56, 8, 17, 132, 137, 255, 55, 116, 128},
		{ 58, 15, 20, 82, 135, 57, 26, 121, 40}
	},
	{
		{ 164, 50, 31, 137, 154, 133, 25, 35, 218},
		{ 51, 103, 44, 131, 131, 123, 31, 6, 158},
		{ 86, 40, 64, 135, 148, 224, 45, 183, 128},
		{ 22, 26, 17, 131, 240, 154, 14, 1, 209},
		{ 83, 12, 13, 54, 192, 255, 68, 47, 28},
		{ 45, 16, 21, 91, 64, 222, 7, 1, 197},
		{ 56, 21, 39, 155, 60, 138, 23, 102, 213},
		{ 85, 26, 85, 85, 128, 128, 32, 146, 171},
		{ 18, 11, 7, 63, 144, 171, 4, 4, 246},
		{ 35, 27, 10, 146, 174, 171, 12, 26, 128}
	},
	{
		{ 190, 80, 35, 99, 180, 80, 126, 54, 45},
		{ 85, 126, 47, 87, 176, 51, 41, 20, 32},
		{ 101, 75, 128, 139, 118, 146, 116, 128, 85},
		{ 56, 41, 15, 176, 236, 85, 37, 9, 62},
		{ 146, 36, 19, 30, 171, 255, 97, 27, 20},
		{ 71, 30, 17, 119, 118, 255, 17, 18, 138},
		{ 101, 38, 60, 138, 55, 70, 43, 26, 142},
		{ 138, 45, 61, 62, 219, 1, 81, 188, 64},
		{ 32, 41, 20, 117, 151, 142, 20, 21, 163},
		{ 112, 19, 12, 61, 195, 128, 48, 4, 24}
	}
};
const Prob bmode_prob [num_intra_bmodes - 1] = { 120, 90, 79, 133, 87, 85, 80, 111, 151 };



typedef enum // MV modes for whole macroblock vectors
{
	NEARESTMV = num_ymodes, /* use "nearest" motion vector for entire MB */
	NEARMV, /* use "next nearest" "" */
	ZEROMV, /* use zero "" */
	NEWMV, /* use explicit offset from implicit "" */
	SPLITMV, /* use multiple motion vectors */
	num_mv_refs = SPLITMV + 1 - NEARESTMV
} mv_ref;
const tree_index mv_ref_tree [2 * (num_mv_refs - 1)] =
{
	-ZEROMV, 2, /* zero = "0" */
	-NEARESTMV, 4, /* nearest = "10" */
	-NEARMV, 6, /* near = "110" */
	-NEWMV, -SPLITMV /* new = "1110", split = "1111" */
};
const int vp8_mode_contexts[6][4] = {	{ 7, 1, 1, 143, },
										{ 14, 18, 14, 107, },
										{ 135, 64, 57, 68, },
										{ 60, 56, 128, 65, },
										{ 159, 134, 128, 34, },
										{ 234, 188, 128, 28, },
									};



typedef enum
{
	MV_TOP_BOTTOM, /* two pieces {0...7} and {8...15} */
	MV_LEFT_RIGHT, /* {0,1,4,5,8,9,12,13} and {2,3,6,7,10,11,14,15} */
	MV_QUARTERS, /* {0,1,4,5}, {2,3,6,7}, {8,9,12,13}, {10,11,14,15} */
	MV_16, /* every subblock gets its own vector {0} ... {15} */ //we have only this one
	mv_num_partitions
} MVpartition;
const tree_index split_mv_tree[2 * (mv_num_partitions - 1)] =
{
	-MV_16, 2, /* MV_16 = "0" */
	-MV_QUARTERS, 4, /* mv_quarters = "10" */
	-MV_TOP_BOTTOM, -MV_LEFT_RIGHT /* top_bottom = "110", left_right = "111" */
};
static const unsigned char split_mv_probs[3] = { 110, 111, 150};

typedef enum
{
	LEFT4x4 = num_intra_bmodes, /* use already-coded MV to my left */
	ABOVE4x4, /* use already-coded MV above me */
	ZERO4x4, /* use zero MV */
	NEW4x4, /* explicit offset from "best" */
	num_sub_mv_ref
} sub_mv_ref;
const tree_index submv_ref_tree [2 * (num_sub_mv_ref - 1)] = {	-LEFT4x4, 2, /* LEFT = "0" */
																-ABOVE4x4, 4, /* ABOVE = "10" */
																-ZERO4x4, -NEW4x4 /* ZERO = "110", NEW = "111" */
															  };
static const unsigned char submv_ref_probs2[5][3] = {	{ 147, 136, 18 },
														{ 106, 145, 1 },
														{ 179, 121, 1 },
														{ 223, 1, 34 },
														{ 208, 1, 1 }
													};



typedef enum
{
	mvpis_short, /* short (<= 7) vs long (>= 8) */
	MVPsign, /* sign for non-zero */
	MVPshort, /* 8 short values = 7-position tree */
	MVPbits = MVPshort + 7, /* 8 long value bits w/independent probs */
	MVPcount = MVPbits + 10 /* 19 probabilities in total */
} MVPindices;
const Prob default_mv_context[2][MVPcount] ={{ // row
											162, // is short
											128, // sign
											225, 146, 172, 147, 214, 39, 156, // short tree
											128, 129, 132, 75, 145, 178, 206, 239, 254, 254 // long bits
											},
											{ // same for column
											164, // is short
											128,
											204, 170, 119, 235, 140, 230, 228,
											128, 130, 130, 74, 148, 180, 203, 236, 254, 254 // long bits
											}};
Prob new_mv_context[2][MVPcount];
cl_uint num_mv_context[2][MVPcount];
cl_uint denom_mv_context[2][MVPcount];
const tree_index small_mvtree [2 * (8 - 1)] = {	2, 8, /* "0" subtree, "1" subtree */
												4, 6, /* "00" subtree, "01" subtree */
												-0, -1, /* 0 = "000", 1 = "001" */
												-2, -3, /* 2 = "010", 3 = "011" */
												10, 12, /* "10" subtree, "11" subtree */
												-4, -5, /* 4 = "100", 5 = "101" */
												-6, -7 /* 6 = "110", 7 = "111" */
											  };



const Prob coeff_update_probs [4] [8] [3] [11] =
{
	{
		{
			{ 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255},
			{ 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255},
			{ 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255}
		},
		{
			{ 176, 246, 255, 255, 255, 255, 255, 255, 255, 255, 255},
			{ 223, 241, 252, 255, 255, 255, 255, 255, 255, 255, 255},
			{ 249, 253, 253, 255, 255, 255, 255, 255, 255, 255, 255}
		},
		{
			{ 255, 244, 252, 255, 255, 255, 255, 255, 255, 255, 255},
			{ 234, 254, 254, 255, 255, 255, 255, 255, 255, 255, 255},
			{ 253, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255}
		},
		{
			{ 255, 246, 254, 255, 255, 255, 255, 255, 255, 255, 255},
			{ 239, 253, 254, 255, 255, 255, 255, 255, 255, 255, 255},
			{ 254, 255, 254, 255, 255, 255, 255, 255, 255, 255, 255}
		},
		{
			{ 255, 248, 254, 255, 255, 255, 255, 255, 255, 255, 255},
			{ 251, 255, 254, 255, 255, 255, 255, 255, 255, 255, 255},
			{ 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255}
		},
		{
			{ 255, 253, 254, 255, 255, 255, 255, 255, 255, 255, 255},
			{ 251, 254, 254, 255, 255, 255, 255, 255, 255, 255, 255},
			{ 254, 255, 254, 255, 255, 255, 255, 255, 255, 255, 255}
		},
		{
			{ 255, 254, 253, 255, 254, 255, 255, 255, 255, 255, 255},
			{ 250, 255, 254, 255, 254, 255, 255, 255, 255, 255, 255},
			{ 254, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255}
		},
		{
			{ 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255},
			{ 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255},
			{ 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255}
		}
	},
	{
		{
			{ 217, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255},
			{ 225, 252, 241, 253, 255, 255, 254, 255, 255, 255, 255},
			{ 234, 250, 241, 250, 253, 255, 253, 254, 255, 255, 255}
		},
		{
			{ 255, 254, 255, 255, 255, 255, 255, 255, 255, 255, 255},
			{ 223, 254, 254, 255, 255, 255, 255, 255, 255, 255, 255},
			{ 238, 253, 254, 254, 255, 255, 255, 255, 255, 255, 255}
		},
		{
			{ 255, 248, 254, 255, 255, 255, 255, 255, 255, 255, 255},
			{ 249, 254, 255, 255, 255, 255, 255, 255, 255, 255, 255},
			{ 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255}
		},
		{
			{ 255, 253, 255, 255, 255, 255, 255, 255, 255, 255, 255},
			{ 247, 254, 255, 255, 255, 255, 255, 255, 255, 255, 255},
			{ 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255}
		},
		{
			{ 255, 253, 254, 255, 255, 255, 255, 255, 255, 255, 255},
			{ 252, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255},
			{ 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255}
		},
		{
			{ 255, 254, 254, 255, 255, 255, 255, 255, 255, 255, 255},
			{ 253, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255},
			{ 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255}
		},
		{
			{ 255, 254, 253, 255, 255, 255, 255, 255, 255, 255, 255},
			{ 250, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255},
			{ 254, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255}
		},
		{
			{ 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255},
			{ 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255},
			{ 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255}
		}
	},
	{
		{
			{ 186, 251, 250, 255, 255, 255, 255, 255, 255, 255, 255},
			{ 234, 251, 244, 254, 255, 255, 255, 255, 255, 255, 255},
			{ 251, 251, 243, 253, 254, 255, 254, 255, 255, 255, 255}
		},
		{
			{ 255, 253, 254, 255, 255, 255, 255, 255, 255, 255, 255},
			{ 236, 253, 254, 255, 255, 255, 255, 255, 255, 255, 255},
			{ 251, 253, 253, 254, 254, 255, 255, 255, 255, 255, 255}
		},
		{
			{ 255, 254, 254, 255, 255, 255, 255, 255, 255, 255, 255},
			{ 254, 254, 254, 255, 255, 255, 255, 255, 255, 255, 255},
			{ 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255}
		},
		{
			{ 255, 254, 255, 255, 255, 255, 255, 255, 255, 255, 255},
			{ 254, 254, 255, 255, 255, 255, 255, 255, 255, 255, 255},
			{ 254, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255}
		},
		{
			{ 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255},
			{ 254, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255},
			{ 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255}
		},
		{
			{ 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255},
			{ 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255},
			{ 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255}
		},
		{
			{ 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255},
			{ 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255},
			{ 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255}
		},
		{
			{ 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255},
			{ 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255},
			{ 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255}
		}
	},
	{
		{
			{ 248, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255},
			{ 250, 254, 252, 254, 255, 255, 255, 255, 255, 255, 255},
			{ 248, 254, 249, 253, 255, 255, 255, 255, 255, 255, 255}
		},
		{
			{ 255, 253, 253, 255, 255, 255, 255, 255, 255, 255, 255},
			{ 246, 253, 253, 255, 255, 255, 255, 255, 255, 255, 255},
			{ 252, 254, 251, 254, 254, 255, 255, 255, 255, 255, 255}
		},
		{
			{ 255, 254, 252, 255, 255, 255, 255, 255, 255, 255, 255},
			{ 248, 254, 253, 255, 255, 255, 255, 255, 255, 255, 255},
			{ 253, 255, 254, 254, 255, 255, 255, 255, 255, 255, 255}
		},
		{
			{ 255, 251, 254, 255, 255, 255, 255, 255, 255, 255, 255},
			{ 245, 251, 254, 255, 255, 255, 255, 255, 255, 255, 255},
			{ 253, 253, 254, 255, 255, 255, 255, 255, 255, 255, 255}
		},
		{
			{ 255, 251, 253, 255, 255, 255, 255, 255, 255, 255, 255},
			{ 252, 253, 254, 255, 255, 255, 255, 255, 255, 255, 255},
			{ 255, 254, 255, 255, 255, 255, 255, 255, 255, 255, 255}
		},
		{
			{ 255, 252, 255, 255, 255, 255, 255, 255, 255, 255, 255},
			{ 249, 255, 254, 255, 255, 255, 255, 255, 255, 255, 255},
			{ 255, 255, 254, 255, 255, 255, 255, 255, 255, 255, 255}
		},
		{
			{ 255, 255, 253, 255, 255, 255, 255, 255, 255, 255, 255},
			{ 250, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255},
			{ 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255}
		},
		{
			{ 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255},
			{ 254, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255},
			{ 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255}
		}
	}
};

const Prob vp8_mv_update_probs[2][19] =
{
    {
        237,
        246,
        253, 253, 254, 254, 254, 254, 254,
        254, 254, 254, 254, 254, 250, 250, 252, 254, 254
    },
    {
        231,
        243,
        245, 253, 254, 254, 254, 254, 254,
        254, 254, 254, 254, 254, 251, 251, 254, 254, 254
    }
};
