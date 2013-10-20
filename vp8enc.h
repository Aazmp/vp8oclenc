#include <CL\cl.h>
#include <conio.h>
#include <stdio.h>
#include <io.h>
#include <fcntl.h>
#include <stdlib.h>
#include <memory.h>
#include <time.h>
#include "stdint1.h"

#define QUANT_TO_FILTER_LEVEL 3

static const uint8_t vp8_dc_qlookup[128] =
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

static const int16_t vp8_ac_qlookup[128] =
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

//static const int32_t zigzag[16] = { 0, 1, 4, 8, 5, 2, 3,  6, 9, 12, 13, 10, 7, 11, 14, 15 };
//static const int32_t inv_zigzag[16] = { 0, 1, 5, 6, 2, 4, 7, 12, 3,  8, 11, 13, 9, 10, 14, 15 };
// not only inv zigzag is inverse for zigzag, but
// A[i] = B[zigzag[i]] === A[inv_zigzag[i]] = B[i]


#define ERRORPATH "clErrors.txt"
#define CPUPATH "CPU_kernels.cl"
#define GPUPATH "GPU_kernels.cl"

union mv {
	uint32_t raw;
	struct {
		int16_t x, y;
	} d;
};

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

typedef struct { //in future resize to short or chars!!!
    int16_t coeffs[25][16];
    int32_t vector_x[4];
    int32_t vector_y[4];
	float SSIM;
	int32_t non_zero_coeffs;
	int32_t parts; //16x16 == 0; 8x8 == 1;
	int32_t reference_frame;
} macroblock;

typedef struct {
	int32_t vector_x;
	int32_t vector_y;
} vector_net;
typedef struct 
{
	union mv base_mv;
	int32_t is_inter_mb;
	int32_t parts;
	int32_t mode[16];
} macroblock_extra_data;

struct deviceContext
{
    cl_context context_gpu;
    cl_context context_cpu;
    cl_platform_id *platforms;
    cl_device_id *device_cpu;
    cl_device_id *device_gpu;
    cl_program program_cpu;
    cl_program program_gpu;
    cl_command_queue commandQueue_cpu;
    cl_command_queue commandQueue_gpu;
    cl_int state_cpu;
    cl_int state_gpu;
	cl_kernel reset_vectors;
	cl_kernel luma_search_1step;
	cl_kernel luma_search_2step;
	cl_kernel downsample;
	cl_kernel try_golden_reference;
	cl_kernel luma_transform_8x8;
	cl_kernel luma_transform_16x16;
	cl_kernel chroma_transform;
	cl_kernel encode_coefficients;
	cl_kernel count_probs;
	cl_kernel num_div_denom;
	cl_kernel luma_interpolate_Hx4;
	cl_kernel luma_interpolate_Vx4;
	cl_kernel chroma_interpolate_Hx8;
	cl_kernel chroma_interpolate_Vx8;
	cl_kernel normal_loop_filter_MBH;
	cl_kernel normal_loop_filter_MBV;
	cl_kernel count_SSIM;
	cl_kernel prepare_filter_mask;
    /* add kernels */

    // these are frame data padded to be devisible by 16 and converted to normalized int16
    cl_mem current_frame_Y;
	cl_mem current_frame_Y_downsampled_by2;
	cl_mem current_frame_Y_downsampled_by4;
	cl_mem current_frame_Y_downsampled_by8;
	cl_mem current_frame_Y_downsampled_by16;
    cl_mem current_frame_U;
    cl_mem current_frame_V;
    cl_mem last_frame_Y_interpolated;
	//instead of original size we use reconstructed frame
	cl_mem last_frame_Y_downsampled_by2;
	cl_mem last_frame_Y_downsampled_by4;
	cl_mem last_frame_Y_downsampled_by8;
	cl_mem last_frame_Y_downsampled_by16;
    cl_mem last_frame_U; // both
    cl_mem last_frame_V; // interpolated
    cl_mem reconstructed_frame_Y;
    cl_mem reconstructed_frame_U;
    cl_mem reconstructed_frame_V;
	cl_mem golden_frame_Y;
    cl_mem golden_frame_U;
    cl_mem golden_frame_V;
	cl_mem vnet1; //vector
	cl_mem vnet2; //nets
	cl_mem mb_metrics;
	cl_mem mb_mask;
	cl_mem third_context;
	cl_mem coeff_probs;
	cl_mem coeff_probs_denom;

    cl_mem transformed_blocks_cpu;
	cl_mem transformed_blocks_gpu;
    cl_mem partitions;
	cl_mem partitions_sizes;

	size_t gpu_work_items_limit;
	size_t gpu_work_items_per_dim[1];
	size_t gpu_work_group_size_per_dim[1];
	size_t cpu_work_items_per_dim[1];
	size_t cpu_work_group_size_per_dim[1];

};

struct videoContext
{
    // size of input frame
    int32_t src_width;
    int32_t src_height;
    int32_t src_frame_size_luma;
    int32_t src_frame_size_chroma;
    // size of output frame
    int32_t dst_width;
    int32_t dst_height;
    int32_t dst_frame_size_luma;
    int32_t dst_frame_size_chroma;
    // size of padded frame
    int32_t wrk_width;
    int32_t wrk_height;
    int32_t wrk_frame_size_luma;
    int32_t wrk_frame_size_chroma;

    int32_t mb_width;
    int32_t mb_height;
    int32_t mb_count;
    int32_t GOP_size;

    int32_t quantizer_index_y_dc_i;
    int32_t quantizer_index_y_ac_i;
    int32_t quantizer_index_y2_dc_i;
    int32_t quantizer_index_y2_ac_i;
    int32_t quantizer_index_uv_dc_i;
    int32_t quantizer_index_uv_ac_i;

	int32_t quantizer_index_y_dc_p;
    int32_t quantizer_index_y_ac_p;
    int32_t quantizer_index_y2_dc_p;
    int32_t quantizer_index_y2_ac_p;
    int32_t quantizer_index_uv_dc_p;
    int32_t quantizer_index_uv_ac_p;

	int32_t quantizer_index_y_dc_p_c;
    int32_t quantizer_index_y_ac_p_c;
    int32_t quantizer_index_y2_dc_p_c;
    int32_t quantizer_index_y2_ac_p_c;
    int32_t quantizer_index_uv_dc_p_c;
    int32_t quantizer_index_uv_ac_p_c;

	int32_t quantizer_index_y_dc_prev;
    int32_t quantizer_index_y_ac_prev;
    int32_t quantizer_index_y2_dc_prev;
    int32_t quantizer_index_y2_ac_prev;
    int32_t quantizer_index_uv_dc_prev;
    int32_t quantizer_index_uv_ac_prev;

    int32_t quantizer_y_dc_i;
    int32_t quantizer_y_ac_i;
    int32_t quantizer_y2_dc_i;
    int32_t quantizer_y2_ac_i;
    int32_t quantizer_uv_dc_i;
    int32_t quantizer_uv_ac_i;

	int32_t quantizer_y_dc_p;
    int32_t quantizer_y_ac_p;
    int32_t quantizer_y2_dc_p;
    int32_t quantizer_y2_ac_p;
    int32_t quantizer_uv_dc_p;
    int32_t quantizer_uv_ac_p; 

	int32_t loop_filter_type;
	int32_t loop_filter_level;
	int32_t loop_filter_sharpness;
	int32_t mbedge_limit;
	int32_t sub_bedge_limit;
	int32_t interior_limit;
	int32_t hev_threshold;

	int32_t number_of_partitions;
	int32_t partition_step;

	uint32_t timestep;
	uint32_t timescale;
	uint32_t framerate;

};

struct hostFrameBuffers
{
    int32_t frame_number;
	int32_t frames_until_key;
	int32_t replaced;
    int32_t input_pack_size;
    uint8_t *input_pack; // allbytes for YUV in one
    uint8_t *current_Y;
    uint8_t *current_U;
    uint8_t *current_V;
    uint8_t *tmp_Y;
    uint8_t *tmp_U;
    uint8_t *tmp_V;
    uint8_t *reconstructed_Y;
    uint8_t *reconstructed_U;
    uint8_t *reconstructed_V;
    uint8_t *last_U;
    uint8_t *last_V;
    macroblock *transformed_blocks;
	macroblock_extra_data *e_data;
    uint8_t *encoded_frame;
	uint32_t encoded_frame_size;
    uint8_t *current_frame_pos_in_pack;
    int32_t current_is_key_frame;
	int32_t prev_is_key_frame;
	int32_t skip_prob;
	float new_SSIM;
	
	int32_t partition_sizes[8];
	uint8_t *partitions;
	uint8_t *partition_0;
	int32_t partition_0_size;

	uint32_t new_probs[4][8][3][11];
	uint32_t new_probs_denom[4][8][3][11];
};

struct fileContext
{
    FILE * handle;
    char * path;
    int cur_pos;
};

struct times 
{
	clock_t init,
			start,
			read, 
			write,
			count_probs,
			bool_encode_header, 
			bool_encode_coeffs,
			inter_transform, 
			intra_transform,
			loop_filter,
			interpolate,
			all;
};



static const unsigned char k_default_coeff_probs [4][8][3][11] =
{
	{ /* block type 0 */
		{ /* coeff band 0 */
			{ 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128},
			{ 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128},
			{ 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128}
		},
		{ /* coeff band 1 */
			{ 253, 136, 254, 255, 228, 219, 128, 128, 128, 128, 128},
			{ 189, 129, 242, 255, 227, 213, 255, 219, 128, 128, 128},
			{ 106, 126, 227, 252, 214, 209, 255, 255, 128, 128, 128}
		},
		{ /* coeff band 2 */
			{ 1, 98, 248, 255, 236, 226, 255, 255, 128, 128, 128},
			{ 181, 133, 238, 254, 221, 234, 255, 154, 128, 128, 128},
			{ 78, 134, 202, 247, 198, 180, 255, 219, 128, 128, 128}
		},
		{ /* coeff band 3 */
			{ 1, 185, 249, 255, 243, 255, 128, 128, 128, 128, 128},
			{ 184, 150, 247, 255, 236, 224, 128, 128, 128, 128, 128},
			{ 77, 110, 216, 255, 236, 230, 128, 128, 128, 128, 128}
		},
		{ /* coeff band 4 */
			{ 1, 101, 251, 255, 241, 255, 128, 128, 128, 128, 128},
			{ 170, 139, 241, 252, 236, 209, 255, 255, 128, 128, 128},
			{ 37, 116, 196, 243, 228, 255, 255, 255, 128, 128, 128}
		},
		{ /* coeff band 5 */
			{ 1, 204, 254, 255, 245, 255, 128, 128, 128, 128, 128},
			{ 207, 160, 250, 255, 238, 128, 128, 128, 128, 128, 128},
			{ 102, 103, 231, 255, 211, 171, 128, 128, 128, 128, 128}
		},
		{ /* coeff band 6 */
			{ 1, 152, 252, 255, 240, 255, 128, 128, 128, 128, 128},
			{ 177, 135, 243, 255, 234, 225, 128, 128, 128, 128, 128},
			{ 80, 129, 211, 255, 194, 224, 128, 128, 128, 128, 128}
		},
		{ /* coeff band 7 */
			{ 1, 1, 255, 128, 128, 128, 128, 128, 128, 128, 128},
			{ 246, 1, 255, 128, 128, 128, 128, 128, 128, 128, 128},
			{ 255, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128}
		}
		},
	{ /* block type 1 */
		{ /* coeff band 0 */
			{ 198, 35, 237, 223, 193, 187, 162, 160, 145, 155, 62},
			{ 131, 45, 198, 221, 172, 176, 220, 157, 252, 221, 1},
			{ 68, 47, 146, 208, 149, 167, 221, 162, 255, 223, 128}
		},
		{ /* coeff band 1 */
			{ 1, 149, 241, 255, 221, 224, 255, 255, 128, 128, 128},
			{ 184, 141, 234, 253, 222, 220, 255, 199, 128, 128, 128},
			{ 81, 99, 181, 242, 176, 190, 249, 202, 255, 255, 128}
		},
		{ /* coeff band 2 */
			{ 1, 129, 232, 253, 214, 197, 242, 196, 255, 255, 128},
			{ 99, 121, 210, 250, 201, 198, 255, 202, 128, 128, 128},
			{ 23, 91, 163, 242, 170, 187, 247, 210, 255, 255, 128}
		},
		{ /* coeff band 3 */
			{ 1, 200, 246, 255, 234, 255, 128, 128, 128, 128, 128},
			{ 109, 178, 241, 255, 231, 245, 255, 255, 128, 128, 128},
			{ 44, 130, 201, 253, 205, 192, 255, 255, 128, 128, 128}
		},
		{ /* coeff band 4 */
			{ 1, 132, 239, 251, 219, 209, 255, 165, 128, 128, 128},
			{ 94, 136, 225, 251, 218, 190, 255, 255, 128, 128, 128},
			{ 22, 100, 174, 245, 186, 161, 255, 199, 128, 128, 128}
		},
		{ /* coeff band 5 */
			{ 1, 182, 249, 255, 232, 235, 128, 128, 128, 128, 128},
			{ 124, 143, 241, 255, 227, 234, 128, 128, 128, 128, 128},
			{ 35, 77, 181, 251, 193, 211, 255, 205, 128, 128, 128}
		},
		{ /* coeff band 6 */
			{ 1, 157, 247, 255, 236, 231, 255, 255, 128, 128, 128},
			{ 121, 141, 235, 255, 225, 227, 255, 255, 128, 128, 128},
			{ 45, 99, 188, 251, 195, 217, 255, 224, 128, 128, 128}
		},
		{ /* coeff band 7 */
			{ 1, 1, 251, 255, 213, 255, 128, 128, 128, 128, 128},
			{ 203, 1, 248, 255, 255, 128, 128, 128, 128, 128, 128},
			{ 137, 1, 177, 255, 224, 255, 128, 128, 128, 128, 128}
		}
		},
	{ /* block type 2 */
		{ /* coeff band 0 */
			{ 253, 9, 248, 251, 207, 208, 255, 192, 128, 128, 128},
			{ 175, 13, 224, 243, 193, 185, 249, 198, 255, 255, 128},
			{ 73, 17, 171, 221, 161, 179, 236, 167, 255, 234, 128}
		},
		{ /* coeff band 1 */
			{ 1, 95, 247, 253, 212, 183, 255, 255, 128, 128, 128},
			{ 239, 90, 244, 250, 211, 209, 255, 255, 128, 128, 128},
			{ 155, 77, 195, 248, 188, 195, 255, 255, 128, 128, 128}
		},
		{ /* coeff band 2 */
			{ 1, 24, 239, 251, 218, 219, 255, 205, 128, 128, 128},
			{ 201, 51, 219, 255, 196, 186, 128, 128, 128, 128, 128},
			{ 69, 46, 190, 239, 201, 218, 255, 228, 128, 128, 128}
		},
		{ /* coeff band 3 */
			{ 1, 191, 251, 255, 255, 128, 128, 128, 128, 128, 128},
			{ 223, 165, 249, 255, 213, 255, 128, 128, 128, 128, 128},
			{ 141, 124, 248, 255, 255, 128, 128, 128, 128, 128, 128}
		},
		{ /* coeff band 4 */
			{ 1, 16, 248, 255, 255, 128, 128, 128, 128, 128, 128},
			{ 190, 36, 230, 255, 236, 255, 128, 128, 128, 128, 128},
			{ 149, 1, 255, 128, 128, 128, 128, 128, 128, 128, 128}
		},
		{ /* coeff band 5 */
			{ 1, 226, 255, 128, 128, 128, 128, 128, 128, 128, 128},
			{ 247, 192, 255, 128, 128, 128, 128, 128, 128, 128, 128},
			{ 240, 128, 255, 128, 128, 128, 128, 128, 128, 128, 128}
		},
		{ /* coeff band 6 */
			{ 1, 134, 252, 255, 255, 128, 128, 128, 128, 128, 128},
			{ 213, 62, 250, 255, 255, 128, 128, 128, 128, 128, 128},
			{ 55, 93, 255, 128, 128, 128, 128, 128, 128, 128, 128}
		},
		{ /* coeff band 7 */
			{ 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128},
			{ 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128},
			{ 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128}
		}
		},
	{ /* block type 3 */
		{ /* coeff band 0 */
			{ 202, 24, 213, 235, 186, 191, 220, 160, 240, 175, 255},
			{ 126, 38, 182, 232, 169, 184, 228, 174, 255, 187, 128},
			{ 61, 46, 138, 219, 151, 178, 240, 170, 255, 216, 128}
		},
		{ /* coeff band 1 */
			{ 1, 112, 230, 250, 199, 191, 247, 159, 255, 255, 128},
			{ 166, 109, 228, 252, 211, 215, 255, 174, 128, 128, 128},
			{ 39, 77, 162, 232, 172, 180, 245, 178, 255, 255, 128}
		},
		{ /* coeff band 2 */
			{ 1, 52, 220, 246, 198, 199, 249, 220, 255, 255, 128},
			{ 124, 74, 191, 243, 183, 193, 250, 221, 255, 255, 128},
			{ 24, 71, 130, 219, 154, 170, 243, 182, 255, 255, 128}
		},
		{ /* coeff band 3 */
			{ 1, 182, 225, 249, 219, 240, 255, 224, 128, 128, 128},
			{ 149, 150, 226, 252, 216, 205, 255, 171, 128, 128, 128},
			{ 28, 108, 170, 242, 183, 194, 254, 223, 255, 255, 128}
		},
		{ /* coeff band 4 */
			{ 1, 81, 230, 252, 204, 203, 255, 192, 128, 128, 128},
			{ 123, 102, 209, 247, 188, 196, 255, 233, 128, 128, 128},
			{ 20, 95, 153, 243, 164, 173, 255, 203, 128, 128, 128}
		},
		{ /* coeff band 5 */
			{ 1, 222, 248, 255, 216, 213, 128, 128, 128, 128, 128},
			{ 168, 175, 246, 252, 235, 205, 255, 255, 128, 128, 128},
			{ 47, 116, 215, 255, 211, 212, 255, 255, 128, 128, 128}
		},
		{ /* coeff band 6 */
			{ 1, 121, 236, 253, 212, 214, 255, 255, 128, 128, 128},
			{ 141, 84, 213, 252, 201, 202, 255, 219, 128, 128, 128},
			{ 42, 80, 160, 240, 162, 185, 255, 205, 128, 128, 128}
		},
		{ /* coeff band 7 */
			{ 1, 1, 255, 128, 128, 128, 128, 128, 128, 128, 128},
			{ 244, 1, 255, 128, 128, 128, 128, 128, 128, 128, 128},
			{ 238, 1, 255, 128, 128, 128, 128, 128, 128, 128, 128}
		}
	}
};
