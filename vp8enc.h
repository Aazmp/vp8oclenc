#include <CL\cl.h>
#include <conio.h>
#include <stdio.h>
#include <io.h>
#include <fcntl.h>
#include <stdlib.h>
#include <memory.h>
#include <time.h>
#include "stdint1.h"

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
typedef struct {
    int16_t coeffs[25][16];
    int32_t vector_x[4];
    int32_t vector_y[4];
	float SSIM;
} macroblock;
typedef struct 
{
	union mv base_mv;
	int32_t is_inter_mb;
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
	cl_kernel luma_search;
	cl_kernel luma_transform;
	cl_kernel chroma_transform;
	cl_kernel encode_coefficients;
	cl_kernel count_probs;
	cl_kernel num_div_denom;
	cl_kernel luma_interpolate_Hx4;
	cl_kernel luma_interpolate_Vx4;
	cl_kernel chroma_interpolate_Hx8;
	cl_kernel chroma_interpolate_Vx8;
	cl_kernel simple_loop_filter_MBH;
	cl_kernel simple_loop_filter_MBV;
	cl_kernel normal_loop_filter_MBH;
	cl_kernel normal_loop_filter_MBV;
	cl_kernel count_SSIM;
    /* add kernels */

    // these are frame data padded to be devisible by 16 and converted to normalized int16
    cl_mem current_frame_Y;
    cl_mem current_frame_U;
    cl_mem current_frame_V;
    cl_mem last_frame_Y;
    cl_mem last_frame_U;
    cl_mem last_frame_V;
    cl_mem reconstructed_frame_Y;
    cl_mem reconstructed_frame_U;
    cl_mem reconstructed_frame_V;
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
	int32_t max_X_vector_length;
	int32_t max_Y_vector_length;

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

	int32_t quantizer_index_y_dc_p_l;
    int32_t quantizer_index_y_ac_p_l;
    int32_t quantizer_index_y2_dc_p_l;
    int32_t quantizer_index_y2_ac_p_l;
    int32_t quantizer_index_uv_dc_p_l;
    int32_t quantizer_index_uv_ac_p_l;

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
	// uint8_t *last_Y; don'y use now
    uint8_t *last_U;
    uint8_t *last_V;
    macroblock *transformed_blocks;
	macroblock_extra_data *e_data;
    uint8_t *encoded_frame;
	uint32_t encoded_frame_size;
    uint8_t *current_frame_pos_in_pack;
    int32_t current_is_key_frame;
	int32_t prev_is_key_frame;
	float new_SSIM;
	
	int32_t partition_sizes[8];
	uint8_t *partitions;
	uint8_t *partition_0;
	int32_t partition_0_size;

	uint32_t new_probs[4][8][3][11];
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

