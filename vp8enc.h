#include <CL\cl.h>
#include <conio.h>
#include <stdio.h>
#include <io.h>
#include <fcntl.h>
#include <stdlib.h>
#include <memory.h>
#include <time.h>

#define QUANT_TO_FILTER_LEVEL 4
#define DEFAULT_ALTREF_RANGE 16

static const cl_uchar vp8_dc_qlookup[128] =
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

static const cl_short vp8_ac_qlookup[128] =
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

//static const cl_int zigzag[16] = { 0, 1, 4, 8, 5, 2, 3,  6, 9, 12, 13, 10, 7, 11, 14, 15 };
//static const cl_int inv_zigzag[16] = { 0, 1, 5, 6, 2, 4, 7, 12, 3,  8, 11, 13, 9, 10, 14, 15 };
// not only inv zigzag is inverse for zigzag, but
// A[i] = B[zigzag[i]] === A[inv_zigzag[i]] = B[i]


#define ERRORPATH "clErrors.txt"
#define DUMPPATH "dump.y4m"
//#define CPUPATH "..\\Release\\CPU_kernels.cl"
//#define GPUPATH "..\\Release\\GPU_kernels.cl"
#define CPUPATH "CPU_kernels.cl"
#define GPUPATH "GPU_kernels.cl"

union mv {
	cl_uint raw;
	struct {
		cl_short x, y;
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

typedef enum {
	intra_segment = 0,
	UQ_segment = 0,
	HQ_segment = 1,
	AQ_segment = 2,
	LQ_segment = 3
} segment_ids;

typedef struct {
	cl_int y_ac_i; 
	cl_int y_dc_idelta;
	cl_int y2_dc_idelta;
	cl_int y2_ac_idelta;
	cl_int uv_dc_idelta;
	cl_int uv_ac_idelta;
	cl_int loop_filter_level;
	cl_int mbedge_limit;
	cl_int sub_bedge_limit;
	cl_int interior_limit;
	cl_int hev_threshold;
} segment_data;

typedef struct { //in future resize to short or chars!!!
    cl_short coeffs[25][16];
    cl_int vector_x[4];
    cl_int vector_y[4];
	float SSIM;
	cl_int non_zero_coeffs;
	cl_int parts; //16x16 == 0; 8x8 == 1;
	cl_int reference_frame;
	segment_ids segment_id;
} macroblock;

typedef struct {
	cl_int vector_x;
	cl_int vector_y;
} vector_net;
typedef struct 
{
	union mv base_mv;
	cl_int is_inter_mb;
	cl_int parts;
	cl_int mode[16];
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
    cl_command_queue boolcoder_commandQueue_cpu;
	cl_command_queue loopfilterY_commandQueue_cpu;
	cl_command_queue loopfilterU_commandQueue_cpu;
	cl_command_queue loopfilterV_commandQueue_cpu;
    cl_command_queue commandQueue1_gpu;
	cl_command_queue commandQueue2_gpu;
	cl_command_queue commandQueue3_gpu;
    cl_int state_cpu;
    cl_int state_gpu;
	cl_kernel reset_vectors;
	cl_kernel luma_search_1step;
	cl_kernel luma_search_2step;
	cl_kernel downsample;
	cl_kernel select_reference;
	cl_kernel prepare_predictors_and_residual;
	cl_kernel pack_8x8_into_16x16;
	cl_kernel dct4x4;
	cl_kernel wht4x4_iwht4x4;
	cl_kernel idct4x4;
	cl_kernel chroma_transform;
	cl_kernel encode_coefficients;
	cl_kernel count_probs;
	cl_kernel num_div_denom;
	cl_kernel normal_loop_filter_MBH;
	cl_kernel normal_loop_filter_MBV;
	cl_kernel loop_filter_frame;
	cl_kernel count_SSIM_luma;
	cl_kernel count_SSIM_chroma;
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
	//images
	cl_image_format image_format;
	cl_mem last_frame_Y_image;
	cl_mem last_frame_U_image;
	cl_mem last_frame_V_image;
	cl_mem altref_frame_Y_image;
	cl_mem altref_frame_U_image;
	cl_mem altref_frame_V_image;
	cl_mem golden_frame_Y_image;
	cl_mem golden_frame_U_image;
	cl_mem golden_frame_V_image;
	//instead of original size we use reconstructed frame
	cl_mem last_frame_Y_downsampled_by2;
	cl_mem last_frame_Y_downsampled_by4;
	cl_mem last_frame_Y_downsampled_by8;
	cl_mem last_frame_Y_downsampled_by16;
	cl_mem golden_frame_Y_downsampled_by2;
	cl_mem golden_frame_Y_downsampled_by4;
	cl_mem golden_frame_Y_downsampled_by8;
	cl_mem golden_frame_Y_downsampled_by16;
	cl_mem altref_frame_Y_downsampled_by2;
	cl_mem altref_frame_Y_downsampled_by4;
	cl_mem altref_frame_Y_downsampled_by8;
	cl_mem altref_frame_Y_downsampled_by16;
    cl_mem reconstructed_frame_Y;
    cl_mem reconstructed_frame_U;
    cl_mem reconstructed_frame_V;
	cl_mem predictors_Y;
	cl_mem predictors_U;
	cl_mem predictors_V;
	cl_mem residual_Y;
	cl_mem residual_U;
	cl_mem residual_V;
	cl_mem golden_frame_Y;
    cl_mem altref_frame_Y;
	cl_mem cpu_frame_Y; //for a filter
	cl_mem cpu_frame_U;
	cl_mem cpu_frame_V;
    cl_mem last_vnet1;
	cl_mem golden_vnet1;
	cl_mem altref_vnet1;
	cl_mem last_vnet2;
	cl_mem golden_vnet2;
	cl_mem altref_vnet2;
	cl_mem last_metrics;
	cl_mem golden_metrics;
	cl_mem altref_metrics;
	cl_mem mb_mask;
	cl_mem segments_data_gpu;
	cl_mem segments_data_cpu;
	cl_mem third_context;
	cl_mem coeff_probs;
	cl_mem coeff_probs_denom;

    cl_mem transformed_blocks_cpu;
	cl_mem transformed_blocks_gpu;
    cl_mem partitions;
	cl_mem partitions_sizes;

	size_t gpu_work_items_per_dim[1];
	size_t gpu_work_group_size_per_dim[1];
	size_t cpu_work_items_per_dim[1];
	size_t cpu_work_group_size_per_dim[1];

};

struct videoContext
{
    // size of input frame
    cl_int src_width;
    cl_int src_height;
    cl_int src_frame_size_luma;
    cl_int src_frame_size_chroma;
    // size of output frame
    cl_int dst_width;
    cl_int dst_height;
    cl_int dst_frame_size_luma;
    cl_int dst_frame_size_chroma;
    // size of padded frame
    cl_int wrk_width;
    cl_int wrk_height;
    cl_int wrk_frame_size_luma;
    cl_int wrk_frame_size_chroma;

    cl_int mb_width;
    cl_int mb_height;
    cl_int mb_count;
    cl_int GOP_size;
	cl_int altref_range;

    cl_int qi_min; 
    cl_int qi_max;
	cl_int altrefqi[4];
	cl_int lastqi[4];
    
	cl_int loop_filter_type;
	cl_int loop_filter_sharpness;

	cl_int number_of_partitions;
	cl_int partition_step;

	cl_uint timestep;
	cl_uint timescale;
	cl_uint framerate;

	int do_loop_filter_on_gpu;
	int thread_limit;

	float SSIM_target;

};

struct hostFrameBuffers
{
    cl_int frame_number;
	cl_int altref_frame_number;
	cl_int golden_frame_number;
	cl_int last_key_detect;
	cl_int frames_until_key;
	cl_int frames_until_altref;
	cl_int replaced;
    cl_int input_pack_size;
    cl_uchar *input_pack; // allbytes for YUV in one
    cl_uchar *current_Y;
    cl_uchar *current_U;
    cl_uchar *current_V;
    cl_uchar *tmp_Y;
    cl_uchar *tmp_U;
    cl_uchar *tmp_V;
    cl_uchar *reconstructed_Y;
    cl_uchar *reconstructed_U;
    cl_uchar *reconstructed_V;
    cl_uchar *last_U;
    cl_uchar *last_V;
    macroblock *transformed_blocks;
	macroblock_extra_data *e_data;
	segment_data segments_data[4];
    cl_uchar *encoded_frame;
	cl_uint encoded_frame_size;
    cl_uchar *current_frame_pos_in_pack;
    cl_int current_is_key_frame;
	cl_int current_is_altref_frame;
	cl_int current_is_golden_frame;
	cl_int prev_is_key_frame;
	cl_int prev_is_altref_frame;
	cl_int prev_is_golden_frame;
	cl_int skip_prob;
	float new_SSIM;
	
	cl_int partition_sizes[8];
	cl_uchar *partitions;
	cl_uchar *partition_0;
	cl_int partition_0_size;

	cl_uint new_probs[4][8][3][11];
	cl_uint new_probs_denom[4][8][3][11];

	cl_int y_dc_q[4], y_ac_q[4], y2_dc_q[4], y2_ac_q[4], uv_dc_q[4], uv_ac_q[4];

	cl_uchar header[128];
	int header_sz;

	int threads_free;
};

struct fileContext
{
    FILE * handle;
    char * path;
    int cur_pos;
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
