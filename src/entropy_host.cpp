// A mix from Multimedia Mike's encoder, spec, vpxenc and new parts

#include "vp8enc.h"
#include "entropy_host.h"

extern struct fileContext input_file, reconstructed_file;
extern struct deviceContext device;
extern struct videoContext video;
extern struct hostFrameBuffers frames;

typedef struct {
	cl_uchar *output; /* ptr to next byte to be written */
	cl_uint range; /* 128 <= range <= 255 */
	cl_uint bottom; /* minimum value of remaining output */
	cl_int bit_count; /* # of shifts before an output byte is available */
	cl_uint count;
} vp8_bool_encoder;


static void init_bool_encoder(vp8_bool_encoder *e, cl_uchar *start_partition)
{
    e->output = start_partition;
    e->range = 255;
    e->bottom = 0;
    e->bit_count = 24;
    e->count = 0;
}

static void add_one_to_output(cl_uchar *q)
{
    while( *--q == 255)
        *q = 0;
    ++*q;
}

static void write_bool(vp8_bool_encoder *e, int prob, int bool_value)
{
    /* split is approximately (range * prob) / 256 and, crucially,
    is strictly bigger than zero and strictly smaller than range */
    cl_uint split = 1 + ( ((e->range - 1) * prob) >> 8);
    if( bool_value) {
        e->bottom += split; /* move up bottom of interval */
        e->range -= split; /* with corresponding decrease in range */
    } else
        e->range = split;
    while( e->range < 128)
    {
        e->range <<= 1;
        if( e->bottom & (1 << 31)) /* detect carry */
            add_one_to_output(e->output);
        e->bottom <<= 1;
        if( !--e->bit_count) {
            *e->output++ = (cl_uchar) (e->bottom >> 24);
            e->count++;
            e->bottom &= (1 << 24) - 1;
            e->bit_count = 8;
        }
    }
}

static void write_flag(vp8_bool_encoder *e, int b)
{
    write_bool(e, 128, (b)?1:0);
}

static void write_literal(vp8_bool_encoder *e, int i, int size)
{
    int mask = 1 << (size - 1);
    while (mask)
    {
        write_flag(e, !((i & mask) == 0));
        mask >>= 1;
    }
}

static void write_quantizer_delta(vp8_bool_encoder *e, int delta)
{
    int sign;
    if (delta < 0)
    {
        delta *= -1;
        sign = 1;
    }
    else
        sign = 0;

    if (!delta)
        write_flag(e, 0);
    else
    {
        write_flag(e, 1);
        write_literal(e, delta, 4);
        write_flag(e, sign);
    }
}

/* Call this function (exactly once) after encoding the last bool value
for the partition being written */
static void flush_bool_encoder(vp8_bool_encoder *e)
{
    int c = e->bit_count;
    cl_uint v = e->bottom;
    if( v & (1 << (32 - c)))
        add_one_to_output(e->output);
    v <<= c & 7;
    c >>= 3;
    while( --c >= 0)
        v <<= 8;
    c = 4;
    while( --c >= 0) {
        /* write remaining data, possibly padded */
        *e->output++ = (cl_uchar) (v >> 24);
        e->count++;
        v <<= 8;
    }
}

static void write_symbol( vp8_bool_encoder *vbe, encoding_symbol symbol, const Prob *const p, Tree t)
{
    tree_index i = 0;

    do
    {
        const int b = (symbol.bits >> --symbol.size) & 1;
        write_bool(vbe, p[i>>1], b);
        i = t[i+b];
    }
    while (symbol.size);
}

static void write_mv(vp8_bool_encoder *vbe, union mv v, const Prob mvc[2][MVPcount])
{
	cl_short abs_v;
	// short values are 0..7
	// long are 8..1023
	// sizes in luma quarter-pixel or chroma eighth-pixel

	enum {IS_SHORT, SIGN, SHORT, BITS = SHORT + 8 - 1, LONG_WIDTH = 10}; //dublicate one in entropy_host.h
	//       0    ,  1  ,   2  ,   9  =   2   + 8 - 1,    10
	// Y(Row) goes first and uses mvc[0]

	abs_v = (v.d.y < 0) ? (-v.d.y) : v.d.y; // sign and absolute value encoded
	if (abs_v <= 7)
	{
		write_bool(vbe, mvc[0][IS_SHORT], 0); //according to spec-decoder flag '0' for a short range
		encoding_symbol s_tmp;
		s_tmp.bits = abs_v;
		s_tmp.size = 3;// they all 000..111	
		const Prob *const p = &(mvc[0][SHORT]);
		tree_index i = 0;
	    do
		{
			const int b = (s_tmp.bits >> --s_tmp.size) & 1;
			write_bool(vbe, p[i>>1], b);
			i = small_mvtree[i+b];
		}
		while (s_tmp.size);

		if (abs_v != 0)
			write_bool(vbe, mvc[0][SIGN], (v.d.y < 0)); // no sign for zero
	}
	else
	{
		write_bool(vbe, mvc[0][IS_SHORT], 1); //'1' for long range
		int i;
		for(i = 0; i < 3; ++i)
			write_bool(vbe, mvc[0][BITS + i], ((abs_v >> i) & 1));
		for(i = LONG_WIDTH - 1; i > 3; --i)
			write_bool(vbe, mvc[0][BITS + i], ((abs_v >> i) & 1));
		if (abs_v & 0xFFF0)
			write_bool(vbe, mvc[0][BITS + 3], ((abs_v >> 3) & 1));
		// we have to ranges <=7 and >7
		// if in all higher bits (except least 4) there are zeros
		// then bit-3 has to be one or it would be another range 
		// and another encoding sequence
		write_bool(vbe, mvc[0][SIGN], (v.d.y < 0)); 
	}

	// X(Collumn) goes next and uses mvc[1]
	abs_v = (v.d.x < 0) ? (-v.d.x) : v.d.x; // sign and absolute value encoded
	if (abs_v <= 7)
	{
		write_bool(vbe, mvc[1][IS_SHORT], 0); 
		encoding_symbol s_tmp;
		s_tmp.bits = abs_v;
		s_tmp.size = 3;// they all 000..111
		const Prob *const p = &(mvc[1][SHORT]);
		tree_index i = 0;
	    do
		{
			const int b = (s_tmp.bits >> --s_tmp.size) & 1;
			write_bool(vbe, p[i>>1], b);
			i = small_mvtree[i+b];
		}
		while (s_tmp.size);

		if (abs_v == 0)	return; // no sign for zero, no encoding
	}
	else
	{
		write_bool(vbe, mvc[1][IS_SHORT], 1); //'1' for long range
		int i;
		for(i = 0; i < 3; ++i)
			write_bool(vbe, mvc[1][BITS + i], ((abs_v >> i) & 1));
		for(i = LONG_WIDTH - 1; i > 3; --i)
			write_bool(vbe, mvc[1][BITS + i], ((abs_v >> i) & 1));
		if (abs_v & 0xFFF0)
			write_bool(vbe, mvc[1][BITS + 3], ((abs_v >> 3) & 1));
	}
	write_bool(vbe, mvc[1][SIGN], (v.d.x < 0));

    return;
}

static void bool_encode_inter_mb_modes_and_mvs(vp8_bool_encoder *vbe, cl_int mb_num) // mostly copied from guide.pdf (converted to encoder)
{
	cl_int mb_row = mb_num / video.mb_width;
	cl_int mb_col = mb_num % video.mb_width;
	macroblock_extra_data *mb_edata, *above_edata, *left_edata, *above_left_edata;
	macroblock_extra_data imaginary_edata;

	imaginary_edata.base_mv.raw = 0;
	imaginary_edata.is_inter_mb = 0; 
	imaginary_edata.parts = are16x16;

	mb_edata = &(frames.e_data[mb_num]);
	mb_edata->parts=frames.transformed_blocks[mb_num].parts;
	mb_edata->base_mv.d.x = frames.transformed_blocks[mb_num].vector_x[3];
	mb_edata->base_mv.d.y = frames.transformed_blocks[mb_num].vector_y[3];
	if (mb_row>0) above_edata = &(frames.e_data[mb_num-video.mb_width]);
	else above_edata = &imaginary_edata;
	if (mb_col>0) left_edata = &(frames.e_data[mb_num-1]);
	else left_edata = &imaginary_edata;
	if ((mb_col>0) && (mb_row>0)) above_left_edata = &(frames.e_data[mb_num-video.mb_width-1]);
	else above_left_edata = &imaginary_edata;
	// we begin at spot, where spec decoder calls find near mvs
	// but we do an encoder
	//there only two types of macroblocks: SPLITMV (real ones), INTRA_like (imaginary blocks above the frame and to the left)
	// for each macroblock there is a list of three vectors mv[3] (above, left, above_left),
	// and "weights" cnt[4];
	/* "The first three entries in the return value cnt are (in order)
		weighted census values for "zero", "nearest", and "near" vectors.
		The final value indicates the extent to which SPLITMV was used by the
		neighboring macroblocks. The largest possible "weight" value in each
		case is 5." */
	// in reference decoder "raw" stands for putting X and Y component together as int32
	union mv mb_mv_list[4];
	cl_int cnt[4];
	union mv *mb_mv = mb_mv_list;
	mb_mv[0].raw = mb_mv[1].raw = mb_mv[2].raw = 0;
	cl_int *cntx  = cnt;
	cntx[0] = cntx[1] = cntx[2] = cntx[3] = 0;

	// process above 
	// if above is INTRA-like then no action is taken, else above is SPLITMV (all we have)
	// the same for other 2 processes
	if (above_edata->is_inter_mb == 1)
	{
		if (above_edata->base_mv.raw)
		{
			++mb_mv; mb_mv->raw = above_edata->base_mv.raw;
			++cntx;
		}
		*cntx += 2;
	}

	// process left
	if (left_edata->is_inter_mb == 1)
	{
		if (left_edata->base_mv.raw)
		{
		// not using golden or altref, no sign correction(?)
			if (left_edata->base_mv.raw != mb_mv->raw)
			{
				++mb_mv; mb_mv->raw = left_edata->base_mv.raw;
				++cntx;
			}
			*cntx += 2;
		} 
		else cnt[0] += 2;
	}

	// process above_left
	if (above_left_edata->is_inter_mb == 1)
	{
		if (above_left_edata->base_mv.raw)
		{
			if (above_left_edata->base_mv.raw != mb_mv->raw)
			{
				++mb_mv; mb_mv->raw = above_left_edata->base_mv.raw;
				++cntx;
			}
			*cntx += 1;
		} else cnt[0] += 1;
	}

	// if we have three distinct MVs
	if (cnt[3]) /* See if above-left MV can be merged with NEAREST */	
		if (mb_mv->raw == mb_mv_list[1].raw)
			cnt[1] += 1;
	// cnt[CNT_SPLITMV] = ((above->base.y_mode == SPLITMV) + (left->base.y_mode == SPLITMV)) * 2 + (aboveleft->base.y_mode == SPLITMV);
	cnt[3] = (
		((above_edata->is_inter_mb == 1)&&(above_edata->parts!=are16x16)) + 
		((left_edata->is_inter_mb == 1)&&(left_edata->parts!=are16x16))
		)*2 + 
		((above_left_edata->is_inter_mb == 1)&&(above_left_edata->parts != are16x16)); 
	if (cnt[2] > cnt[1])
	{
		int tmp; tmp = cnt[1]; cnt[1] = cnt[2]; cnt[2] = tmp;
		tmp = mb_mv_list[1].raw; mb_mv_list[1].raw = mb_mv_list[2].raw; mb_mv_list[2].raw = tmp;
	}
	// Use near_mvs[CNT_BEST] to store the "best" MV. Note that this storage shares the same address as near_mvs[CNT_ZEROZERO].
	if (cnt[1] >= cnt[0])	mb_mv_list[0].raw = mb_mv_list[1].raw;
	// since we never use NEARMV or NEARESTMV modes, we need only cnt[0] (BEST)
	// this position equals end of dixie.c function find_near_mvs with best_mv in mb_mv_list[0];
	//also we have cnt[] array as index-set for probabilities array
	Prob mv_ref_p[4];
	mv_ref_p[0] = vp8_mode_contexts[cnt[0]][0]; // vp8_mode_contexts[6][4]
	mv_ref_p[1] = vp8_mode_contexts[cnt[1]][1]; // defined in
	mv_ref_p[2] = vp8_mode_contexts[cnt[2]][2]; // entropy_host.h
	mv_ref_p[3] = vp8_mode_contexts[cnt[3]][3];

	encoding_symbol s_tmp;

	if (frames.transformed_blocks[mb_num].parts == are8x8)
	{
		// encode SPLITMV mode
		s_tmp.bits = 15; // "1111"
		s_tmp.size = 4;
		write_symbol(vbe, s_tmp, mv_ref_p, mv_ref_tree);

		// encode sub_mv_mode
		s_tmp.bits = 2; // mv_quarters = "10" 
		s_tmp.size = 2;
		write_symbol(vbe, s_tmp, split_mv_probs, split_mv_tree);
		
		cl_int b_num;
		for (b_num = 0; b_num < 4; ++b_num)
		{
			// b_num being part number and block number
			union mv left_mv, above_mv, this_mv;
			cl_int b_col, b_row;
			b_row = b_num / 2; b_col = b_num % 2;
			// read previous vectors (they are already updated, or zeros if it's border case)
			if (b_col > 0) {
				left_mv.d.x = frames.transformed_blocks[mb_num].vector_x[b_num - 1];
				left_mv.d.y = frames.transformed_blocks[mb_num].vector_y[b_num - 1];
			}
			else if (left_edata->is_inter_mb == 1) {
				left_mv.d.x = frames.transformed_blocks[mb_num - 1].vector_x[b_num + 1];
				left_mv.d.y = frames.transformed_blocks[mb_num - 1].vector_y[b_num + 1];
			} 
			else {
				left_mv.d.x = 0; // because there are zeroes there
				left_mv.d.y = 0;
			}
			if (b_row > 0) {
				above_mv.d.x = frames.transformed_blocks[mb_num].vector_x[b_num - 2];
				above_mv.d.y = frames.transformed_blocks[mb_num].vector_y[b_num - 2];
			}
			else if (above_edata->is_inter_mb == 1) {
				above_mv.d.x = frames.transformed_blocks[mb_num - video.mb_width].vector_x[b_num + 2];
				above_mv.d.y = frames.transformed_blocks[mb_num - video.mb_width].vector_y[b_num + 2];
			} 
			else {
				above_mv.d.x = 0; 
				above_mv.d.y = 0;
			}
			// here we take out computed vectors and update them to refence to previous
			this_mv.d.x = frames.transformed_blocks[mb_num].vector_x[b_num];
			this_mv.d.y = frames.transformed_blocks[mb_num].vector_y[b_num];
			cl_int lez = !(left_mv.raw); // flags for context for decoding submv
			cl_int aez = !(above_mv.raw);
			cl_int lea = (left_mv.raw == above_mv.raw); //l = left, a = above, z = zero, e = equals
			cl_int ctx = 0;
			if (lea&&lez) ctx = 4; 
			else if (lea) ctx = 3;
			else if (aez) ctx = 2; // it seems above ha higher priority here
			else if (lez) ctx = 1;
			if (this_mv.raw == left_mv.raw) {// LEFT = "0" 
				s_tmp.bits = 0;
				s_tmp.size = 1;
				//write_symbol(encoder, symbol, *probs, tree[])
				write_symbol(vbe, s_tmp, submv_ref_probs2[ctx], submv_ref_tree);
			}
			else if (this_mv.raw == above_mv.raw) { // ABOVE = "10" 
				s_tmp.bits = 2;
				s_tmp.size = 2;
				write_symbol(vbe, s_tmp, submv_ref_probs2[ctx], submv_ref_tree);
			}
			else if (this_mv.raw == 0) { // ZERO = "110"
				s_tmp.bits = 6;
				s_tmp.size = 3;
				write_symbol(vbe, s_tmp, submv_ref_probs2[ctx], submv_ref_tree);
			}
			else { // NEW = "111" 
				s_tmp.bits = 7;
				s_tmp.size = 3;
				write_symbol(vbe, s_tmp, submv_ref_probs2[ctx], submv_ref_tree);
				// and here we subtract best_mv found for this macroblock and stored in mb_mv_list[0]
				this_mv.d.x -= mb_mv_list[0].d.x;
				this_mv.d.y -= mb_mv_list[0].d.y;
				write_mv(vbe, this_mv, new_mv_context);
			}

		}
	}
	else if (frames.transformed_blocks[mb_num].parts == are16x16)
	{
		if (mb_edata->base_mv.raw == 0) {
			s_tmp.bits = 0;
			s_tmp.size = 1;
			write_symbol(vbe, s_tmp, mv_ref_p, mv_ref_tree);
		} else if (mb_edata->base_mv.raw == mb_mv_list[1].raw) {
			s_tmp.bits = 2; 
			s_tmp.size = 2;
			write_symbol(vbe, s_tmp, mv_ref_p, mv_ref_tree);
		} else if (mb_edata->base_mv.raw == mb_mv_list[2].raw) {
			s_tmp.bits = 6;
			s_tmp.size = 3;
			write_symbol(vbe, s_tmp, mv_ref_p, mv_ref_tree);
		} else {
			s_tmp.bits = 14; 
			s_tmp.size = 4;
			write_symbol(vbe, s_tmp, mv_ref_p, mv_ref_tree);
			union mv mv_delta;
			mv_delta.d.x = mb_edata->base_mv.d.x - mb_mv_list[0].d.x;
			mv_delta.d.y = mb_edata->base_mv.d.y - mb_mv_list[0].d.y;
			write_mv(vbe, mv_delta, new_mv_context);
		}
	}
	else printf("\nUnsupported macroblock partitioning %d! (encoding)\n",frames.transformed_blocks[mb_num].parts);


	// according to spec vector [15] is set as base (we have 3th on that place)
	// that will be referenced by below, right and below_right macroblocks
	mb_edata->base_mv.d.x = (cl_short)frames.transformed_blocks[mb_num].vector_x[3];
	mb_edata->base_mv.d.y = (cl_short)frames.transformed_blocks[mb_num].vector_y[3]; 

	return;
}

static void count_mv(vp8_bool_encoder *vbe, union mv v, cl_uint num[2][MVPcount], cl_uint denom[2][MVPcount])
{
	cl_short abs_v;

	enum {IS_SHORT, SIGN, SHORT, BITS = SHORT + 8 - 1, LONG_WIDTH = 10}; 

	++(denom[0][IS_SHORT]);
	++(denom[1][IS_SHORT]);

	abs_v = (v.d.y < 0) ? (-v.d.y) : v.d.y;
	if (abs_v <= 7)
	{
		++(num[0][IS_SHORT]);
		encoding_symbol s_tmp;
		s_tmp.bits = abs_v;
		s_tmp.size = 3;
		
		cl_uint * const pn = &(num[0][SHORT]);
		cl_uint * const pd = &(denom[0][SHORT]);
		tree_index i = 0;
		do	{
			const int b = (s_tmp.bits >> --s_tmp.size) & 1;
			pn[i>>1] += (1 - b);
			++(pd[i>>1]);
			i = small_mvtree[i+b];
		}
		while (s_tmp.size);

		if (abs_v != 0) {
			num[0][SIGN] += (v.d.y > 0);
			++(denom[0][SIGN]);
		}
	}
	else
	{
		//no numerator increment for long
		int i;
		for(i = 0; i < 3; ++i) {
			num[0][BITS + i] += (1 - ((abs_v >> i) & 1));
			++(denom[0][BITS + i]);
		}
		for(i = LONG_WIDTH - 1; i > 3; --i) {
			num[0][BITS + i] += (1 - ((abs_v >> i) & 1));
			++(denom[0][BITS + i]);
		}
		if (abs_v & 0xFFF0) {
			num[0][BITS + 3] += (1 - ((abs_v >> 3) & 1));
			++(denom[0][BITS + 3]);
		}
		num[0][SIGN] += (v.d.y > 0);
		++(denom[0][SIGN]);
	}
	

	abs_v = (v.d.x < 0) ? (-v.d.x) : v.d.x; 
	if (abs_v <= 7)
	{
		++(num[1][IS_SHORT]);
		encoding_symbol s_tmp;
		s_tmp.bits = abs_v;
		s_tmp.size = 3;
		cl_uint * const pn = &(num[1][SHORT]);
		cl_uint * const pd = &(denom[1][SHORT]);
		tree_index i = 0;
		do	{
			const int b = (s_tmp.bits >> --s_tmp.size) & 1;
			pn[i>>1] += (1 - b);
			++(pd[i>>1]);
			i = small_mvtree[i+b];
		}
		while (s_tmp.size);

		if (abs_v == 0)
			return; 
	}
	else
	{
		int i;
		for(i = 0; i < 3; ++i) {
			num[1][BITS + i] += (1 - ((abs_v >> i) & 1));
			++(denom[1][BITS + i]);
		}
		for(i = LONG_WIDTH - 1; i > 3; --i) {
			num[1][BITS + i] += (1 - ((abs_v >> i) & 1));
			++(denom[1][BITS + i]);
		}
		if (abs_v & 0xFFF0) {
			num[1][BITS + 3] += (1 - ((abs_v >> 3) & 1));
			++(denom[1][BITS + 3]);
		}

	}
	num[1][SIGN] += (v.d.x > 0);
	++(denom[1][SIGN]);
    return;
}

static void count_mv_probs(vp8_bool_encoder *vbe, cl_int mb_num) 
{
	// it looks similar to funtion where we encode vectors
	// BUT
	// we don't write anything here
	// just count probs
	cl_int mb_row = mb_num / video.mb_width;
	cl_int mb_col = mb_num % video.mb_width;
	macroblock_extra_data *mb_edata, *above_edata, *left_edata, *above_left_edata;
	macroblock_extra_data imaginary_edata;

	imaginary_edata.base_mv.raw = 0;
	imaginary_edata.is_inter_mb = 0;
	imaginary_edata.parts = are16x16;

	mb_edata = &(frames.e_data[mb_num]);
	mb_edata->parts = frames.transformed_blocks[mb_num].parts;
	mb_edata->base_mv.d.x = frames.transformed_blocks[mb_num].vector_x[3];
	mb_edata->base_mv.d.y = frames.transformed_blocks[mb_num].vector_y[3];
	if (mb_row>0) above_edata = &(frames.e_data[mb_num-video.mb_width]);
	else above_edata = &imaginary_edata;
	if (mb_col>0) left_edata = &(frames.e_data[mb_num-1]);
	else left_edata = &imaginary_edata;
	if ((mb_col>0) && (mb_row>0)) above_left_edata = &(frames.e_data[mb_num-video.mb_width-1]);
	else above_left_edata = &imaginary_edata;
	union mv mb_mv_list[4];
	cl_int cnt[4];
	union mv *mb_mv = mb_mv_list;
	mb_mv[0].raw = mb_mv[1].raw = mb_mv[2].raw = 0;
	cl_int *cntx  = cnt;
	cntx[0] = cntx[1] = cntx[2] = cntx[3] = 0;
	if (above_edata->is_inter_mb == 1)
	{
		if (above_edata->base_mv.raw)
		{
			++mb_mv; mb_mv->raw = above_edata->base_mv.raw;
			++cntx;
		}
		*cntx += 2;
	}
	if (left_edata->is_inter_mb == 1)
	{
		if (left_edata->base_mv.raw)
		{
			if (left_edata->base_mv.raw != mb_mv->raw)
			{
				++mb_mv; mb_mv->raw = left_edata->base_mv.raw;
				++cntx;
			}
			*cntx += 2;
		} 
		else cnt[0] += 2;
	}
	if (above_left_edata->is_inter_mb == 1)
	{
		if (above_left_edata->base_mv.raw)
		{
			if (above_left_edata->base_mv.raw != mb_mv->raw)
			{
				++mb_mv; mb_mv->raw = above_left_edata->base_mv.raw;
				++cntx;
			}
			*cntx += 1;
		} else cnt[0] += 1;
	}

	if (cnt[3]) 
		if (mb_mv->raw == mb_mv_list[1].raw)
			cnt[1] += 1;
	cnt[3] = (
		((above_edata->is_inter_mb == 1)&&(above_edata->parts!=are16x16)) + 
		((left_edata->is_inter_mb == 1)&&(left_edata->parts!=are16x16))
		)*2 + 
		((above_left_edata->is_inter_mb == 1)&&(above_left_edata->parts != are16x16)); 
	if (cnt[2] > cnt[1])
	{
		cl_int tmp; tmp = cnt[1]; cnt[1] = cnt[2]; cnt[2] = tmp;
		tmp = mb_mv_list[1].raw; mb_mv_list[1].raw = mb_mv_list[2].raw; mb_mv_list[2].raw = tmp;
	}
	if (cnt[1] >= cnt[0])	mb_mv_list[0].raw = mb_mv_list[1].raw;
	Prob mv_ref_p[4];
	mv_ref_p[0] = vp8_mode_contexts[cnt[0]][0]; 
	mv_ref_p[1] = vp8_mode_contexts[cnt[1]][1]; 
	mv_ref_p[2] = vp8_mode_contexts[cnt[2]][2];
	mv_ref_p[3] = vp8_mode_contexts[cnt[3]][3];

	// encode SPLITMV mode
	// -		we count only now
	// encode sub_mv_mode
	// -
	if (frames.transformed_blocks[mb_num].parts == are8x8)
	{
		cl_int b_num;
		for (b_num = 0; b_num < 4; ++b_num)
		{
			// on block level imaginary blocks above and to the left of frame are blocks with ZERO MV 0,0
			// b_num being part number and block number
			union mv left_mv, above_mv, this_mv;
			cl_int b_col, b_row;
			b_row = b_num / 2; b_col = b_num % 2;
			// read previous vectors (they are already updated, or zeros if it's border case)
			if (b_col > 0) {
				left_mv.d.x = frames.transformed_blocks[mb_num].vector_x[b_num - 1];
				left_mv.d.y = frames.transformed_blocks[mb_num].vector_y[b_num - 1];
			}
			else if (left_edata->is_inter_mb == 1) {
				left_mv.d.x = frames.transformed_blocks[mb_num - 1].vector_x[b_num + 1];
				left_mv.d.y = frames.transformed_blocks[mb_num - 1].vector_y[b_num + 1];
			} 
			else {
				left_mv.d.x = 0; // because there are zeroes there
				left_mv.d.y = 0;
			}
			if (b_row > 0) {
				above_mv.d.x = frames.transformed_blocks[mb_num].vector_x[b_num - 2];
				above_mv.d.y = frames.transformed_blocks[mb_num].vector_y[b_num - 2];
			}
			else if (above_edata->is_inter_mb == 1) {
				above_mv.d.x = frames.transformed_blocks[mb_num - video.mb_width].vector_x[b_num + 2];
				above_mv.d.y = frames.transformed_blocks[mb_num - video.mb_width].vector_y[b_num + 2];
			} 
			else {
				above_mv.d.x = 0; 
				above_mv.d.y = 0;
			}
			// here we take out computed vectors and update them to refence to previous
			this_mv.d.x = frames.transformed_blocks[mb_num].vector_x[b_num];
			this_mv.d.y = frames.transformed_blocks[mb_num].vector_y[b_num];
			cl_int lez = !(left_mv.raw); // flags for context for decoding submv
			cl_int aez = !(above_mv.raw);
			cl_int lea = (left_mv.raw == above_mv.raw); //l = left, a = above, z = zero, e = equals
			cl_int ctx = 0;
			if (lea&&lez) ctx = 4; 
			else if (lea) ctx = 3;
			else if (aez) ctx = 2;
			else if (lez) ctx = 1;
			if ((this_mv.raw != left_mv.raw) &&
				(this_mv.raw != above_mv.raw) &&
				(this_mv.raw != 0)) 
			{ // NEW = "111" 
				this_mv.d.x -= mb_mv_list[0].d.x;
				this_mv.d.y -= mb_mv_list[0].d.y;
				count_mv(vbe, this_mv, num_mv_context, denom_mv_context);
			}
		}
	}
	else if (frames.transformed_blocks[mb_num].parts == are16x16)
	{
		if ((mb_edata->base_mv.raw != 0) && 
			(mb_edata->base_mv.raw != mb_mv_list[1].raw) &&
			(mb_edata->base_mv.raw != mb_mv_list[2].raw)) {
			union mv mv_delta;
			mv_delta.d.x = mb_edata->base_mv.d.x - mb_mv_list[0].d.x;
			mv_delta.d.y = mb_edata->base_mv.d.y - mb_mv_list[0].d.y;
			count_mv(vbe, mv_delta, num_mv_context, denom_mv_context);
		}
	}
	else printf("\nUnsupported macroblock partitioning %d! (counting)\n",frames.transformed_blocks[mb_num].parts);
	// according to spec vector [15] is set as base (we have 3th on that place)
	// that will be referenced by below, right and below_right macroblocks
	mb_edata->base_mv.d.x = (cl_short)frames.transformed_blocks[mb_num].vector_x[3];
	mb_edata->base_mv.d.y = (cl_short)frames.transformed_blocks[mb_num].vector_y[3]; 
	return;
}


void encode_header(cl_uchar* partition) // return  size of encoded header
{
	// TODO: move all functions used to count probabilities from root to additional function

	cl_int prob_intra, prob_last, prob_gf, mb_num, segmentation_enabled;
	const Prob *new_ymode_prob = ymode_prob;
	const Prob *new_uv_mode_prob = uv_mode_prob;

	// uncompressed first part(frame tag..) of frame header will be encoded last
	// because it has size of 1st partition

	//using functions taken from Multimedia Mike's encoder
	//write_bool(encoder, prob, 1bit_value)
		//encoded 1bit value with Prob probability
	//write_flag(encoder, 1_bit_value)
		//encodes 1bit with probability 128
	//write_literal(encoder, value, N)
		//writes N bits of value with probabilities 128 each
	//write_quantizer_delta(encoder, delta-value)
		// delta-value encoded as 4 bit absolute value first and 1 bit sign next. Probabilities are 128
	
	vp8_bool_encoder *vbe = (vp8_bool_encoder*)malloc(sizeof(vp8_bool_encoder));
	// start of encoding header
	// at the start of every partition - init
	cl_int bool_offset;
	if (!frames.current_is_key_frame) 
		bool_offset = 3;
	else 
		bool_offset = 10;
	init_bool_encoder(vbe, partition+bool_offset); // 10(3) bytes for uncompressed data chunk

	/*------------------------------------------------- | ----- |
    |   if (key_frame) {                                |       |
    |       color_space                                 | L(1)  |
    |       clamping_type                               | L(1)  |
    |   }                                               |       |*/
	if (frames.current_is_key_frame)
	{
		write_flag(vbe, 0); //YUV (no other options at this time)
		write_flag(vbe, 0); // decoder have to clamp reconstructed values
	}

	/*  segmentation_enabled                            | L(1)  | 
	|  if (segmentation_enabled)						|       | 
    | start of update_segmentation() block	
    |		update_segmentation()						| Type	|
	| ------------------------------------------------- | ----- | 
	|		update_mb_segmentation_map					| L(1)	| 
	|		update_segment_feature_data					| L(1)	| 
	|		if (update_segment_feature_data) {			|		| 
	|			segment_feature_mode					| L(1)	| 
	|			for (i = 0; i < 4; i++) {				|		| 
	|				quantizer_update					| L(1)	| 
	|				if (quantizer_update) {				|		| 
	|					quantizer_update_value			| L(7)	| 
	|					quantizer_update_sign			| L(1)	| 
	|				}									|		| 
	|			}										|		| 
	|			for (i = 0; i < 4; i++) {				|		| 
	|				loop_filter_update					| L(1)	|
	|				if (loop_filter_update) {			|		|
	|					lf_update_value					| L(6)	|
	|					lf_update_sign					| L(1)	|
	|				}									|		|
	|			}										|		| 
	|		}											|		|
	|		if (update_mb_segmentation_map) {			|		|
	|			for (i = 0; i < 3; i++) {				|		|
	|				segment_prob_update					| L(1)	|
	|				if (segment_prob_update)			|		|
	|					segment_prob					| L(8)	|
	|			}										|		|
	|		}											|		| */
	segmentation_enabled = !frames.current_is_key_frame; //only for inter frames;
    write_flag(vbe, segmentation_enabled);     // segmentation test
	if (segmentation_enabled) 
	{
		cl_int update_mb_segmentation_map, update_segment_feature_data;
		update_mb_segmentation_map = 1;
		update_segment_feature_data = 1;
		write_flag(vbe, update_mb_segmentation_map);
		write_flag(vbe, update_segment_feature_data);
		if (update_segment_feature_data) 
		{
			write_flag(vbe, 1); // 1 - absolute; 0 - delta
			int i;
			for (i = 0; i < 4; ++i)
			{ 
				write_flag(vbe, 1);//always update
				write_literal(vbe, frames.segments_data[i].y_ac_i, 7);
				write_flag(vbe, 0); // sign for absolute values is 0
			}
			for (i = 0; i < 4; ++i)
			{ 
				write_flag(vbe, 1);//always update
				write_literal(vbe, frames.segments_data[i].loop_filter_level, 6);
				write_flag(vbe, 0); // sign for absolute values is 0
			}
		}
		if (update_mb_segmentation_map) 
		{
			int i,seg_count[4];
			seg_count[0] = seg_count[1] = seg_count[2] = seg_count[3] = 0;
			for (i = 0; i < video.mb_count; ++i)
				++seg_count[frames.transformed_blocks[i].segment_id];
			// 1 bit choose between 0,1 (0) and 2,3 (1) segments
			i = (seg_count[0]+seg_count[1])*255/video.mb_count;
			new_segment_prob[0] = i;
			// 2 bit :  0 (0) and 1 (1) segments
			i = (seg_count[0]+seg_count[1]); i+=(i==0); i = seg_count[0]*255/i;
			new_segment_prob[1] = i;
			// 3 bit :  2 (0) and 3 (1) segments
			i = (seg_count[2]+seg_count[3]); i+=(i==0); i = seg_count[2]*255/i;
			new_segment_prob[2] = i;
			
			write_flag(vbe, 1); write_literal(vbe, new_segment_prob[0], 8); 
			write_flag(vbe, 1); write_literal(vbe, new_segment_prob[1], 8); 
			write_flag(vbe, 1); write_literal(vbe, new_segment_prob[2], 8); 
		}
	}
    /* end of update_segmentation() block	*/

	/*  filter_type                                     | L(1)  | */
	write_flag(vbe, video.loop_filter_type);
    /*  loop_filter_level                               | L(6)  | */
	write_literal(vbe, frames.segments_data[0].loop_filter_level, 6);
    /*  sharpness_level                                 | L(3)  | */
	write_literal(vbe, video.loop_filter_sharpness, 3);

	/*  start of mb_lf_adjustments() block */
    /*  loop_filter_adj_enable                          | L(1)  | */
    write_flag(vbe, 0); // do not adjust loop filter
    /*  if (loop_filter_adj_enable) ...                 |       |
    // but we do not adjust here						|		|
    /*  end of mb_lf_adjustments() block */

	/*  log2_nbr_of_dct_partitions                      | L(2)  | */
    if (video.number_of_partitions == 1)
		write_literal(vbe, 0, 2); // only 1 partition (2 bits)
	else 
		if (video.number_of_partitions == 2)
			write_literal(vbe, 1, 2);
		else 
			if (video.number_of_partitions == 4)
				write_literal(vbe, 2, 2);
			else 
				write_literal(vbe, 3, 2);

	/*  start of quant_indices() block */
    /*  y_ac_qi                                         | L(7)  |
    |   y_dc_delta_present                              | L(1)  |
    |   if (y_dc_delta_present) {                       |       |
    |       y_dc_delta_magnitude                        | L(4)  |
    |       y_dc_delta_sign                             | L(1)  |
    |   }                                               |       |
    |   y2_dc_delta_present                             | L(1)  |
    |   if (y2_dc_delta_present) {                      |       |
    |       y2_dc_delta_magnitude                       | L(4)  |
    |       y2_dc_delta_sign                            | L(1)  |
    |   }                                               |       |
    |   y2_ac_delta_present                             | L(1)  |
    |   if (y2_ac_delta_present) {                      |       |
    |       y2_ac_delta_magnitude                       | L(4)  |
    |       y2_ac_delta_sign                            | L(1)  |
    |   }                                               |       |
    |   uv_dc_delta_present                             | L(1)  |
    |   if (uv_dc_delta_present) {                      |       |
    |       uv_dc_delta_magnitude                       | L(4)  |
    |       uv_dc_delta_sign                            | L(1)  |
    |   }                                               |       |
    |   uv_ac_delta_present                             | L(1)  |
    |   if (uv_ac_delta_present) {                      |       |
    |       uv_ac_delta_magnitude                       | L(4)  |
    |       uv_ac_delta_sign                            | L(1)  |
    |   }                                               |       | */
    // encode quantizers
	write_literal(vbe, frames.segments_data[0].y_ac_i, 7); // Y AC quantizer index (full 7 bits)
	write_quantizer_delta(vbe, frames.segments_data[0].y_dc_idelta); // Y DC index delta
	write_quantizer_delta(vbe, frames.segments_data[0].y2_dc_idelta); // Y2 DC index delta
	write_quantizer_delta(vbe, frames.segments_data[0].y2_ac_idelta); // Y2 AC index delta
	write_quantizer_delta(vbe, frames.segments_data[0].uv_dc_idelta); // UV DC index delta
	write_quantizer_delta(vbe, frames.segments_data[0].uv_ac_idelta); // UV AC index delta

    /* end of quant_indices() block */

	/*  if (key_frame)                                  |       |
    |       refresh_entropy_probs                       | L(1)  | */
    // do not update coefficient probabilities
    // refresh_entropy_probs determines whether updated token probabilities are used only for this frame or until further update.
    // Explanation found in google.groups:
    //On a key frame, all probabilities are reset to default baseline probabilities, then on each subsequent frame,
    //these probabilities are combined with individual updates for use in coefficient decoding within the frame.
    //1. When refresh_entropy_probs flag is 1, the updated combined probabilities become the new baseline for next frame.
    //2. When refresh_entropy_probs flag is 0, current frame's probability updates are discarded after coefficient decoding is completed for the frame .
    //      The baseline probabilities prior to this frame's probability updates are then used as the baseline for the next frame.
    if (frames.current_is_key_frame)
		write_flag(vbe, 0); // no updatng probabilities now
	else
	{
    /*  else {											|		|
    |       refresh_golden_frame                        | L(1)  |*/
		write_flag(vbe, frames.current_is_golden_frame);
    /*      refresh_alternate_frame                     | L(1)  |*/
		write_flag(vbe, frames.current_is_altref_frame);
    /*      if (!refresh_golden_frame)                  |       |
    |           copy_buffer_to_golden                   | L(2)  | */// 0: no copying; 1: last -> golden; 2: altref -> golden
		if (!frames.current_is_golden_frame)
			write_literal(vbe, 0, 2);
    /*      if (!refresh_alternate_frame)               |       |
    |           copy_buffer_to_alternate                | L(2)  | */// 0: no; 1: last -> altref; 2: golden -> altref
		if (!frames.current_is_altref_frame)
			write_literal(vbe, 0, 2);
    /*      sign_bias_golden                            | L(1)  |
    |       sign_bias_alternate                         | L(1)  |*/
	//These values are used to control the sign of the motion vectors when
	// golden or altref frames are referenced
		write_flag(vbe, 0);
		write_flag(vbe, 0); 
    /*      refresh_entropy_probs                       | L(1)  | */
		write_flag(vbe, 0); // no updating probabilities now
    /*      refresh_last                                | L(1)  | */
		write_flag(vbe, 1); // we will always refresh  last
    /*   }												|		| */
	}

	/*  start of token_prob_update() block */
    /* bitstream from page 120+ version */
    /*  for (i = 0; i < 4; i++) {                       |       |
    |       for (j = 0; j < 8; j++) {                   |       |
    |           for (k = 0; k < 3; k++) {               |       |
    |               for (l = 0; l < 11; l++) {          |       |
    |                   coeff_prob_update_flag          | L(1)  |
    |                   if (coeff_prob_update_flag)     |       |
    |                       coeff_prob                  | L(8)  |
    |               }                                   |       |
    |           }                                       |       |
    |       }                                           |       |
    |   }                                               |       |
    | ------------------------------------------------- | ----- |    */
    /* BUT in example with read probabilities from middle of the spec (explanation parts):
    ---- Begin code block --------------------------------------
    int i = 0; do {
        int j = 0; do {
            int k = 0; do {
                int t = 0; do {
                    if (read_bool(d, coeff_update_probs [i] [j] [k] [t]))
                        coeff_probs [i] [j] [k] [t] = read_literal(d, 8);
                } while (++t < num_dct_tokens - 1);
            } while (++k < 3);
        } while (++j < 8);
    } while (++i < 4);
    ---- End code block ---------------------------------------- */
    // so Flag is B(coeff_probs [i] [j] [k] [t]), but prob is L(8)
    // last one (and the code version) is right. Bitstream.pdf mistake there is.
	{ cl_int i,j,k,l;
    for (i = 0; i < 4; i++)
        for (j = 0; j < 8; j++)
            for (k = 0; k < 3; k++)
				for (l = 0; l < 11; l++) {
					if (frames.new_probs_denom[i][j][k][l] < 2) 
						write_bool(vbe, coeff_update_probs[i][j][k][l], 0); 
					else {
						write_bool(vbe, coeff_update_probs[i][j][k][l], 1); 
						write_literal(vbe, frames.new_probs[i][j][k][l], 8);
					}
				}
	}
    /*  end of token_prob_update() block */

    /*  mb_no_skip_coeff                                | L(1)  | */
    /*  if (mb_no_skip_coeff)                           |       |
    |       prob_skip_false                             | L(8)  | */
	write_flag(vbe, 1);
	write_literal(vbe, frames.skip_prob, 8);

	/*  if (!key_frame) {                               |       |*/
	if (!frames.current_is_key_frame)
	{
		int i;
    /*      prob_intra                                  | L(8)  |*/
		// probability, that block intra encoded. We use only inter in not-key frames
		prob_intra = frames.replaced*255/video.mb_count;
		if ((frames.replaced > 0) && (prob_intra < 2)) prob_intra = 2;
		if ((frames.replaced < video.mb_count) && (prob_intra > 254)) prob_intra = 254;
		write_literal(vbe, prob_intra, 8); 
    /*      prob_last                                   | L(8)  |*/
		// probability of last frame used as reference  for inter encoding
		prob_last = 0; // flag value for last is ZERO; prob of flag being ZERO 
		prob_gf = 0;
		for (i = 0; i < video.mb_count; ++i)
		{
			prob_last += (frames.transformed_blocks[i].reference_frame == LAST);
			prob_gf += (frames.transformed_blocks[i].reference_frame == GOLDEN);
		}
		prob_gf = (prob_gf*256)/(video.mb_count-prob_last+1);
		prob_last = (prob_last*256)/video.mb_count;
		prob_gf = (prob_gf > 255) ? 255 : ((prob_gf < 1) ? 1 : prob_gf);
		prob_last = (prob_last > 255) ? 255 : ((prob_last < 1) ? 1 : prob_last);
		//prob_last = prob_gf = 128;
		write_literal(vbe, prob_last, 8); // we use lasts and goldens
    /*      prob_gf                                     | L(8)  |*/
		// probability of golden frame used as reference  for inter encoding
		write_literal(vbe, prob_gf, 8); 
    /*      intra_16x16_prob_update_flag                | L(1)  |*/
		// indicates if the branch probabilities used in the decoding of the luma intra-prediction(for inter-frames only) mode are updated
    /*      if (intra_16x16_prob_update_flag) {         |       |
    |           for (i = 0; i < 4; i++)                 |       |
    |               intra_16x16_prob                    | L(8)  |
    |       }                                           |       |*/
	/*      intra_chroma prob_update_flag               | L(1)  |*/
    /*      if (intra_chroma_prob_update_flag) {        |       |
    |           for (i = 0; i < 3; i++)                 |       |
    |               intra_chroma_prob                   | L(8)  |
    |       }                                           |       |*/
		if (frames.replaced > 7) 
		{
			write_flag(vbe, 1);
			new_ymode_prob = B_ymode_prob;
			for (i = 0; i < 4; ++i)
				write_literal(vbe, 0, 8); //fixed on B_PRED 
			write_flag(vbe, 1);
			new_uv_mode_prob = TM_uv_mode_prob;
			for (i = 0; i < 3; ++i)
				write_literal(vbe, 0, 8); //fixed on TM_PRED 

		}
		else {
			write_flag(vbe, 0);	
			write_flag(vbe, 0);
		}
    /*		for (i = 0; i < 2; i++) {					|		| // it is a start of mv_prob_update()
	|			for (j = 0; j < 19; j++) {				|		|
	|				mv_prob_update_flag					| L(1)	|
	|				if (mv_prob_update_flag)			|		|
	|					prob							| L(7)	|
	|			}										|		|
	|		}											|		|*/
		// same here. spec says L(1), but it's B(p) !!11!!1one!!1oneone
		{ cl_int i,j, mb_num;
		for (i = 0; i < 2; ++i)
			for (j = 0; j < 19; ++j) {
				num_mv_context[i][j] = 0;
				denom_mv_context[i][j] = 1;
			}
		for (mb_num = 0; mb_num < video.mb_count; ++mb_num) {
			if (frames.e_data[mb_num].is_inter_mb) count_mv_probs(vbe, mb_num);
		}
		for (i = 0; i < 2; ++i)
			for (j = 0; j < 19; ++j) {
				write_bool(vbe, vp8_mv_update_probs[i][j], 1);
				new_mv_context[i][j] = (cl_uchar)((num_mv_context[i][j] << 8) / denom_mv_context[i][j]);
				// standard says that these probs are stored as 7 bit values (7 most significant of total 8)
				// so we have to set LSB = 0;
				new_mv_context[i][j] &= (~(0x1));
				// spec decoder sets prob to 1 when pulls 0 from header
				// but we'll clamp for safety to 2..254
				new_mv_context[i][j] = ((new_mv_context[i][j] < 2) ? 2 : new_mv_context[i][j]);
				new_mv_context[i][j] = ((new_mv_context[i][j] > 254) ? 254 : new_mv_context[i][j]);
				write_literal(vbe, (new_mv_context[i][j]>>1), 7);
			}
		}
    /*  }                                               |       | *///end of  /*  if (!key_frame) { prob_intra....
	}

								//// Now per MacroBlock data

    //each macroblock(MB) will have these
    /*  Macroblock Data                                 | Type  |
    | ------------------------------------------------- | ----- |
    |   macroblock_header()                             |       | <---- this in 1st  partition
    |   residual_data()                                 |       | <---- this in 1..8 next patitions */
    // encode mode for each macroblock
    // vertical mode for the first pass
	for (mb_num = 0; mb_num < video.mb_count; ++mb_num)
	{
        /*  start of macroblock_header() block  */
        /*  macroblock_header()                         | Type  |
        | --------------------------------------------- | ----- |
        |   if (update_mb_segmentation_map)             |       |
        |       segment_id                              |   T   | */
        if (segmentation_enabled)
		{
			encoding_symbol s_tmp;
			s_tmp.bits = frames.transformed_blocks[mb_num].segment_id;
			s_tmp.size = 2;
			write_symbol(vbe, s_tmp, new_segment_prob, mb_segment_tree);
		}

        /*  if (mb_no_skip_coeff)                       |       |
        |       mb_skip_coeff                           | B(p)  | */
		if (frames.transformed_blocks[mb_num].non_zero_coeffs != 0) 
			write_bool(vbe, frames.skip_prob, 0);
		else write_bool(vbe, frames.skip_prob, 1);

        /*  if (!key_frame)                             |       |
        |       is_inter_mb                             | B(p)  | */
			//  from spec:
			/*After the feature specification (which is described in Section 10 and
					is identical for intraframes and interframes), there comes a
					Bool(prob_intra), which indicates inter-prediction (i.e., prediction
					from prior frames) when true and intra-prediction (i.e., prediction
					from already-coded portions of the current frame) when false. The
					zero-probability prob_intra is set by field J of the frame header.*/
		if (frames.current_is_key_frame == 0) 
			write_bool(vbe, prob_intra, (frames.e_data[mb_num].is_inter_mb == 1));
		if ((frames.current_is_key_frame == 0) && (frames.e_data[mb_num].is_inter_mb == 1))
		{
		/*  if (is_inter_mb) {                          |       |
        |       mb_ref_frame_sel1                       | B(p)  |*/
			//selects the reference frame to be used; last frame (0), golden/alternate (1)
			int ref = (frames.transformed_blocks[mb_num].reference_frame != LAST);
			write_bool(vbe, prob_last, ref);

		/*      if (mb_ref_frame_sel1)                  |       |
        |           mb_ref_frame_sel2                   | B(p)  |*/
			if (ref == 1) {
				ref = (frames.transformed_blocks[mb_num].reference_frame == ALTREF);
				write_bool(vbe, prob_gf, ref);

			}

        /*      mv_mode                                 |   T   |// determines the macroblock motion vectormode
        |       if (mv_mode == SPLITMV) {               |       |
        |           mv_split_mode                       |   T   |
        |           for (i = 0; i < numMvs; i++) {      |       |
        |               sub_mv_mode                     |   T   |
        |               if (sub_mv_mode == NEWMV4x4) {  |       |
        |                   read_mvcomponent()          |       |
        |                   read_mvcomponent()          |       |
        |               }                               |       |
        |           }                                   |       |
        |       } else if (mv_mode == NEWMV) {          |       |
        |           read_mvcomponent()                  |       |
        |       }                                       |       |*/
			//we'll do all these left inter encodings in distinct function
			bool_encode_inter_mb_modes_and_mvs(vbe, mb_num); 
			// part for only inter over
		}
        /*  } else { /* intra mb                        |       | */
		else if (frames.current_is_key_frame == 1)
		{
        /*      intra_y_mode                            |   T   | */
        /*      if (intra_y_mode == B_PRED) {           |       |
        |           for (i = 0; i < 16; i++)            |       |
        |               intra_b_mode                    |   T   |
        |       }                                       |       | */
        /*      intra_uv_mode                           |   T   |
        |   }                                           |       |  */
			encoding_symbol s_tmp;
			// now encode B_PRED as luma mode
			// prefix kf_ (!!!) for probabilitiess in key frame (and kf_ymode_tree)
			s_tmp.bits = 0; // B_PRED = "0" in spec in kf_mode
			s_tmp.size = 1; 
			write_symbol(vbe, s_tmp, kf_ymode_prob, kf_ymode_tree);
			{ int b_num, ctx1, ctx2, bp, mbp; //"p" for "previous"
			for(b_num = 0; b_num < 16; ++b_num)
			{
				// imaginary blocks outside frame - B_DC_PRED
				if ( (mb_num < video.mb_width) && (b_num < 4) )
					ctx1 = B_DC_PRED;
				else {
					if (b_num < 4) {
						mbp = mb_num - video.mb_width;
						bp = b_num + 12;
					} 
					else {
						mbp = mb_num;
						bp = b_num - 4;
					}
					ctx1 = frames.e_data[mbp].mode[bp];
				}
				if ( ((mb_num % video.mb_width) == 0) && ((b_num & 0x3) == 0) )
					ctx2 = B_DC_PRED;
				else {
					if ((b_num & 0x3) == 0) {
						mbp = mb_num - 1;
						bp = b_num + 3;
					} 
					else {
						mbp = mb_num;
						bp = b_num - 1;
					}
					ctx2 = frames.e_data[mbp].mode[bp];
				}

				// in spec there is a bmode_tree
				// while the tree itself is correct, 
				// HU, HD, VL and LD tokens in comments have one extra "1"
				// so read the trees, not the spec :)
				const int bmode_bits[num_intra_bmodes] = {0, 2, 6, 28, 30, 58, 59, 62, 126, 127};
				const int bmode_size[num_intra_bmodes] = {1, 2, 3,  5,  5,  6,  6,  6,   7,   7};

				s_tmp.bits = bmode_bits[frames.e_data[mb_num].mode[b_num]];
				s_tmp.size = bmode_size[frames.e_data[mb_num].mode[b_num]];
				write_symbol(vbe, s_tmp, kf_bmode_prob[ctx1][ctx2], bmode_tree);
			} }


			// and now TM_PRED as chroma (different bits value!!!);
			s_tmp.bits = 7; //TM_PRED = "111"
			s_tmp.size = 3;
			write_symbol(vbe, s_tmp, kf_uv_mode_prob, uv_mode_tree);
		}
		else // intra MB in inter-frame
		{
			// similar but different context
			encoding_symbol s_tmp;
			// now encode B_PRED as luma mode
			s_tmp.bits = 7; // B_PRED = "111" for P-frames
			s_tmp.size = 3; 
			write_symbol(vbe, s_tmp, new_ymode_prob, ymode_tree);
			cl_int b_num;
			const int bmode_bits[num_intra_bmodes] = {0, 2, 6, 28, 30, 58, 59, 62, 126, 127};
			const int bmode_size[num_intra_bmodes] = {1, 2, 3,  5,  5,  6,  6,  6,   7,   7};
			for(b_num = 0; b_num < 16; ++b_num)
			{
				s_tmp.bits = bmode_bits[frames.e_data[mb_num].mode[b_num]];
				s_tmp.size = bmode_size[frames.e_data[mb_num].mode[b_num]];
				write_symbol(vbe, s_tmp, bmode_prob, bmode_tree);
			}
			// chroma tree is the same for I and P, but probs are different
			s_tmp.bits = 7; //TM_PRED = "111"
			s_tmp.size = 3;
			write_symbol(vbe, s_tmp, new_uv_mode_prob, uv_mode_tree);
		}
        /*  end of macroblock_header() block  */
    } // end of per macroblock cycle

	// at the end of each partition - flush
	flush_bool_encoder(vbe);
	frames.encoded_frame_size = vbe->count + bool_offset;

	//now we put uncompressed data at the first 10 bytes of partition
	/*frame_tag											| f(24) |
	| if (key_frame) {									|		|
	|	start_code										| f(24) |
	|	horizontal_size_code							| f(16) |
	|	vertical_size_code								| f(16) |
	| }													|		| */
	// frame tag : 1 bit = key(0), not-key(1)
	// The start_code is a constant 3-byte pattern having value 0x9d012a. (byte0->2)

	// 0(1) - (not)key | 010 - version2 | 1 - show frame : 0x5 = 0101
	cl_uint buf;
	buf = (frames.current_is_key_frame) ? 0 : 1; /* indicate keyframe via the lowest bit */
	buf |= (0 << 1); /* version 0 in bits 3-1 */
	// version 0 - bicubic interpolation
	// version 1-2 - bilinear
	// version 3 - no interpolation
	// doesn't make difference for decoder's loop filter
	buf |= 0x10; /* this bit indicates that the frame should be shown */
	buf |= (vbe->count << 5); 

	partition[0] = (cl_uchar)(buf & 0xff);
	partition[1] = (cl_uchar)((buf>>8) & 0xff);
	partition[2] = (cl_uchar)((buf>>16) & 0xff);

	if (frames.current_is_key_frame)
	{
		partition[3] = 0x9d;
		partition[4] = 0x01;
		partition[5] = 0x2a;
		// upscaling == 0
		partition[6] = (cl_uchar)(video.dst_width & 0x00FF);
		partition[7] = (cl_uchar)((video.dst_width >> 8) & 0x00FF);
		partition[8] = (cl_uchar)(video.dst_height & 0x00FF);
		partition[9] = (cl_uchar)((video.dst_height >> 8) & 0x00FF);
	}

	return;
}
