vp8oclenc
=========

upd: full search has been replaced with hierarchical search - lowest limit of bitrate become lower, but still is rather high(just bitrate, not quality :) ). 
vx,vy,ll options removed. 
Filter level is set according to quantizer value for a frame;

main:

Don't know what to write here...
...

This is a VP8 encoder.
Simple and not effective.

Used (and copied :)) sources: 
http://www.webmproject.org/; http://multimedia.cx/eggs/category/vp8/;

Uses OpenCL. CPU for coefficient partitions boolean coding.GPU for motion vector search, transform for inter-frames, interpolation and loop filters.

Launched only on AMD+AMD+Win7.

Intra coding is done in usual host code part. Has almost no error checking. 

  -h gives a list of options

If input_file is set as @ it will be set to stdin

Features:
Only one reference frame - last one;
Fixed size of blocks for ME - 8x8;
MV search - hierarchical search with fullsearch in small areal on downsampled areas (1/4, 1, 2, 4, 8, 16);
Normal loop filter (simple is present in code) with loop_filter_level set according to uantizer value;
Bicubic interpolation (bilinear is in code too);
Used probabilities are calculated and set in each frame.

P.S. No benchmarks, because there is no need in them :) Quality of material can't compete with any good encoder.
No adaptive quant even on frame level.


