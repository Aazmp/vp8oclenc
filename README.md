vp8oclenc
=========

upd: 1) added holden frame referencing. Now for each P-frame it search for vector best in LAST buffer and compare it to (0;0 - displacement) macroblock in GOLDEN. 
Only key frames are set as GOLDEN now. Gives much better quality at static background with moving object over it. And little less bits to encode.
2) Vector search still for 8x8 pixel blocks, but macroblocks with four equal vectors are not partitioned any more (not split_mv mode). Gives a little bit less bits and little higher quality when DC-residual is present(quantizing of WHT a little bit more sensitive).
3) Deleted bilinear interpolation filter and simple loop filter.

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

Features.
Two reference frames, but only one is search for vectors, the other one uses ZEROMV only.
Motion estimation for 8x8 blocks, but with grouping them into 16x16 if they have equal vectors. 
MV search - hierarchical search with fullsearch in small area on downsampled areas (1/4, 1, 2, 4, 8, 16).
Normal loop filter with loop_filter_level set according to quantizer value.
Bicubic interpolation.
Used probabilities are calculated and set in each frame.

P.S. No benchmarks, because there is no need in them :) Quality of material can't compete with any good encoder.
No adaptive quant even on frame level.


