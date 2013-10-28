vp8oclenc
=========

upd: 
1) Added segmentation (now it's used only on intrA MB in intEr frame).
-qi -qp options deleted; 
-qmin -qmax added (set a range for Y AC index quantizer); 
for intEr frame UV indexes are set as (Y_AC_index-15) and Y_DC_index =(Y_AC_index+15);
intrA frames are not segmented;
intEr MBs use last 4th [segment index 3] segment.
2) deleted pipelining bool_coder with GPU computings - now it converts frame after frame with no overlapping.
(makes easier to improve)
other) some code rewritings, some bugs fixed, etc

TODO:
1) Interpolation. (on HD7850 ~1/4 of encoding time is spent here)
Consumes a lot(!) of memory because of keeping interpolated buffers. 
Threads are not distributed across stream cores effectively (one row per core).
=> solution: integrate interpolation into motion vector search

2) More segmentation. 
Maybe variance based (just don't know what variance values should it be based on)
Maybe quality based (redo macroblocks with bad SSIM with lower quantizer index) 
Both a easy to implement in code, but i'm lazy at the moment

main:
Don't know what to write here...
This is a VP8 encoder. Simple and not effective.

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


