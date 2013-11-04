vp8oclenc
=========

upd: 
0) There was a shameful bug in reading ARGV[] part. Also some other bug there was. But not anymore.
1) Added segmentation for inter macroblocks too. 
Now it tries to quant with highest quantizer and if result fails to pass SSIM-target value it's being requantized.
2) Added ALTREF referencing. 
Every <-altref-range param> encoder quantize a frame with lower quantizers and saves it as ALTREF buffer.
ALTREF frames are being predicted from previous ALTREF frames. Others from LAST.

TODO:
1) Interpolation. (on HD7850 ~1/4 of encoding time is spent here)
Consumes a lot(!) of memory because of keeping interpolated buffers. 
Threads are not distributed across stream cores effectively (one row per core).
=> solution: integrate interpolation into motion vector search

2) Make prediction from both LAST, GOLDEN and ALTREF frames for each frame and block.
But for this 1) must be done (otherwise it will be either very slow or consume a huge lot of memory)

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


