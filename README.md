vp8oclenc
=========

upd: 
1) Now codec uses image objects for reference frames. Interpolation is made on the run butstill in sofware (OpenCL doesn't offerbicubic interpolation)
2) GPU code divided in some smaller pieces (dct kernel, idct kernel, wht kernel...) no more long kernels => faster compilation and smaller memory usage.
3) all 3 reference buffers used: ALTREF, GOLDEN, LAST - for search

Some results:
1) memory usage decreased by huge amount. now it fits almost every GPU device.
2) speed a little bit slower (but amount of search done is higher)
3)-cl-opt-disable slows down working with images by 13x times (maybe because there is no option to read 32 bit from one channel image in OpenCL language and only compiler could improve this, maybe not)
4) on E350 (HD6410) performance is veeeeery slow (again because of image usage)

TODO: pure C optimization (LUT instead of IFs, etc...), asm, visual scene changeand effects detection...

main:
Don't know what to write here...
This is a VP8 encoder. Simple and not effective.

Used sources: 
http://www.webmproject.org/; http://multimedia.cx/eggs/category/vp8/;

Uses OpenCL. CPU for coefficient partitions boolean coding.
GPU for motion vector search, transform for inter-frames, interpolation and loop filters.

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


