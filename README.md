vp8oclenc
=========

upd: 
now changlog on changelog.txt

main:
Don't know what to write here...
This is a VP8 encoder. Simple and not effective.

Used sources: 
http://www.webmproject.org/; http://multimedia.cx/eggs/category/vp8/;

Uses OpenCL. CPU for coefficient partitions boolean coding.
GPU for motion vector search, transform for inter-frames, interpolation and loop filters.

Launched only on AMD+AMD+Win7(x32).
And with some changes (\ to /, delte getch() or switch from conio.h to ncurses, delete io.h, delete setmode()) tested on AMD+AMD+Linux 32 and 64.
Strange part is:
output files on 64 and 32 a little different (less then 2KB difference for 32.6 MB video)

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


