vp8oclenc
=========

upd: 
now changуlog in changelog.txt

main:
Don't know what to write here...
This is a VP8 encoder. Simple and not effective.

Used sources: 
http://www.webmproject.org/; http://multimedia.cx/eggs/category/vp8/;

Uses OpenCL. CPU for coefficient partitions boolean coding, loop filter(if CPU is choosen for the task).
GPU for motion vector search, transform for inter-frames, interpolation and loop filters(if GPU is chosen for the task).

Launched only on AMD+AMD+Win7(x32).
And with some changes (\ to /, delte getch() or switch from conio.h to ncurses, delete io.h, delete setmode()) tested on AMD+AMD+Linux 32 and 64.
Working binaries in "bin" with corresponding kernels.
Strange part is:
output files on linux 64 and 32 a little different (less then 2KB difference for 32.6 MB video). Maybe it's because of different precision with float SSIM (for x32 compiler can choose x87 command set and for x64 - SSE2). 

Intra coding is done in usual host code part. Has almost no error checking. 

  -h gives a list of options

If input_file is set as @ it will be set to stdin

Features.
All three reference frames: LAST(updated with each frame), ALTREF(updated with interval set in parameters), GOLDEN (only key).
Motion estimation for 8x8 blocks, but with grouping them into 16x16 if they have equal vectors. 
MV search - hierarchical search with fullsearch in small area on downsampled areas (1/4, 1, 2, 4, 8, 16).
Normal loop filter with loop_filter_level set according to quantizer value. Loop filter could be done on GPU and CPU (CPU is faster on almost all frame sizes, maybe 4K+ would benefit from GPU).
Bicubic interpolation (in OpenCL 2D images but is software itself).
Used probabilities are calculated and set in each frame.

P.S. No benchmarks, because there is no need in them :) Quality of material can't compete with any good encoder.


