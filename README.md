vp8oclenc
=========

Chroma interpolation is broken... Trying to find out why...


Don't know what to write here...
...

This is a VP8 encoder.
Simple and not effective. Just a toy project.

Used (and copied :)) sources: 
http://www.webmproject.org/; http://multimedia.cx/eggs/category/vp8/;

pure C. Cpp just as extension to make Visual Studio compile not in C89....

Uses OpenCL:
CL_DEVICE_TYPE_CPU - for coefficient partitions boolean coding.

CL_DEVICE_TYPE_GPU - for motion vector search and transform for inter-frames.
Also GPU does simple "simple" loop filter, but it does only harm (may be because of errors in code).

Launched only on AMD+AMD+Win7.

Intra coding is done in usual host code part.

Has almost no error checking. 

Command line options:

-i input file = YUV4MPEG2 with 420 chroma sub-sampling

-o output file = IVF file

-g value = GOP size

-qi value = Constant quantizer INDEX for all I-frames (UV has qi minus 15)

-qp value = Same for P-frames

-vx value = dX(collumns) limit for MV search (in quarter pixel). Should be lower than 512

-vy value = dY(rows) -//-

-lt value = filter type (0 - normal, 1 - simple)

-ll value = Loop filter level (0 - disabled)

-ls value = Loop filter sharpness

-w value = Number of GPU device threads (the more the better)

-t value = Number of CPU device(!) threads (also the number of partitions)

If input_file is set as @
it will be set to stdin

... sorry for bad English and bad programming

P.S. 
No benchmarks, because there is no need in them :)
Quality of material can't compete with any good encoder of any standard.
It's just bad now.
No adaptive quant even on frame level.
No skipping macroblocks.
Fixed GOP size, only one reference frame - last one.
Fixed size of blocks for ME - 8x8
MV search - full search (but for GPU it's not that bad)
Encoding is very straightforward, no crafty methods at all.
Just a toy :)

