vp8oclenc
=========

upd:
returned bilinear interpolation filter -> it work's
bicubic filter in the code too -> luma seems to work, but chroma produces traces on moving black edge (with 1..7 fractal vector part)

to enable bicubic filter: 
1. change kernel string in clCreateKernel (suffix _bl for bilinear; _bc for bicubic) : init.h :: init_all()
2. change version number (0 - bc; 1,2 - bl) in entropy_host.c::encode_header();


main:

Don't know what to write here...
...

This is a VP8 encoder.
Simple and not effective. Just a toy project.

Used (and copied :)) sources: 
http://www.webmproject.org/; http://multimedia.cx/eggs/category/vp8/;

Uses OpenCL:
CL_DEVICE_TYPE_CPU - for coefficient partitions boolean coding.

CL_DEVICE_TYPE_GPU - for motion vector search, transform for inter-frames, interpolation and loop filters.

Launched only on AMD+AMD+Win7.

Intra coding is done in usual host code part.
Has almost no error checking. 

  -h gives a list of options

If input_file is set as @ it will be set to stdin

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

