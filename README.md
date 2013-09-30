vp8oclenc
=========

upd: normal filter now works. Also only involved coefficient probabilities updated now

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
Only one reference frame - last one.
Fixed size of blocks for ME - 8x8
MV search - full search (but for GPU it's not that bad)
Encoding is very straightforward, no crafty methods at all

