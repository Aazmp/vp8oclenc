mkdir ../test & ^
g++ -m64 -I"%AMDAPPSDKROOT%/include" -L"%AMDAPPSDKROOT%/lib/x86_64" vp8enc.cpp entropy_host.cpp -lOpenCL -o ../test/vp8oclenc.exe & ^
cp CPU_kernels.cl ../test/ & ^
cp GPU_kernels.cl ../test/