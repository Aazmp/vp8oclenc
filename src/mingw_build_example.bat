mkdir ../test & ^
g++ -m64 -I"C:/Program Files (x86)/AMD APP SDK/2.9/include" -L"C:/Program Files (x86)/AMD APP SDK/2.9/lib/x86_64" vp8enc.cpp entropy_host.cpp -lOpenCL -o ../test/vp8oclenc.exe & ^
cp CPU_kernels.cl ../test/ & ^
cp GPU_kernels.cl ../test/