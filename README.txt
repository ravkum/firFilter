# firFilter
FIR Filter implementation on GPU using OpenCL

FIR Filter:
  𝑦[𝑖]=∑((𝑥[𝑖−(𝑀/2)+ 𝑗] . 𝑎[𝑗]) 
	  for j ranging from 0 to M-1

  x: Input array and 𝑥[𝑖-(𝑀/2)+𝑗] = 0 if 𝑖−(𝑀/2)+ 𝑗 < 0 or 𝑖-(M/2)+𝑗 > length of 𝑥
  a: Filter array
  M: Filter TAP size
  y: Output array

Note:

The sample assumes input data is commplex array with i and q components being fp32 values. SO every sample is total 8 bytes long.

1) The "master" branch implements a simple FIR filter. 
2) The "filter_pipeline_interleaved" branch implements two FIR filters in a pipeline. The second filter upscales the first filter output by a factor of 2. The upscaling
is done in the kernel to avoid extra global memory read/writes. 


Prerequisite:
1) OpenCL header and lib
2) export AMDAPPSDKROOT=/opt/rocm 
    or to any path where the OpenCL SDK is present.

Build:
1) Linux:
  mkdir build
  cd build
  cmake ../
  make clean all
  
2) Windows:
  Open the .sln fileon Visusl Studio and build. 
  
 
Run:
Usage: firfilter.exe

        [-zeroCopy (0 | 1)] //0 (default) - Device buffer, 1 - zero copy buffer
        [-filtSize (Tap size)]
        [-numElements (220000)]
	[-iter (10) number of times the pipeline should run]
        [-verify (0 | 1]
        [-h (help)]

Example:

                firFilter.exe -numElements 256000 -filtSize 335 -verify 1 -iter 10