/*******************************************************************************
Copyright ©2020 Advanced Micro Devices, Inc. All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1   Redistributions of source code must retain the above copyright notice,
this list of conditions and the following disclaimer.
2   Redistributions in binary form must reproduce the above copyright notice,
this list of conditions and the following disclaimer in the
documentation and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF
THE POSSIBILITY OF SUCH DAMAGE.
*******************************************************************************/


__kernel 
__attribute__((reqd_work_group_size(LOCAL_XRES, 1, 1)))
void NaiveMovingAverageFilter(
				    __global float *input,
					size_t numElements,
				    __global float *output,
					__constant float *cFilterWeights)

{
	size_t blockIdx = get_group_id(0);
	size_t blockDimx = get_local_size(0);
	size_t threadIdx = get_local_id(0);

	size_t filterLength = TAP_SIZE;

	long idx = (blockIdx * blockDimx + threadIdx);

    if (idx >= numElements)	
		return;

	long paddedSize = 2 * (numElements + filterLength - 1);
    int halfFilterLength = filterLength / 2;

    long windowIdx = 2*idx;
    float value_r = 0.0f;
	float value_i = 0.0f;
	float fw = 0.0f;
	float sample_r, sample_i;

    for(size_t filterIdx = 0; filterIdx < filterLength; filterIdx++, windowIdx+=2)
    {
		fw = cFilterWeights[filterIdx];

		sample_r = (windowIdx < paddedSize) ? input[windowIdx] : 0.0f;
		sample_i = (windowIdx + 1 < paddedSize) ? input[windowIdx+1] : 0.0f;

		value_r += sample_r * fw;		
		value_i += sample_i * fw;
    }

    output[2*idx] = value_r;
	output[2*idx+1] = value_r;
}


__kernel 
__attribute__((reqd_work_group_size(LOCAL_XRES, 1, 1)))
void MovingAverageFilter(
				    __global float *input,
					size_t numElements,
				    __global float *output,
					__constant float *cFilterWeights)

{
    //int halfFilterLength = filterLength / 2;

	__local float local_Input_r[(LOCAL_XRES + TAP_SIZE - 1)];
	__local float local_Input_i[(LOCAL_XRES + TAP_SIZE - 1)];
	
    size_t threadIdx = get_local_id(0);
	long idx = get_global_id(0);

	size_t filterLength = TAP_SIZE;
	
	
	long paddedSize = 2 * (numElements + filterLength - 1);
    
	
	int local_samples_to_Read = LOCAL_XRES + TAP_SIZE - 1;
	int for_loop_iter = (local_samples_to_Read/LOCAL_XRES);
	int extra_reads = local_samples_to_Read - (for_loop_iter * LOCAL_XRES);

	#pragma unroll 2 //This is 2 for 335 filter size, will be less for small sizes. Curently hardcoding
	for (int i = 0; i < for_loop_iter; i++)	{
		local_Input_r[threadIdx + i * LOCAL_XRES] = (2*((i * LOCAL_XRES) + idx) < paddedSize) ? input[2*((i * LOCAL_XRES) + idx)] : 0.0f;
		local_Input_i[threadIdx + i * LOCAL_XRES] = (2*((i * LOCAL_XRES) + idx) + 1 < paddedSize) ? input[2*((i * LOCAL_XRES) + idx) + 1] : 0.0f;
	}

	if (threadIdx < extra_reads)
    {
		local_Input_r[threadIdx + for_loop_iter*LOCAL_XRES] = (2*((for_loop_iter * LOCAL_XRES) + idx) < paddedSize) ? input[2*((for_loop_iter * LOCAL_XRES) + idx)] : 0.0f;
		local_Input_i[threadIdx + for_loop_iter*LOCAL_XRES] = (2*((for_loop_iter * LOCAL_XRES) + idx) + 1 < paddedSize) ? input[2*((for_loop_iter * LOCAL_XRES) + idx)+ 1] : 0.0f;
    }
	
    barrier(CLK_LOCAL_MEM_FENCE);

    if (idx >= numElements)
		return;

    float value_r = 0.0f;
	float value_i = 0.0f;
	float fw = 0.0f;
	
	#pragma unroll
    for(int sIdx = 0; sIdx < filterLength; sIdx++)
    {
		fw = cFilterWeights[sIdx];
				
		value_r = mad(local_Input_r[threadIdx + sIdx], fw, value_r);
		value_i = mad(local_Input_i[threadIdx + sIdx], fw, value_i);		
    }

	output[2*idx] = value_r;
	output[2*idx + 1] = value_i;
}