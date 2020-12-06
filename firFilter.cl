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
				    __global float *input_r,
					__global float *input_i,
					size_t numElements,
				    __global float *output_r,
					__global float *output_i,
					__constant float *cFilterWeights, 
					size_t filterLength)

{
	size_t blockIdx = get_group_id(0);
	size_t blockDimx = get_local_size(0);
	size_t threadIdx = get_local_id(0);

    size_t idx = blockIdx * blockDimx + threadIdx;

    if (idx >= numElements)
		return;

    size_t halfFilterLength = filterLength / 2;

    size_t windowIdx = idx - halfFilterLength;
    float  value_r = 0.0f;
	float  value_i = 0.0f;
	float fw = 0.0f;
	float sample_r, sample_i;

    for(size_t filterIdx = 0; filterIdx < filterLength; filterIdx++, windowIdx++)
    {
		fw = cFilterWeights[filterIdx];

		sample_r = (0 <= windowIdx && windowIdx < numElements) ? input_r[windowIdx] : 0.0f;
		value_r += sample_r * fw;

		sample_i = (0 <= windowIdx && windowIdx < numElements) ? input_i[windowIdx] : 0.0f;
		value_i += sample_i * fw;
    }

    output_r[idx] = value_r;
	output_i[idx] = value_i;
}


__kernel 
__attribute__((reqd_work_group_size(LOCAL_XRES, 1, 1)))
void MovingAverageFilter(
				    __global float *input_r,
					__global float *input_i,
					size_t numElements,
				    __global float *output_r,
					__global float *output_i,
					__constant float *cFilterWeights, 
					size_t filterLength)

{
    size_t halfFilterLength = filterLength / 2;

	__local float sInput_r[(LOCAL_XRES + TAP_SIZE -1)];
	__local float sInput_i[(LOCAL_XRES + TAP_SIZE -1)];

    size_t blockIdx = get_group_id(0);
	size_t blockDimx = get_local_size(0);
	size_t threadIdx = get_local_id(0);

    size_t idx = blockIdx * blockDimx + threadIdx;

    sInput_r[threadIdx + halfFilterLength] = (idx < numElements) ? input_r[idx] : 0.0f;
	sInput_i[threadIdx + halfFilterLength] = (idx < numElements) ? input_i[idx] : 0.0f;

	if (threadIdx < halfFilterLength)
    {
		sInput_r[threadIdx] = ((int)idx - (int)halfFilterLength >= 0) ? input_r[idx - halfFilterLength] : 0.0f;
		sInput_i[threadIdx] = ((int)idx - (int)halfFilterLength >= 0) ? input_i[idx - halfFilterLength] : 0.0f;
    }

    if (threadIdx >= blockDimx - halfFilterLength)
    {
		sInput_r[threadIdx + 2*halfFilterLength] = (idx + halfFilterLength < numElements) ? input_r[idx + halfFilterLength] : 0.0f;
		sInput_i[threadIdx + 2*halfFilterLength] = (idx + halfFilterLength < numElements) ? input_i[idx + halfFilterLength] : 0.0f;
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    if (idx >= numElements)
		return;

    float value_r = 0.0f;
	float value_i = 0.0f;
	float fw = 0.0f;
    for(size_t sIdx = 0; sIdx < filterLength; sIdx++)
    {
		fw = cFilterWeights[sIdx];
		value_r += fw * sInput_r[threadIdx + sIdx];
		value_i += fw * sInput_i[threadIdx + sIdx];
    }

    output_r[idx] = value_r;
	output_i[idx] = value_i;
}