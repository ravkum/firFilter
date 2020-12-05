//#include <hip/hip_runtime.h>
#include <iostream>
#include <random>
#include <time.h>

#include "utils.h"

#define FIR_FILTER_KERNEL_SOURCE	"firFilter.cl"
#define NAIVE_FIR_FILTER_KERNEL			"NaiveMovingAverageFilter"
#define OPT_FIR_FILTER_KERNEL			"MovingAverageFilter"
#define LOCAL_XRES					256

// Helper function to generate test data for this exercise
void GenerateTestData(size_t const numElements,
		      std::vector<float> const& filterWeights,
		      std::vector<float>& testData,
		      std::vector<float>& reference, bool verify)
{
    testData.resize(numElements);
    reference.resize(numElements);

    std::default_random_engine            generator;
    std::uniform_real_distribution<float> uniform(0.0f, 1.0f);

    for(size_t i = 0; i < numElements; i++)
    {
		testData[i] = uniform(generator);
    }

	if (verify) {
		size_t const filterLength = filterWeights.size();
		size_t const halfFilterLength = filterLength / 2;

		for (size_t i = 0; i < numElements; i++)
		{
			size_t windowIdx = i - halfFilterLength;
			float  value = 0.0f;

			for (size_t filterIdx = 0; filterIdx < filterLength; filterIdx++, windowIdx++)
			{
				float sample = (0 <= windowIdx && windowIdx < numElements) ? testData[windowIdx] : 0.0f;
				value += sample * filterWeights[filterIdx];
			}

			reference[i] = value;
		}
	}
}

bool AlmostEqual(float ref, float value, size_t ulp)
{
    return std::abs(ref-value) <= std::numeric_limits<float>::epsilon() * std::abs(ref) * ulp;
}

// Helper function to compare test data for this exercise
void CompareData(std::vector<float> const& expected, std::vector<float> const& actual, int toleranceInUlp)
{
    size_t const numElements = expected.size();

    bool pass = true;
    for(size_t i = 0; i < numElements; i++)
    {
        if(!AlmostEqual(expected[i], actual[i], toleranceInUlp))
        {
            std::cout << "Mismatch at index " << i << "!\nExpected value: " << expected[i]
                      << "\nActual value: " << actual[i] << std::endl;
            pass = false;
            break;
        }
    }

    if(pass)
        printf("Test Passed! All values match.\n");
}

float cFilterWeights[1024];

#if 0
template <int filterLength>
__global__ void NaiveMovingAverageFilter(size_t const numElements,
					 float const* const __restrict__ input,
					 float      * const __restrict__ output)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= numElements)
	return;

    int const halfFilterLength = filterLength / 2;

    size_t windowIdx = idx - halfFilterLength;
    float  value = 0.0f;

    for(size_t filterIdx = 0; filterIdx < filterLength; filterIdx++, windowIdx++)
    {
	float sample = (0 <= windowIdx && windowIdx < numElements) ? input[windowIdx] : 0.0f;
	value += sample * cFilterWeights[filterIdx];
    }
    output[idx] = value;
}


template <int filterLength>
__global__ void MovingAverageFilter(size_t const numElements,
				    float const* const __restrict__ input,
				    float      * const __restrict__ output)
{
    int const halfFilterLength = filterLength / 2;

    HIP_DYNAMIC_SHARED(float, sInput);

    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    sInput[threadIdx.x + halfFilterLength] = (idx < numElements) ? input[idx] : 0.0f;

    if (threadIdx.x < halfFilterLength)
    {
	sInput[threadIdx.x] = (idx - halfFilterLength > 0) ? input[idx - halfFilterLength] : 0.0f;
    }

    if (threadIdx.x >= blockDim.x - halfFilterLength)
    {
	sInput[threadIdx.x + 2*halfFilterLength] = (idx + halfFilterLength< numElements) ? input[idx + halfFilterLength] : 0.0f;
    }

    __syncthreads();

    size_t windowIdx = idx - halfFilterLength;

    if (idx >= numElements)
	return;

    float value = 0.0f;
    for(int sIdx = 0; sIdx < filterLength; sIdx++)
    {
	value += cFilterWeights[sIdx]*sInput[threadIdx.x + sIdx];
    }
    output[idx] = value;
}
#endif

bool buildKernels(cl_context oclContext, cl_device_id oclDevice, cl_kernel *naiveFirFilterKernel, cl_kernel *optFirFilterKernel, cl_uint filterLength)
{
	cl_int err = CL_SUCCESS;

	/**************************************************************************
	* Read kernel source file into buffer.                                    *
	**************************************************************************/
	const char *filename = FIR_FILTER_KERNEL_SOURCE;
	char *source = NULL;
	size_t sourceSize = 0;
	err = convertToString(filename, &source, &sourceSize);
	CHECK_RESULT(err != CL_SUCCESS, "Error reading file %s ", filename);

	/**************************************************************************
	* Create kernel program.                                                  *
	**************************************************************************/
	cl_program programFirFilter = clCreateProgramWithSource(oclContext, 1, (const char **)&source, (const size_t *)&sourceSize, &err);
	CHECK_RESULT(err != CL_SUCCESS, "clCreateProgramWithSource failed with Error code = %d", err);

	/**************************************************************************
	* Build the kernel and check for errors. If errors are found, it will be  *
	* printed to console                                                      *
	**************************************************************************/
	char option[256];
	sprintf(option, "-DTAP_SIZE=%d -DLOCAL_XRES=%d", filterLength, LOCAL_XRES);

	err = clBuildProgram(programFirFilter, 1, &(oclDevice), option, NULL, NULL);
	free(source);
	if (err != CL_SUCCESS)
	{
		char *buildLog = NULL;
		size_t buildLogSize = 0;
		clGetProgramBuildInfo(programFirFilter, oclDevice,
			CL_PROGRAM_BUILD_LOG, buildLogSize, buildLog,
			&buildLogSize);
		buildLog = (char *)malloc(buildLogSize);
		clGetProgramBuildInfo(programFirFilter, oclDevice,
			CL_PROGRAM_BUILD_LOG, buildLogSize, buildLog, NULL);
		printf("%s\n", buildLog);
		free(buildLog);
		clReleaseProgram(programFirFilter);
		CHECK_RESULT(true, "clCreateProgram failed with Error code = %d", err);
	}

	/**************************************************************************
	* Create kernel                                                           *
	**************************************************************************/
	*naiveFirFilterKernel = clCreateKernel(programFirFilter, NAIVE_FIR_FILTER_KERNEL, &err);
	CHECK_RESULT(err != CL_SUCCESS, "clCreateKernel failed with Error code = %d", err);

	*optFirFilterKernel = clCreateKernel(programFirFilter, OPT_FIR_FILTER_KERNEL, &err);
	CHECK_RESULT(err != CL_SUCCESS, "clCreateKernel failed with Error code = %d", err);

	clReleaseProgram(programFirFilter);
	return true;
}

void usage(const char *prog)
{
	printf("Usage: %s \n\t", prog);
	printf("\n\t[-zeroCopy (0 | 1)] //0 (default) - Device buffer, 1 - zero copy buffer\n\t[-filtSize (Tap size)]\n\t");
	printf("\n\t[-numElements (220000)]\n\t");
	printf("\n\t[-opt (0 | 1)]\n\t");
	printf("\n\t[-batchSize (number of blocks per batch: how much data to be processed in an interval)]\n\t");
	printf("\n\t[-verify (0 | 1]\n\t");
	printf("]\n\t[-h (help)]\n\n");
}

int main(int argc, char **argv)
{
	DeviceInfo infoDeviceOcl;
	bool zeroCopy = true;
	bool dataTransfer = !zeroCopy;
	int iteration = 1;

	cl_mem inputSignal, outputSignal, filterData;
	cl_int err;
	cl_kernel naiveFirFilterKernel, optFirFilterKernel;
	cl_kernel *kernel;
	bool useOptKernel = true;

	size_t filterLength = 335;
	size_t numElements = 2200000;// 600 * 1024 * 1024;

	int x1 = sizeof(size_t);
	int x2 = sizeof(float);

	bool verify = false;

	std::vector<float> data;
	std::vector<float> reference;
	std::vector<float> result(numElements, 0.0f);

	/***************************************************************************
	* Processing the command line arguments                                   *
	**************************************************************************/

	int tmpArgc = argc;
	char **tmpArgv = argv;

	while (tmpArgv[1] && tmpArgv[1][0] == '-')
	{
		if (strncmp(tmpArgv[1], "-filtSize", 9) == 0)
		{
			tmpArgv++;
			tmpArgc--;
			filterLength = atoi(tmpArgv[1]);
		}
		else if (strncmp(tmpArgv[1], "-zeroCopy", 9) == 0)
		{
			tmpArgv++;
			tmpArgc--;
			zeroCopy = atoi(tmpArgv[1]);
		}
		else if (strncmp(tmpArgv[1], "-numElements", 10) == 0)
		{
			tmpArgv++;
			tmpArgc--;
			numElements = atoi(tmpArgv[1]);
		}
		else if (strncmp(tmpArgv[1], "-verify", 7) == 0)
		{
			tmpArgv++;
			tmpArgc--;
			verify = atoi(tmpArgv[1]);
		}
		else if (strncmp(tmpArgv[1], "-opt", 4) == 0)
		{
			tmpArgv++;
			tmpArgc--;
			useOptKernel = atoi(tmpArgv[1]);
		}
		else if (strncmp(tmpArgv[1], "-iter", 5) == 0)
		{
			tmpArgv++;
			tmpArgc--;
			iteration = atoi(tmpArgv[1]);
		}
		else if (strncmp(tmpArgv[1], "-h", 2) == 0)
		{
			usage(argv[0]);
			exit(1);
		}

		else
		{
			printf("Illegal option %s ignored\n", tmpArgv[1]);
			usage(argv[0]);
			exit(1);
		}
		tmpArgv++;
		tmpArgc--;
	}

	if (tmpArgc > 1)
	{
		usage(argv[0]);
		exit(1);
	}

	dataTransfer = !zeroCopy;

	std::vector<float> filterWeights(filterLength, 1.0 / filterLength); // Simple average

	GenerateTestData(numElements, filterWeights, data, reference, verify);

	size_t numBytes = numElements * sizeof(float);

	if (initOpenCl(&infoDeviceOcl, 0) == false)
	{
		printf("Error in initOpenCl.\n");
		return false;
	}

	if (zeroCopy)
	{
		inputSignal = clCreateBuffer(infoDeviceOcl.mCtx, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
			numBytes,
			&data[0], &err);
		CHECK_RESULT(err != CL_SUCCESS, "clCreateBuffer failed with %d\n", err);

		outputSignal = clCreateBuffer(infoDeviceOcl.mCtx, CL_MEM_WRITE_ONLY | CL_MEM_USE_HOST_PTR,
			numBytes,
			&result[0], &err);
		CHECK_RESULT(err != CL_SUCCESS, "clCreateBuffer failed with %d\n", err);

		filterData = clCreateBuffer(infoDeviceOcl.mCtx, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
			filterLength * sizeof(float),
			&filterWeights[0], &err);
		CHECK_RESULT(err != CL_SUCCESS, "clCreateBuffer failed with %d\n", err);
	}
	else
	{
		inputSignal = clCreateBuffer(infoDeviceOcl.mCtx, CL_MEM_READ_ONLY,
			numBytes,
			NULL, &err);
		CHECK_RESULT(err != CL_SUCCESS, "clCreateBuffer failed with %d\n", err);

		outputSignal = clCreateBuffer(infoDeviceOcl.mCtx, CL_MEM_WRITE_ONLY,
			numBytes,
			NULL, &err);
		CHECK_RESULT(err != CL_SUCCESS, "clCreateBuffer failed with %d\n", err);

		filterData = clCreateBuffer(infoDeviceOcl.mCtx, CL_MEM_READ_ONLY,
			filterLength * sizeof(float),
			NULL, &err);
		CHECK_RESULT(err != CL_SUCCESS, "clCreateBuffer failed with %d\n", err);
	}

	/////////////////////
	if (!buildKernels(infoDeviceOcl.mCtx, infoDeviceOcl.mDevice, &naiveFirFilterKernel, &optFirFilterKernel, filterLength))
	{
		printf("Error in buildKernels.\n");
		return false;
	}

	kernel = useOptKernel ? &optFirFilterKernel : &naiveFirFilterKernel;

	//Set kernel argument
	int cnt = 0;
	err = clSetKernelArg(naiveFirFilterKernel, cnt++, sizeof(cl_mem), &(inputSignal));
	err |= clSetKernelArg(naiveFirFilterKernel, cnt++, sizeof(size_t), &(numElements));
	err |= clSetKernelArg(naiveFirFilterKernel, cnt++, sizeof(cl_mem), &(outputSignal));
	err |= clSetKernelArg(naiveFirFilterKernel, cnt++, sizeof(cl_mem), &(filterData));
	err |= clSetKernelArg(naiveFirFilterKernel, cnt++, sizeof(size_t), &(filterLength));
	CHECK_RESULT(err != CL_SUCCESS, "clSetKernelArg failed with Error code = %d", err);


	cnt = 0;
	err = clSetKernelArg(optFirFilterKernel, cnt++, sizeof(cl_mem), &(inputSignal));
	err |= clSetKernelArg(optFirFilterKernel, cnt++, sizeof(size_t), &(numElements));
	err |= clSetKernelArg(optFirFilterKernel, cnt++, sizeof(cl_mem), &(outputSignal));
	err |= clSetKernelArg(optFirFilterKernel, cnt++, sizeof(cl_mem), &(filterData));
	err |= clSetKernelArg(optFirFilterKernel, cnt++, sizeof(size_t), &(filterLength));
	CHECK_RESULT(err != CL_SUCCESS, "clSetKernelArg failed with Error code = %d", err);


	size_t localWorkSize[3] = { LOCAL_XRES, 1, 1 };
	size_t globalWorkSize[3] = { 1, 1, 1 };

	globalWorkSize[0] = numElements;

	cl_int status;


#if 1
	//warm-up run
	err = clEnqueueNDRangeKernel(infoDeviceOcl.mQueue, *kernel, 1, NULL,
		globalWorkSize, localWorkSize, 0, NULL, NULL);
	CHECK_RESULT(err != CL_SUCCESS,
		"clEnqueueNDRangeKernel failed with Error code = %d", err);
	clFinish(infoDeviceOcl.mQueue);
#endif

	timer t_timer;
	timerStart(&t_timer);

	for (int iter = 0; iter < iteration; iter++)
	{
		//Perf test
		if (dataTransfer)
		{
			/**************************************************************************
			* Send the input image data to the device
			***************************************************************************/
			status = clEnqueueWriteBuffer(infoDeviceOcl.mQueue, inputSignal,
				CL_FALSE, 0, numBytes, &data[0], 0,
				NULL, NULL);
			CHECK_RESULT(status != CL_SUCCESS,
				"Error in clEnqueueWriteBuffer. Status: %d\n", status);

			/**************************************************************************
			* Send the filters to the device
			***************************************************************************/
			status = clEnqueueWriteBuffer(infoDeviceOcl.mQueue,
				filterData, CL_FALSE, 0, filterLength * sizeof(float),
				&filterWeights[0], 0, NULL, NULL);
			CHECK_RESULT(status != CL_SUCCESS,
				"Error in clEnqueueWriteBuffer. Status: %d\n", status);
		}


		err = clEnqueueNDRangeKernel(infoDeviceOcl.mQueue, *kernel, 1, NULL,
			globalWorkSize, localWorkSize, 0, NULL, NULL);
		CHECK_RESULT(err != CL_SUCCESS,
			"clEnqueueNDRangeKernel failed with Error code = %d", err);

		if (dataTransfer)
		{
			/**************************************************************************
			* Send the input image data to the device
			***************************************************************************/
			status = clEnqueueReadBuffer(infoDeviceOcl.mQueue, outputSignal,
				CL_FALSE, 0, numBytes, &result[0], 0,
				NULL, NULL);
			CHECK_RESULT(status != CL_SUCCESS,
				"Error in clEnqueueWriteBuffer. Status: %d\n", status);
		}
		clFinish(infoDeviceOcl.mQueue);
	}
	//clFinish(infoDeviceOcl.mQueue);

	double time_ms = timerCurrent(&t_timer);
	time_ms = 1000 * time_ms;

	if (dataTransfer)
		printf("Average time taken per iteration with data transfer: %f msec\n", time_ms/iteration);
	else
		printf("Average time taken per iteration using zero-copy buffer: %f msec\n", time_ms/iteration);

	double throughput = (numBytes * 1000.0f *iteration / (time_ms * 1024 * 1024));
	printf("throughput: %f Mbps\n", throughput);

	if (verify)
		CompareData(reference, result, filterWeights.size());

	//////////////////////
	//Clean up memory
	//////////////////////
	clReleaseMemObject(inputSignal);
	clReleaseMemObject(outputSignal);
	clReleaseMemObject(filterData);
	
#if 0
    HIP_CHECK(hipMalloc(&dData,   numBytes));
    HIP_CHECK(hipMalloc(&dResult, numBytes));

    HIP_CHECK(hipMemcpy(dData, data.data(), numBytes, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpyToSymbol(HIP_SYMBOL(cFilterWeights),
			       filterWeights.data(),
			       filterWeights.size() * sizeof(float)));

    dim3 blockDim(256, 1, 1);
    dim3 gridDim((numElements + blockDim.x - 1) / blockDim.x, 1, 1);

    hipLaunchKernelGGL(NaiveMovingAverageFilter<filterLength>,
		       gridDim,
		       blockDim,
		       0,
		       0,
		       numElements,
		       dData,
		       dResult);

    HIP_CHECK(hipDeviceSynchronize());

    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipGetLastError());

    HIP_CHECK(hipMemcpy(result.data(), dResult, numBytes, hipMemcpyDeviceToHost));

    CompareData(reference, result, filterWeights.size());

    HIP_CHECK(hipMemset(dResult, 0, numBytes));

    size_t dynamicSharedMemoryBytes = sizeof(float) * (blockDim.x + filterWeights.size() - 1);

    hipLaunchKernelGGL(MovingAverageFilter<filterLength>,
		       gridDim,
		       blockDim,
		       dynamicSharedMemoryBytes,
		       0,
		       numElements,
		       dData,
		       dResult);

    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipGetLastError());

    HIP_CHECK(hipMemcpy(result.data(), dResult, numBytes, hipMemcpyDeviceToHost));
    CompareData(reference, result, filterWeights.size());

    HIP_CHECK(hipFree(dData));
    HIP_CHECK(hipFree(dResult));
#endif
    return 0;
}
