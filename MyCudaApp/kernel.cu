
#ifndef CUDACC
#define CUDACC
#endif
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda.h>
#include <device_functions.h>
#include <cuda_runtime_api.h>
#include <stdio.h>
#include <stdlib.h>


typedef unsigned int NUMBER;

const int RADIUS = 3;
const NUMBER N = 2048 * 2048;
const int THREADS_PER_BLOCK = 1024;
const int BLOCKS = N / THREADS_PER_BLOCK;

__global__ void stencil_1d(NUMBER* in, NUMBER* out) {
	__shared__ NUMBER temp[BLOCKS + (2 * RADIUS)];
	NUMBER gindex = threadIdx.x + blockIdx.x * blockDim.x;
	NUMBER lindex = threadIdx.x + RADIUS;

	if (gindex < N)
	{
		temp[lindex] = in[gindex];

		if (threadIdx.x < RADIUS) {
			temp[lindex - RADIUS] = in[lindex - RADIUS];
			temp[lindex + THREADS_PER_BLOCK] = in[gindex + THREADS_PER_BLOCK];
		}

		__syncthreads();

		NUMBER result = 0;
		for (NUMBER offset = -RADIUS; offset <= RADIUS; offset++) {
			result += temp[lindex + offset];
		}

		out[gindex] = result;
	}

}

void print_stencil(int* v) {
	printf("[");
	for (int i = 0; i < N; i++) {
		if (i != N - 1) {
			printf("%d,", v[i]);
		}
		else {
			printf("%d", v[i]);
		}
	}
	printf("]");
}

int main()
{
	NUMBER* h_in, * h_out, * d_in, * d_out;

	h_in = (NUMBER*)malloc(N * sizeof(NUMBER));
	h_out = (NUMBER*)malloc(N * sizeof(NUMBER));

	for (NUMBER i = 0; i < N; i++) {
		h_in[i] = 1;
	}

	cudaEvent_t start = cudaEvent_t();
	cudaEvent_t stop = cudaEvent_t();
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

	cudaMalloc((void**)&d_in, N * sizeof(NUMBER));
	cudaMalloc((void**)&d_out, N * sizeof(NUMBER));

	cudaMemcpy(d_in, h_in, N * sizeof(NUMBER), cudaMemcpyHostToDevice);
	cudaMemcpy(d_out, h_out, N * sizeof(NUMBER), cudaMemcpyHostToDevice);

	stencil_1d<<<BLOCKS, THREADS_PER_BLOCK>>>(d_in, d_out);

	cudaDeviceSynchronize();
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	float time;
	cudaEventElapsedTime(&time, start, stop);
	printf("Time elapsed on CPU: %f ms.\n", time);

	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	return 0;
}





