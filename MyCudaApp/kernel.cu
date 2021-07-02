
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>


#define RADIUS 3
#define GRIDSIZE 2
#define BLOCKSIZE 512
#define THREAD_PER_BLOCK = 32;

typedef unsigned long int NUMBER;
const NUMBER N = 99999999;

__global__ void stencil_1d(NUMBER* in, NUMBER* out) {
	__shared__ NUMBER temp[BLOCKSIZE + 2 * RADIUS];
	NUMBER gindex = threadIdx.x + blockIdx.x * blockDim.x;
	NUMBER lindex = threadIdx.x + RADIUS;

	temp[lindex] = in[gindex];

	if (threadIdx.x < RADIUS) {
		temp[lindex - RADIUS] = in[lindex - RADIUS];
		temp[lindex + BLOCKSIZE] = in[gindex + BLOCKSIZE];
	}

	__syncthreads();

	NUMBER result = 0;
	for (NUMBER offset = -RADIUS; offset <= RADIUS; offset++) {
		result += temp[lindex + offset];
	}

	out[gindex] = result;
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

	cudaMalloc((void**)&d_in, N * sizeof(NUMBER));
	cudaMalloc((void**)&d_out, N * sizeof(NUMBER));

	cudaMemcpy(d_in, h_in, N * sizeof(NUMBER), cudaMemcpyHostToDevice);
	cudaMemcpy(d_out, h_out, N * sizeof(NUMBER), cudaMemcpyHostToDevice);

	cudaEventRecord(start,0);

	stencil_1d<<<GRIDSIZE, BLOCKSIZE>>>(d_in, d_out);
	cudaDeviceSynchronize();
	cudaEventRecord(stop,0);
	cudaEventSynchronize(stop);
	float time;
	cudaEventElapsedTime(&time, start, stop);
	printf("Time elapsed on CPU: %f ms.\n", time);

	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	return 0;
}





