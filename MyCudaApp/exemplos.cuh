
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda.h>
#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <omp.h>


__global__ void sum(int* a, int* b, int* c) {
	*c = *a + *b;
}

__global__ void vector_sum(int* a, int* b, int* c, int n) {
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	if (n > index) {
		c[index] = a[index] + b[index];
	}
}

void exemplo01() {
	int a, b, c;
	int* d_a, * d_b, * d_c;

	cudaDeviceReset();

	a = 10;
	b = 20;

	cudaMalloc((void**)&d_a, sizeof(int));
	cudaMalloc((void**)&d_b, sizeof(int));
	cudaMalloc((void**)&d_c, sizeof(int));

	cudaMemcpy(d_a, &a, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, &b, sizeof(int), cudaMemcpyHostToDevice);

	sum << <1, 1 >> > (d_a, d_b, d_c);
	cudaMemcpy(&c, d_c, sizeof(int), cudaMemcpyDeviceToHost);

	printf("Resultado %d", c);

	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);
}

void cudaDeviceInfo() {
	int nDevices;

	cudaGetDeviceCount(&nDevices);
	for (int i = 0; i < nDevices; i++) {
		cudaDeviceProp prop;
		cudaGetDeviceProperties(&prop, i);
		printf("Device Number: %d\n", i);
		printf("Device name: %s\n", prop.name);
		printf("Memory Clock Rate (KHz): %d\n",
			prop.memoryClockRate / 1000);
		printf("Memory Bus Width (bits): %d\n",
			prop.memoryBusWidth);
		printf("Peak Memory Bandwidth (GB/s): %f\n\n",
			2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1.0e6);

		printf("Total Global Memory:        %d\n", prop.totalGlobalMem);
		printf("Total shared mem per block: %d\n", prop.sharedMemPerBlock);
		printf("Total const mem size:       %d\n", prop.totalConstMem);
		printf("Warp size:                  %d\n", prop.warpSize);
		printf("Maximum block dimensions:   %d x %d x %d\n", prop.maxThreadsDim[0], \
			prop.maxThreadsDim[1], \
			prop.maxThreadsDim[2]);

		printf("Maximum grid dimensions:    %d x %d x %d\n", prop.maxGridSize[0], \
			prop.maxGridSize[1], \
			prop.maxGridSize[2]);
		printf("Clock Rate:                 %d\n", prop.clockRate);
		printf("Number of muliprocessors (SM):   %d\n", prop.multiProcessorCount);
	}

}

void exemplo02() {


	const int THREAD_PER_BLOCK = 32;
	const int N = 2048;
	int* a, * b, * c;
	int* d_a, * d_b, * d_c;
	int size = N * sizeof(int);

	a = (int*)malloc(size);
	b = (int*)malloc(size);
	c = (int*)malloc(size);


	for (int i = 0; i < N; ++i) {
		a[i] = rand() % 100;
		b[i] = rand() % 100;
	}

	cudaMalloc((void**)&d_a, size);
	cudaMalloc((void**)&d_b, size);
	cudaMalloc((void**)&d_c, size);

	cudaMemset((void*)&d_a, 0, size);
	cudaMemset((void*)&d_b, 0, size);
	cudaMemset((void*)&d_c, 0, size);

	cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);

	vector_sum << <N / THREAD_PER_BLOCK, THREAD_PER_BLOCK >> > (d_a, d_b, d_c, N);

	cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);


	for (int i = 0; i < N; i++) {
		printf("VETOR[%d]=%d\n", i, c[i]);
	}

	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);
	free(a);
	free(b);
	free(c);

}


int* init_matrix(int width) {

	int* m = (int*)malloc(sizeof(int) * width * width);
	for (int i = 0; i < width; ++i)
	{
		for (int j = 0; j < width; ++j)
		{
			m[i * width + j] = rand() % 10;
		}
	}

	return m;
}

void print_matrix(int* m, int width)
{
	for (int i = 0; i < width; ++i)
	{
		for (int j = 0; j < width; ++j)
		{
			printf("%d ", m[i * width + j]);
		}
		printf("\n");
	}
	printf("\n");
}

int* matrix_mult(int* A, int* B, int* C, int width)
{
	int i, j, k;
	#pragma omp parallel for private(i,j,k) num_threads(12)
	for (i = 0; i < width; ++i)
	{
		for (j = 0; j < width; ++j)
		{
			int sum = 0;
			for (k = 0; k < width; ++k)
			{
				sum += A[i * width + k] * B[k * width + j];
			}
			C[i * width + j] = sum;
		}
	}
	return C;
}

void execute_matrix_mult_gpu(int width) {
	int* A, * B, * C;
	int* d_A, * d_B, * d_C;
	const int THREAD_PER_BLOCK = width;
	const int N = width * width;

	//HOST
	A = init_matrix(width);
	B = init_matrix(width);
	int size = sizeof(int) * width * width;
	C = (int*)malloc(size);

	//DEVICE
	float gpu_elapsed_time_ms;
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start, 0);

	cudaMalloc((void**)&d_A, size);
	cudaMalloc((void**)&d_B, size);
	cudaMalloc((void**)&d_C, size);

	cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_B, A, size, cudaMemcpyHostToDevice);

	//gpu_matrix_mult << <N / THREAD_PER_BLOCK, THREAD_PER_BLOCK >> > (d_A, d_B, d_C, width);

	cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);

	cudaThreadSynchronize();

	// time counting terminate
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);

	// compute time elapse on GPU computing
	cudaEventElapsedTime(&gpu_elapsed_time_ms, start, stop);
	printf("Time elapsed on GPU: %f ms.\n\n", gpu_elapsed_time_ms);

	cudaDeviceSynchronize();
	print_matrix(C, width);

	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);
	free(A);
	free(B);
	free(C);
}

void execute_matrix_mult_cpu(int width) {
	int* A, * B, * C;

	//HOST
	A = init_matrix(width);
	B = init_matrix(width);
	int size = sizeof(int) * width * width;
	C = (int*)malloc(size);

	float cpu_elapsed_time_ms;
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start, 0);

	matrix_mult(A, B, C, width);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);

	// compute time elapse on GPU computing
	cudaEventElapsedTime(&cpu_elapsed_time_ms, start, stop);
	printf("Time elapsed on CPU: %f ms.\n\n", cpu_elapsed_time_ms);

	//print_matrix(A, width, width);
	free(A);
	free(B);
	free(C);
}



__global__ void gpu_matrix_mult(int* A, int* B, int* C, int N) {
	int row = threadIdx.y + blockIdx.y * blockDim.y;
	int col = threadIdx.x + blockIdx.x * blockDim.x;

	if (row < N && col < N)
	{
		int sum = 0;
		for (int i = 0; i < N; i++)
		{
			sum += A[row * N + i] * B[i * N + col];
		}
		C[row * N + col] = sum;
	}

}

void init_matrix(int* m, int N) {

	for (int i = 0; i < N * N; ++i)
	{
		m[i] = rand() % 10;
	}

}

void print_matrix(int* m, int width)
{
	for (int i = 0; i < width; ++i)
	{
		for (int j = 0; j < width; ++j)
		{
			printf("%d ", m[i * width + j]);
		}
		printf("\n");
	}
	printf("\n");
}


void execute_mult_gpu() {
	int* A, * B, * C;
	int N = 5000;
	int threads = 512;
	int blocks = (N + threads - 1) / threads;
	const size_t bytes = (N * N) * sizeof(int);

	cudaMallocManaged(&A, bytes);
	cudaMallocManaged(&B, bytes);
	cudaMallocManaged(&C, bytes);
	init_matrix(A, N);
	init_matrix(B, N);

	dim3 blocksize(threads, threads);
	dim3 gridsize(blocks, blocks);

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start, 0);

	gpu_matrix_mult << <gridsize, blocksize >> > (A, B, C, N);
	cudaDeviceSynchronize();

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);

	float elapsedtime;
	cudaEventElapsedTime(&elapsedtime, start, stop);
	printf("Total GPU Time: %f ms", elapsedtime);

	//print_matrix(C, N);
}