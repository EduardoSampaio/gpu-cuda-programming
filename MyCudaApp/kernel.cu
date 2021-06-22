
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda.h>
#include <stdio.h>
#include <time.h>
#include <stdlib.h>

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


int main()
{
	cudaDeviceReset();
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


	return 0;
}




