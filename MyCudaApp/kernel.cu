
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda.h>
#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include "exemplos.cuh"


int main()
{
	cudaDeviceReset();
	execute_matrix_mult_gpu(1000);
	execute_matrix_mult_cpu(1000);
	
	return 0;
}




