/**
 * Daeyoun Kim
 * GitHub: https://github.com/daeyoun24/uwgpuclub
 */


#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>

#define WIDTH 244
#define TILE_WIDTH 32
#define MASK_WIDTH 31
#define H_MASK_WIDTH 31 / 2

__constant__ float M[MASK_WIDTH][MASK_WIDTH];

/**
 * This macro checks return value of the CUDA runtime call and exits
 * the application if the call failed.
 *
 * See cuda.h for error code descriptions.
 */
#define CHECK_CUDA_RESULT(N) {											\
	if (N != 0) {														\
		printf("CUDA call on line %d returned error %d\n", __LINE__,N);	\
		exit(1);														\
	} }

__global__ void convKernel(float *A, float *B) {
	unsigned int ty = threadIdx.y;
	unsigned int tx = threadIdx.x;
	unsigned int row = blockIdx.y * blockDim.y + ty;
	unsigned int col = blockIdx.x * blockDim.x + tx;

	__shared__ float s_A[TILE_WIDTH + MASK_WIDTH / 2][TILE_WIDTH + MASK_WIDTH / 2];

	if (row < WIDTH && col < WIDTH) {
		// Top left
		if (row - H_MASK_WIDTH < 0 && col - H_MASK_WIDTH < 0) {
			s_A[ty][tx] = 0;
		}
		// Top
		else if (row - H_MASK_WIDTH < 0) {
			s_A[ty][tx + H_MASK_WIDTH] = 0;
		}
		// Top right
		else if (row - H_MASK_WIDTH < 0 && col + H_MASK_WIDTH >= WIDTH) {
			s_A[ty][tx + H_MASK_WIDTH * 2] = 0;
		}
		// Right
		else if (col + H_MASK_WIDTH >= WIDTH) {
			s_A[ty + H_MASK_WIDTH][tx + H_MASK_WIDTH * 2] = 0;
		}
		// Bottom right
		else if (row + H_MASK_WIDTH >= WIDTH && col + H_MASK_WIDTH >= WIDTH) {
			s_A[ty + H_MASK_WIDTH * 2][tx + H_MASK_WIDTH * 2] = 0;
		}
		// Bottom
		else if (row + H_MASK_WIDTH >= WIDTH) {
			s_A[ty + H_MASK_WIDTH * 2][tx + H_MASK_WIDTH] = 0;
		}
		// Bottom left
		else if (row + H_MASK_WIDTH >= WIDTH && col - H_MASK_WIDTH < 0) {
			s_A[ty + H_MASK_WIDTH * 2][tx] = 0;
		}
		// Left
		else if (col - H_MASK_WIDTH < 0) {
			s_A[ty + H_MASK_WIDTH][tx] = 0;
		}
		// Center
		else {
			s_A[ty + H_MASK_WIDTH][tx + H_MASK_WIDTH] = A[row * WIDTH + col];
		}

		__syncthreads();
	}
}

int main()
{
	float *M_h, *A_h, *B_h, *A_d, *B_d;

	M_h = (float *)malloc(MASK_WIDTH * MASK_WIDTH * sizeof(float));
	A_h = (float *)malloc(WIDTH * WIDTH * sizeof(float));
	B_h = (float *)malloc(WIDTH * WIDTH * sizeof(float));

	srand(time(NULL));

	for (unsigned int i = 0; i < MASK_WIDTH * MASK_WIDTH; ++i) {
		M_h[i] = (float)rand() / RAND_MAX * 5.0f;
	}

	CHECK_CUDA_RESULT(cudaMemcpyToSymbol(M, M_h, (MASK_WIDTH * MASK_WIDTH) * sizeof(float)));

	for (unsigned int i = 0; i < WIDTH * WIDTH; ++i) {
		A_h[i] = (float)rand() % 256;
	}

	CHECK_CUDA_RESULT(cudaMalloc(&A_d, WIDTH * WIDTH * sizeof(float)));
	CHECK_CUDA_RESULT(cudaMalloc(&B_d, WIDTH * WIDTH * sizeof(float)));

	CHECK_CUDA_RESULT(cudaMemcpy(A_d, A_h, WIDTH * WIDTH * sizeof(float), cudaMemcpyHostToDevice));

	dim3 dimBlock (TILE_WIDTH, TILE_WIDTH, 1);
	dim3 dimGrid (ceil((double)WIDTH / TILE_WIDTH), ceil((double)WIDTH / TILE_WIDTH), 1);

	convKernel<<<dimBlock, dimGrid>>>(A_d, B_d);

	CHECK_CUDA_RESULT(cudaMemcpy(B_h, B_d, WIDTH * WIDTH * sizeof(float), cudaMemcpyDeviceToHost));

	// Verify the results here ...

	cudaFree(A_d);
	cudaFree(B_d);
	free(A_h);
	free(B_h);
	free(M_h);

	return 0;
}
