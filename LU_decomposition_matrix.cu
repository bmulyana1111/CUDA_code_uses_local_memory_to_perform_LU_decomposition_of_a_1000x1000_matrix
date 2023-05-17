#include <stdio.h>

#define N 1000

__global__ void LU_decomposition(float* A, float* L, float* U) {
    int i, j, k;
    __shared__ float s[N][N];

    for (k = 0; k < N; k++) {
        if (threadIdx.x == k) {
            U[k * N + k] = A[k * N + k];
            for (j = k + 1; j < N; j++) {
                U[k * N + j] = A[k * N + j];
                L[j * N + k] = A[j * N + k] / U[k * N + k];
            }
        }

        __syncthreads();

        for (i = k + 1; i < N; i++) {
            if (threadIdx.x == k) {
                for (j = k + 1; j < N; j++) {
                    A[i * N + j] -= L[i * N + k] * U[k * N + j];
                }
                U[i * N + k] = A[i * N + k];
            }
            __syncthreads();
        }
    }
}

int main() {
    float *A, *L, *U, *d_A, *d_L, *d_U;
    int i, j;

    A = (float*)malloc(N * N * sizeof(float));
    L = (float*)malloc(N * N * sizeof(float));
    U = (float*)malloc(N * N * sizeof(float));

    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            if (i == j) {
                A[i * N + j] = 1.0f;
            } else if (i > j) {
                A[i * N + j] = 0.0f;
            } else {
                A[i * N + j] = (float)(rand() % 10 + 1);
            }
            L[i * N + j] = 0.0f;
            U[i * N + j] = 0.0f;
        }
    }

    cudaMalloc(&d_A, N * N * sizeof(float));
    cudaMalloc(&d_L, N * N * sizeof(float));
    cudaMalloc(&d_U, N * N * sizeof(float));

    cudaMemcpy(d_A, A, N * N * sizeof(float), cudaMemcpyHostToDevice);

    dim3 dimBlock(N, 1, 1);
    dim3 dimGrid(1, 1, 1);

    LU_decomposition<<<dimGrid, dimBlock>>>(d_A, d_L, d_U);

    cudaMemcpy(L, d_L, N * N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(U, d_U, N * N * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_L);
    cudaFree(d_U);

    free(A);
    free(L);
    free(U);

    return 0;
}
