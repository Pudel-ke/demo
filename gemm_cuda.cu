#include <iostream>
#include <cuda_runtime.h>
#include <iomanip>

using namespace std;

#define N 1024
#define TILE_SIZE 16

// ==========================================
// 宏定义：用于捕捉 CUDA 报错
// ==========================================
#define CUDA_CHECK(call) \
do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        printf("CUDA Error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

__global__ void gemm_tiled(float *A, float *B, float *C)
{
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    float sum = 0.0f;

    for(int t=0; t<N/TILE_SIZE; t++)
    {
        As[threadIdx.y][threadIdx.x] = A[row*N + t*TILE_SIZE + threadIdx.x];
        Bs[threadIdx.y][threadIdx.x] = B[(t*TILE_SIZE + threadIdx.y)*N + col];

        __syncthreads();

        for(int k=0; k<TILE_SIZE; k++)
        {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }

        __syncthreads();
    }

    C[row*N+col] = sum;
}

void init_matrix(float *M)
{
    for(int i=0; i<N*N; i++)
        M[i] = 1.0f;
}

int main()
{
    size_t bytes = N*N*sizeof(float);

    float *h_A = new float[N*N];
    float *h_B = new float[N*N];
    float *h_C = new float[N*N];

    init_matrix(h_A);
    init_matrix(h_B);

    float *d_A, *d_B, *d_C;

    // 分配显存
    CUDA_CHECK(cudaMalloc(&d_A, bytes));
    CUDA_CHECK(cudaMalloc(&d_B, bytes));
    CUDA_CHECK(cudaMalloc(&d_C, bytes));

    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid(N/TILE_SIZE, N/TILE_SIZE);

    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);

    // ==========================================
    // 1. 预热 (Warm-up)：执行完整流程
    // ==========================================
    CUDA_CHECK(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice));
    gemm_tiled<<<grid, block>>>(d_A, d_B, d_C);
    CUDA_CHECK(cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaDeviceSynchronize()); // 确保预热完成

    // ==========================================
    // 2. 循环 20 次并记录完整任务时间
    // ==========================================
    const int iterations = 20;
    float total_ms = 0.0f;

    cout << fixed << setprecision(4);
    cout << "--- Recording End-to-End Time (H2D + Kernel + D2H) ---" << endl;

    for (int i = 0; i < iterations; ++i) {
        cudaEventRecord(start); // 开始计时

        // [H2D] 数据从 Host 拷到 Device
        CUDA_CHECK(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice));

        // [Kernel] 执行 GPU 计算
        gemm_tiled<<<grid, block>>>(d_A, d_B, d_C);
        CUDA_CHECK(cudaGetLastError());

        // [D2H] 结果从 Device 拷回 Host
        CUDA_CHECK(cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost));

        cudaEventRecord(end);   // 结束计时
        cudaEventSynchronize(end);

        float ms = 0.0f;
        cudaEventElapsedTime(&ms, start, end);
        total_ms += ms;
        
        cout << "Iteration " << setw(2) << i + 1 << " Time: " << ms << " ms" << endl;
    }

    float avg_ms = total_ms / iterations;

    // 计算性能
    double flops = 2.0 * N * N * N;
    double gflops = flops / (avg_ms / 1000.0) / 1e9;

    // ==========================================
    // 3. 输出总结
    // ==========================================
    cout << "------------------------------------------" << endl;
    cout << "Avg End-to-End Time: " << avg_ms << " ms" << endl;
    cout << "Effective Perf:      " << gflops << " GFLOPS (includes PCIe transfer)" << endl;
    cout << "C[0] = " << h_C[0] << " (Should be 1024.000)" << endl;

    // 释放资源
    cudaEventDestroy(start);
    cudaEventDestroy(end);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    delete[] h_A;
    delete[] h_B;
    delete[] h_C;

    return 0;
}