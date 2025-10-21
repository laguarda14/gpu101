#include <iostream>
#include <vector>
#include <cassert>

#define TILE_WIDTH 32

#define CUDA_CHECK(err) do { \
   if (err != cudaSuccess) { \
      printf("CUDA error: %s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
      exit(EXIT_FAILURE); \
   } \
} while (0)

// ----------------
// device functions
// ----------------

__global__ void matmul_kernel(float* A, float* B, float* C, int N1, int N2, int N3)
{
   int j = blockDim.y*blockIdx.y + threadIdx.y; // row
   int i = blockDim.x*blockIdx.x + threadIdx.x; // column

   if (i < N1 && j < N3) // bound check
   {
      // value at C[i,j]
      float value = 0;
      for (int k = 0; k < N2; k++)
      {
         value += A[i*N2+k] * B[k*N3+j];
      }

      C[i*N3+j] = value;
   }
}

/* we load each element once per tile, not once per thread. Therefore, global memory traffic is reduced by a factor TILE_WIDTH*/

__global__ void tiled_matmul_kernel(float* A, float* B, float* C, int N1, int N2, int N3)
{
   // ensure that TILE_WIDTH = BLOCK_SIZE
   assert(TILE_WIDTH == blockDim.x);
   assert(TILE_WIDTH == blockDim.y);

   int tx = threadIdx.x;
   int ty = threadIdx.y;
   int i  = TILE_WIDTH*blockIdx.y + ty; // row
   int j  = TILE_WIDTH*blockIdx.x + tx; // column

   // allocate shared memory
   __shared__ float sh_A[TILE_WIDTH][TILE_WIDTH];
   __shared__ float sh_B[TILE_WIDTH][TILE_WIDTH];

   // loop through tiles
   float value = 0;
   for (int phase = 0; phase < ceil((float)N2/TILE_WIDTH); phase++)
   {
      // load tiles from matrix A into shared memory
      // these memory addresses are contiguouss -> coalesced global memory access!
      if ((i < N1) && ((phase*TILE_WIDTH+tx) < N2)) {
         sh_A[ty][tx] = A[(i)*N2 + phase*TILE_WIDTH+tx];
      } else {
         sh_A[ty][tx] = 0.0f;
      }

      // load tiles from matrix B into shared memory
      // these memory addresses are also contiguouss -> coalesced global memory access!
      if (((phase*TILE_WIDTH + ty) < N2) && (j < N3)) {
         sh_B[ty][tx] = B[(phase*TILE_WIDTH + ty)*N3+j];
      } else {
         sh_B[ty][tx] = 0.0f;
      }
      __syncthreads();

      // perform the partial dot product
      for (int k = 0; k < TILE_WIDTH; k++) {
         value += sh_A[ty][k] * sh_B[k][tx];
      }
      __syncthreads();
   }

   if (i < N1 && j < N3) // bound check
   {
      C[i*N3+j] = value;
   }
}

// --------------
// Host functions
// --------------

void printDeviceInfo(int device_id = 0) {

   cudaDeviceProp dev_prop;
   cudaError_t err = cudaGetDeviceProperties(&dev_prop, device_id);
   CUDA_CHECK(err);
   printf("Available Shared Memory per Block: %lu B \n", dev_prop.sharedMemPerBlock);
   printf("Max Threads per Block: %i \n", dev_prop.maxThreadsPerBlock);
   printf("Used Shared Memory per Block: %i B \n", TILE_WIDTH*TILE_WIDTH*8);

}

void matmul(std::vector<float>& A, std::vector<float>& B, std::vector<float>& C, int N1, int N2, int N3)
{
   // print GPU info
   // printDeviceInfo();

   // device array pointers
   float* d_A;
   float* d_B;
   float* d_C;

   // device memory allocation
   cudaError_t err_A = cudaMalloc((void**)&d_A, N1*N2*sizeof(float));
   CUDA_CHECK(err_A);

   cudaError_t err_B = cudaMalloc((void**)&d_B, N2*N3*sizeof(float));
   CUDA_CHECK(err_B);

   cudaError_t err_C = cudaMalloc((void**)&d_C, N1*N3*sizeof(float));
   CUDA_CHECK(err_C);

   // copy A and B to device memory
   cudaError_t err_A_ = cudaMemcpy(d_A, &A[0], N1*N2*sizeof(float), cudaMemcpyHostToDevice);
   CUDA_CHECK(err_A_);

   cudaError_t err_B_ = cudaMemcpy(d_B, &B[0], N2*N3*sizeof(float), cudaMemcpyHostToDevice);
   CUDA_CHECK(err_B_);

   // kernel execution
   dim3 dim_block(32, 32, 1);
   dim3 dim_grid(ceil(N3/32.0), ceil(N1/32.0), 1);
   matmul_kernel<<<dim_grid, dim_block>>>(d_A, d_B, d_C, N1, N2, N3);
   cudaError_t err = cudaGetLastError();
   CUDA_CHECK(err);

   // copy results back to host memory
   cudaError_t err_C_ = cudaMemcpy(&C[0], d_C, N1*N3*sizeof(float), cudaMemcpyDeviceToHost);
   CUDA_CHECK(err_C_);

   // free device memory
   cudaFree(d_A);
   cudaFree(d_B);
   cudaFree(d_C);

}

void tiled_matmul(std::vector<float>& A, std::vector<float>& B, std::vector<float>& C, int N1, int N2, int N3)
{
   // print GPU info
   // printDeviceInfo();

   // device array pointers
   float* d_A;
   float* d_B;
   float* d_C;

   // device memory allocation
   cudaError_t err_A = cudaMalloc((void**)&d_A, N1*N2*sizeof(float));
   CUDA_CHECK(err_A);

   cudaError_t err_B = cudaMalloc((void**)&d_B, N2*N3*sizeof(float));
   CUDA_CHECK(err_B);

   cudaError_t err_C = cudaMalloc((void**)&d_C, N1*N3*sizeof(float));
   CUDA_CHECK(err_C);

   // copy A and B to device memory
   cudaError_t err_A_ = cudaMemcpy(d_A, &A[0], N1*N2*sizeof(float), cudaMemcpyHostToDevice);
   CUDA_CHECK(err_A_);

   cudaError_t err_B_ = cudaMemcpy(d_B, &B[0], N2*N3*sizeof(float), cudaMemcpyHostToDevice);
   CUDA_CHECK(err_B_);

   // kernel execution
   dim3 dim_block(TILE_WIDTH, TILE_WIDTH, 1);
   dim3 dim_grid(ceil(N3/(float)(TILE_WIDTH)), ceil(N1/(float)(TILE_WIDTH)), 1);
   tiled_matmul_kernel<<<dim_grid, dim_block>>>(d_A, d_B, d_C, N1, N2, N3);
   cudaError_t err = cudaGetLastError();
   CUDA_CHECK(err);

   // copy results back to host memory
   cudaError_t err_C_ = cudaMemcpy(&C[0], d_C, N1*N3*sizeof(float), cudaMemcpyDeviceToHost);
   CUDA_CHECK(err_C_);

   // free device memory
   cudaFree(d_A);
   cudaFree(d_B);
   cudaFree(d_C);

}
