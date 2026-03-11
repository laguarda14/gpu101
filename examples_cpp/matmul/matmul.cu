// @file matmul.cu
// @author Luis Laguarda [lluis.laguarda@gmail.com]
#include <iostream>
#include <vector>

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

__global__ void naive_matmul_kernel(float* A, float* B, float* C, int N1, int N2, int N3)
{
   int i = blockDim.y*blockIdx.y + threadIdx.y; // row
   int j = blockDim.x*blockIdx.x + threadIdx.x; // column

   if (i < N1 && j < N3) // bound check
   {
      // value at C[i,j]
      float value = 0.0f;
      for (int k = 0; k < N2; k++)
      {
         value += A[i*N2+k] * B[k*N3+j];
      }
      C[i*N3+j] = value;
   }
}

/* each element is loaded once per tile rather than once per thread,
   reducing global memory traffic by a factor TILE_WIDTH */
__global__ void tiled_matmul_kernel(float* A, float* B, float* C, int N1, int N2, int N3)
{
   // ensure that TILE_WIDTH = BLOCK_SIZE
   // assert(TILE_WIDTH == blockDim.x);
   // assert(TILE_WIDTH == blockDim.y);

   int tx = threadIdx.x;
   int ty = threadIdx.y;
   int i  = TILE_WIDTH*blockIdx.y + ty; // row
   int j  = TILE_WIDTH*blockIdx.x + tx; // column

   // allocate shared memory
   __shared__ float sh_A[TILE_WIDTH][TILE_WIDTH];
   __shared__ float sh_B[TILE_WIDTH][TILE_WIDTH];

   // loop through tiles
   float value = 0.0f;
   int num_phases = (N2 + TILE_WIDTH - 1) / TILE_WIDTH; // integer ceiling, no float cast

   for (int phase = 0; phase < num_phases; phase++)
   {
      // coalesced load of A tile
      if ((i < N1) && ((phase*TILE_WIDTH+tx) < N2)) {
         sh_A[ty][tx] = A[i*N2 + phase*TILE_WIDTH + tx];
      } else {
         sh_A[ty][tx] = 0.0f;
      }

      // coalesced load of B tile
      if (((phase*TILE_WIDTH + ty) < N2) && (j < N3)) {
         sh_B[ty][tx] = B[(phase*TILE_WIDTH + ty)*N3 + j];
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

   if (i < N1 && j < N3) { // bound check
      C[i*N3 + j] = value;
   }
}

// --------------
// Host functions
// --------------

void printDeviceInfo(int device_id = 0)
{
   cudaDeviceProp dev_prop;
   cudaError_t err = cudaGetDeviceProperties(&dev_prop, device_id);
   CUDA_CHECK(err);
   printf("Available Shared Memory per Block: %lu B \n", dev_prop.sharedMemPerBlock);
   printf("Max Threads per Block: %i \n", dev_prop.maxThreadsPerBlock);
   printf("Used Shared Memory per Block: %i B \n", TILE_WIDTH*TILE_WIDTH*8);

}

// allocates device memory, copies A and B to device, returns pointers
// caller is responsible for freeing d_A, d_B, d_C.
static void device_alloc_and_copy(const std::vector<float>& A, const std::vector<float>& B, float** d_A, float** d_B, float** d_C, int N1, int N2, int N3)
{
   cudaError_t err;

   err = cudaMalloc((void**)d_A, N1 * N2 * sizeof(float)); CUDA_CHECK(err);
   err = cudaMalloc((void**)d_B, N2 * N3 * sizeof(float)); CUDA_CHECK(err);
   err = cudaMalloc((void**)d_C, N1 * N3 * sizeof(float)); CUDA_CHECK(err);

   err = cudaMemcpy(*d_A, A.data(), N1 * N2 * sizeof(float), cudaMemcpyHostToDevice); CUDA_CHECK(err);
   err = cudaMemcpy(*d_B, B.data(), N2 * N3 * sizeof(float), cudaMemcpyHostToDevice); CUDA_CHECK(err);
}

// copy C to host and free pointers
static void device_copy_and_free(std::vector<float>& C, float* d_A, float* d_B, float* d_C, int N1, int N3)
{
   cudaError_t err;
   err = cudaMemcpy(C.data(), d_C, N1 * N3 * sizeof(float), cudaMemcpyDeviceToHost); CUDA_CHECK(err);
   cudaFree(d_A);
   cudaFree(d_B);
   cudaFree(d_C);
}

// launch naive matmul, returns kernel-only execution time in milliseconds
float naive_matmul(std::vector<float>& A, std::vector<float>& B, std::vector<float>& C, int N1, int N2, int N3)
{
   float *d_A, *d_B, *d_C;
   device_alloc_and_copy(A, B, &d_A, &d_B, &d_C, N1, N2, N3);

   dim3 dim_block(32, 32, 1);
   dim3 dim_grid((N3 + 31) / 32, (N1 + 31) / 32, 1); // integer ceiling

   // measure kernel time with CUDA events (device-side, excludes transfers)
   cudaEvent_t start, stop;
   cudaEventCreate(&start);
   cudaEventCreate(&stop);

   // kernel execution
   cudaEventRecord(start);
   naive_matmul_kernel<<<dim_grid, dim_block>>>(d_A, d_B, d_C, N1, N2, N3);
   cudaEventRecord(stop);
   cudaEventSynchronize(stop); // blocks host until stop is reached

   CUDA_CHECK(cudaGetLastError());

   float kernel_ms = 0.0f;
   cudaEventElapsedTime(&kernel_ms, start, stop);
   cudaEventDestroy(start);
   cudaEventDestroy(stop);

   // copy results back to host
   device_copy_and_free(C, d_A, d_B, d_C, N1, N3);
   return kernel_ms;
}

// launch tiled matmul, returns kernel-only execution time in milliseconds
float tiled_matmul(std::vector<float>& A, std::vector<float>& B, std::vector<float>& C, int N1, int N2, int N3)
{
   float *d_A, *d_B, *d_C;
   device_alloc_and_copy(A, B, &d_A, &d_B, &d_C, N1, N2, N3);

   dim3 dim_block(TILE_WIDTH, TILE_WIDTH, 1);
   dim3 dim_grid((N3 + TILE_WIDTH - 1) / TILE_WIDTH, (N1 + TILE_WIDTH - 1) / TILE_WIDTH, 1);

   cudaEvent_t start, stop;
   cudaEventCreate(&start);
   cudaEventCreate(&stop);

   cudaEventRecord(start);
   tiled_matmul_kernel<<<dim_grid, dim_block>>>(d_A, d_B, d_C, N1, N2, N3);
   cudaEventRecord(stop);
   cudaEventSynchronize(stop); // blocks host until stop is reached

   CUDA_CHECK(cudaGetLastError());

   float kernel_ms = 0.0f;
   cudaEventElapsedTime(&kernel_ms, start, stop);
   cudaEventDestroy(start);
   cudaEventDestroy(stop);

   device_copy_and_free(C, d_A, d_B, d_C, N1, N3);
   return kernel_ms;
}
