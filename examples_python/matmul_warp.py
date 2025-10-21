# ============================================================
#
#                  Matmul using NVIDIA warp
#
# Demonstrates the impact of tiling on performance
# Compatible with:
# - CPUs
# - CUDA-capable GPUs
# - Apple Silicon (Metal backend)
#
# For more info, see: 
#   https://nvidia.github.io/warp/basics.html
#
# Author : Luis Laguarda
# Contact: lluis.laguarda@gmail.com
#
# ============================================================

import warp as wp
import numpy as np
import time

# explicitely initialize warp (this compiles kernels for all backends)
wp.init()

# constants
M    = 512
N    = 512
K    = 512
TILE = wp.constant(8) # tile size
TILE_THREADS = wp.constant(64)

# ------------
# warp kernels
# ------------

'''
@NOTE: warp translates Python-like kernel functions into CUDA C++ or CPU SIMD code,
       then builds optimized GPU kernels ahead-of-time (doesn't work with jit!).
       The syntax is python but supports shared memory, thread IDs, synchronization, etc.
'''

# naive matmul kernel (no tiling)
@wp.kernel
def matmul_naive( A: wp.array2d(dtype=wp.float32)
                , B: wp.array2d(dtype=wp.float32)
                , C: wp.array2d(dtype=wp.float32) ):

   i, j = wp.tid() # global thread ID, considering entire launch grid. No need to manually combine blockIdx with threadIdx like in CUDA

   M = A.shape[0]
   N = B.shape[1]
   K = A.shape[1]

   if i < M and j < N: # guard against out-of-bounds threads
      acc = wp.float32(0.0) # explicitly tell warp that acc is a mutable float variable (not a constant) by casting it
      for k in range(K):
         acc += A[i, k] * B[k, j]
      C[i, j] = acc

# tile based matmul kernel
@wp.kernel
def matmul_tiled( A: wp.array2d(dtype=wp.float32)
                , B: wp.array2d(dtype=wp.float32)
                , C: wp.array2d(dtype=wp.float32) ):

   '''
   @NOTE: warp allows an explicit return inside a kernel to skip further execution of out-of-bounds threads (unlike CUDA)
   '''

   i, j = wp.tid() # important: if launched tiled, these are now block indices instead of global thread indices

   acc = wp.tile_zeros(shape=(TILE, TILE), dtype=wp.float32)  

   K = A.shape[1]

   # loop through tiles
   for k in range( int(K / TILE) ):

      a = wp.tile_load(A, shape=(TILE, TILE), offset=(i*TILE, k*TILE))
      b = wp.tile_load(B, shape=(TILE, TILE), offset=(k*TILE, j*TILE))

      # sum += a*b
      wp.tile_matmul(a, b, acc)

   wp.tile_store(C, acc, offset=(i*TILE, j*TILE))

# ------------------------
# main tests and profiling
# ------------------------

'''
@NOTE: the lightweight CPU-only build omits ScopedProfiler
       and runtime profiling hooks such profile_start, profile_end, etc.
'''

# profiler
def benchmark(fn, device, label):
   start = time.perf_counter()
   fn(device)
   wp.synchronize_device(device) # make sure all outstanding work on device has completed
   end = time.perf_counter()
   print(f"{label} on {device} took {end - start:.4f} seconds")

# launch naive matmul kernel
def run_naive(device):

   # input matrices
   r = np.random.default_rng(42)
   A_np = r.random((M, K), dtype=np.float32)
   B_np = r.random((K, N), dtype=np.float32)
   C_np = np.zeros((M, N), dtype=np.float32)

   # allocate memory on target device and transfer data from host (numpy arrays) into device
   # - device="cpu"   : allocate on system RAM, accessible directly by CPU threads
   # - device="cuda"  : allocate in GPU VRAM and copy the data there
   # - device="metal" : allocate in Apple GPU memory via Metal buffers
   # also, wrap memory in a wp.array object so warp kernels can access it efficiently
   A = wp.array(A_np, device=device)
   B = wp.array(B_np, device=device)
   C = wp.array(C_np, device=device)

   # launch kernel
   wp.launch( kernel = matmul_naive
            , dim    = (M, N) # important: launch one thread per output element!
            , inputs = [A, B, C]
            , device = device )

   '''
   @NOTE: C.numpy() triggers an implicit device-to-host transfer, 
          and warp automatically synchronizes before performing that copy.
          So no need for wp.synchronize_device
   '''
   #assert(np.allclose(C.numpy(), A_np@B_np))
   #return C.numpy()

# launch tiled matmul kernel
def run_tiled(device):

   # input matrices
   r = np.random.default_rng(42)
   A_np = r.random((M, K), dtype=np.float32)
   B_np = r.random((K, N), dtype=np.float32)
   C_np = np.zeros((M, N), dtype=np.float32) 

    # allocate memory on target device and transfer data from host (numpy arrays) into device
   A = wp.array(A_np, device=device)
   B = wp.array(B_np, device=device)
   C = wp.array(C_np, device=device)

   # launch tiled kernel -> important: use "wp.launch_tiled"
   wp.launch_tiled( kernel    = matmul_tiled
                  , dim       = (int(M / TILE), int(N / TILE)) # number of blocks
                  , inputs    = [A, B, C]
                  , block_dim = TILE_THREADS # number of threads per block
                  , device    = device )

   #assert(np.allclose(C.numpy(), A_np@B_np))
   #return C.numpy()

# --------------
# run benchmarks
# --------------

if __name__ == "__main__":

   print("\nrunning warp matmul benchmarks...")

   devices = ["cpu", "cuda:0"]
   for dev in devices:
      print(f"\n--- device: {dev} ---")
      benchmark(run_naive, dev, "naive kernel")
      benchmark(run_tiled, dev, "tiled kernel")
