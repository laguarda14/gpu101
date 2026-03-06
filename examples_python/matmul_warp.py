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
import torch
import time

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch.cuda")

# explicitely initialize warp (this compiles kernels for all backends)
wp.init()

# constants
SIZES = [512, 1024, 2048, 4096]

# tiling
TILE = wp.constant(16) # larger tile means better GPU occupancy and L1/shared-mem reuse
TILE_THREADS = wp.constant(TILE * TILE) # keep it to TILE * TILE so they stay in sync automatically

# ------------
# warp kernels
# ------------

'''
@NOTE: warp translates Python-like kernel functions into CUDA C++ or CPU SIMD code,
       then builds optimized GPU kernels ahead-of-time (doesn't work with jit!).
       The syntax is Python but supports shared memory, thread IDs, synchronization, etc.
'''

# naive matmul kernel (no tiling)
@wp.kernel
def matmul_naive( A: wp.array2d(dtype=wp.float32)
                , B: wp.array2d(dtype=wp.float32)
                , C: wp.array2d(dtype=wp.float32) ):

   i, j = wp.tid() # global thread ID, considering entire launch grid. No need to manually combine blockIdx with threadIdx like raw CUDA

   M = A.shape[0]
   N = B.shape[1]
   K = A.shape[1]

   if i < M and j < N: # guard against out-of-bounds threads
      acc = wp.float32(0.0) # explicitly tell warp that acc is a mutable float variable (not a constant) by casting it
      for k in range(K):
         acc += A[i, k] * B[k, j]
      C[i, j] = acc

# tile-based matmul kernel
@wp.kernel
def matmul_tiled( A: wp.array2d(dtype=wp.float32)
                , B: wp.array2d(dtype=wp.float32)
                , C: wp.array2d(dtype=wp.float32) ):

   '''
   @NOTE: warp allows an explicit return inside a kernel to skip out-of-bounds threads
   '''

   i, j = wp.tid() # important: when launched with wp.launch_tiled, i/j are block indices, not global thread indices

   acc = wp.tile_zeros(shape=(TILE, TILE), dtype=wp.float32)

   K = A.shape[1]

   # loop through tiles
   for k in range( int(K / TILE) ):

      a = wp.tile_load(A, shape=(TILE, TILE), offset=(i*TILE, k*TILE))
      b = wp.tile_load(B, shape=(TILE, TILE), offset=(k*TILE, j*TILE))

      # sum += a*b
      wp.tile_matmul(a, b, acc)

   wp.tile_store(C, acc, offset=(i*TILE, j*TILE))

# -----------------------------------------------
# helpers: setup and kernel-only launch functions
# -----------------------------------------------

def make_inputs(M, N, K, device):

   # allocate and transfer matrices to device, returns (A, B, C, A_np, B_np)

   # input matrices
   r = np.random.default_rng(42)
   A_np = r.random((M, K), dtype=np.float32)
   B_np = r.random((K, N), dtype=np.float32)
   C_np = np.zeros((M, N), dtype=np.float32)

    # allocate memory on target device and transfer data from host (numpy arrays) into device
   A = wp.array(A_np, device=device)
   B = wp.array(B_np, device=device)
   C = wp.array(C_np, device=device)

   return A, B, C, A_np, B_np

def launch_naive(A, B, C, device):
   wp.launch( kernel = matmul_naive
            , dim    = (A.shape[0], B.shape[1])
            , inputs = [A, B, C]
            , device = device )

def launch_tiled(A, B, C, device):
   wp.launch_tiled( kernel    = matmul_tiled
                  , dim       = (A.shape[0] // TILE, B.shape[1] // TILE)
                  , inputs    = [A, B, C]
                  , block_dim = TILE_THREADS
                  , device    = device)

# this calls into vendor-optimized BLAS and shows the ceiling our custom kernel is aiming for
def launch_cublas(A, B, C, device):
   A_th = wp.to_torch(A)
   B_th = wp.to_torch(B)
   C_th = wp.to_torch(C)
   torch.matmul(A_th, B_th, out=C_th)

# ----------------------
# Correctness validation
# ----------------------

def validate(device, size=512):

   # run both kernels once and compare against numpy reference

   M = N = K = size
   assert M % 16 == 0 and N % 16 == 0 and K % 16 == 0
   A, B, C, A_np, B_np = make_inputs(M, N, K, device)
   ref = A_np @ B_np

   launch_naive(A, B, C, device)
   wp.synchronize_device(device)
   assert np.allclose(C.numpy(), ref, atol=1e-4), "naive kernel result mismatch!"

   # reset C
   C.fill_(0)

   launch_tiled(A, B, C, device)
   wp.synchronize_device(device)
   assert np.allclose(C.numpy(), ref, atol=1e-4), "tiled kernel result mismatch!"

   print(f"  both kernels pass correctness check on {device}")

# ---------------------------------
# benchmarking (kernel-only timing)
# ---------------------------------

def gflops(M, N, K, elapsed_s): return (2.0 * M * N * K) / (elapsed_s * 1.e9)

def benchmark(launch_fn, device, label, M, N, K, warmup=3, runs=10):

   # kernel only to measure compute performance not PCIe bandwidth

   # sanity-check: matrix dims must be divisible by tile size at startup,
   # otherwise run_tiled silently produces wrong results (dropped remainder tiles)
   assert M % 16 == 0 and N % 16 == 0 and K % 16 == 0, \
      "M, N, K must be divisible by 16 for tiled kernel"

   A, B, C, _, _ = make_inputs(M, N, K, device)

   # warmup — let Warp compile and cache the kernel
   for _ in range(warmup): launch_fn(A, B, C, device)
   wp.synchronize_device(device)

   # timed runs
   times = []
   for _ in range(runs):
      start = time.perf_counter()
      launch_fn(A, B, C, device)
      wp.synchronize_device(device) # make sure all outstanding work on device has completed
      times.append(time.perf_counter() - start)

   avg_s  = (sum(times) / len(times))
   best_s  = min(times)
   peak_gf = gflops(M, N, K, best_s)
   print(f"    {label:<22} avg={avg_s*1000.:8.2f}ms  best={best_s*1000.:8.2f}ms  peak={peak_gf:7.1f} GFLOPS")

# ----------
# entrypoint
# ----------

if __name__ == "__main__":

   print("\nrunning warp matmul benchmarks...")

   devices = ["cpu"]
   if wp.is_cuda_available():
      devices.append("cuda:0")
   else:
      print("(no CUDA device found — running CPU only)\n")

   for dev in devices:
      print(f"\n--- device: {dev} ---")

      # correctness check first
      validate(dev, size=512)
      print()

      for size in SIZES:

         # skip large sizes on CPU — they take too long with the naive O(n^3) kernel
         if dev == "cpu" and size > 512:
               print(f"  [skipping size {size} on CPU — use numpy for large CPU matmul]")
               continue

         M = N = K = size
         print(f"  -- {size}x{size} --")
         benchmark(launch_naive, dev, "naive kernel", M, N, K)
         benchmark(launch_tiled, dev, "tiled kernel", M, N, K)

         # cuBLAS reference only available on CUDA
         if dev != "cpu": benchmark(launch_cublas, dev, "cuBLAS (wp.matmul)", M, N, K)
         print()