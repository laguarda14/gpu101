# ============================================================
#
#                   Tiled matmul using numba
#
# Author : Luis Laguarda
# Contact: lluis.laguarda@gmail.com
#
# ============================================================

from numba import cuda, float32
import numpy as np
import time

# constants
M = N = K = 1024
TPB = TILE_SIZE = 32

# ------------
# cuda kernels
# ------------
@cuda.jit
def matmul(A, B, C):

    # 2D thread index (global coordinates)
    row, col = cuda.grid(2)

    if row < C.shape[0] and col < C.shape[1]: # bound check
       value = float32(0.0)
       for k in range(A.shape[1]):  # dot product of A[row, :] and B[:, col]
          value += A[row, k] * B[k, col]
       C[row, col] = value

@cuda.jit
def matmul_tiled(A, B, C):

   # thread index in current block
   tx = cuda.threadIdx.x
   ty = cuda.threadIdx.y

   # block index
   bx = cuda.blockIdx.x
   by = cuda.blockIdx.y

   # row and column of target element of C
   row = by * TILE_SIZE + ty
   col = bx * TILE_SIZE + tx

   # allocate shared memory
   sh_A = cuda.shared.array(shape=(TILE_SIZE, TILE_SIZE), dtype=float32)
   sh_B = cuda.shared.array(shape=(TILE_SIZE, TILE_SIZE), dtype=float32)

   # loop through tiles
   value = float32(0.0)
   n_tiles = (K + TILE_SIZE - 1) // TILE_SIZE

   for m in range(n_tiles):

      # coalesced load: threads in a warp read consecutive columns
      k_A = m * TILE_SIZE + tx
      k_B = m * TILE_SIZE + ty

      # load tiles from A
      sh_A[ty, tx] = A[row, k_A] if (row < M and k_A < K) else float(0.0)

      # load tiles from B
      sh_B[ty, tx] = B[k_B, col] if (k_B < K and col < N) else float(0.0)

      # wait until all threads have loaded their data
      cuda.syncthreads()

      # compute partial dot product for this tile
      for k in range(TILE_SIZE):
         value += sh_A[ty, k] * sh_B[k, tx]

      # wait again before loading new data
      cuda.syncthreads()

   # write
   if row < M and col < N:
      C[row, col] = value

# ------------
# benchmarking
# ------------

def benchmark(fn, label, repeats=5):

   fn()
   times = []
   for _ in range(repeats):
      start = time.perf_counter()
      fn()
      end = time.perf_counter()
      times.append(end - start)
   mean = sum(times) / len(times)

   # TFLOPS: 2*M*N*K FLOPs (multiply-add counts as 2)
   tflops = (2 * M * N * K) / (mean * 1e12)
   print(f"{label:35s}  {mean*1000:7.3f} ms  {tflops:.3f} TFLOPS")

# ------
# kernel
# ------

# non-tiled kernel
def run_naive(A, B, C):
   tpb = (TPB, TPB)
   bpg = ((M + tpb[0] - 1) // tpb[0],
          (M + tpb[1] - 1) // tpb[1])
   matmul[bpg, tpb](A, B, C) # each thread handles one element
   cuda.synchronize()

def run_tiled(A, B, C):
   tpb = (TILE_SIZE, TILE_SIZE)
   bpg = ((N + TILE_SIZE - 1) // TILE_SIZE,
          (M + TILE_SIZE - 1) // TILE_SIZE)
   matmul_tiled[bpg, tpb](A, B, C) # block size = tile width
   cuda.synchronize()

# -------------
# run benchmark
# -------------

if __name__ == "__main__":

   rng = np.random.default_rng(42)
   A_np = rng.random((M, K), dtype=np.float32)
   B_np = rng.random((K, N), dtype=np.float32)
   C_np = np.zeros((M, N), dtype=np.float32)

   # warmup / JIT compile
   A = cuda.to_device(A_np)
   B = cuda.to_device(B_np)
   C = cuda.to_device(C_np)
   for _ in range(3):
      run_naive(A, B, C)
      run_tiled(A, B, C)

   print(f"\nMatrix size: {M}x{K} @ {K}x{N}   Tile: {TILE_SIZE}x{TILE_SIZE}\n")
   benchmark(lambda: run_naive(A, B, C),  "naive (kernel only)")
   benchmark(lambda: run_tiled(A, B, C),  "tiled (kernel only)")

   # Correctness check
   C_naive = cuda.to_device(np.zeros_like(C_np))
   C_tiled = cuda.to_device(np.zeros_like(C_np))
   run_naive(A, B, C_naive)
   run_tiled(A, B, C_tiled)
   assert np.allclose(C_naive.copy_to_host(), C_tiled.copy_to_host(), atol=1e-3), "Results don't match!"
   print("\n Results match")
