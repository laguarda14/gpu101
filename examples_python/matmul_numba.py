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
M = 512
N = 512
K = 512
TILE_SIZE = 16
TPB = 16

# ------------
# cuda kernels
# ------------
@cuda.jit
def matmul(A, B, C):
    
    # 2D thread index (global coordinates)
    row, col = cuda.grid(2) 

    if row < C.shape[0] and col < C.shape[1]: # bound check
       value = 0.0
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

   # allocate shared memory
   sh_A = cuda.shared.array(shape=(TILE_SIZE, TILE_SIZE), dtype=float32)
   sh_B = cuda.shared.array(shape=(TILE_SIZE, TILE_SIZE), dtype=float32)

   # row and column of target element of C
   row = by * TILE_SIZE + ty
   col = bx * TILE_SIZE + tx
   
   # loop through tiles
   value = 0.
   for m in range((A.shape[1] + TILE_SIZE - 1) // TILE_SIZE):

      # load tiles from A
      if row < A.shape[0] and m * TILE_SIZE + tx < A.shape[1]:
         sh_A[ty, tx] = A[row, m * TILE_SIZE + tx]
      else:
         sh_A[ty, tx] = 0.0

      # load tiles from B
      if col < B.shape[1] and m * TILE_SIZE + ty < B.shape[0]:
         sh_B[ty, tx] = B[m * TILE_SIZE + ty, col]
      else:
         sh_B[ty, tx] = 0.0
      
      # wait until all threads have loaded their data
      cuda.syncthreads()

      # compute partial dot product for this tile
      for k in range(TILE_SIZE):
         value += sh_A[ty, k] * sh_B[k, tx]

      # wait again before loading new data
      cuda.syncthreads()

   # write 
   if row < C.shape[0] and col < C.shape[1]:
      C[row, col] = value

# ------------------------
# main tests and profiling
# ------------------------

# profiler
def benchmark(fn, label):
   start = time.perf_counter()
   fn()
   end = time.perf_counter()
   print(f"{label} on took {end - start:.4f} seconds")

# launch non-tiled matmul kernel
def run_naive():

   # input matrices
   r = np.random.default_rng(42)
   A_np = r.random((M, K), dtype=np.float32)
   B_np = r.random((K, N), dtype=np.float32)
   C_np = np.zeros((M, N), dtype=np.float32)

   # transfer data to device
   A = cuda.to_device(A_np)
   B = cuda.to_device(B_np)
   C = cuda.to_device(C_np)

   # launch non-tiled kernel
   th_per_block = (TPB, TPB)
   bl_per_grid_x = (M + th_per_block[0] - 1) // th_per_block[0]
   bl_per_grid_y = (M + th_per_block[1] - 1) // th_per_block[1]
   matmul[(bl_per_grid_x, bl_per_grid_y), th_per_block](A, B, C) # each thread handles one element

   # copy result back
   #C_np = C.copy_to_host()

   # verify correctness
   #assert(np.allclose(C_np, A_np@B_np))

# launch tiled kernel
def run_tiled():

   # input matrices
   r = np.random.default_rng(42)
   A_np = r.random((M, K), dtype=np.float32)
   B_np = r.random((K, N), dtype=np.float32)
   C_np = np.zeros((M, N), dtype=np.float32)

   # transfer data to device
   A = cuda.to_device(A_np)
   B = cuda.to_device(B_np)
   C = cuda.to_device(C_np)

   # launch tiled kernel
   bl_per_grid_x = (N + TILE_SIZE - 1) // TILE_SIZE
   bl_per_grid_y = (M + TILE_SIZE - 1) // TILE_SIZE
   matmul_tiled[(bl_per_grid_x, bl_per_grid_y), (TILE_SIZE, TILE_SIZE)](A, B, C) # block size = tile width

   # copy result back
   #C_np = C.copy_to_host()
    
   # verify correctness
   #assert(np.allclose(C_np, A_np@B_np))

# -------------
# run benchmark
# -------------

if __name__ == "__main__":

   benchmark(run_naive, 'naive matmul')
   benchmark(run_tiled, 'tiled matmul')
