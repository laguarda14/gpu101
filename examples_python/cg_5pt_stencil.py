# ============================================================
#
#          CG for 5-point stencil using NVIDIA warp
#
# Author : Luis Laguarda
# Contact: lluis.laguarda@gmail.com
#
# ============================================================

import time, os
import warp as wp
import numpy as np

from matmul_warp import benchmark

DEBUG = int(os.getenv("DEBUG", "0"))
if DEBUG > 0: np.set_printoptions(linewidth=200)

'''
@NOTE: when calling a warp function like wp.launch() for the first time,
       it will initialize itself and will print some startup information 
       about the compute devices available. 
'''

# also possible to explicitly initialize warp with
# wp.init()

# grid size
if DEBUG > 0:
   Nx = 19
   Ny = 19
else:
   Nx = 128*10
   Ny = 128*10

# grid spacing
dx = 1./(Nx + 1) # face centered, BC within domain
dy = 1./(Ny + 1) # same
dx2 = dx*dx
dy2 = dy*dy

# tiling
st = 6
TILE = wp.constant(st)
TILE_HALO = wp.constant(st+2) # tile size with halo cells
TILE_THREADS = wp.constant(st*st)

# ------------
# warp kernels
# ------------

# naive 5-point stencil operation
@wp.kernel
def apply_A( p      : wp.array2d(dtype=wp.float32)
           , y      : wp.array2d(dtype=wp.float32)
           , inv_dx2: float
           , inv_dy2: float ):

   i, j = wp.tid()
   
   Nx  = p.shape[0]
   Ny  = p.shape[1]
   
   if (i > 0) and (i < Nx - 1) and (j > 0) and (j < Ny - 1):
      y[i,j] = ( (2.0*inv_dx2 + 2.0*inv_dy2) * p[i,j]
               - inv_dx2 * (p[i-1,j] + p[i+1,j])
               - inv_dy2 * (p[i,j-1] + p[i,j+1]) )

# tiled 5-point stencil operation
@wp.kernel
def apply_A_tiled( p      : wp.array2d(dtype=wp.float32)
                 , y      : wp.array2d(dtype=wp.float32)
                 , inv_dx2: float
                 , inv_dy2: float ):

   # again, each thread corresponds to a tile and not an element
   bi, bj = wp.tid()

   # load a tile (plus one-cell halo)
   ofx = bi*TILE
   ofy = bj*TILE
   p_tile = wp.tile_load(p, shape=(TILE_HALO, TILE_HALO), offset=(ofx, ofy)) # TILE_HALO instead TILE+2 because it has to be statically inferred

   acc = wp.tile_zeros((TILE, TILE), dtype=wp.float32)
  
   '''
   @NOTE: the loop is effectively vectorized across the threads
   '''
   # compute Laplacian for interior cells, not halo
   for i in range(TILE):
      for j in range(TILE): 
         acc[i,j] = ( (2.0*inv_dx2 + 2.0*inv_dy2) * p_tile[i+1,j+1]
                    - inv_dx2 * (p_tile[i,j+1] + p_tile[i+2,j+1])
                    - inv_dy2 * (p_tile[i+1,j] + p_tile[i+1,j+2]) )
   
   wp.tile_store(y, acc, offset=(ofx+1, ofy+1))

# --------------------------------------
# serial cpu implementation of cg solver
# --------------------------------------

# apply 5-point operation
def apply_A_cpu(p): 
   
   y = np.zeros_like(p)
   inv_dx2 = 1./dx2
   inv_dy2 = 1./dy2
 
   # note that elements 0 and -1 are untouched on each dimension -> zero Dirichlet BC!
   y[1:-1,1:-1] = ( (2*inv_dx2 + 2*inv_dy2) * p[1:-1,1:-1] 
                - inv_dx2 * (p[ :-2,1:-1] + p[2:  ,1:-1])
                - inv_dy2 * (p[1:-1, :-2] + p[1:-1,2:  ]) )
   return y 

# solve laplacian with 5-point stencil
def cg_5pt_cpu(f, u0, tol=1e-6, max_iter=1000):
   
   u = u0.copy()
   r = f - apply_A_cpu(u, dx2, dy2)
   p = r.copy()

   rho = np.sum(r*r)

   for k in range(max_iter):
      
      q = apply_A_cpu(p)
      alpha = rho / np.sum(p*q)

      u += alpha * p
      r -= alpha * q
      
      rho_new = np.sum(r*r)
      if np.sqrt(rho_new) < tol: print( f"converged in {k+1} iterations" ); break
      
      beta = rho_new / rho
      p = r + beta * p
      rho = rho_new

   return u

# run solver
def run_cg_cpu():

   # constant forcing everywhere 
   f = np.zeros((Nx+2, Ny+2), dtype=np.float32) # include boundary cells
   f[1:-1,1:-1] = 1.

   # initial guess
   u0 = np.zeros_like(f)
    
   # solve 
   u = cg_5pt_cpu( f, u0 )

# ----------
# main tests
# ----------

def run_5pt_stencil(device):

   u = wp.zeros((Nx+2, Ny+2), dtype=wp.float32, device=device)
   y = wp.zeros_like(u)

   inv_dx2 = 1./dx2
   inv_dy2 = 1./dy2

   wp.launch( kernel = apply_A
            , dim    = u.shape
            , inputs = [u, y, inv_dx2, inv_dy2]
            , device = device )    

def run_5pt_stencil_tiled(device):
   
   u = wp.zeros((Nx+2, Ny+2), dtype=wp.float32, device=device)
   y = wp.zeros_like(u)

   u0 = np.zeros((Nx+2, Ny+2), dtype=np.float32) 
   u0[1:-1,1:-1] = 1.0
   wp.copy(u, wp.array(u0, dtype=wp.float32))

   inv_dx2 = 1./dx2
   inv_dy2 = 1./dy2

   dimx = (Nx + TILE - 1) // TILE # make sure we cover the entire grid
   dimy = (Ny + TILE - 1) // TILE # same

   wp.launch_tiled( kernel    = apply_A_tiled
                  , dim       = (dimx, dimy)
                  , inputs    = [u, y, inv_dx2, inv_dy2]
                  , block_dim = TILE_THREADS
                  , device    = device )

   if DEBUG > 0: 
      
      # impose bc again
      sol = np.zeros((Nx+2, Ny+2), dtype=np.float32) 
      sol[1:-1,1:-1] = y.numpy()[1:-1,1:-1]
      
      print('\nsolution computed with tiled kernel')
      print(sol)
      print('\nsolution computed with serial kernel on the CPU')
      print(apply_A_cpu(u0))
      assert(np.allclose(sol, apply_A_cpu(u0)))

# --------------
# run benchmarks
# --------------

if __name__ == "__main__":

   if DEBUG > 0: run_5pt_stencil_tiled("cuda:0"); exit()

   print("\nrunning warp cg benchmarks...")  
   devices = ["cpu", "cuda:0"]

   for dev in devices:
      print(f"\n--- device: {dev} ---")
      benchmark(run_5pt_stencil      , dev, "naive kernel")
      benchmark(run_5pt_stencil_tiled, dev, "tiled kernel")

