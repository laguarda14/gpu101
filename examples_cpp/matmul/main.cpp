// @file main.cpp
// @author Luis Laguarda [lluis.laguarda@gmail.com]
#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <iomanip>
#include <cuda_runtime.h>
#include "matmul.h"

#define MAX_NUM  10
#define MIN_NUM -10
#define CORRECTNESS_TOL 1e-3f

bool check_correctness(const std::vector<float>& C1, const std::vector<float>& C2, int N)
{
   for (int i = 0; i < N; i++)
   {
      if (std::fabs(C1[i] - C2[i]) > CORRECTNESS_TOL * std::fabs(C1[i]) + CORRECTNESS_TOL)
      {
         std::cerr << "Mismatch at index " << i
                   << ": naive=" << C1[i] << " tiled=" << C2[i] << "\n";
         return false;
      }
   }
   return true;
}

int main(int argc, char **argv){

   // size of matrices — optionally override via CLI: ./matmul N1 N2 N3
   int N1 = 2678;
   int N2 = 2678;
   int N3 = 2678;
   if (argc == 4)
   {
      N1 = std::stoi(argv[1]);
      N2 = std::stoi(argv[2]);
      N3 = std::stoi(argv[3]);
   }
   std::cout << "Matrix sizes: A[" << N1 << "x" << N2 << "]"
             << " * B[" << N2 << "x" << N3 << "]\n";

   // generate matrices with uniform random values
   std::mt19937 rng(42); // random number engine following the Mersenne Twister algorithm
   std::uniform_real_distribution<float> dist(MIN_NUM, MAX_NUM);

   std::vector<float> A(N1 * N2);
   std::vector<float> B(N2 * N3);
   for (auto& v : A) v = dist(rng);
   for (auto& v : B) v = dist(rng);

   std::vector<float> C_naive(N1 * N3, 0.0f);
   std::vector<float> C_tiled(N1 * N3, 0.0f);

   // warm up the GPU to avoid measuring driver init in the first timed call
   cudaFree(0);

   // -----------------------------------------------------------------------
   // naive matmul
   // -----------------------------------------------------------------------
   auto  wall_start = std::chrono::high_resolution_clock::now();
   float naive_ms   = naive_matmul(A, B, C_naive, N1, N2, N3);
   auto  wall_end   = std::chrono::high_resolution_clock::now();
   float wall_ms    = std::chrono::duration<float, std::milli>(wall_end - wall_start).count();

   std::cout << "naive  — kernel: " << std::fixed << std::setprecision(2) << naive_ms << " ms"
             << "  |  total (alloc+transfer+kernel): " << std::fixed << std::setprecision(2) << wall_ms << " ms\n";

   // -----------------------------------------------------------------------
   // tiled matmul
   // -----------------------------------------------------------------------
   wall_start       = std::chrono::high_resolution_clock::now();
   float tiled_ms   = tiled_matmul(A, B, C_tiled, N1, N2, N3);
   wall_end         = std::chrono::high_resolution_clock::now();
   wall_ms          = std::chrono::duration<float, std::milli>(wall_end - wall_start).count();

   std::cout << "tiled  — kernel: " << std::fixed << std::setprecision(2) << tiled_ms << " ms"
             << "  |  total (alloc+transfer+kernel): " << std::fixed << std::setprecision(2) << wall_ms << " ms\n";

   // -----------------------------------------------------------------------
   // correctness check
   // -----------------------------------------------------------------------
   if (check_correctness(C_naive, C_tiled, N1 * N3))
      std::cout << "Correctness check PASSED\n";
   else
      std::cout << "Correctness check FAILED\n";

   return 0;
}
