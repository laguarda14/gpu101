// @file main.cpp
// @author Luis Laguarda [lluis.laguarda@gmail.com]
#include <iostream>
#include <vector>
#include <chrono>
#include <cuda_runtime.h>
#include "matmul.h"

#define MAX_NUM  10
#define MIN_NUM -10

int main(int argc, char **argv){

   // size of matrices
   int N1 = 2678;
   int N2 = 2678;
   int N3 = 2678;

   // generate N1xN2 matrix A
   std::vector<float> A(N1*N2);
   for (int i = 0; i < N1; i++)
   {
      for (int j = 0; j < N2; j++)
         A[i*N2+j] = (float)(rand() % (MAX_NUM - MIN_NUM + 1) + MIN_NUM);
   }

   // generate N2xN3 matrix B
   std::vector<float> B(N2*N3);
   for (int i = 0; i < N2; i++)
   {
      for (int j = 0; j < N3; j++)
         B[i*N3+j] = (float)(rand() % (MAX_NUM - MIN_NUM + 1) + MIN_NUM);
   }

   // define target N1xN3 matrix C
   std::vector<float> C(N1*N3);

   // matmul on the GPU
   auto start = std::chrono::high_resolution_clock::now();
   matmul(A, B, C, N1, N2, N3);
   // tiled_matmul(A, B, C, N1, N2, N3);
   auto end = std::chrono::high_resolution_clock::now();

   // peport
   std::cout << "matmul in " << std::chrono::duration_cast<std::chrono::duration<float>>(end - start).count() << " sec\n";


   return 0;
}
