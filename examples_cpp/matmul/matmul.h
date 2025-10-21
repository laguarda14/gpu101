__global__ void matmul_kernel(float* A, float* B, float* C, int N1, int N2, int N3);
__global__ void tiled_matmul_kernel(float* A, float* B, float* C, int N1, int N2, int N3);

void matmul(std::vector<float>& A, std::vector<float>& B, std::vector<float>& C, int N1, int N2, int N3);
void tiled_matmul(std::vector<float>& A, std::vector<float>& B, std::vector<float>& C, int N1, int N2, int N3);