// Common modules could be used in this session.
#pragma once

#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <curand.h>

#define CUDA_CHECK(x)                                                          \
  do {                                                                         \
    cudaError_t e = (x);                                                       \
    if (e != cudaSuccess) {                                                    \
      fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e),       \
              __FILE__, __LINE__);                                             \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

#define CUBLAS_CHECK(x)                                                        \
  do {                                                                         \
    cublasStatus_t s = (x);                                                    \
    if (s != CUBLAS_STATUS_SUCCESS) {                                          \
      fprintf(stderr, "CUBLAS error %d at %s:%d\n", (int)s, __FILE__,          \
              __LINE__);                                                       \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

#define CURAND_CHECK(x)                                                        \
  do {                                                                         \
    curandStatus_t s = (x);                                                    \
    if (s != CURAND_STATUS_SUCCESS) {                                          \
      fprintf(stderr, "CURAND error %d at %s:%d\n", (int)s, __FILE__,          \
              __LINE__);                                                       \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

#define CUDA_KERNEL_LOOP(i, n)                                                 \
  for (int i = blockDim.x * blockIdx.x + threadIdx.x; i < (n);                 \
       i += blockDim.x * gridDim.x)

constexpr int BLOCK = 256;
static inline int GRID(int n) { return (n - 1 + BLOCK) / BLOCK; }

inline cublasHandle_t g_cublas;

static __global__ void fill_kernel(float *x, int n, float v) {
  CUDA_KERNEL_LOOP(i, n) x[i] = v;
}

static __global__ void affine_kernel(float *x, int n, float lo, float hi) {
  CUDA_KERNEL_LOOP(i, n) x[i] = lo + (hi - lo) * x[i];
}

inline void init_uniform(float *d_ptr, int n, float lo, float hi,
                         unsigned long long seed) {
  curandGenerator_t gen;
  CURAND_CHECK(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));
  CURAND_CHECK(curandSetPseudoRandomGeneratorSeed(gen, seed));
  CURAND_CHECK(curandGenerateUniform(gen, d_ptr, n));
  CURAND_CHECK(curandDestroyGenerator(gen));
  affine_kernel<<<GRID(n), BLOCK>>>(d_ptr, n, lo, hi);
  CUDA_CHECK(cudaGetLastError());
}

inline float *dalloc(size_t n) {
  float *p;
  CUDA_CHECK(cudaMalloc(&p, n * sizeof(float)));
  return p;
}

inline std::vector<float> d2h(float *d, size_t n) {
  std::vector<float> h(n);
  CUDA_CHECK(
      cudaMemcpy(h.data(), d, n * sizeof(float), cudaMemcpyDeviceToHost));
  return h;
}

inline void h2d(float *d, const std::vector<float> &h) {
  CUDA_CHECK(cudaMemcpy(d, h.data(), h.size() * sizeof(float),
                        cudaMemcpyHostToDevice));
}

inline double max_abs_diff(const std::vector<float> &a,
                           const std::vector<float> &b) {
  double m = 0;
  for (size_t i = 0; i < a.size(); ++i)
    m = fmax(m, fabs((double)a[i] - (double)b[i]));
  return m;
}

inline const float *get_ones(int n) {
  static float *d_ones = nullptr;
  static int cap = 0;
  if (n > cap) {
    if (d_ones)
      CUDA_CHECK(cudaFree(d_ones));
    d_ones = dalloc(n);
    cap = n;
    fill_kernel<<<GRID(n), BLOCK>>>(d_ones, n, 1.f);
    CUDA_CHECK(cudaGetLastError());
  }
  return d_ones;
}

// 行主序 GEMM 封装
//
// 计算行主序:  C(m x n) = alpha * op(A) * op(B) + beta * C
//
//   op(A) 为 m x k, op(B) 为 k x n
//
//   transA=false: A 在内存中是 m x k;  transA=true: A 在内存中是 k x m
//
//   transB=false: B 在内存中是 k x n;  transB=true: B 在内存中是 n x k
//
// 原理: cuBLAS 列主序下, 行主序矩阵 M 相当于 M^T。
//
//   C = op(A) op(B)  <=>  C^T = op(B)^T op(A)^T
//
// 所以把 B、A 的顺序对调, 并把 m、n 对调即可。
inline void gemm_rowmajor(bool transA, bool transB, int m, int n, int k,
                          const float *A, const float *B, float *C,
                          float alpha = 1.f, float beta = 0.f) {
  int lda = transA ? m : k;
  int ldb = transB ? k : n;
  CUBLAS_CHECK(cublasSgemm(g_cublas, transB ? CUBLAS_OP_T : CUBLAS_OP_N,
                           transA ? CUBLAS_OP_T : CUBLAS_OP_N, n, m, k, &alpha,
                           B, ldb, A, lda, &beta, C, n));
}