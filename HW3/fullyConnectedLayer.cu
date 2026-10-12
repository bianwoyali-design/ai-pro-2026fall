#include "common.cuh"

// CPU reference
void fc_forward_cpu(const float *X, const float *W, const float *b, float *Y,
                    int N, int Cin, int Cout) {
  for (int n = 0; n < N; ++n)
    for (int j = 0; j < Cout; ++j) {
      double s = b[j];
      for (int i = 0; i < Cin; ++i)
        s += X[n * Cin + i] * W[i * Cout + j];
      Y[n * Cout + j] = (float)s;
    }
}

void fc_backward_cpu(const float *X, const float *W, const float *dY, float *dX,
                     float *dW, float *db, int N, int Cin, int Cout) {
  // dW = op(X^T) * op(dY), (Cin, Cout)
  for (int i = 0; i < Cin; ++i)
    for (int j = 0; j < Cout; ++j) {
      double s = 0;
      for (int n = 0; n < N; ++n)
        s += X[n * Cin + i] * dY[n * Cout + j];
      dW[i * Cout + j] = (float)s;
    }

  // dX = op(dY) * op(W^T), (N, Cin)
  for (int n = 0; n < N; ++n)
    for (int i = 0; i < Cin; ++i) {
      double s = 0;
      for (int j = 0; j < Cout; ++j)
        s += dY[n * Cout + j] * W[i * Cout + j];
      dX[n * Cin + i] = (float)s;
    }

  // db = op(dY^T) * 1, (Cout, 1)
  for (int j = 0; j < Cout; ++j) {
    double s = 0;
    for (int n = 0; n < N; ++n)
      s += dY[n * Cout + j];
    db[j] = (float)s;
  }
}

// GPU implementation
void fc_forward(const float *X, const float *W, const float *b, float *Y, int N,
                int Cin, int Cout) {
  gemm_rowmajor(false, false, N, Cout, 1, get_ones(N), b, Y);
  gemm_rowmajor(false, false, N, Cout, Cin, X, W, Y, 1.f, 1.f);
}

void fc_backward(const float *X, const float *W, const float *dY, float *dX,
                 float *dW, float *db, int N, int Cin, int Cout) {
  gemm_rowmajor(true, false, Cin, Cout, N, X, dY, dW);
  gemm_rowmajor(false, true, N, Cin, Cout, dY, W, dX);
  gemm_rowmajor(false, false, 1, Cout, N, get_ones(N), dY, db);
}

// ---- 测试
// 标量 loss: L = sum(Y .* R), 于是 dL/dY = R
static double fc_loss(const float *dX_in, const float *dW_in,
                      const float *db_in, float *dY_buf,
                      const std::vector<float> &R, int N, int Cin, int Cout) {
  fc_forward(dX_in, dW_in, db_in, dY_buf, N, Cin, Cout);
  std::vector<float> y = d2h(dY_buf, (size_t)N * Cout);
  double L = 0;
  for (size_t i = 0; i < y.size(); ++i)
    L += (double)y[i] * R[i];
  return L;
}

// 对 device 上 p[idx] 做中心差分, 返回数值梯度
static double numeric_grad(float *p, int idx, double eps, float *X, float *W,
                           float *b, float *Y, const std::vector<float> &R,
                           int N, int Cin, int Cout) {
  float orig;
  CUDA_CHECK(cudaMemcpy(&orig, p + idx, sizeof(float), cudaMemcpyDeviceToHost));
  float plus = orig + (float)eps, minus = orig - (float)eps;
  CUDA_CHECK(cudaMemcpy(p + idx, &plus, sizeof(float), cudaMemcpyHostToDevice));
  double Lp = fc_loss(X, W, b, Y, R, N, Cin, Cout);
  CUDA_CHECK(
      cudaMemcpy(p + idx, &minus, sizeof(float), cudaMemcpyHostToDevice));
  double Lm = fc_loss(X, W, b, Y, R, N, Cin, Cout);
  CUDA_CHECK(cudaMemcpy(p + idx, &orig, sizeof(float), cudaMemcpyHostToDevice));
  return (Lp - Lm) / (2.0 * eps);
}

void test_fc() {
  // 故意取非 2 的幂, 容易暴露 ld / 转置写错的问题
  const int N = 283, Cin = 1302, Cout = 29;
  printf("=== FC layer (N=%d, Cin=%d, Cout=%d) ===\n", N, Cin, Cout);

  float *X = dalloc(N * Cin), *W = dalloc(Cin * Cout), *b = dalloc(Cout);
  float *Y = dalloc(N * Cout), *dY = dalloc(N * Cout);
  float *dX = dalloc(N * Cin), *dW = dalloc(Cin * Cout), *db = dalloc(Cout);

  init_uniform(X, N * Cin, -1.f, 1.f, 1);
  init_uniform(W, Cin * Cout, -0.1f, 0.1f, 2);
  init_uniform(b, Cout, -0.1f, 0.1f, 3);
  init_uniform(dY, N * Cout, -1.f, 1.f, 4);
  CUDA_CHECK(cudaDeviceSynchronize());

  // GPU
  fc_forward(X, W, b, Y, N, Cin, Cout);
  fc_backward(X, W, dY, dX, dW, db, N, Cin, Cout);
  CUDA_CHECK(cudaDeviceSynchronize());

  // CPU 参考
  auto hX = d2h(X, N * Cin), hW = d2h(W, Cin * Cout), hb = d2h(b, Cout),
       hdY = d2h(dY, N * Cout);
  std::vector<float> rY(N * Cout), rdX(N * Cin), rdW(Cin * Cout), rdb(Cout);
  fc_forward_cpu(hX.data(), hW.data(), hb.data(), rY.data(), N, Cin, Cout);
  fc_backward_cpu(hX.data(), hW.data(), hdY.data(), rdX.data(), rdW.data(),
                  rdb.data(), N, Cin, Cout);

  const double tol = 1e-4;
  double e_y = max_abs_diff(d2h(Y, N * Cout), rY);
  double e_dx = max_abs_diff(d2h(dX, N * Cin), rdX);
  double e_dw = max_abs_diff(d2h(dW, Cin * Cout), rdW);
  double e_db = max_abs_diff(d2h(db, Cout), rdb);
  printf("[vs CPU] max|diff|: Y=%.3e  dX=%.3e  dW=%.3e  db=%.3e  -> %s\n", e_y,
         e_dx, e_dw, e_db,
         (e_y < tol && e_dx < tol && e_dw < tol && e_db < tol) ? "PASS"
                                                               : "FAIL");

  // 数值梯度检验: 以 R = dY 作为 dL/dY, 抽查若干元素
  const double eps = 1e-2;
  double worst = 0;
  auto check = [&](const char *name, float *p, int count,
                   const std::vector<float> &analytic) {
    int samples = count < 64 ? count : 5;
    for (int t = 0; t < samples; ++t) {
      int idx = samples == count ? t : (int)((long long)(t + 1) * 7919 % count);
      double num = numeric_grad(p, idx, eps, X, W, b, Y, hdY, N, Cin, Cout);
      double ana = analytic[idx];
      double rel = fabs(num - ana) / fmax(1e-3, fabs(ana));
      worst = fmax(worst, rel);
      printf("  grad-check %-2s[%4d]: numeric=% .5f analytic=% .5f rel=%.2e\n",
             name, idx, num, ana, rel);
    }
  };
  check("dW", W, Cin * Cout, d2h(dW, Cin * Cout));
  check("dX", X, N * Cin, d2h(dX, N * Cin));
  check("db", b, Cout, d2h(db, Cout));
  printf("[grad check] worst rel err = %.3e -> %s\n", worst,
         worst < 1e-2 ? "PASS" : "FAIL");

  cudaFree(X);
  cudaFree(W);
  cudaFree(b);
  cudaFree(Y);
  cudaFree(dY);
  cudaFree(dX);
  cudaFree(dW);
  cudaFree(db);
}

int main() {
  CUBLAS_CHECK(cublasCreate(&g_cublas));
  test_fc();
  CUBLAS_CHECK(cublasDestroy(g_cublas));
  return 0;
}
