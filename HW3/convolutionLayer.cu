#include "common.cuh"

constexpr int KS = 3;       // kernel size
constexpr int PAD = KS / 2; // zero padding, stride=1 时保持 H, W 不变

// CPU reference: 按定义直接卷积, 不经过 im2col, 与 GPU 实现相互独立
// Y[n][o][h][w] = b[o] + sum_{c,kh,kw} X[n][c][h+kh-PAD][w+kw-PAD] *
// W[o][c][kh][kw]
void conv_forward_cpu(const float *X, const float *Wt, const float *b, float *Y,
                      int N, int Cin, int Cout, int H, int W) {
  for (int n = 0; n < N; ++n)
    for (int o = 0; o < Cout; ++o)
      for (int h = 0; h < H; ++h)
        for (int w = 0; w < W; ++w) {
          double s = b[o];
          for (int c = 0; c < Cin; ++c)
            for (int kh = 0; kh < KS; ++kh)
              for (int kw = 0; kw < KS; ++kw) {
                int ih = h + kh - PAD, iw = w + kw - PAD;
                if (ih < 0 || ih >= H || iw < 0 || iw >= W)
                  continue;
                s += (double)X[((n * Cin + c) * H + ih) * W + iw] *
                     Wt[((o * Cin + c) * KS + kh) * KS + kw];
              }
          Y[((n * Cout + o) * H + h) * W + w] = (float)s;
        }
}

// 与前向相同的循环: 前向每做一次 Y += X * W, 反向就对应
// dW += dY * X, dX += dY * W, db += dY
void conv_backward_cpu(const float *X, const float *Wt, const float *dY,
                       float *dX, float *dW, float *db, int N, int Cin,
                       int Cout, int H, int W) {
  std::vector<double> gx((size_t)N * Cin * H * W, 0.0);
  std::vector<double> gw((size_t)Cout * Cin * KS * KS, 0.0);
  std::vector<double> gb(Cout, 0.0);
  for (int n = 0; n < N; ++n)
    for (int o = 0; o < Cout; ++o)
      for (int h = 0; h < H; ++h)
        for (int w = 0; w < W; ++w) {
          double g = dY[((n * Cout + o) * H + h) * W + w];
          gb[o] += g;
          for (int c = 0; c < Cin; ++c)
            for (int kh = 0; kh < KS; ++kh)
              for (int kw = 0; kw < KS; ++kw) {
                int ih = h + kh - PAD, iw = w + kw - PAD;
                if (ih < 0 || ih >= H || iw < 0 || iw >= W)
                  continue;
                int xi = ((n * Cin + c) * H + ih) * W + iw;
                int wi = ((o * Cin + c) * KS + kh) * KS + kw;
                gw[wi] += g * X[xi];
                gx[xi] += g * Wt[wi];
              }
        }
  for (size_t i = 0; i < gx.size(); ++i)
    dX[i] = (float)gx[i];
  for (size_t i = 0; i < gw.size(); ++i)
    dW[i] = (float)gw[i];
  for (int o = 0; o < Cout; ++o)
    db[o] = (float)gb[o];
}

// GPU implementation
__global__ void im2col_kernel(const float *X, float *col, int Cin, int H,
                              int W) {
  const int HW = H * W;
  CUDA_KERNEL_LOOP(idx, Cin * KS * KS * HW) {
    int row = idx / HW, p = idx % HW;
    int c = row / (KS * KS), kh = row / KS % KS, kw = row % KS;
    int ih = p / W + kh - PAD, iw = p % W + kw - PAD;
    col[idx] = (ih >= 0 && ih < H && iw >= 0 && iw < W)
                   ? X[c * HW + ih * W + iw]
                   : 0.f;
  }
}

__global__ void col2im_kernel(const float *col, float *X, int Cin, int H,
                              int W) {
  const int HW = H * W;
  CUDA_KERNEL_LOOP(idx, Cin * HW) {
    int c = idx / HW, ih = idx / W % H, iw = idx % W;
    float s = 0.f;
    for (int kh = 0; kh < KS; ++kh)
      for (int kw = 0; kw < KS; ++kw) {
        int h = ih - kh + PAD, w = iw - kw + PAD;
        if (h < 0 || h >= H || w < 0 || w >= W)
          continue;
        s += col[(c * KS * KS + kh * KS + kw) * HW + h * W + w];
      }
    X[idx] = s;
  }
}

void conv_forward(const float *X, const float *Wt, const float *b, float *Y,
                  int N, int Cin, int Cout, int H, int W) {
  int HW = H * W, K = Cin * KS * KS;
  float *col = dalloc(K * HW);
  for (int n = 0; n < N; ++n) {
    // X_hat = im2col(X_n), (Cin, H, W) -> (K, HW)
    im2col_kernel<<<GRID(K * HW), BLOCK>>>(X + n * Cin * HW, col, Cin, H, W);
    CUDA_CHECK(cudaGetLastError());
    float *Yn = Y + n * Cout * HW;
    // Y_n = b * 1^T, (Cout, 1) x (1, HW) -> (Cout, HW)
    gemm_rowmajor(false, false, Cout, HW, 1, b, get_ones(HW), Yn);
    // Y_n += W * X_hat, (Cout, K) x (K, HW) -> (Cout, HW)
    gemm_rowmajor(false, false, Cout, HW, K, Wt, col, Yn, 1.f, 1.f);
  }
  CUDA_CHECK(cudaFree(col));
}

void conv_backward(const float *X, const float *Wt, const float *dY, float *dX,
                   float *dW, float *db, int N, int Cin, int Cout, int H,
                   int W) {
  int HW = H * W, K = Cin * KS * KS;
  float *col = dalloc(K * HW);
  for (int n = 0; n < N; ++n) {
    float beta = !n ? 0.f : 1.f; // n=0 时覆盖 dW, db, 之后累加
    // X_hat = im2col(X_n), (Cin, H, W) -> (K, HW)
    im2col_kernel<<<GRID(K * HW), BLOCK>>>(X + n * Cin * HW, col, Cin, H, W);
    CUDA_CHECK(cudaGetLastError());
    // dW += dY_n * X_hat^T, (Cout, HW) x (HW, K) -> (Cout, K), 在 N 上累加
    gemm_rowmajor(false, true, Cout, K, HW, dY + n * Cout * HW, col, dW, 1.f,
                  beta);
    float *dXn = dX + n * Cin * HW;
    // dX_hat = W^T * dY_n, (K, Cout) x (Cout, HW) -> (K, HW), 复用 col 存放
    gemm_rowmajor(true, false, K, HW, Cout, Wt, dY + n * Cout * HW, col);
    // dX_n = col2im(dX_hat), (K, HW) -> (Cin, H, W)
    col2im_kernel<<<GRID(Cin * HW), BLOCK>>>(col, dXn, Cin, H, W);
    CUDA_CHECK(cudaGetLastError());
    // db += dY_n * 1, (Cout, HW) x (HW, 1) -> (Cout, 1), 在 N 上累加
    gemm_rowmajor(false, false, Cout, 1, HW, dY + n * Cout * HW, get_ones(HW),
                  db, 1.f, beta);
  }
  CUDA_CHECK(cudaFree(col));
}

// ---- 测试
// 标量 loss: L = sum(Y .* R), 于是 dL/dY = R
static double conv_loss(const float *X, const float *Wt, const float *b,
                        float *Y, const std::vector<float> &R, int N, int Cin,
                        int Cout, int H, int W) {
  conv_forward(X, Wt, b, Y, N, Cin, Cout, H, W);
  std::vector<float> y = d2h(Y, (size_t)N * Cout * H * W);
  double L = 0;
  for (int i = 0; i < y.size(); ++i)
    L += (double)y[i] * R[i];
  return L;
}

// 对 device 上 p[idx] 做中心差分, 返回数值梯度
static double numeric_grad(float *p, int idx, double eps, const float *X,
                           const float *Wt, const float *b, float *Y,
                           const std::vector<float> &R, int N, int Cin,
                           int Cout, int H, int W) {
  float orig;
  CUDA_CHECK(cudaMemcpy(&orig, p + idx, sizeof(float), cudaMemcpyDeviceToHost));
  float plus = orig + (float)eps, minus = orig - (float)eps;
  CUDA_CHECK(cudaMemcpy(p + idx, &plus, sizeof(float), cudaMemcpyHostToDevice));
  double Lp = conv_loss(X, Wt, b, Y, R, N, Cin, Cout, H, W);
  CUDA_CHECK(
      cudaMemcpy(p + idx, &minus, sizeof(float), cudaMemcpyHostToDevice));
  double Lm = conv_loss(X, Wt, b, Y, R, N, Cin, Cout, H, W);
  CUDA_CHECK(cudaMemcpy(p + idx, &orig, sizeof(float), cudaMemcpyHostToDevice));
  return (Lp - Lm) / (2 * eps);
}

// im2col / col2im 伴随测试: <im2col(x), c> == <x, col2im(c)>
void test_im2col_adjoint() {
  const int Cin = 5, H = 7, W = 11;
  const int HW = H * W, K = Cin * KS * KS;
  printf("=== im2col / col2im adjoint (Cin=%d, H=%d, W=%d) ===\n", Cin, H, W);

  float *x = dalloc(Cin * HW), *c = dalloc(K * HW);
  float *col = dalloc(K * HW), *img = dalloc(Cin * HW);
  init_uniform(x, Cin * HW, -1.f, 1.f, 11);
  init_uniform(c, K * HW, -1.f, 1.f, 12);

  im2col_kernel<<<GRID(K * HW), BLOCK>>>(x, col, Cin, H, W);
  col2im_kernel<<<GRID(Cin * HW), BLOCK>>>(c, img, Cin, H, W);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  std::vector<float> hcol = d2h(col, (size_t)K * HW),
                     hc = d2h(c, (size_t)K * HW), hx = d2h(x, (size_t)Cin * HW),
                     himg = d2h(img, (size_t)Cin * HW);
  double lhs = 0, rhs = 0;
  for (size_t i = 0; i < hcol.size(); ++i)
    lhs += (double)hcol[i] * hc[i];
  for (size_t i = 0; i < hx.size(); ++i)
    rhs += (double)hx[i] * himg[i];
  double rel = fabs(lhs - rhs) / fmax(fabs(lhs), 1e-12);
  printf(
      "<im2col(x), c> = %.10f\n<x, col2im(c)> = %.10f\nrel err = %.3e -> %s\n",
      lhs, rhs, rel, rel < 1e-5 ? "PASS" : "FAIL");

  cudaFree(x);
  cudaFree(c);
  cudaFree(col);
  cudaFree(img);
}

void test_conv() {
  // 故意取非 2 的幂且 H != W, 容易暴露下标 / 转置写错的问题
  const int N = 3, Cin = 5, Cout = 7, H = 7, W = 11;
  const int HW = H * W, K = Cin * KS * KS;
  printf("=== Conv layer (N=%d, Cin=%d, Cout=%d, H=%d, W=%d) ===\n", N, Cin,
         Cout, H, W);

  float *X = dalloc(N * Cin * HW), *Wt = dalloc(Cout * K), *b = dalloc(Cout);
  float *Y = dalloc(N * Cout * HW), *dY = dalloc(N * Cout * HW);
  float *dX = dalloc(N * Cin * HW), *dW = dalloc(Cout * K), *db = dalloc(Cout);

  init_uniform(X, N * Cin * HW, -1.f, 1.f, 1);
  init_uniform(Wt, Cout * K, -0.1f, 0.1f, 2);
  init_uniform(b, Cout, -0.1f, 0.1f, 3);
  init_uniform(dY, N * Cout * HW, -1.f, 1.f, 4);
  CUDA_CHECK(cudaDeviceSynchronize());

  // GPU
  conv_forward(X, Wt, b, Y, N, Cin, Cout, H, W);
  conv_backward(X, Wt, dY, dX, dW, db, N, Cin, Cout, H, W);
  CUDA_CHECK(cudaDeviceSynchronize());

  // CPU 参考
  auto hX = d2h(X, N * Cin * HW), hW = d2h(Wt, Cout * K), hb = d2h(b, Cout),
       hdY = d2h(dY, N * Cout * HW);
  std::vector<float> rY(N * Cout * HW), rdX(N * Cin * HW), rdW(Cout * K),
      rdb(Cout);
  conv_forward_cpu(hX.data(), hW.data(), hb.data(), rY.data(), N, Cin, Cout, H,
                   W);
  conv_backward_cpu(hX.data(), hW.data(), hdY.data(), rdX.data(), rdW.data(),
                    rdb.data(), N, Cin, Cout, H, W);

  const double tol = 1e-4;
  double e_y = max_abs_diff(d2h(Y, N * Cout * HW), rY);
  double e_dx = max_abs_diff(d2h(dX, N * Cin * HW), rdX);
  double e_dw = max_abs_diff(d2h(dW, Cout * K), rdW);
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
      double num =
          numeric_grad(p, idx, eps, X, Wt, b, Y, hdY, N, Cin, Cout, H, W);
      double ana = analytic[idx];
      double rel = fabs(num - ana) / fmax(1e-3, fabs(ana));
      worst = fmax(worst, rel);
      printf("  grad-check %-2s[%4d]: numeric=% .5f analytic=% .5f rel=%.2e\n",
             name, idx, num, ana, rel);
    }
  };

  check("dW", Wt, Cin * Cout, d2h(dW, Cout * K));
  check("dX", X, N * Cin, d2h(dX, N * Cin * HW));
  check("db", b, Cout, d2h(db, Cout));
  printf("[grad check] worst rel err = %.3e -> %s\n", worst,
         worst < 1e-2 ? "PASS" : "FAIL");

  cudaFree(X);
  cudaFree(Wt);
  cudaFree(b);
  cudaFree(Y);
  cudaFree(dY);
  cudaFree(dX);
  cudaFree(dW);
  cudaFree(db);
}

int main() {
  CUBLAS_CHECK(cublasCreate(&g_cublas));
  test_im2col_adjoint();
  test_conv();
  CUBLAS_CHECK(cublasDestroy(g_cublas));
  return 0;
}
