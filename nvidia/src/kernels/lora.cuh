// FP32 rank-1 PEFT delta. One block per token, 256 threads; A is [1,k], B [d,1].
extern "C" __global__ void lora_rank1_dot(const float *a, const float *x, float *dot, int k) {
  __shared__ float sums[256];
  const int tid = threadIdx.x;
  const float *row = x + (size_t)blockIdx.x * k;
  float s = 0.0f;
  for (int j = tid; j < k; j += 256) s = fmaf(a[j], row[j], s);
  sums[tid] = s;
  __syncthreads();
  for (int stride = 128; stride > 0; stride >>= 1) {
    if (tid < stride) sums[tid] += sums[tid + stride];
    __syncthreads();
  }
  if (tid == 0) dot[blockIdx.x] = sums[0];
}

extern "C" __global__ void lora_rank1_add(const float *b, const float *dot, float *y, int d, float scale) {
  const int i = blockIdx.x * 256 + threadIdx.x;
  if (i < d) {
    const size_t p = (size_t)blockIdx.y * d + i;
    y[p] = fmaf(scale * b[i], dot[blockIdx.y], y[p]);
  }
}
