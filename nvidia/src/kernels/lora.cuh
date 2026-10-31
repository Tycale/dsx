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

// Single-token batches: repeat the identical dot reduction in several blocks, then let
// each block write a disjoint strided output partition. This saves one launch
// without serializing every output store onto one block. Do not fold scale into
// the dot or change the reduction tree / FMA expression from dot + add.
extern "C" __global__ void lora_rank1_fused(const float *a, const float *b, const float *x, float *y, int k, int d, float scale) {
  __shared__ float sums[256];
  const int tid = threadIdx.x;
  const float *row = x;
  float s = 0.0f;
  for (int j = tid; j < k; j += 256) s = fmaf(a[j], row[j], s);
  sums[tid] = s;
  __syncthreads();
  for (int stride = 128; stride > 0; stride >>= 1) {
    if (tid < stride) sums[tid] += sums[tid + stride];
    __syncthreads();
  }
  const float dot = sums[0];
  for (int i = blockIdx.x * 256 + tid; i < d; i += gridDim.x * 256) {
    y[i] = fmaf(scale * b[i], dot, y[i]);
  }
}
