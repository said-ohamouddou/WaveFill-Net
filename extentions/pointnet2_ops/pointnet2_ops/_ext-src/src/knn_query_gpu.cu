#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "cuda_utils.h"

// input: xyz(b, n, 3) new_xyz(b, m, 3)
// output: idx(b, m, nsample) dist2(b, m, nsample)
__global__ void knn_query_kernel(int b, int n, int m, int nsample,
                                const float *__restrict__ xyz,
                                const float *__restrict__ new_xyz,
                                int *__restrict__ idx,
                                float *__restrict__ dist2) {
  int batch_index = blockIdx.x;
  xyz += batch_index * n * 3;
  new_xyz += batch_index * m * 3;
  idx += batch_index * m * nsample;
  dist2 += batch_index * m * nsample;

  int index = threadIdx.x;
  int stride = blockDim.x;

  for (int j = index; j < m; j += stride) {
    float new_x = new_xyz[j * 3 + 0];
    float new_y = new_xyz[j * 3 + 1];
    float new_z = new_xyz[j * 3 + 2];
    
    // Initialize output arrays with very large distances
    for (int l = 0; l < nsample; ++l) {
      dist2[j * nsample + l] = 1e40;
      idx[j * nsample + l] = 0;
    }
    
    // Find k nearest neighbors using insertion sort approach
    for (int k = 0; k < n; ++k) {
      float x = xyz[k * 3 + 0];
      float y = xyz[k * 3 + 1];
      float z = xyz[k * 3 + 2];
      float d2 = (new_x - x) * (new_x - x) + (new_y - y) * (new_y - y) +
                 (new_z - z) * (new_z - z);
      
      // Check if this distance should be in the k nearest neighbors
      if (d2 < dist2[j * nsample + nsample - 1]) {
        // Find insertion position
        int pos = nsample - 1;
        while (pos > 0 && d2 < dist2[j * nsample + pos - 1]) {
          pos--;
        }
        
        // Shift elements to make room
        for (int p = nsample - 1; p > pos; p--) {
          dist2[j * nsample + p] = dist2[j * nsample + p - 1];
          idx[j * nsample + p] = idx[j * nsample + p - 1];
        }
        
        // Insert new element
        dist2[j * nsample + pos] = d2;
        idx[j * nsample + pos] = k;
      }
    }
  }
}

void knnquery_cuda_launcher(int b, int n, int m, int nsample,
                           const float *xyz, const float *new_xyz,
                           int *idx, float *dist2) {
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  
  knn_query_kernel<<<b, opt_n_threads(m), 0, stream>>>(
      b, n, m, nsample, xyz, new_xyz, idx, dist2);

  CUDA_CHECK_ERRORS();
}
