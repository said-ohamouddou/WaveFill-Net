#include "knn_query.h"
#include "utils.h"

void knnquery_cuda_launcher(int b, int n, int m, int nsample, 
                           const float *xyz, const float *new_xyz, 
                           int *idx, float *dist2);

std::vector<at::Tensor> knn_query(at::Tensor xyz, at::Tensor new_xyz, const int nsample) {
  CHECK_CONTIGUOUS(xyz);
  CHECK_CONTIGUOUS(new_xyz);
  CHECK_IS_FLOAT(xyz);
  CHECK_IS_FLOAT(new_xyz);

  if (xyz.is_cuda()) {
    CHECK_CUDA(new_xyz);
  }

  at::Tensor idx =
      torch::zeros({new_xyz.size(0), new_xyz.size(1), nsample},
                   at::device(new_xyz.device()).dtype(at::ScalarType::Int));
  
  at::Tensor dist2 =
      torch::zeros({new_xyz.size(0), new_xyz.size(1), nsample},
                   at::device(new_xyz.device()).dtype(at::ScalarType::Float));

  if (xyz.is_cuda()) {
    knnquery_cuda_launcher(xyz.size(0), xyz.size(1), new_xyz.size(1), nsample,
                          xyz.data_ptr<float>(), new_xyz.data_ptr<float>(),
                          idx.data_ptr<int>(), dist2.data_ptr<float>());
  } else {
    AT_ASSERT(false, "CPU not supported");
  }

  return {dist2, idx};
}
