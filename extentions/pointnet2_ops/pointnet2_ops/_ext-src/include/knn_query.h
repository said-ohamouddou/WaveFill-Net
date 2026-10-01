#pragma once
#include <torch/extension.h>
#include <vector>

std::vector<at::Tensor> knn_query(at::Tensor xyz, at::Tensor new_xyz, const int nsample);

