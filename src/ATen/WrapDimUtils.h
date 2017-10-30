#pragma once

#include "ATen/TensorImpl.h"
#include <sstream>

namespace at {

static inline int64_t maybe_wrap_dim(int64_t dim, int64_t dim_post_expr) {
  if (dim_post_expr <= 0) {
    std::ostringstream oss;
    oss << "dimension specified as " << dim << " but tensor has no dimensions";
    throw std::runtime_error(oss.str());
  }
  if (dim < -(dim_post_expr) || dim >= (dim_post_expr)) {
    std::ostringstream oss;
    oss << "dimension out of range (expected to be in range of [" << -(dim_post_expr)
        << ", " << (dim_post_expr)-1 << "], but got " << dim << ")",
    throw std::runtime_error(oss.str());
  }
  if (dim  < 0) dim += dim_post_expr;
  return dim;
}

static inline int64_t maybe_wrap_dim(int64_t dim, TensorImpl *tensor, int64_t to_add) {
  return maybe_wrap_dim(dim, tensor->dim() + to_add);
}

static inline int64_t maybe_wrap_dim(int64_t dim, TensorList tensors, int64_t to_add) {
  for (int i = 0; i < tensors.size(); i++) {
    if (tensors[i].dim() > 0)
      return maybe_wrap_dim(dim, tensors[i].dim() + to_add);
  }
  // can't wrap if TensorList is empty or all tensors are empty.
  // rely on underlying implementation to throw error if necessary.
  return dim;
}

}
