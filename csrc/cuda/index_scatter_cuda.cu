#include "../reduceutils.h"
#include "index_scatter_cuda.h"
#include "index_scatter_kernel.cuh"
#include <ATen/Dispatch.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/detail/IndexUtils.cuh>
#include <ATen/cuda/detail/TensorInfo.cuh>

using namespace at::native;

// policy listed in template
template <typename scalar_t, int NPerThread, int NThreadX, int NnzPerThread,
          int NnzThreadY>
void segscan_sr_sorted(const at::Tensor &index, const at::Tensor &src,
                       const at::Tensor &dst) {
  const auto nnz = index.numel();
  const auto N = src.numel() / nnz;
  const auto key = dst.numel() / N;
  auto indices = index.data_ptr<int64_t>();
  auto src_data = src.data_ptr<scalar_t>();
  auto dst_data = dst.data_ptr<scalar_t>();

  int blockDimX = NThreadX;
  int blockDimY = NnzThreadY;

  dim3 gridDim(CEIL(N, NThreadX * NPerThread),
               CEIL(nnz, NnzThreadY * NnzPerThread), 1);
  dim3 blockDim(blockDimX, blockDimY, 1);

  segscan_sr_sorted_kernel<scalar_t, NPerThread, NThreadX, NnzPerThread,
                           NnzThreadY>
      <<<gridDim, blockDim>>>(nnz, N, src_data, indices, dst_data);
}

template <typename scalar_t, int NPerThread, int NThreadY, int NnzPerThread,
          int RNum, int RSync>
void segscan_pr_sorted(const at::Tensor &index, const at::Tensor &src,
                       const at::Tensor &dst) {
  const auto nnz = index.numel();
  const auto N = src.numel() / nnz;
  const auto key = dst.numel() / N;
  auto indices = index.data_ptr<int64_t>();
  auto src_data = src.data_ptr<scalar_t>();
  auto dst_data = dst.data_ptr<scalar_t>();

  int blockDimX = RSync * RNum;
  int blockDimY = NThreadY;

  dim3 gridDim(CEIL(nnz, RSync * RNum * NnzPerThread),
               CEIL(N, NThreadY * NPerThread), 1);
  dim3 blockDim(blockDimX, blockDimY, 1);

  segscan_pr_sorted_kernel<scalar_t, NPerThread, NThreadY, NnzPerThread, RNum,
                           RSync>
      <<<gridDim, blockDim>>>(nnz, N, src_data, indices, dst_data);
}

template <typename scalar_t, ReductionType reduce>
void index_scatter_sorted_wrapper(const at::Tensor &index,
                                  const at::Tensor &src,
                                  const at::Tensor &dst) {
  const auto nnz = index.numel();
  const auto N = src.numel() / nnz;
  const auto keys = dst.numel() / N;
  if (N >= 1 && N <= 4) {
    segscan_pr_sorted<scalar_t, 1, 1, 2, 4, 32>(index, src, dst);
  } else if (N > 4 && N < 32) {
    segscan_pr_sorted<scalar_t, 2, 2, 2, 4, 32>(index, src, dst);
  } else if (N >= 32 && N < 64) {
    int avg_key_len = nnz / keys;
    if (avg_key_len < 16) {
      segscan_sr_sorted<scalar_t, 2, 16, 32, 1>(index, src, dst);
    } else if (avg_key_len >= 16 && avg_key_len < 64) {
      segscan_sr_sorted<scalar_t, 2, 16, 32, 2>(index, src, dst);
    } else {
      segscan_sr_sorted<scalar_t, 2, 16, 32, 4>(index, src, dst);
    }
  } else if (N >= 64 && N < 128) {
    if (avg_key_len < 16) {
      segscan_sr_sorted<DType, 2, 32, 32, 1>(nnz, N, index, src, dst);
    } else if (avg_key_len >= 16 && avg_key_len < 64) {
      segscan_sr_sorted<DType, 2, 32, 32, 2>(nnz, N, index, src, dst);
    } else {
      segscan_sr_sorted<DType, 2, 32, 32, 4>(nnz, N, index, src, dst);
    }
  } else {
    if (avg_key_len < 16) {
      segscan_sr_sorted<DType, 2, 64, 32, 1>(nnz, N, index, src, dst);
    } else if (avg_key_len >= 16 && avg_key_len < 64) {
      segscan_sr_sorted<DType, 2, 64, 32, 2>(nnz, N, index, src, dst);
    } else {
      segscan_sr_sorted<DType, 2, 64, 32, 4>(nnz, N, index, src, dst);
    }
  }
}

void index_scatter_sorted_dispatch(const at::Tensor &index,
                                   const at::Tensor &src, const at::Tensor &dst,
                                   const ReductionType &reduction) {
  AT_DISPATCH_FLOATING_TYPES(src.scalar_type(), "index_scatter_sorted", [&] {
    DISPATCH_REDUCTION_TYPES(reduction, [&]() {
      index_scatter_sorted_wrapper<scalar_t, reduce>(index, src, dst);
    });
  });
}

at::Tensor index_scatter_cuda(const int64_t dim, const at::Tensor &index,
                              const at::Tensor &src, const at::Tensor &dst,
                              const c10::string_view reduce,
                              const bool sorted) {
  TORCH_CHECK(dim >= 0 && dim < src.dim(),
              "dim must be non-negative and less than input dimensions");
  TORCH_CHECK(index.dim() == 1, "index must be 1 dimensional");
  TORCH_CHECK(src.size(dim) == index.size(0),
              "index length must be equal to src dimension size");

  auto reduce_type = get_reduction_enum(reduce);
  // we will use the src as the output (self in the kernel)

  if (sorted) {
    index_scatter_sorted_dispatch(index, src, dst, reduce_type);
  } else {
    TORCH_CHECK(false, "unsorted index is not supported yet");
  }

  return dst;
}
