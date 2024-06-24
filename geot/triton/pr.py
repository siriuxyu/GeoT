import triton
import triton.language as tl
import torch

@triton.jit
def combine_fn(left_values, left_indices, right_values, right_indices):
    same_segment = left_indices == right_indices
    combined_values = tl.where(same_segment, left_values + right_values, right_values)
    combined_indices = right_indices
    return combined_values, combined_indices

@triton.jit
def parallel_segment_reduction_kernel(
    index,  # the input index tensor
    in_feature,  # the input tensor
    result,  # the output value tensor
    num_edges: tl.constexpr,  # Number of elements in the input tensor (1d)
    feature_size: tl.constexpr,  # Number of features in the input tensor (2d)
    BLOCK_SIZE: tl.constexpr,  # Block size for the scan
):
    pid = tl.program_id(axis=0)
    offset_pid = pid // feature_size
    feature_id = pid % feature_size
    offsets = offset_pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < num_edges

    # Load input data
    values = tl.load(in_feature + offsets * feature_size + feature_id, mask=mask)
    indices = tl.load(index + offsets, mask=mask)
    indices_next = tl.load(index + offsets + 1, offsets < num_edges - 1)

    # Perform an inclusive scan using tl.associative_scan
    result_values, _ = tl.associative_scan(
        (values, indices,), axis=0, combine_fn=combine_fn
    )
    # if offset % BLOCK_SIZE == -1, it means the last element of the segment
    segment_start = (indices != indices_next) | (offsets % BLOCK_SIZE == BLOCK_SIZE - 1)
    tl.atomic_add(result + indices * feature_size + feature_id, result_values, mask & segment_start)
    
    
@triton.jit
def parallel_spmm_sorted_coo_kernel(
    edge_index,  # the input coo sparse matrix
    input,  # the input tensor
    output,  # the output value tensor
    num_edges: tl.constexpr,  # Number of elements in the input tensor (1d)
    feature_size: tl.constexpr,  # Number of features in the input tensor (2d)
    BLOCK_SIZE: tl.constexpr,  # Block size for the scan
):
    pid = tl.program_id(axis=0)
    offset_pid = pid // feature_size
    feature_id = pid % feature_size
    offsets = offset_pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < num_edges

    # Load input data
    in_idx = tl.load(edge_index + offsets, mask=mask)
    values = tl.load(input + in_idx * feature_size + feature_id, mask=mask)
    # values = tl.load(input + offsets * feature_size + feature_id, mask=mask)
    out_idx = tl.load(edge_index + offsets + num_edges, mask=mask)
    out_idx_next = tl.load(edge_index + offsets + num_edges + 1, offsets < num_edges - 1)

    # Perform an inclusive scan using tl.associative_scan
    result_values, _ = tl.associative_scan(
        (values, out_idx,), axis=0, combine_fn=combine_fn
    )
    # if offset % BLOCK_SIZE == -1, it means the last element of the segment
    segment_start = (out_idx != out_idx_next) | (offsets % BLOCK_SIZE == BLOCK_SIZE - 1)
    tl.atomic_add(output + out_idx * feature_size + feature_id, result_values, mask & segment_start)    


def launch_parallel_spmm(indices, input, output, num_edges:tl.constexpr, feature_size: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    # Launch the kernel
    grid = (triton.cdiv(num_edges, BLOCK_SIZE) * feature_size,)
    parallel_spmm_sorted_coo_kernel[grid](indices, input, output, num_edges, feature_size, BLOCK_SIZE)

def launch_parallel_reduction(indices, input, output, num_edges:tl.constexpr, feature_size: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    # Launch the kernel
    grid = (triton.cdiv(num_edges, BLOCK_SIZE) * feature_size,)
    parallel_segment_reduction_kernel[grid](indices, input, output, num_edges, feature_size, BLOCK_SIZE)

if __name__ == '__main__':
    # Create input data
    num_edges = 50
    feature_size = 16
    segs = 5
    input = torch.randn(num_edges, feature_size, dtype=torch.float32, device='cuda')
    out_indices = torch.randint(0, segs, (num_edges,), dtype=torch.int64, device='cuda')
    in_indices = torch.randint(0, segs, (num_edges,), dtype=torch.int64, device='cuda')
    indices, sort_indices = torch.sort(out_indices)

    edges = torch.stack([in_indices, indices], dim=0)
    print(edges.size())
    print(edges)
    # sort

    out_values = torch.zeros(segs, feature_size, dtype=torch.float32, device='cuda')
    # launch_parallel_reduction(indices, input, out_values, num_edges, feature_size, 32)
    launch_parallel_spmm(edges, input, out_values, num_edges, feature_size, 32)
    print(out_values)

    # Set mode

    # Check the result
    # we could use torch.scatter_add to verify the result
    out_values_torch = torch.zeros_like(out_values)
    out_values_torch.scatter_add_(0, indices.unsqueeze(1).expand_as(input), input)
    print(out_values_torch)
    # assert torch.allclose(out_values, out_values_torch), "Error: the output values are not equal!"
