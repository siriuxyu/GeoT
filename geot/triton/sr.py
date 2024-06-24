import triton
import triton.language as tl
import torch

@triton.jit
def serial_spmm_sorted_coo_naive_kernel(
    edge_index, 
    input, 
    output, 
    num_edges: tl.constexpr, 
    feature_size: tl.constexpr, 
    group_size: tl.constexpr
):
    group_id = tl.program_id(0)
    node_offset = group_id * group_size
    f_index = tl.arange(0, feature_size)
    
    accumulate = tl.zeros((feature_size,), dtype=tl.float32)

    for ii in range(group_size):  # Iterate over the group
        xn = ii + node_offset  # Get node index
        mask = xn < num_edges  # Check if the node index is valid
        
        # Load 1st row as src, 2nd row as dst
        out_node = tl.load(edge_index + xn + num_edges, mask=mask)
        next_node = tl.load(edge_index + xn + 1 + num_edges, mask = (xn+1) < num_edges)
        
        in_node = tl.load(edge_index + xn, mask=mask)  # Load the input node
        val = tl.load(input + in_node * feature_size + f_index, mask=mask)
        accumulate += val
        # Check for end of segment
        if out_node != next_node or ii == group_size - 1:
            # Perform atomic addition
            tl.atomic_add(output + out_node * feature_size +
                          f_index, accumulate, mask=mask)
            # Reset val for the next segment
            accumulate = tl.zeros(accumulate.shape, dtype=accumulate.dtype)
   
    
@triton.jit
def serial_segment_reduction_kernel(
        index, 
        in_feature, 
        result, 
        num_edges: tl.constexpr, 
        feature_size: tl.constexpr, 
        group_size: tl.constexpr
):
    group_id = tl.program_id(axis=0)
    node_offset = group_id * group_size
    f_index = tl.arange(0, feature_size)
    
    accumulate = tl.zeros((feature_size,), dtype=tl.float32)
    
    for ii in range(group_size):  # Iterate over the group
        xn = ii + node_offset  # Get node index
        mask = xn < num_edges  # Check if the node index is valid
        
        node_idx = tl.load(index + xn, mask=mask)
        next_idx = tl.load(index + xn + 1, mask = (xn+1) < num_edges)
        
        val = tl.load(in_feature + xn * feature_size + f_index, mask=mask)
        accumulate += val
        # Check for end of segment
        if node_idx != next_idx or ii == group_size - 1:
            # Perform atomic addition
            tl.atomic_add(result + node_idx * feature_size +
                          f_index, accumulate, mask=mask)
            # Clear accumulate for the next segment
            accumulate = tl.zeros(accumulate.shape, dtype=accumulate.dtype)


def launch_serial_spmm(edges, input, output, num_edges, feature_size, group_size):
    # Launch the kernel
    grid = (triton.cdiv(num_edges, group_size),)
    serial_spmm_sorted_coo_naive_kernel[grid](edges, input, output, num_edges, feature_size, group_size)
    

def launch_serial_reduction(edges, input, output, num_edges, feature_size, group_size):
    # Launch the kernel
    grid = (triton.cdiv(num_edges, group_size),)
    serial_segment_reduction_kernel[grid](edges, input, output, num_edges, feature_size, group_size)
    


@triton.jit
def backup_serial_segment_reduction_kernel(
        index, 
        in_feature, 
        result, 
        num_edges: tl.constexpr, 
        feature_size: tl.constexpr, 
        group_size: tl.constexpr
):
    group_id = tl.program_id(axis=0)
    node_offset = group_id * group_size
    f_index = tl.arange(0, feature_size)
    xn = node_offset
    mask = xn < num_edges
    # Load input data
    node_idx = tl.load(index + xn, mask=mask)
    curr_node = node_idx
    val = tl.load(in_feature + node_idx * feature_size + f_index, mask=mask)
    
    for ii in range(1, group_size):  # Iterate over the group
        xn = ii + node_offset  # Get the node index
        mask = xn < num_edges  # Check if the node index is valid
        node_idx = tl.load(index + xn, mask=mask)  # Load the input node
        new_val = tl.load(in_feature + xn * feature_size + f_index, mask=mask)
        if node_idx != curr_node:
            # Perform atomic addition
            tl.atomic_add(result + curr_node * feature_size +
                          f_index, val, mask= (xn-1) < num_edges)
            # Reset val for the new row
            val = new_val
            curr_node = node_idx
        else:
            # Accumulate val
            val += new_val

    tl.atomic_add(result + node_idx * feature_size + f_index, val, mask=mask)


@triton.jit
def backup_spmm_sorted_coo_naive(edge_index, B, C, num_edges, feature_size: tl.constexpr, group_size: tl.constexpr):
    group_id = tl.program_id(0)
    node_offset = group_id * group_size
    f_index = tl.arange(0, feature_size)

    xn = node_offset
    mask = xn < num_edges
    in_node = tl.load(edge_index + xn, mask=mask)  # Load the input node
    out_node = tl.load(edge_index + xn + num_edges, mask=mask)  # Load the output node
    curr_node = out_node
    val = tl.load(B + in_node * feature_size + f_index, mask=mask)
    for ii in range(1, group_size):  # Iterate over the group
        xn = ii + node_offset  # Get the node index
        mask = xn < num_edges  # Check if the node index is valid
        in_node = tl.load(edge_index + xn, mask=mask)  # Load the input node
        out_node = tl.load(edge_index + xn + num_edges, mask=mask)
        new_val = tl.load(B + in_node * feature_size + f_index, mask=mask)
        if out_node != curr_node:
            # Perform atomic addition
            tl.atomic_add(C + curr_node * feature_size +
                          f_index, val)
            # Reset val for the new row
            val = new_val
            curr_node = out_node
        else:
            # Accumulate val
            val += new_val
    tl.atomic_add(C + out_node * feature_size + f_index, val, mask=mask)
 
    
    
if __name__ == "__main__":
    mode = 'reduction'
    # Create input data
    num_edges = 50
    feature_size = 16
    group_size = 32
    num_nodes = 10
    if mode == 'reduction':
        fn = launch_serial_reduction
        edge_raw = torch.randint(0, num_nodes, (num_edges,), dtype=torch.int64, device='cuda')
        edges = edge_raw.sort()[0]
    elif mode == 'spmm':
        fn = launch_serial_spmm
        edge_raw = torch.randint(0, num_nodes, (2, num_edges), dtype=torch.int64, device='cuda')
        _ , indices = torch.sort(edge_raw[1, :])
        edges = edge_raw[ : , indices]

    input = torch.randn(num_edges, feature_size, dtype=torch.float32, device='cuda')
    output = torch.zeros(num_nodes, feature_size, dtype=torch.float32, device='cuda')

    out_values_torch = torch.zeros_like(output)
    out_values_torch.scatter_add_(0, edges.unsqueeze(1).expand_as(input), input)
    fn(edges, input, output, num_edges, feature_size, group_size)

    assert torch.allclose(output, out_values_torch)
    