import importlib
import os.path as osp
import torch
from .index_scatter import index_scatter
from .gather_scatter import gather_scatter
from .gather_weight_scatter import gather_weight_scatter
from .triton import launch_parallel_spmm, launch_parallel_reduction, launch_serial_spmm, launch_serial_reduction
__version__ = '0.0.1'

library = '_C'
spec = importlib.machinery.PathFinder().find_spec(
    library, [osp.dirname(__file__)])
if spec is not None:
    torch.ops.load_library(spec.origin)
else:
    raise ImportError(f"Could not find module '{library}' in "
                      f'{osp.dirname(__file__)}')

__all__ = [index_scatter, gather_scatter, gather_weight_scatter]
