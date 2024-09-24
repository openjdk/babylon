import torch
import intel_extension_for_pytorch  # type: ignore # noqa: F401

import triton
import triton.language as tl
import shutil
import os

# declare the dummy functions
@triton.jit
def add_kernel():
   a = 0

@triton.jit
def matmul_kernel():
   a = 0

@triton.jit
def softmax_kernel():
   a = 0

ADD_KERNEL_MLIR = "/home/intel/babylon/cr-examples/triton/result/add_kernel.mlir"
MATMUL_MLIR = "/home/intel/babylon/cr-examples/triton/result/matmul_kernel.mlir"
SOFTMAX_MLIR = "/home/intel/babylon/cr-examples/triton/result/softmax_kernel.mlir"

if os.path.isdir('/home/intel/.triton/cache'):
   shutil.rmtree('/home/intel/.triton/cache')

triton.compile(triton.compiler.ASTSource(fn=add_kernel, signature={}, constants={}), target_mlir=ADD_KERNEL_MLIR)
triton.compile(triton.compiler.ASTSource(fn=softmax_kernel, signature={}, constants={}), target_mlir=SOFTMAX_MLIR, options={"num_warps":32})
triton.compile(triton.compiler.ASTSource(fn=matmul_kernel, signature={}, constants={}), target_mlir=MATMUL_MLIR, options={"threads_per_warp":16, "num_warps":64})