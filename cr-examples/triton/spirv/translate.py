import torch
import intel_extension_for_pytorch  # type: ignore # noqa: F401

import triton
import triton.language as tl
import shutil
import os

HOME = os.environ['HOME']
BABYLON_PATH = os.path.join(HOME, 'babylon')

@triton.jit
def add_kernel():
   pass

@triton.jit
def matmul_kernel():
   pass

@triton.jit
def softmax_kernel():
   pass

ADD_KERNEL_MLIR = f"{BABYLON_PATH}/cr-examples/triton/target/mlir/add_kernel.mlir"
MATMUL_MLIR = f"{BABYLON_PATH}/cr-examples/triton/target/mlir/matmul_kernel.mlir"
SOFTMAX_MLIR = f"{BABYLON_PATH}/cr-examples/triton/target/mlir/softmax_kernel.mlir"

if os.path.isdir(f'{HOME}/.triton/cache'):
   shutil.rmtree(f'{HOME}/.triton/cache')

triton.compile(triton.compiler.ASTSource(fn=add_kernel, signature={}, constants={}), target_mlir=ADD_KERNEL_MLIR)
triton.compile(triton.compiler.ASTSource(fn=softmax_kernel, signature={}, constants={}), target_mlir=SOFTMAX_MLIR, options={"num_warps":32})
triton.compile(triton.compiler.ASTSource(fn=matmul_kernel, signature={}, constants={}), target_mlir=MATMUL_MLIR, options={"threads_per_warp":16, "num_warps":64})