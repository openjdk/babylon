/*
 * Copyright (c) 2024, Oracle and/or its affiliates. All rights reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * This code is free software; you can redistribute it and/or modify it
 * under the terms of the GNU General Public License version 2 only, as
 * published by the Free Software Foundation.
 *
 * This code is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
 * FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
 * version 2 for more details (a copy is included in the LICENSE file that
 * accompanied this code).
 *
 * You should have received a copy of the GNU General Public License version
 * 2 along with this work; if not, write to the Free Software Foundation,
 * Inc., 51 Franklin St, Fifth Floor, Boston, MA 02110-1301 USA.
 *
 * Please contact Oracle, 500 Oracle Parkway, Redwood Shores, CA 94065 USA
 * or visit www.oracle.com if you need additional information or have any
 * questions.
 */

package oracle.code.triton;

import oracle.code.triton.TritonTestExtension.Kernel;
import oracle.code.triton.TritonTestExtension.TritonTestData;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;

import java.lang.reflect.code.CodeType;
import java.lang.reflect.code.type.JavaType;
import java.lang.runtime.CodeReflection;
import java.util.List;

@ExtendWith(TritonTestExtension.class)
public class TestSoftMax {

    @TritonCodeModel("""
            module ()void -> {
                tt.func @"max_float_float_float" (%0 : float, %1 : float)float -> {
                    %2 : float = arith.maximumf %0 %1;
                    tt.return %2;
                };
                tt.func @"reduce_max_float_float_float_0" (%3 : tensor<x64, float>)float -> {
                    %4 : float = tt.reduce %3 @axis="0" (%5 : float, %6 : float)float -> {
                        %7 : float = tt.call %5 %6 @"max_float_float_float";
                        tt.reduce.return %7;
                    };
                    tt.return %4;
                };
                tt.func @"sum_float_float_float" (%8 : float, %9 : float)float -> {
                    %10 : float = arith.addf %8 %9;
                    tt.return %10;
                };
                tt.func @"reduce_sum_float_float_float_0" (%11 : tensor<x64, float>)float -> {
                    %12 : float = tt.reduce %11 @axis="0" (%13 : float, %14 : float)float -> {
                        %15 : float = tt.call %13 %14 @"sum_float_float_float";
                        tt.reduce.return %15;
                    };
                    tt.return %12;
                };
                tt.func @"softmax_kernel_ptr<float>_ptr<float>_1_1_10_64_void" (%16 : ptr<float>, %17 : ptr<float>)void -> {
                    %18 : int = arith.constant @"1";
                    %19 : int = arith.constant @"1";
                    %20 : int = arith.constant @"10";
                    %21 : int = tt.get_program_id @"0";
                    %22 : int = arith.muli %21 %18;
                    %23 : ptr<float> = tt.addptr %17 %22;
                    %24 : tensor<x64, int> = tt.make_range @start="0" @end="64";
                    %25 : tensor<x64, ptr<float>> = tt.splat %23;
                    %26 : tensor<x64, ptr<float>> = tt.addptr %25 %24;
                    %27 : tensor<x64, int> = tt.splat %20;
                    %28 : tensor<x64, int> = arith.cmpi %24 %27 @"slt";
                    %29 : tensor<x64, float> = tt.load %26 %28;
                    %30 : float = tt.call %29 @"reduce_max_float_float_float_0";
                    %31 : tensor<x64, float> = tt.splat %30;
                    %32 : tensor<x64, float> = arith.subf %29 %31;
                    %33 : tensor<x64, float> = math.exp %32;
                    %34 : float = tt.call %33 @"reduce_sum_float_float_float_0";
                    %35 : tensor<x64, float> = tt.splat %34;
                    %36 : tensor<x64, float> = arith.divf %33 %35;
                    %37 : int = arith.muli %21 %19;
                    %38 : ptr<float> = tt.addptr %16 %37;
                    %39 : tensor<x64, ptr<float>> = tt.splat %38;
                    %40 : tensor<x64, ptr<float>> = tt.addptr %39 %24;
                    tt.store %40 %36 %28;
                    tt.return;
                };
                unreachable;
            };
            """)
    @CodeReflection
    static void softmax_kernel(Ptr output_ptr,
                               Ptr input_ptr,
                               int input_row_stride,
                               int output_row_stride,
                               int n_cols,
                               @Constant int BLOCK_SIZE) {
        // The rows of the softmax are independent, so we parallelize across those
        var row_idx = Triton.programId(0);
        var row_start_ptr = Triton.add(input_ptr, row_idx * input_row_stride);
        // The block size is the next power of two greater than n_cols, so we can fit each
        // row in a single block
        var col_offsets = Triton.arange(0, BLOCK_SIZE);
        var input_ptrs = Triton.add(Triton.broadcast(row_start_ptr, col_offsets.type()), col_offsets);
        // Load the row into SRAM, using a mask since BLOCK_SIZE may be > than n_cols
        var mask = Triton.compare(col_offsets,
                Triton.broadcast(n_cols, col_offsets.type()),
                Triton.CompareKind.LessThan);
        var row = Triton.load(input_ptrs, mask);
        // Subtract maximum for numerical stability
        var row_minus_max = Triton.sub(row, Triton.broadcast(Triton.max(row, 0), row.type()));
        // Note that exponentiation in Triton is fast but approximate (i.e., think __expf in CUDA)
        var numerator = Triton.exp(row_minus_max);
        var denominator = Triton.sum(numerator, 0);
        var softmax_output = Triton.div(numerator, Triton.broadcast(denominator, numerator.type()));
        // Write back output to DRAM
        var output_row_start_ptr = Triton.add(output_ptr, row_idx * output_row_stride);
        var output_ptrs = Triton.add(Triton.broadcast(output_row_start_ptr, col_offsets.type()), col_offsets);
        Triton.store(output_ptrs, softmax_output, mask);
    }

    @Kernel("softmax_kernel")
    @Test
    public void test(TritonTestData t) {
        List<CodeType> argTypes = List.of(
                new PtrType(JavaType.FLOAT),
                new PtrType(JavaType.FLOAT),
                new ConstantType(JavaType.INT, 1),
                new ConstantType(JavaType.INT, 1),
                new ConstantType(JavaType.INT, 10),
                new ConstantType(JavaType.INT, 64));

        t.test(argTypes);
    }

    @TritonCodeModel("""
            module ()void -> {
                tt.func @"max_float_float_float" (%0 : float, %1 : float)float -> {
                    %2 : float = arith.maximumf %0 %1;
                    tt.return %2;
                };
                tt.func @"reduce_max_float_float_float_0" (%3 : tensor<x64, float>)float -> {
                    %4 : float = tt.reduce %3 @axis="0" (%5 : float, %6 : float)float -> {
                        %7 : float = tt.call %5 %6 @"max_float_float_float";
                        tt.reduce.return %7;
                    };
                    tt.return %4;
                };
                tt.func @"sum_float_float_float" (%8 : float, %9 : float)float -> {
                    %10 : float = arith.addf %8 %9;
                    tt.return %10;
                };
                tt.func @"reduce_sum_float_float_float_0" (%11 : tensor<x64, float>)float -> {
                    %12 : float = tt.reduce %11 @axis="0" (%13 : float, %14 : float)float -> {
                        %15 : float = tt.call %13 %14 @"sum_float_float_float";
                        tt.reduce.return %15;
                    };
                    tt.return %12;
                };
                tt.func @"softmax_kernel2_ptr<float>_ptr<float>_1_1_10_64_void" (%16 : ptr<float>, %17 : ptr<float>)void -> {
                    %18 : int = arith.constant @"1";
                    %19 : int = arith.constant @"1";
                    %20 : int = arith.constant @"10";
                    %21 : int = tt.get_program_id @"0";
                    %22 : int = arith.muli %21 %18;
                    %23 : ptr<float> = tt.addptr %17 %22;
                    %24 : tensor<x64, int> = tt.make_range @start="0" @end="64";
                    %25 : tensor<x64, ptr<float>> = tt.splat %23;
                    %26 : tensor<x64, ptr<float>> = tt.addptr %25 %24;
                    %27 : tensor<x64, int> = tt.splat %20;
                    %28 : tensor<x64, int> = arith.cmpi %24 %27 @"slt";
                    %29 : tensor<x64, float> = tt.load %26 %28;
                    %30 : float = tt.call %29 @"reduce_max_float_float_float_0";
                    %31 : tensor<x64, float> = tt.splat %30;
                    %32 : tensor<x64, float> = arith.subf %29 %31;
                    %33 : tensor<x64, float> = math.exp %32;
                    %34 : float = tt.call %33 @"reduce_sum_float_float_float_0";
                    %35 : tensor<x64, float> = tt.splat %34;
                    %36 : tensor<x64, float> = arith.divf %33 %35;
                    %37 : int = arith.muli %21 %19;
                    %38 : ptr<float> = tt.addptr %16 %37;
                    %39 : tensor<x64, ptr<float>> = tt.splat %38;
                    %40 : tensor<x64, ptr<float>> = tt.addptr %39 %24;
                    tt.store %40 %36 %28;
                    tt.return;
                };
                unreachable;
            };
            """)
    @CodeReflection
    static void softmax_kernel2(Ptr output_ptr,
                                Ptr input_ptr,
                                int input_row_stride,
                                int output_row_stride,
                                int n_cols,
                                @Constant int BLOCK_SIZE) {
        // The rows of the softmax are independent, so we parallelize across those
        var row_idx = Triton.programId(0);
        var row_start_ptr = Triton.add(input_ptr, row_idx * input_row_stride);
        // The block size is the next power of two greater than n_cols, so we can fit each
        // row in a single block
        var col_offsets = Triton.arange(0, BLOCK_SIZE);
        var input_ptrs = Triton.add(row_start_ptr, col_offsets);
        // Load the row into SRAM, using a mask since BLOCK_SIZE may be > than n_cols
        var mask = Triton.compare(col_offsets, n_cols, Triton.CompareKind.LessThan);
        var row = Triton.load(input_ptrs, mask);
        // Subtract maximum for numerical stability
        var row_minus_max = Triton.sub(row, Triton.max(row, 0));
        // Note that exponentiation in Triton is fast but approximate (i.e., think __expf in CUDA)
        var numerator = Triton.exp(row_minus_max);
        var denominator = Triton.sum(numerator, 0);
        var softmax_output = Triton.div(numerator, denominator);
        // Write back output to DRAM
        var output_row_start_ptr = Triton.add(output_ptr, row_idx * output_row_stride);
        var output_ptrs = Triton.add(output_row_start_ptr, col_offsets);
        Triton.store(output_ptrs, softmax_output, mask);
    }

    @Kernel("softmax_kernel2")
    @Test
    public void test2(TritonTestData t) {
        List<CodeType> argTypes = List.of(
                new PtrType(JavaType.FLOAT),
                new PtrType(JavaType.FLOAT),
                new ConstantType(JavaType.INT, 1),
                new ConstantType(JavaType.INT, 1),
                new ConstantType(JavaType.INT, 10),
                new ConstantType(JavaType.INT, 64));

        t.test(argTypes);
    }
}

/*
def softmax_kernel(output_ptr, input_ptr, input_row_stride, output_row_stride, n_cols, BLOCK_SIZE: tl.constexpr):
    # The rows of the softmax are independent, so we parallelize across those
    row_idx = tl.program_id(0)
    # The stride represents how much we need to increase the pointer to advance 1 row
    row_start_ptr = input_ptr + row_idx * input_row_stride
    # The block size is the next power of two greater than n_cols, so we can fit each
    # row in a single block
    col_offsets = tl.arange(0, BLOCK_SIZE)
    input_ptrs = row_start_ptr + col_offsets
    # Load the row into SRAM, using a mask since BLOCK_SIZE may be > than n_cols
    row = tl.load(input_ptrs, mask=col_offsets < n_cols, other=-float('inf'))
    # Subtract maximum for numerical stability
    row_minus_max = row - tl.max(row, axis=0)
    # Note that exponentiation in Triton is fast but approximate (i.e., think __expf in CUDA)
    numerator = tl.exp(row_minus_max)
    denominator = tl.sum(numerator, axis=0)
    softmax_output = numerator / denominator
    # Write back output to DRAM
    output_row_start_ptr = output_ptr + row_idx * output_row_stride
    output_ptrs = output_row_start_ptr + col_offsets
    tl.store(output_ptrs, softmax_output, mask=col_offsets < n_cols)
*/

/*
input_row_stride = 1
output_row_stride = 1
n_cols=10
BLOCK_SIZE=64

module {
  tt.func public @softmax_kernel_01(%arg0: !tt.ptr<f32, 1> , %arg1: !tt.ptr<f32, 1> ) attributes {noinline = false} {
    %0 = tt.get_program_id x : i32
    %c1_i32 = arith.constant 1 : i32
    %1 = arith.muli %0, %c1_i32 : i32
    %2 = tt.addptr %arg1, %1 : !tt.ptr<f32, 1>, i32
    %3 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>
    %4 = tt.splat %2 : (!tt.ptr<f32, 1>) -> tensor<64x!tt.ptr<f32, 1>>
    %5 = tt.addptr %4, %3 : tensor<64x!tt.ptr<f32, 1>>, tensor<64xi32>
    %c10_i32 = arith.constant 10 : i32
    %cst = arith.constant dense<10> : tensor<64xi32>
    %6 = arith.cmpi slt, %3, %cst : tensor<64xi32>
    %cst_0 = arith.constant 0xFF800000 : f32
    %cst_1 = arith.constant dense<0xFF800000> : tensor<64xf32>
    %7 = tt.load %5, %6, %cst_1 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<64xf32>
    %8 = tt.call @max__fp32S64S__1cconstexpr_0__2cconstexpr_False__3cconstexpr_True_(%7) : (tensor<64xf32>) -> f32
    %9 = tt.splat %8 : (f32) -> tensor<64xf32>
    %10 = arith.subf %7, %9 : tensor<64xf32>
    %11 = math.exp %10 : tensor<64xf32>
    %12 = tt.call @sum__fp32S64S__1cconstexpr_0_(%11) : (tensor<64xf32>) -> f32
    %13 = tt.splat %12 : (f32) -> tensor<64xf32>
    %14 = arith.divf %11, %13 : tensor<64xf32>
    %c1_i32_2 = arith.constant 1 : i32
    %15 = arith.muli %0, %c1_i32_2 : i32
    %16 = tt.addptr %arg0, %15 : !tt.ptr<f32, 1>, i32
    %17 = tt.splat %16 : (!tt.ptr<f32, 1>) -> tensor<64x!tt.ptr<f32, 1>>
    %18 = tt.addptr %17, %3 : tensor<64x!tt.ptr<f32, 1>>, tensor<64xi32>
    %c10_i32_3 = arith.constant 10 : i32
    %cst_4 = arith.constant dense<10> : tensor<64xi32>
    %19 = arith.cmpi slt, %3, %cst_4 : tensor<64xi32>
    tt.store %18, %14, %19 {cache = 1 : i32, evict = 1 : i32} : tensor<64xf32>
    tt.return
  }
  tt.func private @max__fp32S64S__1cconstexpr_0__2cconstexpr_False__3cconstexpr_True_(%arg0: tensor<64xf32> ) -> f32 attributes {noinline = false} {
    %0 = "tt.reduce"(%arg0) <{axis = 0 : i32}> ({
    ^bb0(%arg1: f32 , %arg2: f32 ):
      %1 = tt.call @maximum__fp32_fp32__(%arg1, %arg2) : (f32, f32) -> f32
      tt.reduce.return %1 : f32
    }) : (tensor<64xf32>) -> f32
    tt.return %0 : f32
  }
  tt.func private @maximum__fp32_fp32__(%arg0: f32 , %arg1: f32 ) -> f32 attributes {noinline = false} {
    %0 = arith.maximumf %arg0, %arg1 : f32
    tt.return %0 : f32
  }
  tt.func private @sum__fp32S64S__1cconstexpr_0_(%arg0: tensor<64xf32> ) -> f32 attributes {noinline = false} {
    %0 = "tt.reduce"(%arg0) <{axis = 0 : i32}> ({
    ^bb0(%arg1: f32 , %arg2: f32 ):
      %1 = tt.call @_sum_combine__fp32_fp32__(%arg1, %arg2) : (f32, f32) -> f32
      tt.reduce.return %1 : f32
    }) : (tensor<64xf32>) -> f32
    tt.return %0 : f32
  }
  tt.func private @_sum_combine__fp32_fp32__(%arg0: f32 , %arg1: f32 ) -> f32 attributes {noinline = false} {
    %0 = arith.addf %arg0, %arg1 : f32
    tt.return %0 : f32
  }
}
*/
