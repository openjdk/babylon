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

import oracle.code.triton.TritonTestExtension.TritonTestData;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;

import java.lang.reflect.Type;
import java.lang.runtime.CodeReflection;
import java.util.List;

@ExtendWith(TritonTestExtension.class)
public class TestAddKernel {

    @TritonCodeModel("""
            module ()void -> {
                tt.func @"add_kernel_ptr<float>_ptr<float>_ptr<float>_int_64_void" (%0 : ptr<float>, %1 : ptr<float>, %2 : ptr<float>, %3 : int)void -> {
                    %4 : int = arith.constant @"64";
                    %5 : int = tt.get_program_id @"0";
                    %6 : int = arith.muli %5 %4;
                    %7 : tensor<x64, int> = tt.make_range @start="0" @end="64";
                    %8 : tensor<x64, int> = tt.splat %6;
                    %9 : tensor<x64, int> = arith.addi %8 %7;
                    %10 : tensor<x64, int> = tt.splat %3;
                    %11 : tensor<x64, int> = arith.cmpi %9 %10 @"slt";
                    %12 : tensor<x64, ptr<float>> = tt.splat %0;
                    %13 : tensor<x64, ptr<float>> = tt.addptr %12 %9;
                    %14 : tensor<x64, float> = tt.load %13 %11;
                    %15 : tensor<x64, ptr<float>> = tt.splat %1;
                    %16 : tensor<x64, ptr<float>> = tt.addptr %15 %9;
                    %17 : tensor<x64, float> = tt.load %16 %11;
                    %18 : tensor<x64, float> = arith.addf %14 %17;
                    %19 : tensor<x64, ptr<float>> = tt.splat %2;
                    %20 : tensor<x64, ptr<float>> = tt.addptr %19 %9;
                    tt.store %20 %18 %11;
                    tt.return;
                };
                unreachable;
            };
            """)
    @CodeReflection
    static void add_kernel(Ptr x_ptr,  // *Pointer* to first input vector.
                           Ptr y_ptr,  // *Pointer* to second input vector.
                           Ptr output_ptr,  // *Pointer* to output vector.
                           int n_elements,  // Size of the vector.
                           @Constant int BLOCK_SIZE)  // Number of elements each program should process.
    // NOTE: @Constant so it can be used as a shape value
    {
        // There are multiple 'programs' processing different data. We identify which program
        // we are here:
        var pid = Triton.programId(0); // We use a 1D launch grid so axis is 0.
        // This program will process inputs that are offset from the initial data.
        // For instance, if you had a vector of length 256 and block_size of 64, the programs
        // would each access the elements [0:64, 64:128, 128:192, 192:256].
        // Note that offsets is a list of pointers:
        var block_start = pid * BLOCK_SIZE;
        var range = Triton.arange(0, BLOCK_SIZE);
        var offsets = Triton.add(Triton.broadcast(block_start, range.type()), range);
        // Create a mask to guard memory operations against out-of-bounds accesses.
        var mask = Triton.compare(offsets, Triton.broadcast(n_elements, offsets.type()), Triton.CompareKind.LessThan);
        // Load x and y from DRAM, masking out any extra elements in case the input is not a
        // multiple of the block size.
        var x = Triton.load(Triton.add(Triton.broadcast(x_ptr, offsets.type()), offsets), mask);
        var y = Triton.load(Triton.add(Triton.broadcast(y_ptr, offsets.type()), offsets), mask);
        var output = Triton.add(x, y);
        // Write x + y back to DRAM.
        Triton.store(Triton.add(Triton.broadcast(output_ptr, offsets.type()), offsets), output, mask);
    }

    @TritonTestExtension.Kernel("add_kernel")
    @Test
    public void test(TritonTestData t) {
        List<Type> argTypes = List.of(
                new PtrType(float.class),
                new PtrType(float.class),
                new PtrType(float.class),
                int.class,
                new ConstantType(int.class, 64));

        t.test(argTypes);
    }


    @TritonCodeModel("""
            module ()void -> {
                tt.func @"add_kernel2_ptr<float>_ptr<float>_ptr<float>_int_64_void" (%0 : ptr<float>, %1 : ptr<float>, %2 : ptr<float>, %3 : int)void -> {
                    %4 : int = arith.constant @"64";
                    %5 : int = tt.get_program_id @"0";
                    %6 : int = arith.muli %5 %4;
                    %7 : tensor<x64, int> = tt.make_range @start="0" @end="64";
                    %8 : tensor<x64, int> = tt.splat %6;
                    %9 : tensor<x64, int> = arith.addi %8 %7;
                    %10 : tensor<x64, int> = tt.splat %3;
                    %11 : tensor<x64, int> = arith.cmpi %9 %10 @"slt";
                    %12 : tensor<x64, ptr<float>> = tt.splat %0;
                    %13 : tensor<x64, ptr<float>> = tt.addptr %12 %9;
                    %14 : tensor<x64, float> = tt.load %13 %11;
                    %15 : tensor<x64, ptr<float>> = tt.splat %1;
                    %16 : tensor<x64, ptr<float>> = tt.addptr %15 %9;
                    %17 : tensor<x64, float> = tt.load %16 %11;
                    %18 : tensor<x64, float> = arith.addf %14 %17;
                    %19 : tensor<x64, ptr<float>> = tt.splat %2;
                    %20 : tensor<x64, ptr<float>> = tt.addptr %19 %9;
                    tt.store %20 %18 %11;
                    tt.return;
                };
                unreachable;
            };
            """)
    @CodeReflection
    static void add_kernel2(Ptr x_ptr,  // *Pointer* to first input vector.
                            Ptr y_ptr,  // *Pointer* to second input vector.
                            Ptr output_ptr,  // *Pointer* to output vector.
                            int n_elements,  // Size of the vector.
                            @Constant int BLOCK_SIZE)  // Number of elements each program should process.
    // NOTE: @Constant so it can be used as a shape value
    {
        // There are multiple 'programs' processing different data. We identify which program
        // we are here:
        var pid = Triton.programId(0); // We use a 1D launch grid so axis is 0.
        // This program will process inputs that are offset from the initial data.
        // For instance, if you had a vector of length 256 and block_size of 64, the programs
        // would each access the elements [0:64, 64:128, 128:192, 192:256].
        // Note that offsets is a list of pointers:
        var block_start = pid * BLOCK_SIZE;
        var range = Triton.arange(0, BLOCK_SIZE);
        var offsets = Triton.add(block_start, range);
        // Create a mask to guard memory operations against out-of-bounds accesses.
        var mask = Triton.compare(offsets, n_elements, Triton.CompareKind.LessThan);
        // Load x and y from DRAM, masking out any extra elements in case the input is not a
        // multiple of the block size.
        var x = Triton.load(Triton.add(x_ptr, offsets), mask);
        var y = Triton.load(Triton.add(y_ptr, offsets), mask);
        var output = Triton.add(x, y);
        // Write x + y back to DRAM.
        Triton.store(Triton.add(output_ptr, offsets), output, mask);
    }

    @TritonTestExtension.Kernel("add_kernel2")
    @Test
    public void test2(TritonTestData t) {
        List<Type> argTypes = List.of(
                new PtrType(float.class),
                new PtrType(float.class),
                new PtrType(float.class),
                int.class,
                new ConstantType(int.class, 64));

        t.test(argTypes);
    }
}

/*
@triton.jit
def add_kernel(x_ptr,  # *Pointer* to first input vector.
               y_ptr,  # *Pointer* to second input vector.
               output_ptr,  # *Pointer* to output vector.
               n_elements,  # Size of the vector.
               BLOCK_SIZE: tl.constexpr,  # Number of elements each program should process.
               # NOTE: `constexpr` so it can be used as a shape value.
               ):
    # There are multiple 'programs' processing different data. We identify which program
    # we are here:
    pid = tl.program_id(axis=0)  # We use a 1D launch grid so axis is 0.
    # This program will process inputs that are offset from the initial data.
    # For instance, if you had a vector of length 256 and block_size of 64, the programs
    # would each access the elements [0:64, 64:128, 128:192, 192:256].
    # Note that offsets is a list of pointers:
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    # Create a mask to guard memory operations against out-of-bounds accesses.
    mask = offsets < n_elements
    # Load x and y from DRAM, masking out any extra elements in case the input is not a
    # multiple of the block size.
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    # Write x + y back to DRAM.
    tl.store(output_ptr + offsets, output, mask=mask)
*/

/*
module {
  tt.func public @add_kernel_0123(%arg0: !tt.ptr<f32, 1> , %arg1: !tt.ptr<f32, 1> , %arg2: !tt.ptr<f32, 1> , %arg3: i32 ) attributes {noinline = false} {
    %0 = tt.get_program_id x : i32
    %c64_i32 = arith.constant 64 : i32
    %1 = arith.muli %0, %c64_i32 : i32
    %2 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>
    %3 = tt.splat %1 : (i32) -> tensor<64xi32>
    %4 = arith.addi %3, %2 : tensor<64xi32>
    %5 = tt.splat %arg3 : (i32) -> tensor<64xi32>
    %6 = arith.cmpi slt, %4, %5 : tensor<64xi32>
    %7 = tt.splat %arg0 : (!tt.ptr<f32, 1>) -> tensor<64x!tt.ptr<f32, 1>>
    %8 = tt.addptr %7, %4 : tensor<64x!tt.ptr<f32, 1>>, tensor<64xi32>
    %9 = tt.load %8, %6 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<64xf32>
    %10 = tt.splat %arg1 : (!tt.ptr<f32, 1>) -> tensor<64x!tt.ptr<f32, 1>>
    %11 = tt.addptr %10, %4 : tensor<64x!tt.ptr<f32, 1>>, tensor<64xi32>
    %12 = tt.load %11, %6 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<64xf32>
    %13 = arith.addf %9, %12 : tensor<64xf32>
    %14 = tt.splat %arg2 : (!tt.ptr<f32, 1>) -> tensor<64x!tt.ptr<f32, 1>>
    %15 = tt.addptr %14, %4 : tensor<64x!tt.ptr<f32, 1>>, tensor<64xi32>
    tt.store %15, %13, %6 {cache = 1 : i32, evict = 1 : i32} : tensor<64xf32>
    tt.return
  }
}
*/
