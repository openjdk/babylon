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

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;

import java.lang.reflect.code.TypeElement;
import java.lang.reflect.code.type.JavaType;
import java.lang.runtime.CodeReflection;
import java.util.List;

import static oracle.code.triton.Triton.*;
import static oracle.code.triton.Triton.CompareKind.*;
import static oracle.code.triton.Triton.compare;
import static oracle.code.triton.Triton.load;

@ExtendWith(TritonTestExtension.class)
public class TestMatrix {

    @TritonCodeModel("""
            module ()void -> {
                tt.func @"cdiv_int_32_int" (%0 : int)int -> {
                    %1 : int = arith.constant @"32";
                    %2 : int = arith.addi %0 %1;
                    %3 : int = arith.constant @"1";
                    %4 : int = arith.subi %2 %3;
                    %5 : int = arith.divsi %4 %1;
                    tt.return %5;
                };
                tt.func @"cdiv_int_64_int" (%6 : int)int -> {
                    %7 : int = arith.constant @"64";
                    %8 : int = arith.addi %6 %7;
                    %9 : int = arith.constant @"1";
                    %10 : int = arith.subi %8 %9;
                    %11 : int = arith.divsi %10 %7;
                    tt.return %11;
                };
                tt.func @"matmul_kernel_broadcast_ptr<float>_ptr<float>_ptr<float>_int_int_int_int_int_int_int_int_int_32_64_32_8_false_void" (%12 : ptr<float>, %13 : ptr<float>, %14 : ptr<float>, %15 : int, %16 : int, %17 : int, %18 : int, %19 : int, %20 : int, %21 : int, %22 : int, %23 : int)void -> {
                    %24 : int = arith.constant @"32";
                    %25 : int = arith.constant @"64";
                    %26 : int = arith.constant @"32";
                    %27 : int = arith.constant @"8";
                    %28 : int = tt.get_program_id @"0";
                    %29 : int = tt.call %15 @"cdiv_int_32_int";
                    %30 : int = tt.call %16 @"cdiv_int_64_int";
                    %31 : int = arith.muli %27 %30;
                    %32 : int = arith.divsi %28 %31;
                    %33 : int = arith.muli %32 %27;
                    %34 : int = arith.subi %29 %33;
                    %35 : int = arith.minsi %34 %27;
                    %36 : int = arith.remsi %28 %35;
                    %37 : int = arith.addi %33 %36;
                    %38 : int = arith.remsi %28 %31;
                    %39 : int = arith.divsi %38 %35;
                    %40 : tensor<x32, int> = tt.make_range @start="0" @end="32";
                    %41 : int = arith.muli %37 %24;
                    %42 : tensor<x32, int> = tt.splat %41;
                    %43 : tensor<x32, int> = arith.addi %42 %40;
                    %44 : tensor<x32, int> = tt.splat %15;
                    %45 : tensor<x32, int> = arith.remsi %43 %44;
                    %46 : tensor<x64, int> = tt.make_range @start="0" @end="64";
                    %47 : int = arith.muli %39 %25;
                    %48 : tensor<x64, int> = tt.splat %47;
                    %49 : tensor<x64, int> = arith.addi %48 %46;
                    %50 : tensor<x64, int> = tt.splat %16;
                    %51 : tensor<x64, int> = arith.remsi %49 %50;
                    %52 : tensor<x32, int> = tt.make_range @start="0" @end="32";
                    %53 : tensor<x32, x1, int> = tt.expand_dims %45 @"1";
                    %54 : tensor<x32, x1, int> = tt.splat %18;
                    %55 : tensor<x32, x1, int> = arith.muli %53 %54;
                    %56 : tensor<x1, x32, int> = tt.expand_dims %52 @"0";
                    %57 : tensor<x1, x32, int> = tt.splat %19;
                    %58 : tensor<x1, x32, int> = arith.muli %56 %57;
                    %59 : tensor<x32, x32, ptr<float>> = tt.splat %12;
                    %60 : tensor<x32, x32, int> = tt.broadcast %55;
                    %61 : tensor<x32, x32, int> = tt.broadcast %58;
                    %62 : tensor<x32, x32, int> = arith.addi %60 %61;
                    %63 : tensor<x32, x32, ptr<float>> = tt.addptr %59 %62;
                    %64 : tensor<x32, x1, int> = tt.expand_dims %52 @"1";
                    %65 : tensor<x32, x1, int> = tt.splat %20;
                    %66 : tensor<x32, x1, int> = arith.muli %64 %65;
                    %67 : tensor<x1, x64, int> = tt.expand_dims %51 @"0";
                    %68 : tensor<x1, x64, int> = tt.splat %21;
                    %69 : tensor<x1, x64, int> = arith.muli %67 %68;
                    %70 : tensor<x32, x64, ptr<float>> = tt.splat %13;
                    %71 : tensor<x32, x64, int> = tt.broadcast %66;
                    %72 : tensor<x32, x64, int> = tt.broadcast %69;
                    %73 : tensor<x32, x64, int> = arith.addi %71 %72;
                    %74 : tensor<x32, x64, ptr<float>> = tt.addptr %70 %73;
                    %75 : tensor<x32, x64, float> = arith.constant @"0.0";
                    %76 : int = arith.constant @"0";
                    %77 : int = tt.call %17 @"cdiv_int_32_int";
                    %78 : int = arith.constant @"1";
                    %79 : Tuple<tensor<x32, x64, float>, tensor<x32, x32, ptr<float>>, tensor<x32, x64, ptr<float>>> = scf.for %76 %77 %78 %75 %63 %74 (%80 : int, %81 : tensor<x32, x64, float>, %82 : tensor<x32, x32, ptr<float>>, %83 : tensor<x32, x64, ptr<float>>)Tuple<tensor<x32, x64, float>, tensor<x32, x32, ptr<float>>, tensor<x32, x64, ptr<float>>> -> {
                        %84 : tensor<x1, x32, int> = tt.expand_dims %52 @"0";
                        %85 : int = arith.muli %80 %26;
                        %86 : int = arith.subi %17 %85;
                        %87 : tensor<x1, x32, int> = tt.splat %86;
                        %88 : tensor<x1, x32, int> = arith.cmpi %84 %87 @"slt";
                        %89 : tensor<x32, x32, int> = tt.broadcast %88;
                        %90 : tensor<x32, x32, float> = tt.load %82 %89;
                        %91 : tensor<x32, x1, int> = tt.expand_dims %52 @"1";
                        %92 : int = arith.muli %80 %26;
                        %93 : int = arith.subi %17 %92;
                        %94 : tensor<x32, x1, int> = tt.splat %93;
                        %95 : tensor<x32, x1, int> = arith.cmpi %91 %94 @"slt";
                        %96 : tensor<x32, x64, int> = tt.broadcast %95;
                        %97 : tensor<x32, x64, float> = tt.load %83 %96;
                        %98 : tensor<x32, x64, float> = tt.dot %90 %97;
                        %99 : tensor<x32, x64, float> = arith.addf %81 %98;
                        %100 : int = arith.muli %26 %19;
                        %101 : tensor<x32, x32, int> = tt.splat %100;
                        %102 : tensor<x32, x32, ptr<float>> = tt.addptr %82 %101;
                        %103 : int = arith.muli %26 %20;
                        %104 : tensor<x32, x64, int> = tt.splat %103;
                        %105 : tensor<x32, x64, ptr<float>> = tt.addptr %83 %104;
                        scf.yield %99 %102 %105;
                    };
                    %106 : tensor<x32, x64, float> = tuple.load %79 @"0";
                    %107 : tensor<x32, x32, ptr<float>> = tuple.load %79 @"1";
                    %108 : tensor<x32, x64, ptr<float>> = tuple.load %79 @"2";
                    %109 : int = arith.muli %37 %24;
                    %110 : tensor<x32, int> = tt.splat %109;
                    %111 : tensor<x32, int> = arith.addi %110 %40;
                    %112 : int = arith.muli %39 %25;
                    %113 : tensor<x64, int> = tt.splat %112;
                    %114 : tensor<x64, int> = arith.addi %113 %46;
                    %115 : tensor<x32, x1, int> = tt.expand_dims %111 @"1";
                    %116 : tensor<x32, x1, int> = tt.splat %22;
                    %117 : tensor<x32, x1, int> = arith.muli %115 %116;
                    %118 : tensor<x1, x64, int> = tt.expand_dims %114 @"0";
                    %119 : tensor<x1, x64, int> = tt.splat %23;
                    %120 : tensor<x1, x64, int> = arith.muli %118 %119;
                    %121 : tensor<x32, x64, ptr<float>> = tt.splat %14;
                    %122 : tensor<x32, x64, int> = tt.broadcast %117;
                    %123 : tensor<x32, x64, int> = tt.broadcast %120;
                    %124 : tensor<x32, x64, int> = arith.addi %122 %123;
                    %125 : tensor<x32, x64, ptr<float>> = tt.addptr %121 %124;
                    %126 : tensor<x32, x1, int> = tt.expand_dims %111 @"1";
                    %127 : tensor<x32, x1, int> = tt.splat %15;
                    %128 : tensor<x32, x1, int> = arith.cmpi %126 %127 @"slt";
                    %129 : tensor<x1, x64, int> = tt.expand_dims %114 @"0";
                    %130 : tensor<x1, x64, int> = tt.splat %16;
                    %131 : tensor<x1, x64, int> = arith.cmpi %129 %130 @"slt";
                    %132 : tensor<x32, x64, int> = tt.broadcast %128;
                    %133 : tensor<x32, x64, int> = tt.broadcast %131;
                    %134 : tensor<x32, x64, int> = arith.andi %132 %133;
                    tt.store %125 %106 %134;
                    tt.return;
                };
                unreachable;
            };
            """)
    @CodeReflection
    static void matmul_kernel_broadcast(
            // Pointers to matrices
            Ptr a_ptr, Ptr b_ptr, Ptr c_ptr,
            // Matrix dimensions
            int M, int N, int K,
            // The stride variables represent how much to increase the ptr by when moving by 1
            // element in a particular dimension. E.g. `stride_am` is how much to increase `a_ptr`
            // by to get the element one row down (A has M rows).
            int stride_am, int stride_ak,
            int stride_bk, int stride_bn,
            int stride_cm, int stride_cn,
            // Meta-parameters
            @Constant int BLOCK_SIZE_M, @Constant int BLOCK_SIZE_N, @Constant int BLOCK_SIZE_K,
            @Constant int GROUP_SIZE_M,
            @Constant boolean ACTIVATION) {

        // """Kernel for computing the matmul C = A x B.
        // A has shape (M, K), B has shape (K, N) and C has shape (M, N)
        // """
        // -----------------------------------------------------------
        // Map program ids `pid` to the block of C it should compute.
        // This is done in a grouped ordering to promote L2 data reuse.
        // See above `L2 Cache Optimizations` section for details.
        var pid = programId(0);

        var num_pid_m = cdiv(M, BLOCK_SIZE_M);
        var num_pid_n = cdiv(N, BLOCK_SIZE_N);
        var num_pid_in_group = GROUP_SIZE_M * num_pid_n;
        var group_id = pid / num_pid_in_group;
        var first_pid_m = group_id * GROUP_SIZE_M;
        var group_size_m = Math.min(num_pid_m - first_pid_m, GROUP_SIZE_M);
        var pid_m = first_pid_m + (pid % group_size_m);
        var pid_n = (pid % num_pid_in_group) / group_size_m;

        // ----------------------------------------------------------
        // Create pointers for the first blocks of A and B.
        // We will advance this pointer as we move in the K direction
        // and accumulate
        // `a_ptrs` is a block of [BLOCK_SIZE_M, BLOCK_SIZE_K] pointers
        // `b_ptrs` is a block of [BLOCK_SIZE_K, BLOCK_SIZE_N] pointers
        // See above `Pointer Arithmetics` section for details
        var offs_m = arange(0, BLOCK_SIZE_M);
        var offs_am = mod(
                add(broadcast(pid_m * BLOCK_SIZE_M, offs_m.type()), offs_m),
                broadcast(M, offs_m.type()));
        var offs_n = arange(0, BLOCK_SIZE_N);
        var offs_bn = mod(
                add(broadcast(pid_n * BLOCK_SIZE_N, offs_n.type()), offs_n),
                broadcast(N, offs_n.type()));
        var offs_k = arange(0, BLOCK_SIZE_K);

        var offs_am_e = expand(offs_am, 1);
        offs_am_e = mul(offs_am_e, broadcast(stride_am, offs_am_e.type()));
        var offs_k_e_0 = expand(offs_k, 0);
        offs_k_e_0 = mul(offs_k_e_0, broadcast(stride_ak, offs_k_e_0.type()));
        TensorType a_ptrs_t = joinShape(offs_am_e.type(), offs_k_e_0.type());
        var a_ptrs = add(broadcast(a_ptr, a_ptrs_t),
                add(broadcast(offs_am_e, a_ptrs_t), broadcast(offs_k_e_0, a_ptrs_t)));

        var offs_k_e_1 = expand(offs_k, 1);
        offs_k_e_1 = mul(offs_k_e_1, broadcast(stride_bk, offs_k_e_1.type()));
        var offs_bn_e = expand(offs_bn, 0);
        offs_bn_e = mul(offs_bn_e, broadcast(stride_bn, offs_bn_e.type()));
        TensorType b_ptrs_t = joinShape(offs_k_e_1.type(), offs_bn_e.type());
        var b_ptrs = add(broadcast(b_ptr, b_ptrs_t),
                add(broadcast(offs_k_e_1, b_ptrs_t), broadcast(offs_bn_e, b_ptrs_t)));

        // -----------------------------------------------------------
        // Iterate to compute a block of the C matrix.
        // We accumulate into a `[BLOCK_SIZE_M, BLOCK_SIZE_N]` block
        // of fp32 values for higher accuracy.
        // `accumulator` will be converted back to fp16 after the loop.
        var accumulator = zeros(float.class, BLOCK_SIZE_M, BLOCK_SIZE_N);
        for (int k = 0; k < cdiv(K, BLOCK_SIZE_K); k++) {
            // Load the next block of A and B, generate a mask by checking the K dimension.
            // If it is out of bounds, set it to 0.
            var offs_k_m_0 = expand(offs_k, 0);
            offs_k_m_0 = compare(offs_k_m_0,
                    broadcast(K - k * BLOCK_SIZE_K, offs_k_m_0.type()),
                    LessThan);
            var a = load(a_ptrs, broadcast(offs_k_m_0, a_ptrs.type()));
            var offs_k_m_1 = expand(offs_k, 1);
            offs_k_m_1 = compare(offs_k_m_1,
                    broadcast(K - k * BLOCK_SIZE_K, offs_k_m_1.type()),
                    LessThan);
            var b = load(b_ptrs, broadcast(offs_k_m_1, b_ptrs.type()));
            // We accumulate along the K dimension.
            accumulator = add(accumulator, dot(a, b));
            // Advance the ptrs to the next K block.
            a_ptrs = add(a_ptrs, broadcast(BLOCK_SIZE_K * stride_ak, a_ptrs.type()));
            b_ptrs = add(b_ptrs, broadcast(BLOCK_SIZE_K * stride_bk, b_ptrs.type()));
        }

        // You can fuse arbitrary activation functions here
        // while the accumulator is still in FP32!
//        if (ACTIVATION) {
//            // ...
//        }
        // c = Triton.to(activation, tl.float16)
        var c = accumulator;

        // -----------------------------------------------------------
        // Write back the block of the output matrix C with masks.
        var offs_cm = add(broadcast(pid_m * BLOCK_SIZE_M, offs_m.type()), offs_m);
        var offs_cn = add(broadcast(pid_n * BLOCK_SIZE_N, offs_n.type()), offs_n);

        var offs_cm_e = expand(offs_cm, 1);
        offs_cm_e = mul(offs_cm_e, broadcast(stride_cm, offs_cm_e.type()));
        var offs_cn_e = expand(offs_cn, 0);
        offs_cn_e = mul(offs_cn_e, broadcast(stride_cn, offs_cn_e.type()));
        TensorType c_ptrs_t = joinShape(offs_cm_e.type(), offs_cn_e.type());
        var c_ptrs = add(broadcast(c_ptr, c_ptrs_t),
                add(broadcast(offs_cm_e, c_ptrs_t), broadcast(offs_cn_e, c_ptrs_t)));

        offs_cm_e = expand(offs_cm, 1);
        var c_mask_l = compare(offs_cm_e, broadcast(M, offs_cm_e.type()), LessThan);
        offs_cn_e = expand(offs_cn, 0);
        var c_mask_r = compare(offs_cn_e, broadcast(N, offs_cn_e.type()), LessThan);
        var c_mask = and(broadcast(c_mask_l, c_ptrs_t), broadcast(c_mask_r, c_ptrs_t));

        store(c_ptrs, c, c_mask);
    }

    @TritonTestExtension.Kernel("matmul_kernel_broadcast")
    @Test
    public void testWithBroadcast(TritonTestExtension.TritonTestData t) {
        List<TypeElement> argTypes = List.of(
                new PtrType(JavaType.FLOAT),
                new PtrType(JavaType.FLOAT),
                new PtrType(JavaType.FLOAT),
                JavaType.INT, JavaType.INT, JavaType.INT,
                JavaType.INT, JavaType.INT,
                JavaType.INT, JavaType.INT,
                JavaType.INT, JavaType.INT,
                new ConstantType(JavaType.INT, 32), new ConstantType(JavaType.INT, 64), new ConstantType(JavaType.INT, 32),
                new ConstantType(JavaType.INT, 8),
                new ConstantType(JavaType.INT, false));

        t.test(argTypes);
    }


    @TritonCodeModel("""
            module ()void -> {
                tt.func @"cdiv_int_32_int" (%0 : int)int -> {
                    %1 : int = arith.constant @"32";
                    %2 : int = arith.addi %0 %1;
                    %3 : int = arith.constant @"1";
                    %4 : int = arith.subi %2 %3;
                    %5 : int = arith.divsi %4 %1;
                    tt.return %5;
                };
                tt.func @"cdiv_int_64_int" (%6 : int)int -> {
                    %7 : int = arith.constant @"64";
                    %8 : int = arith.addi %6 %7;
                    %9 : int = arith.constant @"1";
                    %10 : int = arith.subi %8 %9;
                    %11 : int = arith.divsi %10 %7;
                    tt.return %11;
                };
                tt.func @"matmul_kernel_ptr<oracle.code.triton.Float16>_ptr<oracle.code.triton.Float16>_ptr<oracle.code.triton.Float16>_int_int_int_int_int_int_int_int_int_32_64_32_8_false_void" (%12 : ptr<oracle.code.triton.Float16>, %13 : ptr<oracle.code.triton.Float16>, %14 : ptr<oracle.code.triton.Float16>, %15 : int, %16 : int, %17 : int, %18 : int, %19 : int, %20 : int, %21 : int, %22 : int, %23 : int)void -> {
                    %24 : int = arith.constant @"32";
                    %25 : int = arith.constant @"64";
                    %26 : int = arith.constant @"32";
                    %27 : int = arith.constant @"8";
                    %28 : int = tt.get_program_id @"0";
                    %29 : int = tt.call %15 @"cdiv_int_32_int";
                    %30 : int = tt.call %16 @"cdiv_int_64_int";
                    %31 : int = arith.muli %27 %30;
                    %32 : int = arith.divsi %28 %31;
                    %33 : int = arith.muli %32 %27;
                    %34 : int = arith.subi %29 %33;
                    %35 : int = arith.minsi %34 %27;
                    %36 : int = arith.remsi %28 %35;
                    %37 : int = arith.addi %33 %36;
                    %38 : int = arith.remsi %28 %31;
                    %39 : int = arith.divsi %38 %35;
                    %40 : tensor<x32, int> = tt.make_range @start="0" @end="32";
                    %41 : int = arith.muli %37 %24;
                    %42 : tensor<x32, int> = tt.splat %41;
                    %43 : tensor<x32, int> = arith.addi %42 %40;
                    %44 : tensor<x32, int> = tt.splat %15;
                    %45 : tensor<x32, int> = arith.remsi %43 %44;
                    %46 : tensor<x64, int> = tt.make_range @start="0" @end="64";
                    %47 : int = arith.muli %39 %25;
                    %48 : tensor<x64, int> = tt.splat %47;
                    %49 : tensor<x64, int> = arith.addi %48 %46;
                    %50 : tensor<x64, int> = tt.splat %16;
                    %51 : tensor<x64, int> = arith.remsi %49 %50;
                    %52 : tensor<x32, int> = tt.make_range @start="0" @end="32";
                    %53 : tensor<x32, x1, int> = tt.expand_dims %45 @"1";
                    %54 : tensor<x32, x1, int> = tt.splat %18;
                    %55 : tensor<x32, x1, int> = arith.muli %53 %54;
                    %56 : tensor<x1, x32, int> = tt.expand_dims %52 @"0";
                    %57 : tensor<x1, x32, int> = tt.splat %19;
                    %58 : tensor<x1, x32, int> = arith.muli %56 %57;
                    %59 : tensor<x32, x32, int> = tt.broadcast %55;
                    %60 : tensor<x32, x32, int> = tt.broadcast %58;
                    %61 : tensor<x32, x32, int> = arith.addi %59 %60;
                    %62 : tensor<x32, x32, ptr<oracle.code.triton.Float16>> = tt.splat %12;
                    %63 : tensor<x32, x32, ptr<oracle.code.triton.Float16>> = tt.addptr %62 %61;
                    %64 : tensor<x32, x1, int> = tt.expand_dims %52 @"1";
                    %65 : tensor<x32, x1, int> = tt.splat %20;
                    %66 : tensor<x32, x1, int> = arith.muli %64 %65;
                    %67 : tensor<x1, x64, int> = tt.expand_dims %51 @"0";
                    %68 : tensor<x1, x64, int> = tt.splat %21;
                    %69 : tensor<x1, x64, int> = arith.muli %67 %68;
                    %70 : tensor<x32, x64, int> = tt.broadcast %66;
                    %71 : tensor<x32, x64, int> = tt.broadcast %69;
                    %72 : tensor<x32, x64, int> = arith.addi %70 %71;
                    %73 : tensor<x32, x64, ptr<oracle.code.triton.Float16>> = tt.splat %13;
                    %74 : tensor<x32, x64, ptr<oracle.code.triton.Float16>> = tt.addptr %73 %72;
                    %75 : tensor<x32, x64, float> = arith.constant @"0.0";
                    %76 : int = arith.constant @"0";
                    %77 : int = tt.call %17 @"cdiv_int_32_int";
                    %78 : int = arith.constant @"1";
                    %79 : Tuple<tensor<x32, x64, float>, tensor<x32, x32, ptr<oracle.code.triton.Float16>>, tensor<x32, x64, ptr<oracle.code.triton.Float16>>> = scf.for %76 %77 %78 %75 %63 %74 (%80 : int, %81 : tensor<x32, x64, float>, %82 : tensor<x32, x32, ptr<oracle.code.triton.Float16>>, %83 : tensor<x32, x64, ptr<oracle.code.triton.Float16>>)Tuple<tensor<x32, x64, float>, tensor<x32, x32, ptr<oracle.code.triton.Float16>>, tensor<x32, x64, ptr<oracle.code.triton.Float16>>> -> {
                        %84 : tensor<x1, x32, int> = tt.expand_dims %52 @"0";
                        %85 : int = arith.muli %80 %26;
                        %86 : int = arith.subi %17 %85;
                        %87 : tensor<x1, x32, int> = tt.splat %86;
                        %88 : tensor<x1, x32, int> = arith.cmpi %84 %87 @"slt";
                        %89 : tensor<x32, x32, int> = tt.broadcast %88;
                        %90 : tensor<x32, x32, oracle.code.triton.Float16> = tt.load %82 %89;
                        %91 : tensor<x32, x1, int> = tt.expand_dims %52 @"1";
                        %92 : int = arith.muli %80 %26;
                        %93 : int = arith.subi %17 %92;
                        %94 : tensor<x32, x1, int> = tt.splat %93;
                        %95 : tensor<x32, x1, int> = arith.cmpi %91 %94 @"slt";
                        %96 : tensor<x32, x64, int> = tt.broadcast %95;
                        %97 : tensor<x32, x64, oracle.code.triton.Float16> = tt.load %83 %96;
                        %98 : tensor<x32, x64, float> = tt.dot %90 %97;
                        %99 : tensor<x32, x64, float> = arith.addf %81 %98;
                        %100 : int = arith.muli %26 %19;
                        %101 : tensor<x32, x32, int> = tt.splat %100;
                        %102 : tensor<x32, x32, ptr<oracle.code.triton.Float16>> = tt.addptr %82 %101;
                        %103 : int = arith.muli %26 %20;
                        %104 : tensor<x32, x64, int> = tt.splat %103;
                        %105 : tensor<x32, x64, ptr<oracle.code.triton.Float16>> = tt.addptr %83 %104;
                        scf.yield %99 %102 %105;
                    };
                    %106 : tensor<x32, x64, float> = tuple.load %79 @"0";
                    %107 : tensor<x32, x32, ptr<oracle.code.triton.Float16>> = tuple.load %79 @"1";
                    %108 : tensor<x32, x64, ptr<oracle.code.triton.Float16>> = tuple.load %79 @"2";
                    %109 : tensor<x32, x64, oracle.code.triton.Float16> = arith.truncf %106;
                    %110 : int = arith.muli %37 %24;
                    %111 : tensor<x32, int> = tt.splat %110;
                    %112 : tensor<x32, int> = arith.addi %111 %40;
                    %113 : int = arith.muli %39 %25;
                    %114 : tensor<x64, int> = tt.splat %113;
                    %115 : tensor<x64, int> = arith.addi %114 %46;
                    %116 : tensor<x32, x1, int> = tt.expand_dims %112 @"1";
                    %117 : tensor<x32, x1, int> = tt.splat %22;
                    %118 : tensor<x32, x1, int> = arith.muli %117 %116;
                    %119 : tensor<x1, x64, int> = tt.expand_dims %115 @"0";
                    %120 : tensor<x1, x64, int> = tt.splat %23;
                    %121 : tensor<x1, x64, int> = arith.muli %120 %119;
                    %122 : tensor<x32, x64, int> = tt.broadcast %118;
                    %123 : tensor<x32, x64, int> = tt.broadcast %121;
                    %124 : tensor<x32, x64, int> = arith.addi %122 %123;
                    %125 : tensor<x32, x64, ptr<oracle.code.triton.Float16>> = tt.splat %14;
                    %126 : tensor<x32, x64, ptr<oracle.code.triton.Float16>> = tt.addptr %125 %124;
                    %127 : tensor<x32, x1, int> = tt.expand_dims %112 @"1";
                    %128 : tensor<x32, x1, int> = tt.splat %15;
                    %129 : tensor<x32, x1, int> = arith.cmpi %127 %128 @"slt";
                    %130 : tensor<x1, x64, int> = tt.expand_dims %115 @"0";
                    %131 : tensor<x1, x64, int> = tt.splat %16;
                    %132 : tensor<x1, x64, int> = arith.cmpi %130 %131 @"slt";
                    %133 : tensor<x32, x64, int> = tt.broadcast %129;
                    %134 : tensor<x32, x64, int> = tt.broadcast %132;
                    %135 : tensor<x32, x64, int> = arith.andi %133 %134;
                    tt.store %126 %109 %135;
                    tt.return;
                };
                unreachable;
            };
            """)
    @CodeReflection
    static void matmul_kernel(
            // Pointers to matrices
            Ptr a_ptr, Ptr b_ptr, Ptr c_ptr,
            // Matrix dimensions
            int M, int N, int K,
            // The stride variables represent how much to increase the ptr by when moving by 1
            // element in a particular dimension. E.g. `stride_am` is how much to increase `a_ptr`
            // by to get the element one row down (A has M rows).
            int stride_am, int stride_ak,
            int stride_bk, int stride_bn,
            int stride_cm, int stride_cn,
            // Meta-parameters
            @Constant int BLOCK_SIZE_M, @Constant int BLOCK_SIZE_N, @Constant int BLOCK_SIZE_K,
            @Constant int GROUP_SIZE_M,
            @Constant boolean ACTIVATION) {

        // """Kernel for computing the matmul C = A x B.
        // A has shape (M, K), B has shape (K, N) and C has shape (M, N)
        // """
        // -----------------------------------------------------------
        // Map program ids `pid` to the block of C it should compute.
        // This is done in a grouped ordering to promote L2 data reuse.
        // See above `L2 Cache Optimizations` section for details.
        var pid = programId(0);
        var num_pid_m = cdiv(M, BLOCK_SIZE_M);
        var num_pid_n = cdiv(N, BLOCK_SIZE_N);
        var num_pid_in_group = GROUP_SIZE_M * num_pid_n;
        var group_id = pid / num_pid_in_group;
        var first_pid_m = group_id * GROUP_SIZE_M;
        var group_size_m = Math.min(num_pid_m - first_pid_m, GROUP_SIZE_M);
        var pid_m = first_pid_m + (pid % group_size_m);
        var pid_n = (pid % num_pid_in_group) / group_size_m;

        // ----------------------------------------------------------
        // Create pointers for the first blocks of A and B.
        // We will advance this pointer as we move in the K direction
        // and accumulate
        // `a_ptrs` is a block of [BLOCK_SIZE_M, BLOCK_SIZE_K] pointers
        // `b_ptrs` is a block of [BLOCK_SIZE_K, BLOCK_SIZE_N] pointers
        // See above `Pointer Arithmetics` section for details
        var offs_m = arange(0, BLOCK_SIZE_M);
        var offs_am = mod(add(pid_m * BLOCK_SIZE_M, offs_m), M);
        var offs_n = arange(0, BLOCK_SIZE_N);
        var offs_bn = mod(add(pid_n * BLOCK_SIZE_N, offs_n), N);
        var offs_k = arange(0, BLOCK_SIZE_K);
        var a_ptrs = add(a_ptr, add(
                mul(expand(offs_am, 1), stride_am),
                mul(expand(offs_k, 0), stride_ak)));
        var b_ptrs = add(b_ptr, add(
                        mul(expand(offs_k, 1), stride_bk),
                        mul(expand(offs_bn, 0), stride_bn)));

        // -----------------------------------------------------------
        // Iterate to compute a block of the C matrix.
        // We accumulate into a `[BLOCK_SIZE_M, BLOCK_SIZE_N]` block
        // of fp32 values for higher accuracy.
        // `accumulator` will be converted back to fp16 after the loop.
        var accumulator = zeros(float.class, BLOCK_SIZE_M, BLOCK_SIZE_N);
        for (int k = 0; k < cdiv(K, BLOCK_SIZE_K); k++) {
            // Load the next block of A and B, generate a mask by checking the K dimension.
            // If it is out of bounds, set it to 0.
            var a = load(a_ptrs,
                    compare(expand(offs_k, 0), K - k * BLOCK_SIZE_K, LessThan));
            var b = load(b_ptrs,
                    compare(expand(offs_k, 1), K - k * BLOCK_SIZE_K, LessThan));
            // We accumulate along the K dimension.
            accumulator = add(accumulator, dot(a, b));
            // Advance the ptrs to the next K block.
            a_ptrs = add(a_ptrs, BLOCK_SIZE_K * stride_ak);
            b_ptrs = add(b_ptrs, BLOCK_SIZE_K * stride_bk);
        }

        // You can fuse arbitrary activation functions here
        // while the accumulator is still in FP32!
//        if (ACTIVATION) {
//            // ...
//        }
        var c = Triton.conv(Float16.class, accumulator);

        // -----------------------------------------------------------
        // Write back the block of the output matrix C with masks.
        var offs_cm = add(pid_m * BLOCK_SIZE_M, offs_m);
        var offs_cn = add(pid_n * BLOCK_SIZE_N, offs_n);
        var c_ptrs = add(c_ptr, add(
                        mul(stride_cm, expand(offs_cm, 1)),
                        mul(stride_cn, expand(offs_cn, 0))));
        var c_mask = and(
                compare(expand(offs_cm, 1), M, LessThan),
                compare(expand(offs_cn, 0), N, LessThan));
        store(c_ptrs, c, c_mask);
    }

    @TritonTestExtension.Kernel("matmul_kernel")
    @Test
    public void test(TritonTestExtension.TritonTestData t) {
        List<TypeElement> argTypes = List.of(
                new PtrType(Float16.FLOAT_16_TYPE),
                new PtrType(Float16.FLOAT_16_TYPE),
                new PtrType(Float16.FLOAT_16_TYPE),
                JavaType.INT, JavaType.INT, JavaType.INT,
                JavaType.INT, JavaType.INT,
                JavaType.INT, JavaType.INT,
                JavaType.INT, JavaType.INT,
                new ConstantType(JavaType.INT, 32), new ConstantType(JavaType.INT, 64), new ConstantType(JavaType.INT, 32),
                new ConstantType(JavaType.INT, 8),
                new ConstantType(JavaType.INT, false));

        t.test(argTypes);
    }

}

/*
@triton.jit
def matmul_kernel(
        # Pointers to matrices
        a_ptr, b_ptr, c_ptr,
        # Matrix dimensions
        M, N, K,
        # The stride variables represent how much to increase the ptr by when moving by 1
        # element in a particular dimension. E.g. `stride_am` is how much to increase `a_ptr`
        # by to get the element one row down (A has M rows).
        stride_am, stride_ak,  #
        stride_bk, stride_bn,  #
        stride_cm, stride_cn,
        # Meta-parameters
        BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,  #
        GROUP_SIZE_M: tl.constexpr,  #
        ACTIVATION: tl.constexpr  #
):
    """Kernel for computing the matmul C = A x B.
    A has shape (M, K), B has shape (K, N) and C has shape (M, N)
    """
    # -----------------------------------------------------------
    # Map program ids `pid` to the block of C it should compute.
    # This is done in a grouped ordering to promote L2 data reuse.
    # See above `L2 Cache Optimizations` section for details.
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # ----------------------------------------------------------
    # Create pointers for the first blocks of A and B.
    # We will advance this pointer as we move in the K direction
    # and accumulate
    # `a_ptrs` is a block of [BLOCK_SIZE_M, BLOCK_SIZE_K] pointers
    # `b_ptrs` is a block of [BLOCK_SIZE_K, BLOCK_SIZE_N] pointers
    # See above `Pointer Arithmetics` section for details
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    # -----------------------------------------------------------
    # Iterate to compute a block of the C matrix.
    # We accumulate into a `[BLOCK_SIZE_M, BLOCK_SIZE_N]` block
    # of fp32 values for higher accuracy.
    # `accumulator` will be converted back to fp16 after the loop.
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Load the next block of A and B, generate a mask by checking the K dimension.
        # If it is out of bounds, set it to 0.
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        # We accumulate along the K dimension.
        accumulator += tl.dot(a, b)
        # Advance the ptrs to the next K block.
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
    # You can fuse arbitrary activation functions here
    # while the accumulator is still in FP32!
    if ACTIVATION == "leaky_relu":
        accumulator = leaky_relu(accumulator)
    c = accumulator.to(tl.float16)

    # -----------------------------------------------------------
    # Write back the block of the output matrix C with masks.
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


# We can fuse `leaky_relu` by providing it as an `ACTIVATION` meta-parameter in `_matmul`.
@triton.jit
def leaky_relu(x):
    x = x + 1
    return tl.where(x >= 0, x, 0.01 * x)
*/

/*

 triton/python/triton/tools/compile.py \
    --kernel-name matmul_kernel \
    --signature "*fp16,*fp16,*fp16,i32,i32,i32,i32,i32,i32,i32,i32,i32,32,64,32,8,0" \
    --grid=1024,1024,1024 \
    03-matrix-multiplication.py

BLOCK_SIZE_M = 32
BLOCK_SIZE_N = 64
BLOCK_SIZE_K = 32
GROUP_SIZE_M = 8
ACTIVATION = 0

module {
  tt.func public @matmul_kernel_01234567891011(
            %arg0: !tt.ptr<f16, 1>, %arg1: !tt.ptr<f16, 1>, %arg2: !tt.ptr<f16, 1> ,
            %arg3: i32, %arg4: i32, %arg5: i32 ,
            %arg6: i32, %arg7: i32, %arg8: i32 ,
            %%arg9: i32, %arg10: i32, %arg11: i32 ) attributes {noinline = false} {
    %0 = tt.get_program_id x : i32
    %1 = tt.call @cdiv__i32__1cconstexpr_32_(%arg3) : (i32) -> i32
    %2 = tt.call @cdiv__i32__1cconstexpr_64_(%arg4) : (i32) -> i32
    %c8_i32 = arith.constant 8 : i32
    %3 = arith.muli %2, %c8_i32 : i32
    %4 = arith.divsi %0, %3 : i32
    %c8_i32_0 = arith.constant 8 : i32
    %5 = arith.muli %4, %c8_i32_0 : i32
    %6 = arith.subi %1, %5 : i32
    %7 = tt.call @minimum__i32__1cconstexpr_8_(%6) : (i32) -> i32
    %8 = arith.remsi %0, %7 : i32
    %9 = arith.addi %5, %8 : i32
    %10 = arith.remsi %0, %3 : i32
    %11 = arith.divsi %10, %7 : i32
    %c32_i32 = arith.constant 32 : i32
    %12 = arith.muli %9, %c32_i32 : i32
    %13 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32>
    %14 = tt.splat %12 : (i32) -> tensor<32xi32>
    %15 = arith.addi %14, %13 : tensor<32xi32>
    %16 = tt.splat %arg3 : (i32) -> tensor<32xi32>
    %17 = arith.remsi %15, %16 : tensor<32xi32>
    %c64_i32 = arith.constant 64 : i32
    %18 = arith.muli %11, %c64_i32 : i32
    %19 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>
    %20 = tt.splat %18 : (i32) -> tensor<64xi32>
    %21 = arith.addi %20, %19 : tensor<64xi32>
    %22 = tt.splat %arg4 : (i32) -> tensor<64xi32>
    %23 = arith.remsi %21, %22 : tensor<64xi32>
    %24 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32>
    %25 = tt.expand_dims %17 {axis = 1 : i32} : (tensor<32xi32>) -> tensor<32x1xi32>
    %26 = tt.splat %arg6 : (i32) -> tensor<32x1xi32>
    %27 = arith.muli %25, %26 : tensor<32x1xi32>
    %28 = tt.expand_dims %24 {axis = 0 : i32} : (tensor<32xi32>) -> tensor<1x32xi32>
    %29 = tt.splat %arg7 : (i32) -> tensor<1x32xi32>
    %30 = arith.muli %28, %29 : tensor<1x32xi32>
    %31 = tt.broadcast %27 : (tensor<32x1xi32>) -> tensor<32x32xi32>
    %32 = tt.broadcast %30 : (tensor<1x32xi32>) -> tensor<32x32xi32>
    %33 = arith.addi %31, %32 : tensor<32x32xi32>
    %34 = tt.splat %arg0 : (!tt.ptr<f16, 1>) -> tensor<32x32x!tt.ptr<f16, 1>>
    %35 = tt.addptr %34, %33 : tensor<32x32x!tt.ptr<f16, 1>>, tensor<32x32xi32>
    %36 = tt.expand_dims %24 {axis = 1 : i32} : (tensor<32xi32>) -> tensor<32x1xi32>
    %37 = tt.splat %arg8 : (i32) -> tensor<32x1xi32>
    %38 = arith.muli %36, %37 : tensor<32x1xi32>
    %39 = tt.expand_dims %23 {axis = 0 : i32} : (tensor<64xi32>) -> tensor<1x64xi32>
    %40 = tt.splat %arg9 : (i32) -> tensor<1x64xi32>
    %41 = arith.muli %39, %40 : tensor<1x64xi32>
    %42 = tt.broadcast %38 : (tensor<32x1xi32>) -> tensor<32x64xi32>
    %43 = tt.broadcast %41 : (tensor<1x64xi32>) -> tensor<32x64xi32>
    %44 = arith.addi %42, %43 : tensor<32x64xi32>
    %45 = tt.splat %arg1 : (!tt.ptr<f16, 1>) -> tensor<32x64x!tt.ptr<f16, 1>>
    %46 = tt.addptr %45, %44 : tensor<32x64x!tt.ptr<f16, 1>>, tensor<32x64xi32>
    %47 = tt.call @"zeros____0cconstexpr_(constexpr_32_, constexpr_64_)__1cconstexpr_fp32_"() : () -> tensor<32x64xf32>
    %48 = tt.call @cdiv__i32__1cconstexpr_32_(%arg5) : (i32) -> i32
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %49 = arith.bitcast %c0_i32 : i32 to i32
    %50 = arith.bitcast %48 : i32 to i32
    %51 = arith.bitcast %c1_i32 : i32 to i32
    %52 = llvm.mlir.undef : i32
    %53:3 = scf.for %arg12 = %49 to %50 step %51 iter_args(%arg13 = %47, %arg14 = %35, %arg15 = %46) -> (tensor<32x64xf32>, tensor<32x32x!tt.ptr<f16, 1>>, tensor<32x64x!tt.ptr<f16, 1>>)  : i32 {
      %83 = tt.expand_dims %24 {axis = 0 : i32} : (tensor<32xi32>) -> tensor<1x32xi32>
      %c32_i32_3 = arith.constant 32 : i32
      %84 = arith.muli %arg12, %c32_i32_3 : i32
      %85 = arith.subi %arg5, %84 : i32
      %86 = tt.splat %85 : (i32) -> tensor<1x32xi32>
      %87 = arith.cmpi slt, %83, %86 : tensor<1x32xi32>
      %cst = arith.constant 0.000000e+00 : f32
      %88 = tt.broadcast %87 : (tensor<1x32xi1>) -> tensor<32x32xi1>
      %cst_4 = arith.constant dense<0.000000e+00> : tensor<32x32xf32>
      %89 = arith.truncf %cst_4 : tensor<32x32xf32> to tensor<32x32xf16>
      %90 = tt.load %arg14, %88, %89 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<32x32xf16>
      %91 = tt.expand_dims %24 {axis = 1 : i32} : (tensor<32xi32>) -> tensor<32x1xi32>
      %c32_i32_5 = arith.constant 32 : i32
      %92 = arith.muli %arg12, %c32_i32_5 : i32
      %93 = arith.subi %arg5, %92 : i32
      %94 = tt.splat %93 : (i32) -> tensor<32x1xi32>
      %95 = arith.cmpi slt, %91, %94 : tensor<32x1xi32>
      %cst_6 = arith.constant 0.000000e+00 : f32
      %96 = tt.broadcast %95 : (tensor<32x1xi1>) -> tensor<32x64xi1>
      %cst_7 = arith.constant dense<0.000000e+00> : tensor<32x64xf32>
      %97 = arith.truncf %cst_7 : tensor<32x64xf32> to tensor<32x64xf16>
      %98 = tt.load %arg15, %96, %97 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<32x64xf16>
      %cst_8 = arith.constant 0.000000e+00 : f32
      %cst_9 = arith.constant dense<0.000000e+00> : tensor<32x64xf32>
      %99 = tt.dot %90, %98, %cst_9 {allowTF32 = true, maxNumImpreciseAcc = 0 : i32} : tensor<32x32xf16> * tensor<32x64xf16> -> tensor<32x64xf32>
      %100 = arith.addf %arg13, %99 : tensor<32x64xf32>
      %c32_i32_10 = arith.constant 32 : i32
      %101 = arith.muli %arg7, %c32_i32_10 : i32
      %102 = tt.splat %101 : (i32) -> tensor<32x32xi32>
      %103 = tt.addptr %arg14, %102 : tensor<32x32x!tt.ptr<f16, 1>>, tensor<32x32xi32>
      %c32_i32_11 = arith.constant 32 : i32
      %104 = arith.muli %arg8, %c32_i32_11 : i32
      %105 = tt.splat %104 : (i32) -> tensor<32x64xi32>
      %106 = tt.addptr %arg15, %105 : tensor<32x64x!tt.ptr<f16, 1>>, tensor<32x64xi32>
      scf.yield %100, %103, %106 : tensor<32x64xf32>, tensor<32x32x!tt.ptr<f16, 1>>, tensor<32x64x!tt.ptr<f16, 1>>
    }
    %54 = arith.truncf %53#0 : tensor<32x64xf32> to tensor<32x64xf16>
    %c32_i32_1 = arith.constant 32 : i32
    %55 = arith.muli %9, %c32_i32_1 : i32
    %56 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32>
    %57 = tt.splat %55 : (i32) -> tensor<32xi32>
    %58 = arith.addi %57, %56 : tensor<32xi32>
    %c64_i32_2 = arith.constant 64 : i32
    %59 = arith.muli %11, %c64_i32_2 : i32
    %60 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>
    %61 = tt.splat %59 : (i32) -> tensor<64xi32>
    %62 = arith.addi %61, %60 : tensor<64xi32>
    %63 = tt.expand_dims %58 {axis = 1 : i32} : (tensor<32xi32>) -> tensor<32x1xi32>
    %64 = tt.splat %arg10 : (i32) -> tensor<32x1xi32>
    %65 = arith.muli %64, %63 : tensor<32x1xi32>
    %66 = tt.splat %arg2 : (!tt.ptr<f16, 1>) -> tensor<32x1x!tt.ptr<f16, 1>>
    %67 = tt.addptr %66, %65 : tensor<32x1x!tt.ptr<f16, 1>>, tensor<32x1xi32>
    %68 = tt.expand_dims %62 {axis = 0 : i32} : (tensor<64xi32>) -> tensor<1x64xi32>
    %69 = tt.splat %arg11 : (i32) -> tensor<1x64xi32>
    %70 = arith.muli %69, %68 : tensor<1x64xi32>
    %71 = tt.broadcast %67 : (tensor<32x1x!tt.ptr<f16, 1>>) -> tensor<32x64x!tt.ptr<f16, 1>>
    %72 = tt.broadcast %70 : (tensor<1x64xi32>) -> tensor<32x64xi32>
    %73 = tt.addptr %71, %72 : tensor<32x64x!tt.ptr<f16, 1>>, tensor<32x64xi32>
    %74 = tt.expand_dims %58 {axis = 1 : i32} : (tensor<32xi32>) -> tensor<32x1xi32>
    %75 = tt.splat %arg3 : (i32) -> tensor<32x1xi32>
    %76 = arith.cmpi slt, %74, %75 : tensor<32x1xi32>
    %77 = tt.expand_dims %62 {axis = 0 : i32} : (tensor<64xi32>) -> tensor<1x64xi32>
    %78 = tt.splat %arg4 : (i32) -> tensor<1x64xi32>
    %79 = arith.cmpi slt, %77, %78 : tensor<1x64xi32>
    %80 = tt.broadcast %76 : (tensor<32x1xi1>) -> tensor<32x64xi1>
    %81 = tt.broadcast %79 : (tensor<1x64xi1>) -> tensor<32x64xi1>
    %82 = arith.andi %80, %81 : tensor<32x64xi1>
    tt.store %73, %54, %82 {cache = 1 : i32, evict = 1 : i32} : tensor<32x64xf16>
    tt.return
  }
  tt.func private @cdiv__i32__1cconstexpr_32_(%arg0: i32 ) -> i32 attributes {noinline = false} {
    %c32_i32 = arith.constant 32 : i32
    %0 = arith.addi %arg0, %c32_i32 : i32
    %c1_i32 = arith.constant 1 : i32
    %1 = arith.subi %0, %c1_i32 : i32
    %c32_i32_0 = arith.constant 32 : i32
    %2 = arith.divsi %1, %c32_i32_0 : i32
    tt.return %2 : i32
  }
  tt.func private @cdiv__i32__1cconstexpr_64_(%arg0: i32 ) -> i32 attributes {noinline = false} {
    %c64_i32 = arith.constant 64 : i32
    %0 = arith.addi %arg0, %c64_i32 : i32
    %c1_i32 = arith.constant 1 : i32
    %1 = arith.subi %0, %c1_i32 : i32
    %c64_i32_0 = arith.constant 64 : i32
    %2 = arith.divsi %1, %c64_i32_0 : i32
    tt.return %2 : i32
  }
  tt.func private @minimum__i32__1cconstexpr_8_(%arg0: i32 ) -> i32 attributes {noinline = false} {
    %c8_i32 = arith.constant 8 : i32
    %0 = arith.minsi %arg0, %c8_i32 : i32
    tt.return %0 : i32
  }
  tt.func private @"zeros____0cconstexpr_(constexpr_32_, constexpr_64_)__1cconstexpr_fp32_"() -> tensor<32x64xf32> attributes {noinline = false} {
    %cst = arith.constant 0.000000e+00 : f32
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<32x64xf32>
    tt.return %cst_0 : tensor<32x64xf32>
  }
}
 */
