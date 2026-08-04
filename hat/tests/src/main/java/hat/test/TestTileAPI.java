/*
 * Copyright (c) 2026, Oracle and/or its affiliates. All rights reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * This code is free software; you can redistribute it and/or modify it
 * under the terms of the GNU General Public License version 2 only, as
 * published by the Free Software Foundation.  Oracle designates this
 * particular file as subject to the "Classpath" exception as provided
 * by Oracle in the LICENSE file that accompanied this code.
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
package hat.test;

import hat.Accelerator;
import hat.ComputeContext;
import hat.Constant;
import hat.TileContext;
import hat.TileModel;
import hat.TileOp;
import hat.TileRange;
import hat.backend.Backend;
import hat.buffer.TensorF32;

import hat.test.annotation.HatTest;
import hat.test.exceptions.HATAsserts;
import jdk.incubator.code.Reflect;

import java.lang.invoke.MethodHandles;
import java.util.Random;

import static optkl.ifacemapper.MappableIface.RO;
import static optkl.ifacemapper.MappableIface.WO;

/**
 * How to run?
 *
 * <p>
 *     Empty Tile Kernel
 *     <code>
 *         java @.ffi-opencl-test hat.test.TestTileAPI#test_hat_tile_00
 *     </code>
 * </p>
 *
 * <p>
 *     To run the Vector Addition
 * <code>
 *  java @.ffi-opencl-test hat.test.TestTileAPI#test_hat_tile_01
 * </code>
 * </p>
 *
 * <p>
 *     Matrix Multiplication
 * <code>
 * java @.ffi-opencl-test hat.test.TestTileAPI#test_hat_tile_02
 * </code>
 * </p>
 *
 * <p>
 *     Reduction
 * <code>
 *   java @.ffi-opencl-test hat.test.TestTileAPI#test_hat_tile_03
 * </code>
 * </p>
 *
 * <p>
 *     Transpose Matrix
 * <code>
 *  java @.ffi-opencl-test hat.test.TestTileAPI#test_hat_tile_04
 * </code>
 * </p>
 */
public class TestTileAPI {

    @Reflect
//    @Kernel("""
//            HAT_KERNEL void helloTile(
//                HAT_GLOBAL_MEM TileContext_t* tc,
//                HAT_GLOBAL_MEM F32Array_t* inputA,
//                HAT_GLOBAL_MEM F32Array_t* inputB,
//                HAT_GLOBAL_MEM F32Array_t* output,
//                int tile_size
//            ){
//                int pid = ct::bid().x;
////                auto a = ct::assume_aligned(inputA->array, 16_ic);
////                auto b = ct::assume_aligned(inputB->array, 16_ic);
////                auto c = ct::assume_aligned(output->array, 16_ic);
//
//                auto aTile = ct::partition_view{ct::tensor_span{inputA->array, ct::extents{1024}}, ct::shape{ 16_ic }}.load_masked(pid);
//                auto bTile = ct::partition_view{ct::tensor_span{inputB->array, ct::extents{1024}}, ct::shape{ 16_ic }}.load_masked(pid);
//                auto tileResult = aTile + bTile;
//                ct::partition_view{ct::tensor_span{output->array, ct::extents{1024}}, ct::shape{ 16_ic }}.store_masked(tileResult, pid);
//                return;
//            }∂
//            """)
    public static void helloTile(@RO TileContext tc, @RO TensorF32 inputA, @RO TensorF32 inputB, @WO TensorF32 output, @Constant int tile_size) {
        final var pid = tc.bid(0);
        var aTile = tc.load(inputA, pid, tc.shape(tile_size));
        var bTile = tc.load(inputB, pid, tile_size);
        var tileResult = TileOp.add(aTile, bTile);  // TODO: we need to infer the shape of the resulting tile based on the operands
        tc.store(output, pid, tileResult);
    }

    @Reflect
    public static void computeEmptyTile(@RO ComputeContext computeContext, @RO TensorF32 inputA, @RO TensorF32 inputB, @WO TensorF32 output, @Constant int tile_size) {
        computeContext.dispatchTile(TileRange.of1D(inputA.length(), tile_size),
                tileContext -> helloTile(tileContext, inputA, inputB, output, tile_size));
    }

    @Reflect
    @HatTest
    public void test_hat_tile_00() {
        var accelerator = new Accelerator(MethodHandles.lookup(), Backend.FIRST);
        final int size = 1024;
        final int tile_size = 16;
        TensorF32 inputA = TensorF32.create(accelerator, size);
        TensorF32 inputB = TensorF32.create(accelerator, size);

        // Fill data
        Random r = new Random();
        for (int i = 0; i < size; i++) {
            inputA.array(i, r.nextFloat());
            inputB.array(i, r.nextFloat());
        }

        TensorF32 result = TensorF32.create(accelerator, size);
        accelerator.compute( computeContext -> computeEmptyTile(computeContext, inputA, inputB, result, tile_size));

        for (int i = 0; i < size; i++) {
            HATAsserts.assertEquals((inputA.array(i) + inputB.array(i)), result.array(i), 0.01f);
        }
    }


    // ================================================================================================================
    // Expressing Vector Addition
    // ================================================================================================================


    // Initial code-model
    @TileModel(model="""
            func @loc="29:5:file:///Users/juanfumero/repos/private-babylon/hat/tests/src/main/java/hat/test/TestTileAPI.java" @"vector_add" (%0 : java.type:"hat.TileContext", %1 : java.type:"hat.buffer.TileF32Array", %2 : java.type:"hat.buffer.TileF32Array", %3 : java.type:"hat.buffer.TileF32Array", %4 : java.type:"int")java.type:"void" -> {
                %5 : Var<java.type:"hat.TileContext"> = var %0 @loc="29:5" @"tc";
                %6 : Var<java.type:"hat.buffer.TileF32Array"> = var %1 @loc="29:5" @"inputA";
                %7 : Var<java.type:"hat.buffer.TileF32Array"> = var %2 @loc="29:5" @"inputB";
                %8 : Var<java.type:"hat.buffer.TileF32Array"> = var %3 @loc="29:5" @"output";
                %9 : Var<java.type:"int"> = var %4 @loc="29:5" @"tile_size";
                %10 : java.type:"hat.TileContext" = var.load %5 @loc="33:19";
                %11 : java.type:"int" = constant @loc="33:26" @0;
                %12 : java.type:"int" = invoke %10 %11 @loc="33:19" @java.ref:"hat.TileContext::bid(int):int";
                %13 : Var<java.type:"int"> = var %12 @loc="33:9" @"pid";
                %14 : java.type:"hat.TileContext" = var.load %5 @loc="35:22";
                %15 : java.type:"hat.buffer.TileF32Array" = var.load %6 @loc="35:30";
                %16 : java.type:"int" = var.load %13 @loc="35:38";
                %17 : java.type:"int" = var.load %9 @loc="35:43";
                %18 : java.type:"hat.TileData" = invoke %14 %15 %16 %17 @loc="35:22" @java.ref:"hat.TileContext::load(hat.buffer.Buffer, int, int):hat.TileData";
                %19 : Var<java.type:"hat.TileData"> = var %18 @loc="35:9" @"a_tile";
                %20 : java.type:"hat.TileContext" = var.load %5 @loc="36:22";
                %21 : java.type:"hat.buffer.TileF32Array" = var.load %7 @loc="36:30";
                %22 : java.type:"int" = var.load %13 @loc="36:38";
                %23 : java.type:"int" = var.load %9 @loc="36:43";
                %24 : java.type:"hat.TileData" = invoke %20 %21 %22 %23 @loc="36:22" @java.ref:"hat.TileContext::load(hat.buffer.Buffer, int, int):hat.TileData";
                %25 : Var<java.type:"hat.TileData"> = var %24 @loc="36:9" @"b_tile";
                %26 : java.type:"hat.TileData" = var.load %19 @loc="40:33";
                %27 : java.type:"hat.TileData" = var.load %25 @loc="40:41";
                %28 : java.type:"hat.TileData" = invoke %26 %27 @loc="40:22" @java.ref:"hat.TileOp::add(hat.TileData, hat.TileData):hat.TileData";
                %29 : Var<java.type:"hat.TileData"> = var %28 @loc="40:9" @"result";
                %30 : java.type:"hat.TileContext" = var.load %5 @loc="42:9";
                %31 : java.type:"hat.buffer.TileF32Array" = var.load %8 @loc="42:18";
                %32 : java.type:"int" = var.load %13 @loc="42:26";
                %33 : java.type:"hat.TileData" = var.load %29 @loc="42:31";
                invoke %30 %31 %32 %33 @loc="42:9" @java.ref:"hat.TileContext::store(hat.buffer.Buffer, int, hat.TileData):void";
                return @loc="29:5";
            };
            """)
    @Reflect
    public static void vector_add(TileContext tc, TensorF32 inputA, TensorF32 inputB, TensorF32 output, @Constant int tile_size) {

        // Program id: get tile-id for 1D
        var pid = tc.bid(0);

        var a_tile = tc.load(inputA, pid, tile_size);
        var b_tile = tc.load(inputB, pid, tile_size);

        // This could be a tensor as well
        // var result = Tensor.add(a_tile, b_tile);
        var result = TileOp.add(a_tile, b_tile);

        tc.store(output, pid, result);
    }

    @Reflect
    public static void myComputeWithTile_vector_add(ComputeContext computeContext, TensorF32 inputA, TensorF32 inputB, TensorF32 output, @Constant int tile_size) {
        computeContext.dispatchTile(TileRange.of1D(inputA.length(), tile_size),
                tileContext -> vector_add(tileContext, inputA, inputB, output, tile_size));
    }

    @Reflect
    @HatTest
    public void test_hat_tile_01() {
        // Prototyping vector addition version for tile programming in HAT

        var accelerator = new Accelerator(MethodHandles.lookup(), Backend.FIRST);

        final int size = Math.powExact(2, 12);
        final int tile_size = 64;

        TensorF32 inputA = TensorF32.create(accelerator, size);
        TensorF32 inputB = TensorF32.create(accelerator, size);
        TensorF32 result = TensorF32.create(accelerator, size);

        accelerator.compute( computeContext ->
            myComputeWithTile_vector_add(computeContext, inputA, inputB, result, tile_size));
    }

    // ================================================================================================================
    // Expressing MatMul
    // ================================================================================================================
    public static final int GROUP_SIZE_M = 8;
    @TileModel(model = """
            func @loc="120:5:file:///Users/juanfumero/repos/private-babylon/hat/tests/src/main/java/hat/test/TestTileAPI.java" @"matmul" (%0 : java.type:"hat.TileContext", %1 : java.type:"hat.buffer.TileF32Array", %2 : java.type:"hat.buffer.TileF32Array", %3 : java.type:"hat.buffer.TileF32Array", %4 : java.type:"int", %5 : java.type:"int", %6 : java.type:"int", %7 : java.type:"int", %8 : java.type:"int")java.type:"void" -> {
                %9 : Var<java.type:"hat.TileContext"> = var %0 @loc="120:5" @"tc";
                %10 : Var<java.type:"hat.buffer.TileF32Array"> = var %1 @loc="120:5" @"inputA";
                %11 : Var<java.type:"hat.buffer.TileF32Array"> = var %2 @loc="120:5" @"inputB";
                %12 : Var<java.type:"hat.buffer.TileF32Array"> = var %3 @loc="120:5" @"output";
                %13 : Var<java.type:"int"> = var %4 @loc="120:5" @"tm";
                %14 : Var<java.type:"int"> = var %5 @loc="120:5" @"tn";
                %15 : Var<java.type:"int"> = var %6 @loc="120:5" @"tk";
                %16 : Var<java.type:"int"> = var %7 @loc="120:5" @"M";
                %17 : Var<java.type:"int"> = var %8 @loc="120:5" @"N";
                %18 : java.type:"hat.TileContext" = var.load %9 @loc="124:19";
                %19 : java.type:"int" = constant @loc="124:26" @0;
                %20 : java.type:"int" = invoke %18 %19 @loc="124:19" @java.ref:"hat.TileContext::bid(int):int";
                %21 : Var<java.type:"int"> = var %20 @loc="124:9" @"bid";
                %22 : java.type:"int" = var.load %16 @loc="125:38";
                %23 : java.type:"int" = var.load %13 @loc="125:41";
                %24 : java.type:"int" = invoke %22 %23 @loc="125:25" @java.ref:"java.lang.Math::ceilDiv(int, int):int";
                %25 : Var<java.type:"int"> = var %24 @loc="125:9" @"num_bid_m";
                %26 : java.type:"int" = var.load %17 @loc="126:38";
                %27 : java.type:"int" = var.load %14 @loc="126:41";
                %28 : java.type:"int" = invoke %26 %27 @loc="126:25" @java.ref:"java.lang.Math::ceilDiv(int, int):int";
                %29 : Var<java.type:"int"> = var %28 @loc="126:9" @"num_bid_n";
                %30 : java.type:"int" = field.load @loc="127:32" @java.ref:"hat.test.TestTileAPI::GROUP_SIZE_M:int";
                %31 : java.type:"int" = var.load %29 @loc="127:47";
                %32 : java.type:"int" = mul %30 %31 @loc="127:32";
                %33 : Var<java.type:"int"> = var %32 @loc="127:9" @"num_bid_in_group";
                %34 : java.type:"int" = var.load %21 @loc="128:24";
                %35 : java.type:"int" = var.load %33 @loc="128:30";
                %36 : java.type:"int" = div %34 %35 @loc="128:24";
                %37 : Var<java.type:"int"> = var %36 @loc="128:9" @"group_id";
                %38 : java.type:"int" = var.load %37 @loc="129:27";
                %39 : java.type:"int" = field.load @loc="129:38" @java.ref:"hat.test.TestTileAPI::GROUP_SIZE_M:int";
                %40 : java.type:"int" = mul %38 %39 @loc="129:27";
                %41 : Var<java.type:"int"> = var %40 @loc="129:9" @"first_bid_m";
                %42 : java.type:"int" = var.load %25 @loc="130:37";
                %43 : java.type:"int" = var.load %41 @loc="130:49";
                %44 : java.type:"int" = sub %42 %43 @loc="130:37";
                %45 : java.type:"int" = field.load @loc="130:62" @java.ref:"hat.test.TestTileAPI::GROUP_SIZE_M:int";
                %46 : java.type:"int" = invoke %44 %45 @loc="130:28" @java.ref:"java.lang.Math::min(int, int):int";
                %47 : Var<java.type:"int"> = var %46 @loc="130:9" @"group_size_m";
                %48 : java.type:"int" = var.load %41 @loc="132:20";
                %49 : java.type:"int" = var.load %21 @loc="132:35";
                %50 : java.type:"int" = var.load %47 @loc="132:41";
                %51 : java.type:"int" = mod %49 %50 @loc="132:35";
                %52 : java.type:"int" = add %48 %51 @loc="132:20";
                %53 : Var<java.type:"int"> = var %52 @loc="132:9" @"bidx";
                %54 : java.type:"int" = var.load %21 @loc="133:21";
                %55 : java.type:"int" = var.load %33 @loc="133:27";
                %56 : java.type:"int" = mod %54 %55 @loc="133:21";
                %57 : java.type:"int" = var.load %33 @loc="133:47";
                %58 : java.type:"int" = div %56 %57 @loc="133:20";
                %59 : Var<java.type:"int"> = var %58 @loc="133:9" @"bidy";
                %60 : java.type:"hat.TileContext" = var.load %9 @loc="136:25";
                %61 : java.type:"hat.buffer.TileF32Array" = var.load %10 @loc="136:38";
                %62 : java.type:"int" = constant @loc="136:46" @1;
                %63 : java.type:"hat.TileContext" = var.load %9 @loc="136:49";
                %64 : java.type:"int" = var.load %13 @loc="136:58";
                %65 : java.type:"int" = var.load %15 @loc="136:61";
                %66 : java.type:"hat.TileShape" = invoke %63 %64 %65 @loc="136:49" @java.ref:"hat.TileContext::shape(int, int):hat.TileShape";
                %67 : java.type:"int" = invoke %60 %61 %62 %66 @loc="136:25" @java.ref:"hat.TileContext::num_tiles(hat.buffer.TileF32Array, int, hat.TileShape):int";
                %68 : Var<java.type:"int"> = var %67 @loc="136:9" @"num_tiles";
                %69 : java.type:"hat.TileContext" = var.load %9 @loc="139:27";
                %70 : java.type:"int" = var.load %13 @loc="139:36";
                %71 : java.type:"int" = var.load %15 @loc="139:40";
                %72 : java.type:"hat.TileData" = invoke %69 %70 %71 @loc="139:27" @java.ref:"hat.TileContext::zeros(int, int):hat.TileData";
                %73 : Var<java.type:"hat.TileData"> = var %72 @loc="139:9" @"accumulator";
                java.for @loc="141:9"
                    ()Var<java.type:"int"> -> {
                        %74 : java.type:"int" = constant @loc="141:22" @0;
                        %75 : Var<java.type:"int"> = var %74 @loc="141:14" @"k";
                        yield %75 @loc="141:9";
                    }
                    (%76 : Var<java.type:"int">)java.type:"boolean" -> {
                        %77 : java.type:"int" = var.load %76 @loc="141:25";
                        %78 : java.type:"int" = var.load %68 @loc="141:29";
                        %79 : java.type:"boolean" = lt %77 %78 @loc="141:25";
                        yield %79 @loc="141:9";
                    }
                    (%80 : Var<java.type:"int">)java.type:"void" -> {
                        %81 : java.type:"int" = var.load %80 @loc="141:40";
                        %82 : java.type:"int" = constant @loc="141:40" @1;
                        %83 : java.type:"int" = add %81 %82 @loc="141:40";
                        var.store %80 %83 @loc="141:40";
                        yield @loc="141:9";
                    }
                    (%84 : Var<java.type:"int">)java.type:"void" -> {
                        %85 : java.type:"hat.TileContext" = var.load %9 @loc="142:25";
                        %86 : java.type:"hat.buffer.TileF32Array" = var.load %10 @loc="142:33";
                        %87 : java.type:"hat.TileContext" = var.load %9 @loc="142:41";
                        %88 : java.type:"int" = var.load %53 @loc="142:50";
                        %89 : java.type:"int" = var.load %84 @loc="142:56";
                        %90 : java.type:"hat.TileIndex2D" = invoke %87 %88 %89 @loc="142:41" @java.ref:"hat.TileContext::index(int, int):hat.TileIndex2D";
                        %91 : java.type:"hat.TileContext" = var.load %9 @loc="142:60";
                        %92 : java.type:"int" = var.load %13 @loc="142:69";
                        %93 : java.type:"int" = var.load %15 @loc="142:73";
                        %94 : java.type:"hat.TileShape" = invoke %91 %92 %93 @loc="142:60" @java.ref:"hat.TileContext::shape(int, int):hat.TileShape";
                        %95 : java.type:"hat.TileData" = invoke %85 %86 %90 %94 @loc="142:25" @java.ref:"hat.TileContext::load(hat.buffer.Buffer, hat.TileIndex2D, hat.TileShape):hat.TileData";
                        %96 : Var<java.type:"hat.TileData"> = var %95 @loc="142:13" @"tileA";
                        %97 : java.type:"hat.TileContext" = var.load %9 @loc="143:25";
                        %98 : java.type:"hat.buffer.TileF32Array" = var.load %11 @loc="143:33";
                        %99 : java.type:"hat.TileContext" = var.load %9 @loc="143:41";
                        %100 : java.type:"int" = var.load %84 @loc="143:50";
                        %101 : java.type:"int" = var.load %59 @loc="143:53";
                        %102 : java.type:"hat.TileIndex2D" = invoke %99 %100 %101 @loc="143:41" @java.ref:"hat.TileContext::index(int, int):hat.TileIndex2D";
                        %103 : java.type:"hat.TileContext" = var.load %9 @loc="143:60";
                        %104 : java.type:"int" = var.load %15 @loc="143:69";
                        %105 : java.type:"int" = var.load %14 @loc="143:73";
                        %106 : java.type:"hat.TileShape" = invoke %103 %104 %105 @loc="143:60" @java.ref:"hat.TileContext::shape(int, int):hat.TileShape";
                        %107 : java.type:"hat.TileData" = invoke %97 %98 %102 %106 @loc="143:25" @java.ref:"hat.TileContext::load(hat.buffer.Buffer, hat.TileIndex2D, hat.TileShape):hat.TileData";
                        %108 : Var<java.type:"hat.TileData"> = var %107 @loc="143:13" @"tileB";
                        %109 : java.type:"hat.TileData" = var.load %96 @loc="144:38";
                        %110 : java.type:"hat.TileData" = var.load %108 @loc="144:45";
                        %111 : java.type:"hat.TileData" = var.load %73 @loc="144:52";
                        %112 : java.type:"hat.TileData" = invoke %109 %110 %111 @loc="144:27" @java.ref:"hat.TileOp::mma(hat.TileData, hat.TileData, hat.TileData):hat.TileData";
                        var.store %73 %112 @loc="144:13";
                        java.continue @loc="141:9";
                    };
                %113 : java.type:"hat.TileContext" = var.load %9 @loc="147:9";
                %114 : java.type:"hat.buffer.TileF32Array" = var.load %12 @loc="147:18";
                %115 : java.type:"hat.TileContext" = var.load %9 @loc="147:26";
                %116 : java.type:"int" = var.load %53 @loc="147:35";
                %117 : java.type:"int" = var.load %59 @loc="147:41";
                %118 : java.type:"hat.TileIndex2D" = invoke %115 %116 %117 @loc="147:26" @java.ref:"hat.TileContext::index(int, int):hat.TileIndex2D";
                %119 : java.type:"hat.TileData" = var.load %73 @loc="147:48";
                invoke %113 %114 %118 %119 @loc="147:9" @java.ref:"hat.TileContext::store(hat.buffer.Buffer, hat.TileIndex2D, hat.TileData):void";
                return @loc="120:5";
            };
            """)
    @Reflect
    public static void matmul(TileContext tc, TensorF32 inputA, TensorF32 inputB, TensorF32 output, @Constant int tm, @Constant int tn, @Constant int tk, @Constant int M, @Constant int N) {

        // Calculate bidx and bidy using swizzle
        int bid = tc.bid(0);
        int num_bid_m = Math.ceilDiv(M, tm);
        int num_bid_n = Math.ceilDiv(N, tn);
        int num_bid_in_group = GROUP_SIZE_M * num_bid_n;    // IDEA: to get the constants in, we can do a pass over to transform this GROUP_SIZE_M (field access) into a Constant into the tree!
                                                            // We can implement a similar idea into the main HAT (thread-kernel mode).
        int group_id = bid / num_bid_in_group;
        int first_bid_m = group_id * GROUP_SIZE_M;
        int group_size_m = Math.min(num_bid_m - first_bid_m, GROUP_SIZE_M);

        int bidx = first_bid_m + (bid % group_size_m);
        int bidy = (bid % num_bid_in_group) / num_bid_in_group;

        // Calculate the total number of tiles
        int num_tiles = tc.num_tiles(inputA, 1, tc.shape(tm,tk));

        // Return type should be a TileData
        var accumulator = tc.zeros(tm, tk);

        for (int k = 0; k < num_tiles; k++) {
            var tileA = tc.load(inputA, tc.index(bidx, k), tc.shape(tm, tk));
            var tileB = tc.load(inputB, tc.index(k, bidy), tc.shape(tk, tn));
            accumulator = TileOp.mma(tileA, tileB, accumulator);
        }

        tc.store(output, tc.index(bidx, bidy), accumulator);
    }

    @Reflect
    public static void tile_matmul(ComputeContext computeContext, TensorF32 inputA, TensorF32 inputB, TensorF32 output, @Constant int tm, @Constant int tn, @Constant int tk, @Constant int M, @Constant int N) {
        computeContext.dispatchTile(TileRange.of2D(M, N, tm, tn),
                tileContext -> matmul(tileContext, inputA, inputB, output, tm, tn, tk, M, N));
    }

    @Reflect
    @HatTest
    public void test_hat_tile_02() {

        var accelerator = new Accelerator(MethodHandles.lookup(), Backend.FIRST);

        final int size = 1024;

        TensorF32 matrixA = TensorF32.create(accelerator, size * size);
        TensorF32 matrixB = TensorF32.create(accelerator, size * size);
        TensorF32 matrixC = TensorF32.create(accelerator, size * size);

        int tm = 64;
        int tn = 64;
        int tk = 64;

        accelerator.compute( computeContext -> {
            tile_matmul(computeContext, matrixA, matrixB, matrixC, tm, tn, tk, size, size);
        });
    }

    // -----------------------------------------------------------------------------------------------------
    // To continue, I need to see the mapping into the C++ binding and the C++ runtime to launch the kernel
    // Probably we need:
    // 1. Access to the device/s (CUDA device)
    // 2. Creation of a CUDA Stream
    // 3. Allocation of device buffers on the target device
    // 4. Primitives to copy data in and out
    // 5. API to launch the tile-kernel and setup the grids (as in block grids)
    // 6. Synchronization primitives
    // 7. Release device objects
    // -----------------------------------------------------------------------------------------------------


    // ================================================================================================================
    // Expressing Reductions
    // ================================================================================================================
    @TileModel(model = """
            func @loc="319:5:file:///Users/juanfumero/repos/private-babylon/hat/tests/src/main/java/hat/test/TestTileAPI.java" @"tile_reduction" (%0 : java.type:"hat.TileContext", %1 : java.type:"hat.buffer.TileF32Array", %2 : java.type:"hat.buffer.TileF32Array", %3 : java.type:"int")java.type:"void" -> {
                %4 : Var<java.type:"hat.TileContext"> = var %0 @loc="319:5" @"tileContext";
                %5 : Var<java.type:"hat.buffer.TileF32Array"> = var %1 @loc="319:5" @"input";
                %6 : Var<java.type:"hat.buffer.TileF32Array"> = var %2 @loc="319:5" @"output";
                %7 : Var<java.type:"int"> = var %3 @loc="319:5" @"tile_size";
                %8 : java.type:"hat.TileContext" = var.load %4 @loc="322:19";
                %9 : java.type:"int" = constant @loc="322:35" @0;
                %10 : java.type:"int" = invoke %8 %9 @loc="322:19" @java.ref:"hat.TileContext::bid(int):int";
                %11 : Var<java.type:"int"> = var %10 @loc="322:9" @"pid";
                %12 : java.type:"hat.TileContext" = var.load %4 @loc="325:25";
                %13 : java.type:"hat.buffer.TileF32Array" = var.load %5 @loc="325:47";
                %14 : java.type:"int" = constant @loc="325:54" @0;
                %15 : java.type:"hat.TileContext" = var.load %4 @loc="325:57";
                %16 : java.type:"int" = var.load %7 @loc="325:75";
                %17 : java.type:"hat.TileShape" = invoke %15 %16 @loc="325:57" @java.ref:"hat.TileContext::shape(int):hat.TileShape";
                %18 : java.type:"int" = invoke %12 %13 %14 %17 @loc="325:25" @java.ref:"hat.TileContext::num_tiles(hat.buffer.TileF32Array, int, hat.TileShape):int";
                %19 : Var<java.type:"int"> = var %18 @loc="325:9" @"num_tiles";
                %20 : java.type:"hat.TileContext" = var.load %4 @loc="327:19";
                %21 : java.type:"hat.TileContext" = var.load %4 @loc="327:36";
                %22 : java.type:"int" = constant @loc="327:54" @1;
                %23 : java.type:"hat.TileShape" = invoke %21 %22 @loc="327:36" @java.ref:"hat.TileContext::shape(int):hat.TileShape";
                %24 : java.type:"int" = constant @loc="327:58" @0;
                %25 : java.type:"hat.TileData" = invoke %20 %23 %24 @loc="327:19" @java.ref:"hat.TileContext::full(hat.TileShape, int):hat.TileData";
                %26 : Var<java.type:"hat.TileData"> = var %25 @loc="327:9" @"acc";
                java.for @loc="328:9"
                    ()Var<java.type:"int"> -> {
                        %27 : java.type:"int" = constant @loc="328:22" @0;
                        %28 : Var<java.type:"int"> = var %27 @loc="328:14" @"i";
                        yield %28 @loc="328:9";
                    }
                    (%29 : Var<java.type:"int">)java.type:"boolean" -> {
                        %30 : java.type:"int" = var.load %29 @loc="328:25";
                        %31 : java.type:"int" = var.load %19 @loc="328:29";
                        %32 : java.type:"boolean" = lt %30 %31 @loc="328:25";
                        yield %32 @loc="328:9";
                    }
                    (%33 : Var<java.type:"int">)java.type:"void" -> {
                        %34 : java.type:"int" = var.load %33 @loc="328:40";
                        %35 : java.type:"int" = constant @loc="328:40" @1;
                        %36 : java.type:"int" = add %34 %35 @loc="328:40";
                        var.store %33 %36 @loc="328:40";
                        yield @loc="328:9";
                    }
                    (%37 : Var<java.type:"int">)java.type:"void" -> {
                        %38 : java.type:"hat.TileContext" = var.load %4 @loc="330:25";
                        %39 : java.type:"hat.buffer.TileF32Array" = var.load %5 @loc="330:42";
                        %40 : java.type:"hat.TileContext" = var.load %4 @loc="330:49";
                        %41 : java.type:"int" = var.load %11 @loc="330:67";
                        %42 : java.type:"hat.TileIndex1D" = invoke %40 %41 @loc="330:49" @java.ref:"hat.TileContext::index(int):hat.TileIndex1D";
                        %43 : java.type:"hat.TileContext" = var.load %4 @loc="330:73";
                        %44 : java.type:"int" = var.load %7 @loc="330:91";
                        %45 : java.type:"hat.TileShape" = invoke %43 %44 @loc="330:73" @java.ref:"hat.TileContext::shape(int):hat.TileShape";
                        %46 : java.type:"hat.TileData" = invoke %38 %39 %42 %45 @loc="330:25" @java.ref:"hat.TileContext::load(hat.buffer.Buffer, hat.TileIndex1D, hat.TileShape):hat.TileData";
                        %47 : Var<java.type:"hat.TileData"> = var %46 @loc="330:13" @"tileA";
                        %48 : java.type:"hat.TileContext" = var.load %4 @loc="331:23";
                        %49 : java.type:"hat.TileData" = var.load %47 @loc="331:39";
                        %50 : java.type:"int" = constant @loc="331:46" @0;
                        %51 : java.type:"hat.TileData" = invoke %48 %49 %50 @loc="331:23" @java.ref:"hat.TileContext::sum(hat.TileData, int):hat.TileData";
                        %52 : Var<java.type:"hat.TileData"> = var %51 @loc="331:13" @"res";
                        %53 : java.type:"hat.TileData" = var.load %26 @loc="332:30";
                        %54 : java.type:"hat.TileData" = var.load %52 @loc="332:35";
                        %55 : java.type:"hat.TileData" = invoke %53 %54 @loc="332:19" @java.ref:"hat.TileOp::add(hat.TileData, hat.TileData):hat.TileData";
                        var.store %26 %55 @loc="332:13";
                        java.continue @loc="328:9";
                    };
                %56 : java.type:"hat.TileContext" = var.load %4 @loc="335:9";
                %57 : java.type:"hat.buffer.TileF32Array" = var.load %6 @loc="335:27";
                %58 : java.type:"hat.TileContext" = var.load %4 @loc="335:35";
                %59 : java.type:"int" = constant @loc="335:53" @0;
                %60 : java.type:"hat.TileIndex1D" = invoke %58 %59 @loc="335:35" @java.ref:"hat.TileContext::index(int):hat.TileIndex1D";
                %61 : java.type:"hat.TileData" = var.load %26 @loc="335:57";
                invoke %56 %57 %60 %61 @loc="335:9" @java.ref:"hat.TileContext::store(hat.buffer.Buffer, hat.TileIndex1D, hat.TileData):void";
                return @loc="319:5";
            };
            """)
    @Reflect
    public static void tile_reduction(TileContext tileContext, TensorF32 input, TensorF32 output, @Constant int tile_size) {

        // Obtain the tile-id
        int pid = tileContext.bid(0);

        // Obtain the number of tiles
        int numTiles = tileContext.num_tiles(input, 0, tileContext.shape(tile_size));

        // Initialize a tile
        var acc = tileContext.full(tileContext.shape(1), 0);

        // Perform the sum for all blocks of tiles
        for (int i = 0; i < numTiles; i++) {
            // load tile
            var tileA = tileContext.load(input, tileContext.index(pid), tileContext.shape(tile_size));
            // Perform a sum over the tile
            var res = tileContext.sum(tileA, 0);
            // Store the result into the accumulator
            acc = TileOp.add(acc, res);
        }

        // Store the final result into global memory
        tileContext.store(output, tileContext.index(0), acc);
    }

    @Reflect
    public static void computetile_reduction(ComputeContext computeContext, TensorF32 input, TensorF32 output, @Constant int tileSize) {
        computeContext.dispatchTile(TileRange.of1D(input.length(), tileSize),
                tileContext -> tile_reduction(tileContext, input, output, tileSize));
    }

    @Reflect
    @HatTest
    public void test_hat_tile_03() {
        var accelerator = new Accelerator(MethodHandles.lookup(), Backend.FIRST);

        final int size = Math.powExact(2, 12);
        final int tileSize = 64;

        TensorF32 input = TensorF32.create(accelerator, size);
        TensorF32 result = TensorF32.create(accelerator, size);

        accelerator.compute( computeContext ->
                computetile_reduction(computeContext, input, result, tileSize));
    }

    // Matrix transpose example
    @Reflect
    @TileModel(model = """
            func @loc="454:5:file:///Users/juanfumero/repos/private-babylon/hat/tests/src/main/java/hat/test/TestTileAPI.java" @"transposeKernel" (%0 : java.type:"hat.TileContext", %1 : java.type:"hat.buffer.TileF32Array", %2 : java.type:"hat.buffer.TileF32Array", %3 : java.type:"int", %4 : java.type:"int")java.type:"void" -> {
                %5 : Var<java.type:"hat.TileContext"> = var %0 @loc="454:5" @"tileContext";
                %6 : Var<java.type:"hat.buffer.TileF32Array"> = var %1 @loc="454:5" @"inputMatrix";
                %7 : Var<java.type:"hat.buffer.TileF32Array"> = var %2 @loc="454:5" @"transposedMatrix";
                %8 : Var<java.type:"int"> = var %3 @loc="454:5" @"tm";
                %9 : Var<java.type:"int"> = var %4 @loc="454:5" @"tn";
                %10 : java.type:"hat.TileContext" = var.load %5 @loc="456:20";
                %11 : java.type:"int" = constant @loc="456:36" @0;
                %12 : java.type:"int" = invoke %10 %11 @loc="456:20" @java.ref:"hat.TileContext::bid(int):int";
                %13 : Var<java.type:"int"> = var %12 @loc="456:9" @"bidx";
                %14 : java.type:"hat.TileContext" = var.load %5 @loc="457:20";
                %15 : java.type:"int" = constant @loc="457:36" @1;
                %16 : java.type:"int" = invoke %14 %15 @loc="457:20" @java.ref:"hat.TileContext::bid(int):int";
                %17 : Var<java.type:"int"> = var %16 @loc="457:9" @"bidy";
                %18 : java.type:"hat.TileContext" = var.load %5 @loc="458:25";
                %19 : java.type:"hat.buffer.TileF32Array" = var.load %6 @loc="458:42";
                %20 : java.type:"hat.TileContext" = var.load %5 @loc="458:55";
                %21 : java.type:"int" = var.load %13 @loc="458:73";
                %22 : java.type:"int" = var.load %17 @loc="458:79";
                %23 : java.type:"hat.TileIndex2D" = invoke %20 %21 %22 @loc="458:55" @java.ref:"hat.TileContext::index(int, int):hat.TileIndex2D";
                %24 : java.type:"hat.TileContext" = var.load %5 @loc="458:86";
                %25 : java.type:"int" = var.load %8 @loc="458:104";
                %26 : java.type:"int" = var.load %9 @loc="458:108";
                %27 : java.type:"hat.TileShape" = invoke %24 %25 %26 @loc="458:86" @java.ref:"hat.TileContext::shape(int, int):hat.TileShape";
                %28 : java.type:"hat.TileData" = invoke %18 %19 %23 %27 @loc="458:25" @java.ref:"hat.TileContext::load(hat.buffer.Buffer, hat.TileIndex2D, hat.TileShape):hat.TileData";
                %29 : Var<java.type:"hat.TileData"> = var %28 @loc="458:9" @"inputTile";
                %30 : java.type:"hat.TileContext" = var.load %5 @loc="459:30";
                %31 : java.type:"hat.TileData" = var.load %29 @loc="459:52";
                %32 : java.type:"hat.TileData" = invoke %30 %31 @loc="459:30" @java.ref:"hat.TileContext::transpose(hat.TileData):hat.TileData";
                %33 : Var<java.type:"hat.TileData"> = var %32 @loc="459:9" @"transposedTile";
                %34 : java.type:"hat.TileContext" = var.load %5 @loc="460:9";
                %35 : java.type:"hat.buffer.TileF32Array" = var.load %7 @loc="460:27";
                %36 : java.type:"hat.TileContext" = var.load %5 @loc="460:45";
                %37 : java.type:"int" = var.load %17 @loc="460:63";
                %38 : java.type:"int" = var.load %13 @loc="460:69";
                %39 : java.type:"hat.TileIndex2D" = invoke %36 %37 %38 @loc="460:45" @java.ref:"hat.TileContext::index(int, int):hat.TileIndex2D";
                %40 : java.type:"hat.TileData" = var.load %33 @loc="460:76";
                invoke %34 %35 %39 %40 @loc="460:9" @java.ref:"hat.TileContext::store(hat.buffer.Buffer, hat.TileIndex2D, hat.TileData):void";
                return @loc="454:5";
            };
            """)
    public static void transposeKernel(TileContext tileContext, TensorF32 inputMatrix, TensorF32 transposedMatrix, @Constant int tm, @Constant int tn) {
        // In this example we get a 2D block.
        // The block id 0 maps to a row from the input matrix.
        // the block id 1 maps to a column from the input matrix.
        int bidx = tileContext.bid(0);
        int bidy = tileContext.bid(1);

        // Load the tile with shape tm x tn into memory (e.g., registers, shared memory, or tensor memory)_
        var inputTile = tileContext.load(inputMatrix, tileContext.index(bidx, bidy), tileContext.shape(tm, tn));

        // compute the transpose function.
        var transposedTile = tileContext.transpose(inputTile);

        // store the resulting transposedTile into global memory.
        // Note that the index used are swapped.
        tileContext.store(transposedMatrix, tileContext.index(bidy, bidx), transposedTile);
    }

    @Reflect
    public static void computeTransposeKernel(ComputeContext computeContext, TensorF32 input, TensorF32 output, @Constant int M, @Constant int N, @Constant int tm, @Constant int tn) {
        computeContext.dispatchTile(TileRange.of2D(M, N, tm, tn),
                tileContext -> transposeKernel(tileContext, input, output, tm, tn));
    }

    @Reflect
    @HatTest
    public void test_hat_tile_04() {
        var accelerator = new Accelerator(MethodHandles.lookup(), Backend.FIRST);

        final int M = 2048;
        final int N = 512;
        final int tileSize = 128;

        TensorF32 input = TensorF32.create(accelerator, M * N);
        TensorF32 result = TensorF32.create(accelerator, M * N);

        accelerator.compute( computeContext ->
                computeTransposeKernel(computeContext, input, result, M, N, tileSize, tileSize));
    }
}
