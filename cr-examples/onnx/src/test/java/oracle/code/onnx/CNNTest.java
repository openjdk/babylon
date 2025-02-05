/*
 * Copyright (c) 2025, Oracle and/or its affiliates. All rights reserved.
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

package oracle.code.onnx;

import jdk.incubator.code.Block;
import jdk.incubator.code.CodeReflection;
import jdk.incubator.code.Op;
import jdk.incubator.code.op.CoreOp;
import jdk.incubator.code.type.FunctionType;
import jdk.incubator.code.type.TupleType;
import jdk.incubator.code.writer.OpWriter;
import oracle.code.onnx.compiler.OnnxTransformer;
import oracle.code.onnx.ir.OnnxOps;
import oracle.code.onnx.ir.OnnxType;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Assertions;

import java.io.StringWriter;
import java.lang.invoke.MethodHandles;
import java.lang.reflect.Method;
import java.util.Optional;
import java.util.Set;
import java.util.stream.Stream;

import static java.util.Optional.empty;
import static java.util.Optional.of;
import static oracle.code.onnx.OnnxOperators.*;

// A rough CNN implementation -- uncertain if the padding will line up
// Over time we will improve the operator expressions to reduce
// the verbosity e.g., esp. scalar constant expressions
public class CNNTest {

    private static final int PIXEL_DEPTH = 255;
    private static final int NUM_CHANNELS = 1;
    private static final int IMAGE_SIZE = 28;
    private static final int NUM_LABELS = 10;

    @CodeReflection
    public static Tensor<Float> cnn(
            // Weights and biases
            // (5, 5, NUM_CHANNELS, 32)
            Tensor<Float> conv1Weights,
            // (32)
            Tensor<Float> conv1Biases,
            // (5, 5, 32, 64)
            Tensor<Float> conv2Weights,
            // (64)
            Tensor<Float> conv2Biases,
            // (IMAGE_SIZE * IMAGE_SIZE * 4, 512)
            Tensor<Float> fc1Weights,
            // (512)
            Tensor<Float> fc1Biases,
            // (512, NUM_LABELS)
            Tensor<Float> fc2Weights,
            // (NUM_LABELS)
            Tensor<Float> fc2Biases,
            // Inputs
            Tensor<Float> inputImage) {
        var shape = Constant(new long[]{-1, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS});
        var inputReshaped = Reshape(inputImage, shape, empty());

        // Scaling the features
        var centeringFactor = Constant(PIXEL_DEPTH / 2.0f);
        var scalingFactor = Constant((float) PIXEL_DEPTH);
        var scaledInput = Div(Sub(inputReshaped, centeringFactor), scalingFactor);

        // First conv layer
        var conv1 = Conv(scaledInput, conv1Weights, of(conv1Biases), empty(),
                empty(), of("SAME_UPPER"), of(new long[]{1, 1, 1, 1}),
                empty(), empty());
        var relu1 = Relu(conv1);

        // First pooling layer
        var pool1 = MaxPool(relu1, empty(), empty(), of("SAME_UPPER"),
                empty(), empty(), of(new long[]{1, 2, 2, 1}), new long[]{1, 2, 2, 1});

        // Second conv layer
        var conv2 = Conv(pool1.Y(), conv2Weights, of(conv2Biases), empty(),
                empty(), of("SAME_UPPER"), of(new long[]{1, 1, 1, 1}),
                empty(), empty());
        var relu2 = Relu(conv2);

        // Second pooling layer
        var pool2 = MaxPool(relu2, empty(), empty(), of("SAME_UPPER"),
                empty(), empty(), of(new long[]{1, 2, 2, 1}), new long[]{1, 2, 2, 1});

        // Flatten inputs
        var flatShape = Constant(new long[]{0, 3136});
        var flatten = Reshape(pool2.Y(), flatShape, empty());

        // Fully connected layer
        var fc1 = Gemm(flatten, fc1Weights, of(fc1Biases), of(1f), of(1L), of(1f), empty());
        var relu3 = Relu(fc1);

        // Softmax layer
        var fc2 = Gemm(relu3, fc2Weights, of(fc2Biases), of(1f), of(1L), of(1f), empty());
        var prediction = Softmax(fc2, of(1L));

        return prediction;
    }

    CoreOp.FuncOp cnnModel() {
        // @@@ function type and result types with correct tensor element and shape

        FunctionType functionType = FunctionType.functionType(
                OnnxType.TENSOR_FLOAT32, // return
                // weights & biases
                OnnxType.TENSOR_FLOAT32,
                OnnxType.TENSOR_FLOAT32,
                OnnxType.TENSOR_FLOAT32,
                OnnxType.TENSOR_FLOAT32,
                OnnxType.TENSOR_FLOAT32,
                OnnxType.TENSOR_FLOAT32,
                OnnxType.TENSOR_FLOAT32,
                OnnxType.TENSOR_FLOAT32,
                // input
                OnnxType.TENSOR_FLOAT32
        );

        return CoreOp.func("cnn", functionType).body(b -> {
            // weights & biases
            Block.Parameter conv1Weights = b.parameters().get(0);
            Block.Parameter conv1Biases = b.parameters().get(1);
            Block.Parameter conv2Weights = b.parameters().get(2);
            Block.Parameter conv2Biases = b.parameters().get(3);
            Block.Parameter fc1Weights = b.parameters().get(4);
            Block.Parameter fc1Biases = b.parameters().get(5);
            Block.Parameter fc2Weights = b.parameters().get(6);
            Block.Parameter fc2Biases = b.parameters().get(7);

            Block.Parameter inputImage = b.parameters().get(8);

            var shape = b.op(OnnxOps.Constant(OnnxType.TENSOR_INT64,
                    empty(),
                    empty(),
                    empty(),
                    empty(),
                    empty(),
                    of(new long[]{-1, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS}),
                    empty(),
                    empty()));
            var inputReshaped = b.op(OnnxOps.Reshape(inputImage.type(),
                    inputImage, shape, empty()));

            // Scaling the features
            var centeringFactor = b.op(OnnxOps.Constant(OnnxType.TENSOR_FLOAT32,
                    empty(),
                    empty(),
                    empty(),
                    of(PIXEL_DEPTH / 2.0f),
                    empty(),
                    empty(),
                    empty(),
                    empty()));
            var scalingFactor = b.op(OnnxOps.Constant(OnnxType.TENSOR_FLOAT32,
                    empty(),
                    empty(),
                    empty(),
                    of((float) PIXEL_DEPTH),
                    empty(),
                    empty(),
                    empty(),
                    empty()));
            var scaledInput = b.op(OnnxOps.Div(inputReshaped.type(),
                    b.op(OnnxOps.Sub(inputReshaped.type(),
                            inputReshaped, centeringFactor)), scalingFactor));

            // First conv layer
            var conv1 = b.op(OnnxOps.Conv(scaledInput.type(),
                    scaledInput,
                    conv1Weights,
                    of(conv1Biases),
                    empty(),
                    empty(),
                    of("SAME_UPPER"),
                    of(new long[]{1, 1, 1, 1}),
                    empty(),
                    empty()));
            var relu1 = b.op(OnnxOps.Relu(conv1.type(),
                    conv1));

            // First pooling layer
            // @@@ multiple results?
            var pool1Result = b.op(OnnxOps.MaxPool(TupleType.tupleType(relu1.type(), OnnxType.TENSOR_INT64),
                    Set.of(OnnxOps.MaxPool.OutputParameter.Indices),
                    relu1,
                    empty(),
                    empty(),
                    of("SAME_UPPER"),
                    empty(),
                    empty(),
                    of(new long[]{1, 2, 2, 1}),
                    new long[]{1, 2, 2, 1}));

            // Second conv layer
            var pool1 = b.op(CoreOp.tupleLoad(pool1Result, 0));
            var conv2 = b.op(OnnxOps.Conv(pool1.type(),
                    pool1,
                    conv2Weights,
                    of(conv2Biases),
                    empty(),
                    empty(),
                    of("SAME_UPPER"),
                    of(new long[]{1, 1, 1, 1}),
                    empty(),
                    empty()));
            var relu2 = b.op(OnnxOps.Relu(conv2.type(),
                    conv2));

            // Second pooling layer
            // @@@ multiple results?
            var pool2Result = b.op(OnnxOps.MaxPool(TupleType.tupleType(relu2.type(), OnnxType.TENSOR_INT64),
                    Set.of(OnnxOps.MaxPool.OutputParameter.Indices),
                    relu2,
                    empty(),
                    empty(),
                    of("SAME_UPPER"),
                    empty(),
                    empty(),
                    of(new long[]{1, 2, 2, 1}),
                    new long[]{1, 2, 2, 1}));

            // Flatten inputs
            var flatShape = b.op(OnnxOps.Constant(OnnxType.TENSOR_INT64,
                    empty(),
                    empty(),
                    empty(),
                    empty(),
                    empty(),
                    of(new long[]{0, 3136}),
                    empty(),
                    empty()));
            var pool2 = b.op(CoreOp.tupleLoad(pool2Result, 0));
            var flatten = b.op(OnnxOps.Reshape(pool2.type(),
                    pool2,
                    flatShape,
                    empty()));

            // Fully connected layer
            var fc1 = b.op(OnnxOps.Gemm(flatten.type(),
                    flatten,
                    fc1Weights,
                    of(fc1Biases),
                    of(1f),
                    of(1L),
                    of(1f),
                    empty()));
            var relu3 = b.op(OnnxOps.Relu(fc1.type(),
                    fc1));

            // Softmax layer
            var fc2 = b.op(OnnxOps.Gemm(relu3.type(),
                    relu3,
                    fc2Weights,
                    of(fc2Biases),
                    of(1f),
                    of(1L),
                    of(1f),
                    empty()));
            var prediction = b.op(OnnxOps.Softmax(fc2.type(),
                    fc2,
                    of(1L)));

            b.op(CoreOp._return(prediction));
        });
    }

    @Test
    public void test() {
        CoreOp.FuncOp f = getFuncOp("cnn");
        CoreOp.FuncOp onnxModel = OnnxTransformer.transform(MethodHandles.lookup(), f);
        System.out.println(onnxModel.toText());

        CoreOp.FuncOp expectedOnnxModel = cnnModel();
        System.out.println(expectedOnnxModel.toText());

        Assertions.assertEquals(serialize(expectedOnnxModel), serialize(onnxModel));
    }

    static String serialize(Op o) {
        StringWriter w = new StringWriter();
        OpWriter.writeTo(w, o, OpWriter.LocationOption.DROP_LOCATION);
        return w.toString();
    }

    static CoreOp.FuncOp getFuncOp(String name) {
        Optional<Method> om = Stream.of(CNNTest.class.getDeclaredMethods())
                .filter(m -> m.getName().equals(name))
                .findFirst();

        Method m = om.get();
        return Op.ofMethod(m).get();
    }

    /*
    ONNX code model

func @"cnn" (
%0 : tensor<float32>,
%1 : tensor<float32>,
%2 : tensor<float32>,
%3 : tensor<float32>,
%4 : tensor<float32>,
%5 : tensor<float32>,
%6 : tensor<float32>,
%7 : tensor<float32>,
%8 : tensor<float32>)tensor<float32> -> {
    %9 : tensor<int64> = Constant @value_ints="[I@7b9a4292";
    %10 : tensor<float32> = Reshape %0 %9;
    %11 : tensor<float32> = Constant @value_float="127.5";
    %12 : tensor<float32> = Constant @value_float="255.0";
    %13 : tensor<float32> = Sub %10 %11;
    %14 : tensor<float32> = Div %13 %12;
    %15 : tensor<float32> = Conv %14 %1 %2 @strides="[I@12468a38" @auto_pad="SAME_UPPER" @optional_inputs="[B]";
    %16 : tensor<float32> = Relu %15;
    %17 : tensor<float32> = MaxPool %16 @strides="[I@1aa7ecca" @auto_pad="SAME_UPPER" @kernel_shape="[I@59309333";
    %18 : tensor<float32> = Conv %17 %3 %4 @strides="[I@5876a9af" @auto_pad="SAME_UPPER" @optional_inputs="[B]";
    %19 : tensor<float32> = Relu %18;
    %20 : tensor<float32> = MaxPool %19 @strides="[I@7ec7ffd3" @auto_pad="SAME_UPPER" @kernel_shape="[I@5b239d7d";
    %21 : tensor<int64> = Constant @value_ints="[I@6b81ce95";
    %22 : tensor<float32> = Reshape %20 %21;
    %23 : tensor<float32> = Gemm %22 %5 %6 @optional_inputs="[C]" @transB="1" @beta="1.0" @alpha="1.0";
    %24 : tensor<float32> = Relu %23;
    %25 : tensor<float32> = Gemm %24 %7 %8 @optional_inputs="[C]" @transB="1" @beta="1.0" @alpha="1.0";
    %26 : tensor<float32> = Softmax %25 @axis="1";
    return %26;
};
     */
}