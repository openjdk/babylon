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

package oracle.code.onnx.mnist;

import jdk.incubator.code.CodeReflection;

import java.io.*;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.lang.invoke.MethodHandles;
import oracle.code.onnx.OnnxRuntime;
import oracle.code.onnx.Tensor;

import static java.util.Optional.empty;
import static java.util.Optional.of;
import static oracle.code.onnx.OnnxOperators.*;

public class MNISTDemo {

    static final int IMAGE_SIZE = 28;
    static final long[] IMAGE_SHAPE = new long[]{1, 1, IMAGE_SIZE, IMAGE_SIZE};

    @CodeReflection
    public static Tensor<Float> cnn(Tensor<Float> conv1Weights, Tensor<Float> conv1Biases,
                                    Tensor<Float> conv2Weights, Tensor<Float> conv2Biases,
                                    Tensor<Float> fc1Weights, Tensor<Float> fc1Biases,
                                    Tensor<Float> fc2Weights, Tensor<Float> fc2Biases,
                                    Tensor<Float> fc3Weights, Tensor<Float> fc3Biases,
                                    Tensor<Float> inputImage) {
        // Scaling to 0-1
        var scaledInput = Div(inputImage, Constant(255f));

        // First conv layer
        var conv1 = Conv(scaledInput, conv1Weights, of(conv1Biases), of(new long[4]),
                of(new long[]{1,1}), empty(), of(new long[]{1, 1, 1, 1}),
                of(1L), of(new long[]{5,5}));
        var relu1 = Relu(conv1);

        // First pooling layer
        var pool1 = MaxPool(relu1, of(new long[4]), of(new long[]{1,1}), empty(),
                of(0L), empty(), of(new long[]{2, 2}), new long[]{2, 2});

        // Second conv layer
        var conv2 = Conv(pool1.Y(), conv2Weights, of(conv2Biases), of(new long[4]),
                of(new long[]{1,1}), empty(), of(new long[]{1, 1, 1, 1}),
                of(1L), of(new long[]{5,5}));
        var relu2 = Relu(conv2);

        // Second pooling layer
        var pool2 = MaxPool(relu2, of(new long[4]), of(new long[]{1,1}), empty(),
                of(0L), empty(), of(new long[]{2, 2}), new long[]{2, 2});

        // Flatten inputs
        var flatten = Flatten(pool2.Y(), of(1L));

        // First fully connected layer
        var fc1 = Gemm(flatten, fc1Weights, of(fc1Biases), of(1f), of(1L), of(1f), empty());
        var relu3 = Relu(fc1);

        // Second fully connected layer
        var fc2 = Gemm(relu3, fc2Weights, of(fc2Biases), of(1f), of(1L), of(1f), empty());
        var relu4 = Relu(fc2);

        // Softmax layer
        var fc3 = Gemm(relu4, fc3Weights, of(fc3Biases), of(1f), of(1L), of(1f), empty());
        var prediction = Softmax(fc3, of(1L));

        return prediction;
    }

    private static Tensor<Float> initialize(String resource, long... shape) {
        try (var in = MNISTDemo.class.getResourceAsStream(resource)) {
            return Tensor.ofShape(shape, MemorySegment.ofArray(in.readAllBytes()).toArray(ValueLayout.JAVA_FLOAT_UNALIGNED));
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
    }

    private final Tensor<Float> conv1Weights, conv1Biases,
                                conv2Weights, conv2Biases,
                                fc1Weights, fc1Biases,
                                fc2Weights, fc2Biases,
                                fc3Weights, fc3Biases;

    public MNISTDemo() {
        conv1Weights = initialize("conv1-weight-float-le", 6, 1, 5, 5);
        conv1Biases = initialize("conv1-bias-float-le", 6);
        conv2Weights = initialize("conv2-weight-float-le", 16, 6, 5, 5);
        conv2Biases = initialize("conv2-bias-float-le", 16);
        fc1Weights = initialize("fc1-weight-float-le", 120, 256);
        fc1Biases = initialize("fc1-bias-float-le", 120);
        fc2Weights = initialize("fc2-weight-float-le", 84, 120);
        fc2Biases = initialize("fc2-bias-float-le", 84);
        fc3Weights = initialize("fc3-weight-float-le", 10, 84);
        fc3Biases = initialize("fc3-bias-float-le", 10);
    }

    public float[] classify(float[] imageData) {

        return classify(conv1Weights, conv1Biases,
                        conv2Weights, conv2Biases,
                        fc1Weights, fc1Biases,
                        fc2Weights, fc2Biases,
                        fc3Weights, fc3Biases,
                        imageData);
    }

    float[] classify(Tensor<Float> conv1Weights, Tensor<Float> conv1Biases,
                     Tensor<Float> conv2Weights, Tensor<Float> conv2Biases,
                     Tensor<Float> fc1Weights, Tensor<Float> fc1Biases,
                     Tensor<Float> fc2Weights, Tensor<Float> fc2Biases,
                     Tensor<Float> fc3Weights, Tensor<Float> fc3Biases,
                     float[] imageData) {

        try (var arena = Arena.ofConfined()) {

            Tensor<Float> imageTensor = Tensor.ofShape(arena, IMAGE_SHAPE, imageData);

            return OnnxRuntime.execute(arena, MethodHandles.lookup(), 10,
                    () -> cnn(conv1Weights, conv1Biases,
                              conv2Weights, conv2Biases,
                              fc1Weights, fc1Biases,
                              fc2Weights, fc2Biases,
                              fc3Weights, fc3Biases,
                              imageTensor)).data().toArray(ValueLayout.JAVA_FLOAT);
        }
    }
}
