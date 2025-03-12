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

import java.io.IOException;
import java.io.UncheckedIOException;
import java.lang.foreign.Arena;
import java.lang.foreign.ValueLayout;
import java.lang.invoke.MethodHandles;
import jdk.incubator.code.CodeReflection;
import oracle.code.onnx.OnnxRuntime;
import oracle.code.onnx.Tensor;

import static java.util.Optional.empty;
import static java.util.Optional.of;
import static oracle.code.onnx.OnnxOperators.*;

public class MNISTModel {

    static final int IMAGE_SIZE = 28;

    private static Tensor<Float> load(String resource, long... shape) {
        try (var in = MNISTModel.class.getResourceAsStream(resource)) {
            assert in != null;
            return Tensor.ofShape(shape, in.readAllBytes(), Tensor.ElementType.FLOAT);
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
    }

    // Weights
    final Tensor<Float> conv1Weights;
    final Tensor<Float> conv1Biases;
    final Tensor<Float> conv2Weights;
    final Tensor<Float> conv2Biases;
    final Tensor<Float> fc1Weights;
    final Tensor<Float> fc1Biases;
    final Tensor<Float> fc2Weights;
    final Tensor<Float> fc2Biases;
    final Tensor<Float> fc3Weights;
    final Tensor<Float> fc3Biases;

    public MNISTModel() {
        // Load the weights (constant inputs)
        // The weights were obtained from a trained pytorch model with the MNIST dataset
        conv1Weights = load("conv1-weight-float-le", 6, 1, 5, 5);
        conv1Biases = load("conv1-bias-float-le", 6);
        conv2Weights = load("conv2-weight-float-le", 16, 6, 5, 5);
        conv2Biases = load("conv2-bias-float-le", 16);
        fc1Weights = load("fc1-weight-float-le", 120, 256);
        fc1Biases = load("fc1-bias-float-le", 120);
        fc2Weights = load("fc2-weight-float-le", 84, 120);
        fc2Biases = load("fc2-bias-float-le", 84);
        fc3Weights = load("fc3-weight-float-le", 10, 84);
        fc3Biases = load("fc3-bias-float-le", 10);
    }

    // The machine learning model, a convolutional neural network
    // Annotated with code @CodeReflection so that the method's code is accessible
    @CodeReflection
    public Tensor<Float> cnn(Tensor<Float> inputImage) {
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

    public float[] classify(float[] imageData) {
        // Manage per-execution data in a confined arena
        try (Arena arena = Arena.ofConfined()) {
            // Convert the image to an input tensor
            var imageTensor = Tensor.ofShape(arena, new long[]{1, 1, IMAGE_SIZE, IMAGE_SIZE}, imageData);

            // Execute the machine learning model
            // Translate the Java code to an ONNX model and execute in the ONNX runtime
            var predictionTensor = OnnxRuntime.execute(arena, MethodHandles.lookup(),
                    () -> cnn(imageTensor));

            // Convert the output predication to float[]
            return predictionTensor.data().toArray(ValueLayout.JAVA_FLOAT);
        }
    }
}
