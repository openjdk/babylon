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

package oracle.code.onnx.fer;

import jdk.incubator.code.CodeReflection;
import oracle.code.onnx.OnnxRuntime;
import oracle.code.onnx.Tensor;
import oracle.code.onnx.genai.TensorDataStream;

import java.io.IOException;
import java.lang.foreign.Arena;
import java.lang.foreign.ValueLayout;
import java.lang.invoke.MethodHandles;
import java.net.URL;
import java.util.Objects;

import static java.util.Optional.empty;
import static java.util.Optional.of;
import static oracle.code.onnx.OnnxOperators.*;
import static oracle.code.onnx.fer.FERCoreMLDemo.IMAGE_SIZE;

public class FERModel {

	private final Arena arena;

	// Weights and biases (constant inputs)
    final Tensor<Float> parameter1693;
    final Tensor<Float> parameter1403;
    final Tensor<Float> parameter1367;
    final Tensor<Float> parameter695;
    final Tensor<Float> parameter675;
    final Tensor<Float> parameter655;
    final Tensor<Float> parameter615;
    final Tensor<Float> parameter595;
    final Tensor<Float> parameter575;
    final Tensor<Float> parameter83;
    final Tensor<Float> parameter63;
    final Tensor<Float> parameter23;
    final Tensor<Float> parameter3;
    final Tensor<Float> constant339;
    final Tensor<Float> constant343;
    final Tensor<Float> parameter4;
    final Tensor<Float> parameter24;
    final Tensor<Float> parameter64;
    final Tensor<Float> parameter84;
    final Tensor<Float> parameter576;
    final Tensor<Float> parameter596;
    final Tensor<Float> parameter616;
    final Tensor<Float> parameter656;
    final Tensor<Float> parameter676;
    final Tensor<Float> parameter696;
    final Tensor<Long> dropout612_reshape0_shape;
    final Tensor<Long> parameter1367_reshape1_shape;
    final Tensor<Float> parameter1368;
    final Tensor<Float> parameter1404;
    final Tensor<Float> parameter1694;

	public FERModel(Arena arena) throws IOException {
		this.arena = arena;
        URL resource = Objects.requireNonNull(FERModel.class.getResource("emotion-ferplus-8.onnx.data"));
        var modelData = new TensorDataStream(arena, resource.getPath());
        parameter1693 = modelData.nextTensor(Tensor.ElementType.FLOAT, 1024, 8);
        parameter1403 = modelData.nextTensor(Tensor.ElementType.FLOAT, 1024, 1024);
        parameter1367 = modelData.nextTensor(Tensor.ElementType.FLOAT, 256, 4, 4, 1024);
        parameter695 = modelData.nextTensor(Tensor.ElementType.FLOAT, 256, 256, 3, 3);
        parameter675 = modelData.nextTensor(Tensor.ElementType.FLOAT, 256, 256, 3, 3);
        parameter655 = modelData.nextTensor(Tensor.ElementType.FLOAT, 256, 256, 3, 3);
        parameter615 = modelData.nextTensor(Tensor.ElementType.FLOAT, 256, 256, 3, 3);
        parameter595 = modelData.nextTensor(Tensor.ElementType.FLOAT, 256, 256, 3, 3);
        parameter575 = modelData.nextTensor(Tensor.ElementType.FLOAT, 256, 128, 3, 3);
        parameter83 = modelData.nextTensor(Tensor.ElementType.FLOAT, 128, 128, 3, 3);
        parameter63 = modelData.nextTensor(Tensor.ElementType.FLOAT, 128, 64, 3, 3);
        parameter23 = modelData.nextTensor(Tensor.ElementType.FLOAT, 64, 64, 3, 3);
        parameter3 = modelData.nextTensor(Tensor.ElementType.FLOAT, 64, 1, 3, 3);
        constant339 = Tensor.ofScalar(127.5f);
        constant343 = Tensor.ofScalar(255.0f);
        parameter4 = modelData.nextTensor(Tensor.ElementType.FLOAT, 64, 1, 1);
        parameter24 = modelData.nextTensor(Tensor.ElementType.FLOAT, 64, 1, 1);
        parameter64 = modelData.nextTensor(Tensor.ElementType.FLOAT, 128, 1, 1);
        parameter84 = modelData.nextTensor(Tensor.ElementType.FLOAT, 128, 1, 1);
        parameter576 = modelData.nextTensor(Tensor.ElementType.FLOAT, 256, 1, 1);
        parameter596 = modelData.nextTensor(Tensor.ElementType.FLOAT, 256, 1, 1);
        parameter616 = modelData.nextTensor(Tensor.ElementType.FLOAT, 256, 1, 1);
        parameter656 = modelData.nextTensor(Tensor.ElementType.FLOAT, 256, 1, 1);
        parameter676 = modelData.nextTensor(Tensor.ElementType.FLOAT, 256, 1, 1);
        parameter696 = modelData.nextTensor(Tensor.ElementType.FLOAT, 256, 1, 1);
        dropout612_reshape0_shape = Tensor.ofShape(new long[]{2}, 1, 4096);
        parameter1367_reshape1_shape = Tensor.ofShape(new long[]{2}, 4096, 1024);
        parameter1368 = modelData.nextTensor(Tensor.ElementType.FLOAT, 1024);
        parameter1404 = modelData.nextTensor(Tensor.ElementType.FLOAT, 1024);
        parameter1694 = modelData.nextTensor(Tensor.ElementType.FLOAT, 8);
    }

    @CodeReflection
    public Tensor<Float> cntkGraph(Tensor<Float> input3) {
        // Reshape parameter
        Tensor<Float> parameter1367_reshape1 = Reshape(parameter1367, parameter1367_reshape1_shape, empty());

        // Preprocessing subtraction and division
        Tensor<Float> minus340 = Sub(input3, constant339);
        Tensor<Float> block352 = Div(minus340, constant343);

        // Convolution/ReLU blocks
        Tensor<Float> convolution362 = Conv(block352, parameter3, empty(), empty(), of(new long[]{1L, 1L}), of("SAME_UPPER"), of(new long[]{1L, 1L}), of(1L), of(new long[]{3L, 3L}));
        Tensor<Float> plus364 = Add(convolution362, parameter4);
        Tensor<Float> reLU366 = Relu(plus364);

        Tensor<Float> convolution380 = Conv(reLU366, parameter23, empty(), empty(), of(new long[]{1L, 1L}), of("SAME_UPPER"), of(new long[]{1L, 1L}), of(1L), of(new long[]{3L, 3L}));
        Tensor<Float> plus382 = Add(convolution380, parameter24);
        Tensor<Float> reLU384 = Relu(plus382);

        var pooling398 = MaxPool(reLU384, of(new long[]{0L, 0L, 0L, 0L}), empty(), of("NOTSET"), empty(), empty(), of(new long[]{2L, 2L}), new long[]{2L, 2L});
        var dropout408 = Dropout(pooling398.Y(), empty(), empty(), empty());

        Tensor<Float> convolution418 = Conv(dropout408.output(), parameter63, empty(), empty(), of(new long[]{1L, 1L}), of("SAME_UPPER"), of(new long[]{1L, 1L}), of(1L), of(new long[]{3L, 3L}));
        Tensor<Float> plus420 = Add(convolution418, parameter64);
        Tensor<Float> reLU422 = Relu(plus420);

        Tensor<Float> convolution436 = Conv(reLU422, parameter83, empty(), empty(), of(new long[]{1L, 1L}), of("SAME_UPPER"), of(new long[]{1L, 1L}), of(1L), of(new long[]{3L, 3L}));
        Tensor<Float> plus438 = Add(convolution436, parameter84);
        Tensor<Float> reLU440 = Relu(plus438);

        var pooling454 = MaxPool(reLU440, of(new long[]{0L, 0L, 0L, 0L}), empty(), of("NOTSET"), empty(), empty(), of(new long[]{2L, 2L}), new long[]{2L, 2L});
        var dropout464 = Dropout(pooling454.Y(), empty(), empty(), empty());

        Tensor<Float> convolution474 = Conv(dropout464.output(), parameter575, empty(), empty(), of(new long[]{1L, 1L}), of("SAME_UPPER"), of(new long[]{1L, 1L}), of(1L), of(new long[]{3L, 3L}));
        Tensor<Float> plus476 = Add(convolution474, parameter576);
        Tensor<Float> reLU478 = Relu(plus476);

        Tensor<Float> convolution492 = Conv(reLU478, parameter595, empty(), empty(), of(new long[]{1L, 1L}), of("SAME_UPPER"), of(new long[]{1L, 1L}), of(1L), of(new long[]{3L, 3L}));
        Tensor<Float> plus494 = Add(convolution492, parameter596);
        Tensor<Float> reLU496 = Relu(plus494);

        Tensor<Float> convolution510 = Conv(reLU496, parameter615, empty(), empty(), of(new long[]{1L, 1L}), of("SAME_UPPER"), of(new long[]{1L, 1L}), of(1L), of(new long[]{3L, 3L}));
        Tensor<Float> plus512 = Add(convolution510, parameter616);
        Tensor<Float> reLU514 = Relu(plus512);

        var pooling528 = MaxPool(reLU514, of(new long[]{0L, 0L, 0L, 0L}), empty(), of("NOTSET"), empty(), empty(), of(new long[]{2L, 2L}), new long[]{2L, 2L});
        var dropout538 = Dropout(pooling528.Y(), empty(), empty(), empty());

        Tensor<Float> convolution548 = Conv(dropout538.output(), parameter655, empty(), empty(), of(new long[]{1L, 1L}), of("SAME_UPPER"), of(new long[]{1L, 1L}), of(1L), of(new long[]{3L, 3L}));
        Tensor<Float> plus550 = Add(convolution548, parameter656);
        Tensor<Float> reLU552 = Relu(plus550);

        Tensor<Float> convolution566 = Conv(reLU552, parameter675, empty(), empty(), of(new long[]{1L, 1L}), of("SAME_UPPER"), of(new long[]{1L, 1L}), of(1L), of(new long[]{3L, 3L}));
        Tensor<Float> plus568 = Add(convolution566, parameter676);
        Tensor<Float> reLU570 = Relu(plus568);

        Tensor<Float> convolution584 = Conv(reLU570, parameter695, empty(), empty(), of(new long[]{1L, 1L}), of("SAME_UPPER"), of(new long[]{1L, 1L}), of(1L), of(new long[]{3L, 3L}));
        Tensor<Float> plus586 = Add(convolution584, parameter696);
        Tensor<Float> reLU588 = Relu(plus586);

        var pooling602 = MaxPool(reLU588, of(new long[]{0L, 0L, 0L, 0L}), empty(), of("NOTSET"), empty(), empty(), of(new long[]{2L, 2L}), new long[]{2L, 2L});
        var dropout612 = Dropout(pooling602.Y(), empty(), empty(), empty());

        Tensor<Float> dropout612_reshape0 = Reshape(dropout612.output(), dropout612_reshape0_shape, empty());
        Tensor<Float> times622 = MatMul(dropout612_reshape0, parameter1367_reshape1);
        Tensor<Float> plus624 = Add(times622, parameter1368);
        Tensor<Float> reLU636 = Relu(plus624);
        var dropout646 = Dropout(reLU636, empty(), empty(), empty());

        Tensor<Float> times656 = MatMul(dropout646.output(), parameter1403);
        Tensor<Float> plus658 = Add(times656, parameter1404);
        Tensor<Float> reLU670 = Relu(plus658);
        var dropout680 = Dropout(reLU670, empty(), empty(), empty());

        Tensor<Float> times690 = MatMul(dropout680.output(), parameter1693);
        Tensor<Float> plus692 = Add(times690, parameter1694);

        return Softmax(plus692, of(1L));
    }

    @CodeReflection
    public Tensor<Float> condenseCNTKGraph(Tensor<Float> input3) {

        // Reshape parameter
        Tensor<Float> parameter1367_reshape1 = Reshape(
                parameter1367, parameter1367_reshape1_shape, empty());

        // Preprocessing subtraction and division
        Tensor<Float> minus340 = Sub(input3, constant339);
        Tensor<Float> block352 = Div(minus340, constant343);

        // Convolution/ReLU blocks
        Tensor<Float> reLU366 = convAddRelu(block352, parameter3, parameter4);
        Tensor<Float> reLU384 = convAddRelu(reLU366, parameter23, parameter24);
        Tensor<Float> pooled_dropout1 = maxPoolDropout(reLU384); // After 2nd ConvBlock

        Tensor<Float> reLU422 = convAddRelu(pooled_dropout1, parameter63, parameter64);
        Tensor<Float> reLU440 = convAddRelu(reLU422, parameter83, parameter84);
        Tensor<Float> pooled_dropout2 = maxPoolDropout(reLU440); // After 4th ConvBlock

        Tensor<Float> reLU478 = convAddRelu(pooled_dropout2, parameter575, parameter576);
        Tensor<Float> reLU496 = convAddRelu(reLU478, parameter595, parameter596);
        Tensor<Float> reLU514 = convAddRelu(reLU496, parameter615, parameter616);
        Tensor<Float> pooled_dropout3 = maxPoolDropout(reLU514); // After 6th ConvBlock

        Tensor<Float> reLU552 = convAddRelu(pooled_dropout3, parameter655, parameter656);
        Tensor<Float> reLU570 = convAddRelu(reLU552, parameter675, parameter676);
        Tensor<Float> reLU588 = convAddRelu(reLU570, parameter695, parameter696);
        Tensor<Float> pooled_dropout4 = maxPoolDropout(reLU588); // After 9th ConvBlock

        // Flatten
        Tensor<Float> dropout612_reshape0 = Reshape(
                pooled_dropout4, dropout612_reshape0_shape, empty());

        // Dense/Dropout layer patterns
        Tensor<Float> dense1 = MatMul(dropout612_reshape0, parameter1367_reshape1);
        Tensor<Float> plus624 = Add(dense1, parameter1368);
        Tensor<Float> reLU636 = Relu(plus624);
        var dropout646 = Dropout(reLU636, empty(), empty(), empty());

        Tensor<Float> dense2 = MatMul(dropout646.output(), parameter1403);
        Tensor<Float> plus658 = Add(dense2, parameter1404);
        Tensor<Float> reLU670 = Relu(plus658);
        var dropout680 = Dropout(reLU670, empty(), empty(), empty());

        Tensor<Float> dense3 = MatMul(dropout680.output(), parameter1693);
        Tensor<Float> plus692 = Add(dense3, parameter1694);

        return Softmax(plus692, of(1L));
    }

    // Helper method: Convolution block (Conv -> Add -> Relu)
    @CodeReflection
    private Tensor<Float> convAddRelu(Tensor<Float> input,
            Tensor<Float> weight, Tensor<Float> bias) {
        // Applies convolution, bias addition, and ReLU activation
        Tensor<Float> conv = Conv(input, weight, empty(), empty(),
                of(new long[]{1L, 1L}), of("SAME_UPPER"),
                of(new long[]{1L, 1L}), of(1L), of(new long[]{3L, 3L}));
        Tensor<Float> added = Add(conv, bias);
        return Relu(added);
    }

    // Helper method: MaxPool followed by Dropout (MaxPool -> Dropout)
    @CodeReflection
    private Tensor<Float> maxPoolDropout(Tensor<Float> input) {
        // Applies max pooling, then dropout to the input
        var pooling = MaxPool(input, of(new long[]{0L, 0L, 0L, 0L}), empty(),
                of("NOTSET"), empty(), empty(), of(new long[]{2L, 2L}),
                new long[]{2L, 2L});
        var dropout = Dropout(pooling.Y(), empty(), empty(), empty());
        return Identity(dropout.output());
    }

    public float[] classify(float[] imageData, OnnxRuntime.SessionOptions options, boolean isCondensed) {
        var imageTensor = Tensor.ofShape(arena, new long[]{1, 1, IMAGE_SIZE, IMAGE_SIZE}, imageData);
        Tensor<Float> predictionTensor;
        if (isCondensed) {
            predictionTensor = OnnxRuntime.execute(arena, MethodHandles.lookup(),
                    () -> condenseCNTKGraph(imageTensor), options);
        } else {
            predictionTensor = OnnxRuntime.execute(arena, MethodHandles.lookup(),
                    () -> cntkGraph(imageTensor), options);
        }
        return predictionTensor.data().toArray(ValueLayout.JAVA_FLOAT);
    }

}
