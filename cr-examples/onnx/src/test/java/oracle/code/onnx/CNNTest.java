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

import java.io.*;
import java.lang.foreign.Arena;
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

import java.lang.invoke.MethodHandles;
import java.lang.reflect.Method;
import java.util.Optional;
import java.util.Set;
import java.util.stream.Stream;
import java.lang.foreign.MemorySegment;
import java.nio.ByteBuffer;
import java.nio.FloatBuffer;
import java.nio.channels.FileChannel;
import java.util.List;
import java.util.function.Function;

import static java.util.Optional.empty;
import static java.util.Optional.of;
import static oracle.code.onnx.OnnxOperators.Cast;
import static oracle.code.onnx.OnnxOperators.Constant;
import static oracle.code.onnx.OnnxOperators.Conv;
import static oracle.code.onnx.OnnxOperators.Div;
import static oracle.code.onnx.OnnxOperators.Flatten;
import static oracle.code.onnx.OnnxOperators.Gemm;
import static oracle.code.onnx.OnnxOperators.MaxPool;
import static oracle.code.onnx.OnnxOperators.Relu;
import static oracle.code.onnx.OnnxOperators.Reshape;
import static oracle.code.onnx.OnnxOperators.Softmax;

// A rough CNN implementation which expects a input [batch_size, 1, 28, 28].
// Over time we will improve the operator expressions to reduce
// the verbosity e.g., esp. scalar constant expressions
public class CNNTest {

    private static final String IMAGES_PATH = CNNTest.class.getResource("images-ubyte").getPath();
    private static final String LABELS_PATH = CNNTest.class.getResource("labels-ubyte").getPath();
    private static final int IMAGES_HEADER_SIZE = 0;
    private static final int LABELS_HEADER_SIZE = 0;

//    static final String IMAGES_PATH = CNNTest.class.getResource("t10k-images-idx3-ubyte").getPath();
//    static final String LABELS_PATH = CNNTest.class.getResource("t10k-labels-idx1-ubyte").getPath();
//    static final int IMAGES_HEADER_SIZE = 16;
//    static final int LABELS_HEADER_SIZE = 8;

    private static final String GREY_SCALE = " .'`^\",:;Il!i><~+_-?][}{1)(|\\/tfjrxnuvczXYUJCLQ0OZmwqpdbkhao*#MW&8%B@$";
    private static final Arena ARENA = Arena.ofAuto();

    private static final int PIXEL_DEPTH = 255;
    private static final int NUM_CHANNELS = 1;
    private static final int IMAGE_SIZE = 28;

    @CodeReflection
    public static Tensor<Float> cnn(
            // Weights and biases
            // [6, 1, 5, 5]
            Tensor<Float> conv1Weights,
            // [6]
            Tensor<Float> conv1Biases,
            // [16, 6, 5, 5]
            Tensor<Float> conv2Weights,
            // [16]
            Tensor<Float> conv2Biases,
            // [120, 256]
            Tensor<Float> fc1Weights,
            // [120]
            Tensor<Float> fc1Biases,
            // [84, 120]
            Tensor<Float> fc2Weights,
            // [84]
            Tensor<Float> fc2Biases,
            // [NUM_LABELS, 84]
            Tensor<Float> fc3Weights,
            // [NUM_LABELS]
            Tensor<Float> fc3Biases,
            // Inputs
            Tensor<Byte> ubyteImage) {

        Tensor<Float> inputImage = Cast(ubyteImage, empty(), Tensor.ElementType.FLOAT.id);

        var shape = Constant(new long[]{-1, NUM_CHANNELS, IMAGE_SIZE, IMAGE_SIZE});
        var inputReshaped = Reshape(inputImage, shape, empty());

        // Scaling the features to 0-1
        var scalingFactor = Constant((float) PIXEL_DEPTH);
        var scaledInput = Div(inputReshaped, scalingFactor);

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

    static CoreOp.FuncOp cnnModel() {
        // @@@ function type and result types with correct tensor element and shape

        FunctionType functionType = FunctionType.functionType(
                OnnxType.TENSOR_FLOAT32, // return
                OnnxType.TENSOR_FLOAT32, // conv1Weights
                OnnxType.TENSOR_FLOAT32, // conv1Biases
                OnnxType.TENSOR_FLOAT32, // conv2Weights
                OnnxType.TENSOR_FLOAT32, // conv2Biases
                OnnxType.TENSOR_FLOAT32, // fc1Weights
                OnnxType.TENSOR_FLOAT32, // fc1Biases
                OnnxType.TENSOR_FLOAT32, // fc2Weights
                OnnxType.TENSOR_FLOAT32, // fc2Biases
                OnnxType.TENSOR_FLOAT32, // fc3Weights
                OnnxType.TENSOR_FLOAT32,  // fc3Biases
                OnnxType.TENSOR_UINT8 // input
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
            Block.Parameter fc3Weights = b.parameters().get(8);
            Block.Parameter fc3Biases = b.parameters().get(9);
            Block.Parameter ubyteImage = b.parameters().get(10);

            var inputImage = b.op(OnnxOps.Cast(OnnxType.TENSOR_FLOAT32,
                    ubyteImage,
                    empty(),
                    OnnxType.TENSOR_FLOAT32.eType().id()));

            var shape = b.op(OnnxOps.Constant(OnnxType.TENSOR_INT64,
                    empty(),
                    empty(),
                    empty(),
                    empty(),
                    empty(),
                    of(new long[]{-1, NUM_CHANNELS, IMAGE_SIZE, IMAGE_SIZE}),
                    empty(),
                    empty()));
            var inputReshaped = b.op(OnnxOps.Reshape(inputImage.type(),
                    inputImage, shape, empty()));

            // Scaling the features
            var scalingFactor = b.op(OnnxOps.Constant(OnnxType.TENSOR_FLOAT32,
                    empty(),
                    empty(),
                    empty(),
                    of((float) PIXEL_DEPTH),
                    empty(),
                    empty(),
                    empty(),
                    empty()));
            var scaledInput = b.op(OnnxOps.Div(inputReshaped.type(), inputReshaped, scalingFactor));

            // First conv layer
            var conv1 = b.op(OnnxOps.Conv(scaledInput.type(),
                    scaledInput,
                    conv1Weights,
                    of(conv1Biases),
                    of(new long[4]),
                    of(new long[]{1,1}),
                    empty(),
                    of(new long[]{1, 1, 1, 1}),
                    of(1L),
                    of(new long[]{5,5})));
            var relu1 = b.op(OnnxOps.Relu(conv1.type(),
                    conv1));

            // First pooling layer
            // @@@ multiple results?
            var pool1Result = b.op(OnnxOps.MaxPool(TupleType.tupleType(relu1.type(), OnnxType.TENSOR_INT64),
                    Set.of(OnnxOps.MaxPool.OutputParameter.Indices),
                    relu1,
                    of(new long[4]),
                    of(new long[]{1,1}),
                    empty(),
                    of(0L),
                    empty(),
                    of(new long[]{2, 2}),
                    new long[]{2, 2}));

            // Second conv layer
            var pool1 = b.op(CoreOp.tupleLoad(pool1Result, 0));
            var conv2 = b.op(OnnxOps.Conv(pool1.type(),
                    pool1,
                    conv2Weights,
                    of(conv2Biases),
                    of(new long[4]),
                    of(new long[]{1,1}),
                    empty(),
                    of(new long[]{1, 1, 1, 1}),
                    of(1L),
                    of(new long[]{5,5})));
            var relu2 = b.op(OnnxOps.Relu(conv2.type(),
                    conv2));

            // Second pooling layer
            // @@@ multiple results?
            var pool2Result = b.op(OnnxOps.MaxPool(TupleType.tupleType(relu2.type(), OnnxType.TENSOR_INT64),
                    Set.of(OnnxOps.MaxPool.OutputParameter.Indices),
                    relu2,
                    of(new long[4]),
                    of(new long[]{1,1}),
                    empty(),
                    of(0L),
                    empty(),
                    of(new long[]{2, 2}),
                    new long[]{2, 2}));

            // Flatten inputs
            var pool2 = b.op(CoreOp.tupleLoad(pool2Result, 0));
            var flatten = b.op(OnnxOps.Flatten(pool2.type(),
                    pool2,
                    of(1L)));

            // First fully connected layer
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

            // Second fully connected layer
            var fc2 = b.op(OnnxOps.Gemm(relu3.type(),
                    relu3,
                    fc2Weights,
                    of(fc2Biases),
                    of(1f),
                    of(1L),
                    of(1f),
                    empty()));
            var relu4 = b.op(OnnxOps.Relu(fc2.type(),
                    fc2));

            // Softmax layer
            var fc3 = b.op(OnnxOps.Gemm(relu4.type(),
                    relu4,
                    fc3Weights,
                    of(fc3Biases),
                    of(1f),
                    of(1L),
                    of(1f),
                    empty()));
            var prediction = b.op(OnnxOps.Softmax(fc3.type(),
                    fc3,
                    of(1L)));

            b.op(CoreOp._return(prediction));
        });
    }

    static void printImage(int imageIndex, ByteBuffer bb) {
        System.out.println("Image #" + imageIndex + " :");
        int offset = imageIndex * 28 * 28;
        for (int y = 0; y < 28; y++) {
            for (int x = 0; x < 28; x++) {
                System.out.print(GREY_SCALE.charAt(GREY_SCALE.length() * (0xff & bb.get(offset + y * 28 + x)) / 256));
            }
            System.out.println();
        }
    }

    private static Tensor<Float> floatTensor(String resource, long... shape) throws IOException {
        try (var file = new RandomAccessFile(CNNTest.class.getResource(resource).getPath(), "r")) {
            return new Tensor(file.getChannel().map(FileChannel.MapMode.READ_ONLY, 0, file.length(), ARENA), Tensor.ElementType.FLOAT, shape);
        }
    }

    static List<Tensor<Float>> loadWeights() throws IOException {
        return List.of(floatTensor("conv1-weight-float-le", 6, 1, 5, 5),
                       floatTensor("conv1-bias-float-le", 6),
                       floatTensor("conv2-weight-float-le", 16, 6, 5, 5),
                       floatTensor("conv2-bias-float-le", 16),
                       floatTensor("fc1-weight-float-le", 120, 256),
                       floatTensor("fc1-bias-float-le", 120),
                       floatTensor("fc2-weight-float-le", 84, 120),
                       floatTensor("fc2-bias-float-le", 84),
                       floatTensor("fc3-weight-float-le", 10, 84),
                       floatTensor("fc3-bias-float-le", 10));
    }

    static int nextBestMatch(FloatBuffer fb) {
        float maxW = fb.get();
        int maxI = 0;
        for (int i = 1; i < 10; i++) {
            float w = fb.get();
            if (w > maxW) {
                maxW = w;
                maxI = i;
            }
        }
        return maxI;
    }

    @Test
    public void testModels() {
        CoreOp.FuncOp f = getFuncOp("cnn");
        CoreOp.FuncOp onnxModel = OnnxTransformer.transform(MethodHandles.lookup(), f);
        System.out.println(onnxModel.toText());

        CoreOp.FuncOp expectedOnnxModel = cnnModel();
        System.out.println(expectedOnnxModel.toText());

        Assertions.assertEquals(serialize(expectedOnnxModel), serialize(onnxModel));
    }

    @Test
    public void testInterpreter() throws Exception {
        var weights = loadWeights();
        test(inputImage -> cnn(weights.get(0), weights.get(1), weights.get(2), weights.get(3), weights.get(4),
                               weights.get(5), weights.get(6), weights.get(7), weights.get(8), weights.get(9),
                               inputImage));
    }

    @Test
    public void testProtobufModel() throws Exception {
        var weights = loadWeights();
        test(inputImage -> new Tensor(OnnxRuntime.getInstance().runFunc(
                    OnnxTransformer.transform(MethodHandles.lookup(), getFuncOp("cnn")),
                    Stream.concat(weights.stream(), Stream.of(inputImage))
                            .map(t -> t.tensorAddr).toList()).getFirst()));
    }

    private void test(Function<Tensor<Byte>, Tensor<Float>> executor) throws Exception {
        try (RandomAccessFile imagesF = new RandomAccessFile(IMAGES_PATH, "r");
             RandomAccessFile labelsF = new RandomAccessFile(LABELS_PATH, "r")) {

            ByteBuffer imagesIn = imagesF.getChannel().map(FileChannel.MapMode.READ_ONLY, IMAGES_HEADER_SIZE, imagesF.length() - IMAGES_HEADER_SIZE);
            ByteBuffer labelsIn = labelsF.getChannel().map(FileChannel.MapMode.READ_ONLY, LABELS_HEADER_SIZE, labelsF.length() - LABELS_HEADER_SIZE);

            Tensor<Byte> inputImage = new Tensor(MemorySegment.ofBuffer(imagesIn), Tensor.ElementType.UINT8, new long[]{imagesF.length() - IMAGES_HEADER_SIZE});

            FloatBuffer result = executor.apply(inputImage).asByteBuffer().asFloatBuffer();

            int matched = 0, mismatched = 0;
            while (result.remaining() > 0) {
                int expected = labelsIn.get();
                int actual = nextBestMatch(result);
                if (expected == actual) {
                    matched++;
                } else {
                    int imageIndex = labelsIn.position() - 1;
                    printImage(imageIndex, imagesIn);
                    System.out.println("expected: " + expected + " actual: " + actual);
                    System.out.println("-".repeat(28));
                    mismatched++;
                }
            }
            System.out.println("matched: " + matched + " mismatched: " + mismatched);
            Assertions.assertTrue(mismatched / matched < 0.05);
        }
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

//    @CodeReflection
//    public Tensor<Float> loadWeight(Initializer init) {
//        var buf = ByteBuffer.allocate(init.values().length).order(ByteOrder.nativeOrder());
//        buf.put(init.values());
//        buf.rewind();
//        var floatBuf = buf.asFloatBuffer();
//        var floatArr = new float[floatBuf.remaining()];
//        floatBuf.get(floatArr);
//        Tensor<Long> shape = Constant(
//                empty(), empty(), empty(), empty(), empty(), of(init.shape()), empty(), empty()
//        );
//        Tensor<Float> floats = Constant(
//                empty(), of(floatArr), empty(), empty(), empty(), empty(), empty(), empty()
//        );
//        var shaped = Reshape(floats, shape, empty());
//        return shaped;
//    }
//
//    public static void extractWeights(Path inputOnnx, Path outputSerialized) throws IOException  {
//        try (InputStream is = Files.newInputStream(inputOnnx)) {
//            OnnxMl.ModelProto model = OnnxMl.ModelProto.parseFrom(is);
//            OnnxMl.GraphProto graph = model.getGraph();
//            List<Initializer> initList = new ArrayList<>();
//            for (var init : graph.getInitializerList()) {
//                var name = init.getName();
//                var type = init.getDataType();
//                var shape = init.getDimsList().stream().mapToLong(a -> a).toArray();
//                var valuesBuf = init.getRawData().asReadOnlyByteBuffer();
//                var valuesArr = new byte[valuesBuf.remaining()];
//                valuesBuf.get(valuesArr);
//                var initializer = new Initializer(name, type, shape, valuesArr);
//                System.out.println(initializer);
//                initList.add(initializer);
//            }
//            try (ObjectOutputStream oos = new ObjectOutputStream(Files.newOutputStream(outputSerialized))) {
//                oos.writeObject(initList);
//            }
//        }
//    }
//
//    public record Initializer(String name, int type, long[] shape, byte[] values) implements java.io.Serializable {
//        @Override
//        public String toString() {
//            return "Initializer{" +
//                    "name='" + name + '\'' +
//                    ", type=" + type +
//                    ", shape=" + Arrays.toString(shape) +
//                    ", values.length=" + values.length +
//                    '}';
//        }
//    }
//
//    public static void main(String[] args) throws IOException {
//        Path inputPath = Path.of(args[0]);
//
//        Path outputPath = Path.of(args[1]);
//
//        extractWeights(inputPath, outputPath);
//    }

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
%8 : tensor<float32>,
%9 : tensor<float32>,
%10 : tensor<float32>)tensor<float32> -> {
    %11 : tensor<int64> = Constant @value_ints="[I@32910148";
    %12 : tensor<float32> = Reshape %0 %11;
    %13 : tensor<float32> = Constant @value_float="255.0";
    %14 : tensor<float32> = Div %12 %13;
    %15 : tensor<float32> = Conv %14 %1 %2 @optional_inputs="[B]" @strides="[I@2b4bac49" @pads="[I@fd07cbb" @dilations="[I@3571b748" @group="1" @kernel_shape="[I@3e96bacf";
    %16 : tensor<float32> = Relu %15;
    %17 : tensor<float32> = MaxPool %16 @ceil_mode="0" @strides="[I@484970b0" @pads="[I@4470f8a6" @dilations="[I@7c83dc97" @kernel_shape="[I@7748410a";
    %18 : tensor<float32> = Conv %17 %3 %4 @optional_inputs="[B]" @strides="[I@740773a3" @pads="[I@37f1104d" @dilations="[I@55740540" @group="1" @kernel_shape="[I@60015ef5";
    %19 : tensor<float32> = Relu %18;
    %20 : tensor<float32> = MaxPool %19 @ceil_mode="0" @strides="[I@2f54a33d" @pads="[I@1018bde2" @dilations="[I@65b3f4a4" @kernel_shape="[I@f2ff811";
    %21 : tensor<float32> = Flatten %20 @axis="1";
    %22 : tensor<float32> = Gemm %21 %5 %6 @optional_inputs="[C]" @transB="1" @beta="1.0" @alpha="1.0";
    %23 : tensor<float32> = Relu %22;
    %24 : tensor<float32> = Gemm %23 %7 %8 @optional_inputs="[C]" @transB="1" @beta="1.0" @alpha="1.0";
    %25 : tensor<float32> = Relu %24;
    %26 : tensor<float32> = Gemm %25 %9 %10 @optional_inputs="[C]" @transB="1" @beta="1.0" @alpha="1.0";
    %27 : tensor<float32> = Softmax %26 @axis="1";
    return %27;
};
     */
}