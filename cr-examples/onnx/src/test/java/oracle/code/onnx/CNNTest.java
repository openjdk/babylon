package oracle.code.onnx;

import jdk.incubator.code.CodeReflection;

import java.util.Optional;

import static java.util.Optional.*;
import static oracle.code.onnx.OnnxOperators.*;

// A rough CNN implementation -- uncertain if the padding will line up
// Over time we will improve the operator expressions to reduce
// the verbosity e.g., esp. scalar constant expressions
public class CNNTest {

    private static final int PIXEL_DEPTH = 255;
    private static final int NUM_CHANNELS = 1;
    private static final int IMAGE_SIZE = 28;
    private static final int NUM_LABELS = 10;

    // (5, 5, NUM_CHANNELS, 32)
    private Tensor<Float> conv1Weights;
    // (32)
    private Tensor<Float> conv1Biases;
    // (5, 5, 32, 64)
    private Tensor<Float> conv2Weights;
    // (64)
    private Tensor<Float> conv2Biases;
    // (IMAGE_SIZE * IMAGE_SIZE * 4, 512)
    private Tensor<Float> fc1Weights;
    // (512)
    private Tensor<Float> fc1Biases;
    // (512, NUM_LABELS)
    private Tensor<Float> fc2Weights;
    // (NUM_LABELS)
    private Tensor<Float> fc2Biases;

    @CodeReflection
    public Tensor<Float> cnn(Tensor<Float> inputImage) {
        var shape = Constant(new int[]{-1, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS});
        var inputReshaped = Reshape(inputImage, shape, empty());

        // Scaling the features
        var centeringFactor = Constant(PIXEL_DEPTH / 2.0f);
        var scalingFactor = Constant((float) PIXEL_DEPTH);
        var scaledInput = Div(Sub(inputReshaped, centeringFactor), scalingFactor);

        // First conv layer
        var conv1 = Conv(scaledInput, conv1Weights, of(conv1Biases), empty(),
                empty(), of("SAME_UPPER"), of(new int[]{1, 1, 1, 1}),
                empty(), empty());
        var relu1 = Relu(conv1);

        // First pooling layer
        var pool1 = MaxPool(relu1, empty(), empty(), of("SAME_UPPER"),
                empty(), empty(), of(new int[]{1, 2, 2, 1}), new int[]{1, 2, 2, 1});

        // Second conv layer
        var conv2 = Conv(pool1.getFirst(), conv2Weights, of(conv2Biases), empty(),
                empty(), of("SAME_UPPER"), of(new int[]{1, 1, 1, 1}),
                empty(), empty());
        var relu2 = Relu(conv2);

        // Second pooling layer
        var pool2 = MaxPool(relu2, empty(), empty(), of("SAME_UPPER"),
                empty(), empty(), of(new int[]{1, 2, 2, 1}), new int[]{1, 2, 2, 1});

        // Flatten inputs
        var flatShape = Constant(new int[]{0, 3136});
        var flatten = Reshape(pool2.getFirst(), flatShape, empty());

        // Fully connected layer
        var fc1 = Gemm(flatten, fc1Weights, of(fc1Biases), of(1f), of(1), of(1f), empty());
        var relu3 = Relu(fc1);

        // Softmax layer
        var fc2 = Gemm(relu3, fc2Weights, of(fc2Biases), of(1f), of(1), of(1f), empty());
        var prediction = Softmax(fc2, of(1));

        return prediction;
    }
}