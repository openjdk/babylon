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

import java.awt.*;
import java.awt.event.*;
import java.awt.image.BufferedImage;
import java.io.*;
import jdk.incubator.code.CodeReflection;
import java.util.Optional;
import java.lang.foreign.MemorySegment;
import java.lang.invoke.MethodHandles;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.util.List;
import java.util.concurrent.atomic.AtomicBoolean;
import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.JPanel;
import jdk.incubator.code.Op;
import oracle.code.onnx.compiler.OnnxTransformer;

import static java.util.Optional.empty;
import static java.util.Optional.of;
import static oracle.code.onnx.OnnxOperators.*;
import static oracle.code.onnx.Tensor.ElementType.*;

public class DigitRecognizer {

    private static final int PIXEL_DEPTH = 255;

    private static float[] loadConstant(String resource) throws IOException {
        var bb = ByteBuffer.wrap(DigitRecognizer.class.getResourceAsStream(resource).readAllBytes());
        return FloatBuffer.allocate(bb.capacity() / 4).put(bb.asFloatBuffer()).array();
    }

    @CodeReflection
    public static Tensor<Float> cnn(Tensor<Float> inputImage) throws IOException {

        // Scaling and inverting the grayscale to 0-1
        var scalingFactor = Constant((float) PIXEL_DEPTH);
        var scaledInput = Div(Sub(scalingFactor, inputImage), scalingFactor);

        // First conv layer
        var conv1Weights = Reshape(Constant(loadConstant("conv1-weight-float")), Constant(new long[]{6, 1, 5, 5}), empty());
        var conv1Biases = Reshape(Constant(loadConstant("conv1-bias-float")), Constant(new long[]{6}), empty());
        var conv1 = Conv(scaledInput, conv1Weights, of(conv1Biases), of(new long[4]),
                of(new long[]{1,1}), empty(), of(new long[]{1, 1, 1, 1}),
                of(1L), of(new long[]{5,5}));
        var relu1 = Relu(conv1);

        // First pooling layer
        var pool1 = MaxPool(relu1, of(new long[4]), of(new long[]{1,1}), empty(),
                of(0L), empty(), of(new long[]{2, 2}), new long[]{2, 2});

        // Second conv layer
        var conv2Weights = Reshape(Constant(loadConstant("conv2-weight-float")), Constant(new long[]{16, 6, 5, 5}), empty());
        var conv2Biases = Reshape(Constant(loadConstant("conv2-bias-float")), Constant(new long[]{16}), empty());
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
        var fc1Weights = Reshape(Constant(loadConstant("fc1-weight-float")), Constant(new long[]{120, 256}), empty());
        var fc1Biases = Reshape(Constant(loadConstant("fc1-bias-float")), Constant(new long[]{120}), empty());
        var fc1 = Gemm(flatten, fc1Weights, of(fc1Biases), of(1f), of(1L), of(1f), empty());
        var relu3 = Relu(fc1);

        // Second fully connected layer
        var fc2Weights = Reshape(Constant(loadConstant("fc2-weight-float")), Constant(new long[]{84, 120}), empty());
        var fc2Biases = Reshape(Constant(loadConstant("fc2-bias-float")), Constant(new long[]{84}), empty());
        var fc2 = Gemm(relu3, fc2Weights, of(fc2Biases), of(1f), of(1L), of(1f), empty());
        var relu4 = Relu(fc2);

        // Softmax layer
        var fc3Weights = Reshape(Constant(loadConstant("fc3-weight-float")), Constant(new long[]{10, 84}), empty());
        var fc3Biases = Reshape(Constant(loadConstant("fc3-bias-float")), Constant(new long[]{10}), empty());
        var fc3 = Gemm(relu4, fc3Weights, of(fc3Biases), of(1f), of(1L), of(1f), empty());
        var prediction = Softmax(fc3, of(1L));

        return prediction;
    }

    public static void main(String[] args) throws Exception {
        var frame = new JFrame("Digit Recognizer");
        var pane = new JPanel();
        var status = new JLabel("   Hold SHIFT key to draw with trackpad, click ENTER to run digit recognition.");
        var robot = new Robot();
        var clean = new AtomicBoolean(true);
        var session = OnnxRuntime.getInstance().createSession(
                OnnxProtoBuilder.buildFuncModel(OnnxTransformer.transform(MethodHandles.lookup(),
                        Op.ofMethod(DigitRecognizer.class.getDeclaredMethod("cnn", Tensor.class)).get())));
        var image = new BufferedImage(28, 28, BufferedImage.TYPE_BYTE_GRAY);
        var graphics = image.createGraphics();
        var imageBuffer = ByteBuffer.allocateDirect(28 * 28 * 4).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer();
        var sampleArray = new float[28 * 28];
        var inputTensors = List.of(Optional.of(new Tensor(MemorySegment.ofBuffer(imageBuffer), FLOAT, 1, 1, 28, 28).tensorAddr));

        frame.setLayout(new BorderLayout());
        frame.add(pane, BorderLayout.CENTER);
        frame.add(status, BorderLayout.SOUTH);
        frame.setBackground(Color.WHITE);
        frame.addMouseMotionListener(new MouseAdapter() {
            @Override
            public void mouseMoved(MouseEvent e) {
                if ((e.getModifiersEx() & InputEvent.SHIFT_DOWN_MASK) != 0) {
                    if (clean.getAndSet(false)) {
                        pane.getGraphics().clearRect(0, 0, pane.getWidth(), pane.getHeight());
                    }
                    pane.getGraphics().fillOval(e.getX(), e.getY(), 20, 20);
                }
            }
        });
        frame.addKeyListener(new KeyAdapter(){
            @Override
            public void keyPressed(KeyEvent e) {
                if (e.getKeyCode() == KeyEvent.VK_ENTER) {
                    graphics.drawImage(robot.createScreenCapture(new Rectangle(pane.getLocationOnScreen(), pane.getSize()))
                                     .getScaledInstance(28, 28, Image.SCALE_SMOOTH), 0, 0, null);
                    imageBuffer.put(0, image.getData().getSamples(0, 0, 28, 28, 0, sampleArray));
                    FloatBuffer result = OnnxRuntime.getInstance().tensorBuffer(session.run(inputTensors).getFirst()).asFloatBuffer();
                    int max = 0;
                    for (int i = 1; i < 10; i++) {
                        if (result.get(i) > result.get(max)) max = i;
                    }
                    var msg = new StringBuilder("<html>&nbsp;");
                    for (int i = 0; i < 10; i++) {
                        if (max == i) {
                            msg.append("&nbsp;&nbsp;<b>%d:&nbsp;%.1f%%</b>".formatted(i, 100 * result.get(i)));
                        } else {
                            msg.append("&nbsp;&nbsp;%d:&nbsp;%.1f%%".formatted(i, result.get(i)));

                        }
                    }
                    status.setText(msg.toString());
                    clean.set(true);
                }
            }
        });
        frame.setSize(600, 600);
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        frame.setLocationRelativeTo(null);
        frame.setVisible(true);
    }
}
