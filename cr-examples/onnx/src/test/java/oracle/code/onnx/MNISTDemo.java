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
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.concurrent.atomic.AtomicBoolean;
import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.JPanel;

import static java.util.Optional.empty;
import static java.util.Optional.of;
import static oracle.code.onnx.OnnxOperators.*;

public class MNISTDemo {
    public static float[] loadConstant(String resource) {
        try {
            return MemorySegment.ofArray(MNISTDemo.class.getResourceAsStream(resource).readAllBytes())
                    .toArray(ValueLayout.JAVA_FLOAT_UNALIGNED);
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
    }

    public static Tensor<Float> cnn(Tensor<Float> inputImage) {
        return OnnxRuntime.execute(() -> {
            // Scaling to 0-1
            var scaledInput = Div(inputImage, Constant(255f));

            // First conv layer
            var conv1Weights = Reshape(Constant(loadConstant("conv1-weight-float-le")), Constant(new long[]{6, 1, 5, 5}), empty());
            var conv1Biases = Reshape(Constant(loadConstant("conv1-bias-float-le")), Constant(new long[]{6}), empty());
            var conv1 = Conv(scaledInput, conv1Weights, of(conv1Biases), of(new long[4]),
                    of(new long[]{1,1}), empty(), of(new long[]{1, 1, 1, 1}),
                    of(1L), of(new long[]{5,5}));
            var relu1 = Relu(conv1);

            // First pooling layer
            var pool1 = MaxPool(relu1, of(new long[4]), of(new long[]{1,1}), empty(),
                    of(0L), empty(), of(new long[]{2, 2}), new long[]{2, 2});

            // Second conv layer
            var conv2Weights = Reshape(Constant(loadConstant("conv2-weight-float-le")), Constant(new long[]{16, 6, 5, 5}), empty());
            var conv2Biases = Reshape(Constant(loadConstant("conv2-bias-float-le")), Constant(new long[]{16}), empty());
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
            var fc1Weights = Reshape(Constant(loadConstant("fc1-weight-float-le")), Constant(new long[]{120, 256}), empty());
            var fc1Biases = Reshape(Constant(loadConstant("fc1-bias-float-le")), Constant(new long[]{120}), empty());
            var fc1 = Gemm(flatten, fc1Weights, of(fc1Biases), of(1f), of(1L), of(1f), empty());
            var relu3 = Relu(fc1);

            // Second fully connected layer
            var fc2Weights = Reshape(Constant(loadConstant("fc2-weight-float-le")), Constant(new long[]{84, 120}), empty());
            var fc2Biases = Reshape(Constant(loadConstant("fc2-bias-float-le")), Constant(new long[]{84}), empty());
            var fc2 = Gemm(relu3, fc2Weights, of(fc2Biases), of(1f), of(1L), of(1f), empty());
            var relu4 = Relu(fc2);

            // Softmax layer
            var fc3Weights = Reshape(Constant(loadConstant("fc3-weight-float-le")), Constant(new long[]{10, 84}), empty());
            var fc3Biases = Reshape(Constant(loadConstant("fc3-bias-float-le")), Constant(new long[]{10}), empty());
            var fc3 = Gemm(relu4, fc3Weights, of(fc3Biases), of(1f), of(1L), of(1f), empty());
            var prediction = Softmax(fc3, of(1L));

            return prediction;
        });
    }

    static final int IMAGE_SIZE = 28;
    static final int DRAW_AREA_SIZE = 600;
    static final int PEN_SIZE = 20;
    static final String[] COLORS = {"1034a6", "412f88", "722b6a", "a2264b", "d3212d", "f62d2d"};

    public static void main(String[] args) throws Exception {
        var frame = new JFrame("CNN MNIST Demo - Handwritten Digit Classification");
        var drawPane = new JPanel(false);
        var results = new JLabel();
        var cleanFlag = new AtomicBoolean(true);
        var drawImage = new BufferedImage(DRAW_AREA_SIZE, DRAW_AREA_SIZE, BufferedImage.TYPE_BYTE_GRAY);

        results.setPreferredSize(new Dimension(100, 0));
        drawPane.setPreferredSize(new Dimension(DRAW_AREA_SIZE, DRAW_AREA_SIZE));
        drawPane.addMouseMotionListener(new MouseAdapter() {
            @Override
            public void mouseMoved(MouseEvent e) {
                if ((e.getModifiersEx() & InputEvent.SHIFT_DOWN_MASK) != 0) {
                    if (cleanFlag.getAndSet(false)) {
                        drawImage.getGraphics().clearRect(0, 0, DRAW_AREA_SIZE, DRAW_AREA_SIZE);
                        drawPane.getGraphics().clearRect(0, 0, DRAW_AREA_SIZE, DRAW_AREA_SIZE);
                    }
                    drawImage.getGraphics().fillOval(e.getX(), e.getY(), PEN_SIZE, PEN_SIZE);
                    drawPane.getGraphics().fillOval(e.getX(), e.getY(), PEN_SIZE, PEN_SIZE);
                }
            }
        });
        frame.setLayout(new BorderLayout());
        frame.add(drawPane, BorderLayout.CENTER);
        frame.add(results, BorderLayout.EAST);
        frame.add(new JLabel("   Hold SHIFT key to draw with trackpad or mouse, click ENTER to run digit classification."), BorderLayout.SOUTH);
        frame.pack();
        frame.setResizable(false);
        frame.addKeyListener(new KeyAdapter(){
            @Override
            public void keyPressed(KeyEvent e) {
                if (e.getKeyCode() == KeyEvent.VK_ENTER) {
                    var scaledImage = new BufferedImage(IMAGE_SIZE, IMAGE_SIZE, BufferedImage.TYPE_BYTE_GRAY);
                    scaledImage.createGraphics().drawImage(drawImage.getScaledInstance(IMAGE_SIZE, IMAGE_SIZE, Image.SCALE_SMOOTH), 0, 0, null);
                    var imageData = new float[IMAGE_SIZE * IMAGE_SIZE];
                    scaledImage.getData().getSamples(0, 0, IMAGE_SIZE, IMAGE_SIZE, 0, imageData);
                    var imageTensor = Tensor.ofShape(new long[]{1, 1, IMAGE_SIZE, IMAGE_SIZE}, imageData);
                    var result = cnn(imageTensor).data().toArray(ValueLayout.JAVA_FLOAT);
                    var report = new StringBuilder("<html>");
                    for (int i = 0; i < result.length; i++) {
                        var w = result[i];
                        report.append("&nbsp;<font size=\"%d\" color=\"#%s\">%d</font>&nbsp;(%.1f%%)&nbsp;<br><br><br>"
                                .formatted((int)(20 * w) + 3, COLORS[(int)(5.99 * w)], i, 100 * w));
                    }
                    results.setText(report.toString());
                    cleanFlag.set(true);
                }
            }
        });
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        frame.setLocationRelativeTo(null);
        frame.setVisible(true);
    }
}
