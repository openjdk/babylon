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


import java.awt.*;
import java.awt.event.*;
import java.awt.image.BufferedImage;
import java.util.concurrent.atomic.AtomicBoolean;
import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.JPanel;

import static oracle.code.onnx.mnist.MNISTModel.IMAGE_SIZE;

public class MNISTDemoUI {

    static final int DRAW_AREA_SIZE = 600;
    static final int PEN_SIZE = 20;
    static final String[] COLORS = {"1034a6", "412f88", "722b6a", "a2264b", "d3212d", "f62d2d"};

    public static void main(String[] args) throws Exception {
        var frame = new JFrame("CNN MNIST Demo - Handwritten Digit Classification");
        var drawPane = new JPanel(false);
        var resultsBoard = new JLabel();
        var cleanFlag = new AtomicBoolean(true);
        var drawImage = new BufferedImage(DRAW_AREA_SIZE, DRAW_AREA_SIZE, BufferedImage.TYPE_BYTE_GRAY);
        var mnist = new MNISTModel();

        resultsBoard.setPreferredSize(new Dimension(100, 0));
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
        frame.add(resultsBoard, BorderLayout.EAST);
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

                    var results = mnist.classify(imageData);

                    var report = new StringBuilder("<html>");
                    for (int i = 0; i < results.length; i++) {
                        var w = results[i];
                        report.append("&nbsp;<font size=\"%d\" color=\"#%s\">%d</font>&nbsp;(%.1f%%)&nbsp;<br><br><br>"
                                .formatted((int)(20 * w) + 3, COLORS[(int)(5.99 * w)], i, 100 * w));
                    }
                    resultsBoard.setText(report.toString());
                    cleanFlag.set(true);
                }
            }
        });
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        frame.setLocationRelativeTo(null);
        frame.setVisible(true);
    }
}
