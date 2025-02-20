/*
 * Copyright (c) 2024, Oracle and/or its affiliates. All rights reserved.
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
package life;

import javax.swing.JButton;
import javax.swing.JComboBox;
import javax.swing.JComponent;
import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.JMenuBar;
import javax.swing.JTextField;
import javax.swing.JToggleButton;
import javax.swing.WindowConstants;
import java.awt.Color;
import java.awt.Dimension;
import java.awt.Graphics;
import java.awt.Graphics2D;
import java.awt.GraphicsEnvironment;
import java.awt.MouseInfo;
import java.awt.Point;
import java.awt.Rectangle;
import java.awt.event.MouseAdapter;
import java.awt.event.MouseEvent;
import java.awt.event.MouseMotionAdapter;
import java.awt.geom.AffineTransform;
import java.awt.image.BufferedImage;
import java.awt.image.DataBufferByte;

public class Viewer extends JFrame {

    private final Object doorBell = new Object();
    final Controls controls;
    final MainPanel mainPanel;
    volatile private boolean started=false;

    static final public class MainPanel extends JComponent {
        enum State {Scheduled, Done};
        public  volatile State state = State.Done;

        final double IN = 1.1;
        final double OUT = 1/IN;
        private final BufferedImage image;
        final byte[] rasterData;
        private final double initialZoomFactor;
        private double zoomFactor;
        private double prevZoomFactor;
        private boolean zooming;
        private boolean released;
        private double xOffset = 0;
        private double yOffset = 0;
        private Point startPoint;


        class Drag{
            public int xDiff;
            public int yDiff;
            Drag(int xDiff, int yDiff) {
                this.xDiff = xDiff;
                this.yDiff = yDiff;
            }
        }
        Drag drag = null;

        @Override
        public Dimension getPreferredSize() {
            return new Dimension((int)(image.getWidth()*zoomFactor), (int)(image.getHeight()*zoomFactor));
        }
        public MainPanel(BufferedImage image) {
            this.image = image;
            Rectangle bounds = GraphicsEnvironment.getLocalGraphicsEnvironment().getMaximumWindowBounds();
            this.initialZoomFactor = Math.min((bounds.width-20)/(float)image.getWidth(),
                    (bounds.height-20)/(float)image.getHeight());
            this.rasterData = ((DataBufferByte) image.getRaster().getDataBuffer()).getData();
            this.prevZoomFactor =initialZoomFactor;
            this.zoomFactor = initialZoomFactor;
            addMouseWheelListener(e -> {
                zooming = true;
                zoomFactor = zoomFactor * ((e.getWheelRotation() < 0)?IN:OUT);
                if (zoomFactor < initialZoomFactor ){
                    zoomFactor = initialZoomFactor;
                    prevZoomFactor = zoomFactor;
                }
                repaint();
            });
            addMouseMotionListener(new MouseMotionAdapter() {
                @Override
                public void mouseDragged(MouseEvent e) {
                    Point curPoint = e.getLocationOnScreen();
                    drag = new Drag(curPoint.x - startPoint.x, curPoint.y - startPoint.y);
                    repaint();
                }
            });
            addMouseListener(new MouseAdapter() {
                @Override
                public void mousePressed(MouseEvent e) {
                    released = false;
                    startPoint = MouseInfo.getPointerInfo().getLocation();
                }

                @Override
                public void mouseReleased(MouseEvent e) {
                    released = true;
                    repaint();
                }
            });
        }

        @Override
        public void paint(Graphics g) {
            super.paint(g);
            Graphics2D g2 = (Graphics2D) g;
            AffineTransform at = new AffineTransform();
            if (zooming) {
                double xRel = MouseInfo.getPointerInfo().getLocation().getX() - getLocationOnScreen().getX();
                double yRel = MouseInfo.getPointerInfo().getLocation().getY() - getLocationOnScreen().getY();
                double zoomDiv = zoomFactor / prevZoomFactor;
                xOffset = (zoomDiv) * (xOffset) + (1 - zoomDiv) * xRel;
                yOffset = (zoomDiv) * (yOffset) + (1 - zoomDiv) * yRel;
                at.translate(xOffset, yOffset);
                prevZoomFactor = zoomFactor;
                zooming = false;
            } else if (drag!= null) {
                at.translate(xOffset +drag.xDiff, yOffset + drag.yDiff);
                if (released) {
                    xOffset += drag.xDiff;
                    yOffset += drag.yDiff;
                    drag = null;
                }
            } else{
                at.translate(xOffset, yOffset);
            }
            at.scale(zoomFactor, zoomFactor);
            g2.transform(at);
            g2.setColor(Color.BLACK);
            g2.fillRect(0-5000, 0-5000, image.getWidth()+10000, image.getHeight()+10000);
            g2.drawImage(image, 0,0, image.getWidth(), image.getHeight(), 0, 0, image.getWidth(), image.getHeight(), this);
            state = State.Done;
        }
    }
    public static class Controls{
        private JTextField generationTextField;
        private  JTextField generationsPerSecondTextField;
        private  JButton startButton;
        private  JToggleButton useGPUToggleButton;
        private  JToggleButton alwaysCopyToggleButton;
        private  JComboBox<String> generationsPerFrameComboBox;
        public volatile boolean updated = false;
        Controls(JMenuBar menuBar){
            ((JButton) menuBar.add(new JButton("Exit"))).addActionListener(_ -> System.exit(0));
            this.startButton = (JButton) menuBar.add(new JButton("Start"));
            this.useGPUToggleButton =addToggle(menuBar, "Java", "GPU");
            this.alwaysCopyToggleButton = addToggle(menuBar,"Minimize Moves","Always Copy");
         //   this.generationsPerFrameComboBox = (JComboBox<String>) menuBar.add(new JComboBox<String>(
           //         new String[]{"1", "10", "20"})
           // );
            generationTextField = addLabelledTextField(menuBar,"Gen");
            generationsPerSecondTextField = addLabelledTextField(menuBar,"Gen/Sec");
        }

        JToggleButton addToggle(JMenuBar menuBar,String def, String alt) {
            var toggleButton = (JToggleButton) menuBar.add(new JToggleButton(def));
            toggleButton.addChangeListener(event -> {
                if (((JToggleButton)event.getSource()).isSelected()){
                    ((JToggleButton)event.getSource()).setText(alt);
                } else {
                    ((JToggleButton)event.getSource()).setText(def);
                }
                updated = true;
            });
            return toggleButton;
        }

        JTextField addLabelledTextField(JMenuBar menuBar, String name){
            menuBar.add(new JLabel(name));
            JTextField textField = (JTextField) menuBar.add(new JTextField("",5));
            textField.setEditable(false);
            menuBar.add(textField);
            return textField;
        }

        public boolean alwaysCopy() {
            return alwaysCopyToggleButton.isSelected();
        }

        public boolean useGPU() {
            return useGPUToggleButton.isSelected();
        }

        public void updateGenerationCounter(long generationCounter, long frameCounter, long msPerFrame) {
            generationTextField.setText(String.format("%8d", generationCounter));
            if (generationCounter>0 && frameCounter>0) {
                generationsPerSecondTextField.setText(
                        String.format("%5.2f", (generationCounter * 1000f) / (frameCounter * msPerFrame))
                );
            }else{
                generationsPerSecondTextField.setText("...");
            }
        }
    }

    Viewer(String title, Main.CLWrapCellGrid CLWrapCellGrid) {
        super(title);
        this.mainPanel = new MainPanel(new BufferedImage(CLWrapCellGrid.width(), CLWrapCellGrid.height(), BufferedImage.TYPE_BYTE_GRAY));
        JMenuBar menuBar = new JMenuBar();
        this.controls = new Controls(menuBar);
        setJMenuBar(menuBar);
        controls.startButton.addActionListener(_ -> {started=true;synchronized (doorBell) {doorBell.notify();}});
        this.getContentPane().add(this.mainPanel);
        this.setLocationRelativeTo(null);
        this.pack();
        this.setVisible(true);
        this.setDefaultCloseOperation(WindowConstants.EXIT_ON_CLOSE);
    }

    public void waitForStart() {
        while (!started) {
            synchronized (doorBell) {
                try {
                    doorBell.wait();
                } catch (final InterruptedException ie) {
                    ie.getStackTrace();
                }
            }
        }
    }

}
