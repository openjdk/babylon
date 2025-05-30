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

import hat.ifacemapper.MappableIface;
import hat.util.ui.SevenSegmentDisplay;

import javax.swing.Box;
import javax.swing.BoxLayout;
import javax.swing.JButton;
import javax.swing.JComponent;
import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.JMenuBar;
import javax.swing.JPanel;
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
import java.awt.image.ImageObserver;

public class Viewer extends JFrame {

    public boolean isReadyForUpdate(long now) {
        incGenerations();
        return  (state.lastUIUpdateCompleted    && ((now - state.timeOfLastUIUpdate) >= state.msPerFrame));
    }

    public boolean stillRunning() {
       return  state.generations < state.maxGenerations;
    }

    public void incGenerations() {
        state.generations++;
        state.generationsSinceLastChange++;
    }

     void update(long now, @MappableIface.RO Main.CellGrid cellGrid, int from) {

        if (state.deviceOrModeModified) {
            state.generationsSinceLastChange = 0;
            state.timeOfLastChange = now;
            state.deviceOrModeModified = false;
        }
        controls.updateCounters(now);
        cellGrid.copySliceTo(mainPanel.rasterData, from);
        state.lastUIUpdateCompleted =false;
        mainPanel.repaint();
        state.timeOfLastUIUpdate = now;
    }

    public static class State {
        public final long requiredFrameRate = 5;
        public final long msPerFrame = 1000/requiredFrameRate;
        public final long maxGenerations = 1000000;
        private final Object doorBell = new Object();
        public long generations = 0;
        public volatile boolean minimizingCopies = false;
        public volatile boolean usingGPU = false;
        volatile private boolean started = false;
        public long generationsSinceLastChange = 0;
        public long timeOfLastChange = 0;
        public long timeOfLastUIUpdate;
        public volatile boolean lastUIUpdateCompleted = false;

        public volatile boolean deviceOrModeModified = false;

        State() {

        }
    }
    final Controls controls;
    final MainPanel mainPanel;
    public final State state;
    static final public class MainPanel extends JComponent implements ImageObserver {
        final double IN = 1.1;
        final double OUT = 1 / IN;
        private final BufferedImage image;
        final byte[] rasterData;
        private final double initialZoomFactor;
        private double zoomFactor;
        private double prevZoomFactor;
        private boolean zooming;
        private boolean mouseReleased;
        private double xOffset = 0;
        private double yOffset = 0;
        private Point startPoint;
        final private State state;

        record Drag(int xDiff, int yDiff) { }

        Drag drag = null;

        @Override
        public Dimension getPreferredSize() {
            return new Dimension((int) (image.getWidth() * zoomFactor), (int) (image.getHeight() * zoomFactor));
        }

        public MainPanel(BufferedImage image, State state) {
            this.state = state;
            this.image = image;

            Rectangle bounds = GraphicsEnvironment.getLocalGraphicsEnvironment().getMaximumWindowBounds();
            this.initialZoomFactor = Math.min((bounds.width - 20) / (float) image.getWidth(),
                    (bounds.height - 20) / (float) image.getHeight());
            this.rasterData = ((DataBufferByte) image.getRaster().getDataBuffer()).getData();
            this.prevZoomFactor = initialZoomFactor;
            this.zoomFactor = initialZoomFactor;
            addMouseWheelListener(e -> {
                zooming = true;
                zoomFactor = zoomFactor * ((e.getWheelRotation() < 0) ? IN : OUT);
                if (zoomFactor < initialZoomFactor) {
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
                    mouseReleased = false;
                    startPoint = MouseInfo.getPointerInfo().getLocation();
                    repaint();
                }

                @Override
                public void mouseReleased(MouseEvent e) {
                    mouseReleased = true;
                    repaint();
                }
            });
        }

        @Override
        public void paintComponent(Graphics g) {
            super.paintComponent(g);
            state.lastUIUpdateCompleted =  true;
        }

        @Override
        public void paint(Graphics g) {
            super.paint(g);
            Graphics2D g2 = (Graphics2D) g;
            AffineTransform affineTransform = new AffineTransform();
            if (zooming) {
                double xRel = MouseInfo.getPointerInfo().getLocation().getX() - getLocationOnScreen().getX();
                double yRel = MouseInfo.getPointerInfo().getLocation().getY() - getLocationOnScreen().getY();
                double zoomDiv = zoomFactor / prevZoomFactor;
                xOffset = (zoomDiv) * (xOffset) + (1 - zoomDiv) * xRel;
                yOffset = (zoomDiv) * (yOffset) + (1 - zoomDiv) * yRel;
                affineTransform.translate(xOffset, yOffset);
                prevZoomFactor = zoomFactor;
                zooming = false;
            } else if (drag != null) {
                affineTransform.translate(xOffset + drag.xDiff, yOffset + drag.yDiff);
                if (mouseReleased) {
                    xOffset += drag.xDiff;
                    yOffset += drag.yDiff;
                    drag = null;
                }
            } else {
                affineTransform.translate(xOffset, yOffset);
            }
            affineTransform.scale(zoomFactor, zoomFactor);
            g2.transform(affineTransform);
            g2.setColor(Color.DARK_GRAY);
            g2.fillRect(-image.getWidth(), -image.getHeight(), image.getWidth() * 3, image.getHeight() * 3);
            g2.drawImage(image, 0, 0, image.getWidth(), image.getHeight(), 0, 0, image.getWidth(), image.getHeight(), null);
        }

    }

    public static class Controls {
        public final JMenuBar menuBar;
        private final JButton startButton;
        private JToggleButton useGPUToggleButton;
        private JToggleButton minimizeCopiesToggleButton;
        private SevenSegmentDisplay generationsPerSecondSevenSegment;
        private final SevenSegmentDisplay generationSevenSegment;

        private State state;

        Controls( State state) {
            this.state = state;
            this.menuBar = new JMenuBar();
            JPanel panel = new JPanel();
            panel.setLayout(new BoxLayout(panel, BoxLayout.X_AXIS));

            ((JButton) panel.add(new JButton("Exit"))).addActionListener(_ -> System.exit(0));
            this.startButton = (JButton) panel.add(new JButton("Start"));

            panel.add(new JLabel("Generation"));
            this.generationSevenSegment = (SevenSegmentDisplay)
                    panel.add(new SevenSegmentDisplay(6,30,panel.getForeground(),panel.getBackground()));

            panel.add(new JLabel("Gen/Sec"));

            this.generationsPerSecondSevenSegment = (SevenSegmentDisplay)
                    panel.add(new SevenSegmentDisplay(6,30,panel.getForeground(),panel.getBackground()));
            panel.add(Box.createHorizontalStrut(400));
            menuBar.add(panel);

        }

        JToggleButton addToggle(JComponent component, String def, String alt) {
            var toggleButton = (JToggleButton) component.add(new JToggleButton(def));
            toggleButton.addChangeListener(event -> {
                if (((JToggleButton) event.getSource()).isSelected()) {
                    ((JToggleButton) event.getSource()).setText(alt);
                } else {
                    ((JToggleButton) event.getSource()).setText(def);
                }
                state.deviceOrModeModified = true;
            });
            return toggleButton;
        }

        public void updateCounters(long now) {
            generationSevenSegment.set((int)state.generationsSinceLastChange);
            long interval= (now -state.timeOfLastChange);
            if (state.generationsSinceLastChange > 0 && interval>0) { // no div/0
                int gps = (int)((1000*state.generationsSinceLastChange)/interval);
               /* System.out.println("gps "+(int)gps
                        + " interval="+interval
                        + " state.generationsSinceLastChange="+state.generationsSinceLastChange
                        + " state.timeOfLastChange="+state.timeOfLastChange);*/

                    generationsPerSecondSevenSegment.set( gps);
            }
        }
    }
    Viewer(String title, Main.CellGrid cellGrid, State state) {
        super(title);
        this.state = state;
        this.mainPanel = new MainPanel(new BufferedImage(cellGrid.width(), cellGrid.height(), BufferedImage.TYPE_BYTE_GRAY), state);
        this.controls = new Controls( state);
        setJMenuBar(this.controls.menuBar);
        controls.startButton.addActionListener(_ -> {
            state.started = true;
            synchronized (state.doorBell) {
                state.doorBell.notify();
            }
        });
        this.getContentPane().add(this.mainPanel);
        this.setLocationRelativeTo(null);
        this.pack();
        this.setVisible(true);
        this.setDefaultCloseOperation(WindowConstants.EXIT_ON_CLOSE);

        cellGrid.copySliceTo(mainPanel.rasterData, 0);  // We assume that the original data starts in the lo end of the grid
    }

    public void waitForStart() {
        while (!state.started) {
            synchronized (state.doorBell) {
                try {
                    state.doorBell.wait();
                } catch (final InterruptedException ie) {
                    ie.getStackTrace();
                }
            }
        }
    }

}
