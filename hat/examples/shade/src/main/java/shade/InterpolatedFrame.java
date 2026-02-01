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
package shade;


import javax.swing.JFrame;
import javax.swing.JPanel;
import javax.swing.SwingUtilities;
import java.awt.Color;
import java.awt.Dimension;
import java.awt.Graphics;
import java.awt.Graphics2D;
import java.awt.RenderingHints;

public class InterpolatedFrame extends JPanel implements Runnable {
    // Window Constants
    private final int WIDTH = 800, HEIGHT = 600;

    // Logic Variables
    private boolean running = false;
    private Thread gameThread;

    // Physics Variables (Use doubles for precision)
    private double x = 100, lastX = 100;
    private double y = 100, lastY = 100;
    private double xSpeed = 200; // Pixels per second
    private float renderInterpolation = 0;

    public InterpolatedFrame() {
        this.setPreferredSize(new Dimension(WIDTH, HEIGHT));
        this.setBackground(Color.BLACK);
    }

    public void start() {
        running = true;
        gameThread = new Thread(this);
        gameThread.start();
    }

    @Override
    public void run() {
        double nsPerTick = 1000000000.0 / 60.0; // 60 Fixed Updates per second
        double delta = 0;
        long lastTime = System.nanoTime();

        while (running) {
            long now = System.nanoTime();
            delta += (now - lastTime) / nsPerTick;
            lastTime = now;

            // Fixed Update Loop
            while (delta >= 1) {
                updatePhysics();
                delta--;
            }

            // Calculate Interpolation for rendering
            renderInterpolation = (float) delta;

            // Schedule Render on EDT
            SwingUtilities.invokeLater(this::repaint);

            // Cap the loop to save CPU
            try { Thread.sleep(2); } catch (InterruptedException e) {}
        }
    }

    private void updatePhysics() {
        // Store previous state for interpolation
        lastX = x;
        lastY = y;

        // Move (at 60fps, 1 tick = 1/60th of a second)
        x += xSpeed / 60.0;

        // Wall Bounce
        if (x > WIDTH - 30 || x < 0) xSpeed *= -1;
    }

  //  @Override
    protected void paintComponent(Graphics g) {
        super.paintComponent(g);
        Graphics2D g2d = (Graphics2D) g;
        g2d.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);

        // Interpolated Position: (Current - Last) * Alpha + Last
        int drawX = (int) ((x - lastX) * renderInterpolation + lastX);
        int drawY = (int) ((y - lastY) * renderInterpolation + lastY);

        g2d.setColor(Color.CYAN);
        g2d.fillOval(drawX, drawY, 30, 30);
    }

    public static void main(String[] args) {
        JFrame frame = new JFrame("InterpolatedFrame");
        InterpolatedFrame panel = new InterpolatedFrame();
        frame.add(panel);
        frame.pack();
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        frame.setVisible(true);
        panel.start();
    }
}