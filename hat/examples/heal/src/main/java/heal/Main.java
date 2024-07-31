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
package heal;

import hat.Accelerator;
import hat.backend.Backend;
import hat.buffer.S32Array2D;

import javax.imageio.ImageIO;
import javax.swing.Box;
import javax.swing.JButton;
import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.JMenuBar;
import javax.swing.JTextField;
import java.awt.Rectangle;
import java.awt.image.BufferedImage;
import java.io.IOException;
import java.lang.invoke.MethodHandles;

public class Main {

    public static void main(String[] args) throws IOException {
        Accelerator accelerator = new Accelerator(MethodHandles.lookup(), Backend.FIRST);

        var image= ImageIO.read(Main.class.getResourceAsStream("/images/bolton.png"));
        if (image.getType() != BufferedImage.TYPE_INT_RGB){//Better way?
            var rgbimage = new BufferedImage(image.getWidth(), image.getHeight(), BufferedImage.TYPE_INT_RGB);
            rgbimage.getGraphics().drawImage(image, 0, 0, null);
            image=rgbimage;
        }
        JFrame f = new JFrame("Healing Brush");
        var menuBar = new JMenuBar();
        f.setJMenuBar(menuBar);
        ((JButton) menuBar.add(new JButton("Exit"))).addActionListener(_ -> System.exit(0));
        menuBar.add(Box.createHorizontalStrut(40));
        menuBar.add(new JLabel("Search ms"));
        var searchTB = (JTextField) menuBar.add(new JTextField("",5));
        searchTB.setEditable(false);
        menuBar.add(new JLabel("Mask ms"));
        var maskTB = (JTextField) menuBar.add(new JTextField("",5));
        searchTB.setEditable(false);
        searchTB.setColumns(5);
        menuBar.add(new JLabel("Heal ms"));
        var healTB = (JTextField) menuBar.add(new JTextField("",5));
        healTB.setColumns(5);
        searchTB.setEditable(false);

        Viewer viewer = new Viewer(accelerator,image, searchTB, maskTB, healTB);

        f.setBounds(new Rectangle(image.getWidth(),image.getHeight()));
        f.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        f.setContentPane(viewer);
        f.validate();
        f.setVisible(true);
    }

}
