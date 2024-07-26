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

import javax.swing.JFrame;
import java.awt.Rectangle;
import java.lang.invoke.MethodHandles;

public class Heal {
    public static void main(String[] args) {
        Accelerator accelerator = new Accelerator(MethodHandles.lookup(), Backend.FIRST);
        ImageData imageData = ImageData.of(
                Viewer.class.getResourceAsStream("/images/bolton.png")
        );
        JFrame f = new JFrame("Healing Brush");
        f.setBounds(new Rectangle(imageData.width(), imageData.height()));
        f.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        Viewer viewerDisplay = new Viewer(imageData, accelerator);
        f.setContentPane(viewerDisplay);
        f.validate();
        f.setVisible(true);
    }

}
