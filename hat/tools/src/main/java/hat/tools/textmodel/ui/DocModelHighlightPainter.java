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
package hat.tools.textmodel.ui;

import javax.swing.text.DefaultHighlighter;
import javax.swing.text.JTextComponent;
import javax.swing.text.View;
import java.awt.Color;
import java.awt.Graphics;
import java.awt.Graphics2D;
import java.awt.Rectangle;
import java.awt.Shape;
import java.awt.geom.Rectangle2D;

// A private subclass of the default highlight painter
public class DocModelHighlightPainter extends DefaultHighlighter.DefaultHighlightPainter {

    public DocModelHighlightPainter(Color color) {
        super(color);
    }

    public Shape paintLayer(Graphics g, int offs0, int offs1, Shape bounds, JTextComponent c, View view) {
        g.setColor(c.getSelectionColor());
        if (bounds instanceof Rectangle r) {
            r = bounds.getBounds();
            Graphics2D g2d = (Graphics2D) g;
            //g2d.setColor(Color.YELLOW.darker());
            g2d.fill(bounds);
            g2d.setColor(Color.red);
            //  float[] dash = new float[]{2.0F, 2.0F};

            //Stroke dashedStroke = new BasicStroke(0.5F, 2, 0, 3.0F, dash, 0.0F);
            var r2d = new Rectangle2D.Float((float) r.x, (float) r.y + r.height - 3, (float) (r.width - 1), (float) (3));
            g2d.fill(r2d/*dashedStroke.createStrokedShape(r2d)*/);

            return r;
        }
        return bounds;


    }


}
