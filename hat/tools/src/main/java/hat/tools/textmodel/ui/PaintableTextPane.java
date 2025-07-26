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

import javax.swing.JTextPane;
import javax.swing.text.BadLocationException;
import javax.swing.text.Caret;
import javax.swing.text.DefaultCaret;
import javax.swing.text.JTextComponent;
import java.awt.Color;
import java.awt.Cursor;
import java.awt.FontMetrics;
import java.awt.Graphics;
import java.awt.Graphics2D;
import java.awt.Rectangle;
import java.awt.geom.Rectangle2D;

public class PaintableTextPane extends JTextPane {
    public class MyCaret extends DefaultCaret {

        private String mark = "<";

        public MyCaret() {
            setBlinkRate(500);
        }

        @Override
        protected synchronized void damage(Rectangle r) {
            if (r == null) {
                return;
            }

            JTextComponent comp = getComponent();
            FontMetrics fm = comp.getFontMetrics(comp.getFont());
            int textWidth = fm.stringWidth(">");
            int textHeight = fm.getHeight();
            x = r.x;
            y = r.y;
            width = textWidth;
            height = textHeight;
            repaint(); // calls getComponent().repaint(x, y, width, height)
        }

        @Override
        public void paint(Graphics g) {
            JTextComponent comp = getComponent();
            if (comp == null) {
                return;
            }

            int dot = getCaretPosition();
            System.out.println("dot = "+dot);
            Rectangle2D r = null;
            try {
                r = comp.modelToView2D(dot);
            } catch (BadLocationException e) {
                return;
            }
            if (r == null) {
                return;
            }

            if ((x != (int)r.getX()) || (y != (int)r.getY())) {
                repaint(); // erase previous location of caret
                damage((Rectangle) r);
            }

            if (isVisible()) {
                FontMetrics fm = comp.getFontMetrics(comp.getFont());
                int textWidth = fm.stringWidth(">");
                int textHeight = fm.getHeight();

                g.setColor(Color.ORANGE);//comp.getCaretColor());
                g.drawString(mark, x, y + fm.getAscent());
            }
        }

    }


    public void paintComponent(Graphics g) {
       // this.setCursor(Cursor.getPredefinedCursor(Cursor.TEXT_CURSOR));
      //  this.setCaret(new MyCaret());
        super.paintComponent(g);
        Graphics2D g2d = (Graphics2D) g;

        // Draw Text
        //  g2d.drawString("This is my custom Panel!",10,20);
    }
}
