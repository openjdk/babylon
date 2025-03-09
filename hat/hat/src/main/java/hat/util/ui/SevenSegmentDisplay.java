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
package hat.util.ui;

import javax.swing.JComponent;
import java.awt.Color;
import java.awt.Dimension;
import java.awt.Graphics;
import java.awt.Graphics2D;
import java.awt.Polygon;
import java.util.Arrays;

public class SevenSegmentDisplay extends JComponent {

    private static final boolean OFF = false;
    private static final boolean ON = true;
    record Digit(boolean[] segments) {
        /*
            <  a  >
           ^       ^
           f       b
           v       v
            <  g  >
           ^       ^
           e       c
           v       v
            <  d  >
       */
        static  Digit of(boolean a,boolean b, boolean c,boolean d,boolean e,boolean f, boolean g ){
            return new Digit(new boolean[]{a,b,c,d,e,f,g});
        }
    }
    private static final Digit blankDigit =  new Digit(new boolean[]{OFF, OFF,OFF,OFF,OFF,OFF,OFF});
    private static final Digit[] digits0to9 = new Digit[]{
            Digit.of(ON, ON, ON, ON, ON, ON, OFF),
            Digit.of(OFF, ON, ON, OFF, OFF, OFF, OFF),
            Digit.of(ON, ON, OFF, ON, ON, OFF, ON),
            Digit.of(ON, ON, ON, ON, OFF, OFF, ON),
            Digit.of(OFF, ON, ON, OFF, OFF, ON, ON),
            Digit.of(ON, OFF, ON, ON, OFF, ON, ON),
            Digit.of(ON, OFF, ON, ON, ON, ON, ON),
            Digit.of(ON, ON, ON, OFF, OFF, OFF, OFF),
            Digit.of(ON, ON, ON, ON, ON, ON, ON),
            Digit.of(ON, ON, ON, ON, OFF, ON, ON),
            Digit.of(ON, ON, ON, OFF, ON, ON, ON),//A
            Digit.of(OFF, OFF, ON, ON, ON, ON, ON)//B
    };


    static final Dimension defaultDigitSize = new Dimension(110, 180);
    private final Dimension digitSize;
    record Segment(Polygon polygon) {
        static Segment of(int[] xs, int[] ys) {
            return new Segment(new Polygon(xs, ys, xs.length));
        }
    }

    static final Segment[] segments = new Segment[]{
            Segment.of(new int[]{ 20,  90,  98,  90,  20,  12},
                    new int[]{  8,   8,  15,  22,  22,  15}),
            Segment.of(new int[]{ 91,  98, 105, 105,  98,  91},
                    new int[]{ 23,  18,  23,  81,  89,  81}),
            Segment.of(new int[]{ 91,  98, 105, 105,  98,  91},
                    new int[]{ 97,  89,  97, 154, 159, 154}),
            Segment.of(new int[]{ 20,  90,  98,  90,  20,  12},
                    new int[]{155, 155, 162, 169, 169, 162}),
            Segment.of(new int[]{  5,  12,  19,  19,  12,   5},
                    new int[]{ 97,  89,  97, 154, 159, 154}),
            Segment.of(new int[]{  5,  12,  19,  19,  12,   5},
                    new int[]{ 23,  18,  23,  81,  89,  81}),
            Segment.of(new int[]{ 20,  90,  95,  90,  20,  15},
                    new int[]{ 82,  82,  89,  96,  96,  89})
    };

    private final Digit[] digits;
    private final float digitScale;

    public SevenSegmentDisplay(int digitCount, int digitWidth) {
        digitScale = (float)digitWidth/defaultDigitSize.width;
        digitSize = new Dimension(digitWidth, (int) (defaultDigitSize.height*digitScale));
        var preferredSize = new Dimension(digitSize.width*digitCount,digitSize.height);
        setPreferredSize(preferredSize);
        setSize(preferredSize);
        setOpaque(true);
        setBackground(Color.black);
        this.digits = new Digit[digitCount];
        Arrays.fill(digits, blankDigit);
        digits[digitCount-1]=digits0to9[0];
        repaint();
    }

    public SevenSegmentDisplay(int digitCount) {
        this(digitCount, defaultDigitSize.width);
    }

    public void set(int n) {
        Arrays.fill(digits, blankDigit);
        int pos = digits.length - 1;
        if (n>0) {
            while (n > 0) {
                if (pos<0){
                    throw new IllegalArgumentException("too many digits");
                }
                digits[pos--] = digits0to9[n % 10];
                n /= 10;
            }
        }else if (n==0){
            digits[pos] = digits0to9[0];
        }
        repaint();
    }


    @Override
    public void paintComponent(Graphics g) {
        super.paintComponent(g);
        final Color off = Color.green.darker().darker().darker().darker();
        final Color on = Color.green.brighter().brighter().brighter().brighter().brighter();
        ((Graphics2D)g).scale(digitScale,digitScale);
        for (int x=0; x<digits.length; x++) {
            for (int i = 0; i < segments.length; i++) {
                g.setColor(digits[x].segments[i] ? on : off);
                g.fillPolygon(segments[i].polygon);
                g.drawPolygon(segments[i].polygon);
            }
            g.translate((int)(digitSize.width/digitScale),0);
        }
    }
}

