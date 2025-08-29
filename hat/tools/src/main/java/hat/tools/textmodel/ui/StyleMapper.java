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

import com.sun.source.doctree.TextTree;
import hat.tools.textmodel.BabylonTextModel;
import hat.tools.textmodel.JavaTextModel;
import hat.tools.textmodel.TextModel;
import hat.tools.textmodel.tokens.Arrow;
import hat.tools.textmodel.tokens.At;
import hat.tools.textmodel.tokens.Ch;
import hat.tools.textmodel.tokens.Comment;
import hat.tools.textmodel.tokens.DottedName;
import hat.tools.textmodel.tokens.FloatConst;
import hat.tools.textmodel.tokens.IntConst;
import hat.tools.textmodel.tokens.Nl;
import hat.tools.textmodel.tokens.ReservedWord;
import hat.tools.textmodel.tokens.Seq;
import hat.tools.textmodel.tokens.StringLiteral;
import hat.tools.textmodel.tokens.Ws;

import javax.swing.JTextPane;
import javax.swing.text.DefaultHighlighter;
import javax.swing.text.Highlighter;
import javax.swing.text.Style;
import javax.swing.text.StyleConstants;
import java.awt.Color;

public abstract class StyleMapper {
    JTextPane jTextPane;
    protected final Style defaultStyle;
    protected final Highlighter.HighlightPainter highlightPainter;
    public Style style(String name, Color color, boolean bold, boolean italic, boolean underline) {
        var s = jTextPane.addStyle(name, null);
        StyleConstants.setForeground(s, color);
        StyleConstants.setBold(s, bold);
        StyleConstants.setItalic(s, italic);
        StyleConstants.setUnderline(s, underline);
        return s;
    }
    StyleMapper( JTextPane jtextPane, Color highlightPainterColor, Color defaultStyleColor, Color backGroundColor) {
        this.jTextPane = jtextPane;
        this.highlightPainter =   new DefaultHighlighter.DefaultHighlightPainter(highlightPainterColor);
        this.defaultStyle = style("Default", defaultStyleColor, false, false, false);
        this.jTextPane.setBackground(backGroundColor);
    }

    public abstract void applyStyles(TextModel textModel);

}
