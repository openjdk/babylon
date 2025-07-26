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

import hat.tools.textmodel.tokens.LineCol;
import hat.tools.textmodel.tokens.Span;

import javax.swing.JScrollPane;
import javax.swing.JTextPane;
import javax.swing.SwingUtilities;
import javax.swing.event.DocumentEvent;
import javax.swing.event.DocumentListener;
import javax.swing.text.BadLocationException;
import javax.swing.text.Element;
import javax.swing.text.Highlighter;
import javax.swing.text.Style;
import java.awt.event.MouseEvent;
import java.util.ArrayList;
import java.util.List;

public abstract class TextViewer {
    public final JTextPane jtextPane;
    //final protected TextLineNumber textLineNumber;
    final public JScrollPane scrollPane;
    protected Highlighter.HighlightPainter highlightPainter;
    private String text;
    protected List<Span> lines;
    protected Style defaultStyle;

    public TextViewer(JTextPane jtextPane) {
        this.jtextPane = jtextPane;
        this.scrollPane = new JScrollPane(  this.jtextPane);
      //  this.textLineNumber = new TextLineNumber(  this.jtextPane);
        //this.scrollPane.setRowHeaderView(textLineNumber);

        this.jtextPane.getStyledDocument().addDocumentListener(new DocumentListener() {
            @Override
            public void insertUpdate(DocumentEvent e) {
                SwingUtilities.invokeLater(() -> applyHighlighting());
            }

            @Override
            public void removeUpdate(DocumentEvent e) {
                SwingUtilities.invokeLater(() -> applyHighlighting());
            }

            @Override
            public void changedUpdate(DocumentEvent e) {
                // Plain text attributes changed, not relevant for this highlighter
            }
        });


    }



    public Element getElement(int offset) {
       return jtextPane.getStyledDocument().getCharacterElement(offset);
   }

    public Element getElementFromMouseEvent(MouseEvent e) {
        return getElement(getOffset(e));
    }
    public void scrollTo(Element funcOpElement) {
        try {
            var rectangle2D = jtextPane.modelToView2D(funcOpElement.getStartOffset());
            jtextPane.scrollRectToVisible(rectangle2D.getBounds());
        } catch (BadLocationException e) {
            throw new RuntimeException(e);
        }
    }


    public void highLightLines(LineCol first, LineCol last) {
        var highlighter = jtextPane.getHighlighter();
        try {
            var range = getLineRange( first, last);
            highlighter.addHighlight(range.startOffset(), range.endOffset(), highlightPainter);
        } catch (BadLocationException e) {
            throw new IllegalStateException();
        }
    }
    public int getOffset(MouseEvent e) {
        return jtextPane.viewToModel2D(e.getPoint());
    }
    public void highLight(int from, int to) {
        var highlighter =  jtextPane.getHighlighter();
        try {
            highlighter.addHighlight(from,to, highlightPainter);
        } catch (BadLocationException e) {
            throw new IllegalStateException();
        }
    }
    public void highLight(Element element) {
        var highlighter =  jtextPane.getHighlighter();
        try {
            highlighter.addHighlight(element.getStartOffset(),element.getEndOffset(),  highlightPainter);
        } catch (BadLocationException e) {
            throw new IllegalStateException();
        }
    }

    String setText(String text) {
        this.text = text;
        String[] linesOfText = text.split("\n");
        lines = new ArrayList<>();
        int accumOffset = 0;
        for (int currentLine = 0; currentLine < linesOfText.length; currentLine++) {
            Span line = new Span.Impl(accumOffset, accumOffset + linesOfText[currentLine].length() + 1);// +1 for newline
            lines.add(line);
            accumOffset = line.endOffset();
        }
        return text;
    }


    public void removeHighlights() {
        var highlighter = jtextPane.getHighlighter();
        highlighter.removeAllHighlights();
        //for (var highlight : highlighter.getHighlights()) {
          //      highlighter.removeHighlight(highlight);
       // }
    }


    protected abstract void applyStyles();

    protected abstract String plainText();


    void applyHighlighting() {
        SwingUtilities.invokeLater(() -> {
            this.jtextPane.getStyledDocument().setCharacterAttributes(0, text.length(),  defaultStyle, true);
            applyStyles();
        });
    }

    protected void setTextFromDocModel() {
        try {
            if (this.text != null && !text.equals("")) {
                this.jtextPane.getStyledDocument().remove(0, text.length());
            }
            setText(plainText());
            this.jtextPane.getStyledDocument().insertString(0, text, defaultStyle);
        } catch (BadLocationException e) {
            e.printStackTrace();
        }
    }

    public Span getLineRange(LineCol start, LineCol end) {
            Span startLine = lines.get(start.line());
            Span endLine = lines.get(end.line());
            return new Span.Impl(startLine.startOffset(),endLine.endOffset());
    }

    public int getOffset(LineCol lineCol) {
        return lines.get(lineCol.line()-1).startOffset()+ lineCol.col();
    }

}
