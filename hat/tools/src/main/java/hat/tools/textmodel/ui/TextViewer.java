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
import java.awt.Point;
import java.awt.event.MouseEvent;
import java.awt.geom.Rectangle2D;
import java.util.ArrayList;
import java.util.List;
import java.util.TreeMap;

public abstract class TextViewer {
    public final JTextPane jtextPane;
    final public JScrollPane scrollPane;
    protected Highlighter.HighlightPainter highlightPainter;
    private String text;
    public record Line(int line, int startOffset, int endOffset) implements Span {}
    protected List<Line> lines;
    protected TreeMap<Integer,Line> offsetToLineTreeMap;
    protected Style defaultStyle;

    public TextViewer(JTextPane jtextPane) {
        this.jtextPane = jtextPane;
        this.scrollPane = new JScrollPane(  this.jtextPane);

        this.jtextPane.getStyledDocument().addDocumentListener(new DocumentListener() {
            @Override
            public void insertUpdate(DocumentEvent e) {
                SwingUtilities.invokeLater(() -> applyHighlighting());
            }

            @Override
            public void removeUpdate(DocumentEvent e) {
      //          SwingUtilities.invokeLater(() -> applyHighlighting());
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
    public int getOffset(Point p) {
        return jtextPane.viewToModel2D(p);
    }
    public int getOffset(MouseEvent e) {
        return getOffset(e.getPoint());
    }

    public void highLight(Element element) {
        var highlighter =  jtextPane.getHighlighter();
        try {
            highlighter.addHighlight(element.getStartOffset(),element.getEndOffset(),  highlightPainter);
        } catch (BadLocationException e) {
            throw new IllegalStateException();
        }
    }

    public int getLine(Element element) {
       var lineSpan = offsetToLineTreeMap.ceilingEntry(element.getStartOffset());
       return lineSpan.getValue().line+1;
    }

    String setText(String text) {
        this.text = text;
        String[] linesOfText = text.split("\n");
        lines = new ArrayList<>();
        offsetToLineTreeMap = new TreeMap<>();
        int accumOffset = 0;
        for (int currentLine = 0; currentLine < linesOfText.length; currentLine++) {
            Line line = new Line(lines.size(), accumOffset, accumOffset + linesOfText[currentLine].length() + 1);// +1 for newline
            lines.add(line);
            accumOffset = line.endOffset();
            offsetToLineTreeMap.put(accumOffset, line);
        }
        return text;
    }


    public void removeHighlights() {
        jtextPane.getHighlighter().removeAllHighlights();
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

    public Rectangle2D.Double getRect(Element from) {
        try {
            var fromPoint1 = jtextPane.modelToView2D(from.getStartOffset());
            var fromPoint2 = jtextPane.modelToView2D(from.getEndOffset());
            return new Rectangle2D.Double(fromPoint1.getBounds().getMinX(), fromPoint1.getMinY()
                    , fromPoint2.getBounds().getWidth(), fromPoint2.getBounds().getHeight());
        }catch (Exception e){
            return null;
        }
    }

    public void highlight(ElementSpan fromElementSpan, List<ElementSpan> toElementSpans) {
        highLight(fromElementSpan.element());
        toElementSpans.forEach(targetElementSpan -> {
            var targetTextViewer = targetElementSpan.textViewer();
            var targetElement = targetElementSpan.element();
            targetTextViewer.highLight(targetElement);
            targetTextViewer.scrollTo(targetElement);
        });
    }
}
