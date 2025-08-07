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

import hat.tools.textmodel.JavaTextModel;
import hat.tools.textmodel.TextModel;
import hat.tools.textmodel.tokens.LineCol;
import hat.tools.textmodel.tokens.Span;

import javax.swing.JScrollPane;
import javax.swing.JTextPane;
import javax.swing.SwingUtilities;
import javax.swing.event.DocumentEvent;
import javax.swing.event.DocumentListener;
import javax.swing.text.BadLocationException;
import javax.swing.text.DefaultHighlighter;
import javax.swing.text.Element;
import javax.swing.text.Highlighter;
import javax.swing.text.Style;
import javax.swing.text.StyleConstants;
import java.awt.Color;
import java.awt.Font;
import java.awt.Graphics;
import java.awt.Graphics2D;
import java.awt.Point;
import java.awt.event.MouseEvent;
import java.awt.geom.Rectangle2D;
import java.nio.file.Files;
import java.util.ArrayList;
import java.util.List;
import java.util.TreeMap;

public abstract class TextViewer {
    static class TextViewerPane<T extends TextViewer> extends JTextPane {
        protected T viewer;
        TextViewerPane(Font font, boolean editable) {
            setFont(font);
            setEditable(editable);
        }
        void setViewer(T viewer) {
            this.viewer = viewer;
        }
    }


    public TextModel textModel;
    protected final JTextPane jTextPane;
    protected final Style defaultStyle;
    protected final Highlighter.HighlightPainter highlightPainter;
    final public JScrollPane scrollPane;
    private String text;

    public record Line(int line, int startOffset, int endOffset) implements Span {
    }

    protected List<Line> lines;
    protected TreeMap<Integer, Line> offsetToLineTreeMap;

    public Style style(String name, Color color, boolean bold, boolean italic, boolean underline) {
        var s = jTextPane.addStyle(name, null);
        StyleConstants.setForeground(s, color);
        StyleConstants.setBold(s, bold);
        StyleConstants.setItalic(s, italic);
        StyleConstants.setUnderline(s, underline);
        return s;
    }
    public abstract TextModel createTextModel(String text);

    void reparse(String msg) {
        System.out.println(msg);
        var newTextModel = textModel;
        try{
            newTextModel =createTextModel(jTextPane.getText());
            textModel = newTextModel;
            jTextPane.getStyledDocument().removeDocumentListener(documentListener);
            setText(textModel.plainText());
            jTextPane.getStyledDocument().setCharacterAttributes(0, text.length(), defaultStyle, true);
            applyStyles();
            jTextPane.getStyledDocument().addDocumentListener(documentListener);
        }catch (IllegalStateException e){
            System.out.println("Parse failed");
        }
    }

    final DocumentListener documentListener;

    public TextViewer(TextModel textModel, JTextPane jTextPane, Color highlightColor, Color defaultStyleColor) {
        this.textModel = textModel;
        this.jTextPane = jTextPane;

        this.highlightPainter = new DefaultHighlighter.DefaultHighlightPainter(highlightColor);
        this.defaultStyle = style("Default", defaultStyleColor, false, false, false);

        this.scrollPane = new JScrollPane(this.jTextPane);

        this.documentListener = jTextPane.isEditable()
                ?new DocumentListener() {
            @Override
            public void insertUpdate(DocumentEvent e) {
                SwingUtilities.invokeLater(() ->reparse("insert"));
            }

            @Override
            public void removeUpdate(DocumentEvent e) {
                SwingUtilities.invokeLater(() ->reparse("remove"));
            }

            @Override
            public void changedUpdate(DocumentEvent e) {
                SwingUtilities.invokeLater(() ->reparse("changed"));
            }
        }
                :new DocumentListener(){
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
        };
        this.jTextPane.getStyledDocument().addDocumentListener(documentListener);

    }

    public Element getElement(int offset) {
        return this.jTextPane.getStyledDocument().getCharacterElement(offset);
   }

    public Element getElementFromMouseEvent(MouseEvent e) {
        return getElement(getOffset(e));
    }
    public void scrollTo(Element funcOpElement) {
        try {
            var rectangle2D = this.jTextPane.modelToView2D(funcOpElement.getStartOffset());
            this.jTextPane.scrollRectToVisible(rectangle2D.getBounds());
        } catch (BadLocationException e) {
            throw new RuntimeException(e);
        }
    }

    public void highLightLines(LineCol first, LineCol last) {
        var highlighter = this.jTextPane.getHighlighter();
        try {
            var range = getLineRange(first, last);
            highlighter.addHighlight(range.startOffset(), range.endOffset(), highlightPainter);
        } catch (BadLocationException e) {
            throw new IllegalStateException();
        }
    }

    public int getOffset(Point p) {
        return this.jTextPane.viewToModel2D(p);
    }

    public int getOffset(MouseEvent e) {
        return getOffset(e.getPoint());
    }

    public void highLight(Element element) {
        var highlighter = this.jTextPane.getHighlighter();
        try {
            highlighter.addHighlight(element.getStartOffset(), element.getEndOffset(), highlightPainter);
        } catch (BadLocationException e) {
            throw new IllegalStateException();
        }
    }

    public int getLine(Element element) {
       var lineSpan = offsetToLineTreeMap.ceilingEntry(element.getStartOffset());
        return lineSpan.getValue().line + 1;
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
        this.jTextPane.getHighlighter().removeAllHighlights();
    }

    protected abstract void applyStyles();

    protected abstract String plainText();

    void applyHighlighting() {
        SwingUtilities.invokeLater(() -> {
            this.jTextPane.getStyledDocument().setCharacterAttributes(0, text.length(), defaultStyle, true);
            this.applyStyles();

        });
    }

    protected void setTextFromDocModel() {
        try {
            if (this.text != null && !text.equals("")) {
                this.jTextPane.getStyledDocument().remove(0, text.length());
            }
            setText(plainText());
            this.jTextPane.getStyledDocument().insertString(0, text, defaultStyle);
        } catch (BadLocationException e) {
            e.printStackTrace();
        }
    }

    public Span getLineRange(LineCol start, LineCol end) {
        Span startLine = lines.get(start.line());
        Span endLine = lines.get(end.line());
        return new Span.Impl(startLine.startOffset(), endLine.endOffset());
    }

    public int getOffset(LineCol lineCol) {
        return lines.get(lineCol.line() - 1).startOffset() + lineCol.col();
    }

    public Rectangle2D.Double getRect(Element from) {
        try {
            var fromPoint1 = this.jTextPane.modelToView2D(from.getStartOffset());
            var fromPoint2 = this.jTextPane.modelToView2D(from.getEndOffset());
            return new Rectangle2D.Double(fromPoint1.getBounds().getMinX(), fromPoint1.getMinY()
                    , fromPoint2.getBounds().getWidth(), fromPoint2.getBounds().getHeight());
        } catch (Exception e) {
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
