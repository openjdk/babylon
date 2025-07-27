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

import hat.tools.textmodel.BabylonTextModel;
import hat.tools.textmodel.TextModel;
import jdk.incubator.code.dialect.core.CoreOp;

import javax.swing.BoxLayout;
import javax.swing.JFrame;
import javax.swing.JPanel;
import javax.swing.JTextPane;
import javax.swing.SwingUtilities;
import javax.swing.text.Element;
import java.awt.BorderLayout;
import java.awt.Font;
import java.awt.Graphics;
import java.awt.Graphics2D;
import java.awt.event.MouseAdapter;
import java.awt.event.MouseEvent;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class FuncOpViewer extends JPanel {


    public static class FuncOpTextModelViewer extends AbstractTextModelViewer {
        JavaTextModelViewer javaTextModelViewer;
        Map<ElementSpan, List<ElementSpan>> ssaIdFromToMap = new HashMap<>();
        Map<ElementSpan, List<ElementSpan>> ssaIdToFromMap = new HashMap<>();
        Map<Integer, ElementSpan> ssaIdToElement = new HashMap<>();
        Map<ElementSpan, List<ElementSpan>> opToJava = new HashMap<>();
        static class FuncOpTextPane extends JTextPane {
            public void paintComponent(Graphics g) {
                super.paintComponent(g);
                Graphics2D g2d = (Graphics2D) g;
                // So we can overlay with arrows for
                // Draw Text
                //  g2d.drawString("This is my custom Panel!",10,20);
            }
            FuncOpTextPane(Font font) {
                super.setFont(font);
            }
        };
        FuncOpTextModelViewer(TextModel textModel, Font font, boolean dark) {
            super(textModel, new FuncOpTextPane(font), font, dark);
            jtextPane.addMouseListener(new MouseAdapter() {
                @Override
                public void mouseClicked(MouseEvent e) {
                    var clicked = getElementFromMouseEvent(e);
                    removeHighlights();
                    javaTextModelViewer.removeHighlights();
                    if (clicked != null) {
                        if (opToJava.keySet().stream().anyMatch(fromElementSpan -> fromElementSpan.includes(clicked.getStartOffset()))) {
                            opToJava.keySet().stream().
                                    filter(fromElementSpan -> fromElementSpan.includes(clicked.getStartOffset()))
                                    .forEach(fromElementSpan -> {
                                        fromElementSpan.textViewer().highLight(fromElementSpan.element());
                                        opToJava.get(fromElementSpan).forEach(targetElementSpan -> {
                                            Element targetElement = targetElementSpan.element();
                                            targetElementSpan.textViewer().highLight(targetElement);
                                            targetElementSpan.textViewer().scrollTo(targetElement);
                                        });
                                    });
                        } else {
                            System.out.println("not a locationmapping  from op");
                        }
                    } else {
                        System.out.println("nothing from op");
                    }
                }
            });


        }
    }

    public static class JavaTextModelViewer extends AbstractTextModelViewer {
        FuncOpTextModelViewer funcOpTextModelViewer;
        Map<ElementSpan, List<ElementSpan>> javaToOp = new HashMap<>();
        static class JavaTextPane extends JTextPane {
            public void paintComponent(Graphics g) {
                super.paintComponent(g);
                Graphics2D g2d = (Graphics2D) g;
            }
            JavaTextPane(Font font) {
                super.setFont(font);
            }
        };
        JavaTextModelViewer(TextModel textModel, Font font, boolean dark) {
            super(textModel, new JavaTextPane(font), font, dark);
            jtextPane.addMouseListener(new MouseAdapter() {
                @Override
                public void mouseClicked(MouseEvent e) {
                    var clicked = getElementFromMouseEvent(e);
                    funcOpTextModelViewer.removeHighlights();
                    removeHighlights();
                    if (clicked != null) {
                        if (javaToOp.keySet().stream().anyMatch(fromElementSpan -> fromElementSpan.includes(clicked.getStartOffset()))) {
                            javaToOp.keySet().stream().
                                    filter(fromElementSpan -> fromElementSpan.includes(clicked.getStartOffset()))
                                    .forEach(fromElementSpan -> {
                                        fromElementSpan.textViewer().highLight(fromElementSpan.element());
                                        javaToOp.get(fromElementSpan).forEach(targetElementSpan -> {
                                            Element targetElement = targetElementSpan.element();
                                            targetElementSpan.textViewer().highLight(targetElement);
                                            targetElementSpan.textViewer().scrollTo(targetElement);
                                        });
                                    });
                        } else {
                            System.out.println("not a mappable java line  from op");
                        }
                    } else {
                        System.out.println("nothing from java");
                    }
                }
            });
        }

    }

    public FuncOpViewer(BabylonTextModel cr) {
        setLayout(new BoxLayout(this, BoxLayout.X_AXIS));
        var font = new Font("Monospaced", Font.PLAIN, 14);

        var funcOpTextModelViewer = new FuncOpTextModelViewer(cr, font, false);
        var javaTextModelViewer = new JavaTextModelViewer(cr.javaTextModel, font, false);

        var gutter = new TextGutter(  funcOpTextModelViewer,javaTextModelViewer);
        add(funcOpTextModelViewer.scrollPane);
        add(gutter);
        add(javaTextModelViewer.scrollPane);
        // tell each about the other
        funcOpTextModelViewer.javaTextModelViewer = javaTextModelViewer;
        javaTextModelViewer.funcOpTextModelViewer = funcOpTextModelViewer;
        cr.find(true, t -> t instanceof BabylonTextModel.BabylonSSARef, t -> {
            var ssaRef = (BabylonTextModel.BabylonSSARef) t;
            ElementSpan babylonSSARefElement = new ElementSpan.Impl(ssaRef, funcOpTextModelViewer, funcOpTextModelViewer.getElement(t.startOffset()));
            funcOpTextModelViewer.ssaIdToElement.put(ssaRef.id, babylonSSARefElement);
            funcOpTextModelViewer.ssaIdToFromMap.computeIfAbsent(babylonSSARefElement, _ -> new ArrayList<>());
            funcOpTextModelViewer.ssaIdFromToMap.computeIfAbsent(babylonSSARefElement, _ -> new ArrayList<>());
        });
        cr.ssaEdgeList.stream().forEach(edge -> {
            var ssaRef = edge.ssaRef();
            var ssaDef = edge.ssaDef();
            var ssaDefElement = funcOpTextModelViewer.getElement(ssaDef.startOffset());
            var ssaRefElement = funcOpTextModelViewer.getElement(ssaRef.endOffset());

        });

        cr.babylonLocationAttributes.forEach(babylonLocationAttribute->{
            ElementSpan babylonLocationAttributeElement = new ElementSpan.Impl(
                    babylonLocationAttribute, funcOpTextModelViewer, funcOpTextModelViewer.getElement(babylonLocationAttribute.startOffset()));
            var javaPaneOffset = javaTextModelViewer.getOffset(babylonLocationAttribute);
            var javaPaneElement = javaTextModelViewer.getElement(javaPaneOffset);
            var javaSourceElementSpan = new ElementSpan.Impl(babylonLocationAttribute, javaTextModelViewer, javaPaneElement);
            funcOpTextModelViewer.opToJava.computeIfAbsent(babylonLocationAttributeElement, _ -> new ArrayList<>()).add(javaSourceElementSpan);
            javaTextModelViewer.javaToOp.computeIfAbsent(javaSourceElementSpan, _ -> new ArrayList<>()).add(babylonLocationAttributeElement);
        });

        javaTextModelViewer.highLightLines(cr.babylonLocationAttributes.getFirst(), cr.babylonLocationAttributes.getLast());
    }

    public static void launch(BabylonTextModel crDoc) {
            SwingUtilities.invokeLater(() -> {
                var viewer = new FuncOpViewer(crDoc);
                var frame = new JFrame();
                frame.setLayout(new BorderLayout());
                frame.getContentPane().add(viewer);
                frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
                frame.pack();
                frame.setVisible(true);
            });

    }

    public static void launch(CoreOp.FuncOp javaFunc) {
        BabylonTextModel crDoc = BabylonTextModel.of(javaFunc);
        launch(crDoc);
    }
    public static void launch(Path path) throws IOException {
        BabylonTextModel crDoc = BabylonTextModel.of(Files.readString(path));
        launch(crDoc);
    }

    public static void main(String[] args) throws IOException {
        FuncOpViewer.launch(Path.of(args[0]));
    }
}

