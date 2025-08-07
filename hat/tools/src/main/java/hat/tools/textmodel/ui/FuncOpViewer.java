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
import jdk.incubator.code.dialect.core.CoreOp;

import javax.swing.BoxLayout;
import javax.swing.JFrame;
import javax.swing.JPanel;
import javax.swing.JSplitPane;
import javax.swing.SwingUtilities;
import java.awt.BorderLayout;
import java.awt.Font;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;

public class FuncOpViewer extends JPanel {


    public FuncOpViewer(BabylonTextModel cr) {
        setLayout(new BoxLayout(this, BoxLayout.X_AXIS));
        var font = new Font("Monospaced", Font.PLAIN, 14);
        var funcOpTextModelViewer = new FuncOpTextModelViewer(cr, new FuncOpTextModelViewer.FuncOpTextPane(font, false), false);
        var javaTextModelViewer = new JavaTextModelViewer(cr.javaTextModel, new JavaTextModelViewer.JavaTextPane(font, false), false);
        var gutter = new TextGutter(  funcOpTextModelViewer,javaTextModelViewer);
        JSplitPane splitPane = new JSplitPane(JSplitPane.HORIZONTAL_SPLIT);
        splitPane.setLeftComponent(funcOpTextModelViewer.scrollPane);
        JSplitPane rightSplitPane = new JSplitPane(JSplitPane.HORIZONTAL_SPLIT);
        rightSplitPane.setLeftComponent(gutter);
        rightSplitPane.setRightComponent(javaTextModelViewer.scrollPane);
        splitPane.setRightComponent(rightSplitPane);
        add(splitPane);

        // tell each about the other
        funcOpTextModelViewer.javaTextModelViewer = javaTextModelViewer;
        javaTextModelViewer.funcOpTextModelViewer = funcOpTextModelViewer;

        // here we build the links between funcop viewer and java viewer
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

