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

import hat.tools.jdot.DotBuilder;
import hat.tools.jdot.ui.JDot;
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
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class SSAIDViewer extends JPanel {

    public SSAIDViewer(BabylonTextModel cr) {
        setLayout(new BoxLayout(this, BoxLayout.X_AXIS));
        var font = new Font("Monospaced", Font.PLAIN, 14);
        var styleMapper = new BabylonStyleMapper(  new FuncOpTextModelViewer.FuncOpTextPane(font,false), false);
        var funcOpTextModelViewer = new FuncOpTextModelViewer(cr,  styleMapper);

        var dotViewer = new JDot(DotBuilder.dotDigraph("name", g -> {
            g.nodeShape("record");
            cr.ssaEdgeList.forEach(edge -> {
                var ssaDef = edge.ssaDef();
                String def = "%" + ssaDef.id;
                g.record(def, def);
            });
            cr.ssaEdgeList.forEach(edge -> {
                var ssaDef = edge.ssaDef();
                String def = "%" + ssaDef.id;
                int line = ssaDef.pos().line();
                cr.ssaEdgeList.forEach(e -> {
                    var ssaRef = e.ssaRef();
                    if (ssaRef.pos().line() == line) {
                        String ref = "%" + ssaRef.id;
                        g.edge(def, ref);
                    }
                });
            });
        }));
        JSplitPane splitPane = new JSplitPane(JSplitPane.HORIZONTAL_SPLIT);
        splitPane.setLeftComponent(funcOpTextModelViewer.scrollPane);
        splitPane.setRightComponent(dotViewer.pane);
        add(splitPane);
    }

    public static void launch(BabylonTextModel crDoc) {
        SwingUtilities.invokeLater(() -> {
            var viewer = new SSAIDViewer(crDoc);
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
        SSAIDViewer.launch(Path.of(args[0]));
    }
}

