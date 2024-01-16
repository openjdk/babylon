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

package jdk.code.tools.dot;

import jdk.code.tools.renderer.ProcessRunner;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.io.StringWriter;
import java.util.ArrayList;
import java.util.List;

/**
 * Created by gfrost
 * http://graphviz.org/Documentation.php
 * http://graphviz.org/content/arrow-shapes
 * http://graphviz.org/content/rootNode-shapes
 * http://graphviz.org/content/attrs
 * http://www.graphviz.org/Documentation/dotguide.pdf
 */
public final class DotViewer {
    private DotViewer() {
    }

    // Look at Path.of() ..... ;)  I think I may have reimplemented it
    static String[] svgViewers = new String[]{
            System.getenv("SVG_VIEWER_PATH"),
            System.getProperty("SVG_VIEWER_PATH"),
            "/usr/bin/google-chrome",
            "/snap/chromium/1466/usr/lib/chromium-browser/chrome",
            "/usr/bin/gpicview",
            "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
            "/usr/bin/xdg-open"
    };
    static String[] dotLocations = new String[]{
            System.getenv("DOT_PATH"),
            System.getProperty("DOT_PATH"),
            "/usr/bin/dot",
            "/usr/local/bin/dot" // mac
    };

    static String getLocation(String[] possibles) {
        for (String s : possibles) {
            if (s != null && !s.equals("")) {
                File file = new File(s);
                if (file.exists() && file.canExecute()) {
                    return s;
                }
            }
        }
        return null;
    }

    public static void view(String dotSource) {

        String dot = getLocation(dotLocations);
        System.out.println(dotSource);
        String svgViewer = getLocation(svgViewers);

        if (dot != null && svgViewer != null) {
            try {
                File tempFile = File.createTempFile("ast", ".svg");
                ProcessRunner.run(dot)
                        .opt("-Tsvg")
                        .opt("-o").file(tempFile)
                        .temp("ast", "dot", dotSource)
                        .go(false);
                ProcessRunner.run(svgViewer)
                        .file(tempFile)
                        .go(false);

            } catch (IOException ioe) {
                ioe.printStackTrace();
            }
        }

    }

    public static void viewNoWait(String dotSource) {
        String dot = getLocation(dotLocations);
        String svgViewer = getLocation(svgViewers);

        if (dot != null && svgViewer != null) {
            try {
                File tempDotFile = File.createTempFile("ast", "dot");
                FileWriter tempDotFileWriter = new FileWriter(tempDotFile);
                tempDotFileWriter.append(dotSource);
                tempDotFileWriter.close();
                File tempPngFile = File.createTempFile("ast", "svg");
                List<String> dotCommand = new ArrayList<>();
                dotCommand.add(dot);
                dotCommand.add("-Tsvg");
                dotCommand.add("-o");
                dotCommand.add(tempPngFile.getAbsolutePath());
                dotCommand.add(tempDotFile.getAbsolutePath());

                ProcessBuilder dotBuilder = new ProcessBuilder(dotCommand);
                Process dotProcess = dotBuilder.start();
                dotProcess.waitFor();

                List<String> fehCommand = new ArrayList<>();
                fehCommand.add(svgViewer);
                //  fehCommand.add("-t");
                fehCommand.add(tempPngFile.getAbsolutePath());
                ProcessBuilder fehBuilder = new ProcessBuilder(fehCommand);
                Process fehProcess = fehBuilder.start();

            } catch (IOException | InterruptedException ioe) {
                ioe.printStackTrace();
            }
        } else if (dot == null) {
            System.out.println("Sorry can't find /usr/bin/dot (sudo apt-get install graphviz)");
        } else {
            System.out.println("Sorry can't find a suitable SVG Viewer");
        }

    }

    public static void main(String[] args) throws IOException {
        // https://renenyffenegger.ch/notes/tools/Graphviz/examples/index

        StringWriter dotw = new StringWriter();
        // http://magjac.com/graphviz-visual-editor/
        new DotRenderer().writer(dotw).start("mine").graph(
                (g) -> g
                        .box("A",
                                (box) -> box
                                        .label("Snarlywang")
                                        .color("lightyellow")
                                        .style("filled")
                        )
                        .record("B",
                                (record) -> record
                                        .color("lightgreen")
                                        .style("filled")
                                        .label((label) -> label
                                                .port("left", "left")
                                                .box(
                                                        (vertical) -> vertical
                                                                .port("top", "top")
                                                                .label("center")
                                                                .port("bottom", "bottom")
                                                )
                                                .port("right", "right")
                                        )
                        )
                        .edge("A", "B:top:nw", (e) -> e.label("1"))
                        .edge("A", "B:bottom:se", (e) -> e.label("2"))
                        .edge("A", "B:left:w", (e) -> e.label("3"))
                        .edge("A", "B:right:e", (e) -> e.label("4"))
        ).end();
        DotViewer.view(dotw.toString());
    }
}
