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

package jdk.code.tools.renderer;

import java.io.*;
import java.util.ArrayList;
import java.util.List;

/**
 * Created by gfrost
 */
public final class Diff {
    private Diff() {
    }

    public static void annotate(File file, int line, int col, String msg) {
        List<String> lines = getLines(file);
        for (int i = Math.max(line - 2, 0); i < (line - 1); i++) {
            System.out.printf("    %2d:%s\n", i + 1, lines.get(i));
        }
        String text = lines.get(line - 1);
        System.out.printf(" -> %2d:%s\n      ", line, text);
        for (int i = 0; i < col; i++) {
            System.out.print(" ");
        }
        System.out.println("^ " + msg);
        for (int i = line; i < Math.min(line + 2, lines.size() - 1); i++) {
            System.out.printf("    %2d:%s\n", i + 1, lines.get(i));
        }

    }

    static List<String> getLines(File file) {
        List<String> lines = new ArrayList<>();
        try {
            BufferedReader br = new BufferedReader(new InputStreamReader(new FileInputStream(file)));
            for (String line = br.readLine(); line != null; line = br.readLine()) {
                lines.add(line);
            }
            br.close();
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        } catch (IOException e) {
            e.printStackTrace();
        }
        return lines;
    }

    public static class DiffResult {
        String lhs;
        String rhs;
        public String result;
        public int exitStatus;

        public DiffResult(String lhs, String rhs, String result, int exitStatus) {
            this.lhs = lhs;
            this.rhs = rhs;
            this.result = result;
            this.exitStatus = exitStatus;
        }
    }

    public static DiffResult diff(String lhs, String rhs, int width) {
        try {
            File lhsFile = File.createTempFile("lhs", "txt");
            FileWriter lhsw = new FileWriter(lhsFile);
            lhsw.append(lhs);
            lhsw.close();
            File rhsFile = File.createTempFile("rhs", "txt");
            FileWriter rhsw = new FileWriter(rhsFile);
            rhsw.append(rhs);
            rhsw.close();

            List<String> command = new ArrayList<>();
            command.add("sdiff");
            command.add("--expand-tabs");
            command.add("--ignore-all-space");
            command.add("--width=" + width);
            command.add("--ignore-blank-lines");
            command.add(lhsFile.getAbsolutePath());
            command.add(rhsFile.getAbsolutePath());

            ProcessBuilder builder = new ProcessBuilder(command);
            final Process process = builder.start();
            BufferedReader br = new BufferedReader(new InputStreamReader(process.getInputStream()));
            StringBuilder out = new StringBuilder();
            String line;
            while ((line = br.readLine()) != null) {
                if (line.contains("|")) {
                    out.append(TerminalColors.Color.RED.colorize(line)).append("\n");
                } else {
                    out.append(TerminalColors.Color.GREEN.colorize(line)).append("\n");
                }

            }
            process.waitFor();
            br.close();
            lhsFile.delete();
            rhsFile.delete();
            return new DiffResult(lhs, rhs, out.toString(), process.exitValue());
        } catch (IOException | InterruptedException ioe) {
            ioe.printStackTrace();
        }
        return null;
    }


    public static File write(File file, String text) {
        try {
            PrintWriter pw = new PrintWriter(file);
            pw.append(text);
            pw.append("\n");
            pw.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
        return file;
    }


}
