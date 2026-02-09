/*
 * Copyright (c) 2026, Oracle and/or its affiliates. All rights reserved.
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
package hat.examples.common;

public class ParseArgs {

    private final String[] arguments;

    public ParseArgs(String[] arguments) {
        this.arguments = arguments;
    }

    public Options parseWithDefaults(int defaultSize, int defaultIterations) {
        boolean verbose = false;
        int size = defaultSize;
        int iterations = defaultIterations;
        boolean skipSequential = false;
        // process parameters
        for (String arg : arguments) {
            if (arg.equals("--verbose")) {
                verbose = true;
            } else if (arg.startsWith("--size=")) {
                String number = arg.split("=")[1];
                try {
                    size = Integer.parseInt(number);
                } catch (NumberFormatException _) {
                }
            } else if (arg.startsWith("--iterations=")) {
                String number = arg.split("=")[1];
                try {
                    iterations = Integer.parseInt(number);
                } catch (NumberFormatException _) {
                }
            } else if (arg.startsWith("--skip-sequential")) {
                skipSequential = true;
            }
        }
        return new Options(verbose, size, iterations, skipSequential);
    }

    public record Options(boolean verbose, int size, int iterations, boolean skipSequential) {}
}
