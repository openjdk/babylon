/*
 * Copyright (c) 2025, Oracle and/or its affiliates. All rights reserved.
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
package hat.test.engine;

public class HATTestFormatter {

    public static void appendClass(StringBuilder builder, String className) {
        builder.append(Colours.CYAN).append("Class: " + className).append(Colours.RESET).append("\n");;
    }

    public static void testing(StringBuilder builder, String methodName) {
        builder.append(Colours.BLUE)
                .append(String.format("Testing: #%-30s", methodName))
                .append(String.format("%-20s", "..................... "))
                .append(Colours.RESET);
    }

    public static void ok(StringBuilder builder) {
        builder.append(Colours.GREEN)
                .append("[ok]")
                .append(Colours.RESET)
                .append("\n");;
    }

    public static void fail(StringBuilder builder) {
        builder.append(Colours.RED)
                .append("[fail]")
                .append(Colours.RESET)
                .append("\n");;
    }

    public static void failWithReason(StringBuilder builder, String reason) {
        builder.append(Colours.RED)
                .append("[fail]")
                .append(Colours.YELLOW)
                .append(" Reason: ")
                .append(reason)
                .append(Colours.RESET)
                .append("\n");;
    }

    public static void illegal(StringBuilder builder) {
        builder.append(Colours.YELLOW)
                .append("[illegal]")
                .append(Colours.RESET)
                .append("\n");;
    }

}
