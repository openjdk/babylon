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
package hat.text;


public enum TerminalColors {
        // https://www.lihaoyi.com/post/BuildyourownCommandLinewithANSIescapecodes.html#8-colors
        NONE("0"),
        BLACK("38;5;0"), DARKGREEN("38;5;22"), DARKBLUE("38;5;27"),
        GREY("38;5;247"), RED("38;5;1"), GREEN("38;5;77"), YELLOW("38;5;185"),
        BLUE("38;5;31"), WHITE("38;5;251"), ORANGE("38;5;208"), PURPLE("38;5;133");
        public final String escSequence;

        TerminalColors(String seq) {
            escSequence = "\u001b[" + seq + "m";
        }
    }

