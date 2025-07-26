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
package hat.tools.textmodel.tokens;

public interface Pos extends LineCol {
    char[] text();

    int textOffset();


    default int delta(Pos earlier) {
        return textOffset() - earlier.textOffset();
    }

     class Impl implements Pos {
        @Override
        public char[] text() {
            return text;
        }

        @Override
        public int textOffset() {
            return textOffset;
        }

        @Override
        public int line() {
            return line;
        }

        @Override
        public int col() {
            return col;
        }

        char[] text;
        int textOffset;
        int line;
        int col;

        public Impl(char[] text, int textOffset, int line, int col) {
            this.text = text;
            this.textOffset = textOffset;
            this.line = line;
            this.col = col;
        }

        public Impl(Pos pos) {
            this(pos.text(), pos.textOffset(), pos.line(), pos.col());
        }
    }

}
