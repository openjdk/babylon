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
package hat.tools.textmodel;

import hat.tools.textmodel.tokens.Pos;

public class
Cursor {
    final private char[] chars;
    private Loc current;

    public Cursor(String text) {
        this.chars = text.toCharArray();
       // current = new Loc(this, substring(0), 0, 1, 0);
        current = new Loc(this,  0, 1, 0);
    }

    String substring(int offset) {
        return new String(chars, offset, chars.length - offset);
    }
    public record Loc0(Cursor cursor, String lookingAt, int textOffset, int line, int col) implements Pos {
        public Character ch() {
            return (textOffset() < cursor().chars.length) ? cursor().chars[textOffset()] : null;
        }

        public Character peek() {
            return ((textOffset() + 1) < cursor().chars.length) ? cursor().chars[textOffset() + 1] : null;
        }

        public Pos pos(){
            return new Pos.Impl(cursor.chars,textOffset,line,col);
        }

        @Override
        public char[] text() {
            return cursor.chars;
        }
    }

    public record Loc(Cursor cursor,  int textOffset, int line, int col) implements Pos {
        public Character ch() {
            return (textOffset() < cursor().chars.length) ? cursor().chars[textOffset()] : null;
        }

        public Character peek() {
            return ((textOffset() + 1) < cursor().chars.length) ? cursor().chars[textOffset() + 1] : null;
        }

        public Pos pos(){
            return new Pos.Impl(cursor.chars,textOffset,line,col);
        }

        @Override
        public char[] text() {
            return cursor.chars;
        }
    }

    public Loc next() {
        if (current.ch() instanceof Character ch) {
            Loc prev = current;
           // String lookingAt = substring(current.textOffset + 1);
            current = (ch=='\n')
                   // ? new Loc(this,lookingAt, current.textOffset + 1, current.line + 1, 0)
                   // : new Loc(this,lookingAt, current.textOffset + 1, current.line, current.col+1);
               ? new Loc(this,current.textOffset + 1, current.line + 1, 0)
                    : new Loc(this, current.textOffset + 1, current.line, current.col+1);

            return prev;
        }
        return null;
    }

}
