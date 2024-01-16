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

package java.lang.reflect.code.parser.impl;

final class Errors {

    private Errors() {
    }

    static final class Error {
        String s;

        Error(String s) {
            this.s = s;
        }

        @Override
        public String toString() {
            return s;
        }
    }

    static final Error IllegalUnicodeEsc = new Error("illegal unicode escape");
    static final Error IllegalNonasciiDigit = new Error("illegal non-ASCII digit");
    static final Error IllegalEscChar = new Error("illegal escape character");
    static final Error IllegalUnderscore = new Error("illegal underscore");
    static final Error IllegalLineEndInCharLit = new Error("illegal line end in character literal");
    static final Error IllegalDot = new Error("illegal '.'");
    static final Error MalformedFpLit = new Error("malformed floating-point literal");
    static final Error InvalidHexNumber = new Error("hexadecimal numbers must contain at least one hexadecimal digit");
    static final Error InvalidBinaryNumber = new Error("binary numbers must contain at least one binary digit");
    static final Error EmptyCharLit = new Error("empty character literal");
    static final Error UnclosedStrLit = new Error("unclosed string literal");
    static final Error UnclosedComment = new Error("unclosed comment");
    static final Error UnclosedCharLit = new Error("unclosed character literal");

    static Error IllegalChar(String c) {
        return new Error("unmappable character for encoding " + c);
    }
}
