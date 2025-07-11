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

package jdk.incubator.code.extern.impl;

import jdk.incubator.code.extern.impl.Position.LineMap;
import jdk.incubator.code.extern.impl.Tokens.Token;
import java.util.Arrays;

/**
 * The lexical analyzer maps an input stream consisting of ASCII
 * characters and Unicode escapes into a token sequence.
 *
 * <p><b>This is NOT part of any supported API.
 * If you write code that depends on this, you do so at your own risk.
 * This code and its internal interfaces are subject to change or
 * deletion without notice.</b>
 */
public sealed interface Lexer permits Scanner {

    /**
     * Consume the next token.
     */
    void nextToken();

    /**
     * Return current token.
     */
    Tokens.Token token();

    /**
     * Return token with given lookahead.
     */
    Tokens.Token token(int lookahead);

    /**
     * Return the last character position of the previous token.
     */
    Tokens.Token prevToken();

    /**
     * Return the position where a lexical error occurred;
     */
    int errPos();

    /**
     * Set the position where a lexical error occurred;
     */
    void errPos(int pos);

    /**
     * Build a map for translating between line numbers and
     * positions in the input.
     *
     * @return a LineMap
     */
    LineMap getLineMap();

    default boolean is(Tokens.TokenKind tk) {
        Tokens.Token t = token();
        if (t.kind == tk) {
            return true;
        }
        return false;
    }

    default Tokens.Token accept(Tokens.TokenKind tk) {
        Tokens.Token t = token();
        if (t.kind == tk) {
            nextToken();
            return t;
        } else {
            // @@@ Exception
            LineMap lineMap = getLineMap();
            int lineNumber = lineMap.getLineNumber(t.pos);
            int columnNumber = lineMap.getColumnNumber(t.pos);
            throw new IllegalArgumentException("Expected " + tk + " but observed " + t.kind +
                    " " + lineNumber + ":" + columnNumber);
        }
    }

    default Tokens.Token accept(Tokens.TokenKind... tks) {
        Token t = token();
        for (Tokens.TokenKind tk : tks) {
            if (acceptIf(tk)) {
                return t;
            }
        }
        // @@@ Exception
        LineMap lineMap = getLineMap();
        int lineNumber = lineMap.getLineNumber(t.pos);
        int columnNumber = lineMap.getColumnNumber(t.pos);
        throw new IllegalArgumentException("Expected one of " + Arrays.toString(tks) + " but observed " + t.kind +
                " " + lineNumber + ":" + columnNumber);
    }

    default boolean acceptIf(Tokens.TokenKind tk) {
        Tokens.Token t = token();
        if (t.kind == tk) {
            nextToken();
            return true;
        }
        return false;
    }

    default RuntimeException unexpected() {
        Tokens.Token t = token();
        LineMap lineMap = getLineMap();
        int lineNumber = lineMap.getLineNumber(t.pos);
        int columnNumber = lineMap.getColumnNumber(t.pos);
        return new IllegalArgumentException("Unexpected token " + t.kind +
                " " + lineNumber + ":" + columnNumber);
    }
}
