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

import java.lang.reflect.code.parser.impl.Position.LineMap;

import java.nio.CharBuffer;
import java.util.ArrayList;
import java.util.List;

import static java.lang.reflect.code.parser.impl.Tokens.DUMMY;
import static java.lang.reflect.code.parser.impl.Tokens.Token;

/**
 * The lexical analyzer maps an input stream consisting of
 * ASCII characters and Unicode escapes into a token sequence.
 *
 * <p><b>This is NOT part of any supported API.
 * If you write code that depends on this, you do so at your own risk.
 * This code and its internal interfaces are subject to change or
 * deletion without notice.</b>
 */
public final class Scanner implements Lexer {

    private final Tokens tokens;

    /**
     * The token, set by nextToken().
     */
    private Token token;

    /**
     * The previous token, set by nextToken().
     */
    private Token prevToken;

    /**
     * Buffer of saved tokens (used during lookahead)
     */
    private final List<Token> savedTokens = new ArrayList<>();

    private final JavaBasedTokenizer tokenizer;

    Scanner(Factory fac, char[] buf, int inputLength) {
        this(fac, new JavaBasedTokenizer(fac, buf, inputLength));
    }

    Scanner(Factory fac, JavaBasedTokenizer tokenizer) {
        this.tokenizer = tokenizer;
        tokens = fac.tokens;
        token = prevToken = DUMMY;
    }

    public Token token() {
        return token(0);
    }

    public Token token(int lookahead) {
        if (lookahead == 0) {
            return token;
        } else {
            ensureLookahead(lookahead);
            return savedTokens.get(lookahead - 1);
        }
    }

    //where
    private void ensureLookahead(int lookahead) {
        for (int i = savedTokens.size(); i < lookahead; i++) {
            savedTokens.add(tokenizer.readToken());
        }
    }

    public Token prevToken() {
        return prevToken;
    }

    public void nextToken() {
        prevToken = token;
        if (!savedTokens.isEmpty()) {
            token = savedTokens.remove(0);
        } else {
            token = tokenizer.readToken();
        }
    }

    public LineMap getLineMap() {
        return tokenizer.getLineMap();
    }

    public int errPos() {
        return tokenizer.errPos();
    }

    public void errPos(int pos) {
        tokenizer.errPos(pos);
    }

    public static final class Factory {
        final Log log;
        final Tokens tokens;

        /**
         * Create a new scanner factory.
         */
        Factory() {
            this.log = new Log();
            this.tokens = new Tokens();
        }

        public Scanner newScanner(CharSequence input) {
            if (input instanceof CharBuffer charBuffer) {
                return newScanner(toArray(charBuffer), charBuffer.limit());
            } else {
                char[] array = input.toString().toCharArray();
                return newScanner(array, array.length);
            }
        }

        public Scanner newScanner(char[] input, int inputLength) {
            return new Scanner(this, input, inputLength);
        }

        static char[] toArray(CharBuffer buffer) {
            if (buffer.hasArray())
                return buffer.compact().flip().array();
            else
                return buffer.toString().toCharArray();
        }

        static final Factory INSTANCE = new Factory();
    }

    public static Factory factory() {
        return Factory.INSTANCE;
    }
}
