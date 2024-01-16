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

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.function.Predicate;

/**
 * A class that defines codes/utilities for Java source tokens
 * returned from lexical analysis.
 *
 * <p><b>This is NOT part of any supported API.
 * If you write code that depends on this, you do so at your own risk.
 * This code and its internal interfaces are subject to change or
 * deletion without notice.</b>
 */
public final class Tokens {

    /**
     * Keyword array. Maps name indices to Token.
     */
    private final Map<String, TokenKind> keywords = new HashMap<>();

    Tokens() {
        for (TokenKind t : TokenKind.values()) {
            if (t.name != null) {
                keywords.put(t.name, t);
            }
        }
    }

    TokenKind lookupKind(String name, TokenKind identifier) {
        TokenKind t = keywords.get(name);
        return (t != null) ? t : identifier;
    }

    /**
     * This enum defines all tokens used by the javac scanner. A token is
     * optionally associated with a name.
     */
    public enum TokenKind implements Predicate<TokenKind> {
        EOF(),
        ERROR(),
        IDENTIFIER(Token.Tag.NAMED),
        VALUE_IDENTIFIER(Token.Tag.NAMED),
        EXTENDS("extends", Token.Tag.NAMED),
        SUPER("super", Token.Tag.NAMED),
        INTLITERAL(Token.Tag.NUMERIC),
        LONGLITERAL(Token.Tag.NUMERIC),
        FLOATLITERAL(Token.Tag.NUMERIC),
        DOUBLELITERAL(Token.Tag.NUMERIC),
        CHARLITERAL(Token.Tag.NUMERIC),
        STRINGLITERAL(Token.Tag.STRING),
        TRUE("true", Token.Tag.NAMED),
        FALSE("false", Token.Tag.NAMED),
        NULL("null", Token.Tag.NAMED),
        UNDERSCORE("_", Token.Tag.NAMED),
        ARROW("->"),
        LPAREN("("),
        RPAREN(")"),
        LBRACE("{"),
        RBRACE("}"),
        LBRACKET("["),
        RBRACKET("]"),
        COMMA(","),
        DOT("."),
        EQ("="),
        GT(">"),
        LT("<"),
        QUES("?"),
        COLON(":"),
        COLCOL("::"),
        SEMI(";"),
        PLUS("+"),
        SUB("-"),
        AMP("&"),
        CARET("^"),
        MONKEYS_AT("@"),
        CUSTOM;

        public final String name;
        public final Token.Tag tag;

        TokenKind() {
            this(null, Token.Tag.DEFAULT);
        }

        TokenKind(String name) {
            this(name, Token.Tag.DEFAULT);
        }

        TokenKind(Token.Tag tag) {
            this(null, tag);
        }

        TokenKind(String name, Token.Tag tag) {
            this.name = name;
            this.tag = tag;
        }

        public String toString() {
            return switch (this) {
                case IDENTIFIER -> "token.identifier";
                case VALUE_IDENTIFIER -> "token.value-identifier";
                case CHARLITERAL -> "token.character";
                case STRINGLITERAL -> "token.string";
                case INTLITERAL -> "token.integer";
                case LONGLITERAL -> "token.long-integer";
                case FLOATLITERAL -> "token.float";
                case DOUBLELITERAL -> "token.double";
                case ERROR -> "token.bad-symbol";
                case EOF -> "token.end-of-input";
                case DOT, COMMA, LPAREN, RPAREN, LBRACKET, RBRACKET, LBRACE, RBRACE -> "'" + name + "'";
                default -> name;
            };
        }

        @Override
        public boolean test(TokenKind that) {
            return this == that;
        }
    }

    public interface Comment {

        enum CommentStyle {
            LINE,
            BLOCK,
        }

        String text();

        CommentStyle style();
    }

    /**
     * This is the class representing a javac token. Each token has several fields
     * that are set by the javac lexer (i.e. start/end position, string value, etc).
     */
    public static class Token {

        /**
         * tags constants
         **/
        public enum Tag {
            DEFAULT,
            NAMED,
            STRING,
            NUMERIC
        }

        /**
         * The token kind
         */
        public final TokenKind kind;

        /**
         * The start position of this token
         */
        public final int pos;

        /**
         * The end position of this token
         */
        public final int endPos;

        /**
         * Comment reader associated with this token
         */
        public final List<Comment> comments;

        Token(TokenKind kind, int pos, int endPos, List<Comment> comments) {
            this.kind = kind;
            this.pos = pos;
            this.endPos = endPos;
            this.comments = comments == null ? null : List.copyOf(comments);
            checkKind();
        }

        void checkKind() {
            if (kind.tag != Tag.DEFAULT) {
                throw new AssertionError("Bad token kind - expected " + Tag.DEFAULT);
            }
        }

        public String name() {
            throw new UnsupportedOperationException();
        }

        public String stringVal() {
            throw new UnsupportedOperationException();
        }

        public int radix() {
            throw new UnsupportedOperationException();
        }
    }

    static final class NamedToken extends Token {
        /**
         * The name of this token
         */
        public final String name;

        NamedToken(TokenKind kind, int pos, int endPos, String name, List<Comment> comments) {
            super(kind, pos, endPos, comments);
            this.name = name;
        }

        void checkKind() {
            if (kind.tag != Tag.NAMED) {
                throw new AssertionError("Bad token kind - expected " + Tag.NAMED);
            }
        }

        @Override
        public String name() {
            return name;
        }
    }

    static class StringToken extends Token {
        /**
         * The string value of this token
         */
        public final String stringVal;

        StringToken(TokenKind kind, int pos, int endPos, String stringVal, List<Comment> comments) {
            super(kind, pos, endPos, comments);
            this.stringVal = stringVal;
        }

        void checkKind() {
            if (kind.tag != Tag.STRING) {
                throw new AssertionError("Bad token kind - expected " + Tag.STRING);
            }
        }

        @Override
        public String stringVal() {
            return stringVal;
        }
    }

    static final class NumericToken extends StringToken {
        /**
         * The 'radix' value of this token
         */
        public final int radix;

        NumericToken(TokenKind kind, int pos, int endPos, String stringVal, int radix, List<Comment> comments) {
            super(kind, pos, endPos, stringVal, comments);
            this.radix = radix;
        }

        void checkKind() {
            if (kind.tag != Tag.NUMERIC) {
                throw new AssertionError("Bad token kind - expected " + Tag.NUMERIC);
            }
        }

        @Override
        public int radix() {
            return radix;
        }
    }

    public static final Token DUMMY =
            new Token(TokenKind.ERROR, 0, 0, null);
}
