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

import java.io.IOException;
import java.io.UncheckedIOException;
import java.io.Writer;
import java.util.Map;

public class TextRenderer<T extends TextRenderer<T>> {

    public interface NestedRendererSAM<T> {
        T build(T nb);
    }

    public enum TokenType {
        WHITESPACE, OP, TYPE, CONTROL, LITERAL, COMMENT, KEYWORD, IDENTIFIER, LABELTARGET, NONE
    }

    public static class TokenColorMap {


        private final Map<TokenType, TerminalColors.Colorizer> map;

        public TokenColorMap(Map<TokenType, TerminalColors.Colorizer> map) {
            this.map = map;
        }

        public TokenColorMap() {
            this(Map.of(
                    TokenType.NONE, TerminalColors.Color.WHITE,
                    TokenType.IDENTIFIER, TerminalColors.Color.YELLOW,
                    TokenType.LABELTARGET, TerminalColors.Color.BLUE,
                    TokenType.TYPE, TerminalColors.Color.WHITE,
                    TokenType.COMMENT, TerminalColors.Color.GREEN,
                    TokenType.KEYWORD, TerminalColors.Color.ORANGE,
                    TokenType.CONTROL, TerminalColors.Color.GREY,
                    TokenType.LITERAL, TerminalColors.Color.GREEN,
                    TokenType.OP, TerminalColors.Color.WHITE,
                    TokenType.WHITESPACE, TerminalColors.Color.WHITE));
        }

        public String colorize(TokenType tokenType, String string) {
            if (map.containsKey(tokenType)) {
                return map.get(tokenType).colorize(string);
            } else {
                return string;
            }
        }
    }

    public static class State {
        public Writer writer;
        public int indent;
        public TokenColorMap tokenColorMap = null;
        public boolean isFirst = false;

        public boolean newLined = false;

        State() {
            this.writer = null;
            this.indent = 0;
            this.tokenColorMap = null;
            this.isFirst = false;
            this.newLined = true;
        }
    }

    public T writer(Writer writer) {
        this.state.writer = writer;
        return self();
    }

    public T colorize(TokenColorMap tokenColorMap) {
        this.state.tokenColorMap = tokenColorMap;
        return self();
    }

    public T colorize() {
        return colorize(new TokenColorMap());
    }

    public State state;

    protected TextRenderer() {
        this.state = new State();
    }

    protected TextRenderer(TextRenderer<?> renderer) {
        this.state = renderer.state;
    }

    @SuppressWarnings("unchecked")
    public T self() {
        return (T) this;
    }


    public boolean first() {
        var was = state.isFirst;
        state.isFirst = false;
        return was;
    }

    public T startList() {
        state.isFirst = true;
        return self();
    }

    public T append(String text) {
        try {
            // While we do expect appends text to be simple tokens. We can handle newlines.
            var lines = text.split("\n");
            for (int i = 0; i < lines.length - 1; i++) {
                state.writer.append(" ".repeat(state.indent) + lines[i] + "\n");
                state.newLined = true;
            }
            if (state.newLined) {
                state.writer.append(" ".repeat(state.indent));
            }
            state.writer.append(lines[lines.length - 1]);
            state.newLined = false;
            return self();
        } catch (IOException ioe) {
            throw new UncheckedIOException(ioe);
        }
    }

    public T identifier(String ident) {
        if (state.tokenColorMap != null) {
            return append(state.tokenColorMap.colorize(TokenType.IDENTIFIER, ident));
        } else {
            return append(ident);
        }
    }

    public T type(String typeName) {
        if (state.tokenColorMap != null) {
            return append(state.tokenColorMap.colorize(TokenType.TYPE, typeName));
        } else {
            return append(typeName);
        }
    }

    public T keyword(String keyword) {
        if (state.tokenColorMap != null) {
            return append(state.tokenColorMap.colorize(TokenType.KEYWORD, keyword));
        } else {
            return append(keyword);
        }
    }

    public T literal(String literal) {
        if (state.tokenColorMap != null) {
            return append(state.tokenColorMap.colorize(TokenType.LITERAL, literal));
        } else {
            return append(literal);
        }
    }

    public T ws(String whitespace) {
        if (state.tokenColorMap != null) {
            return append(state.tokenColorMap.colorize(TokenType.WHITESPACE, whitespace));
        } else {
            return append(whitespace);
        }
    }

    public T op(String op) {
        if (state.tokenColorMap != null) {
            return append(state.tokenColorMap.colorize(TokenType.OP, op));
        } else {
            return append(op);
        }
    }

    public T control(String control) {
        if (state.tokenColorMap != null) {
            return append(state.tokenColorMap.colorize(TokenType.CONTROL, control));
        } else {
            return append(control);
        }
    }

    public T labelTarget(String labelTarget) {
        if (state.tokenColorMap != null) {
            return append(state.tokenColorMap.colorize(TokenType.LABELTARGET, labelTarget));
        } else {
            return append(labelTarget);
        }
    }

    public T comment(String comment) {
        if (state.tokenColorMap != null) {
            return append(state.tokenColorMap.colorize(TokenType.COMMENT, comment));
        } else {
            return append(comment);
        }
    }

    public T strLiteral(String s) {
        return oquot().literal(s).cquot();
    }

    public T oquot() {
        return literal("\"");
    }

    public T cquot() {
        return literal("\"");
    }

    public T decLiteral(int i) {
        return literal(String.format("%d", i));
    }

    public T hexLiteral(int i) {
        return literal(String.format("%x", i));
    }

    public T in() {
        state.indent += 2;
        return self();
    }

    public T out() {
        state.indent -= 2;
        return self();
    }

    public T flush() {
        try {
            state.writer.flush();
            return self();
        } catch (IOException ioe) {
            throw new RuntimeException(ioe);
        }
    }

    public T nl() {
        try {
            // note we go directly to the underlying writer!
            state.writer.append("\n");
            state.newLined = true;
            return flush().self();
        } catch (IOException ioe) {
            throw new RuntimeException(ioe);
        }
    }

    public T space() {
        return ws(" ");
    }

    public T nest(NestedRendererSAM<T> nb) {
        return nb.build(self());
    }

    public T open(String op) {
        control(op);
        return self();
    }

    public T close(String op) {
        control(op);
        return self();
    }
}

