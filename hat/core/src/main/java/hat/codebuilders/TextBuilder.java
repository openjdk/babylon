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
package hat.codebuilders;

/**
 * The base abstract class for a slew of fluent style builders.
 * <p>
 * At this level the builder just deals with basic appending, indenting, newline handling
 *
 * <pre>
 *     TextBuilder textBuilder = ....;
 *     textBuilder
 *        .nl()
 *        .in()
 *        .append("hello)
 *        .space()
 *        .append("world")
 *        .out();
 *
 * </pre>
 *
 * @author Gary Frost
 */


public abstract class TextBuilder<T extends TextBuilder<T>> implements TextRenderer<T> {


    public static class State {
        private final StringBuilder stringBuilder = new StringBuilder();
        final public boolean indenting = true;
        private int indent = 0;
        private final String indentation = "    ";
        private boolean newLined = true;
        public void indentation() {
            for (int i = 0; i < indent; i++) {
                stringBuilder.append(indentation);
            }
        }

        public void indentIfNeededAndAppend(String text) {
            if (indenting && newLined) {
                indentation();
            }
            newLined = false;
            stringBuilder.append(text);
        }

        public void incIndent() {
            indent++;
        }
        public void decIndent() {
            indent--;
        }

        public void nl() {
            newLined = true;
        }

        @Override
        public String toString(){
            return stringBuilder.toString();
        }
    }

    State state = new State();

    public T reset() {
        state = new State();
        return self();
    }

    public String getText() {
        return toString();
    }

    public String getTextAndReset() {
        String text = getText();
        reset();
        return text;
    }

    @SuppressWarnings("unchecked")
    public T self() {
        return (T) this;
    }

    private static String escape(char ch) {
        return switch (ch) {
            case '\b' -> "\\b";
            case '\f' -> "\\f";
            case '\n' -> "\\n";
            case '\r' -> "\\r";
            case '\t' -> "\\t";
            case '\'' -> "\\'";
            case '\"' -> "\\\"";
            case '\\' -> "\\\\";
            default -> (ch >= ' ' && ch <= '~') ? String.valueOf(ch) : String.format("\\u%04x", (int) ch);
        };
    }

    public T escaped(String text) {
        StringBuilder buf = new StringBuilder();
        for (int i = 0; i < text.length(); i++) {
            buf.append(escape(text.charAt(i)));
        }
        return emitText(text);
    }

    public T indent() {
        state.indentation();
        return self();
    }

     T emitText(String text) {
        state.indentIfNeededAndAppend(text);
        return self();
    }

    @Override
    public final T comment(String text) {
        return emitText(text);
    }
    @Override
    public T identifier(String text) {
        return emitText(text);
    }

    @Override
    public T reserved(String text) {
        return emitText(text);
    }

    @Override
    public T label(String text) {
        return emitText(text);
    }

    public T identifier(String text, int  padWidth) {
        return emitText(text).emitText(" ".repeat(padWidth-text.length()));
    }
    public T intValue(int i) {
        return emitText(Integer.toString(i));
    }
    public T intHexValue(int i) {
        return emitText("0x").emitText(Integer.toHexString(i));
    }

    @Override
    public final T symbol(String text) {
        return emitText(text);
    }
    @Override
    public final T typeName(String text) {
        return emitText(text);
    }
    @Override
    public final T keyword(String text) {
        return emitText(text);
    }

    @Override
    public final T literal(String text) {
        return emitText(text);
    }

    public final T literal(int i) {
        return emitText(Integer.toString(i));
    }

    public final T literal(long i) {
        return emitText(Long.toString(i));
    }

    public T in() {
        state.incIndent();
        return self();
    }

    public T out() {
        state.decIndent();
        return self();
    }

    @Override
    public T nl() {
        emitText("\n");
        state.nl();

        return self();
    }

    @Override
    public T space() {
         return emitText(" ");
    }

    @Override
    public final String toString() {
        return state.toString();
    }

}
