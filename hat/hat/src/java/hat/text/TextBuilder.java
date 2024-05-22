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

/**
 * The base abstract class for a slew of fluent style builders.
 *
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
 * @author Gary Frost
 */


public abstract class TextBuilder<T extends TextBuilder<T>> {
    public static class State {
        private final StringBuilder stringBuilder = new StringBuilder();
        final  public boolean indenting = true;
        private int indent = 0;
        private final String indentation = "    ";
        private boolean newLined = true;
    }
    State state = new State();
    public T reset(){
        state = new State();
       return self();
    }
    public String getText(){
        return toString();
    }
    public String getTextAndReset(){
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
    public T escaped(String text){
        StringBuilder buf = new StringBuilder();
        for (int i = 0; i < text.length(); i++) {
            buf.append(escape(text.charAt(i)));
        }
        return append(text);
    }
     public T indent(){
        for (int i = 0; i < state.indent; i++) {
            state.stringBuilder.append(state.indentation);
        }
        return self();
    }
    final protected T emitText(String text) {
        if (state.indenting && state.newLined) {
            indent();
        }
        state.newLined = false;
        state.stringBuilder.append(text);
        return self();
    }
    public final T commented(String text) {
        return emitText(text);
    }
    public T identifier(String text) {
        return emitText(text);
    }
    public T append(String text) {
        return emitText(text);
    }
    public final  T symbol(String text) {
        return emitText(text);
    }
    public final T typeName(String text) {
        return emitText(text);
    }
    public final T keyword(String text) {
        return emitText(text);
    }
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
        state.indent++;
        return self();
    }

    public T out() {
        state.indent--;
        return self();
    }

    final protected T emitNl() {
        state.stringBuilder.append("\n");
        state.newLined = true;
        return self();
    }
    public T nl() {
        return emitNl();
    }


    public T space() {
        return emitSpace();
    }
    final protected T emitSpace() {
        return emitText(" ");
    }
    @Override
    public final String toString() {
        return state.stringBuilder.toString();
    }
    public static class ConcreteTextBuilder extends TextBuilder<ConcreteTextBuilder>{
    }
    public static ConcreteTextBuilder concreteTextBuilder(){
        return  new ConcreteTextBuilder();
    }
}
