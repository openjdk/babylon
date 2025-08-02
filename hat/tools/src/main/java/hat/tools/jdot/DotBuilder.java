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
package hat.tools.jdot;
import java.util.function.Consumer;

public class DotBuilder<T extends DotBuilder<T>> {
    Consumer<String> consumer;
    T self(){
        return (T) this;
    }
    public DotBuilder(Consumer<String> consumer) {
        this.consumer = consumer;
    }

    public T accept(String s) {
        consumer.accept(s);
        return self();
    }

    public T  digraph(String name, Consumer<T> consumer) {
        accept("strict").space().accept("digraph").space().accept(name).obrace().nl();
        consumer.accept(self());
        return nl().cbrace().nl();
    }

    public T obrace() {
        return accept("{");
    }

    public T cbrace() {
        return accept("}");
    }

    public  T osbrace() {
        return accept("[");
    }

    public T csbrace() {
        return accept("]");
    }

    public T space() {
        return accept(" ");
    }

    public T nl() {
        return accept("\n");
    }

    public  T semicolon() {
        return accept(";");
    }

    public T equals() {
        return accept("=");
    }

    public T arrow() {
        return accept("->");
    }

    public T nodeShape(String shape) {
        return accept("node").space().osbrace().assign("shape", shape).csbrace().semicolon();
    }

    public T assign(String name, String value) {
        return accept(name).equals().dquote(value);
    }
    public T label(String value) {
        return assign("label", value);
    }
    public T record(String nodeName, String labelValue) {
        return node(nodeName, _->sbrace(_->label(labelValue)));
    }

    T dquote(String s) {
        int portIndex= s.indexOf(":");
        if (portIndex != -1) {
            String nodeName = s.substring(0, portIndex);
            String port = s.substring( portIndex);
            return accept("\"").accept(nodeName).accept("\"").accept(port);
        }else {
            return accept("\"").accept(s).accept("\"");
        }
    }

    public T node(String name, Consumer<T> consumer) {
        dquote(name);
        consumer.accept(self());
        return semicolon().nl();
    }

    public T edge(String fromNodeName, String toNodeName) {
        return dquote(fromNodeName).arrow().dquote(toNodeName).semicolon().nl();
    }

    public T sbrace(Consumer<T> consumer) {
        osbrace();
        consumer.accept(self());
        return csbrace();
    }
    public static class StringDotBuilder extends DotBuilder<StringDotBuilder> {
            public StringDotBuilder(Consumer<String> consumer) {
            super(consumer);
        }
    }
    public static String dotDigraph(String name, Consumer<StringDotBuilder> consumer) {
        StringBuilder sb = new StringBuilder();
         var db = new StringDotBuilder(sb::append);
         db.digraph(name, consumer);
         return sb.toString();
    }
}
