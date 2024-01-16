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

package java.lang.reflect.code.impl;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.io.Writer;
import java.lang.reflect.code.Block;
import java.lang.reflect.code.Body;
import java.lang.reflect.code.Op;
import java.lang.reflect.code.Value;
import java.lang.reflect.code.descriptor.TypeDesc;
import java.util.function.Consumer;

public final class OpWriter {

    static class AttributeMapper {
        static String toString(Object value) {
            return value == Op.NULL_ATTRIBUTE_VALUE
                    ? "null"
                    : "\"" + quote(value.toString()) + "\"";
        }
    }

    // Copied from com.sun.tools.javac.util.Convert
    static String quote(String s) {
        StringBuilder buf = new StringBuilder();
        for (int i = 0; i < s.length(); i++) {
            buf.append(quote(s.charAt(i)));
        }
        return buf.toString();
    }

    /**
     * Escapes a character if it has an escape sequence or is
     * non-printable ASCII.  Leaves non-ASCII characters alone.
     */
    static String quote(char ch) {
        return switch (ch) {
            case '\b' -> "\\b";
            case '\f' -> "\\f";
            case '\n' -> "\\n";
            case '\r' -> "\\r";
            case '\t' -> "\\t";
            case '\'' -> "\\'";
            case '\"' -> "\\\"";
            case '\\' -> "\\\\";
            default -> (isPrintableAscii(ch))
                    ? String.valueOf(ch)
                    : String.format("\\u%04x", (int) ch);
        };
    }

    /**
     * Is a character printable ASCII?
     */
    static boolean isPrintableAscii(char ch) {
        return ch >= ' ' && ch <= '~';
    }

    static final class IndentWriter extends Writer {
        static final int INDENT = 2;

        final Writer w;
        int indent;
        boolean writeIndent = true;

        IndentWriter(Writer w) {
            this(w, 0);
        }

        IndentWriter(Writer w, int indent) {
            this.w = w;
            this.indent = indent;
        }

        @Override
        public void write(char[] cbuf, int off, int len) throws IOException {
            if (writeIndent) {
                w.write(" ".repeat(indent));
                writeIndent = false;
            }
            w.write(cbuf, off, len);
            if (len > 0 && cbuf[off + len - 1] == '\n') {
                writeIndent = true;
            }
        }

        @Override
        public void flush() throws IOException {
            w.flush();
        }

        @Override
        public void close() throws IOException {
            w.close();
        }

        void in() {
            in(INDENT);
        }

        void in(int i) {
            indent += i;
        }

        void out() {
            out(INDENT);
        }

        void out(int i) {
            indent -= i;
        }
    }

    public static void writeTo(Writer w, Op op) {
        OpWriter ow = new OpWriter(w);
        ow.writeOp(op);
        ow.write("\n");
        try {
            w.flush();
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
    }

    final java.lang.reflect.code.impl.GlobalValueBlockNaming gn;
    final IndentWriter w;

    public OpWriter(Writer w) {
        this(w, new java.lang.reflect.code.impl.GlobalValueBlockNaming());
    }

    public OpWriter(Writer w, GlobalValueBlockNaming gn) {
        this.gn = gn;
        this.w = new IndentWriter(w);
    }

    public void writeOp(Op op) {
        write(op.opName());

        if (!op.operands().isEmpty()) {
            write(" ");
            writeSpaceSeparatedList(op.operands(), this::writeValueUse);
        }

        if (!op.successors().isEmpty()) {
            write(" ");
            writeSpaceSeparatedList(op.successors(), this::writeSuccessor);
        }

        if (!op.attributes().isEmpty()) {
            write(" ");
            writeSpaceSeparatedList(op.attributes().entrySet(), e -> writeAttribute(e.getKey(), e.getValue()));
        }

        if (!op.bodies().isEmpty()) {
            int nBodies = op.bodies().size();
            if (nBodies == 1) {
                write(" ");
            } else {
                write("\n");
                w.in();
                w.in();
            }
            boolean first = true;
            for (Body body : op.bodies()) {
                if (!first) {
                    write("\n");
                }
                writeBody(body);
                first = false;
            }
            if (nBodies > 1) {
                w.out();
                w.out();
            }
        }

        write(";");
    }

    void writeSuccessor(Block.Reference successor) {
        writeBlockName(successor.targetBlock());
        if (!successor.arguments().isEmpty()) {
            write("(");
            writeCommaSeparatedList(successor.arguments(), this::writeValueUse);
            write(")");
        }
    }

    void writeAttribute(String name, Object value) {
        write("@");
        if (!name.isEmpty()) {
            write(name);
            write("=");
        }
        write(AttributeMapper.toString(value));
    }

    void writeBody(Body body) {
        Block eb = body.entryBlock();
        write("(");
        writeCommaSeparatedList(eb.parameters(), this::writeValueDeclaration);
        write(")");
        write(body.descriptor().returnType().toString());
        write(" -> {\n");
        w.in();
        boolean isEntryBlock = true;
        for (Block b : body.blocks()) {
            if (!isEntryBlock) {
                write("\n");
            }
            writeBlock(b, isEntryBlock);
            isEntryBlock = false;
        }
        w.out();
        write("}");
    }

    void writeBlock(Block block, boolean isEntryBlock) {
        if (!isEntryBlock) {
            writeBlockName(block);
            if (!block.parameters().isEmpty()) {
                write("(");
                writeCommaSeparatedList(block.parameters(), this::writeValueDeclaration);
                write(")");
            }
            write(":\n");
        }
        w.in();
        for (Op op : block.ops()) {
            Op.Result opr = op.result();
            if (!opr.type().equals(TypeDesc.VOID)) {
                writeValueDeclaration(opr);
                write(" = ");
            }
            writeOp(op);
            write("\n");
        }
        w.out();
    }

    void writeBlockName(Block b) {
        writeBlockName(gn.getBlockName(b));
    }

    void writeBlockName(String s) {
        write("^");
        write(s);
    }

    void writeValueUse(Value v) {
        write("%");
        write(gn.getValueName(v));
    }

    void writeValueDeclaration(Value v) {
        write("%");
        write(gn.getValueName(v));
        write(" : ");
        write(v.type().toString());
    }

    <T> void writeSpaceSeparatedList(Iterable<T> l, Consumer<T> c) {
        writeSeparatedList(" ", l, c);
    }

    public <T> void writeCommaSeparatedList(Iterable<T> l, Consumer<T> c) {
        writeSeparatedList(", ", l, c);
    }

    <T> void writeSeparatedList(String separator, Iterable<T> l, Consumer<T> c) {
        boolean first = true;
        for (T t : l) {
            if (!first) {
                write(separator);
            }
            c.accept(t);
            first = false;
        }
    }

    public void write(String s) {
        try {
            w.write(s);
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
    }
}
