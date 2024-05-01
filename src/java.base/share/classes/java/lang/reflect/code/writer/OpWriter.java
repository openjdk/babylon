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

package java.lang.reflect.code.writer;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.io.Writer;
import java.lang.reflect.code.*;
import java.lang.reflect.code.op.ExternalizableOp;
import java.lang.reflect.code.type.JavaType;
import java.util.HashMap;
import java.util.Map;
import java.util.function.Consumer;
import java.util.function.Function;

/**
 * A writer of code models to the textual form.
 * <p>
 * A code model in textual form may be parsed back into the runtime form by parsing it.
 */
// @@@ We cannot link to OpParser since this code is copied into the jdk.compiler module
public final class OpWriter {

    static final class GlobalValueBlockNaming implements Function<CodeItem, String> {
        final Map<CodeItem, String> gn;
        int valueOrdinal = 0;
        int blockOrdinal = 0;

        GlobalValueBlockNaming() {
            this.gn = new HashMap<>();
        }

        @Override
        public String apply(CodeItem codeItem) {
            return switch (codeItem) {
                case Block block -> gn.computeIfAbsent(block, _b -> "block_" + blockOrdinal++);
                case Value value -> gn.computeIfAbsent(value, _v -> String.valueOf(valueOrdinal++));
                default -> throw new IllegalStateException("Unexpected code item: " + codeItem);
            };
        }
    }

    static final class AttributeMapper {
        static String toString(Object value) {
            return value == ExternalizableOp.NULL_ATTRIBUTE_VALUE
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

    /**
     * Computes global names for blocks and values in a code model.
     * <p>
     * The code model is traversed in the same order as if the model
     * was written. Therefore, the names in the returned map will the
     * same as the names that are written. This can be useful for debugging
     * and testing.
     *
     * @param root the code model
     * @return the map of computed names, modifiable
     */
    public static Map<CodeItem, String> computeGlobalNames(Op root) {
        GlobalValueBlockNaming gn = root.traverse(new GlobalValueBlockNaming(), (n, e) -> {
            switch (e) {
                case Op op -> {
                    for (Block.Reference r : op.successors()) {
                        n.apply(r.targetBlock());
                    }

                    if (root != op) {
                        Op.Result opr = op.result();
                        if (!opr.type().equals(JavaType.VOID)) {
                            n.apply(opr);
                        }
                    }
                }
                case Block block -> {
                    if (!block.isEntryBlock()) {
                        n.apply(block);
                    }
                    for (Block.Parameter p : block.parameters()) {
                        n.apply(p);
                    }
                }
                default -> {
                }
            }
            return n;
        });

        return gn.gn;
    }

    /**
     * Writes a code model (an operation) to the character stream.
     * <p>
     * A carriage return will be written after the model is writen, and
     * then character stream will be flushed.
     *
     * @param w the character stream
     * @param op the code model
     */
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

    /**
     * Writes a code model (an operation) to the character stream.
     * <p>
     * A carriage return will be written after the model is writen, and
     * then character stream will be flushed.
     *
     * @param w the character stream
     * @param op the code model
     * @param options the writer options
     */
    public static void writeTo(Writer w, Op op, Option... options) {
        OpWriter ow = new OpWriter(w, options);
        ow.writeOp(op);
        ow.write("\n");
        try {
            w.flush();
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
    }

    /**
     * An option that affects the writing operations.
     */
    public sealed interface Option {
    }

    /**
     * An option describing the function to use for naming code items.
     */
    public sealed interface CodeItemNamerOption extends Option
            permits NamerOptionImpl {

        static CodeItemNamerOption of(Function<CodeItem, String> named) {
            return new NamerOptionImpl(named);
        }

        static CodeItemNamerOption defaultValue() {
            return of(new GlobalValueBlockNaming());
        }

        Function<CodeItem, String> namer();
    }
    private record NamerOptionImpl(Function<CodeItem, String> namer) implements CodeItemNamerOption {
    }

    /**
     * An option describing whether location information should be written or dropped.
     */
    public enum LocationOption implements Option {
        /** Writes location */
        WRITE_LOCATION,
        /** Drops location */
        DROP_LOCATION;

        public static LocationOption defaultValue() {
            return WRITE_LOCATION;
        }
    }

    final Function<CodeItem, String> namer;
    final IndentWriter w;
    final boolean dropLocation;

    /**
     * Creates a writer of code models (operations) to their textual form.
     *
     * @param w the character stream writer to write the textual form.
     */
    public OpWriter(Writer w) {
        this.w = new IndentWriter(w);
        this.namer = new GlobalValueBlockNaming();
        this.dropLocation = false;
    }

    /**
     * Creates a writer of code models (operations) to their textual form.
     *
     * @param w the character stream writer to write the textual form.
     * @param options the writer options
     */
    public OpWriter(Writer w, Option... options) {
        Function<CodeItem, String> namer = null;
        boolean dropLocation = false;
        for (Option option : options) {
            switch (option) {
                case CodeItemNamerOption namerOption -> {
                    namer = namerOption.namer();
                }
                case LocationOption locationOption -> {
                    dropLocation = locationOption ==
                            LocationOption.DROP_LOCATION;
                }
            }
        }

        this.w = new IndentWriter(w);
        this.namer = (namer == null) ? new GlobalValueBlockNaming() : namer;
        this.dropLocation = dropLocation;
    }

    /**
     * {@return the function that names blocks and values.}
     */
    public Function<CodeItem, String> namer() {
        return namer;
    }

    /**
     * Writes a code model, an operation, to the character stream.
     *
     * @param op the code model
     */
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

        Map<String, Object> attributes = op instanceof ExternalizableOp exop ? exop.attributes() : Map.of();
        if (dropLocation && !attributes.isEmpty() &&
                attributes.containsKey(ExternalizableOp.ATTRIBUTE_LOCATION)) {
            attributes = new HashMap<>(attributes);
            attributes.remove(ExternalizableOp.ATTRIBUTE_LOCATION);
        }
        if (!attributes.isEmpty()) {
            write(" ");
            writeSpaceSeparatedList(attributes.entrySet(), e -> writeAttribute(e.getKey(), e.getValue()));
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
        writeType(body.bodyType().returnType());
        write(" -> {\n");
        w.in();
        for (Block b : body.blocks()) {
            if (!b.isEntryBlock()) {
                write("\n");
            }
            writeBlock(b);
        }
        w.out();
        write("}");
    }

    void writeBlock(Block block) {
        if (!block.isEntryBlock()) {
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
            if (!opr.type().equals(JavaType.VOID)) {
                writeValueDeclaration(opr);
                write(" = ");
            }
            writeOp(op);
            write("\n");
        }
        w.out();
    }

    void writeBlockName(Block b) {
        write("^");
        write(namer.apply(b));
    }

    void writeValueUse(Value v) {
        write("%");
        write(namer.apply(v));
    }

    void writeValueDeclaration(Value v) {
        write("%");
        write(namer.apply(v));
        write(" : ");
        writeType(v.type());
    }

    <T> void writeSpaceSeparatedList(Iterable<T> l, Consumer<T> c) {
        writeSeparatedList(" ", l, c);
    }

    <T> void writeCommaSeparatedList(Iterable<T> l, Consumer<T> c) {
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

    void writeType(CodeType te) {
        write(te.externalize().toString());
    }

    void write(String s) {
        try {
            w.write(s);
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
    }
}
