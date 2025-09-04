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

package jdk.incubator.code.extern;

import java.io.*;

import jdk.incubator.code.*;
import jdk.incubator.code.dialect.java.JavaType;
import jdk.incubator.code.dialect.java.impl.JavaTypeUtils;

import java.lang.reflect.Array;
import java.nio.charset.StandardCharsets;
import java.util.HashMap;
import java.util.Map;
import java.util.function.BiFunction;
import java.util.function.Consumer;
import java.util.function.Function;

/**
 * A writer of code models to the textual form.
 * <p>
 * A code model in textual form may be parsed back into the runtime form by parsing it.
 */
public final class OpWriter {

    /**
     * The attribute name associated with the location attribute.
     */
    static final String ATTRIBUTE_LOCATION = "loc";

    static final class GlobalValueBlockNaming implements Function<CodeItem, String> {
        final Map<CodeItem, String> gn;
        int valueOrdinal = 0;

        GlobalValueBlockNaming() {
            this.gn = new HashMap<>();
        }

        private String name(Block b) {
            Block p = b.ancestorBlock();
            return (p == null ? "block_" : name(p) + "_") + b.index();
        }

        @Override
        public String apply(CodeItem codeItem) {
            return switch (codeItem) {
                case Block block -> gn.computeIfAbsent(block, _b -> name(block));
                case Value value -> gn.computeIfAbsent(value, _v -> String.valueOf(valueOrdinal++));
                default -> throw new IllegalStateException("Unexpected code item: " + codeItem);
            };
        }
    }

    static final class AttributeMapper {
        static String toString(Object value) {
            if (value == ExternalizedOp.NULL_ATTRIBUTE_VALUE) {
                return "null";
            }

            StringBuilder sb = new StringBuilder();
            toString(value, sb);
            return sb.toString();
        }

        static void toString(Object o, StringBuilder sb) {
            if (o.getClass().isArray()) {
                // note, while we can't parse back the array representation, this might be useful
                // for non-externalizable ops that want better string representation of array attribute values (e.g. ONNX)
                arrayToString(o, sb);
            } else {
                switch (o) {
                    case Integer i -> sb.append(i);
                    case Long l -> sb.append(l).append('L');
                    case Float f -> sb.append(f).append('f');
                    case Double d -> sb.append(d).append('d');
                    case Character c -> sb.append('\'').append(c).append('\'');
                    case Boolean b -> sb.append(b);
                    case TypeElement te -> sb.append(JavaTypeUtils.flatten(te.externalize()));
                    default -> {  // fallback to a string
                        sb.append('"');
                        quote(o.toString(), sb);
                        sb.append('"');
                    }
                }
            }
        }

        static void arrayToString(Object a, StringBuilder sb) {
            boolean first = true;
            sb.append("[");
            for (int i = 0; i < Array.getLength(a); i++) {
                if (!first) {
                    sb.append(", ");
                }

                toString(Array.get(a, i), sb);
                first = false;
            }
            sb.append("]");
        }
    }

    static void quote(String s, StringBuilder sb) {
        for (int i = 0; i < s.length(); i++) {
            sb.append(quote(s.charAt(i)));
        }
    }

    /**
     * Escapes a character if it has an escape sequence or is
     * non-printable ASCII.  Leaves non-ASCII characters alone.
     */
    // Copied from com.sun.tools.javac.util.Convert
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
    public static Function<CodeItem, String> computeGlobalNames(Op root) {
        OpWriter w = new OpWriter(Writer.nullWriter());
        w.writeOp(root);
        return w.namer();
    }

    /**
     * Writes a code model (an operation) to the output stream, using the UTF-8 character set.
     *
     * @param out the output stream
     * @param op the code model
     */
    public static void writeTo(OutputStream out, Op op, Option... options) {
        writeTo(new OutputStreamWriter(out, StandardCharsets.UTF_8), op, options);
    }

    /**
     * Writes a code model (an operation) to the character stream.
     * <p>
     * The character stream will be flushed after the model is writen.
     *
     * @param w the character stream
     * @param op the code model
     * @param options the writer options
     */
    public static void writeTo(Writer w, Op op, Option... options) {
        OpWriter ow = new OpWriter(w, options);
        ow.writeOp(op);
        try {
            // @@@ Is this needed?
            w.flush();
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
    }

    /**
     * Writes a code model (an operation) to a string.
     *
     * @param op the code model
     * @param options the writer options
     */
    public static String toText(Op op, OpWriter.Option... options) {
        StringWriter w = new StringWriter();
        writeTo(w, op, options);
        return w.toString();
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

    /**
     * An option describing whether an operation's descendant code elements should be written or dropped.
     */
    public enum OpDescendantsOption implements Option {
        /** Writes descendants of an operation, if any */
        WRITE_DESCENDANTS,
        /** Drops descendants of an operation, if any */
        DROP_DESCENDANTS;

        public static OpDescendantsOption defaultValue() {
            return WRITE_DESCENDANTS;
        }
    }

    /**
     * An option describing whether an operation's result be written or dropped if its type is void.
     */
    public enum VoidOpResultOption implements Option {
        /** Writes void operation result */
        WRITE_VOID,
        /** Drops void operation result */
        DROP_VOID;

        public static VoidOpResultOption defaultValue() {
            return DROP_VOID;
        }
    }

    /**
     * An option describing how to color the output.
     */
    public enum ColoringOption implements Option {
        /** Performs no coloring. */
        NONE((_, text) -> text),

        /** Uses ANSI codes to color the output */
        ANSI((itemType, text) -> "\033[3" +
                (itemType == Op.class ? '4' : // blue
                itemType == Block.class ? '5': // purple
                itemType == TypeElement.class ? '2': '1') // green : red
                + "m" + text + "\033[0m"),

        /** Uses HTML elements to color the output */
        HTML((itemType, text) -> "<font color=\"" +
                (itemType == Op.class ? "blue" :
                itemType == Block.class ? "purple":
                itemType == TypeElement.class ? "green" : "red")
                + "\">" + text + "</font>");

        public static ColoringOption defaultValue() {
            return NONE;
        }

        final BiFunction<Class<? extends CodeItem>, String, String> dyer;

        ColoringOption(BiFunction<Class<? extends CodeItem>, String, String> dyer) {
            this.dyer = dyer;
        }
    }

    final Function<CodeItem, String> namer;
    final IndentWriter w;
    final boolean dropLocation;
    final boolean dropOpDescendants;
    final boolean writeVoidOpResult;
    final BiFunction<Class<? extends CodeItem>, String, String> dyer;

    /**
     * Creates a writer of code models (operations) to their textual form.
     *
     * @param w the character stream writer to write the textual form.
     */
    public OpWriter(Writer w) {
        this.w = new IndentWriter(w);
        this.namer = new GlobalValueBlockNaming();
        this.dropLocation = false;
        this.dropOpDescendants = false;
        this.writeVoidOpResult = false;
        this.dyer = ColoringOption.NONE.dyer;
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
        boolean dropOpDescendants = false;
        boolean writeVoidOpResult = false;
        var dyer = ColoringOption.NONE.dyer;
        for (Option option : options) {
            switch (option) {
                case CodeItemNamerOption namerOption -> {
                    namer = namerOption.namer();
                }
                case LocationOption locationOption -> {
                    dropLocation = locationOption ==
                            LocationOption.DROP_LOCATION;
                }
                case OpDescendantsOption opDescendantsOption -> {
                    dropOpDescendants = opDescendantsOption ==
                            OpDescendantsOption.DROP_DESCENDANTS;
                }
                case VoidOpResultOption voidOpResultOption -> {
                    writeVoidOpResult = voidOpResultOption == VoidOpResultOption.WRITE_VOID;
                }
                case ColoringOption colorSchemaOption -> {
                    dyer = colorSchemaOption.dyer;
                }
            }
        }

        this.w = new IndentWriter(w);
        this.namer = (namer == null) ? new GlobalValueBlockNaming() : namer;
        this.dropLocation = dropLocation;
        this.dropOpDescendants = dropOpDescendants;
        this.writeVoidOpResult = writeVoidOpResult;
        this.dyer = dyer;
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
        if (op.parent() != null) {
            Op.Result opr = op.result();
            if (writeVoidOpResult || !opr.type().equals(JavaType.VOID)) {
                writeValueDeclaration(opr);
                write(" = ");
            }
        }
        write(Op.class, op.opName());

        if (!op.operands().isEmpty()) {
            write(" ");
            writeSpaceSeparatedList(op.operands(), this::writeValueUse);
        }

        if (!op.successors().isEmpty()) {
            write(" ");
            writeSpaceSeparatedList(op.successors(), this::writeSuccessor);
        }

        if (!dropLocation) {
            Location location = op.location();
            if (location != null) {
                write(" ");
                writeAttribute(ATTRIBUTE_LOCATION, op.location());
            }
        }
        Map<String, Object> attributes = op.externalize();
        if (!attributes.isEmpty()) {
            write(" ");
            writeSpaceSeparatedList(attributes.entrySet(), e -> writeAttribute(e.getKey(), e.getValue()));
        }

        if (!dropOpDescendants && !op.bodies().isEmpty()) {
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
            writeOp(op);
            write("\n");
        }
        w.out();
    }

    void writeBlockName(Block b) {
        write(Block.class, "^" + namer.apply(b));
    }

    void writeValueUse(Value v) {
        write(Value.class, "%" + namer.apply(v));
    }

    void writeValueDeclaration(Value v) {
        write(Value.class, "%" + namer.apply(v));
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

    void writeType(TypeElement te) {
        write(TypeElement.class, JavaTypeUtils.flatten(te.externalize()).toString());
    }

    void write(Class<? extends CodeItem> itemType, String s) {
        write(dyer.apply(itemType, s));
    }

    void write(String s) {
        try {
            w.write(s);
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
    }
}
