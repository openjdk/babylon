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

package hat.tools.text;

import java.io.*;

import jdk.incubator.code.*;
import jdk.incubator.code.dialect.java.JavaType;
import jdk.incubator.code.dialect.java.WildcardType;
import jdk.incubator.code.extern.ExternalizedOp;
import jdk.incubator.code.extern.ExternalizedTypeElement;

import java.lang.reflect.Array;
import java.nio.charset.StandardCharsets;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.function.Consumer;
import java.util.function.Function;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

/**
 * A writer of code models to the textual form.
 * <p>
 * A code model in textual form may be parsed back into the runtime form by parsing it.
 */
public final class OpCodeBuilder {

    // Hacked from jdk.incubator.code/share/classes/jdk/incubator/code/dialect/java/impl/JavaTypeUtils.java
static class JavaTypeUtils{

    // useful type identifiers

    /**  Inflated Java class type name */
    public static final String JAVA_TYPE_CLASS_NAME = "java.type.class";
    /**  Inflated Java array type name */
    public static final String JAVA_TYPE_ARRAY_NAME = "java.type.array";
    /**  Inflated Java wildcard type name */
    public static final String JAVA_TYPE_WILDCARD_NAME = "java.type.wildcard";
    /**  Inflated Java type var name */
    public static final String JAVA_TYPE_VAR_NAME = "java.type.var";
    /**  Inflated Java primitive type name */
    public static final String JAVA_TYPE_PRIMITIVE_NAME = "java.type.primitive";

    /** Inflated Java field reference name */
    public static final String JAVA_REF_FIELD_NAME = "java.ref.field";
    /** Inflated Java method reference name */
    public static final String JAVA_REF_METHOD_NAME = "java.ref.method";
    /** Inflated Java constructor reference name */
    public static final String JAVA_REF_CONSTRUCTOR_NAME = "java.ref.constructor";
    /** Inflated Java record name */
    public static final String JAVA_REF_RECORD_NAME = "java.ref.record";

    /** Flattened Java type name */
    public static final String JAVA_TYPE_FLAT_NAME_PREFIX = "java.type:";
    /** Flattened Java reference name */
    public static final String JAVA_REF_FLAT_NAME_PREFIX = "java.ref:";

    /**
     * An enum modelling the Java type form kind. Useful for switching.
     */
    public enum Kind {
        /** A flattened type form */
        FLATTENED_TYPE,
        /** A flattened reference form */
        FLATTENED_REF,
        /** An inflated type form */
        INFLATED_TYPE,
        /** An inflated reference form */
        INFLATED_REF,
        /** Some other form */
        OTHER;

        /**
         * Constructs a new kind from an externalized type form
         * @param tree the externalized type form
         * @return the kind modelling {@code tree}
         */
        public static Kind of(ExternalizedTypeElement tree) {
            return switch (tree.identifier()) {
                case JAVA_TYPE_CLASS_NAME, JAVA_TYPE_ARRAY_NAME,
                     JAVA_TYPE_PRIMITIVE_NAME, JAVA_TYPE_WILDCARD_NAME,
                     JAVA_TYPE_VAR_NAME -> INFLATED_TYPE;
                case JAVA_REF_FIELD_NAME, JAVA_REF_CONSTRUCTOR_NAME,
                     JAVA_REF_METHOD_NAME, JAVA_REF_RECORD_NAME -> INFLATED_REF;
                case String s when s.startsWith(JAVA_TYPE_FLAT_NAME_PREFIX) -> FLATTENED_TYPE;
                case String s when s.startsWith(JAVA_REF_FLAT_NAME_PREFIX) -> FLATTENED_REF;
                default -> OTHER;
            };
        }
    }
    private static ExternalizedTypeElement nameToType(String name) {
        return ExternalizedTypeElement.of(name);
    }
    private static <T> T select(ExternalizedTypeElement tree, int index, Function<ExternalizedTypeElement, T> valueFunc) {
        if (index >= tree.arguments().size()) {
            throw new UnsupportedOperationException();
        }
        return valueFunc.apply(tree.arguments().get(index));
    }
    private static <T> List<T> selectFrom(ExternalizedTypeElement tree, int startIncl, Function<ExternalizedTypeElement, T> valueFunc) {
        if (startIncl >= tree.arguments().size()) {
            return List.of();
        }
        return IntStream.range(startIncl, tree.arguments().size())
                .mapToObj(i -> valueFunc.apply(tree.arguments().get(i)))
                .toList();
    }

    private static String typeToName(ExternalizedTypeElement tree) {
        if (!tree.arguments().isEmpty()) {
            throw new UnsupportedOperationException();
        }
        return tree.identifier();
    }

    /**
     * {@return a flat string modelling the provided inflated Java reference form}.
     * @param tree the inflated Java type form
     */
    public static String toExternalRefString(ExternalizedTypeElement tree) {
        return switch (tree.identifier()) {
            case JAVA_REF_FIELD_NAME -> {
                String owner = select(tree, 0, JavaTypeUtils::toExternalTypeString);
                String fieldName = select(tree, 1, JavaTypeUtils::typeToName);
                String fieldType = select(tree, 2, JavaTypeUtils::toExternalTypeString);
                yield String.format("%s::%s:%s", owner, fieldName, fieldType);
            }
            case JAVA_REF_METHOD_NAME -> {
                String owner = select(tree, 0, JavaTypeUtils::toExternalTypeString);
                ExternalizedTypeElement nameAndArgs = select(tree, 1, Function.identity());
                String methodName = nameAndArgs.identifier();
                List<String> paramTypes = selectFrom(nameAndArgs, 0, JavaTypeUtils::toExternalTypeString);
                String restype = select(tree, 2, JavaTypeUtils::toExternalTypeString);
                yield String.format("%s::%s(%s):%s", owner, methodName, String.join(", ", paramTypes), restype);
            }
            case JAVA_REF_CONSTRUCTOR_NAME -> {
                String owner = select(tree, 0, JavaTypeUtils::toExternalTypeString);
                ExternalizedTypeElement nameAndArgs = select(tree, 1, Function.identity());
                List<String> paramTypes = selectFrom(nameAndArgs, 0, JavaTypeUtils::toExternalTypeString);
                yield String.format("%s::(%s)", owner, String.join(", ", paramTypes));
            }
            case JAVA_REF_RECORD_NAME -> {
                String owner = select(tree, 0, JavaTypeUtils::toExternalTypeString);
                List<String> components = selectFrom(tree, 1, Function.identity()).stream()
                        .map(t -> {
                            String componentName = t.identifier();
                            String componentType = select(t, 0, JavaTypeUtils::toExternalTypeString);
                            return String.format("%s %s", componentType, componentName);
                        }).toList();
                yield String.format("(%s)%s", String.join(", ", components), owner);
            }
            default ->  throw new UnsupportedOperationException();
        };
    }

    private static boolean isSameType(ExternalizedTypeElement tree, TypeElement typeElement) {
        return tree.equals(typeElement.externalize());
    }

    /**
     * {@return a flat string modelling the provided inflated Java type form}.
     * @param tree the inflated Java type form
     */
    public static String toExternalTypeString(ExternalizedTypeElement tree) {
        return switch (tree.identifier()) {
            case JAVA_TYPE_CLASS_NAME -> {
                String className = select(tree, 0, JavaTypeUtils::typeToName);
                ExternalizedTypeElement enclosing = select(tree, 1, Function.identity());
                String typeargs = tree.arguments().size() == 2 ?
                        "" :
                        selectFrom(tree, 2, JavaTypeUtils::toExternalTypeString).stream()
                                .collect(Collectors.joining(", ", "<", ">"));
                if (isSameType(enclosing, JavaType.VOID)) {
                    yield String.format("%s%s", className, typeargs);
                } else {
                    String enclosingString = toExternalTypeString(enclosing);
                    yield String.format("%s::%s%s", enclosingString, className, typeargs);
                }
            }
            case JAVA_TYPE_ARRAY_NAME -> {
                String componentType = select(tree, 0, JavaTypeUtils::toExternalTypeString);
                yield String.format("%s[]", componentType);
            }
            case JAVA_TYPE_WILDCARD_NAME -> {
                WildcardType.BoundKind boundKind = select(tree, 0, t -> WildcardType.BoundKind.valueOf(typeToName(t)));
                ExternalizedTypeElement bound = select(tree, 1, Function.identity());
                yield boundKind == WildcardType.BoundKind.EXTENDS && isSameType(bound, JavaType.J_L_OBJECT) ?
                        "?" :
                        String.format("? %s %s", boundKind.name().toLowerCase(), toExternalTypeString(bound));
            }
            case JAVA_TYPE_VAR_NAME -> {
                String tvarName = select(tree, 0, JavaTypeUtils::typeToName);
                String owner = select(tree, 1, t ->
                        switch (Kind.of(t)) {
                            case INFLATED_REF -> "&" + toExternalRefString(t);
                            case INFLATED_TYPE -> toExternalTypeString(t);
                            default ->  throw new UnsupportedOperationException();
                        });
                ExternalizedTypeElement bound = select(tree, 2, Function.identity());
                yield isSameType(bound, JavaType.J_L_OBJECT) ?
                        String.format("%s::<%s>", owner, tvarName) :
                        String.format("%s::<%s extends %s>", owner, tvarName, toExternalTypeString(bound));
            }
            case JAVA_TYPE_PRIMITIVE_NAME -> select(tree, 0, JavaTypeUtils::typeToName);
            default -> throw  new UnsupportedOperationException();
        };
    }

    /**
     * {@return the flat Java form corresponding to the provided inflated Java form}
     * @param tree the inflated Java form
     */
    public static ExternalizedTypeElement flatten(ExternalizedTypeElement tree) {
        return switch (Kind.of(tree)) {
            case INFLATED_TYPE -> nameToType(String.format("%s\"%s\"", JAVA_TYPE_FLAT_NAME_PREFIX, toExternalTypeString(tree)));
            case INFLATED_REF -> nameToType(String.format("%s\"%s\"", JAVA_REF_FLAT_NAME_PREFIX, toExternalRefString(tree)));
            default -> ExternalizedTypeElement.of(tree.identifier(), tree.arguments().stream().map(JavaTypeUtils::flatten).toList());
        };
    }

}
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
                    case TypeElement te -> sb.append(JavaTypeUtils.flatten(te.externalize()).toString());
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

    static final class IndentWriter {
        static final int INDENT = 2;
        private final Writer w;
        private int indent;
        private boolean writeIndent = true;

        IndentWriter(Writer w) {
            this(w, 0);
        }

        IndentWriter(Writer w, int indent) {
            this.w = w;
            this.indent = indent;
        }


        public void write(String s)  {
            try {
                if (writeIndent) {
                    w.write(" ".repeat(indent));
                    writeIndent = false;
                }
                w.write(s);
            } catch (IOException e) {
                throw new UncheckedIOException(e);
            }

        }

        public void nl(){
            write("\n");
            writeIndent=true;
        }

        public void symbol(String symbol){
           write(symbol);
        }

        void space(){
            write(" ");
        }

        void in() {
            indent += INDENT;
        }
        void out() {
            indent -= INDENT;
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
        OpCodeBuilder w = new OpCodeBuilder(Writer.nullWriter());
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
        OpCodeBuilder ow = new OpCodeBuilder(w, options);
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
    public static String toText(Op op, OpCodeBuilder.Option... options) {
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

    final Function<CodeItem, String> namer;
    final IndentWriter w;
    final boolean dropLocation;
    final boolean dropOpDescendants;
    final boolean writeVoidOpResult;

    /**
     * Creates a writer of code models (operations) to their textual form.
     *
     * @param w the character stream writer to write the textual form.
     */
    public OpCodeBuilder(Writer w) {
        this.w = new IndentWriter(w);
        this.namer = new GlobalValueBlockNaming();
        this.dropLocation = false;
        this.dropOpDescendants = false;
        this.writeVoidOpResult = false;
    }

    /**
     * Creates a writer of code models (operations) to their textual form.
     *
     * @param w the character stream writer to write the textual form.
     * @param options the writer options
     */
    public OpCodeBuilder(Writer w, Option... options) {
        Function<CodeItem, String> namer = null;
        boolean dropLocation = false;
        boolean dropOpDescendants = false;
        boolean writeVoidOpResult = false;
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
            }
        }

        this.w = new IndentWriter(w);
        this.namer = (namer == null) ? new GlobalValueBlockNaming() : namer;
        this.dropLocation = dropLocation;
        this.dropOpDescendants = dropOpDescendants;
        this.writeVoidOpResult = writeVoidOpResult;
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
    public OpCodeBuilder writeOp(Op op) {
        if (op.parent() != null) {
            Op.Result opr = op.result();
            if (writeVoidOpResult || !opr.type().equals(JavaType.VOID)) {
                writeValueDeclaration(opr).space().equal().space();
            }
        }
        write(op.externalizeOpName());

        if (!op.operands().isEmpty()) {
            space().writeSpaceSeparatedList(op.operands(), this::writeValueUse);
        }

        if (!op.successors().isEmpty()) {
            space().writeSpaceSeparatedList(op.successors(), this::writeSuccessor);
        }

        if (!dropLocation) {
            Location location = op.location();
            if (location != null) {
                space().writeAttribute(ATTRIBUTE_LOCATION, op.location());
            }
        }
        Map<String, Object> attributes = op.externalize();
        if (!attributes.isEmpty()) {
            space().writeSpaceSeparatedList(attributes.entrySet(), e -> writeAttribute(e.getKey(), e.getValue()));
        }

        if (!dropOpDescendants && !op.bodies().isEmpty()) {
            int nBodies = op.bodies().size();
            if (nBodies == 1) {
                space();
            } else {
                nl().in().in();
            }
            boolean first = true;
            for (Body body : op.bodies()) {
                if (!first) {
                    nl();
                }
                writeBody(body);
                first = false;
            }
            if (nBodies > 1) {
                out().out();
            }
        }
        semicolon();
        return this;
    }

    OpCodeBuilder writeSuccessor(Block.Reference successor) {
        writeBlockName(successor.targetBlock());
        if (!successor.arguments().isEmpty()) {
            oparen().nl().in().writeCommaSeparatedList(successor.arguments(), this::writeValueUse).out().nl().cparen();
        }
        return this;
    }

    OpCodeBuilder writeAttribute(String name, Object value) {
        at();
        if (!name.isEmpty()) {
            write(name);
            equal();
        }
        write(AttributeMapper.toString(value));
        return this;
    }

    OpCodeBuilder writeBody(Body body) {
        Block eb = body.entryBlock();
        oparen();
        if (!eb.parameters().isEmpty()) {
            nl().in().writeCommaSeparatedList(eb.parameters(), this::writeValueDeclaration).out().nl();
        }
        cparen();
        writeType(body.bodyType().returnType());
        space().arrow().space();
        obrace().nl().in();
        for (Block b : body.blocks()) {
            if (!b.isEntryBlock()) {
                nl();
            }
            writeBlock(b);
        }
        out().cbrace();
        return this;
    }

    OpCodeBuilder writeBlock(Block block) {
        if (!block.isEntryBlock()) {
            writeBlockName(block);
            if (!block.parameters().isEmpty()) {
                oparen().nl().in().writeCommaSeparatedList(block.parameters(), this::writeValueDeclaration).out().nl().cparen();
            }
            colon().nl();
        }
        in();
        for (Op op : block.ops()) {
            writeOp(op).nl();
        }
        out();
        return this;
    }

    OpCodeBuilder writeBlockName(Block b) {
       return hat().write(namer.apply(b));
    }

    OpCodeBuilder ssaid(Value v){
       return percent().write(namer.apply(v));
    }

    OpCodeBuilder writeValueUse(Value v) {
       return  ssaid(v);
    }

    OpCodeBuilder writeValueDeclaration(Value v) {
        return ssaid(v).space().colon().space().writeType(v.type());
    }

    <T> OpCodeBuilder writeSpaceSeparatedList(Iterable<T> l, Consumer<T> c) {
        return writeSeparatedList(" ", l, c);
    }

    <T> OpCodeBuilder writeCommaSeparatedList(Iterable<T> l, Consumer<T> c) {
        return writeSeparatedNlList(", ", l, c);
    }

    <T> OpCodeBuilder writeSeparatedList(String separator, Iterable<T> l, Consumer<T> c) {
        boolean first = true;
        for (T t : l) {
            if (!first) {
                write(separator);
            }
            c.accept(t);
            first = false;
        }
        return this;
    }
    <T> OpCodeBuilder writeSeparatedNlList(String separator, Iterable<T> l, Consumer<T> c) {
        boolean first = true;
        for (T t : l) {
            if (!first) {
                write(separator);
                nl();
            }
            c.accept(t);
            first = false;
        }
        return this;
    }
    OpCodeBuilder writeType(TypeElement te) {
        write(JavaTypeUtils.flatten(te.externalize()).toString());
        return this;
    }

    OpCodeBuilder write(String s) {
        w.write(s);
        return this;
    }
    OpCodeBuilder nl() {
          w.nl();
        return this;
    }

    OpCodeBuilder semicolon(){
        w.symbol(";");
        return this;
    }

    OpCodeBuilder space(){
        w.space();
        return this;
    }
    OpCodeBuilder colon(){
        w.symbol(":");
        return this;
    }
    OpCodeBuilder oparen(){
        w.symbol("(");
        return this;
    }
    OpCodeBuilder obrace(){
        w.symbol("{");
        return this;
    }
    OpCodeBuilder cparen(){
        w.symbol(")");
        return this;
    }
    OpCodeBuilder cbrace(){
        w.symbol("}");
        return this;
    }
    OpCodeBuilder equal(){
        w.symbol("=");
        return this;
    }
    OpCodeBuilder in(){
        w.in();
        return this;
    }
    OpCodeBuilder out(){
        w.out();
        return this;
    }
    OpCodeBuilder arrow(){
        w.symbol("->");
        return this;
    }
    OpCodeBuilder at(){
        w.symbol("@");
        return this;
    }
    OpCodeBuilder hat(){
        w.symbol("^");
        return this;
    }
    OpCodeBuilder percent(){
        w.symbol("%");
        return this;
    }
}
