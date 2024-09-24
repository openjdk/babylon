package oracle.code.triton;

import java.io.IOException;
import java.io.StringWriter;
import java.io.UncheckedIOException;
import java.io.Writer;
import java.lang.reflect.code.*;
import java.lang.reflect.code.op.ExternalizableOp;
import java.lang.reflect.code.type.JavaType;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.List;
import java.util.Stack;
import java.util.regex.Pattern;
import java.util.function.Consumer;
import java.util.function.Function;
import java.util.stream.Collectors;

/**
 * A writer of code models to the textual form.
 * <p>
 * A code model in textual form may be parsed back into the runtime form by
 * parsing it.
 */
// @@@ We cannot link to OpParser since this code is copied into the
// jdk.compiler module

public final class MLIRGenerator {

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

    /**
     * A function that will get value from multiple result values
     * 
     * @param codeItem the code item
     * @return the string representation of the code item
     */
    String applyWrapper(CodeItem codeItem) {
        String s = namer.apply(codeItem);
        return m.getOrDefault(s, s);
    }

    /**
     * A mapping function that add additional characters to the attribute
     */
    static final class AttributeMapper {
        static String toString(String name, Object value) {
            if (value == ExternalizableOp.NULL_ATTRIBUTE_VALUE)
                return "null";
            else if (name.equals("function_type") || name.equals("arg_attrs") || name.equals("value"))
                return quote(value.toString());
            else if (name.equals("callee"))
                return "@" + quote(value.toString());
            else if (name.equals("operandSegmentSizes"))
                return value.toString();
            else if (value instanceof String)
                return "\"" + quote(value.toString()) + "\"";
            else if (value instanceof Integer)
                return quote(value.toString()) + ": i32";
            else if (value instanceof Long)
                return quote(value.toString()) + ": i64";
            else if (value instanceof Float)
                return quote(value.toString()) + ": f32";
            else if (value instanceof Double)
                return quote(value.toString()) + ": f64";
            else
                return quote(value.toString());
        }
    }

    /**
     * A mapping function that converts Java Triton types to Triton types.
     */
    static final class TypeConverter {
        static final Map<Pattern, String> typeReplacements = new LinkedHashMap<>();

        static {
            typeReplacements.put(Pattern.compile("boolean"), "i1");
            typeReplacements.put(Pattern.compile("int"), "i32");
            typeReplacements.put(Pattern.compile("float"), "f32");
            typeReplacements.put(Pattern.compile("oracle\\.code\\.triton\\.Float16"), "f16");
            typeReplacements.put(Pattern.compile("ptr"), "!tt.ptr");
            typeReplacements.put(Pattern.compile("x(\\d+)"), "$1");
            typeReplacements.put(Pattern.compile("void"), "()");
        }

        /**
         * Helper function to convert the type to MLIR type
         *
         * @param type    the type
         * @param m       the map of the start index to the end index
         * @param idxToOp the map of the index to the operation
         * @param op      the operation
         * @param start   the start index
         * @param end     the end index
         * @return the string represents MLIR type
         */
        private static String convertType(String type, Map<Integer, Integer> m, Map<Integer, String> idxToOp, String op,
                int start, int end) {
            StringBuilder sb = new StringBuilder();
            for (int i = start; i < end; ++i) {
                if (idxToOp.containsKey(i)) {
                    String nextOp = idxToOp.get(i);
                    if (nextOp.equals("ptr")) {
                        sb.append("ptr<");
                        sb.append(convertType(type, m, idxToOp, nextOp, i + 4, m.get(i + 3)));
                        sb.append(", 1>");
                        i = m.get(i + 3);
                    } else if (nextOp.equals("Tuple")) {
                        sb.append("(");
                        sb.append(convertType(type, m, idxToOp, nextOp, i + 6, m.get(i + 5)));
                        sb.append(")");
                        i = m.get(i + 5);
                    } else if (nextOp.equals("tensor")) {
                        sb.append("tensor<");
                        sb.append(convertType(type, m, idxToOp, nextOp, i + 7, m.get(i + 6)));
                        sb.append(">");
                        i = m.get(i + 6);
                    }
                } else {
                    if (op.equals("tensor") && type.charAt(i) == ',') {
                        sb.append("x");
                        i++; // skip a space
                    } else {
                        sb.append(type.charAt(i));
                    }
                }
            }
            return sb.toString();
        }

        /**
         * Maps Java types to MLIR types.
         *
         * @param type the Java type
         * @return the string represents MLIR type
         */
        public static String mapType(String type) {
            for (Map.Entry<Pattern, String> entry : typeReplacements.entrySet()) {
                type = entry.getKey().matcher(type).replaceAll(entry.getValue());
            }
            StringBuilder sb = new StringBuilder();
            Stack<Integer> s = new Stack<>();
            Map<Integer, Integer> m = new HashMap<>();
            Map<Integer, String> idxToOp = new HashMap<>();

            for (int i = 0; i < type.length(); ++i) {
                if (i + 3 < type.length() && type.substring(i, i + 3).equals("ptr")) {
                    s.push(i + 3);
                    idxToOp.put(i, "ptr");
                } else if (i + 5 < type.length() && type.substring(i, i + 5).equals("Tuple")) {
                    s.push(i + 5);
                    idxToOp.put(i, "Tuple");
                } else if (i + 6 < type.length() && type.substring(i, i + 6).equals("tensor")) {
                    s.push(i + 6);
                    idxToOp.put(i, "tensor");
                } else if (type.charAt(i) == '>') {
                    m.put(s.pop(), i);
                }
            }

            return convertType(type, m, idxToOp, "", 0, type.length());
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
     * non-printable ASCII. Leaves non-ASCII characters alone.
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
    public static Function<CodeItem, String> computeGlobalNames(Op root) {
        MLIRGenerator w = new MLIRGenerator(Writer.nullWriter());
        w.writeOp(root);
        return w.namer();
    }

    /**
     * Writes a code model (an operation) to the character stream.
     * <p>
     * The character stream will be flushed after the model is writen.
     *
     * @param w  the character stream
     * @param op the code model
     */
    public static void writeTo(Writer w, Op op) {
        MLIRGenerator ow = new MLIRGenerator(w);
        ow.writeOp(op);
        try {
            w.flush();
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
    }

    /**
     * Writes a code model (an operation) to the character stream.
     * <p>
     * The character stream will be flushed after the model is writen.
     *
     * @param w       the character stream
     * @param op      the code model
     * @param options the writer options
     */
    public static void writeTo(Writer w, Op op, Option... options) {
        MLIRGenerator ow = new MLIRGenerator(w, options);
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
     */
    public static String transform(Op op) {
        StringWriter w = new StringWriter();
        writeTo(w, op);
        return w.toString();
    }

    /**
     * Writes a code model (an operation) to a string.
     *
     * @param op      the code model
     * @param options the writer options
     */
    public static String toText(Op op, MLIRGenerator.Option... options) {
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
     * An option describing whether location information should be written or
     * dropped.
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
     * An option describing whether an operation's descendant code elements should
     * be written or dropped.
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
     * An option describing whether an operation's result be written or dropped if
     * its type is void.
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
    /* A map store the original variable number to the var#number */
    final HashMap<String, String> m;

    /**
     * Creates a writer of code models (operations) to their textual form.
     *
     * @param w the character stream writer to write the textual form.
     */
    public MLIRGenerator(Writer w) {
        this.w = new IndentWriter(w);
        this.namer = new GlobalValueBlockNaming();
        this.dropLocation = false;
        this.dropOpDescendants = false;
        this.writeVoidOpResult = false;
        this.m = new HashMap<>();
    }

    /**
     * Creates a writer of code models (operations) to their textual form.
     *
     * @param w       the character stream writer to write the textual form.
     * @param options the writer options
     */
    public MLIRGenerator(Writer w, Option... options) {
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
                    dropLocation = locationOption == LocationOption.DROP_LOCATION;
                }
                case OpDescendantsOption opDescendantsOption -> {
                    dropOpDescendants = opDescendantsOption == OpDescendantsOption.DROP_DESCENDANTS;
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
        this.m = new HashMap<>();
    }

    /**
     * {@return the function that names blocks and values.}
     */
    public Function<CodeItem, String> namer() {
        return namer;
    }

    /**
     * Add additional attributes to the operation
     * This can be removed after babylon support these function attributes
     * 
     * @param op the operation
     * @param attributes the attributes of the operation
     * @return the updated attributes
     */
    Map<String, Object> addAditionalAttributes(Op op, Map<String, Object> attributes) {
        if (op.opName().equals("tt.func")) {
            String retType = op.bodies().get(0).bodyType().returnType().toString();
            List<Block.Parameter> parameters = op.bodies().get(0).entryBlock().parameters();
            attributes = new HashMap<>(attributes);
            attributes.put("sym_visibility", retType.equals("void") ? "public" : "private");
            StringBuilder sb = new StringBuilder();
            sb.append("[");
            sb.append(parameters.stream()
                .map(p -> "{tt.divisibility = 16 : i32}")
                .collect(Collectors.joining(", ")));
            sb.append("]");
            attributes.put("arg_attrs", sb.toString());
            sb = new StringBuilder();
            sb.append("(");
            sb.append(parameters.stream()
                .map(v -> TypeConverter.mapType(v.type().externalize().toString()))
                .collect(Collectors.joining(", ")));
            sb.append(") -> ");
            sb.append(TypeConverter.mapType(retType));
            attributes.put("function_type", sb.toString());
        } else if (op.opName().equals("tt.load")) {
            attributes = new HashMap<>(attributes);
            attributes.put("operandSegmentSizes", "array<i32: " + (op.operands().size() < 3 ? "1, 1, 0" : "1, 1, 1") + ">");
		} else if (op.opName().equals("arith.constant")) {
            if (op.result().type() instanceof TensorType) {
                attributes = new HashMap<>(attributes);
                String val = attributes.get("value")
                                       .toString()
                                       .replaceAll("-Infinity", "0xFF800000")
                                       .replaceAll("Infinity", "0x7F800000");
                attributes.put("value", "dense<" + val + ">");
            }
        }
        return attributes;
    }

    /**
     * Writes a code model, an operation, to the character stream.
     *
     * @param op the code model
     */
    public void writeOp(Op op) {
        // We use var#number instead of tuple.load
        if (op.opName().equals("tuple.load")) {
            write("// ");
        }
        if (op.opName() == "unreachable") {
            return;
        }
        if (op.parent() != null) {
            Op.Result opr = op.result();
            if (writeVoidOpResult || !opr.type().equals(JavaType.VOID)) {
                String number = writeValueDeclaration(opr);
                if (op.opName().equals("scf.for")) {
                    write(":" + String.valueOf(op.operands().size() - 3));
                } else if (op.opName().equals("tuple.load")) {
                    Object value = op instanceof ExternalizableOp exop ? exop.attributes().values().toArray()[0] : 0;
                    m.put(number, namer.apply(op.operands().get(0)) + "#" + String.valueOf((int) value));
                }
                write(" = ");
            }
        }
        write("\"");
        if (op.opName().equals("module"))
            write("builtin.");
        write(op.opName());
        write("\"");

        write(" ");
        write("(");
        writeCommaSeparatedList(op.operands(), this::writeValueUse);
        write(")");

        if (!op.successors().isEmpty()) {
            write(" ");
            writeSpaceSeparatedList(op.successors(), this::writeSuccessor);
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

        Map<String, Object> attributes = op instanceof ExternalizableOp exop ? exop.attributes() : Map.of();
        if (dropLocation && !attributes.isEmpty() &&
                attributes.containsKey(ExternalizableOp.ATTRIBUTE_LOCATION)) {
            attributes = new HashMap<>(attributes);
            attributes.remove(ExternalizableOp.ATTRIBUTE_LOCATION);
        }
        attributes = addAditionalAttributes(op, attributes);
        if (!attributes.isEmpty()) {
            write(" ");
            write("{");
            writeCommaSeparatedList(attributes.entrySet(), e -> writeAttribute(e.getKey(), e.getValue()));
            if (op.opName().equals("arith.constant")) {
                // arith.constant verifier needs type information
                write(":");
                writeType(op.resultType());
            }
            write("}");
        }
        write(" : ");
        write("(");
        writeCommaSeparatedList(op.operands(), this::writeValueType);
        write(") -> ");
        writeType(op.resultType());
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
        if (!name.isEmpty()) {
            write(name);
            write("=");
        }
        write(AttributeMapper.toString(name, value));
    }

    // writeRegion
    void writeBody(Body body) {
        write("(");
        Block eb = body.entryBlock();
        write("{\n");
        w.in();
        for (Block b : body.blocks()) {
            if (!b.isEntryBlock()) {
                write("\n");
            }
            writeBlock(b);
        }
        w.out();
        write("}");
        write(")");
    }

    void writeBlock(Block block) {
        writeBlockName(block);
        if (!block.parameters().isEmpty()) {
            write("(");
            writeCommaSeparatedList(block.parameters(), this::writeValueDeclarationWithType);
            write(")");
        }
        write(":\n");
        w.in();
        for (Op op : block.ops()) {
            writeOp(op);
            write("\n");
        }
        w.out();
    }

    void writeBlockName(Block b) {
        write("^");
        write(applyWrapper(b));
    }

    void writeValueUse(Value v) {
        write("%");
        write(applyWrapper(v));
    }

    String writeValueDeclaration(Value v) {
        write("%");
        String ret = namer.apply(v);
        write(ret);
        return ret;
    }

    void writeValueDeclarationWithType(Value v) {
        writeValueDeclaration(v);
        write(": ");
        writeType(v.type());
    }

    void writeValueType(Value v) {
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
        write(TypeConverter.mapType(te.externalize().toString()));
    }

    void write(String s) {
        try {
            w.write(s);
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
    }
}