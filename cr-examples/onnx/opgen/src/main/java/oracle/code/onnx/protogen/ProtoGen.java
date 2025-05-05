/*
 * Copyright (c) 2025, Oracle and/or its affiliates. All rights reserved.
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

package oracle.code.onnx.protogen;

import java.io.FileOutputStream;
import java.io.PrintStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.stream.Stream;

public class ProtoGen {

    static final String COPYRIGHT_NOTICE = """
            /*
             * Copyright (c) 2025, Oracle and/or its affiliates. All rights reserved.
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

            """;

    static final String PROTOGEN_PACKAGE = "oracle.code.onnx.proto";
    static final String PROTOGEN_CONSTANTS_CLASS = "OnnxConstants";
    static final String PROTOGEN_BUILDER_CLASS = "OnnxBuilder";
    static final String PROTOGEN_MODEL_CLASS = "OnnxModel";

    static final String SOURCES_PATH = "src/main/java/" +PROTOGEN_PACKAGE.replace(".", "/") + "/";

    private static final String E = "\\s*";
    private static final String C = "\\s*(?<comment>//.*)";
    private static final String OC = C + "?";
    private static final String NB = "\\s+(?<name>\\w+)\\s*\\{" + OC;

    enum TokenType {
        EMPTY(E),
        COMMENT(E + C),
        FIELD(E + "(?<flag>optional |repeated |)\\s*(?<type>\\w+)\\s+(?<name>\\w+)\\s*=\\s*(?<index>\\d+)\\s*(\\[.*\\])?\\s*;" + OC),
        ENUM_ELEMENT(E + "(?<name>\\w+)\\s*=\\s*(?<value>\\w+)\\s*;" + OC),
        END(E + "\\}\\s*;?" + OC),
        MESSAGE(E + "message" + NB),
        ENUM(E + "enum" + NB),
        ONEOF(E + "oneof" + NB),
        PACKAGE(E + "package\\s+(?<name>\\S+)\\s*;" + OC),
        RESERVED(E + "reserved\\s+(?<words>.+)\\s*;" + OC),
        SYNTAX(E + "syntax\\s*=\\s*(?<version>.+)\\s*;" + OC);

        final Pattern pattern;

        TokenType(String pattern) {
            this.pattern = Pattern.compile(pattern);
        }
    }

    record Token(TokenType type, Matcher matcher) {}

    record TreeNode(List<String> comments, Token token, List<TreeNode> nested) {}

    static Token lineToToken(String line) {
        for (var tt : TokenType.values()) {
            var m = tt.pattern.matcher(line);
            if (m.matches()) {
                return new Token(tt, m);
            }
        }
        throw new IllegalArgumentException(line);
    }

    static void generateConstants(List<TreeNode> tree, PrintStream out) {
        out.print(COPYRIGHT_NOTICE);
        out.println("package " + PROTOGEN_PACKAGE + ";");
        out.print("""

                import java.util.function.IntSupplier;

                // Generated from onnx.in.proto
                """);
        out.println("public final class " + PROTOGEN_CONSTANTS_CLASS + " {");
        for (TreeNode en : tree.stream().flatMap(n -> Stream.concat(Stream.of(n), n.nested().stream())).filter(n -> n.token().type() == TokenType.ENUM).toList()) {
            out.println();
            String name = en.token().matcher().group("name");
            for (String c : en.comments()) {
                out.println("    /" + c);
            }
            out.println("    public enum " + name + " implements IntSupplier {");
            for (TreeNode ev : en.nested()) {
                if (ev.token().type() == TokenType.ENUM_ELEMENT) {
                    out.println();
                    for (String c : ev.comments()) {
                        out.println("        /" + c);
                    }
                    out.println("        " + ev.token().matcher().group("name") + "(" + ev.token().matcher().group("value") + "),");
                }
            }
            out.println("        ;");
            out.println();
            out.println("        final int value;");
            out.println();
            out.println("        " + name + "(int value) {");
            out.println("            this.value = value;");
            out.println("        }");
            out.println();
            out.println("        @Override");
            out.println("        public int getAsInt() {");
            out.println("            return value;");
            out.println("        }");
            out.println("    }");
        }
        out.println("}");
    }

    static void generateBuilder(List<TreeNode> tree, PrintStream out) {
        out.print(COPYRIGHT_NOTICE);
        out.println("package " + PROTOGEN_PACKAGE + ";");
        out.print("""

                import java.io.ByteArrayOutputStream;
                import java.nio.charset.StandardCharsets;
                import java.util.List;
                import java.util.function.BiConsumer;
                import java.util.function.IntSupplier;

                """);
        out.println("import " + PROTOGEN_PACKAGE + "." + PROTOGEN_CONSTANTS_CLASS + ".*;");
        out.print("""

                // Generated from onnx.in.proto
                """);
        out.println("public sealed class " + PROTOGEN_BUILDER_CLASS + "<T extends " + PROTOGEN_BUILDER_CLASS + "> {");
        generateBuilderCode(null, "    ", tree, out);
        out.print("""

                    // Implementation

                    final ByteArrayOutputStream buf = new ByteArrayOutputStream();

                    public byte[] getBytes() {
                        return buf.toByteArray();
                    }

                    @SuppressWarnings("unchecked")
                    public <P> T forEach(Iterable<P> sup, BiConsumer<T, ? super P> cons) {
                        sup.forEach(p -> cons.accept((T)this, p));
                        return (T)this;
                    }

                    void _encode(long number) {
                        for (int i = 64 - Long.numberOfLeadingZeros(number); i > 7; i -= 7) {
                            buf.write(0x80 | (int)number & 0x7f);
                            number >>= 7;
                        }
                        buf.write((int)number & 0x7f);
                    }

                    void _encode(float value) {
                        int bits =  Float.floatToRawIntBits(value);
                        buf.write((byte)bits);
                        buf.write((byte)(bits >> 8));
                        buf.write((byte)(bits >> 16));
                        buf.write((byte)(bits >> 24));
                    }

                    void _encode(double value) {
                        long bits =  Double.doubleToRawLongBits(value);
                        buf.write((byte)bits);
                        buf.write((byte)(bits >> 8));
                        buf.write((byte)(bits >> 16));
                        buf.write((byte)(bits >> 24));
                        buf.write((byte)(bits >> 32));
                        buf.write((byte)(bits >> 40));
                        buf.write((byte)(bits >> 48));
                        buf.write((byte)(bits >> 56));
                    }

                    @SuppressWarnings("unchecked")
                    T _f(int fieldIndex, String value) {
                        return value == null ? (T)this : _f(fieldIndex, value.getBytes(StandardCharsets.UTF_8));
                    }

                    @SuppressWarnings("unchecked")
                    T _f(int fieldIndex, byte[] bytes) {
                        _encode(fieldIndex << 3 | 2);
                        _encode(bytes.length);
                        buf.writeBytes(bytes);
                        return (T)this;
                    }

                    @SuppressWarnings("unchecked")
                    T _f(int fieldIndex, float value) {
                        _encode(fieldIndex << 3 | 5);
                        _encode(value);
                        return (T)this;
                    }

                    @SuppressWarnings("unchecked")
                    T _f(int fieldIndex, float... values) {
                        if (values.length == 1) {
                            return _f(fieldIndex, values[0]);
                        }
                """);
        out.println("        var b = new " + PROTOGEN_BUILDER_CLASS + "();");
        out.print("""
                        for (var v : values) b._encode(v);
                        _f(fieldIndex, b);
                        return (T)this;
                    }

                    @SuppressWarnings("unchecked")
                    T _f(int fieldIndex, double value) {
                        _encode(fieldIndex << 3 | 1);
                        _encode(value);
                        return (T)this;
                    }

                    @SuppressWarnings("unchecked")
                    T _f(int fieldIndex, double... values) {
                        if (values.length == 1) {
                            return _f(fieldIndex, values[0]);
                        }
                """);
        out.println("        var b = new " + PROTOGEN_BUILDER_CLASS + "();");
        out.print("""
                        for (var v : values) b._encode(v);
                        _f(fieldIndex, b);
                        return (T)this;
                    }

                    @SuppressWarnings("unchecked")
                    T _f(int fieldIndex, long value) {
                        _encode(fieldIndex << 3);
                        _encode(value);
                        return (T)this;
                    }

                    @SuppressWarnings("unchecked")
                    T _f(int fieldIndex, long... values) {
                        if (values.length == 1) {
                            return _f(fieldIndex, values[0]);
                        }
                """);
        out.println("        var b = new " + PROTOGEN_BUILDER_CLASS + "();");
        out.print("""
                        for (var v : values) b._encode(v);
                        _f(fieldIndex, b);
                        return (T)this;
                    }

                    @SuppressWarnings("unchecked")
                    T _f(int fieldIndex, int... values) {
                        if (values.length == 1) {
                            return _f(fieldIndex, values[0]);
                        }
                """);
        out.println("        var b = new " + PROTOGEN_BUILDER_CLASS + "();");
        out.print("""
                        for (var v : values) b._encode(v);
                        _f(fieldIndex, b);
                        return (T)this;
                    }

                    @SuppressWarnings("unchecked")
                """);
        out.println("    T _f(int fieldIndex, " + PROTOGEN_BUILDER_CLASS + " value) {");
        out.print("""
                        return _f(fieldIndex, value.buf.toByteArray());
                    }

                    @SuppressWarnings("unchecked")
                    T _f(int fieldIndex, IntSupplier value) {
                        return _f(fieldIndex, value.getAsInt());
                    }
                }
                """);
    }

    static void generateBuilderCode(String parentName, String indent, List<TreeNode> tree, PrintStream out) {
        for (TreeNode n : tree) {
            switch (n.token().type()) {
                case MESSAGE, FIELD -> {
                    out.println();
                    for (String c : n.comments()) out.println(indent + '/' + c);
                    String name = snakeToCamelCase(n.token().matcher().group("name"));
                    if (n.token().type() == TokenType.MESSAGE) {
                        out.println(indent + "public static final class " + name + " extends " + PROTOGEN_BUILDER_CLASS + "<" + name + "> {");
                        generateBuilderCode(name, indent + "    ", n.nested(), out);
                        out.println(indent + "}");
                    } else {
                        String type = n.token().matcher().group("type");
                        type = switch (type) {
                            case "string" -> "String";
                            case "int32" -> "int";
                            case "int64" -> "long";
                            case "uint64" -> "long";
                            case "bytes" -> "byte[]";
                            default -> type;
                        };
                        if (Character.isLowerCase(type.charAt(0)) && !type.equals("byte[]") && n.token().matcher().group("flag").equals("repeated ")) {
                            type += "...";
                        }
                        String index = n.token().matcher().group("index");
                        out.println(indent + "public " + parentName + " " + name + "(" + type + " " + name + ") {return _f(" + index + ", " + name + ");}");
                    }
                }
            }
        }
    }

    static void generateModel(List<TreeNode> tree, PrintStream out) {
        out.print(COPYRIGHT_NOTICE);
        out.println("package " + PROTOGEN_PACKAGE + ";");
        out.print("""

                import java.io.RandomAccessFile;
                import java.lang.annotation.ElementType;
                import java.lang.annotation.Retention;
                import java.lang.annotation.RetentionPolicy;
                import java.lang.annotation.Target;
                import java.lang.reflect.ParameterizedType;
                import java.lang.reflect.RecordComponent;
                import java.nio.ByteBuffer;
                import java.nio.ByteOrder;
                import java.nio.channels.FileChannel;
                import java.util.ArrayList;
                import java.util.Arrays;
                import java.util.List;
                import java.util.function.IntSupplier;
                import java.util.function.Supplier;

                """);
        out.println("import " + PROTOGEN_PACKAGE + "." + PROTOGEN_CONSTANTS_CLASS + ".*;");
        out.print("""

                // Generated from onnx.in.proto
                """);
        out.println("public sealed interface " + PROTOGEN_MODEL_CLASS + " {");
        generateModelCode("    ", tree, out);
        out.print("""

                    // Implementation


                    @Retention(RetentionPolicy.RUNTIME)
                    @Target(ElementType.RECORD_COMPONENT)
                    @interface f {
                        int value();
                    }

                    private static long decodeVarint(ByteBuffer data) {
                        long i, shift = 0, value = 0;
                        do {
                            value |= ((i = data.get()) & 0x7f) << shift;
                            shift += 7;
                        } while ((i & 0x80) != 0);
                        return value;
                    }

                    private static int countVarInts(ByteBuffer data) {
                        long end  = decodeVarint(data);
                        int start = data.position();
                        end += start;
                        int count = 0;
                        while (data.position() < end) {
                            if ((data.get() & 0x80) == 0) count++;
                        }
                        data.position(start);
                        return count;
                    }

                    private static int[] readPackedInts(ByteBuffer data) {
                        var ret = new int[countVarInts(data)];
                        for (int i = 0; i < ret.length; i++) {
                            ret[i] = (int)decodeVarint(data);
                        }
                        return ret;
                    }

                    private static long[] readPackedLongs(ByteBuffer data) {
                        var ret = new long[countVarInts(data)];
                        for (int i = 0; i < ret.length; i++) {
                            ret[i] = decodeVarint(data);
                        }
                        return ret;
                    }

                    private static float[] readPackedFloats(ByteBuffer data) {
                        var ret = new float[(int)(decodeVarint(data)/4)];
                        for (int i = 0; i < ret.length; i++) {
                            ret[i] = data.getFloat();
                        }
                        return ret;
                    }

                    private static double[] readPackedDoubles(ByteBuffer data) {
                        var ret = new double[(int)(decodeVarint(data)/8)];
                        for (int i = 0; i < ret.length; i++) {
                            ret[i] = data.getDouble();
                        }
                        return ret;
                    }

                    private static byte[] readBytes(ByteBuffer data) {
                        var bytes = new byte[(int)decodeVarint(data)];
                        data.get(bytes);
                        return bytes;
                    }

                    private static Object readData(Class<?> baseType, boolean packed, ByteBuffer bb) {
                        if (baseType == Integer.class) {
                            return (int)decodeVarint(bb);
                        } else if (baseType == int[].class) {
                            return packed ? readPackedInts(bb) : new int[]{(int)decodeVarint(bb)};
                        } else if (baseType == Long.class) {
                            return decodeVarint(bb);
                        } else if (baseType == long[].class) {
                            return packed ? readPackedLongs(bb) : new long[]{decodeVarint(bb)};
                        } else if (baseType == Float.class) {
                            return bb.getFloat();
                        } else if (baseType == float[].class) {
                            return packed ? readPackedFloats(bb) : new float[] {bb.getFloat()};
                        } else if (baseType == Double.class) {
                            return bb.getDouble();
                        } else if (baseType == double[].class) {
                            return packed ? readPackedDoubles(bb) : new double[] {bb.getDouble()};
                        } else if (baseType == byte[].class) {
                            return readBytes(bb);
                        } else if (baseType == String.class) {
                            return new String(readBytes(bb));
                        } else if (baseType.getEnclosingClass() == OnnxConstants.class) {
                            int value = (int)decodeVarint(bb);
                            for (Object cs : baseType.getEnumConstants()) {
                                if (cs instanceof IntSupplier is && is.getAsInt() == value) {
                                    return cs;
                                }
                            }
                            throw new IllegalArgumentException(baseType.toString());
                        } else {
                            var size = decodeVarint(bb);
                            int limit = bb.limit();
                            var data = readFrom((Class<Record>)baseType, bb.limit(bb.position() + (int)size));
                            bb.limit(limit);
                            return data;
                        }
                    }

                    private static int getRecordFieldIndex(RecordComponent[] rcs, int fieldIndex) {
                        for (int i = 0; i < rcs.length; i++) {
                            if (rcs[i].getAnnotation(f.class).value() == fieldIndex) {
                                return i;
                            }
                        }
                        throw new IllegalArgumentException("Field index " + fieldIndex + " not found in " + rcs[0].getDeclaringRecord());
                    }

                    private static <T> T readFrom(Class<T> type, ByteBuffer bb) {
                        Object[] fieldsData = new Object[type.getRecordComponents().length];
                        while (bb.remaining() > 0) {
                            long tag = decodeVarint(bb);
                            RecordComponent[] rcs = type.getRecordComponents();
                            int rfi = getRecordFieldIndex(rcs, (int)tag >> 3);
                            boolean packed = (tag & 7) == 2;
                            RecordComponent rc = rcs[rfi];
                            Class<?> rcType = rc.getType();
                            if (rcType == List.class) {
                                List list;
                                if (fieldsData[rfi] instanceof List l) {
                                    list = l;
                                } else {
                                    list = new ArrayList();
                                    fieldsData[rfi] = list;
                                }
                                Class baseType = (Class)((ParameterizedType)rc.getGenericType()).getActualTypeArguments()[0];
                                list.add(readData(baseType, packed, bb));
                            } else {
                                fieldsData[rfi] = readData(rcType, packed, bb);
                            }
                        }
                        try {
                            return (T)type.getDeclaredConstructors()[0].newInstance(fieldsData);
                        } catch (ReflectiveOperationException e) {
                            throw new RuntimeException(e);
                        }
                    }

                    private static void print(StringBuilder out, int indent, String name, Object value, boolean skipBigData) throws ReflectiveOperationException {
                        if (value == null) return;
                        out.append("  ".repeat(indent)).append(name);
                        switch (value) {
                            case List l -> {
                                out.append(name.endsWith("s") ? ":" : "s:").append(System.lineSeparator());
                                for (var el : l) print(out, indent + 1, "- " + (name.endsWith("s") ? name.substring(0, name.length() - 1) : name), el, skipBigData);
                            }
                            case Record r -> {
                                out.append(':').append(System.lineSeparator());
                                for (var rc : r.getClass().getRecordComponents()) {
                                    print(out, indent + 2, rc.getName(), rc.getAccessor().invoke(r), skipBigData);
                                }
                            }
                            case byte[] a ->
                                out.append(checkSize(a.length, () -> Arrays.toString(a), skipBigData));
                            case long[] a ->
                                out.append(checkSize(a.length, () -> Arrays.toString(a), skipBigData));
                            case float[] a ->
                                out.append(checkSize(a.length, () -> Arrays.toString(a), skipBigData));
                            case double[] a ->
                                out.append(checkSize(a.length, () -> Arrays.toString(a), skipBigData));
                            case String s ->
                                out.append(": \\"").append(s).append('"').append(System.lineSeparator());
                            default ->
                                out.append(": ").append(value).append(System.lineSeparator());
                        }
                    }

                    static final int SKIP_LIMIT = 1000;

                    private static String checkSize(int size, Supplier<String> sup, boolean skipBigData) {
                        return ": " + (skipBigData && size > SKIP_LIMIT ? "# skipped " + size + " values" : sup.get()) + System.lineSeparator();
                    }

                    default String toText() {
                        return toText(true);
                    }

                    default String toText(boolean skipBigData) {
                        try {
                            var sb = new StringBuilder();
                            print(sb, 0, "OnnxModel", this, skipBigData);
                            return sb.toString();
                        } catch (ReflectiveOperationException e) {
                            throw new RuntimeException(e);
                        }
                    }

                    public static OnnxModel.ModelProto readFrom(byte[] onnxProtoModel) {
                        return readFrom(ByteBuffer.wrap(onnxProtoModel));
                    }

                    public static OnnxModel.ModelProto readFrom(ByteBuffer onnxProtoModel) {
                        return readFrom(OnnxModel.ModelProto.class, onnxProtoModel.order(ByteOrder.LITTLE_ENDIAN));
                    }

                    public static void main(String... args) throws Exception {
                        for (var fName : args) {
                            try (var in = new RandomAccessFile(fName, "r")) {
                                OnnxModel.ModelProto model = readFrom(in.getChannel().map(FileChannel.MapMode.READ_ONLY, 0, in.length()));
                                System.out.println(model.toText());
                            }
                        }
                    }
                }
                """);
    }

    static void generateModelCode(String indent, List<TreeNode> tree, PrintStream out) {
        for (TreeNode n : tree) {
            if (n.token().type() == TokenType.MESSAGE) {
                out.println();
                for (String c : n.comments()) out.println(indent + '/' + c);
                String recordName = n.token().matcher().group("name");
                out.println(indent + "public record " + recordName + " (");
                boolean first = true;
                for (TreeNode nn : n.nested()) {
                    if (nn.token().type() == TokenType.FIELD) {
                        if (first) {
                            first = false;
                        } else {
                            out.println(",");
                        }
                        out.println();
                        for (String c : nn.comments()) out.println(indent + "    /" + c);
                        String name = snakeToCamelCase(nn.token().matcher().group("name"));
                        String type = nn.token().matcher().group("type");
                        if (nn.token().matcher().group("flag").equals("repeated ")) {
                            type = switch (type) {
                                case "float" -> "List<float[]>";
                                case "double" -> "List<double[]>";
                                case "string" -> "List<String>";
                                case "int32" -> "List<int[]>";
                                case "int64", "uint64" -> "List<long[]>";
                                case "bytes" -> "List<byte[]>";
                                default -> "List<" + type + ">";
                            };
                        } else {
                            type = switch (type) {
                                case "float" -> "Float";
                                case "double" -> "Double";
                                case "string" -> "String";
                                case "int32" -> "Integer";
                                case "int64", "uint64" -> "Long";
                                case "bytes" -> "byte[]";
                                default -> type;
                            };
                        }
                        String index = nn.token().matcher().group("index");
                        out.print(indent + "    @f(" + index + ") " + type + " " + name);
                    }
                }
                out.println(") implements " + PROTOGEN_MODEL_CLASS + " {");
                generateModelCode(indent + "    ", n.nested(), out);
                out.println(indent + "}");
            }
        }
    }

    static List<TreeNode> toTree(Iterator<Token> tokens) {
        List<TreeNode> nodes = new ArrayList<>();
        List<String> comments = new ArrayList<>();
        int oneofs = 0;
        while (tokens.hasNext()) {
            Token t = tokens.next();
            switch (t.type()) {
                case COMMENT -> comments.add(t.matcher().group("comment"));
                case EMPTY -> comments.clear(); // do not merge isolated comment blocks
                case ONEOF -> oneofs++; // flat ONEOF
                case ENUM_ELEMENT, FIELD, RESERVED, SYNTAX, PACKAGE -> {
                    if (t.matcher().group("comment") instanceof String c) comments.add(c);
                    nodes.add(new TreeNode(comments, t, List.of()));
                    comments = new ArrayList<>();
                }
                case ENUM, MESSAGE -> {
                    nodes.add(new TreeNode(comments, t, toTree(tokens)));
                    comments = new ArrayList<>();
                }
                case END -> {
                    if (oneofs-- == 0) return nodes;
                }
            }
        }
        return nodes;
    }

    static final Pattern SNAKE = Pattern.compile("_([a-z])");

    static String snakeToCamelCase(String name) {
        return SNAKE.matcher(name).replaceAll(mr -> mr.group(1).toUpperCase());
    }

    public static void main(String[] args) throws Exception {
        List<TreeNode> tree = toTree(Files.lines(Path.of("opgen/onnx.in.proto")).map(ProtoGen::lineToToken).iterator());
        try (var constants = new PrintStream(new FileOutputStream(SOURCES_PATH + PROTOGEN_CONSTANTS_CLASS + ".java"))) {
            generateConstants(tree, constants);
        }
        try (var builder = new PrintStream(new FileOutputStream(SOURCES_PATH + PROTOGEN_BUILDER_CLASS + ".java"))) {
            generateBuilder(tree, builder);
        }
        try (var model = new PrintStream(new FileOutputStream(SOURCES_PATH + PROTOGEN_MODEL_CLASS + ".java"))) {
            generateModel(tree, model);
        }
    }
}
