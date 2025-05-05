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
                    String name = n.token().matcher().group("name");
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

    public static void main(String[] args) throws Exception {
        List<TreeNode> tree = toTree(Files.lines(Path.of("opgen/onnx.in.proto")).map(ProtoGen::lineToToken).iterator());
        try (var constants = new PrintStream(new FileOutputStream(SOURCES_PATH + PROTOGEN_CONSTANTS_CLASS + ".java"))) {
            generateConstants(tree, constants);
        }
        try (var builder = new PrintStream(new FileOutputStream(SOURCES_PATH + PROTOGEN_BUILDER_CLASS + ".java"))) {
            generateBuilder(tree, builder);
        }
//        try (var model = new PrintStream(new FileOutputStream(SOURCES_PATH + PROTOGEN_MODEL_CLASS + ".java"))) {
//            generateModel(source, model);
//        }
    }
}
