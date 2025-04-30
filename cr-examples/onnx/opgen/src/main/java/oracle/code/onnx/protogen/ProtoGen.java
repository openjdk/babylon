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
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.List;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

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
    static final String PROTOGEN_BUILDER_CLASS = "OnnxBuilder";

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

    static Token parse(String line) {
        for (var tt : TokenType.values()) {
            var m = tt.pattern.matcher(line);
            if (m.matches()) {
                return new Token(tt, m);
            }
        }
        throw new IllegalArgumentException(line);
    }

    static void generateBuilder(List<String> source, PrintStream out) {
        out.print(COPYRIGHT_NOTICE);
        out.println("package " + PROTOGEN_PACKAGE + ";");
        out.print("""

                import java.io.ByteArrayOutputStream;
                import java.nio.charset.StandardCharsets;
                import java.util.List;
                import java.util.function.BiConsumer;
                import java.util.function.IntSupplier;

                // Generated from onnx.in.proto
                """);
        out.println("public sealed class " + PROTOGEN_BUILDER_CLASS + "<T extends " + PROTOGEN_BUILDER_CLASS + "> {");
        out.println();
        var docLines = new ArrayList<String>();
        var stack = new ArrayDeque<Token>();
        String indent = "    ";
        for (String line : source) {
            Token token = parse(line);
            switch (token.type()) {
                case COMMENT -> docLines.add(token.matcher().group("comment"));
                case EMPTY, SYNTAX, PACKAGE, RESERVED -> docLines.clear();
                case END -> {
                    docLines.clear();
                    var b = stack.pop();
                    switch (b.type) {
                        case ENUM -> {
                            out.println(indent + ";");
                            out.println();
                            out.println(indent +"final int value;");
                            out.println();
                            out.println(indent + b.matcher().group("name") + "(int value) {");
                            out.println(indent + "    this.value = value;");
                            out.println(indent + "}");
                            out.println();
                            out.println(indent + "@Override");
                            out.println(indent + "public int getAsInt() {");
                            out.println(indent + "    return value;");
                            out.println(indent + "}");
                            indent = " ".repeat(indent.length() - 4);
                            out.println(indent + "}");
                            out.println();
                        }
                        case MESSAGE -> {
                            indent = " ".repeat(indent.length() - 4);
                            out.println(indent + "}");
                            out.println();
                        }
                        case ONEOF -> {}
                    }
                }
                case ENUM, MESSAGE, ENUM_ELEMENT, FIELD -> {
                    if (token.matcher().group("comment") instanceof String cmt) {
                        docLines.add(cmt);
                    }
                    for (String dl : docLines) out.println(indent + '/' + dl);
                    docLines.clear();
                    String name = token.matcher().group("name");
                    switch (token.type) {
                        case ENUM -> {
                            out.println(indent + "public enum " + name + " implements IntSupplier {");
                            indent += "    ";
                            stack.push(token);
                        }
                        case MESSAGE -> {
                            out.println(indent + "public static final class " + name + " extends " + PROTOGEN_BUILDER_CLASS + "<" + name + "> {");
                            indent += "    ";
                            stack.push(token);
                        }
                        case ENUM_ELEMENT ->
                            out.println(indent + token.matcher().group("name") + "(" + token.matcher().group("value") + "),");
                        case FIELD -> {
                            String type = token.matcher().group("type");
                            type = switch (type) {
                                case "string" -> "String";
                                case "int32" -> "int";
                                case "int64" -> "long";
                                case "uint64" -> "long";
                                case "bytes" -> "byte[]";
                                default -> type;
                            };
                            if (Character.isLowerCase(type.charAt(0)) && !type.equals("byte[]") && token.matcher().group("flag").equals("repeated ")) {
                                type += "...";
                            }
                            String index = token.matcher().group("index");
                            String parentName = stack.stream().filter(t -> t.type() != TokenType.ONEOF).findFirst().orElseThrow().matcher().group("name");
                            out.println(indent + "public " + parentName + " " + name + "(" + type + " " + name + ") {return _f(" + index + ", " + name + ");}");
                        }
                    }
                    out.println();
                }
                case ONEOF -> stack.push(token);
            }
        }
        out.print("""
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

    public static void main(String[] args) throws Exception {
        try (var out = new PrintStream(new FileOutputStream("src/main/java/oracle/code/onnx/proto/" + PROTOGEN_BUILDER_CLASS + ".java"))) {
            generateBuilder(Files.readAllLines(Path.of("opgen/onnx.in.proto")), out);
        }
    }
}
