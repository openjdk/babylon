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
import java.util.regex.Matcher;
import java.util.regex.Pattern;

public class ProtoGen {

    static final String PROTOGEN_PACKAGE = "oracle.code.onnx.proto";
    static final String PROTOGEN_BUILDER_CLASS = "OnnxBuilder";

    enum BlockType { ENUM, MESSAGE, ONEOF }

    record Block(BlockType type, String name) {}

    static final Pattern ENUM_PAT = Pattern.compile("enum\\s+([^\\s]+)\\s*\\{");
    static final Pattern MESSAGE_PAT = Pattern.compile("message\\s+([^\\s]+)\\s*\\{");
    static final Pattern ENUM_EL_PAT = Pattern.compile("([^\\s]+)\\s*=\\s*([^\\s]+);");
    static final Pattern MESSAGE_EL_PAT = Pattern.compile("(optional |repeated |)\\s*([^\\s]+)\\s+([^\\s]+)\\s*=\\s*([0-9]+)\\s*(\\[.*\\])?\\s*;?");

    static void generateBuilder(String source, PrintStream out) {
        out.print("""
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

                    """);
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
        var stack = new ArrayDeque<Block>();
        int indent = 1;
        for (String line : source.split("\\R")) {
            line = line.trim();
            if (line.startsWith("//")) {
                docLines.add(line);
            } else {
                if (line.isEmpty() || line.startsWith("syntax") || line.startsWith("package") || line.startsWith("reserved")) {
                    docLines.clear();
                } else if (line.startsWith("}")) {
                    docLines.clear();
                    var b = stack.pop();
                    switch (b.type) {
                        case ENUM -> {
                            out.println("    ".repeat(indent) + ";");
                            out.println();
                            out.println("    ".repeat(indent) +"final int value;");
                            out.println();
                            out.println("    ".repeat(indent) + b.name() + "(int value) {");
                            out.println("    ".repeat(indent) + "    this.value = value;");
                            out.println("    ".repeat(indent) + "}");
                            out.println();
                            out.println("    ".repeat(indent) + "@Override");
                            out.println("    ".repeat(indent) + "public int getAsInt() {");
                            out.println("    ".repeat(indent) + "    return value;");
                            out.println("    ".repeat(indent) + "}");
                            out.println("    ".repeat(--indent) + "}");
                            out.println("");
                        }
                        case MESSAGE -> {
                            out.println("    ".repeat(--indent) + "}");
                            out.println("");
                        }
                        case ONEOF -> {}
                    }
                } else {
                    int i = line.indexOf("//");
                    if (i > 0) {
                        docLines.add(line.substring(i));
                        line = line.substring(0, i - 1).trim();
                    }
                    for (String dl : docLines) out.println("    ".repeat(indent) + '/' + dl);
                    docLines.clear();
                    Matcher m;
                    if ((m = ENUM_PAT.matcher(line)).matches()) {
                        stack.push(new Block(BlockType.ENUM, m.group(1)));
                        out.println("    ".repeat(indent++) + "public enum " + m.group(1) + " implements IntSupplier {");
                        out.println();
                    } else if ((m = MESSAGE_PAT.matcher(line)).matches()) {
                        stack.push(new Block(BlockType.MESSAGE, m.group(1)));
                        out.println("    ".repeat(indent++) + "public static final class " + m.group(1) + " extends " + PROTOGEN_BUILDER_CLASS + "<" + m.group(1) + "> {");
                        out.println();
                    } else if (line.startsWith("oneof")) {
                        stack.push(new Block(BlockType.ONEOF, stack.peek().name()));
                    } else {
                        var b = stack.peek();
                        if (b.type == BlockType.ENUM) {
                            m = ENUM_EL_PAT.matcher(line);
                            if (!m.matches()) throw new IllegalArgumentException(line);
                            out.println("    ".repeat(indent) + m.group(1) + "(" + m.group(2) + "),");
                            out.println();
                        } else {
                            m = MESSAGE_EL_PAT.matcher(line);
                            if (!m.matches()) throw new IllegalArgumentException(line);
                            String type = m.group(2);
                            type = switch (type) {
                                case "string" -> "String";
                                case "int32" -> "int";
                                case "int64" -> "long";
                                case "uint64" -> "long";
                                case "bytes" -> "byte[]";
                                default -> type;
                            };
                            if (Character.isLowerCase(type.charAt(0)) && !type.equals("byte[]") && m.group(1).equals("repeated ")) {
                                type += "...";
                            }
                            String name = m.group(3);
                            String index = m.group(4);
                            out.println("    ".repeat(indent) + "public " + stack.peek().name() + " " + name + "(" + type + " " + name + ") {return _f(" + index + ", " + name + ");}");
                            out.println();
                        }
                    }
                }
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
            generateBuilder(Files.readString(Path.of("opgen/onnx.in.proto")), out);
        }
    }
}
