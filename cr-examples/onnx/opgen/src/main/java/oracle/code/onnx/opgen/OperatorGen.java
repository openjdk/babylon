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

package oracle.code.onnx.opgen;

import jdk.incubator.code.TypeElement;
import oracle.code.onnx.OpSchema;
import oracle.code.onnx.Tensor;

import java.io.*;
import java.nio.file.Path;
import java.util.*;
import java.util.stream.Collectors;

import static java.util.Comparator.comparing;
import static java.util.stream.Collectors.*;

public class OperatorGen {

    final SortedMap<String, SortedSet<OpSchema>> schemas;

    OperatorGen(List<OpSchema> schemas) {
        this.schemas = schemas.stream().collect(groupingBy(
                OpSchema::name,
                TreeMap::new,
                toCollection(() -> new TreeSet<>(comparing(OpSchema::since_version).reversed())
                )));
    }

    static final String ONNX_PACKAGE = "oracle.code.onnx";
    static final String ONNX_OPERATORS_CLASS = "OnnxOperators";

    void genOpClass(Path dir) throws IOException {
        OutputStreamWriter osw = new OutputStreamWriter(
                new FileOutputStream(dir.resolve(ONNX_OPERATORS_CLASS + ".java").toFile()));
        genOpClass(osw);
    }

    void genOpClass(Writer w_) throws IOException {
        IndentWriter w = new IndentWriter(w_);

        w.write("""
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
        w.write("// Auto-generated from ONNX op schema\n");
        w.write("\n");
        w.write("package " + ONNX_PACKAGE + ";\n");
        w.write("\n");
        w.write("""
                import oracle.code.onnx.ir.OnnxOps;
                
                import java.util.Optional;
                import java.util.List;
                import java.util.Map;
                """);
        w.write("\n");

        w.write("@SuppressWarnings({\"unchecked\", \"OptionalUsedAsFieldOrParameterType\"})\n");
        w.write("public final class " + ONNX_OPERATORS_CLASS + " extends ExplicitOnnxOperators {\n");

        w.in();

        w.write("\n");
        w.write("private " + ONNX_OPERATORS_CLASS + "() {}\n");
        w.write("\n");

        for (OpSchema s : schemas.values().stream().map(SortedSet::getFirst).toList()) {
            if (skip(s)) {
                System.out.println("Skipping " + s.name());
                continue;
            }

            genMethod(w, s);
            w.write("\n");
        }
        w.out();

        w.write("}\n");
        w.flush();
    }

    private boolean skip(OpSchema s) {
        return s.attributes().stream().anyMatch(a ->
                a.type() == OpSchema.AttributeType.GRAPH ||
                        a.type() == OpSchema.AttributeType.GRAPHS);
    }

    private void genMethod(IndentWriter w, OpSchema s) throws IOException {
        Map<String, TypeElement.ExternalizedTypeElement> javaTypeConstraints = javaTypes(typeConstraintMap(s));
        boolean twoOrMoreResults = s.max_output() > 1 && s.outputs().size() > 1;
        if (twoOrMoreResults) {
            System.out.println(s.name());

            w.write("public record " + s.name() + "Result");
            if (!s.type_constraints().isEmpty()) {
                boolean first = true;
                w.write("<");
                for (OpSchema.TypeConstraintParam typeConstraint : s.type_constraints()) {
                    if (!first) {
                        w.write(", ");
                    }
                    w.write(typeConstraint.type_param_str());
                    first = false;
                }
                w.write(">");
            }

            w.write("(");
            boolean first = true;
            for (OpSchema.FormalParameter outParam : s.outputs()) {
                if (!first) {
                    w.write(", ");
                }

                TypeElement.ExternalizedTypeElement outputType = javaTypeConstraints
                        .computeIfAbsent(outParam.type_str(),
                                ts -> javaType("?", parseTypeString(ts)));

                w.write(outputType.toString());
                w.write(" " + outParam.name());

                first = false;
            }

            w.write(") { }\n");
        }

        w.write("public static ");

        if (!s.type_constraints().isEmpty()) {
            boolean first = true;
            w.write("<");
            for (OpSchema.TypeConstraintParam typeConstraint : s.type_constraints()) {
                if (!first) {
                    w.write(", ");
                }
                w.write(typeConstraint.type_param_str());
                first = false;
            }
            w.write(">");
            w.write(" ");
        }

        // @@@ Multiple output parameters - need to return tuple/record
        final TypeElement.ExternalizedTypeElement outputType;
        if (s.min_output() == 1 && s.max_output() == 1) {
            OpSchema.FormalParameter outParam = s.outputs().getFirst();

            outputType = javaTypeConstraints.computeIfAbsent(outParam.type_str(),
                    ts -> javaType("?", parseTypeString(ts)));
            w.write(outputType.toString());
        } else if (s.min_output() == 0 && s.max_output() == 1) {
            // This does not occur
            throw new UnsupportedOperationException();
        } else if (s.outputs().size() == 1) {
            OpSchema.FormalParameter outParam = s.outputs().getFirst();
            assert outParam.option() == OpSchema.FormalParameterOption.Variadic;

            outputType = new TypeElement.ExternalizedTypeElement("List",
                    List.of(javaTypeConstraints.computeIfAbsent(outParam.type_str(),
                            ts -> javaType("?", parseTypeString(ts)))));
            w.write(outputType.toString());
        } else {
            assert twoOrMoreResults;

            outputType = new TypeElement.ExternalizedTypeElement(s.name() + "Result", List.of());
            w.write(outputType.toString());
            if (!s.type_constraints().isEmpty()) {
                boolean first = true;
                w.write("<");
                for (OpSchema.TypeConstraintParam typeConstraint : s.type_constraints()) {
                    if (!first) {
                        w.write(", ");
                    }
                    w.write(typeConstraint.type_param_str());
                    first = false;
                }
                w.write(">");
            }
        }
        w.write(" ");
        w.write(s.name() + "(");

        boolean first = true;
        for (OpSchema.FormalParameter inParam : s.inputs()) {
            if (!first) {
                w.write(", ");
            }

            final TypeElement.ExternalizedTypeElement inputType = javaTypeConstraints
                    .computeIfAbsent(inParam.type_str(),
                            ts -> javaType("?", parseTypeString(ts)));
            switch (inParam.option()) {
                case Single -> {
                    w.write(inputType.toString());
                }
                case Optional -> {
                    w.write("Optional<");
                    w.write(inputType.toString());
                    w.write(">");
                }
                case Variadic -> {
                    w.write("List<");
                    w.write(inputType.toString());
                    w.write(">");
                }
            }
            w.write(" ");
            w.write(inParam.name());

            first = false;
        }

        for (OpSchema.Attribute attribute : s.attributes()) {
            if (!first) {
                w.write(", ");
            }

            OpSchema.AttributeType aType = attribute.type();
            String typeString = switch (aType) {
                default -> {
                    if (attribute.required()) {
                        yield aType.type().getSimpleName();
                    } else {
                        yield toBoxType(aType.type()).getSimpleName();
                    }
                }
            };
            if (attribute.required()) {
                w.write(typeString);
            } else {
                w.write("Optional<");
                w.write(typeString);
                w.write(">");
            }
            w.write(" ");
            w.write(attribute.name());

            first = false;
        }

        w.write(") {\n");
        w.in();

        w.write("Object result = OnnxInterpreter.interpret(");
        w.write("OnnxOps." + s.name() + ".class");
        w.write(", ");

        w.write("List.of(");
        first = true;
        for (OpSchema.FormalParameter inParam : s.inputs()) {
            if (!first) {
                w.write(", ");
            }

            w.write(inParam.name());
            first = false;
        }
        w.write(")");
        w.write(", ");

        w.write("List.of(");
        first = true;
        for (OpSchema.Attribute attribute : s.attributes()) {
            if (!first) {
                w.write(", ");
            }

            w.write(attribute.name());
            first = false;
        }
        w.write(")");
        w.write(");\n");

        if (twoOrMoreResults) {
            w.write("Object[] resultArray = (Object[]) result;\n");
            w.write("return new " + s.name() + "Result");
            if (!s.type_constraints().isEmpty()) {
                w.write("<>");
            }
            w.write("(");
            first = true;
            for (int i = 0; i < s.outputs().size(); i++) {
                if (!first) {
                    w.write(", ");
                }

                w.write("(");
                //
                final TypeElement.ExternalizedTypeElement t = javaTypeConstraints
                        .computeIfAbsent(s.outputs().get(i).type_str(),
                                ts -> javaType("?", parseTypeString(ts)));
                w.write(t.toString());
                w.write(")");
                w.write("resultArray[" + i + "]");
                first = false;
            }
            w.write(");\n");
        } else {
            w.write("return (" + outputType + ") result;\n");
        }
        w.out();
        w.write("}\n");
    }

    static Map<String, TypeElement.ExternalizedTypeElement> javaTypes(Map<String, TypeElement.ExternalizedTypeElement> tcm) {
        return tcm.entrySet().stream().collect(Collectors.toMap(
                e -> e.getKey(),
                e -> javaType(e.getKey(), e.getValue())));
    }

    static TypeElement.ExternalizedTypeElement javaType(String typeVariable, TypeElement.ExternalizedTypeElement ete) {
        String javaIdentifier = switch (ete.identifier()) {
            case "seq" -> "List";
            case "sequence" -> "List";
            case "map" -> "Map";
            case "optional" -> "Optional";
            case "tensor" -> "Tensor";
            case "?" -> typeVariable;

            default -> {
                Tensor.ElementType elementType = Tensor.ElementType.fromOnnxName(ete.identifier());
                Class<?> type = elementType.type();
                if (type.isPrimitive()) {
                    yield toBoxType(type).getSimpleName();
                } else {
                    yield type.getSimpleName();
                }
            }
        };

        if (ete.identifier().equals("map") &&
                ete.arguments().stream().allMatch(t -> t.identifier().equals("?"))) {
            return new TypeElement.ExternalizedTypeElement(javaIdentifier,
                    ete.arguments().stream().map(c -> javaType("?", c)).toList());
        }

        return new TypeElement.ExternalizedTypeElement(javaIdentifier,
                ete.arguments().stream().map(c -> javaType(typeVariable, c)).toList());
    }

    static Map<String, TypeElement.ExternalizedTypeElement> typeConstraintMap(OpSchema s) {
        return s.type_constraints().stream().collect(toMap(
                tc -> tc.type_param_str(),
                tc -> tc.allowed_type_strs().stream().map(OperatorGen::parseTypeString).reduce(OperatorGen::lub).orElseThrow()));
    }

    static TypeElement.ExternalizedTypeElement lub(TypeElement.ExternalizedTypeElement a,
                                                   TypeElement.ExternalizedTypeElement b) {
        if (!a.identifier().equals(b.identifier())) {
            return new TypeElement.ExternalizedTypeElement("?", List.of());
        }

        assert a.arguments().size() == b.arguments().size();

        List<TypeElement.ExternalizedTypeElement> children = new ArrayList<>();
        for (int i = 0; i < a.arguments().size(); i++) {
            children.add(lub(a.arguments().get(i), b.arguments().get(i)));
        }

        return new TypeElement.ExternalizedTypeElement(a.identifier(), children);
    }

    static TypeElement.ExternalizedTypeElement parseTypeString(String type_str) {
        return TypeElement.ExternalizedTypeElement.ofString(
                type_str.replace('(', '<').replace(')', '>'));
    }

    static Class<?> toBoxType(Class<?> pc) {
        if (pc == byte.class) {
            return Byte.class;
        } else if (pc == short.class) {
            return Short.class;
        } else if (pc == int.class) {
            return Integer.class;
        } else if (pc == long.class) {
            return Long.class;
        } else if (pc == float.class) {
            return Float.class;
        } else if (pc == double.class) {
            return Double.class;
        } else if (pc == boolean.class) {
            return Boolean.class;
        } else {
            return pc;
        }
    }

    public static void main(String[] args) throws Exception {
        List<OpSchema> schemas = OpSchemaParser.parse(Path.of(
                "opgen/onnx-schema.json"));
        OperatorGen oprGen = new OperatorGen(schemas);

        oprGen.genOpClass(Path.of("src/main/java/oracle/code/onnx"));
    }
}
