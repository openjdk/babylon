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

import java.io.*;
import java.nio.file.Path;
import java.util.*;

import static java.util.Comparator.comparing;
import static java.util.stream.Collectors.groupingBy;
import static java.util.stream.Collectors.toCollection;

public class OpGen {

    final SortedMap<String, SortedSet<OpSchema>> schemas;

    OpGen(List<OpSchema> schemas) {
        this.schemas = schemas.stream().collect(groupingBy(
                OpSchema::name,
                TreeMap::new,
                toCollection(() -> new TreeSet<>(comparing(OpSchema::since_version).reversed())
                )));
    }

    static final String ONNX_PACKAGE = "oracle.code.onnx";
    static final String ONNX_IR_PACKAGE = ONNX_PACKAGE + ".ir";
    static final String ONNX_OPS_CLASS = "OnnxOps";

    void genOpsClass(Path dir) throws IOException {
        OutputStreamWriter osw = new OutputStreamWriter(
                new FileOutputStream(dir.resolve(ONNX_OPS_CLASS + ".java").toFile()));
        genOpsClass(osw);
    }

    void genOpsClass(Writer w_) throws IOException {
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
        w.write("package " + ONNX_IR_PACKAGE + ";\n");
        w.write("\n");
        w.write("""
                import jdk.incubator.code.*;
                import jdk.incubator.code.op.OpFactory;

                import java.util.*;
                """);
        w.write("\n");

        w.write("@SuppressWarnings({\"OptionalUsedAsFieldOrParameterType\", \"unused\", \"SequencedCollectionMethodCanBeUsed\"})\n");
        w.write("public final class " + ONNX_OPS_CLASS + " {\n");

        w.in();

        w.write("\n");
        w.write("private " + ONNX_OPS_CLASS + "() {}\n");
        w.write("\n");

        w.out();
        for (OpSchema s : schemas.values().stream().map(SortedSet::getFirst).toList()) {
            try {
                if (skip(s)) {
                    System.out.println("Skipping " + s.name());
                    continue;
                }

                String g = genOpClass(s);
                w.write(g);
                w.write("\n");
            } catch (UnsupportedOperationException e) {
                System.err.println("Skipping " + s.name() + ": " + e.getMessage());
            }
        }

        w.write("}\n");
        w.flush();
    }

    private boolean skip(OpSchema s) {
        return s.attributes().stream().anyMatch(a ->
                a.type() == OpSchema.AttributeType.GRAPH ||
                        a.type() == OpSchema.AttributeType.GRAPHS);
    }

    private String genOpClass(OpSchema s) throws IOException {
        StringWriter sw = new StringWriter();
        IndentWriter w = new IndentWriter(sw, 4);

        w.write("@OpFactory.OpDeclaration(" + s.name() + ".NAME)\n");
        w.write("public static final class " + s.name() + " extends OnnxOp {\n");
        w.in();

        w.write("public static final String NAME = \"" + s.name() + "\";\n");
        w.write("\n");

        genAttributeEnum(w, s);

        Map<String, List<TypeElement.ExternalizedTypeElement>> typeConstraints =
                genTypeConstraintEnum(w, s);

        genInputParameterEnum(w, s, typeConstraints);

        genOutputParameterEnum(w, s, typeConstraints);

        genSchemaInstance(w, s);

        genConstructors(w, s);

        genMethods(w, s);

        w.out();
        w.write("}\n");
        w.write("\n");

        genFactoryMethod(w, s);

        return sw.toString();
    }

    private void genAttributeEnum(IndentWriter w, OpSchema s) throws IOException {
        if (s.attributes().isEmpty()) {
            w.write("public enum Attribute implements OnnxAttribute.None { }\n");
            w.write("\n");
            return;
        }

        w.write("public enum Attribute implements OnnxAttribute {\n");
        w.in();

        for (OpSchema.Attribute a : s.attributes()) {
            w.write(a.name());
            w.write("(");
            w.write(toBoxType(a.type().type()).getSimpleName() + ".class");
            w.write(", ");
            w.write(Boolean.toString(!a.required()));
            w.write(", ");

            if (!a.required() && a.default_value() != null) {
                switch (a.type()) {
                    case FLOAT -> w.write(Float.toString((Float) a.default_value()) + "f");
                    case INT -> w.write(Integer.toString((Integer) a.default_value()));
                    case STRING -> w.write("\"" + a.default_value() + "\"");
                    default -> throw new IllegalStateException();
                }
            } else {
                w.write("null");
            }

            w.write("),\n");
        }
        w.write(";\n");
        w.write("\n");

        w.write("""
                    final Class<?> t;
                    final boolean optional;
                    final Object defaultValue;

                    Attribute(Class<?> type, boolean optional, Object defaultValue) {
                        this.t = type;
                        this.optional = optional;
                        this.defaultValue = defaultValue;
                        assert optional || defaultValue == null;
                    }

                    public Class<?> type() {
                        return t;
                    }

                    public boolean isOptional() {
                        return optional;
                    }

                    public Object defaultValue() {
                        return defaultValue;
                    }
                """);

        w.out();
        w.write("}\n");
        w.write("\n");
    }

    private Map<String, List<TypeElement.ExternalizedTypeElement>> genTypeConstraintEnum(IndentWriter w, OpSchema s) throws IOException {
        if (s.type_constraints().isEmpty()) {
            w.write("public enum TypeConstraint implements OnnxTypeConstraint.None { }\n");
            w.write("\n");
            return Map.of();
        }

        Map<String, List<TypeElement.ExternalizedTypeElement>> typeConstraints = new HashMap<>();

        w.write("public enum TypeConstraint implements OnnxTypeConstraint {\n");
        w.in();

        for (OpSchema.TypeConstraintParam tcp : s.type_constraints()) {
            List<TypeElement.ExternalizedTypeElement> types = tcp.allowed_type_strs().stream()
                    .map(OpGen::parseTypeString)
                    .toList();
            typeConstraints.put(tcp.type_param_str(), types);

            w.write(tcp.type_param_str() + "(");

            w.write("new OnnxType.TypeVariable(");
            w.write("\"" + tcp.type_param_str() + "\", ");
            w.write("List.of(");
            genTypes(w, types);
            w.write(")");
            w.write(")");

            w.write("),\n");
        }
        w.write(";\n");
        w.write("\n");

        w.write("""
                final OnnxType.TypeVariable typeVariable;

                TypeConstraint(OnnxType.TypeVariable typeVariable) {
                    assert typeVariable.name().equals(name());
                    this.typeVariable = typeVariable;
                }

                @Override
                public OnnxType.TypeVariable typeVariable() {
                    return typeVariable;
                }
                """);

        w.out();
        w.write("}\n");
        w.write("\n");

        return typeConstraints;
    }

    private void genInputParameterEnum(IndentWriter w, OpSchema s,
                                       Map<String, List<TypeElement.ExternalizedTypeElement>> typeConstraints) throws IOException {
        if (s.inputs().isEmpty()) {
            w.write("public enum InputParameter implements OnnxParameter.None { }\n");
            w.write("\n");
            return;
        }

        w.write("public enum InputParameter implements OnnxParameter {\n");
        w.in();

        for (OpSchema.FormalParameter input : s.inputs()) {
            w.write(input.name() + "(");

            if (typeConstraints.containsKey(input.type_str())) {
                w.write("TypeConstraint." + input.type_str() + ".typeVariable()");
            } else {
                TypeElement.ExternalizedTypeElement type = parseTypeString(input.type_str());
                genType(w, type);
            }
            w.write(", ");
            w.write("Quantifier.");
            switch (input.option()) {
                case Single -> {
                    w.write("REQUIRED");
                }
                case Optional -> {
                    w.write("OPTIONAL");
                }
                case Variadic -> {
                    w.write("VARIADIC");
                }
            }

            w.write("),\n");
        }
        w.write(";\n");
        w.write("\n");

        w.write("""
                final OnnxType type;
                final Quantifier quantifier;

                InputParameter(OnnxType type, Quantifier quantifier) {
                    this.type = type;
                    this.quantifier = quantifier;
                }

                @Override
                public OnnxType type() {
                    return type;
                }

                @Override
                public Quantifier quantifier() {
                    return quantifier;
                }
                """);

        w.out();
        w.write("}\n");
        w.write("\n");
    }

    private void genOutputParameterEnum(IndentWriter w, OpSchema s,
                                        Map<String, List<TypeElement.ExternalizedTypeElement>> typeConstraints) throws IOException {
        if (s.outputs().isEmpty()) {
            w.write("public enum OutputParameter implements OnnxParameter.None { }\n");
            w.write("\n");
            return;
        }

        w.write("public enum OutputParameter implements OnnxParameter {\n");
        w.in();

        for (OpSchema.FormalParameter output : s.outputs()) {
            w.write(output.name() + "(");

            if (typeConstraints.containsKey(output.type_str())) {
                w.write("TypeConstraint." + output.type_str() + ".typeVariable()");
            } else {
                TypeElement.ExternalizedTypeElement type = parseTypeString(output.type_str());
                genType(w, type);
            }
            w.write(", ");
            w.write("Quantifier.");
            switch (output.option()) {
                case Single -> {
                    w.write("REQUIRED");
                }
                case Optional -> {
                    w.write("OPTIONAL");
                }
                case Variadic -> {
                    w.write("VARIADIC");
                }
            }

            w.write("),\n");
        }
        w.write(";\n");
        w.write("\n");

        w.write("""
                final OnnxType type;
                final Quantifier quantifier;

                OutputParameter(OnnxType type, Quantifier quantifier) {
                    this.type = type;
                    this.quantifier = quantifier;
                }

                @Override
                public OnnxType type() {
                    return type;
                }

                @Override
                public Quantifier quantifier() {
                    return quantifier;
                }
                """);

        w.out();
        w.write("}\n");
        w.write("\n");
    }

    private void genSchemaInstance(IndentWriter w, OpSchema s) throws IOException {
        w.write("""
                public static final OnnxSchema SCHEMA = new OnnxSchemaRecord(
                        NAME,
                        List.of(Attribute.values()),
                        List.of(TypeConstraint.values()),
                        List.of(InputParameter.values()),
                        List.of(OutputParameter.values())
                );
                """);
        w.write("\n");
    }

    private void genConstructors(IndentWriter w, OpSchema s) throws IOException {
        w.write("public " + s.name() + "(ExternalizedOp def) {\n");
        w.in();
        w.write("super(SCHEMA, def);\n");
        w.out();
        w.write("}\n");
        w.write("\n");

        w.write(s.name() + "(" + s.name() + " that, CopyContext cc) {\n");
        w.write("    super(that, cc);\n");
        w.write("}\n");
        w.write("\n");

        w.write("@Override\n");
        w.write("public " + s.name() + " transform(CopyContext cc, OpTransformer ot) {\n");
        w.write("    return new " + s.name() + "(this, cc);\n");
        w.write("}\n");
        w.write("\n");


        w.write(s.name() + "(");

        // Result type parameter
        w.write("TypeElement resultType, ");

        boolean hasOptionalOutputs = s.outputs()
                .stream().anyMatch(o -> o.option() == OpSchema.FormalParameterOption.Optional);
        if (hasOptionalOutputs) {
            w.write("Set<OutputParameter> optionalOutputs, ");
        }

        boolean first = true;
        for (OpSchema.FormalParameter inParam : s.inputs()) {
            if (!first) {
                w.write(", ");
            }

            switch (inParam.option()) {
                case Single -> {
                    w.write("Value");
                }
                case Optional -> {
                    w.write("java.util.Optional<Value>");
                }
                case Variadic -> {
                    w.write("List<Value>");
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
                w.write("java.util.Optional<");
                w.write(typeString);
                w.write(">");
            }
            w.write(" ");
            w.write(attribute.name());

            first = false;
        }

        w.write(") {\n");
        w.in();

        w.write("super(SCHEMA, resultType, ");

        if (hasOptionalOutputs) {
            w.write("optionalOutputs, ");
        } else {
            w.write("Set.of(), ");
        }

        w.write("List.of(");
        first = true;
        for (OpSchema.FormalParameter inParam : s.inputs()) {
            if (!first) {
                w.write(", ");
            }
            w.write(inParam.name());

            first = false;
        }
        w.write("), ");

        w.write("List.of(");
        first = true;
        for (OpSchema.Attribute a : s.attributes()) {
            if (!first) {
                w.write(", ");
            }
            w.write(a.name());

            first = false;
        }
        w.write(")");

        w.write(");\n");

        w.out();
        w.write("}\n");
        w.write("\n");
    }

    private void genMethods(IndentWriter w, OpSchema s) throws IOException {
        genResultTypeMethod(w, s);

        genOutputParameterMethods(w, s);

        genInputParameterMethods(w, s);

        genAttributeAccessMethods(w, s);
    }

    private static void genResultTypeMethod(IndentWriter w, OpSchema s) throws IOException {
    }

    private static void genOutputParameterMethods(IndentWriter w, OpSchema s) throws IOException {
        w.write("""
                @Override
                public SequencedSet<OnnxParameter> onnxOutputs() {
                    return onnxOutputs(SCHEMA);
                }
                """);
        w.write("\n");
    }

    private static void genInputParameterMethods(IndentWriter w, OpSchema s) throws IOException {
        w.write("""
                @Override
                public SequencedMap<OnnxParameter, Object> onnxInputs() {
                """);
        w.in();

        w.write("return onnxInputs(SCHEMA, ");
        w.write("List.of(");
        boolean first = true;
        for (OpSchema.FormalParameter p : s.inputs()) {
            if (!first) {
                w.write(", ");
            }

            w.write(p.name() + "()");

            first = false;
            ;
        }
        w.write(")");
        w.write(");\n");

        w.out();
        w.write("}\n");
        w.write("\n");


        int i = 0;
        int rc = 0;
        int oc = 0;
        for (OpSchema.FormalParameter p : s.inputs()) {
            w.write("public ");
            switch (p.option()) {
                case Single -> {
                    w.write("Value ");
                    rc++;
                }
                case Optional -> {
                    w.write("java.util.Optional<Value> ");
                    oc++;
                }
                case Variadic -> {
                    w.write("List<Value> ");
                }
            }
            w.write(p.name() + "() {\n");
            w.in();

            switch (p.option()) {
                case Single -> {
                    w.write("return operands().get(" + (i++) + ");\n");
                }
                case Optional -> {
                    w.write("int i = optionalInputArguments.indexOf(InputParameter." + p.name() + ");\n");
                    w.write("return i != -1 ? java.util.Optional.of(operands().get(" + rc + " + i)) : java.util.Optional.empty();\n");
                }
                case Variadic -> {
                    if (oc > 0) {
                        w.write("int i = " + rc + " + optionalInputArguments.size();\n");
                        w.write("return operands().subList(i, operands().size());\n");
                    } else if (rc > 0) {
                        w.write("return operands().subList(" + rc + ", operands().size());\n");
                    } else {
                        w.write("return operands();\n");
                    }
                }
            }

            w.out();
            w.write("}\n");
            w.write("\n");
        }
    }

    private static void genAttributeAccessMethods(IndentWriter w, OpSchema s) throws IOException {
        for (OpSchema.Attribute a : s.attributes()) {
            w.write("public ");

            OpSchema.AttributeType aType = a.type();
            String typeString = switch (aType) {
                default -> {
                    if (a.required()) {
                        yield aType.type().getSimpleName();
                    } else {
                        yield toBoxType(aType.type()).getSimpleName();
                    }
                }
            };
            String typeLiteralString = switch (aType) {
                // @@@ sub-graphs have inputs and outputs
                default -> {
                    if (a.required()) {
                        yield aType.type().getSimpleName();
                    } else {
                        yield toBoxType(aType.type()).getSimpleName();
                    }
                }
            };
            if (a.required()) {
                w.write(typeString);
            } else {
                w.write("java.util.Optional<");
                w.write(typeString);
                w.write(">");
            }
            w.write(" ");
            w.write(a.name() + "() {\n");
            w.in();

            w.write(typeString + " ");
            w.write(a.name() + " = ");
            w.write("Attribute." + a.name() + ".access(");
            w.write(typeLiteralString + ".class, onnxAttributes");
            w.write(");\n");

            w.write("return ");
            if (a.required()) {
                w.write(a.name());
                if (aType.type().isArray()) {
                    w.write(".clone()");
                }
            } else {
                w.write("java.util.Optional.ofNullable(" + a.name() + ")");
                if (aType.type().isArray()) {
                    w.write(".map(" + typeString + "::clone)");
                }
            }
            w.write(";\n");

            w.out();
            w.write("}\n");
            w.write("\n");
        }
    }

    private void genFactoryMethod(IndentWriter w, OpSchema s) throws IOException {
        w.write("public static " + s.name() + " " + s.name() + "(");

        // Result type parameter
        w.write("TypeElement resultType, ");

        boolean hasOptionalOutputs = s.outputs()
                .stream().anyMatch(o -> o.option() == OpSchema.FormalParameterOption.Optional);
        if (hasOptionalOutputs) {
            w.write("Set<" + s.name() + ".OutputParameter> optionalOutputs, ");
        }

        boolean first = true;
        for (OpSchema.FormalParameter inParam : s.inputs()) {
            if (!first) {
                w.write(", ");
            }

            switch (inParam.option()) {
                case Single -> {
                    w.write("Value");
                }
                case Optional -> {
                    w.write("java.util.Optional<Value>");
                }
                case Variadic -> {
                    w.write("List<Value>");
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
                w.write("java.util.Optional<");
                w.write(typeString);
                w.write(">");
            }
            w.write(" ");
            w.write(attribute.name());

            first = false;
        }

        w.write(") {\n");
        w.in();

        w.write("return new " + s.name() + "(");

        w.write("resultType, ");

        if (hasOptionalOutputs) {
            w.write("optionalOutputs, ");
        }

        first = true;
        for (OpSchema.FormalParameter inParam : s.inputs()) {
            if (!first) {
                w.write(", ");
            }

            w.write(inParam.name());

            first = false;
        }

        for (OpSchema.Attribute attribute : s.attributes()) {
            if (!first) {
                w.write(", ");
            }

            w.write(attribute.name());

            first = false;
        }

        w.write(");\n");
        w.out();
        w.write("}\n");
    }

    private void genTypes(IndentWriter w, List<TypeElement.ExternalizedTypeElement> types) throws IOException {
        boolean first = true;
        for (TypeElement.ExternalizedTypeElement type : types) {
            if (!first) {
                w.write(", ");
            }

            genType(w, type);

            first = false;
        }
    }

    private void genType(IndentWriter w, TypeElement.ExternalizedTypeElement type) throws IOException {
        w.write("OnnxType." + replaceTypeIdentifier(type.identifier()));
        w.write("(");
        genTypes(w, type.arguments());
        w.write(")");
    }

    private String replaceTypeIdentifier(String i) {
        return switch (i) {
            case "float" -> "float32";
            case "double" -> "float64";
            default -> i;
        };
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
        OpGen opGen = new OpGen(schemas);

        opGen.genOpsClass(Path.of("src/main/java/oracle/code/onnx/ir"));
    }
}