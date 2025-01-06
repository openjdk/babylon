package oracle.code.onnx.opgen;

import jdk.incubator.code.TypeElement;
import oracle.code.onnx.OpSchema;
import oracle.code.onnx.Tensor;

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

    void gen(Path dir) throws IOException {
        genOpsClass(dir);
    }

    void genOpsClass(Path dir) throws IOException {
        OutputStreamWriter osw = new OutputStreamWriter(
                new FileOutputStream(dir.resolve(ONNX_OPS_CLASS + ".java").toFile()));
        genOpsClass(osw);
    }

    void genOpsClass(Writer w_) throws IOException {
        IndentWriter w = new IndentWriter(w_);

        w.write("// Auto-generated from ONNX op schema\n");
        w.write("\n");
        w.write("package " + ONNX_IR_PACKAGE + ";\n");
        w.write("\n");
        w.write("import jdk.incubator.code.*;\n");
        w.write("import jdk.incubator.code.op.OpFactory;\n");
        w.write("import oracle.code.onnx.Tensor;\n");
        w.write("\n");
        w.write("import java.util.*;\n");
        w.write("\n");

        w.write("public final class " + ONNX_OPS_CLASS + " {\n");

        w.in();

        w.write("\n");
        w.write("private " + ONNX_OPS_CLASS + "() {}\n");
        w.write("\n");

        w.out();
        for (OpSchema s : schemas.values().stream().map(SortedSet::getFirst).toList()) {
            try {
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

    private String genOpClass(OpSchema s) throws IOException {
        StringWriter sw = new StringWriter();
        IndentWriter w = new IndentWriter(sw, 4);

        w.write("@OpFactory.OpDeclaration(" + s.name() + ".NAME)\n");
        w.write("public static final class "+ s.name() + " extends OnnxOp {\n");
        w.in();

        w.write("public static final String NAME = \"" + s.name() + "\";\n");
        w.write("\n");

        genAttributeEnum(w, s);

        genFields(w, s);

        genConstructors(w, s);

        genMethods(w, s);

        w.out();
        w.write("}\n");
        return sw.toString();
    }

    private void genAttributeEnum(IndentWriter w, OpSchema s) throws IOException {
        if (s.attributes().isEmpty()) {
            return;
        }

        w.write("public enum Attribute implements OnnxAttribute {\n");
        w.in();

        for (OpSchema.Attribute a : s.attributes()) {
            switch (a.type()) {
                case GRAPH -> {
                    throw new UnsupportedOperationException("Graph attribute unsupported, " + a.name());
                }
                case GRAPHS -> {
                    throw new UnsupportedOperationException("Graph attribute unsupported, " + a.name());
                }
                default -> {}
            }
            w.write(a.name());
            w.write("(");
            w.write(toBoxType(a.type().type()).getSimpleName() + ".class");
            w.write(", ");
            w.write(Boolean.toString(!a.required()));
            w.write(", ");
            // @@@ Default value?
            w.write("null");
            w.write("),\n");
        }
        w.write(";\n");
        w.write("\n");

        w.write("final Class<?> type_;\n");
        w.write("final boolean optional;\n");
        w.write("final Object defaultValue;\n");
        w.write("\n");

        w.write("Attribute(Class<?> type_, boolean optional, Object defaultValue) {\n");
        w.write("    this.type_ = type_;\n");
        w.write("    this.optional = optional;\n");
        w.write("    this.defaultValue = defaultValue;\n");
        w.write("    assert optional || defaultValue == null;\n");
        w.write("}\n");
        w.write("\n");

        w.write("public Class<?> type() {\n");
        w.write("    return type_;\n");
        w.write("}\n");
        w.write("\n");
        w.write("public boolean optional() {\n");
        w.write("    return optional;\n");
        w.write("}\n");
        w.write("\n");
        w.write("public Object defaultValue() {\n");
        w.write("    return defaultValue;\n");
        w.write("}\n");

        w.out();
        w.write("}\n");
        w.write("\n");
    }

    private void genFields(IndentWriter w, OpSchema s) throws IOException{
        if (s.attributes().isEmpty()) {
            return;
        }

        w.write("final Map<String, Object> attributes;\n");
        w.write("\n");
    }

    private void genConstructors(IndentWriter w, OpSchema s) throws IOException {
        w.write("public " + s.name() + "(ExternalizedOp def) {\n");
        w.in();

        w.write("super(def);\n");

        if (!s.attributes().isEmpty()) {
            w.write("\n");
            w.write("this.attributes = OnnxAttribute.process(def, Attribute::valueOf);\n");
        }

        w.out();
        w.write("}\n");
        w.write("\n");

        w.write(s.name() + "(" + s.name() + " that, CopyContext cc) {\n");
        w.write("    super(that, cc);\n");
        if (!s.attributes().isEmpty()) {
            w.write("\n");
            w.write("    this.attributes = Map.copyOf(that.attributes);\n");
        }
        w.write("}\n");
        w.write("\n");

        w.write("@Override\n");
        w.write("public " + s.name() + " transform(CopyContext cc, OpTransformer ot) {\n");
        w.write("    return new " + s.name() + "(this, cc);\n");
        w.write("}\n");
        w.write("\n");


        w.write(s.name() + "(");

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
                    throw new UnsupportedOperationException("Optional formal input parameter unsupported, " +
                            inParam.name());
                }
                case Variadic -> {
                    throw new UnsupportedOperationException("Variadic formal input parameter unsupported, " +
                            inParam.name());
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
            Class<?> type = switch (aType) {
                // @@@ sub-graphs have inputs and outputs
                case GRAPH -> throw new UnsupportedOperationException("Graph attribute unsupported");
                default -> aType.type();
            };
            if (attribute.required()) {
                w.write(type.getSimpleName());
            } else {
                w.write("Optional<");
                w.write(toBoxType(type).getSimpleName());
                w.write(">");
            }
            w.write(" ");
            w.write(attribute.name());

            first = false;
        }

        w.write(") {\n");
        w.in();

        w.write("super(NAME, List.of(");
        first = true;
        for (OpSchema.FormalParameter inParam : s.inputs()) {
            if (!first) {
                w.write(", ");
            }
            w.write(inParam.name());

            first = false;
        }
        w.write("));\n");

        if (!s.attributes().isEmpty()) {
            w.write("\n");
            w.write("Map<String, Object> attrs = new HashMap<>();\n");
            for (OpSchema.Attribute a : s.attributes()) {
                w.write("Attribute." + a.name() + ".process(attrs, " + a.name());

                if (!a.required()) {
                    if (a.type().type().isArray()) {
                        w.write(".map(" + a.type().type().getSimpleName() + "::clone)");
                    }
                } else if (a.type().type().isArray()) {
                    w.write(".clone()");
                }
                w.write(");\n");
            }
            w.write("this.attributes = Map.copyOf(attrs);\n");
        }
        
        w.out();
        w.write("}\n");
        w.write("\n");
    }

    private void genMethods(IndentWriter w, OpSchema s) throws IOException {
        OpSchema.FormalParameter outParam = s.outputs().getFirst();
        if (s.min_output() == 1 && s.max_output() == 1) {
            switch (outParam.option()) {
                case Single -> {
                    String s1 = outParam.type_str();
                    int i = 0;
                    for (OpSchema.FormalParameter input : s.inputs()) {
                        if (s1.equals(input.type_str())) {
                            break;
                        }
                        i++;
                    }
                    if (i < s.inputs().size()) {
                        w.write("@Override\n");
                        w.write("public TypeElement resultType() {\n");
                        w.write("    return operands().get(" + i + ").type();\n");
                        w.write("}\n");
                        w.write("\n");
                    } else {
                        throw new UnsupportedOperationException("Result type independent of input types, " + s1);
                    }
                }
                case Optional -> {
                    throw new UnsupportedOperationException("Optional formal output parameter unsupported");
                }
                case Variadic -> {
                    throw new UnsupportedOperationException("Variadic formal output parameter unsupported");
                }
            }
        } else {
            throw new UnsupportedOperationException("Multiple formal output parameters unsupported, " +
                    "min " + s.min_output() + " max " + s.max_output());
        }

        // Parameters

        int i = 0;
        for (OpSchema.FormalParameter p : s.inputs()) {
            w.write("public Value " + p.name() + "() {\n");
            w.write("    return operands().get(" + (i++) + ");\n");
            w.write("}\n");
            w.write("\n");
        }

        // Attributes

        for (OpSchema.Attribute a : s.attributes()) {
            w.write("public ");

            OpSchema.AttributeType aType = a.type();
            Class<?> type = switch (aType) {
                // @@@ sub-graphs have inputs and outputs
                case GRAPH -> throw new UnsupportedOperationException("Graph attribute unsupported");
                default -> aType.type();
            };
            if (a.required()) {
                w.write(type.getSimpleName());
            } else {
                w.write("Optional<");
                w.write(toBoxType(type).getSimpleName());
                w.write(">");
            }
            w.write(" ");
            w.write(a.name() + "() {\n");
            w.in();

            w.write(toBoxType(type).getSimpleName() + " ");
            w.write(a.name() + " = ");
            w.write("Attribute." + a.name() + ".access(");
            w.write(toBoxType(type).getSimpleName() + ".class, attributes");
            w.write(");\n");

            w.write("return ");
            if (a.required()) {
                w.write(a.name());
                if (type.isArray()) {
                    w.write(".clone()");
                }
            } else {
                w.write("Optional.ofNullable(" + a.name() + ")");
                if (type.isArray()) {
                    w.write(".map(" + type.getSimpleName() + "::clone)");
                }
            }
            w.write(";\n");

            w.out();
            w.write("}\n");
            w.write("\n");
        }
    }

    static String toJavaIdentifier(String type_str) {
        TypeElement.ExternalizedTypeElement ete = TypeElement.ExternalizedTypeElement.ofString(
                type_str.replace('(', '<').replace(')', '>'));
        if (ete.arguments().isEmpty()) {
            // Type variable
            return ete.identifier();
        } else if (ete.arguments().size() == 1 && ete.identifier().equals("tensor")) {
            // Concrete tensor
            TypeElement.ExternalizedTypeElement typeArg = ete.arguments().getFirst();
            if (typeArg.arguments().isEmpty()) {
                Tensor.ElementType elementType = Tensor.ElementType.fromOnnxName(typeArg.identifier());
                Class<?> type = elementType.type();
                if (type.isPrimitive()) {
                    return toBoxType(type).getSimpleName();
                } else {
                    return type.getSimpleName();
                }
            }
        }
        throw new UnsupportedOperationException(type_str);
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
                "/Users/sandoz/Projects/jdk/babylon/cr-examples/onnx/opgen/onnx-schema.json"));
        OpGen opGen = new OpGen(schemas);

        opGen.genOpsClass(Path.of("/Users/sandoz/Projects/jdk/babylon/cr-examples/onnx/src/main/java/oracle/code/onnx/ir"));
//        opGen.genOpsClass(new OutputStreamWriter(System.out));
    }
}
