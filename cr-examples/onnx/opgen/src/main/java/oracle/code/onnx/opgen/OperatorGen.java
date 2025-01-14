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

        w.write("// Auto-generated from ONNX op schema\n");
        w.write("\n");
        w.write("package " + ONNX_PACKAGE + ";\n");
        w.write("\n");
        w.write("""
                import oracle.code.onnx.ir.OnnxOps;
                
                import java.util.Optional;
                import java.util.List;
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
        OpSchema.FormalParameter outParam = s.outputs().getFirst();
        String outputTypeString;
        if (s.min_output() == 1 && s.max_output() == 1) {
            outputTypeString = "Tensor<" + toJavaIdentifier(outParam.type_str()) + ">";
        } else if (s.min_output() == 0 && s.max_output() == 1) {
            // @@@ This does not occur
            outputTypeString = "Optional<Tensor<" + toJavaIdentifier(outParam.type_str()) + ">>";
        } else {
            outputTypeString = "List<Tensor<" + toJavaIdentifier(outParam.type_str()) + ">>";
        }
        w.write(outputTypeString);
        w.write(" ");
        w.write(s.name() + "(");

        boolean first = true;
        for (OpSchema.FormalParameter inParam : s.inputs()) {
            if (!first) {
                w.write(", ");
            }

            switch (inParam.option()) {
                case Single -> {
                    w.write("Tensor<" + toJavaIdentifier(inParam.type_str()) + ">");
                }
                case Optional -> {
                    w.write("Optional<");
                    w.write("Tensor<" + toJavaIdentifier(inParam.type_str()) + ">");
                    w.write(">");
                }
                case Variadic -> {
                    w.write("List<");
                    w.write("Tensor<" + toJavaIdentifier(inParam.type_str()) + ">");
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
                case TENSORS -> aType.type().getSimpleName() + "<?>";
                case TENSOR -> aType.type().getSimpleName() + "<?>";
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

        w.write("return (" + outputTypeString + ") result;\n");
        w.out();
        w.write("}\n");
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
                "opgen/onnx-schema.json"));
        OperatorGen oprGen = new OperatorGen(schemas);

        oprGen.genOpClass(Path.of("src/main/java/oracle/code/onnx"));
    }
}
