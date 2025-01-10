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
        w.write("public static final class " + s.name() + " extends OnnxOp {\n");
        w.in();

        w.write("public static final String NAME = \"" + s.name() + "\";\n");
        w.write("\n");

        genAttributeEnum(w, s);

        genFields(w, s);

        genConstructors(w, s);

        genMethods(w, s);

        w.out();
        w.write("}\n");
        w.write("\n");

        genFactory(w, s);

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
                default -> {
                }
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

    private void genFields(IndentWriter w, OpSchema s) throws IOException {
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

        // Result type parameter
        w.write("TypeElement resultType, ");

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

        w.write("super(NAME, resultType, List.of(");
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
        // Result
        // @@@ result type needs to be computed possibly using broadcasting rules, it is not
        // simply the same as an input operand's type
//        OpSchema.FormalParameter outParam = s.outputs().getFirst();
//        if (s.min_output() == 1 && s.max_output() == 1) {
//            switch (outParam.option()) {
//                case Single -> {
//                    // Find if the output type string is also used by a required input
//                    String outputTypeString = outParam.type_str();
//                    int i = 0;
//                    for (OpSchema.FormalParameter input : s.inputs()) {
//                        if (input.option() == Single && outputTypeString.equals(input.type_str())) {
//                            break;
//                        }
//                        i++;
//                    }
//                    if (i < s.inputs().size()) {
//                        // Result type is the same as one of the required input types
//                        // @@@ Does it have the same shape?
//                        w.write("@Override\n");
//                        w.write("public TypeElement resultType() {\n");
//                        w.write("    return operands().get(" + i + ").type();\n");
//                        w.write("}\n");
//                        w.write("\n");
//                    } else {
//                        // Find the output type constraint from output type string
//                        Optional<OpSchema.TypeConstraintParam> oOutTcp = s.type_constraints().stream()
//                                .filter(tc -> tc.type_param_str().equals(outputTypeString)).findFirst();
//                        if (oOutTcp.isEmpty()) {
//                            // Result type is literal, but we don't know its shape
//                            String literalTypeString = outputTypeString;
//                            throw new UnsupportedOperationException("Result type is literal with unknown shape, " + literalTypeString);
//                        } else {
//                            // See And operator for example
//                            // If there is an input with a different type string
//                            // but with the same with same output type constraints
//                            OpSchema.TypeConstraintParam outTcp = oOutTcp.get();
//                            if (outTcp.allowed_type_strs().size() == 1) {
//                                i = 0;
//                                for (OpSchema.FormalParameter input : s.inputs()) {
//                                    // Find the input type constraint from input type string
//                                    Optional<OpSchema.TypeConstraintParam> oInTcp = s.type_constraints().stream()
//                                            .filter(tc -> tc.type_param_str().equals(input.type_str())).findFirst();
//                                    if (input.option() == Single && oInTcp.isPresent()) {
//                                        OpSchema.TypeConstraintParam inTcp = oInTcp.get();
//                                        if (outTcp.allowed_type_strs().equals(inTcp.allowed_type_strs())) {
//                                            break;
//                                        }
//                                    }
//                                    i++;
//                                }
//                                if (i < s.inputs().size()) {
//                                    // Result type is the same as one of the required input types
//                                    // @@@ Does it have the same shape?
//                                    w.write("@Override\n");
//                                    w.write("public TypeElement resultType() {\n");
//                                    w.write("    return operands().get(" + i + ").type();\n");
//                                    w.write("}\n");
//                                    w.write("\n");
//                                    break;
//                                } else {
//                                    // Result type is literal, but we don't know its shape
//                                    String literalTypeString = outTcp.allowed_type_strs().getFirst();
//                                    throw new UnsupportedOperationException("Result type is literal with unknown shape, " + literalTypeString);
//                                }
//                            }
//                        }
//                        throw new UnsupportedOperationException("Result type independent of input types, " + outputTypeString);
//                    }
//                }
//                case Optional -> {
//                    throw new UnsupportedOperationException("Optional formal output parameter unsupported");
//                }
//                case Variadic -> {
//                    throw new UnsupportedOperationException("Variadic formal output parameter unsupported");
//                }
//            }
//        } else {
//            throw new UnsupportedOperationException("Multiple formal output parameters unsupported, " +
//                    "min " + s.min_output() + " max " + s.max_output());
//        }

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

    private void genFactory(IndentWriter w, OpSchema s) throws IOException {
        w.write("public static " + s.name() + " " + s.name() + "(");

        // Result type parameter
        w.write("TypeElement resultType, ");

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

        w.write("return new " + s.name() + "(");

        w.write("resultType, ");

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

/*
# Result types

Skipping And: Result type independent of input types, T1
- Two type constraints that are the same with one type "tensor(bool)"
- Literal type, can be expressed directly

Skipping ArgMax: Result type independent of input types, tensor(int64)
Skipping ArgMin: Result type independent of input types, tensor(int64)
- Literal type, can be expressed directly

Skipping Bernoulli: Result type independent of input types, T2
- Result type needs to be an explicit Java parameter

Skipping Cast: Result type independent of input types, T2
- int attribute "to" declares the element type of the returned tensor.
  Must be one of the types from DataType enum in TensorProto
- Result type needs to be an explicit Java parameter
- How can we ensure attribute "to" and Result type are in sync
- Requires custom code

Skipping CastMap: Result type independent of input types, T2
- String attribute "cast_to" declares output tensor type
  "A string indicating the desired element type of the output tensor,
   one of 'TO_FLOAT', 'TO_STRING', 'TO_INT64'."
- input types are of map(int64, string) or map(int64, float)
- Skip this operation?

Skipping CategoryMapper: Result type independent of input types, T2
- "Output data. If strings are input, the output values are integers, and vice versa."
- Skip this operation?

Skipping ConcatFromSequence: Result type independent of input types, T
- Input is seq(tensor(T)), output is tensor(T)
- Requires custom code

Skipping Constant: Result type independent of input types, T
- Result type derived from constant value
- Constant value declared as specific attribute
- Requires custom code

Skipping ConstantOfShape: Result type independent of input types, T2
- Constant value is one element tensor declared as attribute
- Input, 1D tensor of int64 that is the shape
- Requires custom code

Skipping DictVectorizer: Result type independent of input types, T2
- Input is map(string, T), output is tensor(T)
- Requires custom code

Skipping Equal: Result type independent of input types, T1
- Result type constraint with just one type "tensor(bool)"
- Literal type, can be expressed directly

Skipping EyeLike: Result type independent of input types, T2
- int attribute "dtype" declares the element type of the returned tensor.
- Result type needs to be an explicit Java parameter
- How can we ensure attribute "dtype" and Result type are in sync
- Requires custom code

Skipping Greater: Result type independent of input types, T1
Skipping GreaterOrEqual: Result type independent of input types, T1
- Result type constraint with just one type "tensor(bool)"
- Literal type, can be expressed directly

Skipping HammingWindow: Result type independent of input types, T2
Skipping HannWindow: Result type independent of input types, T2
- int attribute "output_datatype" declares the element type of the returned tensor.
- Result type needs to be an explicit Java parameter
- How can we ensure attribute "output_datatype" and Result type are in sync
- Requires custom code

Skipping ImageDecoder: Result type independent of input types, T2
- Result type constraint with just one type "tensor(uint8)"
- Literal type, can be expressed directly

Skipping IsInf: Result type independent of input types, T2
Skipping IsNaN: Result type independent of input types, T2
- Result type constraint with just one type "tensor(bool)"
- Literal type, can be expressed directly

Skipping LabelEncoder: Result type independent of input types, T2
- Attribute "values_T" is tensor(T) declared as attribute
- Result type is tensor(T)
- Requires custom code

Skipping Less: Result type independent of input types, T1
Skipping LessOrEqual: Result type independent of input types, T1
- Result type constraint with just one type "tensor(bool)"
- Literal type, can be expressed directly

Skipping LinearRegressor: Result type independent of input types, tensor(float)
- Literal type, can be expressed directly

Skipping MelWeightMatrix: Result type independent of input types, T3
- int attribute "output_datatype" declares the element type of the returned tensor.
- Result type needs to be an explicit Java parameter
- How can we ensure attribute "output_datatype" and Result type are in sync
- Requires custom code

Skipping Multinomial: Result type independent of input types, T2
- *Optional* int attribute "dtype" declares the element type of the returned tensor,
  default value is that for int32
  Set is limited to int32 or int64
- Requires custom code

Skipping NonZero: Result type independent of input types, tensor(int64)
Skipping Normalizer: Result type independent of input types, tensor(float)
Skipping OneHotEncoder: Result type independent of input types, tensor(float)
- Literal type, can be expressed directly

Skipping OptionalGetElement: Result type independent of input types, V
- input is tensor(T), seq(tensor(T)), optional(tensor(T)), optional(seq(tensor(T)))
- output is tensor(T), seq(tensor(T))
- Requires custom code

Skipping Or: Result type independent of input types, T1
- Result type constraint with just one type "tensor(bool)"
- Literal type, can be expressed directly

Skipping RandomNormal: Result type independent of input types, T
- *Optional* int attribute "dtype" declares the element type of the returned tensor,
  restricted to floats, default is float32
- Requires custom code

Skipping RandomNormalLike: Result type independent of input types, T2
- *Optional* int attribute "dtype" declares the element type of the returned tensor,
  restricted to floats. "if not specified, we will use the data type of the input tensor."
- Requires custom code

Skipping RandomUniform: Result type independent of input types, T
- *Optional* int attribute "dtype" declares the element type of the returned tensor,
  restricted to floats, default is float32
- Requires custom code

Skipping RandomUniformLike: Result type independent of input types, T2
- *Optional* int attribute "dtype" declares the element type of the returned tensor,
  restricted to floats. "if not specified, we will use the data type of the input tensor."
- Requires custom code

Skipping RegexFullMatch: Result type independent of input types, T2
- Result type constraint with just one type "tensor(bool)"
- Literal type, can be expressed directly

Skipping SVMRegressor: Result type independent of input types, tensor(float)
Skipping Scaler: Result type independent of input types, tensor(float)
- Literal type, can be expressed directly

Skipping SequenceAt: Result type independent of input types, T
- Unclear if output tensor is same as input tensor, seems so
- Requires custom code

Skipping SequenceEmpty: Result type independent of input types, S
- *Optional* int attribute "dtype" declares the element type of the returned tensor,
  default value is that for float32
- Requires custom code

Skipping SequenceLength: Result type independent of input types, I
Skipping Shape: Result type independent of input types, T1
Skipping Size: Result type independent of input types, T1
- Result type constraint with just one type "tensor(int64)"
- Literal type, can be expressed directly

Skipping TfIdfVectorizer: Result type independent of input types, T1
- Result type constraint with just one type "tensor(float)"
- Literal type, can be expressed directly

Skipping TreeEnsembleRegressor: Result type independent of input types, tensor(float)
- Literal type, can be expressed directly

Skipping Xor: Result type independent of input types, T1
- Result type constraint with just one type "tensor(bool)"
- Literal type, can be expressed directly

Skipping ZipMap: Result type independent of input types, T
- Input is tensor(float)
- Output is seq(map(int64,tensor(float))) or seq(map(string,tensor(float)))
  depending on attribute "classlabels_int64s" or "classlabels_strings"
- Requires custom code


# Optional formal input parameter
- optional parameters occur after single parameters in the sequence of input parameters
- for code models we will require a code model operation attribute listing the present
 input parameters. The production of this attribute can be hidden behind the
 corresponding factory method.

Skipping Clip: Optional formal input parameter unsupported, min
Skipping Conv: Optional formal input parameter unsupported, B
Skipping ConvInteger: Optional formal input parameter unsupported, x_zero_point
Skipping ConvTranspose: Optional formal input parameter unsupported, B
Skipping DFT: Optional formal input parameter unsupported, dft_length
Skipping DeformConv: Optional formal input parameter unsupported, B
Skipping DequantizeLinear: Optional formal input parameter unsupported, x_zero_point
Skipping Dropout: Optional formal input parameter unsupported, ratio
Skipping GRU: Optional formal input parameter unsupported, B
Skipping Gemm: Optional formal input parameter unsupported, C
Skipping LSTM: Optional formal input parameter unsupported, B
Skipping LayerNormalization: Optional formal input parameter unsupported, B
Skipping MatMulInteger: Optional formal input parameter unsupported, a_zero_point
Skipping MaxUnpool: Optional formal input parameter unsupported, output_shape
Skipping NegativeLogLikelihoodLoss: Optional formal input parameter unsupported, weight
Skipping NonMaxSuppression: Optional formal input parameter unsupported, max_output_boxes_per_class
Skipping Optional: Optional formal input parameter unsupported, input
Skipping OptionalHasElement: Optional formal input parameter unsupported, input
Skipping Pad: Optional formal input parameter unsupported, constant_value
Skipping QLinearConv: Optional formal input parameter unsupported, B
Skipping QuantizeLinear: Optional formal input parameter unsupported, y_zero_point
Skipping RNN: Optional formal input parameter unsupported, B
Skipping ReduceL1: Optional formal input parameter unsupported, axes
Skipping ReduceL2: Optional formal input parameter unsupported, axes
Skipping ReduceLogSum: Optional formal input parameter unsupported, axes
Skipping ReduceLogSumExp: Optional formal input parameter unsupported, axes
Skipping ReduceMax: Optional formal input parameter unsupported, axes
Skipping ReduceMean: Optional formal input parameter unsupported, axes
Skipping ReduceMin: Optional formal input parameter unsupported, axes
Skipping ReduceProd: Optional formal input parameter unsupported, axes
Skipping ReduceSum: Optional formal input parameter unsupported, axes
Skipping ReduceSumSquare: Optional formal input parameter unsupported, axes
Skipping Resize: Optional formal input parameter unsupported, roi
Skipping STFT: Optional formal input parameter unsupported, window
Skipping SequenceErase: Optional formal input parameter unsupported, position
Skipping SequenceInsert: Optional formal input parameter unsupported, position
Skipping Slice: Optional formal input parameter unsupported, axes
Skipping SoftmaxCrossEntropyLoss: Optional formal input parameter unsupported, weights
Skipping Split: Optional formal input parameter unsupported, split
Skipping SplitToSequence: Optional formal input parameter unsupported, split
Skipping Squeeze: Optional formal input parameter unsupported, axes
Skipping Trilu: Optional formal input parameter unsupported, k


# Variadic formal input parameter
- the single variadic parameter occurs last in the sequence of input parameters
- if is_homogeneous == false, it is presumable a union of any of the set of types
in the given type constraints, otherwise it is just one

Skipping Adagrad: Variadic formal input parameter unsupported, inputs
Skipping Adam: Variadic formal input parameter unsupported, inputs
Skipping Concat: Variadic formal input parameter unsupported, inputs
Skipping Einsum: Variadic formal input parameter unsupported, Inputs
Skipping FeatureVectorizer: Variadic formal input parameter unsupported, X
Skipping Gradient: Variadic formal input parameter unsupported, Inputs
Skipping Max: Variadic formal input parameter unsupported, data_0
Skipping Mean: Variadic formal input parameter unsupported, data_0
Skipping Min: Variadic formal input parameter unsupported, data_0
Skipping Momentum: Variadic formal input parameter unsupported, inputs
Skipping SequenceConstruct: Variadic formal input parameter unsupported, inputs
Skipping Sum: Variadic formal input parameter unsupported, data_0


# Multiple formal output parameters
- Optional parameters are requested by the "caller"
- Some outputs are variadic (confusing term applied to outputs).
  If is_homogeneous == false, it is presumable a union of any of the set of types
  in the given type constraints, otherwise it is just one
  - Represent as sequence
- For non-variadic represent as tuple, which could also identify the present
optional output parameters in its type. For code models we will require a code model
operation attribute listing the present output parameters. The production of
this attribute can be hidden behind the corresponding factory method.

Skipping BatchNormalization: Multiple formal output parameters unsupported, min 1 max 3
Skipping DynamicQuantizeLinear: Multiple formal output parameters unsupported, min 3 max 3
Skipping LinearClassifier: Multiple formal output parameters unsupported, min 2 max 2
Skipping MaxPool: Multiple formal output parameters unsupported, min 1 max 2
Skipping SVMClassifier: Multiple formal output parameters unsupported, min 2 max 2
Skipping StringSplit: Multiple formal output parameters unsupported, min 2 max 2
Skipping TopK: Multiple formal output parameters unsupported, min 2 max 2
Skipping Unique: Multiple formal output parameters unsupported, min 1 max 4
Skipping TreeEnsembleClassifier: Multiple formal output parameters unsupported, min 2 max 2


# Graph attribute unsupported

Skipping If: Graph attribute unsupported, else_branch
- two graphs for then and else.
- each graph has N outputs and the number of outputs must match the number of
  outputs in the then_branch.
- op has one boolean tensor for input representing the condition to check
- op has N outputs matching that of the graph outputs
Skipping Loop: Graph attribute unsupported, body
- body graph has 2+N inputs: (iteration_num, condition, loop carried dependencies…).
- body graph has 1+N+K outputs: (condition, loop carried dependencies…, scan_outputs…)
- op has N inputs for initial loop carried dependencies
- op has N+K outputs for final loop carried dependency values and scan_outputs
Skipping Scan: Graph attribute unsupported, body
- body graph has has N+M inputs: (loop state variables..., scan_input_elts...).
- body graph has N+K outputs: (loop state variables..., scan_output_elts...)
- op has N+M inputs
- op has N+K outputs
Skipping SequenceMap: Graph attribute unsupported, body
- body graph has number number of inputs and outputs as the operation
 */