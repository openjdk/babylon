package oracle.code.onnx.compiler;

import jdk.incubator.code.Op;
import jdk.incubator.code.TypeElement;
import jdk.incubator.code.Value;
import jdk.incubator.code.analysis.SSA;
import jdk.incubator.code.op.CoreOp;
import jdk.incubator.code.type.*;
import oracle.code.onnx.OnnxOperators;
import oracle.code.onnx.Tensor;
import oracle.code.onnx.ir.OnnxOp;
import oracle.code.onnx.ir.OnnxOps;
import oracle.code.onnx.ir.OnnxType;

import java.lang.invoke.MethodHandles;
import java.lang.reflect.*;
import java.util.*;
import java.util.stream.Collectors;
import java.util.stream.Stream;
import jdk.incubator.code.*;
import oracle.code.onnx.LambdaToFunc;
import oracle.code.onnx.ir.ExplicitOnnxOps;

// Transform the Java code model of an ONNX function to an ONNX code model
public class OnnxTransformer {

    static final JavaType ONNX_OPERATORS_CLASS = JavaType.type(OnnxOperators.class);

    private OnnxTransformer() {
    }

    public record OnnxFuncOp(CoreOp.FuncOp func, List<Object> initializers) {}

    public static OnnxFuncOp transform(MethodHandles.Lookup l, Map<Value, Object> evaluatedValues, CoreOp.FuncOp in) {
        OnnxPartialEvaluator pe = new OnnxPartialEvaluator();
        pe.evaluate(l, in, evaluatedValues);

        FunctionType ft = FunctionType.functionType(
                type(in.invokableType().returnType()),
                in.invokableType().parameterTypes().stream().map(OnnxTransformer::type).toList()
        );
        CoreOp.FuncOp onnxModel = CoreOp.func(in.funcName(), ft).body(b -> {
            b.transformBody(in.body(), b.parameters(), (bb, op) -> {
                if (!pe.unevaluatedOperations.contains(op)) {
                    return bb;
                }
                switch (op) {
                    // Transform invocation to ONNX operator to operation modeling the operator
                    case CoreOp.InvokeOp io when io.invokeDescriptor().refType().equals(ONNX_OPERATORS_CLASS) -> {
                        String operatorName = io.invokeDescriptor().name();
                        Class<? extends OnnxOp> opClass = onnxOpClassFromName(operatorName);
                        OnnxOp.OnnxSchema schema = schemaFromOnnxOpClass(opClass);

                        List<Object> attributes = pe.evaluatedAttributes.get(io);

                        Method opMethod = Stream.of(OnnxOps.class.getMethods())
                                .filter(m -> m.getName().equals(operatorName))
                                .findFirst().orElseThrow();

                        List<Object> opArgs = new ArrayList<>();

                        // @@@ Operator API currently requires all optional output parameters are required
                        if (schema.outputs().stream().anyMatch(p -> p.quantifier().isOptional())) {
                            opArgs.add(recordTypeToTupleType(l, (ClassType) op.resultType()));
                            Set<? extends OnnxOp.OnnxParameter> optionalOutputs = schema.outputs().stream()
                                    .filter(p -> p.quantifier().isOptional())
                                    .collect(Collectors.toSet());
                            opArgs.add(optionalOutputs);
                        } else {
                            opArgs.add(type(op.resultType()));
                        }

                        for (int i = 0; i < schema.inputs().size(); i++) {
                            OnnxOp.OnnxParameter p = schema.inputs().get(i);
                            Value v = io.operands().get(i);

                            switch (p.quantifier()) {
                                case REQUIRED -> {
                                    opArgs.add(bb.context().getValue(v));
                                }
                                case OPTIONAL -> {
                                    // Evaluation of expressions Optional.empty and Optional.of() with symbolic values
                                    if (v instanceof Op.Result r && r.op() instanceof CoreOp.InvokeOp optionalInvoke
                                            && optionalInvoke.invokeDescriptor().refType().equals(JavaType.type(Optional.class))) {
                                        switch (optionalInvoke.invokeDescriptor().name()) {
                                            case "of" -> {
                                                opArgs.add(Optional.of(bb.context().getValue(optionalInvoke.operands().getFirst())));
                                            }
                                            case "empty" -> {
                                                opArgs.add(Optional.empty());
                                            }
                                            default -> throw new UnsupportedOperationException();
                                        }
                                    } else {
                                        throw new UnsupportedOperationException();
                                    }
                                }
                                case VARIADIC -> {
                                    throw new UnsupportedOperationException();
                                }
                            }
                        }
                        opArgs.addAll(attributes.stream().map(a -> {
                            if (a instanceof CoreOp.LambdaOp lo) {
                                var ltf = LambdaToFunc.fromLambda(l, lo, evaluatedValues);
                                var cc = bb.context();
                                var lbb = Body.Builder.of(bb.parentBody(), lo.invokableType(), cc);
                                var eb = lbb.entryBlock();
                                var params = ltf.func().func().body().entryBlock().parameters();
                                var captured = lo.capturedValues();
                                for (int i = 0; i < params.size(); i++) {
                                    var param = params.get(i);
                                    cc.mapValue(param, eb.op(OnnxOps.Identity(param.type(), cc.getValue(traverseUp(captured.get(i))))));
                                }
                                ltf.func().func().body().entryBlock().ops().forEach(eb::apply);
                                return lbb;
                            }
                            return a;
                        }).toList());

                        OnnxOp onnxOp;
                        try {
                            onnxOp = (OnnxOp) opMethod.invoke(null, opArgs.toArray());
                        } catch (ReflectiveOperationException | RuntimeException e) {
                            throw new RuntimeException(e);
                        }
                        Op.Result result = bb.op(onnxOp);
                        bb.context().mapValue(io.result(), result);
                    }
                    // Transform access to the result of an operator that is a record access
                    case CoreOp.InvokeOp io when
                            recordComponentAccessToTupleIndex(l, io.invokeDescriptor()) instanceof Integer index -> {
                        Op.Result result = bb.op(CoreOp.tupleLoad(bb.context().getValue(io.operands().getFirst()), index));
                        bb.context().mapValue(io.result(), result);
                    }
                    case CoreOp.FieldAccessOp.FieldLoadOp fo when fo.fieldDescriptor().type() instanceof ClassType ct && ct.rawType().equals(TENSOR_RAW_CLASS) -> {
                        bb.context().mapValue(fo.result(), b.parameter(type(fo.resultType())));
                    }
                    // Copy remaining operations, which may be removed later transformations
                    default -> bb.op(op);
                }
                return bb;
            });
        });

        return new OnnxFuncOp(SSA.transform(onnxModel).transform((b, op) -> {
            // Drop any non-terminating operation whose result is not used
            if (op instanceof Op.Terminating || !op.result().uses().isEmpty()) {
                b.op(op);
            }
            return b;
        }), pe.initializers);
    }

    static Value traverseUp(Value v) {
        // @@@ when captured value is a VaroOp
        return v instanceof Op.Result or && or.op() instanceof CoreOp.VarOp vo && !vo.isUninitialized()? vo.initOperand() : v;
    }

    @SuppressWarnings({"rawtypes", "unchecked"})
    static Class<? extends OnnxOp> onnxOpClassFromName(String operatorName) {
        Class<? extends OnnxOp> opClass;
        try {
            return (Class) Class.forName(OnnxOps.class.getName() + "$" + operatorName);
        } catch (ClassNotFoundException e) {
            try {
                return (Class) Class.forName(ExplicitOnnxOps.class.getName() + "$" + operatorName);
            } catch (ClassNotFoundException _) {}
            throw new InternalError(e);
        }
    }

    static OnnxOp.OnnxSchema schemaFromOnnxOpClass(Class<? extends OnnxOp> opClass) {
        try {
            return (OnnxOp.OnnxSchema) opClass.getField("SCHEMA").get(null);
        } catch (ReflectiveOperationException e) {
            throw new InternalError(e);
        }
    }

    static TupleType recordTypeToTupleType(MethodHandles.Lookup l, ClassType recordType) {
        Class<?> recordClass;
        try {
            recordClass = (Class<?>) recordType.rawType().resolve(l);
        } catch (ReflectiveOperationException e) {
            throw new RuntimeException(e);
        }
        assert recordClass.isRecord();

        List<TypeElement> tupleComponentTypes = new ArrayList<>();
        for (RecordComponent rc : recordClass.getRecordComponents()) {
            switch (rc.getGenericType()) {
                case ParameterizedType pt when pt.getRawType().equals(Tensor.class) -> {
                    Type elementType = pt.getActualTypeArguments()[0];
                    switch (elementType) {
                        case Class<?> _ -> {
                            tupleComponentTypes.add(type(JavaType.type(pt)));
                        }
                        case TypeVariable<?> tv -> {
                            // Resolve type variable
                            JavaType e = null;
                            for (int j = 0; j < recordClass.getTypeParameters().length; j++) {
                                if (recordClass.getTypeParameters()[j].getName().equals(tv.getName())) {
                                    e = recordType.typeArguments().get(j);
                                    break;
                                }
                            }
                            tupleComponentTypes.add(type(JavaType.parameterized(JavaType.type(Tensor.class), e)));
                        }
                        default -> throw new IllegalStateException("Unexpected value: " + elementType);
                    }
                }
                default -> throw new IllegalStateException("Unexpected value: " + rc.getGenericType());
            }
        }

        return TupleType.tupleType(tupleComponentTypes);
    }

    static Integer recordComponentAccessToTupleIndex(MethodHandles.Lookup l, MethodRef ref) {
        if (ref.refType() instanceof ClassType ct && ct.toClassName().startsWith("oracle.code.onnx.OnnxOperators$")) {
            Class<?> refClass;
            try {
                refClass = (Class<?>) ct.resolve(l);
            } catch (ReflectiveOperationException e) {
                throw new RuntimeException(e);
            }

            if (refClass.isRecord()) {
                RecordComponent[] recordComponents = refClass.getRecordComponents();
                for (int i = 0; i < recordComponents.length; i++) {
                    if (recordComponents[i].getName().equals(ref.name())) {
                        return i;
                    }
                }
                throw new InternalError();
            }
        }
        return null;
    }

    static final TypeElement TENSOR_RAW_CLASS = JavaType.type(Tensor.class);

    // @@@ Map of Java tensor types to ONNX tensor types
    // @@@ Shape??
    static TypeElement type(TypeElement type) {
        if (type instanceof ClassType ct && ct.rawType().equals(TENSOR_RAW_CLASS)) {
            JavaType elementType = ct.typeArguments().getFirst();
            if (elementType.equals(JavaType.J_L_INTEGER)) {
                return OnnxType.TENSOR_INT32;
            } else if (elementType.equals(JavaType.J_L_FLOAT)) {
                return OnnxType.TENSOR_FLOAT32;
            } else if (elementType.equals(JavaType.J_L_LONG)) {
                return OnnxType.TENSOR_INT64;
            } else if (elementType.equals(JavaType.J_L_BYTE)) {
                return OnnxType.TENSOR_UINT8;
            } else if (elementType.equals(JavaType.J_L_BOOLEAN)) {
                return OnnxType.TENSOR_BOOL;
            }
        }
        return type;
//        throw new UnsupportedOperationException("Unknown type: " + type);
    }

}
