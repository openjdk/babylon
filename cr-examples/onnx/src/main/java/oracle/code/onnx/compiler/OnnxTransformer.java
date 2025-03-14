package oracle.code.onnx.compiler;

import java.lang.invoke.MethodHandles;
import java.lang.reflect.*;
import java.util.*;
import java.util.stream.Collectors;
import java.util.stream.Stream;
import jdk.incubator.code.*;
import jdk.incubator.code.analysis.SSA;
import jdk.incubator.code.op.CoreOp;
import jdk.incubator.code.type.*;
import oracle.code.onnx.OnnxOperators;
import oracle.code.onnx.Tensor;
import oracle.code.onnx.ir.OnnxOp;
import oracle.code.onnx.ir.OnnxOps;
import oracle.code.onnx.ir.OnnxType;
import oracle.code.onnx.ir.ExplicitOnnxOps;

// Transform the Java code model of an ONNX function to an ONNX code model
public class OnnxTransformer {

    static final JavaType ONNX_OPERATORS_CLASS = JavaType.type(OnnxOperators.class);


    static final JavaType TENSOR_CLASS = JavaType.type(Tensor.class);

    private final MethodHandles.Lookup l;
    private final CoreOp.FuncOp inputFunc;
    private final List<FieldRef> inits;

    public static OnnxTransformer ofLambda(MethodHandles.Lookup lookup, CoreOp.LambdaOp lambda) {
        var qotableLambdaFunc = SSA.transform((CoreOp.FuncOp)lambda.ancestorBody().parentOp().ancestorBody().parentOp());

        var flatLambdaFunc = CoreOp.func(qotableLambdaFunc.funcName(), FunctionType.functionType(lambda.invokableType().returnType(), qotableLambdaFunc.invokableType().parameterTypes())).body(bb -> {
            var cc  = bb.context();
            cc.mapValues(qotableLambdaFunc.parameters(), bb.parameters());
            qotableLambdaFunc.body().entryBlock().firstOp().bodies().getFirst().entryBlock().ops().getFirst().bodies().getFirst().entryBlock().ops().forEach(bb::apply);
        });

        return new OnnxTransformer(lookup, flatLambdaFunc);
    }

    public OnnxTransformer(MethodHandles.Lookup lookup, CoreOp.FuncOp func) {
        l = lookup;

        var inlinedFunc = func.transform(OpTransformer.DROP_LOCATION_TRANSFORMER).transform((bb, op) -> {
            var cc  = bb.context();
            switch (op) {
                case CoreOp.InvokeOp io when resolve(io) instanceof CoreOp.FuncOp inline -> {
                    // @@@ inline the whole FuncOp with nested bodies
                    var entryBlock = inline.transform(OpTransformer.DROP_LOCATION_TRANSFORMER).body().entryBlock();
                    for (int i = 0; i < io.operands().size(); i++) {
                        var out = cc.getValue(io.operands().get(i));
                        cc.mapValue(entryBlock.parameters().get(i), cc.getValueOrDefault(out, out));
                    }
                    for (var o : entryBlock.ops()) {
                        if (o instanceof CoreOp.ReturnOp ro) {
                            cc.mapValue(io.result(), cc.getValue(ro.returnValue()));
                        } else {
                            bb.apply(o);
                        }
                    }
                }
                default -> bb.apply(op);
            }
            return bb;
        });

        inits = new ArrayList<>();
        var top = new Block.Builder[1];
        // turning field loads into additiona arguments
        inputFunc = inlinedFunc.transform((bb, op) -> {
            if (top[0] == null) top[0] = bb;
            var cc  = bb.context();
            switch (op) {
                case CoreOp.FieldAccessOp.FieldLoadOp flo when op.resultType() instanceof ClassType ct && ct.rawType().equals(TENSOR_CLASS) -> {
                    inits.add(flo.fieldDescriptor());
                    // initializers turn into top block parameters
                    cc.mapValue(op.result(), top[0].parameter(op.resultType()));
                }
                default -> bb.apply(op);
            }
            return bb;
        });
    }

    CoreOp.FuncOp resolve(CoreOp.InvokeOp io) {
        try {
            var res = Op.ofMethod(io.invokeDescriptor().resolveToDirectMethod(l));
            if (res.isPresent()) {
                return SSA.transform(res.get());
            }
        } catch (ReflectiveOperationException _) {}
        return null;
    }

    public List<Tensor> initializers(Object receiver) {
        return inits.stream().map(i -> {
            try {
                return (Tensor)i.resolveToHandle(l).get(receiver);
            } catch (ReflectiveOperationException ex) {
                throw new RuntimeException(ex);
            }
        }).toList();
    }

    public CoreOp.FuncOp transform() {

        OnnxPartialEvaluator pe = new OnnxPartialEvaluator();
        pe.evaluate(l, inputFunc);

        FunctionType ft = FunctionType.functionType(type(inputFunc.invokableType().returnType()),
                inputFunc.invokableType().parameterTypes().stream().map(OnnxTransformer::type).toList()
        );
        CoreOp.FuncOp onnxModel = CoreOp.func(inputFunc.funcName(), ft).body(b -> {
            b.transformBody(inputFunc.body(), b.parameters(), bodyTransformer(pe));
        });

        var paramTypes = onnxModel.invokableType().parameterTypes();

        CoreOp.FuncOp stripModel = onnxModel;
        if (!paramTypes.isEmpty() && !(paramTypes.getFirst() instanceof OnnxType.TensorType)) {
            // drop receiver
            stripModel = CoreOp.func(onnxModel.funcName(), FunctionType.functionType(onnxModel.invokableType().returnType(), paramTypes.subList(1, paramTypes.size()))).body(bb -> {
                bb.context().mapValues(onnxModel.parameters().subList(1, paramTypes.size()), bb.parameters());
                onnxModel.body().entryBlock().ops().forEach(bb::apply);
            });
        }

        return SSA.transform(stripModel).transform((b, op) -> {
            // Drop any non-terminating operation whose result is not used
            if (op instanceof Op.Terminating || !op.result().uses().isEmpty()) {
                b.op(op);
            }
            return b;
        });
    }

    OpTransformer bodyTransformer(OnnxPartialEvaluator pe) {
        return (bb, op) -> {
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
                    opArgs.addAll(attributes);
                    if (opClass == ExplicitOnnxOps.If.class) {
                        // Explicit transformation of nested bodies
                        for (int i = 1; i < 3; i++) {
                            var lambda = (CoreOp.LambdaOp)(((Op.Result)op.operands().get(i)).op());
                            opArgs.add(lambda.body().transform(bb.context(), bodyTransformer(pe)));
                        }
                    }
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
                // Skip nested lambdas
                case CoreOp.LambdaOp _ -> {
                }
                // Copy remaining operations, which may be removed later transformations
                default -> bb.op(op);
            }
            return bb;
        };
    }

    @SuppressWarnings({"rawtypes", "unchecked"})
    static Class<? extends OnnxOp> onnxOpClassFromName(String operatorName) {
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
