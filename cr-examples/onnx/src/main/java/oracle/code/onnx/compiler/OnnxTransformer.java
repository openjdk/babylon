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

/*
Analysis and Transformations, in order

- Lambda to function, promoting captures to function parameters.
  (We need to handle captured Var ops modelling Java method parameters.)
- Inline methods.
  (We could first choose to transform into a module op of func ops, similar to HAT might do.)
- Promote (final) tensor field accesses to parameters.
  Each unique field reference is promoted to a function parameter.
  (This also accumulates every unique field reference into a list in encounter order,
   reflection is used to obtain the tensor values for ONNX initializers.)
- Partially evaluate the Java code model, using a clone of the interpreter.
- Java code model to ONNX code model.
  Lambdas expressions transform recursively (need to restrict where expressions are used
  to arguments of invocations)
  (Also transforms records to tuples.)
  (Using results from partial evaluation.)
- Drop unused parameters (i.e., the receiver).
  (Could be merged with droping unused operations.)
- SSA.
- Drop unused operations.

 */


// Transform the Java code model of an ONNX function to an ONNX code model
public class OnnxTransformer {

    static final JavaType ONNX_OPERATORS_CLASS = JavaType.type(OnnxOperators.class);


    static final JavaType TENSOR_CLASS = JavaType.type(Tensor.class);
    static final JavaType LIST_CLASS = JavaType.type(List.class);

    private final MethodHandles.Lookup l;
    private final CoreOp.FuncOp inputFunc;
    private SequencedCollection<FieldRef> inits;

    public static OnnxTransformer ofQuotedLambda(MethodHandles.Lookup lookup, Quoted quotedLambda) {
        CoreOp.LambdaOp lambda = (CoreOp.LambdaOp) quotedLambda.op();
        assert lambda.parameters().isEmpty();

        List<Value> captures = lambda.capturedValues();
        List<TypeElement> normalizedCaptureTypes = captures.stream()
                .map(v -> v instanceof Op.Result r &&
                        r.op() instanceof CoreOp.VarOp vop &&
                        vop.initOperand() instanceof Block.Parameter p ? p : v)
                .map(Value::type)
                .toList();
        FunctionType ft = FunctionType.functionType(lambda.invokableType().returnType(), normalizedCaptureTypes);

        CoreOp.FuncOp f = CoreOp.FuncOp.func("onnx.model", ft).body(b -> {
            // Map input captured values
            for (int i = 0; i < captures.size(); i++) {
                Value inputCapture = captures.get(i);
                Value output;
                if (inputCapture instanceof Op.Result r &&
                        r.op() instanceof CoreOp.VarOp vop &&
                        vop.initOperand() instanceof Block.Parameter) {
                    output = b.op(CoreOp.var(b.parameters().get(i)));
                } else {
                    output = b.parameters().get(i);
                }
                b.context().mapValue(inputCapture, output);
            }

            b.transformBody(lambda.body(), List.of(), OpTransformer.COPYING_TRANSFORMER);
        });

        return new OnnxTransformer(lookup, f);
    }

//    final CoreOp.FuncOp inline(CoreOp.FuncOp func) {
//        return func.transform((bb, op) -> {
//            var cc  = bb.context();
//            switch (op) {
//                case CoreOp.InvokeOp io when resolve(io) instanceof CoreOp.FuncOp inline ->
//                    bb.inline(inline(inline), cc.getValues(io.operands()), (_, v) -> cc.mapValue(io.result(), v));
//                default ->
//                    bb.apply(op);
//            }
//            return bb;
//        });
//    }

    public OnnxTransformer(MethodHandles.Lookup lookup, CoreOp.FuncOp func) {
        this.l = lookup;
        this.inputFunc = func;
    }

    void collectFunctions(SequencedMap<MethodRef, CoreOp.FuncOp> moduleFuncs, CoreOp.FuncOp func) {
        func.traverse(null, (_, op) -> {
            if(op instanceof CoreOp.InvokeOp io && resolve(io) instanceof CoreOp.FuncOp f) {
                collectFunctions(moduleFuncs, f);
                moduleFuncs.putIfAbsent(io.invokeDescriptor(), f);
            }
            return null;
        });
    }

    public CoreOp.ModuleOp transform() {
        var moduleFuncs = new LinkedHashMap<MethodRef, CoreOp.FuncOp>();
        collectFunctions(moduleFuncs, inputFunc);
        moduleFuncs.putLast(null, inputFunc);

        CoreOp.ModuleOp module = CoreOp.module(moduleFuncs.sequencedValues().stream().map(f -> f.transform((bb, op) -> {
            if (op instanceof CoreOp.InvokeOp io && moduleFuncs.get(io.invokeDescriptor()) instanceof CoreOp.FuncOp fo) {
                bb.context().mapValue(op.result(), bb.op(CoreOp.funcCall(fo, bb.context().getValues(op.operands()))));
            } else {
                bb.op(op);
            }
            return bb;
        })).toList());

        record TI(ClassType type, int index) {}
        var initializers = module.traverse(new LinkedHashMap<FieldRef, TI>(), (i, op) -> {
            if (op instanceof CoreOp.FieldAccessOp.FieldLoadOp flo && flo.resultType() instanceof ClassType ct && ct.rawType().equals(TENSOR_CLASS)) {
                i.putIfAbsent(flo.fieldDescriptor(), new TI(ct, i.size()));
            }
            return i;
        });

        CoreOp.ModuleOp initializedModule;
        if (!initializers.isEmpty()) {
            // all initializers are passed to each function as additional tuple argument
            TupleType initializersType = TupleType.tupleType(initializers.sequencedValues().stream().map(ti -> type(l, ti.type())).toList());
            initializedModule = CoreOp.module(module.functionTable().sequencedValues().stream().map(f -> {
                var ft = f.invokableType();
                return CoreOp.func(f.funcName(), FunctionType.functionType(ft.returnType(), Stream.concat(ft.parameterTypes().stream(), Stream.of(initializersType)).toList()))
                        .body(bob -> bob.transformBody(f.body(), bob.parameters(), (bb, op) -> {
                            Block.Parameter initializersArg = bob.parameters().getLast();
                            switch (op) {
                                // field load transformed to tuple load
                                case CoreOp.FieldAccessOp.FieldLoadOp flo when initializers.get(flo.fieldDescriptor()) instanceof TI ti -> {
                                    bb.context().mapValue(op.result(), bb.op(CoreOp.tupleLoad(initializersArg, ti.index())));
                                }
                                case CoreOp.FuncCallOp fco -> {
                                    // attach initializers arg to all func calls
                                    FunctionType newType = FunctionType.functionType(fco.opType().returnType(),
                                            Stream.concat(fco.opType().parameterTypes().stream(), Stream.of(initializersType)).toList());
                                    List<Value> newOperands = Stream.concat(bb.context().getValues(fco.operands()).stream(), Stream.of(initializersArg)).toList();
                                    Op.Result newCall = bb.op(CoreOp.funcCall(fco.funcName(), newType, newOperands));
                                    bb.context().mapValue(op.result(), newCall);
                                }
                                default -> {
                                    bb.op(op);
                                }
                            }
                            return bb;
                        }));
            }).toList());
        } else {
            initializedModule = module;
        }

        CoreOp.ModuleOp transformedModule = CoreOp.module(initializedModule.functionTable().sequencedValues().stream().map(f ->
                transform(initializedModule, f)).toList());

        inits = initializers.sequencedKeySet();

        System.out.println("----------------- transformed module ------------------");
        System.out.println(transformedModule.toText());

        return transformedModule;

// @@@ work in progress - below is still the old code

//        var inlinedFunc = inline(func);
//
//        inits = new ArrayList<>();
//        var initMap = new HashMap<FieldRef, Block.Parameter>();
//        var top = new Block.Builder[1];
//        // turning field loads into additiona arguments
//        inputFunc = inlinedFunc.transform((bb, op) -> {
//            // @@@ This is ugly, in this case we could ask the bb for its furthest ancestor block
//            // when we need it
//            if (top[0] == null) top[0] = bb;
//            var cc  = bb.context();
//            switch (op) {
//                case CoreOp.FieldAccessOp.FieldLoadOp flo when op.resultType() instanceof ClassType ct && ct.rawType().equals(TENSOR_CLASS) -> {
//                    // initializers turn into top block parameters
//                    cc.mapValue(op.result(), initMap.computeIfAbsent(flo.fieldDescriptor(), fd -> {
//                        inits.add(fd);
//                        return top[0].parameter(op.resultType());
//                    }));
//                }
//                default -> bb.apply(op);
//            }
//            return bb;
//        });
    }

    CoreOp.FuncOp resolve(CoreOp.InvokeOp io) {
        try {
            var res = Op.ofMethod(io.invokeDescriptor().resolveToDirectMethod(l));
            if (res.isPresent()) {
                return SSA.transform(res.get());
            }
        } catch (ReflectiveOperationException | IllegalArgumentException _) {}
        return null;
    }

    public List<Tensor> initializers(Object receiver) {
        return inits.stream().map(i -> {
            try {
                return (Tensor)(i.resolveToMember(l).accessFlags().contains(AccessFlag.STATIC) ? i.resolveToHandle(l).get() : i.resolveToHandle(l).get(receiver));
            } catch (ReflectiveOperationException ex) {
                throw new RuntimeException(ex);
            }
        }).toList();
    }

//    public CoreOp.FuncOp transform() {
//        return transform(null, inputFunc);
//    }

    CoreOp.FuncOp transform(CoreOp.ModuleOp module, CoreOp.FuncOp inputFunc) {
        OnnxPartialEvaluator pe = new OnnxPartialEvaluator();
        pe.evaluate(l, inputFunc);

        // ONNX model transformation
        FunctionType ft = type(l, inputFunc.invokableType());
        CoreOp.FuncOp onnxModel = CoreOp.func(inputFunc.funcName(), ft).body(b -> {
            b.transformBody(inputFunc.body(), b.parameters(), bodyTransformer(module, pe));
        });

        // Drop unused parameters transformation, can be merged with drop unused operations transformation
        CoreOp.FuncOp cutModel = onnxModel;
        List<Block.Parameter> usedParameters = onnxModel.parameters().stream()
                .filter(v -> isUsed(module, v))
                .toList();
        if (usedParameters.size() < onnxModel.parameters().size()) {
            List<TypeElement> usedParameterTypes = usedParameters.stream().map(Value::type).toList();

            var funcType = FunctionType.functionType(onnxModel.invokableType().returnType(), usedParameterTypes);
            cutModel = CoreOp.func(onnxModel.funcName(), funcType).body(bob -> {
                bob.context().mapValues(usedParameters, bob.parameters());
                bob.transformBody(onnxModel.body(), List.of(), (bb, op) -> {
                    if (op instanceof CoreOp.FuncCallOp fco) {
                        CopyContext cc = bb.context();
                        List<Value> newOperands = fco.operands().stream().filter(v -> cc.getValueOrDefault(v, null) != null).map(cc::getValue).toList();
                        CoreOp.FuncCallOp newCall = CoreOp.funcCall(fco.funcName(),
                                                                    FunctionType.functionType(fco.opType().returnType(),
                                                                                              newOperands.stream().map(Value::type).toList()),
                                                                    newOperands);
                        cc.mapValue(op.result(), bb.op(newCall));
                    } else {
                        bb.op(op);
                    }
                    return bb;
                });
            });
        }

        // SSA and drop unused operations transformation
        return SSA.transform(cutModel).transform((b, op) -> {
            // Drop any non-terminating operation whose result is not used
            if (op instanceof Op.Terminating || !op.result().uses().isEmpty() || op instanceof CoreOp.FuncOp) {
                b.op(op);
            }
            return b;
        });
    }

    static boolean isUsed(CoreOp.ModuleOp module, Block.Parameter param) {
        Set<Op.Result> uses = param.uses();
        for (Op.Result use : uses) {
            if (!(use.op() instanceof CoreOp.FuncCallOp fcop)
                    || !(module.functionTable().get(fcop.funcName()) instanceof CoreOp.FuncOp fo)
                    || isUsed(module, fo.parameters().get(param.index()))) {
                return true;
            }
        }
        return false;
    }

    OpTransformer bodyTransformer(CoreOp.ModuleOp module, OnnxPartialEvaluator pe) {
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
                        opArgs.add(type(l, op.resultType()));
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
                                // Evaluation of expressions List.of() with symbolic values
                                if (v instanceof Op.Result r && r.op() instanceof CoreOp.InvokeOp listInvoke
                                        && listInvoke.invokeDescriptor().refType().equals(JavaType.type(List.class))) {
                                    switch (listInvoke.invokeDescriptor().name()) {
                                        case "of" -> {
                                            opArgs.add(listInvoke.operands().stream().map(o -> bb.context().getValue(o)).toList());
                                        }
                                        default -> throw new UnsupportedOperationException();
                                    }
                                } else {
                                    // otherwise pass through a single value
                                    opArgs.add(bb.context().getValue(v));
                                }
                            }
                        }
                    }
                    opArgs.addAll(attributes);
                    if (opClass == ExplicitOnnxOps.If.class) {
                        // Explicit transformation of nested bodies
                        for (int i = 1; i < 3; i++) {
                            var lambda = (CoreOp.LambdaOp)(((Op.Result)op.operands().get(i)).op());
                            opArgs.add(transformBodyTranslateTypes(l, lambda, bb, bodyTransformer(module, pe)));
                        }
                    } else if (opClass == ExplicitOnnxOps.Loop.class) {
                        // Explicit transformation of nested body
                        var lambda = (CoreOp.LambdaOp)(((Op.Result)op.operands().get(3)).op());
                        opArgs.add(transformBodyTranslateTypes(l, lambda, bb, bodyTransformer(module, pe)));
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
                // Transform record construction
                case CoreOp.NewOp no when isRecord(l, no.type()) -> {
                    Op.Result result = bb.op(CoreOp.tuple(bb.context().getValues(no.operands())));
                    bb.context().mapValue(no.result(), result);
                }
                // Transform access to the result of an operator that is a list access
                // @@@ raw use of List::get with constant argument
                case CoreOp.InvokeOp io when io.invokeDescriptor().refType().equals(LIST_CLASS) && io.invokeDescriptor().name().equals("get") -> {
                    Op.Result result = bb.op(CoreOp.invoke(
                            io.invokeDescriptor(),
                            bb.context().getValue(io.operands().getFirst()),
                            bb.op(CoreOp.constant(JavaType.INT, pe.evaluatedAttributes.get(io).getLast()))));
                    bb.context().mapValue(io.result(), result);
                }
                // Skip nested lambdas
                case CoreOp.LambdaOp _ -> {
                }
                case CoreOp.FuncCallOp fco -> {
                    Op.Result result = bb.op(CoreOp.funcCall(fco.funcName(), type(l, fco.opType()), bb.context().getValues(fco.operands())));
                    bb.context().mapValue(fco.result(), result);
                }
                case CoreOp.FuncOp func -> {
                    bb.op(transform(module, func));
                }
                // Copy remaining operations, which may be removed later transformations
                default -> bb.op(op);
            }
            return bb;
        };
    }

    // @@@ Copy of Body::transform content to translate types
    static Body.Builder transformBodyTranslateTypes(MethodHandles.Lookup l, Op.Invokable iop,
                                                    Block.Builder ancestor, OpTransformer ot) {
        // @@@ Pass in function type to override that of body's type?
//        return iop.body().transform(cc, ot);
        FunctionType inputType = iop.invokableType();
        FunctionType outputType = FunctionType.functionType(
                type(l, inputType.returnType()),
                inputType.parameterTypes().stream().map(pt -> type(l, pt)).toList());

        // @@@ It's not clear in the API when to pass CopyContext and OpTransformer
        // @@@ create a Body.Builder structurally connected as a descendant of a Block.Builder
        // but not yet connected as the child of an operation
        Body.Builder bb = Body.Builder.of(ancestor.parentBody(),
                outputType, ancestor.context()); // translate types

        bb.entryBlock().transformBody(iop.body(), bb.entryBlock().parameters(), ot);
        return bb;
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
                            tupleComponentTypes.add(type(l, JavaType.type(pt)));
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
                            tupleComponentTypes.add(type(l, JavaType.parameterized(JavaType.type(Tensor.class), e)));
                        }
                        default -> throw new IllegalStateException("Unexpected value: " + elementType);
                    }
                }
                case TypeVariable tv -> {
                    // Resolve type variable
                    JavaType e = null;
                    for (int j = 0; j < recordClass.getTypeParameters().length; j++) {
                        if (recordClass.getTypeParameters()[j].getName().equals(tv.getName())) {
                            e = recordType.typeArguments().get(j);
                            break;
                        }
                    }
                    tupleComponentTypes.add(type(l, e));
                }
                default -> throw new IllegalStateException("Unexpected value: " + rc.getGenericType());
            }
        }

        return TupleType.tupleType(tupleComponentTypes);
    }

    static boolean isRecord(MethodHandles.Lookup l, TypeElement type) {
        try {
            return type instanceof ClassType ct &&
                    ct.erasure().resolve(l) instanceof Class c &&
                    c.isRecord();
        } catch (ReflectiveOperationException e) {
            throw new RuntimeException(e);
        }
    }

    static Integer recordComponentAccessToTupleIndex(MethodHandles.Lookup l, MethodRef ref) {
        if (ref.refType() instanceof ClassType ct) {
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

    static FunctionType type(MethodHandles.Lookup l, FunctionType t) {
        return FunctionType.functionType(type(l, t.returnType()), t.parameterTypes().stream().map(pt -> type(l, pt)).toList());
    }

    // @@@ Map of Java tensor types to ONNX tensor types
    // @@@ Shape??
    static TypeElement type(MethodHandles.Lookup l, TypeElement type) {
        if (type instanceof ClassType ct) {
            if (ct.rawType().equals(TENSOR_RAW_CLASS)) {
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
            } else if (isRecord(l, type)) {
                return recordTypeToTupleType(l, ct);
            }
        }
        return type;
//        throw new UnsupportedOperationException("Unknown type: " + type);
    }
}
