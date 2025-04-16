package oracle.code.onnx.compiler;

import java.lang.invoke.MethodHandles;
import java.lang.reflect.*;
import java.util.*;
import java.util.stream.Collectors;
import java.util.stream.IntStream;
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
public final class OnnxTransformer {

    static final JavaType ONNX_OPERATORS_CLASS = JavaType.type(OnnxOperators.class);
    static final JavaType TENSOR_CLASS = JavaType.type(Tensor.class);
    static final JavaType LIST_CLASS = JavaType.type(List.class);

    public record ModuleAndInitializers(CoreOp.ModuleOp module, SequencedCollection<FieldRef> initializers) {}

    public static ModuleAndInitializers transform(MethodHandles.Lookup l, Quoted quotedLambda) {
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

        CoreOp.FuncOp f = CoreOp.FuncOp.func("", ft).body(b -> {
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

        return OnnxTransformer.transform(l, f);
    }

    public static ModuleAndInitializers transform(MethodHandles.Lookup l, CoreOp.FuncOp inputFunc) {
        CoreOp.ModuleOp m = collectModuleFunctions(l, inputFunc);
        ModuleAndInitializers mi = remapInitializers(l, m);
        return new ModuleAndInitializers(transformModule(l, mi.module()), mi.initializers());
    }

    static void collectModuleFunctions(MethodHandles.Lookup l, SequencedMap<MethodRef, CoreOp.FuncOp> funcs, Set<CoreOp.FuncOp> doNotInline, CoreOp.FuncOp func) {
        func.traverse(null, (_, op) -> {
            if(op instanceof CoreOp.InvokeOp io && resolve(l, io) instanceof CoreOp.FuncOp f) {
                collectModuleFunctions(l, funcs, doNotInline, f);
                doNotInline.add(funcs.putIfAbsent(io.invokeDescriptor(), f));
            }
            return null;
        });
    }

    static CoreOp.ModuleOp collectModuleFunctions(MethodHandles.Lookup l, CoreOp.FuncOp inputFunc) {
        // traverse inputFunc and collect all functions to construct module
        var funcs = new LinkedHashMap<MethodRef, CoreOp.FuncOp>();
        var doNotInline = new HashSet<CoreOp.FuncOp>();
        doNotInline.add(inputFunc);
        collectModuleFunctions(l, funcs, doNotInline, inputFunc);
        funcs.putLast(null, inputFunc);

        return CoreOp.module(funcs.sequencedValues().stream()
                .filter(f -> doNotInline.contains(f))
                .map(f -> mapOrInline(f, funcs, doNotInline)).toList());
    }

    static String findBetterName(SequencedMap<MethodRef, CoreOp.FuncOp> funcs, Set<CoreOp.FuncOp> doNotInline) {
        // find the last inlined func name
        return funcs.sequencedValues().reversed().stream().filter(f -> !doNotInline.contains(f)).findFirst().map(CoreOp.FuncOp::funcName).orElse("");
    }

    // transform all relevant invocations to func calls or inline
    static CoreOp.FuncOp mapOrInline(CoreOp.FuncOp f, SequencedMap<MethodRef, CoreOp.FuncOp> funcs, Set<CoreOp.FuncOp> doNotInline) {
        return f.transform(f.funcName().isEmpty() ? findBetterName(funcs, doNotInline): f.funcName(), (bb, op) -> {
            if (op instanceof CoreOp.InvokeOp io && funcs.get(io.invokeDescriptor()) instanceof CoreOp.FuncOp fo) {
                if (doNotInline.contains(fo)) {
                    bb.context().mapValue(op.result(), bb.op(CoreOp.funcCall(fo, bb.context().getValues(op.operands()))));
                } else {
                    bb.inline(mapOrInline(fo, funcs, doNotInline), bb.context().getValues(io.operands()), (_, v) -> bb.context().mapValue(io.result(), v));
                }
            } else {
                bb.op(op);
            }
            return bb;
        });
    }


    static ModuleAndInitializers remapInitializers(MethodHandles.Lookup l, CoreOp.ModuleOp module) {
        // collect initializers (field load ops of tensors)
        record TI(OnnxType type, int index) {}
        var initializers = module.traverse(new LinkedHashMap<FieldRef, TI>(), (i, op) -> {
            if (op instanceof CoreOp.FieldAccessOp.FieldLoadOp flo && flo.resultType() instanceof ClassType ct && ct.rawType().equals(TENSOR_CLASS)) {
                i.putIfAbsent(flo.fieldDescriptor(), new TI((OnnxType)convertType(l, ct), i.size()));
            }
            return i;
        });

        if (initializers.isEmpty()) {
            return new ModuleAndInitializers(module, List.of());
        }

        // map all initializers field loads into additional arguments
        List<OnnxType> initTypes = initializers.sequencedValues().stream().map(TI::type).toList();
        return new ModuleAndInitializers(CoreOp.module(module.functionTable().sequencedValues().stream().map(f -> {
            var ft = f.invokableType();
            int argsSize = ft.parameterTypes().size();
            return CoreOp.func(f.funcName(), FunctionType.functionType(ft.returnType(), Stream.concat(ft.parameterTypes().stream(), initTypes.stream()).toList()))
                    .body(bob -> bob.transformBody(f.body(), bob.parameters(), (bb, op) -> {
                        List<Block.Parameter> initArgs = bob.parameters().subList(argsSize, bob.parameters().size());
                        switch (op) {
                            // field loads mapped to initializers args
                            case CoreOp.FieldAccessOp.FieldLoadOp flo when initializers.get(flo.fieldDescriptor()) instanceof TI ti -> {
                                bb.context().mapValue(op.result(), initArgs.get(ti.index()));
                            }
                            case CoreOp.FuncCallOp fco -> {
                                // attach initializers args to all func calls
                                FunctionType newType = FunctionType.functionType(fco.opType().returnType(),
                                        Stream.concat(fco.opType().parameterTypes().stream(), initTypes.stream()).toList());
                                List<Value> newOperands = Stream.concat(bb.context().getValues(fco.operands()).stream(), initArgs.stream()).toList();
                                Op.Result newCall = bb.op(CoreOp.funcCall(fco.funcName(), newType, newOperands));
                                bb.context().mapValue(op.result(), newCall);
                            }
                            default -> {
                                bb.op(op);
                            }
                        }
                        return bb;
                    }));
        }).toList()), initializers.sequencedKeySet());
    }

    static CoreOp.FuncOp resolve(MethodHandles.Lookup l, CoreOp.InvokeOp io) {
        try {
            var res = Op.ofMethod(io.invokeDescriptor().resolveToDirectMethod(l));
            if (res.isPresent()) {
                return SSA.transform(res.get());
            }
        } catch (ReflectiveOperationException | IllegalArgumentException _) {}
        return null;
    }

    static CoreOp.ModuleOp transformModule(MethodHandles.Lookup l, CoreOp.ModuleOp module) {
        var paramsToDropMap = new HashMap<String, BitSet>();
        return CoreOp.module(module.functionTable().sequencedValues().stream().map(f
                -> transformFunc(l, f, paramsToDropMap)).toList());
    }

    static CoreOp.FuncOp transformFunc(MethodHandles.Lookup l, CoreOp.FuncOp func, Map<String, BitSet> paramsToDropMap) {
        OnnxPartialEvaluator pe = new OnnxPartialEvaluator();
        pe.evaluate(l, func);

        // ONNX model transformation
        func = transformToOnnx(l, func, pe);

        // remove redundant args from func calls of funcs with already dropped unused parameters
        // functions are listed in post-ordered and recursion is not allowed
        func = removeDropedFuncCallsArgs(func, paramsToDropMap);

        // drop unused parameters and ops
        func = dropUnused(l, func, paramsToDropMap);

        // SSA and drop unused operations transformation
        return SSA.transform(func);
    }

    static CoreOp.FuncOp transformToOnnx(MethodHandles.Lookup l, CoreOp.FuncOp func, OnnxPartialEvaluator pe) {
        FunctionType ft = convertType(l, func.invokableType());
        return CoreOp.func(func.funcName(), ft).body(b -> {
            b.transformBody(func.body(), b.parameters(), toOnnxOpTransformer(l, pe));
        });
    }

    static CoreOp.FuncOp removeDropedFuncCallsArgs(CoreOp.FuncOp func, Map<String, BitSet> paramsToDropMap) {
        return func.transform((bb, op) -> {
            if (op instanceof CoreOp.FuncCallOp fco) {
                BitSet argsToDrop = paramsToDropMap.get(fco.funcName());
                CopyContext cc = bb.context();
                List<Value> newOperands = IntStream.range(0, fco.operands().size()).filter(i -> !argsToDrop.get(i)).mapToObj(i -> cc.getValue(fco.operands().get(i))).toList();
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
    }

    static CoreOp.FuncOp dropUnused(MethodHandles.Lookup l, CoreOp.FuncOp func, Map<String, BitSet> paramsToDropMap) {
        BitSet paramsToDrop = new BitSet();
        paramsToDropMap.put(func.funcName(), paramsToDrop);
        List<Block.Parameter> usedParameters = func.parameters().stream()
                .filter(v -> {
                    if (v.uses().isEmpty()) {
                        paramsToDrop.set(v.index());
                        return false;
                    } else {
                        return true;
                    }
                })
                .toList();

        var funcType = FunctionType.functionType(func.invokableType().returnType(), usedParameters.stream().map(Value::type).toList());
        return CoreOp.func(func.funcName(), funcType).body(bob -> {
            bob.context().mapValues(usedParameters, bob.parameters());
            bob.transformBody(func.body(), List.of(), (b, op) -> {
                // Drop any non-terminating operation whose result is not used
                if (op instanceof Op.Terminating || !op.result().uses().isEmpty() || op instanceof CoreOp.FuncOp) {
                    b.op(op);
                }
                return b;
            });
        });
    }

    static OpTransformer toOnnxOpTransformer(MethodHandles.Lookup l, OnnxPartialEvaluator pe) {
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
                        opArgs.add(convertType(l, op.resultType()));
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
                            opArgs.add(transformBodyTranslateTypes(l, lambda, bb, toOnnxOpTransformer(l, pe)));
                        }
                    } else if (opClass == ExplicitOnnxOps.Loop.class) {
                        // Explicit transformation of nested body
                        var lambda = (CoreOp.LambdaOp)(((Op.Result)op.operands().get(3)).op());
                        opArgs.add(transformBodyTranslateTypes(l, lambda, bb, toOnnxOpTransformer(l, pe)));
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
                    Op.Result result = bb.op(CoreOp.funcCall(fco.funcName(), convertType(l, fco.opType()), bb.context().getValues(fco.operands())));
                    bb.context().mapValue(fco.result(), result);
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
                convertType(l, inputType.returnType()),
                inputType.parameterTypes().stream().map(pt -> convertType(l, pt)).toList());

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
                            tupleComponentTypes.add(convertType(l, JavaType.type(pt)));
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
                            tupleComponentTypes.add(convertType(l, JavaType.parameterized(JavaType.type(Tensor.class), e)));
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
                    tupleComponentTypes.add(convertType(l, e));
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

    static FunctionType convertType(MethodHandles.Lookup l, FunctionType t) {
        return FunctionType.functionType(convertType(l, t.returnType()), t.parameterTypes().stream().map(pt -> convertType(l, pt)).toList());
    }

    // @@@ Map of Java tensor types to ONNX tensor types
    // @@@ Shape??
    static TypeElement convertType(MethodHandles.Lookup l, TypeElement type) {
        if (type instanceof ClassType ct) {
            if (ct.rawType().equals(TENSOR_CLASS)) {
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
