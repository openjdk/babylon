package oracle.code.onnx.compiler;

import java.lang.invoke.MethodHandles;
import java.lang.reflect.*;
import java.util.*;
import java.util.function.Predicate;
import java.util.stream.Collectors;
import java.util.stream.IntStream;
import java.util.stream.Stream;
import jdk.incubator.code.*;
import jdk.incubator.code.analysis.NormalizeBlocksTransformer;
import jdk.incubator.code.analysis.SSA;
import jdk.incubator.code.dialect.core.CoreOp;
import jdk.incubator.code.dialect.core.CoreType;
import jdk.incubator.code.dialect.core.FunctionType;
import jdk.incubator.code.dialect.core.TupleType;
import jdk.incubator.code.dialect.java.*;
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

    public record ModuleAndInitializers(CoreOp.ModuleOp module, SequencedCollection<FieldRef> initializers, Map<Value, String> namesMap) {}

    public static ModuleAndInitializers transform(MethodHandles.Lookup l, Quoted quotedLambda) {
        JavaOp.LambdaOp lambda = (JavaOp.LambdaOp) quotedLambda.op();
        assert lambda.parameters().isEmpty();

        List<Value> captures = lambda.capturedValues();
        List<TypeElement> normalizedCaptureTypes = captures.stream()
                .map(v -> v instanceof Op.Result r &&
                        r.op() instanceof CoreOp.VarOp vop &&
                        vop.initOperand() instanceof Block.Parameter p ? p : v)
                .map(Value::type)
                .toList();
        FunctionType ft = CoreType.functionType(lambda.invokableType().returnType(), normalizedCaptureTypes);

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
        Map<Value, String> namesMap = new HashMap<>();
        return new ModuleAndInitializers(transformModule(l, mi.module(), namesMap), mi.initializers(), namesMap);
    }

    static void collectModuleFunctions(MethodHandles.Lookup l, SequencedMap<MethodRef, CoreOp.FuncOp> funcs, Set<CoreOp.FuncOp> doNotInline, CoreOp.FuncOp func) {
        func.traverse(null, (_, op) -> {
            if(op instanceof JavaOp.InvokeOp io && resolve(l, io) instanceof CoreOp.FuncOp f) {
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
            if (op instanceof JavaOp.InvokeOp io && funcs.get(io.invokeDescriptor()) instanceof CoreOp.FuncOp fo) {
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
        record TI(TypeElement type, int index) {}
        var initializers = module.traverse(new LinkedHashMap<FieldRef, TI>(), (i, op) -> {
            if (op instanceof JavaOp.FieldAccessOp.FieldLoadOp flo
                    && (flo.resultType() instanceof ClassType ct && ct.rawType().equals(TENSOR_CLASS)
                     || isRecord(l, flo.resultType())
                     || flo.resultType() instanceof ArrayType at && at.componentType() instanceof ClassType ct && ct.rawType().equals(TENSOR_CLASS)
                    )) {
                var targetType = convertType(l, flo.result());
                // computataion of the tuple size created out of the static array initializer field
                i.compute(flo.fieldDescriptor(), (fd, ti) -> ti == null
                        ? new TI(targetType, i.size())
                        : targetType instanceof TupleType newTt && ti.type() instanceof TupleType oldTt && newTt.componentTypes().size() > oldTt.componentTypes().size()
                                ? new TI(newTt, ti.index())
                                : ti);
            }
            return i;
        });

        if (initializers.isEmpty()) {
            return new ModuleAndInitializers(module, List.of(), null);
        }

        // map all initializers field loads into additional arguments
        List<TypeElement> initTypes = initializers.sequencedValues().stream().map(TI::type).toList();
        return new ModuleAndInitializers(CoreOp.module(module.functionTable().sequencedValues().stream().map(f -> {
            var ft = f.invokableType();
            int argsSize = ft.parameterTypes().size();
            return CoreOp.func(f.funcName(), CoreType.functionType(ft.returnType(), Stream.concat(ft.parameterTypes().stream(), initTypes.stream()).toList()))
                    .body(bob -> bob.transformBody(f.body(), bob.parameters(), (bb, op) -> {
                        List<Block.Parameter> initArgs = bob.parameters().subList(argsSize, bob.parameters().size());
                        switch (op) {
                            // field loads mapped to initializers args
                            case JavaOp.FieldAccessOp.FieldLoadOp flo when initializers.get(flo.fieldDescriptor()) instanceof TI ti -> {
                                bb.context().mapValue(op.result(), initArgs.get(ti.index()));
                            }
                            case CoreOp.FuncCallOp fco -> {
                                // attach initializers args to all func calls
                                FunctionType newType = CoreType.functionType(fco.opType().returnType(),
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
        }).toList()), initializers.sequencedKeySet(), null);
    }

    static CoreOp.FuncOp resolve(MethodHandles.Lookup l, JavaOp.InvokeOp io) {
        try {
            var res = Op.ofMethod(io.invokeDescriptor().resolveToDirectMethod(l));
            if (res.isPresent()) {
                return SSA.transform(evaluate(l, res.get()));
            }
        } catch (ReflectiveOperationException | IllegalArgumentException _) {}
        return null;
    }

    public static CoreOp.FuncOp evaluate(MethodHandles.Lookup l, CoreOp.FuncOp f) {
        try {
            f = f.transform(OpTransformer.LOWERING_TRANSFORMER);
            f = PartialEvaluator.evaluate(l,
                    op -> switch (op) {
                        case CoreOp.ConstantOp _ -> true;
                        case JavaOp.FieldAccessOp.FieldLoadOp _ -> false;
                        case JavaOp.InvokeOp _ -> false;
                        case CoreOp.ReturnOp _ -> false;
                        case JavaOp.NewOp _ -> false;
                        default -> op.result() != null;
                    },
                    new HashSet<>(), f);
            f = cleanUp(f);
        } catch (PartialEvaluator.EvaluationException ee) {
            if (!(ee.getCause() instanceof UnsupportedOperationException)) {
                throw ee;
            }
        }
        return f;
    }

    static CoreOp.FuncOp cleanUp(CoreOp.FuncOp f) {
        return removeUnusedOps(NormalizeBlocksTransformer.transform(f));
    }

    static CoreOp.FuncOp removeUnusedOps(CoreOp.FuncOp f) {
        Predicate<Op> unused = op -> (op instanceof Op.Pure || op instanceof CoreOp.VarOp) &&
                op.result().uses().isEmpty();
        while (f.elements().skip(1).anyMatch(ce -> ce instanceof Op op && unused.test(op))) {
            f = f.transform((block, op) -> {
                if (!unused.test(op)) {
                    block.op(op);
                }
                return block;
            });
        }
        return f;
    }

    static CoreOp.ModuleOp transformModule(MethodHandles.Lookup l, CoreOp.ModuleOp module, Map<Value, String> namesMap) {
        var paramsToDropMap = new HashMap<String, BitSet>();
        return CoreOp.module(module.functionTable().sequencedValues().stream().map(f
                -> transformFunc(l, f, paramsToDropMap, namesMap)).toList());
    }

    static CoreOp.FuncOp transformFunc(MethodHandles.Lookup l, CoreOp.FuncOp func, Map<String, BitSet> paramsToDropMap, Map<Value, String> namesMap) {
        // get original return record class
        Class<?> returnRecordClass = null;
        try {
            if (func.invokableType().returnType() instanceof ClassType ct && ct.rawType().resolve(l) instanceof Class cls && cls.isRecord()) {
                returnRecordClass = cls;
            }
        } catch (ReflectiveOperationException e) {
            throw new RuntimeException(e);
        }

        OnnxPartialEvaluator pe = new OnnxPartialEvaluator();
        pe.evaluate(l, func);

        // ONNX model transformation
        func = transformToOnnx(l, func, pe);

        // remove redundant args from func calls of funcs with already dropped unused parameters
        // functions are listed in post-ordered and recursion is not allowed
        func = removeDropedFuncCallsArgs(func, paramsToDropMap);

        // drop unused parameters and ops
        func = dropUnused(l, func, paramsToDropMap);

        // collect param names
        String[] paramNames = new String[func.parameters().size()];
        for (int i = 0; i < paramNames.length; i++) {
            if (func.parameters().get(i).uses().iterator().next().op() instanceof CoreOp.VarOp vo && !vo.varName().isEmpty()) {
                paramNames[i] = vo.varName();
            }
        }

        // SSA and drop unused operations transformation
        func = SSA.transform(func);

        // map param names
        for (int i = 0; i < paramNames.length; i++) {
            if (paramNames[i] != null) {
                namesMap.put(func.parameters().get(i), paramNames[i]);
            }
        }
        // map return tuple names from the original record components
        if (returnRecordClass != null
                && func.body().entryBlock().terminatingOp() instanceof CoreOp.ReturnOp ro
                && ro.operands().getFirst() instanceof Op.Result or
                && or.op() instanceof CoreOp.TupleOp to) {
            var rcs = returnRecordClass.getRecordComponents();
            for (int i = 0; i < to.operands().size(); i++) {
                namesMap.put(to.operands().get(i), rcs[i].getName());
            }
        }

        return func;
    }

    static CoreOp.FuncOp transformToOnnx(MethodHandles.Lookup l, CoreOp.FuncOp func, OnnxPartialEvaluator pe) {
        FunctionType ft = convertType(l, func);
        var func2 = CoreOp.func(func.funcName(), ft).body(b -> {
            b.transformBody(func.body(), b.parameters(), toOnnxOpTransformer(l, pe));
        });
        // double transformation to fix return type by the returned tuple type
        return CoreOp.func(func2.funcName(), convertType(l, func2)).body(b -> b.transformBody(func2.body(), b.parameters(), OpTransformer.COPYING_TRANSFORMER));
    }

    static CoreOp.FuncOp removeDropedFuncCallsArgs(CoreOp.FuncOp func, Map<String, BitSet> paramsToDropMap) {
        return func.transform((bb, op) -> {
            if (op instanceof CoreOp.FuncCallOp fco) {
                BitSet argsToDrop = paramsToDropMap.get(fco.funcName());
                CopyContext cc = bb.context();
                List<Value> newOperands = IntStream.range(0, fco.operands().size()).filter(i -> !argsToDrop.get(i)).mapToObj(i -> cc.getValue(fco.operands().get(i))).toList();
                CoreOp.FuncCallOp newCall = CoreOp.funcCall(fco.funcName(),
                                                            CoreType.functionType(fco.opType().returnType(),
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

        var funcType = CoreType.functionType(func.invokableType().returnType(), usedParameters.stream().map(Value::type).toList());
        return CoreOp.func(func.funcName(), funcType).body(bob -> {
            bob.context().mapValues(usedParameters, bob.parameters());
            bob.transformBody(func.body(), List.of(), (b, op) -> {
                // Drop any non-terminating operation whose result is not used
                if (op instanceof Op.Terminating || !op.result().uses().isEmpty() || op instanceof CoreOp.FuncOp || op instanceof CoreOp.VarAccessOp.VarStoreOp) {
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
                case JavaOp.InvokeOp io when io.invokeDescriptor().refType().equals(ONNX_OPERATORS_CLASS) -> {
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
                                if (v instanceof Op.Result r && r.op() instanceof JavaOp.InvokeOp optionalInvoke
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
                                if (v instanceof Op.Result r && r.op() instanceof JavaOp.InvokeOp listInvoke
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
                            var lambda = (JavaOp.LambdaOp)(((Op.Result)op.operands().get(i)).op());
                            opArgs.add(transformBodyTranslateTypes(l, lambda, bb, toOnnxOpTransformer(l, pe)));
                        }
                    } else if (opClass == ExplicitOnnxOps.Loop.class) {
                        // Explicit transformation of nested body
                        var lambda = (JavaOp.LambdaOp)(((Op.Result)op.operands().get(3)).op());
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
                case JavaOp.InvokeOp io when
                        recordComponentAccessToTupleIndex(l, io.invokeDescriptor()) instanceof Integer index -> {
                    Op.Result result = bb.op(CoreOp.tupleLoad(bb.context().getValue(io.operands().getFirst()), index));
                    bb.context().mapValue(io.result(), result);
                }
                // Transform constant array load access
                case JavaOp.ArrayAccessOp.ArrayLoadOp alo -> {
                    var tuple = bb.context().getValue(alo.operands().getFirst());
                    int index = (Integer)((CoreOp.ConstantOp)((Op.Result)alo.operands().get(1)).op()).value();
                    Op.Result result = bb.op(CoreOp.tupleLoad(tuple, index));
                    bb.context().mapValue(alo.result(), result);
                }
                // Transform record construction
                case JavaOp.NewOp no when isRecord(l, no.type()) -> {
                    Op.Result result = bb.op(CoreOp.tuple(no.operands().stream().map(v -> {
                        Value mv = bb.context().getValueOrDefault(v, null);
                        if (mv == null && bb.context().getProperty(skipVars(v)) instanceof List list) {
                            mv = bb.op(CoreOp.tuple(bb.context().getValues((List<Value>) list)));
                        }
                        if (mv == null) System.out.println(no.toText());
                        return mv;
                    }).toList()));
                    bb.context().mapValue(no.result(), result);
                }
                // Transform access to the result of an operator that is a list access
                // @@@ raw use of List::get with constant argument
                case JavaOp.InvokeOp io when io.invokeDescriptor().refType().equals(LIST_CLASS) && io.invokeDescriptor().name().equals("get") -> {
                    Op.Result result = bb.op(JavaOp.invoke(
                            io.invokeDescriptor(),
                            bb.context().getValue(io.operands().getFirst()),
                            bb.op(CoreOp.constant(JavaType.INT, pe.evaluatedAttributes.get(io).getLast()))));
                    bb.context().mapValue(io.result(), result);
                }
                // Skip nested lambdas
                case JavaOp.LambdaOp _ -> {
                }
                case CoreOp.FuncCallOp fco -> {
                    Op.Result result = bb.op(CoreOp.funcCall(fco.funcName(), convertType(l, fco.opType()), bb.context().getValues(fco.operands())));
                    bb.context().mapValue(fco.result(), result);
                }
                case JavaOp.FieldAccessOp.FieldLoadOp flo when flo.operands().isEmpty() -> {
                    Op.Result result = bb.op(JavaOp.fieldLoad(convertType(l, flo.result()), flo.fieldDescriptor()));
                    bb.context().mapValue(flo.result(), result);
                }
                case JavaOp.FieldAccessOp.FieldLoadOp flo -> {
                    Op.Result result = bb.op(JavaOp.fieldLoad(convertType(l, flo.result()), flo.fieldDescriptor(), bb.context().getValue(flo.operands().getFirst())));
                    bb.context().mapValue(flo.result(), result);
                }
                case JavaOp.ArrayAccessOp.ArrayStoreOp aso when aso.operands().get(1) instanceof Op.Result or && or.op() instanceof CoreOp.ConstantOp cop -> {
                    var list  = (List<Value>)bb.context().computePropertyIfAbsent(skipVars(aso.operands().getFirst()), _ -> new ArrayList<Value>());
                    int index = (Integer)cop.value();
                    while (index >= list.size()) list.add(null);
                    list.set(index, aso.operands().get(2));
                }
                case CoreOp.ReturnOp ro when bb.context().getProperty(ro.operands().getFirst()) instanceof List list -> {
                    bb.op(CoreOp._return(bb.op(CoreOp.tuple(bb.context().getValues(list)))));
                }
                // Copy remaining operations, which may be removed later transformations
                default -> {
                    bb.op(op);
                }
            }
            return bb;
        };
    }

    static Value skipVars(Value v) {
        return v instanceof Op.Result or && or.op() instanceof CoreOp.VarAccessOp.VarLoadOp vlo ? vlo.varOp().initOperand() : v;
    }

    // @@@ Copy of Body::transform content to translate types
    static Body.Builder transformBodyTranslateTypes(MethodHandles.Lookup l, Op.Invokable iop,
                                                    Block.Builder ancestor, OpTransformer ot) {
        // @@@ Pass in function type to override that of body's type?
//        return iop.body().transform(cc, ot);
        FunctionType inputType = iop.invokableType();
        FunctionType outputType = CoreType.functionType(
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
            Type type = rc.getGenericType();
            if (type instanceof ParameterizedType pt && pt.getRawType().equals(Optional.class)) {
                type = pt.getActualTypeArguments()[0];
            }
            switch (type) {
                case ParameterizedType pt -> {
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
                case GenericArrayType gat when rc.getAnnotation(OnnxOperators.ArrayLen.class) instanceof OnnxOperators.ArrayLen al-> {
                    var cType = convertType(l, JavaType.type(gat.getGenericComponentType()));
                    var tContent = new TypeElement[al.value()];
                    Arrays.fill(tContent, cType);
                    tupleComponentTypes.add(CoreType.tupleType(tContent));
                }
                default -> throw new IllegalStateException("Unexpected value: " + rc.getGenericType());
            }
        }

        return CoreType.tupleType(tupleComponentTypes);
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
        return CoreType.functionType(convertType(l, t.returnType()), t.parameterTypes().stream().map(pt -> convertType(l, pt)).toList());
    }

    static FunctionType convertType(MethodHandles.Lookup l, CoreOp.FuncOp fo) {
        return CoreType.functionType(convertType(l, fo.body().entryBlock().terminatingOp().operands().getFirst()), fo.parameters().stream().map(p -> convertType(l, p)).toList());
    }

    static TypeElement convertType(MethodHandles.Lookup l, Value value) {
        // convert 1-dimensional constantly accessed constant arrays into tuples
        if (value.type() instanceof ArrayType at && at.dimensions() == 1) {
            int size = countConstantArraySize(value.uses());
            if (size >= 0) {
                var targs = new TypeElement[size];
                Arrays.fill(targs, convertType(l, at.componentType()));
                return CoreType.tupleType(targs);
            }
        }
        return convertType(l, value.type());
    }

    static int countConstantArraySize(Set<Op.Result> uses) {
        int size = 0;
        for (var use : uses) {
            int s = switch (use.op()) {
                case JavaOp.ArrayAccessOp aao when aao.operands().get(1) instanceof Op.Result or && or.op() instanceof CoreOp.ConstantOp co ->
                    (Integer)co.value() + 1;
                case CoreOp.VarOp _, CoreOp.VarAccessOp.VarLoadOp _ ->
                    countConstantArraySize(use.op().result().uses());
                default -> -1;
            };
            if (s < 0) return -1;
            size = Integer.max(size, s);
        }
        return size;
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
