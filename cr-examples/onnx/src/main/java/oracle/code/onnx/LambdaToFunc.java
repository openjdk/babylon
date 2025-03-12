package oracle.code.onnx;

import java.lang.invoke.MethodHandles;
import java.lang.reflect.Method;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.function.Predicate;
import jdk.incubator.code.Op;
import jdk.incubator.code.Value;
import jdk.incubator.code.op.CoreOp;
import jdk.incubator.code.type.*;
import oracle.code.onnx.compiler.OnnxTransformer;

public record LambdaToFunc(OnnxTransformer.OnnxFuncOp func, int[] operandsMapping) {

    static final boolean DEBUG = Boolean.getBoolean("oracle.code.onnx.OnnxRuntime.DEBUG");

    static final JavaType TENSOR_CLASS = JavaType.type(Tensor.class);

    public static LambdaToFunc fromLambda(MethodHandles.Lookup l, CoreOp.LambdaOp lambda, Object initializersReceiver) {
        // Shortcut for lambda expressions that call just one method
        if (singleMethodInvocation(lambda) instanceof
                SingleMethod(CoreOp.InvokeOp iop, Map<Value, Value> valueMapping)) {
            System.out.println("!!" + lambda.toText());
            Method m;
            try {
                m = iop.invokeDescriptor().resolveToMethod(l, iop.invokeKind());
            } catch (ReflectiveOperationException e) {
                throw new RuntimeException(e);
            }

            var fOpt = Op.ofMethod(m);
            if (fOpt.isPresent()) {
                CoreOp.FuncOp f = Op.ofMethod(m).orElseThrow();
                var operands = iop.operands();
                var captured = lambda.capturedValues();
                var operandsMapping = new int[iop.operands().size()];
                var fParams = f.parameters();
                for (int i = 0; i < operandsMapping.length; i++) {
                    var opValue = valueMapping.get(operands.get(i));
                    operandsMapping[i] = captured.indexOf(opValue);
                }
                if (DEBUG) {
                    System.out.println(f.toText());
                }
                OnnxTransformer.OnnxFuncOp onnxFunc = OnnxTransformer.transform(l, initializersReceiver, f);
                return new LambdaToFunc(onnxFunc, operandsMapping);
            }
        }
        System.out.println("??" + lambda.toText());

        var capturedValues = lambda.capturedValues();
        var functionType = FunctionType.functionType(lambda.invokableType().returnType(),
                capturedValues.stream().map(Value::type)
                        .map(t -> t instanceof VarType vt ? vt.valueType() : t).toList());
        CoreOp.FuncOp f = CoreOp.func("onnxCode", functionType)
                .body(bb -> {
                    bb.context().mapValues(capturedValues, bb.parameters());
                    for (Op op : lambda.body().entryBlock().ops()) {
                        int i;
                        if (op instanceof CoreOp.VarAccessOp.VarLoadOp load &&
                                (i = capturedValues.indexOf(load.varOp().result())) >= 0) {
                            bb.context().mapValue(op.result(), bb.parameters().get(i)); // remap var load result to block param
                        } else {
                            bb.apply(op);
                        }
                    }
                });
        if (DEBUG) {
            System.out.println(f.toText());
        }
        OnnxTransformer.OnnxFuncOp onnxFunc = OnnxTransformer.transform(l, initializersReceiver, f);

        var operandsMapping = new int[capturedValues.size()];
        for (int i = 0; i < operandsMapping.length; i++) {
            operandsMapping[i] = i;
        }
        return new LambdaToFunc(onnxFunc, operandsMapping);
    }

    record SingleMethod(CoreOp.InvokeOp iop, Map<Value, Value> valueMapping) {}
    static SingleMethod singleMethodInvocation(CoreOp.LambdaOp lop) {
        // Single block
        if (lop.body().blocks().size() > 1) {
            return null;
        }

        Map<Value, Value> valueMapping = new HashMap<>();
        CoreOp.InvokeOp methodRefInvokeOp = extractMethodInvoke(valueMapping, lop.body().entryBlock().ops());
        if (methodRefInvokeOp == null) {
            return null;
        }

        return new SingleMethod(methodRefInvokeOp, valueMapping);
    }

    static CoreOp.InvokeOp extractMethodInvoke(Map<Value, Value> valueMapping, List<Op> ops) {
        CoreOp.InvokeOp methodRefInvokeOp = null;
        for (Op op : ops) {
            switch (op) {
                case CoreOp.VarOp varOp -> {
                    if (isValueUsedWithOp(varOp.result(), o -> o instanceof CoreOp.VarAccessOp.VarStoreOp)) {
                        return null;
                    }
                }
                case CoreOp.VarAccessOp.VarLoadOp varLoadOp -> {
                    Value v = varLoadOp.varOp().result();
                    valueMapping.put(varLoadOp.result(), valueMapping.getOrDefault(v, v));
                }
                case CoreOp.InvokeOp iop when isBoxOrUnboxInvocation(iop) -> {
                    Value v = iop.operands().getFirst();
                    valueMapping.put(iop.result(), valueMapping.getOrDefault(v, v));
                }
                case CoreOp.InvokeOp iop -> {
                    if (methodRefInvokeOp != null) {
                        return null;
                    }

                    for (Value o : iop.operands()) {
                        valueMapping.put(o, valueMapping.getOrDefault(o, o));
                    }
                    methodRefInvokeOp = iop;
                }
                case CoreOp.ReturnOp rop -> {
                    if (methodRefInvokeOp == null) {
                        return null;
                    }
                    Value r = rop.returnValue();
                    if (!(valueMapping.getOrDefault(r, r) instanceof Op.Result invokeResult)) {
                        return null;
                    }
                    if (invokeResult.op() != methodRefInvokeOp) {
                        return null;
                    }
                    assert methodRefInvokeOp.result().uses().size() == 1;
                }
                default -> {
                    return null;
                }
            }
        }

        return methodRefInvokeOp;
    }

    private static boolean isValueUsedWithOp(Value value, Predicate<Op> opPredicate) {
        for (Op.Result user : value.uses()) {
            if (opPredicate.test(user.op())) {
                return true;
            }
        }
        return false;
    }

    // @@@ Move to functionality on JavaType(s)
    static final Set<String> UNBOX_NAMES = Set.of(
            "byteValue",
            "shortValue",
            "charValue",
            "intValue",
            "longValue",
            "floatValue",
            "doubleValue",
            "booleanValue");

    private static boolean isBoxOrUnboxInvocation(CoreOp.InvokeOp iop) {
        MethodRef mr = iop.invokeDescriptor();
        return mr.refType() instanceof ClassType ct && ct.unbox().isPresent() &&
                (UNBOX_NAMES.contains(mr.name()) || mr.name().equals("valueOf"));
    }
}
