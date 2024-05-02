/*
 * Copyright (c) 2024, Oracle and/or its affiliates. All rights reserved.
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

package oracle.code.triton;

import java.lang.invoke.MethodHandle;
import java.lang.invoke.MethodHandles;
import java.lang.reflect.Field;
import java.lang.reflect.Method;
import java.lang.reflect.code.*;
import java.lang.reflect.code.analysis.SSA;
import java.lang.reflect.code.op.CoreOp;
import java.lang.reflect.code.op.ExtendedOp;
import java.lang.reflect.code.type.JavaType;
import java.lang.reflect.code.type.VarType;
import java.util.*;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.stream.Stream;

import static java.lang.reflect.code.op.CoreOp.*;
import static java.lang.reflect.code.type.FunctionType.functionType;

public final class TritonTransformer {
    private TritonTransformer() {}

    static final JavaType TYPE_Triton = JavaType.type(Triton.class);

    static final JavaType TYPE_Triton_Test = JavaType.ofString("oracle.code.triton.TritonTest");

    static final JavaType TYPE_Tensor = JavaType.type(Tensor.class);

    static final JavaType TYPE_J_L_MATH = JavaType.type(Math.class);

    public static <O extends Op & Op.Invokable>
    TritonOps.ModuleOp tritonModule(O kernel,
                                    TypeElement rType,
                                    List<? extends TypeElement> argTypes) {
        Map<String, TritonOps.FuncOp> fsymTable = new LinkedHashMap<>();
        tritonFunction(kernel, rType, argTypes, fsymTable);
        return TritonOps.module(fsymTable.values().stream().toList());
    }

    public static <O extends Op & Op.Invokable>
    TritonOps.FuncOp tritonFunction(O javaKernel,
                                    TypeElement rType,
                                    List<? extends TypeElement> argTypes,
                                    Map<String, TritonOps.FuncOp> fsymTable) {
        String name = (javaKernel instanceof FuncOp f) ? f.funcName() : "kernel";
        String signature = signature(name, rType, argTypes);
        if (fsymTable.containsKey(signature)) {
            return fsymTable.get(signature);
        }

        System.out.println(javaKernel.toText());

        Map<Value, TypeElement> valueTypeMap = new HashMap<>();
        Map<Op, Object> opData = new HashMap<>();
        TritonTransformer.typeCheckKernel(javaKernel, argTypes, valueTypeMap, opData);
        TritonTransformer.printTypeMap(javaKernel, valueTypeMap);

        return TritonTransformer.transformToTritonFunction(javaKernel, signature,
                rType, valueTypeMap, opData,
                fsymTable);
    }

    static String signature(String name, TypeElement rType, List<? extends TypeElement> argTypes) {
        StringBuilder sb = new StringBuilder(name);

        for (TypeElement argType : argTypes) {
            sb.append("_");
            if (argType instanceof ConstantType ct) {
                sb.append(ct.value());
            } else {
                sb.append(argType);
            }
        }
        sb.append("_");
        sb.append(rType);
        return sb.toString();
    }

    public static <O extends Op & Op.Invokable> void typeCheckKernel(
            O kernel, List<? extends TypeElement> argTypes,
            Map<Value, TypeElement> valueTypeMap, Map<Op, Object> opData) {
        kernel.traverse(null, CodeElement.opVisitor((o, op) -> {
            switch (op) {
                case Op.Invokable fop -> {
                    List<Block.Parameter> parameters = fop.body().entryBlock().parameters();
                    for (int i = 0; i < parameters.size(); i++) {
                        valueTypeMap.put(parameters.get(i), argTypes.get(i));
                    }
                }
                case VarOp _, VarAccessOp.VarLoadOp _ -> {
                    Value init = op.operands().get(0);
                    valueTypeMap.put(op.result(), valueTypeMap.get(init));
                }
                case VarAccessOp.VarStoreOp _ -> {
                    Value var = op.operands().get(0);
                    TypeElement varType = valueTypeMap.get(var);
                    Value v = op.operands().get(1);
                    TypeElement vType = valueTypeMap.get(v);
                    if (!varType.equals(vType)) {
                        throw new IllegalStateException("Storing to variable with different type: "
                                + varType + " <- " + vType);
                    }

                    valueTypeMap.put(op.result(), valueTypeMap.get(var));
                }
                case ConstantOp cop -> {
                    valueTypeMap.put(op.result(), new ConstantType(op.result().type(), cop.value()));
                }
                case ArithmeticOperation _ -> {
                    TypeElement t = checkWithTypeInterpreter(op, op.opName(), valueTypeMap);
                    valueTypeMap.put(op.result(), t);
                }
                case FieldAccessOp.FieldLoadOp flop -> {
                    if (!flop.operands().isEmpty()) {
                        throw new IllegalStateException("Unsupported field load: " + flop.fieldDescriptor());
                    }

                    Field f;
                    try {
                        f = flop.fieldDescriptor().resolveToMember(MethodHandles.lookup());
                    } catch (ReflectiveOperationException e) {
                        throw new IllegalStateException("Unsupported field load: " + flop.fieldDescriptor(), e);
                    }
                    Object value;
                    try {
                        value = f.get(null);
                    } catch (IllegalAccessException e) {
                        throw new IllegalStateException("Unsupported field load: " + f, e);
                    }
                    valueTypeMap.put(op.result(), new ConstantType(JavaType.type(f.getType()), value));
                }
                case InvokeOp iop when iop.invokeDescriptor().refType().equals(JavaType.J_L_INTEGER) -> {
                    // Box
                    if (iop.invokeDescriptor().name().equals("valueOf")) {
                        Value a = op.operands().get(0);
                        valueTypeMap.put(op.result(), valueTypeMap.get(a));
                    } else {
                        throw new UnsupportedOperationException("Unsupported invocation on Integer: " + iop.invokeDescriptor());
                    }
                }
                case InvokeOp iop when iop.invokeDescriptor().refType().equals(TYPE_J_L_MATH) -> {
                    String name = iop.invokeDescriptor().name();
                    if (name.equals("max") || name.equals("min")) {
                        Value a = op.operands().get(0);
                        valueTypeMap.put(op.result(), valueTypeMap.get(a));
                    } else {
                        throw new UnsupportedOperationException("Unsupported invocation on Math: " + iop.invokeDescriptor());
                    }
                }
                case InvokeOp iop when iop.invokeDescriptor().refType().equals(TYPE_Tensor) -> {
                    if (iop.invokeDescriptor().name().equals("type")) {
                        Value a = op.operands().get(0);
                        valueTypeMap.put(op.result(), valueTypeMap.get(a));
                    } else {
                        throw new UnsupportedOperationException("Unsupported invocation on Tensor: " + iop.invokeDescriptor());
                    }
                }
                case InvokeOp iop when iop.invokeDescriptor().refType().equals(TYPE_Triton) -> {
                    TypeElement t = checkWithTypeInterpreter(op, iop.invokeDescriptor().name(), valueTypeMap);
                    valueTypeMap.put(op.result(), t);
                }
                case InvokeOp iop when iop.invokeDescriptor().refType().equals(TYPE_Triton_Test) -> {
                    TypeElement t = checkWithTypeInterpreter(op, iop.invokeDescriptor().name(), valueTypeMap);
                    valueTypeMap.put(op.result(), t);
                }
                case ExtendedOp.JavaForOp fop -> {
                    SimpleCountedForLoopInfo li = new SimpleCountedForLoopInfo(fop);
                    opData.put(fop, li);

                    TypeElement type = fop.init().yieldType();
                    if (type instanceof VarType vt && vt.valueType().equals(JavaType.INT)) {
                        for (Body b : List.of(fop.cond(), fop.update(), fop.loopBody())) {
                            valueTypeMap.put(b.entryBlock().parameters().get(0), JavaType.INT);
                        }
                    } else {
                        throw new IllegalStateException();
                    }
                }
                case TestOperation _ -> {
                }
                case ExtendedOp.JavaContinueOp _ -> {
                }
                case YieldOp _ -> {
                }
                case ReturnOp _ -> {
                }
                default -> throw new UnsupportedOperationException("Unsupported operation: " + op);
            }

            return null;
        }));
    }

    static TypeElement checkWithTypeInterpreter(Op op, String name, Map<Value, TypeElement> valueTypeMap) {
        // Obtain associated type-based method
        MethodHandle mh;
        try {
            Optional<Method> om = Stream.of(TritonTypeInterpreter.class.getDeclaredMethods())
                    .filter(m -> m.getName().equals(name))
                    .findFirst();
            mh = MethodHandles.lookup().unreflect(
                    om.orElseThrow(() -> new NoSuchMethodException(name)));
        } catch (ReflectiveOperationException e) {
            throw new IllegalStateException(name, e);
        }

        // Invoke with the values' types
        List<TypeElement> operandTypes = op.operands().stream().map(valueTypeMap::get).toList();
        try {
            return (TypeElement) mh.invokeWithArguments(operandTypes.toArray(Object[]::new));
        } catch (Throwable e) {
            throw new IllegalStateException(mh.toString(), e);
        }
    }

    // @@@ type check tensor shapes
    static class TritonTypeInterpreter {
        private TritonTypeInterpreter() {
        }

        //                 int programId(@Constant int axis) {
        public static JavaType programId(ConstantType axis) {
            assert axis.cType().equals(JavaType.INT);
            int axisValue = (int) axis.value();
            if (axisValue < 0 || axisValue > 3) {
                throw new IllegalStateException();
            }

            return JavaType.INT;
        }

        //                Tensor arange(@Constant int start, @Constant int end)
        public static TensorType arange(ConstantType start, ConstantType end) {
            assert start.cType().equals(JavaType.INT);
            assert end.cType().equals(JavaType.INT);

            int startValue = (int) start.value();
            int endValue = (int) end.value();

            return new TensorType(JavaType.INT, List.of(endValue - startValue));
        }

        //                Tensor expand(Tensor a, int axis) {
        public static TensorType expand(TensorType a, ConstantType axis) {
            assert axis.cType().equals(JavaType.INT);
            int axisValue = (int) axis.value();

            List<Integer> s = new ArrayList<>(a.shape());
            if (axisValue < s.size()) {
                s.add(axisValue, 1);
            } else {
                for (int i = 0; i <= (axisValue - s.size()); i++) {
                    s.add(1);
                }
            }
            return new TensorType(a.eType(), s);
        }

        //                Tensor load(Tensor ptr, Tensor mask)
        public static TensorType load(TensorType ptr, TensorType mask) {
            checkTensorShape(ptr, mask);
            if (ptr.eType() instanceof PtrType eptr) {
                return new TensorType(eptr.rType(), ptr.shape());
            }

            throw new IllegalStateException();
        }

        //            void store(Tensor ptr, Tensor value, Tensor mask)
        public static void store(TensorType ptr, TensorType value, TensorType mask) {
            if (!(ptr.eType() instanceof PtrType)) {
                throw new IllegalStateException();
            }
        }

        //                Tensor zeros(TensorType type)
        public static TensorType zeros(ConstantType eType, ConstantType... cShape) {
            List<Integer> shape = Stream.of(cShape).map(s -> (int) s.value()).toList();
            return new TensorType((TypeElement) eType.value(), shape);
        }

        //                Tensor broadcast(Object o, TensorType type)
        public static TensorType broadcast(TypeElement o, TensorType type) {
            if (o instanceof TensorType ot) {
                // @@@
                if (ot.shape().size() != type.shape().size()) {
                    throw new IllegalStateException();
                }
                o = ot.eType();
            } if (o instanceof ConstantType oc) {
                o = oc.cType();
            }
            return new TensorType(o, type.shape());
        }

        public static TensorType joinShape(TensorType a, TensorType b) {
            return checkTensorTypes(a, b);
        }

        //          Tensor add(Number a, Number b)
        //             Ptr add(Ptr a, int offset)
        public static TypeElement add(TypeElement a, TypeElement b) {
            // @@@ Pass additional argument for checking ptr
            return binary(a, b);
        }

        public static TypeElement sub(TypeElement a, TypeElement b) {
            return binary(a, b);
        }

        public static TypeElement mul(TypeElement a, TypeElement b) {
            return binary(a, b);
        }

        public static TypeElement div(TypeElement a, TypeElement b) {
            return binary(a, b);
        }

        public static TypeElement mod(TypeElement a, TypeElement b) {
            return binary(a, b);
        }

        public static TypeElement and(TypeElement a, TypeElement b) {
            return binary(a, b);
        }

        public static TypeElement cdiv(TypeElement a, TypeElement b) {
            a = reduceScalarType(a);
            b = reduceScalarType(b);
            if (!a.equals(JavaType.INT) && !b.equals(JavaType.INT)) {
                throw new IllegalStateException();
            }
            return a;
        }

        //          Number conv(Type t, Number a) {
        public static TypeElement conv(ConstantType eType, TypeElement a) {
            return convTypes(eType, a);
        }

        public static TypeElement convTypes(ConstantType eType, TypeElement a) {
            if (a instanceof TensorType tb) {
                TypeElement e = convScalarTypes(eType, tb.eType());
                return new TensorType(e, tb.shape());
            } else {
                return convScalarTypes(eType, a);
            }
        }

        public static TypeElement convScalarTypes(ConstantType eType, TypeElement a) {
            TypeElement t = (TypeElement) eType.value();
            if (t.equals(Float16.FLOAT_16_TYPE) && a.equals(JavaType.FLOAT)) {
                return Float16.FLOAT_16_TYPE;
            } else if (t.equals(a)) {
                return t;
            } else {
                // @@@ Conversions;
                throw new IllegalStateException();
            }
        }

        //          Tensor exp(Tensor a)
        public static TypeElement exp(TypeElement a) {
            return unary(a);
        }

        static TypeElement unary(TypeElement a) {
            return a;
        }

        //                Tensor compare(Number a, Number b, @Constant CompareKind ck) {
        public static TypeElement compare(TypeElement a, TypeElement b, ConstantType kind) {
            assert kind.cType().equals(JavaType.type(Triton.CompareKind.class));

            return binary(a, b);
        }

        //                Tensor dot(Tensor a, Tensor b)
        public static TensorType dot(TensorType a, TensorType b) {
            if (a.shape().size() != 2 || b.shape().size() != 2) {
                throw new IllegalStateException();
            }

            if (!a.shape().get(1).equals(b.shape().get(0))) {
                throw new IllegalStateException();
            }

            if (a.eType() != b.eType()) {
                // @@@ Conversion, type checking
                throw new IllegalStateException();
            }

            // Computed result is tensor of floats, regardless of inputs
            return new TensorType(JavaType.FLOAT, List.of(a.shape().get(0), b.shape().get(1)));
        }


        //                Tensor max(Tensor a, @Constant int axis) {
        public static TypeElement max(TensorType a, ConstantType axis) {
            return reduce(a, axis);
        }

        //                Tensor sum(Tensor a, @Constant int axis) {
        public static TypeElement sum(TensorType a, ConstantType axis) {
            return reduce(a, axis);
        }

        static TypeElement reduce(TensorType a, ConstantType axis) {
            assert axis.cType().equals(JavaType.INT);
            int axisValue = (int) axis.value();
            if (axisValue < 0 || axisValue > 3) {
                throw new IllegalStateException();
            }

            List<Integer> reduceShape = new ArrayList<>();
            for (int i = 0; i < a.shape().size(); i++) {
                if (i != axisValue) {
                    reduceShape.add(a.shape().get(i));
                } else {
                    reduceShape.add(1);
                }
            }

            if (reduceShape.size() == 1 && reduceShape.getFirst() == 1) {
                return a.eType();
            } else {
                return new TensorType(a.eType(), reduceShape);
            }
        }

        // @@@ Test
        public static void consume(TypeElement a) {
        }


        static TypeElement binary(TypeElement a, TypeElement b) {
            if (a instanceof TensorType ta && b instanceof TensorType tb) {
                return checkTensorTypes(ta, tb);
            } else if (a instanceof TensorType ta) {
                return new TensorType(checkScalarTypes(ta.eType(), b), ta.shape());
            } else if (b instanceof TensorType tb) {
                return new TensorType(checkScalarTypes(a, tb.eType()), tb.shape());
            } else {
                return checkScalarTypes(a, b);
            }
        }

        static TensorType checkTensorTypes(TensorType a, TensorType b) {
            List<Integer> s = checkTensorShape(a, b);
            TypeElement e = checkScalarTypes(a.eType(), b.eType());
            return new TensorType(e, s);
        }

        static List<Integer> checkTensorShape(TensorType a, TensorType b) {
            if (a.shape().size() != b.shape().size()) {
                // Shape mismatch
                throw new IllegalStateException();
            }

            List<Integer> s = new ArrayList<>();
            for (int i = 0; i < a.shape().size(); i++) {
                int ad = a.shape().get(i);
                int bd = b.shape().get(i);

                // Expand dimensions
                int d;
                if (ad == bd) {
                    d = ad;
                } else {
                    if (ad != 1 && bd == 1) {
                        d = ad;
                    } else if (ad == 1) {
                        d = bd;
                    } else {
                        // Shape mismatch
                        throw new IllegalStateException();
                    }
                }

                s.add(d);
            }

            return s;
        }

        static TypeElement checkScalarTypes(TypeElement a, TypeElement b) {
            // @@@ Optional ptr checking
            if (a instanceof PtrType) {
                if (!b.equals(JavaType.INT)) {
                    throw new IllegalStateException();
                }
            } else if (b instanceof PtrType) {
                // Pointer must be first argument
                throw new IllegalStateException();
            } else if (a instanceof ConstantType || b instanceof ConstantType) {
                return checkScalarTypes(reduceScalarType(a), reduceScalarType(b));
            } else if (!a.equals(b)) {
                // @@@ Conversion
                throw new IllegalStateException();
            }
            return a;
        }

        static TypeElement reduceScalarType(TypeElement a) {
            return a instanceof ConstantType ct ? ct.cType() : a;
        }
    }

    public static <O extends Op & Op.Invokable> TritonOps.FuncOp transformToTritonFunction(
            O kernel,
            String signature,
            TypeElement rType,
            Map<Value, TypeElement> valueTypeMap, Map<Op, Object> opData,
            Map<String, TritonOps.FuncOp> fsymTable) {
        TritonOps.FuncOp ttKernel = TritonOps.func(signature, functionType(rType))
                .body(fblock -> {
                    // Process kernel parameters
                    List<Value> args = new ArrayList<>();
                    for (Block.Parameter kp : kernel.body().entryBlock().parameters()) {
                        TypeElement type = valueTypeMap.get(kp);
                        if (type instanceof ConstantType ct) {
                            // Constant
                            Op.Result cr = fblock.op(ArithMathOps.constant(
                                    ct.cType(), ct.value()));
                            args.add(cr);
                        } else {
                            args.add(fblock.parameter(type));
                        }
                    }

                    // Transform kernel body
                    fblock.transformBody(kernel.body(), args, (kblock, op) -> {
                        return transformToTritonOperation(kblock, op, valueTypeMap, opData, fsymTable);
                    });
                });

        ttKernel = cleanup(ttKernel);
        fsymTable.put(ttKernel.funcName(), ttKernel);
        return ttKernel;
    }

    static Block.Builder transformToTritonOperation(Block.Builder kblock, Op op,
                                                    Map<Value, TypeElement> valueTypeMap, Map<Op, Object> opData,
                                                    Map<String, TritonOps.FuncOp> fsymTable) {
        // @@@ Avoid constructing for each operation -- block builder passed as argument or a scoped value
        TritonBuilderInterpreter tbi = new TritonBuilderInterpreter(fsymTable, kblock);
        CopyContext cc = kblock.context();
        switch (op) {
            case VarOp varOp -> {
                // @@@ Cannot copy op because the result type
                //     is derived from init type
                Value init = cc.getValue(op.operands().get(0));
                Op.Result r = kblock.op(var(varOp.varName(), init));
                cc.mapValue(op.result(), r);
            }
            case ConstantOp cop -> {
                TypeElement t = valueTypeMap.get(cop.result());
                if (t instanceof ConstantType ct) {
                    Op.Result r = kblock.op(ArithMathOps.constant(
                            ct.cType(), ct.value()));
                    cc.mapValue(op.result(), r);
                } else {
                    kblock.op(op);
                }
            }
            case ArithmeticOperation _ -> {
                Value result = tbi.build(op, op.opName(), valueTypeMap);
                if (result != null) {
                    cc.mapValue(op.result(), result);
                }
            }
            case InvokeOp iop when iop.invokeDescriptor().refType().equals(JavaType.J_L_INTEGER) -> {
                // Replace box with its value
                Value a = cc.getValue(op.operands().get(0));
                cc.mapValue(op.result(), a);
            }
            case InvokeOp iop when iop.invokeDescriptor().refType().equals(TYPE_J_L_MATH) -> {
                String name = iop.invokeDescriptor().name();
                if (name.equals("max")) {
                    Value a = cc.getValue(op.operands().get(0));
                    Value b = cc.getValue(op.operands().get(1));

                    Op.Result result = kblock.op(ArithMathOps.maximum(a, b));
                    cc.mapValue(op.result(), result);
                } else if (name.equals("min")) {
                    Value a = cc.getValue(op.operands().get(0));
                    Value b = cc.getValue(op.operands().get(1));

                    Op.Result result = kblock.op(ArithMathOps.minimum(a, b));
                    cc.mapValue(op.result(), result);
                }
            }
            case InvokeOp iop when iop.invokeDescriptor().refType().equals(TYPE_Tensor) -> {
                if (iop.invokeDescriptor().name().equals("type")) {
                    // Replace with constant operation to produce tensor type.
                    // Result may be used, but transitively it will be removed due to no uses
                    // contributing to the computation
                    Value a = op.operands().get(0);
                    TensorType aType = (TensorType) valueTypeMap.get(a);
                    Op.Result result = kblock.op(CoreOp.constant(iop.resultType(), aType));
                    cc.mapValue(op.result(), result);
                    valueTypeMap.put(result, aType);
                }
                // Remove
            }
            case InvokeOp iop when iop.invokeDescriptor().refType().equals(TYPE_Triton) -> {
                Value result = tbi.build(op, iop.invokeDescriptor().name(), valueTypeMap);
                if (result != null) {
                    cc.mapValue(op.result(), result);
                }
            }
            case InvokeOp iop when iop.invokeDescriptor().refType().equals(TYPE_Triton_Test) -> {
                Value result = tbi.build(op, iop.invokeDescriptor().name(), valueTypeMap);
                if (result != null) {
                    cc.mapValue(op.result(), result);
                }
            }
            case ExtendedOp.JavaForOp fop -> {
                transformToSCFFor(cc, kblock, fop, valueTypeMap, opData, fsymTable);
            }
            case ReturnOp rop -> {
                if (rop.operands().isEmpty()) {
                    kblock.op(TritonOps.return_());
                } else {
                    kblock.op(TritonOps.return_(
                            cc.getValue(rop.returnValue())));
                }
            }
            default -> kblock.op(op);
        }
        return kblock;
    }

    static void transformToSCFFor(CopyContext cc, Block.Builder kblock, ExtendedOp.JavaForOp fop,
                                  Map<Value, TypeElement> valueTypeMap, Map<Op, Object> opData,
                                  Map<String, TritonOps.FuncOp> fsymTable) {
        Body body = fop.loopBody();

        // Hoist expressions for start, end, and step
        SimpleCountedForLoopInfo li = (SimpleCountedForLoopInfo) opData.get(fop);
        Value start = null;
        for (Op o : li.startExpression()) {
            transformToTritonOperation(kblock, o, valueTypeMap, opData, fsymTable);
            start = cc.getValue(o.result());
        }
        Value end = null;
        for (Op o : li.endExpression()) {
            transformToTritonOperation(kblock, o, valueTypeMap, opData, fsymTable);
            end = cc.getValue(o.result());
        }
        Value step = null;
        for (Op o : li.stepExpression()) {
            transformToTritonOperation(kblock, o, valueTypeMap, opData, fsymTable);
            step = cc.getValue(o.result());
        }

        // Obtain captured vars
        // true == stores
        // false == loads only
        Map<Boolean, Set<Value>> capturedVars = capturedVars(body);
        Set<Value> capturedAndStoredVars = capturedVars.get(true);

        // Get load values
        // Loaded values are hoisted out of the loop body
        Map<Value, Value> loadValues = new HashMap<>();
        for (Value v : capturedVars.get(false)) {
            Value load = kblock.op(varLoad(cc.getValue(v)));
            valueTypeMap.put(load, valueTypeMap.get(v));
            loadValues.put(v, load);
        }

        // Get iteration values -- represented by captured vars that are stored to in the loop
        // The SCF for operation returns the iteration values of the last loop iteration, which
        // are then to be stored to the iteration variables
        List<Value> iterValues = new ArrayList<>();
        for (Value v : capturedAndStoredVars) {
            iterValues.add(kblock.op(varLoad(cc.getValue(v))));
        }

        // @@@ Build in java code model, then transform?
        SCFOps.ForOp scffor = SCFOps.for_(kblock.parentBody(), start, end, step, iterValues)
                // Ensure existing context is used
                .body(CopyContext.create(cc), builder -> {
                    // Create index var initialized from entry block parameter
                    Value index = builder.parameters().get(0);
                    valueTypeMap.put(index, JavaType.INT);
                    Value varIndex = builder.op(var("index", index));
                    valueTypeMap.put(varIndex, JavaType.INT);
                    builder.context().mapValue(body.entryBlock().parameters().get(0), varIndex);

                    // Create iter vars initialized from entry block parameters
                    int pi = 1;
                    for (Value v : capturedAndStoredVars) {
                        TypeElement type = valueTypeMap.get(v);
                        Value iter = builder.parameters().get(pi++);
                        valueTypeMap.put(iter, type);
                        Value varIter = builder.op(var(Integer.toString(pi), iter));
                        valueTypeMap.put(varIter, type);
                        builder.context().mapValue(v, varIter);
                    }

                    // Transform the Java for body into the SCF for body
                    builder.transformBody(body, List.of(), (block, op) -> {
                        // Yield iter values
                        if (op instanceof ExtendedOp.JavaContinueOp) {
                            // Replace with yield of loaded vars
                            List<Value> yieldValues = new ArrayList<>();
                            for (Value value : capturedAndStoredVars) {
                                Value varIter = block.context().getValue(value);
                                Value v = block.op(varLoad(varIter));
                                yieldValues.add(v);
                            }
                            block.op(SCFOps.yield_(yieldValues));
                        } else if (op instanceof VarAccessOp.VarLoadOp) {
                            // Replace with value loaded immediately before loop
                            Value v = op.operands().get(0);
                            if (capturedVars.get(false).contains(v)) {
                                block.context().mapValue(op.result(), loadValues.get(v));
                            } else {
                                block.op(op);
                            }
                        } else {
                            block = transformToTritonOperation(block, op, valueTypeMap, opData, fsymTable);
                        }
                        return block;
                    });
                });
        Op.Result forResult = kblock.op(scffor);

        // Assign back result to iter vars
        if (capturedAndStoredVars.size() == 1) {
            for (Value v : capturedAndStoredVars) {
                kblock.op(varStore(cc.getValue(v), forResult));
            }
        } else {
            int i = 0;
            for (Value v : capturedAndStoredVars) {
                kblock.op(varStore(cc.getValue(v),
                        kblock.op(tupleLoad(forResult, i++))));
            }
        }
    }

    static Map<Boolean, Set<Value>> capturedVars(Body body) {
        Map<Boolean, Set<Value>> capturedValues = new HashMap<>();
        capturedValues.put(false, new LinkedHashSet<>());
        capturedValues.put(true, new LinkedHashSet<>());

        capturedVars(capturedValues, new ArrayDeque<>(), body);
        return capturedValues;
    }

    static void capturedVars(Map<Boolean, Set<Value>> capturedVars, Deque<Body> bodyStack, Body body) {
        bodyStack.push(body);

        for (Block b : body.blocks()) {
            for (Op op : b.ops()) {
                // @@@ Nested bodies
                if (!op.bodies().isEmpty()) {
                    throw new IllegalStateException();
                }
//                for (Body childBody : op.bodies()) {
//                    capturedAndUpdatedVars(capturedValues, bodyStack, childBody);
//                }

                if (op instanceof VarAccessOp) {
                    Value v = op.operands().get(0);
                    if (!bodyStack.contains(v.declaringBlock().parentBody())) {
                        if (op instanceof VarAccessOp.VarStoreOp) {
                            capturedVars.get(true).add(v);
                            capturedVars.get(false).remove(v);
                        } else if (!capturedVars.get(true).contains(v)) {
                            capturedVars.get(false).add(v);
                        }
                    }
                }
            }
        }

        bodyStack.pop();
    }

    public static final ScopedValue<Boolean> SV_SSA = ScopedValue.newInstance();

    static TritonOps.FuncOp cleanup(TritonOps.FuncOp f) {
        // Remove var ops
        boolean doSSA = SV_SSA.isBound() ? SV_SSA.get() : true;
        if (doSSA) {
            f = SSA.transform(f);
        }
        // Remove unused ops
        f = f.transform((fblock, op) -> {
            if (op instanceof Op.Pure && op.result().uses().isEmpty()) {
                return fblock;
            } else if (op instanceof VarAccessOp.VarLoadOp && op.result().uses().isEmpty()) {
                return fblock;
            }

            fblock.op(op);
            return fblock;
        });
        return f;
    }

    static class TritonBuilderInterpreter {
        final Map<String, TritonOps.FuncOp> fsymTable;
        final Block.Builder block;

        TritonBuilderInterpreter(Map<String, TritonOps.FuncOp> fsymTable, Block.Builder block) {
            this.fsymTable = fsymTable;
            this.block = block;
        }

        Value build(Op op, String name, Map<Value, TypeElement> valueTypeMap) {
            // Obtain associated type-based method
            MethodHandle mh;
            try {
                Optional<Method> om = Stream.of(TritonBuilderInterpreter.class.getDeclaredMethods())
                        .filter(m -> m.getName().equals(name))
                        .findFirst();
                mh = MethodHandles.lookup().unreflect(
                        om.orElseThrow(() -> new NoSuchMethodException(name)));
            } catch (ReflectiveOperationException e) {
                throw new IllegalStateException(e);
            }

            List<Object> iArgs = new ArrayList<>();
            iArgs.add(this);
            iArgs.add(valueTypeMap.get(op.result()));
            iArgs.add(op.result());
            for (Value o : op.operands()) {
                iArgs.add(valueTypeMap.get(o));
                iArgs.add(o);
            }
            try {
                return (Value) mh.invokeWithArguments(iArgs.toArray(Object[]::new));
            } catch (Throwable e) {
                throw new IllegalStateException(e);
            }
        }


        public Value programId(TypeElement rType, Op.Result r,
                               ConstantType axisType, Value axis) {
            return block.op(TritonOps.getProgramId(
                    (int) axisType.value()));
        }

        public Value arange(TensorType rType, Op.Result r,
                            ConstantType startType, Value start,
                            ConstantType endType, Value end) {
            return block.op(TritonOps.makeRange(
                    (int) startType.value(),
                    (int) endType.value()));
        }

        public Value expand(TensorType rType, Op.Result r,
                            TensorType aType, Value a,
                            ConstantType axisType, Value axis) {
            return block.op(TritonOps.expand(
                    (int) axisType.value(),
                    rType,
                    block.context().getValue(a)));
        }

        public Value zeros(TensorType rType, Op.Result r,
                           ConstantType aType, Value a,
                           Object... constantsAndValues) {
            Object zero;
            try {
                JavaType zeroType = (JavaType) aType.value();
                zero = MethodHandles.zero(zeroType.resolve(MethodHandles.lookup())).invoke();
            } catch (Throwable e) {
                throw new RuntimeException(e);
            }
            return block.op(ArithMathOps.constant(rType, zero));
        }

        public Value load(TensorType rType, Op.Result r,
                          TensorType ptrType, Value ptr,
                          TensorType maskType, Value mask) {
            broadcastConversionRight(ptrType, maskType, mask);
            return block.op(TritonOps.load(
                    rType,
                    block.context().getValue(ptr),
                    block.context().getValue(mask)));
        }

        public Value store(TensorType rType, Op.Result r,
                           TensorType ptrType, Value ptr,
                           TensorType valueType, Value value,
                           TensorType maskType, Value mask) {
            broadcastConversionRight(ptrType, valueType, value);
            broadcastConversionRight(ptrType, maskType, mask);
            return block.op(TritonOps.store(
                    block.context().getValue(ptr),
                    block.context().getValue(value),
                    block.context().getValue(mask)));
        }

        public Value broadcast(TensorType rType, Op.Result r,
                               TypeElement oType, Value o,
                               TensorType tensorTypeType, Value tensorType) {
            // @@@ tt.splat with scalar operand, tt.broadcast with tensor operand
            if (oType instanceof TensorType) {
                return block.op(TritonOps.broadcast(
                        rType,
                        block.context().getValue(o)));
            } else {
                return block.op(TritonOps.splat(
                        rType,
                        block.context().getValue(o)));
            }
        }

        public Value joinShape(TensorType rType, Op.Result r,
                               TensorType aType, Value a,
                               TensorType bType, Value b) {
            // Replace with constant operation to produce tensor type.
            // Result may be used, but transitively it will be removed due to no uses
            // contributing to the computation
            return block.op(CoreOp.constant(JavaType.type(TensorType.class), r.type()));
        }


        public Value add(TypeElement rType, Op.Result r,
                         TypeElement aType, Value a,
                         TypeElement bType, Value b) {
            broadcastConversion(rType, aType, a, bType, b);
            a = block.context().getValue(a);
            b = block.context().getValue(b);

            if (rType instanceof PtrType ||
                    rType instanceof TensorType t && t.eType() instanceof PtrType) {
                return block.op(TritonOps.addptr(a, b));
            } else {
                return block.op(ArithMathOps.add(a, b));
            }
        }

        public Value sub(TypeElement rType, Op.Result r,
                         TypeElement aType, Value a,
                         TypeElement bType, Value b) {
            broadcastConversion(rType, aType, a, bType, b);
            a = block.context().getValue(a);
            b = block.context().getValue(b);

            return block.op(ArithMathOps.sub(a, b));
        }

        public Value mul(TypeElement rType, Op.Result r,
                         TypeElement aType, Value a,
                         TypeElement bType, Value b) {
            broadcastConversion(rType, aType, a, bType, b);
            a = block.context().getValue(a);
            b = block.context().getValue(b);

            return block.op(ArithMathOps.mul(a, b));
        }

        public Value div(TypeElement rType, Op.Result r,
                         TypeElement aType, Value a,
                         TypeElement bType, Value b) {
            broadcastConversion(rType, aType, a, bType, b);
            a = block.context().getValue(a);
            b = block.context().getValue(b);

            return block.op(ArithMathOps.div(a, b));
        }

        public Value mod(TypeElement rType, Op.Result r,
                         TypeElement aType, Value a,
                         TypeElement bType, Value b) {
            broadcastConversion(rType, aType, a, bType, b);
            a = block.context().getValue(a);
            b = block.context().getValue(b);

            return block.op(ArithMathOps.rem(a, b));
        }

        public Value and(TypeElement rType, Op.Result r,
                         TypeElement aType, Value a,
                         TypeElement bType, Value b) {
            broadcastConversion(rType, aType, a, bType, b);
            a = block.context().getValue(a);
            b = block.context().getValue(b);

            return block.op(ArithMathOps.and(a, b));
        }

        public Value dot(TensorType rType, Op.Result r,
                         TypeElement aType, Value a,
                         TypeElement bType, Value b) {
            a = block.context().getValue(a);
            b = block.context().getValue(b);

            return block.op(TritonOps.dot(rType, a, b));
        }

        public Value cdiv(TypeElement rType, Op.Result r,
                          TypeElement aType, Value a,
                          TypeElement bType, Value b) {
            a = block.context().getValue(a);
            b = block.context().getValue(b);

            TritonOps.FuncOp cdiv = tritonFunction(Functions.getJavaCodeModel("cdiv"),
                    rType, List.of(aType, bType),
                    fsymTable);
            // @@@ Generalize
            List<Value> args = new ArrayList<>();
            if (!(aType instanceof ConstantType)) {
                args.add(a);
            }
            if (!(bType instanceof ConstantType)) {
                args.add(b);
            }
            return block.op(TritonOps.call(cdiv, args));
        }

        public Value conv(TypeElement rType, Op.Result r,
                          ConstantType tType, Value t,
                          TypeElement aType, Value a) {
            a = block.context().getValue(a);

            TypeElement rScalarType;
            TypeElement aScalarType;
            if (rType instanceof TensorType rTensorType && aType instanceof TensorType aTensorType) {
                rScalarType = rTensorType.eType();
                aScalarType = aTensorType.eType();
            } else {
                rScalarType = rType;
                aScalarType = aType;
            }

            if (rScalarType.equals(Float16.FLOAT_16_TYPE) && aScalarType.equals(JavaType.FLOAT)) {
                return block.op(ArithMathOps.trunc(rType, a));
            } else if (rType.equals(aType)) {
                return a;
            } else {
                throw new IllegalStateException();
            }
        }

        public Value exp(TritonType rType, Op.Result r,
                         TritonType aType, Value a) {
            return block.op(ArithMathOps.exp(
                    block.context().getValue(a)));
        }

        public Value compare(TensorType rType, Op.Result r,
                             TypeElement aType, Value a,
                             TypeElement bType, Value b,
                             ConstantType compareType, Value compare) {
            Triton.CompareKind ck = (Triton.CompareKind) compareType.value();

            ArithMathOps.CompareOp.CompareKind ack = switch (ck) {
                case LessThan -> ArithMathOps.CompareOp.CompareKind.slt;
                default -> throw new UnsupportedOperationException("Unsupported comparison: " + ck);
            };

            broadcastConversion(rType, aType, a, bType, b);
            a = block.context().getValue(a);
            b = block.context().getValue(b);

            return block.op(ArithMathOps.cmp(ack, a, b));
        }


        public Value max(TypeElement rType, Op.Result r,
                         TensorType xType, Value x,
                         ConstantType axisType, Value axis) {
            TritonOps.FuncOp f = tritonFunction(Functions.getJavaCodeModel("max"),
                    rType, List.of(rType, rType), fsymTable);
            return reduce(rType, r, xType, x, axisType, axis, f);
        }

        public Value sum(TypeElement rType, Op.Result r,
                         TensorType xType, Value x,
                         ConstantType axisType, Value axis) {
            TritonOps.FuncOp f = tritonFunction(Functions.getJavaCodeModel("sum"),
                    rType, List.of(rType, rType), fsymTable);
            return reduce(rType, r, xType, x, axisType, axis, f);
        }

        Value reduce(TypeElement rType, Op.Result r,
                     TensorType xType, Value x,
                     ConstantType axisType, Value axis,
                     TritonOps.FuncOp f) {
            int axisConstant = (int) axisType.value();

            String signature = "reduce_" + f.funcName() + "_" + axisConstant;
            TritonOps.FuncOp rf = fsymTable.computeIfAbsent(signature,
                    s -> reduce(rType, xType, axisConstant, s, f));

            return block.op(TritonOps.call(rf, block.context().getValue(x)));
        }

        static TritonOps.FuncOp reduce(TypeElement elementType,
                                       TensorType tensorType,
                                       int axisConstant,
                                       String name, TritonOps.FuncOp scalarFunc) {
            return TritonOps.func(name,
                            functionType(elementType, tensorType))
                    .body(fblock -> {
                        TritonOps.ReduceOp reduceOp = TritonOps.reduce(fblock.parentBody(),
                                        axisConstant, fblock.parameters().get(0),
                                        functionType(elementType, elementType, elementType))
                                .body(rblock -> {
                                    Block.Parameter a = rblock.parameters().get(0);
                                    Block.Parameter b = rblock.parameters().get(1);
                                    Op.Result _r = rblock.op(TritonOps.call(scalarFunc, a, b));
                                    rblock.op(TritonOps.reduceReturn(_r));
                                });

                        Op.Result opr = fblock.op(reduceOp);
                        fblock.op(TritonOps.return_(opr));
                    });
        }

        // @@@ Test
        public Value consume(TypeElement rType, Op.Result r,
                             TypeElement aType, Value a) {
            return block.op(TritonTestOps.consume(block.context().getValue(a)));
        }

        void broadcastConversion(TypeElement rType,
                                 TypeElement aType, Value a,
                                 TypeElement bType, Value b) {
            Value ma = block.context().getValue(a);
            Value mb = block.context().getValue(b);
            if (aType instanceof TensorType at && bType instanceof TensorType bTensorType) {
                TensorType rTensorType = (TensorType) rType;
                if (!at.shape().equals(rTensorType.shape())) {
                    ma = block.op(TritonOps.broadcast(rTensorType, ma));
                }
                if (!bTensorType.shape().equals(rTensorType.shape())) {
                    if (rTensorType.eType() instanceof PtrType) {
                        bTensorType = new TensorType(bType, rTensorType.shape());
                        mb = block.op(TritonOps.broadcast(bTensorType, mb));
                    } else {
                        mb = block.op(TritonOps.broadcast(rTensorType, mb));
                    }
                }
            } else if (aType instanceof TensorType) {
                TensorType rTensorType = (TensorType) rType;
                if (rTensorType.eType() instanceof PtrType) {
                    TensorType bTensorType = new TensorType(bType, rTensorType.shape());
                    mb = block.op(TritonOps.splat(bTensorType, mb));
                } else {
                    mb = block.op(TritonOps.splat(rTensorType, mb));
                }
            } else if (bType instanceof TensorType) {
                TensorType rTensorType = (TensorType) rType;
                ma = block.op(TritonOps.splat(rTensorType, ma));
            }
            block.context().mapValue(a, ma);
            block.context().mapValue(b, mb);
        }

        void broadcastConversionRight(TypeElement aType,
                                      TypeElement bType, Value b) {
            Value mb = block.context().getValue(b);
            if (aType instanceof TensorType aTensorType && bType instanceof TensorType bTensorType) {
                if (!bTensorType.shape().equals(aTensorType.shape())) {
                    if (aTensorType.eType() instanceof PtrType) {
                        bTensorType = new TensorType(bTensorType.eType(), aTensorType.shape());
                        mb = block.op(TritonOps.broadcast(bTensorType, mb));
                    } else {
                        mb = block.op(TritonOps.broadcast(aTensorType, mb));
                    }
                }
            } else if (aType instanceof TensorType rTensorType) {
                if (rTensorType.eType() instanceof PtrType) {
                    TensorType bTensorType = new TensorType(bType, rTensorType.shape());
                    mb = block.op(TritonOps.splat(bTensorType, mb));
                } else {
                    mb = block.op(TritonOps.splat(rTensorType, mb));
                }
            }
            block.context().mapValue(b, mb);
        }
    }

    public static <O extends Op & Op.Invokable> void printTypeMap(
            O kernel, Map<Value, TypeElement> valueTypeMap) {
        AtomicInteger valueId = new AtomicInteger();
        Map<Value, Integer> valueIdMap = new LinkedHashMap<>();
        kernel.traverse(null, (o, codeElement) -> {
            switch (codeElement) {
                case FuncOp _ -> {
                    // Ignore
                }
                case Op op when !op.result().type().equals(JavaType.VOID) -> {
                    valueIdMap.put(op.result(), valueId.getAndIncrement());
                }
                case Block block -> {
                    for (Block.Parameter parameter : block.parameters()) {
                        valueIdMap.put(parameter, valueId.getAndIncrement());
                    }
                }
                default -> {
                }
            }
            return null;
        });

        valueIdMap.forEach((value, id) -> {
            TypeElement type = valueTypeMap.get(value);
            if (type != null) {
                System.out.println("%" + id + " : " + value.type() + " -> " + type);
            }
        });
    }
}
