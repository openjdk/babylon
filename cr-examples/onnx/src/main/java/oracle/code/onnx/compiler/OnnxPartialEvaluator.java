/*
 * Copyright (c) 2025, Oracle and/or its affiliates. All rights reserved.
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

package oracle.code.onnx.compiler;

import jdk.incubator.code.*;
import jdk.incubator.code.op.CoreOp;
import jdk.incubator.code.type.*;
import oracle.code.onnx.OnnxOperators;
import oracle.code.onnx.ir.OnnxOp;
import oracle.code.onnx.ir.OnnxOps;

import java.lang.invoke.*;
import java.lang.reflect.Array;
import java.util.*;
import java.util.stream.Collectors;
import java.util.stream.Stream;
import oracle.code.onnx.Tensor;
import oracle.code.onnx.ir.ExplicitOnnxOps;

final class OnnxPartialEvaluator {

    static final JavaType ONNX_OPERATORS_CLASS = JavaType.type(OnnxOperators.class);
    static final TypeElement TENSOR_RAW_CLASS = JavaType.type(Tensor.class);

    // Map from ONNX operator invocation to evaluated attributes
    final Map<CoreOp.InvokeOp, List<Object>> evaluatedAttributes;

    // Operations that depend directly or indirectly on input parameters
    // The operations' results are not evaluated
    final Set<Op> unevaluatedOperations;

    final Object initializersReceiver;
    final List<Object> initializers;

    public OnnxPartialEvaluator(Object initializersReceiver) {
        this.evaluatedAttributes = new HashMap<>();
        this.unevaluatedOperations = new HashSet<>();
        this.initializersReceiver = initializersReceiver;
        this.initializers = new ArrayList<>();
    }

    public <T extends Op & Op.Invokable>
    void evaluate(MethodHandles.Lookup l, T op) {
        var ev = new HashMap();

        interpretEntryBlock(l, op.body().entryBlock(), new OpContext(), ev);

//        evaluatedAttributes.forEach((invokeOp, objects) -> {
//            System.out.println(invokeOp.invokeDescriptor().name() + " -> " + objects);
//        });
    }


    @SuppressWarnings("serial")
    public static final class InterpreterException extends RuntimeException {
        private InterpreterException(Throwable cause) {
            super(cause);
        }
    }

    static InterpreterException interpreterException(Throwable cause) {
        return new InterpreterException(cause);
    }

    record BlockContext(Block b, Map<Value, Object> evaluatedValues) {
    }

    static final class OpContext {
        final Deque<BlockContext> stack = new ArrayDeque<>();

        boolean isValueDefined(Value v) {
            // @@@ Only dominating values are accessible
            BlockContext bc = findContext(v);
            return bc != null;
        }

        Object getValue(Value v) {
            // @@@ Only dominating values are accessible
            BlockContext bc = findContext(v);
            if (bc != null) {
                return bc.evaluatedValues.get(v);
            } else {
                throw interpreterException(new IllegalArgumentException("Undefined value: " + v));
            }
        }

        Object setValue(Value v, Object o) {
            BlockContext bc = findContext(v);
            if (bc != null) {
                throw interpreterException(new IllegalArgumentException("Value already defined: " + v));
            }
            stack.peek().evaluatedValues.put(v, o);
            return o;
        }

        BlockContext findContext(Value v) {
            Optional<BlockContext> ob = stack.stream().filter(b -> b.evaluatedValues.containsKey(v))
                    .findFirst();
            return ob.orElse(null);
        }

        boolean contains(Block.Reference s) {
            Block sb = s.targetBlock();
            return stack.stream().anyMatch(bc -> bc.b.equals(sb));
        }

        void successor(Block.Reference sb) {
            List<Object> sbValues = sb.arguments().stream().map(this::getValue).toList();

            Block b = sb.targetBlock();
            Map<Value, Object> bValues = new HashMap<>();
            for (int i = 0; i < sbValues.size(); i++) {
                bValues.put(b.parameters().get(i), sbValues.get(i));
            }

            if (contains(sb)) {
                // if block is already dominating pop back up from the back branch to the block
                // before the successor block
                while (!stack.peek().b.equals(sb.targetBlock())) {
                    stack.pop();
                }
                stack.pop();
            }
            stack.push(new BlockContext(b, bValues));
        }

        void popTo(BlockContext bc) {
            while (!stack.peek().equals(bc)) {
                stack.pop();
            }
        }
    }

    static final class VarBox
            implements CoreOp.Var<Object> {
        Object value;

        public Object value() {
            return value;
        }

        VarBox(Object value) {
            this.value = value;
        }

        static final Object UINITIALIZED = new Object();
    }

    record TupleRecord(List<Object> components) {
        Object getComponent(int index) {
            return components.get(index);
        }

        TupleRecord with(int index, Object value) {
            List<Object> copy = new ArrayList<>(components);
            copy.set(index, value);
            return new TupleRecord(copy);
        }
    }

    void interpretBody(MethodHandles.Lookup l, Body body,
                       OpContext oc,
                       List<Object> args) {
        List<Block.Parameter> parameters = body.entryBlock().parameters();
        if (parameters.size() != args.size()) {
            throw interpreterException(new IllegalArgumentException(
                    "Incorrect number of arguments arguments"));
        }

        // Map symbolic parameters to runtime arguments
        Map<Value, Object> arguments = new HashMap<>();
        for (int i = 0; i < parameters.size(); i++) {
            arguments.put(parameters.get(i), args.get(i));
        }

        interpretEntryBlock(l, body.entryBlock(), oc, arguments);
    }

    void interpretEntryBlock(MethodHandles.Lookup l, Block entry,
                             OpContext oc,
                             Map<Value, Object> evaluatedValues) {
        assert entry.isEntryBlock();

        // If the stack is not empty it means we are interpreting
        // an entry block with a parent body whose nearest ancestor body
        // is the current context block's parent body
        BlockContext yieldContext = oc.stack.peek();
        assert yieldContext == null ||
                yieldContext.b().parentBody() == entry.parentBody().parentOp().ancestorBody();

        // Note that first block cannot have any successors so the queue will have at least one entry
        oc.stack.push(new BlockContext(entry, evaluatedValues));
        while (true) {
            BlockContext bc = oc.stack.peek();

            // Execute all but the terminating operation
            int nops = bc.b.ops().size();
            try {
                for (int i = 0; i < nops - 1; i++) {
                    Op op = bc.b.ops().get(i);
                    assert !(op instanceof Op.Terminating) : op.opName();

                    Object result = interpretOp(l, oc, op);
                    if (result != null) {
                        oc.setValue(op.result(), result);
                    }
                }
            } catch (InterpreterException e) {
                throw e;
            }

            // Execute the terminating operation
            Op to = bc.b.terminatingOp();
            if (!to.operands().stream().allMatch(oc::isValueDefined)) {
                // Ignore operation if any value is undefined, meaning it is not part of the attribute value space
                unevaluatedOperations.add(to);
            }

            if (to instanceof CoreOp.ConditionalBranchOp cb) {
                boolean p;
                Object bop = oc.getValue(cb.predicate());
                if (bop instanceof Boolean bp) {
                    p = bp;
                } else if (bop instanceof Integer ip) {
                    // @@@ This is required when lifting up from bytecode, since boolean values
                    // are erased to int values, abd the bytecode lifting implementation is not currently
                    // sophisticated enough to recover the type information
                    p = ip != 0;
                } else {
                    throw interpreterException(
                            new UnsupportedOperationException("Unsupported type input to operation: " + cb));
                }
                Block.Reference sb = p ? cb.trueBranch() : cb.falseBranch();
                oc.successor(sb);
            } else if (to instanceof CoreOp.BranchOp b) {
                Block.Reference sb = b.branch();

                oc.successor(sb);
            } else if (to instanceof CoreOp.ReturnOp ret) {
                // @@@ value should not be in scope
                // return rv == null ? null : oc.getValue(rv);
                return;
            } else {
                throw interpreterException(
                        new UnsupportedOperationException("Unsupported terminating operation: " + to.opName()));
            }
        }
    }


    @SuppressWarnings("unchecked")
    public static <E extends Throwable> void eraseAndThrow(Throwable e) throws E {
        throw (E) e;
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

    Object interpretOp(MethodHandles.Lookup l, OpContext oc, Op o) {
        // Invocation to ONNX operator
        // The input operands will be left unevaluated
        // The attribute operands will be evaluated
        // @@@ Clone attributes or disallow subsequent operation
        if (o instanceof CoreOp.InvokeOp io && io.invokeDescriptor().refType().equals(ONNX_OPERATORS_CLASS)) {
            String operatorName = io.invokeDescriptor().name();

            Class<? extends OnnxOp> opClass = onnxOpClassFromName(operatorName);
            OnnxOp.OnnxSchema schema = schemaFromOnnxOpClass(opClass);

            List<OnnxOp.OnnxParameter> inputs = schema.inputs();
//            assert o.operands().subList(0, inputs.size()).stream().noneMatch(oc::isValueDefined);
            List<OnnxOp.OnnxAttribute> attributes = schema.attributes();

            if (opClass == OnnxOps.Constant.class && o.operands().size() == 1) {
                // Specialized one argument invocations
                List<Object> attrs = new ArrayList<>();
                for (OnnxOp.OnnxAttribute attribute : attributes) {
                    if (JavaType.type(attribute.type()).equals(o.operands().getFirst().type())) {
                        attrs.add(Optional.of(oc.getValue(o.operands().getFirst())));
                    } else {
                        attrs.add(Optional.empty());
                    }
                }
                evaluatedAttributes.put(io, attrs);
            } else if (opClass == ExplicitOnnxOps.If.class) {
                // @@@ hard-coded 2 extra undeclared attributes
                List<Object> attrs = o.operands().subList(inputs.size(), inputs.size() + 2).stream()
                        .map(oc::getValue)
                        .toList();
                evaluatedAttributes.put(io, attrs);
            } else {
                for (int i = 0; i < attributes.size(); i++) {
                    assert oc.isValueDefined(o.operands().get(inputs.size() + i)) : operatorName;
                }
                List<Object> attrs = o.operands().subList(inputs.size(), inputs.size() + attributes.size()).stream()
                        .map(oc::getValue)
                        .toList();
                evaluatedAttributes.put(io, attrs);
            }

            unevaluatedOperations.add(o);
            return null;
        } else if (o instanceof CoreOp.FieldAccessOp.FieldLoadOp fo && fo.fieldDescriptor().type() instanceof ClassType ct && ct.rawType().equals(TENSOR_RAW_CLASS)) {
            try {
                if (fo.operands().isEmpty()) {
                    initializers.add(fo.fieldDescriptor().resolveToHandle(l).get());
                } else {
                    try {
                        initializers.add(fo.fieldDescriptor().resolveToHandle(l).get(initializersReceiver));
                    } catch (Exception e) {
                        System.out.println(fo.parentBlock().parentBody().parentOp().toText());
                        System.out.println(fo);
                        System.out.println(initializersReceiver);
                        throw e;
                    }
                }
            } catch (ReflectiveOperationException ex) {
                throw interpreterException(ex);
            }
            unevaluatedOperations.add(o);
            return null;
        } else if (!o.operands().stream().allMatch(oc::isValueDefined)) {
            // Ignore operation if any value is undefined, meaning it is not part of the attribute value space
            unevaluatedOperations.add(o);
            return null;
        }

        switch (o) {
            case CoreOp.ConstantOp co -> {
                if (co.resultType().equals(JavaType.J_L_CLASS)) {
                    return resolveToClass(l, (JavaType) co.value());
                } else {
                    return co.value();
                }
            }
            case CoreOp.InvokeOp co -> {
                MethodType target = resolveToMethodType(l, o.opType());
                MethodHandles.Lookup il = switch (co.invokeKind()) {
                    case STATIC, INSTANCE -> l;
                    case SUPER -> l.in(target.parameterType(0));
                };
                MethodHandle mh = resolveToMethodHandle(il, co.invokeDescriptor(), co.invokeKind());

                mh = mh.asType(target).asFixedArity();
                Object[] values = o.operands().stream().map(oc::getValue).toArray();
                return invoke(mh, values);
            }
            case CoreOp.NewOp no -> {
                Object[] values = o.operands().stream().map(oc::getValue).toArray();
                JavaType nType = (JavaType) no.constructorType().returnType();
                if (nType instanceof ArrayType at) {
                    if (values.length > at.dimensions()) {
                        throw interpreterException(new IllegalArgumentException("Bad constructor NewOp: " + no));
                    }
                    int[] lengths = Stream.of(values).mapToInt(v -> (int) v).toArray();
                    for (int length : lengths) {
                        nType = ((ArrayType) nType).componentType();
                    }
                    return Array.newInstance(resolveToClass(l, nType), lengths);
                } else {
                    MethodHandle mh = constructorHandle(l, no.constructorType());
                    return invoke(mh, values);
                }
            }
            case CoreOp.VarOp vo -> {
                Object v = vo.isUninitialized()
                        ? VarBox.UINITIALIZED
                        : oc.getValue(o.operands().get(0));
                return new VarBox(v);
            }
            case CoreOp.VarAccessOp.VarLoadOp vlo -> {
                // Cast to CoreOp.Var, since the instance may have originated as an external instance
                // via a captured value map
                CoreOp.Var<?> vb = (CoreOp.Var<?>) oc.getValue(o.operands().get(0));
                Object value = vb.value();
                if (value == VarBox.UINITIALIZED) {
                    throw interpreterException(new IllegalStateException("Loading from uninitialized variable"));
                }
                return value;
            }
            case CoreOp.VarAccessOp.VarStoreOp vso -> {
                VarBox vb = (VarBox) oc.getValue(o.operands().get(0));
                vb.value = oc.getValue(o.operands().get(1));
                return null;
            }
            case CoreOp.TupleOp to -> {
                List<Object> values = o.operands().stream().map(oc::getValue).toList();
                return new TupleRecord(values);
            }
            case CoreOp.TupleLoadOp tlo -> {
                TupleRecord tb = (TupleRecord) oc.getValue(o.operands().get(0));
                return tb.getComponent(tlo.index());
            }
            case CoreOp.TupleWithOp two -> {
                TupleRecord tb = (TupleRecord) oc.getValue(o.operands().get(0));
                return tb.with(two.index(), oc.getValue(o.operands().get(1)));
            }
            case CoreOp.FieldAccessOp.FieldLoadOp fo -> {
                if (fo.operands().isEmpty()) {
                    VarHandle vh = fieldStaticHandle(l, fo.fieldDescriptor());
                    return vh.get();
                } else {
                    Object v = oc.getValue(o.operands().get(0));
                    VarHandle vh = fieldHandle(l, fo.fieldDescriptor());
                    return vh.get(v);
                }
            }
            case CoreOp.FieldAccessOp.FieldStoreOp fo -> {
                if (fo.operands().size() == 1) {
                    Object v = oc.getValue(o.operands().get(0));
                    VarHandle vh = fieldStaticHandle(l, fo.fieldDescriptor());
                    vh.set(v);
                } else {
                    Object r = oc.getValue(o.operands().get(0));
                    Object v = oc.getValue(o.operands().get(1));
                    VarHandle vh = fieldHandle(l, fo.fieldDescriptor());
                    vh.set(r, v);
                }
                return null;
            }
            case CoreOp.InstanceOfOp io -> {
                Object v = oc.getValue(o.operands().get(0));
                return isInstance(l, io.type(), v);
            }
            case CoreOp.CastOp co -> {
                Object v = oc.getValue(o.operands().get(0));
                return cast(l, co.type(), v);
            }
            case CoreOp.ArrayLengthOp arrayLengthOp -> {
                Object a = oc.getValue(o.operands().get(0));
                return Array.getLength(a);
            }
            case CoreOp.ArrayAccessOp.ArrayLoadOp arrayLoadOp -> {
                Object a = oc.getValue(o.operands().get(0));
                Object index = oc.getValue(o.operands().get(1));
                return Array.get(a, (int) index);
            }
            case CoreOp.ArrayAccessOp.ArrayStoreOp arrayStoreOp -> {
                Object a = oc.getValue(o.operands().get(0));
                Object index = oc.getValue(o.operands().get(1));
                Object v = oc.getValue(o.operands().get(2));
                Array.set(a, (int) index, v);
                return null;
            }
            case CoreOp.ArithmeticOperation arithmeticOperation -> {
                MethodHandle mh = opHandle(l, o.opName(), o.opType());
                Object[] values = o.operands().stream().map(oc::getValue).toArray();
                return invoke(mh, values);
            }
            case CoreOp.TestOperation testOperation -> {
                MethodHandle mh = opHandle(l, o.opName(), o.opType());
                Object[] values = o.operands().stream().map(oc::getValue).toArray();
                return invoke(mh, values);
            }
            case CoreOp.ConvOp convOp -> {
                MethodHandle mh = opHandle(l, o.opName() + "_" + o.opType().returnType(), o.opType());
                Object[] values = o.operands().stream().map(oc::getValue).toArray();
                return invoke(mh, values);
            }
            case CoreOp.ConcatOp concatOp -> {
                return o.operands().stream()
                        .map(oc::getValue)
                        .map(String::valueOf)
                        .collect(Collectors.joining());
            }
            case CoreOp.LambdaOp lambdaOp -> {
                return lambdaOp;
            }
            case null, default -> throw interpreterException(
                    new UnsupportedOperationException("Unsupported operation: " + o.opName()));
        }
    }

    static MethodHandle opHandle(MethodHandles.Lookup l, String opName, FunctionType ft) {
        MethodType mt = resolveToMethodType(l, ft).erase();
        try {
            return MethodHandles.lookup().findStatic(InvokableLeafOps.class, opName, mt);
        } catch (NoSuchMethodException | IllegalAccessException e) {
            throw interpreterException(e);
        }
    }

    static MethodHandle constructorHandle(MethodHandles.Lookup l, FunctionType ft) {
        MethodType mt = resolveToMethodType(l, ft);

        if (mt.returnType().isArray()) {
            if (mt.parameterCount() != 1 || mt.parameterType(0) != int.class) {
                throw interpreterException(new IllegalArgumentException("Bad constructor descriptor: " + ft));
            }
            return MethodHandles.arrayConstructor(mt.returnType());
        } else {
            try {
                return l.findConstructor(mt.returnType(), mt.changeReturnType(void.class));
            } catch (NoSuchMethodException | IllegalAccessException e) {
                throw interpreterException(e);
            }
        }
    }

    static VarHandle fieldStaticHandle(MethodHandles.Lookup l, FieldRef d) {
        return resolveToVarHandle(l, d);
    }

    static VarHandle fieldHandle(MethodHandles.Lookup l, FieldRef d) {
        return resolveToVarHandle(l, d);
    }

    static Object isInstance(MethodHandles.Lookup l, TypeElement d, Object v) {
        Class<?> c = resolveToClass(l, d);
        return c.isInstance(v);
    }

    static Object cast(MethodHandles.Lookup l, TypeElement d, Object v) {
        Class<?> c = resolveToClass(l, d);
        return c.cast(v);
    }

    static MethodHandle resolveToMethodHandle(MethodHandles.Lookup l, MethodRef d, CoreOp.InvokeOp.InvokeKind kind) {
        try {
            return d.resolveToHandle(l, kind);
        } catch (ReflectiveOperationException e) {
            throw interpreterException(e);
        }
    }

    static VarHandle resolveToVarHandle(MethodHandles.Lookup l, FieldRef d) {
        try {
            return d.resolveToHandle(l);
        } catch (ReflectiveOperationException e) {
            throw interpreterException(e);
        }
    }

    public static MethodType resolveToMethodType(MethodHandles.Lookup l, FunctionType ft) {
        try {
            return MethodRef.toNominalDescriptor(ft).resolveConstantDesc(l);
        } catch (ReflectiveOperationException e) {
            throw interpreterException(e);
        }
    }

    public static Class<?> resolveToClass(MethodHandles.Lookup l, TypeElement d) {
        try {
            if (d instanceof JavaType jt) {
                return (Class<?>) jt.erasure().resolve(l);
            } else {
                throw new ReflectiveOperationException();
            }
        } catch (ReflectiveOperationException e) {
            throw interpreterException(e);
        }
    }

    static Object invoke(MethodHandle m, Object... args) {
        try {
            return m.invokeWithArguments(args);
        } catch (RuntimeException | Error e) {
            throw e;
        } catch (Throwable e) {
            eraseAndThrow(e);
            throw new InternalError("should not reach here");
        }
    }
}
