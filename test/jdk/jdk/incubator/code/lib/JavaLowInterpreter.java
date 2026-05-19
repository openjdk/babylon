import jdk.incubator.code.*;
import jdk.incubator.code.dialect.core.CoreOp;
import jdk.incubator.code.dialect.core.CoreType;
import jdk.incubator.code.dialect.core.FunctionType;
import jdk.incubator.code.dialect.core.VarType;
import jdk.incubator.code.dialect.java.*;

import java.lang.invoke.*;
import java.lang.reflect.Array;
import java.lang.reflect.InvocationHandler;
import java.lang.reflect.Method;
import java.lang.reflect.Proxy;
import java.util.*;
import java.util.stream.Collectors;
import java.util.stream.Stream;

import static java.util.stream.Collectors.toMap;

public class JavaLowInterpreter extends Interpreter {
    public JavaLowInterpreter() {
    }

    static MethodType resolveToMethodType(MethodHandles.Lookup l, FunctionType ft) {
        try {
            return MethodRef.toNominalDescriptor(ft).resolveConstantDesc(l);
        } catch (ReflectiveOperationException e) {
            throw new RuntimeException(e);
        }
    }

    static MethodHandle resolveToMethodHandle(MethodHandles.Lookup l, MethodRef d, JavaOp.InvokeOp.InvokeKind kind) {
        try {
            return d.resolveToHandle(l, kind);
        } catch (ReflectiveOperationException e) {
            throw new RuntimeException(e);
        }
    }

    static Class<?> resolveToClass(MethodHandles.Lookup l, CodeType d) throws ReflectiveOperationException {
        if (!(d instanceof JavaType jt)) {
            throw new InternalError(); // @@@ can be Interpreter exception
        }
        return (Class<?>) jt.erasure().resolve(l);
    }

    static VarHandle resolveToVarHandle(MethodHandles.Lookup l, FieldRef d) throws ReflectiveOperationException {
        return d.resolveToHandle(l);
    }

    static MethodHandle resolveToConstructorHandle(MethodHandles.Lookup l, MethodRef d) throws ReflectiveOperationException {
        return d.resolveToHandle(l, JavaOp.InvokeOp.InvokeKind.SUPER);
    }

    @SuppressWarnings("serial")
    static class OpInterpretationException extends Throwable { // indicate the interpretation of an op throws
        OpInterpretationException(Throwable cause) {
            super(cause);
        }
    }

    static MethodHandle opHandle(MethodHandles.Lookup l, String opName, FunctionType ft) {
        MethodType mt = resolveToMethodType(l, ft).erase();
        try {
            return MethodHandles.lookup().findStatic(ArithmeticAndConvOpImpls.class, opName, mt);
        } catch (NoSuchMethodException | IllegalAccessException e) {
            throw new RuntimeException(e);
        }
    }

    @SuppressWarnings("unchecked")
    static <E extends Throwable> void eraseAndThrow(Throwable e) throws E {
        throw (E) e;
    }

    @Override
    public BlockEffect executeBlock(Block block, Env env) {
        var op = block.firstOp();
        for (; !(op instanceof Op.Terminating); op = block.nextOp(op)) {
            switch (executeOp(op, env)) {
                case TerminatingOpEffect e when e.terminatingOp().equals(fakeThrowOp) -> {
                    // implicit throw
                    JavaEnv jenv = (JavaEnv) env;
                    R r = jenv.findCatchBlock((Throwable) e.operands().getFirst());
                    if (r.catchBlock().isPresent()) {
                        return new SuccessorEffect(r.catchBlock().get(), e.operands(), r.javaEnv());
                    } else {
                        return new TerminatingOpEffect(e.terminatingOp(), e.operands(), r.javaEnv());
                    }
                }
                case TerminatingOpEffect e -> {
                    return e;
                }
                // op completed normally, bind op result in new env, pass control to next op
                case OpResultEffect e -> env = env.bind(op.result(), e.result());
            }
        }

        return executeTerminatingOp((Op & Op.Terminating) op, env);
    }

    private static final class VarBox
            implements CoreOp.Var<Object> {
        Object value;

        public Object value() {
            return value;
        }

        VarBox(Object value) {
            this.value = value;
        }

        static final Object UNINITIALIZED = new Object();
    }

    @Override
    public OpEffect executeOp(Op op, Env e) {
        Object result;
        switch (op) {
            case CoreOp.VarOp o -> {
                Object init = o.isUninitialized() ? VarBox.UNINITIALIZED : e.valueOf(o.initOperand());
                result = new VarBox(init);
            }
            case CoreOp.VarAccessOp.VarLoadOp o -> {
                CoreOp.Var<?> variable = (CoreOp.Var<?>) e.valueOf(o.varOperand());
                Object value = variable.value();
                if (value == VarBox.UNINITIALIZED) {
                    throw new InterpreterException("Loading from uninitialized variable");
                }
                result = value;
            }
            case CoreOp.VarAccessOp.VarStoreOp o -> {
                VarBox variable = (VarBox) e.valueOf(o.varOperand());
                Object v = e.valueOf(o.storeOperand());
                variable.value = v;
                result = null;
            }
            case JavaOp.InvokeOp o -> {
                JavaEnv je = (JavaEnv) e;
                MethodType target = resolveToMethodType(je.l, o.opSignature());
                MethodHandles.Lookup il = switch (o.invokeKind()) {
                    case STATIC, INSTANCE -> je.l;
                    case SUPER -> je.l.in(target.parameterType(0));
                };
                MethodHandle mh = resolveToMethodHandle(il, o.invokeReference(), o.invokeKind());

                mh = mh.asType(target).asFixedArity();
                List<Object> operands = e.valuesOf(o.operands());
                try {
                    result = mh.invokeWithArguments(operands.toArray());
                } catch (Throwable t) {
                    return new TerminatingOpEffect(fakeThrowOp, List.of(t), e);
                }
            }
            case JavaOp.ArithmeticOperation _ -> {
                JavaEnv je = (JavaEnv) e;
                MethodHandle mh = opHandle(je.l, op.externalizeOpName(), op.opSignature());
                List<Object> operands = e.valuesOf(op.operands());
                try {
                    result = mh.invokeWithArguments(operands.toArray());
                } catch (Throwable t) {
                    return new TerminatingOpEffect(fakeThrowOp, List.of(t), e);
                }
            }
            case JavaOp.ConvOp _ -> {
                JavaEnv je = (JavaEnv) e;
                MethodHandle mh = opHandle(je.l, op.externalizeOpName() + "_" + op.opSignature().returnType(), op.opSignature());
                List<Object> operands = e.valuesOf(op.operands());
                try {
                    result = mh.invokeWithArguments(operands.toArray());
                } catch (Throwable t) {
                    return new TerminatingOpEffect(fakeThrowOp, List.of(t), e);
                }
            }
            case CoreOp.ConstantOp o -> {
                if (o.resultType().equals(JavaType.J_L_CLASS)) {
                    try {
                        result = resolveToClass(((JavaEnv) e).l, (JavaType) o.value());
                    } catch (ReflectiveOperationException ex) {
                        return new TerminatingOpEffect(fakeThrowOp, List.of(ex), e);
                    }
                } else {
                    result = o.value();
                }
            }
            case JavaOp.AssertOp o -> {
                TerminatingOpEffect perdEffect = executeBody(o.predicateBody(), List.of(), e);
                boolean b = switch (perdEffect.terminatingOp()) {
                    case CoreOp.YieldOp _ when perdEffect.operands().getFirst() instanceof Boolean av -> av;
                    default -> throw new InternalError();
                };
                if (!b) {
                    Body detailsBody = o.detailsBody();
                    AssertionError ae;
                    if (detailsBody != null) {
                        TerminatingOpEffect messEffect = executeBody(detailsBody, List.of(), e);
                        Object message = switch (messEffect.terminatingOp()) {
                            case CoreOp.YieldOp _ -> messEffect.operands().getFirst();
                            default -> throw new InternalError();
                        };
                        ae = new AssertionError(message);
                    } else {
                        ae = new AssertionError();
                    }
                    return new TerminatingOpEffect(fakeThrowOp, List.of(ae), e);
                }
                result = null;
            }
            case CoreOp.FuncCallOp o -> {
                String name = o.funcName();

                // Find top-level op
                Op top = o;
                while (top.ancestorBody() != null) {
                    top = top.ancestorOp();
                }

                // Ensure top-level op is a module and function name
                // is in the module's function table
                if (top instanceof CoreOp.ModuleOp mop) {
                    CoreOp.FuncOp funcOp = mop.functionTable().get(name);
                    if (funcOp == null) {
                        throw new InterpreterException("Function " + name + " cannot be resolved: not in module's function table");
                    }
                    try {
                        JavaEnv je = (JavaEnv) e;
                        result = new JavaLowInterpreter().executeFuncOp(funcOp, e.valuesOf(o.operands()), je.l);
                    } catch (InterpreterException ex) {
                        throw ex;
                    } catch (Throwable t) {
                        return new TerminatingOpEffect(fakeThrowOp, List.of(t), e);
                    }
                } else {
                    throw new InterpreterException("Function " + name + " cannot be resolved: top level op is not a module");
                }
            }
            case CoreOp.QuotedOp o -> {
                SequencedMap<Value, Object> capturedValues = o.capturedValues().stream()
                        .collect(toMap(v -> v, e::valueOf, (v, _) -> v, LinkedHashMap::new));
                result = new Quoted<>(o.quotedOp(), capturedValues);
            }
            case JavaOp.LambdaOp o -> {
                JavaEnv je = (JavaEnv) e;
                Class<?> fi;
                try {
                    fi = resolveToClass(je.l, o.functionalInterface());
                } catch (ReflectiveOperationException ex) {
                    return new TerminatingOpEffect(fakeThrowOp, List.of(ex), e);
                }

                SequencedMap<Value, Object> capturedValuesAndArguments = o.capturedValues().stream()
                        .collect(toMap(v -> v, e::valueOf, (v, _) -> v, LinkedHashMap::new));
                Object[] capturedArguments = capturedValuesAndArguments.sequencedValues().toArray(Object[]::new);

                MethodHandle execLambdaOpMH;
                try {
                    execLambdaOpMH = MethodHandles.lookup().findVirtual(JavaLowInterpreter.class, "executeLambdaOp",
                            MethodType.methodType(Object.class, JavaOp.LambdaOp.class, MethodHandles.Lookup.class, Object[].class, Object[].class));
                } catch (Throwable t) {
                    throw new InterpreterException(t);
                }
                MethodHandle fProxy = execLambdaOpMH.bindTo(this).bindTo(o).bindTo(je.l).bindTo(capturedArguments)
                        .asCollector(Object[].class, o.parameters().size());
                Object fiInstance = MethodHandleProxies.asInterfaceInstance(fi, fProxy);

                // If a reflectable lambda proxy again to add method Quoted quoted()
                if (o.isReflectable()) {
                    result = Proxy.newProxyInstance(je.l.lookupClass().getClassLoader(), new Class<?>[]{fi},
                            new InvocationHandler() {
                                private final Quoted<JavaOp.LambdaOp> quoted = new Quoted<>(o, capturedValuesAndArguments);

                                @Override
                                public Object invoke(Object proxy, Method method, Object[] args) throws Throwable {
                                    if (Objects.equals(method.getName(), "quoted") && method.getParameterCount() == 0) {
                                        return __internal_quoted();
                                    } else {
                                        // Delegate to FI instance
                                        return method.invoke(fiInstance, args);
                                    }
                                }

                                private Quoted<JavaOp.LambdaOp> __internal_quoted() {
                                    return quoted;
                                }
                            });
                } else {
                    result = fiInstance;
                }
            }
            case CoreOp.TupleOp o -> {
                List<Object> values = o.operands().stream().map(e::valueOf).toList();
                result = values.toArray();
            }
            case CoreOp.TupleLoadOp o -> {
                Object[] arr = (Object[]) e.valueOf(o.operands().getFirst());
                try {
                    result = arr[o.index()];
                } catch (ArrayIndexOutOfBoundsException ex) {
                    return new TerminatingOpEffect(fakeThrowOp, List.of(ex), e);
                }
            }
            case CoreOp.TupleWithOp o -> {
                Object[] arr = (Object[]) e.valueOf(o.operands().getFirst());
                Object[] newArr = Arrays.copyOf(arr, arr.length);
                try {
                    newArr[o.index()] = e.valueOf(o.operands().get(1));
                } catch (ArrayIndexOutOfBoundsException ex) {
                    return new TerminatingOpEffect(fakeThrowOp, List.of(ex), e);
                }
                result = newArr;
            }
            case JavaOp.FieldAccessOp.FieldLoadOp o -> {
                JavaEnv je = (JavaEnv) e;
                VarHandle vh;
                try {
                    vh = resolveToVarHandle(je.l, o.fieldReference());
                } catch (ReflectiveOperationException ex) {
                    return new TerminatingOpEffect(fakeThrowOp, List.of(ex), e);
                }
                try {
                    if (o.operands().isEmpty()) {
                        result = vh.get();
                    } else {
                        Object v = e.valueOf(o.operands().get(0));
                        result = vh.get(v);
                    }
                } catch (RuntimeException ex) {
                    return new TerminatingOpEffect(fakeThrowOp, List.of(ex), e);
                }
            }
            case JavaOp.FieldAccessOp.FieldStoreOp o -> {
                JavaEnv je = (JavaEnv) e;
                VarHandle vh;
                try {
                    vh = resolveToVarHandle(je.l, o.fieldReference());
                } catch (ReflectiveOperationException ex) {
                    return new TerminatingOpEffect(fakeThrowOp, List.of(ex), e);
                }
                try {
                    if (o.operands().size() == 1) {
                        Object v = e.valueOf(o.operands().get(0));
                        vh.set(v);
                    } else {
                        Object r = e.valueOf(o.operands().get(0));
                        Object v = e.valueOf(o.operands().get(1));
                        vh.set(r, v);
                    }
                } catch (RuntimeException ex) {
                    return new TerminatingOpEffect(fakeThrowOp, List.of(ex), e);
                }
                result = null;
            }
            case JavaOp.InstanceOfOp o -> {
                JavaEnv je = (JavaEnv) e;
                Object obj = e.valueOf(o.operands().get(0));
                Class<?> c;
                try {
                    c = resolveToClass(je.l, o.targetType());
                } catch (ReflectiveOperationException ex) {
                    return new TerminatingOpEffect(fakeThrowOp, List.of(ex), e);
                }
                result = c.isInstance(obj);
            }
            case JavaOp.CastOp o  -> {
                Class<?> c;
                try {
                    JavaEnv je = (JavaEnv) e;
                    c = resolveToClass(je.l, o.targetType());
                } catch (ReflectiveOperationException ex) {
                    return new TerminatingOpEffect(fakeThrowOp, List.of(ex), e);
                }
                try {
                    Object v = e.valueOf(o.operands().get(0));
                    result = c.cast(v);
                } catch (ClassCastException ex) {
                    return new TerminatingOpEffect(fakeThrowOp, List.of(ex), e);
                }
            }
            case JavaOp.NewOp o  -> {
                Object[] values = o.operands().stream().map(e::valueOf).toArray();
                MethodHandle mh;
                try {
                    JavaEnv je = (JavaEnv) e;
                    mh = resolveToConstructorHandle(je.l, o.constructorReference());
                } catch (ReflectiveOperationException ex) {
                    return new TerminatingOpEffect(fakeThrowOp, List.of(ex), e);
                }
                try {
                    result = mh.invokeWithArguments(values);
                } catch (Throwable t) {
                    return new TerminatingOpEffect(fakeThrowOp, List.of(t), e);
                }
            }
            case JavaOp.ArrayLengthOp o -> {
                Object a = e.valueOf(o.operands().get(0));
                try {
                    result = Array.getLength(a);
                } catch (RuntimeException ex) {
                    return new TerminatingOpEffect(fakeThrowOp, List.of(ex), e);
                }
            }
            case JavaOp.ArrayAccessOp.ArrayLoadOp o -> {
                Object a = e.valueOf(o.operands().get(0));
                Object index = e.valueOf(o.operands().get(1));
                try {
                    result = Array.get(a, (int) index);
                } catch (RuntimeException ex) {
                    return new TerminatingOpEffect(fakeThrowOp, List.of(ex), e);
                }
            }
            case JavaOp.ArrayAccessOp.ArrayStoreOp o -> {
                Object a = e.valueOf(o.operands().get(0));
                Object index = e.valueOf(o.operands().get(1));
                Object v = e.valueOf(o.operands().get(2));
                try {
                    Array.set(a, (int) index, v);
                } catch (RuntimeException ex) {
                    return new TerminatingOpEffect(fakeThrowOp, List.of(ex), e);
                }
                result = null;
            }
            case JavaOp.ConcatOp o -> {
                result = o.operands().stream()
                        .map(e::valueOf)
                        .map(String::valueOf)
                        .collect(Collectors.joining());
            }
            default -> throw new UnsupportedOperationException(op.toString());
        }
        return new OpResultEffect(result, e);
    }

    /**
     * Exception thrown by the interpreter when execution fails.
     */
    @SuppressWarnings("serial")
    public static final class InterpreterException extends RuntimeException {
        private InterpreterException(Throwable cause) {
            super(cause);
        }
        private InterpreterException(String message) {
            super(message);
        }
    }


    private static final CoreOp.FuncOp fop = CoreOp.func("f",
            CoreType.functionType(JavaType.type(void.class), JavaType.type(Throwable.class))).body(b -> {
       b.op(JavaOp.throw_(b.parameters().get(0)));
    });
    // to treat implicit and explicit exceptions the same
    private static final JavaOp.ThrowOp fakeThrowOp = (JavaOp.ThrowOp) fop.body().entryBlock().terminatingOp();

    @Override
    public <O extends Op & Op.Terminating> BlockEffect executeTerminatingOp(O op, Env e) {
        return switch (op) {
            case CoreOp.BranchOp o -> {
                Block.Reference r = o.successors().getFirst();
                List<Object> arguments = e.valuesOf(r.arguments());
                yield new SuccessorEffect(r.targetBlock(), arguments, e);
            }
            case CoreOp.ConditionalBranchOp o -> {
                boolean p = (boolean) e.valueOf(o.predicateOperand());
                Block.Reference r = p ? o.trueBranch() : o.falseBranch();
                List<Object> arguments = e.valuesOf(r.arguments());
                yield new SuccessorEffect(r.targetBlock(), arguments, e);
            }
            case CoreOp.ReturnOp o -> {
                List<Object> operands = e.valuesOf(o.operands());
                yield new TerminatingOpEffect(o, operands, e);
            }
            case JavaOp.ThrowOp o -> {
                JavaEnv jenv = (JavaEnv) e;
                List<Object> operands = e.valuesOf(o.operands());
                R r = jenv.findCatchBlock((Throwable) operands.getFirst());
                if (r.catchBlock().isPresent()) {
                    yield new SuccessorEffect(r.catchBlock().get(), operands, r.javaEnv());
                }
                yield new TerminatingOpEffect(o, operands, r.javaEnv());
            }
            case CoreOp.YieldOp o -> {
                if (o.ancestorBody().ancestorBody() == null) {
                    throw new IllegalStateException("Yielding to no parent body");
                }
                List<Object> operands = e.valuesOf(o.operands());
                yield new TerminatingOpEffect(o, operands, e);
            }
            case JavaOp.ExceptionRegionEnter o -> {
                JavaEnv je = (JavaEnv) e;
                List<Block> catchBlocks = o.catchReferences().stream().map(Block.Reference::targetBlock).toList();
                je = je.registerCatchBlocks(catchBlocks);
                yield new SuccessorEffect(o.startReference().targetBlock(), je.valuesOf(o.startReference().arguments()), je);
            }
            case JavaOp.ExceptionRegionExit o -> {
                JavaEnv je = (JavaEnv) e;
                je = je.removeCatchBlocks(o.catchReferences().stream().map(Block.Reference::targetBlock).toList());
                yield new SuccessorEffect(o.endReference().targetBlock(), je.valuesOf(o.endReference().arguments()), je);
            }
            default -> throw new UnsupportedOperationException(op.toString());
        };
    }

    private Object executeLambdaOp(JavaOp.LambdaOp op, MethodHandles.Lookup l, Object[] captures, Object[] args) throws Throwable {
        Env e = new JavaEnv(new HashMap<>(), l);
        e = e.bind(op.capturedValues(), Arrays.asList(captures));
        var effect = executeBody(op.body(), Arrays.asList(args), e);
        switch (effect.terminatingOp()) {
            case CoreOp.ReturnOp rop -> {
                return rop.operands().isEmpty() ? null : effect.operands().getFirst();
            }
            case JavaOp.ThrowOp _ -> throw (Throwable) effect.operands().getFirst();
            default -> throw new InternalError(effect.toString());
        }
    }

    private <T extends Op & Op.Invokable> void validateTypes(T op, List<Object> args, MethodHandles.Lookup l) {
        List<Block.Parameter> parameters = op.parameters();
        List<Value> capturedValues = op.capturedValues();
        if (parameters.size() + capturedValues.size() != args.size()) {
            throw new InterpreterException(
                    String.format("Actual #arguments (%d) differs from #parameters (%d) plus #captured arguments (%d)",
                            args.size(), parameters.size(), capturedValues.size()));
        }
        // validate runtime args types
        List<Value> symbolicValues = Stream.concat(parameters.stream(), capturedValues.stream()).toList();
        for (int i = 0; i < symbolicValues.size(); i++) {
            Value sv = symbolicValues.get(i);
            Object rv = args.get(i);
            try {
                JavaType typeToResolve = switch (sv.type()) {
                    // @@@ Deconstruct and test what the var holds
                    case VarType _ -> JavaType.type(CoreOp.Var.class);
                    // Allow reflection to convert between primitive values
                    // @@@ Check conversion compatible
                    case PrimitiveType _ -> JavaType.J_L_OBJECT;
                    case JavaType jt -> jt;
                    default -> throw new InterpreterException("Unexpected type: " + sv.type());
                };
                Class<?> c = typeToResolve.toNominalDescriptor().resolveConstantDesc(l);
                if (rv != null && !c.isInstance(rv)) {
                    throw new InterpreterException(("Runtime argument at position %d has type %s " +
                            "but the corresponding symbolic value has type %s").formatted(i, rv.getClass(), sv.type()));
                }
            } catch (ReflectiveOperationException e) {
                throw new InterpreterException(e);
            }
        }
    }

    public Object executeLambdaOp(JavaOp.LambdaOp op, List<Object> args, MethodHandles.Lookup l) throws Throwable {
        validateTypes(op, args, l);

        Env e = new JavaEnv(new HashMap<>(), l);
        // args = op args + op captures
        e = e.bind(op.capturedValues(), args.subList(op.parameters().size(), args.size()));
        List<Object> arguments = args.subList(0, op.parameters().size());
        var effect = executeBody(op.body(), arguments, e);
        switch (effect.terminatingOp()) {
            case CoreOp.ReturnOp rop -> {
                return rop.operands().isEmpty() ? null : effect.operands().getFirst();
            }
            case JavaOp.ThrowOp _ -> throw (Throwable) effect.operands().getFirst();
            default -> throw new InternalError(effect.toString());
        }
    }

    public Object executeFuncOp(CoreOp.FuncOp op, List<Object> args, MethodHandles.Lookup l) throws Throwable {
        validateTypes(op, args, l);

        Env e = new JavaEnv(new HashMap<>(), l);
        var effect = executeBody(op.body(), args, e);
        switch (effect.terminatingOp()) {
            case CoreOp.ReturnOp rop -> {
                return rop.operands().isEmpty() ? null : effect.operands().getFirst();
            }
            case JavaOp.ThrowOp _ -> throw (Throwable) effect.operands().getFirst();
            default -> throw new InternalError(effect.toString());
        }
    }

    static final class JavaEnv implements Env {
        final Map<Value, Object> bindings;
        final MethodHandles.Lookup l;
        final Deque<List<Block>> catchBlocks;

        public JavaEnv(Map<Value, Object> bindings, MethodHandles.Lookup l) {
            this.bindings = bindings;
            this.l = l;
            this.catchBlocks = new ArrayDeque<>();
        }
        private JavaEnv(Map<Value, Object> bindings, MethodHandles.Lookup l, Deque<List<Block>> catchBlocks) {
            this.bindings = bindings;
            this.l = l;
            this.catchBlocks = catchBlocks;
        }

        Map<Value, Object> newBindings() {
            return new HashMap<>(bindings);
        }

        @Override
        public Env bind(List<? extends Value> symbolicValues, List<Object> runtimeValues) {
            Map<Value, Object> m = newBindings();
            int l = symbolicValues.size();
            for (int i = 0; i < l; i++) {
                m.put(symbolicValues.get(i), runtimeValues.get(i));
            }
            return new JavaEnv(m, this.l, this.catchBlocks);
        }

        @Override
        public Env bind(Value symbolicValue, Object runtimeValue) {
            Map<Value, Object> m = newBindings();
            m.put(symbolicValue, runtimeValue);
            return new JavaEnv(m, l, this.catchBlocks);
        }

        @Override
        public List<Object> valuesOf(List<? extends Value> symbolicValues) {
            List<Object> runtimeValues = new ArrayList<>();
            for (Value symbolicValue : symbolicValues) {
                runtimeValues.add(valueOf(symbolicValue));
            }

            return runtimeValues;
        }

        @Override
        public Object valueOf(Value symbolicValue) {
            if (!bindings.containsKey(symbolicValue)) {
                throw new IllegalArgumentException("Unknown binding for " + symbolicValue);
            }
            return bindings.get(symbolicValue);
        }

        public JavaEnv registerCatchBlocks(List<Block> catchBlocks) {
            var stack = new ArrayDeque<>(this.catchBlocks);
            stack.addFirst(catchBlocks.reversed()); // store catch block from specific to general
            return new JavaEnv(bindings, l, stack);
        }

        public JavaEnv removeCatchBlocks(List<Block> catchBlocks) {
            var stack = new ArrayDeque<>(this.catchBlocks);
            List<Block> l = stack.removeFirst();
            if (!l.equals(catchBlocks)) {
                throw new InternalError();
            }
            return new JavaEnv(bindings, this.l, stack);
        }

        public R findCatchBlock(Throwable t) {
            Block cb = null;
            int blockListToRemove = 0;
            l:
            for (List<Block> blocks : catchBlocks) {
                blockListToRemove++;
                for (Block block : blocks) {
                    Class<?> c;
                    try {
                        c = resolveToClass(l, block.parameters().getFirst().type());
                    } catch (ReflectiveOperationException ex) {
                        throw new InterpreterException(ex);
                    }
                    if (c.isInstance(t)) {
                        cb = block;
                        break l;
                    }
                }
            }

            var cbs = new ArrayDeque<>(catchBlocks);
            while (blockListToRemove > 0) {
                cbs.removeFirst();
                blockListToRemove--;
            }

            return new R(new JavaEnv(bindings, l, cbs), Optional.ofNullable(cb));
        }
    }

    // @@@ find a meaningful name
    record R(JavaEnv javaEnv, Optional<Block> catchBlock) {};
}
