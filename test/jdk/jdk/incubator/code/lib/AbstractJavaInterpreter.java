import jdk.incubator.code.*;
import jdk.incubator.code.dialect.core.CoreOp;
import jdk.incubator.code.dialect.core.CoreType;
import jdk.incubator.code.dialect.core.TupleType;
import jdk.incubator.code.dialect.core.VarType;
import jdk.incubator.code.dialect.java.ClassType;
import jdk.incubator.code.dialect.java.JavaOp;
import jdk.incubator.code.dialect.java.JavaType;
import jdk.incubator.code.dialect.java.PrimitiveType;

import java.lang.invoke.MethodHandle;
import java.lang.invoke.MethodHandleProxies;
import java.lang.invoke.MethodHandles;
import java.lang.invoke.MethodType;
import java.lang.reflect.InvocationHandler;
import java.lang.reflect.Method;
import java.lang.reflect.Proxy;
import java.util.*;
import java.util.stream.IntStream;
import java.util.stream.Stream;

import static java.util.stream.Collectors.toMap;

public abstract class AbstractJavaInterpreter extends Interpreter {
    abstract Env newEnv(MethodHandles.Lookup l);

    <T extends Op & Op.Invokable> Object interpret(T op, List<Object> argsAndCaptures, MethodHandles.Lookup l) {
        validateTypes(op, argsAndCaptures, l);

        return interpret_(op, l,
                argsAndCaptures.subList(op.parameters().size(), argsAndCaptures.size()).toArray(),
                argsAndCaptures.subList(0, op.parameters().size()).toArray());
    }

    <T extends Op & Op.Invokable> Object interpret_(T op, MethodHandles.Lookup l, Object[] captures, Object[] args) {
        Env e = newEnv(l);
        e = e.bind(op.capturedValues(), Arrays.asList(captures));
        var effect = executeBody(op.body(), Arrays.asList(args), e);
        switch (effect.terminatingOp()) {
            case CoreOp.ReturnOp rop -> {
                return rop.operands().isEmpty() ? null : effect.operands().getFirst();
            }
            case JavaOp.ThrowOp _ -> {
                eraseAndThrow((Throwable) effect.operands().getFirst());
                throw new InternalError(); // @@@ shouldn't reach here
            }
            default -> throw new InternalError(effect.toString());
        }
    }

    Object interpretLambdaBody(JavaOp.LambdaOp lambdaOp, MethodHandles.Lookup l, Object[] captures, Object[] args) {
        return interpret_(lambdaOp, l, captures, args);
    }

    static final MethodHandle lambdaBodyInterpreter;
    static {
        try {
            lambdaBodyInterpreter = MethodHandles.lookup().findVirtual(AbstractJavaInterpreter.class, "interpretLambdaBody",
                    MethodType.methodType(Object.class, JavaOp.LambdaOp.class, MethodHandles.Lookup.class, Object[].class, Object[].class));
        } catch (Throwable t) {
            throw new InternalError();
        }
    }

    OpEffect executeLambdaOp(JavaOp.LambdaOp o, Env env, MethodHandle lambdaBodyInterpreter) {
        JavaEnv je = (JavaEnv) env;
        Class<?> fi;
        try {
            fi = resolveToClass(je.l, o.functionalInterface());
        } catch (ReflectiveOperationException ex) {
            return new TerminatingOpEffect(fakeThrowOp, List.of(ex), env);
        }

        SequencedMap<Value, Object> capturedValuesAndArguments = o.capturedValues().stream()
                .collect(toMap(v -> v, env::valueOf, (v, _) -> v, LinkedHashMap::new));
        Object[] capturedArguments = capturedValuesAndArguments.sequencedValues().toArray(Object[]::new);

        MethodHandle fProxy = lambdaBodyInterpreter.bindTo(o).bindTo(je.l).bindTo(capturedArguments)
                .asCollector(Object[].class, o.parameters().size());
        Object fiInstance = MethodHandleProxies.asInterfaceInstance(fi, fProxy);

        Object result;
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
        return new OpResultEffect(result, env);
    }

    private static final CoreOp.FuncOp fop = CoreOp.func("f",
            CoreType.functionType(JavaType.type(void.class), JavaType.type(Throwable.class))).body(b -> {
        b.add(JavaOp.throw_(b.parameters().get(0)));
    });
    // to treat implicit and explicit exceptions the same
    static final JavaOp.ThrowOp fakeThrowOp = (JavaOp.ThrowOp) fop.body().entryBlock().terminatingOp();

    @SuppressWarnings("unchecked")
    private static <E extends Throwable> void eraseAndThrow(Throwable e) throws E {
        throw (E) e;
    }

    private static <T extends Op & Op.Invokable> void validateTypes(T op, List<Object> argsAndCaptures, MethodHandles.Lookup l) {
        List<Block.Parameter> parameters = op.parameters();
        List<Value> capturedValues = op.capturedValues();
        if (parameters.size() + capturedValues.size() != argsAndCaptures.size()) {
            throw new InterpreterException(
                    String.format("Actual #arguments (%d) differs from #parameters (%d) plus #captured arguments (%d)",
                            argsAndCaptures.size(), parameters.size(), capturedValues.size()));
        }
        // validate runtime args and captures types
        List<Value> symbolicValues = Stream.concat(parameters.stream(), capturedValues.stream()).toList();
        for (int i = 0; i < symbolicValues.size(); i++) {
            Value sv = symbolicValues.get(i);
            Object rv = argsAndCaptures.get(i);
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

    static Class<?> resolveToClass(MethodHandles.Lookup l, CodeType d) throws ReflectiveOperationException {
        if (!(d instanceof JavaType jt)) {
            throw new InternalError(); // @@@ can be Interpreter exception
        }
        return (Class<?>) jt.erasure().resolve(l);
    }

    static class JavaEnv implements Env {
        final Map<Value, Object> bindings;
        final MethodHandles.Lookup l;
        final Deque<List<CatchHandler>> catchHandlers;

        JavaEnv(Map<Value, Object> bindings, MethodHandles.Lookup l, Deque<List<CatchHandler>> catchHandlers) {
            this.bindings = bindings;
            this.l = l;
            this.catchHandlers = catchHandlers;
        }

        Env newEnv(Map<Value, Object> m) {
            return new JavaEnv(m, l, catchHandlers);
        }

        JavaEnv newEnv(Deque<List<CatchHandler>> catchHandlers) {
            return new JavaEnv(bindings, l, catchHandlers);
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
            return newEnv(m);
        }

        @Override
        public Env bind(Value symbolicValue, Object runtimeValue) {
            Map<Value, Object> m = newBindings();
            m.put(symbolicValue, runtimeValue);
            return newEnv(m);
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

        public JavaEnv registerCatchHandlers(List<CatchHandler> handlers) {
            var stack = new ArrayDeque<>(catchHandlers);
            stack.addFirst(handlers);
            return newEnv(stack);
        }

        public JavaEnv removeCatchHandlers(List<CatchHandler> handlers) {
            var stack = new ArrayDeque<>(catchHandlers);
            if (!stack.removeFirst().equals(handlers)) {
                throw new InternalError();
            }
            return newEnv(stack);
        }

        @Override
        public BlockEffect onAbruptCompletion(Op op, TerminatingOpEffect eff) {
            Optional<SuccessorEffect> opt = this.findCatchBlock(op.parent(), (Throwable) eff.operands().getFirst());
            if (opt.isPresent()) {
                return opt.get();
            } else {
                JavaEnv newEnv = this.removeAllCatchBlocks();
                return new TerminatingOpEffect(eff.terminatingOp(), eff.operands(), newEnv);
            }
        }

        // @@@ review this area and improve the code
        private Optional<SuccessorEffect> findCatchBlock(Block executedBlock, Throwable t) {
            Block cb = null;
            int handlerListsToRemove = 0;
            l:
            for (List<CatchHandler> handlers : catchHandlers) {
                handlerListsToRemove++;
                for (CatchHandler handler : handlers) {
                    Block block = handler.block();
                    // make sure we are searching for catch block within the same body
                    if (block.parent() != executedBlock.parent()) {
                        break l;
                    }
                    try {
                        if (handler.matches(l, t)) {
                            cb = block;
                            break l;
                        }
                    } catch (ReflectiveOperationException ex) {
                        throw new InterpreterException(ex);
                    }
                }
            }

            if (cb == null) {
                return Optional.empty();
            }

            var rhs = new ArrayDeque<>(catchHandlers);
            while (handlerListsToRemove-- > 0) {
                rhs.removeFirst();
            }

            return Optional.of(new SuccessorEffect(cb, List.of(t), new JavaEnv(bindings, l, rhs)));
        }

        private JavaEnv removeAllCatchBlocks() {
            return newEnv(new ArrayDeque<>());
        }
    }

    record CatchHandler(CodeType catchType, Block block) {

        static List<CatchHandler> of(JavaOp.ExceptionRegionEnter op) {
            List<Block.Reference> references = op.catchReferences();
            List<CodeType> types = op.catchTypes();
            return IntStream.range(0, references.size())
                    .mapToObj(i -> new CatchHandler(types.get(i), references.get(i).targetBlock()))
                    .toList().reversed();
        }

        boolean matches(MethodHandles.Lookup l, Throwable t) throws ReflectiveOperationException {
            return matches(l, catchType, t);
        }

        private static boolean matches(MethodHandles.Lookup l, CodeType catchType, Throwable t)
                throws ReflectiveOperationException {
            return switch (catchType) {
                case TupleType tt -> {
                    boolean matched = false;
                    for (CodeType componentType : tt.componentTypes()) {
                        if (matches(l, componentType, t)) {
                            matched = true;
                            break;
                        }
                    }
                    yield matched;
                }
                case ClassType ct -> resolveToClass(l, ct).isInstance(t);
                case PrimitiveType pt when pt.equals(JavaType.VOID) -> true;
                default -> throw new InterpreterException("Unexpected catch type: " + catchType);
            };
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

        static final Object UNINITIALIZED = new Object();
    }
}
