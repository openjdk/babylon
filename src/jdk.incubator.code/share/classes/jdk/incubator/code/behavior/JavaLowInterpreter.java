package jdk.incubator.code.behavior;

import jdk.incubator.code.Block;
import jdk.incubator.code.CodeType;
import jdk.incubator.code.Op;
import jdk.incubator.code.Value;
import jdk.incubator.code.dialect.core.CoreOp;
import jdk.incubator.code.dialect.core.FunctionType;
import jdk.incubator.code.dialect.java.JavaOp;
import jdk.incubator.code.dialect.java.JavaType;
import jdk.incubator.code.dialect.java.MethodRef;

import java.lang.invoke.MethodHandle;
import java.lang.invoke.MethodHandles;
import java.lang.invoke.MethodType;
import java.util.*;

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

    static Class<?> resolveToClass(MethodHandles.Lookup l, CodeType d) {
        if (!(d instanceof JavaType jt)) {
            throw new InternalError();
        }
        try {
            return (Class<?>) jt.erasure().resolve(l);
        } catch (ReflectiveOperationException e) {
            throw new InternalError(e);
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
                case TerminatingOpEffect e -> {
                    // not returnOp, either explicit throw or implicit throw
                    if (!e.operands().isEmpty() && e.operands().getFirst() instanceof Throwable t &&
                            !(e.terminatingOp() instanceof CoreOp.ReturnOp)) {
                        JavaEnv jenv = (JavaEnv) env;
                        Optional<Block> cb = jenv.catchBlocks.stream()
                                .filter(b -> resolveToClass(jenv.l, b.parameters().getFirst().type()).isInstance(t)).findFirst();
                        if (cb.isPresent()) {
                            return new SuccessorEffect(cb.get(), List.of(t), jenv);
                        }
                    }
                    return e;
                }
                // op completed normally, bind op result in new env, pass control to next op
                case OpResultEffect e -> env = env.bind(op.result(), e.result());
            }
        }

        return executeTerminatingOp((Op & Op.Terminating) op, env);
    }

    @Override
    public OpEffect executeOp(Op op, Env e) {
        Object result;
        try {
            switch (op) {
                case CoreOp.VarOp o -> {
                    Object init = e.valueOf(o.initOperand());
                    result = new Object[]{init};
                }
                case CoreOp.VarAccessOp.VarLoadOp o -> {
                    Object[] variable = (Object[]) e.valueOf(o.varOperand());
                    result = variable[0];
                }
                case CoreOp.VarAccessOp.VarStoreOp o -> {
                    Object[] variable = (Object[]) e.valueOf(o.varOperand());
                    Object v = e.valueOf(o.storeOperand());
                    variable[0] = v;
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
                    result = invoke(mh, operands.toArray());
                }
                case JavaOp.ArithmeticOperation _, JavaOp.ConvOp _ -> {
                    JavaEnv je = (JavaEnv) e;
                    MethodHandle mh = opHandle(je.l, op.externalizeOpName(), op.opSignature());
                    List<Object> operands = e.valuesOf(op.operands());
                    result = invoke(mh, operands.toArray());
                }
                case CoreOp.ConstantOp o -> result = o.value();
                case JavaOp.AssertOp o -> {
                    TerminatingOpEffect p = executeBody(o.predicateBody(), List.of(), e);
                    // TODO check if p.op is YieldOp
                    Boolean b = (Boolean) p.operands().getFirst();
                    if (!b) {
                        if (o.bodies().size() > 1) {
                            TerminatingOpEffect m = executeBody(o.bodies().get(1), List.of(), e);
                            String message = (String) m.operands().getFirst();
                            // @@@ we may need to fake out a ThrowOp or a new kind of TerminatingOpEffect
                            return new TerminatingOpEffect(o, List.of(new AssertionError(message)), e);
                        } else {
                            return new TerminatingOpEffect(o, List.of(new AssertionError()), e);
                        }
                    }
                    result = null;
                }
            }
            return new OpResultEffect(result, e);
        } catch (Throwable t) {
            // execution of op throws
            return new TerminatingOpEffect(op, List.of(t), e);
        }
    }

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
                List<Object> operands = e.valuesOf(o.operands());
                yield new TerminatingOpEffect(o, operands, e);
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
            // TODO explicit exception + implicit ones
            default -> throw new UnsupportedOperationException(op.toString());
        };
    }

    public Optional<Object> executeFuncOp(CoreOp.FuncOp op, List<Object> args, MethodHandles.Lookup l) throws Throwable {
        Env e = new JavaEnv(new HashMap<>(), l);

        var ancestorOpEffect = executeBody(op.body(), args, e);
        switch (ancestorOpEffect.terminatingOp()) {
            case CoreOp.ReturnOp rop -> {
                return rop.operands().isEmpty() ? Optional.empty() : Optional.ofNullable(ancestorOpEffect.operands().getFirst());
            }
            case JavaOp.ThrowOp _ -> throw (Throwable) ancestorOpEffect.operands().getFirst();
            // implicit throw e.g. AssertOp or execution of an op throws
            case Op _ when !ancestorOpEffect.operands().isEmpty() && ancestorOpEffect.operands().getFirst() instanceof Throwable t -> throw t;
            default -> throw new InternalError(ancestorOpEffect.toString());
        }
    }

    static final class JavaEnv implements Env {
        final Map<Value, Object> bindings;
        final MethodHandles.Lookup l;
        final Deque<Block> catchBlocks;

        public JavaEnv(Map<Value, Object> bindings, MethodHandles.Lookup l) {
            this.bindings = bindings;
            this.l = l;
            this.catchBlocks = new ArrayDeque<>();
        }
        private JavaEnv(Map<Value, Object> bindings, MethodHandles.Lookup l, Deque<Block> catchBlocks) {
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
            for (Block b : catchBlocks) {
                stack.addFirst(b);
            }
            return new JavaEnv(bindings, l, stack);
        }
    }
}
