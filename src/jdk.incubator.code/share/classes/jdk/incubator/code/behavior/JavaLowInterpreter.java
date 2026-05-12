package jdk.incubator.code.behavior;

import jdk.incubator.code.Block;
import jdk.incubator.code.Body;
import jdk.incubator.code.CodeType;
import jdk.incubator.code.Op;
import jdk.incubator.code.Value;
import jdk.incubator.code.dialect.core.CoreOp;
import jdk.incubator.code.dialect.core.CoreType;
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

    @Override
    public OpEffect executeOp(Op op, Env e) {
        Object result;
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
                try {
                    result = mh.invokeWithArguments(operands.toArray());
                } catch (Throwable t) {
                    return new TerminatingOpEffect(fakeThrowOp, List.of(t), e);
                }
            }
            case JavaOp.ArithmeticOperation _, JavaOp.ConvOp _ -> {
                JavaEnv je = (JavaEnv) e;
                MethodHandle mh = opHandle(je.l, op.externalizeOpName(), op.opSignature());
                List<Object> operands = e.valuesOf(op.operands());
                try {
                    result = mh.invokeWithArguments(operands.toArray());
                } catch (Throwable t) {
                    return new TerminatingOpEffect(fakeThrowOp, List.of(t), e);
                }
            }
            case CoreOp.ConstantOp o -> result = o.value();
            case JavaOp.AssertOp o -> {
                TerminatingOpEffect perdEffect = executeBody(o.predicateBody(), List.of(), e);
                boolean b = switch (perdEffect.terminatingOp()) {
                    case CoreOp.YieldOp _ -> (boolean) perdEffect.operands().getFirst();
                    default -> throw new InternalError();
                };
                if (!b) {
                    Body detailsBody = o.detailsBody();
                    AssertionError ae;
                    if (detailsBody != null) {
                        TerminatingOpEffect messEffect = executeBody(detailsBody, List.of(), e);
                        String message = switch (messEffect.terminatingOp()) {
                            case JavaOp.YieldOp _ -> (String) messEffect.operands().getFirst();
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
                        Optional<Object> r = new JavaLowInterpreter().executeFuncOp(funcOp, e.valuesOf(o.operands()), ((JavaEnv) e).l);
                        result = r.orElse(null);
                    } catch (InterpreterException ex) {
                        throw ex;
                    } catch (Throwable t) {
                        return new TerminatingOpEffect(fakeThrowOp, List.of(t), e);
                    }
                } else {
                    throw new InterpreterException("Function " + name + " cannot be resolved: top level op is not a module");
                }
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
                // catch blocks order in ExceptionRegionExit is the inverse of the blocks in ExceptionRegionEnter
                je = je.removeCatchBlocks(o.catchReferences().stream().map(Block.Reference::targetBlock).toList().reversed());
                yield new SuccessorEffect(o.endReference().targetBlock(), je.valuesOf(o.endReference().arguments()), je);
            }
            default -> throw new UnsupportedOperationException(op.toString());
        };
    }

    public Optional<Object> executeFuncOp(CoreOp.FuncOp op, List<Object> args, MethodHandles.Lookup l) throws Throwable {
        Env e = new JavaEnv(new HashMap<>(), l);

        var effect = executeBody(op.body(), args, e);
        switch (effect.terminatingOp()) {
            case CoreOp.ReturnOp rop -> {
                return rop.operands().isEmpty() ? Optional.empty() : Optional.ofNullable(effect.operands().getFirst());
            }
            case JavaOp.ThrowOp _ -> throw (Throwable) effect.operands().getFirst();
            // implicit throw e.g. AssertOp or execution of an op throws
            case Op o when o.equals(fakeThrowOp) -> throw (Throwable) effect.operands().getFirst();
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
            stack.addFirst(catchBlocks);
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
                    if (resolveToClass(l, block.parameters().getFirst().type()).isInstance(t)) {
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
