package hat;

import hat.buffer.Buffer;
import hat.ifacemapper.MappableIface;
import jdk.incubator.code.*;
import jdk.incubator.code.analysis.Inliner;
import jdk.incubator.code.analysis.SSA;
import jdk.incubator.code.dialect.core.CoreOp;
import jdk.incubator.code.dialect.java.*;

import java.lang.invoke.MethodHandles;
import java.lang.reflect.Method;
import java.util.*;
import java.util.concurrent.atomic.AtomicBoolean;

public class BufferTagger {
    static HashMap<Value, AccessType> accessMap = new HashMap<>();
    static HashMap<Value, Value> remappedVals = new HashMap<>();
    static HashMap<Block, List<Block.Parameter>> blockParams = new HashMap<>();

    public enum AccessType {
        NA(1),
        RO(2),
        WO(4),
        RW(6),
        NOT_BUFFER(0);

        public final int value;

        AccessType(int i) {
            value = i;
        }
    }

    public static void main(String[] args) {
        // Optional<Method> om = Stream.of(ViolaJonesCoreCompute.class.getDeclaredMethods())
        //         .filter(m -> m.getName().equals("rgbToGreyKernel"))
        //         .findFirst();
        //
        // Method m;
        // if (om.isPresent()) {
        //     m = om.get();
        //     Optional<CoreOp.FuncOp> f = Op.ofMethod(m);
        //     f.ifPresent(func -> {
        //         SSA.transform(func.transform(OpTransformer.LOWERING_TRANSFORMER));
        //         BufferTagging.buildAccessMap(accelerator.lookup, BufferTagging.inlineLoop(accelerator.lookup, func));
        //     });
        // }
    }

    public static ArrayList<AccessType> getAccessList(MethodHandles.Lookup l, CoreOp.FuncOp f) {
        CoreOp.FuncOp inlinedFunc = inlineLoop(l, f);
        buildAccessMap(l, inlinedFunc);
        ArrayList<AccessType> accessList = new ArrayList<>();
        for (Block.Parameter p : inlinedFunc.body().entryBlock().parameters()) {
            if (accessMap.containsKey(p)) {
                accessList.add(accessMap.get(p));
            } else if (getClass(l, p.type()) instanceof Class<?> c &&
                    MappableIface.class.isAssignableFrom(c)) {
                accessList.add(AccessType.NA);
            } else {
                accessList.add(AccessType.NOT_BUFFER);
            }
        }
        return accessList;
    }

    public static CoreOp.FuncOp inlineLoop(MethodHandles.Lookup l, CoreOp.FuncOp f) {
        CoreOp.FuncOp func = SSA.transform(f.transform(OpTransformer.LOWERING_TRANSFORMER));
        AtomicBoolean changed = new AtomicBoolean(true);
        while (changed.get()) {
            changed.set(false);
            func = func.transform((bb, op) -> {
                if (op instanceof JavaOp.InvokeOp iop) {
                    MethodRef methodRef = iop.invokeDescriptor();
                    Method invokeOpCalledMethod;
                    try {
                        invokeOpCalledMethod = methodRef.resolveToMethod(l, iop.invokeKind());
                    } catch (ReflectiveOperationException _) {
                        throw new IllegalStateException("Could not resolve invokeOp to method");
                    }
                    if (invokeOpCalledMethod instanceof Method method) {
                        // only works if method isn't a buffer access (is code reflected)
                        if (Op.ofMethod(method).isPresent()) {
                            CoreOp.FuncOp inlineFunc = Op.ofMethod(method).get();
                            CoreOp.FuncOp ssa = SSA.transform(inlineFunc.transform(OpTransformer.LOWERING_TRANSFORMER));

                            Block.Builder exit = Inliner.inline(bb, ssa, bb.context().getValues(iop.operands()), (builder, v) -> {
                                if (v != null) bb.context().mapValue(iop.result(), v);
                            });
                            if (!exit.parameters().isEmpty()) {
                                bb.context().mapValue(iop.result(), exit.parameters().getFirst());
                            }
                            changed.set(true);
                            return exit.rebind(bb.context(), bb.transformer());
                        }
                    }
                }
                bb.op(op);
                return bb;
            });
        }
        return func;
    }

    public static void buildAccessMap(MethodHandles.Lookup l, CoreOp.FuncOp f) {
        // build blockParams so that we can map params to the root param later
        for (Body b : f.bodies()) {
            for (Block block : b.blocks()) {
                if (!block.parameters().isEmpty()) {
                    blockParams.put(block, block.parameters());
                }
            }
        }

        f.traverse(null, (map, op) -> {
            if (op instanceof CoreOp.BranchOp b) {
                List<Value> args = b.branch().arguments();
                for (int i = 0; i < args.size(); i++) {
                    Value key = blockParams.get(b.branch().targetBlock()).get(i);
                    Value val = args.get(i);

                    if (val instanceof Op.Result) {
                        // either find root param or it doesnt exist (is a constant for example)
                        Class<?> flopClass = getClass(l, val.type());
                        if (flopClass != null && (MappableIface.class.isAssignableFrom(flopClass))) {
                            val = getRootValue(l, ((Op.Result) val).op());
                            if (val instanceof Block.Parameter p) {
                                val = remappedVals.getOrDefault(val, val);
                            }
                        }
                    }

                    remappedVals.put(key, val);
                }
            } else if (op instanceof JavaOp.InvokeOp iop) {
                // all the buffer accesses happen here
                Class<?> iopClass = getClass(l, iop.invokeDescriptor().refType());
                // if the VarOp is initialized by an InvokeOp, don't access - otherwise do
                if (iopClass != null && MappableIface.class.isAssignableFrom(iopClass)) {
                    if (Buffer.class.isAssignableFrom(iopClass)) {
                        updateAccessType(getRootValue(l, iop), getAccessType(iop));
                        if (iop.result() != null && !(iop.resultType() instanceof PrimitiveType)
                                && MappableIface.class.isAssignableFrom(getClass(l, iop.resultType()))) {
                            remappedVals.put(iop.result(), getRootValue(l, iop));
                        }
                    } else {
                        Value val = iop.operands().getFirst();
                        while (!(val instanceof Block.Parameter param)) {
                            val = ((Op.Result) val).op().operands().getFirst();
                        }
                        updateAccessType(val, getAccessType(iop));
                    }
                }
            } else if (op instanceof CoreOp.VarOp vop) {
                try {
                    if (vop.resultType().valueType() instanceof ClassType classType
                            && Buffer.class.isAssignableFrom((Class<?>) classType.resolve(l))) {
                        remappedVals.put(vop.initOperand(), getRootValue(l, vop));
                    }
                } catch (ReflectiveOperationException e) {
                    throw new RuntimeException(e);
                }
            } else if (op instanceof JavaOp.FieldAccessOp.FieldLoadOp flop) {
                Class<?> flopClass = getClass(l, flop.fieldDescriptor().refType());
                if (flopClass != null && (KernelContext.class.isAssignableFrom(flopClass))) {
                    updateAccessType(getRootValue(l, flop), AccessType.RO);
                }
            }
            return map;
        });
    }

    public static Class<?> getClass(MethodHandles.Lookup l, TypeElement type) {
        if (type instanceof ClassType classType) {
            try {
                return (Class<?>) classType.resolve(l);
            } catch (ReflectiveOperationException e) {
                throw new RuntimeException(e);
            }
        }
        return null;
    }

    public static Value getRootValue(MethodHandles.Lookup l, Op op) {
        // if the VarOp is already the root VarOp, return
        if (op.operands().isEmpty()) {
            return op.result();
        }
        if (op.operands().getFirst() instanceof Block.Parameter param) {
            return param;
        }
        do {
            Op tempOp = ((Op.Result) op.operands().getFirst()).op();
            // or if the "root VarOp" is an invoke (not sure how to tell)
            // if (tempOp instanceof JavaOp.InvokeOp iop
            //        && ((TypeElement) iop.resultType()) instanceof ClassType classType
            //        && !hasOperandType(iop, classType)) return ((CoreOp.VarOp) op);
            op = tempOp;
        } while (!(op instanceof CoreOp.VarOp &&
                        // we stop when we find the root VarOp
                        (op.operands().getFirst() instanceof Block.Parameter)));
        return ((CoreOp.VarOp) op).initOperand();
    }

    public static boolean hasOperandType(Op op, ClassType classType) {
        for (Value v : op.operands()) {
            if (v instanceof Op.Result r
                    && ((TypeElement) r.op().resultType()) instanceof ClassType operandType
                    && operandType.equals(classType)) {
                return true;
            }
        }
        return false;
    }

    public static AccessType getAccessType(JavaOp.InvokeOp iop) {
        return iop.invokeDescriptor().type().returnType().equals(JavaType.VOID) ? AccessType.WO : AccessType.RO;
    }

    public static void updateAccessType(Value val, AccessType curAccess) {
        Value remappedVal = remappedVals.getOrDefault(val, val);
        AccessType storedAccess = accessMap.get(remappedVal);
        if (storedAccess == null) {
            accessMap.put(remappedVal, curAccess);
        } else if (curAccess != storedAccess && storedAccess != AccessType.RW) {
            accessMap.put(remappedVal, AccessType.RW);
        }
    }

    public static void printAccessMap() {
        System.out.println("access map output:");
        for (Value val : accessMap.keySet()) {
            if (val instanceof Block.Parameter param) {
                System.out.println("\t" + ((CoreOp.FuncOp) param.declaringBlock().parent().parent()).funcName() + " " + param.toString() + " idx " + param.index() + ": " + accessMap.get(val));
            } else {
                System.out.println("\t" + val.toString() + ": " + accessMap.get(val));
            }
        }
    }
}