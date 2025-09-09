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
    static HashMap<Value, Value> remappedVals = new HashMap<>(); // maps values to their "root" parameter/value
    static HashMap<Block, List<Block.Parameter>> blockParams = new HashMap<>(); // holds block parameters for easy lookup

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

    // generates a list of AccessTypes matching the given FuncOp's parameter order
    public static ArrayList<AccessType> getAccessList(MethodHandles.Lookup l, CoreOp.FuncOp f) {
        CoreOp.FuncOp inlinedFunc = inlineLoop(l, f);
        buildAccessMap(l, inlinedFunc);
        ArrayList<AccessType> accessList = new ArrayList<>();
        for (Block.Parameter p : inlinedFunc.body().entryBlock().parameters()) {
            if (accessMap.containsKey(p)) {
                accessList.add(accessMap.get(p)); // is an accessed buffer
            } else if (getClass(l, p.type()) instanceof Class<?> c && MappableIface.class.isAssignableFrom(c)) {
                accessList.add(AccessType.NA); // is a buffer but not accessed
            } else {
                accessList.add(AccessType.NOT_BUFFER); // is not a buffer
            }
        }
        return accessList;
    }

    // inlines functions found in FuncOp f until no more inline-able functions are present
    public static CoreOp.FuncOp inlineLoop(MethodHandles.Lookup l, CoreOp.FuncOp f) {
        CoreOp.FuncOp ssaFunc = SSA.transform(f.transform(OpTransformer.LOWERING_TRANSFORMER));
        AtomicBoolean changed = new AtomicBoolean(true);
        while (changed.get()) { // loop until no more inline-able functions
            changed.set(false);
            ssaFunc = ssaFunc.transform((bb, op) -> {
                if (op instanceof JavaOp.InvokeOp iop) {
                    MethodRef methodRef = iop.invokeDescriptor();
                    Method invokeOpCalledMethod;
                    try {
                        invokeOpCalledMethod = methodRef.resolveToMethod(l, iop.invokeKind());
                    } catch (ReflectiveOperationException _) {
                        throw new IllegalStateException("Could not resolve invokeOp to method");
                    }
                    if (invokeOpCalledMethod instanceof Method method) { // if method isn't a buffer access (is code reflected)
                        if (Op.ofMethod(method).isPresent()) {
                            CoreOp.FuncOp inline = Op.ofMethod(method).get(); // method to be inlined
                            CoreOp.FuncOp ssaInline = SSA.transform(inline.transform(OpTransformer.LOWERING_TRANSFORMER));

                            Block.Builder exit = Inliner.inline(bb, ssaInline, bb.context().getValues(iop.operands()), (_, v) -> {
                                if (v != null) bb.context().mapValue(iop.result(), v);
                            });

                            if (!exit.parameters().isEmpty()) {
                                bb.context().mapValue(iop.result(), exit.parameters().getFirst());
                            }
                            changed.set(true);
                            return exit.rebind(bb.context(), bb.transformer()); // return exit in same context as block
                        }
                    }
                }
                bb.op(op);
                return bb;
            });
        }
        return ssaFunc;
    }

    // creates the access map
    public static void buildAccessMap(MethodHandles.Lookup l, CoreOp.FuncOp f) {
        // build blockParams so that we can map params to "root" params later
        for (Body b : f.bodies()) {
            for (Block block : b.blocks()) {
                if (!block.parameters().isEmpty()) {
                    blockParams.put(block, block.parameters());
                }
            }
        }

        f.traverse(null, (map, op) -> {
            if (op instanceof CoreOp.BranchOp b) {
                mapBranch(l, b.branch());
            } else if (op instanceof  CoreOp.ConditionalBranchOp cb) {
                mapBranch(l, cb.trueBranch()); // handle true branch
                mapBranch(l, cb.falseBranch()); // handle false branch
            } else if (op instanceof JavaOp.InvokeOp iop) { // (almost) all the buffer accesses happen here
                if (isAssignable(l, iop.invokeDescriptor().refType(), MappableIface.class)) {
                    updateAccessType(getRootValue(iop), getAccessType(iop)); // update buffer access
                    if (isAssignable(l, iop.invokeDescriptor().refType(), Buffer.class)
                            && iop.result() != null && !(iop.resultType() instanceof PrimitiveType)
                            && isAssignable(l, iop.resultType(), MappableIface.class)) {
                        // if we access a struct/union from a buffer, we map the struct/union to the buffer root
                        remappedVals.put(iop.result(), getRootValue(iop));
                    }
                }
            } else if (op instanceof CoreOp.VarOp vop) { // map the new VarOp to the "root" param
                if (isAssignable(l, vop.resultType().valueType(), Buffer.class)) {
                    remappedVals.put(vop.initOperand(), getRootValue(vop));
                }
            } else if (op instanceof JavaOp.FieldAccessOp.FieldLoadOp flop) {
                if (isAssignable(l, flop.fieldDescriptor().refType(), KernelContext.class)) {
                    updateAccessType(getRootValue(flop), AccessType.RO); // handle kc access
                }
            }
            return map;
        });
    }

    // maps the parameters of a block to the values passed to a branch
    public static void mapBranch(MethodHandles.Lookup l, Block.Reference b) {
        List<Value> args = b.arguments();
        for (int i = 0; i < args.size(); i++) {
            Value key = blockParams.get(b.targetBlock()).get(i);
            Value val = args.get(i);

            if (val instanceof Op.Result) {
                // either find root param or it doesnt exist (is a constant for example)
                if (isAssignable(l, val.type(), MappableIface.class)) {
                    val = getRootValue(((Op.Result) val).op());
                    if (val instanceof Block.Parameter) {
                        val = remappedVals.getOrDefault(val, val);
                    }
                }
            }
            remappedVals.put(key, val);
        }
    }

    // checks if a TypeElement is assignable to a certain class
    public static boolean isAssignable(MethodHandles.Lookup l, TypeElement type, Class<?> clazz) {
        Class<?> fopClass = getClass(l, type);
        return (fopClass != null && (clazz.isAssignableFrom(fopClass)));
    }

    // retrieves the class of a TypeElement
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

    // retrieves "root" value of an op, the origin of the parameter (or value) used by the op
    public static Value getRootValue(Op op) {
        if (op.operands().isEmpty()) {
            return op.result();
        } else if (op.operands().getFirst() instanceof Block.Parameter param) {
            return param;
        }
        Value val = op.operands().getFirst();
        while (!(val instanceof Block.Parameter)) {
            Op root = ((Op.Result) val).op();
            if (root.operands().isEmpty()) { // if the "root op" is an invoke
                return root.result();
            }
            val = root.operands().getFirst();
        }
        return val;
    }

    // retrieves accessType based on return value of InvokeOp
    public static AccessType getAccessType(JavaOp.InvokeOp iop) {
        return iop.invokeDescriptor().type().returnType().equals(JavaType.VOID) ? AccessType.WO : AccessType.RO;
    }

    // updates accessMap
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
                System.out.println("\t" + ((CoreOp.FuncOp) param.declaringBlock().parent().parent()).funcName()
                        + " param w/ idx " + param.index() + ": " + accessMap.get(val));
            } else {
                System.out.println("\t" + val.toString() + ": " + accessMap.get(val));
            }
        }
    }
}