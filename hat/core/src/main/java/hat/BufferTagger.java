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
import hat.optools.OpTk;
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

    public static String convertAccessType(int i) {
        switch (i) {
            case 0 -> {return "NOT_BUFFER";}
            case 1 -> {return "NA";}
            case 2 -> {return "RO";}
            case 4 -> {return "WO";}
            case 6 -> {return "RW";}
            default -> {return "";}
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
            } else if (OpTk.isAssignable(l, (JavaType) p.type(), MappableIface.class)) {
                accessList.add(AccessType.NA); // is a buffer but not accessed
            } else {
                accessList.add(AccessType.NOT_BUFFER); // is not a buffer
            }
        }
        return accessList;
    }

    // inlines functions found in FuncOp f until no more inline-able functions are present
    public static CoreOp.FuncOp inlineLoop(MethodHandles.Lookup l, CoreOp.FuncOp f) {

        var here = OpTk.CallSite.of(BufferTagger.class, "inlineLoop");
        CoreOp.FuncOp ssaFunc = OpTk.SSATransformLower(here, f); // do we need this nesting?
        AtomicBoolean changed = new AtomicBoolean(true);
        while (changed.get()) { // loop until no more inline-able functions
            changed.set(false);
            ssaFunc = OpTk.transform(OpTk.CallSite.of(BufferTagger.class, "inlineLoop"),ssaFunc,(bb, op) -> {
                if (op instanceof JavaOp.InvokeOp iop) {
                    MethodRef methodRef = iop.invokeDescriptor();
                    Method invokeOpCalledMethod;
                    try {
                        invokeOpCalledMethod = methodRef.resolveToMethod(l);
                    } catch (ReflectiveOperationException _) {
                        throw new IllegalStateException("Could not resolve invokeOp to method");
                    }
                    if (invokeOpCalledMethod instanceof Method method) { // if method isn't a buffer access (is code reflected)
                        if (Op.ofMethod(method).isPresent()) {
                            CoreOp.FuncOp inline = Op.ofMethod(method).get(); // method to be inlined
                            CoreOp.FuncOp ssaInline = OpTk.SSATransformLower(here, inline);
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
        var here = OpTk.CallSite.of(BufferTagger.class, "buildAccessMap");
        OpTk.elements(here, f).filter(elem -> elem instanceof Block)
                .forEach(b -> blockParams.put((Block) b, ((Block) b).parameters()));

        f.elements().forEach(op -> {
            switch (op) {
                case CoreOp.BranchOp b -> {
                    mapBranch(l, b.branch());
                }
                case CoreOp.ConditionalBranchOp cb -> {
                    mapBranch(l, cb.trueBranch()); // handle true branch
                    mapBranch(l, cb.falseBranch()); // handle false branch
                }
                case JavaOp.InvokeOp iop -> { // (almost) all the buffer accesses happen here
                    // actually now that we have arrayview we'll need to map the corresponding arrays too
                    if (OpTk.isAssignable(l, (JavaType) iop.invokeDescriptor().refType(), MappableIface.class)) {
                        updateAccessType(getRootValue(iop), getAccessType(iop)); // update buffer access
                        if (OpTk.isAssignable(l, (JavaType) iop.invokeDescriptor().refType(), Buffer.class)
                                && iop.result() != null && !(iop.resultType() instanceof PrimitiveType)
                                && (OpTk.isAssignable(l, (JavaType) iop.resultType(), MappableIface.class)
                                    || iop.resultType() instanceof ArrayType)) {
                            // if we access a struct/union from a buffer, we map the struct/union to the buffer root
                            remappedVals.put(iop.result(), getRootValue(iop));
                        }
                    }
                }
                case CoreOp.VarOp vop -> { // map the new VarOp to the "root" param
                    if (OpTk.isAssignable(l, (JavaType) vop.resultType().valueType(), Buffer.class)) {
                        remappedVals.put(vop.initOperand(), getRootValue(vop));
                    }
                }
                case JavaOp.FieldAccessOp.FieldLoadOp flop -> {
                    if (OpTk.isAssignable(l, (JavaType) flop.fieldDescriptor().refType(), KernelContext.class)) {
                        updateAccessType(getRootValue(flop), AccessType.RO); // handle kc access
                    }
                }
                case JavaOp.ArrayAccessOp.ArrayLoadOp alop -> {
                    updateAccessType(getRootValue(alop), AccessType.RO);
                }
                case JavaOp.ArrayAccessOp.ArrayStoreOp asop -> {
                    updateAccessType(getRootValue(asop), AccessType.WO);
                }
                default -> {}
            }
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
                if (OpTk.isAssignable(l, (JavaType) val.type(), MappableIface.class)) {
                    val = getRootValue(((Op.Result) val).op());
                    if (val instanceof Block.Parameter) {
                        val = remappedVals.getOrDefault(val, val);
                    }
                }
            }
            remappedVals.put(key, val);
        }
    }

    // retrieves "root" value of an op, the origin of the parameter (or value) used by the op
    public static Value getRootValue(Op op) {
        if (op.operands().isEmpty()) {
            return op.result();
        } else if (op.operands().getFirst() instanceof Block.Parameter param) {
            return param;
        }
        while (op.operands().getFirst() instanceof Op.Result r) {
            op = r.op();
            if (op.operands().isEmpty()) { // if the "root op" is an invoke
                return op.result();
            }
        }
        return op.operands().getFirst();
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
}