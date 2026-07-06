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

import hat.phases.HATPhaseUtils;
import jdk.incubator.code.dialect.java.JavaOp;
import optkl.IfaceValue;
import optkl.OpHelper;
import optkl.ifacemapper.AccessType;
import optkl.ifacemapper.Buffer;
import optkl.ifacemapper.MappableIface;
import jdk.incubator.code.Op;
import jdk.incubator.code.Value;
import jdk.incubator.code.Block;
import jdk.incubator.code.dialect.core.CoreOp;

import java.lang.invoke.MethodHandles;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import static optkl.OpHelper.Invoke;
import static optkl.OpHelper.Invoke.invoke;

public class BufferTagger {
    static HashMap<Value, AccessType> accessMap = new HashMap<>();
    static HashMap<Value, Value> remappedVals = new HashMap<>(); // maps values to their "root" parameter/value
    static HashMap<Block, List<Block.Parameter>> blockParams = new HashMap<>(); // holds block parameters for easy lookup

    // generates a list of AccessTypes matching the given FuncOp's parameter order
    public static ArrayList<AccessType> getAccessList(MethodHandles.Lookup lookup, CoreOp.FuncOp inlinedEntryPoint) {
        buildAccessMap(lookup, inlinedEntryPoint);
        ArrayList<AccessType> accessList = new ArrayList<>();
        for (Block.Parameter p : inlinedEntryPoint.body().entryBlock().parameters()) {
            if (accessMap.containsKey(p)) {
                accessList.add(accessMap.get(p)); // is an accessed buffer
            } else if (OpHelper.isAssignable(lookup, p.type(), MappableIface.class)) {
                // accessList.add(AccessType.NA); // is a buffer but not accessed
                // TODO: shouldn't be RO as default
                accessList.add(AccessType.RO);
            } else {
                accessList.add(AccessType.NOT_BUFFER); // is not a buffer
            }
        }
        return accessList;
    }
    private static boolean isReference(Invoke ioh) {
        return ioh.returns(IfaceValue.class)
                && ioh.opFromOnlyUseOrNull() instanceof JavaOp.InvokeOp nextInvoke
                && invoke(ioh.lookup(), nextInvoke) instanceof Invoke nextIoh
                && nextIoh.refIs(IfaceValue.class)
                && nextIoh.returnsVoid();
    }

    // creates the access map
    private static void buildAccessMap(MethodHandles.Lookup lookup, CoreOp.FuncOp funcOp) {
        // build blockParams so that we can map params to "root" params later
        funcOp.elements()
                .filter(Block.class::isInstance)
                .map(Block.class::cast)
                .forEach(block -> blockParams.put(block, block.parameters()));

        funcOp.elements().forEach(op -> {
            switch (op) {
                case CoreOp.BranchOp b -> mapBranch(lookup, b.branch());
                case CoreOp.ConditionalBranchOp cb -> {
                    mapBranch(lookup, cb.trueBranch()); // handle true branch
                    mapBranch(lookup, cb.falseBranch()); // handle false branch
                }
                case JavaOp.InvokeOp invokeOp -> {
                    var ioh =  invoke(lookup,invokeOp);
                    // we have to deal with  array views  too
                    // should .arrayview() calls be marked as reads?
                    if ( ioh.refIs(IfaceValue.class)) {
                        // updateAccessType(getRootValue(invokeOp), ioh.returnsVoid()? AccessType.WO : AccessType.RO); // update buffer access
                        // if the invokeOp retrieves an element that is only written to, don't update the access type
                        // (i.e. the only use is an invoke, the invoke is of MappableIface/HAType class, and is a write)
                        if (!isReference(ioh)) { //     value retrieved and not just referenced?
                            updateAccessType(getRootValue(invokeOp), ioh.returnsVoid()? AccessType.WO : AccessType.RO); // update buffer access
                        }
                        if (ioh.refIs(IfaceValue.class) && (ioh.returns(IfaceValue.class) || ioh.returnsArray())) {
                            // if we access a struct/union from a buffer, we map the struct/union to the buffer root
                            remappedVals.put(invokeOp.result(), getRootValue(invokeOp));
                        }
                    }
                }
                case CoreOp.VarOp vop -> { // map the new VarOp to the "root" param
                    if (OpHelper.isAssignable(lookup,  vop.resultType().valueType(), Buffer.class)) {
                        remappedVals.put(vop.initOperand(), getRootValue(vop));
                    }else{
                        // or else maybe CoreOp.VarOp vop when ??? ->
                    }
                }
                case JavaOp.FieldAccessOp.FieldLoadOp flop -> {
                    if (OpHelper.isAssignable(lookup,  flop.fieldReference().refType(), KernelContext.class)) {
                        updateAccessType(getRootValue(flop), AccessType.RO); // handle kc access
                    }else{
                        // or else
                    }
                }
                case JavaOp.ArrayAccessOp.ArrayLoadOp alop -> updateAccessType(getRootValue(alop), AccessType.RO);
                case JavaOp.ArrayAccessOp.ArrayStoreOp asop -> updateAccessType(getRootValue(asop), AccessType.WO);
                default -> {}
            }
        });
    }

    // maps the parameters of a block to the values passed to a branch
    private static void mapBranch(MethodHandles.Lookup lookup, Block.Reference blockReference) {
        List<Value> args = blockReference.arguments();
        for (int i = 0; i < args.size(); i++) {
            Value key = blockParams.get(blockReference.targetBlock()).get(i);
            Value value = args.get(i);
            if (value instanceof Op.Result result) {
                // either find root param or it doesn't exist (is a constant for example)
                if (OpHelper.isAssignable(lookup, value.type(), MappableIface.class)) {
                    value = getRootValue(result.op());
                    if (value instanceof Block.Parameter) {
                        value = remappedVals.getOrDefault(value, value);
                    }
                }else{
                    // or else
                }
            }else{
               // or else?
            }
            remappedVals.put(key, value);
        }
    }

    // retrieves "root" value of an op, which is how we track accesses
    // we will map the return value of this method to the accessType
    private static Value getRootValue(Op op) {
        // the op is a field load, an invoke, or something that reduces to one or the other
        // first, check if we can retrieve a fieldloadop from the given op
        Op fieldOp = HATPhaseUtils.findOpInResultFromFirstOperandsOrNull(op, JavaOp.FieldAccessOp.FieldLoadOp.class);
        if (fieldOp != null) {
            return fieldOp.operands().isEmpty() ? fieldOp.result() : fieldOp.operands().getFirst();
        }

        // we then check if there's an invokeop that has no operands (meaning a shared or private buffer that was created)
        // or if there's an invokeop with a parameter as its first operation (this is a global buffer)
        Op invokeOp = HATPhaseUtils.findOpInResultFromFirstOperandsOrNull(op, JavaOp.InvokeOp.class);
        while (invokeOp != null && !invokeOp.operands().isEmpty()) {
            if (invokeOp.operands().getFirst() instanceof Block.Parameter p) return p; // return the parameter that is the global buffer
            invokeOp = HATPhaseUtils.findOpInResultFromFirstOperandsOrNull(invokeOp.operands().getFirst().asResult().op(), JavaOp.InvokeOp.class);
        }
        return (invokeOp == null) ? null : invokeOp.result(); // return the shared/private buffer invokeop that creates the buffer
    }

    // updates accessMap
    private static void updateAccessType(Value value, AccessType currentAccess) {
        Value remappedValue = remappedVals.getOrDefault(value, value);
        AccessType storedAccess = accessMap.get(remappedValue);
        if (storedAccess == null) {
            accessMap.put(remappedValue, currentAccess);
        } else if (currentAccess != storedAccess && storedAccess != AccessType.RW) {
            accessMap.put(remappedValue, AccessType.RW);
        } else {
            // this is the same access type as what's already stored
        }
    }

    private BufferTagger() {}
}