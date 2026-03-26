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
import jdk.incubator.code.dialect.java.ArrayType;
import jdk.incubator.code.dialect.java.JavaOp;
import jdk.incubator.code.dialect.java.PrimitiveType;
import optkl.IfaceValue;
import optkl.OpHelper;
import optkl.ifacemapper.AccessType;
import jdk.incubator.code.Op;
import jdk.incubator.code.Value;
import jdk.incubator.code.Block;
import jdk.incubator.code.dialect.core.CoreOp;

import java.lang.invoke.MethodHandles;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;
import java.util.List;
import static optkl.OpHelper.Invoke.invoke;

public class BufferTagger {
    static Map<Value, AccessType> accessMap = new HashMap<>(); // mapping of parameters/buffers to access type
    static Map<Value, Value> rootValues = new HashMap<>(); // maps values to their "root" parameter/value

    // generates a list of AccessTypes matching the given FuncOp's parameter order
    public static List<AccessType> getAccessList(MethodHandles.Lookup lookup, CoreOp.FuncOp funcOp) {
        buildAccessMap(lookup, funcOp);
        List<AccessType> accessList = new ArrayList<>();
        for (Block.Parameter p : funcOp.body().entryBlock().parameters()) {
            if (accessMap.containsKey(p)) {
                accessList.add(accessMap.get(p)); // is an accessed buffer
            } else if (OpHelper.isAssignable(lookup, p.type(), IfaceValue.class)) {
                accessList.add(AccessType.NA); // is a buffer but not accessed
            } else {
                accessList.add(AccessType.NOT_BUFFER); // is not a buffer
            }
        }
        return accessList;
    }

    // creates the access map
    private static void buildAccessMap(MethodHandles.Lookup lookup, CoreOp.FuncOp funcOp) {
        funcOp.elements().forEach(op -> {
            switch (op) {
                case CoreOp.BranchOp b -> mapBranch(lookup, b.branch());
                case CoreOp.ConditionalBranchOp cb -> {
                    mapBranch(lookup, cb.trueBranch()); // handle true branch
                    mapBranch(lookup, cb.falseBranch()); // handle false branch
                }
                case JavaOp.InvokeOp invokeOp -> {
                    var ioh = invoke(lookup,invokeOp);
                    if (ioh.refIs(KernelContext.class)) break; // if this is not referencing a buffer, we break
                    if (ioh.returns(IfaceValue.class) || ioh.returnsArray()) { // if we receive a buffer from this invoke, we save its root value
                        for (Value operand : ioh.op().operands()) {
                            if (!(operand.type() instanceof PrimitiveType) && rootValues.containsKey(operand)) {
                                if (operand instanceof Block.Parameter) updateAccessType(operand, AccessType.RO);
                                else updateAccessType(getRootValue(operand.result().op()), AccessType.RO);
                            }
                        }
                        rootValues.put(invokeOp.result(), getRootValue(invokeOp));
                    } else { // if we actually operate on a buffer instead of storing an element in a variable
                        updateAccessType(rootValues.getOrDefault(invokeOp.result(), getRootValue(invokeOp)), ioh.returnsVoid() ? AccessType.WO : AccessType.RO); // update buffer access
                    }
                }
                case CoreOp.VarOp vop -> { // map the new VarOp to the "root" param
                    if (!OpHelper.isAssignable(lookup,  vop.resultType().valueType(), IfaceValue.class)) break;
                    rootValues.put(vop.initOperand(), getRootValue(vop));
                }
                case JavaOp.FieldAccessOp.FieldLoadOp flop -> {
                    if (!OpHelper.isAssignable(lookup,  flop.fieldReference().refType(), KernelContext.class)) break;
                    updateAccessType(getRootValue(flop), AccessType.RO); // handle kc access
                }
                case JavaOp.ArrayAccessOp.ArrayLoadOp alop -> {
                    if (alop.resultType() instanceof ArrayType) break;
                    updateAccessType(getRootValue(alop), AccessType.RO);
                }
                case JavaOp.ArrayAccessOp.ArrayStoreOp asop -> updateAccessType(getRootValue(asop), AccessType.WO);
                default -> {}
            }
        });
    }

    // maps the parameters of a block to the values passed to a branch
    private static void mapBranch(MethodHandles.Lookup lookup, Block.Reference blockReference) {
        List<Value> inputArgs = blockReference.arguments();
        List<Block.Parameter> targetArgs = blockReference.targetBlock().parameters();
        for (int i = 0; i < inputArgs.size(); i++) {
            Value target = targetArgs.get(i);
            Value input = inputArgs.get(i);
            if (!(input instanceof Op.Result result && OpHelper.isAssignable(lookup, input.type(), IfaceValue.class))) break;
            input = getRootValue(result.op());
            rootValues.put(target, rootValues.getOrDefault(input, input));
        }
    }

    // retrieves "root" value of an op, which is how we track accesses
    private static Value getRootValue(Op op) {
        // the op is a field load, an invoke, or something that reduces to one or the other
        Op rootOp = HATPhaseUtils.findOpInResultFromFirstOperandsOrNull(op, JavaOp.FieldAccessOp.FieldLoadOp.class, JavaOp.InvokeOp.class);
        switch (rootOp) {
            case JavaOp.FieldAccessOp.FieldLoadOp fieldOp -> {
                if (fieldOp.operands().isEmpty()) break; // e.g. handling kc.warpSize
                return fieldOp.operands().getFirst();
            }
            case JavaOp.InvokeOp invokeOp -> {
                while (invokeOp != null && !invokeOp.operands().isEmpty()) { // we look for either the parameter or initialization for the buffer
                    if (invokeOp.operands().getFirst() instanceof Block.Parameter p) return p; // return the parameter that is the global buffer
                    invokeOp = (JavaOp.InvokeOp) HATPhaseUtils.findOpInResultFromFirstOperandsOrNull(invokeOp.operands().getFirst().result().op(), JavaOp.InvokeOp.class);
                }
                if (invokeOp != null) return invokeOp.result();
            }
            case null, default -> {}
        }
        return null;
    }

    // updates the access map
    private static void updateAccessType(Value value, AccessType currentAccess) {
        Value remappedValue = rootValues.getOrDefault(value, value);
        AccessType storedAccess = accessMap.get(remappedValue);
        if (storedAccess == null) {
            accessMap.put(remappedValue, currentAccess);
        } else if (currentAccess != storedAccess && storedAccess != AccessType.RW) {
            accessMap.put(remappedValue, AccessType.RW);
        } // otherwise this is the same access type as what's already stored
    }

    public static void printAccessList(CoreOp.FuncOp inlinedEntryPoint, List<AccessType> accessList) {
        System.out.print("func " + inlinedEntryPoint.funcName() + " has parameters");
        for (AccessType at : accessList) {
            System.out.print(" " + at);
        }
        System.out.println();
    }
}