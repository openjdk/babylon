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
import jdk.incubator.code.CodeItem;
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
import java.util.stream.IntStream;
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
                case JavaOp.InvokeOp $ when invoke(lookup, $) instanceof OpHelper.Invoke ioh && !ioh.refIs(KernelContext.class) -> {
                    if (ioh.returns(IfaceValue.class) || ioh.returnsArray()) { // if we receive a buffer from this invoke, we save its root value
                        ioh.op().operands().stream()
                                .filter(operand -> !(operand.type() instanceof PrimitiveType) && rootValues.containsKey(operand))
                                .forEach(operand -> {
                                    if (operand instanceof Block.Parameter) {
                                        updateAccessType(operand, AccessType.RO);
                                    } else {
                                        updateAccessType(operand.result().op(), AccessType.RO);
                                    }
                                });
                        rootValues.put(ioh.returnResult(), getRootValue(ioh.op()));
                    } else { // if we actually operate on a buffer instead of storing an element in a variable
                        updateAccessType(ioh.op(), ioh.returnsVoid() ? AccessType.WO : AccessType.RO); // update buffer access
                    }
                }
                case CoreOp.VarOp vop when OpHelper.isAssignable(lookup, vop.varValueType(), IfaceValue.class) ->
                        rootValues.put(vop.initOperand(), getRootValue(vop)); // map the new VarOp to the "root" param
                case JavaOp.FieldAccessOp.FieldLoadOp flop when OpHelper.isAssignable(lookup, flop.fieldReference().refType(), KernelContext.class) ->
                        updateAccessType(flop, AccessType.RO); // handle kc access
                case JavaOp.ArrayAccessOp.ArrayLoadOp alop when !(alop.resultType() instanceof ArrayType) ->
                        updateAccessType(alop, AccessType.RO);
                case JavaOp.ArrayAccessOp.ArrayStoreOp asop ->
                        updateAccessType(asop, AccessType.WO);
                default -> {}
            }
        });
    }

    // maps the parameters of a block to the values passed to a branch
    private static void mapBranch(MethodHandles.Lookup lookup, Block.Reference blockReference) {
        List<Value> inputArgs = blockReference.arguments();
        List<Block.Parameter> targetArgs = blockReference.targetBlock().parameters();
        IntStream.range(0, inputArgs.size()).filter(i ->
                inputArgs.get(i) instanceof Op.Result && OpHelper.isAssignable(lookup, inputArgs.get(i).type(), IfaceValue.class))
                .forEach(i -> {
                    Value input = inputArgs.get(i);
                    input = getRootValue(input.result().op());
                    rootValues.put(targetArgs.get(i), rootValues.getOrDefault(input, input));
                });
    }

    // retrieves "root" value of an op, which is how we track accesses
    private static Value getRootValue(Op op) {
        // the op is a field load, an invoke, or something that reduces to one or the other
        Op rootOp = HATPhaseUtils.findOpInResultFromFirstOperandsOrNull(op, JavaOp.FieldAccessOp.FieldLoadOp.class, JavaOp.InvokeOp.class);
        switch (rootOp) {
            case JavaOp.FieldAccessOp.FieldLoadOp fieldOp when !fieldOp.operands().isEmpty() -> {
                return fieldOp.operands().getFirst();
            }
            case JavaOp.InvokeOp invokeOp -> {
                while (invokeOp != null && !invokeOp.operands().isEmpty()) { // we look for either the parameter or initialization for the buffer
                    if (invokeOp.operands().getFirst() instanceof Block.Parameter p) {
                        return p; // return the parameter that is the global buffer
                    }
                    invokeOp = (JavaOp.InvokeOp) HATPhaseUtils.findOpInResultFromFirstOperandsOrNull(invokeOp.operands().getFirst().result().op(), JavaOp.InvokeOp.class);
                }
                if (invokeOp != null) {
                    return invokeOp.result();
                }
            }
            case null, default -> {}
        }
        return null;
    }

    // retrieves root value of op before updating the access map
    private static void updateAccessType(Op op, AccessType currentAccess) {
        updateAccessType(getRootValue(op), currentAccess);
    }

    // updates the access map
    private static void updateAccessType(Value value, AccessType currentAccess) {
        AccessType storedAccess = accessMap.get(value);
        if (storedAccess == null) {
            accessMap.put(value, currentAccess);
        } else if (currentAccess != storedAccess && storedAccess != AccessType.RW) {
            accessMap.put(value, AccessType.RW);
        } // otherwise this is the same access type as what's already stored
    }

    public static void printAccessList(CoreOp.FuncOp funcOp, List<AccessType> accessList) {
        StringBuilder output = new StringBuilder();
        output.append("func ").append(funcOp.funcName()).append(" has parameters");
        for (AccessType at : accessList) {
            output.append(" ").append(at);
        }
        System.out.println(output);
    }
}