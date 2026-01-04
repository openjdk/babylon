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

import jdk.incubator.code.analysis.SSA;
import optkl.Invoke;
import optkl.Trxfmr;
import optkl.ifacemapper.AccessType;
import optkl.ifacemapper.Buffer;
import optkl.ifacemapper.MappableIface;
import jdk.incubator.code.*;
import jdk.incubator.code.analysis.Inliner;
import jdk.incubator.code.dialect.core.CoreOp;
import jdk.incubator.code.dialect.java.*;
import optkl.util.CallSite;
import optkl.util.StreamMutable;

import java.lang.invoke.MethodHandles;
import java.lang.reflect.Method;
import java.util.*;

import static optkl.Invoke.invokeOpHelper;
import static optkl.OpTkl.isAssignable;

public class BufferTagger {
    static HashMap<Value, AccessType> accessMap = new HashMap<>();
    static HashMap<Value, Value> remappedVals = new HashMap<>(); // maps values to their "root" parameter/value
    static HashMap<Block, List<Block.Parameter>> blockParams = new HashMap<>(); // holds block parameters for easy lookup

    // generates a list of AccessTypes matching the given FuncOp's parameter order
    public static ArrayList<AccessType> getAccessList(MethodHandles.Lookup lookup, CoreOp.FuncOp funcOp) {
        CoreOp.FuncOp inlinedFunc = inlineLoop(lookup, funcOp);
        buildAccessMap(lookup, inlinedFunc);
        ArrayList<AccessType> accessList = new ArrayList<>();
        for (Block.Parameter p : inlinedFunc.body().entryBlock().parameters()) {
            if (accessMap.containsKey(p)) {
                accessList.add(accessMap.get(p)); // is an accessed buffer
            } else if (isAssignable(lookup, p.type(), MappableIface.class)) {
                accessList.add(AccessType.NA); // is a buffer but not accessed
            } else {
                accessList.add(AccessType.NOT_BUFFER); // is not a buffer
            }
        }
        return accessList;
    }

    // inlines functions found in FuncOp f until no more inline-able functions are present
    public static CoreOp.FuncOp inlineLoop(MethodHandles.Lookup lookup, CoreOp.FuncOp funcOp) {
        CoreOp.FuncOp ssaFunc =  SSA.transform( funcOp.transform(CodeTransformer.LOWERING_TRANSFORMER)) ;
        var changed  = StreamMutable.of(true);
        while (changed.get()) { // loop until no more inline-able functions
            changed.set(false);
            ssaFunc = ssaFunc.transform( (blockbuilder, op) -> {
                if (invokeOpHelper(lookup, op) instanceof Invoke invoke                         // always but pattern friendly
                        && invoke.resolvedMethodOrNull() instanceof Method method
                        && Op.ofMethod(method) instanceof Optional<CoreOp.FuncOp> optionalFuncOp // always but pattern friendly
                        && optionalFuncOp.isPresent()
                        && optionalFuncOp.get() instanceof CoreOp.FuncOp inline                  // always we just want var in scope
                ){
                    var ssaInline =SSA.transform(inline.transform(CodeTransformer.LOWERING_TRANSFORMER));
                    var exitBlockBuilder = Inliner.inline(
                            blockbuilder, ssaInline,
                            blockbuilder.context().getValues(invoke.op().operands()), (_, _value) -> {
                                // intellij doesnt like value as var name so we use _value
                            if (_value == null) {
                               //   What is special about TestArrayView.Compute.lifePerIdx? it reaches here
                                // I think its because it is void ? no return type.
                                    //   throw new IllegalStateException("inliner returned  null processing "+method);
                            }else{
                                blockbuilder.context().mapValue(invoke.op().result(), _value);
                            }
                    });
                    if (!exitBlockBuilder.parameters().isEmpty()) {
                        blockbuilder.context().mapValue(invoke.op().result(), exitBlockBuilder.parameters().getFirst());
                    }
                    changed.set(true);
                    return exitBlockBuilder.rebind(blockbuilder.context(), blockbuilder.transformer());
                }
                blockbuilder.op(op);
               return blockbuilder;
            });
        }
        return ssaFunc;
    }

    // creates the access map
    public static void buildAccessMap(MethodHandles.Lookup lookup, CoreOp.FuncOp funcOp) {
        // build blockParams so that we can map params to "root" params later
        funcOp.elements()
                .filter(elem -> elem instanceof Block)
                .map(elem->(Block)elem)
                .forEach(block -> blockParams.put(block, block.parameters()));

        funcOp.elements().forEach(op -> {
            switch (op) {
                case CoreOp.BranchOp b -> mapBranch(lookup, b.branch());
                case CoreOp.ConditionalBranchOp cb -> {
                    mapBranch(lookup, cb.trueBranch()); // handle true branch
                    mapBranch(lookup, cb.falseBranch()); // handle false branch
                }
                case JavaOp.InvokeOp invokeOp -> {
                    var ioh =  invokeOpHelper(lookup,invokeOp);
                    // we have to deal with  array views  too
                    if ( ioh.refIs(MappableIface.class)) {
                        updateAccessType(getRootValue(invokeOp), ioh.returnsVoid()? AccessType.WO : AccessType.RO); // update buffer access
                        if (ioh.refIs(Buffer.class) && (ioh.returns(MappableIface.class) || ioh.returnsArray())) {
                            // if we access a struct/union from a buffer, we map the struct/union to the buffer root
                            remappedVals.put(invokeOp.result(), getRootValue(invokeOp));
                        }
                    }
                }
                case CoreOp.VarOp vop -> { // map the new VarOp to the "root" param
                    if (isAssignable(lookup,  vop.resultType().valueType(), Buffer.class)) {
                        remappedVals.put(vop.initOperand(), getRootValue(vop));
                    }else{
                        // or else maybe CoreOp.VarOp vop when ??? ->
                    }
                }
                case JavaOp.FieldAccessOp.FieldLoadOp flop -> {
                    if (isAssignable(lookup,  flop.fieldDescriptor().refType(), KernelContext.class)) {
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
    public static void mapBranch(MethodHandles.Lookup lookup, Block.Reference blockReference) {
        List<Value> args = blockReference.arguments();
        for (int i = 0; i < args.size(); i++) {
            Value key = blockParams.get(blockReference.targetBlock()).get(i);
            Value value = args.get(i);
            if (value instanceof Op.Result result) {
                // either find root param or it doesn't exist (is a constant for example)
                if (isAssignable(lookup, value.type(), MappableIface.class)) {
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

    // retrieves "root" value of an op, the origin of the parameter (or value) used by the op
    public static Value getRootValue(Op op) {
        if (op.operands().isEmpty()) {
            return op.result();
        } else if (op.operands().getFirst() instanceof Block.Parameter param) {
            return param;
        }

        while (op.operands().getFirst() instanceof Op.Result result) { // Only first?
            op = result.op(); // we are changing our  par here I assume intended
            if (op.operands().isEmpty()) { // if the "root op" is an invoke
                return op.result();
            }else{
                // or else
            }
        }
        return op.operands().getFirst();
    }

    // updates accessMap
    public static void updateAccessType(Value value, AccessType currentAccess) {
        Value remappedValue = remappedVals.getOrDefault(value, value);
        AccessType storedAccess = accessMap.get(remappedValue);
        if (storedAccess == null) {
            accessMap.put(remappedValue, currentAccess);
        } else if (currentAccess != storedAccess && storedAccess != AccessType.RW) {
            accessMap.put(remappedValue, AccessType.RW);
        } else {
            // or else
        }
    }
}