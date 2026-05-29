/*
 * Copyright (c) 2025, Oracle and/or its affiliates. All rights reserved.
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
package hat.phases;

import hat.device.NonMappableIface;
import jdk.incubator.code.CodeElement;
import jdk.incubator.code.Op;
import jdk.incubator.code.dialect.core.CoreOp;
import jdk.incubator.code.dialect.java.JavaOp;
import optkl.IfaceValue;
import optkl.OpHelper;
import optkl.Trxfmr;
import optkl.VarTable;

import java.lang.invoke.MethodHandles;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Set;

import static optkl.OpHelper.Invoke;
import static optkl.OpHelper.Invoke.invoke;

public final class HATMemoryPhase implements HATPhase {

    String functionName;

    boolean isIntrinsicForDeviceMemoryType(Invoke invoke, String intrinsicName) {
        return invoke.refIs(IfaceValue.class) && invoke.named(intrinsicName);
    }

    private CoreOp.FuncOp analyzeOnChipDataStructures(MethodHandles.Lookup lookup, CoreOp.FuncOp funcOp, VarTable varTable, String intrinsicName, VarTable.HATOpAttribute attribute) {
        functionName = funcOp.funcName();
        Set<CodeElement<?, ?>> nodesInvolved = new LinkedHashSet<>();
        OpHelper.Variable.stream(lookup, funcOp)
                .forEach(variable -> variable.op().operands().stream()
                        .filter(operand -> operand instanceof Op.Result result
                                && invoke(lookup, result.op()) instanceof Invoke invoke
                                && isIntrinsicForDeviceMemoryType(invoke, intrinsicName))
                        .map(r -> (JavaOp.InvokeOp) (((Op.Result) r).op()))
                        .findFirst().ifPresent(remove -> nodesInvolved.add(variable.op()))
                );

        return Trxfmr.of(lookup, funcOp).transform(nodesInvolved::contains, (blockBuilder, op) -> {
            if (OpHelper.Named.Variable.var(lookup, op) instanceof OpHelper.Named.Variable variable) {
                Op.Result op1 = blockBuilder.add(variable.op());
                varTable.addIfNeededOrThrow(functionName, op1.op(), attribute);
            } else {
                blockBuilder.add(op);
            }
            return blockBuilder;
        }, varTable).funcOp();
    }

    private CoreOp.FuncOp initOnChipDataStructures(MethodHandles.Lookup lookup, CoreOp.FuncOp funcOp, VarTable varTable, String intrinsincName, VarTable.HATOpAttribute attribute) {
        this.functionName = funcOp.funcName();
        Set<CodeElement<?, ?>> nodesInvolved = new HashSet<>();
        Invoke.stream(lookup, funcOp)
                .filter(invoke -> invoke.refIs(NonMappableIface.class) && invoke.returnsClassType() && !invoke.nameMatchesRegex(OpHelper.RESERVED_METHODS_MEMORY_REGIONS))
                .forEach(invoke -> invoke.op().result().uses().stream()
                        .filter(use -> use.op() instanceof CoreOp.VarOp)
                        .map(use -> (CoreOp.VarOp) use.op())
                        .forEach(nodesInvolved::add)
                );

        return Trxfmr.of(lookup, funcOp).transform(nodesInvolved::contains, (blockBuilder, op) -> {
            if (op instanceof CoreOp.VarOp varOp) {
                Op.Result opResult = blockBuilder.add(varOp);
                varTable.addIfNeededOrThrow(functionName, opResult.op(), VarTable.HATOpAttribute.INIT_SHARED);
            }
            return blockBuilder;
        }, varTable).funcOp();
    }

    @FunctionalInterface
    public interface MemoryRegionTransformer {
        CoreOp.FuncOp apply(MethodHandles.Lookup lookup, CoreOp.FuncOp funcOp, VarTable varTable, String intrinsicName, VarTable.HATOpAttribute attribute);
    }

    private record MemoryRegion(String intrinsicName, VarTable.HATOpAttribute attribute, MemoryRegionTransformer f){
    }

    private final List<MemoryRegion> containers;

    public HATMemoryPhase() {
        this.containers = new ArrayList<>();
        containers.add(new MemoryRegion("createPrivate", VarTable.HATOpAttribute.PRIVATE, this::analyzeOnChipDataStructures));
        containers.add(new MemoryRegion("createLocal", VarTable.HATOpAttribute.SHARED, this::analyzeOnChipDataStructures));
        containers.add(new MemoryRegion("createLocal", VarTable.HATOpAttribute.INIT_SHARED, this::initOnChipDataStructures));
    }

    @Override
    public CoreOp.FuncOp transform(MethodHandles.Lookup lookup, CoreOp.FuncOp funcOp, VarTable varTable) {
        for (MemoryRegion container : containers) {
            funcOp = container.f().apply(lookup, funcOp, varTable, container.intrinsicName, container.attribute);
        }
        return funcOp;
    }
}
