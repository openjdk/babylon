/*
 * Copyright (c) 2026, Oracle and/or its affiliates. All rights reserved.
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

import hat.types.Tensor;
import jdk.incubator.code.Block;
import jdk.incubator.code.Op;
import jdk.incubator.code.Value;
import jdk.incubator.code.dialect.core.CoreOp;
import jdk.incubator.code.dialect.java.JavaOp;
import optkl.OpHelper;
import optkl.Trxfmr;

import java.lang.invoke.MethodHandles;
import java.util.HashSet;
import java.util.List;
import java.util.Objects;
import java.util.Set;
import java.util.function.BiConsumer;

import static hat.dialect.HATTensorOp.TensorCreateOp;
import static hat.dialect.HATTensorOp.TensorFillOp;
import static hat.dialect.HATTensorOp.TensorLoadOp;
import static hat.dialect.HATTensorOp.TensorMMAOp;
import static hat.dialect.HATTensorOp.TensorStoreLoadOp;
import static hat.dialect.HATTensorOp.TensorStoreOp;
import static hat.dialect.HATTensorOp.TensorVarLoadOp;
import static hat.dialect.HATTensorOp.TensorVarOp;

public record HATTensorsPhase() implements HATPhase {

    private interface TensorTransformer {

        void transform(Block.Builder blockBuilder, Op op);

        default void replaceOp(Block.Builder blockBuilder, Op oldOp, Op newOp) {
            newOp.setLocation(oldOp.location());
            Op.Result newOpResult = blockBuilder.op(newOp);
            blockBuilder.context().mapValue(oldOp.result(), newOpResult);
        }

    }

    private static class TensorView implements TensorTransformer {

        @Override
        public void transform(Block.Builder blockBuilder, Op op) {
            List<Value> operands = blockBuilder.context().getValues(op.operands());
            switch (op) {
                case CoreOp.VarOp varOp -> replaceOp(blockBuilder, varOp, new TensorVarOp(varOp.varName(), varOp.resultType(), operands));
                case JavaOp.InvokeOp invokeOp -> replaceOp(blockBuilder, invokeOp, new TensorCreateOp(invokeOp.resultType(), operands));
                default -> blockBuilder.op(op);
            }
        }
    }

    private static class TensorFill implements TensorTransformer {

        @Override
        public void transform(Block.Builder blockBuilder, Op op) {
            List<Value> operands = blockBuilder.context().getValues(op.operands());
            switch (op) {
                case CoreOp.VarAccessOp.VarLoadOp loadOp -> replaceOp(blockBuilder, loadOp, new TensorVarLoadOp(loadOp.resultType(), operands));
                case JavaOp.InvokeOp invokeOp -> replaceOp(blockBuilder, invokeOp, new TensorFillOp(invokeOp.resultType(), operands));
                default -> blockBuilder.op(op);
            }
        }
    }

    private static class TensorMMA implements TensorTransformer {

        @Override
        public void transform(Block.Builder blockBuilder, Op op) {
            List<Value> operands = blockBuilder.context().getValues(op.operands());
            switch (op) {
                case CoreOp.VarAccessOp.VarLoadOp loadOp -> replaceOp(blockBuilder, loadOp, new TensorVarLoadOp(loadOp.resultType(), operands));
                case JavaOp.InvokeOp invokeOp -> replaceOp(blockBuilder, invokeOp, new TensorMMAOp(invokeOp.resultType(), operands));
                default -> blockBuilder.op(op);
            }
        }
    }

    private static class TensorLoad implements TensorTransformer {

        @Override
        public void transform(Block.Builder blockBuilder, Op op) {
            List<Value> operands = blockBuilder.context().getValues(op.operands());
            switch (op) {
                case CoreOp.VarAccessOp.VarStoreOp storeOp ->
                        replaceOp(blockBuilder, storeOp, new TensorStoreLoadOp(storeOp.resultType(), operands));
                case JavaOp.InvokeOp invokeOp ->
                        replaceOp(blockBuilder, invokeOp, new TensorLoadOp(invokeOp.resultType(), operands));
                default -> blockBuilder.op(op);
            }
        }
    }

    private static class TensorStore implements TensorTransformer {

        @Override
        public void transform(Block.Builder blockBuilder, Op op) {
            if (Objects.requireNonNull(op) instanceof JavaOp.InvokeOp invokeOp) {
                replaceOp(blockBuilder, invokeOp, new TensorStoreOp(invokeOp.resultType(), blockBuilder.context().getValues(invokeOp.operands())));
            } else {
                blockBuilder.op(op);
            }
        }
    }

    private CoreOp.FuncOp transformWithPredicate(MethodHandles.Lookup lookup, CoreOp.FuncOp funcOp, BiConsumer<Block.Builder, Op> function, Set<Op> opsToProcess ) {
        return Trxfmr.of(lookup, funcOp).transform(opsToProcess::contains, (blockBuilder, op) -> {
            function.accept(blockBuilder, op);
            return blockBuilder;
        }).funcOp();
    }

    private CoreOp.FuncOp createTensors(MethodHandles.Lookup lookup, CoreOp.FuncOp funcOp) {
        Set<Op> opsToProcess = new HashSet<>();
        OpHelper.Invoke.stream(lookup, funcOp)
                .filter(invoke -> !invoke.returnsVoid())
                .filter(invoke -> invoke.refIs(Tensor.class))
                .filter(invoke -> invoke.name().equals("create") || invoke.name().equals("of"))
                .forEach( invoke -> {
                    opsToProcess.add(invoke.op());
                    invoke.op().result().uses().stream()
                            .filter(result -> (result.op() instanceof CoreOp.VarOp))
                            .map(result -> (CoreOp.VarOp) result.op())
                            .findFirst()
                            .ifPresent(opsToProcess::add);
                });

        return transformWithPredicate(lookup, funcOp, new TensorView()::transform, opsToProcess);
    }

    private Set<Op> filterOps(MethodHandles.Lookup lookup, CoreOp.FuncOp funcOp, String methodIntrinsicName) {
        Set<Op> opsToProcess = new HashSet<>();
        OpHelper.Invoke.stream(lookup, funcOp)
                .filter(OpHelper.Invoke::returnsVoid)
                .filter(invoke -> invoke.refIs(Tensor.class))
                .filter(invoke -> invoke.name().equals(methodIntrinsicName))
                .forEach(invoke -> {
                    opsToProcess.add(invoke.op());
                    Value varLoadValue = invoke.op().operands().getFirst();
                    if (varLoadValue.declaringElement() instanceof CoreOp.VarAccessOp.VarLoadOp varLoadOp) {
                        opsToProcess.add(varLoadOp);
                    }
                });
        return opsToProcess;
    }

    private CoreOp.FuncOp fillTensors(MethodHandles.Lookup lookup, CoreOp.FuncOp funcOp) {
        Set<Op> opsToProcess = filterOps(lookup, funcOp, "fill");
        return transformWithPredicate(lookup, funcOp, new TensorFill()::transform, opsToProcess);
    }

    private CoreOp.FuncOp mmaTensor(MethodHandles.Lookup lookup, CoreOp.FuncOp funcOp) {
        Set<Op> opsToProcess = filterOps(lookup, funcOp, "mma");
        return transformWithPredicate(lookup, funcOp, new TensorMMA()::transform, opsToProcess);
    }

    private CoreOp.FuncOp tensorLoad(MethodHandles.Lookup lookup, CoreOp.FuncOp funcOp) {
        Set<Op> opsToProcess = new HashSet<>();
        OpHelper.Invoke.stream(lookup, funcOp)
                .filter(invoke -> !invoke.returnsVoid())
                .filter(invoke -> invoke.refIs(Tensor.class))
                .filter(invoke -> invoke.name().equals("load"))
                .forEach(invoke -> {
                    opsToProcess.add(invoke.op());
                    invoke.op().result().uses().stream()
                            .filter(result -> (result.op() instanceof CoreOp.VarAccessOp.VarStoreOp))
                            .map(result -> (CoreOp.VarAccessOp.VarStoreOp) result.op())
                            .forEach(opsToProcess::add);
                });
        return transformWithPredicate(lookup, funcOp, new TensorLoad()::transform, opsToProcess);
    }

    private CoreOp.FuncOp tensorStoreOp(MethodHandles.Lookup lookup, CoreOp.FuncOp funcOp) {
        Set<Op> opsToProcess = new HashSet<>();
        OpHelper.Invoke.stream(lookup, funcOp)
                .filter(OpHelper.Invoke::returnsVoid)
                .filter(invoke -> invoke.refIs(Tensor.class))
                .filter(invoke -> invoke.name().equals("store"))
                .forEach(invoke -> opsToProcess.add(invoke.op()));
        return transformWithPredicate(lookup, funcOp, new TensorStore()::transform, opsToProcess);
    }

    @Override
    public CoreOp.FuncOp transform(MethodHandles.Lookup lookup, CoreOp.FuncOp funcOp) {
        funcOp = createTensors(lookup, funcOp);
        funcOp = fillTensors(lookup, funcOp);
        funcOp = mmaTensor(lookup, funcOp);
        funcOp = tensorLoad(lookup, funcOp);
        funcOp = tensorStoreOp(lookup, funcOp);
        return funcOp;
    }
}
