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

import hat.callgraph.KernelCallGraph;
import hat.dialect.HATTensorOp;
import hat.types.Tensor;
import jdk.incubator.code.Block;
import jdk.incubator.code.Op;
import jdk.incubator.code.Value;
import jdk.incubator.code.dialect.core.CoreOp;
import jdk.incubator.code.dialect.java.JavaOp;
import optkl.OpHelper;
import optkl.Trxfmr;

import java.util.HashSet;
import java.util.Set;

import static hat.dialect.HATTensorOp.*;

public record HATTensorsPhase(KernelCallGraph kernelCallGraph) implements HATPhase {

    private void transformTensorDeclWithVarOp(Block.Builder blockBuilder, Op op) {
        switch (op) {
            case CoreOp.VarOp varOp -> {
                HATTensorOp tensorVarOp = new TensorVarOp(varOp.varName(), varOp.resultType(), blockBuilder.context().getValues(varOp.operands()));
                tensorVarOp.setLocation(varOp.location());
                Op.Result opResult = blockBuilder.op(tensorVarOp);
                blockBuilder.context().mapValue(varOp.result(), opResult);
            }
            case JavaOp.InvokeOp invokeOp -> {
                TensorCreateOp tensorOf = new TensorCreateOp(invokeOp.resultType(), //
                        blockBuilder.context().getValues(invokeOp.operands()));
                tensorOf.setLocation(invokeOp.location());
                Op.Result opResult = blockBuilder.op(tensorOf);
                blockBuilder.context().mapValue(invokeOp.result(), opResult);
            }
            case null, default -> blockBuilder.op(op);
        }
    }

    private void transformTensorFillOp(Block.Builder blockBuilder, Op op) {
        switch (op) {
            case CoreOp.VarAccessOp.VarLoadOp loadOp -> {
                TensorVarLoadOp varLoad = new TensorVarLoadOp(loadOp.resultType(), //
                        blockBuilder.context().getValues(loadOp.operands()));
                varLoad.setLocation(loadOp.location());
                Op.Result opResult = blockBuilder.op(varLoad);
                blockBuilder.context().mapValue(loadOp.result(), opResult);
            }
            case JavaOp.InvokeOp invokeOp -> {
                TensorFillOp fillOp = new TensorFillOp(invokeOp.resultType(), //
                        blockBuilder.context().getValues(invokeOp.operands()));
                fillOp.setLocation(invokeOp.location());
                Op.Result opResult = blockBuilder.op(fillOp);
                blockBuilder.context().mapValue(invokeOp.result(), opResult);
            }
            case null, default -> blockBuilder.op(op);
        }
    }

    private void transformTensorMMAOp(Block.Builder blockBuilder, Op op) {
        switch (op) {
            case CoreOp.VarAccessOp.VarLoadOp loadOp -> {
                TensorVarLoadOp varLoad = new TensorVarLoadOp(loadOp.resultType(), //
                        blockBuilder.context().getValues(loadOp.operands()));
                varLoad.setLocation(loadOp.location());
                Op.Result opResult = blockBuilder.op(varLoad);
                blockBuilder.context().mapValue(loadOp.result(), opResult);
            }
            case JavaOp.InvokeOp invokeOp -> {
                TensorMMAOp mmaOp = new TensorMMAOp(invokeOp.resultType(), //
                        blockBuilder.context().getValues(invokeOp.operands()));
                mmaOp.setLocation(invokeOp.location());
                Op.Result opResult = blockBuilder.op(mmaOp);
                blockBuilder.context().mapValue(invokeOp.result(), opResult);
            }
            case null, default -> blockBuilder.op(op);
        }
    }

    private CoreOp.FuncOp createTensors(CoreOp.FuncOp funcOp) {
        // 1. Analyse IR calls to Tensor.create
        Set<Op> opsToProcess = new HashSet<>();
        OpHelper.Invoke.stream(lookup(), funcOp)
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

        // 2. Transform the IR
        return Trxfmr.of(this, funcOp).transform(opsToProcess::contains, (blockBuilder, op) -> {
            transformTensorDeclWithVarOp(blockBuilder, op);
            return blockBuilder;
        }).funcOp();
    }

    private void transformTensorLoadOp(Block.Builder blockBuilder, Op op) {
        switch (op) {
            case CoreOp.VarAccessOp.VarStoreOp storeOp -> {
                TensorStoreLoadOp tensorStoreLoadOp = new TensorStoreLoadOp(storeOp.resultType(), //
                        blockBuilder.context().getValues(storeOp.operands()));
                tensorStoreLoadOp.setLocation(tensorStoreLoadOp.location());
                Op.Result opResult = blockBuilder.op(tensorStoreLoadOp);
                blockBuilder.context().mapValue(tensorStoreLoadOp.result(), opResult);
            }
            case JavaOp.InvokeOp invokeOp -> {
                TensorLoadOp mmaOp = new TensorLoadOp(invokeOp.resultType(), //
                        blockBuilder.context().getValues(invokeOp.operands()));
                mmaOp.setLocation(invokeOp.location());
                Op.Result opResult = blockBuilder.op(mmaOp);
                blockBuilder.context().mapValue(invokeOp.result(), opResult);
            }
            case null, default -> blockBuilder.op(op);
        }
    }

    private void transformTensorStoreOp(Block.Builder blockBuilder, Op op) {
        switch (op) {
            case JavaOp.InvokeOp invokeOp -> {
                TensorStoreOp storeOp = new TensorStoreOp(invokeOp.resultType(), //
                        blockBuilder.context().getValues(invokeOp.operands()));
                storeOp.setLocation(invokeOp.location());
                Op.Result opResult = blockBuilder.op(storeOp);
                blockBuilder.context().mapValue(invokeOp.result(), opResult);
            }
            case null, default -> blockBuilder.op(op);
        }
    }

    private CoreOp.FuncOp fillTensors(CoreOp.FuncOp funcOp) {
        // 1. Analyse IR calls for Tensor.fill
        Set<Op> opsToProcess = new HashSet<>();
        OpHelper.Invoke.stream(lookup(), funcOp)
                .filter(OpHelper.Invoke::returnsVoid)
                .filter(invoke -> invoke.refIs(Tensor.class))
                .filter(invoke -> invoke.name().equals("fill"))
                .forEach(invoke -> {
                    opsToProcess.add(invoke.op());
                    Value varLoadValue = invoke.op().operands().getFirst();
                    if (varLoadValue.declaringElement() instanceof CoreOp.VarAccessOp.VarLoadOp varLoadOp) {
                        opsToProcess.add(varLoadOp);
                    }
                });

        // 2. Transform the IR (tensor.fill into dialect.tensor.fill)
        return Trxfmr.of(this, funcOp).transform(opsToProcess::contains, (blockBuilder, op) -> {
            transformTensorFillOp(blockBuilder, op);
            return blockBuilder;
        }).funcOp();
    }

    private CoreOp.FuncOp mmaTensor(CoreOp.FuncOp funcOp) {
        Set<Op> opsToProcess = new HashSet<>();
        OpHelper.Invoke.stream(lookup(), funcOp)
                .filter(OpHelper.Invoke::returnsVoid)
                .filter(invoke -> invoke.refIs(Tensor.class))
                .filter(invoke -> invoke.name().equals("mma"))
                .forEach(invoke -> {
                    opsToProcess.add(invoke.op());
                    Value varLoadValue = invoke.op().operands().getFirst();
                    if (varLoadValue.declaringElement() instanceof CoreOp.VarAccessOp.VarLoadOp varLoadOp) {
                        opsToProcess.add(varLoadOp);
                    }
                });
        return Trxfmr.of(this, funcOp).transform(opsToProcess::contains, (blockBuilder, op) -> {
            transformTensorMMAOp(blockBuilder, op);
            return blockBuilder;
        }).funcOp();
    }

    private CoreOp.FuncOp tensorLoad(CoreOp.FuncOp funcOp) {
        Set<Op> opsToProcess = new HashSet<>();
        OpHelper.Invoke.stream(lookup(), funcOp)
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
        return Trxfmr.of(this, funcOp).transform(opsToProcess::contains, (blockBuilder, op) -> {
            transformTensorLoadOp(blockBuilder, op);
            return blockBuilder;
        }).funcOp();
    }

    private CoreOp.FuncOp tensorStoreOp(CoreOp.FuncOp funcOp) {
        Set<Op> opsToProcess = new HashSet<>();
        OpHelper.Invoke.stream(lookup(), funcOp)
                .filter(OpHelper.Invoke::returnsVoid)
                .filter(invoke -> invoke.refIs(Tensor.class))
                .filter(invoke -> invoke.name().equals("store"))
                .forEach(invoke -> opsToProcess.add(invoke.op()));
        return Trxfmr.of(this, funcOp).transform(opsToProcess::contains, (blockBuilder, op) -> {
            transformTensorStoreOp(blockBuilder, op);
            return blockBuilder;
        }).funcOp();
    }

    @Override
    public CoreOp.FuncOp apply(CoreOp.FuncOp funcOp) {

        // Traverse the code-model and analyse Tensor.create / Tensor.of
        // This will trigger a wma::fragment in the case of NVIDIA/CUDA
        // and a private memory allocation in the case of OpenCL.

        // The CUDA generator can ignore a TensorVarOp
        // The OpenCL generator can generate the private allocation in FP16
        // Thus, it should enable FP16 as well.
        funcOp = createTensors(funcOp);
        funcOp = fillTensors(funcOp);
        funcOp = mmaTensor(funcOp);
        funcOp = tensorLoad(funcOp);
        funcOp = tensorStoreOp(funcOp);
        return funcOp;
    }
}
