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

import hat.callgraph.KernelCallGraph;
import hat.dialect.HATMemoryVarOp;
import hat.dialect.HATVectorOp;
import hat.dialect.HATPhaseUtils;
import hat.types._V;
import jdk.incubator.code.CodeContext;
import jdk.incubator.code.CodeElement;
import jdk.incubator.code.Op;
import jdk.incubator.code.TypeElement;
import jdk.incubator.code.Value;
import jdk.incubator.code.dialect.core.CoreOp;
import jdk.incubator.code.dialect.java.JavaOp;
import optkl.Invoke;
import optkl.Trxfmr;

import java.util.List;
import java.util.Set;
import java.util.stream.Collectors;
import java.util.stream.Stream;

import static optkl.Invoke.invokeOpHelper;
import static optkl.OpTkl.asResultOrNull;
import static optkl.OpTkl.isAssignable;

public abstract sealed class HATVectorStorePhase implements HATPhase
        permits HATVectorStorePhase.Float2StorePhase, HATVectorStorePhase.Float4StorePhase{

    protected final KernelCallGraph kernelCallGraph;
    @Override  public KernelCallGraph kernelCallGraph(){
        return this.kernelCallGraph;
    }
    public HATVectorStorePhase(KernelCallGraph kernelCallGraph/*, StoreView vectorOperation*/) {
        this.kernelCallGraph= kernelCallGraph;
    }

    //recursive
    private String findNameVector(Value v) {
        if (v instanceof Op.Result r && r.op() instanceof CoreOp.VarAccessOp.VarLoadOp varLoadOp) {
            return findNameVector(varLoadOp.operands().getFirst());
        } else if (v instanceof CoreOp.Result r && r.op() instanceof HATVectorOp hatVectorOp) {
            return hatVectorOp.varName();
        }else{
            throw new IllegalStateException("no name");
        }
    }

    //recursive
    private boolean findIsSharedOrPrivateSpace(Value v) {
        if (v instanceof Op.Result r && r.op() instanceof CoreOp.VarAccessOp.VarLoadOp varLoadOp) {
            return findIsSharedOrPrivateSpace(varLoadOp.operands().getFirst());
        } else{
            return (v instanceof CoreOp.Result r && (r.op() instanceof HATMemoryVarOp.HATLocalVarOp || r.op() instanceof HATMemoryVarOp.HATPrivateVarOp));
        }
    }

    @Override
    public CoreOp.FuncOp apply(CoreOp.FuncOp funcOp) {

        Stream<CodeElement<?, ?>> vectorNodesInvolved = funcOp.elements()
                .mapMulti((codeElement, consumer) -> {
                    if (invokeOpHelper(lookup(),codeElement)instanceof Invoke invoke
                            && (invoke.op().operands().size() >2)
                            && invoke.named(
                            switch (HATVectorStorePhase.this) {
                               case Float2StorePhase _ -> "storeFloat2View";
                               case Float4StorePhase _ -> "storeFloat4View";
                            })
                            && asResultOrNull(invoke.op().operands().get(1)) instanceof Op.Result result
                            && result.op() instanceof CoreOp.VarAccessOp.VarLoadOp varLoadOp
                            && isAssignable(lookup(),varLoadOp.resultType(), _V.class)){
                            consumer.accept(invoke.op());
                        }
                });

        Set<CodeElement<?, ?>> nodesInvolved = vectorNodesInvolved.collect(Collectors.toSet());
           return new Trxfmr(funcOp).transform(nodesInvolved::contains, (blockBuilder, op) -> {
            CodeContext context = blockBuilder.context();
            if (op instanceof JavaOp.InvokeOp invokeOp) {
                List<Value> inputOperandsVarOp = invokeOp.operands();
                List<Value> outputOperandsVarOp = context.getValues(inputOperandsVarOp);

                boolean isSharedOrPrivate = findIsSharedOrPrivateSpace(invokeOp.operands().get(0));

                HATPhaseUtils.VectorMetaData vectorMetaData  = HATPhaseUtils.getVectorTypeInfo(lookup(),invokeOp, 1);
                TypeElement vectorElementType = vectorMetaData.vectorTypeElement();
                HATVectorOp storeView = new HATVectorOp.HATVectorStoreView(findNameVector(invokeOp.operands().get(1)), invokeOp.resultType(), vectorMetaData.lanes(),
                        vectorElementType, isSharedOrPrivate,  outputOperandsVarOp);
                Op.Result hatLocalResult = blockBuilder.op(storeView);
                storeView.setLocation(invokeOp.location());
                context.mapValue(invokeOp.result(), hatLocalResult);
            } else if (op instanceof CoreOp.VarAccessOp.VarLoadOp varLoadOp) {
                // pass value
                context.mapValue(varLoadOp.result(), context.getValue(varLoadOp.operands().getFirst()));
            }
            return blockBuilder;
        }).funcOp();
    }

    public static final class Float4StorePhase extends HATVectorStorePhase {
        public Float4StorePhase(KernelCallGraph kernelCallGraph) {
            super(kernelCallGraph);
        }
    }

    public static final class Float2StorePhase extends HATVectorStorePhase {
        public Float2StorePhase(KernelCallGraph kernelCallGraph) {
            super(kernelCallGraph);
        }
    }
}
