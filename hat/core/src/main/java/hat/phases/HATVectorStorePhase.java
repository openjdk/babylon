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
import hat.types._V;
import jdk.incubator.code.CodeContext;
import jdk.incubator.code.CodeElement;
import jdk.incubator.code.Op;
import jdk.incubator.code.Value;
import jdk.incubator.code.dialect.core.CoreOp;
import optkl.Trxfmr;

import java.util.HashSet;
import java.util.Set;
import static optkl.OpHelper.Invoke;
import static optkl.OpHelper.Invoke.invoke;
import static optkl.OpHelper.VarAccess;
import static optkl.OpHelper.VarAccess.varAccess;
import static optkl.OpHelper.copyLocation;

public abstract sealed class HATVectorStorePhase implements HATPhase
        permits HATVectorStorePhase.Float2StorePhase, HATVectorStorePhase.Float4StorePhase{

    protected final KernelCallGraph kernelCallGraph;
    @Override  public KernelCallGraph kernelCallGraph(){
        return this.kernelCallGraph;
    }
    public HATVectorStorePhase(KernelCallGraph kernelCallGraph/*, StoreView vectorOperation*/) {
        this.kernelCallGraph= kernelCallGraph;
    }

    abstract String storeViewName();

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
            return findIsSharedOrPrivateSpace(varLoadOp.operands().getFirst()); //recurses here
        } else{
            return (v instanceof CoreOp.Result r && (r.op() instanceof HATMemoryVarOp.HATLocalVarOp || r.op() instanceof HATMemoryVarOp.HATPrivateVarOp));
        }
    }

    @Override
    public CoreOp.FuncOp apply(CoreOp.FuncOp funcOp) {
        Set<CodeElement<?,?>> nodesInvolved = new HashSet<>();
        Invoke.stream(lookup(),funcOp).forEach(invoke->{
              if ( invoke.named(storeViewName())
                   && varAccess(lookup(),invoke.opFromOperandNOrNull(1)) instanceof VarAccess varAccess
                   && varAccess.isLoad() && varAccess.isAssignable( _V.class)){
                   nodesInvolved.add(invoke.op());
              }
        });

        return Trxfmr.of(this,funcOp).transform(nodesInvolved::contains, (blockBuilder, op) -> {
            CodeContext context = blockBuilder.context();
            if (invoke(lookup(),op) instanceof Invoke invoke) {
                HATPhaseUtils.VectorMetaData vectorMetaData  = HATPhaseUtils.getVectorTypeInfo(lookup(),invoke.op(), 1);
                HATVectorOp storeView = new HATVectorOp.HATVectorStoreView(
                        findNameVector(invoke.resultFromOperandNOrThrow(1)),
                        invoke.returnType(),
                        vectorMetaData.lanes(),
                        vectorMetaData.vectorTypeElement(),
                        findIsSharedOrPrivateSpace(invoke.op().operands().getFirst()),
                        context.getValues(invoke.op().operands())
                );
                context.mapValue(invoke.op().result(), blockBuilder.op(copyLocation(invoke.op(),storeView)));
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
        @Override
         String storeViewName() {
            return "storeFloat4View";
        }
    }

    public static final class Float2StorePhase extends HATVectorStorePhase {
        public Float2StorePhase(KernelCallGraph kernelCallGraph) {
            super(kernelCallGraph);
        }
        @Override
        String storeViewName() {
            return "storeFloat2View";
        }
    }
}
