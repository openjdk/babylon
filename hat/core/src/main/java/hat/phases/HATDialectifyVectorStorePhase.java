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

import hat.Accelerator;
import hat.dialect.HATLocalVarOp;
import hat.dialect.HATPrivateVarOp;
import hat.dialect.HATVectorStoreView;
import hat.dialect.HATVectorOp;
import hat.dialect.HATPhaseUtils;
import hat.optools.OpTk;
import hat.types._V;
import jdk.incubator.code.CodeElement;
import jdk.incubator.code.CopyContext;
import jdk.incubator.code.Op;
import jdk.incubator.code.TypeElement;
import jdk.incubator.code.Value;
import jdk.incubator.code.dialect.core.CoreOp;
import jdk.incubator.code.dialect.java.JavaOp;

import java.util.List;
import java.util.Set;
import java.util.stream.Collectors;
import java.util.stream.Stream;

public abstract  class HATDialectifyVectorStorePhase implements HATDialect {

    protected final Accelerator accelerator;
    @Override  public Accelerator accelerator(){
        return this.accelerator;
    }
    private final StoreView vectorOperation;

    public HATDialectifyVectorStorePhase(Accelerator accelerator, StoreView vectorOperation) {
        this.accelerator= accelerator;
        this.vectorOperation = vectorOperation;
    }

    public enum StoreView {
        FLOAT4_STORE("storeFloat4View");
        final String methodName;
        StoreView(String methodName) {
            this.methodName = methodName;
        }
    }

    private boolean isVectorOperation(JavaOp.InvokeOp invokeOp, Value varValue) {
        if (varValue instanceof Op.Result r && r.op() instanceof CoreOp.VarAccessOp.VarLoadOp varLoadOp) {
            TypeElement typeElement = varLoadOp.resultType();
            Set<Class<?>> interfaces = Set.of();
            try {
                Class<?> aClass = Class.forName(typeElement.toString());
                interfaces = inspectAllInterfaces(aClass);
            } catch (ClassNotFoundException _) {
            }
            return interfaces.contains(_V.class) && isMethod(invokeOp, vectorOperation.methodName);
        }
        return false;
    }

    private String findNameVector(CoreOp.VarAccessOp.VarLoadOp varLoadOp) {
        return findNameVector(varLoadOp.operands().get(0));
    }

    private String findNameVector(Value v) {
        if (v instanceof Op.Result r && r.op() instanceof CoreOp.VarAccessOp.VarLoadOp varLoadOp) {
            return findNameVector(varLoadOp);
        } else if (v instanceof CoreOp.Result r && r.op() instanceof HATVectorOp hatVectorOp) {
            return hatVectorOp.varName();
        }else{
            return null;
        }
    }

    private boolean findIsSharedOrPrivateSpace(CoreOp.VarAccessOp.VarLoadOp varLoadOp) {
        return findIsSharedOrPrivateSpace(varLoadOp.operands().get(0));
    }

    private boolean findIsSharedOrPrivateSpace(Value v) {
        if (v instanceof Op.Result r && r.op() instanceof CoreOp.VarAccessOp.VarLoadOp varLoadOp) {
            return findIsSharedOrPrivateSpace(varLoadOp);
        } else  if (v instanceof CoreOp.Result r && (r.op() instanceof HATLocalVarOp || r.op() instanceof HATPrivateVarOp)) {
            return true;
        }else{
            return false;
        }
    }

    @Override
    public CoreOp.FuncOp apply(CoreOp.FuncOp funcOp) {
        var here = OpTk.CallSite.of(this.getClass(), "apply");
        before(here,funcOp);
        Stream<CodeElement<?, ?>> vectorNodesInvolved = funcOp.elements()
                .mapMulti((codeElement, consumer) -> {
                    if (codeElement instanceof JavaOp.InvokeOp invokeOp) {
                        if ((invokeOp.operands().size() >= 3) && (isVectorOperation(invokeOp, invokeOp.operands().get(1)))) {
                            consumer.accept(invokeOp);
                        }
                    }
                });

        Set<CodeElement<?, ?>> nodesInvolved = vectorNodesInvolved.collect(Collectors.toSet());
           funcOp = OpTk.transform(here, funcOp, (blockBuilder, op) -> {
            CopyContext context = blockBuilder.context();
            if (!nodesInvolved.contains(op)) {
                blockBuilder.op(op);
            } else if (op instanceof JavaOp.InvokeOp invokeOp) {
                List<Value> inputOperandsVarOp = invokeOp.operands();
                List<Value> outputOperandsVarOp = context.getValues(inputOperandsVarOp);

                // Find the name of the vector view variable
                String name = findNameVector(invokeOp.operands().get(1));

                boolean isSharedOrPrivate = findIsSharedOrPrivateSpace(invokeOp.operands().get(0));

                HATPhaseUtils.VectorMetaData vectorMetaData  = HATPhaseUtils.getVectorTypeInfo(invokeOp, 1);
                TypeElement vectorElementType = vectorMetaData.vectorTypeElement();
                HATVectorOp storeView = new HATVectorStoreView(name, invokeOp.resultType(), vectorMetaData.lanes(), vectorElementType, isSharedOrPrivate,  outputOperandsVarOp);
                Op.Result hatLocalResult = blockBuilder.op(storeView);
                storeView.setLocation(invokeOp.location());
                context.mapValue(invokeOp.result(), hatLocalResult);
            } else if (op instanceof CoreOp.VarAccessOp.VarLoadOp varLoadOp) {
                // pass value
                context.mapValue(varLoadOp.result(), context.getValue(varLoadOp.operands().getFirst()));
            }
            return blockBuilder;
        });
        after(here, funcOp);
        return funcOp;
    }

    public static class Float4StorePhase extends HATDialectifyVectorStorePhase{
        public Float4StorePhase(Accelerator accelerator) {
            super(accelerator, StoreView.FLOAT4_STORE);
        }
    }
}
