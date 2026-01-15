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
import hat.device.DeviceType;
import hat.dialect.HATMemoryDefOp;
import hat.dialect.HATMemoryVarOp;
import hat.types.HAType;
import jdk.incubator.code.Block;
import jdk.incubator.code.CodeElement;
import jdk.incubator.code.Op;
import jdk.incubator.code.dialect.core.CoreOp;
import jdk.incubator.code.dialect.java.ClassType;
import jdk.incubator.code.dialect.java.JavaOp;
import optkl.OpHelper;
import optkl.Trxfmr;
import optkl.ifacemapper.MappableIface;
import optkl.util.Regex;

import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashSet;
import java.util.Map;
import java.util.Set;

import static optkl.OpHelper.Named.NamedStaticOrInstance.Invoke;
import static optkl.OpHelper.Named.NamedStaticOrInstance.Invoke.invoke;
import static optkl.OpHelper.copyLocation;

public abstract sealed class HATMemoryPhase implements HATPhase {

    protected final KernelCallGraph kernelCallGraph;

    @Override
    public KernelCallGraph kernelCallGraph(){
        return this.kernelCallGraph;
    }


    protected abstract HATMemoryVarOp create(Block.Builder builder, CoreOp.VarOp varOp, JavaOp.InvokeOp invokeOp);

    protected abstract boolean isIfaceBufferInvokeWithName(Invoke invoke);

    public HATMemoryPhase(KernelCallGraph kernelCallGraph) {
        this.kernelCallGraph = kernelCallGraph;
    }

    @Override
    public CoreOp.FuncOp apply(CoreOp.FuncOp funcOp) {
        Set<CodeElement<?,?>> nodesInvolved = new LinkedHashSet<>();
        Set<JavaOp.InvokeOp> mapMe = new LinkedHashSet<>();
        OpHelper.Named.Var.stream(lookup(),funcOp)
                .forEach(varHelper->varHelper.op().operands().stream()
                        .filter(operand -> operand instanceof Op.Result result
                                && invoke(lookup(),result.op()) instanceof Invoke invoke
                                && isIfaceBufferInvokeWithName(invoke))
                        .map(r -> (JavaOp.InvokeOp) (((Op.Result) r).op()))
                        .findFirst().ifPresent(remove-> {
                            nodesInvolved.add(varHelper.op());
                            mapMe.add(remove);
                    })
                );

        return Trxfmr.of(this,funcOp).transform(ce->mapMe.contains(ce)||nodesInvolved.contains(ce), (blockBuilder, op) -> {
            if (invoke(lookup(),op) instanceof Invoke invoke && mapMe.contains(invoke.op())) {
                invoke.op().result().uses().stream()
                        .filter(result->result.op() instanceof CoreOp.VarOp)
                        .map(r->(CoreOp.VarOp)r.op())
                        .forEach(varOp->
                            blockBuilder.context().mapValue(invoke.op().result(), blockBuilder.op(create(blockBuilder, varOp, invoke.op())))
                        );
            } else if (OpHelper.Named.Var.var(lookup(),op) instanceof OpHelper.Named.Var varHelper && nodesInvolved.contains(varHelper.op())) {
                blockBuilder.context().mapValue(varHelper.op().result(), blockBuilder.context().getValue(varHelper.op().operands().getFirst()));
            } else {
                blockBuilder.op(op);
            }
            return blockBuilder;
        }).funcOp();
    }


    public static final class PrivateMemoryPhase extends HATMemoryPhase {
        public PrivateMemoryPhase(KernelCallGraph kernelCallGraph) {
            super(kernelCallGraph);
        }
        public static final String INTRINSIC_NAME = "createPrivate";
        @Override
        protected boolean isIfaceBufferInvokeWithName(Invoke invoke) {
            return invoke.refIs( DeviceType.class, MappableIface.class, HAType.class)
                    && invoke.named(INTRINSIC_NAME);

        }

        @Override
        protected HATMemoryVarOp create(Block.Builder builder, CoreOp.VarOp varOp, JavaOp.InvokeOp invokeOp) {
            var op = new HATMemoryVarOp.HATPrivateVarOp(
                    varOp.varName(),
                    (ClassType) varOp.varValueType(),
                    varOp.resultType(),
                    invokeOp.resultType(),
                    builder.context().getValues(invokeOp.operands())
            );
            op.setLocation(varOp.location());
            return op;
        }
    }

    public static final  class LocalMemoryPhase extends HATMemoryPhase {
        public LocalMemoryPhase(KernelCallGraph kernelCallGraph) {
            super(kernelCallGraph);
        }
        public static final String INTRINSIC_NAME = "createLocal";
        @Override
        protected boolean isIfaceBufferInvokeWithName(Invoke invoke){
            return invoke.refIs( DeviceType.class, MappableIface.class, HAType.class) && invoke.named(INTRINSIC_NAME);

        }

        @Override
        protected HATMemoryVarOp create(Block.Builder builder, CoreOp.VarOp varOp, JavaOp.InvokeOp invokeOp) {
            return  copyLocation(varOp,new HATMemoryVarOp.HATLocalVarOp(
                    varOp.varName(),
                    (ClassType) varOp.varValueType(),
                    varOp.resultType(),
                    invokeOp.resultType(),
                    builder.context().getValues(invokeOp.operands())
            ));
        }
    }

    public static final class DeviceTypePhase extends HATMemoryPhase {

        public DeviceTypePhase(KernelCallGraph kernelCallGraph) {
            super(kernelCallGraph);
        }
        public static final String INTRINSIC_NAME = "createLocal";
        @Override
        protected boolean isIfaceBufferInvokeWithName(Invoke invoke){
            return invoke.refIs( DeviceType.class, MappableIface.class, HAType.class) && invoke.named(INTRINSIC_NAME);
        }

        static private Regex reservedMethods = Regex.of("(createLocal|createPrivate|create|float2View|float4View)");
        @Override
        public CoreOp.FuncOp apply(CoreOp.FuncOp funcOp) {
            Map<CoreOp.VarOp, JavaOp.InvokeOp> varTable = new HashMap<>();
            Set<CodeElement<?, ?>> nodesInvolved = new HashSet<>();
            Invoke.stream(lookup(),funcOp)
                    .filter(invoke->invoke.refIs(DeviceType.class) && invoke.returnsClassType() && !invoke.named(reservedMethods))
                    .forEach(invoke -> invoke.op().result().uses().stream()
                           .filter(use->use.op() instanceof CoreOp.VarOp)
                           .map(use->(CoreOp.VarOp)use.op())
                           .forEach(varOp -> {
                                varTable.put(varOp, invoke.op());
                                nodesInvolved.add(invoke.op());
                                nodesInvolved.add(varOp);
                           })
                    );

            return Trxfmr.of(this,funcOp).transform(nodesInvolved::contains, (blockBuilder, op) -> {
               if (invoke(lookup(),op) instanceof Invoke invoke) {
                   blockBuilder.context().mapValue(invoke.op().result(),
                           blockBuilder.op(invoke.copyLocationTo(
                                   new HATMemoryDefOp.HATMemoryLoadOp(invoke.returnType(),
                                           invoke.refType(),
                                           invoke.name(),
                                           blockBuilder.context().getValues(invoke.op().operands())))
                           )
                   );
                } else if (op instanceof CoreOp.VarOp varOp) {
                    create(blockBuilder, varOp, varTable.get(varOp));
                }
                return blockBuilder;
            }).funcOp();
        }
        @Override
        protected HATMemoryVarOp create(Block.Builder blockBuilder, CoreOp.VarOp varOp, JavaOp.InvokeOp invokeOp) {
            var  privateVarOp = copyLocation(varOp,new HATMemoryVarOp.HATPrivateInitVarOp(varOp.varName(),
                    (ClassType) varOp.varValueType(),
                    varOp.resultType(),
                    invokeOp.invokeDescriptor().refType(),
                    blockBuilder.context().getValues(varOp.operands())));
            blockBuilder.context().mapValue(varOp.result(), blockBuilder.op(privateVarOp));
            return privateVarOp;
        }
    }
}
