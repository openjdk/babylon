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
import jdk.incubator.code.TypeElement;
import jdk.incubator.code.dialect.core.CoreOp;
import jdk.incubator.code.dialect.java.ClassType;
import jdk.incubator.code.dialect.java.JavaOp;
import jdk.incubator.code.dialect.java.JavaType;
import optkl.Invoke;
import optkl.Trxfmr;
import optkl.ifacemapper.MappableIface;
import optkl.util.CallSite;

import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashSet;
import java.util.Map;
import java.util.Set;
import java.util.stream.Collectors;
import java.util.stream.Stream;

import static optkl.Invoke.invokeOpHelper;
import static optkl.Trxfmr.copyLocation;

public abstract sealed class HATMemoryPhase implements HATPhase {

    protected final KernelCallGraph kernelCallGraph;

    @Override
    public KernelCallGraph kernelCallGraph(){
        return this.kernelCallGraph;
    }

    private static final Set<String> reservedMethods = new HashSet<>();

    static {
        reservedMethods.add("createLocal");
        reservedMethods.add("createPrivate");
        reservedMethods.add("create");
        reservedMethods.add("float2View");
        reservedMethods.add("float4View");
    }

    protected abstract HATMemoryVarOp factory(Block.Builder builder, CoreOp.VarOp varOp, JavaOp.InvokeOp invokeOp);

    protected abstract boolean isIfaceBufferInvokeWithName(Invoke invoke);

    public HATMemoryPhase(KernelCallGraph kernelCallGraph) {
        this.kernelCallGraph = kernelCallGraph;
    }

    @Override
    public CoreOp.FuncOp apply(CoreOp.FuncOp funcOp) {
        Set<CoreOp.VarOp> removeMe = new LinkedHashSet<>();
        Set<JavaOp.InvokeOp> mapMe = new LinkedHashSet<>();
        funcOp.elements()
                .filter(e -> e instanceof CoreOp.VarOp )
                .map(e-> (CoreOp.VarOp) e)
                .forEach(varOp->varOp
                        .operands()
                        .stream()
                        .filter(o -> o instanceof Op.Result result
                                && result.op() instanceof JavaOp.InvokeOp invokeOp
                                && isIfaceBufferInvokeWithName(invokeOpHelper(lookup(),invokeOp)))
                        .map(r -> (JavaOp.InvokeOp) (((Op.Result) r).op()))
                        .findFirst().ifPresent(remove-> {
                            removeMe.add(varOp);
                            mapMe.add(remove);
                    })
                );

        return new Trxfmr(funcOp).transform(ce->mapMe.contains(ce)||removeMe.contains(ce), (blockBuilder, op) -> {
            if (op instanceof JavaOp.InvokeOp invokeOp && mapMe.contains(invokeOp)) {
                invokeOp.result().uses().stream()
                        .filter(result->result.op() instanceof CoreOp.VarOp)
                        .map(r->(CoreOp.VarOp)r.op())
                        .forEach(varOp->
                            blockBuilder.context().mapValue(invokeOp.result(), blockBuilder.op(factory(blockBuilder, varOp, invokeOp)))
                        );
            } else if (op instanceof CoreOp.VarOp varOp && removeMe.contains(varOp)) {
                blockBuilder.context().mapValue(varOp.result(), blockBuilder.context().getValue(varOp.operands().getFirst()));
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

        @Override
        protected boolean isIfaceBufferInvokeWithName(Invoke invoke) {
            return invoke.refIs( DeviceType.class, MappableIface.class, HAType.class)
                    && invoke.named(HATMemoryVarOp.HATPrivateVarOp.INTRINSIC_NAME);

        }

        @Override
        protected HATMemoryVarOp factory(Block.Builder builder, CoreOp.VarOp varOp, JavaOp.InvokeOp invokeOp) {
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

        @Override
        protected boolean isIfaceBufferInvokeWithName(Invoke invoke){
            return invoke.refIs( DeviceType.class, MappableIface.class, HAType.class)
                    && invoke.named(HATMemoryVarOp.HATLocalVarOp.INTRINSIC_NAME);

        }

        @Override
        protected HATMemoryVarOp factory(Block.Builder builder, CoreOp.VarOp varOp, JavaOp.InvokeOp invokeOp) {
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

        @Override
        protected boolean isIfaceBufferInvokeWithName(Invoke invoke){
            return invoke.refIs( DeviceType.class, MappableIface.class, HAType.class)
                 || invoke.named(HATMemoryVarOp.HATLocalVarOp.INTRINSIC_NAME);
        }

        @Override
        public CoreOp.FuncOp apply(CoreOp.FuncOp funcOp) {
            Map<CoreOp.VarOp, JavaOp.InvokeOp> varTable = new HashMap<>();
            Stream<CodeElement<?, ?>> memoryLoadOps = funcOp.elements()
                    .mapMulti((codeElement, consumer) -> {
                        if (invokeOpHelper(lookup(),codeElement) instanceof Invoke invoke
                             && invoke.refIs(DeviceType.class)
                                    && !invoke.returnsVoid()
                                    && !invoke.returnsPrimitive()
                                    && !reservedMethods.contains(invoke.name())) {
                                Op.Result result = invoke.op().result();
                                Set<Op.Result> uses = result.uses();
                                for (Op.Result use : uses) {
                                    if (use.op() instanceof CoreOp.VarOp varOp) {
                                        varTable.put(varOp, invoke.op());
                                        consumer.accept(invoke.op());
                                        consumer.accept(varOp);
                                    }
                                }

                        }
                    });

            Set<CodeElement<?, ?>> nodesInvolved = memoryLoadOps.collect(Collectors.toSet());
            return new Trxfmr(funcOp).transform(nodesInvolved::contains, (blockBuilder, op) -> {
               if (op instanceof JavaOp.InvokeOp invokeOp) {
                    insertHatMemoryLoadOp(blockBuilder, invokeOp);
                } else if (op instanceof CoreOp.VarOp varOp) {
                    factory(blockBuilder, varOp, varTable.get(varOp));
                }
                return blockBuilder;
            }).funcOp();
        }

        private void insertHatMemoryLoadOp(Block.Builder blockBuilder, JavaOp.InvokeOp invokeOp) {
            HATMemoryDefOp.HATMemoryLoadOp loadOp = new HATMemoryDefOp.HATMemoryLoadOp(invokeOp.resultType(),
                    invokeOp.invokeDescriptor().refType(),
                    invokeOp.invokeDescriptor().name(),
                    blockBuilder.context().getValues(invokeOp.operands()));
            Op.Result resultLoad = blockBuilder.op(loadOp);
            loadOp.setLocation(invokeOp.location());
            blockBuilder.context().mapValue(invokeOp.result(), resultLoad);
        }

        @Override
        protected HATMemoryVarOp factory(Block.Builder blockBuilder, CoreOp.VarOp varOp, JavaOp.InvokeOp invokeOp) {
            HATMemoryVarOp.HATPrivateInitVarOp privateVarOp = new HATMemoryVarOp.HATPrivateInitVarOp(varOp.varName(),
                    (ClassType) varOp.varValueType(),
                    varOp.resultType(),
                    invokeOp.invokeDescriptor().refType(),
                    blockBuilder.context().getValues(varOp.operands()));
            Op.Result op1 = blockBuilder.op(privateVarOp);
            privateVarOp.setLocation(varOp.location());
            blockBuilder.context().mapValue(varOp.result(), op1);
            return privateVarOp;
        }
    }
}
