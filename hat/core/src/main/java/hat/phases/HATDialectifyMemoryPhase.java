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
import hat.device.DeviceType;
import hat.dialect.HATLocalVarOp;
import hat.dialect.HATMemoryLoadOp;
import hat.dialect.HATMemoryVarOp;
import hat.dialect.HATPrivateInitVarOp;
import hat.dialect.HATPrivateVarOp;
import hat.optools.OpTk;
import jdk.incubator.code.Block;
import jdk.incubator.code.CodeElement;
import jdk.incubator.code.Op;
import jdk.incubator.code.dialect.core.CoreOp;
import jdk.incubator.code.dialect.java.ClassType;
import jdk.incubator.code.dialect.java.JavaOp;
import jdk.incubator.code.dialect.java.JavaType;
import jdk.incubator.code.dialect.java.PrimitiveType;
import optkl.LookupCarrier;

import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashSet;
import java.util.Map;
import java.util.Set;
import java.util.stream.Collectors;
import java.util.stream.Stream;

public abstract class HATDialectifyMemoryPhase implements HATDialect {

    protected final LookupCarrier lookupCarrier;

    private static final Set<String> reservedMethods = new HashSet<>();

    static {
        reservedMethods.add("createLocal");
        reservedMethods.add("createPrivate");
        reservedMethods.add("create");
        reservedMethods.add("float2View");
        reservedMethods.add("float4View");
    }

    @Override
    public LookupCarrier lookupCarrier(){
        return this.lookupCarrier;
    }

    protected abstract HATMemoryVarOp factory(Block.Builder builder, CoreOp.VarOp varOp, JavaOp.InvokeOp invokeOp);

    protected abstract boolean isIfaceBufferInvokeWithName(JavaOp.InvokeOp invokeOp);

    public HATDialectifyMemoryPhase(LookupCarrier lookupCarrier) {
        this.lookupCarrier = lookupCarrier;
    }

    @Override
    public CoreOp.FuncOp apply(CoreOp.FuncOp funcOp) {
        var here = OpTk.CallSite.of(PrivateMemoryPhase.class, "HATDialectifyMemoryPhase");
        before(here,funcOp);
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
                                && isIfaceBufferInvokeWithName(invokeOp))
                        .map(r -> (JavaOp.InvokeOp) (((Op.Result) r).op()))
                        .findFirst().ifPresent(remove-> {
                            removeMe.add(varOp);
                            mapMe.add(remove);
                    })
                );

        funcOp = OpTk.transform(here, funcOp, (blockBuilder, op) -> {
            if (op instanceof JavaOp.InvokeOp invokeOp && mapMe.contains(invokeOp)) {
                invokeOp.result()
                        .uses()
                        .stream()
                        .filter(r->r.op() instanceof CoreOp.VarOp).map(r->(CoreOp.VarOp)r.op())
                        .forEach(varOp->
                            blockBuilder.context().mapValue(invokeOp.result(), blockBuilder.op(factory(blockBuilder, varOp, invokeOp)))
                        );
            } else if (op instanceof CoreOp.VarOp varOp && removeMe.contains(varOp)) {
                      blockBuilder.context().mapValue(varOp.result(), blockBuilder.context().getValue(varOp.operands().getFirst()));
            } else {
                blockBuilder.op(op);
            }
            return blockBuilder;
        });
        after(here,funcOp );
        return funcOp;
    }


    public static class PrivateMemoryPhase extends HATDialectifyMemoryPhase {
        public PrivateMemoryPhase(LookupCarrier lookupCarrier) {
            super(lookupCarrier);
        }

        @Override
        protected boolean isIfaceBufferInvokeWithName(JavaOp.InvokeOp invokeOp) {
            if (OpTk.isIfaceBufferInvokeOpWithName(lookupCarrier.lookup(), invokeOp, n->n.equals(HATPrivateVarOp.INTRINSIC_NAME))) {
                return true;
            } else {
                return OpTk.isMethod(invokeOp, n->n.equals(HATPrivateVarOp.INTRINSIC_NAME))
                        && OpTk.isAssignable(lookupCarrier.lookup(),invokeOp.invokeDescriptor().refType(),DeviceType.class);
            }
        }

        @Override
        protected HATMemoryVarOp factory(Block.Builder builder, CoreOp.VarOp varOp, JavaOp.InvokeOp invokeOp) {
            var op = new HATPrivateVarOp(
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

    public static class LocalMemoryPhase extends HATDialectifyMemoryPhase {

        public LocalMemoryPhase(LookupCarrier lookupCarrier) {
            super(lookupCarrier);
        }

        @Override
        protected boolean isIfaceBufferInvokeWithName(JavaOp.InvokeOp invokeOp){
            if (OpTk.isIfaceBufferInvokeOpWithName(lookupCarrier.lookup(),invokeOp, n->n.equals(HATLocalVarOp.INTRINSIC_NAME))) {
                return true;
            } else {
                return (OpTk.isMethod(invokeOp, n->n.equals(HATLocalVarOp.INTRINSIC_NAME))
                        && invokeOp.resultType() instanceof JavaType javaType &&
                        OpTk.isAssignable(lookupCarrier.lookup(),javaType,DeviceType.class));
            }
        }

        @Override
        protected HATMemoryVarOp factory(Block.Builder builder, CoreOp.VarOp varOp, JavaOp.InvokeOp invokeOp) {
            var op = new HATLocalVarOp(
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

    public static class DeviceTypePhase extends HATDialectifyMemoryPhase {

        public DeviceTypePhase(LookupCarrier lookupCarrier) {
            super(lookupCarrier);
        }

        @Override
        protected boolean isIfaceBufferInvokeWithName(JavaOp.InvokeOp invokeOp){
            return OpTk.isIfaceBufferInvokeOpWithName(lookupCarrier.lookup(),invokeOp, n->n.equals(HATLocalVarOp.INTRINSIC_NAME))
               || (OpTk.isMethod(invokeOp, n->n.equals(HATLocalVarOp.INTRINSIC_NAME))
                    && invokeOp.resultType() instanceof JavaType javaType &&
                    OpTk.isAssignable(lookupCarrier.lookup(),javaType,DeviceType.class));
        }

         private boolean isDeviceTypeReservedMethod(JavaOp.InvokeOp invokeOp){
            return reservedMethods.contains(invokeOp.invokeDescriptor().name());
        }

        private boolean meetConditionsForMemoryLoadOp(JavaOp.InvokeOp invokeOp) {
            return OpTk.isInvokeDescriptorSubtypeOf(lookupCarrier.lookup(),invokeOp, DeviceType.class)
                    && (invokeOp.resultType() != JavaType.VOID)
                    && (!(invokeOp.resultType() instanceof PrimitiveType))
                    && (!isDeviceTypeReservedMethod(invokeOp));
        }

        @Override
        public CoreOp.FuncOp apply(CoreOp.FuncOp funcOp) {
            var here = OpTk.CallSite.of(PrivateMemoryPhase.class, "HATDialectifyMemoryPhase - memoryLoadOp");
            before(here, funcOp);
            Map<CoreOp.VarOp, JavaOp.InvokeOp> varTable = new HashMap<>();
            Stream<CodeElement<?, ?>> memoryLoadOps = funcOp.elements()
                    .mapMulti((codeElement, consumer) -> {
                        if (codeElement instanceof JavaOp.InvokeOp invokeOp) {
                            if (meetConditionsForMemoryLoadOp(invokeOp)) {
                                Op.Result result = invokeOp.result();
                                Set<Op.Result> uses = result.uses();
                                for (Op.Result use : uses) {
                                    if (use.op() instanceof CoreOp.VarOp varOp) {
                                        varTable.put(varOp, invokeOp);
                                        consumer.accept(invokeOp);
                                        consumer.accept(varOp);
                                    }
                                }
                            }
                        }
                    });

            Set<CodeElement<?, ?>> nodesInvolved = memoryLoadOps.collect(Collectors.toSet());
            funcOp = OpTk.transform(here, funcOp, (blockBuilder, op) -> {
                if (!nodesInvolved.contains(op)) {
                    blockBuilder.op(op);
                } else if (op instanceof JavaOp.InvokeOp invokeOp) {
                    insertHatMemoryLoadOp(blockBuilder, invokeOp);
                } else if (op instanceof CoreOp.VarOp varOp) {
                    JavaOp.InvokeOp invokeOp = varTable.get(varOp);
                    factory(blockBuilder, varOp, invokeOp);
                }

                return blockBuilder;
            });
            after(here, funcOp);
            return funcOp;
        }

        private void insertHatMemoryLoadOp(Block.Builder blockBuilder, JavaOp.InvokeOp invokeOp) {
            HATMemoryLoadOp loadOp = new HATMemoryLoadOp(invokeOp.resultType(),
                    invokeOp.invokeDescriptor().refType(),
                    invokeOp.invokeDescriptor().name(),
                    blockBuilder.context().getValues(invokeOp.operands()));
            Op.Result resultLoad = blockBuilder.op(loadOp);
            loadOp.setLocation(invokeOp.location());
            blockBuilder.context().mapValue(invokeOp.result(), resultLoad);
        }

        @Override
        protected HATMemoryVarOp factory(Block.Builder blockBuilder, CoreOp.VarOp varOp, JavaOp.InvokeOp invokeOp) {
            HATPrivateInitVarOp privateVarOp = new HATPrivateInitVarOp(varOp.varName(),
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
