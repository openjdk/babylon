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
import hat.dialect.HATMemoryLoadOp;
import hat.dialect.HATMemoryOp;
import hat.dialect.HATPhaseUtils;
import hat.dialect.HATPrivateVarInitOp;
import hat.dialect.HATPrivateVarOp;
import hat.optools.OpTk;
import jdk.incubator.code.Block;
import jdk.incubator.code.CodeElement;
import jdk.incubator.code.CopyContext;
import jdk.incubator.code.Op;
import jdk.incubator.code.Value;
import jdk.incubator.code.dialect.core.CoreOp;
import jdk.incubator.code.dialect.java.ClassType;
import jdk.incubator.code.dialect.java.JavaOp;
import jdk.incubator.code.dialect.java.JavaType;

import java.util.HashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.stream.Collectors;
import java.util.stream.Stream;

import static hat.dialect.HATPhaseUtils.isDeviceTypeInvokeDescriptor;

public abstract class HATDialectifyMemoryPhase implements HATDialect {

    protected final Accelerator accelerator;

    @Override
    public Accelerator accelerator(){
        return this.accelerator;
    }

    protected abstract HATMemoryOp factory(Block.Builder builder, CoreOp.VarOp varOp, JavaOp.InvokeOp invokeOp);

    protected abstract boolean isIfaceBufferInvokeWithName(JavaOp.InvokeOp invokeOp);

    public HATDialectifyMemoryPhase(Accelerator accelerator) {
        this.accelerator = accelerator;
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
                        .filter(r->r.op() instanceof CoreOp.VarOp)
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
        });
        after(here,funcOp );
        return funcOp;
    }


    public static class PrivateMemoryPhase extends HATDialectifyMemoryPhase {
        public PrivateMemoryPhase(Accelerator accelerator) {
            super(accelerator);
        }

        @Override
        protected boolean isIfaceBufferInvokeWithName(JavaOp.InvokeOp invokeOp){
            if (isIfaceBufferInvokeWithName(invokeOp, HATPrivateVarOp.INTRINSIC_NAME)) {
                return true;
            } else {
                return isMethod(invokeOp, HATPrivateVarOp.INTRINSIC_NAME) && HATPhaseUtils.isDeviceType(invokeOp);
            }
        }

        @Override
        protected HATMemoryOp factory(Block.Builder builder, CoreOp.VarOp varOp, JavaOp.InvokeOp invokeOp) {
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

        public LocalMemoryPhase(Accelerator accelerator) {
            super(accelerator);
        }

        @Override
        protected boolean isIfaceBufferInvokeWithName(JavaOp.InvokeOp invokeOp){
            if (isIfaceBufferInvokeWithName(invokeOp, HATLocalVarOp.INTRINSIC_NAME)) {
                return true;
            } else {
                return (isMethod(invokeOp, HATLocalVarOp.INTRINSIC_NAME) &&  HATPhaseUtils.isDeviceType(invokeOp));
            }
        }

        @Override
        protected HATMemoryOp factory(Block.Builder builder, CoreOp.VarOp varOp, JavaOp.InvokeOp invokeOp) {
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

        public DeviceTypePhase(Accelerator accelerator) {
            super(accelerator);
        }

        @Override
        protected boolean isIfaceBufferInvokeWithName(JavaOp.InvokeOp invokeOp){
            if (isIfaceBufferInvokeWithName(invokeOp, HATLocalVarOp.INTRINSIC_NAME)) {
                return true;
            } else {
                return (isMethod(invokeOp, HATLocalVarOp.INTRINSIC_NAME) &&  HATPhaseUtils.isDeviceType(invokeOp));
            }
        }

        @Override
        public CoreOp.FuncOp apply(CoreOp.FuncOp funcOp) {
            var here = OpTk.CallSite.of(PrivateMemoryPhase.class, "HATDialectifyMemoryPhase");
            before(here,funcOp);

            IO.println("BEFORE: " + funcOp.toText());
            Map<CoreOp.VarOp, JavaOp.InvokeOp> mapMe = new HashMap<>();

            Stream<CodeElement<?, ?>> memoryLoadOps = funcOp.elements()
                    .mapMulti((codeElement, consumer) -> {
                        if (codeElement instanceof JavaOp.InvokeOp invokeOp) {
                            if (isDeviceTypeInvokeDescriptor(invokeOp) && (invokeOp.resultType() != JavaType.VOID)) {
                                Op.Result result = invokeOp.result();
                                Set<Op.Result> uses = result.uses();
                                for (Op.Result use : uses) {
                                    if (use.op() instanceof CoreOp.VarOp varOp) {
                                        mapMe.put(varOp, invokeOp);
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
                    List<Value> operands = blockBuilder.context().getValues(invokeOp.operands());
                    HATMemoryLoadOp loadOp = new HATMemoryLoadOp("",  invokeOp.resultType(), invokeOp.invokeDescriptor().refType(), invokeOp.invokeDescriptor().name(), operands);
                    Op.Result resultLoad = blockBuilder.op(loadOp);
                    loadOp.setLocation(invokeOp.location());
                    blockBuilder.context().mapValue(invokeOp.result(), resultLoad);
                } else if (op instanceof CoreOp.VarOp varOp) {
                    // Pass the value
                    JavaOp.InvokeOp invokeOp = mapMe.get(varOp);
                    // FIXME: Here we should have a new Op: private with Initialization we know we
                    // can generate the equals expression.
                    HATPrivateVarInitOp privateVarOp = new HATPrivateVarInitOp(varOp.varName(),
                            (ClassType) varOp.varValueType(),
                            varOp.resultType(),
                            invokeOp.invokeDescriptor().refType(),
                            blockBuilder.context().getValues(varOp.operands()));
                    Op.Result op1 = blockBuilder.op(privateVarOp);
                    privateVarOp.setLocation(varOp.location());
                    blockBuilder.context().mapValue(varOp.result(), op1);
                    //context.mapValue(varLoadOp.result(), context.getValue(varLoadOp.operands().getFirst()));
                }
                return blockBuilder;
            });

            after(here, funcOp);

            IO.println("AFTER: " + funcOp.toText());
            after(here,funcOp );
            return funcOp;
        }

        @Override
        protected HATMemoryOp factory(Block.Builder builder, CoreOp.VarOp varOp, JavaOp.InvokeOp invokeOp) {
            throw new RuntimeException("Not implemented");
        }
    }
}
