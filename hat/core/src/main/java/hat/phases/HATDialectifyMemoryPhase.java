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
import hat.dialect.HATMemoryOp;
import hat.dialect.HATPrivateVarOp;
import hat.optools.OpTk;
import jdk.incubator.code.Block;
import jdk.incubator.code.CodeElement;
import jdk.incubator.code.CopyContext;
import jdk.incubator.code.Op;
import jdk.incubator.code.dialect.core.CoreOp;
import jdk.incubator.code.dialect.java.ClassType;
import jdk.incubator.code.dialect.java.JavaOp;
import java.util.Set;
import java.util.stream.Collectors;
import java.util.stream.Stream;

public abstract class HATDialectifyMemoryPhase implements HATDialect {
    protected final Accelerator accelerator;
    @Override  public Accelerator accelerator(){
        return this.accelerator;
    }
    protected abstract HATMemoryOp factory(Block.Builder builder, CoreOp.VarOp varOp, JavaOp.InvokeOp invokeOp);

    protected abstract boolean isIfaceBufferInvokeWithName(JavaOp.InvokeOp invokeOp);
    public HATDialectifyMemoryPhase(Accelerator accelerator) {
        this.accelerator = accelerator;
    }

    @Override
    public CoreOp.FuncOp apply(CoreOp.FuncOp funcOp) {
        if (accelerator.backend.config().showCompilationPhases()) {
            IO.println("[INFO] Code model before HatDialectifyMemoryPhase: " + funcOp.toText());
        }
      //  record BufferIfaceInvokeOpAndVarOpPair(JavaOp.InvokeOp bufferIfaceOp, CoreOp.VarOp varOp){}
       // List<BufferIfaceInvokeOpAndVarOpPair> bufferIfaceInvokeOpAndVarOpPairList = new ArrayList<>();
        Stream<CodeElement<?, ?>> elements = funcOp.elements().filter(e -> e instanceof CoreOp.VarOp ).map(e-> (CoreOp.VarOp) e)
                .mapMulti((varOp, consumer) -> {
                                var bufferIfaceInvokeOp = varOp.operands().stream()
                                        .filter(o -> o instanceof Op.Result result && result.op() instanceof JavaOp.InvokeOp invokeOp && isIfaceBufferInvokeWithName(invokeOp))
                                        .map(r -> (JavaOp.InvokeOp) (((Op.Result) r).op()))
                                        .findFirst();
                                if (bufferIfaceInvokeOp.isPresent()) {
                                    consumer.accept(bufferIfaceInvokeOp.get());
                                    consumer.accept(varOp);
                                }
                        }
                );

        Set<CodeElement<?, ?>> nodesInvolved = elements.collect(Collectors.toSet());

        var here = OpTk.CallSite.of(PrivatePhase.class, "run");
        funcOp = OpTk.transform(here, funcOp, (blockBuilder, op) -> {
            CopyContext context = blockBuilder.context();
            if (!nodesInvolved.contains(op)) {
                blockBuilder.op(op);
            } else if (op instanceof JavaOp.InvokeOp invokeOp) {
                // Don't insert the invoke node we just want the results
                invokeOp.result().uses().stream()
                        .filter(r->r.op() instanceof CoreOp.VarOp)
                        .map(r->(CoreOp.VarOp)r.op())
                        .forEach(varOp->
                            context.mapValue(invokeOp.result(), blockBuilder.op(factory(blockBuilder,varOp,invokeOp)))
                        );
            } else if (op instanceof CoreOp.VarOp varOp) {
                // pass value
                context.mapValue(varOp.result(), context.getValue(varOp.operands().getFirst()));
            }
            return blockBuilder;
        });
        if (accelerator.backend.config().showCompilationPhases()) {
            IO.println("[INFO] Code model after HatDialectifyMemoryPhase: " + funcOp.toText());
        }
        return funcOp;
    }

    public static class PrivatePhase extends HATDialectifyMemoryPhase {
        public PrivatePhase(Accelerator accelerator) {
            super(accelerator);
        }
        @Override protected boolean isIfaceBufferInvokeWithName(JavaOp.InvokeOp invokeOp){
             return isIfaceBufferInvokeWithName(invokeOp, HATPrivateVarOp.INTRINSIC_NAME);
        }

        @Override protected HATMemoryOp factory(Block.Builder builder, CoreOp.VarOp varOp, JavaOp.InvokeOp invokeOp) {
            var op=  new HATPrivateVarOp(
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

    public static class SharedPhase extends HATDialectifyMemoryPhase {

        public SharedPhase(Accelerator accelerator) {
            super(accelerator);
        }
        @Override protected boolean isIfaceBufferInvokeWithName(JavaOp.InvokeOp invokeOp){
            return isIfaceBufferInvokeWithName(invokeOp, HATLocalVarOp.INTRINSIC_NAME);
        }

        @Override protected HATMemoryOp factory(Block.Builder builder, CoreOp.VarOp varOp, JavaOp.InvokeOp invokeOp) {
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
}
