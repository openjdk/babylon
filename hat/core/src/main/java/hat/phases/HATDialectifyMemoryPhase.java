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
import hat.Config;
import hat.dialect.HATLocalVarOp;
import hat.dialect.HATMemoryOp;
import hat.dialect.HATPrivateVarOp;
import hat.optools.OpTk;
import jdk.incubator.code.CodeElement;
import jdk.incubator.code.CopyContext;
import jdk.incubator.code.Op;
import jdk.incubator.code.Value;
import jdk.incubator.code.dialect.core.CoreOp;
import jdk.incubator.code.dialect.java.ClassType;
import jdk.incubator.code.dialect.java.JavaOp;

import java.util.List;
import java.util.Set;
import java.util.stream.Collectors;
import java.util.stream.Stream;

public class HATDialectifyMemoryPhase extends HATDialectAbstractPhase implements HATDialectifyPhase {

    public enum Space {
        PRIVATE,
        SHARED,
    }

    private final Space memorySpace;

    public HATDialectifyMemoryPhase(Accelerator accelerator, Space space) {
        super(accelerator);
        this.memorySpace = space;
    }

    private boolean isMethod(JavaOp.InvokeOp invokeOp, String methodName) {
        return invokeOp.invokeDescriptor().name().equals(methodName);
    }

    @Override
    public CoreOp.FuncOp run(CoreOp.FuncOp funcOp) {
            String nameNode = switch (memorySpace) {
                case PRIVATE -> HATPrivateVarOp.INTRINSIC_NAME;
                case SHARED -> HATLocalVarOp.INTRINSIC_NAME;
            };

            if (Config.SHOW_COMPILATION_PHASES.isSet(accelerator.backend.config())) {
                IO.println("[INFO] Code model before HatDialectifyMemoryPhase: " + funcOp.toText());
            }
            Stream<CodeElement<?, ?>> elements = funcOp.elements()
                    .mapMulti((codeElement, consumer) -> {
                        if (codeElement instanceof CoreOp.VarOp varOp) {
                            List<Value> inputOperandsVarOp = varOp.operands();
                            for (Value inputOperand : inputOperandsVarOp) {
                                if (inputOperand instanceof Op.Result result) {
                                    if (result.op() instanceof JavaOp.InvokeOp invokeOp) {
                                        if (OpTk.isIfaceBufferMethod(accelerator.lookup, invokeOp) && isMethod(invokeOp, nameNode)) {
                                            // It is the node we are looking for
                                            consumer.accept(invokeOp);
                                            consumer.accept(varOp);
                                        }
                                    }
                                }
                            }
                        }
                    });

            Set<CodeElement<?, ?>> nodesInvolved = elements.collect(Collectors.toSet());
            if (nodesInvolved.isEmpty()) {
                // No memory nodes involved
                return funcOp;
            }

        var here = OpTk.CallSite.of(HATDialectifyMemoryPhase.class, "run");
        funcOp = OpTk.transform(here, funcOp,(blockBuilder, op) -> {
                CopyContext context = blockBuilder.context();
                if (!nodesInvolved.contains(op)) {
                    blockBuilder.op(op);
                } else if (op instanceof JavaOp.InvokeOp invokeOp) {
                    // Don't insert the invoke node
                    Op.Result result = invokeOp.result();
                    List<Op.Result> collect = result.uses().stream().toList();
                    for (Op.Result r : collect) {
                        if (r.op() instanceof CoreOp.VarOp varOp) {
                            // That's the node we want
                            List<Value> inputOperandsVarOp = invokeOp.operands();
                            List<Value> outputOperandsVarOp = context.getValues(inputOperandsVarOp);
                            HATMemoryOp memoryOp = switch (memorySpace) {
                                case SHARED ->
                                        new HATLocalVarOp(varOp.varName(), (ClassType) varOp.varValueType(), varOp.resultType(), invokeOp.resultType(), outputOperandsVarOp);
                                default ->
                                        new HATPrivateVarOp(varOp.varName(), (ClassType) varOp.varValueType(), varOp.resultType(), invokeOp.resultType(), outputOperandsVarOp);
                            };

                            Op.Result hatLocalResult = blockBuilder.op(memoryOp);

                            // update location
                            memoryOp.setLocation(varOp.location());

                            context.mapValue(invokeOp.result(), hatLocalResult);
                        }
                    }
                } else if (op instanceof CoreOp.VarOp varOp) {
                    // pass value
                    context.mapValue(varOp.result(), context.getValue(varOp.operands().getFirst()));
                }
                return blockBuilder;
            });
            if (Config.SHOW_COMPILATION_PHASES.isSet(accelerator.backend.config())) {
                IO.println("[INFO] Code model after HatDialectifyMemoryPhase: " + funcOp.toText());
            }
            return funcOp;
    }
}
