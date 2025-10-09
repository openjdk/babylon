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

import hat.Config;
import hat.dialect.HatVectorLoadOp;
import hat.dialect.HatVectorViewOp;
import hat.dialect.HatVectorBinaryOp;
import hat.optools.OpTk;
import jdk.incubator.code.CodeElement;
import jdk.incubator.code.CopyContext;
import jdk.incubator.code.Op;
import jdk.incubator.code.TypeElement;
import jdk.incubator.code.Value;
import jdk.incubator.code.dialect.core.CoreOp;
import jdk.incubator.code.dialect.java.JavaOp;

import java.lang.invoke.MethodHandles;
import java.util.List;
import java.util.Set;
import java.util.stream.Collectors;
import java.util.stream.Stream;

public class HatDialectifyVectorOpPhase extends HatDialectAbstractPhase implements HatDialectifyPhase {

    MethodHandles.Lookup lookup;
    private final LoadView vectorOperation;

    public HatDialectifyVectorOpPhase(MethodHandles.Lookup lookup, LoadView vectorOperation, Config config) {
        super(config);
        this.lookup = lookup;
        this.vectorOperation = vectorOperation;
    }

    private boolean isMethod(JavaOp.InvokeOp invokeOp, String methodName) {
        return invokeOp.invokeDescriptor().name().equals(methodName);
    }

    public enum LoadView {
        FLOAT4_LOAD("float4View"),
        ADD("add");
        String methodName;
        LoadView(String methodName) {
            this.methodName = methodName;
        }
    }

    private boolean isVectorOperation(JavaOp.InvokeOp invokeOp) {
        TypeElement typeElement = invokeOp.resultType();
        boolean isHatVectorType = typeElement.toString().startsWith("hat.buffer.Float");
        return isHatVectorType
                && OpTk.isIfaceBufferMethod(lookup, invokeOp)
                && isMethod(invokeOp, vectorOperation.methodName);
    }

    @Override
    public CoreOp.FuncOp run(CoreOp.FuncOp funcOp) {
        if (Config.SHOW_COMPILATION_PHASES.isSet(config))
            IO.println("BEFORE Vector Types transform: " + funcOp.toText());
        Stream<CodeElement<?, ?>> float4NodesInvolved = funcOp.elements()
                .mapMulti((codeElement, consumer) -> {
                    if (codeElement instanceof CoreOp.VarOp varOp) {
                        List<Value> inputOperandsVarOp = varOp.operands();
                        for (Value inputOperand : inputOperandsVarOp) {
                            if (inputOperand instanceof Op.Result result) {
                                if (result.op() instanceof JavaOp.InvokeOp invokeOp) {
                                    if (isVectorOperation(invokeOp)) {
                                        consumer.accept(invokeOp);
                                        consumer.accept(varOp);
                                    }
                                }
                            }
                        }
                    }
                });

        Set<CodeElement<?, ?>> nodesInvolved = float4NodesInvolved.collect(Collectors.toSet());
        if (nodesInvolved.isEmpty()) {
            return funcOp;
        }

        funcOp = funcOp.transform((blockBuilder, op) -> {
            CopyContext context = blockBuilder.context();
            if (!nodesInvolved.contains(op)) {
                blockBuilder.op(op);
            } else if (op instanceof JavaOp.InvokeOp invokeOp) {
                // Don't insert the invoke node
                Op.Result result = invokeOp.result();
                List<Op.Result> collect = result.uses().stream().toList();
                for (Op.Result r : collect) {
                    if (r.op() instanceof CoreOp.VarOp varOp) {
                        List<Value> inputOperandsVarOp = invokeOp.operands();
                        List<Value> outputOperandsVarOp = context.getValues(inputOperandsVarOp);
                        HatVectorViewOp memoryViewOp = switch (vectorOperation) {
                            case FLOAT4_LOAD -> new HatVectorLoadOp(varOp.varName(), varOp.resultType(), invokeOp.resultType(), 4, outputOperandsVarOp);
                            case ADD ->  new HatVectorBinaryOp(varOp.varName(), varOp.resultType(), HatVectorBinaryOp.OpType.ADD, outputOperandsVarOp);
                        };
                        Op.Result hatLocalResult = blockBuilder.op(memoryViewOp);
                        memoryViewOp.setLocation(varOp.location());
                        context.mapValue(invokeOp.result(), hatLocalResult);
                    }
                }
            } else if (op instanceof CoreOp.VarOp varOp) {
                // pass value
                context.mapValue(varOp.result(), context.getValue(varOp.operands().getFirst()));
            } else if (op instanceof CoreOp.VarAccessOp.VarLoadOp varLoadOp) {
                // pass value
                context.mapValue(varLoadOp.result(), context.getValue(varLoadOp.operands().getFirst()));
            }
            return blockBuilder;
        });
        if (Config.SHOW_COMPILATION_PHASES.isSet(config))
            IO.println("After Vector Types Transform: " + funcOp.toText());
        return funcOp;
    }
}
