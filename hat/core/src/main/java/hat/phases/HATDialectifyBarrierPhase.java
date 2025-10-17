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
import hat.dialect.HATBarrierOp;
import hat.optools.OpTk;
import jdk.incubator.code.Block;
import jdk.incubator.code.CodeElement;
import jdk.incubator.code.CopyContext;
import jdk.incubator.code.Op;
import jdk.incubator.code.Value;
import jdk.incubator.code.dialect.core.CoreOp;
import jdk.incubator.code.dialect.java.JavaOp;

import java.util.List;
import java.util.Set;
import java.util.stream.Collectors;
import java.util.stream.Stream;

public class HATDialectifyBarrierPhase extends HATDialectAbstractPhase implements HATDialectifyPhase {

    public HATDialectifyBarrierPhase(Accelerator accelerator) {
        super(accelerator);
    }

    private boolean isMethodFromHatKernelContext(JavaOp.InvokeOp invokeOp) {
        String kernelContextCanonicalName = hat.KernelContext.class.getName();
        return invokeOp.invokeDescriptor().refType().toString().equals(kernelContextCanonicalName);
    }

    private boolean isMethod(JavaOp.InvokeOp invokeOp, String methodName) {
        return invokeOp.invokeDescriptor().name().equals(methodName);
    }

    private void createBarrierNodeOp(CopyContext context, JavaOp.InvokeOp invokeOp, Block.Builder blockBuilder) {
        List<Value> inputOperands = invokeOp.operands();
        List<Value> outputOperands = context.getValues(inputOperands);
        HATBarrierOp hatBarrierOp = new HATBarrierOp(outputOperands);
        Op.Result outputResult = blockBuilder.op(hatBarrierOp);
        Op.Result inputResult = invokeOp.result();
        hatBarrierOp.setLocation(invokeOp.location());
        context.mapValue(inputResult, outputResult);
    }

    @Override
    public CoreOp.FuncOp run(CoreOp.FuncOp funcOp) {
        if (Config.SHOW_COMPILATION_PHASES.isSet(accelerator.backend.config())) {
            System.out.println("[INFO] Code model before HatDialectifyBarrierPhase: " + funcOp.toText());
        }
        Stream<CodeElement<?, ?>> elements = funcOp
                .elements()
                .mapMulti((element, consumer) -> {
                    if (element instanceof JavaOp.InvokeOp invokeOp) {
                        if (isMethodFromHatKernelContext(invokeOp) && isMethod(invokeOp, HATBarrierOp.INTRINSIC_NAME)) {
                            consumer.accept(invokeOp);
                        }
                    }
                });
        Set<CodeElement<?, ?>> collect = elements.collect(Collectors.toSet());
        if (collect.isEmpty()) {
            // Return the function with no modifications
            return funcOp;
        }
        var here = OpTk.CallSite.of(HATDialectifyBarrierPhase.class, "run");
        funcOp = OpTk.transform(here, funcOp, (blockBuilder, op) -> {
            CopyContext context = blockBuilder.context();
            if (!collect.contains(op)) {
                blockBuilder.op(op);
            } else if (op instanceof JavaOp.InvokeOp invokeOp) {
                createBarrierNodeOp(context, invokeOp, blockBuilder);
            }
            return blockBuilder;
        });
        if (Config.SHOW_COMPILATION_PHASES.isSet(accelerator.backend.config())) {
            System.out.println("[INFO] Code model after HatDialectifyBarrierPhase: " + funcOp.toText());
        }
        return funcOp;
    }

}
