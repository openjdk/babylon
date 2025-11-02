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
import hat.ifacemapper.MappableIface;
import hat.optools.OpTk;
import jdk.incubator.code.CodeElement;
import jdk.incubator.code.Op;
import jdk.incubator.code.TypeElement;
import jdk.incubator.code.dialect.core.CoreOp;
import jdk.incubator.code.dialect.java.JavaType;

import java.lang.invoke.MethodHandles;
import java.util.HashMap;
import java.util.Map;
import java.util.Set;
import java.util.function.Function;
import java.util.stream.Stream;

public class HATFinalDetectionPhase implements HATDialect {

    protected final Accelerator accelerator;
    @Override  public Accelerator accelerator(){
        return this.accelerator;
    }
    private final Map<Op.Result, CoreOp.VarOp> finalVars = new HashMap<>();

    public HATFinalDetectionPhase(Accelerator accelerator) {
       this.accelerator = accelerator;
    }

    @Override
    public CoreOp.FuncOp apply(CoreOp.FuncOp funcOp) {
        Stream<CodeElement<?, ?>> elements = funcOp.elements();
        elements.forEach(codeElement -> {
            if (codeElement instanceof CoreOp.VarOp varOp) {
                Op.Result varResult = varOp.result();
                Set<Op.Result> uses = varResult.uses();

                // Obtain if the varOp comes from a declaration of
                // a var with MappableIface type. If so, we can't
                // generate the constant, because at this point of the analysis
                // after the dialectify, the only accesses left are accesses
                // to global memory.
                TypeElement typeElement = varOp.resultType().valueType();
                boolean isMappableType = false;
                if (typeElement instanceof JavaType javaType) {
                    isMappableType = OpTk.isAssignable(MethodHandles.lookup(), javaType, MappableIface.class);
                }

                if (!isMappableType) {
                    boolean isFinalVarOp = true;
                    for (Op.Result use : uses) {
                        Op op = use.op();
                        switch (op) {
                            case CoreOp.VarAccessOp.VarStoreOp storeOp -> {
                                if (storeOp.operands().stream().anyMatch(operand -> operand.equals(varResult))) {
                                    isFinalVarOp = false;
                                }
                            }
                            case CoreOp.YieldOp yieldOp -> {
                                if (yieldOp.operands().stream().anyMatch(operand -> operand.equals(varResult))) {
                                    isFinalVarOp = false;
                                }
                            }
                            case null, default -> {
                            }
                        }
                    }
                    if (isFinalVarOp) {
                        finalVars.put(varResult, varOp);
                    }
                }
            }
        });
        return funcOp;
    }

    public Map<Op.Result, CoreOp.VarOp> getFinalVars() {
        return finalVars;
    }
}
