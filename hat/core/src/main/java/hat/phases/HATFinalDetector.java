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
import hat.types.BF16;
import hat.types.F16;
import optkl.OpHelper;
import optkl.ifacemapper.MappableIface;
import jdk.incubator.code.Op;
import jdk.incubator.code.dialect.core.CoreOp;

import java.util.HashMap;
import java.util.Map;

public record HATFinalDetector(KernelCallGraph kernelCallGraph){
    public Map<Op.Result, CoreOp.VarOp> applied(CoreOp.FuncOp funcOp) {
        final Map<Op.Result, CoreOp.VarOp> finalVars = new HashMap<>();
        OpHelper.Named.Variable.stream(kernelCallGraph.lookup(),funcOp)
                .filter(variable ->!variable.assignable(MappableIface.class, F16.class, BF16.class))
                .forEach(variable ->{
                    // At this point the varOp DOES NOT come from a declaration of a var with MappableIface type.
                    // For those we can't generate the constant, because at this point of the analysis
                    // the only accesses left are accesses to global memory.
                    Op.Result varResult = variable.op().result();
                    if (!varResult.uses().stream()
                            .map(use->use.op())
                            .anyMatch(op->
                                    (op instanceof CoreOp.VarAccessOp.VarStoreOp storeOp &&
                                            (storeOp.operands().stream().anyMatch(operand -> operand.equals(varResult))))
                                ||
                                    (op instanceof CoreOp.YieldOp yieldOp &&
                                            (yieldOp.operands().stream().anyMatch(operand -> operand.equals(varResult)))))){
                        finalVars.put(varResult, variable.op());
                    }
        });
        return finalVars;
    }
}
