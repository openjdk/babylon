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

import hat.device.NonMappableIface;
import hat.dialect.HATMemoryDefOp;
import hat.dialect.HATMemoryVarOp;
import jdk.incubator.code.Block;
import jdk.incubator.code.CodeElement;
import jdk.incubator.code.Op;
import jdk.incubator.code.dialect.core.CoreOp;
import jdk.incubator.code.dialect.java.JavaOp;
import optkl.IfaceValue;
import optkl.OpHelper;
import optkl.Trxfmr;
import optkl.codebuilders.BabylonOpDispatcher.HATOpAttribute;
import optkl.util.Regex;

import java.lang.invoke.MethodHandles;
import java.util.HashSet;
import java.util.LinkedHashSet;
import java.util.Set;

import static optkl.OpHelper.Invoke;
import static optkl.OpHelper.Invoke.invoke;
import static optkl.OpHelper.copyLocation;
import static optkl.codebuilders.BabylonOpDispatcher.table;

public abstract sealed class HATMemoryPhase implements HATPhase {
    protected abstract Op create(Block.Builder builder, CoreOp.VarOp varOp, JavaOp.InvokeOp invokeOp);

    protected abstract boolean isIfaceBufferInvokeWithName(Invoke invoke);

    protected String functionName;

    protected abstract HATOpAttribute attribute();

    @Override
    public CoreOp.FuncOp transform(MethodHandles.Lookup lookup,CoreOp.FuncOp funcOp) {
        functionName = funcOp.funcName();
        Set<CodeElement<?,?>> nodesInvolved = new LinkedHashSet<>();
        Set<JavaOp.InvokeOp> mapMe = new LinkedHashSet<>();
        OpHelper.Variable.stream(lookup,funcOp)
                .forEach(variable -> variable.op().operands().stream()
                        .filter(operand -> operand instanceof Op.Result result
                                && invoke(lookup,result.op()) instanceof Invoke invoke
                                && isIfaceBufferInvokeWithName(invoke))
                        .map(r -> (JavaOp.InvokeOp) (((Op.Result) r).op()))
                        .findFirst().ifPresent(remove-> {
                            nodesInvolved.add(variable.op());
                            mapMe.add(remove);
                    })
                );

        return Trxfmr.of(lookup,funcOp).transform(ce->mapMe.contains(ce)||nodesInvolved.contains(ce), (blockBuilder, op) -> {
            if (invoke(lookup,op) instanceof Invoke invoke && mapMe.contains(invoke.op())) {
                // pass the invoke for reference. This is important for analysis later
                blockBuilder.op(invoke.op());
            } else if (OpHelper.Named.Variable.var(lookup,op) instanceof OpHelper.Named.Variable variable && nodesInvolved.contains(variable.op())) {
                Op.Result op1 = blockBuilder.op(variable.op());
                if (table.containsKey(functionName)) {
                    table.get(functionName).put(op1.op(), attribute());
                } else {
                    throw new RuntimeException("Function Name: " + functionName + " not present");
                }
            } else {
                blockBuilder.op(op);
            }
            return blockBuilder;
        }).funcOp();
    }


    public static final class PrivateMemoryPhase extends HATMemoryPhase {
        public static final String INTRINSIC_NAME = "createPrivate";
        @Override
        protected boolean isIfaceBufferInvokeWithName(Invoke invoke) {
            return invoke.refIs( IfaceValue.class /*DeviceType.class, MappableIface.class, HAType.class*/)
                    && invoke.named(INTRINSIC_NAME);

        }

        @Override
        protected HATOpAttribute attribute() {
            return HATOpAttribute.PRIVATE;
        }

        @Override
        protected HATMemoryVarOp create(Block.Builder builder, CoreOp.VarOp varOp, JavaOp.InvokeOp invokeOp) {
            // no need
            return null;
        }
    }

    public static final  class LocalMemoryPhase extends HATMemoryPhase {
        public static final String INTRINSIC_NAME = "createLocal";
        @Override
        protected boolean isIfaceBufferInvokeWithName(Invoke invoke){
            return invoke.refIs(IfaceValue.class ) && invoke.named(INTRINSIC_NAME);
        }

        @Override
        protected HATOpAttribute attribute() {
            return HATOpAttribute.SHARED;
        }

        @Override
        protected HATMemoryVarOp create(Block.Builder builder, CoreOp.VarOp varOp, JavaOp.InvokeOp invokeOp) {
            // no need
            return null;
        }
    }

    /**
     * This phase sets the corresponding loadOp for obtaining a value that was stored either in local memory or private memory.
     * Thus, we need to load from local/private into private.
     */
    public static final class DeviceTypePhase extends HATMemoryPhase {
        public static final String INTRINSIC_NAME = "createLocal";

        @Override
        protected boolean isIfaceBufferInvokeWithName(Invoke invoke) {
            return invoke.refIs(IfaceValue.class) && invoke.named(INTRINSIC_NAME);
        }

        @Override
        protected HATOpAttribute attribute() {
            return HATOpAttribute.INIT;
        }

        private static final Regex RESERVED_METHODS = Regex.of("(createLocal|createPrivate|create|float2View|float4View)");

        @Override
        public CoreOp.FuncOp transform(MethodHandles.Lookup lookup, CoreOp.FuncOp funcOp) {
            this.functionName = funcOp.funcName();
            Set<CodeElement<?, ?>> nodesInvolved = new HashSet<>();
            Invoke.stream(lookup, funcOp)
                    .filter(invoke -> invoke.refIs(NonMappableIface.class) && invoke.returnsClassType() && !invoke.nameMatchesRegex(RESERVED_METHODS))
                    .forEach(invoke -> invoke.op().result().uses().stream()
                            .filter(use -> use.op() instanceof CoreOp.VarOp)
                            .map(use -> (CoreOp.VarOp) use.op())
                            .forEach(varOp -> {
                                nodesInvolved.add(invoke.op());
                                nodesInvolved.add(varOp);
                            })
                    );

            return Trxfmr.of(lookup, funcOp).transform(nodesInvolved::contains, (blockBuilder, op) -> {
                if (invoke(lookup, op) instanceof Invoke invoke) {
                    blockBuilder.context().mapValue(invoke.op().result(),
                            blockBuilder.op(copyLocation(invoke.op(),
                                    new HATMemoryDefOp.HATMemoryLoadOp(invoke.returnType(),
                                            invoke.refType(),
                                            invoke.name(),
                                            blockBuilder.context().getValues(invoke.op().operands())))
                            )
                    );
                } else if (op instanceof CoreOp.VarOp varOp) {
                    create(blockBuilder, varOp, null);
                }
                return blockBuilder;
            }).funcOp();
        }

        @Override
        protected Op create(Block.Builder blockBuilder, CoreOp.VarOp varOp, JavaOp.InvokeOp invokeOp) {
            Op.Result opResult = blockBuilder.op(varOp);
            if (table.containsKey(functionName)) {
                table.get(functionName).put(opResult.op(), HATOpAttribute.INIT);
            } else {
                throw new RuntimeException("Function Name: " + functionName + " not present");
            }
            return opResult.op();
        }
    }
}
