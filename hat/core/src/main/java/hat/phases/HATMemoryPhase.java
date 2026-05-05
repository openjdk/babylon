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
import jdk.incubator.code.dialect.java.ClassType;
import jdk.incubator.code.dialect.java.JavaOp;
import optkl.IfaceValue;
import optkl.OpHelper;
import optkl.Trxfmr;
import optkl.codebuilders.BabylonOpDispatcher;
import optkl.codebuilders.BabylonOpDispatcher.HATOpAttribute;
import optkl.util.Regex;

import java.lang.invoke.MethodHandles;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashSet;
import java.util.Map;
import java.util.Set;

import static optkl.OpHelper.Invoke;
import static optkl.OpHelper.Invoke.invoke;
import static optkl.OpHelper.copyLocation;
import static optkl.codebuilders.BabylonOpDispatcher.table;

public abstract sealed class HATMemoryPhase implements HATPhase {
    protected abstract Op create(Block.Builder builder, CoreOp.VarOp varOp, JavaOp.InvokeOp invokeOp);

    protected abstract boolean isIfaceBufferInvokeWithName(Invoke invoke);

    protected String functionName;

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
                invoke.op().result().uses().stream()
                        .filter(result->result.op() instanceof CoreOp.VarOp)
                        .map(r->(CoreOp.VarOp)r.op())
                        .forEach(varOp->
                            blockBuilder.context().mapValue(invoke.op().result(), blockBuilder.op(create(blockBuilder, varOp, invoke.op())))
                        );
            } else if (OpHelper.Named.Variable.var(lookup,op) instanceof OpHelper.Named.Variable variable && nodesInvolved.contains(variable.op())) {
                Op.Result result = variable.op().result();
                blockBuilder.context().mapValue(result, blockBuilder.context().getValue(variable.op().operands().getFirst()));
                if (BabylonOpDispatcher.table.containsKey(functionName)) {
                    BabylonOpDispatcher.table.get(functionName).put(result.op(), BabylonOpDispatcher.table.get(functionName).get(variable.op()));
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
        protected HATMemoryVarOp create(Block.Builder builder, CoreOp.VarOp varOp, JavaOp.InvokeOp invokeOp) {
            var op = new HATMemoryVarOp.HATVarOp(
                    varOp.varName(),
                    (ClassType) varOp.varValueType(),
                    varOp.resultType(),
                    HATOpAttribute.PRIVATE,
                    builder.context().getValues(invokeOp.operands())
            );
            op.setLocation(varOp.location());
            return op;
        }
    }

    public static final  class LocalMemoryPhase extends HATMemoryPhase {
        public static final String INTRINSIC_NAME = "createLocal";
        @Override
        protected boolean isIfaceBufferInvokeWithName(Invoke invoke){
            return invoke.refIs(IfaceValue.class ) && invoke.named(INTRINSIC_NAME);

        }

        @Override
        protected HATMemoryVarOp create(Block.Builder builder, CoreOp.VarOp varOp, JavaOp.InvokeOp invokeOp) {
            return  copyLocation(varOp,new HATMemoryVarOp.HATVarOp(
                    varOp.varName(),
                    (ClassType) varOp.varValueType(),
                    varOp.resultType(),
                    HATOpAttribute.SHARED,
                    builder.context().getValues(invokeOp.operands())
            ));
        }
    }

    public static final class DeviceTypePhase extends HATMemoryPhase {
        public static final String INTRINSIC_NAME = "createLocal";
        @Override
        protected boolean isIfaceBufferInvokeWithName(Invoke invoke){
            return invoke.refIs( IfaceValue.class/*DeviceType.class, MappableIface.class, HAType.class*/) && invoke.named(INTRINSIC_NAME);
        }

        private static Regex reservedMethods = Regex.of("(createLocal|createPrivate|create|float2View|float4View)");
        @Override
        public CoreOp.FuncOp transform(MethodHandles.Lookup lookup,CoreOp.FuncOp funcOp) {
            this.functionName = funcOp.funcName();
            Map<CoreOp.VarOp, JavaOp.InvokeOp> varTable = new HashMap<>();
            Set<CodeElement<?, ?>> nodesInvolved = new HashSet<>();
            Invoke.stream(lookup,funcOp)
                    .filter(invoke->invoke.refIs(NonMappableIface.class) && invoke.returnsClassType() && !invoke.nameMatchesRegex(reservedMethods))
                    .forEach(invoke -> invoke.op().result().uses().stream()
                           .filter(use->use.op() instanceof CoreOp.VarOp)
                           .map(use->(CoreOp.VarOp)use.op())
                           .forEach(varOp -> {
                                varTable.put(varOp, invoke.op());
                                nodesInvolved.add(invoke.op());
                                nodesInvolved.add(varOp);
                           })
                    );

            return Trxfmr.of(lookup,funcOp).transform(nodesInvolved::contains, (blockBuilder, op) -> {
               if (invoke(lookup,op) instanceof Invoke invoke) {
                   blockBuilder.context().mapValue(invoke.op().result(),
                           blockBuilder.op(copyLocation(invoke.op(),
                                   new HATMemoryDefOp.HATMemoryLoadOp(invoke.returnType(),
                                           invoke.refType(),
                                           invoke.name(),
                                           blockBuilder.context().getValues(invoke.op().operands())))
                           )
                   );
                } else if (op instanceof CoreOp.VarOp varOp) {
                    create(blockBuilder, varOp, varTable.get(varOp));
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

//            var  privateVarOp = copyLocation(varOp,new HATMemoryVarOp.HATVarOp(varOp.varName(),
//                    (ClassType) varOp.varValueType(),
//                    varOp.resultType(),
//                    BabylonOpDispatcher.HATOpAttribute.INIT,
//                    blockBuilder.context().getValues(varOp.operands())));
//            blockBuilder.context().mapValue(varOp.result(), blockBuilder.op(privateVarOp));
//            return privateVarOp;
        }
    }
}
