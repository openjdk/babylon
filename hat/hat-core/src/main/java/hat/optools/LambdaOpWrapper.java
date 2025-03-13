/*
 * Copyright (c) 2024, Oracle and/or its affiliates. All rights reserved.
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
package hat.optools;

import hat.util.Result;

import java.lang.invoke.MethodHandles;
import java.lang.reflect.Method;
import jdk.incubator.code.Op;
import jdk.incubator.code.Quoted;
import jdk.incubator.code.Value;
import jdk.incubator.code.op.CoreOp;
import jdk.incubator.code.type.MethodRef;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class LambdaOpWrapper extends OpWrapper<CoreOp.LambdaOp> {
    public LambdaOpWrapper( MethodHandles.Lookup lookup, CoreOp.LambdaOp op) {
        super(lookup,op);
    }

    public InvokeOpWrapper getInvoke(int index) {
        var result = new Result<CoreOp.InvokeOp>();
        selectOnlyBlockOfOnlyBody(blockWrapper ->
                result.of(blockWrapper.op(index))
        );
        return OpWrapper.wrap(lookup, result.get());
    }

    public List<Value> operands() {
        return op().operands();
    }

    public Method getQuotableTargetMethod() {
        return getQuotableTargetInvokeOpWrapper().method();
    }

    public InvokeOpWrapper getQuotableTargetInvokeOpWrapper() {
        return OpWrapper.wrap(lookup, op().body().entryBlock().ops().stream()
                .filter(op -> op instanceof CoreOp.InvokeOp)
                .map(op -> (CoreOp.InvokeOp) op)
                .findFirst().get());
    }

    public MethodRef getQuotableTargetMethodRef() {
        return getQuotableTargetInvokeOpWrapper().methodRef();
    }

    public Object[] getQuotableCapturedValues(Quoted quoted, Method method) {
        var block = op().body().entryBlock();
        var ops = block.ops();
        Object[] varLoadNames = ops.stream()
                .filter(op -> op instanceof CoreOp.VarAccessOp.VarLoadOp)
                .map(op -> (CoreOp.VarAccessOp.VarLoadOp) op)
                .map(varLoadOp -> (Op.Result) varLoadOp.operands().get(0))
                .map(varLoadOp -> (CoreOp.VarOp) varLoadOp.op())
                .map(varOp -> varOp.varName()).toArray();


        Map<String, Object> nameValueMap = new HashMap<>();

        quoted.capturedValues().forEach((k, v) -> {
            if (k instanceof Op.Result result) {
                if (result.op() instanceof CoreOp.VarOp varOp) {
                    nameValueMap.put(varOp.varName(), v);
                }
            }
        });
        Object[] args = new Object[method.getParameterCount()];
        if (args.length != varLoadNames.length) {
            throw new IllegalStateException("Why don't we have enough captures.!! ");
        }
        for (int i = 1; i < args.length; i++) {
            args[i] = nameValueMap.get(varLoadNames[i].toString());
            if (args[i] instanceof CoreOp.Var varbox) {
                args[i] = varbox.value();
            }
        }
        return args;
    }
}
