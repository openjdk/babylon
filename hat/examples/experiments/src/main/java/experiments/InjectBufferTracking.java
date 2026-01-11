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
package experiments;

import hat.ComputeContext;
import hat.KernelContext;
import hat.NDRange;
import hat.buffer.S32Array;
import jdk.incubator.code.Block;
import jdk.incubator.code.Op;
import jdk.incubator.code.Reflect;
import jdk.incubator.code.Value;
import jdk.incubator.code.dialect.core.CoreOp;
import jdk.incubator.code.dialect.java.JavaOp;
import optkl.OpHelper;
import optkl.Trxfmr;
import optkl.ifacemapper.AccessType;
import optkl.ifacemapper.MappableIface;
import optkl.ifacemapper.MappableIface.RO;
import optkl.ifacemapper.MappableIface.RW;

import java.lang.invoke.MethodHandles;
import java.util.List;

import static hat.ComputeContext.WRAPPER.ACCESS;
import static hat.ComputeContext.WRAPPER.MUTATE;
import static optkl.OpHelper.Named.NamedStaticOrInstance.Invoke.invoke;

public class InjectBufferTracking {

    @Reflect
    public static void inc(@RO KernelContext kc, @RW S32Array s32Array, int len) {
        if (kc.gix < kc.gsx) {
            s32Array.array(kc.gix, s32Array.array(kc.gix) + 1);
        }
    }

    @Reflect
    public static void add(ComputeContext cc, @RW S32Array s32Array, int len, int n) {

        int l = s32Array.length();

        System.out.println("l = "+l);

        for (int i = 0; i < n; i++) {
            cc.dispatchKernel(NDRange.of1D(len), kc -> inc(kc, s32Array, len));
            System.out.println(i);//s32Array.array(0));
        }
    }

    static Block.Parameter getFuncParamOrNull(Op op, int n){
        while (op != null && !(op instanceof CoreOp.FuncOp)) {
            System.out.println(op);
            op = op.ancestorOp();
        }
        if (op instanceof CoreOp.FuncOp funcOp) {
            return funcOp.bodies().get(0).entryBlock().parameters().get(n);
        }else{
            return null;
        }
    }

    static Block.Parameter getFuncParamOrThrow(Op op, int n){
        if (getFuncParamOrNull(op, n) instanceof Block.Parameter parameter){
            return parameter;
        }else {
            throw new IllegalStateException("cant find func parameter parameter "+n);
        }
    }


    public static void main(String[] args) throws NoSuchMethodException {
        var lookup = MethodHandles.lookup();
        var addMethod = Op.ofMethod(
                InjectBufferTracking.class.getDeclaredMethod("add", ComputeContext.class, S32Array.class, int.class, int.class)
        ).orElseThrow();
        Trxfmr.of(lookup, addMethod)
                .toText("COMPUTE before injecting buffer tracking...")
                .toJava("COMPUTE (Java) before injecting buffer tracking...")
                .transform(ce -> ce instanceof JavaOp.InvokeOp, c -> {
                    var invoke = invoke(lookup, c.op());
                    if (invoke.isMappableIface() && (invoke.returns(MappableIface.class) || invoke.returnsPrimitive())) {
                        Value computeContext =c.getValue(getFuncParamOrThrow(invoke.op(),0));
                        Value ifaceMappedBuffer = c.mappedOperand(0);
                        c.add(JavaOp.invoke(invoke.returnsVoid() ? MUTATE.pre : ACCESS.pre, computeContext, ifaceMappedBuffer));
                        c.retain();
                        c.add(JavaOp.invoke(invoke.returnsVoid() ? MUTATE.post : ACCESS.post, computeContext, ifaceMappedBuffer));
                    } else if (!invoke.refIs(ComputeContext.class) && invoke.operandCount() > 0) {
                        List<AccessType.TypeAndAccess> typeAndAccesses = invoke.paramaterAccessList();
                        Value computeContext =c.getValue(getFuncParamOrThrow(invoke.op(),0));// c.getValue(invoke.op().operands().getFirst());
                        typeAndAccesses.stream()
                                .filter(typeAndAccess -> typeAndAccess.isIface(lookup))
                                .forEach(typeAndAccess ->
                                        c.add(JavaOp.invoke(
                                                typeAndAccess.ro() ? ACCESS.pre : MUTATE.pre,
                                                computeContext, c.getValue(typeAndAccess.value()))
                                        )
                                );
                        c.retain();
                        typeAndAccesses.stream()
                                .filter(typeAndAccess -> OpHelper.isAssignable(lookup, typeAndAccess.javaType(), MappableIface.class))
                                .forEach(typeAndAccess ->
                                        c.add(JavaOp.invoke(
                                                typeAndAccess.ro() ? ACCESS.post : MUTATE.post,
                                                computeContext, c.getValue(typeAndAccess.value()))
                                        )
                                );
                    }
                })
                .toText("COMPUTE after injecting buffer tracking...")
                .toJava();
    }

}
