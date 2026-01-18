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
import jdk.incubator.code.Op;
import jdk.incubator.code.Reflect;
import jdk.incubator.code.Value;
import jdk.incubator.code.dialect.core.CoreOp;
import jdk.incubator.code.dialect.java.JavaOp;
import jdk.incubator.code.dialect.java.JavaType;
import jdk.incubator.code.dialect.java.MethodRef;
import optkl.MappedIfaceBufferInvokeQuery;
import optkl.MappedIfaceBufferInvokeQuery.Match;
import optkl.OpHelper;
import optkl.Trxfmr;
import optkl.ifacemapper.MappableIface;
import optkl.ifacemapper.MappableIface.RO;
import optkl.ifacemapper.MappableIface.RW;
import optkl.util.Mutable;

import java.lang.invoke.MethodHandles;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

import static optkl.OpHelper.Named.NamedStaticOrInstance.Invoke;
import static optkl.OpHelper.Named.NamedStaticOrInstance.Func.*;
import static optkl.OpHelper.Statement.createOpToStatementSpanMap;
import static optkl.OpHelper.getFuncParamOrNull;

public class InjectBufferTracking {

    @Reflect
    public static void inc(@RO KernelContext kc, @RW S32Array s32Array, int len) {
        if (kc.gix < kc.gsx) {
            s32Array.array(kc.gix, s32Array.array(kc.gix) + 1);
        }
    }

    @Reflect
    public static void add(ComputeContext cc, @RW S32Array s32Array, int len, int n) {
        int l = 40*s32Array.length()+25;
        System.out.println("l = " + l);
        s32Array.array(0,s32Array.array(0)+1);
        for (int i = 0; i < n; i++) {
            cc.dispatchKernel(NDRange.of1D(len), kc -> inc(kc, s32Array, len));
            System.out.println(i);
        }
    }


    public static void main(String[] args) throws NoSuchMethodException {
        var lookup = MethodHandles.lookup();
        var func = func(lookup, InjectBufferTracking.class,"add",ComputeContext.class, S32Array.class, int.class, int.class);

        record StatementSpanImpl(Set<Value> mutates, Set<Value> accesses, // Either Access or Mutate
                                 Mutable<Value> mutableIfaceBuffer,
                                 List<Op> ops) implements Statement.Span {
            void put(Value value, boolean mutate){
                if (mutate){
                    mutates.add(value);
                }else {
                    accesses.add(value);
                }
            }
        }

        // The resulting map, maps all ops to their enclosing statements but only if the statement contains an invokeOp.
        // Here we look for iface->set/get and add value -> access type mappings
        Map<Op,StatementSpanImpl> opToStatementSpans = createOpToStatementSpanMap(func.op(),
                op->op instanceof JavaOp.InvokeOp, // we only care if the statement actually contains an invoke
                ops-> new StatementSpanImpl(new HashSet<>(), new HashSet<>(), Mutable.of(null),ops)
        );

        // This query helps locate mappedIfaceBuffer accessors or mutators
        var mappedIfaceBufferInvokeQuery = MappedIfaceBufferInvokeQuery.create(lookup);

        opToStatementSpans.values().forEach(statementSpan -> {
            statementSpan.ops().forEach(opInStatement -> {
                if (mappedIfaceBufferInvokeQuery.matches(opInStatement) instanceof Match match) {
                    statementSpan.put(match.helper().instance(), match.mutatesBuffer());
                }else if (Invoke.invoke(lookup, opInStatement) instanceof Invoke invoke
                        && invoke.isInstance() && !invoke.refIs(ComputeContext.class) && invoke.operandCount() > 0) {
                    invoke.paramaterAccessList().stream()
                            .filter(typeAndAccess -> typeAndAccess.isIface(lookup))
                            .forEach(typeAndAccess -> statementSpan.put(typeAndAccess.value(), typeAndAccess.mutatesBuffer()));
                }
            });
        });

        // Now we have enough info to transform.  We now look like statement first and last ops and inject before or after ,
        MethodRef Println = MethodRef.method(IO.class, "println", void.class, Object.class);
        Trxfmr.of(lookup,func.op())
               // .toText("COMPUTE before injecting buffer tracking...")
                .toJava("COMPUTE (Java) before injecting buffer tracking...")
                .transform((block, op) ->{
                    if (opToStatementSpans.containsKey(op)) {
                        var statementSpan = opToStatementSpans.get(op);
                        if (statementSpan.firstOrLast(op)) {
                            var computeContext = getFuncParamOrNull(op, 0);
                            //Value mappedComputeContext = block.context().mapValue(getFuncParamOrNull(op, 0));
                            if (!OpHelper.isAssignable(lookup, computeContext.type(), ComputeContext.class)) {
                             //   System.out.println("ok we found the compute context");
                            //}else {
                                throw new RuntimeException("parameter 0 is not compute context "+computeContext.type());
                            }
                            if (statementSpan.isFrom(op)) {
                                statementSpan.ops.stream()
                                        .filter(o->o instanceof JavaOp.InvokeOp)
                                        .map(o->Invoke.invoke(lookup,o))
                                        .filter(Invoke::isInstance)
                                        .forEach(invoke -> {
                                            if (OpHelper.isAssignable(lookup,invoke.refType(),MappableIface.class)){
                                                boolean mutates = statementSpan.mutates.contains(invoke.op().operands().getFirst());
                                                boolean accesses  = statementSpan.accesses.contains(invoke.op().operands().getFirst());
                                               var before = block.op(CoreOp.constant(JavaType.J_L_STRING, "The following statement "
                                                       +(mutates?"mutates ":"")+ ((mutates&accesses)?"and ":"")+(accesses?"accesses ":"")+
                                                       "iface mapped buffer "));
                                               block.op(JavaOp.invoke( JavaType.VOID, Println, before));
                                            } else {
                                                // System.out.println("nope");
                                                }

                                    // var mappedIfaceBuffer =c.getValue(ifaceBuffer);
                                    // c.add(JavaOp.invoke(accessOrMutateWrapper.pre, mappedComputeContext,ifaceBuffer));
                                    //c.retain();
                                });
                                block.op(op);
                            } else if (statementSpan.isTo(op)) {
                                block.op(op);
                                statementSpan.ops.stream()
                                        .filter(o->o instanceof JavaOp.InvokeOp)
                                        .map(o->Invoke.invoke(lookup,o))
                                        .filter(Invoke::isInstance)
                                        .forEach(invoke -> {
                                            if (OpHelper.isAssignable(lookup,invoke.refType(),MappableIface.class)){
                                                boolean mutates = statementSpan.mutates.contains(invoke.op().operands().getFirst());
                                                boolean accesses  = statementSpan.accesses.contains(invoke.op().operands().getFirst());
                                                var after = block.op(CoreOp.constant(JavaType.J_L_STRING,
                                                        "The previous statement "
                                                                +(mutates?"mutates ":"")+ ((mutates&accesses)?"and ":"")+(accesses?"accesses ":"")+
                                                                "iface mapped buffer "));
                                                block.op(JavaOp.invoke( JavaType.VOID, Println, after));
                                            } else {
                                                //  System.out.println("nope");
                                            }
                                    // var mappedIfaceBuffer =c.getValue(ifaceBuffer);
                                    //  c.retain();
                                    // c.add(JavaOp.invoke(accessorMutateWrapper.post, mappedComputeContext,mappedIfaceBuffer));
                                });
                            }
                        }else {
                            block.op(op);
                        }
                    }else {
                        block.op(op);
                    }
                    return block;
                })

             //   .toText("COMPUTE after injecting buffer tracking...")
                .toJava("COMPUTE (java) after injecting buffer tracking...");
    }

}
