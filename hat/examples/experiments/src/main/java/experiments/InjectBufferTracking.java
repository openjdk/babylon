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

import static optkl.OpHelper.Func.func;
import static optkl.OpHelper.Invoke;
import static optkl.OpHelper.Invoke.invoke;
import static optkl.OpHelper.Statement.createOpToStatementSpanMap;

public class InjectBufferTracking {


    static class IfaceBufferAccessStatementSpan implements OpHelper.OpSpan {
        enum Acc{NONE,ACCESSES,MUTATES;

            public boolean accessesOrMutates() {
                return equals(ACCESSES)|equals(MUTATES);
            }
        }
        private final List<Op> ops;
        private final Set<Value> mutates = new HashSet<>();
        private final Set<Value> accesses = new HashSet<>();
        private  Value ifaceBuffer;

        IfaceBufferAccessStatementSpan(List<Op> ops){
            this.ops = ops;
        }
        @Override public  List<Op> ops(){
            return ops;
        }

        void put(Value value, boolean mutate) {
            if (mutate) {
                mutates.add(value);
            } else {
                accesses.add(value);
            }
        }
        boolean mutates(Value value){
            return IfaceBufferAccessStatementSpan.this.mutates.contains(value);
        }
        boolean accesses(Value value){
            return IfaceBufferAccessStatementSpan.this.accesses.contains(value);
        }
        boolean accessesOrMutates(Value value){
            return mutates(value)|accesses(value);
        }
        boolean accessesAndMutates(Value value){
            return mutates(value)&accesses(value);
        }
    }


    @Reflect
    public static void inc(@RO KernelContext kc, @RW S32Array s32Array, int len) {
        if (kc.gix < kc.gsx) {
            s32Array.array(kc.gix, s32Array.array(kc.gix) + 1);
        }
    }

    @Reflect
    public static void add(ComputeContext cc, @RW S32Array s32Array, int len, int n) {
        int l = 40 * s32Array.length() + 25;
        int[] arr  = new int[2 * s32Array.length()];
        System.out.println("l = " + l);
        s32Array.array(0, s32Array.array(0) + 1);
        KernelContext kcNull = null;
        inc(kcNull,s32Array,20);
        for (int i = 0; i < n; i++) {
            cc.dispatchKernel(NDRange.of1D(len), kc -> inc(kc, s32Array, len));
            s32Array.array(0,1);
            System.out.println("Weird "+s32Array.array(0)+s32Array.length());
        }
    }
    public static void main(String[] args) throws NoSuchMethodException {
        var lookup = MethodHandles.lookup();
        var func = func(lookup, InjectBufferTracking.class, "add", ComputeContext.class, S32Array.class, int.class, int.class);

        // The resulting map, maps all ops to their enclosing statement (spans)
        // This will be useful later as we mark these spans with information regarding how the enclosing ops access ifacemapped buffers
        Map<Op, IfaceBufferAccessStatementSpan> opToStatementSpans = createOpToStatementSpanMap(func.op(),
                o->o instanceof JavaOp.InvokeOp, // only care if we span an invoke of some kind.
                IfaceBufferAccessStatementSpan::new);

        // This query is useful helps locating access of mappedIfaceBuffer accessors or mutators
       // var mappedIfaceBufferInvokeQuery = MappedIfaceBufferInvokeQuery.create(lookup);

        // We walk over each span and determine whether it has such an invoke.
        opToStatementSpans.values().forEach(statementSpan ->
            statementSpan.ops().forEach(opInStatement -> {
                if (invoke(lookup,opInStatement) instanceof Invoke.Virtual virtual
                        &&  (virtual.refIs(MappableIface.class) || virtual.returnsPrimitive())
                ) {
                    statementSpan.put(virtual.instance(), !virtual.returnsVoid());
                } else if (invoke(lookup, opInStatement) instanceof Invoke.Virtual invoke
                        && !invoke.refIs(ComputeContext.class) && invoke.operandCount() > 0) {
                    invoke.paramaterAccessList().stream()
                            .filter(typeAndAccess -> typeAndAccess.isIface(lookup))
                            .forEach(typeAndAccess -> statementSpan.put(typeAndAccess.value(), typeAndAccess.mutatesBuffer()));
                }
            })
        );

        MethodRef Println = MethodRef.method(IO.class, "println", void.class, Object.class);
        // We finally have have enough information  to transform.
        // We are looking at the edges of statements ,
        Trxfmr.of(lookup, func.op())
                // .toText("COMPUTE before injecting buffer tracking...")
                .toJava("COMPUTE (Java) before injecting buffer tracking...")
                .transform(ce -> ce instanceof Op op // only want ops the leading or trailing edge of the statement
                        && opToStatementSpans.containsKey(op) && opToStatementSpans.get(ce).firstOrLast(op),
                        c -> {
                    var statementSpan = opToStatementSpans.get(c.op());
                   // var computeContext = getFuncParamOrNull(c.op(), 0);
                   // Value mappedComputeContext = c.getValue(getFuncParamOrNull(c.op(), 0));
                    if (statementSpan.isFirst(c.op())) {
                        c.retain();
                    }
                    var acc = Mutable.of(IfaceBufferAccessStatementSpan.Acc.NONE);
                            statementSpan.ops.stream()
                                    .filter(o -> invoke(lookup, o) instanceof Invoke.Virtual virtual && virtual.refIs(MappableIface.class))
                                    .map(o -> (Invoke.Virtual) invoke(lookup, o))
                                    .forEach(invoke -> {
                                        acc.set(switch(acc.get()) {
                                            case NONE -> {
                                                if (statementSpan.mutates(invoke.instance())) {
                                                    yield IfaceBufferAccessStatementSpan.Acc.MUTATES;
                                                } else if (statementSpan.accesses(invoke.instance())) {
                                                    yield IfaceBufferAccessStatementSpan.Acc.ACCESSES;
                                                } else {
                                                    yield acc.get();
                                                }
                                            }
                                            case ACCESSES -> {
                                                if (statementSpan.mutates(invoke.instance())) {
                                                    yield IfaceBufferAccessStatementSpan.Acc.MUTATES;
                                                } else {
                                                    yield acc.get();
                                                }
                                            }
                                            default -> acc.get();
                                        });

                                    });
                            if (acc.get().accessesOrMutates()) {
                                var msg = (statementSpan.isFirst(c.op()) ? "Prev" : "Next") + " statement " +acc.get() + " iface mapped buffer ";
                                c.add(JavaOp.invoke(JavaType.VOID, Println, c.add(CoreOp.constant(JavaType.J_L_STRING, msg))));
                            }
                    if (statementSpan.isLast(c.op())){
                        c.retain();
                    }
                })
                //   .toText("COMPUTE after injecting buffer tracking...")
                .toJava("COMPUTE (java) after injecting buffer tracking...");
    }

}
