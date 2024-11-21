/*
 * Copyright (c) 2024, Oracle and/or its affiliates. All rights reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * This code is free software; you can redistribute it and/or modify it
 * under the terms of the GNU General Public License version 2 only, as
 * published by the Free Software Foundation.
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

/*
 * @test
 * @modules jdk.incubator.code
 * @run testng TestTransitiveInvokeModule
 * @run testng/othervm -Dbabylon.ssa=cytron TestTransitiveInvokeModule
 */

import jdk.incubator.code.Op;
import org.testng.Assert;
import org.testng.annotations.Test;

import java.lang.invoke.MethodHandles;
import java.lang.reflect.Method;
import jdk.incubator.code.OpTransformer;
import jdk.incubator.code.analysis.SSA;
import jdk.incubator.code.interpreter.Interpreter;
import jdk.incubator.code.op.CoreOp;
import jdk.incubator.code.type.MethodRef;
import jdk.incubator.code.CodeReflection;
import java.util.*;
import java.util.stream.Stream;

public class TestTransitiveInvokeModule {

    @CodeReflection
    static void m(int i, List<Integer> l) {
        if (i < 0) {
            return;
        }

        n(i - 1, l);
    }

    @CodeReflection
    static void n(int i, List<Integer> l) {
        l.add(i);
        m(i - 1, l);
    }

    @Test
    public void test() {
        Optional<Method> om = Stream.of(TestTransitiveInvokeModule.class.getDeclaredMethods())
                .filter(m -> m.getName().equals("m"))
                .findFirst();

        CoreOp.ModuleOp module = createTransitiveInvokeModule(MethodHandles.lookup(), om.get());
        System.out.println(module.toText());
        module = module.transform(OpTransformer.LOWERING_TRANSFORMER);
        System.out.println(module.toText());
        module = SSA.transform(module);
        System.out.println(module.toText());

        module.functionTable().forEach((s, funcOp) -> {
            System.out.println(s + " -> " + funcOp);
        });

        List<Integer> r = new ArrayList<>();
        Interpreter.invoke(MethodHandles.lookup(), module.functionTable().firstEntry().getValue(), 10, r);
        Assert.assertEquals(r, List.of(9, 7, 5, 3, 1, -1));
    }

    static CoreOp.ModuleOp createTransitiveInvokeModule(MethodHandles.Lookup l,
                                                        Method m) {
        Optional<CoreOp.FuncOp> codeModel = Op.ofMethod(m);
        if (codeModel.isPresent()) {
            return createTransitiveInvokeModule(l, MethodRef.method(m), codeModel.get());
        } else {
            return CoreOp.module(List.of());
        }
    }

    static CoreOp.ModuleOp createTransitiveInvokeModule(MethodHandles.Lookup l,
                                                        MethodRef entryRef, CoreOp.FuncOp entry) {
        LinkedHashSet<MethodRef> funcsVisited = new LinkedHashSet<>();
        List<CoreOp.FuncOp> funcs = new ArrayList<>();

        record RefAndFunc(MethodRef r, CoreOp.FuncOp f) {
        }
        Deque<RefAndFunc> work = new ArrayDeque<>();
        work.push(new RefAndFunc(entryRef, entry));
        while (!work.isEmpty()) {
            RefAndFunc rf = work.pop();
            if (!funcsVisited.add(rf.r)) {
                continue;
            }

            CoreOp.FuncOp tf = rf.f.transform(rf.r.toString(), (block, op) -> {
                if (op instanceof CoreOp.InvokeOp iop) {
                    MethodRef r = iop.invokeDescriptor();
                    Method em = null;
                    try {
                        em = r.resolveToMethod(l, iop.invokeKind());
                    } catch (ReflectiveOperationException _) {
                    }
                    if (em instanceof Method m) {
                        Optional<CoreOp.FuncOp> f = Op.ofMethod(m);
                        if (f.isPresent()) {
                            RefAndFunc call = new RefAndFunc(r, f.get());
                            // Place model on work queue
                            work.push(call);

                            // Replace invocation with function call
                            block.op(CoreOp.funcCall(
                                    call.r.toString(),
                                    call.f.invokableType(),
                                    block.context().getValues(iop.operands())));
                            return block;
                        }
                    }
                }
                block.op(op);
                return block;
            });
            funcs.add(tf);
        }

        return CoreOp.module(funcs);
    }
}
