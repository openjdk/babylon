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
import hat.buffer.S32Array;
import jdk.incubator.code.Reflect;
import jdk.incubator.code.bytecode.BytecodeGenerator;
import jdk.incubator.code.dialect.core.CoreOp;
import jdk.incubator.code.dialect.java.JavaOp;
import jdk.incubator.code.dialect.java.JavaType;
import jdk.incubator.code.dialect.java.MethodRef;
import optkl.Trxfmr;

import java.lang.invoke.MethodHandles;

import static optkl.OpHelper.Named.NamedStaticOrInstance.Invoke;
import static optkl.OpHelper.Named.NamedStaticOrInstance.Invoke.invoke;


public class AddArbitraryBlock {

    @Reflect
    static void hackMe(ComputeContext cc, S32Array s32Array){
        System.out.println("Ignore this");
        System.out.printf("        But wrap this\n");
        System.out.println("Also ignore");
    }

    public static void main(String[] args) throws Throwable {
        var hackMeMethodRef= MethodRef.method(AddArbitraryBlock.class, "hackMe", void.class, ComputeContext.class, S32Array.class);
        var lookup = MethodHandles.lookup();
        var hackMeFuncOp = CoreOp.FuncOp.ofMethod(hackMeMethodRef.resolveToMethod(lookup)).get();
        MethodRef Println = MethodRef.method(IO.class, "println", void.class, Object.class);
        Trxfmr.of(lookup,hackMeFuncOp)
                .toJava("Before injecting")
                .transform("withNewBlock", ce -> invoke(lookup,ce) instanceof Invoke $ && $.named("printf"), c -> {
                    var beforeString = c.builder().op(CoreOp.constant(JavaType.J_L_STRING, "Before ...."));
                    var afterString = c.builder().op(CoreOp.constant(JavaType.J_L_STRING, "After ...."));
                    c.add(JavaOp.if_(c.builder().parentBody()).if_(b -> {
                        b.op(CoreOp.core_yield(b.op(CoreOp.constant(JavaType.BOOLEAN, true))));
                    }).then(b -> {
                        b.op(JavaOp.invoke( JavaType.VOID, Println, beforeString));
                        b.op(CoreOp.core_yield());
                    }).else_(e->
                            e.op(CoreOp.core_yield()))
                    );
                    c.retain();
                    c.add(JavaOp.if_(c.builder().parentBody()).if_(b -> {
                        b.op(CoreOp.core_yield(b.op(CoreOp.constant(JavaType.BOOLEAN, true))));
                    }).then(b -> {
                        b.op(JavaOp.invoke( JavaType.VOID, Println, afterString));
                        b.op(CoreOp.core_yield());
                    }).else_(e->
                            e.op(CoreOp.core_yield()))
                    );
                })
                .toJava( "After injecting")
                .run(txfmr-> {
                    try {
                        BytecodeGenerator.generate(txfmr.lookup(), txfmr.funcOp()).invoke(null, null);
                    } catch (Throwable throwable) {
                        throw new RuntimeException(throwable);
                    }
                });

    }
}

