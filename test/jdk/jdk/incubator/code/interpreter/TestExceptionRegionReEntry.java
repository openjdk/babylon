/*
 * Copyright (c) 2026, Oracle and/or its affiliates. All rights reserved.
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

import java.lang.invoke.MethodHandles;
import jdk.incubator.code.dialect.java.MethodRef;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

import static jdk.incubator.code.dialect.core.CoreOp.*;
import static jdk.incubator.code.dialect.core.CoreType.*;
import static jdk.incubator.code.dialect.java.JavaOp.*;
import static jdk.incubator.code.dialect.java.JavaType.*;

/*
 * @test
 * @modules jdk.incubator.code
 * @library ../lib
 * @run junit TestExceptionRegionReEntry
 */
public class TestExceptionRegionReEntry {

    @Test
    void test() throws NoSuchMethodException {

        //func @"f" ()java.type:"boolean" -> {
        //    %0 : java.type:"boolean" = constant @false;
        //    %1 : Var<java.type:"boolean"> = var %0;
        //    exception.region.enter ^block_1 ^block_2;
        //
        //  ^block_1:
        //    %2 : java.type:"java.lang.RuntimeException" = new @java.ref:"java.lang.RuntimeException::()";
        //    throw %2;
        //
        //  ^block_2(%3 : java.type:"java.lang.RuntimeException"):
        //    %4 : java.type:"boolean" = var.load %1;
        //    %5 : java.type:"boolean" = constant @true;
        //    var.store %1 %5;
        //    cbranch %4 ^block_3 ^block_4;
        //
        //  ^block_3:
        //    return %4;
        //
        //  ^block_4:
        //    exception.region.enter ^block_1 ^block_2;
        //};
        FuncOp funcOp = func("f", functionType(BOOLEAN)).body(body -> {
            var entry = body.entryBlock();
            var tryBlock = entry.block();
            var catchBlock = entry.block(type(RuntimeException.class));
            var retry = entry.block();
            var done = entry.block();
            var v = entry.add(var(entry.add(constant(BOOLEAN, false))));
            entry.add(exceptionRegionEnter(tryBlock.reference(), catchBlock.reference()));
            tryBlock.add(throw_(tryBlock.add(new_(MethodRef.constructor(RuntimeException.class)))));
            var seen = catchBlock.add(varLoad(v));
            var t = catchBlock.add(constant(BOOLEAN, true));
            catchBlock.add(varStore(v, t));
            catchBlock.add(conditionalBranch(seen, done.reference(), retry.reference()));
            retry.add(exceptionRegionEnter(tryBlock.reference(), catchBlock.reference()));
            done.add(return_(seen));
        });

        System.out.println(funcOp.toText());

        Assertions.assertEquals(true, (boolean) Interpreter.invoke(MethodHandles.lookup(), funcOp));
    }
}
