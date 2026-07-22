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

/*
 * @test
 * @modules jdk.incubator.code
 * @run junit TestSuccessorValidation
 */

import jdk.incubator.code.Block;
import jdk.incubator.code.dialect.core.CoreOp;
import jdk.incubator.code.dialect.core.CoreType;
import jdk.incubator.code.dialect.java.JavaType;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

public class TestSuccessorValidation {
    @Test
    void testInvalid() {
        Assertions.assertThrows(IllegalStateException.class, TestSuccessorValidation::numArgsGtNumParams);
    }

    private static CoreOp.FuncOp numArgsGtNumParams() {
        return CoreOp.func("invalid", CoreType.functionType(JavaType.INT, JavaType.INT)).body(eb -> {
            Block.Builder b1 = eb.block(JavaType.INT);
            b1.add(CoreOp.return_(b1.parameters().get(0)));

            eb.add(CoreOp.branch(b1.reference(eb.parameters().get(0), eb.parameters().get(0))));
        });
    }

    @Test
    void testValid() {
        numArgsLeqNumParams();
    }

    private static CoreOp.FuncOp numArgsLeqNumParams() {
        return CoreOp.func("valid", CoreType.functionType(JavaType.INT, JavaType.INT, JavaType.BOOLEAN))
                .body(eb -> {
                    Block.Builder b1 = eb.block(JavaType.INT);
                    b1.add(CoreOp.return_(b1.parameters().get(0)));

                    eb.add(CoreOp.conditionalBranch(
                            eb.parameters().get(1),
                            b1.reference(eb.parameters().get(0)),
                            b1.reference()
                    ));
                });
    }
}
