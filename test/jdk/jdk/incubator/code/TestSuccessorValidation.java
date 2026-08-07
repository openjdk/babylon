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
import jdk.incubator.code.Op;
import jdk.incubator.code.dialect.core.CoreOp;
import jdk.incubator.code.dialect.core.CoreType;
import jdk.incubator.code.dialect.java.JavaType;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

import java.util.Collections;

public class TestSuccessorValidation {

    @Test
    void testNumArgsLtNumParams() {
        Assertions.assertThrows(IllegalStateException.class, () -> generateModel(0));
    }

    @Test
    void testNumArgsGtNumParams() {
        Assertions.assertThrows(IllegalStateException.class, () -> generateModel(2));
    }

    @Test
    void testNumArgsEqNumParams() {
        Assertions.assertDoesNotThrow(() -> generateModel(1));
    }

    private static CoreOp.FuncOp generateModel(int numOfArgs) {
        // generates a model with two blocks, entry block and another block with one param
        // in the entry block we branch to the other block with the specified number of args
        return CoreOp.func("m", CoreType.FUNCTION_TYPE_VOID).body(eb -> {
            Block.Builder b2 = eb.block(JavaType.INT);
            Op.Result v = eb.add(CoreOp.constant(JavaType.INT, 1));
            eb.add(CoreOp.branch(b2.reference(Collections.nCopies(numOfArgs, v))));

            b2.add(CoreOp.return_());
        });
    }
}
