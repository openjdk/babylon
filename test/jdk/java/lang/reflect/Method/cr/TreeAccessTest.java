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

import org.testng.Assert;
import org.testng.annotations.Test;

import java.lang.reflect.code.op.CoreOps;
import java.lang.reflect.code.op.ExtendedOps;
import java.lang.reflect.code.parser.OpParser;
import java.lang.reflect.Method;
import java.lang.runtime.CodeReflection;
import java.util.Optional;

/**
 * @test
 * @summary Test symbolic access of method
 * @run testng TreeAccessTest
 */

public class TreeAccessTest {

    @CodeReflection
    public int m(String s) {
        return s.length();
    }

    @Test
    void testTreeAccess() throws Exception {
        Method m = TreeAccessTest.class.getDeclaredMethod("m", String.class);

        Optional<CoreOps.FuncOp> tree = m.getCodeModel();
        Assert.assertTrue(tree.isPresent());

        CoreOps.FuncOp methodTree = tree.get();

        String expectedTree = """
                func @"m" (%0 : TreeAccessTest, %1 : java.lang.String)int -> {
                      %2 : Var<java.lang.String> = var %1 @"s";
                      %3 : java.lang.String = var.load %2;
                      %4 : int = invoke %3 @"java.lang.String::length()int";
                      return %4;
                };
                """;

        Assert.assertEquals(canonicalizeDescription(methodTree.toText()), canonicalizeDescription(expectedTree));
    }

    static String canonicalizeDescription(String d) {
        return OpParser.fromString(ExtendedOps.FACTORY, d).get(0).toText();
    }

    @Test
    public int n(String s) {
        return s.length();
    }

    @Test
    void testNoTree() throws Exception {
        Method m = TreeAccessTest.class.getDeclaredMethod("n", String.class);

        Optional<CoreOps.FuncOp> tree = m.getCodeModel();
        Assert.assertTrue(tree.isEmpty());
    }
}