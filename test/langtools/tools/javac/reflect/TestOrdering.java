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
 * @bug 8392141
 * @library /tools/lib
 * @modules
 *      jdk.compiler/com.sun.tools.javac.api
 *      jdk.compiler/com.sun.tools.javac.main
 *      jdk.incubator.code
 * @build toolbox.ToolBox toolbox.JavacTask
 * @run junit TestOrdering
 */

import java.net.URL;
import java.net.URLClassLoader;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.function.Supplier;

import jdk.incubator.code.Op;
import org.junit.jupiter.api.Assertions;
import toolbox.JavacTask;
import toolbox.ToolBox;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.TestInfo;

public class TestOrdering {

    Path base;
    ToolBox tb = new ToolBox();

    @Test
    void test() throws Exception {
        Path classes = base.resolve("classes");
        Files.createDirectories(classes);
        new JavacTask(tb)
                .options("-d", classes.toString(), "--add-modules", "jdk.incubator.code", "-XDreflectAll")
                .sources("""
                             import java.util.function.Supplier;

                             abstract class Base<T> {
                                 abstract T lazy(Supplier<T> supplier);
                             }
                         """,
                         """
                             import java.util.function.Supplier;

                             abstract class A extends Base<String> {
                                 @Override
                                 final String lazy(Supplier<String> supplier) {
                                     return supplier.get();
                                 }

                                 abstract static class Op extends A {}

                                 Object toB() {
                                     return new B.Op() {};
                                 }
                             }
                         """,
                         """
                             import java.util.function.Supplier;

                             abstract class B extends Base<Integer> {
                                 @Override
                                 final Integer lazy(Supplier<Integer> supplier) {
                                     return supplier.get();
                                 }

                                 abstract static class Op extends B {}

                                 Object toA() {
                                     return new A.Op() {};
                                 }
                             }
                         """)
                .run()
                .writeAll();

        try (URLClassLoader loader = new URLClassLoader(new URL[] { classes.toUri().toURL() })) {
            Class<?> clsA = loader.loadClass("A");
            var methodA = clsA.getDeclaredMethod("lazy", Supplier.class);
            Assertions.assertEquals("""
                                    func @loc="4:9:A.java" @func.source=java.ref:"A::lazy(java.util.function.Supplier<java.lang.String>):java.lang.String" (%0 : java.type:"A", %1 : java.type:"java.util.function.Supplier<java.lang.String>")java.type:"java.lang.String" -> {
                                        %2 : Var<java.type:"java.util.function.Supplier<java.lang.String>"> = var %1 @loc="4:9" @"supplier";
                                        %3 : java.type:"java.util.function.Supplier<java.lang.String>" = var.load %2 @loc="6:20";
                                        %4 : java.type:"java.lang.String" = invoke %3 @loc="6:20" @java.ref:"java.util.function.Supplier::get():java.lang.Object";
                                        return %4 @loc="6:13";
                                    };""", Op.ofMethod(methodA).orElseThrow().toText());
            Class<?> clsB = loader.loadClass("B");
            var methodB = clsB.getDeclaredMethod("lazy", Supplier.class);
            Assertions.assertEquals("""
                                    func @loc="4:9:B.java" @func.source=java.ref:"B::lazy(java.util.function.Supplier<java.lang.Integer>):java.lang.Integer" (%0 : java.type:"B", %1 : java.type:"java.util.function.Supplier<java.lang.Integer>")java.type:"java.lang.Integer" -> {
                                        %2 : Var<java.type:"java.util.function.Supplier<java.lang.Integer>"> = var %1 @loc="4:9" @"supplier";
                                        %3 : java.type:"java.util.function.Supplier<java.lang.Integer>" = var.load %2 @loc="6:20";
                                        %4 : java.type:"java.lang.Integer" = invoke %3 @loc="6:20" @java.ref:"java.util.function.Supplier::get():java.lang.Object";
                                        return %4 @loc="6:13";
                                    };""", Op.ofMethod(methodB).orElseThrow().toText());

        }
    }

    @BeforeEach
    public void setUp(TestInfo info) {
        base = Paths.get(".")
                    .resolve(info.getTestMethod()
                                 .orElseThrow()
                                 .getName());
    }
}
