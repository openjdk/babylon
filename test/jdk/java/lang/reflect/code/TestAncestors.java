/*
 * Copyright (c) 2025, Oracle and/or its affiliates. All rights reserved.
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
 * @run junit TestAncestors
 */

import jdk.incubator.code.*;
import jdk.incubator.code.dialect.core.CoreOp;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

import java.lang.reflect.Method;
import java.util.ArrayList;
import java.util.List;
import java.util.Optional;
import java.util.stream.Stream;

import static java.lang.System.out;

public class TestAncestors {

    // Model with sufficient nested structure
    @CodeReflection
    static void f() {
        out.println("X");
        {
            out.println("X");
            {
                out.println("X");
            }
            out.println("X");
            {
                out.println("X");
            }
            out.println("X");
        }
        out.println("X");
        {
            out.println("X");
            {
                out.println("X");
            }
            out.println("X");
            {
                out.println("X");
            }
            out.println("X");
        }
        out.println("X");
    }

    @Test
    public void test() {
        CoreOp.FuncOp f = getFuncOp("f");
        out.println(f.toText());

        List<List<CodeElement<?, ?>>> paths = new ArrayList<>();
        // path has pattern of [op, body, block, ... ,body, block, op]
        computedPaths(paths,List.of(), f);

        testPathPrefix(paths.getFirst());
        paths.forEach(TestAncestors::testPathAncestors);
    }

    static void testPathPrefix(List<CodeElement<?, ?>> path) {
        for (int i = 0; i < 3; i++) {
            CodeElement<?, ?> a = path.get(i);
            testTopElements(a);
        }
    }

    static void testTopElements(CodeElement<?, ?> a) {
        switch (a) {
            case Op op -> {
                Assertions.assertNull(op.ancestorOp());
                Assertions.assertNull(op.ancestorBody());
                Assertions.assertNull(op.ancestorBlock());
            }
            case Body body -> {
                Assertions.assertNotNull(body.ancestorOp());
                Assertions.assertNull(body.ancestorBody());
                Assertions.assertNull(body.ancestorBlock());
            }
            case Block block -> {
                Assertions.assertNotNull(block.ancestorOp());
                Assertions.assertNotNull(block.ancestorBody());
                Assertions.assertNull(block.ancestorBlock());
            }
        }
    }

    static void testPathAncestors(List<CodeElement<?, ?>> path) {
        Assertions.assertTrue(path.size() > 3);
        for (int i = 0; i < 3; i++) {
            CodeElement<?, ?> a = path.get(i);
            int size = path.size() - 1;
            for (int j = size; j > size - 3; j--) {
                if (j < i) {
                    continue;
                }

                CodeElement<?, ?> e = path.get(j);
                testAncestors(a, e);
            }
        }
    }

    static void testAncestors(CodeElement<?, ?> a, CodeElement<?, ?> e) {
        Assertions.assertTrue(isSameOrAncestorUsingParent(e, a));
        if (a != e) {
            Assertions.assertTrue(a.isAncestorOf(e));
        }

        switch (a) {
            case Op op -> {
                Assertions.assertTrue(isSameOrAncestorOfOp(op, e));
            }
            case Body body -> {
                Assertions.assertTrue(isSameOrAncestorOfBody(body, e));
            }
            case Block block -> {
                Assertions.assertTrue(isSameOrAncestorOfBlock(block, e));
            }
        }
    }

    static boolean isSameOrAncestorUsingParent(CodeElement<?, ?> e, CodeElement<?, ?> a) {
        while (e != null && e != a) {
            e = e.parent();
        }
        return e != null;
    }

    static boolean isSameOrAncestorOfOp(Op a, CodeElement<?, ?> e) {
        while (e != null && e != a) {
            e = e.ancestorOp();
        }
        return e != null;
    }

    static boolean isSameOrAncestorOfBody(Body a, CodeElement<?, ?> e) {
        while (e != null && e != a) {
            e = e.ancestorBody();
        }
        return e != null;
    }

    static boolean isSameOrAncestorOfBlock(Block a, CodeElement<?, ?> e) {
        while (e != null && e != a) {
            e = e.ancestorBlock();
        }
        return e != null;
    }

    static void computedPaths(List<List<CodeElement<?, ?>>> paths, List<CodeElement<?, ?>> path, CodeElement<?, ?> e) {
        ArrayList<CodeElement<?, ?>> p = new ArrayList<>(path);
        p.add(e);

        if (e.children().isEmpty()) {
            paths.add(p);
            return;
        }

        for (CodeElement<?, ?> child : e.children()) {
            computedPaths(paths, p, child);
        }
    }

    static CoreOp.FuncOp getFuncOp(String name) {
        Optional<Method> om = Stream.of(TestAncestors.class.getDeclaredMethods())
                .filter(m -> m.getName().equals(name))
                .findFirst();

        Method m = om.get();
        return Op.ofMethod(m).get();
    }
}
