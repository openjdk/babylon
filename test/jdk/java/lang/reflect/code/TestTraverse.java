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
 * @run testng TestTraverse
 */

import org.testng.Assert;
import org.testng.annotations.Test;

import java.lang.reflect.Method;
import java.lang.reflect.code.CodeElement;
import java.lang.reflect.code.Op;
import java.lang.reflect.code.analysis.SSA;
import java.lang.reflect.code.op.CoreOps;
import java.lang.runtime.CodeReflection;
import java.util.ArrayList;
import java.util.List;
import java.util.Optional;
import java.util.stream.Stream;

public class TestTraverse {

    @CodeReflection
    private static int f(String s, int i, List<Object> acc) {
        char c = s.charAt(i);
        int d = (c - '0');
        int n = s.length();
        while (++i < n) {
            c = s.charAt(i);
            if (c >= '0' && c <= '9') {
                d = d * 10 + (c - '0');
                continue;
            }
            break;
        }
        acc.add(d);
        return i;
    }

    @Test
    public void test() {
        CoreOps.FuncOp f = getFuncOp("f");
        testTraverse(f);

        f = f.transform((b, o) -> {
            if (o instanceof Op.Lowerable l) {
                return l.lower(b);
            } else {
                b.op(o);
                return b;
            }
        });
        testTraverse(f);

        f = SSA.transform(f);
        testTraverse(f);
    }

    void testTraverse(Op op) {
        List<CodeElement<?, ?>> tl = op.traverse(new ArrayList<>(), (l, e) -> {
            l.add(e);
            return l;
        });
        Assert.assertEquals(op.elements().toList(), tl);

        Assert.assertEquals(op.elements().limit(2).toList(), tl.subList(0, 2));
    }

    static CoreOps.FuncOp getFuncOp(String name) {
        Optional<Method> om = Stream.of(TestTraverse.class.getDeclaredMethods())
                .filter(m -> m.getName().equals(name))
                .findFirst();

        Method m = om.get();
        return m.getCodeModel().get();
    }
}
