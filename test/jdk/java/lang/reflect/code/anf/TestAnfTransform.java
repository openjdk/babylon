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

/*
 * @test
 * @run main TestAnfTransform
 */

import java.lang.reflect.Method;
import java.lang.reflect.code.OpTransformer;
import java.lang.reflect.code.analysis.AnfTransformer;
import java.lang.reflect.code.analysis.SSA;
import java.lang.reflect.code.analysis.NormalizeBlocksTransformer;
import java.lang.reflect.code.op.CoreOp;
import java.lang.runtime.CodeReflection;
import java.util.List;

public class TestAnfTransform {

    @CodeReflection
    public static int test1(int arg1, int arg2) {
       return arg1 + arg2 - 52;
    }

    @CodeReflection
    public static int test2(int arg1, int arg2) {
        if (arg1 > arg2) {
            return arg1 + 21;
        } else {
            return arg2 + 42;
        }
    }

    @CodeReflection
    public static int fibonacci(int v) {
        if (v == 0) {
            return 0;
        } else if (v == 1) {
            return 1;
        }

        v = v - 2;
        int s1, s2, t;
        s1 = 0;
        s2 = 1;

        while(v > 0) {
            t = s1 + s2;
            s1 = s2;
            s2 = t;
            v--;
        }

        return s1 + s2;

    }

    public static void main(String[] args) {

        //testRun("test2", List.of(int.class, int.class), 20, 1);
        testRun("fibonacci", List.of(int.class), 20);

    }

    private static void testRun(String methodName, List<Class<?>> params, Object...args) {
        try {
            Class<TestAnfTransform> clazz = TestAnfTransform.class;
            Method method = clazz.getDeclaredMethod(methodName,params.toArray(new Class[params.size()]));
            CoreOp.FuncOp f = method.getCodeModel().orElseThrow();

            //Ensure we're fully lowered before testing.
            var fz = f.transform(OpTransformer.LOWERING_TRANSFORMER);
            fz = SSA.transform(fz);
            fz = NormalizeBlocksTransformer.transform(fz);

            System.out.println(fz.toText());

            var res = new AnfTransformer(fz).transform();
            //var labeled = new Labeler().label(res);
            //p.print(res);
            //p.print(labeled);

            System.out.println("---------------------");
            System.out.println(res.toText());

        } catch (NoSuchMethodException e) {
            throw new RuntimeException(e);
        }
    }
}
