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
 * @run testng TestPatterns
 * @enablePreview
 */

import org.testng.Assert;
import org.testng.annotations.Test;

import java.lang.reflect.code.OpTransformer;
import java.lang.reflect.code.op.CoreOp;
import java.lang.reflect.code.Op;
import java.lang.reflect.code.interpreter.Interpreter;
import java.lang.invoke.MethodHandles;
import java.lang.reflect.Method;
import java.lang.runtime.CodeReflection;
import java.util.Optional;
import java.util.stream.Stream;

public class TestPatterns {

    interface Point {
    }

    record ConcretePoint(int x, int y) implements Point {
    }

    enum Color {RED, GREEN, BLUE}

    record ColoredPoint(ConcretePoint p, Color c) implements Point {
    }

    record Rectangle(Point upperLeft, Point lowerRight) {
    }


    @CodeReflection
    public static String recordPatterns(Object r) {
        if (r instanceof Rectangle(
                ColoredPoint(ConcretePoint p, Color c),
                ColoredPoint lr)) {
            return p.toString();
        } else {
            return "";
        }
    }

    @Test
    public void testRecordPatterns() {
        CoreOp.FuncOp f = getFuncOp("recordPatterns");

        f.writeTo(System.out);

        CoreOp.FuncOp lf = f.transform(OpTransformer.LOWERING_TRANSFORMER);

        lf.writeTo(System.out);

        {
            Rectangle r = new Rectangle(
                    new ColoredPoint(new ConcretePoint(1, 2), Color.RED),
                    new ColoredPoint(new ConcretePoint(3, 4), Color.BLUE));
            Assert.assertEquals(Interpreter.invoke(MethodHandles.lookup(), lf, r), recordPatterns(r));
        }

        {
            Rectangle r = new Rectangle(
                    new ColoredPoint(new ConcretePoint(1, 2), Color.RED),
                    new ConcretePoint(3, 4));
            Assert.assertEquals(Interpreter.invoke(MethodHandles.lookup(), lf, r), recordPatterns(r));
        }

        {
            Rectangle r = new Rectangle(
                    new ConcretePoint(1, 2),
                    new ConcretePoint(3, 4));
            Assert.assertEquals(Interpreter.invoke(MethodHandles.lookup(), lf, r), recordPatterns(r));
        }

        {
            String r = "";;
            Assert.assertEquals(Interpreter.invoke(MethodHandles.lookup(), lf, r), recordPatterns(r));
        }
    }


    static CoreOp.FuncOp getFuncOp(String name) {
        Optional<Method> om = Stream.of(TestPatterns.class.getDeclaredMethods())
                .filter(m -> m.getName().equals(name))
                .findFirst();

        Method m = om.get();
        return m.getCodeModel().get();
    }

}
