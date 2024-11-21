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
 * @modules jdk.incubator.code
 * @run testng TestLocation
 */

import org.testng.Assert;
import org.testng.annotations.Test;

import java.io.StringWriter;
import java.lang.reflect.Method;
import jdk.incubator.code.Location;
import jdk.incubator.code.Op;
import jdk.incubator.code.OpTransformer;
import jdk.incubator.code.op.CoreOp;
import jdk.incubator.code.op.ExtendedOp;
import jdk.incubator.code.parser.OpParser;
import jdk.incubator.code.writer.OpWriter;
import jdk.incubator.code.CodeReflection;
import java.util.Optional;
import java.util.stream.Stream;

public class TestLocation {
    @Test
    public void testLocation() {
        CoreOp.FuncOp f = getFuncOp(ClassWithReflectedMethod.class, "f");
        f.traverse(null, (o, ce) -> {
            if (ce instanceof CoreOp.ConstantOp cop) {
                Location loc = cop.location();
                Assert.assertNotNull(loc);

                int actualLine = loc.line();
                int expectedLine = Integer.parseInt((String) cop.value());
                Assert.assertEquals(actualLine, expectedLine);
            }
            return null;
        });
    }

    @CodeReflection
    static int f(int m, int n) {
        int sum = 0;
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                sum++;
            }
        }
        return sum;
    }

    @Test
    public void dropLocationTransform() {
        CoreOp.FuncOp f = getFuncOp(TestLocation.class, "f");

        CoreOp.FuncOp tf = f.transform(OpTransformer.DROP_LOCATION_TRANSFORMER);
        tf.setLocation(Location.NO_LOCATION);
        testNoLocations(tf);

        CoreOp.FuncOp tlf = lower(f).transform(OpTransformer.DROP_LOCATION_TRANSFORMER);
        tlf.setLocation(Location.NO_LOCATION);
        testNoLocations(tlf);
    }

    @Test
    public void dropLocationWriter() {
        CoreOp.FuncOp f = getFuncOp(TestLocation.class, "f");

        StringWriter w = new StringWriter();
        OpWriter.writeTo(w, f, OpWriter.LocationOption.DROP_LOCATION);
        String tfText = w.toString();
        CoreOp.FuncOp tf = (CoreOp.FuncOp) OpParser.fromString(ExtendedOp.FACTORY, tfText).getFirst();
        testNoLocations(tf);
    }

    static CoreOp.FuncOp lower(CoreOp.FuncOp f) {
        return f.transform(OpTransformer.LOWERING_TRANSFORMER);
    }

    static void testNoLocations(Op op) {
        boolean noLocations = op.elements().filter(ce -> ce instanceof Op)
                .allMatch(ce -> ((Op) ce).location() == Location.NO_LOCATION);
        Assert.assertTrue(noLocations);
    }


    static CoreOp.FuncOp getFuncOp(Class<?> c, String name) {
        Optional<Method> om = Stream.of(c.getDeclaredMethods())
                .filter(m -> m.getName().equals(name))
                .findFirst();

        Method m = om.get();
        return Op.ofMethod(m).get();
    }
}
