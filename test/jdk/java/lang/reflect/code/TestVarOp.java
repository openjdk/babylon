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
 * @run testng TestVarOp
 */

import org.testng.Assert;
import org.testng.annotations.Test;

import java.lang.reflect.Method;
import java.lang.reflect.code.Op;
import java.lang.reflect.code.OpTransformer;
import java.lang.reflect.code.Value;
import java.lang.reflect.code.op.CoreOp;
import java.lang.reflect.code.parser.OpParser;
import java.lang.reflect.code.type.CoreTypeFactory;
import java.lang.reflect.code.type.FunctionType;
import java.lang.reflect.code.type.JavaType;
import java.lang.runtime.CodeReflection;
import java.util.List;
import java.util.Optional;
import java.util.stream.Stream;

public class TestVarOp {

    @CodeReflection
    static Object f(String s) {
        Object o = s;
        return o;
    }

    @Test
    public void testTypeSubstitutionAndPreserve() {
        CoreOp.FuncOp f = getFuncOp("f");
        CoreOp.FuncOp ft = CoreOp.func("f", FunctionType.functionType(JavaType.J_L_OBJECT, JavaType.type(CharSequence.class)))
                .body(fb -> {
                    fb.transformBody(f.body(), fb.parameters(), OpTransformer.COPYING_TRANSFORMER);
                });

        List<CoreOp.VarOp> vops = ft.elements()
                .flatMap(ce -> ce instanceof CoreOp.VarOp vop ? Stream.of(vop) : null)
                .toList();
        // VarOp for block parameter, translate from String to CharSequence
        Assert.assertEquals(vops.get(0).resultType().valueType(), JavaType.type(CharSequence.class));
        // VarOp for local variable, preserve Object
        Assert.assertEquals(vops.get(1).resultType().valueType(), JavaType.J_L_OBJECT);
    }

    @Test
    public void testNoName() {
        CoreOp.FuncOp f = getFuncOp("f");
        f = f.transform((block, op) -> {
            if (op instanceof CoreOp.VarOp vop) {
                Value init = block.context().getValue(vop.initOperand());
                Op.Result v = block.op(CoreOp.var(init));
                block.context().mapValue(vop.result(), v);
            } else {
                block.op(op);
            }
            return block;
        });

        Op op = OpParser.fromString(CoreOp.FACTORY, CoreTypeFactory.CORE_TYPE_FACTORY, f.toText()).get(0);
        boolean allNullNames = op.elements()
                .flatMap(ce -> ce instanceof CoreOp.VarOp vop ? Stream.of(vop) : null)
                .allMatch(vop -> vop.varName() == null);
        Assert.assertTrue(allNullNames);
    }

    static CoreOp.FuncOp getFuncOp(String name) {
        Optional<Method> om = Stream.of(TestVarOp.class.getDeclaredMethods())
                .filter(m -> m.getName().equals(name))
                .findFirst();

        Method m = om.get();
        return m.getCodeModel().get();
    }
}
