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
 * @run testng TestParse
 */

import jdk.incubator.code.dialect.java.JavaOp;
import org.testng.Assert;
import org.testng.annotations.Test;

import jdk.incubator.code.Block;
import jdk.incubator.code.dialect.core.CoreOp;
import jdk.incubator.code.Op;
import jdk.incubator.code.dialect.java.MethodRef;
import jdk.incubator.code.extern.OpParser;
import java.util.List;
import java.util.function.IntUnaryOperator;

import static jdk.incubator.code.dialect.core.CoreOp.return_;
import static jdk.incubator.code.dialect.java.JavaOp.add;
import static jdk.incubator.code.dialect.core.CoreOp.constant;
import static jdk.incubator.code.dialect.core.CoreOp.func;
import static jdk.incubator.code.dialect.java.JavaOp.lambda;
import static jdk.incubator.code.dialect.core.CoreType.functionType;
import static jdk.incubator.code.dialect.java.JavaType.INT;
import static jdk.incubator.code.dialect.java.JavaType.type;

public class TestParse {

    static final MethodRef INT_UNARY_OPERATOR_METHOD = MethodRef.method(
            IntUnaryOperator.class, "applyAsInt",
            int.class, int.class);

    @Test
    public void testParseLambdaOp() {
        // functional type = (int)int
        CoreOp.FuncOp f = func("f", functionType(INT, INT))
                .body(block -> {
                    Block.Parameter i = block.parameters().get(0);

                    // functional type = (int)int
                    // op type = ()IntUnaryOperator
                    //   captures i
                    JavaOp.LambdaOp lambda = lambda(block.parentBody(),
                            functionType(INT, INT), type(IntUnaryOperator.class))
                            .body(lbody -> {
                                Block.Builder lblock = lbody.entryBlock();
                                Block.Parameter li = lblock.parameters().get(0);

                                lblock.op(return_(
                                        lblock.op(add(i, li))));
                            });

                    Op.Result fi = block.op(lambda);
                    Op.Result fortyTwo = block.op(constant(INT, 42));
                    Op.Result or = block.op(JavaOp.invoke(INT_UNARY_OPERATOR_METHOD, fi, fortyTwo));
                    block.op(return_(or));
                });

        List<Op> ops = OpParser.fromString(JavaOp.JAVA_DIALECT_FACTORY, f.toText());
        assertTextEquals(f, ops.get(0));
    }


    static final String NAMED_BODY = """
            func @"test" ^body1(%0 : java.type:"int", %1 : java.type:"int")java.type:"int" -> {
                %2 : java.type:"int" = constant @5;
                %3 : java.type:"int" = constant @2;
                branch ^b1(%2, %3);

              ^b1(%0 : java.type:"int", %1 : java.type:"int"):
                return %0;
            };
            """;
    @Test
    void testParseNamedBody() {
        Op opE = OpParser.fromString(JavaOp.JAVA_DIALECT_FACTORY, NAMED_BODY).get(0);
        Op opA = OpParser.fromString(JavaOp.JAVA_DIALECT_FACTORY, opE.toText()).get(0);
        assertTextEquals(opA, opE);
    }


    static final String ESCAPED_STRING = """
            func @"test" ()java.type:"java.lang.String" -> {
                %0 : java.type:"java.lang.String" = constant @"\\b \\f \\n \\r \\t \\' \\" \\\\";
                return %0;
            };
            """;
    @Test
    void testEscapedString() {
        Op opE = OpParser.fromString(JavaOp.JAVA_DIALECT_FACTORY, ESCAPED_STRING).get(0);
        Op opA = OpParser.fromString(JavaOp.JAVA_DIALECT_FACTORY, opE.toText()).get(0);
        assertTextEquals(opA, opE);

        CoreOp.ConstantOp cop = (CoreOp.ConstantOp) opE.bodies().get(0).entryBlock().firstOp();
        String v = (String) cop.value();
        Assert.assertEquals(v, "\b \f \n \r \t \' \" \\");
    }

    static void assertTextEquals(Op a, Op b) {
        Assert.assertEquals(a.toText(), b.toText());
    }
}
