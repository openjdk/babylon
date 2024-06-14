package hat;

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

import org.testng.annotations.Test;

import java.lang.reflect.code.CopyContext;
import java.lang.reflect.code.OpTransformer;
import java.lang.reflect.code.TypeElement;
import java.lang.reflect.code.Value;
import java.lang.reflect.code.op.CoreOp;
import java.lang.reflect.code.Op;
import java.lang.reflect.Method;
import java.lang.reflect.code.type.JavaType;
import java.lang.runtime.CodeReflection;
import java.util.List;


/*
 * @test
 * @run testng hat.CustomOpTest
 */

public class CustomOpTest {


    static int myFunc(int i) {
        return 0;
    }

    @CodeReflection
    public static void addMul(int add, int mul) {
        int len = myFunc(add);
    }

    public static class DNAOp extends Op { // externalized
        private final TypeElement type;

        DNAOp(String opName, TypeElement type, List<Value> operands) {
            super(opName, operands);
            this.type = type;
        }

        @Override
        public Op transform(CopyContext copyContext, OpTransformer opTransformer) {
            throw new IllegalStateException("in transform");

        }


        @Override
        public TypeElement resultType() {
            System.out.println("in result type");
            return type;
        }
    }

    @Test
    public void testDNAOp() throws NoSuchMethodException {
        Method method = CustomOpTest.class.getDeclaredMethod("addMul", int.class, int.class);
        var funcOp = method.getCodeModel().get();
        var transformed = funcOp.transform((builder, op) -> {
            CopyContext cc = builder.context();
            if (op instanceof CoreOp.InvokeOp invokeOp) {
                List<Value> inputOperands = invokeOp.operands();
                List<Value> outputOperands = cc.getValues(inputOperands);
                Op.Result inputResult = invokeOp.result();
                Op.Result outputResult = builder.op(new DNAOp("dna", JavaType.INT, outputOperands));
                cc.mapValue(inputResult, outputResult);
            } else {
                builder.op(op);
            }
            return builder;
        });

        System.out.println(transformed.toText());
    }


}