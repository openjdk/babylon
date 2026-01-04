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

package experiments;

import java.lang.reflect.Method;

import jdk.incubator.code.CodeContext;
import jdk.incubator.code.Op;
import jdk.incubator.code.CodeTransformer;
import jdk.incubator.code.TypeElement;
import jdk.incubator.code.Value;
import jdk.incubator.code.Reflect;
import jdk.incubator.code.dialect.java.JavaOp;
import jdk.incubator.code.dialect.java.JavaType;
import optkl.Trxfmr;
import optkl.util.CallSite;

import java.util.List;


public class DNA {
    static int myFunc(int i) {
        return 0;
    }

    @Reflect
    public static void addMul(int add, int mul) {
        int len = myFunc(add);
    }

    public static class DNAOp extends Op { // externalized
        private final String opName;
        private final TypeElement type;

        DNAOp(String opName, TypeElement type, List<Value> operands) {
            super(operands);
            this.opName = opName;
            this.type = type;
        }

        @Override
        public Op transform(CodeContext copyContext, CodeTransformer opTransformer) {
            throw new IllegalStateException("in transform");
            //  return null;
        }


        @Override
        public TypeElement resultType() {
            System.out.println("in result type");
            return type;
        }
    }


    static public void main(String[] args) throws Exception {
        Method method = DNA.class.getDeclaredMethod("addMul", int.class, int.class);
        var funcOp = Op.ofMethod(method).get();
        var here = CallSite.of(DNA.class, "main");
        var transformed = new Trxfmr(funcOp).transform(_->true,(builder, op) -> {
            CodeContext cc = builder.context();
            if (op instanceof JavaOp.InvokeOp invokeOp) {
               // List<Value> operands = new ArrayList<>();
                //builder.op(new DNAOp("dna", JavaType.INT, operands));
                List<Value> inputOperands = invokeOp.operands();
                List<Value> outputOperands = cc.getValues(inputOperands);
                Op.Result inputResult = invokeOp.result();
                Op.Result outputResult = builder.op(new DNAOp("dna", JavaType.INT, outputOperands));
                cc.mapValue(inputResult, outputResult);
            } else {
                builder.op(op);
            }
            return builder;
        }).funcOp();


        System.out.println(transformed.toText());

    }
}

