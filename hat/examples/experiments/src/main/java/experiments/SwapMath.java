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


import hat.codebuilders.JavaHATCodeBuilder;
import optkl.Invoke;
import optkl.Trxfmr;
import static optkl.Invoke.invokeOpHelper;
import jdk.incubator.code.Op;
import jdk.incubator.code.bytecode.BytecodeGenerator;
import jdk.incubator.code.dialect.core.CoreOp;
import jdk.incubator.code.dialect.core.CoreType;
import jdk.incubator.code.dialect.java.JavaOp;
import jdk.incubator.code.dialect.java.JavaOp.InvokeOp.InvokeKind;
import jdk.incubator.code.dialect.java.JavaType;
import jdk.incubator.code.dialect.java.MethodRef;
import optkl.util.Regex;

import java.lang.invoke.MethodHandles;

public class SwapMath {
    public static void main(String[] args) throws Throwable {
        var lookup = MethodHandles.lookup();
        MethodRef MathSqrt = MethodRef.method(Math.class, "sqrt", double.class, double.class);
        MethodRef MathAbs = MethodRef.method(Math.class, "abs", double.class, double.class);

        CoreOp.FuncOp rsqrt= CoreOp.func("rsqrt", CoreType.functionType(JavaType.DOUBLE, JavaType.DOUBLE))
                .body(builder -> {// double rsqrt(double arg){return 1 / Math.sqrt(qrg)}
                   // var arg = builder.parameters().getFirst();
                    var argOp = CoreOp.var("arg", builder.parameters().getFirst());
                    var arg = builder.op(argOp);

                    // We can pass builder.parameters().getFirst() directly as arg below.  But then we don't know the name
                    var sqrtInvoke = JavaOp.invoke(InvokeKind.STATIC, false, JavaType.DOUBLE, MathSqrt, arg);
                    var _1f = builder.op(CoreOp.constant(JavaType.DOUBLE, 1.0));

                    Op.Result invokeResult = builder.op(sqrtInvoke);
                    Op.Result divResult = builder.op(
                            JavaOp.div(_1f, invokeResult)
                    );
                    builder.op(CoreOp.return_(divResult));
                });
        var javaCodeBuilder = new JavaHATCodeBuilder<>(lookup,rsqrt);
        System.out.println(rsqrt.toText());
        System.out.println(javaCodeBuilder.toText());
        System.out.println(" 1/sqrt(100) = " + BytecodeGenerator.generate(lookup, rsqrt).invoke(100));



        System.out.println("--------------------------");
        var abs = rsqrt.transform((builder,op)->{
            if (invokeOpHelper(lookup,op) instanceof Invoke ih
                    && ih.named(Regex.of("sqrt")) && ih.isStatic() && ih.returns(double.class) && ih.receives(double.class)){
                var absStaticMethod = MethodRef.method(Math.class, "abs", double.class, double.class);
                var absInvoke =  JavaOp.invoke(InvokeKind.STATIC, false, absStaticMethod.type().returnType(), absStaticMethod,
                        builder.context().getValue(op.operands().get(0)));
                var absResult= builder.op(absInvoke);
                builder.context().mapValue(op.result(), absResult);
            }else{
                builder.op(op);
            }
            return builder;
        });

        System.out.println(abs.toText());
        javaCodeBuilder = new JavaHATCodeBuilder<>(lookup,abs);
        System.out.println(" 1/abs(100) = " + BytecodeGenerator.generate(MethodHandles.lookup(), abs).invoke(100));


        System.out.println("Now using txfmr--------------------------");
        var newAbs =Trxfmr.of(rsqrt)
                .transform(ce-> Invoke.invokeOpHelper(lookup,ce) instanceof Invoke $
                                && $.named(Regex.of("sqrt"))
                                && $.isStatic()
                                && $.returns(double.class)
                                && $.receives(double.class), c->
                        c.replace(
                                JavaOp.invoke(InvokeKind.STATIC, false, JavaType.DOUBLE, MathAbs, c.mappedOperand( 0))
                        )
                )
                .funcOp();


        System.out.println(newAbs.toText());
        javaCodeBuilder = new JavaHATCodeBuilder<>(lookup,newAbs);
        System.out.println(javaCodeBuilder.toText());
        System.out.println(" 1/abs(100) = " + BytecodeGenerator.generate(MethodHandles.lookup(), newAbs).invoke(100));


    }
}

