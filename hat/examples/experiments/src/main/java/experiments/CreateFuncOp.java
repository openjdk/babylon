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
import optkl.Trxfmr;
import jdk.incubator.code.CodeContext;
import jdk.incubator.code.CodeTransformer;
import jdk.incubator.code.Op;
import jdk.incubator.code.TypeElement;
import jdk.incubator.code.Value;
import jdk.incubator.code.bytecode.BytecodeGenerator;
import jdk.incubator.code.dialect.core.CoreOp;
import jdk.incubator.code.dialect.core.CoreType;
import jdk.incubator.code.dialect.java.JavaOp;
import jdk.incubator.code.dialect.java.JavaOp.InvokeOp.InvokeKind;
import jdk.incubator.code.dialect.java.JavaType;
import jdk.incubator.code.dialect.java.MethodRef;
import optkl.util.OpCodeBuilder;

import java.lang.invoke.MethodHandles;
import java.util.List;
import java.util.Map;

/**
 * Demonstrates how to dynamically build a new function using the code reflection API.
 * <p>
 * This example creates an <code>rsqrt</code> function, which computes the inverse of a square root.
 * The function takes one argument of type <code>double</code> and returns a <code>double</code>.
 * The implementation uses {@link Math#sqrt(double)} for the square root calculation.
 * </p>
 *
 * <p>
 * In this example, you will learn how to:
 * <ol>
 *   <li>Create a function dynamically</li>
 *   <li>Append new Op nodes in the builder</li>
 *   <li>Compose operations in the code tree</li>
 *   <li>Create nodes to call static methods</li>
 *   <li>Evaluate the composed method in the interpreter</li>
 * </ol>
 * </p>
 *
 * <p>
 * After building the code model for the function, it will be executed both in the code reflection interpreter and in the bytecode interpreter.
 * </p>
 *
 * <p>
 * <b>How to run:</b><br>
 * <code>
 * java --add-modules jdk.incubator.code -cp target/crsamples-1.0-SNAPSHOT.jar oracle.code.samples.DynamicFunctionBuild
 * </code>
 * </p>
 */
public class CreateFuncOp {

    public abstract static class Inject extends Op {
        public Inject(List<Value> operands) {
            super(operands);
        }

        protected Inject(Op that, CodeContext cc) {
            super(that, cc);
        }
    }

    public final static class Pre extends Inject {

        public Pre(List<Value> operands) {
            super(operands);
        }

        public Pre(Pre pre, CodeContext copyContext) {
            super(pre, copyContext);
        }

        @Override
        public Op transform(CodeContext copyContext, CodeTransformer opTransformer) {
            return new Pre(this, copyContext);
        }

        @Override
        public TypeElement resultType() {
            return JavaType.VOID;
        }

        public Map<String, Object> externalize() {
            return Map.of(Pre.class.getSimpleName(), JavaOp.InvokeOp.InvokeKind.INSTANCE);
        }

    }

    public final static class Post extends Inject {

        public Post(List<Value> operands) {
            super(operands);
        }

        public Post(Post pre, CodeContext copyContext) {
            super(pre, copyContext);
        }

        @Override
        public Op transform(CodeContext copyContext, CodeTransformer opTransformer) {
            return new Post(this, copyContext);
        }

        @Override
        public TypeElement resultType() {
            return JavaType.VOID;
        }

        public Map<String, Object> externalize() {
            return Map.of(Post.class.getSimpleName(), JavaOp.InvokeOp.InvokeKind.INSTANCE);
        }

    }

    public static void main(String[] args) throws Throwable {

        var lookup = MethodHandles.lookup();
        MethodRef MathSqrt = MethodRef.method(Math.class, "sqrt", double.class, double.class);
        MethodRef MathAbs = MethodRef.method(Math.class, "abs", double.class, double.class);
        MethodRef Println = MethodRef.method(IO.class, "println", void.class, Object.class);

        CoreOp.FuncOp rsqrtFuncOp = CoreOp.func("rsqrt", CoreType.functionType(JavaType.DOUBLE, JavaType.DOUBLE))
                .body(builder -> {// double rsqrt(double arg){return 1 / Math.sqrt(qrg)}
                    // var arg = builder.parameters().getFirst();
                    var argOp = CoreOp.var("arg", builder.parameters().getFirst());
                    var arg = builder.op(argOp);
                   // var arg = builder.parameters().getFirst();
                    var sqrtInvoke = JavaOp.invoke(InvokeKind.STATIC, false, JavaType.DOUBLE, MathSqrt, arg);
                    var _1f = builder.op(CoreOp.constant(JavaType.DOUBLE, 1.0));

                    Op.Result invokeResult = builder.op(sqrtInvoke);
                    Op.Result divResult = builder.op(
                            JavaOp.div(_1f, invokeResult)
                    );
                    builder.op(CoreOp.return_(divResult));
                });

        System.out.println( OpCodeBuilder.toText(rsqrtFuncOp));
        System.out.println(" 1/sqrt(100) = " + BytecodeGenerator.generate(lookup, rsqrtFuncOp).invoke(100));
        var trxfmr = new Trxfmr(rsqrtFuncOp);
        trxfmr.transform(ce -> ce instanceof JavaOp.InvokeOp, c -> {
            c.add(JavaOp.if_(c.builder().parentBody()).if_(b -> {
                var lhs = b.op(CoreOp.constant(JavaType.BOOLEAN, false));
                var rhs = b.op(CoreOp.constant(JavaType.BOOLEAN, true));
                b.op(CoreOp.core_yield(b.op(JavaOp.or(lhs, rhs))));
            }).then(b -> {
                var msg = b.op(CoreOp.constant(JavaType.J_L_STRING, "Then"));
                b.op(new Pre(List.of()));
                b.op(JavaOp.invoke(InvokeKind.STATIC, false, JavaType.VOID, Println, msg));
                b.op(new Post(List.of()));
                b.op(CoreOp.core_yield());
            }).else_(b -> {
                var msg = b.op(CoreOp.constant(JavaType.J_L_STRING, "Else"));
                b.op(new Pre(List.of()));
                b.op(JavaOp.invoke(InvokeKind.STATIC, false, JavaType.VOID, Println, msg));
                b.op(new Post(List.of()));
                b.op(CoreOp.core_yield());
            }));
             c.add(new Pre(List.of()));
             c.replace(JavaOp.invoke(InvokeKind.STATIC, false, JavaType.DOUBLE, MathAbs, c.mappedOperand(0)));
             c.add(new Post(List.of()));
        });
        System.out.println( OpCodeBuilder.toText(trxfmr.funcOp()));

        // We need to remove our injected ops from the model to execute
        trxfmr.transform(ce -> ce instanceof Inject, c -> c.remove()).funcOp();

        var javaCodeBuilder = new JavaHATCodeBuilder<>(lookup,trxfmr.funcOp());
        System.out.println(javaCodeBuilder.toText());
        System.out.println( OpCodeBuilder.toText(trxfmr.funcOp()));
        System.out.println(" 1/abs(100) = " + BytecodeGenerator.generate(lookup, trxfmr.funcOp()).invoke(100));
    }
}

