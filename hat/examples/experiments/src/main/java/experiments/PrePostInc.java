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


import java.lang.invoke.MethodHandles;
import java.lang.reflect.Method;
import jdk.incubator.code.CopyContext;
import jdk.incubator.code.Op;
import jdk.incubator.code.OpTransformer;
import jdk.incubator.code.TypeElement;
import jdk.incubator.code.Value;
import jdk.incubator.code.interpreter.Interpreter;
import jdk.incubator.code.op.CoreOp;
import jdk.incubator.code.type.JavaType;
import jdk.incubator.code.type.MethodRef;
import jdk.incubator.code.CodeReflection;
import java.util.List;
import java.util.Map;

public class PrePostInc {
        @CodeReflection
        public static int  preInc(int value) {
            int pre = 25 + ++value;
            return pre;
        }

        @CodeReflection
        public static int  postInc(int value) {
           int post = 25 + value++;
           return post;
        }

        static public void main(String[] args) throws Exception {
            Method pre = PrePostInc.class.getDeclaredMethod("preInc",  int.class);
            Method post = PrePostInc.class.getDeclaredMethod("postInc",  int.class);
            CoreOp.FuncOp preFunc = Op.ofMethod(pre).get();
            CoreOp.FuncOp postFunc = Op.ofMethod(post).get();

            Object preResult = Interpreter.invoke(MethodHandles.lookup(),preFunc,5);
            System.out.println("Pre "+ preResult);
            Object postResult = Interpreter.invoke(MethodHandles.lookup(),postFunc,5);
            System.out.println("Pre "+ postResult);
          //  javaFunc.writeTo(System.out);

        }
}

