


    /*
     * Copyright (c) 2024 Intel Corporation. All rights reserved.
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
    import java.lang.reflect.code.CopyContext;
    import java.lang.reflect.code.Op;
    import java.lang.reflect.code.OpTransformer;
    import java.lang.reflect.code.TypeElement;
    import java.lang.reflect.code.Value;
    import java.lang.reflect.code.interpreter.Interpreter;
    import java.lang.reflect.code.op.CoreOp;
    import java.lang.reflect.code.type.JavaType;
    import java.lang.reflect.code.type.MethodRef;
    import java.lang.runtime.CodeReflection;
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
            CoreOp.FuncOp preFunc = pre.getCodeModel().get();
            CoreOp.FuncOp postFunc = post.getCodeModel().get();

            Object preResult = Interpreter.invoke(preFunc,5);
            System.out.println("Pre "+ preResult);
            Object postResult = Interpreter.invoke(postFunc,5);
            System.out.println("Pre "+ postResult);
          //  javaFunc.writeTo(System.out);

        }
    }

