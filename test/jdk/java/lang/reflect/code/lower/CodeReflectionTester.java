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

import java.io.StringWriter;
import java.lang.reflect.Member;
import java.lang.reflect.Method;
import java.lang.reflect.code.Op;
import java.lang.reflect.code.analysis.SSA;
import java.lang.reflect.code.op.CoreOp;
import java.lang.reflect.code.op.ExtendedOp;
import java.lang.reflect.code.parser.OpParser;
import java.lang.reflect.code.writer.OpWriter;
import java.lang.runtime.CodeReflection;

public class CodeReflectionTester {

    public static void main(String[] args) throws ReflectiveOperationException {
        if (args.length != 1) {
            System.err.println("Usage: CodeReflectionTester <classname>");
            System.exit(1);
        }
        Class<?> clazz = Class.forName(args[0]);
        for (Method m : clazz.getDeclaredMethods()) {
            check(m);
        }
    }

    static void check(Method method) throws ReflectiveOperationException {
        if (!method.isAnnotationPresent(CodeReflection.class)) {
            return;
        }

        LoweredModel lma = method.getAnnotation(LoweredModel.class);
        if (lma == null) {
            throw new AssertionError("No @IR annotation found on reflective method");
        }

        CoreOp.FuncOp f = method.getCodeModel().orElseThrow(() ->
                new AssertionError("No code model for reflective method"));
        f = lower(f, lma.ssa());

        String actual = canonicalizeModel(method, f);
        String expected = canonicalizeModel(method, lma.value());
        if (!actual.equals(expected)) {
            throw new AssertionError(String.format("Bad code model\nFound:\n%s\n\nExpected:\n%s", actual, expected));
        }
    }

    static CoreOp.FuncOp lower(CoreOp.FuncOp f, boolean ssa) {
        f = f.transform((block, op) -> {
            if (op instanceof Op.Lowerable lop) {
                return lop.lower(block);
            } else {
                block.op(op);
                return block;
            }
        });
        f.writeTo(System.out);

        if (ssa) {
            f = SSA.transform(f);
            f.writeTo(System.out);
        }

        return f;
    }

    // serializes dropping location information, parses, and then serializes, dropping location information
    static String canonicalizeModel(Member m, Op o) {
        return canonicalizeModel(m, serialize(o));
    }

    // parses, and then serializes, dropping location information
    static String canonicalizeModel(Member m, String d) {
        Op o;
        try {
            o = OpParser.fromString(ExtendedOp.FACTORY, d).get(0);
        } catch (Exception e) {
            throw new IllegalStateException(m.toString(), e);
        }
        return serialize(o);
    }

    // serializes, dropping location information
    static String serialize(Op o) {
        StringWriter w = new StringWriter();
        OpWriter.writeTo(w, o, OpWriter.LocationOption.DROP_LOCATION);
        return w.toString();
    }
}
