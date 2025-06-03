/*
 * Copyright (c) 2025, Oracle and/or its affiliates. All rights reserved.
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

import jdk.incubator.code.*;
import jdk.incubator.code.analysis.NormalizeBlocksTransformer;
import jdk.incubator.code.analysis.SSA;
import jdk.incubator.code.op.CoreOp;
import jdk.incubator.code.op.ExtendedOp;
import jdk.incubator.code.parser.OpParser;
import jdk.incubator.code.writer.OpWriter;

import java.io.StringWriter;
import java.lang.invoke.MethodHandles;
import java.lang.reflect.Member;
import java.lang.reflect.Method;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.function.Predicate;
import java.util.stream.Collectors;

public class CodeReflectionTester {

    public static void main(String[] args) throws ReflectiveOperationException {
        if (args.length != 1) {
            System.err.println("Usage: CodeReflectionTester <classname>");
            System.exit(1);
        }
        Class<?> clazz = Class.forName(args[0]);

        Method lookupMethod = clazz.getMethod("lookup");
        MethodHandles.Lookup lookup = (MethodHandles.Lookup) lookupMethod.invoke(null);

        Method opConstantsMethod = clazz.getMethod("opConstants");
        @SuppressWarnings("unchecked")
        Predicate<Op> opConstants = (Predicate<Op>) opConstantsMethod.invoke(null);

        for (Method m : clazz.getDeclaredMethods()) {
            check(lookup, opConstants, m);
        }
    }

    static void check(MethodHandles.Lookup l, Predicate<Op> opConstants, Method method) throws ReflectiveOperationException {
        if (!method.isAnnotationPresent(CodeReflection.class)) {
            return;
        }

        for (EvaluatedModel em : getModels(method)) {
            CoreOp.FuncOp f = Op.ofMethod(method).orElseThrow(() ->
                    new AssertionError("No code model for reflective method"));
            f = evaluate(l, opConstants, f, em.ssa());

            String actual = canonicalizeModel(method, f);
            System.out.println(actual);
            String expected = canonicalizeModel(method, em.value());
            if (!actual.equals(expected)) {
                throw new AssertionError(String.format("Bad code model\nFound:\n%s\n\nExpected:\n%s", actual, expected));
            }
        }
    }

    static EvaluatedModel[] getModels(Method method) {
        EvaluatedModels ems = method.getAnnotation(EvaluatedModels.class);
        if (ems != null) {
            return ems.value();
        }

        EvaluatedModel em = method.getAnnotation(EvaluatedModel.class);
        if (em != null) {
            return new EvaluatedModel[] { em };
        }

        throw new AssertionError("No @EvaluatedModel annotation found on reflective method");
    }

    static CoreOp.FuncOp evaluate(MethodHandles.Lookup l, Predicate<Op> opConstants, CoreOp.FuncOp f, boolean ssa) {
        f = f.transform(OpTransformer.LOWERING_TRANSFORMER);

        if (ssa) {
            f = SSA.transform(f);
        }

        List<LoopAnalyzer.Loop> loops = LoopAnalyzer.findLoops(f.body());
        Map<Block, LoopAnalyzer.Loop> loopMap = loops.stream().collect(Collectors.toMap(LoopAnalyzer.Loop::header, loop -> loop));
        loops.forEach(System.out::println);
        Set<Value> constants = LoopAnalyzer.analyzeConstants(loopMap, opConstants, f);

        f = PartialEvaluator.evaluate(l, opConstants, constants, f);

        return cleanUp(f);
    }

    static CoreOp.FuncOp cleanUp(CoreOp.FuncOp f) {
        return removeUnusedOps(NormalizeBlocksTransformer.transform(f));
    }

    static CoreOp.FuncOp removeUnusedOps(CoreOp.FuncOp f) {
        Predicate<Op> unused = op -> (op instanceof Op.Pure || op instanceof CoreOp.VarOp) &&
                op.result().uses().isEmpty();
        while (f.elements().skip(1).anyMatch(ce -> ce instanceof Op op && unused.test(op))) {
            f = f.transform((block, op) -> {
                if (!unused.test(op)) {
                    block.op(op);
                }
                return block;
            });
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
