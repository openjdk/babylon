/*
 * Copyright (c) 2026, Oracle and/or its affiliates. All rights reserved.
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

import java.io.IOException;
import java.lang.classfile.ClassFile;
import java.lang.classfile.MethodModel;
import java.lang.classfile.Opcode;
import java.lang.classfile.instruction.BranchInstruction;
import java.lang.invoke.MethodHandles;
import java.lang.reflect.Method;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.Comparator;
import java.util.Map;
import java.util.function.Function;
import java.util.regex.Pattern;
import java.util.stream.Collectors;
import java.util.stream.Stream;
import jdk.incubator.code.CodeTransformer;
import jdk.incubator.code.Op;
import jdk.incubator.code.Reflect;
import jdk.incubator.code.bytecode.BytecodeGenerator;
import jdk.incubator.code.dialect.core.CoreOp.FuncOp;
import jdk.incubator.code.dialect.core.SSA;
import jdk.test.lib.compiler.CompilerUtils;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;
/*
 * @test
 * @summary Compare bytecode generated from CFP shapes (direct, lowered, and SSA)
 * @library /test/lib
 * @modules jdk.compiler jdk.incubator.code/jdk.incubator.code.internal
 * @run junit TestControlFlowPreparation
 */
public class TestControlFlowPreparation {

    static final String TEST_CASES = """
            false
            true
            (!false)
            (!true)
            (false && false)
            (false && true)
            (false && boolean0)
            (true && false)
            (true && true)
            (boolean0 && false)
            (false || false)
            (false || true)
            (true || false)
            (true || true)
            (true || boolean0)
            (boolean0 || true)
            (false ? false : false)
            (false ? false : true)
            (false ? true : false)
            (false ? true : true)
            (false ? boolean0 : false)
            (false ? boolean0 : true)
            (true ? false : false)
            (true ? false : true)
            (true ? false : boolean0)
            (true ? true : false)
            (true ? true : true)
            (true ? true : boolean0)
            (boolean0 ? false : false)
            (boolean0 ? true : true)
            (!(!false))
            (!(!true))
            (!(boolean0 && false))
            (!(boolean0 || true))
            ((!false) && false)
            ((!false) && true)
            ((!false) && boolean0)
            ((!true) && false)
            ((!true) && true)
            ((!true) && boolean0)
            ((!boolean0) && false)
            (true && (!false))
            (true && (!true))
            (boolean0 && (!false))
            (boolean0 && (!true))
            ((false && boolean0) && boolean1)
            ((true && true) && false)
            ((true && true) && true)
            ((true && boolean0) && false)
            ((boolean0 && false) && boolean1)
            ((boolean0 && true) && false)
            ((boolean0 && boolean1) && false)
            ((boolean0 || true) && false)
            ((boolean0 || true) && true)
            ((boolean0 || boolean1) && false)
            (true && (boolean0 || true))
            ((false ? boolean0 : false) && boolean1)
            ((false ? boolean0 : true) && false)
            ((false ? boolean0 : true) && true)
            ((false ? boolean0 : boolean1) && false)
            ((true ? boolean0 : boolean1) && false)
            ((boolean0 ? false : false) && boolean1)
            ((boolean0 ? false : true) && false)
            ((boolean0 ? false : boolean1) && false)
            ((boolean0 ? true : false) && false)
            ((boolean0 ? true : true) && false)
            ((boolean0 ? true : true) && true)
            ((boolean0 ? true : boolean1) && false)
            ((boolean0 ? boolean1 : false) && false)
            ((boolean0 ? boolean1 : true) && false)
            ((boolean0 ? boolean1 : boolean2) && false)
            (false && (boolean0 ? boolean1 : boolean2))
            (true && (boolean0 ? false : false))
            (true && (boolean0 ? true : true))
            (boolean0 && (false ? boolean1 : false))
            (boolean0 && (boolean1 ? false : false))
            ((!false) || false)
            ((!false) || true)
            ((!false) || boolean0)
            ((!true) || false)
            ((!true) || true)
            ((!true) || boolean0)
            ((!boolean0) || true)
            (boolean0 || (!false))
            (boolean0 || (!true))
            ((boolean0 && false) || false)
            ((boolean0 && false) || true)
            ((boolean0 && boolean1) || true)
            (true || (boolean0 && boolean1))
            (boolean0 || (true && true))
            ((boolean0 || true) || boolean1)
            ((boolean0 || boolean1) || true)
            ((false ? boolean0 : true) || boolean1)
            ((false ? boolean0 : boolean1) || true)
            ((true ? boolean0 : boolean1) || true)
            ((boolean0 ? false : true) || true)
            ((boolean0 ? false : boolean1) || true)
            ((boolean0 ? true : false) || true)
            ((boolean0 ? true : true) || boolean1)
            ((boolean0 ? true : boolean1) || true)
            ((boolean0 ? boolean1 : false) || true)
            ((boolean0 ? boolean1 : true) || true)
            ((boolean0 ? boolean1 : boolean2) || true)
            (true || (boolean0 ? boolean1 : boolean2))
            (boolean0 || (false ? boolean1 : true))
            (boolean0 || (boolean1 ? true : true))
            ((!false) ? false : false)
            ((!false) ? false : true)
            ((!false) ? false : boolean0)
            ((!false) ? true : false)
            ((!false) ? true : true)
            ((!false) ? true : boolean0)
            ((!false) ? boolean0 : false)
            ((!false) ? boolean0 : true)
            ((!false) ? boolean0 : boolean1)
            ((!true) ? false : false)
            ((!true) ? false : true)
            ((!true) ? false : boolean0)
            ((!true) ? true : false)
            ((!true) ? true : true)
            ((!true) ? true : boolean0)
            ((!true) ? boolean0 : false)
            ((!true) ? boolean0 : true)
            ((!true) ? boolean0 : boolean1)
            ((!boolean0) ? false : false)
            ((!boolean0) ? true : true)
            (true ? (!false) : boolean0)
            (true ? (!true) : boolean0)
            (boolean0 ? (!false) : false)
            (boolean0 ? (!false) : true)
            (boolean0 ? (!false) : boolean1)
            (boolean0 ? (!true) : false)
            (boolean0 ? (!true) : true)
            (boolean0 ? (!true) : boolean1)
            (boolean0 ? false : (!false))
            (boolean0 ? false : (!true))
            (boolean0 ? true : (!false))
            (boolean0 ? true : (!true))
            (boolean0 ? boolean1 : (!false))
            (boolean0 ? boolean1 : (!true))
            ((boolean0 && false) ? boolean1 : false)
            ((boolean0 && false) ? boolean1 : true)
            ((boolean0 && boolean1) ? false : false)
            ((boolean0 && boolean1) ? true : true)
            ((boolean0 || true) ? false : boolean1)
            ((boolean0 || true) ? true : boolean1)
            ((boolean0 || boolean1) ? false : false)
            ((boolean0 || boolean1) ? true : true)
            ((false ? boolean0 : false) ? boolean1 : false)
            ((false ? boolean0 : false) ? boolean1 : true)
            ((false ? boolean0 : boolean1) ? false : false)
            ((false ? boolean0 : boolean1) ? true : true)
            ((true ? boolean0 : boolean1) ? false : false)
            ((true ? boolean0 : boolean1) ? true : true)
            ((boolean0 ? false : false) ? boolean1 : false)
            ((boolean0 ? false : false) ? boolean1 : true)
            ((boolean0 ? false : true) ? false : false)
            ((boolean0 ? false : true) ? true : true)
            ((boolean0 ? false : boolean1) ? false : false)
            ((boolean0 ? false : boolean1) ? true : true)
            ((boolean0 ? true : boolean1) ? false : false)
            ((boolean0 ? true : boolean1) ? true : true)
            ((boolean0 ? boolean1 : false) ? false : false)
            ((boolean0 ? boolean1 : false) ? true : true)
            ((boolean0 ? boolean1 : true) ? false : false)
            ((boolean0 ? boolean1 : true) ? true : true)
            ((boolean0 ? boolean1 : boolean2) ? false : false)
            ((boolean0 ? boolean1 : boolean2) ? true : true)
            (false ? (boolean0 ? boolean1 : boolean2) : false)
            (false ? (boolean0 ? boolean1 : boolean2) : true)
            (boolean0 ? (false ? boolean1 : false) : false)
            (boolean0 ? (false ? boolean1 : true) : true)
            (boolean0 ? (boolean1 ? false : false) : false)
            (boolean0 ? (boolean1 ? true : true) : true)
            (boolean0 ? false : (false ? boolean1 : false))
            (boolean0 ? false : (boolean1 ? false : false))
            (boolean0 ? true : (false ? boolean1 : true))
            (boolean0 ? true : (boolean1 ? true : true))
            (+1) == 1
            (-1) < 0
            (~1) == -2
            (2 * 3) == 6
            (6 / 2) == 3
            (7 % 3) == 1
            (1 + 2) == 3
            (3 - 1) == 2
            (1 << 2) == 4
            (4 >> 1) == 2
            (-1 >>> 31) == 1
            1 < 2
            1 <= 1
            2 > 1
            2 >= 2
            1 == 1
            1 != 2
            (3 & 1) == 1
            (1 ^ 3) == 2
            (1 | 2) == 3
            (true & false) == false
            (true ^ false) == true
            (true | false) == true
            ((int) 1.0) == 1
            ((String) "a") == "a"
            ("a" + 1) == "a1"
            \"""\\u000aa\\u000a\""" == "a\\n"
            Integer.BYTES == 4
            (true ? (byte) 1 : (byte) 2) == (byte) 1
            (false ? (byte) 1 : (byte) 2) == (byte) 1
            (boolean0 ? (byte) 1 : (byte) 1) == (byte) 1
            (false ? byte0 : (byte) 2) == (byte) 1
            (true ? (short) 1 : (short) 2) == (short) 1
            (false ? (short) 1 : (short) 2) == (short) 1
            (boolean0 ? (short) 1 : (short) 1) == (short) 1
            (false ? short0 : (short) 2) == (short) 1
            (true ? 'a' : 'b') == 'a'
            (false ? 'a' : 'b') == 'a'
            (boolean0 ? 'a' : 'a') == 'a'
            (false ? char0 : 'b') == 'a'
            (true ? 1 : 2) == 1
            (false ? 1 : 2) == 1
            (boolean0 ? 1 : 1) == 1
            (false ? int0 : 2) == 1
            (true ? 1L : 2L) == 1L
            (false ? 1L : 2L) == 1L
            (boolean0 ? 1L : 1L) == 1L
            (false ? long0 : 2L) == 1L
            (true ? 1.0F : 2.0F) == 1.0F
            (false ? 1.0F : 2.0F) == 1.0F
            (boolean0 ? 1.0F : 1.0F) == 1.0F
            (false ? float0 : 2.0F) == 1.0F
            (true ? 1.0 : 2.0) == 1.0
            (false ? 1.0 : 2.0) == 1.0
            (boolean0 ? 1.0 : 1.0) == 1.0
            (false ? double0 : 2.0) == 1.0
            (true ? "a" : "b") == "a"
            (false ? "a" : "b") == "a"
            (boolean0 ? "a" : "a") == "a"
            (false ? string0 : "b") == "a"
            int v0; if ((true ? 1 : 2) == 1) { v0 = 1; } return v0;
            int v0; if ((true ? "a" : "b") == "a") { v0 = 1; } return v0;
            final int v0 = 1 + Integer.BYTES; int v1; if (v0 == 5) { v1 = 1; } return v1;
            final String v0 = "a" + 1; int v1; if (v0 == "a1") { v1 = 1; } return v1;
            int v0; if (boolean0 && ((v0 = 1) > 0)) { return v0; } return 0;
            int v0; if (boolean0 || ((v0 = 1) > 0)) { return 0; } return v0;
            int v0; if (!(boolean0 && ((v0 = 1) > 0))) { return 0; } return v0;
            int v0; if (boolean0 ? ((v0 = 1) > 0) : false) { return v0; } return 0;
            int v0; while ((true ? 1 : 2) == 1) { v0 = 1; break; } return v0;
            int v0; do { v0 = 1; } while ((true ? 1 : 2) != 1); return v0;
            int v0; for (; (true ? 1 : 2) == 1;) { v0 = 1; break; } return v0;
            int v0; if (((true ? 1 : 2) == 1) && ((true ? 3 : 4) == 3)) { v0 = 1; } return v0;
            """;

    static final Class<?> CASE_CLASS = compileTestCases();

    static Class<?> compileTestCases() {
        try {
            var parameters = Pattern.compile("\\b(boolean|byte|short|char|int|long|float|double|string)(\\d+)\\b");
            var testCases = TEST_CASES.split("\n");
            for (int i = 0; i < testCases.length; i++) {
                String testCase = testCases[i];
                String signature = parameters.matcher(testCase).results()
                        .map(m -> (m.group(1).equals("string") ? "String" : m.group(1)) + " " + m.group())
                        .distinct().sorted().collect(Collectors.joining(", "));
                testCases[i] = "@Reflect\nstatic int case" + (i + 1) + "(" + signature + ") {\n"
                        + (testCase.contains(";") ? testCase.indent(4)
                        : "    if (((" + testCase + ") ? 1 : 0) == 1) return 1; else return 0;\n")
                        + "}";
            }
            Path sourceDirectory = Path.of("cfp-src");
            Files.createDirectories(sourceDirectory);
            Files.writeString(sourceDirectory.resolve("CFPTestCases.java"),
                    "import jdk.incubator.code.Reflect;\nclass CFPTestCases {\n"
                            + String.join("\n", testCases) + "}\n");
            Assertions.assertTrue(CompilerUtils.compile(sourceDirectory,
                    Path.of(System.getProperty("test.classes", ".")),
                    "--enable-preview", "--source", Integer.toString(Runtime.version().feature()),
                    "--add-modules", "jdk.incubator.code"), "Test cases compilation failed");
            return Class.forName("CFPTestCases");
        } catch (IOException | ClassNotFoundException e) {
            throw new RuntimeException(e);
        }
    }

    static Stream<Method> testMethods() {
        return Stream.of(CASE_CLASS.getDeclaredMethods())
                .filter(m -> m.isAnnotationPresent(Reflect.class))
                .sorted(Comparator.comparingInt(m -> Integer.parseInt(m.getName().substring(4))));
    }

    @ParameterizedTest
    @MethodSource("testMethods")
    void testSemanticMatch(Method method) throws Throwable {
        FuncOp model = Op.ofMethod(method).orElseThrow();
        var lookup = MethodHandles.lookup();
        var direct = BytecodeGenerator.generate(lookup, model);
        model = model.transform(CodeTransformer.LOWERING_TRANSFORMER);
        var lowered = BytecodeGenerator.generate(lookup, model);
        var lowered_ssa = BytecodeGenerator.generate(lookup, SSA.transform(model));
        var values = Map.of(
                boolean.class, new Object[]{false, true},
                byte.class, new Object[]{(byte) 0, (byte) 1, (byte) 2},
                short.class, new Object[]{(short) 0, (short) 1, (short) 2},
                char.class, new Object[]{' ', 'a', 'b'},
                int.class, new Object[]{0, 1, 2},
                long.class, new Object[]{0L, 1L, 2L},
                float.class, new Object[]{0F, 1F, 2F},
                double.class, new Object[]{0D, 1D, 2D},
                String.class, new Object[]{"a", "b", new String("a"), null});
        var pools = Arrays.stream(method.getParameterTypes()).map(values::get).toArray(Object[][]::new);
        var indices = new int[pools.length];
        var args = new Object[pools.length];
        while (true) {
            for (int i = 0; i < args.length; i++) args[i] = pools[i][indices[i]];
            var expected = method.invoke(null, args);
            Assertions.assertEquals(expected, direct.invokeWithArguments(args), "direct " + Arrays.toString(args));
            Assertions.assertEquals(expected, lowered.invokeWithArguments(args), "lowered " + Arrays.toString(args));
            Assertions.assertEquals(expected, ssa.invokeWithArguments(args), "SSA " + Arrays.toString(args));
            int i = args.length - 1;
            while (i >= 0 && ++indices[i] == pools[i].length) indices[i--] = 0;
            if (i < 0) break;
        }
    }

    static final Map<String, MethodModel> REFERENCE_METHODS = parseReferenceMethods();

    static Map<String, MethodModel> parseReferenceMethods(){
        try {
            return ClassFile.of().parse(CASE_CLASS.getResourceAsStream("CFPTestCases.class").readAllBytes())
                    .methods().stream().collect(Collectors.toMap(mm -> mm.methodName().stringValue(), Function.identity()));
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    @ParameterizedTest(name = "case{index}")
    @MethodSource("testMethods")
    void testConstantExpressionsCleanup(Method method) throws Throwable {
        FuncOp func = Op.ofMethod(method).orElseThrow();
        assertConstantExpressionsCleanup(func, "direct");
        assertConstantExpressionsCleanup(func = func.transform(CodeTransformer.LOWERING_TRANSFORMER), "lowered");
        assertConstantExpressionsCleanup(SSA.transform(func), "SSA");
    }

    static void assertConstantExpressionsCleanup(FuncOp func, String label) {
        MethodModel reference = REFERENCE_METHODS.get(func.funcName());
        MethodModel actual = ClassFile.of().parse(BytecodeGenerator.generateClassData(MethodHandles.lookup(), func)).methods().getFirst();
        // assert that our constant expression cleanup leaves no more conditional branches than javac
        Assertions.assertTrue(countConditionalBranches(actual) <= countConditionalBranches(reference),
                              () -> label + " conditional branches count exceeds the javac reference\n"
                                    + "actual:" + actual.toDebugString()
                                    + "javac reference:" + reference.toDebugString());
    }

    static int countConditionalBranches(MethodModel mm) {
        return (int)mm.code().orElseThrow().elementStream().filter(
                e -> e instanceof BranchInstruction bi
                        && bi.opcode() != Opcode.GOTO
                        && bi.opcode() != Opcode.GOTO_W).count();
    }
}
