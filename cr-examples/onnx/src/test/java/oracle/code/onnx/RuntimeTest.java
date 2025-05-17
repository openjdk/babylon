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

package oracle.code.onnx;

import java.lang.foreign.Arena;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Test;

import static oracle.code.onnx.OnnxProtoBuilder.*;
import static oracle.code.onnx.proto.OnnxBuilder.*;
import static oracle.code.onnx.Tensor.ElementType.*;

import static org.junit.jupiter.api.Assertions.*;

public class RuntimeTest {

    @Test
    public void test() throws Exception {
        var ort = OnnxRuntime.getInstance();
        try (Arena arena = Arena.ofConfined()) {
            var absOp = ort.createSession(arena, buildModel(
                    List.of(),
                    List.of(tensorInfo("x", FLOAT.id)),
                    List.of(node("Abs", List.of("x"), List.of("y"), Map.of())),
                    List.of("y")));

            var addOp = ort.createSession(arena, buildModel(
                    List.of(),
                    List.of(tensorInfo("a", FLOAT.id), tensorInfo("b", FLOAT.id)),
                    List.of(node("Add", List.of("a", "b"), List.of("y"), Map.of())),
                    List.of("y")));

            assertEquals(1, absOp.getNumberOfInputs());
            assertEquals(1, absOp.getNumberOfOutputs());

            assertEquals(2, addOp.getNumberOfInputs());
            assertEquals(1, addOp.getNumberOfOutputs());

            var inputTensor = Tensor.ofFlat(arena, -1f, 2, -3, 4, -5, 6);

            var absExpectedTensor = Tensor.ofFlat(arena, 1f, 2, 3, 4, 5, 6);

            var absResult = absOp.run(arena, List.of(inputTensor));

            assertEquals(1, absResult.size());

            var absOutputTensor = absResult.getFirst();

            SimpleTest.assertEquals(absExpectedTensor, absOutputTensor);

            var addResult = addOp.run(arena, List.of(inputTensor, absOutputTensor));

            assertEquals(1, addResult.size());

            var addOutputTensor = addResult.getFirst();

            var addExpectedTensor = Tensor.ofFlat(arena, 0f, 4, 0, 8, 0, 12);

            SimpleTest.assertEquals(addExpectedTensor, addOutputTensor);
        }
    }

    @Test
    public void testIf() throws Exception {
        var ort = OnnxRuntime.getInstance();
        try (Arena arena = Arena.ofConfined()) {
            var ifOp = ort.createSession(arena, buildModel(
                    List.of(),
                    List.of(tensorInfo("cond", BOOL.id), tensorInfo("a", INT64.id), tensorInfo("b", INT64.id)),
                    List.of(node("If", List.of("cond"), List.of("y"), Map.of(
                            "then_branch", graph(
                                    null,
                                    List.of(),
                                    List.of(),
                                    List.of(node("Identity", List.of("a"), List.of("y"), Map.of())),
                                    List.of("y")),
                            "else_branch", graph(
                                    null,
                                    List.of(),
                                    List.of(),
                                    List.of(node("Identity", List.of("b"), List.of("y"), Map.of())),
                                    List.of("y"))))),
                    List.of("y")));

            var a = Tensor.ofScalar(arena, 1l);
            var b = Tensor.ofScalar(arena, 2l);
            SimpleTest.assertEquals(a, ifOp.run(arena, List.of(Tensor.ofScalar(arena, true), a, b)).getFirst());
            SimpleTest.assertEquals(b, ifOp.run(arena, List.of(Tensor.ofScalar(arena, false), a, b)).getFirst());
        }
    }

    @Test
    public void testLoop() throws Exception {
        var ort = OnnxRuntime.getInstance();
        try (Arena arena = Arena.ofConfined()) {
            var forOp = ort.createSession(arena, buildModel(
                    List.of(),
                    List.of(tensorInfo("max", INT64.id), tensorInfo("cond", BOOL.id), tensorInfo("a", INT64.id)),
                    List.of(node("Loop", List.of("max", "cond", "a"), List.of("a_out"), Map.of(
                            "body", graph(
                                    null,
                                    List.of(),
                                    List.of(tensorInfo("i", INT64.id, true), tensorInfo("cond_in", BOOL.id, true), tensorInfo("a_in", INT64.id)),
                                    List.of(node("Identity", List.of("cond_in"), List.of("cond_out"), Map.of()),
                                            node("Add", List.of("a_in", "a_in"), List.of("a_out"), Map.of())),
                                    List.of("cond_out", "a_out"))))),
                    List.of("a_out")));

            SimpleTest.assertEquals(Tensor.ofScalar(arena, 65536l),
                    forOp.run(arena, List.of(Tensor.ofScalar(arena, 15l), Tensor.ofScalar(arena, true), Tensor.ofScalar(arena, 2l))).getFirst());
        }
    }


    @Test
    public void testCustomFunction() throws Exception {
        String customDomain = RuntimeTest.class.getName();
        var ort = OnnxRuntime.getInstance();
        try (Arena arena = Arena.ofConfined()) {
            var customFunction = ort.createSession(arena, buildModel(
                    List.of(),
                    List.of(tensorInfo("x", INT64.id)),
                    List.of(node(customDomain + ".CustomFunction", List.of("x"), List.of("y"), Map.of())),
                    List.of("y"),
                    List.of(customDomain),
                    List.of(new FunctionProto()
                            .name("CustomFunction")
                            .input("a")
                            .output("b")
                            .node(node("Identity", List.of("a"), List.of("b"), Map.of()))
                            .opsetImport(new OperatorSetIdProto().version(OPSET_VERSION))
                            .domain(customDomain))));

            var a = Tensor.ofScalar(arena, 1l);
            SimpleTest.assertEquals(a, customFunction.run(arena, List.of(a)).getFirst());
        }
    }
}
