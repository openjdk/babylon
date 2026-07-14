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
package oracle.code.onnx.metadata;

import oracle.code.onnx.Tensor;
import org.junit.jupiter.api.Test;

import java.lang.reflect.Method;

import static org.junit.jupiter.api.Assertions.*;

class ModelMetadataResolverTest {

    record ForwardResponse(@Shape({1L, -1L, 64L}) Tensor<Float> logits,
                           @ElementShape(value = {1L, 8L, -1L, 64L}, count = 6) Tensor<Float>[] presentKey,
                           Tensor<Float>[] simpleOutput) {
    }

    static ForwardResponse forward(@Shape({1L, -1L})Tensor<Long> inputIds, @ElementShape(value = {1L, 8L, -1L, 64L}, count = 6) Tensor<Float>[] pastKey,
                                   Tensor<Long>[] simpleOutput) {
        return null;
    }

    @Test
    void testFromMethodParameters() throws NoSuchMethodException {
        Method method = ModelMetadataResolverTest.class.getDeclaredMethod("forward", Tensor.class, Tensor[].class, Tensor[].class);

        ModelMetadata modelMetadata = ModelMetadataResolver.from(method);

        assertEquals(2, modelMetadata.parameters().size());
        assertArrayEquals(new long[] {1L, -1L}, modelMetadata.parameters().get(0).shape());
        assertEquals(TensorMetadata.NO_ELEMENT_COUNT, modelMetadata.parameters().get(0).count());
        assertArrayEquals(new long[] {1L, 8L, -1L, 64L}, modelMetadata.parameters().get(1).shape());
        assertEquals(6, modelMetadata.parameters().get(1).count());
        assertFalse(modelMetadata.parameters().containsKey(2));

        assertEquals(2, modelMetadata.values().size());
        assertArrayEquals(new long[] {1L, -1L, 64L}, modelMetadata.values().get("logits").shape());
        assertEquals(TensorMetadata.NO_ELEMENT_COUNT, modelMetadata.values().get("logits").count());
        assertArrayEquals(new long[] {1L, 8L, -1L, 64L}, modelMetadata.values().get("presentKey").shape());
        assertEquals(6, modelMetadata.values().get("presentKey").count());
        assertFalse(modelMetadata.values().containsKey("simpleOutput"));
    }
}