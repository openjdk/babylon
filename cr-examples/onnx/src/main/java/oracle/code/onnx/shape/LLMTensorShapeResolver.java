/*
 * Copyright (c) 2025, 2026, Oracle and/or its affiliates. All rights reserved.
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
package oracle.code.onnx.shape;

import java.util.List;

public record LLMTensorShapeResolver(long vocabsize, long keyValueHeads, long headSize, String inputIdsName, String attentionMask,
                                     String logitsName, String pastKeyTemplate, String pastValueTemplate, String presentKeyTemplate, String presentValueTemplate) implements  TensorShapeResolver {


    public LLMTensorShapeResolver(long vocabsize, long keyValueHeads, long headSize) {
        this(vocabsize, keyValueHeads, headSize,
        "inputIds", "attentionMask", "logits",
        "pastKey.%d", "pastValue.%d", "presentKey.%d", "presentValue.%d");
    }

    @Override
    public List<Object> shape(String name) {
        return switch (name) {
            case String shapeName when (shapeName.equals(inputIdsName) ||  shapeName.equals(attentionMask)) -> List.of("batch_size", "sequence_length");
            case String shapeName when (shapeName.equals(logitsName)) ->  List.of("batch_size", "sequence_length", vocabsize);
            case String shapeName when (shapeName.matches("past(Key|Value)\\.\\d+")) ->  List.of("batch_size", keyValueHeads, "past_sequence_length", headSize);
            case String shapeName when (shapeName.matches("present(Key|Value)\\.\\d+")) -> List.of("batch_size", keyValueHeads, "total_sequence_length", headSize);
            case String _ -> null;
        };
    }
}
