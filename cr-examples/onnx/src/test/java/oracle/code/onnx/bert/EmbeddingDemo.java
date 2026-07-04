/*
 * Copyright (c) 2025, 2026, Oracle and/or its affiliates. All rights reserved.
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
package oracle.code.onnx.bert;

import java.lang.foreign.Arena;
import java.util.Arrays;

import oracle.code.onnx.bert.AllMiniLML6V2EmbeddingModel.Embedding;

public class EmbeddingDemo {

    public static void main(String[] args) throws Exception {
        String[] sentences = args.length == 0
                ? new String[]{"This is an example sentence", "Each sentence is converted"}
                : args;

        try (Arena arena = Arena.ofConfined()) {
            var modelInstance = new AllMiniLML6V2EmbeddingModel(arena);
            Embedding embedding = modelInstance.embed(arena, sentences);
            System.out.println("dims: [" + embedding.rows() + ", " + embedding.columns() + "]");
            for (int i = 0; i < embedding.rows(); i++) {
                float[] embeddings = embedding.row(i);
                System.out.println('"' + sentences[i] + '"');
                System.out.println(Arrays.toString(Arrays.copyOfRange(embeddings, 0, 12)) + " ...");
            }
        }
    }

}

