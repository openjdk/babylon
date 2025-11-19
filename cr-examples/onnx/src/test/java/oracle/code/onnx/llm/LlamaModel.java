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
package oracle.code.onnx.llm;

import java.io.IOException;
import java.lang.foreign.Arena;
import jdk.incubator.code.CodeReflection;
import oracle.code.onnx.Tensor;
import oracle.code.onnx.genai.TensorDataStream;

import static java.util.Optional.*;
import static oracle.code.onnx.OnnxOperators.*;
import static oracle.code.onnx.Tensor.ElementType.*;
import oracle.code.onnx.ir.OnnxType;

public final class LlamaModel {

    public static final int LAYERS = 16;
    public static final long BITS = 4,
                             BLOCK_SIZE = 32,
                             NUM_KEY_VALUE_HEADS = 8,
                             ACCURACY_LEVEL = 4,
                             VOCAB_SIZE = 128256,
                             HEAD_SIZE = 64,
                             HIDEN_SIZE = 2048,
                             CONTEXT_SIZE = 131072,
                             INTERMEDIATE_SIZE = 8192,
                             ATTN_WEIGHTS_SIZE = 3072;
    public static final float EPSILON = 1.0E-5f,
                              SCALE = 0.125f;

    public final Tensor<Long> flat1, scalar1;
    public final Tensor<Float> tokensWeights, initWeight, cosCache, sinCache, headScales;
    public final Tensor<Float>[] postAttentionWeights = new Tensor[LAYERS],
                                 inputWeights = new Tensor[LAYERS],
                                 attnQkvScales = new Tensor[LAYERS],
                                 attnOScales = new Tensor[LAYERS],
                                 mlpGateScales = new Tensor[LAYERS],
                                 mlpUpScales = new Tensor[LAYERS],
                                 mlpDownScales = new Tensor[LAYERS];
    public final Tensor<Byte>[] attnQkvWeight = new Tensor[LAYERS],
                                attnOWeight = new Tensor[LAYERS],
                                mlpGateWeight = new Tensor[LAYERS],
                                mlpUpWeight = new Tensor[LAYERS],
                                mlpDownWeight = new Tensor[LAYERS];
    public final Tensor<Byte> headWeight;

    public LlamaModel(Arena arena) throws IOException {
        flat1 = Tensor.ofFlat(arena, 1l);
        scalar1 = Tensor.ofScalar(arena, 1l);
        var modelData = new TensorDataStream(arena, LlamaModel.class.getResource("model.onnx.data").getPath());
        tokensWeights = modelData.nextTensor(FLOAT, VOCAB_SIZE, HIDEN_SIZE);
        initWeight = modelData.nextTensor(FLOAT, HIDEN_SIZE);
        cosCache = modelData.nextTensor(FLOAT, CONTEXT_SIZE, HEAD_SIZE / 2);
        sinCache = modelData.nextTensor(FLOAT, CONTEXT_SIZE, HEAD_SIZE / 2);
        for (int i = 0; i < LAYERS; i++) {
            postAttentionWeights[i] = modelData.nextTensor(FLOAT, HIDEN_SIZE);
            inputWeights[i] = modelData.nextTensor(FLOAT, HIDEN_SIZE);
        }
        for (int i = 0; i < LAYERS; i++) {
            attnQkvWeight[i] = modelData.nextTensor(UINT8, ATTN_WEIGHTS_SIZE, HEAD_SIZE, 16);
            attnQkvScales[i] = modelData.nextTensor(FLOAT, ATTN_WEIGHTS_SIZE * HEAD_SIZE);
            attnOWeight[i] = modelData.nextTensor(UINT8, HIDEN_SIZE, HEAD_SIZE, 16);
            attnOScales[i] = modelData.nextTensor(FLOAT, HIDEN_SIZE * HEAD_SIZE);
            mlpGateWeight[i] = modelData.nextTensor(UINT8, INTERMEDIATE_SIZE, HEAD_SIZE, 16);
            mlpGateScales[i] = modelData.nextTensor(FLOAT, INTERMEDIATE_SIZE * HEAD_SIZE);
            mlpUpWeight[i] = modelData.nextTensor(UINT8, INTERMEDIATE_SIZE, HEAD_SIZE, 16);
            mlpUpScales[i] = modelData.nextTensor(FLOAT, INTERMEDIATE_SIZE * HEAD_SIZE);
            mlpDownWeight[i] = modelData.nextTensor(UINT8, HIDEN_SIZE, 256, 16);
            mlpDownScales[i] = modelData.nextTensor(FLOAT, INTERMEDIATE_SIZE * HEAD_SIZE);
        }
        headWeight = modelData.nextTensor(UINT8, VOCAB_SIZE, HEAD_SIZE, 16);
        headScales = modelData.nextTensor(FLOAT, VOCAB_SIZE * HEAD_SIZE);
    }

    public record ForwardResponse(Tensor<Float> logits,
                                  Tensor<Float>[] presentKey,
                                  Tensor<Float>[] presentValue) {
    }

    @CodeReflection
    public ForwardResponse forward(Tensor<Long> inputIds, Tensor<Long> attentionMask, Tensor<Float>[] pastKey, Tensor<Float>[] pastValue) {

        Tensor<Integer> amSL = Cast(Sub(ReduceSum(attentionMask, of(flat1), empty(), empty()), flat1), empty(), OnnxType.INT32.id(), empty());
        Tensor<Integer> amTSL = Cast(Gather(Shape(attentionMask, empty(), empty()), scalar1, of(0l)), empty(), OnnxType.INT32.id(), empty());
        Tensor<Float> skipBias = Gather(tokensWeights, inputIds, empty());
        Tensor<Float> input = LayerNormalization(skipBias, initWeight, empty(), of(EPSILON), of(1l), of(-1l)).Y();

        Tensor<Float>[] presentKeys = new Tensor[LAYERS];
        Tensor<Float>[] presentValues = new Tensor[LAYERS];

        for (int i = 0; i < LAYERS; i++) {
            GroupQueryAttention<Float> attn = GroupQueryAttention(
                    MatMulNBits(input,
                                attnQkvWeight[i],
                                attnQkvScales[i], empty(), empty(), empty(), HIDEN_SIZE, ATTN_WEIGHTS_SIZE, of(ACCURACY_LEVEL), BITS, BLOCK_SIZE),
                    empty(),
                    empty(),
                    of(pastKey[i]),
                    of(pastValue[i]),
                    amSL,
                    amTSL,
                    of(cosCache),
                    of(sinCache), of(1l), NUM_KEY_VALUE_HEADS, empty(), BLOCK_SIZE, of(0l), of(SCALE));

            SkipSimplifiedLayerNormalization<Float> postAttnLayernorm = SkipSimplifiedLayerNormalization(
                    skipBias,
                    MatMulNBits(attn.output(),
                                attnOWeight[i],
                                attnOScales[i], empty(), empty(), empty(), HIDEN_SIZE, HIDEN_SIZE, of(ACCURACY_LEVEL), BITS, BLOCK_SIZE),
                    postAttentionWeights[i], empty(), of(EPSILON));

            Tensor<Float> mlpGateProj = MatMulNBits(postAttnLayernorm.output(),
                                                    mlpGateWeight[i],
                                                    mlpGateScales[i], empty(), empty(), empty(), HIDEN_SIZE, INTERMEDIATE_SIZE, of(ACCURACY_LEVEL), BITS, BLOCK_SIZE);

            SkipSimplifiedLayerNormalization<Float> norm = SkipSimplifiedLayerNormalization(postAttnLayernorm.input_skip_bias_sum(),
                    MatMulNBits(Mul(Mul(mlpGateProj,
                                        Sigmoid(mlpGateProj)),
                                    MatMulNBits(postAttnLayernorm.output(),
                                                mlpUpWeight[i],
                                                mlpUpScales[i], empty(), empty(), empty(), HIDEN_SIZE, INTERMEDIATE_SIZE, of(ACCURACY_LEVEL), BITS, BLOCK_SIZE)),
                                mlpDownWeight[i],
                                mlpDownScales[i], empty(), empty(), empty(), INTERMEDIATE_SIZE, HIDEN_SIZE, of(ACCURACY_LEVEL), BITS, BLOCK_SIZE),
                    inputWeights[i], empty(), of(EPSILON));

            input = norm.output();
            skipBias = norm.input_skip_bias_sum();
            presentKeys[i] = attn.present_key();
            presentValues[i] = attn.present_value();
        }

        Tensor<Float> logits = MatMulNBits(input,
                                           headWeight,
                                           headScales, empty(), empty(), empty(), HIDEN_SIZE, VOCAB_SIZE, of(ACCURACY_LEVEL), BITS, BLOCK_SIZE);

        return new ForwardResponse(logits, presentKeys, presentValues);
    }
}
