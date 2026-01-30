/*
 * Copyright (c) 2026, Oracle and/or its affiliates. All rights reserved.
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
package flashattention;

import hat.Accelerator;
import hat.ComputeContext;
import hat.HATMath;
import hat.KernelContext;
import hat.backend.Backend;
import hat.buffer.F16Array;
import hat.buffer.F32Array;
import hat.device.DeviceSchema;
import hat.device.DeviceType;
import hat.types.F16;
import jdk.incubator.code.Reflect;
import optkl.ifacemapper.MappableIface.RW;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.lang.invoke.MethodHandles;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.stream.IntStream;

import static hat.Accelerator.Compute;
import static hat.NDRange.Global1D;
import static hat.NDRange.Local1D;
import static hat.NDRange.NDRange1D;
import static optkl.ifacemapper.MappableIface.RO;
import static optkl.ifacemapper.MappableIface.WO;

/**
 * How to run?
 *
 * <p>
 * <code>
 * # Using the OpenCL Backend:
 * java -cp hat/job.jar hat.java run ffi-opencl flashattention <--verbose> <--size=SEQ_SIZE>
 * </code>
 * </p>
 *
 * <p>
 * <code>
 * # Using the CUDA Backend:
 * java -cp hat/job.jar hat.java run ffi-cuda flashattention <--verbose> <--size=SEQ_SIZE>
 * </code>
 * </p>
 *
 * <p>
 * This version of FlashAttention corresponds with a simplification of version 1 for the forward pass only from the
 * original article presented in this paper:
 * <a href="https://arxiv.org/pdf/2205.14135">FlashAttention: Fast and Memory-Efficient Exact Attention
 * with IO-Awareness</a>.
 * </p>
 *
 * <p>
 * This version is used for demonstration purposes. It computes the self-attention on CPU with Java and on GPU with HAT.
 * Then, it implements a simplified version of flash-attention (single-head, in FP32, with no tensors).
 * This example demonstrates how to perform loop-tiling and private/shared-memory to improve performance with a
 * well-known kernel in the LLM and NLP domains.
 * </p>
 */
public class Main {

    public static final int ITERATIONS = 100;

    /**
     * Computes self-attention with HAT in a 1D parallel kernel. It fuses:
     * - Matmul: Q @ K^T with a scale factor
     * - Softmax
     * - Final matmul: attention ^ V
     * In a single kernel. But it does not apply the techniques for the self-attention using tiling and shared-memory.
     *
     * @param kernelContext
     * @param Q
     * @param K
     * @param V
     * @param attentionMatrix
     * @param O
     * @param N
     * @param d
     * @param softMaxScale
     */
    @Reflect
    public static void selfAttentionV2HAT(@RO KernelContext kernelContext,
                                          @RO F32Array Q, @RO F32Array K, @RO F32Array V,
                                          @WO F32Array attentionMatrix, @WO F32Array O,
                                          @RO final int N, @RO final int d, @RO final float softMaxScale) {
        int idx = kernelContext.gix;
        if (idx < N) {
            // Compute the attention scores: Q * K^T and scale it to sqrt(d) => softMaxScale
            for (int j = 0; j < N; j++) {
                float acc = 0.0f;
                for (int k = 0; k < d; k++) {
                    acc += Q.array(idx * d + k) * K.array(j * d + k);
                }
                // multiply by the scale factor
                acc *= softMaxScale;
                // store partial results in the temporary matrix for flash-attention
                attentionMatrix.array(idx * N + j, acc);
            }

            // SoftMax: apply softmax function to the attention score to normalize them
            float maxVal = Float.MIN_VALUE;
            // Compute max
            for (int j = 0; j < N; j++) {
                maxVal = Math.max(maxVal, attentionMatrix.array(idx * N + j));
            }
            // Compute exp()
            float sum = 0.0f;
            for (int j = 0; j < N; j++) {
                float p = (float) Math.exp(attentionMatrix.array(idx * N + j) - maxVal);
                attentionMatrix.array(idx * N + j, p);
                sum += p;
            }

            // normalization by the sum compute in the prev. step
            for (int j = 0; j < N; j++) {
                float val = attentionMatrix.array(idx * N + j) / sum;
                attentionMatrix.array(idx * N + j, val);
            }

            // Final matmul: O = attention * V
            for (int j = 0; j < d; j++) {
                float acc = 0.0f;
                for (int k = 0; k < N; k++) {
                    acc += attentionMatrix.array(idx * N + k) * V.array(k * d + j);
                }
                O.array(idx * d + j, acc);
            }
        }
    }

    public static void selfAttentionV2(F32Array Q, F32Array K, F32Array V,
                                       F32Array attentionMatrix, F32Array O,
                                       final int N, final int d, final float softMaxScale) {

        // Compute the attention scores: Q * K^T and scale it to sqrt(d) => softMaxScale
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                float acc = 0.0f;
                for (int k = 0; k < d; k++) {
                    acc += Q.array(i * d + k) * K.array(j * d + k);
                }
                // multiply by the scale factor
                acc *= softMaxScale;
                // store partial results in the temporary matrix for flash-attention
                attentionMatrix.array(i * N + j, acc);
            }

            // SoftMax: apply softmax function to the attention score to normalize them
            float max = Float.MIN_VALUE;
            // Compute max
            for (int j = 0; j < N; j++) {
                max = Math.max(max, attentionMatrix.array(i * N + j));
            }
            float sum = 0.0f;
            // Compute exp()
            for (int j = 0; j < N; j++) {
                float p = (float) Math.exp(attentionMatrix.array(i * N + j) - max);
                attentionMatrix.array(i * N + j, p);
                sum += p;
            }
            // normalization by the sum compute in the prev. step
            for (int j = 0; j < N; j++) {
                float val = attentionMatrix.array(i * N + j) / sum;
                attentionMatrix.array(i * N + j, val);
            }

            // Final matmul: O = attention * V
            for (int j = 0; j < d; j++) {
                float acc = 0.0f;
                for (int k = 0; k < N; k++) {
                    acc += attentionMatrix.array(i * N + k) * V.array(k * d + j);
                }
                O.array(i * d + j, acc);
            }
        }
    }

    public static void selfAttentionStreamsV2(F32Array Q, F32Array K, F32Array V,
                                       F32Array attentionMatrix, F32Array O,
                                       final int N, final int d, final float softMaxScale) {

        // Compute the attention scores: Q * K^T and scale it to sqrt(d) => softMaxScale
        IntStream.range(0, N).parallel().forEach(i -> {
        //for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                float acc = 0.0f;
                for (int k = 0; k < d; k++) {
                    acc += Q.array(i * d + k) * K.array(j * d + k);
                }
                // multiply by the scale factor
                acc *= softMaxScale;
                // store partial results in the temporary matrix for flash-attention
                attentionMatrix.array(i * N + j, acc);
            }

            // SoftMax: apply softmax function to the attention score to normalize them
            float max = Float.MIN_VALUE;
            // Compute max
            for (int j = 0; j < N; j++) {
                max = Math.max(max, attentionMatrix.array(i * N + j));
            }
            float sum = 0.0f;
            // Compute exp()
            for (int j = 0; j < N; j++) {
                float p = (float) Math.exp(attentionMatrix.array(i * N + j) - max);
                attentionMatrix.array(i * N + j, p);
                sum += p;
            }
            // normalization by the sum compute in the prev. step
            for (int j = 0; j < N; j++) {
                float val = attentionMatrix.array(i * N + j) / sum;
                attentionMatrix.array(i * N + j, val);
            }

            // Final matmul: O = attention * V
            for (int j = 0; j < d; j++) {
                float acc = 0.0f;
                for (int k = 0; k < N; k++) {
                    acc += attentionMatrix.array(i * N + k) * V.array(k * d + j);
                }
                O.array(i * d + j, acc);
            }
        });
    }

    /**
     * Single-head scale-dot product attention. It is currently not invoked. We just keep it as a reference.
     *
     * @param Q
     * @param K
     * @param V
     * @param attentionMatrix
     * @param O
     * @param N
     * @param d
     * @param softMaxScale
     */
    public static void selfAttention(F32Array Q, F32Array K, F32Array V,
                                     F32Array attentionMatrix, F32Array O,
                                     final int N, final int d, final float softMaxScale) {

        // Compute attention scores: Q @ K^T and scale it to (1/sqrt(head_dim))
        // In this example, the parameter d already computes 1/sqrt(head_dim)
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                float acc = 0.0f;
                for (int k = 0; k < d; k++) {
                    acc += Q.array(i * d + k) * K.array(j * d + k);
                }
                acc *= softMaxScale;
                attentionMatrix.array(i * N + j, acc);
            }
        }

        // SoftMax: apply softmax function to the attention score to normalize them
        for (int i = 0; i < N; i++) {
            // Compute max
            float max = Float.MIN_VALUE;
            for (int j = 0; j < N; j++) {
                max = Math.max(max, attentionMatrix.array(i * N + j));
            }

            // exp(attention[i][j] - max)
            // compute total sum
            float sum = 0.0f;
            for (int j = 0; j < N; j++) {
                float p = (float) Math.exp(attentionMatrix.array(i * N + j) - max);
                attentionMatrix.array(i * N + j, p);
                sum += p;
            }

            // normalize:
            // attention[i][j] /= sum
            for (int j = 0; j < N; j++) {
                float val = attentionMatrix.array(i * N + j) / sum;
                attentionMatrix.array(i * N + j, val);
            }
        }

        // Final matmul: O = attention @ V
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < d; j++) {
                float acc = 0.0f;
                for (int k = 0; k < N; k++) {
                    acc += attentionMatrix.array(i * N + k) * V.array(k * d + j);
                }
                O.array(i * d + j, acc);
            }
        }
    }

    @Reflect
    public static void selfAttentionCompute(@RO ComputeContext computeContext, @RO F32Array Q, @RO F32Array K, @RO F32Array V,
                                            @WO F32Array attentionMatrix, @WO F32Array O,
                                            final int N, final int d, final float softmaxScale) {
        var ndRange = NDRange1D.of(Global1D.of(N), Local1D.of(256));
        computeContext.dispatchKernel(ndRange, kernelContext -> selfAttentionV2HAT(kernelContext, Q, K, V, attentionMatrix, O, N, d, softmaxScale));
    }

    // Express a float array in shared memory with HAT
    private interface SharedFloatArray extends DeviceType {
        void array(long index, float value);

        float array(long index);

        DeviceSchema<SharedFloatArray> schema = DeviceSchema.of(SharedFloatArray.class,
                arr -> arr
                        // final int sharedMemorySize = block_m * head_dim
                        //                + block_n * head_dim
                        //                + block_n * head_dim
                        //                + block_m * block_n;
                        .withArray("array", 7168));

        static SharedFloatArray createLocal() {
            return null;
        }
    }

    // Express an array of floats in private memory with HAT
    private interface PrivateFloatArray extends DeviceType {
        void array(long index, float value);

        float array(long index);

        DeviceSchema<PrivateFloatArray> schema = DeviceSchema.of(PrivateFloatArray.class,
                arr -> arr
                        // SIZE = HEAD_DIM (e.g., 64)
                        .withArray("array", 64));

        static PrivateFloatArray createPrivate() {
            return null;
        }
    }

    @Reflect
    public static int ceilFunction(int N, int blockN) {
        return (N + blockN - 1) / blockN;
    }

    /**
     * Flash-attention: simplification of the original paper {@url https://arxiv.org/abs/2205.14135}
     * using a single-head. It computes the local max and sums using the m and l arrays.
     *
     * <p>This example is mainly used for illustration purposes, showcasing how to achieve
     * a naive version of flash-attention using tiling, private and shared memory on the GPU
     * with HAT.</p>
     *
     * @param kernelContext
     * @param Q
     * @param K
     * @param V
     * @param O
     * @param m
     * @param l
     * @param N
     * @param d
     * @param softmaxScale
     */
    @Reflect
    public static void flashAttention(@RO KernelContext kernelContext,
                                      @RO F32Array Q, @RO F32Array K, @RO F32Array V,
                                      @WO F32Array O, @RW F32Array m, @RW F32Array l,
                                      final int N, final int d, final float softmaxScale) {
        int bx = kernelContext.bix;
        int tid = kernelContext.lix;

        // Parameters used
        final int headDim = 64;
        final int blockM = 32;
        final int blockN = 32;

        int startIndex = bx * blockM;

        // We use a unique space in shared memory to compute matrices
        // Q, K, V and the intermediate one (S).
        // The way we distinguish the matrices are by using different
        // indexes.
        SharedFloatArray sharedArray = SharedFloatArray.createLocal();
        int sQ_index = 0;
        int baseIndex = blockN * headDim;
        int sK_index = baseIndex;
        int sV_index = baseIndex * 2;
        int sS_index = baseIndex * 3;

        // Load Q into shared memory (sQ_index)
        for (int k = 0; k < d; k++) {
            sharedArray.array((tid * d + k) + sQ_index,
                    Q.array((startIndex + (tid * d + k) * d + k)));
        }
        kernelContext.barrier();

        int numBlocks = ceilFunction(N, blockN);
        for (int tileId = 0; tileId < numBlocks; tileId++) {

            int kvTileRow = (tileId * blockN) + tid;

            // Load the tiles K and V into shared memoru
            for (int k = 0; k < d; k++) {
                sharedArray.array((tid * d + k) + sK_index, K.array(kvTileRow * d + k));
                sharedArray.array((tid + d + k) + sV_index, V.array(kvTileRow * d + k));
            }
            kernelContext.barrier();

            // m we accumulate the max values
            float m_prev = m.array(tileId * blockN + tid);
            // in l we accumulate the sum values
            float l_prev = l.array(tileId * blockN + tid);
            float m_block = Float.MIN_VALUE; // for calculating max
            float l_block = 0.0f; // for sum

            // Compute attention scores: S = Qi @ Kj^T * scale
            // Then: rowmax(m_block)
            PrivateFloatArray privateFloatArray = PrivateFloatArray.createPrivate();
            for (int t = 0; t < blockN; t++) {
                float score = 0.0f;
                for (int k = 0; k < d; k++) {
                    score += sharedArray.array((tid * d + k) + sQ_index)
                            * sharedArray.array((t * d + k) + sK_index);
                }
                score *= softmaxScale;
                privateFloatArray.array((t) + sS_index, score);
                m_block = Math.max(m_block, score);
            }

            // Compute local sum of Math.exp(p_i - m_block)
            for (int t = 0; t < blockN; t++) {
                float p = (float) Math.exp(privateFloatArray.array(t) - m_block);
                privateFloatArray.array(t, p);
                l_block += p;
            }

            // Update m and l with the new values
            float m_new = Math.max(m_prev, m_block);
            float l_new = (float) (Math.exp(m_prev - m_new) * l_prev + Math.exp(m_block - m_new) * l_block);

            // Update the Output (O)
            for (int k = 0; k < d; k++) {
                float pv = 0.0f;
                for (int t = 0; t < blockN; t++) {
                    // MMA: P @ V (V in shared memory)
                    pv += privateFloatArray.array(t) * sharedArray.array(t * d + k + sV_index);
                }

                // compute the output value using the formula:
                // diag(l_new)^-1 (diag(l_prev)*exp(m_prev-m_new) * O(current) + exp(m_block - m_new) * pv
                int oldIndex = startIndex + (tileId * blockN + tid) * d + k;
                int oIndex = (bx * blockN + tid) * d + k;
                float value = O.array(oIndex);
                float outVal = (float) ((l_prev * Math.exp(m_prev - m_new) * value +
                        Math.exp(m_block - m_new) * pv) / l_new);
                // write output
                O.array(oIndex, outVal);
            }

            // update m and l in global memory
            m.array(tileId * blockN + tid, m_new);
            l.array(tileId * blockN + tid, l_new);

            kernelContext.barrier();
        }
    }

    @Reflect
    public static void computeFlashAttention(@RO ComputeContext computeContext,
                                             @RO F32Array Q, @RO F32Array K, @RO F32Array V,
                                             @WO F32Array O, @RW F32Array m, @RW F32Array l,
                                             final int N, final int d, final float scale, final int blockSize) {
        var ndRange = NDRange1D.of(Global1D.of(N), Local1D.of(blockSize));
        computeContext.dispatchKernel(ndRange, kernelContext -> flashAttention(kernelContext, Q, K, V, O, m, l, N, d, scale));
    }

    private interface SharedF16Array extends DeviceType {
        F16 array(int index);

        DeviceSchema<SharedF16Array> schema = DeviceSchema.of(SharedF16Array.class,
                // final int sharedMemorySize = block_m * head_dim
                //                + block_n * head_dim
                //                + block_n * head_dim
                //                + block_m * block_n;
                arr -> arr.withArray("array", 7168)
                .withDeps(F16.class, half -> half.withField("value")));

        static SharedF16Array createLocal() {
            return null;
        }
    }

    private interface PrivateF16Array extends DeviceType {

        F16 array(int index);

        DeviceSchema<PrivateF16Array> schema = DeviceSchema.of(PrivateF16Array.class,
                // SIZE = HEAD_DIM (e.g., 64)
                arr -> arr.withArray("array", 64)
                 .withDeps(F16.class, half -> half.withField("value")));

        static PrivateF16Array createPrivate() {
            return null;
        }
    }

    @Reflect
    public static void flashAttentionF16(@RO KernelContext kernelContext,
                                      @RO F16Array Q, @RO F16Array K, @RO F16Array V,
                                      @WO F16Array O, @RW F16Array m, @RW F16Array l,
                                      final int N, final int d, final float softmaxScale) {
        int bx = kernelContext.bix;
        int tid = kernelContext.lix;

        // Parameters used
        final int headDim = 64;
        final int blockM = 32;
        final int blockN = 32;

        int startIndex = bx * blockM;

        // scaling factor
        F16 scale = F16.of(softmaxScale);

        // We use a unique space in shared memory to compute matrices
        // Q, K, V and the intermediate one (S).
        // The way we distinguish the matrices are by using different
        // indexes.
        SharedF16Array sharedArray = SharedF16Array.createLocal();
        int sQ_index = 0;
        int baseIndex = blockN * headDim;
        int sK_index = baseIndex;
        int sV_index = baseIndex * 2;
        int sS_index = baseIndex * 3;

        // Load Q into shared memory (sQ_index)
        for (int k = 0; k < d; k++) {
            F16 valQ = Q.array((startIndex + (tid * d + k) * d + k));
            sharedArray.array((tid * d + k) + sQ_index).value(valQ.value());
        }

        kernelContext.barrier();

        int numBlocks = ceilFunction(N, blockN);
        for (int tileId = 0; tileId < numBlocks; tileId++) {

            int kvTileRow = (tileId * blockN) + tid;

            // Load the tiles K and V into shared memory
            for (int k = 0; k < d; k++) {
                F16 kVal = K.array(kvTileRow * d + k);
                F16 vVal = V.array(kvTileRow * d + k);
                sharedArray.array((tid * d + k) + sK_index).value(kVal.value());
                sharedArray.array((tid + d + k) + sV_index).value(vVal.value());
            }
            kernelContext.barrier();

            // m we accumulate the max values
            F16 m_prev = m.array(tileId * blockN + tid);
            // in l we accumulate the sum values
            F16 l_prev = l.array(tileId * blockN + tid);
            F16 m_block = F16.of(-100f); // for calculating max
            F16 l_block = F16.of(0.0f); // for sum

            // Compute attention scores: S = Qi @ Kj^T * scale
            // Then: rowmax(m_block)
            PrivateF16Array privateFloatArray = PrivateF16Array.createPrivate();
            for (int t = 0; t < blockN; t++) {
                F16 score = F16.of(0.0f);
                for (int k = 0; k < d; k++) {
                    F16 valQ = sharedArray.array((tid * d + k) + sQ_index);
                    F16 valK = sharedArray.array((t * d + k) + sK_index);
                    F16 mul = F16.mul(valQ, valK);
                    score = F16.add(score, mul);
                }
                score = F16.mul(score, scale);
                privateFloatArray.array((t) + sS_index).value(score.value());
                m_block = HATMath.max(m_block, score);
            }

            // Compute local sum of Math.exp(p_i - m_block)
            for (int t = 0; t < blockN; t++) {
                F16 privateVal = privateFloatArray.array(t);
                F16 sub = F16.sub(privateVal, m_block);
                F16 p = HATMath.exp(sub);
                privateFloatArray.array(t).value(p.value());
                l_block = F16.add(l_block, p);
            }

            // Update m and l with the new values
            F16 m_new = HATMath.max(m_prev, m_block);

            //F16 l_new = (float) (HATMath.exp(m_prev - m_new) * l_prev + HATMath.exp(m_block - m_new) * l_block);
            F16 exp1 = HATMath.exp(F16.sub(m_prev , m_new));
            F16 mul1 = F16.mul(exp1, l_prev);
            F16 exp2 = HATMath.exp(F16.sub(m_block , m_new));
            F16 mul2 = F16.mul(exp2, l_block);
            F16 l_new = F16.add(mul1, mul2);

            // Update the Output (O)
            for (int k = 0; k < d; k++) {
                F16 pv = F16.of(0.0f);
                for (int t = 0; t < blockN; t++) {
                    // MMA: P @ V (V in shared memory)
                    F16 aux1 = sharedArray.array(t * d + k + sV_index);
                    F16 aux2 = privateFloatArray.array(t);
                    F16 mul = F16.mul(aux1, aux2);
                    pv = F16.add(pv, mul);
                }

                // compute the output value using the formula:
                // diag(l_new)^-1 (diag(l_prev)*exp(m_prev-m_new) * O(current) + exp(m_block - m_new) * pv
                int oIndex = (bx * blockN + tid) * d + k;
                F16 value = O.array(oIndex);

                //F16 outVal = ((l_prev * Math.exp(m_prev - m_new) * value + Math.exp(m_block - m_new) * pv) / l_new);

                F16 expOut1 = HATMath.exp(F16.sub(m_prev, m_new));
                F16 multOut1 = F16.mul(l_prev, expOut1);
                multOut1 = F16.mul(multOut1, value);

                F16 expOut2 = HATMath.exp(F16.sub(m_block, m_new));
                F16 multOut2 = F16.mul(expOut2, pv);
                F16 addOut = F16.add(multOut1, multOut2);
                F16 outVal = F16.div(addOut, l_new);

                // write output
                O.array(oIndex).value(outVal.value());
            }

            // update m and l in global memory
            m.array(tileId * blockN + tid).value(m_new.value());
            l.array(tileId * blockN + tid).value(l_new.value());

            kernelContext.barrier();
        }
    }

    @Reflect
    public static void computeFlashAttentionF16(@RO ComputeContext computeContext,
                                                @RO F16Array Q, @RO F16Array K, @RO F16Array V,
                                                @WO F16Array O, @RW F16Array m, @RW F16Array l,
                                                final int N, final int d, final float scale, final int blockSize) {
        var ndRange = NDRange1D.of(Global1D.of(N), Local1D.of(blockSize));
        computeContext.dispatchKernel(ndRange, kernelContext -> flashAttentionF16(kernelContext, Q, K, V, O, m, l, N, d, scale));
    }

    public static boolean checkResult(F32Array O_reference, F32Array O, final int matrixSize) {
        for (int i = 0; i < matrixSize; i++) {
            if (Math.abs(O_reference.array(i) - O.array(i)) > 0.1f) {
                IO.println("Iteration: #" + i + " " + O_reference.array(i) + " != " + O.array(i));
                return false;
            }
        }
        return true;
    }

    public static void dumpStatsToCSVFile(List<List<Long>> listOfTimers, List<String> header, final String fileName) {
        final int numColumns = listOfTimers.size();
        if (numColumns != header.size()) {
            throw new RuntimeException("Header size and List of timers need to be the same size");
        }
        StringBuilder builder = new StringBuilder();
        IntStream.range(0, header.size()).forEach(i -> {
            builder.append(header.get(i));
            if (i != header.size() - 1) {
                builder.append(",");
            }
        });
        builder.append(System.lineSeparator());

        final int numRows = listOfTimers.getFirst().size();
        for (int row = 0; row < numRows; row++) {
            for (int col = 0; col < numColumns; col++) {
                // all lists must be of the same size:
                if (listOfTimers.get(col).size() != numRows) {
                    throw new RuntimeException("[ERROR] Result List: " + col + " has a different size");
                }
                Long timer = listOfTimers.get(col).get(row);
                builder.append(timer);
                if (col != header.size() - 1) {
                    builder.append(",");
                }
            }
            builder.append(System.lineSeparator());
        }
        builder.append(System.lineSeparator());

        IO.println("[INFO] Saving results into file: " + fileName);
        try(BufferedWriter writer = new BufferedWriter(new FileWriter(fileName))) {
            writer.append(builder.toString());
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    public static double computeAverage(List<Long> timers, int discard) {
        double sum = timers.stream().skip(discard).reduce(0L, Long::sum).doubleValue();
        int totalCountedValues = timers.size() - discard;
        return (sum / totalCountedValues);
    }

    private static double computeSpeedup(double baseline, double measured) {
        return (Math.ceil(baseline / measured * 100) / 100);
    }

    @Reflect
    static void main(String[] args) {
        IO.println("Example of Flash-Attention in HAT");

        var lookup = MethodHandles.lookup();
        var accelerator = new Accelerator(lookup, Backend.FIRST);

        boolean verbose = false;
        int size = 512;
        // process parameters
        for (String arg : args) {
            if (arg.equals("--verbose")) {
                verbose = true;
                IO.println("Verbose mode on? " + verbose);
            } else if (arg.startsWith("--size=")) {
                String number = arg.split("=")[1];
                try {
                    size = Integer.parseInt(number);
                } catch (NumberFormatException _) {
                }
            }
        }
        IO.println("Sequence Length = " + size);

        // Configuration parameters
        final int sequenceLen = size;   // represent the number of tokens (or words)
        //final int sequenceLen = 16384;   // represent the number of tokens (or words)
        final int headDim = 64;        // vector representation for a single token
        final int blockM = 32;         // tile size
        final int blockN = 32;         // tile size
        final float softmaxScale = (float) (1.0f / Math.sqrt(headDim));

        final int sharedMemorySize = blockM * headDim
                + blockN * headDim
                + blockN * headDim
                + blockM * blockN;
        IO.println("Shared Array Size: " + sharedMemorySize);
        IO.println("");

        // Inputs preparation
        final int matrixSize = sequenceLen * headDim;
        var Q = F32Array.create(accelerator, matrixSize);
        var K = F32Array.create(accelerator, matrixSize);
        var V = F32Array.create(accelerator, matrixSize);
        var m = F32Array.create(accelerator, matrixSize);
        var l = F32Array.create(accelerator, matrixSize);
        var O_java = F32Array.create(accelerator, matrixSize);
        var O_streams = F32Array.create(accelerator, matrixSize);
        var O_selfAttention = F32Array.create(accelerator, matrixSize);
        var O_flashAttention = F32Array.create(accelerator, matrixSize);

        var Q16 = F16Array.create(accelerator, matrixSize);
        var K16 = F16Array.create(accelerator, matrixSize);
        var V16 = F16Array.create(accelerator, matrixSize);
        var m16 = F16Array.create(accelerator, matrixSize);
        var l16 = F16Array.create(accelerator, matrixSize);
        var O_flashAttention16 = F16Array.create(accelerator, matrixSize);

        F32Array attentionMatrix = F32Array.create(accelerator, sequenceLen * sequenceLen);

        // Initialize matrices with random values
        // In the real-world, this will be calculated from the input embeddings
        Random r = new Random(71);
        for (int i = 0; i < matrixSize; i++) {
            Q.array(i, r.nextFloat(1));
            K.array(i, r.nextFloat(1));
            V.array(i, r.nextFloat(1));

            Q16.array(i).value(F16.floatToF16(Q.array(i)).value());
            K16.array(i).value(F16.floatToF16(K.array(i)).value());
            V16.array(i).value(F16.floatToF16(V.array(i)).value());

        }

        IntStream.range(0, m.length()).forEach(k -> m.array(k, 0.0f));
        IntStream.range(0, l.length()).forEach(k -> l.array(k, 1.0f));

        IntStream.range(0, m.length()).forEach(k -> m16.array(k).value(F16.of(0.0f).value()));
        IntStream.range(0, m.length()).forEach(k -> l16.array(k).value(F16.of(1.0f).value()));

        List<Long> timersSelfAttentionJava = new ArrayList<>();
        List<Long> timersSelfAttentionStream = new ArrayList<>();
        List<Long> timersSelfAttentionHAT = new ArrayList<>();
        List<Long> timersFlashAttentionHAT = new ArrayList<>();
        List<Long> timersFlashAttentionHAT16 = new ArrayList<>();

        // Run the CPU version with Java:
        for (int i = 0; i < ITERATIONS; i++) {
            long start = System.nanoTime();
            selfAttentionV2(Q, K, V, attentionMatrix, O_java, sequenceLen, headDim, softmaxScale);
            long end = System.nanoTime();
            timersSelfAttentionJava.add((end - start));
            if (verbose) {
                IO.println("Java-Sequential (FP32) elapsed time: " + (end - start) + " ns");
            }
        }

        // Run the Parallel Stream version with Java:
        for (int i = 0; i < ITERATIONS; i++) {
            long start = System.nanoTime();
            selfAttentionStreamsV2(Q, K, V, attentionMatrix, O_streams, sequenceLen, headDim, softmaxScale);
            long end = System.nanoTime();
            timersSelfAttentionStream.add((end - start));
            if (verbose) {
                IO.println("Java-Parallel-Stream (FP32) elapsed time: " + (end - start) + " ns");
            }
        }

        // Run self-attention algorithm with HAT
        for (int i = 0; i < ITERATIONS; i++) {
            long start = System.nanoTime();
            accelerator.compute((@Reflect Compute)
                    cc -> Main.selfAttentionCompute(
                            cc,
                            Q,
                            K,
                            V,
                            attentionMatrix,
                            O_selfAttention,
                            sequenceLen,
                            headDim,
                            softmaxScale));

            long end = System.nanoTime();
            timersSelfAttentionHAT.add((end - start));
            if (verbose) {
                IO.println("HAT-Self-Attention (FP32) elapsed time: " + (end - start) + " ns");
            }
        }

        // Run flashAttention in HAT
        for (int i = 0; i < ITERATIONS; i++) {
            long start = System.nanoTime();
            accelerator.compute((@Reflect Compute)
                    cc -> Main.computeFlashAttention(
                            cc,
                            Q,
                            K,
                            V,
                            O_flashAttention,
                            m, l,
                            sequenceLen,
                            headDim,
                            softmaxScale,
                            blockM));

            long end = System.nanoTime();
            timersFlashAttentionHAT.add((end - start));
            if (verbose) {
                IO.println("HAT-Flash-Attention (FP32) elapsed time: " + (end - start) + " ns");
            }
        }

        // Run flashAttention in HAT
        for (int i = 0; i < ITERATIONS; i++) {
            long start = System.nanoTime();
            accelerator.compute((@Reflect Compute)
                    cc -> Main.computeFlashAttentionF16(
                            cc,
                            Q16,
                            K16,
                            V16,
                            O_flashAttention16,
                            m16, l16,
                            sequenceLen,
                            headDim,
                            softmaxScale,
                            blockM));

            long end = System.nanoTime();
            timersFlashAttentionHAT16.add((end - start));
            if (verbose) {
                IO.println("HAT-Flash-Attention (FP16) elapsed time: " + (end - start) + " ns");
            }
        }

        // Check results
        boolean isStreamsCorrect          = checkResult(O_java, O_streams, matrixSize);
        boolean isHATSelfAttentionCorrect = checkResult(O_java, O_selfAttention, matrixSize);
        boolean isFlashAttentionCorrect   = checkResult(O_java, O_flashAttention, matrixSize);

        if (isStreamsCorrect) {
            IO.println("Self-Attention Parallel Stream is correct");
        } else {
            IO.println("Self-Attention Parallel Stream  is wrong");
        }
        if (isHATSelfAttentionCorrect) {
            IO.println("HAT-Self-Attention Result is correct");
        } else {
            IO.println("HAT-Self-Attention is wrong");
        }
        if (isFlashAttentionCorrect) {
            IO.println("HAT-Flash-Attention is correct");
        } else {
            IO.println("HAT_Flash-Attention is wrong. Note: expected due to use of multiple Math.exp operations " +
                    "not present in the self-attention version.");
        }

        // Print Performance Metrics
        // skip 50% of first timers -> we evaluate in peak,
        // or closed to peak, performance.
        final int skip = ITERATIONS / 2;
        double averageJavaTimer = computeAverage(timersSelfAttentionJava, skip);
        double averageStreamTimer = computeAverage(timersSelfAttentionStream, skip);
        double averageSelfAttentionHAT = computeAverage(timersSelfAttentionHAT, skip);
        double averageFlashAttentionHAT = computeAverage(timersFlashAttentionHAT, skip);
        double averageFlashAttentionHAT16 = computeAverage(timersFlashAttentionHAT16, skip);

        IO.println("\nAverage elapsed time:");
        IO.println("Average Java Self-Attention       : " + averageJavaTimer);
        IO.println("Average Stream Self-Attention     : " + averageStreamTimer);
        IO.println("Average HAT Self-Attention        : " + averageSelfAttentionHAT);
        IO.println("Average HAT Flash-Attention       : " + averageFlashAttentionHAT);
        IO.println("Average HAT Flash-Attention (FP16): " + averageFlashAttentionHAT16);

        IO.println("\nSpeedups vs Java:");
        IO.println("Java / Java Parallel Stream       = " + computeSpeedup(averageJavaTimer, averageStreamTimer) + "x");
        IO.println("Java / HAT-Self-Attention         = " + computeSpeedup(averageJavaTimer, averageSelfAttentionHAT) + "x");
        IO.println("Java / HAT-Flash-Attention        = " + computeSpeedup(averageJavaTimer, averageFlashAttentionHAT) + "x");
        IO.println("Java / HAT-Flash-Attention (FP16) = " + computeSpeedup(averageJavaTimer, averageFlashAttentionHAT16) + "x");

        IO.println("\nSpeedups vs Streams:");
        IO.println("Java Streams / HAT-Self-Attention         = " + computeSpeedup(averageStreamTimer, averageSelfAttentionHAT) + "x");
        IO.println("Java Streams / HAT-Flash-Attention        = " + computeSpeedup(averageStreamTimer, averageFlashAttentionHAT) + "x");
        IO.println("Java Streams / HAT-Flash-Attention (FP16) = " + computeSpeedup(averageStreamTimer, averageFlashAttentionHAT16) + "x");

        IO.println("\nSpeedups vs HAT-Self-Attention:");
        IO.println("HAT-Self-Attention / HAT-Flash-Attention = " + computeSpeedup(averageSelfAttentionHAT, averageFlashAttentionHAT) + "x");
        IO.println("HAT-Self-Attention / HAT-Flash-Attention (FP16) = " + computeSpeedup(averageSelfAttentionHAT, averageFlashAttentionHAT16) + "x");

        // Write CSV table with all results
        dumpStatsToCSVFile(List.of(timersSelfAttentionJava,
                        timersSelfAttentionStream,
                        timersSelfAttentionHAT,
                        timersFlashAttentionHAT,
                        timersFlashAttentionHAT16),
                List.of("Java-fp32",
                        "Streams-fp32",
                        "HAT-Self-Attention-fp32",
                        "HAT-Flash-Attention-fp32",
                        "HAT-Flash-Attenttion-fp16"),
                "table-flash-attention-" + size + ".csv");
    }
}
