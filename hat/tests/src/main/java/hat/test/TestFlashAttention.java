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
package hat.test;

import hat.Accelerator;
import hat.Accelerator.Compute;
import hat.ComputeContext;
import hat.HATMath;

import static hat.KernelContext.*;
import hat.backend.Backend;
import hat.buffer.F16Array;
import hat.buffer.F32Array;
import hat.device.DeviceSchema;
import hat.device.NonMappableIface;
import hat.test.annotation.HatTest;
import hat.test.exceptions.HATAssertionError;
import hat.test.exceptions.HATAsserts;
import hat.types.F16;
import jdk.incubator.code.Reflect;

import java.lang.invoke.MethodHandles;
import java.util.Random;
import java.util.stream.IntStream;

import static hat.NDRange.Global1D;
import static hat.NDRange.Local1D;
import static hat.NDRange.NDRange1D;
import static hat.buffer.F16Array.create;

/**
 * How to run?
 *
 * <p>Testing with the OpenCL backend:</p>
 * <code>java @.ffi-opencl-test hat.test.TestFlashAttention</code>
 *
 * <p>Testing with the CUDA backend:</p>
 * <code>java @.ffi-cuda-test hat.test.TestFlashAttention</code>
 */
public class TestFlashAttention {

    private static final float DELTA = 0.01f;

    // Sequential implementation of the standard attention
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

    private interface SharedF16Array extends NonMappableIface {
        F16 array(int index);

        DeviceSchema<SharedF16Array> deviceSchema = DeviceSchema.of(SharedF16Array.class,
                arr -> arr.array("array", 7168, half -> half.field("value")));

        static SharedF16Array createLocal() {
            return null;
        }
    }

    private interface PrivateF16Array extends NonMappableIface {
        F16 array(int index);

        DeviceSchema<PrivateF16Array> deviceSchema = DeviceSchema.of(PrivateF16Array.class,
                arr -> arr.array("array", 64, half -> half.field("value")));

        static PrivateF16Array createPrivate() {
            return null;
        }
    }

    @Reflect
    public static int ceilFunction(int N, int blockN) {
        return (N + blockN - 1) / blockN;
    }

    @Reflect
    public static void flashAttentionF16(
                                         F16Array Q, F16Array K, F16Array V,
                                         F16Array O, F16Array m, F16Array l,
                                         final int N, final int d, final float softmaxScale) {
        int bx = BIX();
        int tid = LIX();

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
            F16 valQ = Q.array((startIndex + tid) * d + k);
            sharedArray.array((tid * d + k) + sQ_index).value(valQ.value());
        }

        barrier();

        int numBlocks = ceilFunction(N, blockN);
        for (int tileId = 0; tileId < numBlocks; tileId++) {

            int kvTileRow = (tileId * blockN) + tid;

            // Load the tiles K and V into shared memory
            for (int k = 0; k < d; k++) {
                F16 kVal = K.array(kvTileRow * d + k);
                F16 vVal = V.array(kvTileRow * d + k);
                sharedArray.array((tid * d + k) + sK_index).value(kVal.value());
                sharedArray.array((tid * d + k) + sV_index).value(vVal.value());
            }
            barrier();

            // m we accumulate the max values for all tiles
            F16 m_prev = m.array(startIndex + tid);
            // in l we accumulate the sum values for all tiles
            F16 l_prev = l.array(startIndex + tid);
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
                privateFloatArray.array(t).value(score.value());
                m_block = HATMath.maxf16(m_block, score);
            }

            // Compute local sum of Math.exp(p_i - m_block)
            for (int t = 0; t < blockN; t++) {
                F16 privateVal = privateFloatArray.array(t);
                F16 sub = F16.sub(privateVal, m_block);
                F16 p = HATMath.expf16(sub);
                privateFloatArray.array(t).value(p.value());
                l_block = F16.add(l_block, p);
            }

            // Update m and l with the new values
            F16 m_new = HATMath.maxf16(m_prev, m_block);

            F16 exp1 = HATMath.expf16(F16.sub(m_prev, m_new));
            F16 mul1 = F16.mul(exp1, l_prev);
            F16 exp2 = HATMath.expf16(F16.sub(m_block, m_new));
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

                F16 expOut1 = HATMath.expf16(F16.sub(m_prev, m_new));
                F16 multOut1 = F16.mul(l_prev, expOut1);
                multOut1 = F16.mul(multOut1, value);

                F16 expOut2 = HATMath.expf16(F16.sub(m_block, m_new));
                F16 multOut2 = F16.mul(expOut2, pv);
                F16 addOut = F16.add(multOut1, multOut2);
                F16 outVal = F16.div(addOut, l_new);

                // write output
                O.array(oIndex).value(outVal.value());
            }

            // update m and l in global memory
            m.array(startIndex + tid).value(m_new.value());
            l.array(startIndex + tid).value(l_new.value());

            barrier();
        }
    }

    @Reflect
    public static void computeFlashAttentionF16(ComputeContext computeContext,
                                                F16Array Q, F16Array K, F16Array V,
                                                F16Array O, F16Array m, F16Array l,
                                                final int N, final int d, final float scale, final int blockSize) {
        var ndRange = NDRange1D.of(Global1D.of(N), Local1D.of(blockSize));
        computeContext.dispatchKernel(ndRange, () -> flashAttentionF16( Q, K, V, O, m, l, N, d, scale));
    }

    // Express a float array in shared memory with HAT
    private interface SharedFloatArray extends NonMappableIface {
        void array(long index, float value);

        float array(long index);

        DeviceSchema<SharedFloatArray> deviceSchema = DeviceSchema.of(SharedFloatArray.class,
                arr -> arr
                        // final int sharedMemorySize = block_m * head_dim
                        //                + block_n * head_dim
                        //                + block_n * head_dim
                        //                + block_m * block_n;
                        .array("array", 7168));

        static SharedFloatArray createLocal() {
            return null;
        }
    }

    // Express an array of floats in private memory with HAT
    private interface PrivateFloatArray extends NonMappableIface {
        void array(long index, float value);

        float array(long index);

        DeviceSchema<PrivateFloatArray> deviceSchema = DeviceSchema.of(PrivateFloatArray.class,
                arr -> arr
                        // SIZE = HEAD_DIM (e.g., 64)
                        .array("array", 64));

        static PrivateFloatArray createPrivate() {
            return null;
        }
    }

    @Reflect
    public static void flashAttention(
            F32Array Q, F32Array K, F32Array V,
            F32Array O, F32Array m, F32Array l,
            final int N, final int d, final float softmaxScale) {
        int bx = BIX();
        int tid = LIX();

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
                    Q.array((startIndex + tid) * d + k));
        }
        barrier();

        int numBlocks = ceilFunction(N, blockN);
        for (int tileId = 0; tileId < numBlocks; tileId++) {

            int kvTileRow = (tileId * blockN) + tid;

            // Load the tiles K and V into shared memory
            for (int k = 0; k < d; k++) {
                final int storeSharedMemIndex = tid * d + k;
                sharedArray.array(storeSharedMemIndex + sK_index, K.array(kvTileRow * d + k));
                sharedArray.array(storeSharedMemIndex + sV_index, V.array(kvTileRow * d + k));
            }
            barrier();

            // m we accumulate the max values for all tiles
            float m_prev = m.array(startIndex + tid);
            // in l we accumulate the sum values for all tiles
            float l_prev = l.array(startIndex + tid);
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
                privateFloatArray.array(t, score);
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
                int oIndex = (bx * blockN + tid) * d + k;
                float value = O.array(oIndex);
                float outVal = (float) ((l_prev * Math.exp(m_prev - m_new) * value +
                        Math.exp(m_block - m_new) * pv) / l_new);
                // write output
                O.array(oIndex, outVal);
            }

            // update m and l in global memory
            m.array(startIndex + tid, m_new);
            l.array(startIndex + tid, l_new);

            barrier();
        }
    }

    @Reflect
    public static void computeFlashAttention(ComputeContext computeContext,
                                             F32Array Q, F32Array K, F32Array V,
                                             F32Array O, F32Array m, F32Array l,
                                             final int N, final int d, final float scale, final int blockSize) {
        var ndRange = NDRange1D.of(Global1D.of(N), Local1D.of(blockSize));
        computeContext.dispatchKernel(ndRange, () -> flashAttention( Q, K, V, O, m, l, N, d, scale));
    }

    public static void checkResult(F32Array O_reference, F32Array O, final int matrixSize) {
        for (int i = 0; i < matrixSize; i++) {
            HATAsserts.assertEquals(O_reference.array(i), O.array(i), DELTA);
        }
    }

    public static void checkResult(F32Array O_reference, F16Array O, final int matrixSize) {
        for (int i = 0; i < matrixSize; i++) {
            HATAsserts.assertEquals(O_reference.array(i), F16.f16ToFloat(O.array(i)), DELTA);
        }
    }

    @HatTest
    public void testFlashAttentionF32() {

        var lookup = MethodHandles.lookup();
        var accelerator = new Accelerator(lookup, Backend.FIRST);

        final int sequenceLen = 512;   // represent the number of tokens (or words)
        final int headDim = 64;        // vector representation for a single token
        final int blockM = 32;         // tile size
        final float softmaxScale = (float) (1.0f / Math.sqrt(headDim));

        // Inputs preparation
        final int matrixSize = sequenceLen * headDim;
        var Q = F32Array.create(accelerator, matrixSize);
        var K = F32Array.create(accelerator, matrixSize);
        var V = F32Array.create(accelerator, matrixSize);
        var m = F32Array.create(accelerator, matrixSize);
        var l = F32Array.create(accelerator, matrixSize);
        var outputF32 = F32Array.create(accelerator, matrixSize);

        var Q16 = create(accelerator, matrixSize);
        var K16 = create(accelerator, matrixSize);
        var V16 = create(accelerator, matrixSize);

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

        IntStream.range(0, m.length()).forEach(k -> m.array(k, Float.MIN_VALUE));
        IntStream.range(0, l.length()).forEach(k -> l.array(k, 0.0f));

        // HAT Accelerated Version Using FP32
        accelerator.compute((@Reflect Compute)
                cc -> computeFlashAttention(
                        cc,
                        Q,
                        K,
                        V,
                        outputF32,
                        m, l,
                        sequenceLen,
                        headDim,
                        softmaxScale,
                        blockM));

        F32Array attentionMatrix = F32Array.create(accelerator, sequenceLen * sequenceLen);
        var outputJava = F32Array.create(accelerator, matrixSize);
        selfAttentionV2(Q, K, V, attentionMatrix, outputJava, sequenceLen, headDim, softmaxScale);
        checkResult(outputJava, outputF32, matrixSize);
    }

    @HatTest
    public void testFlashAttentionF16() {
        var lookup = MethodHandles.lookup();
        var accelerator = new Accelerator(lookup, Backend.FIRST);
        final int sequenceLen = 512;   // represent the number of tokens (or words)
        final int headDim = 64;        // vector representation for a single token
        final int blockM = 32;         // tile size
        final float softmaxScale = (float) (1.0f / Math.sqrt(headDim));

        // Inputs preparation
        final int matrixSize = sequenceLen * headDim;
        var Q = F32Array.create(accelerator, matrixSize);
        var K = F32Array.create(accelerator, matrixSize);
        var V = F32Array.create(accelerator, matrixSize);

        var Q16 = create(accelerator, matrixSize);
        var K16 = create(accelerator, matrixSize);
        var V16 = create(accelerator, matrixSize);
        var m16 = create(accelerator, matrixSize);
        var l16 = create(accelerator, matrixSize);
        var outputF16 = create(accelerator, matrixSize);

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

        IntStream.range(0, m16.length()).forEach(k -> m16.array(k).value(F16.of(-100).value()));
        IntStream.range(0, l16.length()).forEach(k -> l16.array(k).value(F16.of(0.0f).value()));

        // HAT Accelerated Version Using FP16
        accelerator.compute((@Reflect Compute)
                cc -> computeFlashAttentionF16(
                        cc,
                        Q16,
                        K16,
                        V16,
                        outputF16,
                        m16, l16,
                        sequenceLen,
                        headDim,
                        softmaxScale,
                        blockM));

        F32Array attentionMatrix = F32Array.create(accelerator, sequenceLen * sequenceLen);
        var outputJava = F32Array.create(accelerator, matrixSize);
        selfAttentionV2(Q, K, V, attentionMatrix, outputJava, sequenceLen, headDim, softmaxScale);
        checkResult(outputJava, outputF16, matrixSize);
    }
}
