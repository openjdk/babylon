/*
 * Copyright (c) 2024 Intel Corporation. All rights reserved.
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

package intel.code.spirv;

import java.lang.reflect.Method;
import java.lang.foreign.MemorySegment;
import jdk.incubator.code.Op;
import jdk.incubator.code.CodeReflection;
import jdk.incubator.code.dialect.core.CoreOp;
import org.junit.jupiter.api.Test;

public class ExampleUseTest {

    @CodeReflection
    public static void matrixMultiply(float[] a, float[] b, float[] c, int size) {
        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++) {
                float sum = 0f;
                for (int k = 0; k < size; k++) {
                    sum += a[i * size + k] * b[k * size + j];
                }
                c[i * size + j] = sum;
            }
        }
    }

    @Test
    public void test() throws Exception {
        String methodName = "matrixMultiply";
        Method method = ExampleUseTest.class.getDeclaredMethod(methodName, float[].class, float[].class, float[].class, int.class);

        CoreOp.FuncOp javaFunc = Op.ofMethod(method).get();
        SpirvOps.FuncOp spirvFunc = TranslateToSpirvModel.translateFunction(javaFunc);
        MemorySegment spirvBinary = SpirvModuleGenerator.generateModule(methodName, spirvFunc);

        System.out.println("\n------- Java Model -------");
        System.out.println(javaFunc.toText());
        System.out.println("------- SPIR-V Model -------");
        System.out.println(spirvFunc.toText());
        System.out.println("------- SPIR-V Module -------");
        System.out.println(SpirvModuleGenerator.disassembleModule(spirvBinary));
   }
}