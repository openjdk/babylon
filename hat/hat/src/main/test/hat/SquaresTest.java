package hat;

/*
 * Copyright (c) 2024, Oracle and/or its affiliates. All rights reserved.
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

import hat.backend.Backend;
import hat.backend.JavaMultiThreadedBackend;
import hat.buffer.S32Array;
import org.testng.Assert;
import org.testng.annotations.Test;

import java.lang.reflect.Method;
import java.lang.reflect.code.CopyContext;
import java.lang.reflect.code.Op;
import java.lang.reflect.code.OpTransformer;
import java.lang.reflect.code.TypeElement;
import java.lang.reflect.code.Value;
import java.lang.reflect.code.op.CoreOp;
import java.lang.reflect.code.type.JavaType;
import java.lang.runtime.CodeReflection;
import java.util.List;


/*
 * @test
 * @run testng hat.CustomOpTest
 */

public class SquaresTest {
    @CodeReflection
    public static void squareKernel(KernelContext kc, S32Array s32Array) {
        if (kc.x<kc.maxX){
            int value = s32Array.array(kc.x);     // arr[cc.x]
            s32Array.array(kc.x, value * value);  // arr[cc.x]=value*value
        }
    }

    @CodeReflection
    public static void square(ComputeContext cc, S32Array s32Array) {
        cc.dispatchKernel(s32Array.length(),
                kc -> squareKernel(kc, s32Array)
        );
    }

  @Test
       void    testSquares(){
        var lookup = java.lang.invoke.MethodHandles.lookup();
        var accelerator = new Accelerator(lookup, new JavaMultiThreadedBackend());
        var arr = S32Array.create(accelerator, 32);
        for (int i = 0; i < arr.length(); i++) {
            arr.array(i, i);
        }
        accelerator.compute(
                cc -> SquaresTest.square(cc, arr)  //QuotableComputeContextConsumer
        );                                     //   extends Quotable, Consumer<ComputeContext>
        for (int i = 0; i < arr.length(); i++) {
            Assert.assertEquals(i*i, arr.array(i));
        }
    }

}