/*
 * Copyright (c) 2024, Oracle and/or its affiliates. All rights reserved.
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
package experiments.spirv;

import jdk.incubator.code.Op;
import jdk.incubator.code.Quotable;
import jdk.incubator.code.Quoted;
import java.util.function.Consumer;

public class Bad {
    public static class AcceleratorProxy {
        public interface QuotableComputeConsumer extends Quotable, Consumer<ComputeClosureProxy> {
        }

        public static class ComputeClosureProxy {
        }

        public void compute(AcceleratorProxy.QuotableComputeConsumer cqr) {
            Quoted quoted = cqr.quoted();
            Op op = quoted.op();
            System.out.println(op.toText());
        }

    }

    public static class MatrixMultiplyCompute {
        static void compute(AcceleratorProxy.ComputeClosureProxy computeContext, float[] a, float[] b, float[] c, int size) {
        }
    }

    //static final int size = 100; // works
    public static void main(String[] args) {
        AcceleratorProxy accelerator = new AcceleratorProxy();
        final int size = 100; // breaks!!!!
        //int size = 100;  // works
        var a = new float[]{};
        var b = new float[]{};
        var c = new float[]{};
        accelerator.compute(cc -> MatrixMultiplyCompute.compute(cc, a, b, c, size));
    }
}
