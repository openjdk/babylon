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
package hat;

/**
 * Created by a dispatch call to a kernel from within a Compute method and 'conceptually' passed to a kernel.
 * <p>
 * In reality the backend decides how to pass the information contained in the KernelContext.
 *
 * <pre>
 *     @ CodeReflection
 *      static public void doSomeWork(final ComputeContext cc, S32Array arrayOfInt) {
 *         cc.dispatchKernel(KernelContext kc -> addDeltaKernel(kc,arrayOfInt.length(), 5, arrayOfInt);
 *      }
 * </pre>
 *
 * @author Gary Frost
 */
public class KernelContext {

    public final NDRange ndRange;

    public int x;
    public int y;
    public int z;

    final public int maxX;
    final public int maxY;
    final public int maxZ;

    final int dimensions;

    /**
     * 1D Kernel
     * @param ndRange {@link NDRange}
     * @param maxX Global number of threads for the first dimension (1D)
     */
    public KernelContext(NDRange ndRange, int maxX) {
        this.ndRange = ndRange;
        this.maxX = maxX;
        this.maxY = 0;
        this.maxZ = 0;
        this.dimensions = 1;
    }

    /**
     * 1D Kernel
     * @param ndRange {@link NDRange}
     * @param maxX Global number of threads for the first dimension (1D)
     * @param maxY Global number of threads for the second dimension (2D)
     */
    public KernelContext(NDRange ndRange, int maxX, int maxY) {
        this.ndRange = ndRange;
        this.maxX = maxX;
        this.maxY = maxY;
        this.maxZ = 0;
        this.dimensions = 2;
    }

    /**
     * 1D Kernel
     * @param ndRange {@link NDRange}
     * @param maxX Global number of threads for the first dimension (1D)
     * @param maxY Global number of threads for the second dimension (2D)
     * @param maxZ Global number of threads for the second dimension (3D)
     */
    public KernelContext(NDRange ndRange, int maxX, int maxY, int maxZ) {
        this.ndRange = ndRange;
        this.maxX = maxX;
        this.maxY = maxY;
        this.maxZ = maxZ;
        this.dimensions = 3;
    }

    public int getDimensions() {
        return this.dimensions;
    }

}
