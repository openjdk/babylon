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
package hat.buffer;

import hat.Accelerator;
import hat.ifacemapper.Schema;

public interface KernelBufferContext extends Buffer {
    int x();
    void x(int x);

    int y();
    void y(int y);

    int z();
    void z(int z);

    int maxX();
    void maxX(int maxX);

    int maxY();
    void maxY(int maxY);

    int maxZ();
    void maxZ(int maxZ);

    int dimensions();
    void dimensions(int numDimensions);

    int lx();
    void lx(int lx);
    int ly();
    void ly(int ly);
    int lz();
    void lz(int lz);

    int lsx();
    void lsx(int lsx);
    int lsy();
    void lsy(int lsy);
    int lsz();
    void lsz(int lsz);

    // Important part here! do not forget the new fields.

    Schema<KernelBufferContext> schema = Schema.of(KernelBufferContext.class,
            s -> s.fields(
                    "x","maxX", "y", "maxY", "z", "maxZ",
                    "dimensions",
                    "lx", "ly", "lz", "lsx",  "lsy", "lsz"));

    static KernelBufferContext create(Accelerator accelerator, int x, int maxX) {
        KernelBufferContext kernelContext =  schema.allocate(accelerator);
        kernelContext.x(x);
        kernelContext.maxX(maxX);
        kernelContext.dimensions(1);
        return kernelContext;
    }

    static KernelBufferContext create(Accelerator accelerator, int x, int y, int z, int maxX, int maxY, int maxZ, int[] locals, int[] sizeLocals) {
        KernelBufferContext kernelBufferContext =  schema.allocate(accelerator);

        kernelBufferContext.x(x);
        kernelBufferContext.y(y);
        kernelBufferContext.z(z);
        kernelBufferContext.maxX(maxX);
        kernelBufferContext.maxY(maxY);
        kernelBufferContext.maxZ(maxZ);
        kernelBufferContext.dimensions(3);

        if (locals != null) {
            kernelBufferContext.lsx(locals[0]);
            kernelBufferContext.lsy(locals[1]);
            kernelBufferContext.lsz(locals[2]);
        }

        if (sizeLocals != null) {
            kernelBufferContext.lsx(sizeLocals[0]);
            kernelBufferContext.lsy(sizeLocals[1]);
            kernelBufferContext.lsz(sizeLocals[2]);
        }
        return kernelBufferContext;
    }

}
