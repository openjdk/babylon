/*
 * Copyright (c) 2025, Oracle and/or its affiliates. All rights reserved.
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
 * A compute range holds the number of threads to run on an accelerator.
 * A compute range has two main properties:
 * - The global number of threads: this means the total number of threads to run per dimension.
 * This is specified by instancing a new object of type {@link ThreadMesh}.
 * - A local group size: this is specified by instancing an object of type {@link ThreadMesh}.
 * A local group size is optional. If it is not specified, the HAT runtime may device a default
 * value.
 */
public class ComputeRange {

    final private ThreadMesh globalMesh;
    final private ThreadMesh localMesh;

    /**
     * Total number of threads to run in 1D.
     * @param globalMesh {@link ThreadMesh1D}
     */
    public ComputeRange(GlobalMesh1D globalMesh) {
        this.globalMesh = globalMesh;
        this.localMesh = null;
    }

    /**
     * Total number of threads to run in 1D for global and local mesh.
     * @param globalMesh {@link ThreadMesh1D}
     * @param localMesh {@link ThreadMesh1D}
     */
    public ComputeRange(GlobalMesh1D globalMesh, LocalMesh1D localMesh) {
        this.globalMesh = globalMesh;
        this.localMesh = localMesh;
    }


    /**
     * Defines a compute range for a 2D mesh. The parameter specifies the
     * global mesh (total number of threads to run).
     * @param globalMesh {@link ThreadMesh2D}
     */
    public ComputeRange(GlobalMesh2D globalMesh) {
        this.globalMesh = globalMesh;
        this.localMesh = null;
    }

    /**
     * Defines a compute range for a 2D mesh. The parameters specify the
     * global mesh (total number of threads to run) and the local mesh.
     * @param globalMesh {@link ThreadMesh2D}
     * @param localMesh {@link ThreadMesh2D}
     */
    public ComputeRange(GlobalMesh2D globalMesh, LocalMesh2D localMesh) {
        this.globalMesh = globalMesh;
        this.localMesh = localMesh;
    }

    /**
     * Defines a compute range for a 3D mesh. The parameter specifies the
     * global mesh (total number of threads to run).
     * @param globalMesh {@link ThreadMesh3D}
     */
    public ComputeRange(GlobalMesh3D globalMesh) {
        this.globalMesh = globalMesh;
        this.localMesh = null;
    }

    /**
     * Defines a compute range for a 3D mesh. The parameters specify the
     * global mesh (total number of threads to run) and the local mesh.
     * @param globalMesh {@link ThreadMesh3D}
     * @param localMesh {@link ThreadMesh3D}
     */
    public ComputeRange(GlobalMesh3D globalMesh, LocalMesh3D localMesh) {
        this.globalMesh = globalMesh;
        this.localMesh = localMesh;
    }

    /**
     * Factory method to run a single thread on a target accelerator. Although for some accelerators this could be
     * beneficial (e.g., FPGAs), in general, use only for debugging purposes.
     */
    public static final ComputeRange SINGLE_THREADED = new ComputeRange(new GlobalMesh1D(1));

    /**
     * Obtain the total number of threads per dimension. The number of threads
     * per dimension is stored in a {@link ThreadMesh}
     * @return {@link ThreadMesh}
     */
    public ThreadMesh getGlobalMesh() {
        return globalMesh;
    }

    /**
     * Obtain the local group size per dimension. The group size per dimension is stored
     * in a {@link ThreadMesh}.
     * @return {@link ThreadMesh}
     */
    public ThreadMesh getLocalMesh() {
        return localMesh;
    }
}