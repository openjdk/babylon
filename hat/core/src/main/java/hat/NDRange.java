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
 * This is specified by instancing a new object of type {@link Range}.
 * - A local group size: this is specified by instancing an object of type {@link Range}.
 * A local group size is optional. If it is not specified, the HAT runtime may device a default
 * value.
 */
public class NDRange {

    /**
     * Total number of threads to run in 1D.
     * @param global {@link Global1D}
     */
    public static NDRange of(Global1D global) {
        return new NDRange(global);
    }

    /**
     * Total number of threads to run in 1D for global and local mesh.
     * @param global {@link Global1D}
     * @param local {@link Local1D}
     */
    public static NDRange of(Global1D global, Local1D local) {
        return new NDRange(global, local);
    }

    /**
     * Defines a compute range for a 2D mesh. The parameter specifies the
     * global mesh (total number of threads to run).
     * @param global {@link Global2D}
     */
    public static NDRange of(Global2D global) {
        return new NDRange(global);
    }

    /**
     * Defines a compute range for a 2D mesh. The parameters specify the
     * global mesh (total number of threads to run) and the local mesh.
     * @param global {@link Global2D}
     * @param local {@link Local2D}
     */
    public static NDRange of(Global2D global, Local2D local) {
        return new NDRange(global, local);
    }

    /**
     * Defines a compute range for a 3D mesh. The parameter specifies the
     * global mesh (total number of threads to run).
     * @param global {@link Global3D}
     */
    public static NDRange of(Global3D global) {
        return new NDRange(global);
    }

    /**
     * Defines a compute range for a 3D mesh. The parameters specify the
     * global mesh (total number of threads to run) and the local mesh.
     * @param global {@link Global3D}
     * @param local {@link Local3D}
     */
    public static NDRange of(Global3D global, Local3D local) {
        return new NDRange(global, local);
    }

    /**
     * Factory method to run a single thread on a target accelerator. Although for some accelerators this could be
     * beneficial (e.g., FPGAs), in general, use only for debugging purposes.
     */
    public static final NDRange SINGLE_THREADED = new NDRange(new Global1D(1));

    /**
     * Obtain the total number of threads per dimension. The number of threads
     * per dimension is stored in a {@link Range}
     * @return {@link Range}
     */
    public Range getGlobal() {
        return global;
    }

    /**
     * Obtain the local group size per dimension. The group size per dimension is stored
     * in a {@link Range}.
     * @return {@link Range}
     */
    public Range getLocal() {
        return local;
    }

    public boolean isSpecificRange() {
        return false;
    }

    /**
     * Utility method to create a 1D program with only global thread size.
     * @param numThreadsGlobal
     * @return {@link NDRange}
     */
    public static NDRange of(int numThreadsGlobal) {
        return new NDRange(new Global1D(numThreadsGlobal));
    }

    /**
     * Utility method to create a 1D program with global and local thread sizes.
     * @param numThreadsGlobal
     * @return {@link NDRange}
     */
    public static NDRange of(int numThreadsGlobal, int numThreadLocal) {
        return new NDRange(new Global1D(numThreadsGlobal), new Local1D(numThreadLocal));
    }

    private final Range global;
    private final Range local;

    private NDRange(Global1D global) {
        this.global = global;
        this.local = null;
    }

    private NDRange(Global1D global, Local1D local) {
        this.global = global;
        this.local = local;
    }

    private NDRange(Global2D global) {
        this.global = global;
        this.local = null;
    }

    private NDRange(Global2D global, Local2D local) {
        this.global = global;
        this.local = local;
    }

    private NDRange(Global3D global) {
        this.global = global;
        this.local = null;
    }

    private NDRange(Global3D global, Local3D local) {
        this.global = global;
        this.local = local;
    }
}