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
 * Interface that specifies the number of threads per dimension.
 * The Thread Mesh can be used to store the global number of threads,
 * local group sizes and offsets.
 */
public interface Range {

    /**
     * Obtain the number of threads in the first dimension of the thread-mesh.
     * @return
     */
    int getX();

    /**
     * Obtain the number of threads in the second dimension of the thread-mesh.
     * @return
     */
    int getY();

    /**
     * Obtain the number of threads in the third dimension of the thread-mesh.
     * @return
     */
    int getZ();

    /**
     * Return the mesh dimension. It could be 1, 2 or 3.
     * @return int value
     */
    int getDims();

}
