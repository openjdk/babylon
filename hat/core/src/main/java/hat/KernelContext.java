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
 *     @ Reflect
 *      static public void doSomeWork(final ComputeContext cc, S32Array arrayOfInt) {
 *         cc.dispatchKernel(KernelContext kc -> addDeltaKernel(kc,arrayOfInt.length(), 5, arrayOfInt);
 *      }
 * </pre>
 *
 * @author Gary Frost
 */
public class KernelContext {
    // Global accesses
    public int gix;
    public int giy;
    public int giz;

    public final int gsx;
    public final int gsy;
    public final int gsz;

    // Local accesses within a group
    public int lix;
    public int liy;
    public int liz;

    // Specify sizes for the local group sizes
    public int lsx;
    public int lsy;
    public int lsz;

    // Specify group/block index
    public int bix;
    public int biy;
    public int biz;

    final int dimensions;

    final public NDRange<?,?> ndRange;

    public KernelContext(NDRange<?,?> ndRange) {
        this.ndRange = ndRange;
        switch (ndRange) {
            case NDRange.NDRange1D ndRange1D -> {
                this.gsx = ((NDRange._1DX)(ndRange1D.global())).x();
                this.gsy = 1;
                this.gsz = 1;
                this.dimensions = ((NDRange._1DX)(ndRange1D.global())).dimension();
            }
            case NDRange.NDRange2D ndRange2D -> {
                this.gsx = ((NDRange._2DXY)(ndRange2D.global())).x();
                this.gsy = ((NDRange._2DXY)(ndRange2D.global())).y();
                this.gsz = 1;
                this.dimensions = ((NDRange._2DXY)(ndRange2D.global())).dimension();
            }
            case NDRange.NDRange3D ndRange3D -> {
                this.gsx = ((NDRange._3DXYZ)(ndRange3D.global())).x();
                this.gsy = ((NDRange._3DXYZ)(ndRange3D.global())).y();
                this.gsz = ((NDRange._3DXYZ)(ndRange3D.global())).z();
                this.dimensions = ((NDRange._3DXYZ)(ndRange3D.global())).dimension();
            }
            case null, default ->
                throw new IllegalArgumentException("Unknown NDRange type: "  + ndRange.getClass());

        }
    }

  //  public int getDimensions() {
    //    return this.dimensions;
   // }

   // public NDRange<?,?> getNDRange() {
     //   return this.ndRange;
   // }

    // This is a marker called by kernel code which is mapped to a barrier implementatiom
    public void barrier() {

    }
}
