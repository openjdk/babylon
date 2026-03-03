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

import optkl.util.Regex;

/**
 * An NDRange specifies the number of threads to deploy on an hardware accelerator.
 * An NDRange has two main properties:
 * <ul>
 * <li>The global number of threads: this means the total number of threads to run per dimension.
 * This is specified by instancing a new object of type {@link Global}.</li>
 * <li>A local group size: this is specified by instancing an object of type {@link Local}.</li>
 * A local group size is optional. If it is not specified, the HAT runtime may device a default
 * value.
 * </ul>
 */
public interface NDRange<G extends NDRange.Global, L extends NDRange.Local> {
    Local local();
    Global global();
    boolean hasLocal();

    sealed interface Dim permits _1D,_2D,_3D{
        default int dimension(){
            return switch ((Dim)this){
                case _1D _ -> 1;
                case _2D _ -> 2;
                case _3D _ -> 3;
            };
        }
    }

    sealed interface _1D extends Dim  {
    }

    sealed interface _2D extends Dim {
    }

    sealed interface _3D extends Dim {
    }

    sealed  interface _1DX extends _1D {
        int x();
    }

    sealed interface _2DXY extends _2D {
        int x();
        int y();
    }

    sealed interface _3DXYZ extends _3D {
        int x();
        int y();
        int z();
    }

    sealed interface Range permits Local, Global, Block {
    }
    sealed interface Local extends Range {

    }

    sealed interface Block extends Range{
        // We need this to seal the interface hierarchy
        record Impl() implements Block {
        }

    }

    sealed interface Global extends Range {

    }

    sealed interface Global1D extends _1DX, Global {
        record Impl(int x) implements Global1D {
        }
        static Global1D of(int x) {

            return new Impl(x);
        }
    }

    sealed interface Global2D extends _2DXY, Global {
        record Impl(int x, int y) implements Global2D {
        }
        static Global2D of(int x, int y) {
            return new Impl(x, y);
        }
    }

    sealed interface Global3D extends _3DXYZ, Global {
        record Impl(int x, int y, int z) implements Global3D {
        }
        static Global3D of(int x, int y, int z) {
            return new Impl(x, y, z);
        }
    }




  sealed  interface Local1D extends _1DX, Local {
      record Impl(int x) implements Local1D {
      }
        static Local1D of(int x) {

            return new Impl(x);
        }

        Local1D EMPTY = Local1D.of(0);
    }

    sealed interface Local2D extends _2DXY, Local {
        record Impl(int x, int y) implements Local2D {
        }
        static Local2D of(int x, int y) {

            return new Impl(x, y);
        }

        Local2D EMPTY = Local2D.of(0, 0);

    }

    sealed interface Local3D extends _3DXYZ, Local {
        record Impl(int x, int y, int z) implements Local3D {
        }
        static Local3D of(int x, int y, int z) {

            return new Impl(x, y, z);
        }

        Local3D EMPTY = Local3D.of(0, 0, 0);
    }

    sealed interface NDRange1D extends NDRange<Global1D, Local1D>, _1D {
        @Override
        default boolean hasLocal() {
            return local() != Local1D.EMPTY;
        }

        record Impl(int dimension, Global1D global, Local1D local) implements NDRange1D {
        }

        static NDRange1D of(Global1D global, Local1D local) {
            return new Impl(1, global, local);
        }

        static NDRange1D of(Global1D global) {
            return new Impl(1, global, Local1D.EMPTY);
        }
    }


    static NDRange1D of1D(int gsx, int lsx) {
        return NDRange1D.of(Global1D.of(gsx), Local1D.of(lsx));
    }

    static NDRange1D of1D(int gsx) {
        return NDRange1D.of(Global1D.of(gsx), Local1D.EMPTY);
    }

    sealed interface NDRange2D extends NDRange<Global2D, Local2D>, _2D {
        @Override
        default boolean hasLocal() {
            return local() != Local2D.EMPTY;
        }

        record Impl(Global2D global, Local2D local) implements NDRange2D {
        }

        static NDRange2D of(Global2D global, Local2D local) {
            return new Impl(global, local);
        }

        static NDRange2D of(Global2D global) {
            return new Impl(global, Local2D.EMPTY);
        }
    }


    static NDRange2D of2D(int gsx, int gsy, int lsx, int lsy) {
        return NDRange2D.of(Global2D.of(gsx, gsy), Local2D.of(lsx, lsy));
    }

    static NDRange2D of2D(int gsx, int gsy) {
        return NDRange2D.of(Global2D.of(gsx, gsy), Local2D.EMPTY);
    }

   sealed interface NDRange3D extends NDRange<Global3D, Local3D>, _3D {
        @Override
        default boolean hasLocal() {
            return local() != Local3D.EMPTY;
        }

        record Impl(Global3D global, Local3D local) implements NDRange3D {
        }

        static NDRange3D of(Global3D global, Local3D local) {
            return new Impl(global, local);
        }

        static NDRange3D of(Global3D global) {
            return new Impl(global, Local3D.EMPTY);
        }
    }

    static NDRange3D of3D(int gsx, int gsy, int gsz, int lsx, int lsy, int lsz) {
        return NDRange3D.of(Global3D.of(gsx, gsy, gsz), Local3D.of(lsx, lsy, lsz));
    }

    static NDRange3D of3D(int gsx, int gsy, int gsz) {
        return NDRange3D.of(Global3D.of(gsx, gsy, gsz), Local3D.EMPTY);
    }

}