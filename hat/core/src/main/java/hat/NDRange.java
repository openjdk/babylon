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
public interface NDRange {

    int dimension();
    interface _1D extends NDRange {
        @Override
        default int dimension() {
            return 1;
        }
    }
    interface _2D extends NDRange {
        @Override
        default int dimension() {
            return 2;
        }
    }
    interface _3D extends NDRange {
        @Override
        default int dimension() {
            return 3;
        }
    }

    interface _1DX extends _1D {
        int x();
    }

    interface _2DXY extends _2D {
        int x();
        int y();
    }

    interface _3DXYZ extends _3D {
        int x();
        int y();
        int z();
    }

    interface Global {}

    interface Global1D extends  _1DX, Global{
        record Impl(int x) implements Global1D{};
        static Global1D of(int x) {
            return new Impl(x);
        }
    }
    interface Global2D extends  _2DXY, Global{
        record Impl(int x, int y) implements Global2D{};
        static Global2D of(int x,int y) {
            return new Impl(x, y);
        }
    }

    interface Global3D extends  _3DXYZ, Global{
        record Impl(int x, int y, int z) implements Global3D{};
        static Global3D of(int x, int y, int z) {
            return new Impl(x,y,z);
        }
    }

    interface Local{}

    interface Local1D extends  _1DX, Local{
        record Impl(int x) implements Local1D{};
        static Local1D of(int x) {
            return new Impl(x);
        }
    }
    interface Local2D extends  _2DXY, Local{
        record Impl(int x, int y) implements Local2D{};
        static Local2D of(int x,int y) {
            return new Impl(x, y);
        }
    }

    interface Local3D extends  _3DXYZ, Local{
        record Impl(int x, int y, int z) implements Local3D{};
        static Local3D of(int x, int y, int z) {
            return new Impl(x,y,z);
        }
    }

    interface Range extends NDRange {
        Global global();
        Local local();
    }

    record NDRange1D(Global1D global, Local1D local) implements Range, _1D { }

    record NDRange2D(Global2D global, Local2D local) implements Range, _2D { }

    record NDRange3D(Global3D global, Local3D local) implements Range, _3D { }

    static NDRange1D of(int x) {
        return new NDRange1D(Global1D.of(x), NDRange.EMPTY_LOCAL_1D);
    }

    static NDRange1D of(Global1D global) {
        return new NDRange1D(global, NDRange.EMPTY_LOCAL_1D);
    }

    static NDRange1D of(Global1D global, Local1D local) {
        return new NDRange1D(global, local);
    }


    static NDRange2D of(Global2D global) {
        return new NDRange2D(global, NDRange.EMPTY_LOCAL_2D);
    }

    static NDRange2D of(Global2D global, Local2D local) {
        return new NDRange2D(global, local);
    }


    static NDRange3D of(Global3D global) {
        return new NDRange3D(global, NDRange.EMPTY_LOCAL_3D);
    }

    static NDRange3D of(Global3D global, Local3D local) {
        return new NDRange3D(global, local);
    }

    Local1D EMPTY_LOCAL_1D = Local1D.of(0);
    Local2D EMPTY_LOCAL_2D = Local2D.of(0, 0);
    Local3D EMPTY_LOCAL_3D = Local3D.of(0, 0, 0);

}