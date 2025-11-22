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
public interface NDRange<G extends NDRange.Global, L extends NDRange.Local> {


    int dimension();

    default boolean hasLocal(){
        return (this instanceof NDRange.NDRange1D r1 && r1.local() != Local1D.EMPTY)||
                (this instanceof NDRange.NDRange2D r2 && r2.local() != Local2D.EMPTY)||
                (this instanceof NDRange.NDRange3D r3 && r3.local() != Local3D.EMPTY);
    }

    interface _1D extends NDRange<NDRange.Global1D,NDRange.Local1D> {
        @Override
        default int dimension() {
            return 1;
        }

    }
    interface _2D extends NDRange<NDRange.Global2D,NDRange.Local2D> {
        @Override
        default int dimension() {
            return 2;
        }
    }
    interface _3D extends NDRange<NDRange.Global3D,NDRange.Local3D> {
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
        private static Global1D of(int x) {
            record Impl(int x) implements Global1D{ }
            return new Impl(x);
        }
    }

    interface Global2D extends  _2DXY, Global{
        private static Global2D of(int x,int y) {
            record Impl(int x, int y) implements Global2D{};
            return new Impl(x, y);
        }
    }

    interface Global3D extends  _3DXYZ, Global{
        private static Global3D of(int x, int y, int z) {
            record Impl(int x, int y, int z) implements Global3D{};
            return new Impl(x,y,z);
        }
    }

    interface Local{}

    interface Local1D extends  _1DX, Local{
        private static Local1D of(int x) {
            record Impl(int x) implements Local1D{};
            return new Impl(x);
        }
        Local1D EMPTY = Local1D.of(0);
    }
    interface Local2D extends  _2DXY, Local{
        private static Local2D of(int x,int y) {
            record Impl(int x, int y) implements Local2D{};
            return new Impl(x, y);
        }
        Local2D EMPTY = Local2D.of(0, 0);

    }

    interface Local3D extends  _3DXYZ, Local{
        private static Local3D of(int x, int y, int z) {
            record Impl(int x, int y, int z) implements Local3D{};
            return new Impl(x,y,z);
        }
        Local3D EMPTY = Local3D.of(0, 0, 0);
    }

    interface Range<G extends NDRange.Global,L extends NDRange.Local> extends NDRange<G,L> {
        Global global();
        Local local();
    }

    record  NDRange1D(Global1D global, Local1D local) implements Range<Global1D,Local1D>, _1D {
    }
    static NDRange<Global1D,Local1D> of1D(int gsx,  int lsx) {
        return  new NDRange1D(Global1D.of(gsx), Local1D.of(lsx));
    }
    static NDRange<Global1D,Local1D> of1D(int gsx) {
        return new NDRange1D(Global1D.of(gsx), Local1D.EMPTY);
    }
    record NDRange2D(Global2D global, Local2D local) implements Range<Global2D,Local2D>, _2D {
    }

    static NDRange<Global2D,Local2D> of2D(int gsx, int gsy, int lsx, int lsy) {
        return new NDRange2D(Global2D.of(gsx,gsy), Local2D.of(lsx,lsy));
    }
    static NDRange<Global2D,Local2D> of2D(int gsx,int gsy) {
        return new NDRange2D(Global2D.of(gsx,gsy), Local2D.EMPTY);
    }

    record NDRange3D(Global3D global, Local3D local) implements Range<Global3D,Local3D>, _3D {
    }

    static NDRange<Global3D,Local3D> of3D(int gsx, int gsy, int gsz, int lsx, int lsy, int lsz) {
        return new NDRange3D(Global3D.of(gsx,gsy,gsz), Local3D.of(lsx,lsy,lsz));
    }
    static NDRange<Global3D,Local3D> of3D(int gsx,int gsy, int gsz) {
        return new NDRange3D(Global3D.of(gsx,gsy,gsz), Local3D.EMPTY);
    }
}