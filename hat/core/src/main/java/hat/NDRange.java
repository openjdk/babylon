/*
 * Copyright (c) 2025-2026, Oracle and/or its affiliates. All rights reserved.
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

    Local local();

    Global global();

    Tile tile();

    Warp warp();

    boolean hasLocal();

    boolean hasTile();

    boolean hasWarp();

    sealed interface Dim permits Marker1D, Marker2D, Marker3D {
        default int dimension() {
            return switch (this) {
                case Marker1D _ -> 1;
                case Marker2D _ -> 2;
                case Marker3D _ -> 3;
            };
        }
    }

    sealed interface Marker1D extends Dim {
    }

    sealed interface Marker2D extends Dim {
    }

    sealed interface Marker3D extends Dim {
    }

    sealed interface M1D extends Marker1D {
        int x();
    }

    sealed interface M2D extends Marker2D {
        int x();

        int y();
    }

    sealed interface M3D extends Marker3D {
        int x();

        int y();

        int z();
    }

    sealed interface B1D extends Marker1D {
        boolean x();
    }

    sealed interface B2D extends Marker2D {
        boolean x();
        boolean y();
    }

    sealed interface B3D extends Marker3D {
        boolean x();
        boolean y();
        boolean z();
    }

    sealed interface Range permits Global, Local, Tile, Warp {
    }


    sealed interface Global extends Range {

    }

    sealed interface Local extends Range {

    }

    sealed interface Tile extends Range {

    }

    sealed interface Warp extends Range {

    }

    sealed interface Global1D extends M1D, Global {
        record Impl(int x) implements Global1D {
        }

        static Global1D of(int x) {

            return new Impl(x);
        }
    }

    sealed interface Global2D extends M2D, Global {
        record Impl(int x, int y) implements Global2D {
        }

        static Global2D of(int x, int y) {
            return new Impl(x, y);
        }
    }

    sealed interface Global3D extends M3D, Global {
        record Impl(int x, int y, int z) implements Global3D {
        }

        static Global3D of(int x, int y, int z) {
            return new Impl(x, y, z);
        }
    }

    sealed interface Local1D extends M1D, Local {
        record Impl(int x) implements Local1D {
        }

        static Local1D of(int x) {

            return new Impl(x);
        }

        Local1D EMPTY = Local1D.of(0);
    }

    sealed interface Local2D extends M2D, Local {
        record Impl(int x, int y) implements Local2D {
        }

        static Local2D of(int x, int y) {

            return new Impl(x, y);
        }

        Local2D EMPTY = Local2D.of(0, 0);

    }

    sealed interface Local3D extends M3D, Local {
        record Impl(int x, int y, int z) implements Local3D {
        }

        static Local3D of(int x, int y, int z) {

            return new Impl(x, y, z);
        }

        Local3D EMPTY = Local3D.of(0, 0, 0);
    }

    sealed interface Tile1D extends M1D, Tile {
        record Impl(int x) implements Tile1D {
        }

        static Tile1D of(int x) {
            return new Impl(x);
        }

        Tile1D EMPTY = Tile1D.of(0);
    }

    sealed interface Tile2D extends M2D, Tile {
        record Impl(int x, int y) implements Tile2D {
        }

        static Tile2D of(int x, int y) {
            return new Impl(x, y);
        }

        Tile2D EMPTY = Tile2D.of(0, 0);

    }

    sealed interface Tile3D extends M3D, Tile {
        record Impl(int x, int y, int z) implements Tile3D {
        }

        static Tile3D of(int x, int y, int z) {
            return new Impl(x, y, z);
        }

        Tile3D EMPTY = Tile3D.of(0, 0, 0);
    }

    sealed interface Warp1D extends B1D, Warp {
        record Impl(boolean x) implements Warp1D {
        }

        static Warp1D of(boolean x) {
            return new Impl(x);
        }

        Warp1D EMPTY = Warp1D.of(false);
    }

    sealed interface Warp2D extends B2D, Warp {
        record Impl(boolean x, boolean y) implements Warp2D {
        }

        static Warp2D of(boolean x, boolean y) {
            return new Impl(x, y);
        }

        Warp2D EMPTY = Warp2D.of(false, false);
    }

    sealed interface Warp3D extends B3D, Warp {
        record Impl(boolean x, boolean y, boolean z) implements Warp3D {
        }

        static Warp3D of(boolean x, boolean y, boolean z) {
            return new Impl(x, y, z);
        }

        Warp3D EMPTY = Warp3D.of(false, false, false);
    }


    sealed interface NDRange1D extends NDRange, Marker1D {
        @Override
        default boolean hasLocal() {
            return local() != Local1D.EMPTY;
        }

        @Override
        default boolean hasTile() {
            return tile() != Tile1D.EMPTY;
        }

        @Override
        default boolean hasWarp() {
            return warp() != Warp1D.EMPTY;
        }

        record Impl(Global1D global, Local1D local, Tile1D tile, Warp1D warp) implements NDRange1D {
        }

        static NDRange1D of(Global1D global, Local1D local, Tile1D tile, Warp1D warp) {
            return new Impl( global, local, tile, warp);
        }

        static NDRange1D of(Global1D global, Local1D local) {
            return new Impl(global, local, Tile1D.EMPTY, Warp1D.EMPTY);
        }

        static NDRange1D of(Global1D global) {
            return new Impl(global, Local1D.EMPTY, Tile1D.EMPTY, Warp1D.EMPTY);
        }
    }

    static NDRange1D of1D(int gsx, int lsx) {
        return NDRange1D.of(Global1D.of(gsx), Local1D.of(lsx));
    }

    static NDRange1D of1D(int gsx) {
        return NDRange1D.of(Global1D.of(gsx), Local1D.EMPTY);
    }

    sealed interface NDRange2D extends NDRange, Marker2D {
        @Override
        default boolean hasLocal() {
            return local() != Local2D.EMPTY;
        }

        @Override
        default boolean hasTile() {
            return tile() != Tile2D.EMPTY;
        }

        @Override
        default boolean hasWarp() {
            return warp() != Warp2D.EMPTY;
        }

        record Impl(Global2D global, Local2D local, Tile2D tile, Warp2D warp) implements NDRange2D {
        }

        static NDRange2D of(Global2D global, Local2D local, Tile2D tile, Warp2D warp) {
            return new Impl(global, local, tile, warp);
        }

        static NDRange2D of(Global2D global, Local2D local) {
            return new Impl(global, local, Tile2D.EMPTY, Warp2D.EMPTY);
        }

        static NDRange2D of(Global2D global) {
            return new Impl(global, Local2D.EMPTY, Tile2D.EMPTY, Warp2D.EMPTY);
        }
    }

    static NDRange2D of2D(int gsx, int gsy, int lsx, int lsy) {
        return NDRange2D.of(Global2D.of(gsx, gsy), Local2D.of(lsx, lsy));
    }

    static NDRange2D of2D(int gsx, int gsy) {
        return NDRange2D.of(Global2D.of(gsx, gsy), Local2D.EMPTY);
    }

    sealed interface NDRange3D extends NDRange, Marker3D {
        @Override
        default boolean hasLocal() {
            return local() != Local3D.EMPTY;
        }

        @Override
        default boolean hasTile() {
            return tile() != Tile3D.EMPTY;
        }

        @Override
        default boolean hasWarp() {
            return warp() != Warp3D.EMPTY;
        }

        record Impl(Global3D global, Local3D local, Tile3D tile, Warp3D warp) implements NDRange3D {
        }

        static NDRange3D of(Global3D global, Local3D local, Tile3D tile, Warp3D warp) {
            return new Impl(global, local, tile, warp);
        }

        static NDRange3D of(Global3D global, Local3D local) {
            return new Impl(global, local, Tile3D.EMPTY, Warp3D.EMPTY);
        }

        static NDRange3D of(Global3D global) {
            return new Impl(global, Local3D.EMPTY, Tile3D.EMPTY, Warp3D.EMPTY);
        }
    }

    static NDRange3D of3D(int gsx, int gsy, int gsz, int lsx, int lsy, int lsz) {
        return NDRange3D.of(Global3D.of(gsx, gsy, gsz), Local3D.of(lsx, lsy, lsz));
    }

    static NDRange3D of3D(int gsx, int gsy, int gsz) {
        return NDRange3D.of(Global3D.of(gsx, gsy, gsz), Local3D.EMPTY);
    }

}