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
package view.f32;


public interface F32Matrix4x4 {
    float x0y0();

    float x1y0();

    float x2y0();

    float x3y0();

    float x0y1();

    float x1y1();

    float x2y1();

    float x3y1();

    float x0y2();

    float x1y2();

    float x2y2();

    float x3y2();

    float x0y3();

    float x1y3();

    float x2y3();

    float x3y3();

    int idx();
    default String asString() {

        return String.format("""
                        |%5.2f, %5.2f, %5.2f, %5.2f|
                        |%5.2f, %5.2f, %5.2f, %5.2f|
                        |%5.2f, %5.2f, %5.2f, %5.2f|
                        |%5.2f, %5.2f, %5.2f, %5.2f|
                        """,
                x0y0(), x1y0(), x2y0(), x3y0(),
                x0y1(), x1y1(), x2y1(), x3y1(),
                x0y2(), x1y2(), x2y2(), x3y2(),
                x0y3(), x1y3(), x2y3(), x3y3());
    }
    class F32Matrix4x4Pool extends FloatPool<F32Matrix4x4Pool> {
        static int X0Y0 = 0;
        static  int X1Y0 = 1;
        static  int X2Y0 = 2;
        static  int X3Y0 = 3;
        static  int X0Y1 = 4;
        static  int X1Y1 = 5;
        static  int X2Y1 = 6;
        static  int X3Y1 = 7;
        static  int X0Y2 = 8;
        static  int X1Y2 = 9;
        static  int X2Y2 = 10;
        static  int X3Y2 = 11;
        static  int X0Y3 = 12;
        static  int X1Y3 = 13;
        static  int X2Y3 = 14;
        static  int X3Y3 = 15;
        public record Idx(F32Matrix4x4Pool pool, int idx) implements Pool.Idx<F32Matrix4x4Pool>, F32Matrix4x4 {
            private int x0y0Idx() {
                return idx * pool.stride + X0Y0;
            }

            private int x1y0Idx() {
                return idx * pool.stride + X1Y0;
            }

            private int x2y0Idx() {
                return idx * pool.stride + X2Y0;
            }

            private int x3y0Idx() {
                return idx * pool.stride + X3Y0;
            }

            private int x0y1Idx() {
                return idx * pool.stride + X0Y1;
            }

            private int x1y1Idx() {
                return idx * pool.stride + X1Y1;
            }

            private int x2y1Idx() {
                return idx * pool.stride + X2Y1;
            }

            private int x3y1Idx() {
                return idx * pool.stride + X3Y1;
            }

            int x0y2Idx() {
                return idx * pool.stride + X0Y2;
            }

            int x1y2Idx() {
                return idx * pool.stride + X1Y2;
            }

            int x2y2Idx() {
                return idx * pool.stride + X2Y2;
            }

            int x3y2Idx() {
                return idx * pool.stride + X3Y2;
            }

            int x0y3Idx() {
                return idx * pool.stride + X0Y3;
            }

            int x1y3Idx() {
                return idx * pool.stride + X1Y3;
            }

            int x2y3Idx() {
                return idx * pool.stride + X2Y3;
            }

            int x3y3Idx() {
                return idx * pool.stride + X3Y3;
            }

            @Override
            public float x0y0() {
                return pool.entries[x0y0Idx()];
            }

            @Override
            public float x1y0() {
                return pool.entries[x1y0Idx()];
            }

            @Override
            public float x2y0() {
                return pool.entries[x2y0Idx()];
            }

            @Override
            public float x3y0() {
                return pool.entries[x3y0Idx()];
            }

            @Override
            public float x0y1() {
                return pool.entries[x0y1Idx()];
            }

            @Override
            public float x1y1() {
                return pool.entries[x1y1Idx()];
            }

            @Override
            public float x2y1() {
                return pool.entries[x2y1Idx()];
            }

            @Override
            public float x3y1() {
                return pool.entries[x3y1Idx()];
            }

            @Override
            public float x0y2() {
                return pool.entries[x0y2Idx()];
            }

            @Override
            public float x1y2() {
                return pool.entries[x1y2Idx()];
            }

            @Override
            public float x2y2() {
                return pool.entries[x2y2Idx()];
            }

            @Override
            public float x3y2() {
                return pool.entries[x3y2Idx()];
            }

            @Override
            public float x0y3() {
                return pool.entries[x0y3Idx()];
            }

            @Override
            public float x1y3() {
                return pool.entries[x1y3Idx()];
            }

            @Override
            public float x2y3() {
                return pool.entries[x2y3Idx()];
            }

            @Override
            public float x3y3() {
                return pool.entries[x3y3Idx()];
            }
        }

        F32Matrix4x4Pool(int max) {
            super(16, max);
        }

        @Override
        Idx idx(int idx) {
            return new Idx(this, idx);
        }

        F32Matrix4x4 of(float x0y0, float x1y0, float x2y0, float x3y0,
                                float x0y1, float x1y1, float x2y1, float x3y1,
                                float x0y2, float x1y2, float x2y2, float x3y2,
                                float x0y3, float x1y3, float x2y3, float x3y3) {
            var i = idx(f32matrix4x4Pool.count++);
            entries[i.x0y0Idx()] = x0y0;
            entries[i.x1y0Idx()] = x1y0;
            entries[i.x2y0Idx()] = x2y0;
            entries[i.x3y0Idx()] = x3y0;
            entries[i.x0y1Idx()] = x0y1;
            entries[i.x1y1Idx()] = x1y1;
            entries[i.x2y1Idx()] = x2y1;
            entries[i.x3y1Idx()] = x3y1;
            entries[i.x0y2Idx()] = x0y2;
            entries[i.x1y2Idx()] = x1y2;
            entries[i.x2y2Idx()] = x2y2;
            entries[i.x3y2Idx()] = x3y2;
            entries[i.x0y3Idx()] = x0y3;
            entries[i.x1y3Idx()] = x1y3;
            entries[i.x2y3Idx()] = x2y3;
            entries[i.x3y3Idx()] = x3y3;
            return i;
        }

    }

    F32Matrix4x4Pool f32matrix4x4Pool = new F32Matrix4x4Pool(100);

    interface Impl extends F32Matrix4x4 {
        F32Matrix4x4 id();

        @Override
        default float x0y0() {
            return id().x0y0();
        }

        @Override
        default float x1y0() {
            return id().x1y0();
        }

        @Override
        default float x2y0() {
            return id().x2y0();
        }

        @Override
        default float x3y0() {
            return id().x3y0();
        }

        @Override
        default float x0y1() {
            return id().x0y1();
        }

        @Override
        default float x1y1() {
            return id().x1y1();
        }

        @Override
        default float x2y1() {
            return id().x2y1();
        }

        @Override
        default float x3y1() {
            return id().x3y1();
        }

        @Override
        default float x0y2() {
            return id().x0y2();
        }

        @Override
        default float x1y2() {
            return id().x1y2();
        }

        @Override
        default float x2y2() {
            return id().x2y2();
        }

        @Override
        default float x3y2() {
            return id().x3y2();
        }

        @Override
        default float x0y3() {
            return id().x0y3();
        }

        @Override
        default float x1y3() {
            return id().x1y3();
        }

        @Override
        default float x2y3() {
            return id().x2y3();
        }

        @Override
        default float x3y3() {
            return id().x3y3();
        }

        @Override
        default int idx() {
            return id().idx();
        }
    }



    //  https://stackoverflow.com/questions/28075743/how-do-i-compose-a-rotation-matrix-with-human-readable-angles-from-scratch/28084380#28084380
    static F32Matrix4x4 mulMat4(F32Matrix4x4 lhs, F32Matrix4x4 rhs) {
        return f32matrix4x4Pool.of(
                lhs.x0y0() * rhs.x0y0() + lhs.x1y0() * rhs.x0y1() + lhs.x2y0() * rhs.x0y2() + lhs.x3y0() * rhs.x0y3(),
                lhs.x0y0() * rhs.x1y0() + lhs.x1y0() * rhs.x1y1() + lhs.x2y0() * rhs.x1y2() + lhs.x3y0() * rhs.x1y3(),
                lhs.x0y0() * rhs.x2y0() + lhs.x1y0() * rhs.x2y1() + lhs.x2y0() * rhs.x2y2() + lhs.x3y0() * rhs.x2y3(),
                lhs.x0y0() * rhs.x3y0() + lhs.x1y0() * rhs.x3y1() + lhs.x2y0() * rhs.x3y2() + lhs.x3y0() * rhs.x3y3(),

                lhs.x0y1() * rhs.x0y0() + lhs.x1y1() * rhs.x0y1() + lhs.x2y1() * rhs.x0y2() + lhs.x3y1() * rhs.x0y3(),
                lhs.x0y1() * rhs.x1y0() + lhs.x1y1() * rhs.x1y1() + lhs.x2y1() * rhs.x1y2() + lhs.x3y1() * rhs.x1y3(),
                lhs.x0y1() * rhs.x2y0() + lhs.x1y1() * rhs.x2y1() + lhs.x2y1() * rhs.x2y2() + lhs.x3y1() * rhs.x2y3(),
                lhs.x0y1() * rhs.x3y0() + lhs.x1y1() * rhs.x3y1() + lhs.x2y1() * rhs.x3y2() + lhs.x3y1() * rhs.x3y3(),

                lhs.x0y2() * rhs.x0y0() + lhs.x1y2() * rhs.x0y1() + lhs.x2y2() * rhs.x0y2() + lhs.x3y2() * rhs.x0y3(),
                lhs.x0y2() * rhs.x1y0() + lhs.x1y2() * rhs.x1y1() + lhs.x2y2() * rhs.x1y2() + lhs.x3y2() * rhs.x1y3(),
                lhs.x0y2() * rhs.x2y0() + lhs.x1y2() * rhs.x2y1() + lhs.x2y2() * rhs.x2y2() + lhs.x3y2() * rhs.x2y3(),
                lhs.x0y2() * rhs.x3y0() + lhs.x1y2() * rhs.x3y1() + lhs.x2y2() * rhs.x3y2() + lhs.x3y2() * rhs.x3y3(),

                lhs.x0y3() * rhs.x0y0() + lhs.x1y3() * rhs.x0y1() + lhs.x2y3() * rhs.x0y2() + lhs.x3y3() * rhs.x0y3(),
                lhs.x0y3() * rhs.x1y0() + lhs.x1y3() * rhs.x1y1() + lhs.x2y3() * rhs.x1y2() + lhs.x3y3() * rhs.x1y3(),
                lhs.x0y3() * rhs.x2y0() + lhs.x1y3() * rhs.x2y1() + lhs.x2y3() * rhs.x2y2() + lhs.x3y3() * rhs.x2y3(),
                lhs.x0y3() * rhs.x3y0() + lhs.x1y3() * rhs.x3y1() + lhs.x2y3() * rhs.x3y2() + lhs.x3y3() * rhs.x3y3()

        );
    }




    //https://medium.com/swlh/understanding-3d-matrix-transforms-with-pixijs-c76da3f8bd8
    // record Transformation(F32Matrix4x4Pool id) implements Impl {
    static F32Matrix4x4 transformation(float x, float y, float z) {
        return f32matrix4x4Pool.of(
                1f, 0f, 0f, 0f,
                0f, 1f, 0f, 0f,
                0f, 0f, 1f, 0f,
                x, y, z, 1f
        );
    }

    static F32Matrix4x4 transformation(float v) {
        return transformation(v, v, v);
    }


    // https://medium.com/swlh/understanding-3d-matrix-transforms-with-pixijs-c76da3f8bd8


    static F32Matrix4x4 scale(float x, float y, float z) {
        return f32matrix4x4Pool.of(
                x, 0f, 0f, 0f,
                0f, y, 0f, 0f,
                0f, 0f, z, 0f,
                0f, 0f, 0f, 1f

        );
    }

    static F32Matrix4x4 scale(float v) {
        return scale(v, v, v);
    }

    static F32Matrix4x4 rotationX(float thetaRadians) {
        float sinTheta = (float) Math.sin(thetaRadians);
        float cosTheta = (float) Math.cos(thetaRadians);
        return f32matrix4x4Pool.of(
                1, 0, 0, 0,
                0, cosTheta, -sinTheta, 0,
                0, sinTheta, cosTheta, 0,
                0, 0, 0, 1

        );
    }

    static F32Matrix4x4 rotationZ(float thetaRadians) {
        float sinTheta = (float) Math.sin(thetaRadians);
        float cosTheta = (float) Math.cos(thetaRadians);
        return f32matrix4x4Pool.of(
                cosTheta, sinTheta, 0, 0,
                -sinTheta, cosTheta, 0, 0,
                0, 0, 1, 0,
                0, 0, 0, 1
        );
    }

    static F32Matrix4x4 rotationY(float thetaRadians) {
        float sinTheta = (float) Math.sin(thetaRadians);
        float cosTheta = (float) Math.cos(thetaRadians);
        return f32matrix4x4Pool.of(
                cosTheta, 0, sinTheta, 0,
                0, 1, 0, 0,
                -sinTheta, 0, cosTheta, 0,
                0, 0, 0, 1
        );
    }


    static F32Matrix4x4 rotation(float thetaX, float thetaY, float thetaZ) {
        return F32Matrix4x4.mulMat4(F32Matrix4x4.mulMat4(rotationX(thetaX), rotationY(thetaY)), rotationZ(thetaZ));
    }



    /*
                 https://youtu.be/ih20l3pJoeU?t=973
                 https://stackoverflow.com/questions/28075743/how-do-i-compose-a-rotation-matrix-with-human-readable-angles-from-scratch/28084380#28084380^
                --------------------            far
                 \                /              ^    ^
                  \              /               |    |   far-near
                   \            /                |    |
                    \__________/         near    |    v
                                          ^      |
                                          v      v
                        \^/
                      [x,y,z]

               */

    static F32Matrix4x4 projection(float width, float height, float near, float far, float fieldOfViewDeg) {
        float aspectRatio = height / width;
        float fieldOfViewRadians = (float) (1.0f / Math.tan((fieldOfViewDeg * 0.5f) / 180.0 * Math.PI));
        return f32matrix4x4Pool.of(
                aspectRatio * fieldOfViewRadians, 0f, 0f, 0f,
                0f, fieldOfViewRadians, 0f, 0f,
                0f, 0f, far / (far - near), (-far * near) / (far - near),
                    0f, 0f, (-far * near) / (far - near), 0f);

    }

    //https://medium.com/swlh/understanding-3d-matrix-transforms-with-pixijs-c76da3f8bd8
    record Transformation(F32Matrix4x4 id) implements Impl {
        public Transformation(float x, float y, float z) {
            this(F32Matrix4x4.f32matrix4x4Pool.of(
                    1f, 0f, 0f, 0f,
                    0f, 1f, 0f, 0f,
                    0f, 0f, 1f, 0f,
                    x, y, z, 1f
            ));
        }

        public static Transformation of(float v) {
            return new Transformation(v, v, v);
        }
    }

    // https://medium.com/swlh/understanding-3d-matrix-transforms-with-pixijs-c76da3f8bd8

    record Scale(F32Matrix4x4 id) implements Impl {
        Scale(float x, float y, float z) {
            this(F32Matrix4x4.f32matrix4x4Pool.of(
                            x, 0f, 0f, 0f,
                            0f, y, 0f, 0f,
                            0f, 0f, z, 0f,
                            0f, 0f, 0f, 1f
                    )
            );
        }

        public static Scale of(float v) {
            return new Scale(v, v, v);
        }
    }

    record Rotation(F32Matrix4x4 id) implements Impl {

        static F32Matrix4x4 ofX(float thetaRadians) {
            float sinTheta = (float) Math.sin(thetaRadians);
            float cosTheta = (float) Math.cos(thetaRadians);
            return F32Matrix4x4.f32matrix4x4Pool.of(
                    1, 0, 0, 0,
                    0, cosTheta, -sinTheta, 0,
                    0, sinTheta, cosTheta, 0,
                    0, 0, 0, 1

            );
        }

        static F32Matrix4x4 ofZ(float thetaRadians) {
            float sinTheta = (float) Math.sin(thetaRadians);
            float cosTheta = (float) Math.cos(thetaRadians);
            return F32Matrix4x4.f32matrix4x4Pool.of(
                    cosTheta, sinTheta, 0, 0,
                    -sinTheta, cosTheta, 0, 0,
                    0, 0, 1, 0,
                    0, 0, 0, 1
            );
        }

        static F32Matrix4x4 ofY(float thetaRadians) {
            float sinTheta = (float) Math.sin(thetaRadians);
            float cosTheta = (float) Math.cos(thetaRadians);
            return F32Matrix4x4.f32matrix4x4Pool.of(
                    cosTheta, 0, sinTheta, 0,
                    0, 1, 0, 0,
                    -sinTheta, 0, cosTheta, 0,
                    0, 0, 0, 1
            );
        }


        public Rotation(float thetaX, float thetaY, float thetaZ) {
            this(F32Matrix4x4.mulMat4(F32Matrix4x4.mulMat4(ofX(thetaX), ofY(thetaY)), ofZ(thetaZ)));
        }

    }

    /*
                 https://youtu.be/ih20l3pJoeU?t=973
                 https://stackoverflow.com/questions/28075743/how-do-i-compose-a-rotation-matrix-with-human-readable-angles-from-scratch/28084380#28084380^
                --------------------            far
                 \                /              ^    ^
                  \              /               |    |   far-near
                   \            /                |    |
                    \__________/         near    |    v
                                          ^      |
                                          v      v
                        \^/
                      [x,y,z]

               */
    record Projection(F32Matrix4x4 id) implements Impl {
        public static Projection of(F32Matrix4x4 id) {
            return new Projection(id);
        }

        public static Projection of(float width, float height, float near, float far, float fieldOfViewDeg) {
            float aspectRatio = height / width;
            float fieldOfViewRadians = (float) (1.0f / Math.tan((fieldOfViewDeg * 0.5f) / 180.0 * Math.PI));
            return of(F32Matrix4x4.f32matrix4x4Pool.of(
                    aspectRatio * fieldOfViewRadians, 0f, 0f, 0f,
                    0f, fieldOfViewRadians, 0f, 0f,
                    0f, 0f, far / (far - near), (-far * near) / (far - near),
                    0f, 0f, (-far * near) / (far - near), 0f));

        }
    }



}
