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
package view;

import java.awt.Image;
import java.util.ArrayList;
import java.util.List;

public interface F32 {
    static Vec3 mulMat4(Vec3 vec3, Mat4x4 m4) {
        var o = Vec3.of(
                vec3.x() * m4.x0y0() + vec3.y() * m4.x0y1() + vec3.z() * m4.x0y2() + m4.x0y3(),
                vec3.x() * m4.x1y0() + vec3.y() * m4.x1y1() + vec3.z() * m4.x1y2() + m4.x1y3(),
                vec3.x() * m4.x2y0() + vec3.y() * m4.x2y1() + vec3.z() * m4.x2y2() + m4.x2y3()
        );

        float w = vec3.x() * m4.x3y0() + vec3.y() * m4.x3y1() + vec3.z() * m4.x3y2() + m4.x3y3();
        if (w != 0.0) {
            o = divScaler(o, w);
        }
        return o;
    }

    static Vec3 divScaler(Vec3 vec3, float s) {
        return Vec3.of(vec3.x() / s, vec3.y() / s, vec3.z() / s);
    }

    static Vec3 add(Vec3 lhs, Vec3 rhs) {
        return Vec3.of(lhs.x() + rhs.x(), lhs.y() + rhs.y(), lhs.z() + rhs.z());
    }

    static Vec3 sub(Vec3 lhs, Vec3 rhs) {
        return Vec3.of(lhs.x() - rhs.x(), lhs.y() - rhs.y(), lhs.z() - rhs.z());
    }

    static float sumOfSquares(Vec3 vec3) {
        return vec3.x() * vec3.x() + vec3.y() * vec3.y() + vec3.z() * vec3.z();
    }

    static Vec3 xprod(Vec3 lhs, Vec3 rhs) {
        return Vec3.of(
                lhs.y() * rhs.z() - lhs.z() * rhs.x(),
                lhs.z() * rhs.x() - lhs.x() * rhs.z(),
                lhs.x() * rhs.y() - lhs.y() * rhs.x());

    }

    static float dotprod(Vec3 lhs, Vec3 rhs) {
        return lhs.x() * rhs.x() + lhs.y() * rhs.y() + lhs.z() * rhs.z();
    }

    static TriangleVec3 mul(TriangleVec3 triangleVec3, Mat4x4 m4) {

        return TriangleVec3.of(mulMat4(triangleVec3.v0(), m4), mulMat4(triangleVec3.v1(), m4), mulMat4(triangleVec3.v2(), m4), triangleVec3.rgb());
    }

    static TriangleVec3 add(TriangleVec3 triangleVec3, Vec3 v3) {
        return TriangleVec3.of(add(triangleVec3.v0(), v3), add(triangleVec3.v1(), v3), add(triangleVec3.v2(), v3), triangleVec3.rgb());
    }

    static Vec3 center(TriangleVec3 triangleVec3) {
        return divScaler(vectorSum(triangleVec3), 3);
    }

    static Vec3 vectorSum(TriangleVec3 triangleVec3) {
        return add(add(triangleVec3.v0(), triangleVec3.v1()), triangleVec3.v2());
    }

    static Vec3 normal(TriangleVec3 triangleVec3) {
        return xprod(sub(triangleVec3.v1(), triangleVec3.v0()), sub(triangleVec3.v2(), triangleVec3.v0()));
    }

    interface Mat4x4 {
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

        interface Mutable extends Mat4x4 {
            void x0y0(float x0y0);

            void x1y0(float x1y0);

            void x2y0(float x2y0);

            void x3y0(float x3y0);

            void x0y1(float x0y1);

            void x1y1(float x1y1);

            void x2y1(float x2y1);

            void x3y1(float x3y1);

            void x0y2(float x0y2);

            void x1y2(float x1y2);

            void x2y2(float x2y2);

            void x3y2(float x3y2);

            void x0y3(float x0y3);

            void x1y3(float x1y3);

            void x2y3(float x2y3);

            void x3y3(float x3y3);
        }



        //  https://stackoverflow.com/questions/28075743/how-do-i-compose-a-rotation-matrix-with-human-readable-angles-from-scratch/28084380#28084380
        static Impl mul(Mat4x4 lhs, Mat4x4 rhs) {
            return new Impl(
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


        static String asString(Mat4x4 mat4X4) {

            return String.format("|%5.2f, %5.2f, %5.2f, %5.2f|\n" +
                            "|%5.2f, %5.2f, %5.2f, %5.2f|\n" +
                            "|%5.2f, %5.2f, %5.2f, %5.2f|\n" +
                            "|%5.2f, %5.2f, %5.2f, %5.2f|\n",
                    mat4X4.x0y0(), mat4X4.x1y0(), mat4X4.x2y0(), mat4X4.x3y0(),
                    mat4X4.x0y1(), mat4X4.x1y1(), mat4X4.x2y1(), mat4X4.x3y1(),
                    mat4X4.x0y2(), mat4X4.x1y2(), mat4X4.x2y2(), mat4X4.x3y2(),
                    mat4X4.x0y3(), mat4X4.x1y3(), mat4X4.x2y3(), mat4X4.x3y3());
        }


        class Impl implements Mat4x4 {

            float x0y0;

            @Override
            public float x0y0() {
                return x0y0;
            }

            float x0y1;

            @Override
            public float x0y1() {
                return x0y1;
            }

            float x0y2;

            @Override
            public float x0y2() {
                return x0y2;
            }

            float x0y3;

            @Override
            public float x0y3() {
                return x0y3;
            }

            float x1y0;

            @Override
            public float x1y0() {
                return x1y0;
            }

            float x1y1;

            @Override
            public float x1y1() {
                return x1y1;
            }

            float x1y2;

            @Override
            public float x1y2() {
                return x1y2;
            }

            float x1y3;

            @Override
            public float x1y3() {
                return x1y3;
            }

            float x2y0;

            @Override
            public float x2y0() {
                return x2y0;
            }

            float x2y1;

            @Override
            public float x2y1() {
                return x2y1;
            }

            float x2y2;

            @Override
            public float x2y2() {
                return x2y2;
            }

            float x2y3;

            @Override
            public float x2y3() {
                return x2y3;
            }

            float x3y0;

            @Override
            public float x3y0() {
                return x3y0;
            }

            float x3y1;

            @Override
            public float x3y1() {
                return x3y1;
            }

            float x3y2;

            @Override
            public float x3y2() {
                return x3y2;
            }

            float x3y3;

            @Override
            public float x3y3() {
                return x3y3;
            }

            Impl(float x0y0, float x1y0, float x2y0, float x3y0,
                 float x0y1, float x1y1, float x2y1, float x3y1,
                 float x0y2, float x1y2, float x2y2, float x3y2,
                 float x0y3, float x1y3, float x2y3, float x3y3) {
                this.x0y0 = x0y0;
                this.x1y0 = x1y0;
                this.x2y0 = x2y0;
                this.x3y0 = x3y0;
                this.x0y1 = x0y1;
                this.x1y1 = x1y1;
                this.x2y1 = x2y1;
                this.x3y1 = x3y1;
                this.x0y2 = x0y2;
                this.x1y2 = x1y2;
                this.x2y2 = x2y2;
                this.x3y2 = x3y2;
                this.x0y3 = x0y3;
                this.x1y3 = x1y3;
                this.x2y3 = x2y3;
                this.x3y3 = x3y3;
            }
        }

        class MutableImpl extends Impl implements Mutable {

            @Override
            public void x0y0(float x0y0) {
                this.x0y0 = x0y0;
            }

            @Override
            public void x0y1(float x0y1) {
                this.x0y1 = x0y1;
            }

            @Override
            public void x0y2(float x0y2) {
                this.x0y2 = x0y2;
            }

            @Override
            public void x0y3(float x0y3) {
                this.x0y3 = x0y3;
            }

            @Override
            public void x1y0(float x1y0) {
                this.x1y0 = x1y0;
            }

            @Override
            public void x1y1(float x1y1) {
                this.x1y1 = x1y1;
            }

            @Override
            public void x1y2(float x1y2) {
                this.x1y2 = x1y2;
            }

            @Override
            public void x1y3(float x1y3) {
                this.x1y3 = x1y3;
            }

            @Override
            public void x2y0(float x2y0) {
                this.x2y0 = x2y0;
            }

            @Override
            public void x2y1(float x2y1) {
                this.x2y1 = x2y1;
            }

            @Override
            public void x2y2(float x2y2) {
                this.x2y2 = x2y2;
            }

            @Override
            public void x2y3(float x2y3) {
                this.x2y3 = x2y3;
            }

            @Override
            public void x3y0(float x3y0) {
                this.x3y0 = x3y0;
            }

            @Override
            public void x3y1(float x3y1) {
                this.x3y1 = x3y1;
            }

            @Override
            public void x3y2(float x3y2) {
                this.x3y2 = x3y2;
            }

            @Override
            public void x3y3(float x3y3) {
                this.x3y3 = x3y3;
            }

            MutableImpl(float x0y0, float x1y0, float x2y0, float x3y0,
                        float x0y1, float x1y1, float x2y1, float x3y1,
                        float x0y2, float x1y2, float x2y2, float x3y2,
                        float x0y3, float x1y3, float x2y3, float x3y3) {
                super(x0y0, x1y0, x2y0, x3y0,
                        x0y1, x1y1, x2y1, x3y1,
                        x0y2, x1y2, x2y2, x3y2,
                        x0y3, x1y3, x2y3, x3y3);
            }
        }


        //https://medium.com/swlh/understanding-3d-matrix-transforms-with-pixijs-c76da3f8bd8
        record Transformation(float x0y0, float x1y0, float x2y0, float x3y0,
                              float x0y1, float x1y1, float x2y1, float x3y1,
                              float x0y2, float x1y2, float x2y2, float x3y2,
                              float x0y3, float x1y3, float x2y3, float x3y3) implements Mat4x4 {
            public Transformation(float x, float y, float z) {
                this(1f, 0f, 0f, 0f, 0f, 1f, 0f, 0f, 0f, 0f, 1f, 0f, x, y, z, 1f);
            }

            public static Transformation of(float v) {
                return new Transformation(v, v, v);
            }
        }

        // https://medium.com/swlh/understanding-3d-matrix-transforms-with-pixijs-c76da3f8bd8

        record Scale(float x0y0, float x1y0, float x2y0, float x3y0,
                     float x0y1, float x1y1, float x2y1, float x3y1,
                     float x0y2, float x1y2, float x2y2, float x3y2,
                     float x0y3, float x1y3, float x2y3, float x3y3) implements Mat4x4 {
            Scale(float x, float y, float z) {
                this(x, 0f, 0f, 0f, 0f, y, 0f, 0f, 0f, 0f, z, 0f, 0f, 0f, 0f, 1f);
            }

            public static Scale of(float v) {
                return new Scale(v, v, v);
            }
        }

        record Rotation(float x0y0, float x1y0, float x2y0, float x3y0,
                        float x0y1, float x1y1, float x2y1, float x3y1,
                        float x0y2, float x1y2, float x2y2, float x3y2,
                        float x0y3, float x1y3, float x2y3, float x3y3) implements Mat4x4 {

            static Rotation ofX(float thetaRadians) {
                float sinTheta = (float) Math.sin(thetaRadians);
                float cosTheta = (float) Math.cos(thetaRadians);
                return new Rotation(1, 0, 0, 0, 0, cosTheta, -sinTheta, 0, 0, sinTheta, cosTheta, 0, 0, 0, 0, 1);
            }

            static Rotation ofZ(float thetaRadians) {
                float sinTheta = (float) Math.sin(thetaRadians);
                float cosTheta = (float) Math.cos(thetaRadians);
                return new Rotation(cosTheta, sinTheta, 0, 0, -sinTheta, cosTheta, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1);
            }

            static Rotation ofY(float thetaRadians) {
                float sinTheta = (float) Math.sin(thetaRadians);
                float cosTheta = (float) Math.cos(thetaRadians);
                return new Rotation(cosTheta, 0, sinTheta, 0, 0, 1, 0, 0, -sinTheta, 0, cosTheta, 0, 0, 0, 0, 1);
            }

            public static Rotation of(float thetaX, float thetaY, float thetaZ) {
                var m4 = Mat4x4.mul(Mat4x4.mul(ofX(thetaX), ofY(thetaY)), ofZ(thetaZ));
                return new Rotation(m4.x0y0(), m4.x1y0(), m4.x2y0(), m4.x3y0(), m4.x0y1(), m4.x1y1(), m4.x2y1(), m4.x3y1(), m4.x0y2(), m4.x1y2(), m4.x2y2(), m4.x3y2(), m4.x0y3(), m4.x1y3(), m4.x2y3(), m4.x3y3());
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
        record Projection(float x0y0, float x1y0, float x2y0, float x3y0,
                          float x0y1, float x1y1, float x2y1, float x3y1,
                          float x0y2, float x1y2, float x2y2, float x3y2,
                          float x0y3, float x1y3, float x2y3, float x3y3) implements Mat4x4 {

            static Projection of(float width, float height, float near, float far, float fieldOfViewDeg) {
                float aspectRatio = height / width;
                float fieldOfViewRadians = (float) (1.0f / Math.tan((fieldOfViewDeg * 0.5f) / 180.0 * Math.PI));
                return new Projection(
                        aspectRatio * fieldOfViewRadians, 0f, 0f, 0f,
                        0f, fieldOfViewRadians, 0f, 0f,
                        0f, 0f, far / (far - near), (-far * near) / (far - near),
                        0f, 0f, (-far * near) / (far - near), 0f);
            }
        }

    }

    interface TriangleVec2 {
        List<TriangleVec2> arr = new ArrayList<>();

        Vec2 v0();

        void v0(Vec2 v0);

        Vec2 v1();

        void v1(Vec2 v1);

        Vec2 v2();

        void v2(Vec2 v2);

        int rgb();

        void rgb(int rgb);

        class Impl implements TriangleVec2 {
            Vec2 v0, v1, v2;
            int rgb;

            @Override
            public Vec2 v0() {
                return v0;
            }

            @Override
            public void v0(Vec2 v0) {
                this.v0 = v0;
            }

            @Override
            public Vec2 v1() {
                return v1;
            }

            @Override
            public void v1(Vec2 v1) {
                this.v1 = v1;
            }

            @Override
            public Vec2 v2() {
                return v2;
            }

            @Override
            public void v2(Vec2 v2) {
                this.v2 = v2;
            }

            @Override
            public int rgb() {
                return rgb;
            }

            @Override
            public void rgb(int rgb) {
                this.rgb = rgb;
            }

            Impl(Vec2 v0, Vec2 v1, Vec2 v2, int rgb) {
                v0(v0);
                v1(v1);
                v2(v2);
                rgb(rgb);
            }
        }

        static float side(float x, float y, float x0, float y0, float x1, float y1) {
            return (y1 - y0) * (x - x0) + (-x1 + x0) * (y - y0);
        }

        static float side(Vec2 v, Vec2 v0, Vec2 v1) {

            return (v1.y() - v0.y() * (v.x() - v0.x()) + (-v1.x() + v0.x()) * (v.y() - v0.y()));
        }

        static boolean intriangle(float x, float y, float x0, float y0, float x1, float y1, float x2, float y2) {
            return side(x, y, x0, y0, x1, y1) >= 0 && side(x, y, x1, y1, x2, y2) >= 0 && side(x, y, x2, y2, x0, y0) >= 0;
        }

        static boolean intriangle(Vec2 v, Vec2 v0, Vec2 v1, Vec2 v2) {
            return side(v, v0, v1) >= 0 && side(v, v1, v2) >= 0 && side(v, v2, v0) >= 0;
        }

        static boolean online(float x, float y, float x0, float y0, float x1, float y1, float deltaSquare) {
            float dxl = x1 - x0;
            float dyl = y1 - y0;
            float cross = (x - x0) * dyl - (y - y0) * dxl;
            if ((cross * cross) < deltaSquare) {
                if (dxl * dxl >= dyl * dyl)
                    return dxl > 0 ? x0 <= x && x <= x1 : x1 <= x && x <= x0;
                else
                    return dyl > 0 ? y0 <= y && y <= y1 : y1 <= y && y <= y0;
            } else {
                return false;
            }
        }

        float deltaSquare = 10000f;

        static boolean onedge(float x, float y, float x0, float y0, float x1, float y1, float x2, float y2) {
            return online(x, y, x0, y0, x1, y1, deltaSquare) || TriangleVec2.online(x, y, x1, y1, x2, y2, deltaSquare) || TriangleVec2.online(x, y, x2, y2, x0, y0, deltaSquare);
        }


        static Impl of(int x0, int y0, int x1, int y1, int x2, int y2, int col) {
            var impl = side(x0, y0, x1, y1, x2, y2) > 0 // We need the triangle to be clock wound
                    ? new Impl(Vec2.of(x0, y0), Vec2.of(x1, y1), Vec2.of(x2, y2), col)
                    : new Impl(Vec2.of(x0, y0), Vec2.of(x2, y2), Vec2.of(x1, y1), col);
            arr.add(impl);
            return impl;
        }

    }

    interface TriangleVec3 {

        List<TriangleVec3> arr = new ArrayList<>();

        Vec3 v0();

        void v0(Vec3 v0);

        Vec3 v1();

        void v1(Vec3 v1);

        Vec3 v2();

        void v2(Vec3 v2);

        int rgb();

        void rgb(int rgb);



         /*
           v0----v1         v0----v2
            \    |           \    |
             \   |            \   |
              \  |    --->     \  |
               \ |              \ |
                \|               \|
                 v2               v1
       */

        static void rewind(TriangleVec3 triangleVec3) {
            Vec3 temp = triangleVec3.v1();
            triangleVec3.v1(triangleVec3.v2());
            triangleVec3.v2(temp);
        }

        static TriangleVec3 of(Vec3 v0, Vec3 v1, Vec3 v2, int rgb) {
            var impl = new Impl(v0, v1, v2, rgb);
            arr.add(impl);
            return impl;
        }

        static String asString(TriangleVec3 triangleVec3) {
            return Vec3.asString(triangleVec3.v0()) + " -> " + Vec3.asString(triangleVec3.v1()) + " -> " + Vec3.asString(triangleVec3.v2()) + " =" + String.format("0x%8x", triangleVec3.rgb());
        }


        class Impl implements TriangleVec3 {
            Vec3 v0, v1, v2;
            int rgb;

            @Override
            public Vec3 v0() {
                return v0;
            }

            @Override
            public void v0(Vec3 v0) {
                this.v0 = v0;
            }

            @Override
            public Vec3 v1() {
                return v1;
            }

            @Override
            public void v1(Vec3 v1) {
                this.v1 = v1;
            }

            @Override
            public Vec3 v2() {
                return v2;
            }

            @Override
            public void v2(Vec3 v2) {
                this.v2 = v2;
            }

            @Override
            public int rgb() {
                return rgb;
            }

            @Override
            public void rgb(int rgb) {
                this.rgb = rgb;
            }

            Impl(Vec3 v0, Vec3 v1, Vec3 v2, int rgb) {
                v0(v0);
                v1(v1);
                v2(v2);
                rgb(rgb);
            }

        }
    }

    interface Vec2 {
        float x();

        void x(float x);

        float y();

        void y(float y);

        class Impl implements Vec2 {
            float x, y;

            @Override
            public float x() {
                return x;
            }

            @Override
            public void x(float x) {
                this.x = x;
            }

            @Override
            public float y() {
                return y;
            }

            @Override
            public void y(float y) {
                this.y = y;
            }

            Impl(float x, float y) {
                x(x);
                y(y);
            }
        }

        static Impl of(float x, float y) {
            return new Impl(x, y);
        }

    }

    interface Vec3 {

        float x();

        void x(float x);

        float y();

        void y(float y);

        float z();

        void z(float z);

        static Vec3 of(float x, float y, float z) {
            return new Impl(z, y, z);
        }


        /*
            lhs= | 1|   rhs= | 2|
                 | 3|        | 7|
                 | 4|        |-5|

            lhs xprod rhs = | x  y  z| =  | 3  4|x - | 1  4|y  | 1  3|
                            | 1  3  4|    | 7 -5|    | 2 -5|   | 2  7|
                            | 2  7 -5|

                          = (-15-28)x - (-5 -8)y + (7 - 6)z

                          = -43x - (-13)y +1z
                          = -43x + 14y +z

         */

        /*
            lhs= | 1|   rhs= | 2|
                 | 3|        | 7|
                 | 4|        |-5|

            lhs0*rhs0 + lhs1*rhs1 + lhs2*rhs2
             1  * 2   +  3  * 7   +  4  *-5

                3     +    21     +   -20

                           4

         */

        static String asString(Vec3 vec3) {
            return vec3.x() + "," + vec3.y() + "," + vec3.z();
        }


        class Impl implements Vec3 {
            float x, y, z;

            @Override
            public float x() {
                return x;
            }

            @Override
            public void x(float x) {
                this.x = x;
            }

            @Override
            public float y() {
                return y;
            }

            @Override
            public void y(float y) {
                this.y = y;
            }

            @Override
            public float z() {
                return z;
            }

            @Override
            public void z(float z) {
                this.z = z;
            }

            Impl(float x, float y, float z) {
                x(x);
                y(y);
                z(z);
            }
        }
    }

    class Mesh {
        String name;

        Vec3 triSum;

        private Mesh(String name) {
            this.name = name;
        }

        public static Mesh of(String name) {
            return new Mesh(name);
        }


        public record triInfo(TriangleVec3 tri, Vec3 centre, Vec3 normal, Vec3 v0) {
            static triInfo of(TriangleVec3 tri) {
                return new triInfo(tri, F32.center(tri), F32.normal(tri), tri.v0());
            }
        }

        public List<triInfo> triInfos = new ArrayList<>();
        public List<Vec3> vecEntries = new ArrayList<>();

        public void tri(Vec3 v0, Vec3 v1, Vec3 v2, int rgb) {
            var trii = triInfo.of(TriangleVec3.of(v0, v1, v2, rgb));
            triSum = (triInfos.isEmpty()) ? trii.centre() : add(triSum, trii.centre());
            triInfos.add(trii);

        }

        public void fin() {
            Vec3 meshCenterVec3 = divScaler(triSum, triInfos.size());
            for (int t = 0; t < triInfos.size(); t++) {
                var trii = triInfos.get(t);
                var v0CenterDiff = sub(meshCenterVec3, trii.centre);
                float normDotProd = dotprod(v0CenterDiff, trii.normal);
                if (normDotProd > 0f) { // the normal from the center from the triangle was pointing out, so re wind it
                    TriangleVec3.rewind(trii.tri);
                }
            }
            cube(meshCenterVec3.x(), meshCenterVec3.y(), meshCenterVec3.z(), .1f);
        }

        public Mesh quad(Vec3 v0, Vec3 v1, Vec3 v2, Vec3 v3, int rgb) {
      /*
           v0-----v1
            |\    |
            | \   |
            |  \  |
            |   \ |
            |    \|
           v3-----v2
       */

            tri(v0, v1, v2, rgb);
            tri(v0, v2, v3, rgb);
            return this;
        }

        public Mesh pent(Vec3 v0, Vec3 v1, Vec3 v2, Vec3 v3, Vec3 v4, int rgb) {
      /*
           v0-----v1
           |\    | \
           | \   |  \
           |  \  |   v2
           |   \ |  /
           |    \| /
           v4-----v3
       */

            tri(v0, v1, v3, rgb);
            tri(v1, v2, v3, rgb);
            tri(v0, v3, v4, rgb);
            return this;
        }

        public Mesh hex(Vec3 v0, Vec3 v1, Vec3 v2, Vec3 v3, Vec3 v4, Vec3 v5, int rgb) {
      /*
           v0-----v1
          / |\    | \
         /  | \   |  \
        v5  |  \  |   v2
         \  |   \ |  /
          \ |    \| /
           v4-----v3
       */

            tri(v0, v1, v3, rgb);
            tri(v1, v2, v3, rgb);
            tri(v0, v3, v4, rgb);
            tri(v0, v4, v5, rgb);
            return this;
        }


        /*
                   a-----------d
                  /|          /|
                 / |         / |
               h------------g  |
               |   |        |  |
               |   b--------|--c
               |  /         | /
               | /          |/
               e------------f

         */


        public Mesh cube(
                float x,
                float y,
                float z,
                float s) {
            var a = vec3(x - (s * .5f), y - (s * .5f), z - (s * .5f));  //000  000 111 111
            var b = vec3(x - (s * .5f), y + (s * .5f), z - (s * .5f));  //010  010 101 101
            var c = vec3(x + (s * .5f), y + (s * .5f), z - (s * .5f));  //110  011 001 100
            var d = vec3(x + (s * .5f), y - (s * .5f), z - (s * .5f));  //100  001 011 110
            var e = vec3(x - (s * .5f), y + (s * .5f), z + (s * .5f));  //011  110 100 001
            var f = vec3(x + (s * .5f), y + (s * .5f), z + (s * .5f));  //111  111 000 000
            var g = vec3(x + (s * .5f), y - (s * .5f), z + (s * .5f));  //101  101 010 010
            var h = vec3(x - (s * .5f), y - (s * .5f), z + (s * .5f));  //001  100 110 011
            quad(a, b, c, d, 0xff0000); //front
            quad(b, e, f, c, 0x0000ff); //top
            quad(d, c, f, g, 0xffff00); //right
            quad(h, e, b, a, 0xffffff); //left
            quad(g, f, e, h, 0x00ff00);//back
            quad(g, h, a, d, 0xffa500);//bottom
            return this;
        }


        /*
        http://paulbourke.net/dataformats/obj/
         */
        public Mesh cubeoctahedron(
                float x,
                float y,
                float z,
                float s) {

            var v1 = vec3(x + (s * .30631559f), y + (s * .20791225f), z + (s * .12760004f));
            var v2 = vec3(x + (s * .12671047f), y + (s * .20791227f), z + (s * .30720518f));
            var v3 = vec3(x + (s * .12671045f), y + (s * .38751736f), z + (s * .12760002f));
            var v4 = vec3(x + (s * .30631556f), y + (s * .20791227f), z + (s * .48681026f));
            var v5 = vec3(x + (s * .48592068f), y + (s * .20791225f), z + (s * .30720514f));
            var v6 = vec3(x + (s * .30631556f), y + (s * .56712254f), z + (s * .48681026f));
            var v7 = vec3(x + (s * .12671047f), y + (s * .56712254f), z + (s * .30720512f));
            var v8 = vec3(x + (s * .12671042f), y + (s * .3875174f), z + (s * .48681026f));
            var v9 = vec3(x + (s * .48592068f), y + (s * .38751736f), z + (s * .1276f));
            var v10 = vec3(x + (s * .30631556f), y + (s * .56712254f), z + (s * .1276f));
            var v11 = vec3(x + (s * .48592068f), y + (s * .56712254f), z + (s * .30720512f));
            var v12 = vec3(x + (s * .48592068f), y + (s * .38751743f), z + (s * .4868103f));

            tri(v1, v2, v3, 0xff0000);
            tri(v4, v2, v5, 0x7f8000);
            tri(v5, v2, v1, 0x3fc000);
            tri(v6, v7, v8, 0x1fe000);
            tri(v9, v10, v11, 0x0ff000);
            tri(v8, v2, v4, 0x07f800);
            tri(v5, v1, v9, 0x03fc00);
            tri(v3, v7, v10, 0x01fe00);
            tri(v8, v7, v2, 0x00ff00);
            tri(v2, v7, v3, 0x007f80);
            tri(v8, v4, v6, 0x003fc0);
            tri(v6, v4, v12, 0x001fe0);
            tri(v11, v12, v9, 0x000ff0);
            tri(v9, v12, v5, 0x0007f8);
            tri(v7, v6, v10, 0x0003fc);
            tri(v6, v11, v10, 0x0001fe);
            tri(v1, v3, v9, 0x0000ff);
            tri(v9, v3, v10, 0x00007f);
            tri(v12, v4, v5, 0x00003f);
            tri(v6, v12, v11, 0x00001f);
            return this;
        }


        public Mesh rubric(float s) {
            for (int x = -1; x < 2; x++) {
                for (int y = -1; y < 2; y++) {
                    for (int z = -1; z < 2; z++) {
                        cube(x * .5f, y * .5f, z * .5f, s);
                    }
                }
            }
            return this;
        }

        public Vec3 vec3(float x, float y, float z) {
            vecEntries.add(Vec3.of(x, y, z));
            return vecEntries.getLast();
        }
    }

    class ZPos implements Comparable<ZPos> {
        public enum ColourMode {NORMALIZED_COLOUR, NORMALIZED_INV_COLOUR, COLOUR, NORMALIZED_WHITE, NORMALIZED_INV_WHITE, WHITE}
        public static final ColourMode colourMode = ColourMode.COLOUR;

        int x0, y0, x1, y1, x2, y2;
        float z0, z1, z2;
        float z;
        float howVisible;
        int rgb;

        @Override
        public int compareTo(ZPos zPos) {
            return Float.compare(z, zPos.z);
        }

        ZPos(TriangleVec3 t, float howVisible) {
            Vec3 v0 = t.v0();
            Vec3 v1 = t.v1();
            Vec3 v2 = t.v2();
            x0 = (int) v0.x();
            y0 = (int) v0.y();
            z0 = v0.z();
            x1 = (int) v1.x();
            y1 = (int) v1.y();
            z1 = v1.z();
            x2 = (int) v2.x();
            y2 = (int) v2.y();
            z2 = v2.z();
            this.rgb = t.rgb();
            this.howVisible = howVisible;
            z = Math.min(z0, Math.min(z1, z2));
        }


        TriangleVec2 create() {
            int r = ((rgb & 0xff0000) >> 16);
            int g = ((rgb & 0x00ff00) >> 8);
            int b = ((rgb & 0x0000ff) >> 0);

            if (colourMode == ColourMode.NORMALIZED_COLOUR) {
                r = r - (int) (20 * howVisible);
                g = g - (int) (20 * howVisible);
                b = b - (int) (20 * howVisible);
            } else if (colourMode == ColourMode.NORMALIZED_INV_COLOUR) {
                r = r + (int) (20 * howVisible);
                g = g + (int) (20 * howVisible);
                b = b + (int) (20 * howVisible);
            } else if (colourMode == ColourMode.NORMALIZED_WHITE) {
                r = g = b = (int) (0x7f - (20 * howVisible));
            } else if (colourMode == ColourMode.NORMALIZED_INV_WHITE) {
                r = g = b = (int) (0x7f + (20 * howVisible));
            } else if (colourMode == ColourMode.WHITE) {
                r = g = b = 0xff;
            }
            return TriangleVec2.of(x0, y0, x1, y1, x2, y2, (r & 0xff) << 16 | (g & 0xff) << 8 | (b & 0xff));
        }
    }

    record ModelHighWaterMark(
            int markedTriangles3D,
            int markedTriangles2D,
            int markedVec2,
            int markedVec3,
            int markedMat4) {

        ModelHighWaterMark() {
            this(TriangleVec3.arr.size(), 0, 0,0,0);
        }

        void resetAll() {
            TriangleVec3.arr.clear();
        }
    }
}
