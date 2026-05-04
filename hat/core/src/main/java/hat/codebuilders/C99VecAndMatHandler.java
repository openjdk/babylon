/*
 * Copyright (c) 2024-2026, Oracle and/or its affiliates. All rights reserved.
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
package hat.codebuilders;

import hat.buffer.Uniforms;
import hat.types.F32;
import hat.types.ivec2;
import hat.types.mat2;
import hat.types.mat3;
import hat.types.vec2;
import hat.types.vec3;
import hat.types.vec4;
import jdk.incubator.code.Op;
import jdk.incubator.code.dialect.java.ClassType;
import jdk.incubator.code.dialect.java.JavaType;
import jdk.incubator.code.dialect.java.PrimitiveType;
import optkl.IfaceValue;
import optkl.OpHelper;
import optkl.ifacemapper.MappableIface;

import java.lang.invoke.MethodHandles;
import java.lang.reflect.Field;
import java.util.List;
import java.util.function.Consumer;
import java.util.stream.Stream;

public class C99VecAndMatHandler {

    static final String SHADER_MAIN_IMAGE = "mainImage";
    static final String SHAPE_FIELD_NAME = "shape";

    static IfaceValue.vec.Shape getVecShape(java.lang.reflect.Type vecClass) {
        try {
            Field field = ((Class<?>) vecClass).getField(SHAPE_FIELD_NAME);
            return (IfaceValue.vec.Shape) field.get(null);
        } catch (Throwable t) {
            throw new RuntimeException(t);
        }
    }

    static IfaceValue.vec.Shape getVecShape(MethodHandles.Lookup lookup, JavaType javaType) {
        var resolved = OpHelper.classTypeToTypeOrThrow(lookup, (ClassType) javaType);
        return getVecShape(resolved);

    }

    static String clName(MethodHandles.Lookup lookup, JavaType javaType) {

        if (OpHelper.isAssignable(lookup, javaType, vec4.class)) {
            return "vec4";
        } else if (OpHelper.isAssignable(lookup, javaType, vec3.class)) {
            return "vec3";
        } else if (OpHelper.isAssignable(lookup, javaType, vec2.class)) {
            return "vec2";
        } else if (OpHelper.isAssignable(lookup, javaType, mat2.class)) {
            return "mat2";
        } else if (OpHelper.isAssignable(lookup, javaType, mat3.class)) {
            return "mat3";
        } else if (OpHelper.isAssignable(lookup, javaType, ivec2.class)) {
            return "ivec2";
        } else if (javaType.equals(PrimitiveType.FLOAT)) {
            return "float";
        } else if (javaType.equals(PrimitiveType.INT)) {
            return "int";
        } else {
            throw new RuntimeException("no cl name mapping for " + javaType);
        }
    }

    public static boolean isVecInvoke(OpHelper.Invoke invoke) {
        return (invoke.named(SHADER_MAIN_IMAGE) || invoke.refIs(F32.class, Uniforms.class, IfaceValue.vec.class, IfaceValue.mat.class));
    }

    public static String mangledName(OpHelper.Invoke invoke) {
        // So invoke  float mod(float lhs, float rhs) -> f32_mod_f32_f32"
        return clName(invoke.lookup(), (JavaType) invoke.returnType()) + "_"
                + invoke.name() + "_"
                + clName(invoke.lookup(), (JavaType) invoke.resultFromOperandNOrNull(0).type()) + "_"
                + clName(invoke.lookup(), (JavaType) invoke.resultFromOperandNOrNull(1).type());
    }

    public static <T extends C99HATKernelBuilder<T>> void args(C99HATKernelBuilder<T> bldr, Stream<Op.Result> argStream) {
        bldr.paren(_ -> bldr.commaSpaceSeparated(argStream, operand -> bldr.recurse(operand.op())));
    }

    public static <T extends C99HATKernelBuilder<T>> void nameAndArgs(String name, C99HATKernelBuilder<T> bldr, Stream<Op.Result> argStream) {
        bldr.funcName(name).paren(_ -> args(bldr, argStream));
    }

    public static <T extends C99HATKernelBuilder<T>> void nameAndArgs(OpHelper.Invoke invoke, String newName, C99HATKernelBuilder<T> bldr) {
        bldr.funcName(newName).paren(_ ->
                bldr.commaSpaceSeparated(invoke.operandsAsResults(), operand -> bldr.recurse(operand.op()))
        );
    }

    public static <T extends C99HATKernelBuilder<T>> void nameAndArgs(OpHelper.Invoke invoke, C99HATKernelBuilder<T> bldr) {
        nameAndArgs(invoke, invoke.name(), bldr);
    }

    public static <T extends C99HATKernelBuilder<T>> void mangledNameAndArgs(OpHelper.Invoke invoke, C99HATKernelBuilder<T> bldr) {
        nameAndArgs(invoke, mangledName(invoke), bldr);
    }

    public static boolean hasVecOperand(OpHelper.Invoke invoke) {
        return invoke.operandsAsResults().anyMatch(r ->
                r.type() instanceof ClassType classType
                        && OpHelper.classTypeToTypeOrThrow(invoke.lookup(), classType) instanceof Class<?> clazz
                        && IfaceValue.mat.class.isAssignableFrom(clazz));

    }

    public static <T extends C99HATKernelBuilder<T>> void handleF32Invoke(C99HATKernelBuilder<T> bldr, OpHelper.Invoke invoke) {
        switch (invoke.name()) {
            case "fract" -> //opencl 's fract wants a ptr as the second arg where it returns fract value.  So we implement our own
                    bldr.paren(_ -> bldr.recurse(invoke.opFromFirstOperandOrNull()).sub().funcName("floor")
                            .paren(_ -> bldr.recurse(invoke.opFromFirstOperandOrNull())));
            case "atan" -> {
                if (invoke.operandCount() > 1) { // atan -> atan2
                    nameAndArgs(invoke, invoke.name() + "2", bldr);  // atan(l,r) ->atan2(l,r)
                } else {
                    nameAndArgs(invoke, bldr); // atan(v)->atan(v)
                }
            }
            case "inversesqrt" -> nameAndArgs(invoke, "rsqrt", bldr);// inversesqrt(...)->rsqrt(...)
            case "cos", "sqrt", "sin", "exp", "pow", "min", "max", "log", "smoothstep", "clamp", "floor", "step",
                 "mix" -> nameAndArgs(invoke, bldr); // asis!
            case "abs" -> nameAndArgs(invoke, "fabs", bldr);// abs(...)->fabs(...)
            case "mod" -> mangledNameAndArgs(invoke, bldr); //mod(...) -> float_mod_float_float(...)
            default -> throw new RuntimeException("unmapped F32 call " + invoke.name());
        }
    }

    public static <T extends C99HATKernelBuilder<T>> void handleUniformsInvoke(C99HATKernelBuilder<T> bldr, OpHelper.Invoke invoke) {
        bldr.cast(_ -> bldr.type((JavaType) invoke.returnType()));
        switch (invoke.name()) {
            case "iResolution" -> bldr.paren(_ ->
                    bldr.sep(List.of("x", "y", "z"), _ -> bldr.csp(), lane -> bldr.id("uniforms").rarrow().id(invoke.name()).dot().id(lane))
            );
            case "iMouse" ->
                    bldr.paren(_ -> bldr.sep(List.of("x", "y"), _ -> bldr.csp(), lane -> bldr.id("uniforms").rarrow().id(invoke.name()).dot().id(lane))
                    );
            case "iTime" -> bldr.id("uniforms").rarrow().id(invoke.name());
            default -> throw new RuntimeException("some other uniform" + invoke.name());
        }
    }

    public static <T extends C99HATKernelBuilder<T>> void handleVecInvoke(C99HATKernelBuilder<T> bldr, OpHelper.Invoke invoke) {
        if (invoke.refIs(IfaceValue.Struct.class)) {
            // A call on a field of the uniforms
            // so uniforms_t.iTime -> uniforms.iTime;
            bldr.recurse(invoke.opFromFirstOperandOrNull()).dot().id(invoke.name());
        } else if (invoke instanceof OpHelper.Invoke.Virtual && invoke.operandCount() == 1 && invoke.refIs(vec3.class, vec2.class, vec4.class)) {
            // an accessor on a vec say v.x() -> v.x
            bldr.recurse(invoke.opFromFirstOperandOrNull()).dot().id(invoke.name());
        } else if (invoke.nameMatchesRegex("vec[234]")) { //  a psuedo vec2 constructor vec2(....) -> (vec2)(....)
            bldr.paren(_ -> bldr.type(invoke.name()));
            args(bldr, invoke.operandsAsResults());
        } else if (invoke.nameMatchesRegex("(mod|reflect)")) {
            mangledNameAndArgs(invoke, bldr);
        } else if (invoke.named("mul") && invoke.operandCount() == 2 && hasVecOperand(invoke)) {
            mangledNameAndArgs(invoke, bldr);
        } else if (invoke.nameMatchesRegex("(mul|add|sub|div)")) {
            // for opencl we can turn these into expressions. So vec3.mul(l,r) -> (l * r)
            bldr.paren(_ -> bldr.recurse(invoke.opFromFirstOperandOrNull()).symbol(switch (invoke.name()) {
                case "mul" -> "*";
                case "add" -> "+";
                case "div" -> "/";
                case "sub" -> "-";
                default -> throw new IllegalStateException("oh my");
            }).recurse(invoke.opFromOperandNOrNull(1)));
        } else if (invoke.named("abs") && invoke.refIs(IfaceValue.vec.class)) {
            bldr.funcName("f" + invoke.name()).paren(_ ->
                    bldr.sep(invoke.op().operands(), _ -> bldr.csp(), v -> bldr.recurse(v.asResult().op()))
            );
        } else if (invoke.named("fract") && invoke.operandCount() == 1) {
            // return x - floor(x);
            bldr.paren(_ -> bldr.recurse(invoke.opFromFirstOperandOrNull()).sub().funcName("floor")
                    .paren(_ -> bldr.recurse(invoke.opFromFirstOperandOrNull())));
        } else if (invoke.nameMatchesRegex("(dot|length|max|mix|min|smoothstep|step|normalize|clamp|pow|cross|distance|floor|fract|round|sin|cos|abs)")) {
            bldr.funcName(invoke.op()).paren(_ ->
                    bldr.sep(invoke.op().operands(), _ -> bldr.csp(), v -> bldr.recurse(v.asResult().op()))
            );
        } else {
            StringBuilder stringBuilder = new StringBuilder("For vec types we need to IMPLEMENT " + invoke.refType() + ":" + invoke.name() + "(");
            invoke.op().operands().forEach(o -> stringBuilder.append(" " + o.asResult().type()));
            stringBuilder.append(")");
            throw new RuntimeException(stringBuilder.toString());
        }
    }

    public static <T extends C99HATKernelBuilder<T>> void handleMatInvoke(C99HATKernelBuilder<T> bldr, OpHelper.Invoke invoke) {
        if (invoke.nameMatchesRegex("mat[234]")) {
            bldr.paren(_ -> bldr.type(invoke.name())).brace(_ ->
                    bldr.commaSpaceSeparated(invoke.operandsAsResults(), operand -> bldr.recurse(operand.op()))
            );
        } else if (invoke.named("mul")) {
            mangledNameAndArgs(invoke, bldr);
        } else {
            StringBuilder stringBuilder = new StringBuilder("For mat types we need to IMPLEMENT " + invoke.refType() + ":" + invoke.name() + "(");
            invoke.op().operands().forEach(o -> stringBuilder.append(" " + o.asResult().type()));
            stringBuilder.append(")");
            throw new RuntimeException(stringBuilder.toString());
        }
    }

    public static <T extends C99HATKernelBuilder<T>> void handleInvoke(C99HATKernelBuilder<T> bldr, OpHelper.Invoke invoke) {
        if (invoke.named(SHADER_MAIN_IMAGE)) {
            nameAndArgs(invoke, bldr);
        } else if (invoke.refIs(F32.class)) {
            handleF32Invoke(bldr, invoke);
        } else if (invoke.refIs(Uniforms.class)) {
            handleUniformsInvoke(bldr, invoke);
        } else if (invoke.refIs(IfaceValue.vec.class)) {
            handleVecInvoke(bldr, invoke);
        } else if (invoke.refIs(IfaceValue.mat.class)) {
            handleMatInvoke(bldr, invoke);
        }
    }

    public static boolean isVecOrMatType(MethodHandles.Lookup lookup, JavaType javaType) {
        return OpHelper.isAssignable(lookup, javaType, MappableIface.vec.class, IfaceValue.mat.class);
    }

    public static <T extends C99HATKernelBuilder<T>> void handleType(C99HATKernelBuilder<T> bldr, JavaType javaType) {
        bldr.type(clName(bldr.scopedCodeBuilderContext().lookup(), javaType));
    }

    public static <T extends C99HATKernelBuilder<T>> void genFunc(C99HATKernelBuilder<T> bldr, String ret, String op, String lhs, String rhs, Consumer<C99HATKernelBuilder<T>> consumer) {
        bldr.func(
                _ -> bldr.type(ret),
                ret + "_" + op + "_" + lhs + "_" + rhs,
                _ -> bldr.type(lhs).sp().id("l").csp().type(rhs).sp().id("r"),
                _ -> consumer.accept(bldr)
        ).semicolon();

    }

    public static <T extends C99HATKernelBuilder<T>> void createVecFunctions(C99HATKernelBuilder<T> builder) {
        record NamedMatShape(String name, IfaceValue.mat.Shape shape) {
        }
        List.of(new NamedMatShape("mat2", mat2.shape), new NamedMatShape("mat3", mat3.shape)).forEach(ns ->
                builder.typedefKeyword().sp().structKeyword().sp().id(ns.name + "_s").braceNlIndented(_ -> {
                    builder.sep(ns.shape.rowColNames(), _ -> builder.nl(), n ->
                            builder.type((JavaType) ns.shape.codeType()).sp().id(n).semicolon()
                    );
                }).sp().id(ns.name).snl()
        );


        record NamedVecShape(String name, IfaceValue.vec.Shape shape) {
        }

        List.of(new NamedVecShape("vec2", vec2.shape), new NamedVecShape("vec3", vec3.shape), new NamedVecShape("vec4", vec4.shape)).forEach(ns ->
                builder.typedefKeyword().sp().type("float" + ns.shape.lanes()).sp().type(ns.name).snl()
        );
        List.of(new NamedVecShape("ivec2", vec2.shape)).forEach(ns ->
                builder.typedefKeyword().sp().type("int" + ns.shape.lanes()).sp().type(ns.name).snl()
        );
/*
2. Vector * Matrix (vec2 * mat2)This treats the vector as a row. Mathematically, this is equivalent to multiplying the transpose of the matrix by the vector.$$\text{result}.x = (v.x \cdot m_{0}) + (v.y \cdot m_{1})$$$$\text{result}.y = (v.x \cdot m_{2}) + (v.y \cdot m_{3})$$Javapublic static float[] multiplyVec2Mat2(float[] v, float[] m) {
    float x = v[0] * m[0] + v[1] * m[1];
    float y = v[0] * m[2] + v[1] * m[3];
     l.x * r.00 + l.y * r.01,l.x * r.10 + l.y * r.11;
    return new float[]{x, y};
}
        */
        genFunc(builder, "vec2", "mul", "vec2", "mat2", _ ->
                builder.returnKeyword().sp().paren(_ -> builder.type("vec2"))
                        .paren(_ -> builder.preformatted("l.x*r._00+l.y*r._01,l.x*r._10+l.y*r._11"))
                        .semicolon()
        );

        /*
        public static float[] multiplyMat2Vec2(float[] m, float[] v) {
    float x = m[0] * v[0] + m[2] * v[1];
    float y = m[1] * v[0] + m[3] * v[1];
     l.00 * r.x + l.10 * r.y,
     l.01 * r.x + l.11 * r.y
    return new float[]{x, y};
}


         */
        genFunc(builder, "vec2", "mul", "mat2", "vec2", _ ->
                builder.returnKeyword().sp().paren(_ -> builder.type("vec2")).paren(_ ->
                        builder.preformatted(" l._00*r.x+l._10*r.y,l._01*r.x+l._11*r.y")
                ).semicolon());

        /*
          public static float[] multiplyVec3Mat3(float[] v, float[] m) {
            float x = v[0] * m[0] + v[1] * m[1] + v[2] * m[2];
            float y = v[0] * m[3] + v[1] * m[4] + v[2] * m[5];
            float z = v[0] * m[6] + v[1] * m[7] + v[2] * m[8];
           l.x * r._00 + l.y * r._01 + l.z * r._02,
           l.x * r._10 + l.y * r._11 + l.z * r._12,
           l.x * r._20 + l.y * r._21 + l.z * r._22,
            return new float[]{x, y, z};
        }
         */
        genFunc(builder, "vec3", "mul", "vec3", "mat3", _ ->
                builder.returnKeyword().sp().paren(_ -> builder.type("vec3")).paren(_ ->
                builder.preformatted("l.x*r._00+l.y*r._01+l.z*r._02,l.x*r._10+l.y*r._11+l.z*r._12,l.x*r._20+l.y*r._21+l.z*r._22")).semicolon()
        );
        genFunc(builder, "vec3", "mul", "mat3", "vec3", _ ->
                builder.returnKeyword().sp().paren(_ -> builder.type("vec3")).paren(_ ->
                        builder.preformatted(
                                "       l._00 * r.x + l._01 * r.y + l._02 * r.z," +
                                "        l._10 * r.x + l._11 * r.y + l._12 * r.z," +
                                "        l._20 * r.x + l._21 * r.y + l._22 * r.z")).semicolon()
        );

        /*
          l._00() * r.x() + l._01() * r.y() + l._02() * r.z(),
        l._10() * r.x() + l._11() * r.y() + l._12() * r.z(),
        l._20() * r.x() + l._21() * r.y() + l._22() * r.z()
         */


        /*
        public static float[] multiplyMat3Vec3(float[] m, float[] v) {
          //  float x = m[0] * v[0] + m[3] * v[1] + m[6] * v[2];
          //  float y = m[1] * v[0] + m[4] * v[1] + m[7] * v[2];
          //  float z = m[2] * v[0] + m[5] * v[1] + m[8] * v[2];
            float x = m[0] * v[0] + m[3] * v[1] + m[6] * v[2];
            float y = m[1] * v[0] + m[4] * v[1] + m[7] * v[2];
            float z = m[2] * v[0] + m[5] * v[1] + m[8] * v[2];
            return new float[]{x, y, z};
        }
        2. Vector * Matrix (vec3 * mat3)
        This treats the vector as a row vector on the left. Effectively, you are calculating the dot product of the vector with each column of the matrix.

        Java
        public static float[] multiplyVec3Mat3(float[] v, float[] m) {
            float x = v[0] * m[0] + v[1] * m[1] + v[2] * m[2];
            float y = v[0] * m[3] + v[1] * m[4] + v[2] * m[5];
            float z = v[0] * m[6] + v[1] * m[7] + v[2] * m[8];
            return new float[]{x, y, z};
        }
*/
        genFunc(builder, "mat3", "mul", "mat3", "mat3", _ ->
                builder.returnKeyword().sp().paren(_ -> builder.type("mat3")).brace(_ ->
                        builder.preformatted("""
                                    l._00*r._00,l._01*r._01,l._02*r._02,
                                    l._10*r._10,l._11*r._11,l._12*r._12,
                                    l._20*r._20,l._21*r._21,l._22*r._22
                                """)
                ).semicolon()
        );
        genFunc(builder, "mat2", "mul", "mat2", "mat2", _ ->
                builder.returnKeyword().sp().paren(_ -> builder.type("mat2")).brace(_ ->
                        builder.preformatted("l._00*r._00,l._01*r._01, l._10*r._10,l._11*r._11")
                ).semicolon()
        );

        genFunc(builder, "float", "mod", "float", "float", _ ->
                builder.returnKeyword().sp().id("l").sp().minus().id("r").sp().mul().sp().id("floor").paren(_ ->
                        builder.id("l").div().id("r")).semicolon());

        genFunc(builder, "vec2", "mod", "vec2", "float", _ ->
                builder.returnKeyword().sp().id("l").sp().minus().id("r").sp().mul().sp().id("floor").paren(_ ->
                        builder.id("l").div().id("r")).semicolon());

        genFunc(builder, "vec3", "reflect", "vec3", "vec3", _ ->
                builder.returnKeyword().sp().id("l").sp().minus().id("r").sp().mul().sp().id("l").sp().mul().floatConst(2).semicolon());
    }
}
