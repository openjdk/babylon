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
import jdk.incubator.code.TypeElement;
import jdk.incubator.code.dialect.java.ClassType;
import jdk.incubator.code.dialect.java.JavaType;
import optkl.IfaceValue;
import optkl.OpHelper;
import optkl.ifacemapper.MappableIface;

import java.lang.invoke.MethodHandles;
import java.lang.reflect.Field;
import java.util.List;

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
        }else if (OpHelper.isAssignable(lookup, javaType, mat2.class)) {
            return "mat2";
        } else if (OpHelper.isAssignable(lookup, javaType, mat3.class)) {
            return "mat3";
        } else if (OpHelper.isAssignable(lookup, javaType, ivec2.class)) {
            return "ivec2";
        } else {
            throw new RuntimeException("no cl name mapping for " + javaType);
        }
    }

    public static boolean isVecInvoke(OpHelper.Invoke invoke) {
        return (invoke.named(SHADER_MAIN_IMAGE) || invoke.refIs(F32.class, Uniforms.class, IfaceValue.vec.class, IfaceValue.mat.class));
    }

    public static <T extends C99HATKernelBuilder<T>> void handleInvoke(C99HATKernelBuilder<T> bldr, OpHelper.Invoke invoke) {
        if (invoke.named(SHADER_MAIN_IMAGE)) {
            bldr.funcName(invoke.op()).paren(_ ->
                    bldr.commaSpaceSeparated(invoke.operandsAsResults(), operand -> bldr.recurse(operand.op()))
            );
        } else if (invoke.refIs(F32.class)) {
           // System.out.println("IMPLEMENT F32." + invoke.name());
            switch (invoke.name()) {
                case "fract" -> bldr.paren(_ -> bldr.recurse(invoke.opFromFirstOperandOrNull()).sub().funcName("floor")
                        .paren(_ -> bldr.recurse(invoke.opFromFirstOperandOrNull())));
                case "atan" -> {
                    if (invoke.operandCount() > 1) {
                        bldr.id(invoke.name() + "2").paren(_ ->
                                bldr.commaSpaceSeparated(invoke.operandsAsResults(), operand -> bldr.recurse(operand.op())));
                    } else {
                        bldr.id(invoke.name()).paren(_ ->
                                bldr.commaSpaceSeparated(invoke.operandsAsResults(), operand -> bldr.recurse(operand.op())));
                    }
                }
                case "inversesqrt" ->
                    bldr.id("rsqrt").paren(_ ->
                            bldr.commaSpaceSeparated(invoke.operandsAsResults(), operand -> bldr.recurse(operand.op())));
                case "cos", "sqrt","sin", "exp", "pow", "min", "max", "log", "smoothstep", "clamp","floor","step","mix" -> bldr.id(invoke.name()).paren(_ ->
                        bldr.commaSpaceSeparated(invoke.operandsAsResults(), operand -> bldr.recurse(operand.op())));
                case "abs" -> bldr.id("f" + invoke.name()).paren(_ ->
                        bldr.commaSpaceSeparated(invoke.operandsAsResults(), operand -> bldr.recurse(operand.op())));
                case "mod"->bldr.id("f32_mod_f32_f32").paren(_->
                    bldr.commaSpaceSeparated(invoke.operandsAsResults(), operand -> bldr.recurse(operand.op())));

                default -> throw new RuntimeException("unmapped F32 call " + invoke.name());
            }
        } else if (invoke.refIs(Uniforms.class)) {
            bldr.cast(_ -> bldr.type((JavaType) invoke.returnType()));
            switch (invoke.name()) {
                case "iResolution" -> bldr.paren(_ ->
                        bldr.sep(List.of("x", "y", "z"), _ -> bldr.csp(), lane ->
                                bldr.id("uniforms").rarrow().id(invoke.name()).dot().id(lane)
                        )
                );
                case "iMouse" -> bldr.paren(_ ->
                        bldr.sep(List.of("x", "y"), _ -> bldr.csp(), lane ->
                                bldr.id("uniforms").rarrow().id(invoke.name()).dot().id(lane)
                        )
                );
                case "iTime" -> bldr.id("uniforms").rarrow().id(invoke.name());
                default -> throw new RuntimeException("some other uniform" + invoke.name());
            }
        } else if (invoke.refIs(IfaceValue.vec.class)) {
            if (invoke.refIs(IfaceValue.Struct.class)) {
                // This is likely a call on a field of the uniform
                bldr.recurse(invoke.opFromFirstOperandOrNull()).dot().id(invoke.name());
            } else if (invoke instanceof OpHelper.Invoke.Virtual && invoke.operandCount() == 1
                    && invoke.refIs(vec3.class, vec2.class, vec4.class)
            ) {
                // This is likely an accessor on a vec say v.x()

                bldr.recurse(invoke.opFromFirstOperandOrNull()).dot().id(invoke.name());
            } else if (invoke.nameMatchesRegex("vec[234]")) {
                bldr.paren(_ -> bldr.type(invoke.name())).paren(_ ->
                        bldr.commaSpaceSeparated(invoke.operandsAsResults(), operand -> bldr.recurse(operand.op()))
                );
            } else {
                if (invoke.named("mod")) {
                    bldr.id("vec2_mod_vec2_f32").paren(_ ->
                            bldr.commaSpaceSeparated(invoke.operandsAsResults(), operand -> bldr.recurse(operand.op())));
                }else
                // We have to catch vecn.mul(x, x)
                if (invoke.named("mul")
                    && invoke.operandCount() == 2
                            && invoke.resultFromOperandNOrNull(1) instanceof Op.Result r
                            && r.type() instanceof ClassType cte
                         && OpHelper.classTypeToTypeOrThrow(invoke.lookup(), cte) instanceof Class<?> c
                    && mat2.class.isAssignableFrom(c)) {
                    bldr.id("vec2_mul_vec2_mat2").paren(_ -> bldr.recurse(invoke.opFromFirstOperandOrNull()).csp().recurse(invoke.opFromOperandNOrNull(1)));
                }else    if (invoke.named("mul")
                            && invoke.operandCount() == 2
                            && invoke.resultFromOperandNOrNull(1) instanceof Op.Result r
                            && r.type() instanceof ClassType cte
                            && OpHelper.classTypeToTypeOrThrow(invoke.lookup(), cte) instanceof Class<?> c
                            && mat3.class.isAssignableFrom(c)){
                        bldr.id("vec3_mul_vec3_mat3").paren(_->bldr.recurse(invoke.opFromFirstOperandOrNull()).csp().recurse(invoke.opFromOperandNOrNull(1)));
                }else if (invoke.nameMatchesRegex("(mul|add|sub|div)")) {
                    // for opencl we can turn these into expressions. So vec3.mul(l,r) -> (l * r)
                    bldr.paren(_ -> bldr.recurse(invoke.opFromFirstOperandOrNull()).symbol(switch (invoke.name()) {
                        case "mul" -> "*";
                        case "add" -> "+";
                        case "div" -> "/";
                        case "sub" -> "-";
                        default -> throw new IllegalStateException("oh my");
                    }).recurse(invoke.opFromOperandNOrNull(1)));
                } else {
                    System.out.print("IMPLEMENT " + invoke.refType() + ":" + invoke.name() + "(");
                    invoke.op().operands().forEach(o -> System.out.print(" " + o.result().type()));
                    System.out.println(")");
                    invoke(bldr, invoke);
                    // throw new RuntimeException("HOW");
                }
            }
        } else if (invoke.refIs(IfaceValue.mat.class)) {
            if (invoke.nameMatchesRegex("mat[234]")) {
                bldr.paren(_ -> bldr.type(invoke.name())).brace(_ ->
                        bldr.commaSpaceSeparated(invoke.operandsAsResults(), operand -> bldr.recurse(operand.op()))
                );
            }else   if (invoke.named("mul")) {
            // for opencl we can turn these into expressions. So vec3.mul(l,r) -> (l * r)
                bldr.id("vec2_mul_mat2_vec2").paren(_->    bldr.commaSpaceSeparated(invoke.operandsAsResults(), operand -> bldr.recurse(operand.op())));
            } else {
                bldr.lineComment("other call through mat !");
                bldr.recurse(invoke.opFromFirstOperandOrNull()).dot().id(invoke.name());
            }
        }

    }


    public static <T extends C99HATKernelBuilder<T>> void invoke(C99HATKernelBuilder<T> bldr, OpHelper.Invoke invoke) {

        // most opencl functions can me called directly...
        if (invoke.named("abs") && invoke.refIs(IfaceValue.vec.class)) {
            bldr.funcName("f" + invoke.name()).paren(_ ->
                    bldr.sep(invoke.op().operands(), _ -> bldr.csp(), v -> bldr.recurse(v.result().op()))
            );
        } else if (invoke.named("fract") && invoke.operandCount() == 1) {
            // return x - floor(x);
            bldr.paren(_ -> bldr.recurse(invoke.opFromFirstOperandOrNull()).sub().funcName("floor")
                    .paren(_ -> bldr.recurse(invoke.opFromFirstOperandOrNull())));
        } else {
            bldr.funcName(invoke.op()).paren(_ ->
                    bldr.sep(invoke.op().operands(), _ -> bldr.csp(), v -> bldr.recurse(v.result().op()))
            );
        }
    }

    public static boolean isVecOrMatType(MethodHandles.Lookup lookup, JavaType javaType) {
        return OpHelper.isAssignable(lookup, javaType, MappableIface.vec.class, IfaceValue.mat.class);
    }

    public static <T extends C99HATKernelBuilder<T>> void handleType(C99HATKernelBuilder<T> bldr, JavaType javaType) {
            bldr.type(clName(bldr.scopedCodeBuilderContext().lookup(), javaType));
    }

    public static <T extends C99HATKernelBuilder<T>> void createVecFunctions(C99HATKernelBuilder<T> builder) {
        record NamedMatShape(String name, IfaceValue.mat.Shape shape) {
        }
        List.of(new NamedMatShape("mat2", mat2.shape), new NamedMatShape("mat3", mat3.shape)).forEach(ns ->
            builder.typedefKeyword().sp().structKeyword().sp().id(ns.name + "_s").braceNlIndented(_ -> {
                builder.sep(ns.shape.rowColNames(), _ -> builder.nl(), n ->
                        builder.type((JavaType) ns.shape.typeElement()).sp().id(n).semicolon()
                );
            }).sp().id(ns.name).snl()
        );
        record NamedVecShape(String name, IfaceValue.vec.Shape shape) {
        }

        List.of(new NamedVecShape("vec2", vec2.shape), new NamedVecShape("vec3", vec3.shape),new NamedVecShape("vec4", vec4.shape)).forEach(ns->
             builder.typedefKeyword().sp().type("float"+ns.shape.lanes()).sp().type(ns.name).snl()
        );
        List.of(new NamedVecShape("ivec2", vec2.shape)).forEach(ns->
                builder.typedefKeyword().sp().type("int"+ns.shape.lanes()).sp().type(ns.name).snl()
        );
        builder.func(
                _->builder.type("vec2"),
                "vec2_mul_vec2_mat2",
                _->builder.type("vec2").sp().id("l").csp().type("mat2").sp().id("r"),
                _->builder.returnKeyword().sp().paren(_->builder.type("vec2")).paren(_->
                                builder.preformatted("""
                                        l.x*r._00+l.x*r._01,
                                        l.y*r._10+l.y*r._11
                                        """)
                ).semicolon());
        builder.func(
                _->builder.type("vec3"),
                "vec3_mul_vec3_mat3",
                _->builder.type("vec3").sp().id("l").csp().type("mat3").sp().id("r"),
                _->builder.returnKeyword().sp().paren(_->builder.type("vec3")).paren(_->
                        builder.preformatted("""
                                l.x*r._00+l.x*r._01+l.x*r._02,
                                l.y*r._10+l.y*r._11+l.y*r._12,
                                l.z*r._20+l.z*r._21+l.z*r._22
                                """)
                ).semicolon());

        builder.func(
                _->builder.type("mat2"),
                "mat2_mul_mat2_mat2",
                _->builder.type("mat2").sp().id("l").csp().type("mat2").sp().id("r"),
                _->builder.returnKeyword().sp().paren(_->builder.type("mat2")).brace(_->
                        builder.preformatted("""
                                l._00*r._00,l._01*r._01, l._10*r._10,l._11*r._11
                                """)
                ).semicolon());
        builder.func(
                _->builder.type("vec2"),
                "vec2_mul_mat2_vec2",
                _->builder.type("mat2").sp().id("l").csp().type("vec2").sp().id("r"),
                _->builder.returnKeyword().sp().paren(_->builder.type("vec2")).paren(_->
                        builder.preformatted("""
                                  l._00*r.x+l._01*r.y,
                                  l._10*r.x+l._11*r.y
                                """)
                ).semicolon());
        builder.func(
                _->builder.type("float"),
                "f32_mod_f32_f32",
                _->builder.type("float").sp().id("l").csp().type("float").sp().id("r"),
                        _->builder.returnKeyword().sp().id("l").sp().minus().id("r").sp().mul().sp().id("floor").paren(_->
                                builder.id("l").div().id("r")).semicolon());

        builder.func(
                _->builder.type("vec2"),
                "vec2_mod_vec2_f32",
                _->builder.type("vec2").sp().id("l").csp().type("float").sp().id("r"),
                _->builder.returnKeyword().sp().id("l").sp().minus().id("r").sp().mul().sp().id("floor").paren(_->
                        builder.id("l").div().id("r")).semicolon());

                     //   "                // static float mod(float x, float y){return x - y * floor(x/y);}"
    }
}
