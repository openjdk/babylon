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
import hat.types.vec2;
import hat.types.vec3;
import hat.types.vec4;
import jdk.incubator.code.dialect.java.ClassType;
import jdk.incubator.code.dialect.java.JavaType;
import optkl.IfaceValue;
import optkl.OpHelper;
import optkl.ifacemapper.MappableIface;

import java.lang.invoke.MethodHandles;
import java.lang.reflect.Field;
import java.util.List;

public class C99VecHandler {

    static final String SHADER_MAIN_IMAGE = "mainImage";
    static final String SHAPE_FIELD_NAME = "shape";
    static IfaceValue.vec.Shape getVecShape(java.lang.reflect.Type vecClass){
        try {
            Field field = ((Class<?>)vecClass).getField(SHAPE_FIELD_NAME);
            return (IfaceValue.vec.Shape) field.get(null);
        }catch (Throwable t){
            throw new RuntimeException(t);
        }
    }
    static IfaceValue.vec.Shape getVecShape(MethodHandles.Lookup lookup,JavaType javaType){
        var resolved = OpHelper.classTypeToTypeOrThrow(lookup,(ClassType) javaType);
        return  getVecShape(resolved);

    }

    static String clName(MethodHandles.Lookup lookup, JavaType javaType){
        if (OpHelper.isAssignable(lookup, javaType, vec4.class)) {
            return "float4";
        } else if (OpHelper.isAssignable( lookup, javaType, vec3.class)) {
            return "float3";
        } else if (OpHelper.isAssignable(lookup, javaType, vec2.class)) {
            return "float2";
        } else {
            throw new RuntimeException("no cl name mapping for "+javaType);
        }
    }

    static public String mapVecName(String vname){
        return "float"+vname.substring(3);
    }

    public static  boolean isVecInvoke(OpHelper.Invoke invoke) {
        return (invoke.named(SHADER_MAIN_IMAGE) || invoke.refIs(F32.class,Uniforms.class,IfaceValue.vec.class,IfaceValue.mat.class));
    }

    public static <T extends C99HATKernelBuilder<T>> void handleInvoke(C99HATKernelBuilder<T> bldr, OpHelper.Invoke invoke) {
        if (invoke.named(SHADER_MAIN_IMAGE)) {
            bldr.funcName(invoke.op()).paren(_ ->
                    bldr.commaSpaceSeparated(invoke.operandsAsResults(), operand -> bldr.recurse(operand.op()))
            );
        } else if (invoke.refIs(F32.class)) {
            switch (invoke.name()) {
                case "cos", "sin" -> bldr.identifier(invoke.name()).paren(_ ->
                        bldr.commaSpaceSeparated(invoke.operandsAsResults(), operand -> bldr.recurse(operand.op())));
                case "abs" -> bldr.identifier("f" + invoke.name()).paren(_ ->
                        bldr.commaSpaceSeparated(invoke.operandsAsResults(), operand -> bldr.recurse(operand.op())));
                default -> throw new RuntimeException("unmapped F32 call " + invoke.name());
            }
        } else if (invoke.refIs(Uniforms.class)) {
            bldr.cast(_ -> bldr.type((JavaType) invoke.returnType()));
            switch(invoke.name()){
                case "iResolution" -> bldr.paren(_ ->
                        bldr.join(List.of("x", "y", "z"), _ -> bldr.commaSpace(), lane ->
                                bldr.identifier("uniforms").rarrow().identifier(invoke.name()).dot().identifier(lane)
                        )
                );
                case "iTime" -> bldr.identifier("uniforms").rarrow().identifier(invoke.name());
                default ->  throw new RuntimeException("some other uniform" + invoke.name());
            }
        } else if (invoke.refIs(IfaceValue.vec.class)) {
            if (invoke.nameMatchesRegex("vec[234]")) {
                bldr.paren(_ -> bldr.typeName(C99VecHandler.mapVecName(invoke.name()))).paren(_ ->
                        bldr.commaSpaceSeparated(invoke.operandsAsResults(), operand -> bldr.recurse(operand.op()))
                );
            } else {
                bldr.recurse(invoke.opFromFirstOperandOrNull()).dot().identifier(invoke.name());
            }
        } else if (invoke.refIs(IfaceValue.mat.class)) {
            bldr.lineComment("call through mat !");
            bldr.recurse(invoke.opFromFirstOperandOrNull()).dot().identifier(invoke.name());
        }
    }

    public static boolean isVecType(MethodHandles.Lookup lookup,JavaType javaType) {
        return OpHelper.isAssignable(lookup, javaType, MappableIface.vec.class);
    }

    public static <T extends C99HATKernelBuilder<T>> void handleType(C99HATKernelBuilder<T> bldr, JavaType javaType) {
        bldr.typeName(clName(bldr.scopedCodeBuilderContext().lookup(), javaType));
    }
}
