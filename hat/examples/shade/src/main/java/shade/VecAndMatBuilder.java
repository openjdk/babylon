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
package shade;

import jdk.incubator.code.dialect.java.JavaType;
import jdk.incubator.code.dialect.java.PrimitiveType;
import optkl.IfaceValue;
import optkl.IfaceValue.vec.Shape;
import optkl.codebuilders.JavaCodeBuilder;

import java.lang.invoke.MethodHandles;
import java.util.List;
import java.util.Map;

public class VecAndMatBuilder extends JavaCodeBuilder<VecAndMatBuilder> {

    VecAndMatBuilder() {
        super(MethodHandles.lookup(), null);
    }

    VecAndMatBuilder shape(Shape shape) {
        return dotted("Shape", "of").paren(_ -> {
                    typeName("JavaType").dot();
                    if (shape.typeElement() instanceof PrimitiveType primitiveType) {
                        if (JavaType.FLOAT.equals(primitiveType)) {
                            typeName("FLOAT");
                        } else {
                            typeName("INT");
                        }
                    } else {
                        blockInlineComment("WHAT");
                    }
                    commaSpace().intValue(shape.lanes());
                }
        );
    }

    VecAndMatBuilder type(IfaceValue.vec.Shape shape) {
        return type((JavaType) shape.typeElement());
    }

    VecAndMatBuilder vec(String vectorName, Shape shape) {
        final String lhs = "l";
        final String rhs = "r";
        oracleCopyright();
        packageDotted("shade");
        nl().nl().lineComment("Auto generated DO NOT EDIT").nl();
        importDotted("hat", "types", "F32");
        importStaticDotted("hat", "types", "F32", "*");
        importDotted("jdk", "incubator", "code", "Reflect");
        importDotted("jdk", "incubator", "code", "dialect", "java", "JavaType");
        importDotted("optkl", "IfaceValue");
        nl();
        interfaceKeyword(vectorName).space().dotted("IfaceValue", "vec").body(_ -> {
            typeName("Shape").space().identifier("shape").equals().shape(shape).semicolonNl().nl();
            shape.laneNames().forEach(laneName -> {
                type(shape).space().call(laneName).semicolonNl();
            });
            nl();
            /*
               float vec4(float x, float y, float z, float w){
                  record Impl(float x, float y, float z, float w){}
                  return new Impl(x,y,z,w);
               }
             */
            func(_ -> type(shape),
                    vectorName,
                    _ -> commaSpaceSeparated(shape.laneNames(), c -> type(shape).space().identifier(c)),
                    _ -> { // We create a record to return
                        record("Impl", _ -> blockInlineComment("implement args"));
                        returnKeyword(_ -> reserved("new").space().call("Impl"));
                    }
            );

            Map.of(
                    "add", "+",
                    "sub", "-",
                    "mul", "*",
                    "div", "/"
            ).forEach((name, symbol) -> {
                        /*
                             vec4 _add(float lx, float ly, float lz, float lw, float rx, float ry, float rz, float rw){
                                 return vec4(lx+rx, ly+ry, lz+rz, lw+rw);
                             }
                        */
                        func(
                                _ -> type(shape),
                                "_" + name,    // core ops are named _add(...)
                                _ -> commaSpaceSeparated(List.of(lhs, rhs), shape.laneNames(), (side, laneName) ->
                                        type(shape).space().identifier(side + laneName)
                                ),
                                _ -> returnCallResult(vectorName, _ ->
                                        commaSpaceSeparated(shape.laneNames(), c ->
                                                identifier(lhs + c).symbol(symbol).identifier(rhs + c)
                                        )
                                )
                        );
                        /*
                           vec4 add(vec4 l, vec4 r){
                              return add(l.x(), r.x(), l.y(), r.y(), l.z(), r.z(), l.w(), r.w());
                           }
                        */
                        func(
                                _ -> type(shape),
                                name,
                                _ -> commaSpaceSeparated(List.of(lhs, rhs), c ->
                                        identifier(vectorName).space().identifier(c)
                                ),
                                _ -> returnCallResult(name, _ ->
                                        commaSpaceSeparated(shape.laneNames(), List.of(lhs, rhs), (laneName, side) ->
                                                identifier(side).dot().call(laneName).ocparen()// l.x() or r.y()
                                        )
                                )
                        );
                         /*
                           vec4 add(float l, vec4 r){
                              return add(l, r.x(), l, r.y(), l, r.z(), l, r.w());
                           }
                        */
                        func(
                                _ -> type(shape),
                                name,
                                _ -> type(shape).space().identifier(lhs).commaSpace().identifier(vectorName).space().identifier(rhs),
                                _ -> returnCallResult(name, _ ->
                                        commaSpaceSeparated(shape.laneNames(), List.of(lhs, rhs), (laneName, side) ->
                                                identifier(side).dot().call(laneName)  // l.x() or r.y()
                                        )
                                )
                        );
                         /*
                           vec4 add(vec4 l, float r){
                              return add( l.x(), r, l.y(), r, l.z(), r, l.w(), r);
                           }
                        */

                    }
            );
            List.of("neg", "sin", "cos", "tan", "sqrt", "invsqrt").forEach(functionName ->
                    /*
                       vec4 sin(vec4 v){
                          return vec4(F32.sin(v.x()), F32.sin(v.y()), F32.sin(v.z()), F32.sin(v.w()));
                       }
                     */
                    func(_ -> type(shape), // type
                            functionName,               // name
                            _ -> identifier(vectorName).space().identifier("v"),
                            _ -> returnCallResult(vectorName, _ ->
                                    commaSpaceSeparated(shape.laneNames(), laneName ->
                                            dotted("F32", functionName).paren(_ ->
                                                    dotted("v", laneName).ocparen()
                                            )
                                    )
                            )
                    )
            );
            List.of("length").forEach(functionName ->
                     /*
                       float length(vec4 v){
                          return F32.length(v.x(), v.y(),v.z(),v.w());
                       }
                     */
                    func(_ -> type(shape), // type
                            functionName,               // name
                            _ -> identifier(vectorName).space().identifier("v"),
                            _ -> returnCallResult(vectorName, _ ->
                                    commaSpaceSeparated(shape.laneNames(), laneName ->
                                            dotted("F32", functionName).paren(_ ->
                                                    dotted("v", laneName).ocparen()
                                            )
                                    )
                            )
                    )
            );
        });
        return self();
    }

    static String createVec(String vectorName, Shape shape) {
        return new VecAndMatBuilder().vec(vectorName, shape).toString();
    }


    static void main(String[] argv) {

        System.out.println(
                createVec("ivec2", Shape.of(JavaType.INT, 2))
                        + createVec("ivec3", Shape.of(JavaType.INT, 3))
                        + createVec("vec2", Shape.of(JavaType.FLOAT, 2))
                        + createVec("vec3", Shape.of(JavaType.FLOAT, 3))
                        + createVec("vec4", Shape.of(JavaType.FLOAT, 4))
        );


    }
}
