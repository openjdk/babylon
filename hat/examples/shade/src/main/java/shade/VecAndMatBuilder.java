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

import hat.types.F32;
import hat.types.vec2;
import hat.types.vec3;
import hat.types.vec4;
import jdk.incubator.code.TypeElement;
import jdk.incubator.code.dialect.java.JavaType;
import optkl.IfaceValue;
import optkl.IfaceValue.vec;
import optkl.codebuilders.JavaCodeBuilder;

import java.io.IOException;
import java.lang.invoke.MethodHandles;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.Consumer;
import java.util.stream.IntStream;

public class VecAndMatBuilder extends JavaCodeBuilder<VecAndMatBuilder> {
    /*
                                         We should be able to use vec16 for mat4


                                         float16 mat4_mul(float16 A, float16 B) {
                                            float16 C;

                                            // We compute C row by row
                                            // Each row of C is the sum of the rows of B scaled by the components of A

                                            // Row 0
                                            C.s0123 = A.s0 * B.s0123 + A.s1 * B.s4567 + A.s2 * B.s89ab + A.s3 * B.scdef;
                                            // Row 1
                                            C.s4567 = A.s4 * B.s0123 + A.s5 * B.s4567 + A.s6 * B.s89ab + A.s7 * B.scdef;
                                            // Row 2
                                            C.s89ab = A.s8 * B.s0123 + A.s9 * B.s4567 + A.sa * B.s89ab + A.sb * B.scdef;
                                            // Row 3
                                            C.scdef = A.sc * B.s0123 + A.sd * B.s4567 + A.se * B.s89ab + A.sf * B.scdef;

                                            return C;
                                        }


                                        #define TS 16 // Tile Size

                                        __kernel void mat4_mul_tiled(__global const float16* A,
                                                                     __global const float16* B,
                                                                     __global float16* C,
                                                                     const int Width) { // Width in terms of float16 units

                                            // Local memory for tiles of float16 matrices
                                            __local float16 tileA[TS][TS];
                                            __local float16 tileB[TS][TS];

                                            int row = get_local_id(1);
                                            int col = get_local_id(0);
                                            int globalRow = get_global_id(1);
                                            int globalCol = get_global_id(0);

                                            float16 accumulated = (float16)(0.0f);

                                            // Loop over tiles
                                            for (int t = 0; t < (Width / TS); t++) {

                                                // Cooperative Load: Each thread loads one float16 into local memory
                                                tileA[row][col] = A[globalRow * Width + (t * TS + col)];
                                                tileB[row][col] = B[(t * TS + row) * Width + globalCol];

                                                // Synchronize to ensure the tile is fully loaded
                                                barrier(CLK_LOCAL_MEM_FENCE);

                                                // Compute partial product for this tile
                                                for (int k = 0; k < TS; k++) {
                                                    accumulated = mat4_mul_core(accumulated, tileA[row][k], tileB[k][col]);
                                                }

                                                // Synchronize before loading the next tile
                                                barrier(CLK_LOCAL_MEM_FENCE);
                                            }

                                            // Write result to global memory
                                            C[globalRow * Width + globalCol] = accumulated;
                                        }
                         */

    private record Config(
            vec.Shape shape,
            boolean collectStats,
             boolean addField,
            List<vec.Shape> composableLanes){
        public String vectorName(){
            return "vec"+shape.lanes();
        }
        public String matName(){
            return "mat"+shape.lanes();
        }
        public static Config fieldButNoStats(TypeElement typeElement, int lanes, List<Integer>composableLaneSizes){
            var composableLanes = new ArrayList<vec.Shape>();
            for (var i:composableLaneSizes){
                composableLanes.add(vec.Shape.of(typeElement,i));
            }
            return new Config(vec.Shape.of(typeElement,lanes),false,true, composableLanes);
        }
    }

    final Config config;


    VecAndMatBuilder( Config config) {
        super(MethodHandles.lookup(), null);
        this.config = config;
    }

    VecAndMatBuilder vName() {
        return id(config.vectorName());
    }

    List<String> lNames() {
        return config.shape.laneNames();
    }

    int lanes() {
        return config.shape.lanes();
    }

    VecAndMatBuilder vType() {
        return typeName(config.vectorName());
    }
    VecAndMatBuilder mType() {
        return typeName(config.matName());
    }
    VecAndMatBuilder lType() {
        return type((JavaType) config.shape.typeElement());
    }

    VecAndMatBuilder vDecl(String id) {
        return vType().space().id(id);
    }
    VecAndMatBuilder mDecl(String id) {
        return mType().space().id(id);
    }

    VecAndMatBuilder lDecl(String id) {
        return lType().space().id(id);
    }

    VecAndMatBuilder simpleClassName(Class<?> clazz) {
        return typeName(clazz.getSimpleName());
    }

    VecAndMatBuilder cs() {
        return commaSpace();
    }

    VecAndMatBuilder id(String name) {
        return identifier(name);
    }

    VecAndMatBuilder idSp(String name) {
        return id(name).space();
    }

    VecAndMatBuilder idParen(String name, Consumer<VecAndMatBuilder> consumer) {
        return id(name).paren(consumer);
    }

    VecAndMatBuilder idDotIdParen(String n1, String n2) {
        return id(n1).dot().idParen(n2);
    }

    VecAndMatBuilder idParen(String name) {
        return id(name).ocparen();
    }

    VecAndMatBuilder snl() {
        return semicolonNl();
    }

    VecAndMatBuilder cnl() {
        return comma().nl();
    }


    VecAndMatBuilder f32Call(String name, Consumer<VecAndMatBuilder> consumer) {
        return simpleClassName(F32.class).dot().idParen(name, consumer);
    }


    //VecAndMatBuilder f32Inversesqrt(Consumer<VecAndMatBuilder> consumer) {
       // return f32Call("inversesqrt", consumer);
    //}


    VecAndMatBuilder staticLaneTypeFunc(
            String name, Consumer<VecAndMatBuilder> args, Consumer<VecAndMatBuilder> body) {
        return staticKwSp().func(_ -> lType(), name, args, body);
    }

    VecAndMatBuilder staticVecTypeFunc(
            String name, Consumer<VecAndMatBuilder> args, Consumer<VecAndMatBuilder> body) {
        return staticKwSp().func(_ -> vType(), name, args, body);
    }

    VecAndMatBuilder vec() {
        final String l = "l";
        final String m = "m";
        final String r = "r";
        final String e = "e";
        final String e1 = "e1";
        final String e2 = "e2";
        final var lnr = List.of(l, r);
        oracleCopyright();
        packageName(F32.class.getPackage());
        autoGenerated();
        importClasses(JavaType.class, IfaceValue.class, vec4.class, vec3.class, vec2.class);
        importStatic(vec4.class,"*");
        importStatic(vec3.class,"*");
        importStatic(vec2.class,"*");
        when(config.collectStats,  _ ->
                importClasses(AtomicInteger.class, AtomicBoolean.class)
        );
        nl();
        publicKwSp().interfaceKwSp().vName().space().extendsKwSp().simpleClassName(IfaceValue.class).dot().id("vec").body(_ -> {
            stmnt(_ ->
                    assign(
                            _ -> simpleClassName(vec.Shape.class).space().id("shape"),
                            _ -> typeName("Shape").dot().idParen("of", _ -> {
                                simpleClassName(JavaType.class).dot();
                                either(config.shape.typeElement() instanceof JavaType javaType && JavaType.FLOAT.equals(javaType),
                                        _ -> typeName("FLOAT"),
                                        _ -> typeName("INT"));
                                cs().intValue(lanes());
                            })
                    )
            ).nl();
            stmnt(_ -> join(lNames(), _ -> snl(), n -> lType().space().idParen(n))).nl(2);

            when(config.collectStats,  _ ->
                    stmnt(_ -> assign(
                                    _ -> simpleClassName(AtomicInteger.class).space().id("count"),
                                    _ -> newKwSp().simpleClassName(AtomicInteger.class).paren(_ -> intConstZero())
                            )).nl().stmnt(_ -> assign(
                                    _ -> simpleClassName(AtomicBoolean.class).space().id("collect"),
                                    _ -> newKwSp().simpleClassName(AtomicBoolean.class).paren(_ -> booleanFalse())
                            )
                    ).nl()
            );
            when(config.addField, _ -> {
                blockComment("This allows us to add this type to interface mapped segments ");
                interfaceKwSp().idSp("Field").extendsKwSp().vType().body(_ ->
                        stmnt(_ -> join(lNames(), _ -> snl(), n -> voidType().space().idParen(n, _ -> lDecl(n)))).nl()
                                .defaultKwSp().func(_ -> vType(), "of", _ -> join(lNames(), _ -> cs(), this::lDecl),
                                        _ -> stmnt(_ -> join(lNames(), _ -> snl(), n -> idParen(n, _ -> id(n))).snl()
                                                .returnKwSp().id("this"))
                                )
                                .defaultKwSp().func(_ -> vType(), "of", _ -> vType().space().vName(),
                                        _ -> stmnt(_ -> idParen("of", _ -> join(lNames(), _ -> cs(), n -> vName().dot().idParen(n))).snl()
                                                .returnKwSp().id("this")
                                        )
                                )
                ).nl(2);
            });

            staticVecTypeFunc(config.vectorName(), _ -> join(lNames(), _ -> cs(), this::lDecl),
                    _ -> record("Impl",
                            _ -> join(lNames(), _ -> cs(), this::lDecl),
                            _ -> vName()
                    )
                            .when(config.collectStats, _ ->
                                    ifKeyword().paren(_ -> idDotIdParen("collect", "get")).braceNlIndented(_ -> stmnt(_ -> idDotIdParen("count", "getAndIncrement")))).nl()
                            .returnKeyword(_ -> newKwSp().typeName("Impl").paren(_ -> join(lNames(), _ -> cs(), this::identifier)))
            );
            staticVecTypeFunc(config.vectorName(), _ -> lDecl("scalar"),
                    _ -> stmnt(_ ->
                            returnKwSp().vName().paren(_ ->
                                    join(lNames(), _ -> cs(), _ -> id("scalar"))
                            )
                    )
            );
            Map.of(
                    "add", "+",
                    "sub", "-",
                    "mul", "*",
                    "div", "/"
            ).forEach((fName, sym) -> {
                        staticVecTypeFunc(fName,
                                _ -> joinX2(lNames(), lnr, _ -> cs(), (side, n) -> lDecl(side + n)),
                                _ -> stmnt(_ ->
                                        returnKwSp().vName().paren(_ ->
                                                join(lNames(), _ -> cs(), n -> id(n + l).symbol(sym).id(n + r))
                                        )
                                )
                        );
                        staticVecTypeFunc(fName, _ -> join(lnr, _ -> cs(), this::vDecl),
                                _ -> stmnt(_ ->
                                        returnKwSp().idParen(fName, _ ->
                                                joinX2(lNames(), lnr, _ -> cs(), (n, side) -> idDotIdParen(side,n))
                                        )
                                )
                        );
                        staticVecTypeFunc(fName, _ -> lDecl(l).cs().vDecl(r),
                                _ -> stmnt(_ ->
                                        returnKwSp().idParen(fName, _ ->
                                                join(lNames(), _ -> cs(), n -> id(l).cs().idDotIdParen(r,n))
                                        )
                                )
                        );
                        staticVecTypeFunc(fName, _ -> vDecl(l).cs().lDecl(r),
                                _ -> stmnt(_ ->
                                        returnKwSp().idParen(fName, _ ->
                                                join(lNames(), _ -> cs(), n -> idDotIdParen(l,n).cs().id(r))
                                        )
                                )
                        );

                    }
            );

            List.of(
                    "pow", "min", "max"
            ).forEach(fName -> {
                        staticVecTypeFunc(fName, _ -> join(lnr, _ -> cs(), this::vDecl),
                                _ -> stmnt(_ ->
                                        returnKwSp().vName().paren(_ -> join(lNames(), _ -> cs(), n ->
                                                f32Call(fName, _ -> join(lnr, _ -> cs(), side -> idDotIdParen(side,n))))
                                        )
                                )
                        );
                        staticVecTypeFunc(fName, _ -> lDecl(l).cs().vDecl(r),
                                _ -> stmnt(_ ->
                                        returnKwSp().vName().paren(_ -> join(lNames(), _ -> cs(), n ->
                                                f32Call(fName, _ -> id(l).cs().idDotIdParen(r,n)))
                                        )
                                )

                        );
                        staticVecTypeFunc(fName, _ -> vDecl(l).cs().lDecl(r),
                                _ -> stmnt(_ ->
                                        returnKwSp().vName().paren(_ -> join(lNames(), _ -> cs(), n ->
                                                f32Call(fName, _ -> idDotIdParen(l,n).cs().id(r)))
                                        )
                                )

                        );
                    }
            );


            List.of("floor", "round", "fract", "abs", "log", "sin", "cos", "tan", "sqrt", "inversesqrt").forEach(fName ->
                    staticVecTypeFunc(fName, _ -> vDecl("v"),
                            _ -> stmnt(_ ->
                                    returnKwSp().vName().paren(_ ->
                                            join(lNames(), _ -> cs(), n -> f32Call(fName, _ -> idDotIdParen("v",n)))
                                    )
                            )
                    )
            );

            staticVecTypeFunc("neg", _ -> vDecl("v"),
                    _ -> stmnt(_ ->
                            returnKwSp().vName().paren(_ ->
                                    join(lNames(), _ -> cs(), n -> floatConstZero().minus().idDotIdParen("v",n))
                            )
                    )
            );
            if (!config.composableLanes.isEmpty()) {
                IntStream.range(2, lanes()).forEach(lane -> {
                    var argVecName = config.vectorName().substring(0, config.vectorName().length() - 1) + lane; //maps vec4 ->  vec2 ,vec3 etc
                    var trailingArgs = lNames().subList(lane, lanes());
                    staticVecTypeFunc(config.vectorName(),                    // name
                            _ -> typeName(argVecName).space().id(argVecName).cs().join(trailingArgs, _ -> cs(), this::lDecl),
                            _ -> stmnt(_ ->
                                    returnKwSp().vName().paren(_ ->
                                            IntStream.range(0, lanes()).forEach(argPos ->
                                                    when(argPos > 0, _ -> cs())
                                                            .either(argPos < lane,
                                                                    _ -> typeName(argVecName).dot().idParen(lNames().get(argPos)),
                                                                    _ -> id(lNames().get(argPos))
                                                            )
                                            )
                                    )
                            )
                    );

                });
            }

            staticLaneTypeFunc("dot", _ -> join(lnr, _ -> cs(), this::vDecl),
                    _ -> stmnt(_ ->
                            returnKwSp().join(lNames(), _ -> add(), n ->
                                    idDotIdParen(l,n).mul().idDotIdParen(r,n)
                            )
                    )
            );

            staticLaneTypeFunc("sumOfSquares", _ -> vDecl("v"),
                    _ -> stmnt(_ ->
                            returnKwSp().idParen("dot", _ ->
                                    id("v").cs().id("v")
                            )
                    )
            );

            staticLaneTypeFunc("length", _ -> vDecl("v"),
                    _ -> stmnt(_ ->
                            returnKwSp().f32Call("sqrt",_ ->
                                    idParen("sumOfSquares", _ -> id("v"))
                            )
                    )
            );
            staticVecTypeFunc("clamp", _ -> vDecl("v").cs().lDecl("min").cs().lDecl("max"),
                    _ -> stmnt(_ ->
                            returnKwSp().vName().paren(_ ->
                                    join(lNames(), _ -> cs(), n -> f32Call("clamp",_ ->
                                            idDotIdParen("v",n).cs().id("min").cs().id("max"))
                                    )
                            )
                    )
            );

            staticVecTypeFunc("normalize", _ -> vDecl("v"),
                    _ -> {
                        stmnt(_ ->
                                assign(_ -> lDecl("lenSq"), _ -> idParen("sumOfSquares", _ -> id("v")))
                        ).nl(2);
                        stmnt(_ ->
                                returnKwSp().tern(
                                        _ -> idSp("lenSq").gt().floatConstZero(),
                                        _ -> idParen("mul", _ -> id("v").cs().f32Call("inversesqrt",_ -> id("lenSq"))),
                                        _ -> vName().paren(_ -> floatConstZero()))
                        );
                    });

            staticVecTypeFunc("reflect", _ -> vDecl(l).cs().vDecl(r),
                    _ -> lineComment("lhs - 2f * dot(rhs, lhs) * rhs")
                            .stmnt(_ -> returnKwSp().vName().dot().idParen("sub", _ ->
                                    id(l).cs().idParen("mul", _ ->
                                            idParen("mul", _ -> id(r).cs().id(l)).cs().constant("2f"))
                            ))
            );

            staticLaneTypeFunc("distance", _ -> join(lnr, _ -> cs(), this::vDecl),
                    _ -> {
                        stmnt(_ -> join(lNames(), _ -> snl(), n ->
                                assign(_ -> lDecl("d" + n), _ -> idDotIdParen(r,n).sub().idDotIdParen(l,n))
                        )).nl();

                        stmnt(_ ->
                                returnKwSp().f32Call("sqrt",_ ->
                                        join(lNames(), _ -> add(), n -> id("d" + n).mul().id("d" + n))
                                )
                        );
                    });

            staticVecTypeFunc("smoothstep", _ -> join(List.of(e1, e2, r), _ -> cs(), this::vDecl),
                    _ -> stmnt(_ ->
                            returnKwSp().vName().paren(_ ->
                                    indent(_ -> nl()
                                            .join(lNames(), _ -> cnl(),
                                                    n -> f32Call("smoothstep",_ ->
                                                            idDotIdParen(e1,n).cs().idDotIdParen(e2,n).cs().idDotIdParen(r,n)
                                                    )
                                            ).nl()
                                    )
                            )
                    )
            );

            staticVecTypeFunc("step", _ -> join(List.of(e, r), _ -> cs(), this::vDecl),
                    _ -> stmnt(_ ->
                            returnKwSp().vName().paren(_ ->
                                    indent(_ -> nl().join(lNames(), _ -> cnl(),
                                                    n -> f32Call("step",_ -> idDotIdParen(e,n).cs().idDotIdParen(r,n))
                                            ).nl()
                                    )
                            )
                    )
            );
            staticVecTypeFunc("mix", _ -> join(lnr, _ -> cs(), this::vDecl).cs().lDecl("v"),
                    _ -> stmnt(_ ->
                            returnKwSp().vName().paren(_ -> indent(_ -> nl()
                                                    .join(lNames(), _ -> cnl(), n ->
                                                            f32Call("mix",_ ->
                                                                    idDotIdParen(l,n).cs().idDotIdParen(r,n).cs().id("v"))
                                                    )
                                    ).nl()
                            )
                    )

            );

            staticVecTypeFunc("mix", _ -> join(lnr, _ -> cs(), this::vDecl).cs().vDecl("v"),
                    _ -> stmnt(_ ->
                            returnKwSp().vName().paren(_ -> indent(_ ->
                                            nl().join(lNames(), _ -> cnl(), n ->
                                                            f32Call("mix",_ ->
                                                                    idDotIdParen(l,n).cs().idDotIdParen(r,n).cs().idDotIdParen("v",n))
                                                    )
                                    ).nl()
                            )
                    )

            );
            staticVecTypeFunc("mod", _ -> join(lnr, _ -> cs(), this::vDecl),
                    _ -> stmnt(_ ->
                            returnKwSp().vName().paren(_ -> indent(_ ->
                                            nl()
                                                    .separated(lNames(), _ ->
                                                            cnl(), n ->
                                                            f32Call("mod",_ -> idDotIdParen(l,n).cs().idDotIdParen(r,n))
                                                    )
                                    )
                                            .nl()
                            )
                    )

            );
            staticVecTypeFunc("mod", _ -> vDecl(l).cs().lDecl(r),
                    _ -> stmnt(_ ->
                            returnKwSp().vName().paren(_ -> indent(_ -> nl()
                                                    .separated(lNames(), _ ->
                                                            cnl(), n ->
                                                            f32Call("mod",_ -> idDotIdParen(l,n).cs().id(r)))
                                    ).nl()
                            )
                    )

            );


            record SideAndPrefix(String side, String... idx){};
            var matMul = switch (lanes()){
                case 2 ->List.of(
                        new SideAndPrefix("x","_00","_01"),
                        new SideAndPrefix("y","_10", "_11")
                );
                case 3 -> List.of(
                        new SideAndPrefix("x","_00","_01","_02"),
                        new SideAndPrefix("y","_10", "_11","_12"),
                        new SideAndPrefix("z","_20", "_21","_22")
                );
                default -> null;
            };
            if (lanes()==2 || lanes()==3){
                staticVecTypeFunc("mul", _ -> vDecl(l).cs().mDecl(r),
                        _ -> stmnt(_ ->
                                returnKwSp().vName().paren(_ -> indent(_ -> nl()
                                                .join( matMul,_->cnl(),i->
                                                        join(List.of(i.idx),_->add(), i2->
                                                                idDotIdParen(l, i.side).mul().idDotIdParen(r, i2)
                                                        )
                                                ).nl()
                                        )
                                )
                        )
                );
            }

            if (lanes()==2) {
                List.of("xx","xy", "yy","xz", "yz").forEach(swizzle -> {
                    char[] chars = new char[swizzle.length()];
                    swizzle.getChars(0, swizzle.length(), chars, 0);
                    var retVecName = config.vectorName().substring(0, config.vectorName().length() - 1) + (config.shape.lanes() +1); //maps vec2 ->  vec3

                    staticVecTypeFunc(swizzle,
                            _ -> typeName(retVecName).space().id("v"),
                            _ -> stmnt(_ ->
                                    returnKwSp().vName().paren(_ ->
                                            idDotIdParen("v", Character.toString(chars[0]))
                                                    .commaSpace()
                                                    .idDotIdParen("v", Character.toString(chars[1]))

                                    )

                            )
                    );
                });
            }else if (lanes()==3){
                //if (lanes()==2) {
                    List.of("xxx","xxy", "xxz","xyy", "xyz", "xzz", "yyy", "yyz","yzz","zzz").forEach(swizzle -> {
                        char[] chars = new char[swizzle.length()];
                        swizzle.getChars(0, swizzle.length(), chars, 0);
                        var retVecName = config.vectorName().substring(0, config.vectorName().length() - 1) + (config.shape.lanes() +1); //maps vec2 ->  vec3

                        staticVecTypeFunc(swizzle,
                                _ -> typeName(retVecName).space().id("v"),
                                _ -> stmnt(_ ->
                                        returnKwSp().vName().paren(_ ->
                                                idDotIdParen("v", Character.toString(chars[0]))
                                                        .commaSpace()
                                                        .idDotIdParen("v", Character.toString(chars[1]))
                                                        .commaSpace()
                                                        .idDotIdParen("v", Character.toString(chars[2]))

                                        )

                                )
                        );
                    });
               // }
            }


            switch (lanes()) {
            //    case 2 ->
                 //   preformatted("""
                    //             static vec2 xy(vec3 vec3) {return vec2(vec3.x(), vec3.y());}
                      //           static vec2 xz(vec3 vec3) {return vec2(vec3.x(), vec3.z());}
                        //         static vec2 yz(vec3 vec3) {return vec2(vec3.y(), vec3.z());}
                          //  """);
                case 3 -> {
                    // Hack for vec3 until we have this working
                    preformatted("""
                                static vec3 vec3(float x, vec2 yz) {return vec3(x, yz.x(), yz.y());}
                            """);

                    staticVecTypeFunc(
                            "cross",                    // name
                            _ -> join(List.of(l, r), _ -> cs(), this::vDecl),
                            _ -> stmnt(_ ->
                                    returnKwSp().vName().paren(_ -> indent(_ -> nl()
                                                    // lhs.y() * rhs.z() - lhs.z() * rhs.y(),
                                                    .idDotIdParen(l, "y").mul().idDotIdParen(r, "z").sub().idDotIdParen(l, "z").mul().idDotIdParen(r, "y").cnl()
                                                    //lhs.z() * rhs.x() - lhs.x() * rhs.z(),
                                                    .idDotIdParen(l, "z").mul().idDotIdParen(r, "x").sub().idDotIdParen(l, "x").mul().idDotIdParen(r, "z").cnl()
                                                    //lhs.x() * rhs.y() - lhs.y() * rhs.x()
                                                    .idDotIdParen(l, "x").mul().idDotIdParen(r, "y").sub().idDotIdParen(l, "y").mul().idDotIdParen(r, "x")
                                            ).nl()
                                    )
                            )

                    );
                }
            }
        });
        return self();
    }
    private void writeToFile(Path path) {
        try {
            Files.writeString(path.resolve(config.vectorName() + ".java"), vec().toString());
        }catch (Throwable t){
            throw new RuntimeException(t);
        }
    }
    static void createSource(Config config,Path path) {
        new VecAndMatBuilder(config).writeToFile(path);
    }

    static void main(String[] argv) throws IOException {

        var path = Files.createDirectories(Path.of(
                "/Users/grfrost/github/babylon-grfrost-fork/hat/vecs/java/hat/types" // to test
                //   "/Users/grfrost/github/babylon-grfrost-fork/hat/core/src/main/java/hat/types" // for hat core
        ));
        List.of(
                VecAndMatBuilder.Config.fieldButNoStats(JavaType.FLOAT, 2, List.of()),
                VecAndMatBuilder.Config.fieldButNoStats(JavaType.FLOAT, 3, List.of(2)),
                VecAndMatBuilder.Config.fieldButNoStats(JavaType.FLOAT, 4, List.of(2,3))
              //  VecAndMatBuilder.Config.fieldButNoStats(JavaType.FLOAT, 8,List.of(2,3,4)),
              //  VecAndMatBuilder.Config.fieldButNoStats(JavaType.FLOAT, 16,List.of(2,3,4,8))
        ).forEach(c->createSource(c,path));

    }


}
