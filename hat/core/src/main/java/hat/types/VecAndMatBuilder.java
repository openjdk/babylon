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
package hat.types;

import jdk.incubator.code.dialect.java.JavaType;
import optkl.IfaceValue;
import optkl.IfaceValue.vec;
import optkl.codebuilders.JavaCodeBuilder;

import java.io.IOException;
import java.lang.invoke.MethodHandles;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.Consumer;
import java.util.stream.IntStream;

public class VecAndMatBuilder {
    private record Config(
            Path path,
            vec.Shape shape,
            boolean collectStats
    ) {
        public String vectorName() {
            return "vec" + shape.lanes();
        }

        public String matName() {
            return "mat" + shape.lanes();
        }

        public static Config noStats(Path path, vec.Shape shape){
            return new Config(path, shape, false);
        }
    }


    static class VecBuilder extends  JavaCodeBuilder<VecBuilder> {
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



      define TS 16 // Tile Size

      _kernel void mat4_mul_tiled(__global const float16* A,
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
     */

      final Config config;


        VecBuilder(Config config) {
            super(MethodHandles.lookup(), null);
            this.config = config;
        }

        VecBuilder vName() {
            return id(config.vectorName());
        }

        List<String> lNames() {
            return config.shape.laneNames();
        }

        int lanes() {
            return config.shape.lanes();
        }

        VecBuilder vType() {
            return type(config.vectorName());
        }

        VecBuilder mType() {
            return type(config.matName());
        }

        VecBuilder lType() {
            return type((JavaType) config.shape.codeType());
        }

        VecBuilder vDecl(String id) {
            return vType().sp().id(id);
        }

        VecBuilder mDecl(String id) {
            return mType().sp().id(id);
        }

        VecBuilder lDecl(String id) {
            return lType().sp().id(id);
        }

        VecBuilder simpleClassName(Class<?> clazz) {
            return type(clazz.getSimpleName());
        }

        VecBuilder idSp(String name) {
            return id(name).sp();
        }

        VecBuilder idParen(String name, Consumer<VecBuilder> consumer) {
            return id(name).paren(consumer);
        }

        VecBuilder idDotIdParen(String n1, String n2) {
            return id(n1).dot().idParen(n2);
        }

        VecBuilder idParen(String name) {
            return id(name).ocparen();
        }

        VecBuilder f32Call(String name, Consumer<VecBuilder> consumer) {
            return simpleClassName(F32.class).dot().idParen(name, consumer);
        }

        private void writeToFile( String suffix) {
            try {
                Files.writeString(config.path.resolve(config.vectorName() + "."+suffix), toString());
            }catch (Throwable t){
                throw new RuntimeException(t);
            }
        }
    }


    static VecBuilder createVecs(Config config) {
        final String l = "l";
        final String m = "m";
        final String r = "r";
        final String e = "e";
        final String e1 = "e1";
        final String e2 = "e2";
        final var lnr = List.of(l, r);
        return new VecBuilder(config)
                .oracleCopyright()
                .packageName(F32.class.getPackage())
                .autoGenerated()
                .importClasses(JavaType.class, IfaceValue.class, vec4.class, vec3.class, vec2.class)
                .importStatic(vec4.class,"*")
                .importStatic(vec3.class,"*")
                .importStatic(vec2.class,"*")
                .when(config.collectStats(),  vb -> vb.importClasses(AtomicInteger.class, AtomicBoolean.class))
                .nl()
                .publicKwSp().interfaceKwSp().vName().sp().extendsKwSp().simpleClassName(IfaceValue.class).dot().id("vec").body(vb -> {
            vb.stmnt(_ ->
                    vb.assign(
                            _ -> vb.simpleClassName(vec.Shape.class).sp().id("shape"),
                            _ -> vb.type("Shape").dot().idParen("of", _ -> {
                                vb.simpleClassName(JavaType.class).dot();
                                vb.either(config.shape.codeType() instanceof JavaType javaType && JavaType.FLOAT.equals(javaType),
                                        _ -> vb.type("FLOAT"),
                                        _ -> vb.type("INT"));
                                vb.csp().intValue(vb.lanes());
                            })
                    )
            ).nl();
            vb.stmnt(_ -> vb.sep(vb.lNames(), _ -> vb.snl(), n -> vb.lType().sp().idParen(n))).nl(2);

            vb.when(config.collectStats,  _ ->
                    vb.stmnt(_ -> vb.assign(
                                    _ -> vb.simpleClassName(AtomicInteger.class).sp().id("count"),
                                    _ -> vb.newKwSp().simpleClassName(AtomicInteger.class).paren(_ -> vb.intConstZero())
                            )).nl().stmnt(_ -> vb.assign(
                                    _ -> vb.simpleClassName(AtomicBoolean.class).sp().id("collect"),
                                    _ -> vb.newKwSp().simpleClassName(AtomicBoolean.class).paren(_ -> vb.boolConst(false))
                            )
                    ).nl()
            );

                vb.blockComment("This allows us to add this type to interface mapped segments ");
                vb.interfaceKwSp().idSp("Field").extendsKwSp().vType().body(_ ->
                        vb.stmnt(_ -> vb.sep(vb.lNames(), _ -> vb.snl(), n -> vb.voidType().sp().idParen(n, _ -> vb.lDecl(n)))).nl()
                                .defaultKwSp().func(_ -> vb.vType(), "of", _ -> vb.sep(vb.lNames(), _ -> vb.csp(), vb::lDecl),
                                        _ -> vb.stmnt(_ -> vb.sep(vb.lNames(), _ -> vb.snl(), n -> vb.idParen(n, _ -> vb.id(n))).snl()
                                                .returnKwSp().id("this"))
                                )
                                .defaultKwSp().func(_ -> vb.vType(), "of", _ -> vb.vType().sp().vName(),
                                        _ -> vb.stmnt(_ -> vb.idParen("of", _ -> vb.sep(vb.lNames(), _ -> vb.csp(), n -> vb.vName().dot().idParen(n))).snl()
                                                .returnKwSp().id("this")
                                        )
                                )
                ).nl(2);


            vb.staticKwSp().func(_ -> vb.vType(),config.vectorName(), _ -> vb.sep(vb.lNames(), _ -> vb.csp(), vb::lDecl),
                    _ -> vb.record("Impl",
                            _ -> vb.sep(vb.lNames(), _ -> vb.csp(), vb::lDecl),
                            _ -> vb.vName()
                    )
                            .when(config.collectStats, _ ->
                                    vb.ifKeyword().paren(_ -> vb.idDotIdParen("collect", "get")).braceNlIndented(_ -> vb.stmnt(_ -> vb.idDotIdParen("count", "getAndIncrement")))).nl()
                            .returnKeyword(_ -> vb.newKwSp().type("Impl").paren(_ -> vb.sep(vb.lNames(), _ -> vb.csp(), vb::id)))
            );
            vb.staticKwSp().func(_ -> vb.vType(),config.vectorName(), _ -> vb.lDecl("scalar"),
                    _ -> vb.stmnt(_ ->
                            vb.returnKwSp().vName().paren(_ ->
                                    vb.sep(vb.lNames(), _ -> vb.csp(), _ -> vb.id("scalar"))
                            )
                    )
            );
            var funcNameToBinaryOpMap = Map.of(
                    "add", "+",
                    "sub", "-",
                    "mul", "*",
                    "div", "/"
            );
            funcNameToBinaryOpMap.forEach((fName, sym) -> {
                vb.staticKwSp().func(_ -> vb.vType(),fName,
                        _ -> vb.joinX2(vb.lNames(), lnr, _ -> vb.csp(), (side, n) -> vb.lDecl(side + n)),
                        _ -> vb.stmnt(_ ->
                                        vb.returnKwSp().vName().paren(_ ->
                                                vb.sep(vb.lNames(), _ -> vb.csp(), n -> vb.id(n + l).symbol(sym).id(n + r))
                                        )
                        )
                );
                vb.staticKwSp().func(_ -> vb.vType(),fName,
                        _ -> vb.sep(lnr, _ -> vb.csp(), vb::vDecl),
                        _ -> vb.stmnt(_ ->
                                        vb.returnKwSp().idParen(fName, _ ->
                                                vb.joinX2(vb.lNames(), lnr, _ -> vb.csp(), (n, side) -> vb.idDotIdParen(side,n))
                                        )
                        )
                );
                vb.staticKwSp().func(_ -> vb.vType(),fName,
                        _ -> vb.lDecl(l).csp().vDecl(r),
                        _ -> vb.stmnt(_ ->
                                        vb.returnKwSp().idParen(fName, _ ->
                                                vb.sep(vb.lNames(), _ -> vb.csp(), n -> vb.id(l).csp().idDotIdParen(r,n))
                                        )
                        )
                );
                vb.staticKwSp().func(_ -> vb.vType(),fName,
                        _ -> vb.vDecl(l).csp().lDecl(r),
                        _ -> vb.stmnt(_ ->
                                        vb.returnKwSp().idParen(fName, _ ->
                                                vb.sep(vb.lNames(), _ -> vb.csp(), n -> vb.idDotIdParen(l,n).csp().id(r))
                                        )
                        )
                );
            });

            List.of(
                    "pow", "min", "max"
            ).forEach(fName -> {
                        vb.staticKwSp().func(_ -> vb.vType(),fName, _ -> vb.sep(lnr, _ -> vb.csp(), vb::vDecl),
                                _ -> vb.stmnt(_ ->
                                        vb.returnKwSp().vName().paren(_ -> vb.sep(vb.lNames(), _ -> vb.csp(), n ->
                                                vb.f32Call(fName, _ -> vb.sep(lnr, _ -> vb.csp(), side -> vb.idDotIdParen(side,n))))
                                        )
                                )
                        );
                        vb.staticKwSp().func(_ -> vb.vType(),fName, _ -> vb.lDecl(l).csp().vDecl(r),
                                _ -> vb.stmnt(_ ->
                                        vb.returnKwSp().vName().paren(_ -> vb.sep(vb.lNames(), _ -> vb.csp(), n ->
                                                vb.f32Call(fName, _ -> vb.id(l).csp().idDotIdParen(r,n)))
                                        )
                                )

                        );
                        vb.staticKwSp().func(_ -> vb.vType(),fName, _ -> vb.vDecl(l).csp().lDecl(r),
                                _ -> vb.stmnt(_ ->
                                        vb.returnKwSp().vName().paren(_ -> vb.sep(vb.lNames(), _ -> vb.csp(), n ->
                                                vb.f32Call(fName, _ -> vb.idDotIdParen(l,n).csp().id(r)))
                                        )
                                )

                        );
                    }
            );


            List.of("floor", "round", "fract", "abs", "log", "sin", "cos", "tan", "sqrt", "inversesqrt").forEach(fName ->
                    vb.staticKwSp().func(_ -> vb.vType(),fName, _ -> vb.vDecl("v"),
                            _ -> vb.stmnt(_ ->
                                    vb.returnKwSp().vName().paren(_ ->
                                            vb.sep(vb.lNames(), _ -> vb.csp(), n -> vb.f32Call(fName, _ -> vb.idDotIdParen("v",n)))
                                    )
                            )
                    )
            );

            vb.staticKwSp().func(_ -> vb.vType(),"neg", _ -> vb.vDecl("v"),
                    _ -> vb.stmnt(_ ->
                            vb.returnKwSp().vName().paren(_ ->
                                    vb.sep(vb.lNames(), _ -> vb.csp(), n -> vb.floatConstZero().minus().idDotIdParen("v",n))
                            )
                    )
            );


            vb.staticKwSp().func(_ -> vb.lType(),"dot", _ -> vb.sep(lnr, _ -> vb.csp(), vb::vDecl),
                    _ -> vb.stmnt(_ ->
                            vb.returnKwSp().sep(vb.lNames(), _ -> vb.add(), n ->
                                    vb.idDotIdParen(l,n).mul().idDotIdParen(r,n)
                            )
                    )
            );

            vb.staticKwSp().func(_ -> vb.lType(),"sumOfSquares", _ -> vb.vDecl("v"),
                    _ -> vb.stmnt(_ ->
                            vb.returnKwSp().idParen("dot", _ ->
                                    vb.id("v").csp().id("v")
                            )
                    )
            );

            vb.staticKwSp().func(_ -> vb.lType(),"length", _ -> vb.vDecl("v"),
                    _ -> vb.stmnt(_ ->
                            vb.returnKwSp().f32Call("sqrt",_ ->
                                    vb.idParen("sumOfSquares", _ -> vb.id("v"))
                            )
                    )
            );
            vb.staticKwSp().func(_ -> vb.vType(),"clamp", _ -> vb.vDecl("v").csp().lDecl("min").csp().lDecl("max"),
                    _ -> vb.stmnt(_ ->
                            vb.returnKwSp().vName().paren(_ ->
                                    vb.sep(vb.lNames(), _ -> vb.csp(), n -> vb.f32Call("clamp", _ ->
                                            vb.idDotIdParen("v",n).csp().id("min").csp().id("max"))
                                    )
                            )
                    )
            );

            vb.staticKwSp().func(_ -> vb.vType(),"normalize", _ -> vb.vDecl("v"),
                    _ -> vb.stmnt(_ ->
                                vb.assign(_ -> vb.lDecl("lenSq"), _ -> vb.idParen("sumOfSquares", _ -> vb.id("v")))
                        ).nl().stmnt(_ ->
                                vb.returnKwSp().tern(
                                        _ -> vb.idSp("lenSq").gt().floatConstZero(),
                                        _ -> vb.idParen("mul", _ -> vb.id("v").csp().f32Call("inversesqrt", _ -> vb.id("lenSq"))),
                                        _ -> vb.vName().paren(_ -> vb.floatConstZero()))
                        )
                    );

            vb.staticKwSp().func(_ -> vb.vType(),"reflect", _ -> vb.vDecl(l).csp().vDecl(r),
                    _ -> vb.stmnt(_ -> vb.returnKwSp().vName().dot().idParen("sub", _ ->
                                    vb.id(l).csp().idParen("mul", _ ->
                                            vb.idParen("mul", _ -> vb.id(r).csp().id(l)).csp().floatConst(2f))
                            ))
            );

            vb.staticKwSp().func(_ -> vb.lType(),"distance", _ -> vb.sep(lnr, _ -> vb.csp(), vb::vDecl), _ ->
                        vb.stmnt(_ -> vb.sep(vb.lNames(), _ -> vb.snl(), n ->
                                vb.assign(_ -> vb.lDecl("d" + n), _ -> vb.idDotIdParen(r,n).sub().idDotIdParen(l,n))
                        )).nl().stmnt(_ -> vb.returnKwSp().f32Call("sqrt",_ ->
                                vb.sep(vb.lNames(), _ -> vb.add(), n -> vb.id("d" + n).mul().id("d" + n))
                        ))
                    );

            vb.staticKwSp().func(_ -> vb.vType(),"smoothstep", _ -> vb.sep(List.of(e1, e2, r), _ -> vb.csp(), vb::vDecl),
                    _ -> vb.stmnt(_ ->
                            vb.returnKwSp().vName().parenNlIndented(_ ->
                                    vb.sep(vb.lNames(), _ -> vb.cnl(), n ->
                                                    vb.f32Call("smoothstep",_ ->
                                                            vb.idDotIdParen(e1,n).csp().idDotIdParen(e2,n).csp().idDotIdParen(r,n)
                                                    )
                                            )
                                    )
                    )
            );

            vb.staticKwSp().func(_ -> vb.vType(),"step", _ -> vb.sep(List.of(e, r), _ -> vb.csp(), vb::vDecl),
                    _ -> vb.stmnt(_ ->
                            vb.returnKwSp().vName().parenNlIndented(_ ->
                                    vb.sep(vb.lNames(), _ -> vb.cnl(), n ->
                                                    vb.f32Call("step", _->
                                                            vb.idDotIdParen(e,n).csp().idDotIdParen(r,n)
                                                    )
                                    )
                            )
                    )
            );
            vb.staticKwSp().func(_ -> vb.vType(),"mix", _ -> vb.sep(lnr, _ -> vb.csp(), vb::vDecl).csp().lDecl("v"),
                    _ -> vb.stmnt(_ ->
                            vb.returnKwSp().vName().parenNlIndented(_ ->
                                            vb.sep(vb.lNames(), _ -> vb.cnl(), n ->
                                                    vb.f32Call("mix",_ ->
                                                            vb.idDotIdParen(l,n).csp().idDotIdParen(r,n).csp().id("v")
                                                    )
                                            )
                            )
                    )

            );

            vb.staticKwSp().func(_ -> vb.vType(),"mix", _ -> vb.sep(lnr, _ -> vb.csp(), vb::vDecl).csp().vDecl("v"),
                    _ -> vb.stmnt(_ ->
                            vb.returnKwSp().vName().parenNlIndented(_ ->
                                            vb.sep(vb.lNames(), _ -> vb.cnl(), n ->
                                                            vb.f32Call("mix",_ ->
                                                                    vb.idDotIdParen(l,n).csp().idDotIdParen(r,n).csp().idDotIdParen("v",n))
                                                    )
                                    )
                    )

            );
            vb.staticKwSp().func(_ -> vb.vType(),"mod", _ -> vb.sep(lnr, _ -> vb.csp(), vb::vDecl),
                    _ -> vb.stmnt(_ ->
                            vb.returnKwSp().vName().parenNlIndented(_->
                                                    vb.sep(vb.lNames(), _ ->
                                                            vb.cnl(), n ->
                                                            vb.f32Call("mod",_ -> vb.idDotIdParen(l,n).csp().idDotIdParen(r,n))
                                                    )
                            )
                    )

            );
            vb.staticKwSp().func(_ -> vb.vType(),"mod", _ ->  vb.vDecl(l).csp().lDecl(r),
                    _ -> vb.stmnt(_ -> vb.returnKwSp().vName()
                            .parenNlIndented(_ ->
                                            vb.sep(vb.lNames(), _ ->
                                                    vb.cnl(), n ->
                                                    vb.f32Call("mod",_ -> vb.idDotIdParen(l,n).csp().id(r)))
                            )
                    )

            );

            IntStream.range(2, vb.lanes()).forEach(lane -> {
                var argVecName = config.vectorName().substring(0, config.vectorName().length() - 1) + lane; //maps vec4 ->  vec2 ,vec3 etc
                var trailingArgs = vb.lNames().subList(lane, vb.lanes());
                vb.staticKwSp().func(_ -> vb.vType(),config.vectorName(),                    // name
                        _ -> vb.type(argVecName).sp().id(argVecName).csp().sep(trailingArgs, _ -> vb.csp(), vb::lDecl),
                        _ -> vb.stmnt(_ ->
                                vb.returnKwSp().vName().paren(_ ->
                                        IntStream.range(0, vb.lanes()).forEach(argPos ->
                                                vb.when(argPos > 0, _ -> vb.csp())
                                                        .either(argPos < lane,
                                                                _ -> vb.type(argVecName).dot().idParen(vb.lNames().get(argPos)),
                                                                _ -> vb.id(vb.lNames().get(argPos))
                                                        )
                                        )
                                )
                        )
                );
            });

            if (vb.lanes()==2 || vb.lanes()==3) {
                if (vb.lanes() == 3) {
                    vb.staticKwSp().func(_ -> vb.vType(),
                            "cross",                    // name
                            _ -> vb.sep(List.of(l, r), _ -> vb.csp(), vb::vDecl),
                            _ -> vb.stmnt(_ ->
                                    vb.returnKwSp().vName().parenNlIndented(_ -> vb
                                            .idDotIdParen(l, "y").mul().idDotIdParen(r, "z").sub().idDotIdParen(l, "z").mul().idDotIdParen(r, "y").cnl()
                                            .idDotIdParen(l, "z").mul().idDotIdParen(r, "x").sub().idDotIdParen(l, "x").mul().idDotIdParen(r, "z").cnl()
                                            .idDotIdParen(l, "x").mul().idDotIdParen(r, "y").sub().idDotIdParen(l, "y").mul().idDotIdParen(r, "x")
                                    )

                            )

                    );


                // Hack for vec3 until we have this working
                vb.preformatted("""
                                    static vec3 vec3(float x, vec2 yz) {
                                       return vec3(x, yz.x(), yz.y());
                                    }
                        """).nl();
            }
                record SideAndPrefix(String side, String... idx) {
                }

                var matMul = switch (vb.lanes()) {
                    case 2 -> List.of(
                            new SideAndPrefix("x", "_00", "_01"),
                            new SideAndPrefix("y", "_10", "_11")
                    );
                    case 3 -> List.of(
                            new SideAndPrefix("x", "_00", "_01", "_02"),
                            new SideAndPrefix("y", "_10", "_11", "_12"),
                            new SideAndPrefix("z", "_20", "_21", "_22")
                    );
                    default -> null;
                };
                vb.staticKwSp().func(_ -> vb.vType(), "mul", _ -> vb.vDecl(l).csp().mDecl(r),
                        _ -> vb.stmnt(_ ->
                                vb.returnKwSp().vName().parenNlIndented(_ ->
                                        vb.sep(matMul, _ -> vb.cnl(), i ->
                                                vb.sep(List.of(i.idx), _ -> vb.add(), i2 ->
                                                        vb.idDotIdParen(l, i.side).mul().idDotIdParen(r, i2)
                                                )
                                        )
                                )
                        )
                );


                var swizzles = (vb.lanes() == 2)
                        ? List.of("xx", "xy", "yy", "xz", "yz")
                        : List.of("xxx", "xxy", "xxz", "xyy", "xyz", "xzz", "yyy", "yyz", "yzz", "zzz");

                swizzles.forEach(swizzle -> {
                    var swizzleLaneName = swizzle.chars().mapToObj(Character::toString).toList();
                    var retVecName = config.vectorName().substring(0, config.vectorName().length() - 1) + (config.shape.lanes() + 1); //maps vec2 ->  vec3

                    vb.staticKwSp().func(_ -> vb.vType(), swizzle,
                            _ -> vb.type(retVecName).sp().id("v"),
                            _ -> vb.stmnt(_ ->
                                    vb.returnKwSp().vName().paren(_ -> vb.sep(swizzleLaneName, _ -> vb.csp(), n -> vb.idDotIdParen("v", n)))
                            )
                    );
                });
            }
        });
    }

    static void main(String[] argv) throws IOException {

        var path = Files.createDirectories(Path.of(
               // "/Users/grfrost/github/babylon-grfrost-fork/hat/vecs/java/hat/types" // to test
                   "/Users/grfrost/github/babylon-grfrost-fork/hat/core/src/main/java/hat/types" // for hat core
        ));
        List.of(
                Config.noStats(path, vec.Shape.of(JavaType.FLOAT, 2)),
                Config.noStats(path,vec.Shape.of(JavaType.FLOAT, 3)),
                Config.noStats(path, vec.Shape.of(JavaType.FLOAT, 4))
        ).forEach(c->createVecs(c).writeToFile( "java"));

    }


}
