package shade;

import jdk.incubator.code.dialect.java.JavaType;
import optkl.IfaceValue;
import optkl.codebuilders.JavaCodeBuilder;

import java.lang.invoke.MethodHandles;
import java.util.List;
import java.util.function.BiConsumer;

public class VecAndMatBuilder extends JavaCodeBuilder<VecAndMatBuilder> {

    static final String lhs = "l";
    static final String rhs = "r";


    VecAndMatBuilder() {
        super(MethodHandles.lookup(), null);
    }

    static void main(String[] argv) {
        var jc = new VecAndMatBuilder();
        record LaneInfo(String name, IfaceValue.vec.Shape shape) {}
        record UnaryFunc(String name, boolean returnsScalar) { }
        record Op(String name, String symbol) { }

        List.of("ivec2", "vec2", "vec3", "vec4").forEach(vec -> {
            jc.interfaceKeyword(vec).space().dotted("IfaceValue", "vec").body(_ -> {
                jc.typeName("Shape").space().identifier("shape").equals();
                jc.dotted("Shape", "of")
                        .paren(_ ->
                                jc.dotted("JavaType", "FLOAT").comma().intValue(3)
                        ).semicolonNl().nl();
                jc.typeName("float").call("x").semicolonNl();
                jc.typeName("float").call("y").semicolonNl();
                jc.nl();



                List.of(
                        new LaneInfo("ivec2", IfaceValue.vec.Shape.of(JavaType.INT, 2)),
                        new LaneInfo("ivec3", IfaceValue.vec.Shape.of(JavaType.INT, 3)),
                        new LaneInfo("vec2", IfaceValue.vec.Shape.of(JavaType.FLOAT, 2)),
                        new LaneInfo("vec3", IfaceValue.vec.Shape.of(JavaType.FLOAT, 3)),
                        new LaneInfo("vec4", IfaceValue.vec.Shape.of(JavaType.FLOAT, 4))
                )
                        .forEach(laneInfo -> {
                            jc.func(_ -> jc.type(laneInfo.shape),
                                    laneInfo.name,
                                    _ -> jc.commaSpaceSeparated(laneInfo.shape.laneNames(), c -> jc.type((JavaType) laneInfo.shape.typeElement()).space().identifier(c)),
                                    _ -> { // We create a record to return
                                        jc.record("Impl", _ -> jc.blockInlineComment("implement args"));
                                        jc.returnKeyword(_ -> jc.reserved("new").space().call("Impl"));
                                    }
                            );

                            List.of(
                                    new Op("add", "+"),
                                    new Op("sub", "-"),
                                    new Op("mul", "*"),
                                    new Op("div", "/")
                            ).forEach(op -> {
                                        jc.func(_ -> jc.type(laneInfo.shape),
                                                "_" + op.name(),    // core ops are named _add(...)
                                                _ -> jc.commaSpaceSeparated(List.of(lhs, rhs), laneInfo.shape.laneNames(),
                                                        (side, laneName) -> jc.type(laneInfo.shape).space().identifier(side + laneName)),
                                                _ -> jc.returnCallResult(laneInfo.name, _ ->
                                                        jc.commaSpaceSeparated(laneInfo.shape.laneNames(), c -> jc.identifier(lhs + c).symbol(op.symbol).identifier(rhs + c))
                                                )
                                        );
                                        jc.func(_ -> jc.type(laneInfo.shape),
                                                op.name(),
                                                _ -> jc.commaSpaceSeparated(List.of(lhs, rhs), c -> jc.identifier(laneInfo.name).space().identifier(c)),
                                                _ -> jc.returnCallResult(op.name, _ ->
                                                                jc.commaSpaceSeparated(laneInfo.shape.laneNames(), List.of(lhs, rhs),
                                                                        (laneName, side) -> jc.identifier(side).dot().call(laneName).ocparen())
                                                        // l.x() or r.y()
                                                )
                                        );
                                        jc.func(_ -> jc.type(laneInfo.shape),
                                                op.name(),
                                                _ -> jc.type((JavaType) laneInfo.shape.typeElement()).space().identifier(lhs).commaSpace().identifier(laneInfo.name).space().identifier(rhs),
                                                _ -> jc.returnCallResult(op.name, _ ->
                                                        jc.commaSpaceSeparated(laneInfo.shape.laneNames(), List.of(lhs, rhs),
                                                                (laneName, side) -> jc.identifier(side).dot().call(laneName))  // l.x() or r.y()
                                                )
                                        );

                                    }
                            );

                            List.of(
                                    new UnaryFunc("neg", false),
                                    new UnaryFunc("sin", false),
                                    new UnaryFunc("cos", false),
                                    new UnaryFunc("tan", false),
                                    new UnaryFunc("sqrt", false),
                                    new UnaryFunc("invsqrt", false),
                                    new UnaryFunc("length", true)
                            ).forEach(unaryFunc -> jc.
                                    func(_ -> jc.type(laneInfo.shape), // type
                                            unaryFunc.name(),               // name
                                            _ -> jc.identifier(laneInfo.name).space().identifier("v"),
                                            _ -> {
                                                if (unaryFunc.returnsScalar()) {
                                                    jc.blockInlineComment("scalar")
                                                            .returnKeyword(_ -> jc.identifier(laneInfo.name)
                                                            .paren(_ ->
                                                                    jc.commaSpaceSeparated(laneInfo.shape.laneNames(), laneName ->
                                                                            jc.dotted("F32", unaryFunc.name).paren(_ ->
                                                                                    jc.dotted("v", laneName).ocparen()
                                                                            )
                                                                    )
                                                            )
                                                    );
                                                } else {
                                                    jc.blockInlineComment("vec")
                                                            .returnKeyword(_ -> jc.identifier(laneInfo.name).paren(_ ->
                                                                    jc.commaSpaceSeparated(laneInfo.shape.laneNames(), laneName ->
                                                                            jc.dotted("F32", unaryFunc.name).paren(_ ->
                                                                                    jc.dotted("v", laneName).ocparen())
                                                                    )
                                                            )
                                                    );
                                                }
                                            }
                                    )
                            );
                        });
            });
            System.out.println(jc);
        });
    }
}
