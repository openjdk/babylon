import jdk.incubator.code.CodeReflection;

/*
 * @test
 * @modules jdk.incubator.code
 * @build PatternTest2
 * @build CodeReflectionTester
 * @run main CodeReflectionTester PatternTest2
 */
public class PatternTest2 {
    record R<T extends Number> (T n) {}

    @IR("""
            func @"f" (%0 : java.type:"java.lang.Object")java.type:"boolean" -> {
                %1 : Var<java.type:"java.lang.Object"> = var %0 @"o";
                %2 : java.type:"java.lang.Object" = var.load %1;
                %3 : java.type:"java.lang.Integer" = constant @null;
                %4 : Var<java.type:"java.lang.Integer"> = var %3 @"i";
                %5 : java.type:"boolean" = pattern.match %2
                    ()java.type:"jdk.incubator.code.op.ExtendedOp$Pattern$Record<PatternTest2$R<PatternTest2$R::<T extends java.lang.Number>>>" -> {
                        %6 : java.type:"jdk.incubator.code.op.ExtendedOp$Pattern$Type<java.lang.Integer>" = pattern.type @"i";
                        %7 : java.type:"jdk.incubator.code.op.ExtendedOp$Pattern$Record<PatternTest2$R<PatternTest2$R::<T extends java.lang.Number>>>" = pattern.record %6 @"(PatternTest2$R::<T extends java.lang.Number> n)PatternTest2$R<PatternTest2$R::<T extends java.lang.Number>>";
                        yield %7;
                    }
                    (%8 : java.type:"java.lang.Integer")java.type:"void" -> {
                        var.store %4 %8;
                        yield;
                    };
                return %5;
            };
            """)
    @CodeReflection
    static boolean f(Object o) {
        return o instanceof R(Integer i);
    }
}
