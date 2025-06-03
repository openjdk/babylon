import jdk.incubator.code.CodeReflection;

/*
 * @test
 * @modules jdk.incubator.code
 * @build StringConcatTest
 * @build CodeReflectionTester
 * @run main CodeReflectionTester StringConcatTest
 */
public class StringConcatTest {

    @IR("""
            func @"test1" (%0 : java.type:"java.lang.String", %1 : java.type:"int")java.type:"java.lang.String" -> {
                %2 : Var<java.type:"java.lang.String"> = var %0 @"a";
                %3 : Var<java.type:"int"> = var %1 @"b";
                %4 : java.type:"java.lang.String" = var.load %2;
                %5 : java.type:"int" = var.load %3;
                %6 : java.type:"java.lang.String" = concat %4 %5;
                return %6;
            };
            """)
    @CodeReflection
    static String test1(String a, int b) {
        return a + b;
    }

    @IR("""
            func @"test2" (%0 : java.type:"java.lang.String", %1 : java.type:"char")java.type:"java.lang.String" -> {
                %2 : Var<java.type:"java.lang.String"> = var %0 @"a";
                %3 : Var<java.type:"char"> = var %1 @"b";
                %4 : java.type:"java.lang.String" = var.load %2;
                %5 : java.type:"char" = var.load %3;
                %6 : java.type:"java.lang.String" = concat %4 %5;
                var.store %2 %6;
                %7 : java.type:"java.lang.String" = var.load %2;
                return %7;
            };
            """)
    @CodeReflection
    static String test2(String a, char b) {
        a += b;
        return a;
    }

    @IR("""
            func @"test3" (%0 : java.type:"java.lang.String", %1 : java.type:"float")java.type:"java.lang.String" -> {
                %2 : Var<java.type:"java.lang.String"> = var %0 @"a";
                %3 : Var<java.type:"float"> = var %1 @"b";
                %4 : java.type:"java.lang.String" = var.load %2;
                %5 : java.type:"float" = var.load %3;
                %6 : java.type:"java.lang.String" = concat %4 %5;
                var.store %2 %6;
                %7 : java.type:"java.lang.String" = var.load %2;
                return %7;
            };
            """)
    @CodeReflection
    static String test3(String a, float b) {
        a = a + b;
        return a;
    }
}
