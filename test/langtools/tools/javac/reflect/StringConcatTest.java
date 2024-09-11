import java.lang.runtime.CodeReflection;

/*
 * @test
 * @build StringConcatTest
 * @build CodeReflectionTester
 * @run main CodeReflectionTester StringConcatTest
 */
public class StringConcatTest {

    @IR("""
            func @"test1" (%0 : java.lang.String, %1 : int)java.lang.String -> {
                %2 : Var<java.lang.String> = var %0 @"a";
                %3 : Var<int> = var %1 @"b";
                %4 : java.lang.String = var.load %2;
                %5 : int = var.load %3;
                %6 : java.lang.String = concat %4 %5;
                return %6;
            };
            """)
    @CodeReflection
    static String test1(String a, int b) {
        return a + b;
    }

    @IR("""
            func @"test2" (%0 : java.lang.String, %1 : char)java.lang.String -> {
                %2 : Var<java.lang.String> = var %0 @"a";
                %3 : Var<char> = var %1 @"b";
                %4 : java.lang.String = var.load %2;
                %5 : char = var.load %3;
                %6 : java.lang.String = concat %4 %5;
                var.store %2 %6;
                %7 : java.lang.String = var.load %2;
                return %7;
            };
            """)
    @CodeReflection
    static String test2(String a, char b) {
        a += b;
        return a;
    }

    @IR("""
            func @"test3" (%0 : java.lang.String, %1 : float)java.lang.String -> {
                %2 : Var<java.lang.String> = var %0 @"a";
                %3 : Var<float> = var %1 @"b";
                %4 : java.lang.String = var.load %2;
                %5 : float = var.load %3;
                %6 : java.lang.String = concat %4 %5;
                var.store %2 %6;
                %7 : java.lang.String = var.load %2;
                return %7;
            };
            """)
    @CodeReflection
    static String test3(String a, float b) {
        a = a + b;
        return a;
    }
}
