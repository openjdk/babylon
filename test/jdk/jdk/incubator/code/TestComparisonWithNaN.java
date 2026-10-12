import jdk.incubator.code.Op;
import jdk.incubator.code.Reflect;
import jdk.incubator.code.bytecode.BytecodeGenerator;
import jdk.incubator.code.dialect.core.CoreOp;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;

import java.lang.invoke.MethodHandle;
import java.lang.invoke.MethodHandles;
import java.lang.reflect.Method;
import java.util.Arrays;
import java.util.stream.Stream;

/*
 * @test
 * @modules jdk.incubator.code
 * @library lib
 * @run junit TestComparisonWithNaN
 */
public class TestComparisonWithNaN {

    @Reflect
    static boolean gtNaN(float f) {
        return f > Float.NaN;
    }

    @Reflect
    static boolean gteNaN(float f) {
        return f >= Float.NaN;
    }

    @Reflect
    static boolean gtNaN(double d) {
        return d > Double.NaN;
    }

    @Reflect
    static boolean gteNaN(double d) {
        return d >= Double.NaN;
    }

    @ParameterizedTest
    @MethodSource("cases")
    void test(Method m) throws Throwable {
        CoreOp.FuncOp op = Op.ofMethod(m).get();
        MethodHandle mh = BytecodeGenerator.generate(MethodHandles.lookup(), op);
        int arg = 1;
        boolean expected = false;
        Assertions.assertEquals(expected, Interpreter.invoke(MethodHandles.lookup(), op, arg));
        Assertions.assertEquals(expected, mh.invoke(arg));
    }

    static Stream<Method> cases() {
        return Arrays.stream(TestComparisonWithNaN.class.getDeclaredMethods())
                .filter(m -> m.isAnnotationPresent(Reflect.class));
    }
}
