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
 * @run junit TestFloatingPointComparison
 */
public class TestFloatingPointComparison {
    // float
    @Reflect
    static boolean gt(float f, float g) {
        return f > g;
    }

    @Reflect
    static boolean gte(float f, float g) {
        return f >= g;
    }

    @Reflect
    static boolean eq(float f, float g) {
        return f == g;
    }

    @Reflect
    static boolean lt(float f, float g) {
        return f < g;
    }

    @Reflect
    static boolean lte(float f, float g) {
        return f <= g;
    }

    // double
    @Reflect
    static boolean gt(double f, double g) {
        return f > g;
    }

    @Reflect
    static boolean gte(double f, double g) {
        return f >= g;
    }

    @Reflect
    static boolean eq(double f, double g) {
        return f == g;
    }

    @Reflect
    static boolean lt(double f, double g) {
        return f < g;
    }

    @Reflect
    static boolean lte(double f, double g) {
        return f <= g;
    }

    @ParameterizedTest
    @MethodSource("cases")
    void test(Method m) throws Throwable {
        CoreOp.FuncOp op = Op.ofMethod(m).get();
        MethodHandle mh = BytecodeGenerator.generate(MethodHandles.lookup(), op);

        Number nan;
        if (m.getParameterTypes()[0].equals(float.class)) {
            nan = Float.NaN;
        } else {
            nan = Double.NaN;
        }
        Number[] args = {nan, Float.NEGATIVE_INFINITY, Float.POSITIVE_INFINITY, +0, -0, 1};
        for (Number a : args) {
            for (Number b : args) {
                Object expected = m.invoke(null, a, b);
                Object actual = mh.invoke(a, b);
                Assertions.assertEquals(expected, actual);
            }
        }
    }

    static Stream<Method> cases() {
        return Arrays.stream(TestFloatingPointComparison.class.getDeclaredMethods())
                .filter(m -> m.isAnnotationPresent(Reflect.class));
    }
}
