import jdk.incubator.code.Op;

import java.lang.reflect.Method;
import jdk.incubator.code.CodeReflection;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

import java.util.Arrays;
import java.util.Optional;
import java.util.stream.IntStream;

/*
 * @test
 * @summary test that invoking Method::getCodeModel returns the same instance.
 * @modules jdk.incubator.code
 * @run junit CodeModelSameInstanceTest
 */
public class CodeModelSameInstanceTest {

    @CodeReflection
    static int add(int a, int b) {
        return a + b;
    }

    @Test
    public void test() {
        Optional<Method> om = Arrays.stream(this.getClass().getDeclaredMethods()).filter(m -> m.getName().equals("add"))
                .findFirst();
        Method m = om.get();
        Object[] codeModels = IntStream.range(0, 1024).mapToObj(_ -> Op.ofMethod(m)).toArray();
        for (int i = 1; i < codeModels.length; i++) {
            Assertions.assertSame(codeModels[i-1], codeModels[i]);
        }
    }
}
