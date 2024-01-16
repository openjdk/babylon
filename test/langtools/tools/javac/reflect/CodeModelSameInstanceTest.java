import org.testng.Assert;
import org.testng.annotations.Test;

import java.lang.reflect.Method;
import java.lang.runtime.CodeReflection;
import java.util.Arrays;
import java.util.Optional;
import java.util.stream.IntStream;

/*
 * @test
 * @summary test that invoking Method::getCodeModel returns the same instance.
 * @run testng CodeModelSameInstanceTest
 */
public class CodeModelSameInstanceTest {

    @CodeReflection
    static int add(int a, int b) {
        return a + b;
    }

    @Test
    void test() {
        Optional<Method> om = Arrays.stream(this.getClass().getDeclaredMethods()).filter(m -> m.getName().equals("add"))
                .findFirst();
        Method m = om.get();
        Object[] codeModels = IntStream.range(0, 1024).mapToObj(_ -> m.getCodeModel()).toArray();
        for (int i = 1; i < codeModels.length; i++) {
            Assert.assertSame(codeModels[i], codeModels[i-1]);
        }
    }
}
