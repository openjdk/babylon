import jdk.incubator.code.Op;
import jdk.incubator.code.Reflect;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.function.IntBinaryOperator;

/*
 * @test
 * @summary Test that java version check we do in op building methods is working
 * @modules jdk.incubator.code
 * @run main TestJavaVersionCheckerForLambdas
 * @run junit/othervm TestJavaVersionCheckerForLambdas
 */
public class TestJavaVersionCheckerForLambdas {

    public static void main(String[] args) throws IOException { // transform $CM classfile
        String testClassName = TestJavaVersionCheckerForLambdas.class.getName();
        Path testClassesDir = Path.of(System.getProperty("test.classes"));
        Path innerClassPath = testClassesDir.resolve(testClassName + "$$CM.class");
        byte[] newInnerBytes = TestJavaVersionCheckerForMethods.changeCompileTimeVersion(innerClassPath, Runtime.version().feature() - 1);
        Files.write(innerClassPath, newInnerBytes);
    }

    @Test
    void test() throws ReflectiveOperationException, IOException {
        IntBinaryOperator l = (@Reflect IntBinaryOperator) (a, b) -> Math.max(a, b);
        Assertions.assertThrows(UnsupportedOperationException.class, () -> Op.ofLambda(l));
    }
}
