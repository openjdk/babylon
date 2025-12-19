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
        // in the lambda class initializer <clinit>, we invoke lambda op method
        // after the changes we made to $CM classfile, the lambda op method throws UOE, causing <clinit> to fails
        // UOE -> ExceptionInInitializerError -> InternalError
        InternalError ie = null;
        try {
            IntBinaryOperator l = (@Reflect IntBinaryOperator) (a, b) -> Math.max(a, b);
        } catch (InternalError e) {
            Assertions.assertInstanceOf(ExceptionInInitializerError.class, e.getCause());
            Assertions.assertInstanceOf(UnsupportedOperationException.class, e.getCause().getCause());
            ie = e;
        }
        Assertions.assertNotNull(ie, "Reflectable lambda creation didn't fail as expected");
    }
}
