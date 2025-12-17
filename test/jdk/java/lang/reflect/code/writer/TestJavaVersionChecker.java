import jdk.incubator.code.Op;
import jdk.incubator.code.Reflect;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

import java.io.IOException;
import java.lang.classfile.*;
import java.lang.classfile.instruction.ConstantInstruction;
import java.lang.classfile.instruction.InvokeInstruction;
import java.lang.reflect.Method;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Optional;

/*
 * @test
 * @summary Test that java version check we do in op building methods is working
 * @run main TestJavaVersionChecker
 * @run junit/othervm TestJavaVersionChecker
 */
public class TestJavaVersionChecker {

    public static void main(String[] args) throws IOException { // transform $CM classfile
        String testClassName = TestJavaVersionChecker.class.getName();
        Path testClassesDir = Path.of(System.getProperty("test.classes"));
        Path innerClassPath = testClassesDir.resolve(testClassName + "$$CM.class");
        byte[] newInnerBytes = changeCompileTimeVersion(innerClassPath, Runtime.version().feature() - 1);
        Files.write(innerClassPath, newInnerBytes);
    }

    @Test
    void test() throws ReflectiveOperationException, IOException {
        Method m = this.getClass().getDeclaredMethod("max", int.class, int.class);
        Assertions.assertThrows(UnsupportedOperationException.class, () -> Op.ofMethod(m));
    }

    // change java compile time version that was embedded in the $checkJavaVersion method
    private static byte[] changeCompileTimeVersion(Path innerClassPath, int newCompileTimeVersion) {
        ClassModel inner = null;
        try {
            inner = ClassFile.of().parse(innerClassPath);
        } catch (IOException e) {
            Assertions.fail("Inner class holding the code model doesn't exist");
        }

        Optional<MethodModel> optional = inner.methods().stream().filter(m -> m.methodName().equalsString("$checkJavaVersion")).findFirst();
        Assertions.assertTrue(optional.isPresent(), "Helper method that checks the java version doesn't exist");
        MethodModel checkerMethod = optional.get();

        for (MethodModel m : inner.methods()) {
            if (m.methodName().stringValue().startsWith("$")) { // not model building method
                continue;
            }
            Assertions.assertTrue(m.code().get().elementList().getFirst() instanceof InvokeInstruction i &&
                            i.method().owner().asSymbol().equals(inner.thisClass().asSymbol()) &&
                            i.name().equals(checkerMethod.methodName()) &&
                            i.type().equals(checkerMethod.methodType()),
                    "model building method doesn't check Java version at the start");
        }

        // @@@ we may want to change the compile time version embedded in the error message ?
        CodeTransform codeTransform = (codeBuilder, codeElement) -> {
            if (codeElement instanceof ConstantInstruction.ArgumentConstantInstruction a) {
                codeBuilder.bipush(newCompileTimeVersion);
            } else {
                codeBuilder.with(codeElement);
            }
        };
        ClassTransform classTransform = (classBuilder, classElement) -> {
            if (classElement instanceof MethodModel m && m.equals(checkerMethod)) {
                classBuilder.transformMethod(m, MethodTransform.transformingCode(codeTransform));
            } else {
                classBuilder.with(classElement);
            }
        };

        return ClassFile.of(ClassFile.ConstantPoolSharingOption.NEW_POOL).transformClass(inner, classTransform);
    }

    @Reflect
    static int max(int a, int b) {
        return Math.max(a, b);
    }
}
