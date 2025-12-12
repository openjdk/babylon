import jdk.incubator.code.Op;
import jdk.incubator.code.Reflect;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

import java.io.IOException;
import java.lang.classfile.*;
import java.lang.classfile.instruction.ConstantInstruction;
import java.lang.classfile.instruction.InvokeInstruction;
import java.lang.reflect.Method;
import java.nio.file.FileSystems;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.HashMap;
import java.util.Map;
import java.util.Optional;

/*
 * @test
 * @summary Test that java version check we do in op building methods is working
 * @run junit TestJavaVersionChecker
 */
public class TestJavaVersionChecker {
    @Reflect
    static int max(int a, int b) {
        return Math.max(a, b);
    }

    @Test
    void test() throws ReflectiveOperationException, IOException {
        String testClassName = this.getClass().getName();
        String innerClassName = testClassName + "$$CM";
        Path testClassfilesDir = Path.of("/Users/mabbay/babylon/JTwork/classes/0/java/lang/reflect/code/writer/" +
                testClassName + ".d");
        Path innerClassPath = FileSystems.getDefault().getPath(testClassfilesDir.toString(), innerClassName + ".class");
        Path testClassPath = FileSystems.getDefault().getPath(testClassfilesDir.toString(), testClassName + ".class");

        byte[] newInner = changeCompileTimeVersion(innerClassPath, Runtime.version().feature() - 1);

        // we must choose a parent loader incapable of loading this test class and its inner class
        // so that we fall back to BytecodeLoader#findClass
        ByteClassLoader byteClassLoader = new ByteClassLoader(ClassLoader.getSystemClassLoader());
        byteClassLoader.registerClass(innerClassName, newInner);
        byteClassLoader.registerClass(testClassName, Files.readAllBytes(testClassPath));

        Class<?> testClass = byteClassLoader.loadClass(this.getClass().getName());
        // using the testClass instance will cause the inner class $CM to be loaded by bytecodeLoader
        Method m = testClass.getDeclaredMethod("max", int.class, int.class);
        Assertions.assertThrows(RuntimeException.class, () -> Op.ofMethod(m));
    }

    public static class ByteClassLoader extends ClassLoader {
        private final Map<String, byte[]> classesBytes = new HashMap<>();

        public ByteClassLoader(ClassLoader parent) {
            super(parent);
        }

        public void registerClass(String className, byte[] classBytes) {
            classesBytes.put(className, classBytes);
        }

        @Override
        protected Class<?> findClass(String className) throws ClassNotFoundException {
            System.out.println("findClass " + className);
            byte[] bytes = classesBytes.get(className);
            if (bytes == null)
                throw new ClassNotFoundException(className + " was not registered in " + this.getClass().getName());

            return defineClass(className, bytes, 0, bytes.length);
        }
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
}
