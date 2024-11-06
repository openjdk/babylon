import org.testng.Assert;
import org.testng.annotations.Test;

import java.lang.classfile.ClassFile;
import java.lang.classfile.components.ClassPrinter;
import java.lang.invoke.MethodHandle;
import java.lang.invoke.MethodHandles;
import java.lang.reflect.Method;
import java.lang.reflect.code.OpTransformer;
import java.lang.reflect.code.bytecode.BytecodeGenerator;
import java.lang.reflect.code.op.CoreOp;
import java.lang.runtime.CodeReflection;
import java.util.Arrays;
import java.util.Optional;
import java.util.stream.Stream;

/*
 * @test
 * @enablePreview
 * @run testng TestVarArg
 *
 */
public class TestVarArg {

    @Test
    void test() throws Throwable {
        var f = getFuncOp("f");
        f.writeTo(System.out);

        var lf = f.transform(OpTransformer.LOWERING_TRANSFORMER);
        lf.writeTo(System.out);

        var bytes = BytecodeGenerator.generateClassData(MethodHandles.lookup(), f);
        var classModel = ClassFile.of().parse(bytes);
        ClassPrinter.toYaml(classModel, ClassPrinter.Verbosity.TRACE_ALL, System.out::print);

        MethodHandle mh = BytecodeGenerator.generate(MethodHandles.lookup(), lf);
        Assert.assertEquals(mh.invoke(), f());
    }

    @CodeReflection
    static String f() {
        String r = "";
        String ls = System.lineSeparator();

        r += ls + h(1);
        r += ls + h(2, 3);
        r += ls + h(4, (byte) 5);

        r += ls + k(Byte.MIN_VALUE, Byte.MAX_VALUE);

        r += ls + j("s1", "s2", "s3");

        r += ls + w(8, 9);

        return r;
    }

    static String h(int i, int... s) {
        return i + ", " + Arrays.toString(s);
    }

    static String k(byte... s) {
        return Arrays.toString(s);
    }

    static String j(String i, String... s) {
        return i + ", " + Arrays.toString(s);
    }

    static <T extends Number> String w(T... ts) {
        return Arrays.toString(ts);
    }

    private CoreOp.FuncOp getFuncOp(String name) {
        Optional<Method> om = Stream.of(this.getClass().getDeclaredMethods())
                .filter(m -> m.getName().equals(name))
                .findFirst();

        Method m = om.get();
        return m.getCodeModel().get();
    }
}
