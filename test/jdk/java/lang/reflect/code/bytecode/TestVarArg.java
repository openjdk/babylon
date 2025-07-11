import jdk.incubator.code.CodeReflection;
import jdk.incubator.code.Op;
import jdk.incubator.code.OpTransformer;
import jdk.incubator.code.bytecode.BytecodeGenerator;
import jdk.incubator.code.dialect.core.CoreOp;
import org.testng.Assert;
import org.testng.annotations.Test;

import java.lang.classfile.ClassFile;
import jdk.internal.classfile.components.ClassPrinter;
import java.lang.invoke.MethodHandle;
import java.lang.invoke.MethodHandles;
import java.lang.reflect.Method;
import java.util.Arrays;
import java.util.Optional;
import java.util.stream.Stream;

/*
 * @test
 * @enablePreview
 * @modules jdk.incubator.code
 * @modules java.base/jdk.internal.classfile.components
 * @run testng TestVarArg
 *
 */
public class TestVarArg {

    @Test
    void test() throws Throwable {
        var f = getFuncOp("f");
        System.out.println(f.toText());

        var lf = f.transform(OpTransformer.LOWERING_TRANSFORMER);
        System.out.println(lf.toText());

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

        r += k();

        r += l(11L, 12L);

        r += d(21.0, 22.0);

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

    static String l(long... a) {
        return Arrays.toString(a);
    }

    static String d(double... a) {
        return Arrays.toString(a);
    }

    private CoreOp.FuncOp getFuncOp(String name) {
        Optional<Method> om = Stream.of(this.getClass().getDeclaredMethods())
                .filter(m -> m.getName().equals(name))
                .findFirst();

        Method m = om.get();
        return Op.ofMethod(m).get();
    }
}
