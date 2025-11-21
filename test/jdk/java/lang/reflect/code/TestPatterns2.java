import jdk.incubator.code.Reflect;
import jdk.incubator.code.CodeTransformer;
import jdk.incubator.code.Op;
import jdk.incubator.code.dialect.core.CoreOp;
import jdk.incubator.code.interpreter.Interpreter;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

import java.lang.invoke.MethodHandles;
import java.lang.reflect.Method;
import java.util.Optional;
import java.util.stream.Stream;

/*
 * @test
 * @modules jdk.incubator.code
 * @run junit TestPatterns2
 * @enablePreview */
public class TestPatterns2 {

    record R<T extends Number> (T n) {}

    @Reflect
    static boolean f(Object o) {
        return o instanceof R(Integer i);
    }

    @Test
    void test() {

        CoreOp.FuncOp f = getFuncOp("f");
        System.out.println(f.toText());

        CoreOp.FuncOp lf = f.transform(CodeTransformer.LOWERING_TRANSFORMER);
        System.out.println(lf.toText());

        R[] args = {new R(1), new R(2d)};
        for (R arg : args) {
            Assertions.assertEquals(f(arg), Interpreter.invoke(MethodHandles.lookup(), lf, arg));
        }
    }

    static CoreOp.FuncOp getFuncOp(String name) {
        Optional<Method> om = Stream.of(TestPatterns2.class.getDeclaredMethods())
                .filter(m -> m.getName().equals(name))
                .findFirst();

        Method m = om.get();
        return Op.ofMethod(m).get();
    }
}
