import org.testng.Assert;
import org.testng.annotations.Test;

import java.lang.reflect.Method;
import java.lang.reflect.code.OpTransformer;
import java.lang.reflect.code.interpreter.Interpreter;
import java.lang.reflect.code.op.CoreOp;
import java.lang.runtime.CodeReflection;
import java.util.Optional;
import java.util.stream.Stream;

/*
 * @test
 * @run testng TestPatternsWithPrimitives
 * @enablePreview
 */

public class TestPatternsWithPrimitives {

    @CodeReflection
    static boolean f(int a) {
        return a instanceof byte _;
    }

    @Test
    void test() {

        CoreOp.FuncOp f = getFuncOp("f");
        f.writeTo(System.out);

        CoreOp.FuncOp lf = f.transform(OpTransformer.LOWERING_TRANSFORMER);
        lf.writeTo(System.out);

        int[] args = {1, 128, -129};
        for (int a : args) {
            Assert.assertEquals(Interpreter.invoke(lf, a), f(a));
        }
    }

    static CoreOp.FuncOp getFuncOp(String name) {
        Optional<Method> om = Stream.of(TestPatternsWithPrimitives.class.getDeclaredMethods())
                .filter(m -> m.getName().equals(name))
                .findFirst();

        Method m = om.get();
        return m.getCodeModel().get();
    }
}
