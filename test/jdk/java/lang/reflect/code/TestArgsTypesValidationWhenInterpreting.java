import jdk.incubator.code.CodeReflection;
import jdk.incubator.code.Op;
import jdk.incubator.code.dialect.core.CoreOp;
import jdk.incubator.code.interpreter.Interpreter;
import org.testng.Assert;
import org.testng.annotations.Test;

import java.lang.invoke.MethodHandles;
import java.lang.reflect.Method;

/*
 * @test
 * @modules jdk.incubator.code
 * @run testng TestArgsTypesValidationWhenInterpreting
 */
public class TestArgsTypesValidationWhenInterpreting {

    @CodeReflection
    private double f(int value) {
        return Math.pow(value, 2);
    }

    @Test
    void test() throws NoSuchMethodException {
        Method m = this.getClass().getDeclaredMethod("f", int.class);
        CoreOp.FuncOp funcOp = Op.ofMethod(m).get();
        System.out.println(funcOp.toText());

        double res = (double) Interpreter.invoke(MethodHandles.lookup(), funcOp, this, 2);
        Assert.assertEquals(res, 4d);

        res = (double) Interpreter.invoke(MethodHandles.lookup(), funcOp,
                new TestArgsTypesValidationWhenInterpreting(), 2);
        Assert.assertEquals(res, 4d);

        Assert.assertThrows(() -> Interpreter.invoke(MethodHandles.lookup(), funcOp, new Object(), 2));

        Assert.assertThrows(() -> Interpreter.invoke(MethodHandles.lookup(), funcOp, null, 2));

        Assert.assertThrows(() -> Interpreter.invoke(MethodHandles.lookup(), funcOp, this, 2d));

    }
}
