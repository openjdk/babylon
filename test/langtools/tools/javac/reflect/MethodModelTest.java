import jdk.incubator.code.CodeReflection;
import jdk.incubator.code.Op;
import jdk.incubator.code.op.CoreOp;
import org.testng.Assert;
import org.testng.annotations.Test;

import java.lang.reflect.Method;

/*
 * @test
 * @modules jdk.incubator.code
 * @run testng MethodModelTest
 */

public class MethodModelTest {

    @CodeReflection
    static void f() {
    }

    @Test
    void test() throws NoSuchMethodException {
        Method f = this.getClass().getDeclaredMethod("f");
        Method f2 = this.getClass().getDeclaredMethod("f");
        CoreOp.FuncOp funcOp = Op.ofMethod(f).orElseThrow();
        CoreOp.FuncOp funcOp2 = Op.ofMethod(f2).orElseThrow();
        Assert.assertSame(funcOp, funcOp2);
    }

    // different Method objects that rep the same java method, have different code models
    // code model not unique accross multiple Method objects that rep the same thing
}
