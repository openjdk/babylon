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
    @CodeReflection
    static void g() {
    }

    @Test
    void testInstancesReflectSameMethodHaveSameModel() throws NoSuchMethodException {
        Method f = this.getClass().getDeclaredMethod("f");
        Method f2 = this.getClass().getDeclaredMethod("f");
        CoreOp.FuncOp fm = Op.ofMethod(f).orElseThrow();
        CoreOp.FuncOp fm2 = Op.ofMethod(f2).orElseThrow();
        Assert.assertSame(fm, fm2);
    }

    @Test
    void testInstancesReflectDiffMethodsHaveDiffModels() throws NoSuchMethodException {
        Method f = this.getClass().getDeclaredMethod("f");
        CoreOp.FuncOp fm = Op.ofMethod(f).orElseThrow();

        Method g = this.getClass().getDeclaredMethod("g");
        CoreOp.FuncOp gm = Op.ofMethod(g).orElseThrow();

        Assert.assertNotSame(gm, fm);
    }
}
