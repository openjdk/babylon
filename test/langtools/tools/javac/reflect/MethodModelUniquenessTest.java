import jdk.incubator.code.CodeReflection;
import jdk.incubator.code.Op;
import jdk.incubator.code.dialect.core.CoreOp;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

import java.lang.reflect.Method;

/*
 * @test
 * @modules jdk.incubator.code
 * @run junit MethodModelUniquenessTest
 */

public class MethodModelUniquenessTest {

    @CodeReflection
    static void f() {
    }
    @CodeReflection
    static void g() {
    }

    @Test
    public void testInstancesReflectSameMethodHaveSameModel() throws NoSuchMethodException {
        Method f = this.getClass().getDeclaredMethod("f");
        Method f2 = this.getClass().getDeclaredMethod("f");
        CoreOp.FuncOp fm = Op.ofMethod(f).orElseThrow();
        CoreOp.FuncOp fm2 = Op.ofMethod(f2).orElseThrow();
        Assertions.assertSame(fm, fm2);
    }

    @Test
    public void testInstancesReflectDiffMethodsHaveDiffModels() throws NoSuchMethodException {
        Method f = this.getClass().getDeclaredMethod("f");
        CoreOp.FuncOp fm = Op.ofMethod(f).orElseThrow();

        Method g = this.getClass().getDeclaredMethod("g");
        CoreOp.FuncOp gm = Op.ofMethod(g).orElseThrow();

        Assertions.assertNotSame(gm, fm);
    }
}
