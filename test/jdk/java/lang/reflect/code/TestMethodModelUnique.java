import jdk.incubator.code.Op;
import jdk.incubator.code.Reflect;
import jdk.incubator.code.dialect.core.CoreOp;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

import java.lang.reflect.Method;
import java.util.List;
import java.util.Optional;
import java.util.stream.IntStream;

/*
 * @test
 * @modules jdk.incubator.code
 * @run junit TestMethodModelUnique
 */
public class TestMethodModelUnique {

    @Reflect
    static void f() {
    }
    @Reflect
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

    @Test
    public void testOpOfMethodIsThreadSafe() throws NoSuchMethodException {
        Method f = this.getClass().getDeclaredMethod("f");
        List<Optional<CoreOp.FuncOp>> fops = IntStream.range(1, 3).parallel().mapToObj(_ -> Op.ofMethod(f)).toList();
        Assertions.assertSame(fops.getFirst(), fops.getLast());
    }
}
