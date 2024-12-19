import jdk.incubator.code.CodeReflection;
import jdk.incubator.code.Op;
import jdk.incubator.code.op.CoreOp;
import org.testng.Assert;
import org.testng.annotations.Test;

import java.lang.reflect.Method;
import java.util.ArrayList;
import java.util.Optional;
import java.util.stream.Stream;

/*
 * @test
 * @modules jdk.incubator.code
 * @run testng TestInvokeOp
 */
public class TestInvokeOp {

    @Test
    void test() {
        var f = getFuncOp(this.getClass(), "f");
        var invokeOps = f.elements().filter(ce -> ce instanceof CoreOp.InvokeOp).map(ce -> ((CoreOp.InvokeOp) ce)).toList();

        Assert.assertEquals(invokeOps.get(0).argOperands(), invokeOps.get(0).operands());

        Assert.assertEquals(invokeOps.get(1).argOperands(), invokeOps.get(1).operands().subList(0, 1));

        Assert.assertEquals(invokeOps.get(2).argOperands(), invokeOps.get(2).operands());

        Assert.assertEquals(invokeOps.get(3).argOperands(), invokeOps.get(3).operands().subList(0, 1));

        for (CoreOp.InvokeOp invokeOp : invokeOps) {
            var l = new ArrayList<>(invokeOp.argOperands());
            if (invokeOp.isVarArgs()) {
                l.addAll(invokeOp.varArgOperands());
            }
            Assert.assertEquals(l, invokeOp.operands());
        }
    }

    @CodeReflection
    void f() {
        s(1);
        s(4, 2, 3);
        i();
        i(0.0, 0.0);
    }

    static void s(int a, long... l) {}
    void i(double... d) {}

    static CoreOp.FuncOp getFuncOp(Class<?> c, String name) {
        Optional<Method> om = Stream.of(c.getDeclaredMethods())
                .filter(m -> m.getName().equals(name))
                .findFirst();

        Method m = om.get();
        return Op.ofMethod(m).get();
    }
}
