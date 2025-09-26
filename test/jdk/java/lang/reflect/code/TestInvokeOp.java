import jdk.incubator.code.CodeReflection;
import jdk.incubator.code.Op;
import jdk.incubator.code.dialect.core.CoreOp;
import jdk.incubator.code.dialect.java.JavaOp;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

import java.lang.reflect.Method;
import java.util.ArrayList;
import java.util.Optional;
import java.util.stream.Stream;

/*
 * @test
 * @modules jdk.incubator.code
 * @run junit TestInvokeOp
 */
public class TestInvokeOp {

    @Test
    void test() {
        var f = getFuncOp(this.getClass(), "f");
        var invokeOps = f.elements().filter(ce -> ce instanceof JavaOp.InvokeOp).map(ce -> ((JavaOp.InvokeOp) ce)).toList();

        Assertions.assertEquals(invokeOps.get(0).operands(), invokeOps.get(0).argOperands());

        Assertions.assertEquals(invokeOps.get(1).operands().subList(0, 1), invokeOps.get(1).argOperands());

        Assertions.assertEquals(invokeOps.get(2).operands(), invokeOps.get(2).argOperands());

        Assertions.assertEquals(invokeOps.get(3).operands().subList(0, 1), invokeOps.get(3).argOperands());

        for (JavaOp.InvokeOp invokeOp : invokeOps) {
            var l = new ArrayList<>(invokeOp.argOperands());
            if (invokeOp.isVarArgs()) {
                l.addAll(invokeOp.varArgOperands());
            }
            Assertions.assertEquals(invokeOp.operands(), l);
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
