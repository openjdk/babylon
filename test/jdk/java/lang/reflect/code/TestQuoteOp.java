import jdk.incubator.code.CodeReflection;
import jdk.incubator.code.Op;
import jdk.incubator.code.op.CoreOp;
import org.testng.annotations.Test;

import java.lang.reflect.Method;
import java.util.Optional;

/*
 * @test
 * @modules jdk.incubator.code
 * @run testng TestQuoteOp
 */
public class TestQuoteOp {

    @CodeReflection
    public void f(int i) {
        String s = "abc";
        Runnable r = () -> {
            System.out.println(i + s + hashCode());
        };
    }

    @Test
    void test() throws NoSuchMethodException {
        Method f = getClass().getDeclaredMethod("f", int.class);
        CoreOp.FuncOp fm = Op.ofMethod(f).orElseThrow();
        Op lop = fm.body().entryBlock().ops().stream().filter(op -> op instanceof CoreOp.LambdaOp).findFirst().orElseThrow();

        fm.writeTo(System.out);
        CoreOp.FuncOp funcOp = CoreOp.quoteOp(lop);
        funcOp.writeTo(System.out);
    }
}
