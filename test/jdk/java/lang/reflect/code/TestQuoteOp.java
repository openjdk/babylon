import jdk.incubator.code.CodeReflection;
import jdk.incubator.code.Op;
import jdk.incubator.code.Quoted;
import jdk.incubator.code.op.CoreOp;
import org.testng.Assert;
import org.testng.annotations.Test;

import java.lang.reflect.Method;

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

        Object[] args = {1, "s", this};
        Quoted q = CoreOp.quotedOp(funcOp, args);

        Assert.assertTrue(lop.getClass().isInstance(q.op()));

        // q.op() must have the same structure as lop
        // for the moment, we don't have utility to check that

        Assert.assertEquals(args, q.capturedValues().values().toArray());

        Assert.assertTrue(q.operands().isEmpty());
    }

    @CodeReflection
    static void g(String s) {
        boolean b = s.startsWith("a");
    }

    @Test
    void testQuoteOpThatHasOperands() throws NoSuchMethodException { // op with operands
        Method g = getClass().getDeclaredMethod("g", String.class);
        CoreOp.FuncOp gm = Op.ofMethod(g).orElseThrow();
        Op op = gm.body().entryBlock().ops().stream().filter(o -> o instanceof CoreOp.InvokeOp).findFirst().orElseThrow();

        gm.writeTo(System.out);

        CoreOp.FuncOp funcOp = CoreOp.quoteOp(op);
        funcOp.writeTo(System.out);

        Object[] args = {"str", "s"};
        Quoted q = CoreOp.quotedOp(funcOp, args);

        Assert.assertTrue(op.getClass().isInstance(q.op()));

        Assert.assertTrue(q.capturedValues().isEmpty());

        Assert.assertEquals(args, q.operands().values().toArray());
    }
}
