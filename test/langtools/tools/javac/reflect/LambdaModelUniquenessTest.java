import jdk.incubator.code.Op;
import jdk.incubator.code.Quotable;
import jdk.incubator.code.Quoted;
import org.testng.Assert;
import org.testng.annotations.Test;

import java.util.function.Function;
import java.util.function.IntUnaryOperator;

/*
 * @test
 * @modules jdk.incubator.code
 * @run testng LambdaModelUniquenessTest
 */
public class LambdaModelUniquenessTest {

    Quotable f() {
        return (Runnable & Quotable) () -> {
            System.out.println("Running...");
        };
    }

    @Test
    void testWithLambdaThatDoesNotCapture() {
        Quotable q1 = f();
        Quotable q2 = f();
        Quoted quoted1 = Op.ofQuotable(q1).orElseThrow();
        Quoted quoted2 = Op.ofQuotable(q2).orElseThrow();
        Assert.assertSame(quoted1.op(), quoted2.op());
    }

    static Quotable g(int i) {
        return (IntUnaryOperator & Quotable) j -> j + i;
    }

    @Test
    void testWithLambdaThatCapture() {
        Quotable q1 = g(1);
        Quotable q2 = g(2);
        Quoted quoted1 = Op.ofQuotable(q1).orElseThrow();
        Quoted quoted2 = Op.ofQuotable(q2).orElseThrow();
        Assert.assertSame(quoted1.op(), quoted2.op());
    }
}
