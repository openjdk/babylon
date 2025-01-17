import jdk.incubator.code.CodeReflection;
import jdk.incubator.code.Op;
import jdk.incubator.code.interpreter.Interpreter;
import org.testng.Assert;
import org.testng.annotations.Test;

import jdk.incubator.code.Quotable;

import java.lang.invoke.MethodHandles;
import java.util.function.IntSupplier;
import java.util.function.IntUnaryOperator;
import java.util.stream.IntStream;

/*
 * @test
 * @summary test that invoking Op#ofQuotable returns the same instance
 * @modules jdk.incubator.code
 * @run testng QuotedSameInstanceTest
 */

public class QuotedSameInstanceTest {

    private static final Quotable q1 = (Quotable & Runnable) () -> {
    };

    @Test
    void testWithOneThread() {
        Assert.assertSame(Op.ofQuotable(q1).get(), Op.ofQuotable(q1).get());
    }

    interface QuotableIntUnaryOperator extends IntUnaryOperator, Quotable { }
    private static final QuotableIntUnaryOperator q2 = x -> x;

    @Test
    void testWithMultiThreads() {
        Object[] quotedObjects = IntStream.range(0, 1024).parallel().mapToObj(__ -> Op.ofQuotable(q2).get()).toArray();
        for (int i = 1; i < quotedObjects.length; i++) {
            Assert.assertSame(quotedObjects[i], quotedObjects[i - 1]);
        }
    }

    public interface QuotableIntSupplier extends IntSupplier, Quotable {}
    @CodeReflection
    static Quotable q() {
        QuotableIntSupplier r = () -> 8;
        return r;
    }

    @Test
    void testMultiThreadsViaInterpreter() throws NoSuchMethodException {
        var qm = this.getClass().getDeclaredMethod("q");
        var q = Op.ofMethod(qm).get();
        Quotable quotable = (Quotable) Interpreter.invoke(MethodHandles.lookup(), q);
        Object[] quotedObjects = IntStream.range(0, 1024).parallel().mapToObj(__ -> Op.ofQuotable(quotable).get()).toArray();
        for (int i = 1; i < quotedObjects.length; i++) {
            Assert.assertSame(quotedObjects[i], quotedObjects[i - 1]);
        }
    }
}
