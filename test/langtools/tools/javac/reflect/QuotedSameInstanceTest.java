import jdk.incubator.code.CodeReflection;
import jdk.incubator.code.Op;
import jdk.incubator.code.interpreter.Interpreter;

import jdk.incubator.code.Quotable;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

import java.lang.invoke.MethodHandles;
import java.util.function.IntSupplier;
import java.util.function.IntUnaryOperator;
import java.util.stream.IntStream;

/*
 * @test
 * @summary test that invoking Op#ofQuotable returns the same instance
 * @modules jdk.incubator.code
 * @run junit QuotedSameInstanceTest
 */

public class QuotedSameInstanceTest {

    private static final Quotable q1 = (Quotable & Runnable) () -> {
    };

    @Test
    public void testWithOneThread() {
        Assertions.assertSame(Op.ofQuotable(q1).get(), Op.ofQuotable(q1).get());
    }

    interface QuotableIntUnaryOperator extends IntUnaryOperator, Quotable { }
    private static final QuotableIntUnaryOperator q2 = x -> x;

    @Test
    public void testWithMultiThreads() {
        Object[] quotedObjects = IntStream.range(0, 1024).parallel().mapToObj(__ -> Op.ofQuotable(q2).get()).toArray();
        for (int i = 1; i < quotedObjects.length; i++) {
            Assertions.assertSame(quotedObjects[i], quotedObjects[i - 1]);
        }
    }

    public interface QuotableIntSupplier extends IntSupplier, Quotable {}
    @CodeReflection
    static Quotable q() {
        QuotableIntSupplier r = () -> 8;
        return r;
    }

    @Test
    public void testMultiThreadsViaInterpreter() throws NoSuchMethodException {
        var qm = this.getClass().getDeclaredMethod("q");
        var q = Op.ofMethod(qm).get();
        Quotable quotable = (Quotable) Interpreter.invoke(MethodHandles.lookup(), q);
        Object[] quotedObjects = IntStream.range(0, 1024).parallel().mapToObj(__ -> Op.ofQuotable(quotable).get()).toArray();
        for (int i = 1; i < quotedObjects.length; i++) {
            Assertions.assertSame(quotedObjects[i], quotedObjects[i - 1]);
        }
    }
}
