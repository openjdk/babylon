import jdk.incubator.code.*;
import jdk.incubator.code.dialect.core.CoreOp;
import jdk.incubator.code.dialect.core.FunctionType;
import jdk.incubator.code.dialect.java.JavaType;
import org.testng.Assert;
import org.testng.annotations.Test;

import java.lang.reflect.Method;
import java.util.ArrayList;
import java.util.List;
import java.util.function.IntUnaryOperator;

/*
 * @test
 * @modules jdk.incubator.code
 * @run testng TestSealOp
 */
public class TestSealOp {

    @CodeReflection
    static List<Integer> f(int i) {
        return new ArrayList<>(i);
    }

    @Test
    void test0() throws NoSuchMethodException {
        Method m = this.getClass().getDeclaredMethod("f", int.class);
        CoreOp.FuncOp f = Op.ofMethod(m).get();
        assertOpIsCopiedWhenAddedToBlock(f);
    }

    @Test
    void test1() {
        Quotable q = (IntUnaryOperator & Quotable) i -> i / 2;
        Quoted quoted = Op.ofQuotable(q).get();
        CoreOp.QuotedOp quotedOp = (CoreOp.QuotedOp) quoted.op().ancestorBody().ancestorOp();
        CoreOp.FuncOp funcOp = (CoreOp.FuncOp) quotedOp.ancestorBody().ancestorOp();
        assertOpIsCopiedWhenAddedToBlock(funcOp);
    }

    @Test
    void test2() {
        CoreOp.ConstantOp constant = CoreOp.constant(JavaType.INT, 7);
        constant.seal();
        assertOpIsCopiedWhenAddedToBlock(constant);
    }

    @Test
    void test3() {
        CoreOp.FuncOp funcOp = CoreOp.func("f", FunctionType.FUNCTION_TYPE_VOID).body(b -> {
            b.op(CoreOp.return_());
        });
        funcOp.seal();
        funcOp.seal();
    }

    @Test
    void test4() {
        Quoted q = (int a, int b) -> {
            return a + b;
        };
        CoreOp.QuotedOp quotedOp = (CoreOp.QuotedOp) q.op().ancestorBody().ancestorOp();
        CoreOp.FuncOp funcOp = (CoreOp.FuncOp) quotedOp.ancestorBody().ancestorOp();
        Assert.assertTrue(funcOp.isSealed());
        assertOpIsCopiedWhenAddedToBlock(funcOp);
    }

    @Test
    void test5() { // freezing an already bound op should throw
        Body.Builder body = Body.Builder.of(null, FunctionType.FUNCTION_TYPE_VOID);
        Op.Result r = body.entryBlock().op(CoreOp.constant(JavaType.DOUBLE, 1d));
        Assert.assertThrows(() -> r.op().seal());
    }

    @Test
    void test6() {
        CoreOp.ConstantOp cop = CoreOp.constant(JavaType.LONG, 1L);
        cop.setLocation(Location.NO_LOCATION);
        cop.seal();
        Assert.assertThrows(() -> cop.setLocation(Location.NO_LOCATION));
    }

    void assertOpIsCopiedWhenAddedToBlock(Op op) {
        Body.Builder body = Body.Builder.of(null, FunctionType.FUNCTION_TYPE_VOID);
        body.entryBlock().op(op);
        body.entryBlock().op(CoreOp.return_());
        CoreOp.FuncOp funcOp = CoreOp.func("t", body);
        boolean b = funcOp.body().entryBlock().ops().stream().allMatch(o -> o != op);
        Assert.assertTrue(b);
    }
}
