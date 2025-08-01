import jdk.incubator.code.*;
import jdk.incubator.code.dialect.core.CoreOp;
import jdk.incubator.code.dialect.core.FunctionType;
import jdk.incubator.code.dialect.java.JavaType;
import org.testng.Assert;
import org.testng.annotations.Test;

import java.lang.reflect.Method;
import java.util.ArrayList;
import java.util.List;
import java.util.Optional;
import java.util.function.IntSupplier;
import java.util.function.IntUnaryOperator;

/*
 * @test
 * @modules jdk.incubator.code
 * @run testng TestFreezeOp
 */
public class TestFreezeOp {

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
        CoreOp.QuotedOp quotedOp = (CoreOp.QuotedOp) quoted.op().ancestorBody().parentOp();
        CoreOp.FuncOp funcOp = (CoreOp.FuncOp) quotedOp.ancestorBody().parentOp();
        assertOpIsCopiedWhenAddedToBlock(funcOp);
    }

    @Test
    void test2() {
        CoreOp.ConstantOp constant = CoreOp.constant(JavaType.INT, 7);
        constant.freeze();
        assertOpIsCopiedWhenAddedToBlock(constant);
    }

    @Test
    void test3() {
        CoreOp.FuncOp funcOp = CoreOp.func("f", FunctionType.FUNCTION_TYPE_VOID).body(b -> {
            b.op(CoreOp._return());
        });
        funcOp.freeze();
        funcOp.freeze();
    }

    @Test
    void test4() {
        Quoted q = (int a, int b) -> {
            return a + b;
        };
        CoreOp.QuotedOp quotedOp = (CoreOp.QuotedOp) q.op().ancestorBody().parentOp();
        CoreOp.FuncOp funcOp = (CoreOp.FuncOp) quotedOp.ancestorBody().parentOp();
        Assert.assertTrue(funcOp.isFrozen());
        assertOpIsCopiedWhenAddedToBlock(funcOp);
    }

    void assertOpIsCopiedWhenAddedToBlock(Op op) {
        Body.Builder body = Body.Builder.of(null, FunctionType.FUNCTION_TYPE_VOID);
        body.entryBlock().op(op);
        body.entryBlock().op(CoreOp._return());
        CoreOp.FuncOp funcOp = CoreOp.func("t", body);
        boolean b = funcOp.body().entryBlock().ops().stream().allMatch(o -> o != op);
        Assert.assertTrue(b);
    }
}
