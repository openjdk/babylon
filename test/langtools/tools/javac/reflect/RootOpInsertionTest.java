import jdk.incubator.code.Block;
import jdk.incubator.code.Body;
import jdk.incubator.code.CodeReflection;
import jdk.incubator.code.Op;
import jdk.incubator.code.op.CoreOp;
import jdk.incubator.code.type.FunctionType;
import org.testng.Assert;
import org.testng.annotations.Test;

import java.lang.reflect.Method;

/*
 * @test
 * @modules jdk.incubator.code
 * @run testng RootOpInsertionTest
 */
public class RootOpInsertionTest {

    @CodeReflection
    static void f() {}

    @Test
    void test() throws NoSuchMethodException {
        Method mf = this.getClass().getDeclaredMethod("f");
        CoreOp.FuncOp funcOp = Op.ofMethod(mf).orElseThrow();

        Body.Builder bodyBuilder = Body.Builder.of(null, FunctionType.VOID);
        Block.Builder entryBlock = bodyBuilder.entryBlock();
        Assert.assertThrows(IllegalStateException.class, () -> entryBlock.op(funcOp));
    }
}
