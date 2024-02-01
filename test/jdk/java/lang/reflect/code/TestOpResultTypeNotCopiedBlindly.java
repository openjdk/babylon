import org.testng.Assert;
import org.testng.annotations.Test;

import java.lang.reflect.code.Op;
import java.lang.runtime.CodeReflection;
import java.util.Arrays;

import static java.lang.reflect.code.descriptor.MethodTypeDesc.methodType;
import static java.lang.reflect.code.descriptor.TypeDesc.DOUBLE;
import static java.lang.reflect.code.op.CoreOps.*;

/*
 * @test
 * @run testng TestOpResultTypeNotCopiedBlindly
 */

public class TestOpResultTypeNotCopiedBlindly {

    @CodeReflection
    static int f(int a, int b) {
        return a + b;
    }

    @Test
    void test() {
        FuncOp f = getCodeModel(this.getClass(), "f");

        FuncOp g = func("g", methodType(DOUBLE, DOUBLE, DOUBLE))
                .body(b -> b.inline(f, b.parameters(), (block, v) -> {
                    block.op(_return(v));
                }));

        g.writeTo(System.out);

        // check that add has a result-type of double
        Assert.assertEquals(
                ((Op.Result) g.body().entryBlock().terminatingOp().operands().get(0)).op().result().type(),
                DOUBLE
        );
    }

    private static FuncOp getCodeModel(Class<?> c, String methodName) {
        return Arrays.stream(c.getDeclaredMethods()).filter(m -> m.getName().equals(methodName))
                .findFirst().get().getCodeModel().get();
    }
}
