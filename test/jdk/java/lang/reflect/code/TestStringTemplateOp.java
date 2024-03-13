import org.testng.Assert;
import org.testng.annotations.Test;

import java.lang.reflect.Method;
import java.lang.reflect.code.Op;
import java.lang.reflect.code.interpreter.Interpreter;
import java.lang.reflect.code.op.CoreOps;
import java.lang.runtime.CodeReflection;
import java.util.List;
import java.util.Optional;
import java.util.stream.Stream;

/*
 * @test
 * @enablePreview
 * @run testng TestStringTemplateOp
 */

public class TestStringTemplateOp {

    @CodeReflection
    static String f(int x, int y) {
        String s = STR."x = \{x}, y = \{y}, x + y = \{x + y}";
        return s;
    }

    @Test
    public void testf() {
        CoreOps.FuncOp f = getFuncOp("f");

        f.writeTo(System.out);

        CoreOps.FuncOp lf = f.transform((block, op) -> {
            if (op instanceof Op.Lowerable lop) {
                return lop.lower(block);
            } else {
                block.op(op);
                return block;
            }
        });

        lf.writeTo(System.out);

        Assert.assertEquals(Interpreter.invoke(lf, 1, 2), f(1, 2));
    }


    static CoreOps.FuncOp getFuncOp(String name) {
        Optional<Method> om = Stream.of(TestStringTemplateOp.class.getDeclaredMethods())
                .filter(m -> m.getName().equals(name))
                .findFirst();

        Method m = om.get();
        return m.getCodeModel().get();
    }
}
