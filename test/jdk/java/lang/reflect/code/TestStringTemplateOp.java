import org.testng.Assert;
import org.testng.annotations.DataProvider;
import org.testng.annotations.Test;

import java.lang.reflect.InvocationTargetException;
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
        return STR."x = \{x}, y = \{y}, x + y = \{x + y}" ;
    }

    @CodeReflection
    static String f2(int a, int b) {
        return STR."\{a > b ? STR."\{a} greater than \{b}" : STR."\{b} greater than or equal \{a}"}";
    }

    @CodeReflection
    static String f3() {
        return STR."\{(byte) 1} \{(short) 2} \{3} \{4L} \{5f} \{6d} \{'c'} \{true} \{List.of()}";
    }

    @CodeReflection
    static String f4() {
        // test with a number of expressions that is greater than the List.of parameter threshold where varargs is used
        return STR."\{(byte) 1} \{(short) 2} \{3} \{4L} \{5f} \{6d} \{'c'} \{true} \{List.of()}, \{!true}, \{System.out}";
    }
    @DataProvider
    public Object[][] cases() {
        return new Object[][] {
                {"f", new Object[] {2, 42}},
                {"f2", new Object[] {13, 7}},
                {"f3", new Object[] {}},
                {"f4", new Object[] {}}
        };
    }
    @Test(dataProvider = "cases")
    public void test(String caseName, Object[] args) throws InvocationTargetException, IllegalAccessException {
        CoreOps.FuncOp f = getFuncOp(caseName);

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

        Assert.assertEquals(Interpreter.invoke(lf, args), getMethod(caseName).invoke(null, args));
    }

    static CoreOps.FuncOp getFuncOp(String name) {
        Method m = getMethod(name);
        return m.getCodeModel().get();
    }

    static Method getMethod(String name) {
        Optional<Method> om = Stream.of(TestStringTemplateOp.class.getDeclaredMethods())
                .filter(m -> m.getName().equals(name))
                .findFirst();

        Method m = om.get();
        return m;
    }
}
