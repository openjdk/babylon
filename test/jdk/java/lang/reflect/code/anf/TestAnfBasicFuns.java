import java.lang.reflect.Method;
import java.lang.reflect.code.OpTransformer;
import java.lang.reflect.code.analysis.AnfTransformer;
import java.lang.reflect.code.analysis.SSA;
import java.lang.reflect.code.op.CoreOp;
import java.lang.runtime.CodeReflection;
import java.util.List;

/*
 * @test
 * @run main TestAnfBasicFuns
 */

public class TestAnfBasicFuns {

    @CodeReflection
    public static int test2(int arg1, int arg2) {
        if (arg1 > arg2) {
            return arg1 + 21;
        } else {
            return arg2 + 42;
        }
    }

    public static void main(String[] args) {

        testRun("test2", List.of(int.class, int.class), 20, 1);

    }

    private static void testRun(String methodName, List<Class<?>> params, Object...args) {
        try {
            Class<TestAnfBasicFuns> clazz = TestAnfBasicFuns.class;
            Method method = clazz.getDeclaredMethod(methodName,params.toArray(new Class[params.size()]));
            CoreOp.FuncOp f = method.getCodeModel().orElseThrow();

            //Ensure we're fully lowered before testing.
            var fz = f.transform(OpTransformer.LOWERING_TRANSFORMER);
            fz = SSA.transform(fz);

            System.out.println(fz.toText());

            var res = new AnfTransformer(fz).transform();
            //var labeled = new Labeler().label(res);
            //p.print(res);
            //p.print(labeled);

            System.out.println("---------------------");
            System.out.println(res.toText());

        } catch (NoSuchMethodException e) {
            throw new RuntimeException(e);
        }
    }
}
