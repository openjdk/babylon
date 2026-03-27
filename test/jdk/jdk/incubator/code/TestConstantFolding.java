import jdk.incubator.code.CodeTransformer;
import jdk.incubator.code.Op;
import jdk.incubator.code.Reflect;
import jdk.incubator.code.dialect.core.CoreOp;
import jdk.incubator.code.dialect.java.JavaOp;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;

import java.lang.invoke.MethodHandles;
import java.lang.reflect.Method;
import java.time.LocalDate;
import java.util.Arrays;
import java.util.Optional;
import java.util.stream.Stream;

import static jdk.incubator.code.dialect.core.CoreOp.constant;

/*
 * @test
 * @modules jdk.incubator.code
 * @library lib
 * @build TestStringConstantExpressionInterning
 * @run junit TestConstantFolding
 */
public class TestConstantFolding {

    // how about evaluate an value and put the result in a map
    // op with bodies ?
    @Reflect
    static int f() {
        return ((1 + 2) + (3 + 4)) + 4 + 5;
    }

    @Reflect
    static void f2() {
        // @@@ we keep the conidtional expr
        String s = 1 < 2 ? "A" + 1 : LocalDate.now().toString();
        boolean b = s == "A1";
    }

    @Reflect
    static boolean t_localVariable() {
        // in the model, we broaden the notion of JLS constant variable to include effectively final variable
        String s = "A";
        return s + 1 == "A1";
    }

    static Stream<Method> cases() {
        return Arrays.stream(TestConstantFolding.class.getDeclaredMethods()).filter(m -> m.isAnnotationPresent(Reflect.class));
    }

    @ParameterizedTest
    @MethodSource("cases")
    void test(Method m) throws NoSuchMethodException {
        CoreOp.FuncOp op = Op.ofMethod(m).get();
        System.out.println(op.toText());
        CoreOp.FuncOp op2 = op.transform(foldConstants);
        System.out.println(op2.toText());
    }

    static final JavaOp.JavaExpression.Evaluator evaluator = new JavaOp.JavaExpression.Evaluator(MethodHandles.lookup());
    static final CodeTransformer foldConstants = (b, op) -> {
        Optional<Object> v = evaluator.evaluate(op.result());
        if (v.isPresent()) {
            Op.Result c = b.op(constant(op.resultType(), v.get()));
            b.context().mapValue(op.result(), c);
        } else {
            b.op(op);
        }
        return b;
    };
}
