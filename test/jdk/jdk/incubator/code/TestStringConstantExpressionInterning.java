import jdk.incubator.code.CodeTransformer;
import jdk.incubator.code.Op;
import jdk.incubator.code.Reflect;
import jdk.incubator.code.Value;
import jdk.incubator.code.bytecode.BytecodeGenerator;
import jdk.incubator.code.dialect.java.JavaOp;
import jdk.incubator.code.dialect.java.MethodRef;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;

import java.lang.invoke.MethodHandles;
import java.lang.reflect.Method;
import java.util.Arrays;
import java.util.Optional;
import java.util.function.BiFunction;
import java.util.stream.Stream;

import static jdk.incubator.code.dialect.core.CoreOp.*;
import static jdk.incubator.code.dialect.java.JavaOp.*;
import static jdk.incubator.code.dialect.java.JavaType.*;

/*
 * @test
 * @modules jdk.incubator.code
 * @library lib
 * @run junit TestStringConstantExpressionInterning
 */
public class TestStringConstantExpressionInterning {

    @Reflect
    static boolean t_strLiteral() {
        return "A" + 1 == "A1";
    }

    static final String B = "B";
    @Reflect
    static boolean t_classVariable() {
        return B + 1 == "B1";
    }

    static final String C;
    static {
        C = "C";
    }
    //@Reflect
    static boolean f_classVariable2() {
        // C + 1 shouldn't be interned, but currently it's
        // that's a limitation in the evaluate API, as the API consider C a constant variable
        return C + 1 == "C1";
    }

    static final String D = "D".toLowerCase();
    //@Reflect
    static boolean f_classVariable3() {
        // D shouldn't be interned, but currently it's
        // that's a limitation in the evaluate API, as the API consider D a constant variable
        return D == "d";
    }

    final String E = "E";
    //@Reflect
    boolean t_instanceVariable() {
        // A + 1 should be interned, but currently it's not
        // because the op evaluate API considers E (and any instance field) a non constant variable
        // it's a limitation in the API
        // if core reflection can tag fields that are constant variables the limitation will be solved
        // or, we can change the model by replacing read of constant variable by constant
        // but the latter only work for compiler generated model
        return E + 1 == "E1";
    }

    @Reflect
    static boolean t_localVariable() {
        // in the model, we broaden the notion of JLS constant variable to include effectively final variable
        String s = "A";
        return s + 1 == "A1";
    }

    static Stream<Method> cases() {
        return Arrays.stream(TestStringConstantExpressionInterning.class.getDeclaredMethods())
                .filter(m -> m.isAnnotationPresent(Reflect.class));
    }

    @ParameterizedTest
    @MethodSource("cases")
    void test(Method m) throws Throwable {
        FuncOp op = Op.ofMethod(m).get();
        Object expected = m.getName().startsWith("t");
        MethodHandles.Lookup l = MethodHandles.lookup();

        Value rv = op.body().entryBlock().terminatingOp().operands().getFirst();
        Object v = JavaExpression.evaluate(l, rv).get();
        Assertions.assertEquals(expected, v, op.toText());

        Assertions.assertNotEquals(expected, Interpreter.invoke(l, op));
        Assertions.assertNotEquals(expected, BytecodeGenerator.generate(l, op).invoke());

        FuncOp op2 = op.transform(internStringConstantExpr);
        Assertions.assertEquals(expected, Interpreter.invoke(l, op2));
        Assertions.assertEquals(expected, BytecodeGenerator.generate(l, op2).invoke());

        FuncOp op3 = op.transform(foldConstants);
        Assertions.assertEquals(expected, Interpreter.invoke(l, op3));
        Assertions.assertEquals(expected, BytecodeGenerator.generate(l, op3).invoke());
    }

    private static final MethodRef STRING_INTERN = MethodRef.method(J_L_STRING, "intern", J_L_STRING);
    CodeTransformer internStringConstantExpr = (b, op) -> {
        Result r = b.op(op);
        Optional<Object> v = JavaExpression.evaluate(MethodHandles.lookup(), op.result());
        if (v.isPresent() && v.get() instanceof String) {
            r = b.op(JavaOp.invoke(STRING_INTERN, r));
        }
        b.context().mapValue(op.result(), r);
        return b;
    };

    CodeTransformer foldConstants = (b, op) -> {
        if (op instanceof JavaExpression) {
            BiFunction<MethodHandles.Lookup, Value, Object> operandEvaluator = (l, operand) -> {
                Value nv = b.context().getValue(operand);
                if (nv instanceof Op.Result opr && opr.op() instanceof ConstantOp cop) {
                    return cop.value();
                }
                return null;
            };
            // we extend JavaExpression.evaluate by passing an operand evaluator, to avoid re-evaluation of operands
            Optional<Object> v = JavaExpression.evaluate(MethodHandles.lookup(), (Op & JavaExpression) op, operandEvaluator);
            if (v.isPresent()) {
                Result c = b.op(constant(op.resultType(), v.get()));
                b.context().mapValue(op.result(), c);
            }
        } else {
            b.op(op);
        }
        return b;
    };
}
