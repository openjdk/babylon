import jdk.incubator.code.CodeTransformer;
import jdk.incubator.code.Op;
import jdk.incubator.code.Reflect;
import jdk.incubator.code.Value;
import jdk.incubator.code.bytecode.BytecodeGenerator;
import jdk.incubator.code.dialect.core.CoreType;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;

import java.lang.invoke.MethodHandles;
import java.lang.reflect.Method;
import java.time.LocalDate;
import java.util.Arrays;
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

    @Reflect
    static boolean t_case7() {
        // @@@ we keep the conditional expr in the model
        String s = 1 < 2 ? "A" + 1 : LocalDate.now().toString();
        return s == "A1";
    }

    @Reflect
    static boolean t_case8() {
        int i = 1;
        String s = switch(i) {
            case 1 -> "A" + 2 + 4;
            default -> "B";
        };
        return s == "A24";
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

        Assertions.assertNotEquals(expected, Interpreter.invoke(l, op.transform(CodeTransformer.LOWERING_TRANSFORMER)));
        Assertions.assertNotEquals(expected, BytecodeGenerator.generate(l, op).invoke());

        // fold constant + lower
        FuncOp transformed = op.transform(TestConstantFolding.foldConstants);
        Assertions.assertEquals(expected, Interpreter.invoke(l, transformed.transform(CodeTransformer.LOWERING_TRANSFORMER)));
        Assertions.assertEquals(expected, BytecodeGenerator.generate(l, transformed).invoke());

        // lower + fold constant
        FuncOp transformed2 = op.transform(CodeTransformer.LOWERING_TRANSFORMER)
                .transform(TestConstantFolding.foldConstants);
        Assertions.assertEquals(expected, Interpreter.invoke(l, transformed2));
        Assertions.assertEquals(expected, BytecodeGenerator.generate(l, transformed2).invoke());
    }

}
