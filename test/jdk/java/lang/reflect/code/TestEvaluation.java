import jdk.incubator.code.CodeTransformer;
import jdk.incubator.code.Op;
import jdk.incubator.code.Reflect;
import jdk.incubator.code.TypeElement;
import jdk.incubator.code.dialect.core.CoreOp;
import jdk.incubator.code.dialect.core.CoreType;
import jdk.incubator.code.dialect.java.JavaOp;
import jdk.incubator.code.dialect.java.PrimitiveType;
import jdk.incubator.code.interpreter.Interpreter;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;

import java.lang.invoke.MethodHandles;
import java.lang.reflect.Method;
import java.time.LocalDate;
import java.util.Arrays;
import java.util.List;
import java.util.Optional;
import java.util.stream.Stream;

import static jdk.incubator.code.dialect.java.PrimitiveType.*;

/*
 * @test
 * @modules jdk.incubator.code
 * @run junit TestEvaluation
 * @run main Unreflect TestEvaluation
 * @run junit TestEvaluation
 */
public class TestEvaluation {
    // play with different ops and see if evaluate returns the correct result
    // TODO tests for constant expr in compiler ?
    @Reflect
    static int constant() {
        return 1;
    }

    @Reflect
    static String strLiteral() {
        return "A";
    }

    @Reflect
    static String concat() {
        return "number " + 1;
    }

    @Reflect
    static int constantVar() {
        final int x = 1;
        return x;
    }
    @Reflect
    static String strConstantVar() {
        final String s = "u";
        return s;
    }
    @Reflect
    static String strConstantVar2() {
        final int x = 1;
        final String s = x == 1 ? "one" : "not one";
        return s;
    }
    @Reflect
    static String strConstantVar3() {
        final String s = 123 + "456";
        return s;
    }
    @Reflect
    static String fcNotConstantVar() {
        // initialized with a non-constant expression
        final String s = LocalDate.now().toString();
        return s;
    }
    @Reflect
    static Object fcNotConstantVar2() {
        // type not primitive nor String
        final Object o = "s";
        return o;
    }
    @Reflect
    static int fcFinalVariableNotInitialized() {
        final int i;
        i = 1;
        return i;
    }

//    @Reflect
//    static int fcBlankFinalVar() {
//        // @@@ should fail
//        // currently we lack sufficent info to determine if a variable was declared final in source code
//        int x = 1;
//        return x;
//    }

    public static final int y = 3;
    @Reflect
    static int staticFinalField() {
        return y;
    }

    @Reflect
    static int unaryOperator() {
        return +1;
    }
    @Reflect
    static int unaryOperator2() {
        return -1;
    }
    @Reflect
    static int unaryOperator3() {
        return ~1;
    }
    @Reflect
    static boolean unaryOperator4() {
        return !false;
    }
    @Reflect
    static int binaryOperator() {
        return 1 + 2;
    }
    @Reflect
    static int binaryOperator2() {
        return 1 - 2;
    }
    @Reflect
    static int binaryOperator3() {
        return 1 * 2;
    }
    @Reflect
    static int binaryOperator4() {
        return 1 / 2;
    }
    @Reflect
    static int binaryOperator5() {
        return 1 % 2;
    }
    @Reflect
    static int binaryOperator6() {
        return 1 << 2;
    }
    @Reflect
    static int binaryOperator7() {
        return 1 >> 2;
    }
    @Reflect
    static int binaryOperator8() {
        return -1 >>> 2;
    }
    @Reflect
    static int binaryOperator9() {
        return 1 & 2;
    }
    @Reflect
    static int binaryOperator10() {
        return 1 | 2;
    }
    @Reflect
    static int binaryOperator11() {
        return 1 ^ 2;
    }
    @Reflect
    static boolean binaryOperator12() {
        return true & false;
    }
    @Reflect
    static boolean binaryOperator13() {
        return false | true;
    }
    @Reflect
    static boolean binaryOperator14() {
        return true ^ false;
    }
    @Reflect
    static boolean binaryTestOp() {
        return 1 < 2;
    }
    @Reflect
    static boolean binaryTestOp2() {
        return 1 <= 2;
    }
    @Reflect
    static boolean binaryTestOp3() {
        return 1 > 2;
    }
    @Reflect
    static boolean binaryTestOp4() {
        return 1 >= 2;
    }
    @Reflect
    static boolean binaryTestOp5() {
        return 1 == 2;
    }
    @Reflect
    static boolean binaryTestOp6() {
        return 1 != 2;
    }
    @Reflect
    static boolean binaryTestOp7() {
        return "A" != "B";
    }
    @Reflect
    static int condExprOp() {
        return 1 != 2 | 2 > 3 ? 3 : 4;
    }
    @Reflect
    static boolean condAndOp() {
        return true && false;
    }
    @Reflect
    static boolean condOrOp() {
        return true || false;
    }

    @ParameterizedTest
    @MethodSource("reflectableMethods")
    void test(Method m) throws NoSuchMethodException {
        CoreOp.FuncOp f = Op.ofMethod(m).get();
        Op op = ((Op.Result) f.body().entryBlock().terminatingOp().operands().getFirst()).op();
        MethodHandles.Lookup l = MethodHandles.lookup();
        Optional<Object> v = JavaOp.JavaExpression.evaluate(l, (Op & JavaOp.JavaExpression) op);
        Assertions.assertTrue(v.isPresent());
        // TODO BytecodeGen
        Object expected = Interpreter.invoke(l, f.transform(CodeTransformer.LOWERING_TRANSFORMER));
        Assertions.assertEquals(expected, v.get());
    }

    static Stream<Method> reflectableMethods() {
        return Arrays.stream(TestEvaluation.class.getDeclaredMethods())
                .filter(m -> m.isAnnotationPresent(Reflect.class))
                .filter(m -> !m.getName().startsWith("fc"));
    }
    @Reflect
    static int fc2() {
        int x = 1;
        x++;
        return x;
    }
    @Reflect
    static boolean fc3() {
        return "abc" instanceof String;
    }

    static int z = 0;
    @Reflect
    static int fc4() {
        return z;
    }
    @Reflect
    static Object fc6() {
        return (Object) 1;
    }
    @Reflect
    static Object fc7() {
        final int x = Math.abs(-1);
        return x;
    }
    @Reflect
    static boolean fc8() {
        return 1d > Math.pow(2, 2);
    }
    @Reflect
    static String fc9() {
        return null;
    }


    @ParameterizedTest
    @MethodSource("falseCases")
    void testFalseCases(Method m) {
        CoreOp.FuncOp f = Op.ofMethod(m).get();
        Op op = ((Op.Result) f.body().entryBlock().terminatingOp().operands().getFirst()).op();
        MethodHandles.Lookup l = MethodHandles.lookup();
        Optional<Object> v = JavaOp.JavaExpression.evaluate(MethodHandles.lookup(), (Op & JavaOp.JavaExpression) op);
        Assertions.assertTrue(v.isEmpty());
    }

    static Stream<Method> falseCases() {
        return Arrays.stream(TestEvaluation.class.getDeclaredMethods())
                .filter(m -> m.isAnnotationPresent(Reflect.class))
                .filter(m -> m.getName().startsWith("fc"));
    }

    static CoreOp.FuncOp conversionModel(TypeElement source, TypeElement target) {
        return CoreOp.func("conv", CoreType.functionType(target)).body(b -> {
            var v = b.op(CoreOp.constant(source, valueOne(source)));
            var r = b.op(JavaOp.conv(target, v));
            b.op(CoreOp.return_(r));
        });
    }

    static Object valueOne(TypeElement t) {
        if (t.equals(BOOLEAN)) {
            return true;
        } if (t.equals(BYTE)) {
            return (byte) 1;
        } else if (t.equals(SHORT)) {
            return (short) 1;
        } else if (t.equals(CHAR)) {
            return (char) 1;
        } else if (t.equals(INT)) {
            return 1;
        } else if (t.equals(LONG)) {
            return 1L;
        } else if (t.equals(FLOAT)) {
            return 1f;
        } else if (t.equals(DOUBLE)) {
            return 1d;
        }
        throw new IllegalArgumentException("Unkown value one for type " + t);
    }

    @Test
    void testConversion() {
        var pt = new PrimitiveType[] {BOOLEAN, BYTE, SHORT, CHAR, INT, LONG, FLOAT, DOUBLE};
        for (PrimitiveType from : pt) {
            for (PrimitiveType to : pt) {
                CoreOp.FuncOp f = conversionModel(from, to);
                Op op = ((Op.Result) f.body().entryBlock().terminatingOp().operands().getFirst()).op();
                MethodHandles.Lookup l = MethodHandles.lookup();
                Optional<Object> v = JavaOp.JavaExpression.evaluate(l, (JavaOp.ConvOp) op);
                if ((to.equals(BOOLEAN) && !from.equals(BOOLEAN)) || (from.equals(BOOLEAN) && !to.equals(BOOLEAN))) {
                    Assertions.assertTrue(v.isEmpty());
                } else {
                    Assertions.assertTrue(v.isPresent());
                    Object expected = Interpreter.invoke(l, f);
                    Assertions.assertEquals(expected, v.get());
                }
            }
        }
    }

    @Test
    void testInvalidConversion() {
        CoreOp.FuncOp funcOp = CoreOp.func("ic", CoreType.FUNCTION_TYPE_VOID).body(b -> {
            // String -> int
            b.op(JavaOp.conv(INT, b.op(CoreOp.constant(J_L_STRING, "A"))));
            // int -> String
            b.op(JavaOp.conv(J_L_STRING, b.op(CoreOp.constant(INT, 1))));
            b.op(CoreOp.return_());
        });

        List<Op> convOps = funcOp.body().entryBlock().ops().stream().filter(op -> op instanceof JavaOp.ConvOp).toList();
        for (Op convOp : convOps) {
            Optional<Object> opt = JavaOp.JavaExpression.evaluate(MethodHandles.lookup(), (JavaOp.ConvOp) convOp);
            Assertions.assertTrue(opt.isEmpty());
        }
    }

    @Test
    void testInvalidConstants() {
        CoreOp.FuncOp funcOp = CoreOp.func("ic", CoreType.FUNCTION_TYPE_VOID).body(b -> {
            // valid constant op result type but invalid values
            b.op(CoreOp.constant(J_L_STRING, null));
            b.op(CoreOp.constant(INT, new Object()));
            // invalid constant op result type but valid values
            b.op(CoreOp.constant(J_L_OBJECT, 1));
            b.op(CoreOp.constant(J_L_BOOLEAN, true));
            b.op(CoreOp.return_());
        });

        List<Op> constantOps = funcOp.body().entryBlock().ops().stream().filter(op -> op instanceof CoreOp.ConstantOp).toList();
        for (Op cop : constantOps) {
            Optional<Object> opt = JavaOp.JavaExpression.evaluate(MethodHandles.lookup(), (CoreOp.ConstantOp) cop);
            Assertions.assertTrue(opt.isEmpty());
        }
    }

    @Test
    void testCasts() {
        // invalid
        CoreOp.FuncOp iv = CoreOp.func("ic", CoreType.FUNCTION_TYPE_VOID).body(b -> {
            // we can have cast to String but the value we cast has a non-type String
            // this will not be allowed by the Java language
            // e.g. String s = (String) 1;
            b.op(JavaOp.cast(J_L_STRING, b.op(CoreOp.constant(INT, 1))));
            b.op(CoreOp.return_());
        });
        List<JavaOp.CastOp> castOps = iv.body().entryBlock().ops().stream().filter(op -> op instanceof JavaOp.CastOp)
                .map(op -> (JavaOp.CastOp) op).toList();
        for (JavaOp.CastOp castOp : castOps) {
            Optional<Object> opt = JavaOp.JavaExpression.evaluate(MethodHandles.lookup(), castOp);
            Assertions.assertTrue(opt.isEmpty());
        }

        // valid
        CoreOp.FuncOp v = CoreOp.func("vc", CoreType.FUNCTION_TYPE_VOID).body(b -> {
            // we can have cast to String but the value we cast has a non-type String
            // this will not be allowed by the Java language
            // e.g. String s = (String) 1;
            b.op(JavaOp.cast(J_L_STRING, b.op(CoreOp.constant(J_L_STRING, "1"))));
            b.op(CoreOp.return_());
        });
        List<JavaOp.CastOp> castOps2 = v.body().entryBlock().ops().stream().filter(op -> op instanceof JavaOp.CastOp)
                .map(op -> (JavaOp.CastOp) op).toList();
        for (JavaOp.CastOp castOp : castOps2) {
            Optional<Object> opt = JavaOp.JavaExpression.evaluate(MethodHandles.lookup(), castOp);
            Assertions.assertTrue(opt.isPresent());
        }
    }
}
