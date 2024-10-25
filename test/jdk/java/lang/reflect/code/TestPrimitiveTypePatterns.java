import org.testng.Assert;
import org.testng.annotations.DataProvider;
import org.testng.annotations.Test;

import java.lang.invoke.MethodHandles;
import java.lang.reflect.Method;
import java.lang.reflect.code.Body;
import java.lang.reflect.code.OpTransformer;
import java.lang.reflect.code.bytecode.BytecodeGenerator;
import java.lang.reflect.code.interpreter.Interpreter;
import java.lang.reflect.code.op.CoreOp;
import java.lang.reflect.code.op.ExtendedOp;
import java.lang.reflect.code.type.JavaType;
import java.lang.reflect.code.type.MethodRef;
import java.lang.runtime.CodeReflection;
import java.lang.runtime.ExactConversionsSupport;
import java.util.List;
import java.util.Optional;
import java.util.stream.Stream;

import static java.lang.reflect.code.op.CoreOp.*;
import static java.lang.reflect.code.op.ExtendedOp.match;
import static java.lang.reflect.code.op.ExtendedOp.typePattern;
import static java.lang.reflect.code.type.FunctionType.functionType;
import static java.lang.reflect.code.type.PrimitiveType.*;

/*
 * @test
 * @run testng TestPrimitiveTypePatterns
 * @enablePreview
 */

public class TestPrimitiveTypePatterns {

    static MethodRef conversionMethodRef(JavaType sourceType, JavaType targetType) {
        if (SHORT.equals(sourceType) || CHAR.equals(sourceType)) {
            sourceType = INT;
        }
        String n = "is%sTo%sExact".formatted(capitalize(sourceType.toString()), capitalize(targetType.toString()));
        JavaType c = JavaType.type(ExactConversionsSupport.class);
        return MethodRef.method(c, n, BOOLEAN, sourceType);
    }

    static String capitalize(String s) {
        return s.substring(0, 1).toUpperCase() + s.substring(1);
    }

    @DataProvider
    public static Object[][] dp() {
        return new Object[][]{
                {JavaType.INT, JavaType.BYTE, new Object[] {
                        Byte.MIN_VALUE - 1, Byte.MIN_VALUE, Byte.MAX_VALUE, Byte.MAX_VALUE + 1
                }},
                {JavaType.INT, JavaType.SHORT, new Object[] {
                        Short.MIN_VALUE - 1, Short.MIN_VALUE, Short.MAX_VALUE, Short.MAX_VALUE + 1
                }},
                {JavaType.INT, JavaType.CHAR, new Object[] {
                        Character.MIN_VALUE - 1, Character.MIN_VALUE, Character.MAX_VALUE, Character.MAX_VALUE + 1
                }},
                // (1<<24) + 1 : first int that's not an instanceof float
                // 1<<31) - (1<<7): largest int that's an instance of float
                {JavaType.INT, JavaType.FLOAT, new Object[] {
                        1<<24, (1<<24) + 1, (1<<31) - (1<<7), (1<<31) - (1<<7) + 1, Integer.MAX_VALUE, Integer.MIN_VALUE
                }},

                {JavaType.SHORT, JavaType.BYTE, new Object[]{
                        (short) (Byte.MIN_VALUE - 1), Byte.MIN_VALUE, Byte.MAX_VALUE, (short) (Byte.MAX_VALUE + 1)
                }},
                {JavaType.SHORT, JavaType.CHAR, new Object[]{
                        Short.MIN_VALUE, (short) -1, (short) 0, Short.MAX_VALUE
                }},

                {JavaType.CHAR, JavaType.BYTE, new Object[]{
                        (char) 0, (char) Byte.MAX_VALUE, (char) (Byte.MAX_VALUE + 1)
                }},
                {JavaType.CHAR, JavaType.SHORT, new Object[]{
                        (char) 0, (char) Short.MAX_VALUE, (char) (Short.MAX_VALUE + 1)
                }},

                {JavaType.LONG, JavaType.BYTE, new Object[] {
                        Byte.MIN_VALUE - 1, Byte.MIN_VALUE, Byte.MAX_VALUE, Byte.MAX_VALUE + 1
                }},
                {JavaType.LONG, JavaType.SHORT, new Object[] {
                        Short.MIN_VALUE - 1, Short.MIN_VALUE, Short.MAX_VALUE, Short.MAX_VALUE + 1
                }},
                {JavaType.LONG, JavaType.CHAR, new Object[] {
                        Character.MIN_VALUE - 1, Character.MIN_VALUE, Character.MAX_VALUE, Character.MAX_VALUE + 1
                }},
                {JavaType.LONG, JavaType.INT, new Object[] {
                        (long)Integer.MIN_VALUE - 1, Integer.MIN_VALUE, Integer.MAX_VALUE, (long)Integer.MAX_VALUE + 1
                }},
                // (1<<24) + 1 : first long that can't be represented as float
                // (1L<<63) - (1L<<39) : largest long that can be represented as float
                {JavaType.LONG, JavaType.FLOAT, new Object[] {
                        Long.MIN_VALUE, (1L<<24), (1<<24) + 1, (1L<<63) - (1L<<39), (1L<<63) - (1L<<39) + 1, Long.MAX_VALUE
                }},
                // (1L<<53) + 1 : first long that can't be represented as double
                // (1L<<63) - (1<<10) : largest long that can be represented as double
                {JavaType.LONG, JavaType.DOUBLE, new Object[] {
                        Long.MIN_VALUE, 1L<<53, (1L<<53) + 1, (1L<<63) - (1<<10), (1L<<63) - (1<<10) + 1, Long.MAX_VALUE
                }},

                {JavaType.FLOAT, JavaType.BYTE, new Object[] {
                        Byte.MIN_VALUE - 1, Byte.MIN_VALUE, Byte.MAX_VALUE, Byte.MIN_VALUE + 1
                }},
                {JavaType.FLOAT, JavaType.SHORT, new Object[] {
                        Short.MIN_VALUE - 1, Short.MIN_VALUE, Short.MAX_VALUE, Short.MAX_VALUE + 1
                }},
                {JavaType.FLOAT, JavaType.CHAR, new Object[] {
                        Character.MIN_VALUE - 1, Character.MIN_VALUE, Character.MAX_VALUE, Character.MAX_VALUE + 1
                }},
                {JavaType.FLOAT, JavaType.INT, new Object[] {
                        Float.MIN_VALUE, Float.NEGATIVE_INFINITY, 0f, Float.POSITIVE_INFINITY, Float.MAX_VALUE
                }},
                {JavaType.FLOAT, JavaType.LONG, new Object[] {
                        Float.MIN_VALUE, Float.NEGATIVE_INFINITY, 0f, Float.POSITIVE_INFINITY, Float.MAX_VALUE
                }},

                {JavaType.DOUBLE, JavaType.BYTE, new Object[] {
                        Double.NEGATIVE_INFINITY, Double.MIN_VALUE, -0d, +0d, Double.MAX_VALUE, Double.POSITIVE_INFINITY, Double.NaN
                }},
                {JavaType.DOUBLE, JavaType.SHORT, new Object[] {
                        Double.NEGATIVE_INFINITY, Double.MIN_VALUE, -0d, +0d, Double.MAX_VALUE, Double.POSITIVE_INFINITY, Double.NaN
                }},
                {JavaType.DOUBLE, JavaType.CHAR, new Object[] {
                        Double.NEGATIVE_INFINITY, Double.MIN_VALUE, -0d, +0d, Double.MAX_VALUE, Double.POSITIVE_INFINITY, Double.NaN
                }},
                {JavaType.DOUBLE, JavaType.INT, new Object[] {
                        Double.NEGATIVE_INFINITY, Double.MIN_VALUE, -0d, +0d, Double.MAX_VALUE, Double.POSITIVE_INFINITY, Double.NaN
                }},
                {JavaType.DOUBLE, JavaType.LONG, new Object[] {
                        Double.NEGATIVE_INFINITY, Double.MIN_VALUE, -0d, +0d, Double.MAX_VALUE, Double.POSITIVE_INFINITY, Double.NaN
                }},
                {JavaType.DOUBLE, JavaType.FLOAT, new Object[] {
                        Double.NEGATIVE_INFINITY, Double.MIN_VALUE, -0d, +0d, Double.MAX_VALUE, Double.POSITIVE_INFINITY, Double.NaN
                }}

        };
    }

    @Test(dataProvider = "dp")
    void test(JavaType sourceType, JavaType targetType, Object[] values) throws Throwable {

        var model = buildTypePatternModel(sourceType, targetType);
        model.writeTo(System.out);

        var lmodel = model.transform(OpTransformer.LOWERING_TRANSFORMER);
        lmodel.writeTo(System.out);


        var expectedConvMethod = conversionMethodRef(sourceType, targetType);
        var actualConvMethod = lmodel.elements()
                .mapMulti((ce, c) -> {
                    if (ce instanceof InvokeOp op) {
                        c.accept(op.invokeDescriptor());
                    }
                })
                .findFirst().orElseThrow();
        Assert.assertEquals(actualConvMethod, expectedConvMethod);

        var mh = BytecodeGenerator.generate(MethodHandles.lookup(), lmodel);

        for (Object v : values) {
            Assert.assertEquals(Interpreter.invoke(lmodel, v), mh.invoke(v));
        }
    }

    @CodeReflection
    static boolean ip(short s) {
        return s instanceof short _;
    }

    @Test
    void test_ip() {
        FuncOp f = getFuncOp("ip");
        f.writeTo(System.out);

        FuncOp lf = f.transform(OpTransformer.LOWERING_TRANSFORMER);
        lf.writeTo(System.out);

        Assert.assertEquals(Interpreter.invoke(lf, Short.MAX_VALUE), true);
    }

    @CodeReflection
    static boolean wnp(byte s) {
        return s instanceof char _;
    }

    @Test
    void test_wnp() {
        FuncOp f = getFuncOp("wnp");
        f.writeTo(System.out);

        FuncOp lf = f.transform(OpTransformer.LOWERING_TRANSFORMER);
        lf.writeTo(System.out);

        Assert.assertEquals(Interpreter.invoke(lf, Byte.MAX_VALUE), true);
        Assert.assertEquals(Interpreter.invoke(lf, Byte.MIN_VALUE), false);
    }

    @CodeReflection
    static boolean b(int s) {
        return s instanceof Integer _;
    }

    @Test
    void test_b() {
        FuncOp f = getFuncOp("b");
        f.writeTo(System.out);

        FuncOp lf = f.transform(OpTransformer.LOWERING_TRANSFORMER);
        lf.writeTo(System.out);

        Assert.assertEquals(Interpreter.invoke(lf, Integer.MAX_VALUE), true);
        Assert.assertEquals(Interpreter.invoke(lf, Integer.MIN_VALUE), true);
    }

    @CodeReflection
    static boolean bw(int s) {
        return s instanceof Number _;
    }

    @Test
    void test_bw() {
        FuncOp f = getFuncOp("bw");
        f.writeTo(System.out);

        FuncOp lf = f.transform(OpTransformer.LOWERING_TRANSFORMER);
        lf.writeTo(System.out);

        Assert.assertEquals(Interpreter.invoke(lf, Integer.MAX_VALUE), true);
        Assert.assertEquals(Interpreter.invoke(lf, Integer.MIN_VALUE), true);
    }

    @CodeReflection
    static boolean nr_unboxing(Number n) {
        return n instanceof int _;
    }

    @Test
    void test_nr_unboxing() {
        FuncOp f = getFuncOp("nr_unboxing");
        f.writeTo(System.out);

        FuncOp lf = f.transform(OpTransformer.LOWERING_TRANSFORMER);
        lf.writeTo(System.out);

        Assert.assertEquals(Interpreter.invoke(lf, 1), true);
        Assert.assertEquals(Interpreter.invoke(lf, (short) 1), false);
    }

    @CodeReflection
    static boolean unboxing(Integer n) {
        return n instanceof int _;
    }

    @Test
    void test_unboxing() {
        FuncOp f = getFuncOp("unboxing");
        f.writeTo(System.out);

        FuncOp lf = f.transform(OpTransformer.LOWERING_TRANSFORMER);
        lf.writeTo(System.out);

        Assert.assertEquals(Interpreter.invoke(lf, Integer.MAX_VALUE), true);
        Assert.assertEquals(Interpreter.invoke(lf, Integer.MIN_VALUE), true);
    }

    @CodeReflection
    static boolean unboxing_wp(Integer n) {
        return n instanceof long _;
    }

    @Test
    void test_unboxing_wp() {
        FuncOp f = getFuncOp("unboxing_wp");
        f.writeTo(System.out);

        FuncOp lf = f.transform(OpTransformer.LOWERING_TRANSFORMER);
        lf.writeTo(System.out);

        Assert.assertEquals(Interpreter.invoke(lf, Integer.MAX_VALUE), true);
        Assert.assertEquals(Interpreter.invoke(lf, Integer.MIN_VALUE), true);
    }

    @CodeReflection
    static boolean wr(String s) {
        return s instanceof Object _;
    }

    @Test
    void test_wr() {
        FuncOp f = getFuncOp("wr");
        f.writeTo(System.out);

        FuncOp lf = f.transform(OpTransformer.LOWERING_TRANSFORMER);
        lf.writeTo(System.out);

        Assert.assertEquals(Interpreter.invoke(lf, (Object) null), false);
        Assert.assertEquals(Interpreter.invoke(lf, "str"), true);
    }

    @CodeReflection
    static boolean ir(Float f) {
        return f instanceof Float _;
    }

    @Test
    void test_ir() {
        FuncOp f = getFuncOp("ir");
        f.writeTo(System.out);

        FuncOp lf = f.transform(OpTransformer.LOWERING_TRANSFORMER);
        lf.writeTo(System.out);

        Assert.assertEquals(Interpreter.invoke(lf, Float.MAX_VALUE), true);
        Assert.assertEquals(Interpreter.invoke(lf, Float.MIN_VALUE), true);
        Assert.assertEquals(Interpreter.invoke(lf, Float.POSITIVE_INFINITY), true);
        Assert.assertEquals(Interpreter.invoke(lf, Float.NEGATIVE_INFINITY), true);
    }

    @CodeReflection
    static boolean nr(Number n) {
        return n instanceof Double _;
    }

    @Test
    void test_nr() {
        FuncOp f = getFuncOp("nr");
        f.writeTo(System.out);

        FuncOp lf = f.transform(OpTransformer.LOWERING_TRANSFORMER);
        lf.writeTo(System.out);

        Assert.assertEquals(Interpreter.invoke(lf, Float.MAX_VALUE), false);
        Assert.assertEquals(Interpreter.invoke(lf, Integer.MIN_VALUE), false);
        Assert.assertEquals(Interpreter.invoke(lf, Double.POSITIVE_INFINITY), true);
        Assert.assertEquals(Interpreter.invoke(lf, Double.NEGATIVE_INFINITY), true);
    }

     private CoreOp.FuncOp getFuncOp(String name) {
        Optional<Method> om = Stream.of(this.getClass().getDeclaredMethods())
                .filter(m -> m.getName().equals(name))
                .findFirst();

        Method m = om.get();
        return m.getCodeModel().get();
    }

    static FuncOp buildTypePatternModel(JavaType sourceType, JavaType targetType) {
        // builds the model of:
        // static boolean f(sourceType a) { return a instanceof targetType _; }
        return func(sourceType + "_" + targetType, functionType(JavaType.BOOLEAN, sourceType)).body(fblock -> {

            var paramVal = fblock.parameters().get(0);

            var patternVar = fblock.op(var(fblock.op(constant(targetType, defaultValue(targetType)))));

            var pattern = Body.Builder.of(fblock.parentBody(), functionType(ExtendedOp.Pattern.bindingType(targetType)));
            pattern.entryBlock().op(_yield(
                    pattern.entryBlock().op(typePattern(targetType, null))
            ));

            var match = Body.Builder.of(fblock.parentBody(), functionType(JavaType.VOID, targetType));
            var binding = match.entryBlock().parameters().get(0);
            match.entryBlock().op(varStore(patternVar, binding));
            match.entryBlock().op(_yield());

            var result = fblock.op(match(paramVal, pattern, match));

            fblock.op(_return(result));
        });
    }

    static Object defaultValue(JavaType t) {
        if (List.of(BYTE, SHORT, CHAR, INT).contains(t)) {
            return 0;
        } else if (LONG.equals(t)) {
            return 0L;
        } else if (FLOAT.equals(t)) {
            return 0f;
        } else if (DOUBLE.equals(t)) {
            return 0d;
        } else if (BOOLEAN.equals(t)) {
            return false;
        }
        return null;
    }
}
