import org.testng.Assert;
import org.testng.annotations.DataProvider;
import org.testng.annotations.Test;

import java.lang.invoke.MethodHandles;
import java.lang.reflect.code.Block;
import java.lang.reflect.code.Body;
import java.lang.reflect.code.Op;
import java.lang.reflect.code.OpTransformer;
import java.lang.reflect.code.bytecode.BytecodeGenerator;
import java.lang.reflect.code.interpreter.Interpreter;
import java.lang.reflect.code.op.ExtendedOp;
import java.lang.reflect.code.type.JavaType;
import java.lang.reflect.code.type.MethodRef;
import java.lang.reflect.code.type.PrimitiveType;
import java.lang.runtime.ExactConversionsSupport;
import java.util.List;
import java.util.function.Predicate;

import static java.lang.reflect.code.op.CoreOp.*;
import static java.lang.reflect.code.op.ExtendedOp.match;
import static java.lang.reflect.code.op.ExtendedOp.typePattern;
import static java.lang.reflect.code.type.FunctionType.functionType;

/*
 * @test
 * @run testng TestPrimitiveTypePatterns
 * @enablePreview
 */

public class TestPrimitiveTypePatterns {

    static final MethodRef intToByte = conversionMethodRef(JavaType.INT, JavaType.BYTE);
    static final MethodRef intToShort = conversionMethodRef(JavaType.INT, JavaType.SHORT);
    static final MethodRef intToChar = conversionMethodRef(JavaType.INT, JavaType.CHAR);
    static final MethodRef intToFloat = conversionMethodRef(JavaType.INT, JavaType.FLOAT);
    static final MethodRef longToByte = conversionMethodRef(JavaType.LONG, JavaType.BYTE);
    static final MethodRef longToShort = conversionMethodRef(JavaType.LONG, JavaType.SHORT);
    static final MethodRef longToChar = conversionMethodRef(JavaType.LONG, JavaType.CHAR);
    static final MethodRef longToInt = conversionMethodRef(JavaType.LONG, JavaType.INT);
    static final MethodRef longToFloat = conversionMethodRef(JavaType.LONG, JavaType.FLOAT);
    static final MethodRef longToDouble = conversionMethodRef(JavaType.LONG, JavaType.DOUBLE);
    static final MethodRef floatToByte = conversionMethodRef(JavaType.FLOAT, JavaType.BYTE);
    static final MethodRef floatToShort = conversionMethodRef(JavaType.FLOAT, JavaType.SHORT);
    static final MethodRef floatToChar = conversionMethodRef(JavaType.FLOAT, JavaType.CHAR);
    static final MethodRef floatToInt = conversionMethodRef(JavaType.FLOAT, JavaType.INT);
    static final MethodRef floatToLong = conversionMethodRef(JavaType.FLOAT, JavaType.LONG);
    static final MethodRef doubleToByte = conversionMethodRef(JavaType.DOUBLE, JavaType.BYTE);
    static final MethodRef doubleToShort = conversionMethodRef(JavaType.DOUBLE, JavaType.SHORT);
    static final MethodRef doubleToChar = conversionMethodRef(JavaType.DOUBLE, JavaType.CHAR);
    static final MethodRef doubleToInt = conversionMethodRef(JavaType.DOUBLE, JavaType.INT);
    static final MethodRef doubleToLong = conversionMethodRef(JavaType.DOUBLE, JavaType.LONG);
    static final MethodRef doubleToFloat = conversionMethodRef(JavaType.DOUBLE, JavaType.FLOAT);




    static MethodRef conversionMethodRef(JavaType sourceType, JavaType targetType) {
        String n = "is%sTo%sExact".formatted(capitalize(sourceType.toString()), capitalize(targetType.toString()));
        JavaType c = JavaType.type(ExactConversionsSupport.class);
        return MethodRef.method(c, n, PrimitiveType.BOOLEAN, sourceType);
    }

    static String capitalize(String s) {
        return s.substring(0, 1).toUpperCase() + s.substring(1);
    }

    @DataProvider
    public static Object[][] dp() {
        return new Object[][]{
                {JavaType.INT, JavaType.BYTE, new Object[] {Byte.MIN_VALUE - 1, Byte.MIN_VALUE, Byte.MAX_VALUE,
                        Byte.MAX_VALUE + 1}, intToByte},
                {JavaType.INT, JavaType.SHORT, new Object[] {Short.MIN_VALUE - 1, Short.MIN_VALUE, Short.MAX_VALUE,
                        Short.MAX_VALUE + 1}, intToShort},
                {JavaType.INT, JavaType.CHAR, new Object[] {Character.MIN_VALUE - 1, Character.MIN_VALUE, Character.MAX_VALUE,
                        Character.MAX_VALUE + 1}, intToChar},
                // (1<<24) + 1 : first int that's not an instanceof float
                // 1<<31) - (1<<7): largest int that's an instance of float
                {JavaType.INT, JavaType.FLOAT, new Object[] {1<<24, (1<<24) + 1, (1<<31) - (1<<7), (1<<31) - (1<<7) + 1,
                        Integer.MAX_VALUE, Integer.MIN_VALUE}, intToFloat},

                {JavaType.SHORT, JavaType.BYTE, new Object[]{(short) (Byte.MIN_VALUE - 1), Byte.MIN_VALUE, Byte.MAX_VALUE,
                        (short) (Byte.MAX_VALUE + 1)}, intToByte},
                {JavaType.SHORT, JavaType.CHAR, new Object[]{Short.MIN_VALUE, (short) -1, (short) 0, Short.MAX_VALUE}, intToChar},

                {JavaType.CHAR, JavaType.BYTE, new Object[]{(char) 0, (char) Byte.MAX_VALUE, (char) (Byte.MAX_VALUE + 1)}, intToByte},
                {JavaType.CHAR, JavaType.SHORT, new Object[]{(char) 0, (char) Short.MAX_VALUE, (char) (Short.MAX_VALUE + 1)}, intToShort},

                {JavaType.LONG, JavaType.BYTE, new Object[] {Byte.MIN_VALUE - 1, Byte.MIN_VALUE, Byte.MAX_VALUE,
                        Byte.MAX_VALUE + 1}, longToByte},
                {JavaType.LONG, JavaType.SHORT, new Object[] {Short.MIN_VALUE - 1, Short.MIN_VALUE, Short.MAX_VALUE,
                        Short.MAX_VALUE + 1}, longToShort},
                {JavaType.LONG, JavaType.CHAR, new Object[] {Character.MIN_VALUE - 1, Character.MIN_VALUE, Character.MAX_VALUE,
                        Character.MAX_VALUE + 1}, longToChar},
                {JavaType.LONG, JavaType.INT, new Object[] {(long)Integer.MIN_VALUE - 1, Integer.MIN_VALUE, Integer.MAX_VALUE,
                        (long)Integer.MAX_VALUE + 1}, longToInt},
                // (1<<24) + 1 : first long that can't be represented as float
                // (1L<<63) - (1L<<39) : largest long that can be represented as float
                {JavaType.LONG, JavaType.FLOAT, new Object[] {Long.MIN_VALUE, (1L<<24), (1<<24) + 1, (1L<<63) - (1L<<39),
                        (1L<<63) - (1L<<39) + 1, Long.MAX_VALUE}, longToFloat},
                // (1L<<53) + 1 : first long that can't be represented as double
                // (1L<<63) - (1<<10) : largest long that can be represented as double
                {JavaType.LONG, JavaType.DOUBLE, new Object[] {Long.MIN_VALUE, 1L<<53, (1L<<53) + 1, (1L<<63) - (1<<10),
                        (1L<<63) - (1<<10) + 1, Long.MAX_VALUE}, longToDouble},

                {JavaType.FLOAT, JavaType.BYTE, new Object[] {Byte.MIN_VALUE - 1, Byte.MIN_VALUE, Byte.MAX_VALUE,
                        Byte.MIN_VALUE + 1}, floatToByte},
                {JavaType.FLOAT, JavaType.SHORT, new Object[] {Short.MIN_VALUE - 1, Short.MIN_VALUE, Short.MAX_VALUE,
                        Short.MAX_VALUE + 1}, floatToShort},
                {JavaType.FLOAT, JavaType.CHAR, new Object[] {Character.MIN_VALUE - 1, Character.MIN_VALUE,
                        Character.MAX_VALUE, Character.MAX_VALUE + 1}, floatToChar},
                {JavaType.FLOAT, JavaType.INT, new Object[] {Float.MIN_VALUE, Float.NEGATIVE_INFINITY, 0f,
                        Float.POSITIVE_INFINITY, Float.MAX_VALUE}, floatToInt},
                {JavaType.FLOAT, JavaType.LONG, new Object[] {Float.MIN_VALUE, Float.NEGATIVE_INFINITY, 0f,
                        Float.POSITIVE_INFINITY, Float.MAX_VALUE}, floatToLong},

                {JavaType.DOUBLE, JavaType.BYTE, new Object[] {Double.NEGATIVE_INFINITY, Double.MIN_VALUE, -0d, +0d,
                        Double.MAX_VALUE, Double.POSITIVE_INFINITY, Double.NaN}, doubleToByte},
                {JavaType.DOUBLE, JavaType.SHORT, new Object[] {Double.NEGATIVE_INFINITY, Double.MIN_VALUE, -0d, +0d,
                        Double.MAX_VALUE, Double.POSITIVE_INFINITY, Double.NaN}, doubleToShort},
                {JavaType.DOUBLE, JavaType.CHAR, new Object[] {Double.NEGATIVE_INFINITY, Double.MIN_VALUE, -0d, +0d,
                        Double.MAX_VALUE, Double.POSITIVE_INFINITY, Double.NaN}, doubleToChar},
                {JavaType.DOUBLE, JavaType.INT, new Object[] {Double.NEGATIVE_INFINITY, Double.MIN_VALUE, -0d, +0d,
                        Double.MAX_VALUE, Double.POSITIVE_INFINITY, Double.NaN}, doubleToInt},
                {JavaType.DOUBLE, JavaType.LONG, new Object[] {Double.NEGATIVE_INFINITY, Double.MIN_VALUE, -0d, +0d,
                        Double.MAX_VALUE, Double.POSITIVE_INFINITY, Double.NaN}, doubleToLong},
                {JavaType.DOUBLE, JavaType.FLOAT, new Object[] {Double.NEGATIVE_INFINITY, Double.MIN_VALUE, -0d, +0d,
                        Double.MAX_VALUE, Double.POSITIVE_INFINITY, Double.NaN}, doubleToFloat}

        };
    }

    @Test(dataProvider = "dp")
    void test(JavaType sourceType, JavaType targetType, Object[] values, MethodRef expectedConversionMethod) throws Throwable {

        var model = buildTypePatternModel(sourceType, targetType);
        model.writeTo(System.out);

        var lmodel = model.transform(OpTransformer.LOWERING_TRANSFORMER);
        lmodel.writeTo(System.out);
        Assert.assertTrue(
                containsOp(lmodel.body(), op -> op instanceof InvokeOp invOp && invOp.invokeDescriptor().equals(expectedConversionMethod))
        );

        var mh = BytecodeGenerator.generate(MethodHandles.lookup(), lmodel);

        for (Object v : values) {
            Assert.assertEquals(Interpreter.invoke(lmodel, v), mh.invoke(v));
        }
    }

    static Op findOp(Body b, Predicate<Op> p) {
        for (Block block : b.blocks()) {
            for (Op op : block.ops()) {
                if (p.test(op)) {
                    return op;
                }
                for (Body body : op.bodies()) {
                    return findOp(body, p);
                }
            }
        }
        return null;
    }

    static boolean containsOp(Body b, Predicate<Op> p) {
        return findOp(b, p) != null;
    }

    static FuncOp buildTypePatternModel(JavaType sourceType, JavaType targetType) {
        // builds the model of:
        // static boolean f(sourceType a) { return a instanceof targetType _; }
        return func("f", functionType(JavaType.BOOLEAN, sourceType)).body(fblock -> {

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
        if (List.of(PrimitiveType.BYTE, PrimitiveType.SHORT, PrimitiveType.CHAR, PrimitiveType.INT).contains(t)) {
            return 0;
        } else if (PrimitiveType.LONG.equals(t)) {
            return 0L;
        } else if (PrimitiveType.FLOAT.equals(t)) {
            return 0f;
        } else if (PrimitiveType.DOUBLE.equals(t)) {
            return 0d;
        } else if (PrimitiveType.BOOLEAN.equals(t)) {
            return false;
        }
        return null;
    }
}
