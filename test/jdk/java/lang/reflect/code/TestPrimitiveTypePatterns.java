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

    static final MethodRef intToByte = MethodRef.method(ExactConversionsSupport.class, "isIntToByteExact",
            boolean.class, int.class);
    static final MethodRef intToShort = MethodRef.method(ExactConversionsSupport.class, "isIntToShortExact",
            boolean.class, int.class);
    static final MethodRef intToChar = MethodRef.method(ExactConversionsSupport.class, "isIntToCharExact",
            boolean.class, int.class);
    static final MethodRef intToFloat = MethodRef.method(ExactConversionsSupport.class, "isIntToFloatExact",
            boolean.class, int.class);

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
