import org.testng.Assert;
import org.testng.annotations.DataProvider;
import org.testng.annotations.Test;

import java.lang.invoke.MethodHandles;
import java.lang.reflect.code.*;
import java.lang.reflect.code.bytecode.BytecodeGenerator;
import java.lang.reflect.code.interpreter.Interpreter;
import java.lang.reflect.code.op.ExtendedOp;
import java.lang.reflect.code.type.JavaType;
import java.lang.reflect.code.type.PrimitiveType;
import java.util.List;

import static java.lang.reflect.code.op.CoreOp.*;
import static java.lang.reflect.code.op.ExtendedOp.*;
import static java.lang.reflect.code.type.FunctionType.*;

/*
 * @test
 * @run testng TestPrimitiveTypePatterns
 * @enablePreview
 */

public class TestPrimitiveTypePatterns {

    @DataProvider
    public static Object[][] patternsOfInt() {
        return new Object[][]{
                {JavaType.BYTE, new int[] {Byte.MIN_VALUE, Byte.MAX_VALUE, Byte.MIN_VALUE -1, Byte.MAX_VALUE + 1}},
                {JavaType.SHORT, new int[] {Short.MIN_VALUE, Short.MAX_VALUE, Short.MIN_VALUE -1, Short.MAX_VALUE + 1}},
                {JavaType.CHAR, new int[] {Character.MIN_VALUE, Character.MAX_VALUE, Character.MIN_VALUE -1, Character.MAX_VALUE + 1}},
        };
    }

    @Test(dataProvider = "patternsOfInt")
    void testPatternsOfInt(JavaType targetType, int[] values) throws Throwable {

        var model = buildTypePatternModel(JavaType.INT, targetType);
        var lmodel = model.transform(OpTransformer.LOWERING_TRANSFORMER);

        var mh = BytecodeGenerator.generate(MethodHandles.lookup(), lmodel);

        for (int v : values) {
            Assert.assertEquals(Interpreter.invoke(lmodel, v), mh.invoke(v));
        }
    }

    static FuncOp buildTypePatternModel(JavaType sourceType, JavaType targetType) {
        // builds the model of:
        // static boolean f(sourceType a) { return a instanceof targetType _; }
        return func("f", functionType(JavaType.BOOLEAN, sourceType)).body(fblock -> {

            var param = fblock.op(var(fblock.parameters().get(0)));
            var paramVal = fblock.op(varLoad(param));

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
