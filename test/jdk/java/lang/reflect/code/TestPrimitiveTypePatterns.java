import org.testng.Assert;
import org.testng.annotations.DataProvider;
import org.testng.annotations.Test;

import java.lang.invoke.MethodHandles;
import java.lang.reflect.Method;
import java.lang.reflect.code.*;
import java.lang.reflect.code.bytecode.BytecodeGenerator;
import java.lang.reflect.code.interpreter.Interpreter;
import java.lang.reflect.code.op.CoreOp;
import java.lang.reflect.code.type.JavaType;
import java.lang.reflect.code.type.PrimitiveType;
import java.lang.runtime.CodeReflection;
import java.util.ArrayList;
import java.util.List;
import java.util.Optional;
import java.util.stream.Stream;

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

        var genericModel = getCodeModel("f");

        var model = buildTypePatternModel(genericModel, JavaType.INT, targetType);
        model.writeTo(System.out);

        var lmodel = model.transform(OpTransformer.LOWERING_TRANSFORMER);
        lmodel.writeTo(System.out);

        var mh = BytecodeGenerator.generate(MethodHandles.lookup(), lmodel);

        for (int v : values) {
            Assert.assertEquals(Interpreter.invoke(lmodel, v), mh.invoke(v));
        }
    }

    @CodeReflection
    // works as generic model that will be transformed to test conversion from a sourceType to targetType
    static boolean f(byte b) {
        return b instanceof byte _;
    }

    static FuncOp buildTypePatternModel(FuncOp genericModel, JavaType sourceType, JavaType targetType) {
        List<VarOp> patternVariables = getPatternVariables(genericModel);
        return func(genericModel.funcName(), functionType(JavaType.BOOLEAN, sourceType)).body(fblock -> {
            fblock.transformBody(genericModel.body(), fblock.parameters(), ((block, op) -> {
               if (op instanceof ConstantOp cop && cop.parentBlock().nextOp(cop) instanceof VarOp vop &&
                       patternVariables.contains(vop)) {
                   var newCop = constant(targetType, defaultValue(targetType));
                   block.op(newCop);
                   block.context().mapValue(cop.result(), newCop.result());
               } else if (op instanceof PatternOps.TypePatternOp tpop) {
                   var newTpop = typePattern(targetType, tpop.bindingName());
                   block.op(newTpop);
                   block.context().mapValue(tpop.result(), newTpop.result());
               } else {
                   block.op(op);
               }
               return block;
           }));
        });
    }

    static List<VarOp> getPatternVariables(FuncOp f) {
        return f.traverse(new ArrayList<>(), (l, e) -> {
            if (e instanceof Block b && b.parentBody().parentOp() instanceof PatternOps.MatchOp) {
                b.ops().forEach(op -> {
                    if (op instanceof VarAccessOp.VarStoreOp vsop) {
                        l.add(vsop.varOp());
                    }
                });
            }
            return l;
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

    private static CoreOp.FuncOp getCodeModel(String methodName) {
        Optional<Method> om = Stream.of(TestPrimitiveTypePatterns.class.getDeclaredMethods())
                .filter(m -> m.getName().equals(methodName))
                .findFirst();

        return om.get().getCodeModel().get();
    }
}
