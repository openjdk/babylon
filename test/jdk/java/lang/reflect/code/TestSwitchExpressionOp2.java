import org.testng.Assert;
import org.testng.annotations.Test;

import java.io.IOException;
import java.io.OutputStream;
import java.io.StringWriter;
import java.lang.reflect.Method;
import java.lang.reflect.code.Block;
import java.lang.reflect.code.OpTransformer;
import java.lang.reflect.code.op.CoreOp;
import java.lang.reflect.code.type.JavaType;
import java.lang.reflect.code.writer.OpWriter;
import java.lang.runtime.CodeReflection;
import java.nio.charset.StandardCharsets;
import java.util.Optional;
import java.util.stream.Stream;

/*
* @test
* @run testng TestSwitchExpressionOp2
* */
public class TestSwitchExpressionOp2 {

    sealed interface S permits A, B { }
    static final class A implements S { }
    record B(int i) implements S { }
    @CodeReflection
     static int f(S s) {
        return switch (s) {
            case A a -> 1;
            case B b -> 2;
        };
    }

    @Test
    void test() {
        CoreOp.FuncOp lf = lower("f");

        Block lastBlock = lf.body().blocks().getLast();
        Assert.assertTrue(lastBlock.terminatingOp() instanceof CoreOp.ThrowOp t &&
                t.argument().type().equals(JavaType.type(MatchException.class)));

        // @@@ test with separate compilation later on
    }

    private static CoreOp.FuncOp lower(String methodName) {
        return lower(getCodeModel(methodName));
    }

    private static CoreOp.FuncOp lower(CoreOp.FuncOp f) {
        writeModel(f, System.out, OpWriter.LocationOption.DROP_LOCATION);

        CoreOp.FuncOp lf = f.transform(OpTransformer.LOWERING_TRANSFORMER);
        writeModel(lf, System.out, OpWriter.LocationOption.DROP_LOCATION);

        return lf;
    }

    private static void writeModel(CoreOp.FuncOp f, OutputStream os, OpWriter.Option... options) {
        StringWriter sw = new StringWriter();
        new OpWriter(sw, options).writeOp(f);
        try {
            os.write(sw.toString().getBytes(StandardCharsets.UTF_8));
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    private static CoreOp.FuncOp getCodeModel(String methodName) {
        Optional<Method> om = Stream.of(TestSwitchExpressionOp2.class.getDeclaredMethods())
                .filter(m -> m.getName().equals(methodName))
                .findFirst();

        return om.get().getCodeModel().get();
    }
}
