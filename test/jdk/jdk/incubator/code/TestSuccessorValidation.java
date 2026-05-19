import jdk.incubator.code.Block;
import jdk.incubator.code.Body;
import jdk.incubator.code.dialect.core.CoreOp;
import jdk.incubator.code.dialect.core.CoreType;
import jdk.incubator.code.dialect.core.FunctionType;
import jdk.incubator.code.dialect.java.JavaType;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

/*
 * @test
 * @modules jdk.incubator.code
 * @run junit TestSuccessorValidation
 */
public class TestSuccessorValidation {
    @Test
    void testInvalid() {
        Assertions.assertThrows(IllegalStateException.class, TestSuccessorValidation::numArgsGtNumParams);
    }

    private static CoreOp.FuncOp numArgsGtNumParams() {
        return CoreOp.func("invalid", CoreType.functionType(JavaType.INT, JavaType.INT)).body(eb -> {
            Block.Builder b1 = eb.block(JavaType.INT);
            b1.op(CoreOp.return_(b1.parameters().get(0)));

            eb.op(CoreOp.branch(b1.reference(eb.parameters().get(0), eb.parameters().get(0))));
        });
    }

    @Test
    void testValid() {
        numArgsLeqNumParams();
    }

    private static CoreOp.FuncOp numArgsLeqNumParams() {
        return CoreOp.func("valid", CoreType.functionType(JavaType.INT, JavaType.INT, JavaType.BOOLEAN))
                .body(eb -> {
                    Block.Builder b1 = eb.block(JavaType.INT);
                    b1.op(CoreOp.return_(b1.parameters().get(0)));

                    eb.op(CoreOp.conditionalBranch(
                            eb.parameters().get(1),
                            b1.reference(eb.parameters().get(0)),
                            b1.reference()
                    ));
                });
    }
}
