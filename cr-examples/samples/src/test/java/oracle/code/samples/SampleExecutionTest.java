package oracle.code.samples;

import org.junit.jupiter.api.Test;

class SampleExecutionTest {

    @Test
    void execute() {
        DialectFMAOp.main();
        DialectWithInvoke.main();
        DynamicFunctionBuild.main();
        HelloCodeReflection.main();
        InlineExample.main();
        MathOptimizer.main();
        MathOptimizerWithInlining.main();
    }
}