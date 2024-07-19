package experiments;

import hat.Accelerator;
import hat.backend.Backend;
import hat.buffer.S32Array;

import java.lang.invoke.MethodHandles;
/*
https://github.com/openjdk/babylon/tree/code-reflection/test/jdk/java/lang/reflect/code
*/

public class LambdaTest {
    public static void main(String[] args) {
        Accelerator accelerator = new Accelerator(MethodHandles.lookup(), Backend.FIRST_NATIVE);

        // TODO: create a test case for these **/
        S32Array s32Array = S32Array.create(accelerator, 10);
    }

}
