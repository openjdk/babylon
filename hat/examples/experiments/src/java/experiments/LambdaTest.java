package experiments;

import hat.Accelerator;

import hat.backend.Backend;
import hat.buffer.S32Array;

import java.lang.invoke.MethodHandles;
/*
https://github.com/openjdk/babylon/tree/code-reflection/test/jdk/java/lang/reflect/code
*/

public class LambdaTest {
    public static void main(String[] args ){
        Accelerator accelerator = new Accelerator(MethodHandles.lookup(), Backend.FIRST_NATIVE);

            // TODO: create a test case for these **/
            S32Array s32Array = S32Array.create(accelerator, 10);
/*
            accelerator.compute(cc->ccargS32Array) -> {
                var range = cc.accelerator.range(argS32Array.length());
                DispatchArity.KernelSam1<S32Array> kernel = (id, kArgS32Array) -> {
                    kArgS32Array.array(id.x, kArgS32Array.array(id.x) * 2);
                };
                cc.dispatchKernel(kernel, range, s32Array);
            }, s32Array);

            accelerator.compute((cc, argS32Array) -> {
                var range = cc.accelerator.range(argS32Array.length());
                cc.dispatchKernel(
                        (id, kArgS32Array) -> {
                            kArgS32Array.array(id.x, kArgS32Array.array(id.x) * 2);
                        }, range, argS32Array);
            }, s32Array);

            accelerator.compute((cc, argS32Array) ->
                            cc.dispatchKernel((id, kArgS32Array) ->
                                    kArgS32Array.array(id.x, kArgS32Array.array(id.x) * 2), cc.accelerator.range(argS32Array.length()), argS32Array)
                    , s32Array);

 */
        }

}
