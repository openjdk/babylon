package oracle.code.hat;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;

import hat.Accelerator;
import hat.ComputeContext;
import hat.KernelContext;
import hat.backend.Backend;
import hat.buffer.S32Array;
import hat.ifacemapper.MappableIface;
import jdk.incubator.code.CodeReflection;
import org.junit.jupiter.api.Test;

import java.lang.invoke.MethodHandles;

public class AppTest {

    @CodeReflection
    public static int squareit(int v) {
        return  v * v;

    }

    @CodeReflection
    public static void squareKernel(@MappableIface.RO KernelContext kc, @MappableIface.RW S32Array s32Array) {
        if (kc.x < kc.gsx){
            int value = s32Array.array(kc.x);       // arr[cc.x]
            s32Array.array(kc.x, squareit(value));  // arr[cc.x]=value*value
        }
    }

    @CodeReflection
    public static void square(@MappableIface.RO ComputeContext cc, @MappableIface.RW S32Array s32Array) {
        cc.dispatchKernel(s32Array.length(),
                kc -> squareKernel(kc, s32Array)
        );
    }

    @Test
    public void testHelloHat() {

        final int size = 64;
        var accelerator = new Accelerator(MethodHandles.lookup(), Backend.FIRST);
        assertNotNull(accelerator);

        var array = S32Array.create(accelerator, size);
        assertNotNull(array);

        // Initialize array
        for (int i = 0; i < array.length(); i++) {
            array.array(i, i);
        }

        // Blocking call
        accelerator.compute(cc -> AppTest.square(cc, array));

        S32Array test = S32Array.create(accelerator, size);
        assertNotNull(test);

        for (int i = 0; i < test.length(); i++) {
            test.array(i, squareit(i));
        }

        for (int i = 0; i < test.length(); i++) {
            assertEquals(test.array(i), array.array(i));
        }

    }
}
