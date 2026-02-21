package shade;

import hat.ComputeContext;
import hat.ComputeContext.Kernel;
import hat.KernelContext;
import hat.NDRange;
import hat.buffer.F32Array;
import hat.types.vec2;
import hat.types.vec4;
import jdk.incubator.code.Reflect;
import optkl.ifacemapper.MappableIface;

import static hat.types.vec2.vec2;
import static hat.types.vec4.vec4;

public class HATShader {
    @Reflect
    public static vec4 mainImage(@MappableIface.RO Uniforms uniforms, vec4 fragColor, vec2 fragCoord) {
        return vec4(0f, 0f, 1f, 0f);
    }

    @Reflect
    public static void penumbra(@MappableIface.RO KernelContext kc, @MappableIface.RO Uniforms uniforms, @MappableIface.RO F32Array image) {
        if (kc.gix < kc.gsx) {
            // The image is essentially a vec3 array
            int width = (int) uniforms.iResolution().x();
            int height = (int) uniforms.iResolution().y();
            var fragCoord = vec2(kc.gix % width, kc.gix / width);
            long offset = ((long) kc.gsx * height * 3) + (kc.gix * 3L);
            var fragColor = mainImage(uniforms,
                    vec4(image.array(offset + 0), image.array(offset + 1), image.array(offset + 2), 0f),
                    fragCoord);
            image.array(offset + 0, fragColor.x());
            image.array(offset + 1, fragColor.y());
            image.array(offset + 2, fragColor.z());
        }
    }


    @Reflect
    static public void compute(final ComputeContext computeContext, @MappableIface.RO Uniforms uniforms, @MappableIface.RO F32Array image, int width, int height) {
        computeContext.dispatchKernel(
                NDRange.of1D(width * height),               //0..S32Array2D.size()
                (@Reflect Kernel) kc -> penumbra(kc, uniforms, image));
    }
}
