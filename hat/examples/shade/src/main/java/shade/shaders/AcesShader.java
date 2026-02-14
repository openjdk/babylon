package shade.shaders;

import hat.types.F32;
import hat.types.mat3;
import hat.types.vec2;
import hat.types.vec3;
import hat.types.vec4;
import shade.Shader;
import shade.Uniforms;

import static hat.types.vec3.clamp;

public class AcesShader implements Shader {
    static vec3 aces_tonemap(vec3 color) {
        mat3 m1 = mat3.mat3(
                0.59719f, 0.07600f, 0.02840f,
                0.35458f, 0.90834f, 0.13383f,
                0.04823f, 0.01566f, 0.83777f
        );
        mat3 m2 = mat3.mat3(
                1.60475f, -0.10208f, -0.00327f,
                -0.53108f, 1.10813f, -0.07276f,
                -0.07367f, -0.00605f, 1.07602f
        );
        vec3 v = color.mul(m1);
        vec3 a = v.mul(v.add(+0.0245786f)).sub(0.000090537f);
        vec3 b = v.mul((v.mul(0.983729f).add(0.4329510f)).add(0.238081f));
        var aOverBMulM2 = a.div(b).mul(m2);
        return vec3.clamp(vec3.pow(aOverBMulM2, 1.0f / 2.2f), 0f, 1f);
    }
    @Override
    public vec4 mainImage(Uniforms uniforms, vec4 fragColor, vec2 fragCoord) {
        // https://www.shadertoy.com/view/XsGfWV
        vec2 position = fragCoord.div(vec2.vec2(uniforms.iResolution())).mul(2f).sub(1f); //fragCoord/iResolution.xy)* 2.0 - 1.0;
        position = vec2.vec2(position.x() + uniforms.iTime() * 0.2f, position.y()); // position.x += iTime * 0.2;

        //vec3 color = pow(
        //     sin(
        //        position.x * 4.0 + vec3(0.0, 1.0, 2.0) * 3.1415 * 2.0 / 3.0
        //     ) * 0.5 + 0.5,
        //     vec3(2.0)
        //     ) * (exp(
        //              abs(position.y) * 4.0
        //              ) - 1.0);
        vec3 v012 = vec3.vec3(0f, 1f, 2f);
        vec3 v0123x2Pi = v012.mul(F32.PI).mul(2f);
        vec3 v0123x2PiDiv3 = v0123x2Pi.div(3f);
        vec3 sinCoef = v0123x2PiDiv3.add(position.x() * 4);
        vec3 color = vec3.pow(
                        vec3.sin(sinCoef).mul(0.5f).add(0.5f),
                        2f
                )
                .mul(
                        F32.exp(
                                F32.abs(position.y()) * 4f
                        ) - 1f
                );
        if (position.y() < 0f) {
            color = aces_tonemap(color);
        }
        fragColor = vec4.vec4(clamp(color, 0f, 1f), 1.0f);
        return fragColor;
    }
}
