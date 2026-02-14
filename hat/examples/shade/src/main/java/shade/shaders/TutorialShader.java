package shade.shaders;

import hat.types.F32;
import hat.types.vec2;
import hat.types.vec3;
import hat.types.vec4;
import shade.Shader;
import shade.Uniforms;

/* This animation is the material of my first youtube tutorial about creative
   coding, which is a video in which I try to introduce programmers to GLSL
   and to the wonderful world of shaders, while also trying to share my recent
   passion for this community.
                                       Video URL: https://youtu.be/f4s1h2YETNY


//https://iquilezles.org/articles/palettes/
vec3 palette( float t ) {
    vec3 a = vec3(0.5, 0.5, 0.5);
    vec3 b = vec3(0.5, 0.5, 0.5);
    vec3 c = vec3(1.0, 1.0, 1.0);
    vec3 d = vec3(0.263,0.416,0.557);

    return a + b*cos( 6.28318*(c*t+d) );
}

        //https://www.shadertoy.com/view/mtyGWy
        void mainImage( out vec4 fragColor, in vec2 fragCoord ) {
            vec2 uv = (fragCoord * 2.0 - iResolution.xy) / iResolution.y;
            vec2 uv0 = uv;
            vec3 finalColor = vec3(0.0);

            for (float i = 0.0; i < 4.0; i++) {
                uv = fract(uv * 1.5) - 0.5;

                float d = length(uv) * exp(-length(uv0));

                vec3 col = palette(length(uv0) + i*.4 + iTime*.4);

                d = sin(d*8. + iTime)/8.;
                d = abs(d);

                d = pow(0.01 / d, 1.2);

                finalColor += col * d;
            }

            fragColor = vec4(finalColor, 1.0);
        }
 */
//https://www.shadertoy.com/view/mtyGWy
public class TutorialShader implements Shader {
    vec3 palette( float t ) {
        vec3 a = vec3.vec3(0.5f, 0.5f, 0.5f);
        vec3 b = vec3.vec3(0.5f, 0.5f, 0.5f);
        vec3 c = vec3.vec3(1.0f, 1.0f, 1.0f);
        vec3 d = vec3.vec3(0.263f,0.416f,0.557f);

        vec3 cxt = vec3.mul(c,vec3.vec3(t));
        vec3 cxtplusd = vec3.add(cxt,d);
         return vec3.add(a, vec3.mul(b,
                vec3.cos(vec3.mul(cxtplusd, vec3.vec3(6.28318f))))
        );
    }
    @Override
    public vec4 mainImage(Uniforms uniforms, vec4 fragColor, vec2 fragCoord) {

        vec2 uv = vec2.div(
                vec2.sub(
                        vec2.mul(
                                fragCoord, 2f),vec2.vec2(uniforms.iResolution()
                        )
                ), uniforms.iResolution().y()
        );
        vec2 uv0 = uv;
        vec3 finalColor = vec3.vec3(0f);

        for (float i = 0f; i < 4f; i++) {
            uv = vec2.sub(vec2.fract(vec2.mul(uv,1.5f)), vec2.vec2(0.5f));

            float d = vec2.length(uv) * F32.exp(-vec2.length(uv0));

            vec3 col = palette(vec2.length(uv0) + i * .4f + uniforms.iTime() * .4f);

            d = F32.sin(d * 8f + uniforms.iTime()) / 8f;
            d = F32.abs(d);

            d = F32.pow(0.01f / d, 1.2f);

            finalColor  = vec3.add(finalColor, vec3.mul(col, d));
        }

        fragColor = vec4.vec4(finalColor, 1.0f);
        return fragColor;
    }
}
