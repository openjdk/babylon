package shade.shaders;

import hat.types.vec2;
import hat.types.vec4;
import shade.Shader;
import shade.Uniforms;

public class Shader1 implements Shader {

    @Override
    public vec4 mainImage(Uniforms uniforms, vec4 fragColor, vec2 fragCoord) {

            int w = uniforms.iResolution().x();
            int wDiv3 = uniforms.iResolution().x() / 3;
            int h = uniforms.iResolution().y();
            int hDiv3 = uniforms.iResolution().y() / 3;
            boolean midx = (fragCoord.x() > wDiv3 && fragCoord.x() < (w - wDiv3));
            boolean midy = (fragCoord.y() > hDiv3 && fragCoord.y() < (h - hDiv3));
            if (uniforms.iMouse().x() > wDiv3) {
                if (midx && midy) {
                    return vec4.vec4(fragCoord.x(), .0f, fragCoord.y(), 0.f);
                } else {
                    return vec4.vec4(0f, 0f, .5f, 0f);
                }
            } else {
                return vec4.vec4(1f, 1f, .5f, 0f);
            }
        }
}
