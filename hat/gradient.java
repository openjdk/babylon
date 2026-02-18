import hat.backend.Backend;

import javax.swing.JFrame;
import java.awt.Rectangle;
import hat.Accelerator;
import hat.backend.Backend;
import hat.types.ivec2;
import hat.types.vec2;
import shade.Shader;
import static hat.types.F32.*;
import static hat.types.vec4.*;
import static hat.types.vec2.*;


static void main(String[] args) throws IOException {
    Shader shader = (uniforms, inFragColor, fragCoord) -> {
            var uv = div(fragCoord,vec2(uniforms.iResolution()));  // normalize between 0->1 vec2 uv = fragCoord/iResolution.xy
            float frame= max(uniforms.iFrame()/1000f,1f);
            return vec4(uv.x(),uv.y(),frame,0f);
    };
    shade.ShaderFrame.of( 1024, 1024, 30, "Gradient", shader);
}
