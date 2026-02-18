import hat.backend.Backend;

import shade.Shader;
import hat.types.vec3;
import hat.types.vec2;
import static hat.types.F32.*;
import static hat.types.vec4.*;
import static hat.types.vec2.*;
import static hat.types.vec3.*;


static void main(String[] args) throws IOException {
    Shader shader = (uniforms, inFragColor, fragCoord) -> {
            var uv = div(fragCoord,vec3.xy(uniforms.iResolution()));  // normalize between 0->1 vec2 uv = fragCoord/iResolution.xy
            float frame= max(uniforms.iFrame()/1000f,1f);
            return vec4(uv.x(),uv.y(),frame,0f);
    };
    shade.ShaderFrame.of( 1024, 1024, 30, "Gradient", shader);
}
