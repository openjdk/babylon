import hat.Accelerator;
import hat.backend.Backend;
import shade.types.ivec2;
import shade.types.vec2;
import shade.types.vec4;
import shade.types.Shader;
import shade.types.Uniforms;
import static shade.types.vec4.*;
import static shade.types.vec2.*;
import shade.Main;

static void main(String[] args) throws IOException {
    var acc =  new Accelerator(MethodHandles.lookup(), Backend.FIRST);
    Shader shader = (uniforms, inFragColor, fragCoord) -> {
            var uv = fragCoord.div(vec2(uniforms.iResolution()));  // normalize between 0->1 vec2 uv = fragCoord/iResolution.xy
            float frame= Math.max(uniforms.iFrame()/1000f,1f);
            return vec4(uv.x(),uv.y(),frame,0f);
    };
    new Main(acc, 1024, 1024, shader);
}
