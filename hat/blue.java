import hat.Accelerator;
import hat.backend.Backend;
import shade.types.ivec2;
import shade.types.vec2;
import shade.types.vec4;
import static shade.types.vec4.*;
import static shade.types.vec2.*;
import shade.Main;
import shade.Main.Shader;

static void main(String[] args) throws IOException {
    var acc =  new Accelerator(MethodHandles.lookup(), Backend.FIRST);
    var shader = new Shader() {
        @Override
        public vec4 mainImage(vec4 inFragColor, vec2 fragCoord, int iTime, int iFrame, ivec2 iMouse) {
            return vec4(0f,0f,5f,0f);
        }
    };
    new Main(acc, 1024, 1024, shader);
}
