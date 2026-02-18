import hat.types.vec4;
import static hat.types.vec4.*;

static void main(String[] args) throws IOException {
    shade.ShaderFrame.of( 1024, 1024, 30, "Blue",
        (uniforms, FragColor, fragCoord) -> 
           vec4(0f,0f,5f,0f) 
    );
}
