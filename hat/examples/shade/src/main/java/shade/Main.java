package shade;

import hat.Accelerator;
import hat.backend.Backend;
import hat.types.F32;
import hat.types.vec2;
import hat.types.vec3;
import shade.shaders.AcesShader;
import shade.shaders.AnimShader;
import shade.shaders.GroovyShader;
import shade.shaders.IntroShader;
import shade.shaders.MobiusShader;
import shade.shaders.MouseSensitiveShader;
import shade.shaders.RandShader;
import shade.shaders.SeaScapeShader;
import shade.shaders.SpiralShader;
import shade.shaders.SquareWaveShader;
import shade.shaders.TruchetShader;
import shade.shaders.TutorialShader;
import shade.shaders.WavesShader;

import java.io.IOException;
import java.lang.invoke.MethodHandles;

import static hat.types.vec4.vec4;

public class Main {


    private enum ShaderEnum {
        Blue((uniform, fragColor, fragCoord) -> {
            return vec4(0f, 0f, 1f, 0f);
        }),
        Gradient((uniforms, fragColor, fragCoord) -> {
            var fResolution = vec3.xy(uniforms.iResolution());
            float fFrame = uniforms.iFrame();
            var uv = vec2.div(fragCoord, fResolution);
            return vec4(uv.x(), uv.y(), F32.max(fFrame / 100f, 1f), 0f);
        }),

        MouseSensitive(new MouseSensitiveShader()),
        Groovy(new GroovyShader()),
        Rand(new RandShader()),
        Spiral(new SpiralShader()),
        Aces(new AcesShader()),
        Anim(new AnimShader()),
        Waves(new WavesShader()),
        Intro(new IntroShader()),
        Tutorial(new TutorialShader()),
        SquareWave(new SquareWaveShader()),
        SeaScape(new SeaScapeShader()),
        Mobius(new MobiusShader()),
        Truchet(new TruchetShader());
        Shader shader;

        ShaderEnum(Shader shader) {
            this.shader = shader;
        }
    }

    static void main(String[] args) throws IOException {
        new ShaderApp(Config.of(
                Boolean.getBoolean("hat") ? new Accelerator(MethodHandles.lookup(), Backend.FIRST) : null,
                Integer.parseInt(System.getProperty("width", System.getProperty("size", "1024"))),
                Integer.parseInt(System.getProperty("height", System.getProperty("size", "1024"))),
                Integer.parseInt(System.getProperty("targetFps", "12")),
                ShaderEnum.valueOf(System.getProperty("shader", "Tutorial")).shader
        ));
    }
}
