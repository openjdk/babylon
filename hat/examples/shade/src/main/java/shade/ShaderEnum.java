package shade;

import hat.types.F32;
import shade.shaders.AcesShader;
import shade.shaders.AnimShader;
import shade.shaders.IntroShader;
import shade.shaders.RandShader;
import shade.shaders.Shader1;
import shade.shaders.Shader25;
import shade.shaders.SpiralShader;
import shade.shaders.TutorialShader;
import shade.shaders.WavesShader;

import static hat.types.vec2.vec2;
import static hat.types.vec4.vec4;

enum ShaderEnum {
    Blue((uniform, fragColor, fragCoord) -> {
        return vec4(0f, 0f, 1f, 0f);
    }),
    Gradient((uniforms, fragColor, fragCoord) -> {
        var fResolution = vec2(uniforms.iResolution());
        float fFrame = uniforms.iFrame();
        var uv = fragCoord.div(fResolution);
        return vec4(uv.x(), uv.y(), F32.max(fFrame / 100f, 1f), 0f);
    }),

    Shader1(new Shader1()),
    Shader25(new Shader25()),
    Rand(new RandShader()),
    Spiral(new SpiralShader()),
    Aces(new AcesShader()),
    Anim(new AnimShader()),
    Waves(new WavesShader()),
    Intro(new IntroShader()),
    Tutrorial(new TutorialShader());
    Shader shader;

    ShaderEnum(Shader shader) {
        this.shader = shader;
    }
}
