package nbody;

import nbody.opencl.NBody;
import wrap.glwrap.GLTexture;

import java.io.IOException;
import java.lang.foreign.Arena;

public class Main {
    public static void main(String[] args) throws IOException {
        int particleCount = args.length > 2 ? Integer.parseInt(args[2]) : 32768;
        NBodyGLWindow.Mode mode = NBodyGLWindow.Mode.of(args.length > 3 ? args[3] : NBodyGLWindow.Mode.OpenCL.toString());
        System.out.println("mode" + mode);
        try (var arena = mode.equals(NBodyGLWindow.Mode.JavaMT4) || mode.equals(NBodyGLWindow.Mode.JavaMT) ? Arena.ofShared() : Arena.ofConfined()) {
            var particleTexture = new GLTexture(arena, NBody.class.getResourceAsStream("/particle.png"));
            new NBody.CLNBodyGLWindow( arena, 1000, 1000, particleTexture, particleCount, mode).bindEvents().mainLoop();
        }
    }
}
