
import hat.backend.Backend;

import javax.swing.JFrame;
import java.awt.Rectangle;
import hat.Accelerator;
import hat.backend.Backend;
import hat.types.ivec2;
import hat.types.vec2;
import hat.types.vec4;
import shade.Controls;
import shade.FloatImagePanel;
import shade.Shader;
import shade.Uniforms;
import static hat.types.F32.*;
import static hat.types.vec4.*;
import static hat.types.vec2.*;
import shade.Main;

static void main(String[] args) throws IOException {
    var acc =  new Accelerator(MethodHandles.lookup(), Backend.FIRST);
    Shader shader = (uniforms, inFragColor, fragCoord) -> {
         return vec4(0f,0f,5f,0f);
    };
    Controls controls = new Controls();
    JFrame frame = new JFrame();
    frame.setJMenuBar(controls.menu.menuBar());
    int width = 1024;
    int height = 1024;

    FloatImagePanel imagePanel = new FloatImagePanel(acc, controls, width, height, false, shader, 30);
    frame.setBounds(new Rectangle(width + 100, height + 100));
    frame.setContentPane(imagePanel);
    frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
    frame.setVisible(true);
    imagePanel.start();
}
