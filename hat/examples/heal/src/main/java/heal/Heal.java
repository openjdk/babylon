package heal;

import hat.Accelerator;
import hat.backend.Backend;

import javax.swing.JFrame;
import java.awt.Rectangle;
import java.lang.invoke.MethodHandles;

public class Heal {
    public static void main(String[] args) {
        Accelerator accelerator = new Accelerator(MethodHandles.lookup(), Backend.FIRST);
        ImageData imageData = ImageData.of(
                Viewer.class.getResourceAsStream("/images/bolton.png")
        );
        JFrame f = new JFrame("Healing Brush");
        f.setBounds(new Rectangle(imageData.width(), imageData.height()));
        f.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        Viewer viewerDisplay = new Viewer(imageData, accelerator);
        f.setContentPane(viewerDisplay);
        f.validate();
        f.setVisible(true);
    }

}
