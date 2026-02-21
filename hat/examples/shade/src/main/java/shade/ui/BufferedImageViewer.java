package shade.ui;

import javax.swing.JComponent;
import java.awt.Graphics;
import java.awt.Point;
import java.awt.event.MouseEvent;
import java.awt.event.MouseMotionListener;
import java.awt.image.BufferedImage;
import java.util.function.Consumer;

public class BufferedImageViewer extends JComponent {
    private final BufferedImage bufferedImage;

    public BufferedImageViewer(BufferedImage bufferedImage, Consumer<Point> mouseLocationConsumer) {
        this.bufferedImage = bufferedImage;
        //this.setPreferredSize(new Dimension(floatImage.width(),floatImage.height()));
        this.addMouseMotionListener(new MouseMotionListener() {
            @Override
            public void mouseDragged(MouseEvent e) {
                mouseLocationConsumer.accept(e.getPoint());
            }

            @Override
            public void mouseMoved(MouseEvent e) {
                mouseLocationConsumer.accept(e.getPoint());
            }
        });
    }

    @Override
    public void paint(Graphics graphics) {
        synchronized (this) {
            graphics.drawImage(bufferedImage, 0, 0, this.getWidth(), this.getHeight(), null);
        }
    }

}
