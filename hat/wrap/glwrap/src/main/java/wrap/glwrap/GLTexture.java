package wrap.glwrap;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.awt.image.DataBufferByte;
import java.io.IOException;
import java.io.InputStream;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;


public class GLTexture {
    public final Arena arena;
    public final MemorySegment data;
    public final int width;
    public final int height;
    public int idx;

    public GLTexture(Arena arena, InputStream textureStream) {
        this.arena = arena;
        BufferedImage img = null;
        try {
            img = ImageIO.read(textureStream);
            this.width = img.getWidth();
            this.height = img.getHeight();
            BufferedImage image = new BufferedImage(width, height, BufferedImage.TYPE_4BYTE_ABGR_PRE);
            image.getGraphics().drawImage(img, 0, 0, null);
            var raster = image.getRaster();
            var dataBuffer = raster.getDataBuffer();
            data = arena.allocateFrom(ValueLayout.JAVA_BYTE, ((DataBufferByte) dataBuffer).getData());
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }
}
