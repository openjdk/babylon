package heal;

import java.awt.Color;
import java.awt.Graphics2D;
import java.awt.Polygon;
import java.awt.image.BufferedImage;
import java.awt.image.DataBufferInt;
import java.util.Arrays;

class Mask {
    public final int[] data;
    public final int width;
    public final int height;

    public Mask(Path path) {
        width = path.width()+2;
        height = path.height()+2;
        Polygon polygon = new Polygon();
        for (int i = 0; i < path.xyList.length(); i++) {
            XYList.XY xy = path.xyList.xy(i);
            polygon.addPoint(xy.x() - path.x1() + 1, xy.y() - path.y1() + 1);
        }
        BufferedImage maskImg = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);
        data = ((DataBufferInt) (maskImg.getRaster().getDataBuffer())).getData();
        Arrays.fill(data, 0);
        Graphics2D g = maskImg.createGraphics();
        g.setColor(Color.WHITE);
        g.fillPolygon(polygon);
    }
}
