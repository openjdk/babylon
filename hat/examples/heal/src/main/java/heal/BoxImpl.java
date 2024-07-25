package heal;

import java.awt.Rectangle;

public class BoxImpl implements Box{
    final Rectangle rectangle;
    BoxImpl(int x1, int y1, int x2, int y2) {
         rectangle =new Rectangle(x1,y1,x2-x1,y2-y1);
    }


    @Override
    public int x1() {
        return rectangle.x;
    }

    @Override
    public void x1(int x1) {
        rectangle.x=x1;
    }

    @Override
    public int y1() {
        return rectangle.y;
    }

    @Override
    public void y1(int y1) {
        rectangle.y = y1;
    }


    @Override
    public int x2() {
        return rectangle.width+rectangle.x;
    }

    @Override
    public int y2() {
        return rectangle.height+rectangle.y;
    }

    @Override
    public void y2(int y2) {
        rectangle.height = y2-rectangle.y;
    }

    @Override
    public void x2(int x2) {
        rectangle.width = x2-rectangle.x;
    }
}
