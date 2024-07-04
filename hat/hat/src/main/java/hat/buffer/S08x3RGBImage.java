package hat.buffer;

import hat.ifacemapper.Schema;

public interface S08x3RGBImage extends ImageIfaceBuffer<S08x3RGBImage> {
    byte data(long idx);
    void data(long idx, byte v);
    int width();
    void width(int width);
    int height();
    void height(int height);
    Schema<S08x3RGBImage> schema = Schema.of(S08x3RGBImage.class, s -> s
            .arrayLen("width", "height").stride(3).array("data")
    );

}
