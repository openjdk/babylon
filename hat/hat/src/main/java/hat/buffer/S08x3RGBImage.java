package hat.buffer;

import hat.Accelerator;
import hat.ifacemapper.Schema;

import java.lang.invoke.MethodHandles;

public interface S08x3RGBImage extends ImageIfaceBuffer<S08x3RGBImage> {
    int width();
    int height();
    byte data(long idx);
    void data(long idx, byte v);

    Schema<S08x3RGBImage> schema = Schema.of(S08x3RGBImage.class, s -> s
            .arrayLen("width", "height").stride(3).array("data")
    );

    static S08x3RGBImage create(Accelerator accelerator, int width, int height){
        return schema.allocate(accelerator,width,height);
    }
}
