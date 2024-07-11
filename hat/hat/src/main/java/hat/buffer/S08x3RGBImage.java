package hat.buffer;

import hat.Accelerator;
import hat.ifacemapper.Schema;

import java.lang.invoke.MethodHandles;

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

    static S08x3RGBImage create(MethodHandles.Lookup lookup, BufferAllocator bufferAllocator, int width, int height){
        var instance = schema.allocate(lookup, bufferAllocator,width,height);
        instance.width(width);
        instance.height(height);
        return instance;
    }
    static S08x3RGBImage create(Accelerator accelerator, int width, int height){
        return create(accelerator.lookup,accelerator,width,height);
    }

}
