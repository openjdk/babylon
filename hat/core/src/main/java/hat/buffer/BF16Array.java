package hat.buffer;

import hat.Accelerator;
import hat.ifacemapper.Schema;

public interface BF16Array extends Buffer {
    int length();

    BF16Impl array(long index);

    interface BF16Impl extends Struct, BF16 {
        String NAME = "F16Impl";

        char value();
        void value(char value);
    }

    Schema<BF16Array> schema = Schema.of(BF16Array.class, bf16array ->
            bf16array.arrayLen("length")
                     .array("array", bfloat16 -> bfloat16.fields("value")));

    static BF16Array create(Accelerator accelerator, int length){
        return schema.allocate(accelerator, length);
    }
}