package hat.buffer;

public interface BF16 {

    String HAT_MAPPING_TYPE = BF16.class.getSimpleName();

    char value();
    void value(char value);

    static BF16 of(float value) {
        return new BF16() {
            @Override
            public char value() {
                int bits = Float.floatToRawIntBits(value);
                bits >>= 16;
                return (char) bits;
            }

            @Override
            public void value(char value) {
            }
        };
    }

    static BF16 of(char value) {
        return new BF16() {
            @Override
            public char value() {
                return value;
            }

            @Override
            public void value(char value) {
            }
        };
    }

    static BF16 float2bfloat16(float value) {
        return of(value);
    }

    static float bfloat162float(BF16 value) {
        return Float.intBitsToFloat(value.value() << 16);
    }

    static BF16 add(BF16 ha, BF16 hb) {
        return BF16.of(bfloat162float(ha) + bfloat162float(hb));
    }

    static BF16 add(float f32, BF16 hb) {
        return BF16.of(f32 + bfloat162float(hb));
    }

    static BF16 sub(BF16 ha, BF16 hb) {
        return BF16.of(bfloat162float(ha) - bfloat162float(hb));
    }

    static BF16 sub(float f32, BF16 hb) {
        return BF16.of(f32 - bfloat162float(hb));
    }

    static BF16 sub(BF16 hb, float f32) {
        return BF16.of(bfloat162float(hb) - f32);
    }

    static BF16 mul(BF16 ha, BF16 hb) {
        return BF16.of(bfloat162float(ha) * bfloat162float(hb));
    }

    static BF16 mul(float f32, BF16 hb) {
        return BF16.of(f32 * bfloat162float(hb));
    }

    static BF16 div(BF16 ha, BF16 hb) {
        return BF16.of(bfloat162float(ha) / bfloat162float(hb));
    }

    static BF16 div(float f32, BF16 hb) {
        return BF16.of(f32 / bfloat162float(hb));
    }

    static BF16 add(BF16 hb, float f32) {
        return BF16.of(bfloat162float(hb) / f32);
    }

    default BF16 add(BF16 ha) {
        return BF16.add(this, ha);
    }

    default BF16 sub(BF16 ha) {
        return BF16.sub(this, ha);
    }

    default BF16 mul(BF16 ha) {
        return BF16.mul(this, ha);
    }

    default BF16 div(BF16 ha) {
        return BF16.div(this, ha);
    }

}
