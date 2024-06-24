package hat.backend;

public class PTXRegister {
    private String name;
    private final Type type;
    private boolean destination;

    // TODO: d actually stands for dest
    public enum Type {
        S8 (8, BasicType.SIGNED, "s8"),
        S16 (16, BasicType.SIGNED, "s16"),
        S32 (32, BasicType.SIGNED, "s32"),
        S64 (64, BasicType.SIGNED, "s64"),
        U8 (8, BasicType.UNSIGNED, "u8"),
        U16 (16, BasicType.UNSIGNED, "u16"),
        U32 (32, BasicType.UNSIGNED, "u32"),
        U64 (64, BasicType.UNSIGNED, "u64"),
        F16 (16, BasicType.FLOATING, "f16"),
        F16X2 (16, BasicType.FLOATING, "f16"),
        F32 (32, BasicType.FLOATING, "f32"),
        F64 (64, BasicType.FLOATING, "f64"),
        B8 (8, BasicType.BIT, "b8"),
        B16 (16, BasicType.BIT, "b16"),
        B32 (32, BasicType.BIT, "b32"),
        B64 (64, BasicType.BIT, "b64"),
        B128 (128, BasicType.BIT, "b128"),
        PREDICATE (1, BasicType.PREDICATE, "pred");

        public enum BasicType {
            SIGNED,
            UNSIGNED,
            FLOATING,
            BIT,
            PREDICATE
        }

        private final int size;
        private final BasicType basicType;
        private final String name;

        Type(int size, BasicType type, String name) {
            this.size = size;
            this.basicType = type;
            this.name = name;
        }

        public int getSize() {
            return this.size;
        }

        public BasicType getBasicType() {
            return this.basicType;
        }

        public String toString() {
            return this.name;
        }
    }

    public PTXRegister(int num, Type type) {
        this(num, type, false);
    }

    public PTXRegister(int num, Type type, boolean destination) {
        this.type = type;
        if (destination) {
            this.name = "%rd" + num;
        } else if (type.size == 1) {
            this.name = "%p" + num;
        } else {
            this.name = "%r" + num;
        }
    }

    public String name() {
        return this.name;
    }

    public Type type() {
        return this.type;
    }
}
