package hat.backend;

public class PTXRegister {
    private String name;
    private final Type type;

    public enum Type {
        S8 (8, BasicType.SIGNED, "s8", "%s"),
        S16 (16, BasicType.SIGNED, "s16", "%s"),
        S32 (32, BasicType.SIGNED, "s32", "%s"),
        S64 (64, BasicType.SIGNED, "s64", "%sd"),
        U8 (8, BasicType.UNSIGNED, "u8", "%r"),
        U16 (16, BasicType.UNSIGNED, "u16", "%r"),
        U32 (32, BasicType.UNSIGNED, "u32", "%r"),
        U64 (64, BasicType.UNSIGNED, "u64", "%rd"),
        F16 (16, BasicType.FLOATING, "f16", "%f"),
        F16X2 (16, BasicType.FLOATING, "f16", "%f"),
        F32 (32, BasicType.FLOATING, "f32", "%f"),
        F64 (64, BasicType.FLOATING, "f64", "%fd"),
        B8 (8, BasicType.BIT, "b8", "%b"),
        B16 (16, BasicType.BIT, "b16", "%b"),
        B32 (32, BasicType.BIT, "b32", "%b"),
        B64 (64, BasicType.BIT, "b64", "%bd"),
        B128 (128, BasicType.BIT, "b128", "%b"),
        PREDICATE (1, BasicType.PREDICATE, "pred", "%p");

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
        private final String regPrefix;

        Type(int size, BasicType type, String name, String regPrefix) {
            this.size = size;
            this.basicType = type;
            this.name = name;
            this.regPrefix = regPrefix;
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

        public String getRegPrefix() {
            return this.regPrefix;
        }
    }

    public PTXRegister(int num, Type type) {
        this.type = type;
        this.name = type.regPrefix + num;
    }

    public String name() {
        return this.name;
    }

    public void name(String name) {
        this.name = name;
    }

    public Type type() {
        return this.type;
    }
}
